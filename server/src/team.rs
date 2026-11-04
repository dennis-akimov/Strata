//! "/team [N] TASK" in a chat: N agents (default 3) on one task at once.  The model first splits the task into N parts
//! that do not overlap; each agent then runs the usual MCP tool loop on its part in its own engine slot (the run config's
//! "parallel" sets how many write at once); their lists are merged here, duplicates dropped by address and by headline.
//!
//! Each agent browses in a browser of its own: Strata.app serves them beside the pane people see (/mcp/1-4, the MCP
//! servers agent1-agent4); agent i gets only agent{i}'s tools, so no two click on each other's pages.
// ponytail: a fixed plan -> agents -> merge; agents that talk to each other mid-run if a task ever needs it

use crate::api::{run_with_mcp, McpRun};
use crate::errors::Fail;
use crate::parser::{Event, Kind};
use crate::run::Item;
use crate::service::Service;
use crate::sync::{Closed, Event as Flag, Stop};
use serde_json::{json, Map, Value};
use std::collections::HashSet;
use std::sync::{mpsc, Arc};

/// "/team TASK" or "/team 4 TASK" as the last user message: (agents, the task), the message left as the task.
pub fn command(messages: &mut [Value]) -> Option<(usize, String)> {
    let last = messages.iter_mut().rev().find(|m| m["role"] == "user")?;
    let text = last["content"].as_str()?.trim_start().to_string();
    let rest = text.strip_prefix("/team")?;
    if !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let rest = rest.trim_start();
    let (n, task) = match rest.split_once(char::is_whitespace) {
        Some((k, t)) if k.parse::<usize>().is_ok() => (k.parse::<usize>().unwrap().clamp(2, 4), t.trim()),
        _ => (3, rest.trim()),
    };
    if task.is_empty() {
        return None;
    }
    last["content"] = json!(task);
    Some((n, task.to_string()))
}

/// The planner's lines ("focus | query") as N parts; short of N, the task itself fills in.
pub fn parts(text: &str, n: usize, task: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = text.lines()
        .map(|l| l.trim().trim_start_matches(|c: char| c.is_ascii_digit() || matches!(c, '.' | ')' | '-' | '*' | ' ')).trim())
        .filter(|l| !l.is_empty())
        .map(|l| {
            let mut f = l.split('|').map(str::trim);
            let focus = f.next().unwrap_or(l).to_string();
            let q = f.next().unwrap_or(l).trim_start_matches("search for").trim_start_matches("query").replace('"', "");
            // "financial news search today": the word "search" in a query only narrows Google to search pages
            let q = q.split_whitespace().filter(|w| !w.eq_ignore_ascii_case("search")).collect::<Vec<_>>().join(" ");
            (focus, q)
        })
        .take(n).collect();
    while out.len() < n {
        out.push((format!("{task} (part {} of {n})", out.len() + 1), task.to_string()));
    }
    out
}

/// One line of an agent's list: (headline, address, the line).
pub type Found = (String, String, String);

/// The list lines of an agent's answer: "- [headline](address) - ...", "- headline (address) - ...", or a headline
/// with no address (then "" : `address_from` finds it among the links its browser showed).
pub fn items(answer: &str) -> Vec<Found> {
    let md = regex::Regex::new(r"\[([^\]]+)\]\((https?://[^)\s]+)\)").unwrap();
    let bare = regex::Regex::new(r"https?://[^)\s\]]+").unwrap();
    let cut = regex::Regex::new(r"\s+(\(|[-–—:]\s)").unwrap();
    answer.lines().map(str::trim).filter(|l| l.starts_with(['-', '*', '•']) || l.starts_with(|c: char| c.is_ascii_digit())).filter_map(|l| {
        let line = l.trim_start_matches(|c: char| matches!(c, '-' | '*' | '•' | '.' | ')' | ' ') || c.is_ascii_digit()).to_string();
        if let Some(c) = md.captures(&line) {
            return Some((c[1].trim().to_string(), c[2].to_string(), line.clone()));
        }
        let url = bare.find(&line).map(|m| m.as_str().to_string()).unwrap_or_default();
        let title = cut.split(&line).next().unwrap_or(&line).trim().trim_matches(['*', '"']).to_string();
        (title.split_whitespace().count() >= 3).then_some((title, url, line))
    }).collect()
}

/// The links the agent's browser showed ('[n] link "headline" -> host/path'): (headline, address).
pub fn links(read: &str) -> Vec<(String, String)> {
    let re = regex::Regex::new(r#"link "([^"]{12,})" -> (\S+)"#).unwrap();
    re.captures_iter(read).map(|c| (c[1].to_string(), format!("https://{}", &c[2]))).collect()
}

/// A headline's address among those links: the one sharing most of its words (6 in 10 at least).
fn address_from(title: &str, links: &[(String, String)]) -> Option<String> {
    let w = words(title);
    links.iter().map(|(t, u)| (w.intersection(&words(t)).count(), u)).filter(|(k, _)| *k >= 3 && k * 10 >= w.len() * 6)
        .max_by_key(|(k, _)| *k).map(|(_, u)| u.clone())
}

/// An address without what does not change the page: scheme, "www.", query, fragment, a trailing slash.
fn page_key(url: &str) -> String {
    let u = url.split(['?', '#']).next().unwrap_or(url).to_lowercase();
    let u = u.trim_start_matches("https://").trim_start_matches("http://").trim_start_matches("www.");
    u.trim_end_matches('/').to_string()
}

fn words(title: &str) -> HashSet<String> {
    title.split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 2).map(str::to_lowercase).collect()
}

/// The same story twice: the same page, or headlines sharing most of their words (two sites on one story).
// ponytail: word overlap of headlines; embeddings if paraphrased duplicates get through
fn same(a: &Found, b: &Found) -> bool {
    if page_key(&a.1) == page_key(&b.1) {
        return true;
    }
    let (x, y) = (words(&a.0), words(&b.0));
    let common = x.intersection(&y).count();
    common >= 4 && common * 10 >= x.len().min(y.len()) * 6
}

/// Each agent's items without the ones an earlier item already holds; and how many were dropped.
pub fn dedupe(lists: Vec<Vec<Found>>) -> (Vec<Vec<Found>>, usize) {
    let mut kept: Vec<Found> = vec![];
    let mut dropped = 0;
    let out = lists.into_iter().map(|list| {
        list.into_iter().filter(|f| {
            if kept.iter().any(|k| same(k, f)) {
                dropped += 1;
                return false;
            }
            kept.push(f.clone());
            true
        }).collect()
    }).collect();
    (out, dropped)
}

/// An ad or a sign-up offer, not an item: a small model lists a page's newsletter boxes as news.
// ponytail: word list; widen it when other ads get through
fn ad(line: &str) -> bool {
    let l = line.to_lowercase();
    ["newsletter", "subscribe", "subscription", "inbox", "sign up", "sponsored", "advertis", "promo code", "free trial"].iter().any(|w| l.contains(w))
}

/// The site of an address (its host without "www."): an item counts when its site is on a page the agent read.
fn site(url: &str) -> String {
    page_key(url).split('/').next().unwrap_or("").to_string()
}

/// The model's whole answer to one prompt, no tools and no thinking (its think closed for it: a plan and a list need
/// none, and LFM's thinking took most of a minute a turn on the M1).
fn ask(svc: &Service, prompt: &str, kw: &Map<String, Value>, max: i64, sampling: &Map<String, Value>, cancel: &Arc<Flag>) -> Result<String, Stop<Fail>> {
    let messages = [json!({"role": "user", "content": prompt})];
    let (mut ids, mut thinking, max_new) = svc.prepare(&messages, None, kw, Some(max), None, None).map_err(Stop::Err)?;
    if thinking && !svc.harmony {
        ids.extend(svc.tok.encode(&format!("{}\n\n", crate::parser::THINK_END), true));
        thinking = false;
    }
    let mut text = String::new();
    svc.run(&ids, thinking, None, max_new, sampling, cancel, None, &mut |item| {
        if let Item::Event(ev) = item && ev.kind == Kind::Content {
            text.push_str(&ev.text);
        }
        Ok(())
    })?;
    Ok(text)
}

/// One agent: the usual tool loop with its own browser, then its list; only items from sites its browser showed are
/// kept (a small model listed made-up example.com news without searching).
#[allow(clippy::too_many_arguments)]
fn agent(svc: &Service, i: usize, n: usize, task: &str, focus: &str, query: &str, others: &str, tools: Vec<Value>, kw: &Map<String, Value>, max_req: i64, sampling: &Map<String, Value>, cancel: &Arc<Flag>, tx: &mpsc::Sender<Value>) -> (Vec<Found>, String) {
    let prompt = format!("You are agent {} of {n} on this task: {task}\nToday is {}. Your part only: {focus}. Leave to the others: {others}.\n\
        Your browser searched \"{query}\" (below). browser_click the two or three most relevant results to read them.\n\
        Then answer with only a list, one item per line: - [headline](address) - one sentence. Only news items from pages you read, \
        each with its own address; the newest first; no ads, newsletters or tables.", i + 1, chrono::Local::now().format("%Y-%m-%d"));
    // the first search run here, not left to the model: told to search, two of three agents made their lists up
    let search = format!("agent{}__browser_search", i + 1);
    let args = json!({"query": query});
    let id = format!("team{}-search", i + 1);
    let _ = tx.send(json!({"event": "call", "id": id, "name": search, "arguments": args, "round": 0, "agent": i + 1}));
    let r = svc.mcp.as_ref().map(|h| h.call(&search, &args, Some(cancel)).unwrap_or_default()).unwrap_or_default();
    let found = r.get("text").and_then(Value::as_str).unwrap_or("").to_string();
    let mut ev = json!({"event": "result", "id": id, "agent": i + 1});
    for k in ["ok", "text", "chars", "truncated", "ms"] {
        ev[k] = r.get(k).cloned().unwrap_or(Value::Null);
    }
    let _ = tx.send(ev);
    let messages = vec![
        json!({"role": "user", "content": prompt}),
        json!({"role": "assistant", "content": "", "tool_calls": [{"function": {"name": search, "arguments": args}}]}),
        json!({"role": "tool", "content": found}),
    ];
    let Ok((ids, thinking, max_new)) = svc.prepare(&messages, Some(&tools), kw, Some(max_req), None, None) else { return (vec![], String::new()) };
    let names = tools.iter().filter_map(|t| t["name"].as_str().map(String::from)).collect();
    let run = McpRun { messages, kw: kw.clone(), max_req, names, team: 0, rounds: Some(3) };
    let (mut text, mut read) = (String::new(), found.clone());
    let r = run_with_mcp(svc, run, ids, thinking, Some(tools), max_new, sampling, cancel, &mut |item| {
        match item {
            Item::Mcp(mut v) => {
                if v["event"] == "result" {
                    read.push_str(v["text"].as_str().unwrap_or(""));
                }
                v["agent"] = json!(i + 1);
                let _ = tx.send(v);
            }
            Item::Ping => {
                let _ = tx.send(Value::Null);
            }
            Item::Event(ev) if ev.kind == Kind::Content => text.push_str(&ev.text),
            _ => {}
        }
        Ok(())
    });
    if let Err(Stop::Err(e)) = r {
        crate::say(&format!("team agent {}: {e}", i + 1));
    }
    let shown = links(&read);
    let read = read.to_lowercase();
    let all: Vec<Found> = items(&text).into_iter().map(|(t, u, l)| {
        let u = if u.is_empty() { address_from(&t, &shown).unwrap_or_default() } else { u };
        // a Chrome text fragment (#:~:text=...) marks a spot on the page, not a page
        let (u, l) = match u.find("#:~:") { Some(k) => (u[..k].to_string(), l.replace(&u, &u[..k])), None => (u, l) };
        let l = if l.contains(&u) || u.is_empty() { l } else { format!("[{}]({u}) - {}", t.replace(['[', ']'], ""), l.strip_prefix(t.as_str()).unwrap_or(&l).trim_start_matches([' ', '-', '–', '—', ':']).trim()) };
        (t, u, l)
    }).collect();
    let kept: Vec<Found> = all.iter().filter(|f| !site(&f.1).is_empty() && read.contains(&site(&f.1)) && !ad(&f.2)).cloned().collect();
    crate::say(&format!("team agent {}: {} items, {} from sites its browser showed", i + 1, all.len(), kept.len()));
    (kept, text)
}

/// The team's run: the plan, the agents at once, their merged list; emitted as one answer.
#[allow(clippy::too_many_arguments)]
pub fn run(svc: &Service, n: usize, task: &str, m: McpRun, tools: Vec<Value>, sampling: &Map<String, Value>, cancel: &Arc<Flag>, emit: &mut dyn FnMut(Item) -> Result<(), Closed>) -> Result<(), Stop<Fail>> {
    let say = |emit: &mut dyn FnMut(Item) -> Result<(), Closed>, t: String| emit(Item::Event(Event::text(Kind::Content, t)));
    // agent i's browser: the MCP server agent{i}; open and click only.  A click answers with the page's first part
    // (browser_read of parts 2 and 3 was a round each of prefill on the M1, for page footers); the search is the
    // server's (given it, two of three agents spent every round searching again and read no page)
    let own = |i: usize| -> Vec<Value> {
        let p = format!("agent{}__", i + 1);
        tools.iter().filter(|t| t["name"].as_str().and_then(|n| n.strip_prefix(&p)).is_some_and(|n| matches!(n, "browser_open" | "browser_click"))).cloned().collect()
    };
    if (0..n).any(|i| own(i).is_empty()) {
        return Err(Stop::Err(Fail::Value(format!("/team needs a browser for each of its {n} agents: Strata.app gives them (agent1-agent4); this server runs without the app"))));
    }
    let t0 = std::time::Instant::now();
    let plan = format!("Today is {}. Split this research task among {n} agents so that no two of them look for the same \
        things. Each gets a Google search of a few words (what a person types, not a site's name).\nThe task: {task}\n\nReply with exactly {n} lines and nothing else. Each line: a short focus, then \" | \", then a web \
        search query for it. For example:\nStock markets | stock market news today\nCentral banks | central bank interest rate news", chrono::Local::now().format("%Y-%m-%d"));
    let plan = ask(svc, &plan, &m.kw, 300, sampling, cancel)?;
    crate::say(&format!("team plan: {plan:?}"));
    let parts = parts(&plan, n, task);
    say(emit, format!("**A team of {n}:**\n{}\n\n", parts.iter().enumerate().map(|(i, (f, q))| format!("{}. {f} (\"{q}\")", i + 1)).collect::<Vec<_>>().join("\n")))?;
    let (tx, rx) = mpsc::channel::<Value>();     // Null: a keep-alive
    let done: Vec<(Vec<Found>, String)> = std::thread::scope(|s| {
        let handles: Vec<_> = parts.iter().enumerate().map(|(i, (focus, query))| {
            let others = parts.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, p)| p.0.as_str()).collect::<Vec<_>>().join("; ");
            let (tx, tools, kw) = (tx.clone(), own(i), m.kw.clone());
            s.spawn(move || agent(svc, i, n, task, focus, query, &others, tools, &kw, m.max_req, sampling, cancel, &tx))
        }).collect();
        drop(tx);
        // the agents' searches go out as they happen (the page shows them); keep-alives while they write; a client
        // gone stops them all
        loop {
            match rx.recv_timeout(std::time::Duration::from_secs(5)) {
                Ok(v) => {
                    if emit(if v.is_null() { Item::Ping } else { Item::Mcp(v) }).is_err() {
                        cancel.set();
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if emit(Item::Ping).is_err() {
                        cancel.set();
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        handles.into_iter().map(|h| h.join().unwrap_or_default()).collect()
    });
    if cancel.is_set() {
        return Err(Stop::Closed);
    }
    let (lists, said): (Vec<Vec<Found>>, Vec<String>) = done.into_iter().unzip();
    let found: usize = lists.iter().map(Vec::len).sum();
    let (lists, dropped) = dedupe(lists);
    crate::say(&format!("team of {n}: {found} items, {dropped} duplicates dropped, {:.0} s", t0.elapsed().as_secs_f64()));
    let mut out = String::new();
    for (i, ((focus, _), list)) in parts.iter().zip(&lists).enumerate() {
        out.push_str(&format!("### {}. {focus}\n", i + 1));
        if list.is_empty() {
            // what it said, marked: none of it is from a page its browser showed
            let raw: String = said[i].trim().chars().take(800).collect();
            out.push_str(&if raw.is_empty() { "(nothing found)\n".to_string() } else { format!("(unchecked: no item named a page its browser showed)\n> {}\n", raw.replace('\n', "\n> ")) });
        }
        for (_, _, line) in list {
            out.push_str(&format!("- {line}\n"));
        }
        out.push('\n');
    }
    out.push_str(&format!("{} items from {n} agents in {:.0} s; {dropped} duplicates removed.", found - dropped, t0.elapsed().as_secs_f64()));
    say(emit, out)?;
    let mut done = Map::new();
    done.insert("finish".into(), json!("stop"));
    emit(Item::Done(done))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_command_and_its_count() {
        let mut m = vec![json!({"role": "user", "content": "/team find financial news"})];
        assert_eq!(command(&mut m), Some((3, "find financial news".into())));
        assert_eq!(m[0]["content"], "find financial news");
        let mut m = vec![json!({"role": "user", "content": "/team 9 x"})];
        assert_eq!(command(&mut m).unwrap().0, 4);
        for t in ["/teams x", "find /team x", "/team   "] {
            assert_eq!(command(&mut [json!({"role": "user", "content": t})]), None, "{t}");
        }
    }

    #[test]
    fn the_plan_parts_fill_in() {
        let p = parts("1. Stocks | stock market news\n2) Crypto | search for \"bitcoin news\" | extra\n", 3, "news");
        assert_eq!(p[0], ("Stocks".into(), "stock market news".into()));
        assert_eq!(p[1].1, "bitcoin news");
        assert_eq!(p[2].1, "news");
        assert_eq!(parts("News | financial news search today", 1, "x")[0].1, "financial news today");
        assert!(ad("Free Money Insights From SoFi, Delivered Straight to Your Inbox") && ad("Briefings, Goldman Sachs' weekly newsletter"));
        assert!(!ad("Stock futures fall as Democrats win big in midterms"));
    }

    #[test]
    fn list_lines_with_and_without_addresses() {
        let it = items("Here:\n- [Fed holds rates](https://a.com/f) - x\n- AI stocks slide as Texas result raises risks (Yahoo Finance) – y\n2. Oil rises on supply fears https://b.com/o\nno list line");
        assert_eq!(it.len(), 3);
        assert_eq!(it[1].0, "AI stocks slide as Texas result raises risks");
        assert_eq!(it[1].1, "");
        assert_eq!(it[2].1, "https://b.com/o");
        let shown = links(r#"[3] link "AI stocks slide as Texas governor result raises data center risks" -> finance.yahoo.com/news/ai-stocks-1.html"#);
        assert_eq!(address_from(&it[1].0, &shown).as_deref(), Some("https://finance.yahoo.com/news/ai-stocks-1.html"));
        assert_eq!(address_from("Gold at a record high today", &shown), None);
    }

    #[test]
    fn duplicates_by_page_and_by_headline() {
        let a = items("- [Fed holds interest rates steady in November meeting](https://www.reuters.com/a?x=1) - x\n- [Oil rises](https://b.com/o)");
        let b = items("* [Fed holds interest rates steady at November meeting](https://cnbc.com/f) - y\n- [Oil rises](http://b.com/o/)\n- [Gold at a record](https://c.com/g)");
        let (lists, dropped) = dedupe(vec![a, b]);
        assert_eq!(dropped, 2);
        assert_eq!(lists[1].len(), 1);
        assert_eq!(lists[1][0].0, "Gold at a record");
    }
}
