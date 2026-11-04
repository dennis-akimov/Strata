//! "/team [N] TASK" in a chat: N agents (default 3) on one task at once.  The model first splits the task into N parts
//! that do not overlap; each agent then runs the usual MCP tool loop on its part in its own engine slot (the run config's
//! "parallel" sets how many write at once); their lists are merged here, duplicates dropped by address and by headline.
//!
//! The agents search with the stateless tools only: the browser tools drive the one pane people see, so three agents
//! there would click on each other's pages.
// ponytail: a fixed plan -> agents -> merge; agents that talk to each other mid-run if a task ever needs it

use crate::api::McpRun;
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
            (focus, q.trim().to_string())
        })
        .take(n).collect();
    while out.len() < n {
        out.push((format!("{task} (part {} of {n})", out.len() + 1), task.to_string()));
    }
    out
}

/// One line of an agent's list: (headline, address, the line).
pub type Found = (String, String, String);

/// The "[headline](address)" lines of an agent's answer.
pub fn items(answer: &str) -> Vec<Found> {
    let re = regex::Regex::new(r"\[([^\]]+)\]\((https?://[^)\s]+)\)").unwrap();
    answer.lines().filter_map(|l| re.captures(l).map(|c| (c[1].trim().to_string(), c[2].to_string(), l.trim().trim_start_matches(['-', '*', ' ']).to_string()))).collect()
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

/// The pages a web search gave ("[k] web://... · headline · address · ..."), each once.
pub fn pages(result: &str) -> Vec<(String, String)> {
    let re = regex::Regex::new(r"(?m)^\[\d+\] web://\S+ · (.+?) · (https?://\S+) · ").unwrap();
    let mut out: Vec<(String, String)> = vec![];
    for c in re.captures_iter(result) {
        if !out.iter().any(|(_, u)| page_key(u) == page_key(&c[2])) {
            out.push((c[1].trim().to_string(), c[2].to_string()));
        }
    }
    out
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

/// One agent: its search (run here, not left to the model: a small model listed made-up example.com news without
/// one), then the model's list from what it found.  Only pages the search gave are kept; the ones the list left
/// out are added as they are ("all" was asked).
#[allow(clippy::too_many_arguments)]
fn agent(svc: &Service, i: usize, n: usize, task: &str, focus: &str, query: &str, others: &str, tool: &str, kw: &Map<String, Value>, sampling: &Map<String, Value>, cancel: &Arc<Flag>, tx: &mpsc::Sender<Value>) -> Vec<Found> {
    let hub = svc.mcp.clone().expect("hub");
    let id = format!("team{}-{}", i + 1, uuid::Uuid::new_v4().simple());
    // the search engines only: "papers" answered news questions with research papers
    let args = json!({"query": format!("web: {query}; engines=google,bing,duckduckgo; fresh=day; n=8")});
    let _ = tx.send(json!({"event": "call", "id": id, "name": tool, "arguments": args, "round": 1, "agent": i + 1}));
    let r = hub.call(tool, &args, Some(cancel)).unwrap_or_default();
    let text = r.get("text").and_then(Value::as_str).unwrap_or("").to_string();
    let mut ev = json!({"event": "result", "id": id, "agent": i + 1});
    for k in ["ok", "text", "chars", "truncated", "ms"] {
        ev[k] = r.get(k).cloned().unwrap_or(Value::Null);
    }
    let _ = tx.send(ev);
    let found = pages(&text);
    crate::say(&format!("team agent {}: {} pages for {query:?}", i + 1, found.len()));
    if found.is_empty() || cancel.is_set() {
        return vec![];
    }
    let room: String = text.chars().take(12000).collect();
    let prompt = format!("You are agent {} of {n} on this task: {task}\nYour part: {focus}. The others cover: {others}.\n\n\
        Your web search found:\n{room}\n\nList every item of your part from these results, one per line: - [headline](address) - \
        one sentence from the result. Only these results' addresses; nothing else.", i + 1);
    let said = ask(svc, &prompt, kw, 1200, sampling, cancel).unwrap_or_default();
    let mut list: Vec<Found> = items(&said).into_iter().filter(|f| found.iter().any(|(_, u)| page_key(u) == page_key(&f.1))).collect();
    for (t, u) in &found {
        if !list.iter().any(|f| page_key(&f.1) == page_key(u)) {
            list.push((t.clone(), u.clone(), format!("[{}]({u})", t.replace(['[', ']'], ""))));
        }
    }
    list
}

/// The team's run: the plan, the agents at once, their merged list; emitted as one answer.
#[allow(clippy::too_many_arguments)]
pub fn run(svc: &Service, n: usize, task: &str, m: McpRun, tools: Vec<Value>, sampling: &Map<String, Value>, cancel: &Arc<Flag>, emit: &mut dyn FnMut(Item) -> Result<(), Closed>) -> Result<(), Stop<Fail>> {
    let say = |emit: &mut dyn FnMut(Item) -> Result<(), Closed>, t: String| emit(Item::Event(Event::text(Kind::Content, t)));
    // ponytail: the web search is bible's find_skill("web: ..."); another stateless search tool would need its own call
    let Some(tool) = tools.iter().filter_map(|t| t["name"].as_str()).find(|n| n.ends_with("__find_skill")).map(String::from) else {
        return Err(Stop::Err(Fail::Value("/team searches with bible's web search (find_skill \"web: ...\"); no MCP server has it".into())));
    };
    let t0 = std::time::Instant::now();
    let plan = format!("Today is {}. Split this research task among {n} agents so that no two of them look for the same \
        things.\nThe task: {task}\n\nReply with exactly {n} lines and nothing else. Each line: a short focus, then \" | \", then a web \
        search query for it. For example:\nStock markets | stock market news today\nCentral banks | central bank interest rate news", chrono::Local::now().format("%Y-%m-%d"));
    let plan = ask(svc, &plan, &m.kw, 300, sampling, cancel)?;
    crate::say(&format!("team plan: {plan:?}"));
    let parts = parts(&plan, n, task);
    say(emit, format!("**A team of {n}:**\n{}\n\n", parts.iter().enumerate().map(|(i, (f, q))| format!("{}. {f} (\"{q}\")", i + 1)).collect::<Vec<_>>().join("\n")))?;
    let (tx, rx) = mpsc::channel::<Value>();
    let lists: Vec<Vec<Found>> = std::thread::scope(|s| {
        let handles: Vec<_> = parts.iter().enumerate().map(|(i, (focus, query))| {
            let others = parts.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, p)| p.0.as_str()).collect::<Vec<_>>().join("; ");
            let (tx, tool, kw) = (tx.clone(), tool.clone(), m.kw.clone());
            s.spawn(move || agent(svc, i, n, task, focus, query, &others, &tool, &kw, sampling, cancel, &tx))
        }).collect();
        drop(tx);
        // the agents' searches go out as they happen (the page shows them); keep-alives while they write; a client
        // gone stops them all
        loop {
            match rx.recv_timeout(std::time::Duration::from_secs(5)) {
                Ok(v) => {
                    if emit(Item::Mcp(v)).is_err() {
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
    let found: usize = lists.iter().map(Vec::len).sum();
    let (lists, dropped) = dedupe(lists);
    crate::say(&format!("team of {n}: {found} items, {dropped} duplicates dropped, {:.0} s", t0.elapsed().as_secs_f64()));
    let mut out = String::new();
    for (i, ((focus, _), list)) in parts.iter().zip(&lists).enumerate() {
        out.push_str(&format!("### {}. {focus}\n", i + 1));
        if list.is_empty() {
            out.push_str("(nothing new found)\n");
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

    #[test]
    fn the_pages_of_a_web_search() {
        let r = "web: x · 3 pages\n[1] web://a/1 · Apple is the priciest · https://finance.yahoo.com/v.html · google · fetched\n<<<\ntext\n>>>\n[2] web://a/1 · Apple is the priciest · https://finance.yahoo.com/v.html · google\n[3] web://a/2 · Oil · https://b.com/o · bing";
        assert_eq!(pages(r), vec![("Apple is the priciest".into(), "https://finance.yahoo.com/v.html".into()), ("Oil".into(), "https://b.com/o".into())]);
    }
}
