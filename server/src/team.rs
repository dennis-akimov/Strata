//! "/team [N] TASK" in a chat: N agents (default 3) on one task at once.  The model first splits the task into N parts
//! that do not overlap; each agent then runs the usual MCP tool loop on its part in its own engine slot (the run config's
//! "parallel" sets how many write at once); their lists are merged here, duplicates dropped by address and by headline.
//!
//! The agents search with the stateless tools only: the browser tools drive the one pane people see, so three agents
//! there would click on each other's pages.
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
        .map(|l| match l.split_once('|') {
            Some((f, q)) => (f.trim().to_string(), q.trim().to_string()),
            None => (l.to_string(), l.to_string()),
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

/// The model's whole answer to one prompt, no tools (the plan).
fn ask(svc: &Service, messages: &[Value], kw: &Map<String, Value>, max: i64, sampling: &Map<String, Value>, cancel: &Arc<Flag>) -> Result<String, Stop<Fail>> {
    let (ids, thinking, max_new) = svc.prepare(messages, None, kw, Some(max), None, None).map_err(Stop::Err)?;
    let mut text = String::new();
    svc.run(&ids, thinking, None, max_new, sampling, cancel, None, &mut |item| {
        if let Item::Event(ev) = item {
            if ev.kind == Kind::Content {
                text.push_str(&ev.text);
            }
        }
        Ok(())
    })?;
    Ok(text)
}

/// The team's run: the plan, the agents at once, their merged list; emitted as one answer.
#[allow(clippy::too_many_arguments)]
pub fn run(svc: &Service, n: usize, task: &str, m: McpRun, tools: Vec<Value>, sampling: &Map<String, Value>, cancel: &Arc<Flag>, emit: &mut dyn FnMut(Item) -> Result<(), Closed>) -> Result<(), Stop<Fail>> {
    let say = |emit: &mut dyn FnMut(Item) -> Result<(), Closed>, t: String| emit(Item::Event(Event::text(Kind::Content, t)));
    let plan = format!("Split this research task among {n} agents so that no two of them look for the same things. \
        Reply with exactly {n} lines and nothing else, one per agent: a short focus, then \" | \", then a web search \
        query for it.\n\nThe task: {task}");
    let plan = ask(svc, &[json!({"role": "user", "content": plan})], &m.kw, 1500, sampling, cancel)?;
    let parts = parts(&plan, n, task);
    say(emit, format!("**A team of {n}:**\n{}\n\n", parts.iter().enumerate().map(|(i, (f, _))| format!("{}. {f}", i + 1)).collect::<Vec<_>>().join("\n")))?;
    crate::say(&format!("team of {n}: {}", parts.iter().map(|p| p.0.as_str()).collect::<Vec<_>>().join(" / ")));
    let search = tools.iter().filter_map(|t| t["name"].as_str()).find(|n| n.ends_with("__find_skill")).map(|n| format!(
        " Search with {n}, its query starting with \"web: \" (for example \"web: QUERY; fresh=day; n=8\"): its results carry the pages' text, so one or two searches are enough."))
        .unwrap_or_default();
    let (tx, rx) = mpsc::channel::<Option<Value>>();
    let answers: Vec<String> = std::thread::scope(|s| {
        let handles: Vec<_> = parts.iter().enumerate().map(|(i, (focus, query))| {
            let others: Vec<&str> = parts.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, p)| p.0.as_str()).collect();
            let prompt = format!("You are agent {} of {n} on this task: {task}\nYour part only: {focus}. Leave these to the others: {}.\
                {search} Start from: {query}\nThen answer with only a list, one item per line: - [headline](address) - one \
                sentence. Only items your results show, each with its own address; the newest first.",
                i + 1, others.join("; "));
            let (tx, tools, kw) = (tx.clone(), tools.clone(), m.kw.clone());
            let names = m.names.clone();
            s.spawn(move || -> String {
                let messages = vec![json!({"role": "user", "content": prompt})];
                let Ok((ids, thinking, max_new)) = svc.prepare(&messages, Some(&tools), &kw, Some(m.max_req), None, None) else { return String::new() };
                let mut text = String::new();
                let run = McpRun { messages, kw, max_req: m.max_req, names, team: 0 };
                let _ = run_with_mcp(svc, run, ids, thinking, Some(tools.clone()), max_new, sampling, cancel, &mut |item| {
                    match item {
                        Item::Mcp(mut v) => {
                            v["agent"] = json!(i + 1);
                            let _ = tx.send(Some(v));
                        }
                        Item::Ping => {
                            let _ = tx.send(None);
                        }
                        Item::Event(ev) if ev.kind == Kind::Content => text.push_str(&ev.text),
                        _ => {}
                    }
                    Ok(())
                });
                text
            })
        }).collect();
        drop(tx);
        // the agents' tool calls go out as they happen (the page shows them); a client gone stops them all
        for v in rx {
            let r = match v {
                Some(v) => emit(Item::Mcp(v)),
                None => emit(Item::Ping),
            };
            if r.is_err() {
                cancel.set();
            }
        }
        handles.into_iter().map(|h| h.join().unwrap_or_default()).collect()
    });
    if cancel.is_set() {
        return Err(Stop::Closed);
    }
    let lists: Vec<Vec<Found>> = answers.iter().map(|a| items(a)).collect();
    let found: usize = lists.iter().map(Vec::len).sum();
    let (lists, dropped) = dedupe(lists);
    crate::say(&format!("team of {n}: {found} items, {dropped} duplicates dropped"));
    let mut out = String::new();
    for (i, ((focus, _), list)) in parts.iter().zip(&lists).enumerate() {
        out.push_str(&format!("### {}. {focus}\n", i + 1));
        if list.is_empty() {
            let raw = answers[i].trim();
            out.push_str(if raw.is_empty() { "(nothing found)" } else { raw });
            out.push('\n');
        }
        for (_, _, line) in list {
            out.push_str(&format!("- {line}\n"));
        }
        out.push('\n');
    }
    out.push_str(&format!("{} items from {n} agents; {dropped} duplicates removed.", found - dropped));
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
        let p = parts("1. Stocks | stock market news\n2) Crypto | bitcoin news\n", 3, "news");
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
}
