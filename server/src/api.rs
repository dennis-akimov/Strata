//! The chat APIs over Service::run (serve/server.py): OpenAI Chat Completions chunks and the whole answer collected
//! from them, the structured-output buffer, Anthropic Messages events and their collected message, and the MCP tool
//! loop (run_with_mcp) the web app opts into.  `out` gets each chunk (None: a keep-alive); Err(Closed) is the client
//! gone.

use crate::errors::Fail;
use crate::parser::Kind;
use crate::py::{get, truthy};
use crate::pyjson;
use crate::run::Item;
use crate::say;
use crate::service::Service;
use crate::structured::{validated_json, Validator};
use crate::sync::{Closed, Event as Flag, Stop};
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// llama.cpp's `prompt_progress` from the last PP line (None: nothing to say, or the read is all but done).
pub fn prompt_progress(svc: &Service) -> Option<Value> {
    if svc.engine.batch() > 0 {
        return None;
    }
    let live = svc.engine.live();
    let (read, total) = live.progress?;
    if total <= 0 || total - read <= 64 {
        return None;
    }
    Some(json!({"total": total, "cache": live.reused, "processed": read, "time_ms": live.progress_ms as i64}))
}

/// What a request runs: the service's events, or the MCP loop's.
pub struct Run<'a> {
    pub svc: &'a Service,
    pub ids: Vec<u32>,
    pub thinking: bool,
    pub tools: Option<Vec<Value>>,
    pub max_new: i64,
    pub sampling: Map<String, Value>,
    pub cancel: Arc<Flag>,
    pub force: Option<String>,
    pub mcp: Option<McpRun>,
}

pub struct McpRun {
    pub messages: Vec<Value>,
    pub kw: Map<String, Value>,
    pub max_req: i64,
    pub names: HashSet<String>,
}

impl Run<'_> {
    pub fn go(&mut self, emit: &mut dyn FnMut(Item) -> Result<(), Closed>) -> Result<(), Stop<Fail>> {
        match self.mcp.take() {
            Some(m) => run_with_mcp(self.svc, m, self.ids.clone(), self.thinking, self.tools.clone(), self.max_new, &self.sampling, &self.cancel, emit),
            None => self.svc.run(&self.ids, self.thinking, self.tools.as_deref(), self.max_new, &self.sampling, &self.cancel, self.force.clone(), emit),
        }
    }
}

/// A tool call's arguments for the log, cut to 300 characters (which columns a write gave is what a wrong sheet asked).
pub fn args_brief(args: &Map<String, Value>) -> String {
    let t = Value::Object(args.clone()).to_string();
    if t.chars().count() > 300 { format!("{}...", t.chars().take(300).collect::<String>()) } else { t }
}

/// A browser result's page (browser_mcp's page_text: its title on the first line, its address on the second; a search
/// result starts with a note in [...] first): (title, address); a Google results page as "Google: <query>".
pub fn page_source(text: &str) -> Option<(String, String)> {
    // the browser's note on top ("[Read this first. ...]", "[Today is ...]") is not the page
    let body = if text.starts_with('[') { text.split_once("\n\n").map(|x| x.1).unwrap_or(text) } else { text };
    if body.contains("\n(no text on this page") {
        return None;                                      // a bot check or a blank page: nothing came from it
    }
    let mut lines = body.lines();
    let title = lines.next()?.trim().to_string();
    let url = lines.next()?.trim().to_string();
    if !url.starts_with("http") {
        return None;
    }
    let title = if url.contains("google.") && url.contains("/search") { title.trim_end_matches(" - Google Search").to_string() } else { title };
    Some((if title.is_empty() { url.clone() } else { title }, url))
}

/// A page the browser gave the model: (title, address, the text it read).
pub type Source = (String, String, String);

fn trigrams(text: &str) -> HashSet<String> {
    let w: Vec<String> = text.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(str::to_lowercase).collect();
    w.windows(3).map(|t| t.join(" ")).collect()
}

/// Whether the answer says what the page says: 3 of its word triples are on the page, the question's own left out
/// ("events in san diego" is on every page the search found).  A soft 404 ("Page not found" served with 200, RFC 9110
/// 15.5.5 has no say there), a bot check or an error page shares none, so it is not cited.
// ponytail: word-triple overlap, not entailment; a model judging each page if wrong citations still get through
pub fn supports(page: &str, answer: &str, question: &str) -> bool {
    let q = trigrams(question);
    let page = trigrams(page);
    trigrams(answer).iter().filter(|t| !q.contains(*t) && page.contains(*t)).count() >= 3
}

/// The Sources list under an answer: the pages it used (what it says is on them) and did not link itself, search pages
/// only when no other page was used, at most 6; None when there is nothing to add.
pub fn sources_text(sources: &[Source], answer: &str, question: &str) -> Option<String> {
    let used: Vec<&Source> = sources.iter().filter(|(_, _, text)| supports(text, answer, question)).collect();
    let search = |u: &str| u.contains("google.") && u.contains("/search");
    let pages: Vec<&Source> = used.iter().copied().filter(|(_, u, _)| !search(u)).collect();
    let list = if pages.is_empty() { used } else { pages };
    let lines: Vec<String> = list.iter().filter(|(_, u, _)| !answer.contains(u.as_str())).take(6)
        .map(|(t, u, _)| format!("- [{}]({u})", t.replace(['[', ']'], ""))).collect();
    (!lines.is_empty()).then(|| format!("\n\n**Sources**\n{}", lines.join("\n")))
}

/// The conversation's tool results, shortened one at a time when they no longer fit the context: the oldest still-whole
/// result becomes a note (its size and first 200 characters); when only the newest is left, it is halved (from 2,000
/// characters on).  The user's and the model's own messages are never touched.  (index, its characters before) or None.
pub fn compact_one(messages: &mut [Value]) -> Option<(usize, usize)> {
    let tools: Vec<usize> = messages.iter().enumerate().filter(|(_, m)| m["role"] == "tool").map(|(i, _)| i).collect();
    let newest = *tools.last()?;
    let text = |m: &Value| m["content"].as_str().unwrap_or("").to_string();
    if let Some(&i) = tools.iter().find(|&&i| i != newest && text(&messages[i]).chars().count() > 300 && !text(&messages[i]).starts_with("[compacted")) {
        let c = text(&messages[i]);
        let head: String = c.chars().take(200).collect();
        messages[i]["content"] = json!(format!("[compacted to fit the context: an earlier tool result of {} characters; it began: {head} ...]", c.chars().count()));
        return Some((i, c.chars().count()));
    }
    let c = text(&messages[newest]);
    let n = c.chars().count();
    if n < 2000 {
        return None;
    }
    let keep: String = c.chars().take(n / 2).collect();
    messages[newest]["content"] = json!(format!("{keep}\n[cut to fit the context: {} of {n} characters shown]", n / 2));
    Some((newest, n))
}

/// Service::run with the MCP tools executed here, up to max_rounds; one Done at the very end.
#[allow(clippy::too_many_arguments)]
pub fn run_with_mcp(svc: &Service, m: McpRun, mut ids: Vec<u32>, mut thinking: bool, tools: Option<Vec<Value>>, mut max_new: i64, sampling: &Map<String, Value>, cancel: &Arc<Flag>, emit: &mut dyn FnMut(Item) -> Result<(), Closed>) -> Result<(), Stop<Fail>> {
    let hub = svc.mcp.clone().expect("hub");
    let max_rounds = hub.settings["max_rounds"].as_i64().unwrap_or(8);
    let (mut total, mut rounds) = (0i64, 0i64);
    let mut messages = m.messages;
    // the round after the limit answers from what it has: its calls are not run
    let mut tools = tools;
    let mut last = false;
    // the pages the browser read or searched in this answer: listed under it (an answer from the web always says
    // where from, even when a small model forgets to)
    let mut sources: Vec<Source> = vec![];
    let question = messages.iter().rev().find(|m| m["role"] == "user").map(|m| crate::py::py_str(&m["content"])).unwrap_or_default();
    let mut answer_text = String::new();
    let mut done: Map<String, Value>;
    loop {
        let (mut text, mut reasoning, mut calls, mut own_calls) = (String::new(), String::new(), vec![], 0);
        let mut round_done = None;
        svc.run(&ids, thinking, tools.as_deref(), max_new, sampling, cancel, None, &mut |item| match item {
            Item::Done(d) => {
                round_done = Some(d);
                Ok(())
            }
            Item::Event(ev) => {
                if let Some(c) = ev.call.as_ref().filter(|c| m.names.contains(&c.name)) {
                    if ev.kind == Kind::ToolStart {
                        return emit(Item::Mcp(json!({"event": "start", "id": c.id, "name": c.name})));
                    }
                    if ev.kind == Kind::ToolCall {
                        calls.push(c.clone());
                    }
                    return Ok(());
                }
                match ev.kind {
                    Kind::ToolCall => own_calls += 1,
                    Kind::Content => text.push_str(&ev.text),
                    Kind::Reasoning => reasoning.push_str(&ev.text),
                    _ => {}
                }
                emit(Item::Event(ev))
            }
            other => emit(other),
        })?;
        done = round_done.unwrap_or_default();
        answer_text.push_str(&text);
        total += done.get("completion_tokens").and_then(Value::as_i64).unwrap_or(0);
        let mut run_them = !calls.is_empty() && own_calls == 0 && done.get("finish").and_then(Value::as_str) == Some("stop") && !cancel.is_set();
        if run_them && rounds >= max_rounds && !last && !cancel.is_set() {
            // the limit: these calls are not run; one more round, without tools, to answer from what it gathered
            emit(Item::Mcp(json!({"event": "limit", "max_rounds": max_rounds})))?;
            for c in &calls {
                emit(Item::Mcp(json!({"event": "result", "id": c.id, "ok": false, "skipped": true, "text": "not run", "chars": 0, "truncated": false, "ms": 0})))?;
            }
            let mut assistant = json!({"role": "assistant", "content": crate::py::strip(&text)}).as_object().cloned().unwrap();
            assistant.insert("tool_calls".into(), Value::Array(calls.iter().map(|c| json!({"function": {"name": c.name, "arguments": Value::Object(c.arguments.clone())}})).collect()));
            messages.push(Value::Object(assistant));
            messages.extend(calls.iter().map(|_| json!({"role": "tool", "content": "not run: the limit of tool calls for one answer is reached. Answer the user now from what you have found (say what is missing)."})));
            last = true;
            // the tools stay in the prompt: its start is unchanged, so the engine's cache holds (dropping them re-read
            // all 10K tokens, 50 s on the M1); a call in this round is not run
            let (i2, t2, m2) = svc.prepare(&messages, tools.as_deref(), &m.kw, Some(m.max_req), None, None).map_err(Stop::Err)?;
            ids = i2;
            thinking = t2;
            max_new = m2;
            continue;
        }
        if run_them && last {
            run_them = false;                             // no tools in the last round: any call there is not run
            if crate::py::strip(&text).is_empty() && tools.is_some() {
                // it called again instead of answering: without the tools, it can only answer (a reply is never empty)
                for c in &calls {
                    emit(Item::Mcp(json!({"event": "result", "id": c.id, "ok": false, "skipped": true, "text": "not run", "chars": 0, "truncated": false, "ms": 0})))?;
                }
                tools = None;
                let (mut i2, t2, m2) = svc.prepare(&messages, None, &m.kw, Some(m.max_req), None, None).map_err(Stop::Err)?;
                // its thinking closed for it: a template that always opens one (LFM's) let it think "let me try another
                // page" to the end, with no answer; after </think> what comes is the answer
                let close = t2 && !svc.harmony;
                if close {
                    i2.extend(svc.tok.encode(&format!("{}\n\n", crate::parser::THINK_END), true));
                }
                (ids, thinking, max_new) = (i2, t2 && !close, m2);
                continue;
            }
        }
        if !run_them {
            for c in &calls {
                emit(Item::Mcp(json!({"event": "result", "id": c.id, "ok": false, "skipped": true, "text": "not run", "chars": 0, "truncated": false, "ms": 0})))?;
            }
            break;
        }
        rounds += 1;
        let mut results = vec![];
        let routes = hub.routes();
        for c in &calls {
            let (server, tool) = routes.get(&c.name).map(|(s, t)| (json!(s.name), t.clone())).unwrap_or((Value::Null, c.name.clone()));
            emit(Item::Mcp(json!({"event": "call", "id": c.id, "name": c.name, "server": server, "tool": tool, "arguments": Value::Object(c.arguments.clone()), "round": rounds})))?;
            // the call runs on a thread while keep-alives go out (how a client that left is noticed during a slow tool)
            let (h, name, args, cn) = (hub.clone(), c.name.clone(), Value::Object(c.arguments.clone()), cancel.clone());
            let worker = std::thread::spawn(move || h.call(&name, &args, Some(&cn)));
            while !worker.is_finished() {
                std::thread::sleep(Duration::from_millis(50));
                let t = Instant::now();
                while !worker.is_finished() && t.elapsed() < Duration::from_secs(1) {
                    std::thread::sleep(Duration::from_millis(20));
                }
                if !worker.is_finished() && emit(Item::Ping).is_err() {
                    cancel.set();
                    return Err(Stop::Closed);
                }
            }
            let Ok(Ok(r)) = worker.join() else { break };
            let num = |k: &str| r.get(k).and_then(Value::as_f64).unwrap_or(0.0);
            say(&format!("tool {}: {}, {} characters in {:.1} s{} {}", c.name, if truthy(&r["ok"]) { "ok" } else { "error" }, crate::mcp::thousands(num("chars") as i64), num("ms") / 1000.0, if truthy(&r["truncated"]) { " (truncated for the model)" } else { "" }, args_brief(&c.arguments)));
            if !truthy(&r["ok"]) {
                say(&format!("tool {} said: {}", c.name, r["text"].as_str().unwrap_or("").chars().take(300).collect::<String>()));
            }
            results.push(r["text"].as_str().unwrap_or("").to_string());
            if server == json!("browser") && truthy(&r["ok"]) && matches!(tool.as_str(), "browser_read" | "browser_search" | "browser_open" | "browser_back") {
                let text = r["text"].as_str().unwrap_or("");
                if let Some((title, url)) = page_source(text) {
                    if !sources.iter().any(|x| x.1 == url) {
                        sources.push((title, url, text.to_string()));
                    }
                }
            }
            let mut ev = json!({"event": "result", "id": c.id}).as_object().cloned().unwrap();
            for k in ["ok", "text", "chars", "truncated", "ms"] {
                ev.insert(k.into(), r[k].clone());
            }
            emit(Item::Mcp(Value::Object(ev)))?;
        }
        if cancel.is_set() || results.len() < calls.len() {
            done.insert("finish".into(), json!("cancel"));
            break;
        }
        let mut assistant = json!({"role": "assistant", "content": crate::py::strip(&text)}).as_object().cloned().unwrap();
        if !reasoning.is_empty() {
            assistant.insert("reasoning_content".into(), json!(crate::py::strip(&reasoning)));
        }
        assistant.insert("tool_calls".into(), Value::Array(calls.iter().map(|c| json!({"function": {"name": c.name, "arguments": Value::Object(c.arguments.clone())}})).collect()));
        messages.push(Value::Object(assistant));
        // the budget on the last result (a model that can see what is left spends it better: Google's "Budget-Aware
        // Tool Use", arXiv 2511.17006); without it, a small model opened page after page into the limit
        let left = max_rounds - rounds;
        if let Some(r) = results.last_mut() {
            r.push_str(&format!("\n\n[tool calls left for this answer: {left} of {max_rounds}{}]",
                                if left <= 2 { ". Answer from what you have unless one more page is essential" } else { "" }));
        }
        messages.extend(results.into_iter().map(|r| json!({"role": "tool", "content": r})));
        let (i2, t2, m2) = loop {
            match svc.prepare(&messages, tools.as_deref(), &m.kw, Some(m.max_req), None, None) {
                Ok(r) => break r,
                Err(Fail::Value(e)) if e.contains("exceeds the context window") => {
                    // auto-compact: tool results fill the context (a browsing session: a page per round); the oldest
                    // one still whole gives way to a note, the newest (the one the model reads now) last, only halved
                    let Some((i, chars)) = compact_one(&mut messages) else { return Err(Stop::Err(Fail::Value(e))) };
                    say(&format!("auto-compact: an earlier tool result ({} characters) shortened to fit the context", crate::mcp::thousands(chars as i64)));
                    emit(Item::Mcp(json!({"event": "compacted", "message": i, "chars": chars})))?;
                }
                Err(e) => return Err(Stop::Err(e)),
            }
        };
        ids = i2;
        thinking = t2;
        max_new = m2;
    }
    if crate::py::strip(&answer_text).is_empty() && !sources.is_empty() && !cancel.is_set() {
        // still no answer: said, with what was read, instead of an empty reply
        let read: Vec<String> = sources.iter().take(6).map(|(t, u, _)| format!("- [{}]({u})", t.replace(['[', ']'], ""))).collect();
        emit(Item::Event(crate::parser::Event::text(Kind::Content, format!("I ran out of steps before I could answer. The pages I read:\n{}\n\nAsk again, more narrowly, and I will answer from them.", read.join("\n")))))?;
    } else if let Some(list) = sources_text(&sources, &answer_text, &question) {
        emit(Item::Event(crate::parser::Event::text(Kind::Content, list)))?;
    }
    done.insert("completion_tokens".into(), json!(total));
    done.insert("prompt_tokens".into(), json!(ids.len()));
    emit(Item::Done(done))?;
    Ok(())
}

pub type Out<'a> = &'a mut dyn FnMut(Option<Value>) -> Result<(), Closed>;

/// OpenAI Chat Completions chunks.  MCP items become chunks with an empty delta and a `strata_mcp` field.
pub fn openai_chunks(req: &Value, run: &mut Run, out: Out) -> Result<(), Stop<Fail>> {
    let svc = run.svc;
    let cid = format!("chatcmpl-{}", &uuid::Uuid::new_v4().simple().to_string()[..24]);
    let created = crate::sync::now() as i64;
    let model = svc.model_for(req);
    let want_progress = truthy(get(req, "return_progress"));
    let chunk = |delta: Value, finish: Value| json!({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
    out(Some(chunk(json!({"role": "assistant", "content": ""}), Value::Null)))?;
    let mut calls = 0i64;
    let mut streamed: HashMap<String, i64> = HashMap::new();
    let mut finished: HashSet<String> = HashSet::new();
    let n_ids = run.ids.len();
    run.go(&mut |item| match item {
        Item::Ping => {
            let pp = if want_progress { prompt_progress(svc) } else { None };
            match pp {
                None => out(None),
                Some(p) => {
                    let mut c = chunk(json!({}), Value::Null);
                    c["prompt_progress"] = p;
                    out(Some(c))
                }
            }
        }
        Item::Mcp(x) => {
            let mut c = chunk(json!({}), Value::Null);
            c["strata_mcp"] = x;
            out(Some(c))
        }
        Item::Event(ev) => match ev.kind {
            Kind::Reasoning if !ev.text.is_empty() => out(Some(chunk(json!({"reasoning_content": ev.text}), Value::Null))),
            Kind::Content if !ev.text.is_empty() => out(Some(chunk(json!({"content": ev.text}), Value::Null))),
            Kind::ToolStart => {
                let c = ev.call.unwrap();
                streamed.insert(c.id.clone(), calls);
                calls += 1;
                out(Some(chunk(json!({"tool_calls": [{"index": streamed[&c.id], "id": c.id, "type": "function", "function": {"name": c.name, "arguments": ""}}]}), Value::Null)))
            }
            Kind::ToolArgs => {
                let c = ev.call.unwrap();
                out(Some(chunk(json!({"tool_calls": [{"index": streamed.get(&c.id).copied().unwrap_or(0), "function": {"arguments": ev.text}}]}), Value::Null)))
            }
            Kind::ToolCall => {
                let c = ev.call.unwrap();
                if streamed.contains_key(&c.id) {
                    finished.insert(c.id);
                    return Ok(());
                }
                let r = out(Some(chunk(json!({"tool_calls": [{"index": calls, "id": c.id, "type": "function", "function": {"name": c.name, "arguments": pyjson::dumps_u(&Value::Object(c.arguments))}}]}), Value::Null)));
                calls += 1;
                r
            }
            _ => Ok(()),
        },
        Item::Done(x) => {
            let whole = calls > 0 && streamed.keys().all(|k| finished.contains(k));
            let f = x.get("finish").and_then(Value::as_str).unwrap_or("stop");
            let finish = if whole && f == "stop" { "tool_calls" } else if f == "cancel" { "stop" } else { f };
            let mut last = chunk(json!({}), json!(finish));
            let pt = x.get("prompt_tokens").and_then(Value::as_i64).unwrap_or(n_ids as i64);
            let ct = x.get("completion_tokens").and_then(Value::as_i64).unwrap_or(0);
            let cached = x.get("reused").filter(|v| truthy(v)).cloned().unwrap_or(json!(0));
            last["usage"] = json!({"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct, "prompt_tokens_details": {"cached_tokens": cached}});
            if let Some(t) = x.get("timings").filter(|t| truthy(t)) {
                last["timings"] = t.clone();
            }
            out(Some(last))
        }
    })
}

/// The whole answer from the chunks (non-streamed requests).
pub fn openai_collect(chunks: &[Value]) -> Value {
    let (mut content, mut reasoning) = (String::new(), String::new());
    let mut by_index: std::collections::BTreeMap<i64, Value> = Default::default();
    let mut mcp = vec![];
    let mut last = &Value::Null;
    for c in chunks {
        if let Some(m) = c.get("strata_mcp").filter(|m| truthy(m)) {
            mcp.push(m.clone());
        }
        let d = &c["choices"][0]["delta"];
        content += d.get("content").and_then(Value::as_str).unwrap_or("");
        reasoning += d.get("reasoning_content").and_then(Value::as_str).unwrap_or("");
        for tc in d.get("tool_calls").and_then(Value::as_array).into_iter().flatten() {
            let idx = tc.get("index").and_then(Value::as_i64).unwrap_or(by_index.len() as i64);
            let cur = by_index.entry(idx).or_insert_with(|| json!({"id": null, "type": "function", "function": {"name": "", "arguments": ""}}));
            if let Some(id) = tc.get("id").filter(|v| truthy(v)) {
                cur["id"] = id.clone();
            }
            let f = tc.get("function").cloned().unwrap_or(json!({}));
            let name = format!("{}{}", cur["function"]["name"].as_str().unwrap_or(""), f.get("name").and_then(Value::as_str).unwrap_or(""));
            let args = format!("{}{}", cur["function"]["arguments"].as_str().unwrap_or(""), f.get("arguments").and_then(Value::as_str).unwrap_or(""));
            cur["function"]["name"] = json!(name);
            cur["function"]["arguments"] = json!(args);
        }
        last = c;
    }
    let mut calls: Vec<Value> = by_index.into_values().collect();
    if last["choices"][0]["finish_reason"] != "tool_calls" {
        calls.retain(|c| serde_json::from_str::<Value>(c["function"]["arguments"].as_str().unwrap_or("")).is_ok());
    }
    let mut msg = json!({"role": "assistant", "content": if content.is_empty() { Value::Null } else { json!(content) }}).as_object().cloned().unwrap();
    if !reasoning.is_empty() {
        msg.insert("reasoning_content".into(), json!(reasoning));
    }
    if !calls.is_empty() {
        msg.insert("tool_calls".into(), Value::Array(calls));
    }
    if !mcp.is_empty() {
        msg.insert("strata_mcp".into(), Value::Array(mcp));
    }
    let mut out = json!({"id": last["id"], "object": "chat.completion", "created": last["created"], "model": last["model"],
        "choices": [{"index": 0, "message": Value::Object(msg), "finish_reason": last["choices"][0]["finish_reason"]}], "usage": last["usage"]});
    if let Some(t) = last.get("timings").filter(|t| truthy(t)) {
        out["timings"] = t.clone();
    }
    out
}

/// Structured streams are buffered so a client never receives unvalidated content.
pub fn structured_chunks(req: &Value, run: &mut Run, validator: &Validator, out: Out) -> Result<(), Stop<Fail>> {
    let mut buffered: Vec<Value> = vec![];
    let mut heartbeat = Instant::now();
    openai_chunks(req, run, &mut |c| {
        let progress = c.as_ref().map(|c| c.get("prompt_progress").is_some()).unwrap_or(false);
        if let Some(ch) = &c {
            if !progress {
                buffered.push(ch.clone());
            }
        }
        if c.is_none() || progress || heartbeat.elapsed() >= Duration::from_secs(1) {
            heartbeat = Instant::now();
            return out(if progress { c } else { None });
        }
        Ok(())
    })?;
    let result = openai_collect(&buffered);
    let choice = &result["choices"][0];
    let content = validated_json(choice["message"]["content"].as_str().unwrap_or(""), validator, choice["finish_reason"].as_str().unwrap_or("")).map_err(Stop::Err)?;
    out(Some(buffered[0].clone()))?;
    let mut delta = json!({"content": content});
    if let Some(r) = choice["message"].get("reasoning_content").filter(|r| truthy(r)) {
        delta["reasoning_content"] = r.clone();
    }
    let mut mid = buffered[0].clone();
    mid["choices"] = json!([{"index": 0, "delta": delta, "finish_reason": null}]);
    out(Some(mid))?;
    out(Some(buffered.last().unwrap().clone()))?;
    Ok(())
}

pub type AOut<'a> = &'a mut dyn FnMut(Option<(&'static str, Value)>) -> Result<(), Closed>;

/// Anthropic Messages events.
pub fn anthropic_events(req: &Value, run: &mut Run, out: AOut) -> Result<(), Stop<Fail>> {
    let svc = run.svc;
    let n_ids = run.ids.len() as i64;
    let mid = format!("msg_{}", &uuid::Uuid::new_v4().simple().to_string()[..24]);
    out(Some(("message_start", json!({"type": "message_start", "message": {"id": mid, "type": "message", "role": "assistant", "model": svc.model_for(req), "content": [], "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": n_ids, "output_tokens": 0}}}))))?;
    let mut index: i64 = -1;
    let mut open_kind: Option<&'static str> = None;
    let mut used_tool = false;
    let (mut streamed, mut finished): (HashSet<String>, HashSet<String>) = Default::default();
    run.go(&mut |item| {
        match item {
            Item::Ping => out(None),
            Item::Mcp(_) => Ok(()),
            Item::Event(ev) => {
                if ev.kind == Kind::ToolArgs {
                    return out(Some(("content_block_delta", json!({"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": ev.text}}))));
                }
                if ev.kind == Kind::ToolCall && ev.call.as_ref().map(|c| streamed.contains(&c.id)).unwrap_or(false) {
                    finished.insert(ev.call.unwrap().id);
                    return Ok(());
                }
                let want = match ev.kind {
                    Kind::Reasoning => "thinking",
                    Kind::Content => "text",
                    _ => "tool_use",
                };
                if !matches!(ev.kind, Kind::ToolCall | Kind::ToolStart) && ev.text.is_empty() {
                    return Ok(());
                }
                if ev.kind == Kind::ToolStart {
                    streamed.insert(ev.call.as_ref().unwrap().id.clone());
                    used_tool = true;
                }
                if open_kind != Some(want) || want == "tool_use" {
                    if open_kind.is_some() {
                        out(Some(("content_block_stop", json!({"type": "content_block_stop", "index": index}))))?;
                    }
                    index += 1;
                    open_kind = Some(want);
                    let block = match want {
                        "thinking" => json!({"type": "thinking", "thinking": "", "signature": ""}),
                        "text" => json!({"type": "text", "text": ""}),
                        _ => json!({"type": "tool_use", "id": ev.call.as_ref().map(|c| c.id.clone()).unwrap_or_default(), "name": ev.call.as_ref().map(|c| c.name.clone()).unwrap_or_default(), "input": {}}),
                    };
                    out(Some(("content_block_start", json!({"type": "content_block_start", "index": index, "content_block": block}))))?;
                }
                match want {
                    "thinking" => out(Some(("content_block_delta", json!({"type": "content_block_delta", "index": index, "delta": {"type": "thinking_delta", "thinking": ev.text}})))),
                    "text" => out(Some(("content_block_delta", json!({"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": ev.text}})))),
                    _ if ev.kind == Kind::ToolStart => Ok(()),
                    _ => {
                        used_tool = true;
                        out(Some(("content_block_delta", json!({"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": pyjson::dumps_u(&Value::Object(ev.call.unwrap().arguments))}}))))
                    }
                }
            }
            Item::Done(x) => {
                if open_kind.is_some() {
                    out(Some(("content_block_stop", json!({"type": "content_block_stop", "index": index}))))?;
                }
                let f = x.get("finish").and_then(Value::as_str).unwrap_or("stop");
                let ss = x.get("stop_sequence").cloned().unwrap_or(Value::Null);
                let stop = if !ss.is_null() {
                    "stop_sequence"
                } else if used_tool && streamed.iter().all(|s| finished.contains(s)) && f == "stop" {
                    "tool_use"
                } else {
                    match f {
                        "length" => "max_tokens",
                        _ => "end_turn",
                    }
                };
                let reused = x.get("reused").and_then(Value::as_i64).unwrap_or(0).min(n_ids - 1).max(0);
                out(Some(("message_delta", json!({"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": ss}, "usage": {"input_tokens": n_ids - reused, "cache_read_input_tokens": reused, "output_tokens": x.get("completion_tokens").cloned().unwrap_or(json!(0))}}))))?;
                out(Some(("message_stop", json!({"type": "message_stop"}))))
            }
        }
    })
}

/// The whole message from the events (non-streamed requests).
pub fn anthropic_collect(events: &[(&'static str, Value)]) -> Value {
    let mut msg = Value::Null;
    let mut blocks: Vec<Map<String, Value>> = vec![];
    for (name, e) in events {
        match *name {
            "message_start" => msg = e["message"].clone(),
            "content_block_start" => blocks.push(e["content_block"].as_object().cloned().unwrap_or_default()),
            "content_block_delta" => {
                let d = &e["delta"];
                let Some(b) = blocks.last_mut() else { continue };
                match d["type"].as_str() {
                    Some("text_delta") => {
                        let t = format!("{}{}", b["text"].as_str().unwrap_or(""), d["text"].as_str().unwrap_or(""));
                        b.insert("text".into(), json!(t));
                    }
                    Some("thinking_delta") => {
                        let t = format!("{}{}", b["thinking"].as_str().unwrap_or(""), d["thinking"].as_str().unwrap_or(""));
                        b.insert("thinking".into(), json!(t));
                    }
                    _ => {
                        let j = format!("{}{}", b.get("_json").and_then(Value::as_str).unwrap_or(""), d["partial_json"].as_str().unwrap_or(""));
                        b.insert("_json".into(), json!(j));
                    }
                }
            }
            "content_block_stop" => {
                if let Some(b) = blocks.last_mut() {
                    if let Some(j) = b.shift_remove("_json") {
                        let s = j.as_str().unwrap_or("");
                        match serde_json::from_str::<Value>(if s.is_empty() { "{}" } else { s }) {
                            Ok(v) => {
                                b.insert("input".into(), v);
                            }
                            Err(_) => {
                                blocks.pop();
                            }
                        }
                    }
                }
            }
            "message_delta" => {
                msg["stop_reason"] = e["delta"]["stop_reason"].clone();
                msg["stop_sequence"] = e["delta"].get("stop_sequence").cloned().unwrap_or(Value::Null);
                if let (Some(u), Some(eu)) = (msg["usage"].as_object_mut(), e["usage"].as_object()) {
                    for (k, v) in eu {
                        u.insert(k.clone(), v.clone());
                    }
                }
            }
            _ => {}
        }
    }
    msg["content"] = Value::Array(blocks.into_iter().map(Value::Object).collect());
    msg
}

#[cfg(test)]
mod compact_tests {
    use super::*;

    #[test]
    fn the_oldest_tool_result_goes_first_the_newest_is_only_halved() {
        let page = |n: usize| "x".repeat(n);
        let mut m = vec![json!({"role": "user", "content": page(5000)}), json!({"role": "tool", "content": page(5000)}),
                         json!({"role": "assistant", "content": "a"}), json!({"role": "tool", "content": page(4000)})];
        assert_eq!(compact_one(&mut m), Some((1, 5000)));
        assert!(m[1]["content"].as_str().unwrap().starts_with("[compacted to fit the context: an earlier tool result of 5000 characters"));
        assert_eq!(m[0]["content"].as_str().unwrap().len(), 5000);           // the user's message stays whole
        assert_eq!(compact_one(&mut m), Some((3, 4000)));                    // only the newest left: halved
        assert!(m[3]["content"].as_str().unwrap().ends_with("[cut to fit the context: 2000 of 4000 characters shown]"));
        let n = m[3]["content"].as_str().unwrap().chars().count();         // halved again on the next overflow
        assert_eq!(compact_one(&mut m), Some((3, n)));
        assert_eq!(compact_one(&mut vec![json!({"role": "user", "content": "hi"})]), None);
    }
}

#[cfg(test)]
mod sources_tests {
    use super::*;

    #[test]
    fn tool_arguments_in_the_log_are_cut() {
        let short = json!({"row": 3, "values": {"Price": "$1"}}).as_object().cloned().unwrap();
        assert_eq!(args_brief(&short), r#"{"row":3,"values":{"Price":"$1"}}"#);
        let long = json!({"rows": "x".repeat(1000)}).as_object().cloned().unwrap();
        let b = args_brief(&long);
        assert!(b.ends_with("...") && b.chars().count() == 303, "{}", b.len());
    }

    #[test]
    fn an_answer_from_the_web_lists_its_pages() {
        let read = "Paris - BBC Weather\nhttps://www.bbc.com/weather/2988507\n\nToday: sunny\n\nElements:\n(none)";
        let search = "[Read this first. Today is 2026-10-30: ...]\n\nparis weather - Google Search\nhttps://www.google.com/search?q=paris+weather\n\nresults";
        assert_eq!(page_source(read), Some(("Paris - BBC Weather".into(), "https://www.bbc.com/weather/2988507".into())));
        assert_eq!(page_source(search), Some(("paris weather".into(), "https://www.google.com/search?q=paris+weather".into())));
        assert_eq!(page_source("opened https://x: browser_read it"), None);
        let opened = "[Today is 2026-10-30. Answer only from what this page says, and link it as [title](address).]\n\nEvents | Reader\nhttps://r.test/events\n\nJazz";
        assert_eq!(page_source(opened), Some(("Events | Reader".into(), "https://r.test/events".into())));
        assert_eq!(page_source("[Today is x.]\n\nJust a moment\nhttps://cf.test/\n\n(no text on this page: a bot check)\n\nElements:\n(none)"), None);
        let src = |t: &str| { let (a, b) = page_source(t).unwrap(); (a, b, t.to_string()) };
        let read = "Paris - BBC Weather\nhttps://www.bbc.com/weather/2988507\n\nToday: sunny intervals and a gentle breeze, 18C\n\nElements:\n(none)";
        let search = "[Read this first. Today is 2026-10-30: ...]\n\nparis weather - Google Search\nhttps://www.google.com/search?q=paris+weather\n\nParis today: sunny intervals and a gentle breeze";
        let both = vec![src(search), src(read)];
        let q = "paris weather";
        let answer = "In Paris it is 18C with sunny intervals and a gentle breeze.";
        // a page was used: the search page is not listed
        assert_eq!(sources_text(&both, answer, q), Some("\n\n**Sources**\n- [Paris - BBC Weather](https://www.bbc.com/weather/2988507)".into()));
        // the answer linked it already: nothing added
        assert_eq!(sources_text(&both, &format!("{answer} ([BBC](https://www.bbc.com/weather/2988507))"), q), None);
        // only a search: its results page is the source
        assert_eq!(sources_text(&both[..1], answer, q), Some("\n\n**Sources**\n- [paris weather](https://www.google.com/search?q=paris+weather)".into()));
        // an answer that says nothing the pages say (from memory) cites none of them
        assert_eq!(sources_text(&both, "It rains a lot in autumn there.", q), None);
        assert_eq!(sources_text(&[], "x", ""), None);
    }

    #[test]
    fn error_pages_bot_checks_and_soft_404s_are_not_cited() {
        // the three pages a family-events answer listed (2026-10-30 chat): fetched, but none of the answer came from them
        let q = "find more family events in san diego today";
        let answer = "1. **Halloween Spooktacular** at the Fairmont Grand Del Mar, 5:00pm to 10:00pm: trick-or-treat stations, a live DJ and a photo station for families in San Diego today.";
        let page = |title: &str, url: &str, body: &str| (title.to_string(), url.to_string(), format!("[Today is 2026-10-30.]\n\n{title}\n{url}\n\n{body}"));
        let bad = vec![
            page("0 - Failed to parse time string (2026-5252-01 00:00:00) at position 6 (2): Unexpected character", "https://www.sandiegofamily.com/x", "0 - Failed to parse time string (2026-5252-01 00:00:00) at position 6 (2): Unexpected character"),
            page("Page not found | Coronado Public Library", "https://coronado.librarycalendar.com/events/2026/10", "Page not found. The page you requested could not be found. Search the library catalog. Family events in San Diego today"),
            page("Just a moment...", "https://sandiegomoms.com/calendar", "Performing security verification. This website uses a security service to protect against malicious bots."),
        ];
        let good = page("Halloween Spooktacular | San Diego", "https://www.sandiego.org/e/1", "Halloween Spooktacular. Fairmont Grand Del Mar. 5:00pm to 10:00pm. Trick-or-treat stations, a live DJ and a photo station.");
        let mut all = bad.clone();
        all.push(good);
        assert_eq!(sources_text(&bad, answer, q), None);
        assert_eq!(sources_text(&all, answer, q), Some("\n\n**Sources**\n- [Halloween Spooktacular | San Diego](https://www.sandiego.org/e/1)".into()));
        // the question's words on a page are not support ("family events in san diego today" is on the 404 page)
        assert!(!supports(&bad[1].2, "Family events in San Diego today: none found.", q));
    }
}

#[cfg(test)]
mod loop_tests {
    use super::*;
    use crate::engine::{Engine, MockEngine};
    use crate::tokenizer::ByteTokenizer;
    use std::io::{BufRead, BufReader, Read, Write};

    /// An MCP server over HTTP with one tool, `tool`, that answers with page N (as the browser's tools do).
    fn fake_mcp(tool: &'static str, reply: fn(usize) -> String) -> u16 {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let mut n = 0;
            for s in l.incoming().flatten() {
                let mut r = BufReader::new(s.try_clone().unwrap());
                let mut len = 0;
                loop {
                    let mut h = String::new();
                    if r.read_line(&mut h).unwrap_or(0) == 0 || h == "\r\n" {
                        break;
                    }
                    if let Some(v) = h.to_ascii_lowercase().strip_prefix("content-length:") {
                        len = v.trim().parse().unwrap_or(0);
                    }
                }
                let mut body = vec![0; len];
                let _ = r.read_exact(&mut body);
                let req: Value = serde_json::from_slice(&body).unwrap_or_default();
                let result = match req["method"].as_str().unwrap_or("") {
                    "initialize" => json!({"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "fake", "version": "1"}}),
                    "tools/list" => json!({"tools": [{"name": tool, "description": "read the page", "inputSchema": {"type": "object", "properties": {}}}]}),
                    "tools/call" => {
                        n += 1;
                        json!({"content": [{"type": "text", "text": reply(n)}]})
                    }
                    _ => Value::Null,                        // a notification: 202, no body (RFC 9110 15.3.3)
                };
                let mut s = s;
                if req.get("id").is_none() {
                    let _ = write!(s, "HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                    continue;
                }
                let out = json!({"jsonrpc": "2.0", "id": req["id"], "result": result}).to_string();
                let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{out}", out.len());
            }
        });
        port
    }

    /// The loop with max_rounds 2 and the model's scripted turns; what it emitted and every prompt the engine got.
    fn run(scripts: &[&str]) -> (Vec<Value>, String, Vec<Vec<u32>>) {
        run_on("web", "read", page, scripts)
    }

    /// Page n, as the browser's tools answer.
    fn page(n: usize) -> String {
        format!("[Today is 2026-10-30.]\n\nPage {n}\nhttps://p.test/{n}\n\nThe Soto Six jazz quartet plays venue {n} at 8 PM")
    }

    fn run_on(server: &str, tool: &'static str, reply: fn(usize) -> String, scripts: &[&str]) -> (Vec<Value>, String, Vec<Vec<u32>>) {
        let tok = Arc::new(ByteTokenizer);
        let mock = Arc::new(MockEngine::new(&*tok, &scripts.iter().map(|s| s.to_string()).collect::<Vec<_>>(), 32768, 0.0));
        let tpl = crate::template::ChatTemplate::load(&crate::root().join("server/chat_template.jinja")).unwrap();
        let mut svc = Service::new(Engine::Mock(mock.clone()), tok, tpl, "m".into(), None, Default::default());
        let servers = indexmap::IndexMap::from([(server.to_string(), json!({"url": format!("http://127.0.0.1:{}/mcp", fake_mcp(tool, reply))}))]);
        let hub = Arc::new(crate::mcp::McpHub::new(servers, json!({"max_rounds": 2}).as_object().cloned().unwrap(), &[]));
        hub.start();
        hub.wait(Some(10.0));
        let tools = hub.template_tools(&[]);
        assert_eq!(tools.len(), 1, "the fake server's tool: {tools:?}");
        let names: HashSet<String> = tools.iter().filter_map(|t| get(t, "name").as_str().map(String::from)).collect();
        svc.mcp = Some(hub);
        let messages = vec![json!({"role": "user", "content": "find events"})];
        let kw = Map::new();
        let (ids, thinking, max_new) = svc.prepare(&messages, Some(&tools), &kw, Some(200), None, None).unwrap();
        let m = McpRun { messages, kw, max_req: 200, names };
        let (mut mcp, mut text) = (vec![], String::new());
        run_with_mcp(&svc, m, ids, thinking, Some(tools), max_new, &Map::new(), &Arc::new(Flag::new()), &mut |item| {
            match item {
                Item::Mcp(v) => mcp.push(v),
                Item::Event(e) if e.kind == Kind::Content => text.push_str(&e.text),
                _ => {}
            }
            Ok(())
        }).map_err(|_| "the loop failed").unwrap();
        let prompts = mock.prompts.lock().unwrap().clone();
        (mcp, text, prompts)
    }

    // the default template opens a thinking block: each turn closes it first
    const CALL: &str = "</think>\n\n<tool_call>\n<function=web__read>\n</function>\n</tool_call>";

    fn not_run(mcp: &[Value]) -> usize {
        mcp.iter().filter(|v| v["event"] == "result" && v["skipped"] == true).count()
    }

    #[test]
    fn at_the_limit_it_answers_with_the_tools_still_in_the_prompt() {
        let (mcp, text, prompts) = run(&[CALL, CALL, CALL, "</think>\n\nTwo pages read."]);
        assert_eq!(mcp.iter().filter(|v| v["event"] == "call").count(), 2, "max_rounds calls run: {mcp:?} {text:?}");
        assert_eq!(mcp.iter().filter(|v| v["event"] == "limit").count(), 1);
        assert_eq!(not_run(&mcp), 1, "the third call is not run");
        assert!(text.contains("Two pages read."), "{text}");
        assert_eq!(prompts.len(), 4);
        // the last round's prompt extends the round before it: the engine's cache holds (the tools were not dropped)
        assert!(prompts[3].starts_with(&prompts[2]), "the final round re-reads the prompt from the start");
    }

    #[test]
    fn a_call_again_instead_of_an_answer_gets_a_round_without_tools() {
        use crate::tokenizer::Tokenize;
        // the last turn has no </think>: the loop closed the thinking for it, so its text is the answer
        let (mcp, text, prompts) = run(&[CALL, CALL, CALL, CALL, "From page 1 and 2."]);
        assert!(ByteTokenizer.decode(&prompts[4]).ends_with("</think>\n\n"), "the thinking is closed in the last prompt");
        assert_eq!(not_run(&mcp), 2, "the limit's call and the final round's call");
        assert!(text.contains("From page 1 and 2."), "a reply is never empty: {text:?}");
        assert_eq!(prompts.len(), 5);
        assert!(!prompts[4].starts_with(&prompts[3]), "the tools are gone from the last prompt");
    }

    #[test]
    fn pages_the_browser_opened_are_listed_under_the_answer() {
        let (_, text, _) = run_on("browser", "browser_open", page, &[OPEN, OPEN, "</think>\n\nThe Soto Six jazz quartet plays tonight."]);
        assert!(text.ends_with("**Sources**\n- [Page 1](https://p.test/1)\n- [Page 2](https://p.test/2)"), "{text:?}");
    }

    const OPEN: &str = "</think>\n\n<tool_call>\n<function=browser__browser_open>\n<parameter=url>\nhttps://p.test/1\n</parameter>\n</function>\n</tool_call>";

    #[test]
    fn a_page_gone_back_to_is_a_source_too() {
        let back = "</think>\n\n<tool_call>\n<function=browser__browser_back>\n</function>\n</tool_call>";
        let (_, text, _) = run_on("browser", "browser_back", page, &[back, "</think>\n\nThe Soto Six jazz quartet plays tonight."]);
        assert!(text.ends_with("**Sources**\n- [Page 1](https://p.test/1)"), "{text:?}");
    }

    #[test]
    fn the_same_page_twice_is_listed_once() {
        let (_, text, _) = run_on("browser", "browser_open", |_| page(7), &[OPEN, OPEN, "</think>\n\nThe Soto Six jazz quartet plays tonight."]);
        assert_eq!(text.matches("https://p.test/7").count(), 1, "{text:?}");
    }

    #[test]
    fn blank_refused_and_linked_pages_add_no_list() {
        // a bot check, then a guessed address held back: nothing came from either
        let (_, text, _) = run_on("browser", "browser_open", |n| if n == 1 {
            "[Today is x.]\n\nJust a moment\nhttps://cf.test/\n\n(no text on this page: a bot check, or still loading.)\n\nElements:\n(none)".into()
        } else {
            "not opened: guess.test is a site none of the results or pages you read showed".into()
        }, &[OPEN, OPEN, "</think>\n\nI could not read a page."]);
        assert!(!text.contains("Sources"), "{text:?}");
        // the answer linked the page itself: no list under it
        let (_, text, _) = run_on("browser", "browser_open", page, &[OPEN, "</think>\n\nJazz ([Page 1](https://p.test/1))."]);
        assert!(!text.contains("Sources"), "{text:?}");
    }

    #[test]
    fn other_servers_results_are_not_sources() {
        // a tool of another server that answers like a page: only the browser's pages are cited
        let call = "</think>\n\n<tool_call>\n<function=web__browser_open>\n</function>\n</tool_call>";
        let (_, text, _) = run_on("web", "browser_open", page, &[call, "</think>\n\nThe Soto Six jazz quartet plays tonight."]);
        assert!(!text.contains("Sources"), "{text:?}");
    }

    #[test]
    fn the_model_sees_its_tool_budget() {
        use crate::tokenizer::Tokenize;
        let (_, _, prompts) = run(&[CALL, CALL, "</think>\n\nDone."]);
        let seen = |i: usize| ByteTokenizer.decode(&prompts[i]);
        assert!(seen(1).contains("[tool calls left for this answer: 1 of 2. Answer from what you have unless one more page is essential]"), "{}", seen(1));
        assert!(seen(2).contains("[tool calls left for this answer: 0 of 2."));
        assert!(!seen(0).contains("tool calls left"));
    }

    #[test]
    fn no_answer_at_all_says_so_with_the_pages_read() {
        let open = "</think>\n\n<tool_call>\n<function=browser__browser_open>\n<parameter=url>\nhttps://p.test/1\n</parameter>\n</function>\n</tool_call>";
        // calls to the end, and an empty last turn
        let (_, text, _) = run_on("browser", "browser_open", page, &[open, open, open, open, ""]);
        assert!(text.starts_with("I ran out of steps before I could answer. The pages I read:\n- [Page 1](https://p.test/1)\n- [Page 2](https://p.test/2)"), "{text:?}");
        // nothing read: no note (the reply is the model's, as it is)
        let (_, text, _) = run(&[CALL, CALL, CALL, CALL, ""]);
        assert!(!text.contains("ran out of steps"), "{text:?}");
    }
}
