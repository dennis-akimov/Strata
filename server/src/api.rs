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

/// Service::run with the MCP tools executed here, up to max_rounds; one Done at the very end.
#[allow(clippy::too_many_arguments)]
pub fn run_with_mcp(svc: &Service, m: McpRun, mut ids: Vec<u32>, mut thinking: bool, tools: Option<Vec<Value>>, mut max_new: i64, sampling: &Map<String, Value>, cancel: &Arc<Flag>, emit: &mut dyn FnMut(Item) -> Result<(), Closed>) -> Result<(), Stop<Fail>> {
    let hub = svc.mcp.clone().expect("hub");
    let max_rounds = hub.settings["max_rounds"].as_i64().unwrap_or(8);
    let (mut total, mut rounds) = (0i64, 0i64);
    let mut messages = m.messages;
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
        total += done.get("completion_tokens").and_then(Value::as_i64).unwrap_or(0);
        let mut run_them = !calls.is_empty() && own_calls == 0 && done.get("finish").and_then(Value::as_str) == Some("stop") && !cancel.is_set();
        if run_them && rounds >= max_rounds {
            emit(Item::Mcp(json!({"event": "limit", "max_rounds": max_rounds})))?;
            run_them = false;
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
            say(&format!("tool {}: {}, {} characters in {:.1} s{}", c.name, if truthy(&r["ok"]) { "ok" } else { "error" }, crate::mcp::thousands(num("chars") as i64), num("ms") / 1000.0, if truthy(&r["truncated"]) { " (truncated for the model)" } else { "" }));
            results.push(r["text"].as_str().unwrap_or("").to_string());
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
        messages.extend(results.into_iter().map(|r| json!({"role": "tool", "content": r})));
        let (i2, t2, m2) = svc.prepare(&messages, tools.as_deref(), &m.kw, Some(m.max_req), None, None).map_err(Stop::Err)?;
        ids = i2;
        thinking = t2;
        max_new = m2;
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
