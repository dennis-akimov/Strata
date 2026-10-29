//! #451: OpenAI's Responses API (POST /v1/responses), stateless, on the chat path (serve/responses.py): the request
//! becomes the same template messages, tools and kwargs; Service::run's events come back as output items and typed
//! SSE events in the official order.  Codex CLI is the first client.

use crate::frontend::{effort_kwargs, parts_of};
use crate::parser::{Event, Kind};
use crate::py::{self, get, truthy};
use crate::pyjson;
use base64::Engine as _;
use serde_json::{json, Map, Value};
use std::sync::Mutex;

pub const ENCRYPTED_PREFIX: &str = "strata.r1:";

/// A request error in the Responses format: {"error": {"message", "type", "param", "code"}}.
#[derive(Debug, Clone)]
pub struct RespErr {
    pub message: String,
    pub param: Option<String>,
    pub code: Option<String>,
    pub status: u16,
    pub kind: String,
}

impl RespErr {
    pub fn new(message: impl Into<String>, param: Option<&str>, code: Option<&str>) -> Self {
        RespErr { message: message.into(), param: param.map(String::from), code: code.map(String::from), status: 400, kind: "invalid_request_error".into() }
    }
    pub fn body(&self) -> Value {
        error_body(&self.message, &self.kind, self.param.as_deref(), self.code.as_deref())
    }
}

pub fn error_body(message: &str, kind: &str, param: Option<&str>, code: Option<&str>) -> Value {
    json!({"error": {"message": message, "type": kind, "param": param, "code": code}})
}

pub fn new_id(prefix: &str) -> String {
    format!("{prefix}_{}", uuid::Uuid::new_v4().simple())
}

fn content(c: &Value, param: &str) -> Result<Value, RespErr> {
    match c {
        Value::Null => Ok(json!("")),
        Value::String(_) => Ok(c.clone()),
        Value::Array(a) => {
            let mut parts = vec![];
            for (j, part) in a.iter().enumerate() {
                if let Value::String(s) = part {
                    parts.push(json!({"type": "text", "text": s}));
                    continue;
                }
                if !part.is_object() {
                    return Err(RespErr::new("expected a content part object", Some(&format!("{param}[{j}]")), None));
                }
                let kind = get(part, "type");
                match kind.as_str() {
                    Some("input_text" | "output_text" | "text" | "summary_text" | "reasoning_text") => parts.push(json!({"type": "text", "text": if truthy(get(part, "text")) { get(part, "text").clone() } else { json!("") }})),
                    Some("refusal") => parts.push(json!({"type": "text", "text": if truthy(get(part, "refusal")) { get(part, "refusal").clone() } else { json!("") }})),
                    Some("input_image" | "image_url") => {
                        if !truthy(get(part, "image_url")) {
                            return Err(RespErr::new("only images given as image_url (a data: or http(s) URL) are supported; this server keeps no files", Some(&format!("{param}[{j}]")), Some("unsupported_parameter")));
                        }
                        parts.push(json!({"type": "input_image", "image_url": part["image_url"]}));
                    }
                    _ => return Err(RespErr::new(format!("content parts of type {} are not supported (text and images are)", py::py_repr(kind)), Some(&format!("{param}[{j}].type")), Some("unsupported_parameter"))),
                }
            }
            Ok(parts_of(&Value::Array(parts)))
        }
        _ => Err(RespErr::new("expected a string or an array of content parts", Some(param), None)),
    }
}

fn reasoning_text(item: &Value, param: &str) -> Result<String, RespErr> {
    let texts: Vec<String> = get(item, "content").as_array().into_iter().flatten().filter(|p| p.is_object() && matches!(get(p, "type").as_str(), Some("reasoning_text" | "text"))).map(|p| get(p, "text").as_str().unwrap_or("").to_string()).collect();
    if !texts.is_empty() {
        return Ok(texts.concat());
    }
    if let Some(enc) = get(item, "encrypted_content").as_str().and_then(|e| e.strip_prefix(ENCRYPTED_PREFIX)) {
        let bad = || RespErr::new("this reasoning item's encrypted_content is damaged", Some(&format!("{param}.encrypted_content")), Some("invalid_encrypted_content"));
        let raw = base64::engine::general_purpose::STANDARD.decode(enc).map_err(|_| bad())?;
        let v: Value = serde_json::from_slice(&raw).map_err(|_| bad())?;
        return v.get("text").and_then(Value::as_str).map(String::from).ok_or_else(bad);
    }
    Ok(String::new())
}

fn arguments(raw: &Value, param: &str) -> Result<Value, RespErr> {
    match raw {
        Value::Object(_) => Ok(raw.clone()),
        Value::Null => Ok(json!({})),
        Value::String(s) if py::strip(s).is_empty() => Ok(json!({})),
        Value::String(s) => Ok(match serde_json::from_str::<Value>(s) {
            Ok(v @ Value::Object(_)) => v,
            Ok(v) => json!({"arguments": v}),
            Err(_) => json!({"arguments": s}),
        }),
        _ => Err(RespErr::new("function_call arguments must be a JSON string", Some(param), None)),
    }
}

fn tool_output(item: &Value, param: &str) -> Result<Value, RespErr> {
    let out = get(item, "output");
    match out {
        Value::Object(_) => Ok(json!(pyjson::dumps_u(out))),
        Value::String(_) => Ok(out.clone()),
        _ => content(out, &format!("{param}.output")),
    }
}

/// `instructions` + `input` -> the chat template's messages.
pub fn input_messages(req: &Value) -> Result<Vec<Value>, RespErr> {
    let mut messages: Vec<Map<String, Value>> = vec![];
    let instructions = get(req, "instructions");
    if !instructions.is_null() && instructions.as_str() != Some("") {
        let c = if instructions.is_string() { instructions.clone() } else { content(instructions, "instructions")? };
        messages.push(json!({"role": "system", "content": c}).as_object().cloned().unwrap());
    }
    let items = match get(req, "input") {
        Value::Null => return Err(RespErr::new("input is required: a string or an array of input items", Some("input"), Some("missing_required_parameter"))),
        Value::String(s) => vec![json!({"type": "message", "role": "user", "content": s})],
        Value::Array(a) => a.clone(),
        _ => return Err(RespErr::new("input must be a string or an array of input items", Some("input"), None)),
    };
    let mut thinking: Vec<String> = vec![];
    let mut open_turn: Option<usize> = None;
    let turn = |messages: &mut Vec<Map<String, Value>>, thinking: &mut Vec<String>, c: Value| -> usize {
        let mut t = json!({"role": "assistant", "content": c}).as_object().cloned().unwrap();
        if !thinking.is_empty() {
            t.insert("reasoning_content".into(), json!(thinking.concat()));
            thinking.clear();
        }
        messages.push(t);
        messages.len() - 1
    };
    for (i, item) in items.iter().enumerate() {
        let param = format!("input[{i}]");
        if !item.is_object() {
            return Err(RespErr::new("expected an input item object", Some(&param), None));
        }
        let kind = if truthy(get(item, "type")) { get(item, "type").as_str().map(String::from) } else if item.get("role").is_some() { Some("message".into()) } else { None };
        match kind.as_deref() {
            Some("message") => {
                let role = get(item, "role");
                if !matches!(role.as_str(), Some("user" | "assistant" | "system" | "developer")) {
                    return Err(RespErr::new(format!("unknown message role {}", py::py_repr(role)), Some(&format!("{param}.role")), None));
                }
                let c = content(get(item, "content"), &format!("{param}.content"))?;
                if role == "assistant" {
                    let text = match &c {
                        Value::String(_) => c.clone(),
                        Value::Array(a) => json!(a.iter().filter(|p| get(p, "type") == "text").map(|p| get(p, "text").as_str().unwrap_or("").to_string()).collect::<String>()),
                        _ => c.clone(),
                    };
                    open_turn = Some(turn(&mut messages, &mut thinking, text));
                } else {
                    open_turn = None;
                    thinking.clear();
                    messages.push(json!({"role": if role == "developer" { "system" } else { role.as_str().unwrap() }, "content": c}).as_object().cloned().unwrap());
                }
            }
            Some("reasoning") => {
                let text = reasoning_text(item, &param)?;
                if !text.is_empty() {
                    if let Some(t) = open_turn {
                        if truthy(messages[t].get("tool_calls").unwrap_or(&Value::Null)) || truthy(&messages[t]["content"]) {
                            open_turn = None;
                        }
                    }
                    thinking.push(text);
                }
            }
            Some(k @ ("function_call" | "custom_tool_call")) => {
                let name = get(item, "name").as_str().filter(|n| !n.is_empty()).ok_or_else(|| RespErr::new("a function_call needs a name", Some(&format!("{param}.name")), None))?;
                let name = if truthy(get(item, "namespace")) { format!("{}.{name}", py::py_str(get(item, "namespace"))) } else { name.to_string() };
                let args = if k == "function_call" { arguments(get(item, "arguments"), &format!("{param}.arguments"))? } else { json!({"input": get(item, "input").as_str().unwrap_or("")}) };
                let target = match open_turn {
                    Some(t) if thinking.is_empty() => t,
                    _ => {
                        let t = turn(&mut messages, &mut thinking, json!(""));
                        open_turn = Some(t);
                        t
                    }
                };
                let calls = messages[target].entry("tool_calls").or_insert_with(|| json!([]));
                calls.as_array_mut().unwrap().push(json!({"function": {"name": name, "arguments": args}, "_call_id": get(item, "call_id")}));
            }
            Some("function_call_output" | "custom_tool_call_output") => {
                messages.push(json!({"role": "tool", "content": tool_output(item, &param)?, "_call_id": get(item, "call_id")}).as_object().cloned().unwrap());
                open_turn = None;
                thinking.clear();
            }
            Some("additional_tools") => continue,
            Some("item_reference") => return Err(RespErr::new("item references need stored responses, and this server keeps none: send the items themselves", Some(&param), Some("unsupported_parameter"))),
            _ => return Err(RespErr::new(format!("input items of type {} are not supported", py::py_repr(&kind.map(Value::from).unwrap_or(Value::Null))), Some(&format!("{param}.type")), Some("unsupported_parameter"))),
        }
    }
    order_tool_results(&mut messages);
    let mut lead = 0;
    while lead < messages.len() && messages[lead]["role"] == "system" && messages[lead]["content"].is_string() {
        lead += 1;
    }
    if lead > 1 {
        let joined = messages[..lead].iter().map(|m| m["content"].as_str().unwrap().to_string()).collect::<Vec<_>>().join("\n\n");
        messages.splice(..lead, [json!({"role": "system", "content": joined}).as_object().cloned().unwrap()]);
    }
    let messages: Vec<Value> = messages.into_iter().map(Value::Object).collect();
    Ok(messages
        .into_iter()
        .enumerate()
        .map(|(i, mut m)| {
            if i > 0 && m["role"] == "system" {
                m["role"] = json!("user");
            }
            m
        })
        .collect())
}

fn order_tool_results(messages: &mut [Map<String, Value>]) {
    let mut i = 0;
    while i < messages.len() {
        if messages[i]["role"] != "tool" {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < messages.len() && messages[j]["role"] == "tool" {
            j += 1;
        }
        let order: Vec<Value> = if i > 0 { messages[i - 1].get("tool_calls").and_then(Value::as_array).map(|a| a.iter().map(|c| get(c, "_call_id").clone()).collect()).unwrap_or_default() } else { vec![] };
        let rank = |m: &Map<String, Value>| -> usize {
            let cid = m.get("_call_id").cloned().unwrap_or(Value::Null);
            if !truthy(&cid) {
                return order.len();
            }
            order.iter().position(|o| *o == cid && truthy(o)).unwrap_or(order.len())
        };
        let mut run: Vec<Map<String, Value>> = messages[i..j].to_vec();
        run.sort_by_key(|m| rank(m));
        for (k, m) in run.into_iter().enumerate() {
            messages[i + k] = m;
        }
        i = j;
    }
    for m in messages.iter_mut() {
        m.shift_remove("_call_id");
        if let Some(calls) = m.get_mut("tool_calls").and_then(Value::as_array_mut) {
            for c in calls {
                if let Some(o) = c.as_object_mut() {
                    o.shift_remove("_call_id");
                }
            }
        }
    }
}

pub type Names = indexmap::IndexMap<String, (Option<String>, String, String)>;

/// -> (template tools or None, {template name: (namespace, name, kind)}, names of hosted tools left out).
pub fn request_tools(req: &Value) -> Result<(Option<Vec<Value>>, Names, Vec<String>), RespErr> {
    let given = get(req, "tools");
    let mut given: Vec<Value> = match given {
        Value::Null => vec![],
        Value::Array(a) => a.clone(),
        v if !truthy(v) => vec![],
        _ => return Err(RespErr::new("tools must be an array", Some("tools"), None)),
    };
    for item in get(req, "input").as_array().into_iter().flatten() {
        if get(item, "type") == "additional_tools" {
            if let Some(t) = get(item, "tools").as_array() {
                given.extend(t.iter().cloned());
            }
        }
    }
    let (mut tools, mut names, mut skipped): (Vec<Value>, Names, Vec<String>) = (vec![], Names::new(), vec![]);
    let mut add = |tool: &Value, param: &str, namespace: Option<String>, ns_desc: &str| -> Result<(), RespErr> {
        let kind = get(tool, "type").as_str().unwrap_or("").to_string();
        let name = get(tool, "name").as_str().filter(|n| !n.is_empty()).ok_or_else(|| RespErr::new("a tool needs a name", Some(&format!("{param}.name")), None))?.to_string();
        let flat = match &namespace {
            Some(ns) if !ns.is_empty() => format!("{ns}.{name}"),
            _ => name.clone(),
        };
        let mut description = if truthy(get(tool, "description")) { py::py_str(get(tool, "description")) } else { String::new() };
        if !ns_desc.is_empty() {
            description = format!("{ns_desc}{}", if description.is_empty() { String::new() } else { format!("\n\n{description}") });
        }
        let params = if kind == "function" {
            if truthy(get(tool, "parameters")) { tool["parameters"].clone() } else { json!({"type": "object", "properties": {}}) }
        } else {
            let fmt = get(tool, "format");
            if get(fmt, "type") == "grammar" && truthy(get(fmt, "definition")) {
                description += &format!("\n\nThe input must follow this {} grammar:\n{}", fmt.get("syntax").map(py::py_str).unwrap_or_default(), py::py_str(&fmt["definition"]));
            }
            json!({"type": "object", "properties": {"input": {"type": "string", "description": "the raw input"}}, "required": ["input"]})
        };
        tools.push(json!({"name": flat, "description": description, "parameters": params}));
        let ns = namespace.filter(|n| !n.is_empty());
        names.insert(flat.clone(), (ns.clone(), name.clone(), kind.clone()));
        if let Some(ns) = &ns {
            names.entry(format!("{ns}__{name}")).or_insert((Some(ns.clone()), name.clone(), kind.clone()));
        }
        Ok(())
    };
    for (i, tool) in given.iter().enumerate() {
        let param = format!("tools[{i}]");
        if !tool.is_object() {
            return Err(RespErr::new("expected a tool object", Some(&param), None));
        }
        match get(tool, "type").as_str() {
            Some("function" | "custom") => add(tool, &param, None, "")?,
            Some("namespace") => {
                for (j, member) in get(tool, "tools").as_array().into_iter().flatten().enumerate() {
                    let mt = member.get("type").and_then(Value::as_str).unwrap_or("function");
                    if !member.is_object() || !(mt == "function" || mt == "custom") {
                        return Err(RespErr::new("a namespace holds function tools", Some(&format!("{param}.tools[{j}]")), None));
                    }
                    let mut m = json!({"type": "function"}).as_object().cloned().unwrap();
                    for (k, v) in member.as_object().unwrap() {
                        m.insert(k.clone(), v.clone());
                    }
                    let ns = get(tool, "name").as_str().map(String::from);
                    let nd = if truthy(get(tool, "description")) { py::py_str(get(tool, "description")) } else { String::new() };
                    add(&Value::Object(m), &format!("{param}.tools[{j}]"), ns, &nd)?;
                }
            }
            Some(k) => skipped.push(k.to_string()),
            None => return Err(RespErr::new("a tool needs a type", Some(&format!("{param}.type")), None)),
        }
    }
    if get(req, "tool_choice") == "none" {
        return Ok((None, names, skipped));
    }
    Ok((if tools.is_empty() { None } else { Some(tools) }, names, skipped))
}

fn turn_body(req: &Value) -> Option<Value> {
    let raw = get(get(req, "client_metadata"), "x-codex-turn-metadata").as_str()?;
    serde_json::from_str::<Value>(raw).ok().filter(Value::is_object)
}

fn title_text(req: &Value) -> String {
    let items = match get(req, "input") {
        Value::String(s) => vec![json!({"role": "user", "content": s})],
        Value::Array(a) => a.clone(),
        _ => vec![],
    };
    let mut parts = vec![];
    for item in &items {
        if get(item, "role") != "user" {
            continue;
        }
        match get(item, "content") {
            Value::String(s) => parts.push(s.clone()),
            Value::Array(a) => parts.extend(a.iter().filter_map(|p| get(p, "text").as_str().map(String::from))),
            _ => {}
        }
    }
    let mut text = parts.join("\n");
    if let Some((_, rest)) = text.split_once("User prompt:") {
        text = rest.to_string();
    }
    let mut line = text.lines().map(py::strip).find(|x| !x.is_empty()).unwrap_or("Untitled").to_string();
    line = line.trim_matches([' ', '"', '\'', '`']).to_string();
    while line.chars().last().map(|c| ".。!！?？,，;；:：".contains(c)).unwrap_or(false) {
        line.pop();
        line = py::rstrip(&line).to_string();
    }
    let cut = py::rstrip(py::chars_prefix(&line, 36)).to_string();
    if cut.is_empty() { "Untitled".into() } else { cut }
}

/// Codex's thread-title turn, answered here; None when this request is not one.
pub fn thread_title_events(req: &Value, model: &str) -> Result<Option<Vec<Value>>, RespErr> {
    if turn_body(req).map(|t| t.get("thread_source").cloned()).flatten().as_ref().and_then(Value::as_str) != Some("thread_title") {
        return Ok(None);
    }
    check_request(req)?;
    let text = pyjson::dumps_u(&json!({"title": title_text(req)}));
    let mut asm = Assembler::new(req, model, 0, Names::new(), false, false);
    let mut events = asm.start();
    events.extend(asm.open(json!({"id": new_id("msg"), "type": "message", "status": "in_progress", "role": "assistant", "content": []})));
    let idx = asm.index();
    asm.output_mut(idx)["content"][0]["text"] = json!(text);
    let id = asm.output(idx)["id"].clone();
    events.push(asm.event("response.output_text.delta", vec![("item_id", id), ("output_index", json!(idx)), ("content_index", json!(0)), ("delta", json!(text)), ("logprobs", json!([]))]));
    events.extend(asm.close("completed"));
    asm.response.insert("usage".into(), json!({"input_tokens": 0, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 0, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 0}));
    asm.response.insert("status".into(), json!("completed"));
    asm.response.insert("completed_at".into(), json!(crate::sync::now() as i64));
    let snap = asm.snapshot();
    events.push(asm.event("response.completed", vec![("response", snap)]));
    Ok(Some(events))
}

/// #924: the tools a Codex conversation's last prompt was rendered with, for its compaction request.
static KEPT_PROMPT_TOOLS: Mutex<Option<((String, String), Option<Value>)>> = Mutex::new(None);

fn codex_turn(req: &Value) -> (Option<String>, Option<(String, String)>) {
    let Some(turn) = turn_body(req) else { return (None, None) };
    let kind = turn.get("request_kind").and_then(Value::as_str).map(String::from);
    let s = turn.get("session_id").and_then(Value::as_str).filter(|x| !x.is_empty());
    let t = turn.get("thread_id").and_then(Value::as_str).filter(|x| !x.is_empty());
    (kind, s.zip(t).map(|(a, b)| (a.to_string(), b.to_string())))
}

pub fn prompt_tools(req: &Value, tools: Option<Vec<Value>>) -> Option<Vec<Value>> {
    let (kind, conv) = codex_turn(req);
    let kept = KEPT_PROMPT_TOOLS.lock().unwrap().clone();
    match (kind.as_deref(), conv, kept) {
        (Some("compaction"), Some(c), Some((owner, Some(k)))) if owner == c && !truthy(get(req, "tools")) => k.as_array().cloned(),
        _ => tools,
    }
}

pub fn prompt_made(req: &Value, tools: Option<&[Value]>) {
    if turn_body(req).and_then(|t| t.get("thread_source").cloned()).as_ref().and_then(Value::as_str) == Some("thread_title") {
        return;
    }
    let (kind, conv) = codex_turn(req);
    if let Some(c) = conv {
        if kind.as_deref() != Some("compaction") {
            *KEPT_PROMPT_TOOLS.lock().unwrap() = Some((c, tools.filter(|t| !t.is_empty()).map(|t| Value::Array(t.to_vec()))));
        }
    }
}

/// `text.format` -> the chat path's response_format (Null: plain text).
pub fn text_format(req: &Value) -> Result<Value, RespErr> {
    let text = get(req, "text");
    if text.is_null() {
        return Ok(Value::Null);
    }
    if !text.is_object() {
        return Err(RespErr::new("text must be an object", Some("text"), None));
    }
    let fmt = get(text, "format");
    if fmt.is_null() {
        return Ok(Value::Null);
    }
    if !fmt.is_object() {
        return Err(RespErr::new("text.format must be an object", Some("text.format"), None));
    }
    let t = get(fmt, "type");
    if t.is_null() || t == "text" {
        return Ok(Value::Null);
    }
    match t.as_str() {
        Some("json_object") => Ok(json!({"type": "json_object"})),
        Some("json_schema") => {
            let spec: Map<String, Value> = ["name", "schema", "strict", "description"].iter().filter_map(|k| fmt.get(*k).map(|v| (k.to_string(), v.clone()))).collect();
            Ok(json!({"type": "json_schema", "json_schema": spec}))
        }
        _ => Err(RespErr::new("text.format.type must be text, json_object or json_schema", Some("text.format.type"), None)),
    }
}

/// What this stateless server cannot do, refused before anything runs.
pub fn check_request(req: &Value) -> Result<(), RespErr> {
    let un = |m: &str, p: &str| Err(RespErr::new(m, Some(p), Some("unsupported_parameter")));
    if truthy(get(req, "previous_response_id")) {
        return un("this server keeps no responses (stateless): send the whole conversation in input instead of previous_response_id", "previous_response_id");
    }
    if truthy(get(req, "conversation")) {
        return un("this server keeps no conversations (stateless): send the whole conversation in input", "conversation");
    }
    if truthy(get(req, "background")) {
        return un("background responses are not supported: use stream or wait for the answer", "background");
    }
    if truthy(get(req, "prompt")) {
        return un("stored prompt templates are not supported: send instructions and input", "prompt");
    }
    let n = get(req, "max_output_tokens");
    if !n.is_null() && !(n.is_i64() && n.as_i64().unwrap() >= 1) {
        return Err(RespErr::new("max_output_tokens must be a positive whole number", Some("max_output_tokens"), None));
    }
    let r = get(req, "reasoning");
    if !r.is_null() && !r.is_object() {
        return Err(RespErr::new("reasoning must be an object", Some("reasoning"), None));
    }
    Ok(())
}

pub fn template_kwargs(req: &Value, shared: &Map<String, Value>) -> Result<Map<String, Value>, RespErr> {
    let mut effort = get(get(req, "reasoning"), "effort").clone();
    if effort.is_null() {
        effort = shared.get("reasoning_effort").cloned().unwrap_or(Value::Null);
    }
    effort_kwargs(&effort).map_err(|e| RespErr::new(e.0, Some("reasoning.effort"), None))
}

/// One response: its output items and typed events, built from Service::run's events.
pub struct Assembler {
    names: Names,
    encrypted: bool,
    json_mode: bool,
    prompt_tokens: i64,
    seq: i64,
    item: Option<usize>,
    json_text: Vec<String>,
    called: bool,
    pub response: Map<String, Value>,
}

impl Assembler {
    pub fn new(req: &Value, model: &str, prompt_tokens: i64, names: Names, encrypted: bool, json_mode: bool) -> Self {
        let reasoning = get(req, "reasoning");
        let response = json!({"id": new_id("resp"), "object": "response", "created_at": crate::sync::now() as i64, "status": "in_progress",
            "background": false, "error": null, "incomplete_details": null,
            "instructions": get(req, "instructions"), "max_output_tokens": get(req, "max_output_tokens"),
            "model": model, "output": [], "parallel_tool_calls": req.get("parallel_tool_calls").cloned().unwrap_or(json!(true)),
            "previous_response_id": null,
            "reasoning": {"effort": get(reasoning, "effort"), "summary": get(reasoning, "summary")},
            "store": false, "temperature": get(req, "temperature"),
            "text": if get(req, "text").is_object() { req["text"].clone() } else { json!({"format": {"type": "text"}}) },
            "tool_choice": req.get("tool_choice").cloned().unwrap_or(json!("auto")), "tools": if truthy(get(req, "tools")) { req["tools"].clone() } else { json!([]) },
            "top_p": get(req, "top_p"), "truncation": "disabled", "usage": null, "user": null,
            "metadata": if truthy(get(req, "metadata")) { req["metadata"].clone() } else { json!({}) }});
        Assembler { names, encrypted, json_mode, prompt_tokens, seq: 0, item: None, json_text: vec![], called: false, response: response.as_object().cloned().unwrap() }
    }

    pub fn event(&mut self, kind: &str, data: Vec<(&str, Value)>) -> Value {
        let mut e = json!({"type": kind, "sequence_number": self.seq}).as_object().cloned().unwrap();
        for (k, v) in data {
            e.insert(k.into(), v);
        }
        self.seq += 1;
        Value::Object(e)
    }

    pub fn snapshot(&self) -> Value {
        Value::Object(self.response.clone())
    }

    fn outputs(&mut self) -> &mut Vec<Value> {
        self.response.get_mut("output").unwrap().as_array_mut().unwrap()
    }

    fn output(&self, i: usize) -> &Value {
        &self.response["output"][i]
    }

    fn output_mut(&mut self, i: usize) -> &mut Value {
        &mut self.outputs()[i]
    }

    pub fn index(&self) -> usize {
        self.response["output"].as_array().map(|a| a.len()).unwrap_or(0).saturating_sub(1)
    }

    pub fn start(&mut self) -> Vec<Value> {
        let s1 = self.snapshot();
        let a = self.event("response.created", vec![("response", s1)]);
        let s2 = self.snapshot();
        let b = self.event("response.in_progress", vec![("response", s2)]);
        vec![a, b]
    }

    pub fn in_progress(&mut self) -> Value {
        let s = self.snapshot();
        self.event("response.in_progress", vec![("response", s)])
    }

    fn open(&mut self, item: Value) -> Vec<Value> {
        let message = item["type"] == "message";
        let id = item["id"].clone();
        self.outputs().push(item.clone());
        let idx = self.index();
        self.item = Some(idx);
        let mut out = vec![self.event("response.output_item.added", vec![("output_index", json!(idx)), ("item", item)])];
        if message {
            let part = json!({"type": "output_text", "text": "", "annotations": [], "logprobs": []});
            self.output_mut(idx)["content"].as_array_mut().unwrap().push(part.clone());
            out.push(self.event("response.content_part.added", vec![("item_id", id), ("output_index", json!(idx)), ("content_index", json!(0)), ("part", part)]));
        }
        out
    }

    pub fn close(&mut self, status: &str) -> Vec<Value> {
        let Some(at) = self.item.take() else { return vec![] };
        self.output_mut(at)["status"] = json!(status);
        let item = self.output(at).clone();
        let mut out = vec![];
        let id = item["id"].clone();
        match item["type"].as_str() {
            Some("message") => {
                let part = item["content"][0].clone();
                out.push(self.event("response.output_text.done", vec![("item_id", id.clone()), ("output_index", json!(at)), ("content_index", json!(0)), ("text", part["text"].clone()), ("logprobs", json!([]))]));
                out.push(self.event("response.content_part.done", vec![("item_id", id), ("output_index", json!(at)), ("content_index", json!(0)), ("part", part)]));
            }
            Some("reasoning") => {
                let text = item["content"][0]["text"].clone();
                out.push(self.event("response.reasoning_text.done", vec![("item_id", id), ("output_index", json!(at)), ("content_index", json!(0)), ("text", text.clone())]));
                if self.encrypted {
                    let enc = format!("{ENCRYPTED_PREFIX}{}", base64::engine::general_purpose::STANDARD.encode(pyjson::dumps_u(&json!({"text": text})).as_bytes()));
                    self.output_mut(at)["encrypted_content"] = json!(enc);
                }
            }
            Some("function_call") => out.push(self.event("response.function_call_arguments.done", vec![("item_id", id), ("output_index", json!(at)), ("name", item["name"].clone()), ("arguments", item["arguments"].clone())])),
            Some("custom_tool_call") => out.push(self.event("response.custom_tool_call_input.done", vec![("item_id", id), ("output_index", json!(at)), ("input", item["input"].clone())])),
            _ => {}
        }
        let item = self.output(at).clone();
        out.push(self.event("response.output_item.done", vec![("output_index", json!(at)), ("item", item)]));
        out
    }

    fn item_type(&self) -> Option<String> {
        self.item.map(|i| self.output(i)["type"].as_str().unwrap_or("").to_string())
    }

    pub fn feed(&mut self, ev: &Event) -> Vec<Value> {
        let mut out = vec![];
        match ev.kind {
            Kind::Reasoning => {
                if ev.text.is_empty() {
                    return out;
                }
                if self.item_type().as_deref() != Some("reasoning") {
                    out.extend(self.close("completed"));
                    out.extend(self.open(json!({"id": new_id("rs"), "type": "reasoning", "status": "in_progress", "summary": [], "content": [{"type": "reasoning_text", "text": ""}]})));
                }
                let i = self.item.unwrap();
                let t = format!("{}{}", self.output(i)["content"][0]["text"].as_str().unwrap_or(""), ev.text);
                self.output_mut(i)["content"][0]["text"] = json!(t);
                let id = self.output(i)["id"].clone();
                out.push(self.event("response.reasoning_text.delta", vec![("item_id", id), ("output_index", json!(i)), ("content_index", json!(0)), ("delta", json!(ev.text))]));
            }
            Kind::Content => {
                if ev.text.is_empty() {
                    return out;
                }
                if self.json_mode {
                    out.extend(self.close("completed"));
                    self.json_text.push(ev.text.clone());
                    return out;
                }
                if self.item_type().as_deref() != Some("message") {
                    out.extend(self.close("completed"));
                    out.extend(self.open(json!({"id": new_id("msg"), "type": "message", "status": "in_progress", "role": "assistant", "content": []})));
                }
                let i = self.item.unwrap();
                let t = format!("{}{}", self.output(i)["content"][0]["text"].as_str().unwrap_or(""), ev.text);
                self.output_mut(i)["content"][0]["text"] = json!(t);
                let id = self.output(i)["id"].clone();
                out.push(self.event("response.output_text.delta", vec![("item_id", id), ("output_index", json!(i)), ("content_index", json!(0)), ("delta", json!(ev.text)), ("logprobs", json!([]))]));
            }
            Kind::ToolStart => {
                out.extend(self.close("completed"));
                out.extend(self.call_started());
                let it = self.call_item(ev);
                out.extend(self.open(it));
            }
            Kind::ToolArgs => {
                let Some(i) = self.item else { return out };
                let c = ev.call.as_ref().unwrap();
                if self.output(i)["call_id"] != c.id.as_str() {
                    return out;
                }
                if self.output(i)["type"] == "function_call" {
                    let a = format!("{}{}", self.output(i)["arguments"].as_str().unwrap_or(""), ev.text);
                    self.output_mut(i)["arguments"] = json!(a);
                    let id = self.output(i)["id"].clone();
                    out.push(self.event("response.function_call_arguments.delta", vec![("item_id", id), ("output_index", json!(i)), ("delta", json!(ev.text))]));
                }
            }
            Kind::ToolCall => {
                let c = ev.call.as_ref().unwrap();
                let matches = self.item.map(|i| self.output(i)["call_id"] == c.id.as_str()).unwrap_or(false);
                if !matches {
                    out.extend(self.close("completed"));
                    out.extend(self.call_started());
                    let it = self.call_item(ev);
                    out.extend(self.open(it));
                    let i = self.item.unwrap();
                    if self.output(i)["type"] == "function_call" {
                        let a = pyjson::dumps_u(&Value::Object(c.arguments.clone()));
                        self.output_mut(i)["arguments"] = json!(a);
                        let id = self.output(i)["id"].clone();
                        out.push(self.event("response.function_call_arguments.delta", vec![("item_id", id), ("output_index", json!(i)), ("delta", json!(a))]));
                    }
                }
                let i = self.item.unwrap();
                if self.output(i)["type"] == "custom_tool_call" {
                    let input = match c.arguments.get("input") {
                        Some(Value::String(s)) => s.clone(),
                        _ => pyjson::dumps_u(&Value::Object(c.arguments.clone())),
                    };
                    self.output_mut(i)["input"] = json!(input);
                    let id = self.output(i)["id"].clone();
                    out.push(self.event("response.custom_tool_call_input.delta", vec![("item_id", id), ("output_index", json!(i)), ("delta", json!(input))]));
                }
                out.extend(self.close("completed"));
            }
        }
        out
    }

    fn release(&mut self, text: &str) -> Vec<Value> {
        if py::strip(text).is_empty() {
            return vec![];
        }
        let mut out = self.open(json!({"id": new_id("msg"), "type": "message", "status": "in_progress", "role": "assistant", "content": []}));
        let i = self.item.unwrap();
        self.output_mut(i)["content"][0]["text"] = json!(text);
        let id = self.output(i)["id"].clone();
        out.push(self.event("response.output_text.delta", vec![("item_id", id), ("output_index", json!(i)), ("content_index", json!(0)), ("delta", json!(text)), ("logprobs", json!([]))]));
        out.extend(self.close("completed"));
        out
    }

    fn call_started(&mut self) -> Vec<Value> {
        let mut out = vec![];
        if self.json_mode && !self.called {
            let t = self.json_text.concat();
            out = self.release(&t);
            self.json_text.clear();
        }
        self.called = true;
        out
    }

    fn call_item(&self, ev: &Event) -> Value {
        let c = ev.call.as_ref().unwrap();
        let (namespace, name, kind) = self.names.get(&c.name).cloned().unwrap_or((None, c.name.clone(), "function".into()));
        let mut item = if kind == "custom" {
            json!({"id": new_id("ctc"), "type": "custom_tool_call", "status": "in_progress", "call_id": c.id, "name": name, "input": ""})
        } else {
            json!({"id": new_id("fc"), "type": "function_call", "status": "in_progress", "call_id": c.id, "name": name, "arguments": ""})
        };
        if let Some(ns) = namespace {
            item["namespace"] = json!(ns);
        }
        item
    }

    /// The end: close what is open, then response.completed (or .incomplete when max_output_tokens ran out).
    pub fn finish(&mut self, done: &Map<String, Value>, validate: &dyn Fn(&str, &str) -> Result<String, crate::errors::Fail>) -> Result<Vec<Value>, crate::errors::Fail> {
        let length = done.get("finish").and_then(Value::as_str) == Some("length");
        let unfinished = matches!(self.item_type().as_deref(), Some("function_call" | "custom_tool_call"));
        let mut out = self.close(if length || unfinished { "incomplete" } else { "completed" });
        if self.json_mode {
            let text = self.json_text.concat();
            self.json_text.clear();
            if self.called {
                out.extend(self.release(&text));
            } else {
                let v = validate(&text, done.get("finish").and_then(Value::as_str).unwrap_or(""))?;
                out.extend(self.release(&v));
            }
        }
        let n = done.get("completion_tokens").and_then(Value::as_i64).unwrap_or(0);
        let prompt = done.get("prompt_tokens").and_then(Value::as_i64).unwrap_or(self.prompt_tokens);
        let cached = done.get("reused").and_then(Value::as_i64).unwrap_or(0).min(prompt);
        self.response.insert("usage".into(), json!({"input_tokens": prompt, "input_tokens_details": {"cached_tokens": cached}, "output_tokens": n, "output_tokens_details": {"reasoning_tokens": done.get("reasoning_tokens").cloned().unwrap_or(json!(0))}, "total_tokens": prompt + n}));
        if length {
            self.response.insert("status".into(), json!("incomplete"));
            self.response.insert("incomplete_details".into(), json!({"reason": "max_output_tokens"}));
            let s = self.snapshot();
            out.push(self.event("response.incomplete", vec![("response", s)]));
            return Ok(out);
        }
        self.response.insert("status".into(), json!("completed"));
        self.response.insert("completed_at".into(), json!(crate::sync::now() as i64));
        let s = self.snapshot();
        out.push(self.event("response.completed", vec![("response", s)]));
        Ok(out)
    }

    /// response.failed: an error after the stream started.
    pub fn failed(&mut self, message: &str, code: &str) -> Value {
        if let Some(i) = self.item.take() {
            self.output_mut(i)["status"] = json!("incomplete");
        }
        self.response.insert("status".into(), json!("failed"));
        self.response.insert("error".into(), json!({"code": code, "message": message}));
        let s = self.snapshot();
        self.event("response.failed", vec![("response", s)])
    }
}

/// The non-streaming answer: the final response object.
pub fn collect(events: &[Value]) -> Value {
    events.last().map(|e| e["response"].clone()).unwrap_or(Value::Null)
}
