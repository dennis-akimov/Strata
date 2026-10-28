//! Requests -> the chat template's messages, tools and kwargs (serve/frontend.py's request half): OpenAI Chat
//! Completions and Anthropic Messages, literal tags in message text kept as text (#537 #554), stable tool order.

use crate::parser::tool_arguments;
use crate::py::{self, get, truthy};
use crate::pyjson;
use indexmap::IndexMap;
use serde_json::{json, Map, Value};
use std::sync::Mutex;

/// A request the server answers with a 400 naming the problem (Python's ValueError).
#[derive(Debug, Clone, thiserror::Error)]
#[error("{0}")]
pub struct BadRequest(pub String);

pub fn bad<T>(msg: impl Into<String>) -> Result<T, BadRequest> {
    Err(BadRequest(msg.into()))
}

/// The content-part types read as text (a part with no type that has a "text" field reads as text too).
const TEXT_PARTS: [Option<&str>; 3] = [Some("text"), Some("input_text"), None];
pub const IMAGE_PARTS: [&str; 3] = ["image_url", "input_image", "image"];

fn part_type(p: &Value) -> Option<&str> {
    get(p, "type").as_str()
}

fn is_text_part(p: &Value) -> bool {
    // Python: part.get("type") in TEXT_PARTS - None (no type, or a null one) counts
    let t = get(p, "type");
    match t {
        Value::Null => true,
        Value::String(s) => TEXT_PARTS.contains(&Some(s.as_str())),
        _ => false,
    }
}

fn is_image_part(p: &Value) -> bool {
    part_type(p).map(|t| IMAGE_PARTS.contains(&t)).unwrap_or(false)
}

/// `str(d.get(key, default))`: the default only for a missing key (a null one is "None", as Python formats it).
fn get_or(d: &Value, key: &str, default: &str) -> String {
    match d.as_object().and_then(|o| o.get(key)) {
        None => default.to_string(),
        Some(v) => py::py_str(v),
    }
}

pub fn text_of(content: &Value) -> String {
    match content {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Array(a) => a
            .iter()
            .filter(|p| p.is_object() && is_text_part(p))
            .map(|p| get_or(p, "text", ""))
            .collect(),
        _ => String::new(),
    }
}

/// Thinking levels: the template knows low, medium and xhigh, and enable_thinking=false for none.
fn effort_level(key: &str) -> Option<Option<&'static str>> {
    Some(match key {
        "none" | "off" | "minimal" | "disabled" | "false" => None,
        "low" => Some("low"),
        "medium" => Some("medium"),
        "high" | "xhigh" | "max" | "maximum" => Some("xhigh"),
        _ => return None,
    })
}

/// A reasoning effort as given by a client -> the template's kwargs.  Unknown values are a 400.
pub fn effort_kwargs(value: &Value) -> Result<Map<String, Value>, BadRequest> {
    match value {
        Value::Null => return Ok(Map::new()),
        Value::String(s) if s.is_empty() => return Ok(Map::new()),
        Value::Bool(false) => return Ok(Map::from_iter([("enable_thinking".into(), json!(false))])),
        _ => {}
    }
    let key = py::strip(&py::py_str(value)).to_lowercase();
    match effort_level(&key) {
        None => bad(format!("unknown reasoning effort {}: use none, low, medium or high", py::py_repr(value))),
        Some(None) => Ok(Map::from_iter([("enable_thinking".into(), json!(false))])),
        Some(Some(level)) => Ok(Map::from_iter([("reasoning_effort".into(), json!(level))])),
    }
}

/// Anthropic's budget_tokens -> a level: under 2K low, under 8K medium, else high.
pub fn budget_effort(tokens: &Value) -> Map<String, Value> {
    let n: Option<i64> = match tokens {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f.trunc() as i64)),
        Value::Bool(b) => Some(*b as i64),
        Value::String(s) => py::strip(s).parse().ok(),
        _ => None,
    };
    match n {
        None => Map::new(),
        Some(n) => Map::from_iter([("reasoning_effort".into(), json!(if n < 2048 { "low" } else if n < 8192 { "medium" } else { "xhigh" }))]),
    }
}

pub fn has_image(content: &Value) -> bool {
    content.as_array().map(|a| a.iter().any(|p| p.is_object() && is_image_part(p))).unwrap_or(false)
}

fn image_source(part: &Value) -> String {
    if part_type(part) == Some("image") {
        let src = get(part, "source");
        if get(src, "type").as_str() == Some("base64") {
            let mt = get_or(src, "media_type", "image/png");
            return format!("data:{mt};base64,{}", get_or(src, "data", ""));
        }
        for k in ["url", "path"] {
            if truthy(get(src, k)) {
                return py::py_str(get(src, k));
            }
        }
        return String::new();
    }
    let mut url = get(part, "image_url");
    if url.is_object() {
        url = get(url, "url");
    }
    if truthy(url) { py::py_str(url) } else { String::new() }
}

/// Message content for the template: a string without images, else the list of text and image items.
pub fn parts_of(content: &Value) -> Value {
    if !has_image(content) {
        return Value::String(text_of(content));
    }
    let mut items = vec![];
    for part in content.as_array().into_iter().flatten() {
        if !part.is_object() {
            continue;
        }
        if is_image_part(part) {
            items.push(json!({"type": "image", "source": image_source(part)}));
        } else if is_text_part(part) && part.as_object().map(|o| o.contains_key("text")).unwrap_or(false) {
            items.push(json!({"type": "text", "text": get(part, "text").clone()}));
        }
    }
    Value::Array(items)
}

/// A refusal part reads as its text; parts of other unread types stay out of the prompt (with a debug line).
fn check_parts(content: &Value) -> Value {
    let Some(a) = content.as_array() else { return content.clone() };
    Value::Array(
        a.iter()
            .map(|p| {
                if part_type(p) == Some("refusal") {
                    let r = get(p, "refusal");
                    json!({"type": "text", "text": if truthy(r) { r.clone() } else { json!("") }})
                } else {
                    p.clone()
                }
            })
            .collect(),
    )
}

/// The image sources of the rendered conversation, in prompt order.
pub fn images_of(messages: &[Value]) -> Vec<String> {
    let mut out = vec![];
    for m in messages {
        if let Some(items) = get(m, "content").as_array() {
            for it in items {
                if get(it, "type").as_str() == Some("image") {
                    out.push(py::py_str(get(it, "source")));
                }
            }
        }
    }
    out
}

// #537 #554: literal tags inside message text are swapped for private-use marks before the template runs, and
// encoded as ordinary text afterwards.
pub const LITERAL_TAGS: [(&str, char); 6] = [
    ("<think>", '\u{F0E01}'),
    ("</think>", '\u{F0E02}'),
    ("<|vision_start|>", '\u{F0E03}'),
    ("<|image_pad|>", '\u{F0E04}'),
    ("<|vision_end|>", '\u{F0E05}'),
    ("<|video_pad|>", '\u{F0E06}'),
];
const CONTROL_MARK0: u32 = 0xF0E10;

/// The tags to mark, longest first (stable): LITERAL_TAGS plus each control token's text.
pub type Tags = Vec<(String, char)>;

pub fn default_tags() -> Tags {
    LITERAL_TAGS.iter().map(|(t, m)| (t.to_string(), *m)).collect()
}

pub fn literal_tags(controls: &[String]) -> Tags {
    let mut tags = default_tags();
    let mut k = 0;
    for c in controls {
        if LITERAL_TAGS.iter().any(|(t, _)| t == c) {
            continue;
        }
        // a duplicate control text keeps its first mark (Python's dict keeps the first key's slot, the last value)
        if let Some(e) = tags.iter_mut().find(|(t, _)| t == c) {
            e.1 = char::from_u32(CONTROL_MARK0 + k).expect("mark");
        } else {
            tags.push((c.clone(), char::from_u32(CONTROL_MARK0 + k).expect("mark")));
        }
        k += 1;
    }
    tags.sort_by_key(|(t, _)| std::cmp::Reverse(t.chars().count()));
    tags
}

fn mark(text: &str, tags: &Tags) -> String {
    let mut s = text.to_string();
    for (tag, m) in tags {
        if s.contains(tag.as_str()) {
            s = s.replace(tag.as_str(), &m.to_string());
        }
    }
    s
}

fn mark_deep(v: &Value, tags: &Tags) -> Value {
    match v {
        Value::String(s) => Value::String(mark(s, tags)),
        Value::Object(o) => Value::Object(o.iter().map(|(k, x)| (k.clone(), mark_deep(x, tags))).collect()),
        Value::Array(a) => Value::Array(a.iter().map(|x| mark_deep(x, tags)).collect()),
        x => x.clone(),
    }
}

fn has_tag(v: &Value, tags: &Tags) -> bool {
    match v {
        Value::String(s) => tags.iter().any(|(t, _)| s.contains(t.as_str())),
        Value::Object(o) => o.values().any(|x| has_tag(x, tags)),
        Value::Array(a) => a.iter().any(|x| has_tag(x, tags)),
        _ => false,
    }
}

/// #537: (messages, tools) with every literal tag in their text swapped for its mark, and whether there was one.
/// An assistant message that opens with a whole <think>...</think> block keeps that block's own two tags.
pub fn mark_think_literals(messages: &[Value], tools: Option<&Value>, tags: &Tags) -> (Vec<Value>, Option<Value>, bool) {
    let msgs = Value::Array(messages.to_vec());
    if !has_tag(&msgs, tags) && !tools.map(|t| has_tag(t, tags)).unwrap_or(false) {
        return (messages.to_vec(), tools.cloned(), false);
    }
    let mut out = vec![];
    for m in messages {
        let Some(o) = m.as_object() else {
            out.push(m.clone());
            continue;
        };
        let content = o.get("content").cloned().unwrap_or(Value::Null);
        let mut n: Map<String, Value> = o.iter().map(|(k, v)| (k.clone(), if k == "role" { v.clone() } else { mark_deep(v, tags) })).collect();
        if get(m, "role").as_str() == Some("assistant") {
            if let Value::String(c) = &content {
                if py::lstrip(c).starts_with("<think>") && c.contains("</think>") {
                    let i = c.find("<think>").expect("think") + "<think>".len();
                    let j = c.find("</think>").expect("end");
                    // Python slices content[i:j] even when j < i (an empty middle then)
                    let mid = if j >= i { &c[i..j] } else { "" };
                    let tail = &c[j + "</think>".len()..];
                    n.insert("content".into(), Value::String(format!("{}{}</think>{}", &c[..i], mark(mid, tags), mark(tail, tags))));
                }
            }
        }
        out.push(Value::Object(n));
    }
    (out, tools.map(|t| mark_deep(t, tags)), true)
}

/// The rendered prompt with the marks turned back into the tags' text, and the byte spans of those tags in it.
pub fn unmark_think_literals(prompt: &str, tags: &Tags) -> (String, Vec<(usize, usize)>) {
    let mut out = String::with_capacity(prompt.len());
    let mut spans = vec![];
    for c in prompt.chars() {
        if let Some((tag, _)) = tags.iter().find(|(_, m)| *m == c) {
            let start = out.len();
            out.push_str(tag);
            spans.push((start, out.len()));
        } else {
            out.push(c);
        }
    }
    (out, spans)
}

/// System messages after the first become user messages, in place.
fn late_system_to_user(messages: Vec<Value>) -> Vec<Value> {
    messages
        .into_iter()
        .enumerate()
        .map(|(i, mut m)| {
            if i > 0 && get(&m, "role").as_str() == Some("system") {
                m.as_object_mut().expect("object").insert("role".into(), json!("user"));
            }
            m
        })
        .collect()
}

/// #460: "messages" (or "tool_calls") as a list of objects; a double-encoded JSON string is decoded.
pub fn object_list(value: &Value, name: &str) -> Result<Vec<Value>, BadRequest> {
    let decoded;
    let v = match value {
        Value::Null => return Ok(vec![]),
        Value::String(s) => {
            decoded = serde_json::from_str::<Value>(s).map_err(|_| BadRequest(format!("{name} must be a list of objects (a string was sent that is not JSON)")))?;
            &decoded
        }
        v => v,
    };
    match v.as_array() {
        Some(a) if a.iter().all(Value::is_object) => Ok(a.clone()),
        _ => bad(format!("{name} must be a list of objects")),
    }
}

/// a set of tool names -> the order it first came in (opencode issue 23571)
static TOOL_ORDERS: Mutex<Option<IndexMap<Vec<String>, Vec<String>>>> = Mutex::new(None);
const TOOL_ORDERS_MAX: usize = 64;

fn stable_tool_order(tools: Vec<Value>, name_of: impl Fn(&Value) -> String) -> Vec<Value> {
    if tools.len() < 2 {
        return tools;
    }
    let names: Vec<String> = tools.iter().map(&name_of).collect();
    let mut key = names.clone();
    key.sort();
    key.dedup();
    if key.len() != names.len() {
        return tools;
    }
    let order = {
        let mut g = TOOL_ORDERS.lock().unwrap();
        let map = g.get_or_insert_with(IndexMap::new);
        match map.shift_remove(&key) {
            None => {
                map.insert(key, names);
                while map.len() > TOOL_ORDERS_MAX {
                    map.shift_remove_index(0);
                }
                return tools;
            }
            Some(order) => {
                map.insert(key, order.clone());
                order
            }
        }
    };
    let mut tools = tools;
    tools.sort_by_key(|t| order.iter().position(|n| *n == name_of(t)).unwrap_or(usize::MAX));
    tools
}

/// #592: "tools" as a list of named tool objects, OpenAI-shaped (`wrapper` "function") or Anthropic-shaped.
fn tool_list(value: &Value, wrapper: Option<&str>) -> Result<Vec<Value>, BadRequest> {
    if !truthy(value) {
        return Ok(vec![]);
    }
    let shape = if wrapper.is_some() {
        r#"{"type": "function", "function": {"name": ..., "parameters": {...}}}"#
    } else {
        r#"{"name": ..., "input_schema": {...}}"#
    };
    let tools = object_list(value, "tools").map_err(|_| BadRequest(format!("tools must be a list of tool objects, each {shape}")))?;
    let key = if wrapper.is_some() { "parameters" } else { "input_schema" };
    let unwrap = |t: &Value| -> Value {
        match wrapper {
            Some(w) if get(t, "type").as_str() == Some(w) => t.as_object().and_then(|o| o.get(w)).cloned().unwrap_or_else(|| t.clone()),
            _ => t.clone(),
        }
    };
    for (i, t) in tools.iter().enumerate() {
        let f = unwrap(t);
        let name = get(&f, "name").as_str().unwrap_or("");
        if !f.is_object() || name.is_empty() {
            return bad(format!("tools[{i}] has no name: each tool must be {shape}"));
        }
        let schema = get(&f, key);
        if !schema.is_null() && !schema.is_object() {
            return bad(format!(
                "tools[{i}] ({name}): \"{key}\" must be an object (the JSON schema of its parameters), not {}",
                py::type_name(schema)
            ));
        }
    }
    Ok(stable_tool_order(tools, |t| get(&unwrap(t), "name").as_str().unwrap_or("").to_string()))
}

pub type Converted = (Vec<Value>, Option<Vec<Value>>, Map<String, Value>);

/// OpenAI Chat Completions -> (template messages, template tools, template kwargs).
pub fn openai_to_messages(req: &Value) -> Result<Converted, BadRequest> {
    let mut messages = vec![];
    for m in object_list(get(req, "messages"), "messages")? {
        let mut role = get(&m, "role").clone();
        if role.as_str() == Some("developer") {
            role = json!("system");
        }
        let images = matches!(role.as_str(), Some("user" | "tool" | "assistant"));
        let content = check_parts(get(&m, "content"));
        let mut out = Map::new();
        out.insert("role".into(), role);
        out.insert("content".into(), if images { parts_of(&content) } else { Value::String(text_of(&content)) });
        if truthy(get(&m, "reasoning_content")) {
            out.insert("reasoning_content".into(), get(&m, "reasoning_content").clone());
        }
        if truthy(get(&m, "tool_calls")) {
            let mut calls = vec![];
            for c in object_list(get(&m, "tool_calls"), "tool_calls")? {
                let f = c.as_object().and_then(|o| o.get("function")).cloned().unwrap_or_else(|| c.clone());
                if !f.is_object() {
                    return bad("tool_calls must be a list of objects (each with a \"function\" object)");
                }
                calls.push(json!({"function": {"name": get(&f, "name").clone(), "arguments": Value::Object(tool_arguments(get(&f, "arguments")))}}));
            }
            out.insert("tool_calls".into(), Value::Array(calls));
        }
        messages.push(Value::Object(out));
    }
    let tools: Vec<Value> = tool_list(get(req, "tools"), Some("function"))?
        .into_iter()
        .map(|t| if get(&t, "type").as_str() == Some("function") { t.as_object().and_then(|o| o.get("function")).cloned().unwrap_or(t) } else { t })
        .collect();
    let mut kwargs = Map::new();
    let reasoning = get(req, "reasoning");
    let effort = if truthy(get(req, "reasoning_effort")) { get(req, "reasoning_effort") } else { get(reasoning, "effort") };
    kwargs.extend(effort_kwargs(effort)?);
    if let Some(ctk) = get(req, "chat_template_kwargs").as_object() {
        for (k, v) in ctk {
            if k == "enable_thinking" && !truthy(v) {
                kwargs = Map::from_iter([("enable_thinking".into(), json!(false))]);
            } else if k == "reasoning_effort" && !kwargs.contains_key("enable_thinking") {
                kwargs.extend(effort_kwargs(v)?);
            }
        }
    }
    Ok((late_system_to_user(messages), if tools.is_empty() { None } else { Some(tools) }, kwargs))
}

const BILLING_HEADER: &str = "x-anthropic-billing-header:";

/// Claude Code's per-request billing stamps (cch, the version's 4th part) pinned to f's, as llama.cpp does.
pub fn pin_billing_stamp(system: &str) -> String {
    if !system.starts_with(BILLING_HEADER) {
        return system.to_string();
    }
    // Python indexes characters; the header and its stamps are ASCII, positions past them are only compared
    let s: Vec<char> = system.chars().collect();
    let find = |sub: &str, from: usize| -> isize {
        let sub: Vec<char> = sub.chars().collect();
        (from..=s.len().saturating_sub(sub.len())).find(|&i| s[i..i + sub.len()] == sub[..]).map(|i| i as isize).unwrap_or(-1)
    };
    let mut out = s.clone();
    let cch = find("cch=", BILLING_HEADER.len());
    if (0..=160).contains(&cch) {
        let v = cch as usize + 4;
        let end = find(";", v);
        if end > v as isize && end - v as isize <= 16 {
            for c in out.iter_mut().take(end as usize).skip(v) {
                *c = 'f';
            }
        }
    }
    let cv = find("cc_version=", 0);
    if (0..=160).contains(&cv) {
        let v = cv as usize + "cc_version=".len();
        let end = find(";", cv as usize);
        if (v as isize) < end && end - v as isize <= 64 {
            let ver: String = s[v..end as usize].iter().collect();
            let parts: Vec<&str> = ver.split('.').collect();
            if parts.len() > 3 {
                let tail = v + parts[..3].join(".").chars().count() + 1;
                for c in out.iter_mut().take(end as usize).skip(tail) {
                    *c = 'f';
                }
            }
        }
    }
    out.into_iter().collect()
}

const ANTHROPIC_BLOCKS: [&str; 8] = ["text", "image_url", "input_image", "image", "thinking", "tool_use", "tool_result", "redacted_thinking"];

/// Anthropic Messages -> (template messages, template tools, template kwargs).  `think_unasked`: a request that asks
/// for no thinking level gets the template's default (#278).
pub fn anthropic_to_messages(req: &Value, think_unasked: bool) -> Result<Converted, BadRequest> {
    let _ = ANTHROPIC_BLOCKS;
    let mut messages = vec![];
    let system = get(req, "system");
    if truthy(system) {
        messages.push(json!({"role": "system", "content": pin_billing_stamp(&text_of(&check_parts(system)))}));
    }
    for m in object_list(get(req, "messages"), "messages")? {
        let content = check_parts(get(&m, "content"));
        if let Value::String(s) = &content {
            messages.push(json!({"role": get(&m, "role").clone(), "content": s}));
            continue;
        }
        let blocks: Vec<Value> = content.as_array().cloned().unwrap_or_default();
        if get(&m, "role").as_str() == Some("user") && has_image(&content) && !blocks.iter().any(|b| get(b, "type").as_str() == Some("tool_result")) {
            messages.push(json!({"role": "user", "content": parts_of(&content)}));
            continue;
        }
        let (mut text, mut reasoning, mut calls, mut parts) = (vec![], vec![], vec![], vec![]);
        for block in &blocks {
            match get(block, "type").as_str() {
                Some("text") => {
                    text.push(get_or(block, "text", ""));
                    parts.push(block.clone());
                }
                Some(t) if IMAGE_PARTS.contains(&t) => parts.push(block.clone()),
                Some("thinking") => reasoning.push(get(block, "thinking").as_str().unwrap_or("").to_string()),
                Some("tool_use") => calls.push(json!({"function": {"name": get(block, "name").clone(), "arguments": Value::Object(tool_arguments(get(block, "input")))}})),
                Some("tool_result") => {
                    messages.push(json!({"role": "tool", "content": parts_of(&check_parts(get(block, "content")))}));
                }
                _ => {}
            }
        }
        if !text.is_empty() || !calls.is_empty() || !reasoning.is_empty() || !parts.is_empty() {
            let parts_v = Value::Array(parts);
            let mut out = Map::new();
            out.insert("role".into(), get(&m, "role").clone());
            out.insert("content".into(), if has_image(&parts_v) { parts_of(&parts_v) } else { Value::String(text.concat()) });
            if !reasoning.is_empty() {
                out.insert("reasoning_content".into(), Value::String(reasoning.concat()));
            }
            if !calls.is_empty() {
                out.insert("tool_calls".into(), Value::Array(calls));
            }
            messages.push(Value::Object(out));
        }
    }
    let tools: Vec<Value> = tool_list(get(req, "tools"), None)?
        .into_iter()
        .map(|t| {
            let desc = t.as_object().and_then(|o| o.get("description")).cloned().unwrap_or(json!(""));
            let params = t.as_object().and_then(|o| o.get("input_schema")).cloned().unwrap_or(json!({}));
            json!({"name": get(&t, "name").clone(), "description": desc, "parameters": params})
        })
        .collect();
    let mut kwargs = Map::new();
    let thinking = get(req, "thinking");
    let effort = get(get(req, "output_config"), "effort");
    if get(thinking, "type").as_str() == Some("disabled") {
        kwargs.insert("enable_thinking".into(), json!(false));
    } else if truthy(effort) {
        kwargs.extend(effort_kwargs(effort)?);
    } else if truthy(get(thinking, "budget_tokens")) {
        kwargs.extend(budget_effort(get(thinking, "budget_tokens")));
    } else if thinking.is_null() && !truthy(get(req, "reasoning_budget_tokens")) && !think_unasked {
        kwargs.insert("enable_thinking".into(), json!(false));
    }
    Ok((late_system_to_user(messages), if tools.is_empty() { None } else { Some(tools) }, kwargs))
}

/// `json.dumps(x)` for the "[strata] ..." lines that quote a request's value.
pub fn quote(v: &Value) -> String {
    pyjson::dumps(v)
}
