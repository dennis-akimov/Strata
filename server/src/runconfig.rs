//! #564: the web page's Settings view of the run config (serve/runconfig.py): a short list of documented keys can be
//! read and changed; everything else in the file is kept as it is.  The earlier file is kept as <name>.bak.

use crate::engine::py_g;
use crate::py;
use crate::pyjson;
use serde_json::{json, Map, Value};
use std::path::Path;

#[derive(Clone, Copy)]
enum K {
    Bool,
    Int0,
    Num0,
    Enum(&'static [&'static str]),
    Names,
    Sampling(&'static str),
    Arg(&'static str),
}

const EDITABLE: [(&str, K, &str); 15] = [
    ("sampling.temperature", K::Sampling("num>=0"), "Default temperature for requests that send none (0 = greedy, the default without a sampling block)"),
    ("sampling.top_p", K::Sampling("0<x<=1"), "Default top_p for requests that send none"),
    ("sampling.top_k", K::Sampling("1..64"), "Default top_k for requests that send none (1-64)"),
    ("sampling.min_p", K::Sampling("0<=x<=1"), "Default min_p for requests that send none"),
    ("reasoning_budget_tokens", K::Int0, "Cap the thinking of every request at this many tokens (0 or empty: no cap)"),
    ("fit_max_tokens", K::Bool, "Shorten a max_tokens that does not fit the context instead of answering 400"),
    ("anthropic_thinking", K::Enum(&["model", "on_request"]), "Anthropic requests that do not ask for thinking: think as the model does (model) or not (on_request)"),
    ("effort_position", K::Enum(&["start", "end"]), "Where a non-default reasoning effort goes: start (the default) or end (keeps the cache when it changes)"),
    ("aliases", K::Names, "Other model names the server lists and answers to (comma-separated)"),
    ("idle_unload_s", K::Num0, "Unload the model after this many seconds without requests (0 or empty: never)"),
    ("lazy_load", K::Bool, "Start without loading the model; the first request loads it (text only)"),
    ("engine_silence_s", K::Num0, "End a request when the engine says nothing for this long (default 300 s, 0 = wait)"),
    ("api_monitor", K::Bool, "Keep the last 100 requests' prompts and answers in memory for /api-monitor"),
    ("open_browser", K::Bool, "Open the chat page in the browser when the model is ready"),
    ("vram_reserve_mib", K::Arg("--vram-reserve-mib"), "VRAM in MiB the engine leaves free for other programs (engine default 700)"),
];

fn spec(key: &str) -> Option<K> {
    EDITABLE.iter().find(|(k, _, _)| *k == key).map(|(_, k, _)| *k)
}

fn arg_of(cfg: &Value, flag: &str) -> Value {
    let a: Vec<Value> = cfg.get("args").and_then(Value::as_array).cloned().unwrap_or_default();
    if let Some(i) = a[..a.len().saturating_sub(1)].iter().position(|x| x == flag) {
        let v = py::py_str(&a[i + 1]);
        return if !v.is_empty() && v.chars().all(|c| c.is_ascii_digit()) { json!(v.parse::<i64>().unwrap_or(0)) } else { json!(v) };
    }
    Value::Null
}

fn value_of(cfg: &Value, key: &str) -> Value {
    match spec(key) {
        Some(K::Sampling(_)) => cfg.get("sampling").and_then(Value::as_object).and_then(|s| s.get(key.split_once('.').unwrap().1)).cloned().unwrap_or(Value::Null),
        Some(K::Arg(flag)) => arg_of(cfg, flag),
        _ => cfg.get(key).cloned().unwrap_or(Value::Null),
    }
}

/// GET /config: the editable keys with their values (null: not set, the default applies).
pub fn view(cfg: &Value, path: &str) -> Value {
    let keys: Vec<Value> = EDITABLE
        .iter()
        .map(|(key, kind, help)| {
            let kn = match kind {
                K::Sampling(_) | K::Arg(_) | K::Int0 | K::Num0 => "number",
                K::Bool => "bool",
                K::Enum(_) => "enum",
                K::Names => "names",
            };
            let mut m = json!({"key": key, "value": value_of(cfg, key), "help": help, "kind": kn}).as_object().cloned().unwrap();
            if let K::Enum(c) = kind {
                m.insert("choices".into(), json!(c));
            }
            Value::Object(m)
        })
        .collect();
    json!({"file": Path::new(path).file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default(), "keys": keys,
        "note": "Saved to the run config; used from the next start of the model."})
}

fn number(key: &str, v: &Value, whole: bool, lo: f64, hi: Option<f64>, lo_open: bool) -> Result<Value, String> {
    let n = v.as_f64().filter(|_| v.is_number());
    let ok = n.map(|n| !(whole && n != n.trunc()) && !(if lo_open { n <= lo } else { n < lo }) && hi.map(|h| n <= h).unwrap_or(true)).unwrap_or(false);
    if !ok {
        let rng = format!("{} {}{}", if lo_open { "more than" } else { "at least" }, py_g(lo), hi.map(|h| format!(" and at most {}", py_g(h))).unwrap_or_default());
        return Err(format!("{key}: expected a {}number, {rng}, not {}", if whole { "whole " } else { "" }, py::py_repr(v)));
    }
    Ok(if whole { json!(n.unwrap() as i64) } else { v.clone() })
}

/// The value to store for `key`, or what is expected; Null removes the key.
fn check(key: &str, v: &Value, cfg: &Value) -> Result<Value, String> {
    let Some(kind) = spec(key) else { return Err(format!("{} cannot be changed here (only the keys the Settings view lists)", py::py_repr(&json!(key)))) };
    if v.is_null() {
        return Ok(Value::Null);
    }
    match kind {
        K::Bool => {
            if !v.is_boolean() {
                return Err(format!("{key}: expected true or false, not {}", py::py_repr(v)));
            }
            if key == "lazy_load" && v == &json!(true) && py::truthy(py::get(cfg, "vision")) {
                return Err("lazy_load: lazy loading is text-only, and this model reads images (\"vision\")".into());
            }
            Ok(v.clone())
        }
        K::Enum(c) => {
            if !v.as_str().map(|s| c.contains(&s)).unwrap_or(false) {
                return Err(format!("{key}: expected one of {}, not {}", c.join(", "), py::py_repr(v)));
            }
            Ok(v.clone())
        }
        K::Names => {
            let list = match v {
                Value::String(s) => json!(s.split(',').map(py::strip).filter(|x| !x.is_empty()).collect::<Vec<_>>()),
                _ => v.clone(),
            };
            let ok = list.as_array().map(|a| a.iter().all(|x| x.as_str().map(|s| !py::strip(s).is_empty() && s.chars().count() <= 128).unwrap_or(false))).unwrap_or(false);
            if !ok {
                return Err(format!("{key}: expected a list of model names, not {}", py::py_repr(v)));
            }
            Ok(if list.as_array().unwrap().is_empty() { Value::Null } else { list })
        }
        K::Int0 => number(key, v, true, 0.0, None, false),
        K::Num0 => number(key, v, false, 0.0, None, false),
        K::Arg(_) => number(key, v, true, 0.0, None, false),
        K::Sampling(rule) => {
            let f = |r: Result<Value, String>| r.map(|x| json!(x.as_f64().unwrap()));
            match rule {
                "num>=0" => f(number(key, v, false, 0.0, None, false)),
                "0<x<=1" => f(number(key, v, false, 0.0, Some(1.0), true)),
                "0<=x<=1" => f(number(key, v, false, 0.0, Some(1.0), false)),
                _ => number(key, v, true, 1.0, Some(64.0), false),
            }
        }
    }
}

/// Python's `==` for JSON values (5 == 5.0).
fn py_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        _ => a == b,
    }
}

/// The config with `changes` made, and the keys that changed; every value is checked first.
pub fn apply(cfg: &Value, changes: &Value) -> Result<(Value, Vec<String>), String> {
    let Some(ch) = changes.as_object().filter(|c| !c.is_empty()) else { return Err(r#"send {"set": {"<key>": <value or null>, ...}}"#.into()) };
    let mut checked = vec![];
    for (k, v) in ch {
        checked.push((k.clone(), check(k, v, cfg)?));
    }
    let mut new = cfg.clone();
    let mut changed = vec![];
    for (key, v) in checked {
        if py_eq(&value_of(&new, &key), &v) {
            continue;
        }
        let o = new.as_object_mut().unwrap();
        match spec(&key).unwrap() {
            K::Sampling(_) => {
                let mut s: Map<String, Value> = o.get("sampling").and_then(Value::as_object).cloned().unwrap_or_default();
                let sub = key.split_once('.').unwrap().1;
                if v.is_null() {
                    s.shift_remove(sub);
                } else {
                    s.insert(sub.into(), v);
                }
                if s.is_empty() {
                    o.shift_remove("sampling");
                } else {
                    o.insert("sampling".into(), Value::Object(s));
                }
            }
            K::Arg(flag) => {
                let mut a: Vec<Value> = o.get("args").and_then(Value::as_array).cloned().unwrap_or_default();
                let pos = a[..a.len().saturating_sub(1)].iter().position(|x| x == flag);
                match (pos, v.is_null()) {
                    (Some(i), true) => {
                        a.drain(i..i + 2);
                    }
                    (Some(i), false) => a[i + 1] = json!(py::py_str(&v)),
                    (None, false) => {
                        a.push(json!(flag));
                        a.push(json!(py::py_str(&v)));
                    }
                    (None, true) => {}
                }
                o.insert("args".into(), Value::Array(a));
            }
            _ if v.is_null() => {
                o.shift_remove(&key);
            }
            _ => {
                o.insert(key.clone(), v);
            }
        }
        changed.push(key);
    }
    Ok((new, changed))
}

/// The config written whole (a temporary file moved over the old one), the earlier one kept as <name>.bak.
pub fn save(path: &str, cfg: &Value) -> std::io::Result<String> {
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    let p = Path::new(path);
    let bak = format!("{path}.bak");
    std::fs::copy(p, &bak)?;
    let mode = std::fs::metadata(p)?.permissions().mode() & 0o777;
    let tmp = format!("{path}.tmp");
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(mode).open(&tmp)?;
        f.write_all(pyjson::dumps_with(cfg, pyjson::Opts { indent: Some(1), ..Default::default() }).as_bytes())?;
    }
    std::fs::rename(&tmp, p)?;
    Ok(Path::new(&bak).file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or(bak))
}

pub fn load(path: &str) -> Result<Value, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let v: Value = serde_json::from_str(text.trim_start_matches('\u{feff}')).map_err(|e| e.to_string())?;
    if !v.is_object() {
        return Err(format!("{} is not a JSON object", Path::new(path).file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default()));
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn changes_are_checked_and_applied() {
        let cfg = json!({"args": ["--gguf", "x", "--vram-reserve-mib", "700"], "sampling": {"temperature": 1.0}});
        let (new, ch) = apply(&cfg, &json!({"sampling.top_p": 0.9, "vram_reserve_mib": 1024, "sampling.temperature": 1})).unwrap();
        assert_eq!(ch, vec!["sampling.top_p", "vram_reserve_mib"]);
        assert_eq!(new["args"], json!(["--gguf", "x", "--vram-reserve-mib", "1024"]));
        assert_eq!(new["sampling"], json!({"temperature": 1.0, "top_p": 0.9}));
        assert_eq!(apply(&cfg, &json!({"sampling.top_k": 99})).unwrap_err(), "sampling.top_k: expected a whole number, at least 1 and at most 64, not 99");
        assert!(apply(&cfg, &json!({"host": "x"})).is_err());
    }
}
