//! The service between the HTTP layer and the engine (serve/server.py's Service): one request at a time behind a FIFO
//! (or several in the engine's batch slots), the prompt prepared from the template, the run's events, and what the
//! Monitor, /metrics and /status show.

use crate::engine::{round1, Engine, Live};
use crate::errors::Fail;
use crate::frontend::{images_of, literal_tags, mark_think_literals, unmark_think_literals, Tags};
use crate::py::{self, get, truthy};
use crate::pyjson;
use crate::say;
use crate::sync::{now, Sem};
use crate::template::ChatTemplate;
use crate::tokenizer::Tokenize;
use crate::vision::{ImgSrc, Vision};
use serde_json::{json, Map, Value};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering::SeqCst};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub type Rec = Arc<Mutex<Map<String, Value>>>;

thread_local! {
    /// #332: this request's monitor record (the request runs on its own thread, as in the Python server)
    pub static TRACE: RefCell<Option<Rec>> = const { RefCell::new(None) };
    /// the current request's combined image embeddings file (GENI)
    pub static EMBEDDINGS: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
}

pub fn trace() -> Option<Rec> {
    TRACE.with(|t| t.borrow().clone())
}

pub const IM_END: &str = "<|im_end|>";
pub const IMAGE_PAD: &str = "<|image_pad|>";
pub const VISION_START: &str = "<|vision_start|>";
pub const EFFORT_TURN: &str = "<|im_start|>system\n{}<|im_end|>\n";
pub fn effort_text(level: &str) -> &'static str {
    match level {
        "low" => "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
        "xhigh" => "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
        _ => "Reasoning effort is set to medium. Think as much as the task needs, without unnecessary elaboration.",
    }
}
pub const REPEAT_STOP_TOKENS: i64 = 256;
pub const ANSWER_RESERVE_MIN: i64 = 512;
pub const REASONING_CLOSE: &str = "\n</think>\n\n";
pub const REASONING_WRAP_UP: &str = "\n\nI have thought about this long enough; time to give my answer.\n</think>\n\n";
pub const HARMONY_CLOSE: &str = "<|end|><|start|>assistant<|channel|>final<|message|>";
pub fn harmony_wrap_up() -> String {
    format!(" I have thought about this long enough; time to give my answer.{HARMONY_CLOSE}")
}
pub const LOOP_CHECK_EVERY: i64 = 512;
pub const LOOP_COVERAGE: f64 = 0.25;
pub const LOOP_HISTORY_WORDS: usize = 30000;
pub const LOOP_TAIL_CHARS: usize = 2048;
pub const LOOP_TAIL_PERIOD: usize = 64;
pub const QUEUE_BEAT_S: f64 = 10.0;
pub const KEEPALIVE_S: f64 = 10.0;
pub const CTX_SLACK: i64 = 8;
pub const RATE_WINDOW_S: f64 = 2.0;
pub const RATE_MIN_SPAN_S: f64 = 0.25;
pub const PREFIX_END: &str = "\u{1}strata-prefix-end\u{1}";

/// #728: coverage of recent words by 12-word passages seen at least three times (1.0 for a periodic tail, #1753).
pub fn reasoning_repeat_coverage(text: &str) -> f64 {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() >= LOOP_TAIL_CHARS {
        let tail: String = chars[chars.len() - LOOP_TAIL_CHARS..].iter().collect::<String>().to_lowercase();
        let t: Vec<char> = tail.chars().collect();
        if t.len() == LOOP_TAIL_CHARS && (1..=LOOP_TAIL_PERIOD).any(|p| t[p..] == t[..t.len() - p]) {
            return 1.0;
        }
    }
    // Python's re: \w is str.isalnum() or "_"
    let lower = text.to_lowercase();
    let mut words: Vec<String> = vec![];
    let mut cur = String::new();
    let isw = |c: char| c.is_alphanumeric() || c == '_';
    for c in lower.chars() {
        if isw(c) {
            cur.push(c);
        } else {
            if !cur.is_empty() {
                words.push(std::mem::take(&mut cur));
            }
            if !py::is_space(c) {
                words.push(c.to_string());
            }
        }
    }
    if !cur.is_empty() {
        words.push(cur);
    }
    let words = &words[words.len().saturating_sub(LOOP_HISTORY_WORDS)..];
    let (width, window) = (12, 2000);
    if words.len() < window {
        return 0.0;
    }
    let grams: Vec<&[String]> = (0..words.len() - width + 1).map(|i| &words[i..i + width]).collect();
    let mut counts: std::collections::HashMap<&[String], usize> = std::collections::HashMap::new();
    for g in &grams {
        *counts.entry(g).or_insert(0) += 1;
    }
    let start = words.len().saturating_sub(window);
    let mut covered = std::collections::HashSet::new();
    for (i, g) in grams.iter().enumerate().skip(start) {
        if counts[g] >= 3 {
            covered.extend(i..i + width);
        }
    }
    covered.len() as f64 / (words.len() - start) as f64
}

/// #728: the prompt with the first system message's xhigh effort sentence swapped for the low one, then the output.
pub fn focused_recovery_prompt(tok: &dyn Tokenize, ids: &[u32], generated: &[u32]) -> Option<Vec<u32>> {
    let end = tok.encode(IM_END, true);
    if end.len() != 1 {
        return None;
    }
    let at = ids.iter().position(|t| *t == end[0])?;
    let head = &ids[..at];
    if !tok.decode(head).starts_with("<|im_start|>system\n") {
        return None;
    }
    let old = tok.encode(effort_text("xhigh"), false);
    let new = tok.encode(effort_text("low"), false);
    if old.len() > head.len() {
        return None;
    }
    for i in 0..=head.len() - old.len() {
        if head[i..i + old.len()] == old[..] {
            let mut out = ids[..i].to_vec();
            out.extend(&new);
            out.extend(&ids[i + old.len()..]);
            out.extend(generated);
            return Some(out);
        }
    }
    None
}

/// The request's stop strings (OpenAI `stop`, Anthropic `stop_sequences`); a bad value is a 400.
pub fn stop_strings(req: &Value) -> Result<Vec<String>, Fail> {
    let as_list = |v: &Value| -> Option<Vec<String>> { v.as_array()?.iter().map(|x| x.as_str().map(String::from)).collect() };
    let stop = get(req, "stop");
    let list = if !stop.is_null() {
        let l = match stop {
            Value::String(s) => vec![s.clone()],
            v => as_list(v).ok_or_else(|| Fail::Value("stop must be a string or a list of strings".into()))?,
        };
        if l.len() > 4 {
            return Err(Fail::Value("stop takes at most 4 strings".into()));
        }
        l
    } else if !get(req, "stop_sequences").is_null() {
        as_list(get(req, "stop_sequences")).ok_or_else(|| Fail::Value("stop_sequences must be a list of strings".into()))?
    } else {
        return Ok(vec![]);
    };
    Ok(list.into_iter().filter(|x| !x.is_empty()).collect())
}

/// Cuts the answer's text at the first stop string, holding back only a tail that could still start one.
pub struct StopMatcher {
    stops: Vec<String>,
    pub held: String,
    pub hit: Option<String>,
}

impl StopMatcher {
    pub fn new(stops: Vec<String>) -> Self {
        StopMatcher { stops, held: String::new(), hit: None }
    }
    pub fn push(&mut self, text: &str) -> String {
        if self.hit.is_some() {
            return String::new();
        }
        let buf = format!("{}{}", self.held, text);
        let mut first: Option<(usize, &String)> = None;
        for s in &self.stops {
            if let Some(i) = buf.find(s.as_str()) {
                if first.map(|f| i < f.0).unwrap_or(true) {
                    first = Some((i, s));
                }
            }
        }
        if let Some((i, s)) = first {
            self.hit = Some(s.clone());
            self.held.clear();
            return buf[..i].to_string();
        }
        let mut keep = 0;
        for s in &self.stops {
            // the longest end of buf that a stop string starts with (in whole characters)
            let ends: Vec<usize> = py::prefix_ends(s).collect();
            for &k in ends.iter().rev() {
                if k >= s.len() || k <= keep {
                    continue;
                }
                if buf.ends_with(&s[..k]) {
                    keep = k;
                    break;
                }
            }
        }
        self.held = buf[buf.len() - keep..].to_string();
        buf[..buf.len() - keep].to_string()
    }
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

/// #471: the prompt tokens a request got through (all, unless a cancel stopped its read part-way).
pub fn prompt_tokens_seen(prompt_tokens: i64, last: &Map<String, Value>) -> i64 {
    let read = last.get("prompt_read").and_then(Value::as_i64);
    match read {
        Some(r) if last.get("finish").and_then(Value::as_str) == Some("cancel") => prompt_tokens.min(last.get("reused").and_then(Value::as_i64).unwrap_or(0) + r),
        _ => prompt_tokens,
    }
}

/// One request's `timings` in llama.cpp's names, from the engine's own clock.
pub fn request_timings(prompt_tokens: i64, generated: i64, last: &Map<String, Value>) -> Option<Value> {
    let prompt_ms = last.get("prompt_ms").and_then(Value::as_f64)?;
    let cache_n = last.get("reused").and_then(Value::as_i64).unwrap_or(0);
    let prompt_n = (prompt_tokens - cache_n).max(0);
    let decode_ms = last.get("decode_ms").and_then(Value::as_f64).unwrap_or(0.0);
    let decoded = last.get("generated").and_then(Value::as_i64).filter(|g| *g != 0).unwrap_or(generated);
    let r = |x: f64, d: i32| pyround(x, d);
    let mut t = json!({"cache_n": cache_n, "prompt_n": prompt_n, "prompt_ms": r(prompt_ms, 1),
        "prompt_per_token_ms": if prompt_n > 0 { json!(r(prompt_ms / prompt_n as f64, 3)) } else { Value::Null },
        "prompt_per_second": if prompt_n > 0 && prompt_ms > 0.0 { json!(r(prompt_n as f64 / (prompt_ms / 1000.0), 1)) } else { Value::Null },
        "predicted_n": generated, "predicted_ms": r(decode_ms, 1),
        "predicted_per_token_ms": if decoded != 0 { json!(r(decode_ms / decoded as f64, 3)) } else { Value::Null },
        "predicted_per_second": if decoded != 0 && decode_ms > 0.0 { json!(r(decoded as f64 / (decode_ms / 1000.0), 1)) } else { Value::Null }});
    if let Some(off) = last.get("drafts_offered").filter(|v| !v.is_null()) {
        t["draft_n"] = json!(off.as_i64().unwrap_or(0));
        t["draft_n_accepted"] = json!(last.get("drafts_accepted").and_then(Value::as_i64).unwrap_or(0));
    }
    Some(t)
}

/// Python's round(x, d) (a float).
pub fn pyround(x: f64, d: i32) -> f64 {
    let m = 10f64.powi(d);
    let y = x * m;
    let r = y.round();
    // round half to even, as Python does for exact halves
    let r = if (y - y.trunc()).abs() == 0.5 { 2.0 * (y / 2.0).round() } else { r };
    r / m
}

/// Input/cache accounting at the API boundary, native work summed across continuations.
pub fn request_stats(segments: &[Map<String, Value>]) -> Map<String, Value> {
    let Some(last) = segments.last() else { return Map::new() };
    let mut result = last.clone();
    for key in ["reused", "prompt_read"] {
        match segments[0].get(key) {
            Some(v) => {
                result.insert(key.into(), v.clone());
            }
            None => {
                result.shift_remove(key);
            }
        }
    }
    for key in ["prompt_ms", "decode_ms", "generated", "drafts_offered", "drafts_accepted", "hits", "lookups", "offloaded", "ram_blobs", "file_blobs", "file_mb"] {
        if segments.iter().any(|s| s.get(key).map(|v| !v.is_null()).unwrap_or(false)) {
            let all_int = segments.iter().all(|s| s.get(key).map(|v| v.is_null() || v.is_i64() || v.is_u64()).unwrap_or(true));
            if all_int {
                result.insert(key.into(), json!(segments.iter().map(|s| s.get(key).and_then(Value::as_i64).unwrap_or(0)).sum::<i64>()));
            } else {
                let sum: f64 = segments.iter().map(|s| s.get(key).and_then(Value::as_f64).unwrap_or(0.0)).sum();
                result.insert(key.into(), json!(sum));
            }
        }
    }
    result
}

/// The Chat settings other apps get (POST /settings): only known keys, each checked.
pub fn clean_shared_defaults(d: &Value) -> Result<Map<String, Value>, Fail> {
    let o = match d {
        Value::Null => return Ok(Map::new()),
        Value::Object(o) => o,
        _ => return Err(Fail::Value("defaults must be an object".into())),
    };
    let mut out = Map::new();
    for (key, value) in o {
        if value.is_null() || value.as_str() == Some("") {
            continue;
        }
        let number = value.as_f64().filter(|_| value.is_number());
        let whole = number.filter(|n| *n == n.trunc());
        let v = match key.as_str() {
            "reasoning_effort" => match value.as_str() {
                Some("none" | "low" | "medium" | "high") => value.clone(),
                _ => return Err(Fail::Value("reasoning_effort: none, low, medium or high".into())),
            },
            "temperature" => match number.filter(|n| (0.0..=2.0).contains(n)) {
                Some(n) => json!(n),
                None => return Err(Fail::Value("temperature: 0..2".into())),
            },
            "top_p" => match number.filter(|n| *n > 0.0 && *n <= 1.0) {
                Some(n) => json!(n),
                None => return Err(Fail::Value("top_p: 0 < top_p <= 1".into())),
            },
            "top_k" => match whole.filter(|n| (1.0..=64.0).contains(n)) {
                Some(n) => json!(n as i64),
                None => return Err(Fail::Value("top_k: an integer 1..64".into())),
            },
            "seed" | "max_tokens" => match whole.filter(|n| *n > 0.0) {
                Some(n) => json!(n as i64),
                None => return Err(Fail::Value(format!("{key}: a positive integer"))),
            },
            "reasoning_budget_tokens" => match whole.filter(|n| *n >= 0.0) {
                Some(n) => json!(n as i64),
                None => return Err(Fail::Value("reasoning_budget_tokens: a whole number of tokens (0: no budget)".into())),
            },
            "experimental_speed_projection" => match value {
                Value::Bool(_) => value.clone(),
                _ => return Err(Fail::Value("experimental_speed_projection: true or false".into())),
            },
            _ => return Err(Fail::Value(format!("unknown setting {}", py::py_repr(&json!(key))))),
        };
        out.insert(key.clone(), v);
    }
    Ok(out)
}

/// A request's sampling field of the wrong JSON type is a 400 naming it; numbers sent as strings are read.
pub fn check_request_sampling(req: &mut Map<String, Value>) -> Result<(), Fail> {
    if req.get("repeat_penalty").map(|v| !v.is_null()).unwrap_or(false) && req.get("repetition_penalty").map(Value::is_null).unwrap_or(true) {
        let v = req["repeat_penalty"].clone();
        req.insert("repetition_penalty".into(), v);
    }
    if let Some(Value::Number(n)) = req.get("repeat_last_n") {
        if let Some(rl) = n.as_i64().filter(|v| *v > 0 && n.is_i64()) {
            if req.get("penalty_last_n").map(Value::is_null).unwrap_or(true) {
                req.insert("penalty_last_n".into(), json!(rl));
            }
        }
    }
    for (key, integer) in [("temperature", false), ("top_p", false), ("min_p", false), ("presence_penalty", false), ("frequency_penalty", false), ("repetition_penalty", false), ("top_k", true), ("penalty_last_n", true), ("seed", true)] {
        let what = if integer { "an integer" } else { "a number" };
        let bad = || Fail::Value(format!("{key}: {what}"));
        let Some(value) = req.get(key).cloned() else { continue };
        if value.is_null() {
            continue;
        }
        let value = if let Value::String(s) = &value {
            if py::strip(s).is_empty() {
                return Err(bad());
            }
            let num: f64 = py::strip(s).parse().map_err(|_| bad())?;
            if !num.is_finite() {
                return Err(bad());
            }
            let v = if integer {
                if num != num.trunc() {
                    return Err(bad());
                }
                json!(num as i64)
            } else if regex::Regex::new(r"^\s*[+-]?\d+\s*$").unwrap().is_match(s) {
                json!(py::strip(s).parse::<i64>().map_err(|_| bad())?)
            } else {
                json!(num)
            };
            req.insert(key.into(), v.clone());
            v
        } else {
            value
        };
        let ok = match &value {
            Value::Number(n) => !integer || n.is_i64() || n.is_u64(),
            _ => false,
        };
        if !ok {
            return Err(bad());
        }
    }
    Ok(())
}

/// The run config's optional `sampling` block: defaults for the fields a request leaves out.
pub fn sampling_defaults_from_config(cfg: &Value) -> Result<Map<String, Value>, String> {
    let mut block: Map<String, Value> = get(cfg, "sampling").as_object().cloned().unwrap_or_default();
    for (alias, canon) in [("repeat_penalty", "repetition_penalty"), ("repeat_last_n", "penalty_last_n")] {
        if block.get(alias).map(|v| !v.is_null()).unwrap_or(false) {
            if block.get(canon).map(|v| !v.is_null()).unwrap_or(false) {
                return Err(format!("[strata] config sampling: both '{alias}' and '{canon}' are set; they are the same setting, keep one"));
            }
            let v = block.shift_remove(alias).unwrap();
            block.insert(canon.into(), v);
        }
    }
    let mut out = Map::new();
    for (key, value) in &block {
        if value.is_null() {
            continue;
        }
        let number = value.as_f64().filter(|_| value.is_number());
        let whole = number.filter(|n| *n == n.trunc());
        let r = py::py_repr(value);
        let err = |m: &str| Err(format!("[strata] config sampling.{key}={r}: {m}"));
        match key.as_str() {
            "temperature" => match number.filter(|n| *n >= 0.0) { Some(n) => { out.insert(key.clone(), json!(n)); } None => return err("expected a number >= 0 (0 = greedy)") },
            "top_p" => match number.filter(|n| *n > 0.0 && *n <= 1.0) { Some(n) => { out.insert(key.clone(), json!(n)); } None => return err("expected 0 < top_p <= 1") },
            "min_p" => match number.filter(|n| (0.0..=1.0).contains(n)) { Some(n) => { out.insert(key.clone(), json!(n)); } None => return err("expected 0 <= min_p <= 1") },
            "top_k" => match whole.filter(|n| (1.0..=64.0).contains(n)) { Some(n) => { out.insert(key.clone(), json!(n as i64)); } None => return err("the sampled path takes an integer 1..64") },
            "presence_penalty" => match number.filter(|n| *n >= 0.0) { Some(n) => { out.insert(key.clone(), json!(n)); } None => return err("expected a number >= 0") },
            "frequency_penalty" => match number.filter(|n| *n >= 0.0) { Some(n) => { out.insert(key.clone(), json!(n)); } None => return err("expected a number >= 0") },
            "repetition_penalty" => match number.filter(|n| *n > 0.0) { Some(n) => { out.insert(key.clone(), json!(n)); } None => return err("expected a number > 0 (1 = off)") },
            "penalty_last_n" => match whole.filter(|n| *n >= 0.0) { Some(n) => { out.insert(key.clone(), json!(n as i64)); } None => return err("expected a non-negative integer") },
            "seed" => match whole.filter(|n| *n > 0.0) { Some(n) => { out.insert(key.clone(), json!(n as i64)); } None => return err("expected a positive integer") },
            "experimental_speed_projection" => match value { Value::Bool(_) => { out.insert(key.clone(), value.clone()); } _ => return err("expected true or false (the default for requests that leave it out, when the engine has the vector)") },
            _ => say(&format!("config sampling.{key}={r}: unknown key, ignored")),
        }
    }
    Ok(out)
}

/// The settings the Service runs with (the run config's keys, set once at the start).
#[derive(Default)]
pub struct Settings {
    pub fit_max_tokens: bool,
    pub aliases: Vec<String>,
    pub reasoning_close_retry: bool,
    pub codex_compaction_cache: bool,
    pub codex_thread_titles: bool,
    pub sampling_defaults: Map<String, Value>,
    pub config_effort: Option<String>,
    pub slot_save_path: Option<String>,
    pub api_key: Vec<String>,
    pub effort_end: bool,
    pub config_path: Option<String>,
    pub cors_origins: Vec<String>,
    pub trusted_origins: Vec<String>,
    pub allowed_hosts: Vec<String>,
    pub host_names: std::collections::HashSet<String>,
    pub api_monitor: bool,
    pub idle_unload_s: f64,
    pub min_free_vram_mib: i64,
    pub before_load: Option<Value>,
    pub reasoning_loop_recovery: Option<String>,
    pub reasoning_budget_tokens: Value,
    pub repeat_stop_tokens: i64,
    pub anthropic_think_unasked: bool,
    pub tool_call_recovery: bool,
    pub shared_path: Option<String>,
    pub gpu_index: i64,
}

pub struct Totals(pub Map<String, Value>);

pub struct State {
    pub status: Map<String, Value>,
    pub rate: VecDeque<(f64, i64)>,
    /// "parallel": every running request's own status and rate window, by request id (insertion order = age)
    pub live_reqs: indexmap::IndexMap<u64, (Map<String, Value>, VecDeque<(f64, i64)>)>,
    pub entering: i64,
    pub api_requests: VecDeque<Rec>,
    pub history: VecDeque<Value>,
    pub totals: Map<String, Value>,
    pub last_timings: Option<Value>,
    pub last_request_at: Option<f64>,
}

pub struct Service {
    pub engine: Engine,
    pub tok: Arc<dyn Tokenize>,
    pub template: ChatTemplate,
    pub model: String,
    pub vision: Option<Arc<Vision>>,
    pub literals: Tags,
    pub set: Settings,
    pub shared: Mutex<Map<String, Value>>,
    pub fifo: Sem,
    pub state: Mutex<State>,
    pub image_gate: crate::sync::Event,
    pub latencies: crate::prometheus::Latencies,
    pub started_at: f64,
    pub vram_reserve: Mutex<Option<Option<i64>>>,
    pub conv_log: Mutex<crate::engine::ConvCacheLog>,
    pub harmony: bool,
    pub glm: bool,
    pub stop_ids: Vec<u32>,
    pub mcp: Option<Arc<crate::mcp::McpHub>>,
    pub telemetry: Mutex<Option<Arc<crate::telemetry::Telemetry>>>,
    pub image: Option<Arc<crate::imagegen::ImageGen>>,
    pub video: Option<Arc<crate::videogen::VideoGen>>,
    pub config_lock: Mutex<()>,
    pub hosted_tools_noted: Mutex<std::collections::HashSet<String>>,
    pub next_req: std::sync::atomic::AtomicU64,
    pub stopping: AtomicBool,
}

impl Service {
    pub fn new(engine: Engine, tok: Arc<dyn Tokenize>, template: ChatTemplate, model: String, vision: Option<Arc<Vision>>, set: Settings) -> Self {
        let controls = tok.control_tokens();
        let literals = literal_tags(&controls);
        let special = |s: &str| tok.encode(s, true).len() == 1;
        // GPT-OSS: harmony (its tokenizer has <|channel|>); GLM: <|observation|>
        let harmony = special("<|channel|>") && is_special(&*tok, "<|channel|>");
        let glm = is_special(&*tok, "<|observation|>");
        let ends: Vec<&str> = if harmony { vec!["<|return|>", "<|call|>", "<|endoftext|>"] } else if glm { vec!["<|endoftext|>", "<|user|>", "<|observation|>"] } else { vec![IM_END, "<|endoftext|>"] };
        let mut stop_ids = vec![];
        for s in ends {
            let ids = tok.encode(s, true);
            if ids.len() == 1 && !stop_ids.contains(&ids[0]) {
                stop_ids.push(ids[0]);
            }
        }
        let totals = json!({"since": now(), "requests": 0, "prompt_tokens": 0, "reused": 0, "output_tokens": 0, "prompt_ms": 0.0, "decode_ms": 0.0, "drafts_offered": 0, "drafts_accepted": 0});
        Service {
            engine,
            tok,
            template,
            model,
            vision,
            literals,
            set,
            shared: Mutex::new(Map::new()),
            fifo: Sem::new(),
            state: Mutex::new(State {
                status: json!({"busy": false, "queued": 0}).as_object().cloned().unwrap(),
                rate: VecDeque::with_capacity(32),
                live_reqs: indexmap::IndexMap::new(),
                entering: 0,
                api_requests: VecDeque::new(),
                history: VecDeque::new(),
                totals: totals.as_object().cloned().unwrap(),
                last_timings: None,
                last_request_at: None,
            }),
            image_gate: crate::sync::Event::new(),
            latencies: crate::prometheus::Latencies::default(),
            started_at: now(),
            vram_reserve: Mutex::new(None),
            conv_log: Mutex::new(crate::engine::ConvCacheLog::default()),
            harmony,
            glm,
            stop_ids,
            mcp: None,
            telemetry: Mutex::new(None),
            image: None,
            video: None,
            config_lock: Mutex::new(()),
            hosted_tools_noted: Mutex::new(Default::default()),
            next_req: std::sync::atomic::AtomicU64::new(1),
            stopping: AtomicBool::new(false),
        }
    }

    pub fn loaded(&self) -> bool {
        self.engine.loaded()
    }

    pub fn model_names(&self) -> Vec<String> {
        std::iter::once(self.model.clone()).chain(self.set.aliases.iter().cloned()).collect()
    }

    /// The name to answer with: the request's own when it is an alias (#297), else the model's.
    pub fn model_for(&self, req: &Value) -> String {
        match get(req, "model").as_str() {
            Some(m) if self.set.aliases.iter().any(|a| a == m) => m.to_string(),
            _ => self.model.clone(),
        }
    }

    pub fn reported_ctx(&self) -> i64 {
        let m = self.engine.max_context();
        if m != 0 { m } else { self.engine.known_ctx() }
    }

    /// #123: the most tokens this request may think, or None; ValueError for anything but a whole number.
    pub fn reasoning_budget(&self, req: &Value) -> Result<Option<i64>, Fail> {
        let mut value = get(req, "reasoning_budget_tokens").clone();
        if value.is_null() {
            value = self.set.reasoning_budget_tokens.clone();
        }
        if value.is_null() {
            value = json!(0);
        }
        let n = match &value {
            Value::Number(n) if n.is_i64() => n.as_i64(),
            Value::Number(n) if n.as_f64().map(|f| f == f.trunc()).unwrap_or(false) && n.is_f64() => n.as_f64().map(|f| f as i64),
            _ => None,
        };
        match n {
            None => Err(Fail::Value(format!("reasoning_budget_tokens={}: expected a whole number of tokens (0: no budget)", py::py_repr(&value)))),
            Some(n) => Ok(if n > 0 { Some(n) } else { None }),
        }
    }

    fn vision_down(&self) -> bool {
        self.vision.as_ref().map(|v| !v.alive()).unwrap_or(false)
    }

    pub fn status_get(&self, k: &str) -> Value {
        self.state.lock().unwrap().status.get(k).cloned().unwrap_or(Value::Null)
    }

    fn queued_add(&self, d: i64) {
        let mut s = self.state.lock().unwrap();
        let q = s.status.get("queued").and_then(Value::as_i64).unwrap_or(0) + d;
        s.status.insert("queued".into(), json!(q));
    }

    pub fn touch(&self) {
        self.state.lock().unwrap().last_request_at = Some(now());
    }

    /// Start the engine if it is not running (unloaded, or it died - issue #27); the caller holds the FIFO.
    pub fn ensure_loaded(&self) -> Result<(), Fail> {
        if self.loaded() && !self.vision_down() {
            return Ok(());
        }
        if let Some(cmd) = &self.set.before_load {
            let shown = match cmd {
                Value::String(s) => s.clone(),
                Value::Array(a) => a.iter().map(py::py_str).collect::<Vec<_>>().join(" "),
                v => py::py_str(v),
            };
            say(&format!("before loading: {shown}"));
            let r = match cmd {
                Value::String(s) => std::process::Command::new("/bin/sh").arg("-c").arg(s).stdin(std::process::Stdio::null()).status(),
                Value::Array(a) if !a.is_empty() => std::process::Command::new(py::py_str(&a[0])).args(a[1..].iter().map(py::py_str)).stdin(std::process::Stdio::null()).status(),
                _ => Err(std::io::Error::other("no command")),
            };
            if let Err(e) = r {
                say(&format!("the before_load command failed ({e}); loading anyway"));
            }
        }
        let mut vision_started = false;
        if self.vision_down() {
            say("starting the vision encoder ...");
            self.vision.as_ref().unwrap().restart().map_err(|e| Fail::Died(format!("the vision encoder failed to start: {e}")))?;
            vision_started = true;
        }
        if self.loaded() {
            return Ok(());
        }
        let Some(s) = self.engine.strata() else { return Ok(()) };
        if s.unloaded.load(SeqCst) {
            say("loading the model again (it was unloaded) ...");
        } else {
            let code = s.exit_code().map(|c| c.to_string()).unwrap_or_else(|| "None".into());
            say(&format!("the engine had stopped (exit code {code}); starting it again (a minute or two) ..."));
        }
        if let Err(e) = s.restart(3) {
            if vision_started {
                self.vision.as_ref().unwrap().unload();
            }
            return Err(e);
        }
        say("the engine is running again");
        if let Some(r) = *self.vram_reserve.lock().unwrap() {
            if let Err(e) = s.vram(r, 120.0) {
                say(&format!("the VRAM reserve ({} MiB) was not applied: {e}", r.map(|v| v.to_string()).unwrap_or_else(|| "None".into())));
            }
        }
        Ok(())
    }

    /// #533, POST /v1/vram: keep `reserve_mib` of VRAM free for other programs, applied between requests.
    pub fn vram(&self, reserve_mib: Option<i64>) -> Result<Value, Fail> {
        let Some(s) = self.engine.strata() else { return Err(Fail::Value("this engine cannot resize its VRAM use".into())) };
        if !self.fifo.acquire_timeout(Duration::from_secs(300)) {
            return Err(Fail::Busy("a request is still running; try again when it has finished".into()));
        }
        let _f = crate::sync::SemGuard(&self.fifo);
        if !self.loaded() {
            *self.vram_reserve.lock().unwrap() = Some(reserve_mib);
            return Ok(json!({"status": "not loaded", "reserve_mib": reserve_mib, "note": "applied when the model loads"}));
        }
        let out = s.vram(reserve_mib, 120.0)?;
        *self.vram_reserve.lock().unwrap() = Some(reserve_mib);
        let g = |k: &str| py::py_str(out.get(k).unwrap_or(&Value::Null));
        say(&format!("VRAM: {} MiB free, expert cache {} of {} MiB ({} experts)", g("vram_free_mib"), g("expert_cache_mib"), g("expert_cache_full_mib"), g("expert_slots")));
        let mut r = Map::new();
        r.insert("status".into(), json!("ok"));
        r.extend(out);
        Ok(Value::Object(r))
    }

    pub fn say_died(&self, e: &Fail) {
        let note = self.engine.strata().map(|s| s.death_note()).unwrap_or_default();
        let log = self.engine.strata().and_then(|s| s.log_path.clone());
        say(&format!("{e}. {note} The next request starts the engine again.{}", log.map(|l| format!(" Its log: {l}")).unwrap_or_default()));
    }

    /// POST /load and every generation request: start the engine now if it is unloaded.
    pub fn load(&self) -> Result<(), Fail> {
        self.touch();
        if self.loaded() && !self.vision_down() {
            return Ok(());
        }
        let trace = trace();
        let waiting = Instant::now();
        self.queued_add(1);
        self.fifo.acquire();
        let _f = crate::sync::SemGuard(&self.fifo);
        self.queued_add(-1);
        let loading = Instant::now();
        if let Some(t) = &trace {
            let mut t = t.lock().unwrap();
            add_f(&mut t, "queue_s", pyround(waiting.elapsed().as_secs_f64(), 3));
            t.insert("state".into(), json!("loading"));
        }
        let r = self.ensure_loaded();
        if let Some(t) = &trace {
            let mut t = t.lock().unwrap();
            add_f(&mut t, "load_s", pyround(loading.elapsed().as_secs_f64(), 3));
            t.insert("state".into(), json!("queued"));
        }
        r
    }

    /// "unloaded", "not loaded", "busy" or "unsupported".
    pub fn unload(&self, idle_for: Option<f64>) -> Result<&'static str, Fail> {
        if self.engine.strata().is_none() {
            return Ok("unsupported");
        }
        if !self.fifo.try_acquire() {
            return Ok("busy");
        }
        let _f = crate::sync::SemGuard(&self.fifo);
        self.unload_held(idle_for, false)
    }

    /// unload() for a caller that holds the FIFO already (a picture making room for itself).
    pub fn unload_held(&self, idle_for: Option<f64>, waiting_ok: bool) -> Result<&'static str, Fail> {
        let Some(s) = self.engine.strata() else { return Ok("unsupported") };
        if !s.alive() {
            return Ok("not loaded");
        }
        {
            let st = self.state.lock().unwrap();
            let busy = truthy(st.status.get("busy").unwrap_or(&Value::Null));
            let queued = st.status.get("queued").and_then(Value::as_i64).unwrap_or(0);
            if busy || st.entering != 0 || (queued != 0 && !waiting_ok) {
                return Ok("busy");
            }
            if let Some(idle) = idle_for {
                if now() - st.last_request_at.unwrap_or(self.started_at) < idle {
                    return Ok("busy");
                }
            }
        }
        s.unload()?;
        if let Some(v) = &self.vision {
            v.unload();
        }
        say(&format!("model unloaded{}; the next request loads it again", idle_for.filter(|i| *i != 0.0).map(|i| format!(" after {i:.0} s idle")).unwrap_or_default()));
        Ok("unloaded")
    }

    pub fn start_idle_unload(self: &Arc<Self>) {
        let idle = self.set.idle_unload_s;
        if idle == 0.0 || self.engine.strata().is_none() {
            return;
        }
        say(&format!("the model unloads after {} s without requests", py_num(idle)));
        let me = self.clone();
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_secs_f64((idle / 4.0).clamp(1.0, 30.0)));
            if let Err(e @ Fail::Stuck(_)) = me.unload(Some(idle)) {
                say(&format!("idle unload: {e}"));
            }
        });
    }

    /// The Chat settings every client gets for what it leaves out; {} / null = clients use their own again.
    pub fn set_shared(&self, defaults: &Value) -> Result<Map<String, Value>, Fail> {
        let shared = clean_shared_defaults(defaults)?;
        *self.shared.lock().unwrap() = shared.clone();
        if let Some(p) = &self.set.shared_path {
            let r = if shared.is_empty() {
                match std::fs::remove_file(p) {
                    Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e),
                    _ => Ok(()),
                }
            } else {
                std::fs::write(p, pyjson::dumps_with(&Value::Object(shared.clone()), pyjson::Opts { indent: Some(1), ..Default::default() }))
            };
            if let Err(e) = r {
                say(&format!("could not save the shared settings: {e}"));
            }
        }
        Ok(shared)
    }

    /// The request with the shared thinking level and max tokens filled in where it has none of its own.
    pub fn with_shared(&self, req: &Map<String, Value>, api: &str) -> Map<String, Value> {
        let s = self.shared.lock().unwrap().clone();
        let effort = s.get("reasoning_effort").and_then(Value::as_str).map(String::from).filter(|e| !e.is_empty()).or_else(|| self.set.config_effort.clone());
        if s.is_empty() && effort.is_none() {
            return req.clone();
        }
        let mut req = req.clone();
        let r = Value::Object(req.clone());
        if let Some(mt) = s.get("max_tokens") {
            if !truthy(get(&r, "max_tokens")) && !truthy(get(&r, "max_completion_tokens")) {
                req.insert("max_tokens".into(), mt.clone());
            }
        }
        if let Some(effort) = effort {
            if api == "openai" {
                let ctk = get(&r, "chat_template_kwargs");
                let ctk_has = |k: &str| ctk.as_object().map(|o| o.contains_key(k)).unwrap_or(false);
                if !truthy(get(&r, "reasoning_effort")) && !truthy(get(&r, "reasoning")) && !ctk_has("enable_thinking") && !ctk_has("reasoning_effort") {
                    req.insert("reasoning_effort".into(), json!(effort));
                }
            } else if !truthy(get(&r, "thinking")) && !truthy(get(&r, "output_config")) {
                if effort == "none" {
                    req.insert("thinking".into(), json!({"type": "disabled"}));
                } else {
                    req.insert("output_config".into(), json!({"effort": effort}));
                }
            }
        }
        req
    }

    /// tok/s over the last RATE_WINDOW_S seconds (with parallel requests: the sum of theirs).
    pub fn tok_s(&self) -> f64 {
        let st = self.state.lock().unwrap();
        if !st.live_reqs.is_empty() {
            return st.live_reqs.values().map(|(s, r)| rate_of(s, r)).sum();
        }
        rate_of(&st.status, &st.rate)
    }

    pub fn tok_s_mean(&self) -> f64 {
        let st = self.state.lock().unwrap();
        let many: Vec<&Map<String, Value>> = if st.live_reqs.is_empty() { vec![&st.status] } else { st.live_reqs.values().map(|(s, _)| s).collect() };
        let t = now();
        many.iter()
            .filter(|s| truthy(s.get("busy").unwrap_or(&Value::Null)) && s.get("first_token").map(truthy).unwrap_or(false))
            .map(|s| s.get("generated").and_then(Value::as_f64).unwrap_or(0.0) / (t - s["first_token"].as_f64().unwrap_or(t)).max(1e-6))
            .sum()
    }

    fn prefill_tok_s_mean(&self) -> Option<f64> {
        let reading = {
            let st = self.state.lock().unwrap();
            truthy(st.status.get("busy").unwrap_or(&Value::Null)) && st.status.get("first_token").map(Value::is_null).unwrap_or(true)
        };
        if reading { self.engine.live().prefill_tok_s_mean } else { Some(0.0) }
    }

    /// The hardware sampler behind GET /metrics.
    pub fn start_telemetry(self: &Arc<Self>) {
        let mut t = self.telemetry.lock().unwrap();
        if t.is_some() {
            return;
        }
        let me = Arc::downgrade(self);
        let tel = crate::telemetry::Telemetry::start(Box::new(move || {
            let Some(s) = me.upgrade() else { return Map::new() };
            let mut m = Map::new();
            m.insert("tok_s".into(), json!(s.tok_s()));
            m.insert("tok_s_mean".into(), json!(s.tok_s_mean()));
            m.insert("prefill_tok_s_mean".into(), s.prefill_tok_s_mean().map(Value::from).unwrap_or(Value::Null));
            m
        }));
        if let Some(e) = self.engine.strata() {
            let tw = Arc::downgrade(&tel);
            *e.gpu_busy.lock().unwrap() = Some(Box::new(move || tw.upgrade().map(|t| t.snapshot()["now"]["gpu_util"].as_f64().unwrap_or(0.0) >= 10.0).unwrap_or(false)));
        }
        *t = Some(tel);
    }

    fn telemetry_snapshot(&self) -> Value {
        match self.telemetry.lock().unwrap().clone() {
            Some(t) => t.snapshot(),
            None => json!({"now": {}, "history": {}, "static": {}}),
        }
    }

    /// #332: a monitor record for this request, or None when the monitor is off.
    pub fn begin_request(&self, path: &str, req: &Value) -> Option<Rec> {
        if !self.set.api_monitor {
            return None;
        }
        let raw = pyjson::dumps_with(req, pyjson::Opts { ensure_ascii: false, indent: Some(2), ..Default::default() });
        let input: String = raw.chars().take(262144).collect();
        let rf = get(req, "response_format");
        let rec = json!({"id": &uuid::Uuid::new_v4().simple().to_string()[..12], "path": path,
            "model": if truthy(get(req, "model")) { get(req, "model").clone() } else { json!(self.model) },
            "started_at": now(), "state": "queued", "stream": truthy(get(req, "stream")),
            "response_format": if rf.is_object() { get(rf, "type").clone() } else { Value::Null },
            "input": input, "input_truncated": raw.chars().count() > 262144,
            "output": "", "reasoning": "", "output_truncated": false, "reasoning_truncated": false,
            "queue_s": 0.0, "load_s": 0.0, "first_token_s": null, "_clock": clock()});
        let rec: Rec = Arc::new(Mutex::new(rec.as_object().cloned().unwrap()));
        let mut st = self.state.lock().unwrap();
        st.api_requests.push_back(rec.clone());
        while st.api_requests.len() > 100 {
            st.api_requests.pop_front();
        }
        drop(st);
        TRACE.with(|t| *t.borrow_mut() = Some(rec.clone()));
        Some(rec)
    }

    pub fn request_records(&self, request_id: Option<&str>) -> Option<Value> {
        let st = self.state.lock().unwrap();
        let records: Vec<Rec> = st.api_requests.iter().cloned().collect();
        drop(st);
        if let Some(id) = request_id {
            let r = records.iter().find(|r| r.lock().unwrap().get("id").and_then(Value::as_str) == Some(id))?;
            let r = r.lock().unwrap();
            return Some(Value::Object(r.iter().filter(|(k, _)| !k.starts_with('_')).map(|(k, v)| (k.clone(), v.clone())).collect()));
        }
        let c = clock();
        Some(Value::Array(
            records
                .iter()
                .rev()
                .map(|r| {
                    let r = r.lock().unwrap();
                    let mut m: Map<String, Value> = r.iter().filter(|(k, _)| !["input", "output", "reasoning", "response"].contains(&k.as_str()) && !k.starts_with('_')).map(|(k, v)| (k.clone(), v.clone())).collect();
                    let wc = r.get("wallclock_s").cloned().unwrap_or_else(|| json!(pyround(c - r["_clock"].as_f64().unwrap_or(c), 3)));
                    m.insert("wallclock_s".into(), wc);
                    Value::Object(m)
                })
                .collect(),
        ))
    }

    /// GET /metrics: the engine's facts, what it is doing, the last requests, and the hardware.
    pub fn metrics(&self, all_requests: bool) -> Value {
        let (mut s, hist, totals, running_n) = {
            let st = self.state.lock().unwrap();
            let mut s = st.status.clone();
            if let Some((newest, _)) = st.live_reqs.values().last() {
                for (k, v) in newest {
                    s.insert(k.clone(), v.clone());
                }
                s.insert("queued".into(), st.status.get("queued").cloned().unwrap_or(json!(0)));
            }
            (s, st.history.iter().cloned().collect::<Vec<_>>(), st.totals.clone(), st.live_reqs.len())
        };
        let t = now();
        let live_e: Live = self.engine.live();
        let busy = truthy(s.get("busy").unwrap_or(&Value::Null));
        let state = if busy && s.get("first_token").map(Value::is_null).unwrap_or(true) {
            "reading"
        } else if busy {
            "generating"
        } else if !self.loaded() {
            "unloaded"
        } else {
            "idle"
        };
        let b = |k: &str| if busy { s.get(k).cloned().unwrap_or(Value::Null) } else { Value::Null };
        let mut live = json!({"state": state, "queued": s.get("queued").cloned().unwrap_or(json!(0)), "phase": b("phase"),
            "prompt_tokens": b("prompt_tokens"), "prompt_read": null, "prompt_total": null, "generated": b("generated"),
            "max_tokens": b("max_tokens"),
            "elapsed_s": if busy && s.get("started").map(truthy).unwrap_or(false) { json!(round1(t - s["started"].as_f64().unwrap_or(t))) } else { Value::Null },
            "tok_s": if state == "generating" { json!(round1(self.tok_s())) } else { Value::Null },
            "tok_s_mean": if state == "generating" { json!(round1(self.tok_s_mean())) } else { Value::Null },
            "prefill_tok_s_mean": if busy { live_e.prefill_tok_s_mean.map(Value::from).unwrap_or(Value::Null) } else { Value::Null },
            "tok_s_window_s": if state == "generating" { json!(RATE_WINDOW_S) } else { Value::Null }});
        if state == "reading" {
            if let Some((a, b)) = live_e.progress {
                live["prompt_read"] = json!(a);
                live["prompt_total"] = json!(b);
            }
        }
        let par = self.engine.batch();
        if par > 0 {
            if let Some(e) = self.engine.strata() {
                let slots = e.slots_view();
                let in_slots = slots.iter().filter(|x| x["state"] != "idle").count();
                let lo = live.as_object_mut().unwrap();
                lo.insert("parallel".into(), json!(par));
                lo.insert("running".into(), json!(running_n));
                lo.insert("slots".into(), json!(slots));
                lo.insert("outside_slots".into(), json!(running_n.saturating_sub(in_slots)));
                lo.insert("waiting".into(), json!(e.waiting()));
                lo.insert("admitting".into(), json!(e.admitting()));
                if running_n > 0 && state == "idle" {
                    lo.insert("state".into(), json!("generating"));
                }
            }
        }
        let mut engine = json!({"model": self.model, "max_context": self.reported_ctx(), "images": self.vision.is_some()}).as_object().cloned().unwrap();
        for (k, v) in self.engine.info() {
            engine.insert(k, v);
        }
        let tel = self.telemetry_snapshot();
        let parked = {
            let e = self.engine.strata();
            self.conv_log.lock().unwrap().poll(e.and_then(|e| e.log_path.as_deref()), e.map(|e| e.log_start.load(SeqCst)))
        };
        s.shift_remove("x");
        let n = if all_requests { hist.len() } else { 12 };
        let requests: Vec<Value> = hist.iter().rev().take(n).cloned().collect();
        json!({"engine": engine, "live": live, "requests": requests,
            "conversation_cache": conversation_cache_view(&engine, &hist, &totals, &parked),
            "requests_kept": hist.len(), "totals": totals, "hardware": tel["now"], "hardware_static": tel["static"],
            "history": tel["history"], "time": t})
    }

    /// GET /metrics/prometheus: llama-server's metric names (llamacpp:*), the drafts and requests as strata:*.
    pub fn prometheus(&self) -> String {
        let (s, t, running) = {
            let st = self.state.lock().unwrap();
            let busy = truthy(st.status.get("busy").unwrap_or(&Value::Null));
            (st.status.clone(), st.totals.clone(), if st.live_reqs.is_empty() { busy as usize } else { st.live_reqs.len() })
        };
        let i = |k: &str| t.get(k).and_then(Value::as_i64).unwrap_or(0);
        let f = |k: &str| t.get(k).and_then(Value::as_f64).unwrap_or(0.0);
        let prompt_n = i("prompt_tokens") - i("reused");
        let (prompt_s, decode_s) = (f("prompt_ms") / 1000.0, f("decode_ms") / 1000.0);
        let fr = crate::pyjson::float_repr;
        let rows: Vec<(&str, &str, &str, String)> = vec![
            ("llamacpp:prompt_tokens_total", "counter", "Prompt tokens read", prompt_n.to_string()),
            ("llamacpp:prompt_tokens_cached_total", "counter", "Prompt tokens the conversation cache already held", i("reused").to_string()),
            ("llamacpp:prompt_seconds_total", "counter", "Prompt read time", fr(prompt_s)),
            ("llamacpp:tokens_predicted_total", "counter", "Tokens generated", i("output_tokens").to_string()),
            ("llamacpp:tokens_predicted_seconds_total", "counter", "Generation time", fr(decode_s)),
            ("llamacpp:prompt_tokens_seconds", "gauge", "Average prompt read throughput in tokens/s", if prompt_s != 0.0 { fr(prompt_n as f64 / prompt_s) } else { "0".into() }),
            ("llamacpp:predicted_tokens_seconds", "gauge", "Average generation throughput in tokens/s", if decode_s != 0.0 { fr(i("output_tokens") as f64 / decode_s) } else { "0".into() }),
            ("llamacpp:requests_processing", "gauge", "Requests running", running.to_string()),
            ("llamacpp:requests_deferred", "gauge", "Requests waiting their turn", s.get("queued").and_then(Value::as_i64).unwrap_or(0).to_string()),
            ("strata:requests_total", "counter", "Requests finished", i("requests").to_string()),
            ("strata:drafts_offered_total", "counter", "Speculative draft tokens offered", i("drafts_offered").to_string()),
            ("strata:drafts_accepted_total", "counter", "Speculative draft tokens accepted", i("drafts_accepted").to_string()),
            ("strata:model_loaded", "gauge", "1 while the model is loaded", (self.loaded() as i32).to_string()),
        ];
        rows.iter().map(|(n, k, h, v)| format!("# HELP {n} {h}\n# TYPE {n} {k}\n{n} {v}\n")).collect()
    }

    /// GET /v1/status: what this server is and does.
    pub fn v1_status(&self) -> Value {
        let (s, totals, last_t, last_at) = {
            let st = self.state.lock().unwrap();
            (st.status.clone(), st.totals.clone(), st.last_timings.clone(), st.last_request_at)
        };
        let tel = self.telemetry_snapshot();
        let (hw, stat) = (&tel["now"], &tel["static"]);
        let scaled = |v: &Value, unit: f64, digits: i32| -> Value {
            match v.as_f64() {
                Some(x) if v.is_number() => json!(pyround(x / unit, digits)), // round(x, n) is a float in Python
                _ => Value::Null,
            }
        };
        let busy = truthy(s.get("busy").unwrap_or(&Value::Null));
        let ctx = self.reported_ctx();
        let images = self.vision.is_some();
        let info = self.engine.info();
        let conc = self.engine.batch().max(1);
        let mut vram = info.get("vram").and_then(Value::as_object).cloned().unwrap_or_default();
        vram.insert("elastic".into(), json!(info.get("vram_elastic").map(truthy).unwrap_or(false)));
        let queued = s.get("queued").and_then(Value::as_i64).unwrap_or(0);
        json!({"service": "strata", "model": self.model, "loaded": self.loaded(), "auto_load": self.engine.strata().is_some(),
            "structured_output": {"formats": ["json_object", "json_schema"], "method": "prompt_and_validate", "constrained_decoding": false, "stream_buffered": true},
            "engine": info.get("version").cloned().unwrap_or(Value::Null),
            "started": self.started_at as i64, "uptime_s": (now() - self.started_at) as i64,
            "cache_max_tokens": ctx, "context": {"native": ctx, "max_positions": ctx},
            "concurrency": {"serving": conc, "requested": conc},
            "dialects": ["/v1/chat/completions", "/v1/messages", "/v1/responses"],
            "vision": {"enabled": images, "available": images, "error": null},
            "activity": {"requests": totals.get("requests").and_then(Value::as_i64).unwrap_or(0) + busy as i64, "in_flight": busy as i64 + queued,
                         "last_request_at": last_at.map(|a| json!(a as i64)).unwrap_or(Value::Null)},
            "last_timings": last_t, "vram": vram,
            "machine": {"at": now() as i64,
                "gpu": if truthy(&stat["gpu_name"]) { json!({"name": stat["gpu_name"], "used_mib": scaled(&hw["gpu_mem_used"], 1048576.0, 0), "total_mib": scaled(&hw["gpu_mem_total"], 1048576.0, 0), "util_pct": hw["gpu_util"], "temp_c": hw["gpu_temp"], "power_w": hw["gpu_power"]}) } else { Value::Null },
                "ram": if truthy(&hw["ram_total"]) { json!({"used_gib": scaled(&hw["ram_used"], 1073741824.0, 1), "total_gib": scaled(&hw["ram_total"], 1073741824.0, 1)}) } else { Value::Null }}})
    }

    fn render(&self, messages: &[Value], tools: Option<&[Value]>, add: bool, kwargs: &Map<String, Value>) -> Result<String, Fail> {
        let mut kw = kwargs.clone();
        let add = match kw.shift_remove("add_generation_prompt") {
            Some(v) => truthy(&v),
            None => add,
        };
        self.template.render(messages, tools, add, &kw).map_err(|e| match e {
            crate::template::TemplateError::Request(m) => Fail::Value(m),
            crate::template::TemplateError::Broken(m) => Fail::Value(format!("the request is malformed (TemplateError: {m})")),
        })
    }

    /// The template rendered; #458: with effort_end, a non-default effort goes in a short system turn before the answer.
    pub fn render_prompt(&self, messages: &[Value], tools: Option<&[Value]>, kwargs: &Map<String, Value>) -> Result<String, Fail> {
        let effort = kwargs.get("reasoning_effort").and_then(Value::as_str);
        let off = kwargs.get("enable_thinking") == Some(&Value::Bool(false));
        if !self.set.effort_end || (!off && matches!(effort, None | Some("" | "xhigh" | "high"))) {
            return self.render(messages, tools, true, kwargs);
        }
        let base: Map<String, Value> = kwargs.iter().filter(|(k, _)| *k != "reasoning_effort" && *k != "enable_thinking").map(|(k, v)| (k.clone(), v.clone())).collect();
        let history = self.render(messages, tools, false, &base)?;
        let full = self.render(messages, tools, true, kwargs)?;
        let mine = self.render(messages, tools, false, kwargs)?;
        if !full.starts_with(&mine) {
            return Ok(full);
        }
        let tail = if off { String::new() } else { EFFORT_TURN.replace("{}", effort_text(effort.unwrap_or("medium"))) };
        Ok(format!("{history}{tail}{}", &full[mine.len()..]))
    }

    /// The request's prompt ids: the template rendered and tokenized, literal tags in message text kept as text.
    pub fn encode_prompt(&self, messages: &[Value], tools: Option<&[Value]>, kwargs: &Map<String, Value>) -> Result<Vec<u32>, Fail> {
        let tv = tools.map(|t| Value::Array(t.to_vec()));
        let (marked, marked_tools, changed) = mark_think_literals(messages, tv.as_ref(), &self.literals);
        let mt: Option<Vec<Value>> = marked_tools.and_then(|v| v.as_array().cloned());
        let prompt = self.render_prompt(&marked, mt.as_deref(), kwargs)?;
        if !changed {
            return Ok(self.tok.encode(&prompt, true));
        }
        let (prompt, plain) = unmark_think_literals(&prompt, &self.literals);
        Ok(self.tok.encode_plain(&prompt, true, &plain))
    }

    /// A picture a tool returned that this server cannot read becomes a short note in its place instead of a 400.
    fn note_unreadable_tool_images(&self, messages: &mut [Value]) -> std::collections::HashMap<String, Vec<u8>> {
        let mut fetched = std::collections::HashMap::new();
        for m in messages.iter_mut() {
            if get(m, "role").as_str() != Some("tool") {
                continue;
            }
            let Some(content) = m.get_mut("content").and_then(Value::as_array_mut) else { continue };
            for item in content.iter_mut() {
                if get(item, "type").as_str() != Some("image") {
                    continue;
                }
                let mut why = if self.vision.is_none() { Some("this server has no image encoder".to_string()) } else { None };
                if why.is_none() {
                    let src = py::py_str(get(item, "source"));
                    let r = (|| -> Result<(), String> {
                        let s = if src.starts_with("http://") || src.starts_with("https://") {
                            let b = Vision::download(&src)?;
                            fetched.insert(src.clone(), b.clone());
                            ImgSrc::Bytes(b)
                        } else {
                            ImgSrc::Str(src.clone())
                        };
                        self.fifo.acquire();
                        let _f = crate::sync::SemGuard(&self.fifo);
                        self.vision.as_ref().unwrap().encode(&s, &[]).map(|_| ())
                    })();
                    if let Err(e) = r {
                        why = Some(e);
                    }
                }
                if let Some(w) = why {
                    say(&format!("a picture a tool returned was left out of the prompt: {w}"));
                    *item = json!({"type": "text", "text": format!("[image omitted: {w}]")});
                }
            }
        }
        fetched
    }

    /// -> (ids, thinking, max_new).  An unset or non-positive max_new means the rest of the context.
    pub fn prepare(&self, messages: &[Value], tools: Option<&[Value]>, kwargs: &Map<String, Value>, max_new: Option<i64>, force: Option<&str>, req: Option<&mut Map<String, Value>>) -> Result<(Vec<u32>, bool, i64), Fail> {
        let mut messages = messages.to_vec();
        let fetched = self.note_unreadable_tool_images(&mut messages);
        let mut kwargs = kwargs.clone();
        let mut force = force.map(String::from);
        if self.harmony || self.glm {
            let off = kwargs.get("enable_thinking") == Some(&Value::Bool(false));
            let effort: Option<String> = if off { Some("low".into()) } else { kwargs.get("reasoning_effort").and_then(|v| v.as_str().map(String::from)) };
            let mapped: Value = if self.harmony {
                match effort.as_deref() {
                    Some("xhigh") => json!("high"),
                    Some("none") => json!("low"),
                    None | Some("") => json!("medium"),
                    Some(e) => json!(e),
                }
            } else {
                match effort.as_deref() {
                    Some("none") => json!("low"),
                    Some("medium") => json!("high"),
                    Some("xhigh") => json!("max"),
                    Some("") => Value::Null,
                    None => Value::Null,
                    Some(e) => json!(e),
                }
            };
            kwargs.insert("enable_thinking".into(), json!(true));
            kwargs.insert("reasoning_effort".into(), mapped);
            force = None;
        }
        let mut ids = self.encode_prompt(&messages, tools, &kwargs)?;
        let thinking_off = kwargs.get("enable_thinking") == Some(&Value::Bool(false));
        if let Some(f) = &force {
            if thinking_off {
                ids.extend(self.tok.encode(f, true));
            }
        }
        EMBEDDINGS.with(|e| *e.borrow_mut() = None);
        let images = images_of(&messages);
        let mut encoded: Vec<(PathBuf, i64)> = vec![];
        if !images.is_empty() {
            let Some(vision) = &self.vision else {
                return Err(Fail::Value("this server was started without the vision encoder (run setup again and choose 'vision'), so it cannot read images".into()));
            };
            let pad = self.tok.encode(IMAGE_PAD, true)[0];
            let start = self.tok.encode(VISION_START, true)[0];
            let mut srcs = vec![];
            for src in &images {
                srcs.push(if let Some(b) = fetched.get(src) {
                    ImgSrc::Bytes(b.clone())
                } else if src.starts_with("http://") || src.starts_with("https://") {
                    ImgSrc::Bytes(Vision::download(src).map_err(Fail::Value)?)
                } else {
                    ImgSrc::Str(src.clone())
                });
            }
            {
                self.fifo.acquire();
                let _f = crate::sync::SemGuard(&self.fifo);
                encoded = vision.encode_all(&srcs).map_err(Fail::Value)?;
            }
            let literal = self.tok.encode(IMAGE_PAD, false);
            let mut out = vec![];
            let mut k = 0;
            for (j, &t) in ids.iter().enumerate() {
                if t == pad && j > 0 && ids[j - 1] == start {
                    if k == encoded.len() {
                        return Err(Fail::Value("the prompt and its images do not match".into()));
                    }
                    out.extend(std::iter::repeat_n(pad, encoded[k].1 as usize));
                    k += 1;
                } else if t == pad {
                    out.extend(&literal);
                } else {
                    out.push(t);
                }
            }
            if k != encoded.len() {
                return Err(Fail::Value("the prompt and its images do not match".into()));
            }
            ids = out;
        }
        let mut ctx = self.engine.max_context();
        if ctx <= 0 {
            if self.engine.starting() {
                return Err(Fail::Starting("the engine is starting (a minute or two); try again shortly".into()));
            }
            ctx = self.engine.known_ctx();
            if ctx <= 0 {
                return Err(Fail::Starting("the engine is starting (a minute or two); try again shortly".into()));
            }
        }
        let n = ids.len() as i64;
        let room = ctx - CTX_SLACK - n;
        let max_new = match max_new {
            Some(m) if m > 0 && !(self.set.fit_max_tokens && room < 1) => {
                if m > room {
                    if !self.set.fit_max_tokens {
                        return Err(Fail::Value(format!("request exceeds the context window: prompt ({n} tokens) + max tokens ({m}) exceeds the context ({ctx}); requests are never truncated. Send a smaller max_tokens (at most {} here), or add \"fit_max_tokens\": true to the model's strata-<model>.json to shorten it to the room left (#545)", room.max(0))));
                    }
                    room.max(1)
                } else {
                    m
                }
            }
            _ => {
                if room < 1 {
                    return Err(Fail::Value(format!("request exceeds the context window: prompt ({n} tokens) leaves no room to answer in the context ({ctx}); requests are never truncated")));
                }
                room
            }
        };
        if !images.is_empty() {
            let vision = self.vision.as_ref().unwrap();
            let size: u64 = encoded.iter().map(|(p, _)| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)).sum();
            let combined = crate::vision::combined_embeddings_path(&vision.dir, size);
            EMBEDDINGS.with(|e| *e.borrow_mut() = Some(combined.clone()));
            let mut all = vec![];
            for (p, _) in &encoded {
                all.extend(std::fs::read(p).map_err(|e| Fail::Value(format!("the image could not be read ({e})")))?);
            }
            crate::vision::write_private(&combined, &all).map_err(|e| Fail::Value(format!("the image could not be written ({e})")))?;
        }
        if let Some(req) = req {
            if req.get("strata_prefix").map(|v| !v.is_null()).unwrap_or(false) {
                let spec = req["strata_prefix"].clone();
                let r = self.resolve_prefix(&spec, &messages, tools, &kwargs, &ids, !images.is_empty())?;
                req.insert("strata_prefix".into(), r);
            }
        }
        Ok((ids, !thinking_off, max_new))
    }

    /// R1: `"strata_prefix"` marks the start of the prompt as a shared read-only prefix -> {"tokens": N} or {}.
    pub fn resolve_prefix(&self, spec: &Value, messages: &[Value], tools: Option<&[Value]>, kwargs: &Map<String, Value>, ids: &[u32], has_images: bool) -> Result<Value, Fail> {
        let keys = ["messages", "message", "chars", "tokens"];
        let shape = r#"strata_prefix: {"messages": K} (the first K messages), {"message": I, "chars": N} (the first N characters of message I) or {"tokens": N}"#;
        let Some(o) = spec.as_object().filter(|o| !o.is_empty()) else { return Err(Fail::Value(shape.into())) };
        let mut set: Vec<&str> = o.keys().map(String::as_str).collect();
        set.sort();
        if o.keys().any(|k| !keys.contains(&k.as_str())) || !(set == ["messages"] || set == ["tokens"] || set == ["chars", "message"]) {
            return Err(Fail::Value(shape.into()));
        }
        for (k, v) in o {
            let lo = if k == "message" { 0 } else { 1 };
            if !(v.is_i64() || v.is_u64()) || v.as_i64().unwrap_or(-1) < lo {
                return Err(Fail::Value(format!("strata_prefix.{k}: a whole number ({lo} or more)")));
            }
        }
        if has_images {
            say("strata_prefix ignored: the request has images");
            return Ok(json!({}));
        }
        let what = o.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(",");
        let lcp = |a: &[u32], b: &[u32]| a.iter().zip(b).take_while(|(x, y)| x == y).count();
        let head_ids = |msgs: &[Value]| -> Result<Vec<u32>, Fail> {
            let tv = tools.map(|t| Value::Array(t.to_vec()));
            let (marked, mtools, changed) = mark_think_literals(msgs, tv.as_ref(), &self.literals);
            let mt: Option<Vec<Value>> = mtools.and_then(|v| v.as_array().cloned());
            let effort = kwargs.get("reasoning_effort").and_then(Value::as_str);
            let off = kwargs.get("enable_thinking") == Some(&Value::Bool(false));
            let mut kw: Map<String, Value> = if self.set.effort_end && (off || !matches!(effort, None | Some("" | "xhigh" | "high"))) {
                kwargs.iter().filter(|(k, _)| *k != "reasoning_effort" && *k != "enable_thinking").map(|(k, v)| (k.clone(), v.clone())).collect()
            } else {
                kwargs.clone()
            };
            kw.insert("add_generation_prompt".into(), json!(false));
            let mut text = self.render(&marked, mt.as_deref(), false, &kw)?;
            if let Some(cut) = text.find(PREFIX_END) {
                text.truncate(cut);
            }
            if changed {
                let (t, plain) = unmark_think_literals(&text, &self.literals);
                return Ok(self.tok.encode_plain(&t, true, &plain));
            }
            Ok(self.tok.encode(&text, true))
        };
        let n = if let Some(t) = o.get("tokens") {
            t.as_i64().unwrap() as usize
        } else if let Some(k) = o.get("messages") {
            let k = k.as_i64().unwrap() as usize;
            if k >= messages.len() {
                say(&format!("strata_prefix.messages={k} ignored: the request has {} message(s) and the prefix must leave at least the last one", messages.len()));
                return Ok(json!({}));
            }
            lcp(&head_ids(&messages[..k])?, ids)
        } else {
            let i = o["message"].as_i64().unwrap() as usize;
            let chars = o["chars"].as_i64().unwrap() as usize;
            let text = messages.get(i).map(|m| get(m, "content").clone()).unwrap_or(Value::Null);
            let Some(text) = text.as_str().filter(|t| chars <= t.chars().count()) else {
                say(&format!("strata_prefix ignored: message {i} has no text of {chars} characters"));
                return Ok(json!({}));
            };
            let mut cut: Vec<Value> = messages[..i].to_vec();
            let mut m = messages[i].as_object().cloned().unwrap_or_default();
            m.insert("content".into(), json!(format!("{}{PREFIX_END}", py::chars_prefix(text, chars))));
            cut.push(Value::Object(m));
            lcp(&head_ids(&cut)?, ids)
        };
        if n < 1 || n >= ids.len().saturating_sub(1) {
            say(&format!("strata_prefix {what} ignored: it leaves no usable prefix of the {}-token prompt (common start {n})", ids.len()));
            return Ok(json!({}));
        }
        Ok(json!({"tokens": n}))
    }

    /// Delete the combined image file prepare() wrote when no run() took it over.
    pub fn drop_embeddings(&self) {
        if let Some(p) = EMBEDDINGS.with(|e| e.borrow_mut().take()) {
            let _ = std::fs::remove_file(p);
        }
    }
}

fn is_special(tok: &dyn Tokenize, s: &str) -> bool {
    // getattr(tokenizer, "special_tokens", {}): a literal the vocabulary has as one CONTROL / USER_DEFINED token
    tok.control_tokens().iter().any(|c| c == s) || (tok.encode(s, true).len() == 1 && tok.encode(s, false).len() == 1 && s.starts_with("<|"))
}

fn rate_of(s: &Map<String, Value>, rate: &VecDeque<(f64, i64)>) -> f64 {
    if !truthy(s.get("busy").unwrap_or(&Value::Null)) || !s.get("first_token").map(truthy).unwrap_or(false) {
        return 0.0;
    }
    let t = now();
    let newest = rate.back();
    let oldest = rate.iter().find(|(ts, _)| t - ts <= RATE_WINDOW_S);
    if let (Some(n), Some(o)) = (newest, oldest) {
        if n.0 - o.0 >= RATE_MIN_SPAN_S {
            return ((n.1 - o.1) as f64 / (n.0 - o.0)).max(0.0);
        }
    }
    s.get("generated").and_then(Value::as_f64).unwrap_or(0.0) / RATE_MIN_SPAN_S.max(t - s["first_token"].as_f64().unwrap_or(t))
}

/// #596: the Monitor's Conversation cache card.
pub fn conversation_cache_view(info: &Map<String, Value>, hist: &[Value], totals: &Map<String, Value>, parked: &Map<String, Value>) -> Value {
    let mib = info.get("conversation_cache_mib").filter(|v| v.is_i64());
    let last = hist.last();
    let mut out = json!({"enabled": mib.and_then(Value::as_i64).map(|m| m > 0).unwrap_or(false), "budget_mib": mib.cloned().unwrap_or(Value::Null),
        "slots": info.get("conversation_cache_slots").cloned().unwrap_or(Value::Null)}).as_object().cloned().unwrap();
    for (k, v) in parked {
        out.insert(k.clone(), v.clone());
    }
    out.insert("requests".into(), json!(hist.len()));
    out.insert("requests_reused".into(), json!(hist.iter().filter(|r| r["reused"].as_i64().unwrap_or(0) > 0).count()));
    out.insert("reused_tokens".into(), totals.get("reused").cloned().unwrap_or(json!(0)));
    out.insert("prompt_tokens".into(), totals.get("prompt_tokens").cloned().unwrap_or(json!(0)));
    out.insert("last_reused".into(), last.map(|l| l["reused"].clone()).unwrap_or(Value::Null));
    out.insert("last_prompt".into(), last.map(|l| l["prompt_tokens"].clone()).unwrap_or(Value::Null));
    Value::Object(out)
}

/// time.perf_counter()
pub fn clock() -> f64 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64()
}

pub fn add_f(m: &mut Map<String, Value>, k: &str, d: f64) {
    let v = m.get(k).and_then(Value::as_f64).unwrap_or(0.0) + d;
    m.insert(k.into(), json!(v));
}

/// A number as Python formats it in an f-string (int stays int, float repr).
pub fn py_num(x: f64) -> String {
    if x == x.trunc() && x.abs() < 1e16 { format!("{}", x as i64) } else { crate::pyjson::float_repr(x) }
}
