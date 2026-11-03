//! What the Monitor shows beyond speed (docs/OBSERVABILITY.md): the requests the server refused, latency as percentiles
//! (vLLM's and OpenTelemetry's GenAI metrics: time to first token, time per output token, end to end), and each MCP
//! tool's calls.  Process-wide: a refusal can come before there is a Service to hold it.

use serde_json::{json, Map, Value};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::Mutex;

const RECENT: usize = 50;

#[derive(Default)]
pub struct Failures {
    /// (status, error type) -> count, in the order first seen
    counts: Vec<((u16, String), u64)>,
    recent: VecDeque<Value>,
}

impl Failures {
    pub fn add(&mut self, time: f64, path: &str, status: u16, typ: &str, message: &str) {
        match self.counts.iter_mut().find(|(k, _)| k.0 == status && k.1 == typ) {
            Some((_, n)) => *n += 1,
            None => self.counts.push(((status, typ.to_string()), 1)),
        }
        if self.recent.len() == RECENT {
            self.recent.pop_front();
        }
        let message: String = message.chars().take(500).collect();
        self.recent.push_back(json!({"time": time, "path": path, "status": status, "type": typ, "message": message}));
    }

    pub fn view(&self) -> Value {
        json!({"total": self.counts.iter().map(|(_, n)| n).sum::<u64>(),
               "by_status": self.counts.iter().map(|((s, t), n)| json!({"status": s, "type": t, "count": n})).collect::<Vec<_>>(),
               "recent": self.recent.iter().rev().cloned().collect::<Vec<_>>()})
    }
}

pub static FAILURES: Mutex<Failures> = Mutex::new(Failures { counts: Vec::new(), recent: VecDeque::new() });

/// A JSON answer with status >= 400: counted, kept, and said on stdout (the server's log)
pub fn failed(path: &str, status: u16, typ: &str, message: &str) {
    crate::say(&format!("HTTP {status} {path}: {}{message}", if typ.is_empty() { String::new() } else { format!("{typ}: ") }));
    FAILURES.lock().unwrap().add(crate::sync::now(), path, status, typ, message);
    client_failed();
}

/// p50 / p95 (nearest rank) of the values, rounded to ms
pub fn percentiles(mut xs: Vec<f64>) -> Value {
    xs.retain(|x| x.is_finite());
    if xs.is_empty() {
        return Value::Null;
    }
    xs.sort_by(|a, b| a.total_cmp(b));
    let at = |p: f64| xs[((p * xs.len() as f64).ceil() as usize).clamp(1, xs.len()) - 1];
    let r = |x: f64| (x * 1000.0).round() / 1000.0;
    json!({"p50": r(at(0.50)), "p95": r(at(0.95)), "n": xs.len()})
}

/// The kept requests' latencies: time to first token and queue (both from arrival), end to end, time per output token
pub fn latency_view(hist: &[Value]) -> Value {
    let col = |k: &str| hist.iter().filter_map(|r| r.get(k).and_then(Value::as_f64)).collect::<Vec<_>>();
    let tpot = hist.iter().filter_map(|r| r.get("decode_tok_s").and_then(Value::as_f64)).filter(|v| *v > 0.0).map(|v| 1.0 / v).collect();
    json!({"ttft_s": percentiles(col("ttft_s")), "queue_s": percentiles(col("queue_s")), "e2e_s": percentiles(col("duration_s")),
           "tpot_s": percentiles(tpot)})
}

#[derive(Default, Clone)]
struct ToolStat {
    calls: u64,
    errors: u64,
    ms: u64,
    last_error: Option<String>,
}

static TOOLS: Mutex<Vec<(String, ToolStat)>> = Mutex::new(Vec::new());

/// One MCP tool call (McpHub::call): its outcome and time
pub fn tool_call(name: &str, ok: bool, ms: u64, error: Option<&str>) {
    let mut t = TOOLS.lock().unwrap();
    let i = match t.iter().position(|(n, _)| n == name) {
        Some(i) => i,
        None => {
            t.push((name.to_string(), ToolStat::default()));
            t.len() - 1
        }
    };
    let s = &mut t[i].1;
    s.calls += 1;
    s.ms += ms;
    if !ok {
        s.errors += 1;
        s.last_error = error.map(|e| e.chars().take(300).collect());
    }
}

/// tool name -> {calls, errors, ms_mean, last_error}
pub fn tools_view() -> Map<String, Value> {
    TOOLS.lock().unwrap().iter().map(|(n, s)| (n.clone(), json!({"calls": s.calls, "errors": s.errors,
        "ms_mean": if s.calls > 0 { s.ms / s.calls } else { 0 }, "last_error": s.last_error}))).collect()
}

/// A client: the address a request came from and its User-Agent (Claude Code, a script, a browser)
#[derive(Default, Clone)]
struct ClientStat {
    active: u64,
    requests: u64,
    errors: u64,
    prompt_tokens: i64,
    reused: i64,
    output_tokens: i64,
    ttft_s: f64,
    ttft_n: u64,
    decode_ms: f64,
    generated: i64,
    first: f64,
    last: f64,
    path: String,
}

// ponytail: kept in memory until the server restarts, unbounded by client (one row per address and User-Agent)
static CLIENTS: Mutex<Vec<((String, String), ClientStat)>> = Mutex::new(Vec::new());

thread_local! {
    /// The client of the request this thread serves (http.rs dispatch sets it; a request runs on one thread)
    static CLIENT: RefCell<Option<((String, String), String)>> = const { RefCell::new(None) };
}

pub fn set_client(peer: &str, agent: Option<&str>, path: &str) {
    let agent: String = agent.unwrap_or("").chars().take(120).collect();
    CLIENT.with(|c| *c.borrow_mut() = Some(((peer.to_string(), agent), path.to_string())));
}

fn with_client(f: impl FnOnce(&mut ClientStat)) {
    let Some((key, path)) = CLIENT.with(|c| c.borrow().clone()) else { return };
    let mut cs = CLIENTS.lock().unwrap();
    let i = match cs.iter().position(|(k, _)| *k == key) {
        Some(i) => i,
        None => {
            cs.push((key, ClientStat { first: crate::sync::now(), ..Default::default() }));
            cs.len() - 1
        }
    };
    let s = &mut cs[i].1;
    s.last = crate::sync::now();
    s.path = path;
    f(s);
}

/// A finished generation (run.rs settle's request record) for this thread's client
pub fn client_request(rec: &Value) {
    let i = |k: &str| rec.get(k).and_then(Value::as_i64).unwrap_or(0);
    with_client(|s| {
        s.requests += 1;
        s.prompt_tokens += i("prompt_tokens");
        s.reused += i("reused");
        s.output_tokens += i("output_tokens");
        if let Some(t) = rec.get("ttft_s").and_then(Value::as_f64) {
            s.ttft_s += t;
            s.ttft_n += 1;
        }
        if let (Some(g), Some(d)) = (rec.get("engine_generated").and_then(Value::as_i64), rec.get("decode_ms").and_then(Value::as_f64)) {
            s.generated += g;
            s.decode_ms += d;
        }
    });
}

/// A POST arrived (http.rs dispatch): the client shows from now, active until `client_end`
pub fn client_begin() {
    with_client(|s| s.active += 1);
}

pub fn client_end() {
    with_client(|s| s.active = s.active.saturating_sub(1));
}

fn client_failed() {
    with_client(|s| s.errors += 1);
}

/// One row per client, the most recently seen first
pub fn clients_view() -> Value {
    let mut cs = CLIENTS.lock().unwrap().clone();
    cs.sort_by(|a, b| b.1.last.total_cmp(&a.1.last));
    let r = |x: f64, d: f64| (x * d).round() / d;
    Value::Array(cs.into_iter().map(|((peer, agent), s)| json!({"peer": peer, "agent": agent, "active": s.active, "requests": s.requests, "errors": s.errors,
        "prompt_tokens": s.prompt_tokens, "reused": s.reused, "output_tokens": s.output_tokens,
        "ttft_s": if s.ttft_n > 0 { json!(r(s.ttft_s / s.ttft_n as f64, 1000.0)) } else { Value::Null },
        "decode_tok_s": if s.decode_ms > 0.0 { json!(r(s.generated as f64 / (s.decode_ms / 1000.0), 10.0)) } else { Value::Null },
        "first": s.first, "last": s.last, "path": s.path})).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failures_are_counted_by_status_and_type_and_the_newest_comes_first() {
        let mut f = Failures::default();
        f.add(1.0, "/v1/chat/completions", 400, "invalid_request_error", "too long");
        f.add(2.0, "/v1/chat/completions", 400, "invalid_request_error", "bad json");
        f.add(3.0, "/v1/models", 401, "authentication_error", "missing key");
        let v = f.view();
        assert_eq!(v["total"], 3);
        assert_eq!(v["by_status"][0], json!({"status": 400, "type": "invalid_request_error", "count": 2}));
        assert_eq!(v["recent"][0]["status"], 401);
        assert_eq!(v["recent"][2]["message"], "too long");
        for i in 0..60 {
            f.add(i as f64, "/x", 503, "server_error", "busy");
        }
        assert_eq!(f.view()["recent"].as_array().unwrap().len(), RECENT);
    }

    #[test]
    fn percentiles_by_nearest_rank() {
        assert_eq!(percentiles(vec![]), Value::Null);
        let v = percentiles((1..=100).map(|i| i as f64).collect());
        assert_eq!((v["p50"].as_f64(), v["p95"].as_f64(), v["n"].as_u64()), (Some(50.0), Some(95.0), Some(100)));
        assert_eq!(percentiles(vec![0.4, f64::NAN])["p95"], 0.4);
    }

    #[test]
    fn latency_reads_the_request_records() {
        let hist = vec![json!({"ttft_s": 0.5, "queue_s": 0.1, "duration_s": 3.0, "decode_tok_s": 50.0}),
                        json!({"ttft_s": 1.5, "queue_s": 0.0, "duration_s": 9.0, "decode_tok_s": null})];
        let v = latency_view(&hist);
        assert_eq!(v["ttft_s"]["p95"], 1.5);
        assert_eq!(v["tpot_s"]["p50"], 0.02);
        assert_eq!(v["e2e_s"]["n"], 2);
    }

    #[test]
    fn tool_calls_add_up_per_tool() {
        tool_call("t_test__a", true, 100, None);
        tool_call("t_test__a", false, 300, Some("timeout"));
        let v = tools_view();
        assert_eq!(v["t_test__a"], json!({"calls": 2, "errors": 1, "ms_mean": 200, "last_error": "timeout"}));
    }

    #[test]
    fn clients_add_up_per_address_and_agent() {
        let mine = |v: &Value, agent: &str| v.as_array().unwrap().iter().find(|c| c["peer"] == "10.9.9.9" && c["agent"] == agent).cloned();
        set_client("10.9.9.9", Some("test-a/1"), "/v1/messages");
        client_request(&json!({"prompt_tokens": 100, "reused": 60, "output_tokens": 10, "ttft_s": 0.5, "engine_generated": 10, "decode_ms": 500.0}));
        client_request(&json!({"prompt_tokens": 50, "reused": 0, "output_tokens": 20, "ttft_s": 1.5, "engine_generated": 20, "decode_ms": 1000.0}));
        set_client("10.9.9.9", Some("test-a/1"), "/v1/chat/completions");
        failed("/v1/chat/completions", 400, "invalid_request_error", "too long");
        set_client("10.9.9.9", Some("test-b/1"), "/v1/completions");
        client_request(&json!({"prompt_tokens": 7, "output_tokens": 1}));
        let v = clients_view();
        let a = mine(&v, "test-a/1").unwrap();
        assert_eq!((a["requests"].as_u64(), a["errors"].as_u64(), a["prompt_tokens"].as_i64(), a["reused"].as_i64(), a["output_tokens"].as_i64()), (Some(2), Some(1), Some(150), Some(60), Some(30)));
        assert_eq!((a["ttft_s"].as_f64(), a["decode_tok_s"].as_f64(), a["path"].as_str()), (Some(1.0), Some(20.0), Some("/v1/chat/completions")));
        assert_eq!(a["active"], 0);
        let b = mine(&v, "test-b/1").unwrap();
        assert_eq!((b["requests"].as_u64(), b["ttft_s"].clone(), b["decode_tok_s"].clone()), (Some(1), Value::Null, Value::Null));
        set_client("10.9.9.9", Some("test-c/1"), "/v1/messages");
        client_begin(); // still reading its prompt: shown, active
        assert_eq!((mine(&clients_view(), "test-c/1").unwrap()["active"].as_u64(), mine(&clients_view(), "test-c/1").unwrap()["requests"].as_u64()), (Some(1), Some(0)));
        client_end();
        assert_eq!(mine(&clients_view(), "test-c/1").unwrap()["active"], 0);
        CLIENT.with(|c| *c.borrow_mut() = None);
        client_request(&json!({"prompt_tokens": 1})); // no client on this thread: not counted anywhere
        assert_eq!(mine(&clients_view(), "test-b/1").unwrap()["requests"], 1);
    }
}
