//! What the Monitor shows beyond speed (docs/OBSERVABILITY.md): the requests the server refused, latency as percentiles
//! (vLLM's and OpenTelemetry's GenAI metrics: time to first token, time per output token, end to end), and each MCP
//! tool's calls.  Process-wide: a refusal can come before there is a Service to hold it.

use serde_json::{json, Map, Value};
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
}
