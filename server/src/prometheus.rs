//! GET /metrics in the Prometheus text format, with vLLM's metric names (serve/prometheus.py): everything from the
//! server's own record (Service::metrics) plus three latency histograms observed when a request finishes.

use crate::pyjson::float_repr;
use serde_json::Value;
use std::sync::Mutex;

pub const BUCKETS: [f64; 22] = [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0];
pub const CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

/// Prometheus asks with Accept: text/plain or application/openmetrics-text; a person can add ?format=prometheus.
pub fn wants_prometheus(accept: &str, query: &str) -> bool {
    query.contains("format=prometheus") || accept.contains("openmetrics") || accept.contains("text/plain")
}

#[derive(Clone, Default)]
pub struct Histogram {
    pub counts: [u64; 22],
    pub sum: f64,
    pub n: u64,
}

impl Histogram {
    fn observe(&mut self, v: Option<f64>, times: u64) {
        let Some(v) = v else { return };
        if times == 0 {
            return;
        }
        for (i, b) in BUCKETS.iter().enumerate() {
            if v <= *b {
                self.counts[i] += times;
            }
        }
        self.sum += v * times as f64;
        self.n += times;
    }
}

/// Time to first token, the time between tokens and the whole request, per finished request.
#[derive(Default)]
pub struct Latencies(Mutex<(Histogram, Histogram, Histogram)>);

impl Latencies {
    pub fn observe(&self, ttft_s: Option<f64>, output_tokens: i64, decode_s: f64, total_s: f64) {
        let mut h = self.0.lock().unwrap();
        h.0.observe(ttft_s, 1);
        if output_tokens > 1 && decode_s != 0.0 {
            h.1.observe(Some(decode_s / (output_tokens - 1) as f64), (output_tokens - 1) as u64);
        }
        h.2.observe(Some(total_s), 1);
    }
    pub fn snapshot(&self) -> [Histogram; 3] {
        let h = self.0.lock().unwrap();
        [h.0.clone(), h.1.clone(), h.2.clone()]
    }
}

/// A sample value as Python prints it (int stays int, float repr; a bool counts as no number).
fn num(v: &Value) -> Option<String> {
    match v {
        Value::Number(n) if n.is_i64() || n.is_u64() => Some(n.to_string()),
        Value::Number(n) => Some(float_repr(n.as_f64().unwrap())),
        _ => None,
    }
}

fn round_py(x: f64, d: i32) -> String {
    float_repr(crate::service::pyround(x, d))
}

pub fn render(m: &Value, lat: &[Histogram; 3]) -> String {
    let model = m["engine"]["model"].as_str().filter(|s| !s.is_empty()).unwrap_or("strata").replace('\\', "\\\\").replace('"', "\\\"");
    let lab = format!("model_name=\"{model}\"");
    let (live, totals) = (&m["live"], &m["totals"]);
    let mut out: Vec<String> = vec![];
    let metric = |out: &mut Vec<String>, name: &str, kind: &str, help: &str, value: String| {
        out.push(format!("# HELP {name} {help}\n# TYPE {name} {kind}\n{name}{{{lab}}} {value}"));
    };
    let n0 = |v: &Value| num(v).unwrap_or_else(|| "0".into());
    let f = |v: &Value| v.as_f64().filter(|_| v.is_number()).unwrap_or(0.0);
    let busy = matches!(live["state"].as_str(), Some("reading" | "generating"));
    let ctx = f(&m["engine"]["max_context"]);
    let used = if busy { f(&live["prompt_tokens"]) + f(&live["generated"]) } else { 0.0 };
    metric(&mut out, "vllm:num_requests_running", "gauge", "Requests reading their prompt or generating.", num(&live["running"]).unwrap_or_else(|| (busy as i32).to_string()));
    let waiting = |v: &Value| v.as_i64().unwrap_or(0);
    metric(&mut out, "vllm:num_requests_waiting", "gauge", "Requests waiting for their turn.", (waiting(&live["queued"]) + waiting(&live["waiting"])).to_string());
    metric(&mut out, "vllm:kv_cache_usage_perc", "gauge", "The running request's share of the context (1 = full).", if ctx != 0.0 { round_py((used / ctx).min(1.0), 4) } else { "0".into() });
    metric(&mut out, "vllm:prompt_tokens_total", "counter", "Prompt tokens of the finished requests.", n0(&totals["prompt_tokens"]));
    metric(&mut out, "vllm:generation_tokens_total", "counter", "Generated tokens.", n0(&totals["output_tokens"]));
    metric(&mut out, "vllm:request_success_total", "counter", "Finished requests.", n0(&totals["requests"]));
    metric(&mut out, "vllm:prefix_cache_queries_total", "counter", "Prompt tokens looked up in the prompt cache.", n0(&totals["prompt_tokens"]));
    metric(&mut out, "vllm:prefix_cache_hits_total", "counter", "Prompt tokens the prompt cache already held.", n0(&totals["reused"]));
    metric(&mut out, "vllm:spec_decode_num_draft_tokens_total", "counter", "MTP draft tokens offered.", n0(&totals["drafts_offered"]));
    metric(&mut out, "vllm:spec_decode_num_accepted_tokens_total", "counter", "MTP draft tokens accepted.", n0(&totals["drafts_accepted"]));
    metric(&mut out, "vllm:num_preemptions_total", "counter", "Preempted requests (Strata does not preempt).", "0".into());
    for (i, (name, help)) in [("vllm:time_to_first_token_seconds", "Time to the first token."), ("vllm:inter_token_latency_seconds", "Time between two tokens."), ("vllm:e2e_request_latency_seconds", "A request from start to end.")].iter().enumerate() {
        let h = &lat[i];
        out.push(format!("# HELP {name} {help}\n# TYPE {name} histogram"));
        for (b, c) in BUCKETS.iter().zip(h.counts.iter()) {
            out.push(format!("{name}_bucket{{{lab},le=\"{}\"}} {c}", float_repr(*b)));
        }
        out.push(format!("{name}_bucket{{{lab},le=\"+Inf\"}} {}", h.n));
        out.push(format!("{name}_sum{{{lab}}} {}\n{name}_count{{{lab}}} {}", round_py(h.sum, 6), h.n));
    }
    strata(m, &mut out, &lab);
    out.join("\n") + "\n"
}

/// The JSON's facts vLLM has no name for, under `strata:` and the JSON's key.
fn strata(m: &Value, out: &mut Vec<String>, lab: &str) {
    let (live, totals, hw, eng) = (&m["live"], &m["totals"], &m["hardware"], &m["engine"]);
    let mut typed = std::collections::HashSet::new();
    let mut metric = |name: &str, kind: &str, help: &str, value: &Value, labels: &str| {
        let Some(v) = num(value) else { return };
        if typed.insert(name.to_string()) {
            out.push(format!("# HELP strata:{name} {help}\n# TYPE strata:{name} {kind}"));
        }
        out.push(format!("strata:{name}{{{lab}{labels}}} {v}"));
    };
    let or0 = |v: &Value| if crate::py::truthy(v) { v.clone() } else { serde_json::json!(0) };
    for state in ["unloaded", "idle", "reading", "generating"] {
        metric("live_state", "gauge", "1 for what the engine is doing (live.state).", &serde_json::json!((live["state"] == state) as i32), &format!(",state=\"{state}\""));
    }
    metric("live_tok_s", "gauge", "Decode rate over the last seconds (live.tok_s).", &or0(&live["tok_s"]), "");
    metric("live_prefill_tok_s_mean", "gauge", "Prompt reading rate of the running request (live.prefill_tok_s_mean).", &or0(&live["prefill_tok_s_mean"]), "");
    metric("live_prompt_read", "gauge", "Prompt tokens read so far by the running request (live.prompt_read).", &or0(&live["prompt_read"]), "");
    for state in ["idle", "reading", "generating"] {
        if let Some(slots) = live["slots"].as_array().filter(|s| !s.is_empty()) {
            let n = slots.iter().filter(|x| x["state"] == state).count();
            metric("live_slots", "gauge", "The batch slots in each state (live.slots).", &serde_json::json!(n), &format!(",state=\"{state}\""));
        }
    }
    metric("engine_max_context", "gauge", "The engine's context (engine.max_context).", &eng["max_context"], "");
    let secs = |v: &Value| serde_json::json!(crate::service::pyround(v.as_f64().unwrap_or(0.0) / 1000.0, 3));
    metric("totals_prompt_seconds_total", "counter", "Time spent reading prompts (totals.prompt_ms).", &secs(&totals["prompt_ms"]), "");
    metric("totals_decode_seconds_total", "counter", "Time spent generating (totals.decode_ms).", &secs(&totals["decode_ms"]), "");
    let last = m["requests"].as_array().and_then(|r| r.first()).cloned().unwrap_or(Value::Null);
    metric("last_hit_rate", "gauge", "The last request's expert cache hit rate (requests[0].hit_rate).", &last["hit_rate"], "");
    metric("last_decode_tok_s", "gauge", "The last request's decode rate (requests[0].decode_tok_s).", &last["decode_tok_s"], "");
    let gpus: Vec<Value> = match hw["gpus"].as_array() {
        Some(g) if !g.is_empty() => g.clone(),
        _ if !hw["gpu_util"].is_null() => vec![serde_json::json!({"index": 0, "util": hw["gpu_util"], "mem_used": hw["gpu_mem_used"], "mem_total": hw["gpu_mem_total"], "temp": hw["gpu_temp"], "power": hw["gpu_power"]})],
        _ => vec![],
    };
    for (name, key, help) in [("gpu_util", "util", "GPU busy, percent (hardware.gpu_util)."), ("gpu_mem_used_bytes", "mem_used", "GPU memory in use (hardware.gpu_mem_used)."), ("gpu_mem_total_bytes", "mem_total", "GPU memory (hardware.gpu_mem_total)."), ("gpu_temp_celsius", "temp", "GPU temperature (hardware.gpu_temp)."), ("gpu_power_watts", "power", "GPU power draw (hardware.gpu_power).")] {
        for g in &gpus {
            metric(name, "gauge", help, &g[key], &format!(",gpu=\"{}\"", crate::py::py_str(&g["index"])));
        }
    }
    metric("cpu", "gauge", "CPU busy, percent (hardware.cpu).", &hw["cpu"], "");
    metric("ram_used_bytes", "gauge", "RAM in use (hardware.ram_used).", &hw["ram_used"], "");
    metric("ram_total_bytes", "gauge", "RAM (hardware.ram_total).", &hw["ram_total"], "");
}
