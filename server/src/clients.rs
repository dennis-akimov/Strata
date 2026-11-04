//! A client's own engine (docs/CLIENT_ENGINES.md): the run config's "clients" route a client, by its User-Agent, to a
//! small engine of its own - a second strata-server this one starts (or a server at a "url") - with its own context.

use serde_json::{json, Map, Value};
use std::sync::{Mutex, OnceLock};

pub const GENERATION_PATHS: [&str; 4] = ["/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/responses"];

#[derive(Debug, Clone, PartialEq)]
pub struct Route {
    pub name: String,
    /// lowercase: a User-Agent containing it (case ignored) matches
    pub user_agent: String,
    pub paths: Vec<String>,
    pub url: String,
    /// "vip" (default true): while its requests run, the main engine gives this client the GPU (docs/CLIENT_ENGINES.md)
    pub vip: bool,
    /// the child server's config and port (None: "url", a server already running)
    pub spawn: Option<(Value, u16)>,
}

static ROUTES: OnceLock<Vec<Route>> = OnceLock::new();
static CHILDREN: Mutex<Vec<std::process::Child>> = Mutex::new(Vec::new());

/// The config's "clients": each entry's route, and for one without "url" the child server's config.
pub fn parse(cfg: &Value, port: u16) -> Result<Vec<Route>, String> {
    let Some(list) = cfg.get("clients") else { return Ok(vec![]) };
    let list = list.as_array().ok_or("\"clients\" must be a list of {\"user_agent\": ..., \"max_context\": ...}")?;
    let mut out = vec![];
    for (i, e) in list.iter().enumerate() {
        let name = e.get("name").and_then(Value::as_str).map(String::from).unwrap_or_else(|| format!("client{}", i + 1));
        let ua = e.get("user_agent").and_then(Value::as_str).filter(|s| !s.trim().is_empty())
            .ok_or_else(|| format!("\"clients\" {name}: \"user_agent\" (the text its User-Agent contains) is required"))?;
        let paths = match e.get("paths") {
            None => GENERATION_PATHS.iter().map(|p| p.to_string()).collect(),
            Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).map(|p| p.trim_end_matches('/').to_string()).collect(),
            Some(_) => return Err(format!("\"clients\" {name}: \"paths\" must be a list of API paths")),
        };
        let num = |k: &str, d: i64, lo: i64| -> Result<i64, String> {
            match e.get(k) {
                None => Ok(d),
                Some(v) => v.as_i64().filter(|n| *n >= lo).ok_or_else(|| format!("\"clients\" {name}: \"{k}\" must be a whole number >= {lo}")),
            }
        };
        let (url, spawn) = match e.get("url").and_then(Value::as_str) {
            Some(u) => (u.trim_end_matches('/').to_string(), None),
            None => {
                let p = num("port", port as i64 + 100 + i as i64, 1)?;
                let p = u16::try_from(p).map_err(|_| format!("\"clients\" {name}: \"port\" {p} is not a port"))?;
                let child = child_config(cfg, &name, num("max_context", 8192, 256)?, num("parallel", 1, 1)?, num("cache_mib", 0, 0)?);
                (format!("http://127.0.0.1:{p}"), Some((child, p)))
            }
        };
        let vip = match e.get("vip") {
            None => true,
            Some(Value::Bool(b)) => *b,
            Some(_) => return Err(format!("\"clients\" {name}: \"vip\" must be true or false")),
        };
        out.push(Route { name, user_agent: ua.to_lowercase(), paths, url, vip, spawn });
    }
    Ok(out)
}

/// The main config for a client's engine: its context, slots and cache; no pool, vision, MCP or "clients"
fn child_config(cfg: &Value, name: &str, max_context: i64, parallel: i64, cache_mib: i64) -> Value {
    let mut c: Map<String, Value> = cfg.as_object().cloned().unwrap_or_default();
    let args: Vec<String> = c.get("args").and_then(Value::as_array).map(|a| a.iter().map(crate::py::py_str).collect()).unwrap_or_default();
    let mut kept = vec![];
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--max-context" | "--kv-pool" | "--conversation-cache-mib" | "--batch" | "--slots" => { it.next(); }
            "--kv-unified" => {}
            _ => kept.push(a),
        }
    }
    kept.extend(["--max-context".into(), max_context.to_string(), "--conversation-cache-mib".into(), cache_mib.to_string()]);
    c.insert("args".into(), json!(kept));
    for k in ["clients", "vision", "mcp", "port", "parallel", "open_browser"] {
        c.remove(k);
    }
    if parallel > 1 {
        c.insert("parallel".into(), json!(parallel));
    }
    c.insert("open_browser".into(), json!(false));
    if let Some(log) = c.get("log").and_then(Value::as_str).map(String::from) {
        let log = match log.strip_suffix(".log") { Some(b) => format!("{b}-{name}.log"), None => format!("{log}-{name}") };
        c.insert("log".into(), json!(log));
    }
    Value::Object(c)
}

/// Whether the main engine runs with --vip: any client engine that takes the GPU first
pub fn any_vip(cfg: &Value) -> bool {
    parse(cfg, 0).map(|r| r.iter().any(|r| r.vip)).unwrap_or(false)
}

/// The route for a request, the first entry that matches its path and User-Agent
pub fn pick<'a>(routes: &'a [Route], path: &str, user_agent: Option<&str>) -> Option<&'a Route> {
    let ua = user_agent?.to_lowercase();
    routes.iter().find(|r| r.paths.iter().any(|p| p == path) && ua.contains(&r.user_agent))
}

pub fn set(routes: Vec<Route>) {
    let _ = ROUTES.set(routes);
}

pub fn routes() -> &'static [Route] {
    ROUTES.get().map(Vec::as_slice).unwrap_or(&[])
}

/// Start each entry's server (this binary, its config in the temp folder); `handoff`: beside an old one on that port
pub fn spawn_all(main_port: u16, handoff: bool) {
    let exe = std::env::current_exe().unwrap_or_else(|_| "strata-server".into());
    for r in routes() {
        let Some((cfg, port)) = &r.spawn else { continue };
        let path = std::env::temp_dir().join(format!("strata-client-{main_port}-{}.json", r.name));
        if let Err(e) = std::fs::write(&path, cfg.to_string()) {
            crate::say(&format!("client engine {}: could not write its config ({e}); its requests get HTTP 503", r.name));
            continue;
        }
        let mut cmd = std::process::Command::new(&exe);
        cmd.args(["--engine", "strata", "--config"]).arg(&path).args(["--port", &port.to_string(), "--host", "127.0.0.1"]);
        cmd.arg(if handoff { "--handoff" } else { "--replace" });
        cmd.env("STRATA_ALLOW_SECOND_ENGINE", "1").env("STRATA_PARENT_PID", std::process::id().to_string());
        cmd.stdin(std::process::Stdio::null());
        match cmd.spawn() {
            Ok(c) => {
                crate::say(&format!("client engine {}: User-Agent \"{}\" -> {} (pid {})", r.name, r.user_agent, r.url, c.id()));
                CHILDREN.lock().unwrap().push(c);
            }
            Err(e) => crate::say(&format!("client engine {}: could not start ({e}); its requests get HTTP 503", r.name)),
        }
    }
}

/// SIGTERM to every client engine, then wait for them (they QUIT their own engines)
pub fn stop_all() {
    let mut cs = CHILDREN.lock().unwrap();
    for c in cs.iter() {
        unsafe { libc::kill(c.id() as i32, libc::SIGTERM) };
    }
    for c in cs.iter_mut() {
        let _ = c.wait();
    }
    cs.clear();
}

/// In a client engine (STRATA_PARENT_PID): stop when the main server is gone (SIGTERM to self: the usual way out)
pub fn watch_parent() -> bool {
    let Some(parent) = std::env::var("STRATA_PARENT_PID").ok().and_then(|p| p.parse::<i32>().ok()) else { return false };
    std::thread::spawn(move || loop {
        std::thread::sleep(std::time::Duration::from_secs(2));
        if unsafe { libc::getppid() } != parent {
            crate::say("the main server is gone: stopping");
            unsafe { libc::kill(libc::getpid(), libc::SIGTERM) };
            return;
        }
    });
    true
}

/// The Monitor's view of the client engines (GET /metrics "client_engines"): each one's route and its own /metrics'
/// state, totals and newest requests; `up: false` while it loads or when it is gone
pub fn status(key: Option<&str>) -> Value {
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    let agent = AGENT.get_or_init(|| ureq::Agent::config_builder().timeout_global(Some(std::time::Duration::from_millis(1500))).http_status_as_error(false).build().into());
    Value::Array(routes().iter().map(|r| {
        let mut q = agent.get(format!("{}/metrics", r.url));
        if let Some(k) = key {
            q = q.header("Authorization", format!("Bearer {k}"));
        }
        let m = q.call().ok().filter(|x| x.status() == 200).and_then(|mut x| x.body_mut().read_to_string().ok()).and_then(|t| serde_json::from_str::<Value>(&t).ok());
        view(r, m.as_ref())
    }).collect())
}

fn view(r: &Route, m: Option<&Value>) -> Value {
    let mut v = json!({"name": r.name, "user_agent": r.user_agent, "paths": r.paths, "url": r.url, "up": m.is_some()});
    if let Some(m) = m {
        v["max_context"] = m.pointer("/engine/max_context").cloned().unwrap_or(Value::Null);
        v["live"] = m.get("live").cloned().unwrap_or(Value::Null);
        v["totals"] = m.get("totals").cloned().unwrap_or(Value::Null);
        v["requests"] = json!(m.get("requests").and_then(Value::as_array).map(|a| a.iter().take(5).cloned().collect::<Vec<_>>()).unwrap_or_default());
    }
    v
}

/// A routed answer's numbers for the Clients tab (observe::client_request's record): OpenAI's or Anthropic's usage,
/// Strata's timings
pub fn record_from(answer: &Value, ttft_s: Option<f64>) -> Value {
    let u = answer.get("usage").or_else(|| answer.get("response").and_then(|r| r.get("usage"))).or_else(|| answer.get("message").and_then(|m| m.get("usage")));
    let n = |v: Option<&Value>| v.and_then(Value::as_i64);
    let u = u.unwrap_or(&Value::Null);
    let t = answer.get("timings").unwrap_or(&Value::Null);
    json!({
        "prompt_tokens": n(u.get("prompt_tokens")).or(n(u.get("input_tokens"))),
        "reused": n(u.pointer("/prompt_tokens_details/cached_tokens")).or(n(u.get("cache_read_input_tokens"))).or(n(t.get("cache_n"))),
        "output_tokens": n(u.get("completion_tokens")).or(n(u.get("output_tokens"))),
        "ttft_s": ttft_s,
        "engine_generated": n(t.get("predicted_n")),
        "decode_ms": t.get("predicted_ms").and_then(Value::as_f64),
    })
}

/// The numbers in a streamed answer's events (each `data: {...}` line): the last usage and timings seen win
pub fn merge_event(acc: &mut Map<String, Value>, line: &str) {
    let Some(j) = line.strip_prefix("data: ").and_then(|d| serde_json::from_str::<Value>(d.trim()).ok()) else { return };
    for k in ["usage", "timings"] {
        if let Some(v) = j.get(k).filter(|v| v.is_object()) {
            let slot = acc.entry(k).or_insert_with(|| json!({}));
            for (kk, vv) in v.as_object().unwrap() {
                slot[kk] = vv.clone();
            }
        }
    }
    // Anthropic: message_start carries the input tokens, message_delta the output; Responses: response.completed
    for p in ["/message/usage", "/response/usage"] {
        if let Some(v) = j.pointer(p).and_then(Value::as_object) {
            let slot = acc.entry("usage").or_insert_with(|| json!({}));
            for (kk, vv) in v {
                slot[kk] = vv.clone();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Value {
        json!({"args": ["--gguf", "m.gguf", "--max-context", "32768", "--kv-unified", "--kv-pool", "98304", "--conversation-cache-mib", "16384", "--kv", "f16"],
               "parallel": 4, "log": "/x/strata-q.log", "vision": {"exe": "v"}, "mcp": {"a": {}}, "model_name": "q",
               "clients": [{"name": "zed", "user_agent": "Zed/", "max_context": 6144},
                           {"user_agent": "curl", "paths": ["/v1/completions"], "url": "http://10.0.0.2:9000/"}]})
    }

    #[test]
    fn an_entry_gets_its_own_small_engine_config() {
        let r = parse(&cfg(), 8090).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!((r[0].name.as_str(), r[0].user_agent.as_str(), r[0].url.as_str()), ("zed", "zed/", "http://127.0.0.1:8190"));
        assert_eq!(r[0].paths, GENERATION_PATHS.to_vec());
        let (c, port) = r[0].spawn.clone().unwrap();
        assert_eq!(port, 8190);
        assert_eq!(c["args"], json!(["--gguf", "m.gguf", "--kv", "f16", "--max-context", "6144", "--conversation-cache-mib", "0"]));
        for k in ["clients", "vision", "mcp", "parallel"] {
            assert!(c.get(k).is_none(), "{k} left in");
        }
        assert_eq!((c["log"].as_str(), c["model_name"].as_str(), c["open_browser"].as_bool()), (Some("/x/strata-q-zed.log"), Some("q"), Some(false)));
        // "url": a server already running, nothing started
        assert_eq!((r[1].name.as_str(), r[1].url.as_str(), r[1].spawn.is_none()), ("client2", "http://10.0.0.2:9000", true));
    }

    #[test]
    fn bad_entries_are_refused_with_the_reason() {
        assert!(parse(&json!({"clients": {}}), 1).unwrap_err().contains("must be a list"));
        assert!(parse(&json!({"clients": [{"max_context": 6144}]}), 1).unwrap_err().contains("user_agent"));
        assert!(parse(&json!({"clients": [{"user_agent": "z", "max_context": 10}]}), 1).unwrap_err().contains("max_context"));
        assert!(parse(&json!({"clients": [{"user_agent": "z", "port": 70000}]}), 1).unwrap_err().contains("not a port"));
        assert!(parse(&json!({"clients": [{"user_agent": "z", "vip": "yes"}]}), 1).unwrap_err().contains("vip"));
        assert!(parse(&json!({"clients": [{"user_agent": "z"}]}), 1).unwrap()[0].vip);         // on by default
        assert!(!any_vip(&json!({"clients": [{"user_agent": "z", "vip": false}]})));
        assert!(any_vip(&json!({"clients": [{"user_agent": "z"}]})) && !any_vip(&json!({})));
        assert_eq!(parse(&json!({}), 1).unwrap(), vec![]);
        let r = parse(&json!({"clients": [{"user_agent": "z", "parallel": 2, "cache_mib": 512}]}), 8000).unwrap();
        let (c, _) = r[0].spawn.clone().unwrap();
        assert_eq!((c["parallel"].as_i64(), c["args"][3].as_str()), (Some(2), Some("512")));
    }

    #[test]
    fn a_request_goes_to_the_first_matching_route() {
        let r = parse(&cfg(), 8090).unwrap();
        let zed = Some("Zed/1.28.0+dev.33 (macos; aarch64)");
        assert_eq!(pick(&r, "/v1/chat/completions", zed).map(|r| r.name.as_str()), Some("zed"));
        assert_eq!(pick(&r, "/v1/chat/completions", Some("zed/2")).map(|r| r.name.as_str()), Some("zed")); // case ignored
        assert_eq!(pick(&r, "/v1/models", zed), None);                          // not a generation path
        assert_eq!(pick(&r, "/v1/chat/completions", Some("curl/8.7")), None);   // its paths are /v1/completions only
        assert_eq!(pick(&r, "/v1/completions", Some("curl/8.7")).map(|r| r.name.as_str()), Some("client2"));
        assert_eq!(pick(&r, "/v1/chat/completions", None), None);
        assert_eq!(pick(&r, "/v1/chat/completions", Some("claude-cli/2")), None);
    }

    #[test]
    fn the_monitor_sees_an_engine_up_or_not() {
        let r = &parse(&cfg(), 8090).unwrap()[0];
        let down = view(r, None);
        assert_eq!((down["name"].as_str(), down["up"].as_bool(), down.get("live")), (Some("zed"), Some(false), None));
        let m = json!({"engine": {"max_context": 6144}, "live": {"state": "generating"}, "totals": {"requests": 3},
                       "requests": [1, 2, 3, 4, 5, 6, 7]});
        let up = view(r, Some(&m));
        assert_eq!((up["up"].as_bool(), up["max_context"].as_i64(), up["live"]["state"].as_str(), up["totals"]["requests"].as_i64()),
                   (Some(true), Some(6144), Some("generating"), Some(3)));
        assert_eq!(up["requests"].as_array().unwrap().len(), 5);
        assert_eq!(up["paths"], json!(GENERATION_PATHS));
    }

    #[test]
    fn a_routed_answer_counts_its_tokens() {
        let openai = json!({"usage": {"prompt_tokens": 100, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 90}},
                            "timings": {"predicted_n": 7, "predicted_ms": 140.0}});
        assert_eq!(record_from(&openai, Some(0.2)), json!({"prompt_tokens": 100, "reused": 90, "output_tokens": 7, "ttft_s": 0.2, "engine_generated": 7, "decode_ms": 140.0}));
        let anthropic = json!({"usage": {"input_tokens": 50, "output_tokens": 3, "cache_read_input_tokens": 40}});
        let r = record_from(&anthropic, None);
        assert_eq!((r["prompt_tokens"].as_i64(), r["reused"].as_i64(), r["output_tokens"].as_i64()), (Some(50), Some(40), Some(3)));
        // streamed: OpenAI's last chunk; Anthropic's message_start then message_delta
        let mut acc = Map::new();
        merge_event(&mut acc, r#"data: {"choices":[{"delta":{"content":"hi"}}]}"#);
        merge_event(&mut acc, r#"data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":2},"timings":{"predicted_n":2,"predicted_ms":30}}"#);
        merge_event(&mut acc, "data: [DONE]");
        let r = record_from(&Value::Object(acc), None);
        assert_eq!((r["prompt_tokens"].as_i64(), r["output_tokens"].as_i64(), r["engine_generated"].as_i64()), (Some(12), Some(2), Some(2)));
        let mut acc = Map::new();
        merge_event(&mut acc, r#"data: {"type":"message_start","message":{"usage":{"input_tokens":30,"output_tokens":1}}}"#);
        merge_event(&mut acc, r#"data: {"type":"message_delta","usage":{"output_tokens":9}}"#);
        let r = record_from(&Value::Object(acc), None);
        assert_eq!((r["prompt_tokens"].as_i64(), r["output_tokens"].as_i64()), (Some(30), Some(9)));
    }
}
