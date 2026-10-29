//! The HTTP layer (serve/server.py's make_handler): every route, the API key, the Host check (DNS rebinding), the
//! Origin checks, CORS for configured origins, the web app's files with ETags, SSE streams with keep-alives, and the
//! request monitor.  Each request runs on its own thread (as the Python server's), writing to the connection through
//! a channel; a client that hangs up sets `gone`, which the request's watcher turns into its cancel.

use crate::api::{self, anthropic_collect, anthropic_events, openai_chunks, openai_collect, structured_chunks, McpRun, Run};
use crate::errors::Fail;
use crate::frontend::{anthropic_to_messages, images_of, openai_to_messages};
use crate::parser::{forced_call, tool_choice_of, CALL_START};
use crate::py::{self, get, truthy};
use crate::pyjson;
use crate::responses::{self as rs, RespErr};
use crate::say;
use crate::service::{check_request_sampling, stop_strings, Rec, Service, TRACE};
use crate::structured::{prepare_format, validated_json};
use crate::sync::{Closed, Event as Flag, Stop};
use axum::body::Body;
use axum::extract::{ConnectInfo, Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use bytes::Bytes;
use futures::StreamExt;
use serde_json::{json, Map, Value};
use sha2::Digest;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

pub struct App {
    pub svc: Arc<Service>,
    cancels: Mutex<HashMap<String, Arc<Flag>>>,
    early: Mutex<indexmap::IndexSet<String>>,
}

impl App {
    pub fn new(svc: Arc<Service>) -> Arc<Self> {
        Arc::new(App { svc, cancels: Mutex::new(HashMap::new()), early: Mutex::new(Default::default()) })
    }
}

enum Frame {
    Head(u16, Vec<(String, String)>),
    Data(Bytes),
}

/// #893: the most a request body may hold (STRATA_MAX_BODY_MIB, default 256 MiB).
pub fn body_limit() -> usize {
    match std::env::var("STRATA_MAX_BODY_MIB").ok().and_then(|v| v.parse::<usize>().ok()) {
        Some(m) if m > 0 => m << 20,
        _ => 256 << 20,
    }
}

/// Sets `gone` when dropped while armed: the client left before the answer was handed over.
struct GoneOnDrop(Arc<Flag>, bool);

impl Drop for GoneOnDrop {
    fn drop(&mut self) {
        if self.1 {
            self.0.set();
        }
    }
}

pub async fn handle(State(app): State<Arc<App>>, ConnectInfo(peer): ConnectInfo<SocketAddr>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let limit = body_limit();
    let chunked = parts.headers.get("transfer-encoding").and_then(|v| v.to_str().ok()).map(|v| v.rsplit(',').next().unwrap_or("").trim().eq_ignore_ascii_case("chunked")).unwrap_or(false);
    let declared = parts.headers.get("content-length").and_then(|v| v.to_str().ok()).map(|v| v.trim().parse::<i64>());
    // the body, read here (the answer may need it whole); a size over the limit is refused before it is read
    let body: Result<Bytes, (u16, String)> = match (chunked, &declared) {
        (false, Some(Err(_))) => Err((400, "invalid Content-Length".into())),
        (false, Some(Ok(n))) if *n < 0 => Err((400, "invalid Content-Length".into())),
        (false, Some(Ok(n))) if *n as usize > limit => Err((413, format!("the request body is larger than {} MiB (STRATA_MAX_BODY_MIB raises the limit)", limit >> 20))),
        _ => {
            let mut stream = body.into_data_stream();
            let mut buf: Vec<u8> = vec![];
            let mut over = false;
            let mut broken = false;
            while let Some(c) = stream.next().await {
                match c {
                    Ok(c) => {
                        if buf.len() + c.len() > limit {
                            over = true;
                            break;
                        }
                        buf.extend_from_slice(&c);
                    }
                    Err(_) => {
                        broken = true;
                        break;
                    }
                }
            }
            if over {
                Err((413, format!("the request body is larger than {} MiB", limit >> 20)))
            } else if broken {
                Err((400, "malformed chunked request body".into()))
            } else {
                Ok(Bytes::from(buf))
            }
        }
    };
    let gone = Arc::new(Flag::new());
    let (tx, mut rx) = mpsc::channel::<Frame>(64);
    let ctx = Ctx {
        app: app.clone(),
        method: parts.method.as_str().to_string(),
        raw_path: parts.uri.path_and_query().map(|p| p.as_str().to_string()).unwrap_or_else(|| "/".into()),
        headers: parts.headers.clone(),
        body,
        peer: peer.ip().to_string(),
        tx,
        head_sent: false,
        gone: gone.clone(),
        record: None,
        cancel_id: None,
        watch_done: None,
    };
    let mut guard = GoneOnDrop(gone.clone(), true);
    let worker = tokio::task::spawn_blocking(move || dispatch(ctx));
    let Some(Frame::Head(code, headers)) = rx.recv().await else {
        let _ = worker.await;
        guard.1 = false;
        return Response::builder().status(500).body(Body::empty()).unwrap();
    };
    guard.1 = false;
    let mut b = Response::builder().status(StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR));
    for (k, v) in headers {
        b = b.header(k, v);
    }
    let body_guard = GoneOnDrop(gone, true);
    let stream = futures::stream::unfold((rx, body_guard), |(mut rx, mut g)| async move {
        match rx.recv().await {
            Some(Frame::Data(d)) => Some((Ok::<Bytes, std::io::Error>(d), (rx, g))),
            Some(Frame::Head(..)) => Some((Ok(Bytes::new()), (rx, g))),
            None => {
                g.1 = false; // the answer is complete: no hang-up
                None
            }
        }
    });
    b.body(Body::from_stream(stream)).unwrap()
}

pub struct Ctx {
    app: Arc<App>,
    method: String,
    raw_path: String,
    headers: HeaderMap,
    body: Result<Bytes, (u16, String)>,
    peer: String,
    tx: mpsc::Sender<Frame>,
    head_sent: bool,
    gone: Arc<Flag>,
    record: Option<Rec>,
    cancel_id: Option<String>,
    watch_done: Option<Arc<Flag>>,
}

fn status_ok(code: u16) -> bool {
    code < 600
}

impl Ctx {
    fn svc(&self) -> &Arc<Service> {
        &self.app.svc
    }

    fn header(&self, k: &str) -> Option<String> {
        self.headers.get(k).map(|v| String::from_utf8_lossy(v.as_bytes()).into_owned())
    }

    fn path(&self) -> String {
        self.raw_path.split('?').next().unwrap_or("").trim_end_matches('/').to_string()
    }

    fn query(&self) -> String {
        self.raw_path.split_once('?').map(|(_, q)| q.to_string()).unwrap_or_default()
    }

    fn query_param(&self, k: &str) -> Option<String> {
        form_urlencoded::parse(self.query().as_bytes()).find(|(n, _)| n == k).map(|(_, v)| v.into_owned())
    }

    fn send_head(&mut self, code: u16, headers: Vec<(String, String)>) -> Result<(), Closed> {
        self.head_sent = true;
        self.tx.blocking_send(Frame::Head(if status_ok(code) { code } else { 500 }, headers)).map_err(|_| Closed)
    }

    fn write(&mut self, data: &[u8]) -> Result<(), Closed> {
        if self.gone.is_set() {
            return Err(Closed);
        }
        self.tx.blocking_send(Frame::Data(Bytes::copy_from_slice(data))).map_err(|_| Closed)
    }

    fn note(&self, values: Vec<(&str, Value)>) {
        if let Some(r) = &self.record {
            let mut r = r.lock().unwrap();
            for (k, v) in values {
                r.insert(k.into(), v);
            }
        }
    }

    /// #321: CORS headers for an API path and an origin the config lists in cors_origins.
    fn cors(&self) -> Vec<(String, String)> {
        let origins = &self.svc().set.cors_origins;
        if origins.is_empty() || !self.raw_path.split('?').next().unwrap_or("").starts_with("/v1/") {
            return vec![];
        }
        let origin = self.header("origin").unwrap_or_default().trim_end_matches('/').to_string();
        let mut h = vec![];
        let allow = if origins.iter().any(|o| o == "*") {
            "*".to_string()
        } else if !origin.is_empty() && origins.contains(&origin) {
            h.push(("Vary".into(), "Origin".into()));
            origin
        } else {
            return vec![];
        };
        h.push(("Access-Control-Allow-Origin".into(), allow));
        h.push(("Access-Control-Allow-Methods".into(), "GET, POST, DELETE, OPTIONS".into()));
        h.push(("Access-Control-Allow-Headers".into(), self.header("access-control-request-headers").unwrap_or_else(|| "Authorization, Content-Type, x-api-key, anthropic-version, anthropic-beta".into())));
        h.push(("Access-Control-Max-Age".into(), "600".into()));
        h
    }

    fn json_h(&mut self, code: u16, obj: &Value, extra: Vec<(String, String)>) {
        if let Some(r) = &self.record {
            let mut r = r.lock().unwrap();
            r.insert("http_status".into(), json!(code));
            let raw = pyjson::dumps_with(obj, pyjson::Opts { ensure_ascii: false, indent: Some(2), ..Default::default() });
            r.insert("response".into(), json!(raw.chars().take(262144).collect::<String>()));
            r.insert("response_truncated".into(), json!(raw.chars().count() > 262144));
            for key in ["error", "usage", "timings"] {
                if let Some(v) = obj.get(key) {
                    r.insert(key.into(), v.clone());
                }
            }
        }
        let body = pyjson::dumps_u(obj).into_bytes();
        let mut h = vec![("Content-Type".to_string(), "application/json".to_string())];
        h.extend(extra);
        h.extend(self.cors());
        h.push(("Content-Length".into(), body.len().to_string()));
        if self.send_head(code, h).is_ok() {
            let _ = self.write(&body);
        }
    }

    fn json(&mut self, code: u16, obj: Value) {
        self.json_h(code, &obj, vec![]);
    }

    /// A JSON answer as it is (pictures); a client that has gone is no error here.
    fn raw(&mut self, code: u16, data: &[u8], retry: Option<String>) {
        let mut h = vec![];
        if let Some(r) = retry {
            h.push(("Retry-After".to_string(), r));
        }
        h.push(("Content-Type".into(), "application/json".into()));
        h.push(("Content-Length".into(), data.len().to_string()));
        if self.send_head(code, h).is_err() || self.write(data).is_err() {
            self.note(vec![("outcome", json!("disconnected"))]);
        }
    }

    /// One of the web app's files, with a strong ETag (a 304 when the client has it).
    fn static_(&mut self, body: &[u8], ctype: &str, cache: &str) {
        let tag = format!("\"{}\"", &hex::encode(sha2::Sha256::digest(body))[..32]);
        let asked: String = self.headers.get_all("if-none-match").iter().map(|v| String::from_utf8_lossy(v.as_bytes()).into_owned()).collect::<Vec<_>>().join(", ");
        let re = regex::Regex::new(r#"(?:W/)?("[^"]*")"#).unwrap();
        let fresh = py::strip(&asked) == "*" || re.captures_iter(&asked).any(|c| c[1] == tag);
        let mut h = vec![("ETag".to_string(), tag), ("Cache-Control".into(), cache.into())];
        h.extend(self.cors());
        if !fresh {
            h.push(("Content-Type".into(), ctype.into()));
            h.push(("Content-Length".into(), body.len().to_string()));
        }
        if self.send_head(if fresh { 304 } else { 200 }, h).is_ok() && !fresh {
            let _ = self.write(body);
        }
    }

    fn authorized(&mut self) -> bool {
        let keys = self.svc().set.api_key.clone();
        if keys.is_empty() {
            return true;
        }
        let auth = self.headers.get("authorization").map(|v| v.as_bytes().to_vec()).unwrap_or_default();
        let given: Vec<u8> = if auth.len() >= 7 && auth[..7].eq_ignore_ascii_case(b"bearer ") {
            let s = &auth[7..];
            let start = s.iter().position(|b| !b.is_ascii_whitespace()).unwrap_or(s.len());
            let end = s.iter().rposition(|b| !b.is_ascii_whitespace()).map(|e| e + 1).unwrap_or(start);
            s[start..end].to_vec()
        } else {
            self.headers.get("x-api-key").map(|v| v.as_bytes().to_vec()).unwrap_or_default()
        };
        if crate::net::key_matches(&given, &keys) {
            return true;
        }
        let challenge = format!("Bearer realm=\"strata\"{}", if given.is_empty() { "" } else { ", error=\"invalid_token\"" });
        self.json_h(401, &json!({"error": {"type": "authentication_error", "message": "missing or wrong API key"}}), vec![("WWW-Authenticate".into(), challenge)]);
        false
    }

    /// Without an API key, a /v1 request from another site's page is refused (and must be JSON).
    fn foreign_page(&mut self, body: bool) -> bool {
        let Some(origin) = self.header("origin") else { return false };
        if !self.svc().set.api_key.is_empty() {
            return false;
        }
        let s = self.svc().clone();
        let allowed: Vec<String> = s.set.trusted_origins.iter().chain(s.set.cors_origins.iter()).cloned().collect();
        if !crate::net::origin_allowed(&origin, self.header("host").as_deref(), &s.set.host_names, &allowed) {
            say(&format!("refused an API request from the web page {} (no API key; add its host to \"allowed_hosts\" or its origin to \"cors_origins\" in the config)", py::py_repr(&json!(origin))));
            self.json(403, json!({"error": {"type": "forbidden", "message": format!("web pages of {origin} may not use this server without an API key; set \"api_key\", or add the page's host to \"allowed_hosts\" (or its origin to \"cors_origins\") in the config")}}));
            return true;
        }
        if body && !self.header("content-type").unwrap_or_default().starts_with("application/json") {
            self.json(415, json!({"error": {"message": "send application/json"}}));
            return true;
        }
        false
    }

    fn foreign_origin(&self) -> bool {
        let origin = self.header("origin").unwrap_or_default().trim_end_matches('/').to_string();
        if origin.is_empty() {
            return false;
        }
        let host_part = origin.split_once("://").map(|p| p.1).unwrap_or(&origin);
        host_part != self.header("host").unwrap_or_default() && !self.svc().set.trusted_origins.contains(&origin)
    }

    fn own_page(&mut self, what: &str) -> bool {
        if !self.header("content-type").unwrap_or_default().starts_with("application/json") {
            self.json(415, json!({"error": {"message": "send application/json"}}));
            return false;
        }
        if self.foreign_origin() {
            self.json(403, json!({"error": {"message": format!("{what} only from Strata's own page (or an origin in the config's trusted_origins)")}}));
            return false;
        }
        true
    }

    fn body(&self) -> Result<Bytes, Fail> {
        self.body.clone().map_err(|(code, m)| Fail::Session { kind: format!("http{code}"), message: m, published: false })
    }

    /// #430 #431: cancel the request as soon as its client hangs up (or POST /cancel names it).
    fn watch_client(&mut self, cancel: &Arc<Flag>) {
        let done = Arc::new(Flag::new());
        self.watch_done = Some(done.clone());
        let rid = self.header("x-strata-request-id").unwrap_or_default();
        if rid.len() == 32 && rid.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()) {
            let mut c = self.app.cancels.lock().unwrap();
            let registered = c.entry(rid.clone()).or_insert_with(|| cancel.clone());
            if Arc::ptr_eq(registered, cancel) {
                drop(c);
                self.cancel_id = Some(rid.clone());
                if self.app.early.lock().unwrap().shift_remove(&rid) {
                    cancel.set();
                }
            }
        }
        let (gone, cancel, rec) = (self.gone.clone(), cancel.clone(), self.record.clone());
        std::thread::spawn(move || {
            while !done.is_set() && !cancel.is_set() {
                if gone.wait(Some(Duration::from_millis(500))) {
                    if let Some(r) = &rec {
                        r.lock().unwrap().insert("outcome".into(), json!("disconnected"));
                    }
                    cancel.set();
                    return;
                }
            }
        });
    }

    fn sse(&mut self) -> Result<(), Closed> {
        self.note(vec![("http_status", json!(200))]);
        let mut h = vec![("Content-Type".to_string(), "text/event-stream".to_string()), ("Cache-Control".into(), "no-cache".into()), ("X-Accel-Buffering".into(), "no".into())];
        h.extend(self.cors());
        self.send_head(200, h)
    }

    /// #332: the monitor's copy of what went out (nothing when the monitor is off).
    fn capture(&self, api: &str, item: &Value, name: Option<&str>) {
        let Some(rec) = &self.record else { return };
        let (content, reasoning, usage, timings) = match api {
            "openai" => {
                let d = &item["choices"][0]["delta"];
                (d.get("content").and_then(Value::as_str).unwrap_or(""), d.get("reasoning_content").and_then(Value::as_str).unwrap_or(""), item.get("usage"), item.get("timings"))
            }
            "responses" => {
                let kind = item["type"].as_str().unwrap_or("");
                (if kind == "response.output_text.delta" { item["delta"].as_str().unwrap_or("") } else { "" }, if kind == "response.reasoning_text.delta" { item["delta"].as_str().unwrap_or("") } else { "" }, item.get("response").and_then(|r| r.get("usage")), None)
            }
            _ => {
                let _ = name;
                let d = item.get("delta");
                (d.and_then(|d| d.get("text")).and_then(Value::as_str).unwrap_or(""), d.and_then(|d| d.get("thinking")).and_then(Value::as_str).unwrap_or(""), item.get("usage"), None)
            }
        };
        let mut r = rec.lock().unwrap();
        for (nm, value) in [("output", content), ("reasoning", reasoning)] {
            if !value.is_empty() {
                let combined = format!("{}{}", r[nm].as_str().unwrap_or(""), value);
                let n = combined.chars().count();
                r.insert(nm.into(), json!(combined.chars().take(262144).collect::<String>()));
                let t = r[&format!("{nm}_truncated")].as_bool().unwrap_or(false) || n > 262144;
                r.insert(format!("{nm}_truncated"), json!(t));
            }
        }
        if let Some(u) = usage.filter(|u| truthy(u)) {
            r.insert("usage".into(), u.clone());
        }
        if let Some(t) = timings.filter(|t| truthy(t)) {
            r.insert("timings".into(), t.clone());
        }
    }
}

fn dispatch(mut c: Ctx) {
    // the Host check (DNS rebinding) before anything, unless there is an API key
    let host = c.header("host");
    let any = c.svc().set.allowed_hosts.iter().any(|h| h == "*");
    if c.svc().set.api_key.is_empty() && !crate::net::host_allowed(host.as_deref(), &c.svc().set.host_names, any) {
        let h = py::py_repr(&host.clone().map(Value::from).unwrap_or(Value::Null));
        say(&format!("refused a request for Host {h} from {}: not a name this server answers to (add it to \"allowed_hosts\" in the config or STRATA_ALLOWED_HOSTS, or set an API key)", c.peer));
        c.json(403, json!({"error": {"type": "forbidden", "message": format!("Host {h} is not allowed (DNS rebinding protection). Reaching Strata under this name on purpose? Add it to \"allowed_hosts\" in the config (strata-<model>.json) or to the STRATA_ALLOWED_HOSTS environment variable, or set an API key (\"api_key\"), which turns this check off")}}));
        return;
    }
    match c.method.as_str() {
        "GET" => get_(&mut c),
        "POST" => post(&mut c),
        "DELETE" => delete(&mut c),
        "OPTIONS" => {
            let mut h = c.cors();
            h.push(("Content-Length".into(), "0".into()));
            let _ = c.send_head(204, h);
        }
        "HEAD" => get_(&mut c),
        _ => {
            let body = b"<!DOCTYPE HTML>\n<html><body><h1>Error response</h1><p>Error code: 501</p><p>Message: Unsupported method.</p></body></html>\n";
            let _ = c.send_head(501, vec![("Content-Type".into(), "text/html;charset=utf-8".into()), ("Content-Length".into(), body.len().to_string())]);
            let _ = c.write(body);
        }
    }
}

fn web_dir() -> std::path::PathBuf {
    crate::root().join("server").join("web")
}

/// The web app's files as one short hash (a tab left open across an update reloads itself).
pub fn web_version() -> String {
    let mut h = sha2::Sha256::new();
    for name in ["index.html", "app.js", "app.css"] {
        h.update(std::fs::read(web_dir().join(name)).unwrap_or_default());
    }
    hex::encode(h.finalize())[..16].to_string()
}

fn get_(c: &mut Ctx) {
    let path = c.path();
    let svc = c.svc().clone();
    if let Some(name) = path.strip_prefix("/fonts/") {
        let f = web_dir().join("fonts").join(name);
        if name.contains(['/', '\\']) || !name.ends_with(".woff2") || !f.is_file() {
            return c.json(404, json!({"error": {"message": "not found"}}));
        }
        return c.static_(&std::fs::read(f).unwrap_or_default(), "font/woff2", "max-age=86400");
    }
    if let Some(name) = path.strip_prefix("/web/") {
        let ext = std::path::Path::new(name).extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default();
        let ctype = match ext.as_str() {
            "css" => "text/css; charset=utf-8",
            "js" => "text/javascript; charset=utf-8",
            "svg" => "image/svg+xml",
            _ => "",
        };
        let f = web_dir().join(name);
        if name.contains(['/', '\\']) || ctype.is_empty() || !f.is_file() {
            return c.json(404, json!({"error": {"message": "not found"}}));
        }
        return c.static_(&std::fs::read(f).unwrap_or_default(), ctype, "no-cache");
    }
    if path == "/metrics/prometheus" {
        if c.authorized() {
            let body = svc.prometheus().into_bytes();
            if c.send_head(200, vec![("Content-Type".into(), "text/plain; version=0.0.4; charset=utf-8".into()), ("Content-Length".into(), body.len().to_string())]).is_ok() {
                let _ = c.write(&body);
            }
        }
        return;
    }
    if path == "/metrics" {
        if crate::prometheus::wants_prometheus(&c.header("accept").unwrap_or_default(), &c.query()) {
            if c.authorized() {
                let body = crate::prometheus::render(&svc.metrics(false), &svc.latencies.snapshot()).into_bytes();
                if c.send_head(200, vec![("Content-Type".into(), crate::prometheus::CONTENT_TYPE.into()), ("Content-Length".into(), body.len().to_string())]).is_ok() {
                    let _ = c.write(&body);
                }
            }
            return;
        }
        if c.authorized() {
            let m = svc.metrics(c.raw_path.contains("requests=all"));
            c.json(200, m);
        }
        return;
    }
    if path == "/firewall" {
        if c.authorized() {
            let limit = c.query_param("limit").and_then(|v| v.parse().ok()).unwrap_or(500);
            c.json(200, crate::firewall::history(limit));
        }
        return;
    }
    if path == "/connections" {
        if c.authorized() {
            let all = c.query_param("all").is_some_and(|v| v == "1");
            c.json(200, crate::firewall::connections(all));
        }
        return;
    }
    if path == "/api/requests" && svc.set.api_monitor {
        if c.authorized() {
            let id = c.query_param("id");
            match svc.request_records(id.as_deref()) {
                None => c.json(404, json!({"error": {"message": "request no longer retained"}})),
                Some(r) if id.is_some() => c.json(200, r),
                Some(r) => c.json(200, json!({"requests": r, "retention": 100, "persistent": false, "loaded": svc.loaded(), "auto_load": svc.engine.strata().is_some()})),
            }
        }
        return;
    }
    if path == "/settings" {
        if c.authorized() {
            let s = svc.shared.lock().unwrap().clone();
            c.json(200, json!({"shared": !s.is_empty(), "defaults": s}));
        }
        return;
    }
    if path == "/config" {
        if c.authorized() {
            config_get(c);
        }
        return;
    }
    if path == "/mcp" {
        if c.authorized() {
            let s = svc.mcp.as_ref().map(|m| m.status()).unwrap_or(json!({"servers": [], "tools": 0}));
            c.json(200, s);
        }
        return;
    }
    if path.is_empty() || (path == "/api-monitor" && svc.set.api_monitor) {
        let body = std::fs::read(web_dir().join(if path.is_empty() { "index.html" } else { "monitor.html" })).unwrap_or_default();
        return c.static_(&body, "text/html; charset=utf-8", "no-cache");
    }
    match path.as_str() {
        "/health" | "/api/health" => c.json(200, json!({"status": "ok", "max_context": svc.reported_ctx(), "model": svc.model, "images": svc.vision.is_some(), "api_key": !svc.set.api_key.is_empty(), "image_generation": svc.image.is_some(), "video_generation": svc.video.is_some(), "web": web_version(), "loaded": svc.loaded(), "service": "strata"})),
        "/status" => {
            if !c.authorized() {
                return;
            }
            let mut s = svc.state.lock().unwrap().status.clone();
            let t = crate::sync::now();
            if truthy(s.get("busy").unwrap_or(&Value::Null)) {
                s.insert("elapsed_s".into(), json!(crate::engine::round1(t - s.get("started").and_then(Value::as_f64).unwrap_or(t))));
                if s.get("first_token").map(truthy).unwrap_or(false) {
                    s.insert("tokens_per_s".into(), json!(crate::engine::round1(svc.tok_s())));
                    s.insert("tokens_per_s_mean".into(), json!(crate::engine::round1(svc.tok_s_mean())));
                }
            }
            s.shift_remove("started");
            s.shift_remove("first_token");
            c.json(200, Value::Object(s));
        }
        "/v1/models" | "/models" => {
            if !c.authorized() {
                return;
            }
            let mut loaded = svc.loaded();
            let mut model = json!({"id": svc.model, "object": "model", "status": {"value": "loaded"}, "meta": {"n_ctx": svc.reported_ctx()}, "architecture": {"input_modalities": if svc.vision.is_some() { json!(["text", "image"]) } else { json!(["text"]) }, "output_modalities": ["text"]}});
            if !loaded && (svc.set.idle_unload_s != 0.0 || svc.engine.unloaded()) {
                model["status"] = json!({"value": "unloaded"});
                loaded = true;
            }
            if !svc.set.aliases.is_empty() {
                model["aliases"] = json!(svc.set.aliases);
            }
            let mut data = vec![model.clone()];
            for a in &svc.set.aliases {
                let mut m = model.clone();
                m["id"] = json!(a);
                m["alias_of"] = json!(svc.model);
                data.push(m);
            }
            c.json(200, json!({"object": "list", "data": if loaded { json!(data) } else { json!([]) }}));
        }
        "/props" => {
            if c.authorized() {
                props(c);
            }
        }
        "/slots" => {
            if !c.authorized() {
                return;
            }
            let loaded = svc.loaded();
            let (busy, in_use) = {
                let s = svc.state.lock().unwrap();
                (truthy(s.status.get("busy").unwrap_or(&Value::Null)), s.status.get("prompt_tokens").and_then(Value::as_i64).unwrap_or(0))
            };
            let slots: Vec<Value> = match svc.engine.strata().filter(|e| e.batch() > 0) {
                Some(e) => e.slots_view().iter().map(|s| json!({"id": s["slot"], "n_ctx": svc.reported_ctx(), "is_processing": s["state"] != "idle", "n_prompt_tokens": s["prompt_tokens"].as_i64().filter(|n| *n != 0).or(s["held_tokens"].as_i64()).unwrap_or(0)})).collect(),
                None => vec![json!({"id": 0, "n_ctx": svc.reported_ctx(), "is_processing": busy, "n_prompt_tokens": in_use})],
            };
            c.json(200, if loaded { json!(slots) } else { json!([]) });
        }
        "/v1/status" => {
            if c.authorized() {
                let v = svc.v1_status();
                c.json(200, v);
            }
        }
        p if p == "/v1/videos" || p.starts_with("/v1/videos/") => {
            if c.authorized() && !c.foreign_page(false) {
                videos_get(c, &path);
            }
        }
        _ => c.json(404, json!({"error": {"message": "not found"}})),
    }
}

fn props(c: &mut Ctx) {
    let svc = c.svc().clone();
    let model = c.query_param("model").unwrap_or_else(|| svc.model.clone());
    if !svc.model_names().contains(&model) {
        return c.json(404, json!({"error": {"message": "model not found"}}));
    }
    if !svc.loaded() && !svc.engine.unloaded() {
        return c.json(503, json!({"error": {"message": "the engine is not running"}}));
    }
    let shared = svc.shared.lock().unwrap().clone();
    let mut defaults = svc.set.sampling_defaults.clone();
    for (k, v) in &shared {
        defaults.insert(k.clone(), v.clone());
    }
    let mut params = Map::new();
    for (k, v) in &defaults {
        if ["temperature", "top_p", "top_k", "min_p", "seed", "repetition_penalty", "presence_penalty", "frequency_penalty", "penalty_last_n"].contains(&k.as_str()) {
            let name = match k.as_str() {
                "repetition_penalty" => "repeat_penalty",
                "penalty_last_n" => "repeat_last_n",
                x => x,
            };
            params.insert(name.into(), v.clone());
        }
    }
    params.insert("n_predict".into(), shared.get("max_tokens").cloned().unwrap_or(json!(-1)));
    let mut p = json!({"default_generation_settings": {"n_ctx": svc.reported_ctx(), "params": params}, "total_slots": svc.engine.batch().max(1),
        "model_alias": svc.model, "chat_template": svc.template.source, "chat_template_caps": svc.template.caps,
        "modalities": {"vision": svc.vision.is_some()}, "models_autoload": svc.engine.strata().is_some(), "is_sleeping": !svc.loaded()});
    if let Some(e) = svc.engine.strata() {
        p["model_path"] = json!(e.model_path);
    }
    if let Some(v) = svc.engine.info().get("version").filter(|v| truthy(v)) {
        p["build_info"] = json!(format!("Strata {}", py::py_str(v)));
    }
    c.json(200, p);
}

fn config_get(c: &mut Ctx) {
    let Some(path) = c.svc().set.config_path.clone() else { return c.json(404, json!({"error": {"message": "this server was started without a run config"}})) };
    match crate::runconfig::load(&path) {
        Err(e) => c.json(500, json!({"error": {"type": "server_error", "message": format!("the run config cannot be read: {e}")}})),
        Ok(cfg) => c.json(200, crate::runconfig::view(&cfg, &path)),
    }
}

fn videos_get(c: &mut Ctx, path: &str) {
    let Some(g_) = c.svc().video.clone() else { return c.json(404, json!({"error": {"message": "this server makes no videos"}})) };
    let parts: Vec<&str> = path.split('/').skip(3).collect();
    if parts.is_empty() {
        return c.json(200, json!({"object": "list", "data": g_.list()}));
    }
    let job = g_.get(parts[0]);
    match job {
        Some(j) if parts.len() == 1 => c.json(200, j.view(crate::videogen::MODEL, g_.keep_s)),
        Some(j) if parts.len() == 2 && parts[1] == "content" => {
            let st = crate::videogen::status_of(&j);
            if st != "completed" {
                return c.json(409, json!({"error": {"type": "invalid_request_error", "message": format!("the video is {st}, not completed")}}));
            }
            let f = crate::videogen::job_file(&j);
            match std::fs::read(&f) {
                Ok(data) => c.static_(&data, if f.extension().map(|e| e == "mp4").unwrap_or(false) { "video/mp4" } else { "video/webm" }, "no-store"),
                Err(_) => c.json(404, json!({"error": {"type": "invalid_request_error", "message": "no such video"}})),
            }
        }
        _ => c.json(404, json!({"error": {"type": "invalid_request_error", "message": "no such video"}})),
    }
}

fn delete(c: &mut Ctx) {
    let path = c.path();
    if !c.authorized() || c.foreign_page(false) {
        return;
    }
    if let Some(name) = path.strip_prefix("/mcp/servers/") {
        let name = percent_encoding::percent_decode_str(name).decode_utf8_lossy().into_owned();
        return mcp_delete(c, &name);
    }
    if path.starts_with("/v1/videos/") && path.matches('/').count() == 3 {
        if let Some(v) = c.svc().video.clone() {
            let jid = path.rsplit('/').next().unwrap_or("").to_string();
            if v.delete(&jid) {
                return c.json(200, json!({"id": jid, "object": "video.deleted", "deleted": true}));
            }
        }
    }
    c.json(404, json!({"error": {"type": "invalid_request_error", "message": "no such video"}}));
}

fn post(c: &mut Ctx) {
    if let Err((code, m)) = c.body.clone() {
        if c.authorized() {
            c.json(code, json!({"error": {"type": "invalid_request_error", "message": m}}));
        }
        return;
    }
    if !c.authorized() {
        return;
    }
    let path = c.path();
    if path.starts_with("/v1/") && c.foreign_page(true) {
        return;
    }
    match path.as_str() {
        "/settings" => return settings(c),
        "/config" => return config_post(c),
        "/mcp/servers" => return mcp_put(c),
        _ => {}
    }
    if (path == "/unload" || path == "/load") && c.body.as_ref().map(|b| b.len() > 65536).unwrap_or(false) {
        return c.json(413, json!({"error": {"message": "control request body is limited to 64 KiB"}}));
    }
    if (path == "/unload" || path == "/load") && !c.own_page("the model can be loaded or unloaded") {
        return;
    }
    if path.starts_with("/slots/") && !c.own_page("conversations can be saved or restored") {
        return;
    }
    let svc = c.svc().clone();
    if path == "/cancel" {
        if !c.own_page("a request can be cancelled") {
            return;
        }
        let rid = c.body().ok().and_then(|b| serde_json::from_slice::<Value>(if b.is_empty() { b"{}" } else { &b }).ok()).and_then(|v| v.get("id").and_then(Value::as_str).map(String::from));
        let event = rid.as_ref().and_then(|r| c.app.cancels.lock().unwrap().get(r).cloned());
        match (&event, &rid) {
            (Some(e), _) => e.set(),
            (None, Some(r)) if r.len() == 32 && r.chars().all(|ch| ch.is_ascii_hexdigit() && !ch.is_ascii_uppercase()) => {
                let mut early = c.app.early.lock().unwrap();
                early.insert(r.clone());
                while early.len() > 256 {
                    early.shift_remove_index(0);
                }
            }
            _ => {}
        }
        return c.json(if event.is_some() { 200 } else { 404 }, json!({"cancelled": event.is_some()}));
    }
    if path == "/unload" {
        match svc.unload(None) {
            Err(e) => c.json(503, json!({"error": {"type": "server_error", "message": e.to_string()}})),
            Ok(r) => c.json(if r == "busy" { 409 } else { 200 }, json!({"status": r})),
        }
        return;
    }
    if path == "/load" {
        match svc.load() {
            Ok(()) => c.json(200, json!({"status": "loaded"})),
            Err(e @ Fail::GpuBusy(_)) => c.json(503, json!({"error": {"type": "server_error", "message": e.to_string()}})),
            Err(e) => error_answer(c, &path, e),
        }
        return;
    }
    let r = post_api(c, &path);
    if let Err(e) = r {
        error_answer(c, &path, e);
    }
    // finally: the images' file of a request that never got to run(), the watcher, the cancel id, the record
    svc.drop_embeddings();
    if let Some(w) = c.watch_done.take() {
        w.set();
    }
    if let Some(id) = c.cancel_id.take() {
        c.app.cancels.lock().unwrap().remove(&id);
    }
    if let Some(rec) = c.record.take() {
        let mut r = rec.lock().unwrap();
        let wc = crate::service::pyround(crate::service::clock() - r.get("_clock").and_then(Value::as_f64).unwrap_or(0.0), 3);
        r.insert("wallclock_s".into(), json!(wc));
        r.insert("finished_at".into(), json!(crate::sync::now()));
        let state = if r.get("error").map(|e| !e.is_null()).unwrap_or(false) { json!("error") } else { r.get("outcome").cloned().unwrap_or(json!("completed")) };
        r.insert("state".into(), state);
        TRACE.with(|t| *t.borrow_mut() = None);
    }
}

/// _post's except clauses: the status each failure answers with, before an answer started.
fn error_answer(c: &mut Ctx, path: &str, e: Fail) {
    if c.head_sent {
        return;
    }
    let responses = path == "/v1/responses";
    match e {
        Fail::Value(m) | Fail::Session { message: m, .. } => {
            say(&format!("400 invalid request: {m}"));
            if responses {
                c.json(400, rs::error_body(&m, "invalid_request_error", None, None));
            } else {
                c.json(400, json!({"error": {"type": "invalid_request_error", "message": m}}));
            }
        }
        Fail::Busy(m) => c.json(409, json!({"error": {"type": "model_busy", "message": m}})),
        Fail::Structured(m) => c.json(502, json!({"error": {"type": "structured_output_failed", "code": "structured_output_failed", "message": m}})),
        Fail::GpuBusy(m) | Fail::Starting(m) | Fail::Stuck(m) => c.json(503, json!({"error": {"type": "server_error", "message": m}})),
        e @ (Fail::Died(_) | Fail::Silent(_)) => c.json(503, json!({"error": {"type": "server_error", "message": format!("{e}; the next request restarts it")}})),
        Fail::McpCancelled => {}
    }
}

fn parse_req(c: &Ctx) -> Result<Map<String, Value>, Fail> {
    let b = c.body()?;
    let v: Value = serde_json::from_slice(if b.is_empty() { b"{}" } else { &b }).map_err(|e| Fail::Value(e.to_string()))?;
    match v {
        Value::Object(o) => Ok(o),
        _ => Err(Fail::Value("send a JSON object".into())),
    }
}

fn post_api(c: &mut Ctx, path: &str) -> Result<(), Fail> {
    let svc = c.svc().clone();
    let req = parse_req(c)?;
    let rv = Value::Object(req.clone());
    if path.starts_with("/v1/responses/") {
        c.json(404, rs::error_body("this server keeps no responses (stateless); send the whole conversation to POST /v1/responses", "invalid_request_error", None, Some("not_found")));
        return Ok(());
    }
    if path == "/v1/load" || path == "/v1/unload" {
        if !c.own_page("the model can be loaded or unloaded") {
            return Ok(());
        }
        if !get(&rv, "model").is_null() && get(&rv, "model").as_str() != Some(svc.model.as_str()) {
            c.json(404, json!({"error": {"message": "model not found"}}));
            return Ok(());
        }
        let result = if path == "/v1/unload" {
            let r = svc.unload(None)?;
            if r == "busy" {
                return Err(Fail::Busy("a request is running or queued".into()));
            }
            r
        } else {
            if !svc.fifo.try_acquire() {
                return Err(Fail::Busy("a request is running or queued".into()));
            }
            let _g = crate::sync::SemGuard(&svc.fifo);
            {
                let s = svc.state.lock().unwrap();
                if truthy(s.status.get("busy").unwrap_or(&Value::Null)) || truthy(s.status.get("queued").unwrap_or(&Value::Null)) {
                    return Err(Fail::Busy("a request is running or queued".into()));
                }
            }
            svc.ensure_loaded()?;
            svc.touch();
            "loaded"
        };
        let mut out = json!({"status": result}).as_object().cloned().unwrap();
        for (k, v) in svc.v1_status().as_object().cloned().unwrap() {
            out.insert(k, v);
        }
        c.json(200, Value::Object(out));
        return Ok(());
    }
    if path == "/v1/vram" {
        if !c.own_page("the VRAM reserve can be changed") {
            return Ok(());
        }
        let r = get(&rv, "reserve_mib");
        let reserve = match r {
            Value::Null => None,
            v if v.is_i64() && v.as_i64().unwrap() >= 0 => v.as_i64(),
            _ => return Err(Fail::Value("reserve_mib: a whole number of MiB (0 or more), or null for the start's".into())),
        };
        match svc.vram(reserve) {
            Ok(v) => c.json(200, v),
            Err(e) if e.is_died() => c.json(503, json!({"error": {"type": "server_error", "message": e.to_string()}})),
            Err(e) => return Err(e),
        }
        return Ok(());
    }
    if ["/v1/chat/completions", "/v1/messages", "/v1/responses"].contains(&path) {
        c.record = svc.begin_request(path, &rv);
    }
    match path {
        "/v1/responses" => responses(c, req),
        p if p.starts_with("/slots/") => {
            let (code, body) = slot_action(&svc, &p["/slots/".len()..], &c.query_param("action").unwrap_or_default(), get(&rv, "filename"));
            c.json(code, body);
            Ok(())
        }
        "/v1/chat/completions" => openai(c, req),
        "/v1/messages" => anthropic(c, req),
        "/v1/messages/count_tokens" => {
            let req = svc.with_shared(&req, "anthropic");
            let rv = Value::Object(req.clone());
            let (m, t, kw) = anthropic_to_messages(&rv, svc.set.anthropic_think_unasked)?;
            stop_strings(&rv)?;
            check_request_sampling(&mut req.clone())?;
            let n = svc.encode_prompt(&m, t.as_deref(), &kw)?.len();
            c.json(200, json!({"input_tokens": n}));
            Ok(())
        }
        "/context" => context(c, req),
        "/v1/images/generations" => {
            images(c, &rv);
            Ok(())
        }
        "/v1/videos" => {
            match svc.video.clone() {
                None => c.json(404, json!({"error": {"type": "invalid_request_error", "message": "this server makes no videos: add \"video\" to its run config (docs/MACOS.md)"}})),
                Some(v) => {
                    let (code, body) = v.create(svc.clone(), &rv).map_err(Fail::Value)?;
                    let loc = if code == 202 { vec![("Location".to_string(), format!("/v1/videos/{}", body["id"].as_str().unwrap_or("")))] } else { vec![] };
                    c.json_h(code, &body, loc);
                }
            }
            Ok(())
        }
        _ => {
            c.json(404, json!({"error": {"message": "not found"}}));
            Ok(())
        }
    }
}

fn no_local_images(c: &Ctx, messages: &[Value]) -> Result<(), Fail> {
    if c.foreign_origin() && images_of(messages).iter().any(|s| !(s.starts_with("data:") || s.starts_with("http://") || s.starts_with("https://"))) {
        return Err(Fail::Value("a web page of another origin cannot have a file on this computer read as an image: send it as a data: URL (or add the page's origin to the config's trusted_origins)".into()));
    }
    Ok(())
}

/// A page older than this server sends /video or /image as chat: say so rather than run it on the chat model.
fn stale_page_command(c: &Ctx, req: &Value) -> Result<(), Fail> {
    if c.header("x-strata-request-id").is_none() || c.header("x-strata-web").as_deref() == Some(web_version().as_str()) {
        return Ok(());
    }
    let last = get(req, "messages").as_array().and_then(|m| m.iter().rev().find(|m| get(m, "role") == "user").cloned());
    let text = last.map(|l| match get(&l, "content") {
        Value::Array(a) => a.iter().filter(|p| p.is_object()).map(|p| get(p, "text").as_str().unwrap_or("").to_string()).collect::<Vec<_>>().join(" "),
        v => v.as_str().unwrap_or("").to_string(),
    });
    let re = regex::Regex::new(r"^[\s\x1c-\x1f]*/(video|image)[\s\x1c-\x1f]").unwrap();
    if let Some(m) = text.as_deref().and_then(|t| re.captures(t)) {
        let kind = m[1].to_string();
        let available = if kind == "video" { c.svc().video.is_some() } else { c.svc().image.is_some() };
        if available {
            return Err(Fail::Value(format!("this page is older than the server, so it sent /{kind} to the chat model: reload the page (Cmd+R or Ctrl+R) and send it again")));
        }
    }
    Ok(())
}

fn int_of(v: &Value) -> Result<i64, Fail> {
    match v {
        Value::Null => Ok(0),
        Value::Number(n) => Ok(n.as_i64().or_else(|| n.as_f64().map(|f| f.trunc() as i64)).unwrap_or(0)),
        Value::Bool(b) => Ok(*b as i64),
        Value::String(s) => py::strip(s).parse::<i64>().map_err(|_| Fail::Value(format!("invalid literal for int() with base 10: {}", py::py_repr(v)))),
        _ => Err(Fail::Value(format!("int() argument must be a string, a bytes-like object or a real number, not '{}'", py::type_name(v)))),
    }
}

fn openai(c: &mut Ctx, req: Map<String, Value>) -> Result<(), Fail> {
    let svc = c.svc().clone();
    stale_page_command(c, &Value::Object(req.clone()))?;
    let mut req = svc.with_shared(&req, "openai");
    let vocab = Some(svc.tok.vocab_size());
    crate::bias::normalize(req.get("logit_bias"), vocab)?;
    let rv = Value::Object(req.clone());
    let (messages, tools, kw) = openai_to_messages(&rv)?;
    no_local_images(c, &messages)?;
    let prompt_tools = tools.clone();
    let mut tools = tools;
    if tool_choice_of(get(&rv, "tool_choice")).0 == "none" {
        tools = None;
        let stop = stop_strings(&json!({"stop": get(&rv, "stop")}))?;
        if stop.len() < 4 && !stop.iter().any(|s| s == CALL_START) {
            let mut s = stop.clone();
            s.push(CALL_START.into());
            req.insert("stop".into(), json!(s));
        }
    }
    let force = forced_call(get(&rv, "tool_choice"), tools.as_deref());
    let (messages, validator) = prepare_format(get(&rv, "response_format"), messages, false)?;
    if validator.is_some() && (tools.is_some() || truthy(get(&rv, "strata_mcp"))) {
        return Err(Fail::Value("structured response_format with tools/MCP is not supported".into()));
    }
    svc.load()?;
    let mt = if truthy(get(&rv, "max_completion_tokens")) { get(&rv, "max_completion_tokens") } else { get(&rv, "max_tokens") };
    let max_req = int_of(if truthy(mt) { mt } else { &Value::Null })?;
    let info_bias = svc.engine.info().get("logit_bias").and_then(Value::as_i64);
    crate::bias::validate_request(&req, info_bias, svc.engine.batch(), vocab)?;
    let mut use_mcp = get(&rv, "strata_mcp") == &json!(true) && svc.mcp.is_some();
    let own: Vec<String> = tools.iter().flatten().filter_map(|t| get(t, "name").as_str().map(String::from)).collect();
    let mut extra = vec![];
    if use_mcp {
        if !c.own_page("MCP tools can be used") {
            return Ok(());
        }
        let hub = svc.mcp.clone().unwrap();
        hub.wait(Some(10.0));
        extra = hub.template_tools(&own);
        use_mcp = !extra.is_empty();
        let mut all = tools.clone().unwrap_or_default();
        all.extend(extra.iter().cloned());
        tools = if all.is_empty() { None } else { Some(all) };
    }
    if force.is_some() && use_mcp {
        return Err(Fail::Value("a forced tool_choice with MCP tools is not supported".into()));
    }
    svc.reasoning_budget(&Value::Object(req.clone()))?;
    stop_strings(&Value::Object(req.clone()))?;
    check_request_sampling(&mut req)?;
    let shown = tools.clone().or(prompt_tools);
    let (ids, thinking, max_new) = svc.prepare(&messages, shown.as_deref(), &kw, Some(max_req), force.as_deref(), Some(&mut req))?;
    debug_req("openai", &Value::Object(req.clone()), &messages, tools.as_deref(), max_new, thinking, ids.len());
    let cancel = Arc::new(Flag::new());
    c.watch_client(&cancel);
    let mcp = if use_mcp { Some(McpRun { messages: messages.clone(), kw: kw.clone(), max_req, names: extra.iter().filter_map(|t| get(t, "name").as_str().map(String::from)).collect() }) } else { None };
    let rv = Value::Object(req.clone());
    let mut run = Run { svc: &svc, ids, thinking, tools, max_new, sampling: req, cancel: cancel.clone(), force, mcp };
    let stream = truthy(get(&rv, "stream"));
    if !stream {
        let mut chunks = vec![];
        let mut keep = |ch: Option<Value>| -> Result<(), Closed> {
            if let Some(ch) = ch {
                c.capture("openai", &ch, None);
                chunks.push(ch);
            }
            Ok(())
        };
        let r = match &validator {
            Some(v) => structured_chunks(&rv, &mut run, v, &mut keep),
            None => openai_chunks(&rv, &mut run, &mut keep),
        };
        match r {
            Ok(()) => {
                let body = openai_collect(&chunks);
                c.json(200, body);
                Ok(())
            }
            Err(Stop::Closed) => Ok(()),
            Err(Stop::Err(e)) => Err(e),
        }
    } else {
        if c.sse().is_err() {
            cancel.set();
            return Ok(());
        }
        let mut send = |ch: Option<Value>| -> Result<(), Closed> {
            let r = match &ch {
                None => c.write(b": keep-alive\n\n"),
                Some(v) => {
                    c.capture("openai", v, None);
                    c.write(format!("data: {}\n\n", pyjson::dumps_u(v)).as_bytes())
                }
            };
            if r.is_err() {
                c.note(vec![("outcome", json!("disconnected"))]);
                cancel.set();
            }
            r
        };
        let r = match &validator {
            Some(v) => structured_chunks(&rv, &mut run, v, &mut send),
            None => openai_chunks(&rv, &mut run, &mut send),
        };
        match r {
            Ok(()) => {
                let _ = c.write(b"data: [DONE]\n\n");
            }
            Err(Stop::Closed) => {}
            Err(Stop::Err(e)) => {
                let err = match &e {
                    Fail::Died(_) | Fail::Silent(_) => json!({"error": {"type": "server_error", "message": format!("{e}; the next request restarts it")}}),
                    Fail::Structured(m) => json!({"error": {"type": "structured_output_failed", "code": "structured_output_failed", "message": m}}),
                    _ => json!({"error": {"type": "server_error", "message": e.to_string()}}),
                };
                c.note(vec![("error", err["error"].clone())]);
                let _ = c.write(format!("data: {}\n\ndata: [DONE]\n\n", pyjson::dumps(&err)).as_bytes());
            }
        }
        Ok(())
    }
}

fn anthropic(c: &mut Ctx, req: Map<String, Value>) -> Result<(), Fail> {
    let svc = c.svc().clone();
    svc.load()?;
    let mut req = svc.with_shared(&req, "anthropic");
    let rv = Value::Object(req.clone());
    let (messages, tools, kw) = anthropic_to_messages(&rv, svc.set.anthropic_think_unasked)?;
    no_local_images(c, &messages)?;
    let tools = if tool_choice_of(get(&rv, "tool_choice")).0 == "none" { None } else { tools };
    let force = forced_call(get(&rv, "tool_choice"), tools.as_deref());
    let max_new = int_of(if truthy(get(&rv, "max_tokens")) { get(&rv, "max_tokens") } else { &Value::Null })?;
    svc.reasoning_budget(&rv)?;
    stop_strings(&rv)?;
    check_request_sampling(&mut req)?;
    let (ids, thinking, max_new) = svc.prepare(&messages, tools.as_deref(), &kw, Some(max_new), force.as_deref(), Some(&mut req))?;
    debug_req("anthropic", &Value::Object(req.clone()), &messages, tools.as_deref(), max_new, thinking, ids.len());
    let cancel = Arc::new(Flag::new());
    c.watch_client(&cancel);
    let rv = Value::Object(req.clone());
    let mut run = Run { svc: &svc, ids, thinking, tools, max_new, sampling: req, cancel: cancel.clone(), force, mcp: None };
    if !truthy(get(&rv, "stream")) {
        let mut events = vec![];
        let r = anthropic_events(&rv, &mut run, &mut |item| {
            if let Some((name, e)) = item {
                c.capture("anthropic", &e, Some(name));
                events.push((name, e));
            }
            Ok(())
        });
        return match r {
            Ok(()) => {
                let m = anthropic_collect(&events);
                c.json(200, m);
                Ok(())
            }
            Err(Stop::Closed) => Ok(()),
            Err(Stop::Err(e)) => Err(e),
        };
    }
    if c.sse().is_err() {
        cancel.set();
        return Ok(());
    }
    let r = anthropic_events(&rv, &mut run, &mut |item| {
        let r = match &item {
            None => c.write(b": keep-alive\n\n"),
            Some((name, e)) => {
                c.capture("anthropic", e, Some(name));
                c.write(format!("event: {name}\ndata: {}\n\n", pyjson::dumps_u(e)).as_bytes())
            }
        };
        if r.is_err() {
            c.note(vec![("outcome", json!("disconnected"))]);
            cancel.set();
        }
        r
    });
    if let Err(Stop::Err(e)) = r {
        let msg = if e.is_died() { format!("{e}; the next request restarts it") } else { e.to_string() };
        let err = json!({"type": "error", "error": {"type": "api_error", "message": msg}});
        c.note(vec![("error", err["error"].clone())]);
        let _ = c.write(format!("event: error\ndata: {}\n\n", pyjson::dumps(&err)).as_bytes());
    }
    Ok(())
}

fn responses(c: &mut Ctx, req: Map<String, Value>) -> Result<(), Fail> {
    let svc = c.svc().clone();
    let rv = Value::Object(req.clone());
    let model = svc.model_for(&rv);
    if svc.set.codex_thread_titles {
        match rs::thread_title_events(&rv, &model) {
            Err(e) => {
                c.json(e.status, e.body());
                return Ok(());
            }
            Ok(Some(events)) => {
                if !truthy(get(&rv, "stream")) {
                    c.json(200, rs::collect(&events));
                } else if c.sse().is_ok() {
                    for e in events {
                        if c.write(format!("event: {}\ndata: {}\n\n", e["type"].as_str().unwrap_or(""), pyjson::dumps_u(&e)).as_bytes()).is_err() {
                            c.note(vec![("outcome", json!("disconnected"))]);
                            break;
                        }
                    }
                }
                return Ok(());
            }
            Ok(None) => {}
        }
    }
    let prep = responses_prepare(c, req);
    let (ids, thinking, tools, max_new, mut asm, validator, req) = match prep {
        Ok(p) => p,
        Err(Prep::Resp(e)) => {
            c.json(e.status, e.body());
            return Ok(());
        }
        Err(Prep::Fail(e)) => {
            match e {
                Fail::Busy(m) => c.json(409, rs::error_body(&m, "server_error", None, Some("model_busy"))),
                Fail::GpuBusy(m) | Fail::Starting(m) | Fail::Stuck(m) => c.json(503, rs::error_body(&m, "server_error", None, Some("server_error"))),
                e @ (Fail::Died(_) | Fail::Silent(_)) => c.json(503, rs::error_body(&e.to_string(), "server_error", None, Some("server_error"))),
                e => return Err(e),
            }
            return Ok(());
        }
    };
    let cancel = Arc::new(Flag::new());
    c.watch_client(&cancel);
    let check = |text: &str, finish: &str| -> Result<String, Fail> {
        match &validator {
            Some(v) => validated_json(text, v, finish),
            None => Ok(text.to_string()),
        }
    };
    let rv = Value::Object(req.clone());
    let stream = truthy(get(&rv, "stream"));
    // the events: start, the run's items fed to the assembler, then the end (unless cancelled).  Streamed, a quiet
    // spell becomes response.in_progress every 15 s (Codex's idle timeout counts events, not SSE comments)
    let produce = |send: &mut dyn FnMut(Option<Value>) -> Result<(), Closed>, asm: &mut rs::Assembler| -> Result<(), Stop<Fail>> {
        let mut last = Instant::now();
        for e in asm.start() {
            send(Some(e))?;
        }
        let mut done = None;
        svc.run(&ids, thinking, tools.as_deref(), max_new, &req, &cancel, None, &mut |item| match item {
            crate::run::Item::Ping => {
                if stream && last.elapsed() >= Duration::from_secs(15) {
                    last = Instant::now();
                    send(Some(asm.in_progress()))
                } else {
                    send(None)
                }
            }
            crate::run::Item::Event(ev) => {
                for e in asm.feed(&ev) {
                    last = Instant::now();
                    send(Some(e))?;
                }
                Ok(())
            }
            crate::run::Item::Done(d) => {
                done = Some(d);
                Ok(())
            }
            crate::run::Item::Mcp(_) => Ok(()),
        })?;
        if let Some(d) = done {
            if d.get("finish").and_then(Value::as_str) != Some("cancel") && !cancel.is_set() {
                for e in asm.finish(&d, &check).map_err(Stop::Err)? {
                    send(Some(e))?;
                }
            }
        }
        Ok(())
    };
    if !stream {
        let mut items: Vec<Value> = vec![];
        let r = produce(&mut |e| {
            if let Some(e) = e {
                c.capture("responses", &e, None);
                items.push(e);
            }
            Ok(())
        }, &mut asm);
        match r {
            Ok(()) => c.json(200, rs::collect(&items)),
            Err(Stop::Closed) => {}
            Err(Stop::Err(Fail::Structured(m))) => c.json(502, rs::error_body(&m, "server_error", None, Some("structured_output_failed"))),
            Err(Stop::Err(e)) if e.is_died() => c.json(503, rs::error_body(&format!("{e}; the next request restarts it"), "server_error", None, Some("server_error"))),
            Err(Stop::Err(e)) => c.json(500, rs::error_body(&e.to_string(), "server_error", None, Some("server_error"))),
        }
        return Ok(());
    }
    if c.sse().is_err() {
        cancel.set();
        return Ok(());
    }
    let r = produce(&mut |e: Option<Value>| -> Result<(), Closed> {
        let r = match e {
            None => c.write(b": keep-alive\n\n"),
            Some(ev) => {
                c.capture("responses", &ev, None);
                c.write(format!("event: {}\ndata: {}\n\n", ev["type"].as_str().unwrap_or(""), pyjson::dumps_u(&ev)).as_bytes())
            }
        };
        if r.is_err() {
            c.note(vec![("outcome", json!("disconnected"))]);
            cancel.set();
        }
        r
    }, &mut asm);
    if let Err(Stop::Err(e)) = r {
        let (msg, code) = match &e {
            Fail::Structured(m) => (m.clone(), "structured_output_failed"),
            e if e.is_died() => (format!("{e}; the next request restarts it"), "server_error"),
            e => (e.to_string(), "server_error"),
        };
        let ev = asm.failed(&msg, code);
        c.note(vec![("error", ev["response"]["error"].clone())]);
        let _ = c.write(format!("event: {}\ndata: {}\n\n", ev["type"].as_str().unwrap_or(""), pyjson::dumps_u(&ev)).as_bytes());
    }
    Ok(())
}

enum Prep {
    Resp(RespErr),
    Fail(Fail),
}

impl From<RespErr> for Prep {
    fn from(e: RespErr) -> Self {
        Prep::Resp(e)
    }
}

type Prepared = (Vec<u32>, bool, Option<Vec<Value>>, i64, rs::Assembler, Option<crate::structured::Validator>, Map<String, Value>);

fn responses_prepare(c: &mut Ctx, req: Map<String, Value>) -> Result<Prepared, Prep> {
    let svc = c.svc().clone();
    let mut req = req;
    let rv = Value::Object(req.clone());
    rs::check_request(&rv)?;
    let messages = rs::input_messages(&rv)?;
    let (tools, names, skipped) = rs::request_tools(&rv)?;
    let shared = svc.shared.lock().unwrap().clone();
    let kw = rs::template_kwargs(&rv, &shared)?;
    no_local_images(c, &messages).map_err(|e| RespErr::new(e.to_string(), Some("input"), None))?;
    let (messages, validator) = prepare_format(&rs::text_format(&rv)?, messages, tools.is_some()).map_err(|e| RespErr::new(e.to_string().replace("response_format", "text.format"), Some("text.format"), None))?;
    {
        let mut noted = svc.hosted_tools_noted.lock().unwrap();
        let mut fresh: Vec<String> = skipped.iter().filter(|s| !noted.contains(*s)).cloned().collect();
        if !fresh.is_empty() {
            fresh.sort();
            fresh.dedup();
            say(&format!("/v1/responses: left out the hosted tools {} (this server cannot run them)", fresh.join(", ")));
            noted.extend(skipped.iter().cloned());
        }
    }
    if get(&rv, "max_output_tokens").is_null() {
        if let Some(mt) = shared.get("max_tokens") {
            req.insert("max_output_tokens".into(), mt.clone());
        }
    }
    let rv = Value::Object(req.clone());
    svc.reasoning_budget(&rv).map_err(|e| RespErr::new(e.to_string(), Some("reasoning_budget_tokens"), None))?;
    stop_strings(&rv).map_err(|e| RespErr::new(e.to_string(), Some("stop"), None))?;
    check_request_sampling(&mut req).map_err(|e| {
        let m = e.to_string();
        let p = m.split(':').next().unwrap_or("").to_string();
        RespErr::new(m, Some(&p), None)
    })?;
    svc.load().map_err(Prep::Fail)?;
    let rv = Value::Object(req.clone());
    let shown = if svc.set.codex_compaction_cache { rs::prompt_tools(&rv, tools.clone()) } else { tools.clone() };
    let max_out = get(&rv, "max_output_tokens").as_i64().unwrap_or(0);
    let to_resp = |e: Fail| -> Prep {
        match e {
            Fail::Value(m) => {
                let code = if m.contains("context") { Some("context_length_exceeded") } else { None };
                Prep::Resp(RespErr::new(m, Some("input"), code))
            }
            e => Prep::Fail(e),
        }
    };
    let mut shown_used = shown.clone();
    let (ids, thinking, max_new) = match svc.prepare(&messages, shown.as_deref(), &kw, Some(max_out), None, Some(&mut req)) {
        Ok(r) => r,
        Err(Fail::Value(_)) if shown != tools => {
            shown_used = tools.clone();
            svc.prepare(&messages, tools.as_deref(), &kw, Some(max_out), None, Some(&mut req)).map_err(to_resp)?
        }
        Err(e) => return Err(to_resp(e)),
    };
    if svc.set.codex_compaction_cache {
        rs::prompt_made(&Value::Object(req.clone()), shown_used.as_deref());
    }
    debug_req("responses", &Value::Object(req.clone()), &messages, tools.as_deref(), max_new, thinking, ids.len());
    let include = get(&Value::Object(req.clone()), "include").as_array().cloned().unwrap_or_default();
    let asm = rs::Assembler::new(&Value::Object(req.clone()), &svc.model_for(&Value::Object(req.clone())), ids.len() as i64, names, include.contains(&json!("reasoning.encrypted_content")), validator.is_some());
    Ok((ids, thinking, tools, max_new, asm, validator, req))
}

fn context(c: &mut Ctx, req: Map<String, Value>) -> Result<(), Fail> {
    let svc = c.svc().clone();
    let req = svc.with_shared(&req, "openai");
    let rv = Value::Object(req.clone());
    let (mut messages, mut tools, kw) = openai_to_messages(&rv)?;
    if messages.is_empty() {
        messages = vec![json!({"role": "user", "content": ""})];
    }
    if get(&rv, "strata_mcp") == &json!(true) {
        if let Some(hub) = &svc.mcp {
            let own: Vec<String> = tools.iter().flatten().filter_map(|t| get(t, "name").as_str().map(String::from)).collect();
            let mut all = tools.clone().unwrap_or_default();
            all.extend(hub.template_tools(&own));
            tools = if all.is_empty() { None } else { Some(all) };
        }
    }
    let total = svc.encode_prompt(&messages, tools.as_deref(), &kw)?.len() as i64;
    let tools_n = if tools.is_some() { total - svc.encode_prompt(&messages, None, &kw)?.len() as i64 } else { 0 };
    let mut parts: indexmap::IndexMap<&str, i64> = [("tools", tools_n), ("system", 0), ("user", 0), ("assistant", 0), ("tool_io", 0), ("draft", 0)].into_iter().collect();
    let draft = if truthy(get(&rv, "strata_draft")) && get(messages.last().unwrap(), "role") == "user" { messages.len() as i64 - 1 } else { -1 };
    let n = |t: &str| if t.is_empty() { 0 } else { svc.tok.encode(t, false).len() as i64 };
    for (i, m) in messages.iter().enumerate() {
        let cn = get(m, "content");
        let text = match cn {
            Value::String(s) => s.clone(),
            Value::Array(a) => a.iter().filter(|x| x.is_object()).map(|x| get(x, "text").as_str().unwrap_or("").to_string()).collect(),
            _ => String::new(),
        };
        let role = get(m, "role").as_str().unwrap_or("");
        let key = if i as i64 == draft { "draft" } else if role == "system" { "system" } else if role == "tool" { "tool_io" } else if role == "assistant" { "assistant" } else { "user" };
        *parts.get_mut(key).unwrap() += n(&text);
        let tio: i64 = get(m, "tool_calls").as_array().into_iter().flatten().map(|x| n(&pyjson::dumps_u(get(get(x, "function"), "arguments")))).sum();
        *parts.get_mut("tool_io").unwrap() += tio;
    }
    let sum: i64 = parts.values().sum();
    let mut po: Map<String, Value> = parts.iter().map(|(k, v)| (k.to_string(), json!(v))).collect();
    po.insert("formatting".into(), json!((total - sum).max(0)));
    let images: usize = messages.iter().filter_map(|m| get(m, "content").as_array()).flatten().filter(|x| x.is_object() && get(x, "type") != "text").count();
    c.json(200, json!({"max_context": svc.reported_ctx(), "total": total, "parts": po, "images": images}));
    Ok(())
}

fn images(c: &mut Ctx, req: &Value) {
    let svc = c.svc().clone();
    let Some(g_) = svc.image.clone() else { return c.json(404, json!({"error": {"type": "invalid_request_error", "message": "this server makes no pictures: add \"image\" to its run config (docs/MACOS.md)"}})) };
    if let Some((code, body)) = g_.validate(req) {
        return c.raw(code, &body, None);
    }
    let cancel = Arc::new(Flag::new());
    c.watch_client(&cancel);
    let deadline = Instant::now() + Duration::from_secs_f64(g_.queue_s);
    let mut got = false;
    while !got && !cancel.is_set() && Instant::now() < deadline {
        got = svc.fifo.acquire_timeout(deadline.saturating_duration_since(Instant::now()).min(Duration::from_secs(1)));
    }
    let err = |m: &str| crate::imagegen::error(m, "server_error");
    if !got {
        if cancel.is_set() {
            return c.raw(499, &err("the picture was cancelled before it started"), None);
        }
        return c.raw(503, &err(&format!("busy: no turn within {:.0} s", g_.queue_s)), Some("60".into()));
    }
    svc.image_gate.set();
    let room = crate::imagegen::make_room(&svc, g_.need_gb, deadline, &cancel);
    let (code, retry, data) = match room {
        "cancelled" => (499, None, err("the picture was cancelled before it started")),
        "enough" | "unloaded" => {
            say(&format!("picture {}: {}", get(req, "size").as_str().unwrap_or("1024x1024"), if room == "unloaded" { "the chat model unloaded for it" } else { "memory enough" }));
            g_.forward(req, &cancel)
        }
        r => {
            let why = match r {
                "busy" => "the chat model is answering and memory is short",
                "unsupported" => "the chat model cannot be unloaded and memory is short",
                _ => "memory is short even without the chat model",
            };
            let avail = crate::imagegen::available_gb().map(|a| crate::pyjson::float_repr(crate::service::pyround(a, 1))).unwrap_or_else(|| "None".into());
            (503, Some("60".to_string()), err(&format!("{why} ({avail} GB free, {:.0} GB needed)", g_.need_gb)))
        }
    };
    svc.image_gate.clear();
    svc.fifo.release();
    c.raw(code, &data, retry);
}

fn settings(c: &mut Ctx) {
    if !c.own_page("settings can be changed") {
        return;
    }
    let svc = c.svc().clone();
    let req = c.body().ok().and_then(|b| serde_json::from_slice::<Value>(if b.is_empty() { b"{}" } else { &b }).ok());
    let Some(req) = req.filter(|r| r.get("defaults").is_some()) else {
        return c.json(400, json!({"error": {"type": "invalid_request_error", "message": "the body must be {\"defaults\": {...}} to set them, or {\"defaults\": null} to clear them"}}));
    };
    match svc.set_shared(&req["defaults"]) {
        Err(e) => c.json(400, json!({"error": {"type": "invalid_request_error", "message": e.to_string()}})),
        Ok(shared) => {
            if shared.is_empty() {
                say("other apps use their own settings again");
            } else {
                say(&format!("other apps now use the Chat settings: {}", shared.iter().map(|(k, v)| format!("{k}={}", py::py_str(v))).collect::<Vec<_>>().join(", ")));
            }
            c.json(200, json!({"shared": !shared.is_empty(), "defaults": shared}));
        }
    }
}

fn config_post(c: &mut Ctx) {
    if !c.own_page("the run config can be changed") {
        return;
    }
    let svc = c.svc().clone();
    let Some(path) = svc.set.config_path.clone() else { return c.json(404, json!({"error": {"message": "this server was started without a run config"}})) };
    let req: Value = c.body().ok().and_then(|b| serde_json::from_slice(if b.is_empty() { b"{}" } else { &b }).ok()).unwrap_or(Value::Null);
    let _g = svc.config_lock.lock().unwrap();
    let cfg = match crate::runconfig::load(&path) {
        Ok(c) => c,
        Err(e) => return c.json(500, json!({"error": {"type": "server_error", "message": format!("the run config cannot be written: {e}")}})),
    };
    let (new, changed) = match crate::runconfig::apply(&cfg, get(&req, "set")) {
        Ok(r) => r,
        Err(e) => return c.json(400, json!({"error": {"type": "invalid_request_error", "message": e}})),
    };
    if !changed.is_empty() {
        match crate::runconfig::save(&path, &new) {
            Err(e) => return c.json(500, json!({"error": {"type": "server_error", "message": format!("the run config cannot be written: {e}")}})),
            Ok(bak) => say(&format!("the Settings view changed {} in {} (the earlier file: {bak}); used from the next start", changed.join(", "), std::path::Path::new(&path).file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default())),
        }
    }
    let mut v = crate::runconfig::view(&new, &path).as_object().cloned().unwrap();
    v.insert("changed".into(), json!(changed));
    c.json(200, Value::Object(v));
}

fn mcp_save(svc: &Service, name: &str, entry: Option<&Value>) -> Result<bool, String> {
    let Some(path) = svc.set.config_path.clone() else { return Ok(false) };
    let _g = svc.config_lock.lock().unwrap();
    let cfg = crate::runconfig::load(&path)?;
    let block = get(&cfg, "mcp_servers");
    if !block.is_null() && !block.is_object() {
        return Err("the run config's mcp_servers is not an object".into());
    }
    let mut block = block.as_object().cloned().unwrap_or_default();
    match entry {
        None => {
            block.shift_remove(name);
        }
        Some(e) => {
            block.insert(name.into(), e.clone());
        }
    }
    let mut new = cfg.as_object().cloned().unwrap();
    new.insert("mcp_servers".into(), Value::Object(block));
    crate::runconfig::save(&path, &Value::Object(new)).map_err(|e| e.to_string())?;
    Ok(true)
}

fn mcp_put(c: &mut Ctx) {
    if !c.own_page("MCP servers can be changed") {
        return;
    }
    let svc = c.svc().clone();
    let Some(hub) = svc.mcp.clone() else { return c.json(404, json!({"error": {"message": "not found"}})) };
    let req: Value = c.body().ok().and_then(|b| serde_json::from_slice(if b.is_empty() { b"{}" } else { &b }).ok()).unwrap_or(Value::Null);
    let old = get(&req, "name").as_str().and_then(|n| hub.servers.lock().unwrap().get(n).cloned());
    if let Some(o) = &old {
        if !o.editable {
            return c.json(409, json!({"error": {"type": "invalid_request_error", "message": format!("{} comes from the config file or --mcp-config, not from this page: change it there", py::py_repr(&json!(o.name)))}}));
        }
    }
    let (name, entry) = match crate::mcp::page_entry(&req, old.as_ref().map(|o| &o.cfg)) {
        Ok(r) => r,
        Err(e) => return c.json(400, json!({"error": {"type": "invalid_request_error", "message": e}})),
    };
    let saved = match mcp_save(&svc, &name, Some(&entry)) {
        Ok(s) => s,
        Err(e) => return c.json(500, json!({"error": {"type": "server_error", "message": format!("the run config cannot be written: {e}")}})),
    };
    hub.put(&name, entry.clone());
    let url = entry["url"].as_str().unwrap_or("").split('?').next().unwrap_or("").to_string();
    say(&format!("the Settings view {} the MCP server {} ({url}){}", if old.is_some() { "changed" } else { "added" }, py::py_repr(&json!(name)), if saved { "" } else { "; no run config, so only until this server stops" }));
    let mut st = hub.status().as_object().cloned().unwrap();
    st.insert("saved".into(), json!(saved));
    c.json(200, Value::Object(st));
}

fn mcp_delete(c: &mut Ctx, name: &str) {
    if c.foreign_origin() {
        return c.json(403, json!({"error": {"message": "MCP servers can be changed only from Strata's own page"}}));
    }
    let svc = c.svc().clone();
    let s = svc.mcp.as_ref().and_then(|h| h.servers.lock().unwrap().get(name).cloned());
    match s {
        None => return c.json(404, json!({"error": {"type": "invalid_request_error", "message": format!("no MCP server {}", py::py_repr(&json!(name)))}})),
        Some(s) if !s.editable => return c.json(409, json!({"error": {"type": "invalid_request_error", "message": format!("{} comes from the config file or --mcp-config, not from this page", py::py_repr(&json!(name)))}})),
        _ => {}
    }
    let saved = match mcp_save(&svc, name, None) {
        Ok(s) => s,
        Err(e) => return c.json(500, json!({"error": {"type": "server_error", "message": format!("the run config cannot be written: {e}")}})),
    };
    let hub = svc.mcp.clone().unwrap();
    hub.remove(name);
    say(&format!("the Settings view removed the MCP server {}", py::py_repr(&json!(name))));
    let mut st = hub.status().as_object().cloned().unwrap();
    st.insert("saved".into(), json!(saved));
    c.json(200, Value::Object(st));
}

const WIN_DEVICES: [&str; 4] = ["con", "prn", "aux", "nul"];

/// /slots/0 {"filename": NAME}: why NAME is not one plain file name inside the slot save path (None when it is).
pub fn slot_filename_problem(name: &Value) -> Option<String> {
    let Some(n) = name.as_str().filter(|n| !n.is_empty()) else { return Some("must be a non-empty string".into()) };
    if n.len() > 200 {
        return Some("is too long (at most 200 bytes)".into());
    }
    if n.chars().any(|c| (c as u32) < 32 || c as u32 == 127) {
        return Some("must not contain control characters".into());
    }
    if n.contains(['/', '\\', ':', '*', '?', '"', '<', '>', '|']) {
        return Some("must be a plain file name (no / \\ : * ? \" < > |)".into());
    }
    if n.starts_with('.') {
        return Some("must not start with a dot".into());
    }
    if n.ends_with(['.', ' ']) {
        return Some("must not end with a dot or a space".into());
    }
    let stem = n.split('.').next().unwrap_or("").trim_end_matches(' ').to_lowercase();
    let device = WIN_DEVICES.contains(&stem.as_str()) || stem == "conin$" || stem == "conout$" || ((stem.starts_with("com") || stem.starts_with("lpt")) && stem.chars().count() == 4 && "123456789\u{b9}\u{b2}\u{b3}".contains(stem.chars().nth(3).unwrap()));
    if device {
        return Some("names a Windows device".into());
    }
    None
}

/// llama-server's POST /slots/{id}?action=save|restore {"filename": ...} -> (status, body).
pub fn slot_action(svc: &Arc<Service>, slot: &str, action: &str, filename: &Value) -> (u16, Value) {
    let error = |code: u16, m: String| (code, json!({"error": {"code": code, "message": m, "type": if code < 500 { "invalid_request_error" } else { "server_error" }}}));
    let (Some(dir), Some(engine)) = (svc.set.slot_save_path.clone(), svc.engine.strata().cloned()) else {
        return error(501, "slot save/restore is disabled (start the server with --slot-save-path DIR)".into());
    };
    if engine.batch() > 0 {
        return error(501, "slot save/restore is not available with parallel requests (\"parallel\" / --batch)".into());
    }
    if slot != "0" {
        return error(400, "this server has one slot: 0".into());
    }
    if action != "save" && action != "restore" {
        return error(400, "action must be save or restore".into());
    }
    if let Some(why) = slot_filename_problem(filename) {
        return error(400, format!("filename {why}"));
    }
    let name = filename.as_str().unwrap().to_string();
    let path = std::path::Path::new(&dir).join(&name).to_string_lossy().into_owned();
    if action == "restore" && std::fs::symlink_metadata(&path).is_err() {
        return error(404, format!("no saved session named {name}"));
    }
    svc.touch();
    {
        let mut s = svc.state.lock().unwrap();
        let q = s.status.get("queued").and_then(Value::as_i64).unwrap_or(0) + 1;
        s.status.insert("queued".into(), json!(q));
    }
    svc.fifo.acquire();
    let _g = crate::sync::SemGuard(&svc.fifo);
    {
        let mut s = svc.state.lock().unwrap();
        let q = s.status.get("queued").and_then(Value::as_i64).unwrap_or(0) - 1;
        s.status.insert("queued".into(), json!(q));
    }
    if !svc.loaded() {
        return error(503, "the model is not loaded".into());
    }
    {
        let mut s = svc.state.lock().unwrap();
        for (k, v) in [("busy", json!(true)), ("phase", json!(if action == "save" { "saving the session" } else { "restoring a session" })), ("started", json!(crate::sync::now())), ("first_token", Value::Null), ("prompt_tokens", Value::Null), ("generated", Value::Null), ("max_tokens", Value::Null)] {
            s.status.insert(k.into(), v);
        }
    }
    let r = if action == "save" && engine.session_save_reclaim { session_save_retry(&engine, &path) } else { engine.session_file(action, &path) };
    svc.state.lock().unwrap().status.insert("busy".into(), json!(false));
    svc.touch();
    match r {
        Ok((tokens, bytes, ms)) => {
            if action == "save" {
                (200, json!({"id_slot": 0, "filename": name, "n_saved": tokens, "n_written": bytes, "timings": {"save_ms": ms}}))
            } else {
                (200, json!({"id_slot": 0, "filename": name, "n_restored": tokens, "n_read": bytes, "timings": {"restore_ms": ms}}))
            }
        }
        Err(Fail::Session { kind, message, published }) => {
            let code = Fail::session_status(&kind).unwrap_or(500);
            let (c2, mut body) = error(code, message);
            body["error"]["kind"] = json!(kind);
            if published {
                body["error"]["published"] = json!(true);
            }
            (c2, body)
        }
        Err(Fail::Value(m)) => error(400, m),
        Err(e) if e.is_died() => (500, json!({"error": {"code": 500, "message": format!("{e}; the next request starts the engine again"), "type": "server_error"}})),
        Err(e) => error(500, e.to_string()),
    }
}

/// Bounded SAVE retries after the engine reclaimed cache memory (serve/session_save_retry.py).
fn session_save_retry(engine: &Arc<crate::engine::StrataEngine>, path: &str) -> Result<(i64, i64, f64), Fail> {
    let required = |e: &Fail| -> Option<u64> {
        let Fail::Session { message, .. } = e else { return None };
        let re = regex::Regex::new(r"^not enough RAM to save the session \((\d+) MiB plus a floor of (\d+) MiB needed,").unwrap();
        re.captures(message).map(|c| c[1].parse::<u64>().unwrap_or(0) + c[2].parse::<u64>().unwrap_or(0) + 2)
    };
    let retryable = |e: &Fail| matches!(e, Fail::Session { kind, published: false, .. } if kind == "memory") && required(e).is_some();
    let pid = engine.pid();
    let first = engine.session_file("save", path);
    let Err(mut last) = first else { return first };
    if !retryable(&last) || pid.is_none() {
        return Err(last);
    }
    let deadline = Instant::now() + Duration::from_secs(30);
    let same = || engine.pid() == pid && engine.alive();
    say(&format!("SAVE waiting for RAM after cache reclaim: need={} MiB, limit=30s; floor unchanged", required(&last).unwrap_or(0)));
    for attempt in 0..3 {
        loop {
            if !same() || Instant::now() >= deadline {
                return Err(last);
            }
            let (Some(avail), Some(need)) = (crate::telemetry::available_mib(), required(&last)) else { return Err(last) };
            if avail >= need {
                break;
            }
            std::thread::sleep(deadline.saturating_duration_since(Instant::now()).min(Duration::from_millis(500)));
        }
        if !same() {
            return Err(last);
        }
        match engine.session_file("save", path) {
            Ok(r) => return Ok(r),
            Err(e) => {
                if !retryable(&e) {
                    return Err(e);
                }
                last = e;
                if attempt == 2 || Instant::now() >= deadline {
                    return Err(last);
                }
                std::thread::sleep(deadline.saturating_duration_since(Instant::now()).min(Duration::from_millis(500)));
            }
        }
    }
    Err(last)
}

fn debug_req(api: &str, req: &Value, messages: &[Value], tools: Option<&[Value]>, max_new: i64, thinking: bool, prompt_tokens: usize) {
    if !std::env::var("STRATA_DEBUG").map(|v| !v.is_empty()).unwrap_or(false) {
        return;
    }
    let last = messages.last().cloned().unwrap_or(json!({}));
    let body = match get(&last, "content") {
        Value::Array(a) => a.iter().filter(|p| p.is_object()).map(|p| get(p, "text").as_str().unwrap_or("").to_string()).collect::<Vec<_>>().join(" "),
        v if truthy(v) => py::py_str(v),
        _ => String::new(),
    };
    let preview = py::chars_prefix(&body, 80).replace('\n', " ");
    say(&format!("req {api}: msgs={} tools={} max_tokens_raw={}/{} max_new={max_new} thinking={} stream={} prompt_tokens={prompt_tokens} last={}:{}", messages.len(), tools.map(|t| t.len()).unwrap_or(0), py::py_repr(get(req, "max_tokens")), py::py_repr(get(req, "max_completion_tokens")), if thinking { "True" } else { "False" }, if truthy(get(req, "stream")) { "True" } else { "False" }, py::py_repr(get(&last, "role")), py::py_repr(&json!(preview))));
}

pub use api::prompt_progress;
