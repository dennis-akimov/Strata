//! Tools from MCP (Model Context Protocol) servers for the web app's chats (serve/mcp.py): a program Strata starts
//! (stdio: newline-delimited JSON-RPC 2.0) or an address (Streamable HTTP: JSON-RPC POSTed, answered as JSON or as an
//! event stream).  The hub offers their tools as `<server>__<tool>`, routes a call back, caps a result's text, and
//! turns a failed call into a result that starts with "error:".

use crate::py::{self, get};
use crate::pyjson;
use crate::say;
use crate::sync::Event;
use regex::Regex;
use serde_json::{json, Map, Value};
use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const PROTOCOL: &str = "2025-06-18";

#[derive(Debug, Clone)]
pub enum McpErr {
    Err(String),
    Timeout(String),
    Cancelled,
}

impl std::fmt::Display for McpErr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            McpErr::Err(m) | McpErr::Timeout(m) => write!(f, "{m}"),
            McpErr::Cancelled => write!(f, "stopped"),
        }
    }
}

type Answer = Result<Value, McpErr>;

/// Wait for a response, checking the chat's stop button and the deadline a few times a second.
fn wait(rx: &crossbeam_channel::Receiver<Answer>, timeout: f64, cancel: Option<&Event>, what: &str) -> Result<Value, McpErr> {
    let deadline = Instant::now() + Duration::from_secs_f64(timeout);
    loop {
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(Ok(msg)) => {
                if let Some(e) = msg.get("error") {
                    let err = if e.is_object() { e.clone() } else { json!({"message": py::py_str(e)}) };
                    let m = get(&err, "message");
                    return Err(McpErr::Err(format!("{} (code {})", if py::truthy(m) { py::py_str(m) } else { "error".into() }, py::py_str(get(&err, "code")).replace("None", "None"))));
                }
                let r = get(&msg, "result");
                return Ok(if py::truthy(r) { r.clone() } else { json!({}) });
            }
            Ok(Err(e)) => return Err(e),
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => return Err(McpErr::Err("the server stopped".into())),
            Err(_) => {
                if cancel.map(|c| c.is_set()).unwrap_or(false) {
                    return Err(McpErr::Cancelled);
                }
                if Instant::now() > deadline {
                    return Err(McpErr::Timeout(format!("{what} timed out after {} s", crate::engine::py_g(timeout))));
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ transports
struct Stdio_ {
    name: String,
    cfg: Value,
    child: Mutex<Option<Child>>,
    stdin: Mutex<Option<ChildStdin>>,
    pending: Arc<Mutex<HashMap<i64, crossbeam_channel::Sender<Answer>>>>,
    next_id: Mutex<i64>,
    stderr: Arc<Mutex<VecDeque<String>>>,
    ended: Arc<std::sync::atomic::AtomicBool>,
}

struct Http {
    url: String,
    headers: Vec<(String, String)>,
    session: Mutex<Option<String>>,
    protocol: Mutex<Option<String>>,
    next_id: Mutex<i64>,
    broken: Mutex<Option<String>>,
}

enum Transport {
    Stdio(Arc<Stdio_>),
    Http(Arc<Http>),
}

impl Stdio_ {
    fn tail(&self) -> String {
        self.stderr.lock().unwrap().back().map(|l| format!(": {}", py::chars_prefix(l, 300))).unwrap_or_default()
    }

    fn start(self: &Arc<Self>) -> Result<(), McpErr> {
        let mut env: HashMap<String, String> = std::env::vars().collect();
        if let Some(e) = get(&self.cfg, "env").as_object() {
            for (k, v) in e {
                env.insert(k.clone(), py::py_str(v));
            }
        }
        let command = py::py_str(get(&self.cfg, "command"));
        let path = env.get("PATH").cloned().unwrap_or_default();
        let exe = which(&command, &path).unwrap_or_else(|| command.clone());
        let args: Vec<String> = get(&self.cfg, "args").as_array().map(|a| a.iter().map(py::py_str).collect()).unwrap_or_default();
        let mut cmd = Command::new(&exe);
        cmd.args(&args).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).env_clear().envs(&env);
        if let Some(cwd) = get(&self.cfg, "cwd").as_str().filter(|c| !c.is_empty()) {
            cmd.current_dir(cwd);
        }
        use std::os::unix::process::CommandExt;
        cmd.process_group(0); // its own process group, so close() ends its children too
        let mut child = cmd.spawn().map_err(|e| McpErr::Err(format!("could not start {}: {e}", py::py_repr(&json!(command)))))?;
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();
        *self.stdin.lock().unwrap() = child.stdin.take();
        *self.child.lock().unwrap() = Some(child);
        let (buf, name) = (self.stderr.clone(), self.name.clone());
        let err_reader = std::thread::spawn(move || {
            for line in BufReader::new(stderr).split(b'\n').map_while(Result::ok) {
                let line = String::from_utf8_lossy(&line).trim_end().to_string();
                if !line.is_empty() {
                    if std::env::var("STRATA_DEBUG").map(|v| !v.is_empty()).unwrap_or(false) {
                        println!("[mcp {name}] {line}");
                    }
                    let mut b = buf.lock().unwrap();
                    if b.len() == 40 {
                        b.pop_front();
                    }
                    b.push_back(line);
                }
            }
        });
        let me = self.clone();
        std::thread::spawn(move || {
            for raw in BufReader::new(stdout).split(b'\n').map_while(Result::ok) {
                let line = String::from_utf8_lossy(&raw).trim().to_string();
                if line.is_empty() {
                    continue;
                }
                let Ok(msg) = serde_json::from_str::<Value>(&line) else {
                    me.stderr.lock().unwrap().push_back(line);
                    continue;
                };
                let items = if let Value::Array(a) = msg { a } else { vec![msg] };
                for m in items.into_iter().filter(Value::is_object) {
                    me.dispatch(m);
                }
            }
            me.ended.store(true, std::sync::atomic::Ordering::SeqCst);
            let t = Instant::now();
            while !err_reader.is_finished() && t.elapsed() < Duration::from_secs(1) {
                std::thread::sleep(Duration::from_millis(10));
            }
            let code = me.child.lock().unwrap().as_mut().and_then(|c| c.try_wait().ok().flatten()).and_then(|s| s.code());
            let err = format!("the server stopped{}{}", code.map(|c| format!(" (exit code {c})")).unwrap_or_default(), me.tail());
            let slots: Vec<_> = me.pending.lock().unwrap().drain().map(|(_, s)| s).collect();
            for s in slots {
                let _ = s.send(Err(McpErr::Err(err.clone())));
            }
        });
        Ok(())
    }

    fn alive(&self) -> bool {
        !self.ended.load(std::sync::atomic::Ordering::SeqCst) && self.child.lock().unwrap().as_mut().map(|c| matches!(c.try_wait(), Ok(None))).unwrap_or(false)
    }

    fn dispatch(&self, m: Value) {
        if m.get("method").is_some() {
            if let Some(id) = m.get("id") {
                let r = if m["method"] == "ping" {
                    json!({"jsonrpc": "2.0", "id": id, "result": {}})
                } else {
                    json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": format!("method not supported: {}", py::py_str(&m["method"]))}})
                };
                let _ = self.send(&r);
            }
            return;
        }
        let Some(id) = m.get("id").and_then(Value::as_i64) else { return };
        if let Some(s) = self.pending.lock().unwrap().remove(&id) {
            let _ = s.send(Ok(m));
        }
    }

    fn send(&self, msg: &Value) -> Result<(), McpErr> {
        let data = format!("{}\n", pyjson::dumps_u(msg));
        let mut s = self.stdin.lock().unwrap();
        let ok = s.as_mut().map(|w| w.write_all(data.as_bytes()).and_then(|_| w.flush()).is_ok()).unwrap_or(false);
        if ok { Ok(()) } else { Err(McpErr::Err(format!("the server stopped{}", self.tail()))) }
    }

    fn request(&self, method: &str, params: Value, timeout: f64, cancel: Option<&Event>) -> Result<Value, McpErr> {
        if !self.alive() {
            return Err(McpErr::Err(format!("the server is not running{}", self.tail())));
        }
        let (tx, rx) = crossbeam_channel::bounded(1);
        let rid = {
            let mut n = self.next_id.lock().unwrap();
            *n += 1;
            *n
        };
        self.pending.lock().unwrap().insert(rid, tx);
        let r = self.send(&json!({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})).and_then(|_| wait(&rx, timeout, cancel, method));
        if let Err(e @ (McpErr::Timeout(_) | McpErr::Cancelled)) = &r {
            self.pending.lock().unwrap().remove(&rid);
            let _ = self.notify("notifications/cancelled", Some(json!({"requestId": rid, "reason": e.to_string()})));
        }
        r
    }

    fn notify(&self, method: &str, params: Option<Value>) -> Result<(), McpErr> {
        let mut m = json!({"jsonrpc": "2.0", "method": method});
        if let Some(p) = params {
            m["params"] = p;
        }
        self.send(&m)
    }

    /// As the spec says: close its stdin, give it a moment to exit, then end it (and what it started).
    fn close(&self) {
        *self.stdin.lock().unwrap() = None;
        if let Some(c) = self.child.lock().unwrap().as_mut() {
            if !crate::engine::wait_timeout(c, 2.0) {
                let pg = c.id() as i32;
                unsafe { libc::killpg(pg, libc::SIGTERM) };
                if !crate::engine::wait_timeout(c, 2.0) {
                    unsafe { libc::killpg(pg, libc::SIGKILL) };
                }
                let _ = c.kill();
            }
        }
    }
}

fn which(cmd: &str, path: &str) -> Option<String> {
    if cmd.contains('/') {
        return None;
    }
    path.split(':').map(|d| std::path::Path::new(d).join(cmd)).find(|p| p.is_file()).map(|p| p.to_string_lossy().into_owned())
}

impl Http {
    fn headers(&self) -> Vec<(String, String)> {
        let mut h = vec![("Content-Type".to_string(), "application/json".to_string()), ("Accept".into(), "application/json, text/event-stream".into()), ("User-Agent".into(), "strata".into())];
        for (k, v) in &self.headers {
            h.retain(|(x, _)| !x.eq_ignore_ascii_case(k));
            h.push((k.clone(), v.clone()));
        }
        if let Some(s) = self.session.lock().unwrap().clone() {
            h.push(("Mcp-Session-Id".into(), s));
        }
        if let Some(p) = self.protocol.lock().unwrap().clone() {
            h.push(("MCP-Protocol-Version".into(), p));
        }
        h
    }

    /// One POST; for a request (rid Some) the message that answers it.
    fn post(&self, msg: &Value, rid: Option<i64>, timeout: f64) -> Result<Option<Value>, McpErr> {
        let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs_f64(timeout))).http_status_as_error(false).build().into();
        let mut req = agent.post(&self.url);
        for (k, v) in self.headers() {
            req = req.header(&k, &v);
        }
        let r = req.send(pyjson::dumps_u(msg).as_bytes()).map_err(|e| McpErr::Err(format!("could not reach {}: {e}", self.url)))?;
        if let Some(s) = r.headers().get("Mcp-Session-Id").and_then(|v| v.to_str().ok()) {
            *self.session.lock().unwrap() = Some(s.to_string());
        }
        let status = r.status();
        if status.as_u16() >= 400 {
            let reason = status.canonical_reason().unwrap_or("");
            let mut detail = String::new();
            let _ = r.into_body().into_reader().take(4096).read_to_string(&mut detail);
            let detail: String = detail.chars().take(300).collect();
            if status.as_u16() == 404 && self.session.lock().unwrap().is_some() {
                *self.broken.lock().unwrap() = Some("the server ended the session".into());
            }
            return Err(McpErr::Err(format!("HTTP {} {reason}{}", status.as_u16(), if detail.is_empty() { String::new() } else { format!(": {detail}") })));
        }
        let Some(rid) = rid else { return Ok(None) };
        let ctype = r.headers().get("Content-Type").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
        let matches = |x: &Value| x.is_object() && x.get("id").and_then(Value::as_i64) == Some(rid) && x.get("method").is_none();
        let found = if ctype.contains("text/event-stream") {
            let mut data: Vec<String> = vec![];
            let mut found = None;
            for line in BufReader::new(r.into_body().into_reader()).split(b'\n').map_while(Result::ok) {
                let line = String::from_utf8_lossy(&line).trim_end_matches(['\r', '\n']).to_string();
                if let Some(d) = line.strip_prefix("data:") {
                    data.push(d.trim_start_matches(' ').to_string());
                } else if line.is_empty() && !data.is_empty() {
                    let m: Value = serde_json::from_str(&data.join("\n")).unwrap_or(Value::Null);
                    data.clear();
                    let items = if let Value::Array(a) = m { a } else { vec![m] };
                    if let Some(x) = items.into_iter().find(|x| matches(x)) {
                        found = Some(x);
                        break;
                    }
                }
            }
            found
        } else {
            let mut body = String::new();
            r.into_body().into_reader().read_to_string(&mut body).map_err(|e| McpErr::Err(format!("could not reach {}: {e}", self.url)))?;
            let parsed: Value = if py::strip(&body).is_empty() { Value::Null } else { serde_json::from_str(&body).map_err(|e| McpErr::Err(format!("could not reach {}: {e}", self.url)))? };
            let items = if let Value::Array(a) = parsed { a } else { vec![parsed] };
            items.into_iter().filter(|x| matches(x)).last()
        };
        found.map(Some).ok_or_else(|| McpErr::Err("the server's answer did not contain a response".into()))
    }

    fn request(self: &Arc<Self>, method: &str, params: Value, timeout: f64, cancel: Option<&Event>) -> Result<Value, McpErr> {
        if let Some(b) = self.broken.lock().unwrap().clone() {
            return Err(McpErr::Err(b));
        }
        let rid = {
            let mut n = self.next_id.lock().unwrap();
            *n += 1;
            *n
        };
        let (tx, rx) = crossbeam_channel::bounded(1);
        let me = self.clone();
        let msg = json!({"jsonrpc": "2.0", "id": rid, "method": method, "params": params});
        std::thread::spawn(move || {
            let r = me.post(&msg, Some(rid), timeout + 5.0).map(|m| m.unwrap_or(Value::Null));
            let _ = tx.send(r);
        });
        let r = wait(&rx, timeout, cancel, method);
        if let Err(e @ (McpErr::Timeout(_) | McpErr::Cancelled)) = &r {
            let me = self.clone();
            let reason = e.to_string();
            std::thread::spawn(move || {
                let _ = me.notify("notifications/cancelled", Some(json!({"requestId": rid, "reason": reason})));
            });
        }
        r
    }

    fn notify(&self, method: &str, params: Option<Value>) -> Result<(), McpErr> {
        let mut m = json!({"jsonrpc": "2.0", "method": method});
        if let Some(p) = params {
            m["params"] = p;
        }
        self.post(&m, None, 10.0).map(|_| ())
    }

    fn close(&self) {
        if self.session.lock().unwrap().is_none() {
            return;
        }
        let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(3))).build().into();
        let mut req = agent.delete(&self.url);
        for (k, v) in self.headers() {
            req = req.header(&k, &v);
        }
        let _ = req.call();
    }
}

impl Transport {
    fn alive(&self) -> bool {
        match self {
            Transport::Stdio(s) => s.alive(),
            Transport::Http(h) => h.broken.lock().unwrap().is_none(),
        }
    }
    fn request(&self, method: &str, params: Value, timeout: f64, cancel: Option<&Event>) -> Result<Value, McpErr> {
        match self {
            Transport::Stdio(s) => s.request(method, params, timeout, cancel),
            Transport::Http(h) => h.request(method, params, timeout, cancel),
        }
    }
    fn notify(&self, method: &str) -> Result<(), McpErr> {
        match self {
            Transport::Stdio(s) => s.notify(method, None),
            Transport::Http(h) => h.notify(method, None),
        }
    }
    fn set_protocol(&self, p: String) {
        if let Transport::Http(h) = self {
            *h.protocol.lock().unwrap() = Some(p);
        }
    }
    fn close(&self) {
        match self {
            Transport::Stdio(s) => s.close(),
            Transport::Http(h) => h.close(),
        }
    }
}

// ------------------------------------------------------------------------------------------------ one server
pub struct ServerState {
    pub status: String,
    pub error: Option<String>,
    pub tools: Vec<Value>,
    pub info: Map<String, Value>,
    transport: Option<Arc<Transport>>,
    last_start: Instant,
}

pub struct McpServer {
    pub name: String,
    pub cfg: Value,
    pub kind: &'static str,
    pub editable: bool,
    settings: Map<String, Value>,
    pub st: Mutex<ServerState>,
}

impl McpServer {
    fn new(name: &str, cfg: Value, settings: Map<String, Value>, editable: bool) -> Self {
        let kind = if py::truthy(get(&cfg, "url")) { "http" } else { "stdio" };
        McpServer {
            name: name.into(),
            editable: editable && kind == "http",
            cfg,
            kind,
            settings,
            st: Mutex::new(ServerState { status: "idle".into(), error: None, tools: vec![], info: Map::new(), transport: None, last_start: Instant::now() }),
        }
    }

    pub fn start(&self) -> bool {
        let mut st = self.st.lock().unwrap();
        self.start_locked(&mut st)
    }

    fn start_locked(&self, st: &mut ServerState) -> bool {
        if let Some(t) = st.transport.take() {
            t.close();
        }
        st.status = "starting".into();
        st.error = None;
        st.last_start = Instant::now();
        let t = Arc::new(if self.kind == "http" {
            Transport::Http(Arc::new(Http {
                url: py::py_str(get(&self.cfg, "url")),
                headers: get(&self.cfg, "headers").as_object().map(|o| o.iter().map(|(k, v)| (k.clone(), py::py_str(v))).collect()).unwrap_or_default(),
                session: Mutex::new(None),
                protocol: Mutex::new(None),
                next_id: Mutex::new(0),
                broken: Mutex::new(None),
            }))
        } else {
            Transport::Stdio(Arc::new(Stdio_ {
                name: self.name.clone(),
                cfg: self.cfg.clone(),
                child: Mutex::new(None),
                stdin: Mutex::new(None),
                pending: Arc::new(Mutex::new(HashMap::new())),
                next_id: Mutex::new(0),
                stderr: Arc::new(Mutex::new(VecDeque::new())),
                ended: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            }))
        });
        st.transport = Some(t.clone());
        let timeout = self.settings["start_timeout_s"].as_f64().unwrap_or(120.0);
        let r = (|| -> Result<Vec<Value>, McpErr> {
            if let Transport::Stdio(s) = &*t {
                s.start()?;
            }
            let res = t.request("initialize", json!({"protocolVersion": PROTOCOL, "capabilities": {}, "clientInfo": {"name": "strata", "title": "Strata", "version": "1"}}), timeout, None)?;
            let proto = if py::truthy(get(&res, "protocolVersion")) { py::py_str(get(&res, "protocolVersion")) } else { PROTOCOL.into() };
            t.set_protocol(proto.clone());
            let mut info: Map<String, Value> = get(&res, "serverInfo").as_object().map(|o| o.iter().filter(|(k, _)| ["name", "title", "version"].contains(&k.as_str())).map(|(k, v)| (k.clone(), v.clone())).collect()).unwrap_or_default();
            info.insert("protocol".into(), json!(proto));
            st.info = info;
            t.notify("notifications/initialized")?;
            let mut tools: Vec<Value> = vec![];
            let mut cursor: Option<Value> = None;
            for _ in 0..100 {
                let params = match &cursor {
                    Some(c) => json!({"cursor": c}),
                    None => json!({}),
                };
                let page = match t.request("tools/list", params, timeout, None) {
                    Ok(p) => p,
                    Err(e) if e.to_string().contains("code -32601") && tools.is_empty() => break,
                    Err(e) => return Err(e),
                };
                tools.extend(get(&page, "tools").as_array().into_iter().flatten().filter(|x| x.is_object() && py::truthy(get(x, "name"))).cloned());
                let next = get(&page, "nextCursor");
                if !py::truthy(next) {
                    break;
                }
                cursor = Some(next.clone());
            }
            if let Some(allow) = get(&self.cfg, "tools").as_array() {
                tools.retain(|x| allow.contains(get(x, "name")));
            }
            Ok(tools)
        })();
        match r {
            Ok(tools) => {
                let names: Vec<String> = tools.iter().take(8).map(|x| py::py_str(get(x, "name"))).collect();
                let more = if tools.len() > 8 { ", ..." } else { "" };
                say(&format!("MCP server {}: {} tool{}{}", py::py_repr(&json!(self.name)), tools.len(), if tools.len() != 1 { "s" } else { "" }, if tools.is_empty() { String::new() } else { format!(" ({}{more})", names.join(", ")) }));
                st.tools = tools;
                st.status = "ready".into();
                true
            }
            Err(e) => {
                st.status = "failed".into();
                st.tools.clear();
                let mut err = e.to_string();
                let url = py::py_str(get(&self.cfg, "url"));
                let base = url.split('?').next().unwrap_or("").to_string();
                if self.kind == "http" && err.starts_with("HTTP 404") && Regex::new(r"^https?://[^/?#]+/?$").unwrap().is_match(&base) {
                    err += &format!(" - the URL has no path; most MCP servers are at {}/mcp", base.trim_end_matches('/'));
                }
                st.error = Some(err);
                t.close();
                say(&format!("MCP server {} did not start: {e}. Its tools are left out; the chat works without them.", py::py_repr(&json!(self.name))));
                false
            }
        }
    }

    pub fn call(&self, tool: &str, arguments: &Value, timeout: f64, cancel: Option<&Event>) -> Result<Value, McpErr> {
        let t = {
            let mut st = self.st.lock().unwrap();
            let up = st.transport.as_ref().map(|t| t.alive()).unwrap_or(false);
            if !up {
                if st.status == "failed" && st.last_start.elapsed() < Duration::from_secs(10) {
                    return Err(McpErr::Err(format!("the server is not running ({})", st.error.clone().unwrap_or_else(|| "it stopped".into()))));
                }
                say(&format!("MCP server {} had stopped; starting it again", py::py_repr(&json!(self.name))));
                if !self.start_locked(&mut st) {
                    return Err(McpErr::Err(format!("the server could not be started again: {}", st.error.clone().unwrap_or_default())));
                }
            }
            st.transport.clone().unwrap()
        };
        let r = t.request("tools/call", json!({"name": tool, "arguments": arguments}), timeout, cancel);
        if let Err(e) = &r {
            if !t.alive() {
                let mut st = self.st.lock().unwrap();
                st.status = "stopped".into();
                st.error = Some(e.to_string());
            }
        }
        r
    }

    pub fn close(&self) {
        let mut st = self.st.lock().unwrap();
        if let Some(t) = &st.transport {
            t.close();
        }
        if st.status == "ready" || st.status == "starting" {
            st.status = "stopped".into();
        }
    }
}

/// An MCP tool result as the text the model reads.
pub fn result_text(result: &Value) -> String {
    let mut parts = vec![];
    for block in get(result, "content").as_array().into_iter().flatten() {
        if !block.is_object() {
            continue;
        }
        match get(block, "type").as_str() {
            Some("text") => parts.push(block.get("text").map(py::py_str).unwrap_or_default()),
            Some("resource") => {
                let res = get(block, "resource");
                if res.get("text").is_some() {
                    parts.push(py::py_str(&res["text"]));
                } else {
                    let uri = res.get("uri").map(py::py_str).unwrap_or_default();
                    let mime = res.get("mimeType").map(py::py_str).unwrap_or_else(|| "binary".into());
                    parts.push(format!("[resource {uri} ({mime}), not shown]"));
                }
            }
            Some("resource_link") => {
                let name = if py::truthy(get(block, "name")) { py::py_str(get(block, "name")) } else { String::new() };
                parts.push(format!("[resource link: {name} {}]", block.get("uri").map(py::py_str).unwrap_or_default()).replace("  ", " "));
            }
            Some(k @ ("image" | "audio")) => {
                let size = get(block, "data").as_str().map(|d| d.chars().count()).unwrap_or(0) * 3 / 4;
                parts.push(format!("[{k} ({}, {} bytes): not shown to the model]", block.get("mimeType").map(py::py_str).unwrap_or_else(|| "?".into()), thousands(size as i64)));
            }
            _ => parts.push(pyjson::dumps_u(block)),
        }
    }
    if parts.is_empty() && !get(result, "structuredContent").is_null() {
        parts.push(pyjson::dumps_u(&result["structuredContent"]));
    }
    parts.join("\n")
}

/// Python's `f"{n:,}"`.
pub fn thousands(n: i64) -> String {
    let s = n.abs().to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    if n < 0 { format!("-{out}") } else { out }
}

fn clean(name: &str) -> String {
    let s: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect();
    if s.is_empty() { "x".into() } else { s }
}

fn defaults() -> Map<String, Value> {
    json!({"timeout_s": 60.0, "max_result_chars": 20000, "max_rounds": 8, "start_timeout_s": 120.0}).as_object().cloned().unwrap()
}

/// Every configured server, and the merged, namespaced tool list the model sees.
pub struct McpHub {
    pub settings: Map<String, Value>,
    pub servers: Mutex<indexmap::IndexMap<String, Arc<McpServer>>>,
    threads: Mutex<Vec<std::thread::JoinHandle<()>>>,
    routes_cache: Mutex<indexmap::IndexMap<String, (Arc<McpServer>, String)>>,
}

impl McpHub {
    pub fn new(servers: indexmap::IndexMap<String, Value>, settings: Map<String, Value>, editable: &[String]) -> Self {
        let mut s = defaults();
        for (k, v) in settings {
            s.insert(k, v);
        }
        let servers = servers.into_iter().map(|(n, c)| (n.clone(), Arc::new(McpServer::new(&n, c, s.clone(), editable.contains(&n))))).collect();
        McpHub { settings: s, servers: Mutex::new(servers), threads: Mutex::new(vec![]), routes_cache: Mutex::new(Default::default()) }
    }

    fn spawn_start(&self, s: Arc<McpServer>) {
        let h = std::thread::Builder::new().name(format!("mcp-{}", s.name)).spawn(move || {
            s.start();
        });
        if let Ok(h) = h {
            self.threads.lock().unwrap().push(h);
        }
    }

    /// Add a URL server, or replace one the page may change, and start it in the background.
    pub fn put(&self, name: &str, cfg: Value) {
        let server = Arc::new(McpServer::new(name, cfg, self.settings.clone(), true));
        let old = self.servers.lock().unwrap().insert(name.to_string(), server.clone());
        if let Some(o) = old {
            std::thread::spawn(move || o.close());
        }
        self.spawn_start(server);
    }

    pub fn remove(&self, name: &str) -> bool {
        let old = self.servers.lock().unwrap().shift_remove(name);
        match old {
            None => false,
            Some(o) => {
                std::thread::spawn(move || o.close());
                true
            }
        }
    }

    pub fn start(&self) {
        let all: Vec<_> = self.servers.lock().unwrap().values().cloned().collect();
        for s in all {
            self.spawn_start(s);
        }
    }

    /// Wait (up to `timeout` s in all) for servers that are still starting.
    pub fn wait(&self, timeout: Option<f64>) {
        let end = timeout.map(|t| Instant::now() + Duration::from_secs_f64(t));
        loop {
            let pending = self.threads.lock().unwrap().iter().any(|h| !h.is_finished());
            if !pending || end.map(|e| Instant::now() >= e).unwrap_or(false) {
                return;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// tool name the model sees -> (server, the tool's own name).
    pub fn routes(&self) -> indexmap::IndexMap<String, (Arc<McpServer>, String)> {
        let mut out: indexmap::IndexMap<String, (Arc<McpServer>, String)> = indexmap::IndexMap::new();
        let servers: Vec<_> = self.servers.lock().unwrap().values().cloned().collect();
        for s in servers {
            let st = s.st.lock().unwrap();
            if st.status != "ready" && st.status != "stopped" {
                continue;
            }
            for t in &st.tools {
                let tool = py::py_str(get(t, "name"));
                let full = format!("{}__{}", clean(&s.name), clean(&tool));
                let mut name: String = full.chars().take(64).collect();
                let mut n = 2;
                while out.contains_key(&name) {
                    name = format!("{}_{n}", name.chars().take(60).collect::<String>());
                    n += 1;
                }
                out.insert(name, (s.clone(), tool));
            }
        }
        *self.routes_cache.lock().unwrap() = out.clone();
        out
    }

    /// The tools in the chat template's form ({name, description, parameters}); `exclude`: the request's own names.
    pub fn template_tools(&self, exclude: &[String]) -> Vec<Value> {
        let mut out = vec![];
        for (name, (s, tool)) in self.routes() {
            if exclude.contains(&name) {
                continue;
            }
            let st = s.st.lock().unwrap();
            let Some(t) = st.tools.iter().find(|x| py::py_str(get(x, "name")) == tool) else { continue };
            let schema = get(t, "inputSchema").as_object().cloned().unwrap_or_default();
            let desc = if py::truthy(get(t, "description")) { py::py_str(get(t, "description")) } else if py::truthy(get(t, "title")) { py::py_str(get(t, "title")) } else { String::new() };
            out.push(json!({"name": name, "description": desc, "parameters": if schema.is_empty() { json!({"type": "object", "properties": {}}) } else { Value::Object(schema) }}));
        }
        out
    }

    /// Run one tool -> {"server", "tool", "ok", "text", "chars", "truncated", "ms"}; only a cancel is an Err.
    pub fn call(&self, name: &str, arguments: &Value, cancel: Option<&Event>) -> Result<Map<String, Value>, McpErr> {
        let t0 = Instant::now();
        let route = self.routes_cache.lock().unwrap().get(name).cloned().or_else(|| self.routes().get(name).cloned());
        let mut out = Map::new();
        out.insert("server".into(), route.as_ref().map(|(s, _)| json!(s.name)).unwrap_or(Value::Null));
        out.insert("tool".into(), json!(route.as_ref().map(|(_, t)| t.clone()).unwrap_or_else(|| name.to_string())));
        let (text, ok) = match &route {
            None => (format!("error: there is no tool named {}", py::py_repr(&json!(name))), false),
            Some((s, tool)) => {
                let args = if arguments.is_object() { arguments.clone() } else { json!({}) };
                match s.call(tool, &args, self.settings["timeout_s"].as_f64().unwrap_or(60.0), cancel) {
                    Ok(res) => {
                        let text = result_text(&res);
                        let ok = !py::truthy(get(&res, "isError"));
                        if ok { (text, true) } else { (format!("error: {}", if text.is_empty() { "the tool reported an error".into() } else { text }), false) }
                    }
                    Err(McpErr::Cancelled) => return Err(McpErr::Cancelled),
                    Err(e) => (format!("error: {e}"), false),
                }
            }
        };
        let cap = self.settings["max_result_chars"].as_i64().unwrap_or(20000);
        let full = text.chars().count() as i64;
        let truncated = cap > 0 && full > cap;
        let text = if truncated { format!("{}\n\n[... truncated: the tool returned {} characters; only the first {} are shown]", py::chars_prefix(&text, cap as usize), thousands(full), thousands(cap)) } else { text };
        out.insert("ok".into(), json!(ok));
        out.insert("text".into(), json!(text));
        out.insert("chars".into(), json!(full));
        out.insert("truncated".into(), json!(truncated));
        out.insert("ms".into(), json!(t0.elapsed().as_millis() as i64));
        Ok(out)
    }

    /// GET /mcp: every server, its state and its tools.
    pub fn status(&self) -> Value {
        let routes = self.routes();
        let servers: Vec<_> = self.servers.lock().unwrap().values().cloned().collect();
        let list: Vec<Value> = servers
            .iter()
            .map(|s| {
                let st = s.st.lock().unwrap();
                let names: HashMap<String, String> = routes.iter().filter(|(_, (srv, _))| Arc::ptr_eq(srv, s)).map(|(n, (_, t))| (t.clone(), n.clone())).collect();
                let mut m = json!({"name": s.name, "transport": s.kind, "status": st.status, "error": st.error, "info": st.info, "editable": s.editable}).as_object().cloned().unwrap();
                if s.kind == "http" {
                    let url = py::py_str(get(&s.cfg, "url"));
                    let mut headers: Vec<String> = get(&s.cfg, "headers").as_object().map(|o| o.keys().cloned().collect()).unwrap_or_default();
                    headers.sort();
                    m.insert("url".into(), json!(format!("{}{}", url.split('?').next().unwrap_or(""), if url.contains('?') { "?…" } else { "" })));
                    m.insert("headers".into(), json!(headers));
                }
                m.insert("tools".into(), Value::Array(st.tools.iter().map(|t| {
                    let tn = py::py_str(get(t, "name"));
                    json!({"name": names.get(&tn).cloned().unwrap_or(tn.clone()), "tool": tn, "description": py::chars_prefix(&if py::truthy(get(t, "description")) { py::py_str(get(t, "description")) } else { String::new() }, 300)})
                }).collect()));
                Value::Object(m)
            })
            .collect();
        let settings: Map<String, Value> = ["timeout_s", "max_result_chars", "max_rounds"].iter().map(|k| (k.to_string(), self.settings[*k].clone())).collect();
        json!({"servers": list, "tools": routes.len(), "settings": settings})
    }

    pub fn close(&self) {
        let servers: Vec<_> = self.servers.lock().unwrap().values().cloned().collect();
        for s in servers {
            s.close();
        }
    }
}

// ------------------------------------------------------------------------------------------------ config
fn check_server(name: &str, cfg: &Value, where_: &str) -> Result<Value, String> {
    let n = py::py_repr(&json!(name));
    if !cfg.is_object() {
        return Err(format!("[strata] {where_}: MCP server {n} must be an object with \"command\" or \"url\""));
    }
    if let Some(t) = cfg.get("tools") {
        if !t.as_array().map(|a| a.iter().all(Value::is_string)).unwrap_or(false) {
            return Err(format!("[strata] {where_}: MCP server {n}: \"tools\" must be a list of tool names"));
        }
    }
    if py::truthy(get(cfg, "url")) {
        if get(cfg, "type").as_str() == Some("sse") {
            return Err(format!("[strata] {where_}: MCP server {n} uses the old SSE transport (\"type\": \"sse\"); Strata speaks Streamable HTTP - most servers offer it at /mcp"));
        }
        if py::truthy(get(cfg, "headers")) && !get(cfg, "headers").is_object() {
            return Err(format!("[strata] {where_}: MCP server {n}: \"headers\" must be an object"));
        }
        return Ok(cfg.clone());
    }
    if !get(cfg, "command").as_str().map(|c| !py::strip(c).is_empty()).unwrap_or(false) {
        return Err(format!("[strata] {where_}: MCP server {n} needs \"command\" (a program to start) or \"url\""));
    }
    if py::truthy(get(cfg, "args")) && !get(cfg, "args").is_array() {
        return Err(format!("[strata] {where_}: MCP server {n}: \"args\" must be a list"));
    }
    if py::truthy(get(cfg, "env")) && !get(cfg, "env").is_object() {
        return Err(format!("[strata] {where_}: MCP server {n}: \"env\" must be an object"));
    }
    Ok(cfg.clone())
}

pub fn servers_from(block: &Value, where_: &str) -> Result<indexmap::IndexMap<String, Value>, String> {
    match block {
        Value::Null => Ok(Default::default()),
        Value::Object(o) => {
            let mut out = indexmap::IndexMap::new();
            for (n, c) in o {
                if c.is_object() && c.get("disabled") == Some(&Value::Bool(true)) {
                    continue;
                }
                out.insert(n.clone(), check_server(n, c, where_)?);
            }
            Ok(out)
        }
        _ => Err(format!("[strata] {where_}: the MCP servers must be an object {{\"name\": {{...}}}}")),
    }
}

/// The run config's optional `"mcp"` block.
pub fn settings_from(cfg: &Value) -> Result<Map<String, Value>, String> {
    let mut out = Map::new();
    for (key, value) in get(cfg, "mcp").as_object().into_iter().flatten() {
        let number = value.as_f64().filter(|_| value.is_number());
        let r = py::py_repr(value);
        match key.as_str() {
            "timeout_s" | "start_timeout_s" => match number.filter(|n| *n > 0.0) {
                Some(n) => {
                    out.insert(key.clone(), json!(n));
                }
                None => return Err(format!("[strata] config mcp.{key}={r}: expected a number of seconds > 0")),
            },
            "max_result_chars" | "max_rounds" => match number.filter(|n| *n == n.trunc() && *n >= 1.0) {
                Some(n) => {
                    out.insert(key.clone(), json!(n as i64));
                }
                None => return Err(format!("[strata] config mcp.{key}={r}: expected a positive integer")),
            },
            _ => say(&format!("config mcp.{key}={r}: unknown key, ignored")),
        }
    }
    Ok(out)
}

const RESERVED_HEADERS: [&str; 10] = ["accept", "content-type", "content-length", "host", "connection", "transfer-encoding", "mcp-session-id", "mcp-protocol-version", "user-agent", "cookie"];

/// POST /mcp/servers from the Settings view -> (name, the "mcp_servers" entry); Err with what is wrong.
pub fn page_entry(req: &Value, old: Option<&Value>) -> Result<(String, Value), String> {
    if !req.is_object() {
        return Err("send a JSON object".into());
    }
    let name = get(req, "name").as_str().filter(|n| Regex::new(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,39}$").unwrap().is_match(n)).ok_or("name: 1-40 letters, digits, '_', '.', '-' (starting with a letter or digit)")?;
    let url = get(req, "url").as_str().filter(|u| u.chars().count() <= 2000 && Regex::new(r"^https?://[^\s/?#]+[^\s]*$").unwrap().is_match(py::strip(u))).ok_or("url: an http:// or https:// address (Streamable HTTP; most servers offer it at /mcp)")?;
    let header = match req.get("header") {
        None => "Authorization".to_string(),
        Some(h) => h.as_str().map(String::from).unwrap_or_default(),
    };
    if !Regex::new(r"^[A-Za-z0-9-]{1,64}$").unwrap().is_match(&header) || RESERVED_HEADERS.contains(&header.to_lowercase().as_str()) {
        return Err("header: a header name such as Authorization or X-API-Key".into());
    }
    let key = req.get("key").filter(|k| !k.is_null());
    if let Some(k) = key {
        if !k.as_str().map(|s| s.chars().count() <= 4096 && !s.contains(['\r', '\n', '\0'])).unwrap_or(false) {
            return Err("key: one line of text, at most 4096 characters".into());
        }
    }
    let mut entry = json!({"url": py::strip(url)});
    let auth = header.eq_ignore_ascii_case("authorization");
    match key {
        None => {
            if let Some((old_name, value)) = old.and_then(|o| get(o, "headers").as_object()).and_then(|h| h.iter().next()) {
                let mut value = py::py_str(value);
                let old_auth = old_name.eq_ignore_ascii_case("authorization");
                if old_auth && !auth {
                    value = value.split_once(' ').map(|x| x.1.to_string()).unwrap_or(value);
                } else if auth && !old_auth && !value.contains(' ') {
                    value = format!("Bearer {value}");
                }
                entry["headers"] = json!({header.clone(): value});
            }
        }
        Some(k) => {
            let k = py::strip(k.as_str().unwrap()).to_string();
            if !k.is_empty() {
                if (py::split_ws(&k).len() > 2 || !auth) && k.chars().any(py::is_space) {
                    return Err("key: paste only the key itself (it has spaces; for Authorization \"<scheme> <token>\" is fine)".into());
                }
                let value = if !auth || k.contains(' ') { k } else { format!("Bearer {k}") };
                entry["headers"] = json!({header.clone(): value});
            }
        }
    }
    Ok((name.to_string(), entry))
}

/// The run config's `"mcp_servers"` (or `"mcpServers"`) plus the --mcp-config file; None when there are none.
pub fn hub_from_config(cfg: &Value, mcp_config_path: Option<&str>) -> Result<Option<McpHub>, String> {
    let mut servers = servers_from(get(cfg, "mcp_servers"), "config mcp_servers")?;
    servers.extend(servers_from(get(cfg, "mcpServers"), "config mcpServers")?);
    if let Some(p) = mcp_config_path {
        let text = std::fs::read_to_string(p).map_err(|e| format!("[strata] --mcp-config {p}: {e}"))?;
        let data: Value = serde_json::from_str(text.trim_start_matches('\u{feff}')).map_err(|e| format!("[strata] --mcp-config {p}: {e}"))?;
        let block = data.get("mcpServers").or_else(|| data.get("mcp_servers")).cloned();
        let Some(block) = block else { return Err(format!("[strata] --mcp-config {p}: expected {{\"mcpServers\": {{...}}}}")) };
        servers.extend(servers_from(&block, &format!("--mcp-config {p}"))?);
    }
    if servers.is_empty() {
        return Ok(None);
    }
    let own = servers_from(get(cfg, "mcp_servers"), "config mcp_servers")?;
    let editable: Vec<String> = own.iter().filter(|(n, c)| py::truthy(get(c, "url")) && servers.get(*n) == Some(c)).map(|(n, _)| n.clone()).collect();
    Ok(Some(McpHub::new(servers, settings_from(cfg)?, &editable)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn result_text_and_page_entries() {
        let r = json!({"content": [{"type": "text", "text": "a"}, {"type": "image", "mimeType": "image/png", "data": "AAAA"}, {"type": "resource", "resource": {"uri": "x://y"}}]});
        assert_eq!(result_text(&r), "a\n[image (image/png, 3 bytes): not shown to the model]\n[resource x://y (binary), not shown]");
        assert_eq!(thousands(1234567), "1,234,567");
        let (n, e) = page_entry(&json!({"name": "bible", "url": "http://localhost:3460/mcp", "key": "k1"}), None).unwrap();
        assert_eq!((n.as_str(), e), ("bible", json!({"url": "http://localhost:3460/mcp", "headers": {"Authorization": "Bearer k1"}})));
        let (_, e) = page_entry(&json!({"name": "bible", "url": "http://x/mcp", "header": "X-API-Key"}), Some(&json!({"headers": {"Authorization": "Bearer k1"}}))).unwrap();
        assert_eq!(e["headers"], json!({"X-API-Key": "k1"}));
    }
}
