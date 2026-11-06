//! Hub (the web page's tab beside About): this server dials an AI-cluster (bible's /hub/ws) and offers its models
//! there; the cluster sends it jobs, many at once on the one connection, and this server answers each as its own API
//! would.  Wire: one WebSocket (RFC 6455; ws:// or wss://), every message one binary protobuf Frame - hub.proto, the
//! same as bible's src/hub.rs (a_job_frame_has_the_bytes_both_sides_agree_on holds both to it).  A job is an HTTP
//! request that this server's own API answers on 127.0.0.1, so every feature works through the hub with no code here;
//! the Clients tab lists those requests as "strata-hub".  The key is the cluster's node key, sent as
//! `Authorization: Bearer` (RFC 6750 §2.1); it is kept in .strata-hub.json (mode 0600), never shown to the page again.

use crate::say;
use futures::{SinkExt, StreamExt};
use prost::Message as _;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message;

pub mod wire {
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct Frame {
        #[prost(oneof = "frame::Kind", tags = "1, 2, 3, 4, 5, 6")]
        pub kind: Option<frame::Kind>,
    }
    pub mod frame {
        #[derive(Clone, PartialEq, prost::Oneof)]
        pub enum Kind {
            #[prost(message, tag = "1")]
            Hello(super::Hello),
            #[prost(message, tag = "2")]
            Job(super::Job),
            #[prost(message, tag = "3")]
            Head(super::Head),
            #[prost(message, tag = "4")]
            Chunk(super::Chunk),
            #[prost(message, tag = "5")]
            End(super::End),
            #[prost(message, tag = "6")]
            Cancel(super::Cancel),
        }
    }
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct Hello {
        #[prost(string, tag = "1")]
        pub name: String,
        #[prost(string, tag = "2")]
        pub version: String,
        #[prost(message, repeated, tag = "3")]
        pub models: Vec<Model>,
        #[prost(uint32, tag = "4")]
        pub slots: u32,
    }
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct Model {
        #[prost(string, tag = "1")]
        pub id: String,
        #[prost(uint32, tag = "2")]
        pub context: u32,
    }
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct Job {
        #[prost(uint64, tag = "1")]
        pub id: u64,
        #[prost(string, tag = "2")]
        pub path: String,
        #[prost(bytes = "bytes", tag = "3")]
        pub body: prost::bytes::Bytes,
    }
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct Head {
        #[prost(uint64, tag = "1")]
        pub id: u64,
        #[prost(uint32, tag = "2")]
        pub status: u32,
        #[prost(string, tag = "3")]
        pub content_type: String,
    }
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct Chunk {
        #[prost(uint64, tag = "1")]
        pub id: u64,
        #[prost(bytes = "bytes", tag = "2")]
        pub data: prost::bytes::Bytes,
    }
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct End {
        #[prost(uint64, tag = "1")]
        pub id: u64,
        #[prost(string, tag = "2")]
        pub error: String,
    }
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct Cancel {
        #[prost(uint64, tag = "1")]
        pub id: u64,
    }
}

use wire::frame::Kind;

/// The paths a job may ask (the API's generation paths; nothing that changes this server).
const PATHS: [&str; 5] = ["/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/responses", "/v1/embeddings"];
/// bible pings every 20 s: nothing heard for this long is a dead connection.
const SILENT: Duration = Duration::from_secs(60);

#[derive(Default)]
struct State {
    state: &'static str,
    detail: String,
    since: i64,
    /// bumped by every change of the settings: the connection made under an older one closes
    generation: u64,
}

static STATE: LazyLock<Mutex<State>> = LazyLock::new(|| Mutex::new(State { state: "off", ..Default::default() }));
static ACTIVE: AtomicUsize = AtomicUsize::new(0);
static SERVED: AtomicU64 = AtomicU64::new(0);

fn path() -> PathBuf {
    crate::root().join(".strata-hub.json")
}

fn load() -> Value {
    std::fs::read_to_string(path()).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(json!({}))
}

fn set(state: &'static str, detail: &str) {
    let mut s = STATE.lock().unwrap();
    if s.state != state || s.detail != detail {
        s.state = state;
        s.detail = detail.to_string();
        s.since = crate::sync::now() as i64;
    }
}

/// GET /hub: the settings (the key only as its first characters) and the connection's state.
pub fn status() -> Value {
    let c = load();
    let key = c["key"].as_str().unwrap_or("");
    let s = STATE.lock().unwrap();
    json!({"url": c["url"], "name": c["name"], "enabled": c["enabled"].as_bool().unwrap_or(false),
           "key": if key.is_empty() { Value::Null } else { json!(format!("{}…", key.chars().take(12).collect::<String>())) },
           "state": s.state, "detail": s.detail, "since": s.since,
           "active": ACTIVE.load(Ordering::Relaxed), "served": SERVED.load(Ordering::Relaxed)})
}

/// POST /hub {url, key, name, enabled}: saved (a missing or empty key keeps the saved one); the connection follows.
pub fn put(req: &Value) -> Result<Value, String> {
    let mut c = load();
    if let Some(u) = req["url"].as_str() {
        let u = u.trim();
        if !(u.is_empty() || u.starts_with("ws://") || u.starts_with("wss://")) {
            return Err("the address starts with ws:// or wss:// (the AI-cluster page shows it)".into());
        }
        c["url"] = json!(u);
    }
    if let Some(k) = req["key"].as_str().map(str::trim).filter(|k| !k.is_empty()) {
        c["key"] = json!(k);
    }
    if let Some(n) = req["name"].as_str() {
        c["name"] = json!(n.trim().chars().take(100).collect::<String>());
    }
    if let Some(e) = req["enabled"].as_bool() {
        c["enabled"] = json!(e);
    }
    let tmp = path().with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(&c).unwrap_or_default()).map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600));
    }
    std::fs::rename(&tmp, path()).map_err(|e| e.to_string())?;
    STATE.lock().unwrap().generation += 1;
    Ok(status())
}

/// The hub connection, for as long as the server runs: (re)connects whenever it is on, with a growing wait after a
/// failure (1 s doubling to 30 s), at once after a change of the settings.
pub fn start(port: u16, key: Option<String>) {
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().expect("hub runtime");
        rt.block_on(async move {
            let mut wait = Duration::from_secs(1);
            loop {
                let c = load();
                let generation = STATE.lock().unwrap().generation;
                let (url, nkey) = (c["url"].as_str().unwrap_or("").to_string(), c["key"].as_str().unwrap_or("").to_string());
                if !c["enabled"].as_bool().unwrap_or(false) || url.is_empty() || nkey.is_empty() {
                    set("off", if url.is_empty() || nkey.is_empty() { "set the address and the key" } else { "" });
                } else {
                    set("connecting", &url);
                    let name = c["name"].as_str().filter(|n| !n.is_empty()).map(String::from).unwrap_or_else(host_name);
                    match session(&url, &nkey, &name, port, key.as_deref(), generation).await {
                        Ok(()) => wait = Duration::from_secs(1),
                        Err(e) => {
                            say(&format!("hub: {e}"));
                            // a connection that worked (bible redeploys: 1001) is dialled again at once, not after the last wait
                            if STATE.lock().unwrap().state == "connected" {
                                wait = Duration::from_millis(500);
                            }
                            set("error", &e);
                        }
                    }
                }
                // until the wait is over or the settings change
                let until = tokio::time::Instant::now() + wait;
                while tokio::time::Instant::now() < until && STATE.lock().unwrap().generation == generation {
                    tokio::time::sleep(Duration::from_millis(250)).await;
                }
                if STATE.lock().unwrap().generation == generation {
                    wait = (wait * 2).min(Duration::from_secs(30));
                } else {
                    wait = Duration::from_secs(1);
                }
            }
        });
    });
}

fn host_name() -> String {
    std::process::Command::new("hostname").output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).filter(|h| !h.is_empty()).unwrap_or_else(|| "strata".into())
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(3600))).http_status_as_error(false).build().into()
}

/// This server's models and slots, from its own /v1/models and /slots (what a client sees).
fn hello(name: &str, port: u16, key: Option<&str>) -> wire::Hello {
    let get = |p: &str| -> Value {
        let mut r = agent().get(&format!("http://127.0.0.1:{port}{p}")).header("User-Agent", "strata-hub");
        if let Some(k) = key {
            r = r.header("Authorization", &format!("Bearer {k}"));
        }
        r.call().ok().and_then(|r| serde_json::from_reader(r.into_body().into_reader()).ok()).unwrap_or(Value::Null)
    };
    let models = get("/v1/models")["data"].as_array().cloned().unwrap_or_default().iter()
        .filter_map(|m| Some(wire::Model { id: m["id"].as_str()?.to_string(), context: m.pointer("/meta/n_ctx").and_then(Value::as_u64).unwrap_or(0) as u32 }))
        .collect();
    let slots = get("/slots").as_array().map_or(1, |a| a.len().max(1)) as u32;
    wire::Hello { name: name.to_string(), version: env!("CARGO_PKG_VERSION").to_string(), models, slots }
}

fn bin(k: Kind) -> Message {
    Message::Binary(wire::Frame { kind: Some(k) }.encode_to_vec().into())
}

/// One connection, until it closes, falls silent or the settings change (Ok: reconnect at once).
async fn session(url: &str, nkey: &str, name: &str, port: u16, key: Option<&str>, generation: u64) -> Result<(), String> {
    let mut req = url.into_client_request().map_err(|e| format!("address: {e}"))?;
    let h = req.headers_mut();
    h.insert("Authorization", format!("Bearer {nkey}").parse().map_err(|_| "the key has characters a header cannot carry")?);
    h.insert("User-Agent", format!("strata/{}", env!("CARGO_PKG_VERSION")).parse().unwrap());
    let (ws, _) = tokio_tungstenite::connect_async(req).await.map_err(|e| match e {
        tokio_tungstenite::tungstenite::Error::Http(r) if r.status() == 401 => "the AI-cluster refused the key (unknown or revoked)".to_string(),
        e => format!("connect: {e}"),
    })?;
    let (mut sink, mut stream) = ws.split();
    let (k2, n2) = (key.map(String::from), name.to_string());
    let mut last = tokio::task::spawn_blocking(move || hello(&n2, port, k2.as_deref())).await.map_err(|e| e.to_string())?;
    sink.send(bin(Kind::Hello(last.clone()))).await.map_err(|e| e.to_string())?;
    set("connected", url);
    say(&format!("hub: connected to {url} as {name} ({} model(s))", last.models.len()));
    let (tx, mut rx) = mpsc::unbounded_channel::<Kind>();
    let jobs: Arc<Mutex<HashMap<u64, Arc<AtomicBool>>>> = Arc::default();
    let mut refresh = tokio::time::interval(Duration::from_secs(10));
    let mut tick = tokio::time::interval(Duration::from_millis(500));
    let mut heard = tokio::time::Instant::now();
    let out = loop {
        tokio::select! {
            m = stream.next() => {
                let m = match m { Some(Ok(m)) => m, Some(Err(e)) => break Err(e.to_string()), None => break Err("the AI-cluster closed the connection".into()) };
                heard = tokio::time::Instant::now();
                match m {
                    Message::Binary(b) => match wire::Frame::decode(b) {
                        Ok(wire::Frame { kind: Some(Kind::Job(j)) }) => {
                            let cancel = Arc::new(AtomicBool::new(false));
                            jobs.lock().unwrap().insert(j.id, cancel.clone());
                            let (tx, jobs, key) = (tx.clone(), jobs.clone(), key.map(String::from));
                            tokio::task::spawn_blocking(move || {
                                ACTIVE.fetch_add(1, Ordering::Relaxed);
                                let id = j.id;
                                let err = run_job(j, port, key.as_deref(), &cancel, &tx);
                                ACTIVE.fetch_sub(1, Ordering::Relaxed);
                                if err.is_empty() {
                                    SERVED.fetch_add(1, Ordering::Relaxed);
                                }
                                jobs.lock().unwrap().remove(&id);
                                let _ = tx.send(Kind::End(wire::End { id, error: err }));
                            });
                        }
                        Ok(wire::Frame { kind: Some(Kind::Cancel(c)) }) => {
                            if let Some(f) = jobs.lock().unwrap().get(&c.id) {
                                f.store(true, Ordering::Relaxed);
                            }
                        }
                        Ok(_) => {}
                        Err(e) => break Err(format!("a frame that is not hub.proto: {e}")),
                    },
                    Message::Close(f) => break Err(format!("the AI-cluster closed the connection{}", f.map(|f| format!(" ({} {})", u16::from(f.code), f.reason)).unwrap_or_default())),
                    _ => {}
                }
            }
            k = rx.recv() => {
                let Some(k) = k else { break Ok(()) };
                if let Err(e) = sink.send(bin(k)).await { break Err(e.to_string()) }
            }
            _ = refresh.tick() => {
                let (k2, n2) = (key.map(String::from), name.to_string());
                if let Ok(h) = tokio::task::spawn_blocking(move || hello(&n2, port, k2.as_deref())).await
                    && h != last
                {
                    last = h.clone();
                    if let Err(e) = sink.send(bin(Kind::Hello(h))).await { break Err(e.to_string()) }
                }
            }
            _ = tick.tick() => {
                if STATE.lock().unwrap().generation != generation {
                    let _ = sink.send(Message::Close(None)).await;
                    break Ok(());
                }
                if heard.elapsed() > SILENT { break Err("the AI-cluster fell silent".into()) }
            }
        }
    };
    // the jobs still running stop: their caller is on the far side of this connection
    for f in jobs.lock().unwrap().values() {
        f.store(true, Ordering::Relaxed);
    }
    out
}

/// One job against this server's own API: Head, then each piece of the body as it is read, until the end or a Cancel
/// (dropping the response closes the local connection, which cancels the generation). The error is "" when done.
fn run_job(j: wire::Job, port: u16, key: Option<&str>, cancel: &AtomicBool, tx: &mpsc::UnboundedSender<Kind>) -> String {
    if !PATHS.contains(&j.path.as_str()) {
        return format!("this node answers {}", PATHS.join(", "));
    }
    let mut r = agent().post(&format!("http://127.0.0.1:{port}{}", j.path)).header("Content-Type", "application/json").header("User-Agent", "strata-hub");
    if let Some(k) = key {
        r = r.header("Authorization", &format!("Bearer {k}"));
    }
    let resp = match r.send(&j.body[..]) {
        Ok(r) => r,
        Err(e) => return e.to_string(),
    };
    let ctype = resp.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("application/json").to_string();
    let _ = tx.send(Kind::Head(wire::Head { id: j.id, status: resp.status().as_u16() as u32, content_type: ctype }));
    let mut body = resp.into_body().into_reader();
    let mut buf = vec![0u8; 16 * 1024];
    loop {
        if cancel.load(Ordering::Relaxed) {
            return "cancelled".into();
        }
        match body.read(&mut buf) {
            Ok(0) => return String::new(),
            Ok(n) => {
                let _ = tx.send(Kind::Chunk(wire::Chunk { id: j.id, data: buf[..n].to_vec().into() }));
            }
            Err(e) => return e.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Live (cargo test --lib hub::tests::live_node -- --ignored): this test is a node for STRATA_HUB_SECS (60) s, the
    /// running server on STRATA_PORT (8090, key STRATA_KEY) its local API, the AI-cluster at STRATA_HUB_URL with
    /// STRATA_HUB_KEY.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn live_node() {
        let env = |k: &str| std::env::var(k).unwrap_or_default();
        let port = env("STRATA_PORT").parse().unwrap_or(8090);
        let key = Some(env("STRATA_KEY")).filter(|k| !k.is_empty());
        let secs = env("STRATA_HUB_SECS").parse().unwrap_or(60);
        let (url, nkey) = (env("STRATA_HUB_URL"), env("STRATA_HUB_KEY"));
        let run = session(&url, &nkey, "live-test", port, key.as_deref(), 0);
        let r = tokio::time::timeout(Duration::from_secs(secs), run).await;
        assert!(r.is_err(), "the session ended early: {r:?}");
    }

    /// The bytes of a Job frame: bible's src/hub.rs asserts the same hex (the two sides of hub.proto).
    #[test]
    fn a_job_frame_has_the_bytes_both_sides_agree_on() {
        let f = wire::Frame { kind: Some(Kind::Job(wire::Job { id: 7, path: "/v1/chat/completions".into(), body: prost::bytes::Bytes::from_static(b"{}") })) };
        assert_eq!(hex::encode(f.encode_to_vec()), "121c080712142f76312f636861742f636f6d706c6574696f6e731a027b7d");
    }

    /// A fake AI-cluster and a fake local API: the node says Hello with the local models, sends the bearer key, answers
    /// a job with the local status, content type and body, refuses a path that is not a generation path, and a
    /// change of the settings closes the connection.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_job_is_answered_by_the_local_api_over_the_hub() {
        use axum::routing::{get, post};
        let api = axum::Router::new()
            .route("/v1/models", get(|| async { axum::Json(json!({"data": [{"id": "qwen", "meta": {"n_ctx": 8192}}]})) }))
            .route("/slots", get(|| async { axum::Json(json!([{}, {}])) }))
            .route("/v1/chat/completions", post(|b: String| async move { ([("content-type", "text/event-stream")], format!("data: {b}\n\n")) }));
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = l.local_addr().unwrap().port();
        tokio::spawn(async move { axum::serve(l, api).await.unwrap() });

        let hub = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("ws://{}/hub/ws", hub.local_addr().unwrap());
        let generation = STATE.lock().unwrap().generation;
        tokio::spawn(async move { session(&url, "hub_test", "m1", port, None, generation).await });
        let (tcp, _) = hub.accept().await.unwrap();
        let auth = Arc::new(Mutex::new(String::new()));
        let a2 = auth.clone();
        let mut ws = tokio_tungstenite::accept_hdr_async(tcp, move |r: &tokio_tungstenite::tungstenite::handshake::server::Request, resp| {
            *a2.lock().unwrap() = r.headers().get("authorization").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
            Ok(resp)
        }).await.unwrap();
        assert_eq!(*auth.lock().unwrap(), "Bearer hub_test");
        let next = |m: Message| match m { Message::Binary(b) => wire::Frame::decode(b).unwrap().kind.unwrap(), m => panic!("{m:?}") };
        let Kind::Hello(h) = next(ws.next().await.unwrap().unwrap()) else { panic!("hello first") };
        assert_eq!((h.name.as_str(), h.slots, h.models[0].id.as_str(), h.models[0].context), ("m1", 2, "qwen", 8192));

        ws.send(bin(Kind::Job(wire::Job { id: 5, path: "/v1/chat/completions".into(), body: prost::bytes::Bytes::from_static(b"{\"x\":1}") }))).await.unwrap();
        ws.send(bin(Kind::Job(wire::Job { id: 6, path: "/hub".into(), body: Default::default() }))).await.unwrap();
        let (mut body, mut ends) = (Vec::new(), HashMap::new());
        while ends.len() < 2 {
            match next(ws.next().await.unwrap().unwrap()) {
                Kind::Head(h) => assert_eq!((h.id, h.status, h.content_type.as_str()), (5, 200, "text/event-stream")),
                Kind::Chunk(c) => body.extend_from_slice(&c.data),
                Kind::End(e) => { ends.insert(e.id, e.error); }
                k => panic!("{k:?}"),
            }
        }
        assert_eq!(String::from_utf8(body).unwrap(), "data: {\"x\":1}\n\n");
        assert_eq!(ends[&5], "");
        assert!(ends[&6].starts_with("this node answers"), "{}", ends[&6]);

        STATE.lock().unwrap().generation += 1;
        loop {
            match ws.next().await {
                Some(Ok(Message::Close(_))) | None => break,
                Some(Ok(_)) => continue,
                Some(Err(_)) => break,
            }
        }
    }
}
