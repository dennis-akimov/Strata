//! strata-tui: Strata.app in a terminal, for Linux on an Apple Silicon Mac (and any terminal, over SSH too).
//!
//! The same as the app: a bar with the state, the model's run config, its context, the address, the speed and what
//! the model needs against what this Mac has; under it Chat, Monitor, Memory and Log.  The server is this program's
//! child (engine-metal/strata-server on the config's port): quitting stops it the server's own way (SIGTERM: the
//! engine frees its memory first).
//!
//!   Ctrl+S start / stop   Ctrl+R restart   Ctrl+L load / unload   Ctrl+N next config   Ctrl+T thinking effort
//!   Ctrl+O new chat   Esc stop the answer   PgUp / PgDn scroll   Tab or F1-F5 view   ? keys   Ctrl+Q quit
//!   (outside Chat the same letters work without Ctrl; Memory: Up / Down and Enter pick the context; Models: Up / Down
//!   and Enter switch to a model, or download one that is not here with make pull)

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Cell, Clear, Paragraph, Row, Table, Tabs, Wrap};
use ratatui::Frame;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const GB: f64 = 1e9;
const CONTEXTS: [u64; 12] = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];
const TABS: [&str; 5] = ["Chat", "Monitor", "Memory", "Models", "Log"];
const EFFORTS: [(&str, &str); 4] = [("none", "off"), ("low", "low"), ("medium", "medium"), ("high", "high")];
/// The models `make pull` knows (tools/pull_model.py): (its name, the run config it writes, a title, the download in GB)
const CATALOG: [(&str, &str, &str, f64); 5] = [
    ("GPT-OSS-20B", "gpt-oss-20b", "GPT-OSS 20B", 12.1),
    ("QWEN3.6-35B-IQ2", "qwen3.6-35b-a3b-iq2", "Qwen3.6-35B-A3B 2-bit", 11.5),
    ("QWEN3.6-35B", "qwen3.6-35b-a3b", "Qwen3.6-35B-A3B", 21.7),
    ("GPT-OSS-120B", "gpt-oss-120b", "GPT-OSS 120B", 63.4),
    ("GLM-5.3", "glm-5.3-flash-maya-s24", "GLM-5.3-Flash", 94.7),
];

enum Msg {
    Log(String),
    Status(Value),
    Plan(Value),
    Delta { think: bool, text: String },
    ChatEnd(Option<String>),
    Apps(Vec<(String, f64)>),
    Pulled(String, bool),
}

/// Strata's folder: $STRATA_ROOT, else the folder above engine-metal/ (where make installs this), else the source's.
fn root() -> PathBuf {
    if let Ok(r) = std::env::var("STRATA_ROOT") {
        return r.into();
    }
    let exe = std::env::current_exe().ok().and_then(|e| e.parent().and_then(|p| p.parent()).map(Path::to_path_buf));
    match exe {
        Some(r) if r.join("engine-metal").is_dir() => r,
        _ => Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf(),
    }
}

fn read_json(p: &Path) -> Value {
    std::fs::read_to_string(p).ok().and_then(|s| serde_json::from_str(s.trim_start_matches('\u{feff}')).ok()).unwrap_or(json!({}))
}

/// The run configs setup wrote (strata-<model>.json with an "exe"), newest first.
fn configs(root: &Path) -> Vec<PathBuf> {
    let mut v: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(root)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("strata-") && n.ends_with(".json")))
        .filter(|p| read_json(p).get("exe").is_some())
        .map(|p| (std::fs::metadata(&p).and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH), p))
        .collect();
    v.sort_by(|a, b| b.0.cmp(&a.0));
    v.into_iter().map(|(_, p)| p).collect()
}

fn arg(cfg: &Value, flag: &str) -> Option<String> {
    let a = cfg["args"].as_array()?;
    let i = a.iter().position(|x| x == flag)?;
    a.get(i + 1)?.as_str().map(String::from)
}

fn api_key(cfg: &Value) -> String {
    match &cfg["api_key"] {
        Value::String(s) => s.split(',').next().unwrap_or("").trim().to_string(),
        Value::Array(a) => a.first().and_then(Value::as_str).unwrap_or("").to_string(),
        _ => String::new(),
    }
}

// ------------------------------------------------------------------ memory (the engine's --plan, as Strata.app does)

fn meminfo(key: &str) -> f64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with(&format!("{key}:"))).and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok()))
        .unwrap_or(0.0)
        * 1024.0
}

fn sysctl(k: &str) -> f64 {
    Command::new("sysctl").args(["-n", k]).output().ok().and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse().ok()).unwrap_or(0.0)
}

/// Memory free or reclaimable now: Linux MemAvailable; macOS vm_stat's free + inactive + speculative + purgeable.
fn free_now() -> f64 {
    if cfg!(target_os = "linux") {
        return meminfo("MemAvailable");
    }
    let out = Command::new("vm_stat").output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
    let page = out.split("page size of ").nth(1).and_then(|r| r.split_whitespace().next()?.parse::<f64>().ok()).unwrap_or(16384.0);
    let pages = |n: &str| out.lines().find(|l| l.starts_with(n)).and_then(|l| l.rsplit(':').next()?.trim().trim_end_matches('.').parse::<f64>().ok()).unwrap_or(0.0);
    (pages("Pages free") + pages("Pages inactive") + pages("Pages speculative") + pages("Pages purgeable")) * page
}

/// What the GPU may hold.  Linux (Vulkan): Honeykrisp's heap, HK_SYSMEM from the run config's "env" (setup sets it to
/// the memory less what Linux needs), else Vulkan's own (about half the memory).  macOS: iogpu.wired_limit_mb, else
/// about two thirds of the memory up to 36 GB, else Metal's; less 1 GB for the display and the driver.
fn gpu_limit(recommended: f64, cfg: &Value) -> f64 {
    if cfg!(target_os = "linux") {
        return cfg["env"]["HK_SYSMEM"].as_str().and_then(|v| v.parse().ok()).unwrap_or(recommended);
    }
    let ram = sysctl("hw.memsize");
    let limit = {
        let wired = sysctl("iogpu.wired_limit_mb");
        if wired > 0.0 { wired * 1048576.0 } else if ram <= 36.0 * 1073741824.0 { (ram * 2.0 / 3.0).min(recommended) } else { recommended }
    };
    limit - 1073741824.0
}

/// The config's memory against this Mac: the engine measures the model (`<exe> --plan <args>`, no weights read).
fn memory(cfg_path: &Path) -> Value {
    let cfg = read_json(cfg_path);
    let exe = cfg["exe"].as_str().unwrap_or("").to_string();
    let args: Vec<String> = cfg["args"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default();
    let envs: Vec<(String, String)> = cfg["env"].as_object().map(|e| e.iter().map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string())).collect()).unwrap_or_default();
    let out = match Command::new(&exe).arg("--plan").args(&args).envs(envs).stderr(Stdio::null()).output() {
        Ok(o) => String::from_utf8_lossy(&o.stdout).into_owned(),
        Err(e) => return json!({"error": format!("{exe}: {e} (make build)")}),
    };
    let Some(p) = out.lines().find(|l| l.starts_with('{')).and_then(|l| serde_json::from_str::<Value>(l).ok()) else {
        return json!({"error": "the engine could not measure this model (is the model downloaded? an engine without --plan: make build)"});
    };
    let n = |v: &Value| v.as_f64().unwrap_or(0.0);
    let (g, h) = (&p["gpu"], &p["host"]);
    let ctx = n(&p["context"]);
    let per_token = n(&p["per_token"]);
    let limit = gpu_limit(n(&g["limit"]), &cfg);
    // Metal maps the whole model file: the CPU side's share of the weights counts on the GPU too (on Vulkan it stays
    // in system memory)
    let host_on_gpu = if cfg!(target_os = "linux") { 0.0 } else { n(&h["model"]) };
    let gpu_need = n(&g["model"]) + host_on_gpu + n(&g["context"]) + n(&g["compute"]);
    let need = gpu_need + n(&h["model"]) - host_on_gpu + n(&h["context"]) + n(&h["compute"]);
    let train = n(&p["n_ctx_train"]);
    let mut options: Vec<Value> = CONTEXTS
        .iter()
        .filter(|c| train <= 0.0 || (**c as f64) <= train)
        .map(|&c| {
            let d = per_token * (c as f64 - ctx);
            json!({"ctx": c, "need": need + d, "fits": gpu_need + d <= limit})
        })
        .collect();
    if let Some(i) = options.iter().rposition(|o| o["fits"] == true) {
        options[i]["recommended"] = json!(true);
    }
    json!({"context": ctx, "need": need, "gpu_need": gpu_need, "limit": limit, "free": free_now(), "options": options,
           "weights": n(&g["model"]) + n(&h["model"]), "cache": n(&g["context"]) + n(&h["context"]), "work": n(&g["compute"]) + n(&h["compute"])})
}

/// The programs using the most memory now (each name's processes added up; 300 MB or more), largest first.
fn memory_apps() -> Vec<(String, f64)> {
    let out = Command::new("ps").args(["-axo", "uid=,rss=,comm="]).output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
    let me = unsafe { libc::getuid() }.to_string();
    let mut by: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for l in out.lines() {
        let mut f = l.split_whitespace();
        let (Some(uid), Some(rss)) = (f.next(), f.next()) else { continue };
        let comm: String = f.collect::<Vec<_>>().join(" ");
        let name = match comm.find(".app/") {
            Some(i) => Path::new(&comm[..i + 4]).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or(comm.clone()),
            None => Path::new(&comm).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or(comm.clone()),
        };
        if uid != me || name.starts_with("strata-") || comm.starts_with("/System/") {
            continue;
        }
        *by.entry(name).or_default() += rss.parse::<f64>().unwrap_or(0.0) * 1024.0;
    }
    let mut v: Vec<(String, f64)> = by.into_iter().filter(|(_, b)| *b >= 300e6).collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    v.truncate(8);
    v
}

/// Rewrite the config's --max-context (and --prefill-batch 512 for a tight fit), keeping its key order.
fn set_context(cfg_path: &Path, ctx: u64, batch512: bool) -> Result<(), String> {
    let mut cfg = read_json(cfg_path);
    let a = cfg["args"].as_array_mut().ok_or("the config has no args")?;
    let mut set = |flag: &str, val: String| match a.iter().position(|x| x == flag) {
        Some(i) if i + 1 < a.len() => a[i + 1] = json!(val),
        _ => a.extend([json!(flag), json!(val)]),
    };
    set("--max-context", ctx.to_string());
    if batch512 {
        set("--prefill-batch", "512".into());
    }
    std::fs::write(cfg_path, serde_json::to_string_pretty(&cfg).unwrap() + "\n").map_err(|e| e.to_string())
}

// ------------------------------------------------------------------ the server

struct Server {
    child: Option<Child>,
    port: u16,
    key: String,
    stopping: Option<Instant>,
}

/// Start the server with this config; its output goes to the log.  On Linux before the next login, the GPU's group
/// (render) is not in this session yet: sg gives it.
fn start(srv: &Arc<Mutex<Server>>, root: &Path, cfg: &Path, tx: &Sender<Msg>) -> Result<(), String> {
    let mut s = srv.lock().unwrap();
    if s.child.is_some() {
        return Err("the server is already running".into());
    }
    let exe = root.join("engine-metal/strata-server");
    if !exe.is_file() {
        return Err(format!("{} is missing: make server", exe.display()));
    }
    let port = s.port.to_string();
    let base = [exe.to_string_lossy().into_owned(), "--engine".into(), "strata".into(), "--config".into(), cfg.to_string_lossy().into_owned(), "--port".into(), port, "--replace".into()];
    let in_render = cfg!(target_os = "linux") && {
        let gid = std::fs::read_to_string("/etc/group").ok().and_then(|g| g.lines().find(|l| l.starts_with("render:")).and_then(|l| l.split(':').nth(2)?.parse::<u32>().ok()));
        let mut groups = vec![0 as libc::gid_t; 64];
        let k = unsafe { libc::getgroups(64, groups.as_mut_ptr()) };
        gid.is_some_and(|g| !groups[..k.max(0) as usize].contains(&g))
    };
    let mut cmd = if in_render {
        let mut c = Command::new("sg");
        c.args(["render", "-c", &base.iter().map(|a| format!("'{}'", a.replace('\'', "'\\''"))).collect::<Vec<_>>().join(" ")]);
        c
    } else {
        let mut c = Command::new(&base[0]);
        c.args(&base[1..]);
        c
    };
    let mut child = cmd.current_dir(root).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().map_err(|e| format!("the server could not start: {e}"))?;
    let outs: Vec<Box<dyn Read + Send>> = vec![Box::new(child.stdout.take().unwrap()), Box::new(child.stderr.take().unwrap())];
    for out in outs {
        let tx = tx.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(out).lines().map_while(Result::ok) {
                if tx.send(Msg::Log(line)).is_err() {
                    break;
                }
            }
        });
    }
    let _ = tx.send(Msg::Log(format!("── started {} on port {} ──", cfg.file_name().unwrap_or_default().to_string_lossy(), s.port)));
    s.child = Some(child);
    s.stopping = None;
    Ok(())
}

/// The server's own way to stop: SIGTERM (QUIT to the engine, which frees its memory); SIGKILL after 120 s.
fn stop(srv: &Arc<Mutex<Server>>) {
    let mut s = srv.lock().unwrap();
    if let Some(c) = &s.child {
        unsafe { libc::kill(c.id() as i32, libc::SIGTERM) };
        s.stopping = Some(Instant::now());
    }
}

fn stop_and_wait(srv: &Arc<Mutex<Server>>) {
    stop(srv);
    loop {
        {
            let mut s = srv.lock().unwrap();
            match s.child.as_mut().map(|c| c.try_wait()) {
                None | Some(Ok(Some(_))) | Some(Err(_)) => {
                    s.child = None;
                    return;
                }
                _ if s.stopping.is_some_and(|t| t.elapsed() > Duration::from_secs(120)) => {
                    let _ = s.child.as_mut().unwrap().kill();
                }
                _ => {}
            }
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn agent(timeout: Option<Duration>) -> ureq::Agent {
    ureq::Agent::config_builder().timeout_global(timeout).http_status_as_error(false).build().into()
}

fn get(url: &str, key: &str) -> Option<Value> {
    let mut r = agent(Some(Duration::from_secs(3))).get(url);
    if !key.is_empty() {
        r = r.header("Authorization", &format!("Bearer {key}"));
    }
    let mut resp = r.call().ok()?;
    (resp.status() == 200).then(|| serde_json::from_str(&resp.body_mut().read_to_string().ok()?).ok())?
}

fn post(url: &str, key: &str) -> String {
    let mut r = agent(Some(Duration::from_secs(600))).post(url).header("Content-Type", "application/json");
    if !key.is_empty() {
        r = r.header("Authorization", &format!("Bearer {key}"));
    }
    match r.send("{}") {
        Ok(mut resp) => resp.body_mut().read_to_string().unwrap_or_default(),
        Err(e) => e.to_string(),
    }
}

/// Every 1.5 s: is the process alive, /health, /metrics -> one status.
fn poll(srv: Arc<Mutex<Server>>, tx: Sender<Msg>) {
    loop {
        let (alive, port, key) = {
            let mut s = srv.lock().unwrap();
            if let Some(Ok(Some(code))) = s.child.as_mut().map(|c| c.try_wait()) {
                s.child = None;
                let _ = tx.send(Msg::Log(format!("── the server exited ({}) ──", code.code().map(|c| c.to_string()).unwrap_or("signal".into()))));
            }
            (s.child.is_some(), s.port, s.key.clone())
        };
        let base = format!("http://127.0.0.1:{port}");
        let health = if alive { get(&format!("{base}/health"), "") } else { None };
        let metrics = health.as_ref().and_then(|_| get(&format!("{base}/metrics"), &key));
        if tx.send(Msg::Status(json!({"alive": alive, "health": health, "metrics": metrics}))).is_err() {
            return;
        }
        std::thread::sleep(Duration::from_millis(1500));
    }
}

/// One chat turn, streamed (OpenAI chat completions): the thinking and the answer as they come.
fn chat(port: u16, key: String, messages: Vec<Value>, effort: &str, stop: Arc<AtomicBool>, tx: Sender<Msg>) {
    let effort = effort.to_string();
    std::thread::spawn(move || {
        let body = json!({"model": "strata", "stream": true, "reasoning_effort": effort, "messages": messages}).to_string();
        let mut r = agent(None).post(&format!("http://127.0.0.1:{port}/v1/chat/completions")).header("Content-Type", "application/json");
        if !key.is_empty() {
            r = r.header("Authorization", &format!("Bearer {key}"));
        }
        let mut resp = match r.send(&body) {
            Ok(r) => r,
            Err(e) => return drop(tx.send(Msg::ChatEnd(Some(e.to_string())))),
        };
        if resp.status() != 200 {
            let t = resp.body_mut().read_to_string().unwrap_or_default();
            let msg = serde_json::from_str::<Value>(&t).ok().and_then(|v| v["error"]["message"].as_str().map(String::from)).unwrap_or(t);
            return drop(tx.send(Msg::ChatEnd(Some(msg))));
        }
        let reader = BufReader::new(resp.into_body().into_reader());
        for line in reader.lines().map_while(Result::ok) {
            if stop.load(Ordering::Relaxed) {
                return drop(tx.send(Msg::ChatEnd(Some("stopped".into())))); // the connection closes: the server stops it
            }
            let Some(data) = line.strip_prefix("data: ") else { continue };
            if data == "[DONE]" {
                break;
            }
            let Ok(v) = serde_json::from_str::<Value>(data) else { continue };
            if let Some(m) = v["error"]["message"].as_str() {
                return drop(tx.send(Msg::ChatEnd(Some(m.to_string()))));
            }
            let d = &v["choices"][0]["delta"];
            for (k, think) in [("reasoning_content", true), ("content", false)] {
                if let Some(t) = d[k].as_str().filter(|t| !t.is_empty()) {
                    let _ = tx.send(Msg::Delta { think, text: t.to_string() });
                }
            }
        }
        let _ = tx.send(Msg::ChatEnd(None));
    });
}

// ------------------------------------------------------------------ the screen

struct Turn {
    user: String,
    think: String,
    answer: String,
    error: Option<String>,
}

struct App {
    root: PathBuf,
    configs: Vec<PathBuf>,
    cfg: usize,
    srv: Arc<Mutex<Server>>,
    tx: Sender<Msg>,
    tab: usize,
    status: Value,
    plan: Value,
    apps: Vec<(String, f64)>,
    log: VecDeque<String>,
    log_scroll: u16,
    turns: Vec<Turn>,
    input: String,
    busy: bool,
    pick: usize,
    note: String,
    quit: bool,
    effort: usize,
    stop: Arc<AtomicBool>,
    chat_scroll: u16,
    mpick: usize,
    pulling: Option<String>,
    help: bool,
    power_peak: f64,
}

/// A line of the Models view: a run config here, or a model make pull can bring.
struct ModelRow {
    title: String,
    config: Option<usize>,
    pull: Option<&'static str>,
    size: f64,
}

impl App {
    fn cfg_path(&self) -> Option<&PathBuf> {
        self.configs.get(self.cfg)
    }

    fn running(&self) -> bool {
        self.srv.lock().unwrap().child.is_some()
    }

    fn phase(&self) -> &'static str {
        let s = self.srv.lock().unwrap();
        if s.stopping.is_some() && s.child.is_some() {
            return "stopping";
        }
        if s.child.is_none() {
            return "stopped";
        }
        match self.status["metrics"]["live"]["state"].as_str() {
            _ if self.status["health"].is_null() => "starting",
            Some("reading" | "generating") => "busy",
            Some("unloaded") => "unloaded",
            _ => "ready",
        }
    }

    /// The selected config: its port and key for the server, its memory measured off the screen's thread.
    fn select(&mut self) {
        let Some(p) = self.cfg_path().cloned() else { return };
        let cfg = read_json(&p);
        {
            let mut s = self.srv.lock().unwrap();
            s.port = cfg["port"].as_u64().unwrap_or(8090) as u16;
            s.key = api_key(&cfg);
        }
        self.plan = json!({"measuring": true});
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            let _ = tx.send(Msg::Plan(memory(&p)));
            let _ = tx.send(Msg::Apps(memory_apps()));
        });
    }

    fn toggle(&mut self) {
        if self.running() {
            stop(&self.srv);
            self.note = "stopping: the engine frees its memory first".into();
        } else if let Some(p) = self.cfg_path().cloned() {
            match start(&self.srv, &self.root, &p, &self.tx) {
                Ok(()) => self.note = "starting: the model loads (a few seconds to a minute)".into(),
                Err(e) => self.note = e,
            }
        }
    }

    fn restart(&mut self) {
        let (srv, root, tx) = (self.srv.clone(), self.root.clone(), self.tx.clone());
        let Some(p) = self.cfg_path().cloned() else { return };
        self.note = "restarting ...".into();
        std::thread::spawn(move || {
            stop_and_wait(&srv);
            if let Err(e) = start(&srv, &root, &p, &tx) {
                let _ = tx.send(Msg::Log(format!("── {e} ──")));
            }
        });
    }

    fn load_unload(&mut self) {
        let what = if self.status["health"]["loaded"] == true { "/unload" } else { "/load" };
        let (port, key, tx) = {
            let s = self.srv.lock().unwrap();
            (s.port, s.key.clone(), self.tx.clone())
        };
        std::thread::spawn(move || {
            let r = post(&format!("http://127.0.0.1:{port}{what}"), &key);
            let _ = tx.send(Msg::Log(format!("── {what}: {r} ──")));
        });
    }

    fn send_chat(&mut self) {
        let text = self.input.trim().to_string();
        if text.is_empty() || self.busy {
            return;
        }
        if self.status["health"].is_null() {
            self.note = "start the server first (Ctrl+S)".into();
            return;
        }
        self.input.clear();
        self.turns.push(Turn { user: text, think: String::new(), answer: String::new(), error: None });
        let mut messages = vec![];
        for t in &self.turns {
            messages.push(json!({"role": "user", "content": t.user}));
            if !t.answer.is_empty() {
                messages.push(json!({"role": "assistant", "content": t.answer}));
            }
        }
        self.busy = true;
        self.chat_scroll = 0;
        self.stop = Arc::new(AtomicBool::new(false));
        let (port, key) = {
            let s = self.srv.lock().unwrap();
            (s.port, s.key.clone())
        };
        chat(port, key, messages, EFFORTS[self.effort].0, self.stop.clone(), self.tx.clone());
    }

    /// `p` on Monitor: the CPU governor to performance (clocks stay up) or back (Linux, sudo; macOS: Strata.app's
    /// System card has High Power Mode).
    fn performance(&mut self) {
        if !cfg!(target_os = "linux") {
            self.note = "on macOS: Strata.app > System > High Power Mode (or sudo pmset -a powermode 2)".into();
            return;
        }
        let cur = std::fs::read_to_string("/sys/devices/system/cpu/cpufreq/policy0/scaling_governor").unwrap_or_default();
        let to = governor_target(cur.trim());
        let cmd = format!("for f in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do echo {to} > $f; done");
        let ok = Command::new("sudo").args(["-n", "sh", "-c", &cmd]).status().map(|s| s.success()).unwrap_or(false);
        self.note = if ok { format!("CPU governor: {to}") } else { format!("needs sudo: sudo sh -c '{cmd}'") };
    }

    /// The Models view's lines: every run config here, then what make pull can download that has no config yet.
    fn models(&self) -> Vec<ModelRow> {
        let mut rows: Vec<ModelRow> = self
            .configs
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let c = read_json(p);
                let gguf = arg(&c, "--gguf").unwrap_or_default();
                ModelRow {
                    title: c["model_name"].as_str().map(String::from).unwrap_or_else(|| p.file_stem().unwrap_or_default().to_string_lossy().into_owned()),
                    config: Some(i),
                    pull: None,
                    size: std::fs::metadata(&gguf).map(|m| m.len() as f64 / GB).unwrap_or(0.0),
                }
            })
            .collect();
        for (pull, cfg, title, size) in CATALOG {
            if !self.root.join(format!("strata-{cfg}.json")).exists() {
                rows.push(ModelRow { title: title.into(), config: None, pull: Some(pull), size });
            }
        }
        rows
    }

    /// Enter in Models: switch to a config (the server restarts on it when running), or download a model.
    fn activate_model(&mut self) {
        let rows = self.models();
        let Some(row) = rows.get(self.mpick) else { return };
        if let Some(i) = row.config {
            let was = self.running();
            self.cfg = i;
            self.select();
            self.note = format!("{} selected{}", row.title, if was { ": restarting on it" } else { ": Ctrl+S starts it" });
            if was {
                self.restart();
            }
        } else if let Some(name) = row.pull {
            if self.pulling.is_some() {
                self.note = "a download is already running (F5 Log)".into();
                return;
            }
            self.pulling = Some(name.to_string());
            self.note = format!("downloading {} ({:.1} GB): progress in Log (F5); it writes the run config", row.title, row.size);
            let (root, tx, name) = (self.root.clone(), self.tx.clone(), name.to_string());
            std::thread::spawn(move || {
                let child = Command::new("make").args(["--no-print-directory", "pull", &format!("MODEL={name}")]).current_dir(&root).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn();
                let ok = match child {
                    Ok(mut c) => {
                        let err = c.stderr.take().unwrap();
                        let t2 = tx.clone();
                        std::thread::spawn(move || BufReader::new(err).lines().map_while(Result::ok).for_each(|l| drop(t2.send(Msg::Log(l)))));
                        BufReader::new(c.stdout.take().unwrap()).lines().map_while(Result::ok).for_each(|l| drop(tx.send(Msg::Log(l))));
                        c.wait().map(|s| s.success()).unwrap_or(false)
                    }
                    Err(e) => {
                        let _ = tx.send(Msg::Log(format!("make pull: {e}")));
                        false
                    }
                };
                let _ = tx.send(Msg::Pulled(name, ok));
            });
        }
    }

    fn apply_context(&mut self) {
        let Some(o) = self.plan["options"].as_array().and_then(|o| o.get(self.pick)).cloned() else { return };
        let Some(p) = self.cfg_path().cloned() else { return };
        let ctx = o["ctx"].as_u64().unwrap_or(0);
        match set_context(&p, ctx, false) {
            Ok(()) => {
                self.note = format!("context {} written to {}", k(ctx), p.file_name().unwrap_or_default().to_string_lossy());
                if self.running() {
                    self.restart();
                }
                self.select();
            }
            Err(e) => self.note = e,
        }
    }

    fn key(&mut self, code: KeyCode, mods: KeyModifiers) {
        let ctrl = mods.contains(KeyModifiers::CONTROL);
        let typing = self.tab == 0 && !ctrl;
        let letter = |c: char| matches!(code, KeyCode::Char(x) if x == c) && (ctrl || self.tab != 0);
        if self.help {
            self.help = false; // any key closes the keys' list
            return;
        }
        match code {
            KeyCode::Char('?') if self.tab != 0 => self.help = true,
            _ if letter('t') => {
                self.effort = (self.effort + 1) % EFFORTS.len();
                self.note = format!("thinking effort: {} (from the next message)", EFFORTS[self.effort].1);
            }
            _ if letter('o') => {
                if self.busy {
                    self.note = "an answer is being written: Esc stops it".into();
                } else {
                    self.turns.clear();
                    self.chat_scroll = 0;
                    self.note = "new chat".into();
                }
            }
            KeyCode::Esc if self.tab == 0 && self.busy => {
                self.stop.store(true, Ordering::Relaxed);
                self.note = "stopping the answer ...".into();
            }
            KeyCode::PageUp if self.tab == 0 => self.chat_scroll = self.chat_scroll.saturating_add(10),
            KeyCode::PageDown if self.tab == 0 => self.chat_scroll = self.chat_scroll.saturating_sub(10),
            KeyCode::Char('p') if self.tab == 1 => self.performance(),
            KeyCode::Enter if self.tab == 3 => self.activate_model(),
            KeyCode::Up if self.tab == 3 => self.mpick = self.mpick.saturating_sub(1),
            KeyCode::Down if self.tab == 3 => self.mpick = (self.mpick + 1).min(self.models().len().saturating_sub(1)),
            KeyCode::Tab => self.tab = (self.tab + 1) % TABS.len(),
            KeyCode::BackTab => self.tab = (self.tab + TABS.len() - 1) % TABS.len(),
            KeyCode::F(n) if (1..=5).contains(&n) => self.tab = n as usize - 1,
            _ if letter('q') || (ctrl && code == KeyCode::Char('c')) => self.quit = true,
            _ if letter('s') => self.toggle(),
            _ if letter('r') => self.restart(),
            _ if letter('l') => self.load_unload(),
            _ if letter('n') => {
                if self.running() {
                    self.note = "stop the server to pick another config".into();
                } else if !self.configs.is_empty() {
                    self.cfg = (self.cfg + 1) % self.configs.len();
                    self.select();
                }
            }
            KeyCode::Enter if self.tab == 0 => self.send_chat(),
            KeyCode::Enter if self.tab == 2 => self.apply_context(),
            KeyCode::Up if self.tab == 2 => self.pick = self.pick.saturating_sub(1),
            KeyCode::Down if self.tab == 2 => self.pick = (self.pick + 1).min(self.plan["options"].as_array().map(|o| o.len().saturating_sub(1)).unwrap_or(0)),
            KeyCode::Up | KeyCode::PageUp if self.tab == 4 => self.log_scroll = self.log_scroll.saturating_add(if code == KeyCode::PageUp { 20 } else { 1 }),
            KeyCode::Down | KeyCode::PageDown if self.tab == 4 => self.log_scroll = self.log_scroll.saturating_sub(if code == KeyCode::PageDown { 20 } else { 1 }),
            KeyCode::Backspace if typing => drop(self.input.pop()),
            KeyCode::Esc if typing => self.input.clear(),
            KeyCode::Char(c) if typing => self.input.push(c),
            _ => {}
        }
    }

    fn on(&mut self, m: Msg) {
        match m {
            Msg::Log(l) => {
                if self.log.len() >= 3000 {
                    self.log.pop_front();
                }
                if l.contains("ran out of memory") || l.contains("Insufficient Memory") || l.contains("OutOfDeviceMemory") {
                    self.note = "the GPU ran out of memory: pick a smaller context in Memory (F3), or close programs".into();
                    self.tab = 2;
                }
                self.log.push_back(l);
            }
            Msg::Status(s) => {
                let hw = &s["metrics"]["hardware"];
                let w = hw["system_power"].as_f64().or(hw["gpu_power"].as_f64()).unwrap_or(0.0);
                self.power_peak = self.power_peak.max(w);
                self.status = s;
            }
            Msg::Plan(p) => {
                self.pick = p["options"].as_array().and_then(|o| o.iter().position(|x| x["ctx"].as_f64() == p["context"].as_f64())).unwrap_or(0);
                self.plan = p;
            }
            Msg::Apps(a) => self.apps = a,
            Msg::Pulled(name, ok) => {
                self.pulling = None;
                self.configs = configs(&self.root);
                let cfg = CATALOG.iter().find(|c| c.0 == name).map(|c| format!("strata-{}.json", c.1)).unwrap_or_default();
                if let Some(i) = self.configs.iter().position(|p| p.file_name().is_some_and(|n| n.to_string_lossy() == cfg)) {
                    if !self.running() {
                        self.cfg = i;
                        self.select();
                    }
                }
                self.note = if ok { format!("{name} downloaded: pick it in Models (F4), then Ctrl+S") } else { format!("{name}: the download stopped (Log, F5); run it again to continue") };
            }
            Msg::Delta { think, text } => {
                if let Some(t) = self.turns.last_mut() {
                    if think { t.think.push_str(&text) } else { t.answer.push_str(&text) }
                }
            }
            Msg::ChatEnd(err) => {
                self.busy = false;
                if let (Some(e), Some(t)) = (err, self.turns.last_mut()) {
                    if e.contains("out of memory") {
                        self.tab = 2;
                    }
                    t.error = Some(e);
                }
            }
        }
    }
}

/// How much of the context window the chat uses: (tokens, the window).  Writing: the prompt plus what is written so
/// far; else the last request's prompt and answer (the next message starts from there).
fn context_fill(status: &Value, busy: bool, chatting: bool) -> Option<(u64, u64)> {
    let max = status["health"]["max_context"].as_u64()?;
    let m = &status["metrics"];
    let used = if busy {
        let l = &m["live"];
        l["prompt_tokens"].as_u64().or(l["prompt_total"].as_u64()).unwrap_or(0) + l["generated"].as_u64().unwrap_or(0)
    } else if chatting {
        let r = &m["requests"][0];
        r["prompt_tokens"].as_u64()? + r["output_tokens"].as_u64().unwrap_or(0)
    } else {
        return None;
    };
    Some((used, max))
}

fn thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// The clusters' clocks in words: "E-cores 0.6 of 2.1 GHz · P-cores 3.0 of 3.0 GHz".
fn clusters_text(v: &Value) -> String {
    let Some(c) = v.as_array().filter(|c| !c.is_empty()) else { return String::new() };
    let low = c.iter().filter_map(|x| x["max_mhz"].as_f64()).fold(f64::MAX, f64::min);
    c.iter()
        .map(|x| {
            let name = if c.len() > 1 && x["max_mhz"].as_f64() == Some(low) { "E-cores" } else { "P-cores" };
            format!("{name} {:.1} of {:.1} GHz", x["mhz"].as_f64().unwrap_or(0.0) / 1000.0, x["max_mhz"].as_f64().unwrap_or(0.0) / 1000.0)
        })
        .collect::<Vec<_>>()
        .join(" · ")
}

/// A bar of `width` cells for a share 0..1: filled █, empty ░.
fn bar(ratio: f64, width: usize) -> String {
    let full = (ratio.clamp(0.0, 1.0) * width as f64).round() as usize;
    format!("{}{}", "█".repeat(full), "░".repeat(width - full))
}

/// One Monitor line: the name, a grey bar, the value in words.
fn meter(name: &str, ratio: f64, value: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{name:<12}"), Style::new().bold()),
        Span::styled(bar(ratio, 22), Style::new().fg(Color::Gray)),
        Span::raw(format!("  {value}")),
    ])
}

/// The Power bar: watts against the highest seen this session (at least 20 W, about an M1's chip under load).
/// Linux: the whole system's, with the chip's (CPU + GPU, the SMC heatpipe) beside it; macOS: the GPU's.
fn power_gauge(hw: &Value, peak: f64) -> (f64, String) {
    let w = |x: f64| format!("{x:.1} W");
    let (now, label) = match (hw["system_power"].as_f64(), hw["gpu_power"].as_f64()) {
        (Some(sys), chip) => (sys, format!("{} system{}", w(sys), chip.map(|c| format!(" · {} chip (CPU + GPU)", w(c))).unwrap_or_default())),
        (None, Some(gpu)) => (gpu, format!("{} GPU", w(gpu))),
        (None, None) => return (0.0, "–".into()),
    };
    let top = peak.max(now).max(20.0);
    ((now / top).clamp(0.0, 1.0), format!("{label} · peak {}", w(peak.max(now))))
}

/// The CPU governor `p` switches to: performance, or back to the kernel's usual one.
fn governor_target(current: &str) -> &'static str {
    if current == "performance" { "schedutil" } else { "performance" }
}

fn gb(b: f64) -> String {
    format!("{:.1} GB", b / GB)
}

fn k(c: u64) -> String {
    if c >= 1048576 && c % 1048576 == 0 { format!("{}M", c / 1048576) } else if c >= 1024 { format!("{}K", c / 1024) } else { c.to_string() }
}

fn colour(phase: &str) -> Color {
    match phase {
        "ready" => Color::Green,
        "busy" => Color::Blue,
        "starting" | "stopping" => Color::Yellow,
        "unloaded" => Color::Magenta,
        _ => Color::DarkGray,
    }
}

fn draw(f: &mut Frame, app: &App) {
    let [bar, tabs, body, help] = Layout::vertical([Constraint::Length(3), Constraint::Length(1), Constraint::Min(5), Constraint::Length(1)]).areas(f.area());
    // the bar: as the app's
    let phase = app.phase();
    let cfg = app.cfg_path().map(|p| read_json(p)).unwrap_or(json!({}));
    let name = app.cfg_path().and_then(|p| p.file_stem()).map(|s| s.to_string_lossy().trim_start_matches("strata-").to_string()).unwrap_or_else(|| "no run config: make setup".into());
    let port = app.srv.lock().unwrap().port;
    let live = &app.status["metrics"]["live"];
    let rate = live["tok_s"].as_f64().map(|r| format!("{r:.1} tok/s")).unwrap_or_default();
    let mem = if app.plan["need"].is_number() {
        let (need, limit, free) = (app.plan["need"].as_f64().unwrap(), app.plan["limit"].as_f64().unwrap_or(0.0), app.plan["free"].as_f64().unwrap_or(0.0));
        let c = if app.plan["gpu_need"].as_f64().unwrap_or(need) > limit { Color::Red } else if !app.running() && need > free { Color::Yellow } else { Color::Green };
        Span::styled(format!("needs {} · GPU has {}{}", gb(need), gb(limit), if app.running() { String::new() } else { format!(" · {} free", gb(free)) }), Style::new().fg(c))
    } else if let Some(e) = app.plan["error"].as_str() {
        Span::styled(format!("memory ? ({e})"), Style::new().fg(Color::Red))
    } else {
        Span::raw("memory …")
    };
    let line = Line::from(vec![
        Span::styled("● ", Style::new().fg(colour(phase))),
        Span::styled(format!("{phase:<9}"), Style::new().bold()),
        Span::raw(format!("{name}  ")),
        Span::styled(format!("context {}  ", arg(&cfg, "--max-context").and_then(|c| c.parse().ok()).map(k).unwrap_or("?".into())), Style::new().fg(Color::Cyan)),
        Span::raw(format!("127.0.0.1:{port}  ")),
        Span::styled(format!("thinking {}  ", EFFORTS[app.effort].1), Style::new().fg(Color::Magenta)),
        Span::styled(format!("{rate:<12}"), Style::new().fg(Color::Blue)),
        mem,
    ]);
    f.render_widget(Paragraph::new(line).block(Block::bordered().title(" Strata ".bold())), bar);
    f.render_widget(Tabs::new(TABS.iter().enumerate().map(|(i, t)| format!("F{} {t}", i + 1))).select(app.tab).highlight_style(Style::new().reversed()), tabs);
    match app.tab {
        0 => draw_chat(f, app, body),
        1 => draw_monitor(f, app, body),
        2 => draw_memory(f, app, body),
        3 => draw_models(f, app, body),
        _ => draw_log(f, app, body),
    }
    let keys = if app.tab == 0 { "Enter send · Esc stop · Ctrl+S start/stop · Ctrl+T thinking · Ctrl+O new chat · PgUp/PgDn · Tab view · Ctrl+Q quit" } else { "s start/stop · r restart · l load/unload · t thinking · F4 models · ? keys · q quit" };
    let help_line = if app.note.is_empty() { Line::from(keys.dim()) } else { Line::from(vec![Span::styled(app.note.clone(), Style::new().fg(Color::Yellow)), Span::raw("   "), keys.dim()]) };
    f.render_widget(Paragraph::new(help_line), help);
    if app.help {
        let lines = [
            "Ctrl+S   start / stop the server (the model loads; quitting stops it)",
            "Ctrl+R   restart            Ctrl+L   load / unload the model",
            "Ctrl+N   next run config    F4 Models   switch, or download one (make pull)",
            "Ctrl+T   thinking effort: off, low, medium, high",
            "Enter    send               Esc   stop the answer      Ctrl+O   new chat",
            "PgUp / PgDn   scroll the chat or the log      p (Monitor)   CPU performance mode",
            "F3 Memory: Up / Down, Enter   the context (written to the run config)",
            "Tab, F1-F5   views          ?   these keys         Ctrl+Q   quit",
            "",
            "Outside Chat the letters work without Ctrl (s, r, l, n, t, o, q).   Any key closes this.",
        ];
        let w = 84.min(f.area().width.saturating_sub(4));
        let h = (lines.len() as u16 + 2).min(f.area().height.saturating_sub(2));
        let r = Rect::new((f.area().width.saturating_sub(w)) / 2, (f.area().height.saturating_sub(h)) / 2, w, h);
        f.render_widget(Clear, r);
        f.render_widget(Paragraph::new(lines.iter().map(|l| Line::from(*l)).collect::<Vec<_>>()).block(Block::bordered().title(" Keys ")), r);
    }
}

fn draw_models(f: &mut Frame, app: &App, area: Rect) {
    let ram = if cfg!(target_os = "linux") { meminfo("MemTotal") } else { sysctl("hw.memsize") };
    let room = ram - (3.0 * 1073741824.0f64).max(ram * 0.15);
    let rows: Vec<Row> = app.models().iter().enumerate().map(|(i, m)| {
        let current = m.config == Some(app.cfg);
        let (state, colour) = match (m.config, m.pull) {
            _ if current && app.running() => ("running".to_string(), Color::Green),
            _ if current => ("selected".to_string(), Color::Green),
            (Some(_), _) => ("installed".to_string(), Color::Reset),
            (_, Some(p)) if app.pulling.as_deref() == Some(p) => ("downloading …".to_string(), Color::Yellow),
            _ => (format!("download {:.1} GB", m.size), Color::Cyan),
        };
        let fits = if m.size <= 0.0 { "" } else if m.size * GB + 1e9 <= room { "fits" } else { "too big for this Mac" };
        let style = if i == app.mpick { Style::new().reversed() } else { Style::new().fg(colour) };
        Row::new([format!("{}{}", if current { "▸ " } else { "  " }, m.title), state, if m.size > 0.0 { format!("{:.1} GB", m.size) } else { "–".into() }, fits.to_string()]).style(style)
    }).collect();
    f.render_widget(
        Table::new(rows, [Constraint::Min(28), Constraint::Length(20), Constraint::Length(10), Constraint::Length(22)])
            .header(Row::new(["model", "", "file", "for this Mac"]).style(Style::new().bold()))
            .block(Block::bordered().title(" Models: Up / Down, Enter to switch (or download) ")),
        area,
    );
}

/// How many screen lines a text takes at this width (wrapped).
fn height(text: &str, width: u16) -> u16 {
    let w = width.max(1) as usize;
    text.split('\n').map(|l| (l.chars().count().max(1)).div_ceil(w) as u16).sum()
}

fn draw_chat(f: &mut Frame, app: &App, area: Rect) {
    let [hist, input] = Layout::vertical([Constraint::Min(3), Constraint::Length(3)]).areas(area);
    let mut lines: Vec<Line> = vec![];
    let mut raw = String::new();
    for t in &app.turns {
        lines.push(Line::from(vec!["you  ".bold().cyan(), Span::raw(t.user.clone())]));
        raw.push_str(&format!("you  {}\n", t.user));
        if !t.think.is_empty() {
            for l in t.think.trim().lines() {
                lines.push(Line::from(Span::styled(format!("     {l}"), Style::new().fg(Color::DarkGray).add_modifier(Modifier::ITALIC))));
                raw.push_str(&format!("     {l}\n"));
            }
        }
        let first = t.answer.trim_start();
        for (i, l) in first.lines().enumerate() {
            lines.push(if i == 0 { Line::from(vec!["model ".bold().green(), Span::raw(l.to_string())]) } else { Line::from(format!("      {l}")) });
            raw.push_str(&format!("model {l}\n"));
        }
        match t.error.as_deref() {
            Some("stopped") => {
                lines.push(Line::from("      (stopped)".dim()));
                raw.push_str("      (stopped)\n");
            }
            Some(e) => {
                lines.push(Line::from(Span::styled(format!("error {e}"), Style::new().fg(Color::Red))));
                raw.push_str(&format!("error {e}\n"));
            }
            None => {}
        }
        lines.push(Line::raw(""));
        raw.push('\n');
    }
    if app.turns.is_empty() {
        lines.push(Line::from("Type a message and press Enter. The answer streams here; the model's thinking is shown dimmed.".dim()));
    }
    let inner_w = hist.width.saturating_sub(2);
    let scroll = height(&raw, inner_w).saturating_sub(hist.height.saturating_sub(2)).saturating_sub(app.chat_scroll);
    let fill = context_fill(&app.status, app.busy, !app.turns.is_empty());
    let mut chat_title = " Chat ".to_string();
    if let Some((used, max)) = fill {
        chat_title = format!(" Chat · context {} / {} ({:.1}%) ", thousands(used), thousands(max), 100.0 * used as f64 / max.max(1) as f64);
    }
    if app.busy {
        if let Some(t) = app.status["metrics"]["live"]["elapsed_s"].as_f64() {
            chat_title.push_str(&format!("· {t:.1} s "));
        }
    }
    f.render_widget(Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false }).scroll((scroll, 0)).block(Block::bordered().title(chat_title)), hist);
    let title = if app.busy { " answering … " } else { " message " };
    f.render_widget(Paragraph::new(format!("{}▏", app.input)).block(Block::bordered().title(title)), input);
}

fn draw_monitor(f: &mut Frame, app: &App, area: Rect) {
    let m = &app.status["metrics"];
    if m.is_null() {
        f.render_widget(Paragraph::new("The server is not running, or not answering yet. Ctrl+S / s starts it.").block(Block::bordered().title(" Monitor ")), area);
        return;
    }
    let [top, gauges, reqs] = Layout::vertical([Constraint::Length(4), Constraint::Length(9), Constraint::Min(4)]).areas(area);
    let (l, hw) = (&m["live"], &m["hardware"]);
    let s = |v: &Value| match v {
        Value::Null => "–".to_string(),
        Value::Number(n) => format!("{}", n.as_f64().map(|x| (x * 10.0).round() / 10.0).unwrap_or(0.0)),
        v => v.as_str().map(String::from).unwrap_or(v.to_string()),
    };
    let state = format!(
        "state {}   speed {} tok/s   prompt {}/{}   generated {}   queued {}   requests {}",
        s(&l["state"]), s(&l["tok_s"]), s(&l["prompt_read"]), s(&l["prompt_total"]), s(&l["generated"]), s(&l["queued"]), s(&m["totals"]["requests"])
    );
    let model = format!("model {}   context {}   engine {}", s(&app.status["health"]["model"]), s(&app.status["health"]["max_context"]), s(&m["engine"]["engine"]));
    f.render_widget(Paragraph::new(vec![Line::from(state), Line::from(model.dim())]).block(Block::bordered().title(" Model ")), top);
    // one line each: the name, a bar, and the value in words beside it (a value printed inside a bar is unreadable
    // over its filled part)
    let gbs = |v: &Value| v.as_f64().unwrap_or(0.0) / GB;
    let ratio = |a: &Value, b: &Value| (a.as_f64().unwrap_or(0.0) / b.as_f64().unwrap_or(1.0).max(1.0)).clamp(0.0, 1.0);
    let mut lines: Vec<Line> = vec![];
    let used_of = |a: &Value, b: &Value| format!("{:.1} of {:.1} GB  ({:.0}%)", gbs(a), gbs(b), 100.0 * ratio(a, b));
    lines.push(meter("GPU memory", ratio(&hw["gpu_mem_used"], &hw["gpu_mem_total"]), &used_of(&hw["gpu_mem_used"], &hw["gpu_mem_total"])));
    lines.push(if hw["gpu_util"].is_null() {
        // Asahi Linux: the driver reports no GPU load or temperature
        Line::from(vec![Span::styled(format!("{:<12}", "GPU"), Style::new().bold()), "load not reported by Linux's Apple GPU driver".dim()])
    } else {
        meter("GPU load", hw["gpu_util"].as_f64().unwrap_or(0.0) / 100.0, &format!("{}%   {} °C", s(&hw["gpu_util"]), s(&hw["gpu_temp"])))
    });
    lines.push(meter("RAM", ratio(&hw["ram_used"], &hw["ram_total"]), &used_of(&hw["ram_used"], &hw["ram_total"])));
    let cpu = hw["cpu"].as_f64().unwrap_or(0.0);
    lines.push(meter("CPU", cpu / 100.0, &format!("{cpu:.0}%   {}", clusters_text(&hw["cpu_clusters"]))));
    let (pr, pl) = power_gauge(hw, app.power_peak);
    lines.push(meter("Power", pr, &pl));
    if let Some(g) = hw["cpu_clusters"][0]["governor"].as_str() {
        let next = governor_target(g);
        lines.push(Line::from(vec![Span::styled(format!("{:<12}", "Governor"), Style::new().bold()), Span::raw(format!("{g}   ")), format!("(p: {next})").dim()]));
    }
    if !hw["disk_read_mb"].is_null() {    // macOS reads the disk's counters; Linux leaves them out
        lines.push(Line::from(vec![Span::styled(format!("{:<12}", "Disk"), Style::new().bold()),
            Span::raw(format!("read {} MB/s · write {} MB/s", s(&hw["disk_read_mb"]), s(&hw["disk_write_mb"])))]));
    }
    f.render_widget(Paragraph::new(lines).block(Block::bordered().title(" Hardware ")), gauges);
    let header = Row::new(["status", "prompt", "reused", "output", "prefill t/s", "decode t/s", "seconds"]).style(Style::new().bold());
    let list: Vec<Row> = m["requests"].as_array().map(|r| r.iter().take(12).map(|q| {
        Row::new([s(&q["finish"]), s(&q["prompt_tokens"]), s(&q["reused"]), s(&q["output_tokens"]), s(&q["prefill_tok_s"]), s(&q["decode_tok_s"]), s(&q["duration_s"])].map(Cell::from))
    }).collect()).unwrap_or_default();
    f.render_widget(Table::new(list, [Constraint::Length(10); 7]).header(header).block(Block::bordered().title(" Recent requests ")), reqs);
}

fn draw_memory(f: &mut Frame, app: &App, area: Rect) {
    let p = &app.plan;
    let [sum, opts, apps] = Layout::vertical([Constraint::Length(5), Constraint::Min(5), Constraint::Length(10)]).areas(area);
    let summary = if let Some(e) = p["error"].as_str() {
        vec![Line::from(Span::styled(e.to_string(), Style::new().fg(Color::Red)))]
    } else if p["need"].is_number() {
        let n = |k: &str| p[k].as_f64().unwrap_or(0.0);
        vec![
            Line::from(format!("at a context of {} tokens this model needs {}: weights {} + conversation cache {} + work buffers {}", n("context") as u64, gb(n("need")), gb(n("weights")), gb(n("cache")), gb(n("work")))),
            Line::from(format!("the GPU can use {} · {} free right now", gb(n("limit")), gb(n("free")))),
            Line::from(if n("gpu_need") > n("limit") { Span::styled("too big at this context: pick a smaller one below".to_string(), Style::new().fg(Color::Red)) } else { Span::styled("fits".to_string(), Style::new().fg(Color::Green)) }),
        ]
    } else {
        vec![Line::from("measuring the model … (the engine's --plan)".dim())]
    };
    f.render_widget(Paragraph::new(summary).block(Block::bordered().title(" Memory ")), sum);
    let options: Vec<Row> = p["options"].as_array().map(|o| o.iter().enumerate().map(|(i, x)| {
        let ctx = x["ctx"].as_u64().unwrap_or(0);
        let mark = if x["recommended"] == true { "recommended" } else if x["fits"] == true { "fits" } else { "too big" };
        let current = x["ctx"].as_f64() == p["context"].as_f64();
        let style = if i == app.pick { Style::new().reversed() } else if x["fits"] == true { Style::new().fg(Color::Green) } else { Style::new().fg(Color::Red) };
        Row::new([format!("{}{}", if current { "▸ " } else { "  " }, k(ctx)), gb(x["need"].as_f64().unwrap_or(0.0)), mark.to_string()]).style(style)
    }).collect()).unwrap_or_default();
    f.render_widget(Table::new(options, [Constraint::Length(10), Constraint::Length(10), Constraint::Length(14)])
        .header(Row::new(["context", "needs", ""]).style(Style::new().bold()))
        .block(Block::bordered().title(" Context: Up / Down, Enter to use it (written to the run config; a running server restarts) ")), opts);
    let rows: Vec<Row> = app.apps.iter().map(|(n, b)| Row::new([n.clone(), gb(*b)])).collect();
    f.render_widget(Table::new(rows, [Constraint::Min(20), Constraint::Length(10)]).block(Block::bordered().title(" Using the most memory now (close what you do not need) ")), apps);
}

fn draw_log(f: &mut Frame, app: &App, area: Rect) {
    let h = area.height.saturating_sub(2) as usize;
    let end = app.log.len().saturating_sub(app.log_scroll as usize);
    let start = end.saturating_sub(h);
    let lines: Vec<Line> = app.log.iter().skip(start).take(end - start).map(|l| Line::from(l.clone())).collect();
    f.render_widget(Paragraph::new(lines).block(Block::bordered().title(" The server's log (Up / Down, PageUp / PageDown) ")), area);
}

fn main() {
    let root = root();
    let configs = configs(&root);
    let (tx, rx): (Sender<Msg>, Receiver<Msg>) = channel();
    let srv = Arc::new(Mutex::new(Server { child: None, port: 8090, key: String::new(), stopping: None }));
    let mut app = App {
        root, configs, cfg: 0, srv: srv.clone(), tx: tx.clone(), tab: 0, status: json!({}), plan: json!({}), apps: vec![], log: VecDeque::new(),
        log_scroll: 0, turns: vec![], input: String::new(), busy: false, pick: 0, note: String::new(), quit: false,
        effort: 2, stop: Arc::new(AtomicBool::new(false)), chat_scroll: 0, mpick: 0, pulling: None, help: false, power_peak: 0.0,
    };
    app.select();
    std::thread::spawn({
        let (srv, tx) = (srv.clone(), tx.clone());
        move || poll(srv, tx)
    });
    let mut term = ratatui::init();
    while !app.quit {
        let _ = term.draw(|f| draw(f, &app));
        if event::poll(Duration::from_millis(200)).unwrap_or(false) {
            if let Ok(Event::Key(k)) = event::read() {
                if k.kind == KeyEventKind::Press {
                    app.note.clear();
                    app.key(k.code, k.modifiers);
                }
            }
        }
        while let Ok(m) = rx.try_recv() {
            app.on(m);
        }
    }
    ratatui::restore();
    if app.running() {
        println!("stopping the server (the engine frees its memory first) ...");
        stop_and_wait(&srv);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_fill_counts_the_conversation() {
        let st = json!({"health": {"max_context": 32768}, "metrics": {"live": {"prompt_tokens": 1200, "generated": 300},
                        "requests": [{"prompt_tokens": 1000, "output_tokens": 234}]}});
        assert_eq!(context_fill(&st, true, true), Some((1500, 32768))); // writing: prompt + so far
        assert_eq!(context_fill(&st, false, true), Some((1234, 32768))); // between messages: the last request
        assert_eq!(context_fill(&st, false, false), None); // a new chat uses nothing yet
        assert_eq!(context_fill(&json!({}), true, true), None); // no server
    }

    #[test]
    fn formats_the_clusters_and_numbers() {
        let c = json!([{"cpus": "0 1 2 3", "mhz": 1704.0, "max_mhz": 2064.0, "governor": "schedutil"},
                       {"cpus": "4 5 6 7", "mhz": 2988.0, "max_mhz": 2988.0, "governor": "schedutil"}]);
        assert_eq!(clusters_text(&c), "E-cores 1.7 of 2.1 GHz · P-cores 3.0 of 3.0 GHz");
        assert_eq!(clusters_text(&json!(null)), "");
        assert_eq!(thousands(32768), "32,768");
        assert_eq!(thousands(1048576), "1,048,576");
        assert_eq!(thousands(7), "7");
    }

    #[test]
    fn power_bar_in_watts() {
        let linux = json!({"gpu_power": 10.4, "system_power": 18.3});
        let (r, l) = power_gauge(&linux, 21.0);
        assert_eq!(l, "18.3 W system · 10.4 W chip (CPU + GPU) · peak 21.0 W");
        assert!((r - 18.3 / 21.0).abs() < 1e-9);
        let (r, l) = power_gauge(&json!({"gpu_power": 8.3}), 0.0); // macOS: the GPU's; the bar's top is at least 20 W
        assert_eq!(l, "8.3 W GPU · peak 8.3 W");
        assert!((r - 8.3 / 20.0).abs() < 1e-9);
        assert_eq!(power_gauge(&json!({}), 5.0), (0.0, "–".to_string()));
    }

    #[test]
    fn readable_meters() {
        assert_eq!(bar(0.5, 10), "█████░░░░░");
        assert_eq!(bar(0.0, 4), "░░░░");
        assert_eq!(bar(1.7, 4), "████"); // clamped
        let c = json!([{"mhz": 600.0, "max_mhz": 2064.0}, {"mhz": 2988.0, "max_mhz": 2988.0}]);
        assert_eq!(clusters_text(&c), "E-cores 0.6 of 2.1 GHz · P-cores 3.0 of 3.0 GHz");
        let line = meter("RAM", 0.5, "8.0 of 16.0 GB  (50%)");
        assert_eq!(line.spans.iter().map(|s| s.content.as_ref()).collect::<String>(), "RAM         ███████████░░░░░░░░░░░  8.0 of 16.0 GB  (50%)");
    }

    #[test]
    fn performance_toggles_the_governor() {
        assert_eq!(governor_target("schedutil"), "performance");
        assert_eq!(governor_target("performance"), "schedutil");
        assert_eq!(governor_target(""), "performance");
    }
}
