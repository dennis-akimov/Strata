//! strata-tui: Strata.app in a terminal, for Linux on an Apple Silicon Mac (and any terminal, over SSH too).
//!
//! The same as the app: a bar with the state, the model's run config, its context, the address, the speed and what
//! the model needs against what this Mac has; under it Chat, Monitor, Memory and Log.  The server is this program's
//! child (engine-metal/strata-server on the config's port): quitting stops it the server's own way (SIGTERM: the
//! engine frees its memory first).
//!
//!   Ctrl+S start / stop   Ctrl+R restart   Ctrl+L load / unload   Ctrl+N next config   Tab next view   Ctrl+Q quit
//!   (outside Chat the same letters work without Ctrl; in Memory, Up / Down and Enter pick the context)

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Cell, Gauge, Paragraph, Row, Table, Tabs, Wrap};
use ratatui::Frame;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const GB: f64 = 1e9;
const CONTEXTS: [u64; 12] = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];
const TABS: [&str; 4] = ["Chat", "Monitor", "Memory", "Log"];

enum Msg {
    Log(String),
    Status(Value),
    Plan(Value),
    Delta { think: bool, text: String },
    ChatEnd(Option<String>),
    Apps(Vec<(String, f64)>),
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

/// What the GPU may hold, less 1 GB for the display and the driver.  Linux (Vulkan on the unified memory): the memory
/// less 3 GB or 15%; macOS: iogpu.wired_limit_mb, else about two thirds of the memory up to 36 GB, else Metal's.
fn gpu_limit(recommended: f64) -> f64 {
    let ram = if cfg!(target_os = "linux") { meminfo("MemTotal") } else { sysctl("hw.memsize") };
    let limit = if cfg!(target_os = "linux") {
        ram - (3.0 * 1073741824.0f64).max(ram * 0.15)
    } else {
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
    let out = match Command::new(&exe).arg("--plan").args(&args).stderr(Stdio::null()).output() {
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
    let limit = gpu_limit(n(&g["limit"]));
    // Metal maps the whole model file: the CPU side's share of the weights counts on the GPU too
    let gpu_need = n(&g["model"]) + n(&h["model"]) + n(&g["context"]) + n(&g["compute"]);
    let need = gpu_need + n(&h["context"]) + n(&h["compute"]);
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
fn chat(port: u16, key: String, messages: Vec<Value>, tx: Sender<Msg>) {
    std::thread::spawn(move || {
        let body = json!({"model": "strata", "stream": true, "messages": messages}).to_string();
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
        let (port, key) = {
            let s = self.srv.lock().unwrap();
            (s.port, s.key.clone())
        };
        chat(port, key, messages, self.tx.clone());
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
        match code {
            KeyCode::Tab => self.tab = (self.tab + 1) % TABS.len(),
            KeyCode::BackTab => self.tab = (self.tab + TABS.len() - 1) % TABS.len(),
            KeyCode::F(n) if (1..=4).contains(&n) => self.tab = n as usize - 1,
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
            KeyCode::Up | KeyCode::PageUp if self.tab == 3 => self.log_scroll = self.log_scroll.saturating_add(if code == KeyCode::PageUp { 20 } else { 1 }),
            KeyCode::Down | KeyCode::PageDown if self.tab == 3 => self.log_scroll = self.log_scroll.saturating_sub(if code == KeyCode::PageDown { 20 } else { 1 }),
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
            Msg::Status(s) => self.status = s,
            Msg::Plan(p) => {
                self.pick = p["options"].as_array().and_then(|o| o.iter().position(|x| x["ctx"].as_f64() == p["context"].as_f64())).unwrap_or(0);
                self.plan = p;
            }
            Msg::Apps(a) => self.apps = a,
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
        Span::styled(format!("{rate:<12}"), Style::new().fg(Color::Blue)),
        mem,
    ]);
    f.render_widget(Paragraph::new(line).block(Block::bordered().title(" Strata ".bold())), bar);
    f.render_widget(Tabs::new(TABS.iter().enumerate().map(|(i, t)| format!("F{} {t}", i + 1))).select(app.tab).highlight_style(Style::new().reversed()), tabs);
    match app.tab {
        0 => draw_chat(f, app, body),
        1 => draw_monitor(f, app, body),
        2 => draw_memory(f, app, body),
        _ => draw_log(f, app, body),
    }
    let keys = if app.tab == 0 { "Enter send · Ctrl+S start/stop · Ctrl+R restart · Ctrl+L load/unload · Ctrl+N config · Tab view · Ctrl+Q quit" } else { "s start/stop · r restart · l load/unload · n config · Tab view · q quit" };
    let help_line = if app.note.is_empty() { Line::from(keys.dim()) } else { Line::from(vec![Span::styled(app.note.clone(), Style::new().fg(Color::Yellow)), Span::raw("   "), keys.dim()]) };
    f.render_widget(Paragraph::new(help_line), help);
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
        if let Some(e) = &t.error {
            lines.push(Line::from(Span::styled(format!("error {e}"), Style::new().fg(Color::Red))));
            raw.push_str(&format!("error {e}\n"));
        }
        lines.push(Line::raw(""));
        raw.push('\n');
    }
    if app.turns.is_empty() {
        lines.push(Line::from("Type a message and press Enter. The answer streams here; the model's thinking is shown dimmed.".dim()));
    }
    let inner_w = hist.width.saturating_sub(2);
    let scroll = height(&raw, inner_w).saturating_sub(hist.height.saturating_sub(2));
    f.render_widget(Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false }).scroll((scroll, 0)).block(Block::bordered().title(" Chat ")), hist);
    let title = if app.busy { " answering … " } else { " message " };
    f.render_widget(Paragraph::new(format!("{}▏", app.input)).block(Block::bordered().title(title)), input);
}

fn draw_monitor(f: &mut Frame, app: &App, area: Rect) {
    let m = &app.status["metrics"];
    if m.is_null() {
        f.render_widget(Paragraph::new("The server is not running, or not answering yet. Ctrl+S / s starts it.").block(Block::bordered().title(" Monitor ")), area);
        return;
    }
    let [top, gauges, reqs] = Layout::vertical([Constraint::Length(4), Constraint::Length(6), Constraint::Min(4)]).areas(area);
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
    let rows = Layout::vertical([Constraint::Length(1); 4]).split(Block::bordered().title(" Hardware ").inner(gauges));
    f.render_widget(Block::bordered().title(" Hardware "), gauges);
    let ratio = |a: &Value, b: &Value| (a.as_f64().unwrap_or(0.0) / b.as_f64().unwrap_or(1.0).max(1.0)).clamp(0.0, 1.0);
    let bars = [
        ("GPU memory", ratio(&hw["gpu_mem_used"], &hw["gpu_mem_total"]), format!("{} / {}", gb(hw["gpu_mem_used"].as_f64().unwrap_or(0.0)), gb(hw["gpu_mem_total"].as_f64().unwrap_or(0.0)))),
        ("GPU load", hw["gpu_util"].as_f64().unwrap_or(0.0) / 100.0, format!("{}%  {} °C  {} W", s(&hw["gpu_util"]), s(&hw["gpu_temp"]), s(&hw["gpu_power"]))),
        ("RAM", ratio(&hw["ram_used"], &hw["ram_total"]), format!("{} / {}", gb(hw["ram_used"].as_f64().unwrap_or(0.0)), gb(hw["ram_total"].as_f64().unwrap_or(0.0)))),
        ("CPU", hw["cpu"].as_f64().unwrap_or(0.0) / 100.0, format!("{}%", s(&hw["cpu"]))),
    ];
    for (i, (name, r, label)) in bars.into_iter().enumerate() {
        f.render_widget(Gauge::default().ratio(r.clamp(0.0, 1.0)).label(format!("{name}  {label}")).gauge_style(Style::new().fg(Color::Green)), rows[i]);
    }
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
            Line::from(format!("the GPU can use {} (1 GB kept for the display and the driver) · {} free right now", gb(n("limit")), gb(n("free")))),
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
