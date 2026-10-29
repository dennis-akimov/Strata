//! Strata.app: the server under one hand.  A control window (pick the model's run config and port, start / stop /
//! restart the server, load / unload the model, live status, the server's log) above the server's own page (Chat, Monitor, About), and a
//! menu-bar icon that shows the state (grey stopped, orange starting, green ready, blue generating).  The server is
//! the app's child: quitting the app stops it the server's own way (SIGTERM: QUIT to the engine, which frees its
//! memory), a stray Strata server on the port is replaced (--replace).

use serde_json::{json, Value};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::os::unix::process::CommandExt;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tao::event::{Event, WindowEvent};
use tao::event_loop::{ControlFlow, EventLoopBuilder, EventLoopProxy};
use tao::window::WindowBuilder;
use tray_icon::menu::{accelerator::{Accelerator, Code, Modifiers}, Menu, MenuEvent, MenuItem, PredefinedMenuItem, Submenu};
use tray_icon::{Icon, TrayIcon, TrayIconBuilder};
use wry::{WebView, WebViewBuilder};

const PANEL: &str = include_str!("panel.html");
const LOG_KEEP: usize = 2000;
const BAR: f64 = 44.0;
// in the server's page: the config's API key (handed over as ?strata-key=) goes where Chat and Monitor read it, and
// leaves the address before the page's own scripts run
const KEY_SCRIPT: &str = r#"{ const u = new URL(location.href), k = u.searchParams.get("strata-key"); if (k !== null) { try { localStorage.setItem("strata.apikey", JSON.stringify(k)); sessionStorage.setItem("strata.monitor.key", k); } catch (e) {} u.searchParams.delete("strata-key"); history.replaceState(null, "", u); } }"#;

enum Ev {
    Status(Value),
    Memory(Value),
    Alert(Value),
    Page(Value),
    Log(String),
    Ipc(Value),
    Menu(MenuEvent),
    Quit,
}

/// Strata's folder: $STRATA_ROOT, else the folder this app was built in.
fn root() -> PathBuf {
    match std::env::var("STRATA_ROOT") {
        Ok(r) if !r.is_empty() => r.into(),
        _ => Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf(),
    }
}

fn settings_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    Path::new(&home).join("Library/Application Support/Strata/app.json")
}

fn load_settings() -> Value {
    std::fs::read_to_string(settings_path()).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(json!({}))
}

fn save_settings(v: &Value) {
    let p = settings_path();
    let _ = std::fs::create_dir_all(p.parent().unwrap());
    let _ = std::fs::write(p, serde_json::to_string_pretty(v).unwrap_or_default());
}

/// The run configs setup wrote (strata-<model>.json) that name an engine, newest first; [`problem`] says which of
/// them cannot run yet.
fn configs(root: &Path) -> Vec<PathBuf> {
    let mut v: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(root)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            let n = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            n.starts_with("strata-") && n.ends_with(".json") && !n.contains(".shared-settings")
        })
        .filter(|p| read_config(p).get("exe").is_some())
        .map(|p| (std::fs::metadata(&p).and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH), p))
        .collect();
    v.sort_by(|a, b| b.0.cmp(&a.0));
    v.into_iter().map(|(_, p)| p).collect()
}

/// A config the server can run: it names an engine and every file it needs is on disk.
fn runnable(c: &Value) -> bool {
    c.get("exe").is_some() && problem(c).is_none()
}

/// Why a config cannot run yet, with the command that fixes it.
fn problem(c: &Value) -> Option<String> {
    let args: Vec<&str> = c["args"].as_array().map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
    let exe = c["exe"].as_str().unwrap_or("");
    if !Path::new(exe).is_file() {
        return Some(format!("the engine {} is missing: run make build", Path::new(exe).file_name().unwrap_or_default().to_string_lossy()));
    }
    let pull = |path: &str| {
        // the folder tools/pull_model.py downloads each model into
        let dir = Path::new(path).parent().and_then(|d| d.file_name()).map(|d| d.to_string_lossy().to_lowercase()).unwrap_or_default();
        let model = [("glm-5.3", "GLM-5.3"), ("gpt-oss-120b-heretic", "GPT-OSS-120B-HERETIC"), ("gpt-oss-120b-dflash", "GPT-OSS-120B-DFLASH"),
            ("gpt-oss-120b", "GPT-OSS-120B"), ("gpt-oss-20b", "GPT-OSS-20B"), ("qwen3.6-35b", "QWEN3.6-35B"), ("nucleus", "NUCLEUS"),
            ("hunyuan", "HUNYUAN"), ("wan2.2", "WAN2.2"), ("iq3_xxs", "IQ3_XXS"), ("iq3_s", "IQ3_S"), ("iq2_xs", "IQ2_XS"), ("q2_0", "Q2_0")]
            .iter()
            .find(|(d, _)| dir.starts_with(d))
            .map(|(_, m)| *m);
        match model {
            Some(m) => format!("the model is not downloaded: run make pull MODEL={m}"),
            None => format!("{path} is missing"),
        }
    };
    if let Some(w) = args.windows(2).find(|w| matches!(w[0], "--gguf" | "--model" | "--pack") && !Path::new(w[1]).exists()) {
        return Some(pull(w[1]));
    }
    match c["tokenizer"].as_str() {
        Some(t) if !Path::new(t).exists() => Some(format!("its tokenizer pack is missing ({t}): run make pull for this model again")),
        _ => None,
    }
}

// ---------------------------------------------------------------------------------------------- memory

/// Whether this engine binary knows --plan (an older one would take it as an unknown flag and load the whole model).
fn plans(exe: &Path) -> bool {
    std::fs::read(exe).map(|b| b.windows(6).any(|w| w == b"--plan")).unwrap_or(false)
}

fn mtime(p: &Path) -> std::time::SystemTime {
    std::fs::metadata(p).and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH)
}

/// The engine's own measure of what this config needs (`<exe> --plan <args>`: llama.cpp lays the model and its
/// buffers out without reading the weights, < 1 s), cached per config and engine file.
fn plan(cfg_path: &Path) -> Result<Value, String> {
    static CACHE: std::sync::LazyLock<Mutex<std::collections::HashMap<(PathBuf, std::time::SystemTime, std::time::SystemTime), Value>>> =
        std::sync::LazyLock::new(Default::default);
    let c = read_config(cfg_path);
    let exe = PathBuf::from(c["exe"].as_str().unwrap_or(""));
    let key = (cfg_path.to_path_buf(), mtime(cfg_path), mtime(&exe));
    if let Some(v) = CACHE.lock().unwrap().get(&key) {
        return Ok(v.clone());
    }
    if let Some(p) = problem(&c) {
        return Err(p);
    }
    if !plans(&exe) {
        return Err("this engine is older than its memory check (--plan): run make build".into());
    }
    let args: Vec<String> = c["args"].as_array().map(|a| a.iter().filter_map(Value::as_str).filter(|a| *a != "--serve").map(String::from).collect()).unwrap_or_default();
    let mut cmd = Command::new(&exe);
    cmd.arg("--plan").args(&args).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null());
    if let Some(cwd) = c["cwd"].as_str().filter(|d| Path::new(d).is_dir()) {
        cmd.current_dir(cwd);
    }
    let out = cmd.output().map_err(|e| format!("the engine could not be asked: {e}"))?;
    let text = String::from_utf8_lossy(&out.stdout);
    let v: Value = text.lines().find(|l| l.starts_with('{')).and_then(|l| serde_json::from_str(l).ok()).ok_or_else(|| {
        let err = text.lines().find(|l| l.starts_with("ERR")).unwrap_or("no answer").trim_start_matches("ERR ").to_string();
        format!("the engine could not measure this model: {err}")
    })?;
    CACHE.lock().unwrap().insert(key, v.clone());
    Ok(v)
}

/// Memory macOS can give a program now without swapping much: free, inactive (file cache, idle apps), speculative
/// and purgeable pages (vm_stat).
fn ram_available() -> u64 {
    let out = Command::new("/usr/bin/vm_stat").output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
    let page = out.split("page size of ").nth(1).and_then(|r| r.split_whitespace().next()).and_then(|n| n.parse::<u64>().ok()).unwrap_or(16384);
    let pages = |name: &str| out.lines().find(|l| l.starts_with(name)).and_then(|l| l.rsplit(':').next()).and_then(|n| n.trim().trim_end_matches('.').parse::<u64>().ok()).unwrap_or(0);
    (pages("Pages free") + pages("Pages inactive") + pages("Pages speculative") + pages("Pages purgeable")) * page
}

// up to 1M: GLM-5.3 is trained for 1,048,576; the menu stops at each model's own (gpt-oss 131,072, Qwen 262,144)
const CONTEXTS: [u64; 10] = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];

/// What a config needs against what this Mac has: the GPU's share must fit Metal's working set, and everything must
/// fit the memory free now (when the server is not running: a running engine already holds its share).  For each
/// context size the need follows the engine's per-token cost; a smaller --prefill-batch shrinks the work buffers
/// (their size is the batch x the vocabulary's logits).
fn memory(cfg_path: &Path, running: bool) -> Value {
    let p = match plan(cfg_path) {
        Ok(p) => p,
        Err(e) => return json!({"config": cfg_path, "error": e}),
    };
    let n = |v: &Value| v.as_f64().unwrap_or(0.0);
    let (g, h) = (&p["gpu"], &p["host"]);
    let ctx = n(&p["context"]);
    let per_token = n(&p["per_token"]);
    let limit = n(&g["limit"]);
    let gpu_need = n(&g["model"]) + n(&g["context"]) + n(&g["compute"]);
    let need = gpu_need + n(&h["model"]) + n(&h["context"]) + n(&h["compute"]);
    let pb = n(&p["prefill_batch"]).max(1.0);
    // the work buffers at a batch of 512: they scale with the batch (measured 1702 -> 424 MiB, 2048 -> 512)
    let saved_512 = if pb > 512.0 { (n(&g["compute"]) + n(&h["compute"])) * (1.0 - 512.0 / pb) } else { 0.0 };
    let avail = if running { f64::INFINITY } else { ram_available() as f64 };
    let train = n(&p["n_ctx_train"]);
    let mut sizes: Vec<u64> = CONTEXTS.iter().copied().filter(|c| train <= 0.0 || (*c as f64) <= train).collect();
    if !sizes.contains(&(ctx as u64)) {
        sizes.push(ctx as u64);
    }
    // the largest context that fits exactly, in 16K steps, when it falls between two of the steps above (GLM: ~384K)
    if per_token > 0.0 {
        let room = (limit - gpu_need).min(avail - need);
        let exact = ((ctx + room / per_token).min(if train > 0.0 { train } else { f64::MAX }) as u64) / 16384 * 16384;
        if exact > 16384 && !sizes.contains(&exact) {
            sizes.push(exact);
        }
    }
    sizes.sort();
    let fits = |gpu: f64, total: f64| gpu <= limit && total <= avail;
    let mut options: Vec<Value> = sizes
        .iter()
        .map(|&c| {
            let d = per_token * (c as f64 - ctx);
            let (gpu, total) = (gpu_need + d, need + d);
            let ok = fits(gpu, total);
            let ok512 = !ok && saved_512 > 0.0 && fits(gpu - saved_512, total - saved_512);
            json!({"ctx": c, "need": if ok512 { total - saved_512 } else { total }, "fits": ok || ok512, "batch512": ok512})
        })
        .collect();
    let best = options.iter().rposition(|o| o["fits"] == true);
    if let Some(i) = best {
        options[i]["recommended"] = json!(true);
    }
    let state = if gpu_need > limit {
        "red"
    } else if need > avail {
        "amber"
    } else {
        "green"
    };
    let gb = |b: f64| format!("{:.1} GB", b / 1e9);
    let advice = match state {
        "green" if running => "running".to_string(),
        "green" => "fits".to_string(),
        "amber" => format!("fits the GPU, but only {} is free now: close other apps to free {}", gb(avail), gb(need - avail)),
        _ => match best {
            Some(i) => format!("too big for this Mac at this context (the GPU can use {}): use a context of {}{}", gb(limit), options[i]["ctx"], if options[i]["batch512"] == true { " with a prefill batch of 512" } else { "" }),
            None if n(&g["model"]) > limit => format!("the model alone ({}) is more than this Mac's GPU can use ({}): pick a smaller model", gb(n(&g["model"])), gb(limit)),
            None => "no context size fits: close other apps or pick a smaller model".into(),
        },
    };
    json!({"config": cfg_path, "context": ctx, "n_ctx_train": train, "prefill_batch": pb, "need": need, "gpu_need": gpu_need, "limit": limit,
        "avail": if running { Value::Null } else { json!(avail) }, "running": running,
        "weights": n(&g["model"]) + n(&h["model"]), "cache": n(&g["context"]) + n(&h["context"]), "work": n(&g["compute"]) + n(&h["compute"]),
        "state": state, "advice": advice, "options": options, "recommended": best.map(|i| options[i].clone())})
}

/// Rewrite a config's --max-context (and --prefill-batch), keeping its key order and indentation.
fn set_context(cfg_path: &Path, ctx: u64, batch: Option<u64>) -> Result<(), String> {
    let text = std::fs::read_to_string(cfg_path).map_err(|e| e.to_string())?;
    let mut v: Value = serde_json::from_str(text.trim_start_matches('\u{feff}')).map_err(|e| e.to_string())?;
    let args = v["args"].as_array_mut().ok_or("the config has no \"args\"")?;
    for (flag, val) in [("--max-context", Some(ctx)), ("--prefill-batch", batch)] {
        let Some(val) = val else { continue };
        match args.iter().position(|a| a == flag) {
            Some(i) if i + 1 < args.len() => args[i + 1] = json!(val.to_string()),
            _ => args.extend([json!(flag), json!(val.to_string())]),
        }
    }
    let indent = text.lines().nth(1).map(|l| l.len() - l.trim_start().len()).filter(|n| *n > 0).unwrap_or(1);
    let mut out = vec![];
    let fmt = serde_json::ser::PrettyFormatter::with_indent(&b"        "[..indent.min(8)]);
    let mut ser = serde_json::Serializer::with_formatter(&mut out, fmt);
    serde::Serialize::serialize(&v, &mut ser).map_err(|e| e.to_string())?;
    out.push(b'\n');
    std::fs::write(cfg_path, out).map_err(|e| e.to_string())
}

fn config_port(c: &Path) -> u16 {
    read_config(c)["port"].as_u64().map(|p| p as u16).unwrap_or(8090)
}

fn read_config(p: &Path) -> Value {
    std::fs::read_to_string(p).ok().and_then(|s| serde_json::from_str(s.trim_start_matches('\u{feff}')).ok()).unwrap_or(json!({}))
}

struct State {
    root: PathBuf,
    config: Option<PathBuf>,
    port: u16,
    child: Option<Child>,
    phase: &'static str,
    stopping_since: Option<Instant>,
    log: VecDeque<String>,
    last: Value,
}

impl State {
    fn api_key(&self) -> String {
        let cfg = self.config.as_deref().map(read_config).unwrap_or(json!({}));
        match &cfg["api_key"] {
            Value::String(s) => s.split(',').next().unwrap_or("").trim().to_string(),
            Value::Array(a) => a.first().and_then(Value::as_str).unwrap_or("").to_string(),
            _ => String::new(),
        }
    }

    fn base(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    fn push_log(&mut self, line: String) {
        if self.log.len() == LOG_KEEP {
            self.log.pop_front();
        }
        self.log.push_back(line);
    }
}

type Shared = Arc<Mutex<State>>;

fn server_exe(root: &Path) -> PathBuf {
    root.join("engine-metal/strata-server")
}

/// Start the server with the chosen config; its output goes to the log view.
fn start(st: &Shared, proxy: &EventLoopProxy<Ev>) -> Result<(), String> {
    let mut s = st.lock().unwrap();
    if s.child.is_some() {
        return Err("the server is already running".into());
    }
    let cfg = s.config.clone().ok_or("pick a model config first")?;
    let exe = server_exe(&s.root);
    if !exe.is_file() {
        return Err(format!("{} is missing: build it with `make server` in {}", exe.display(), s.root.display()));
    }
    let mut cmd = Command::new(&exe);
    cmd
        .args(["--engine", "strata", "--config"])
        .arg(&cfg)
        .args(["--port", &s.port.to_string(), "--replace"])
        .current_dir(&s.root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // the app blocks SIGTERM/SIGINT for its waiter thread and a spawned child keeps that mask: unblock them, or the
    // server never hears its SIGTERM (stop = 120 s, then SIGKILL)
    unsafe {
        cmd.pre_exec(|| {
            let mut set: libc::sigset_t = std::mem::zeroed();
            libc::sigemptyset(&mut set);
            libc::pthread_sigmask(libc::SIG_SETMASK, &set, std::ptr::null_mut());
            Ok(())
        });
    }
    let mut child = cmd.spawn().map_err(|e| format!("the server could not start: {e}"))?;
    for out in [child.stdout.take().map(|o| Box::new(o) as Box<dyn std::io::Read + Send>), child.stderr.take().map(|e| Box::new(e) as Box<dyn std::io::Read + Send>)].into_iter().flatten() {
        let p = proxy.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(out).lines().map_while(Result::ok) {
                if p.send_event(Ev::Log(line)).is_err() {
                    break;
                }
            }
        });
    }
    let port = s.port;
    s.push_log(format!("── started {} on port {} ──", cfg.file_name().unwrap_or_default().to_string_lossy(), port));
    s.child = Some(child);
    s.phase = "starting";
    s.stopping_since = None;
    Ok(())
}

/// The server's own way to stop: SIGTERM (it sends QUIT to the engine, which frees its memory); SIGKILL after 120 s.
fn stop(st: &Shared) {
    let mut s = st.lock().unwrap();
    if let Some(c) = s.child.as_ref() {
        unsafe { libc::kill(c.id() as i32, libc::SIGTERM) };
        s.phase = "stopping";
        s.stopping_since = Some(Instant::now());
        s.push_log("── stopping (the engine frees its memory first) ──".into());
    }
}

/// Wait until the server has exited (quitting the app).
fn stop_and_wait(st: &Shared) {
    stop(st);
    let t = Instant::now();
    loop {
        {
            let mut s = st.lock().unwrap();
            match s.child.as_mut().map(|c| c.try_wait()) {
                None | Some(Ok(Some(_))) | Some(Err(_)) => {
                    s.child = None;
                    return;
                }
                Some(Ok(None)) if t.elapsed() > Duration::from_secs(120) => {
                    let _ = s.child.as_mut().unwrap().kill();
                }
                _ => {}
            }
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(3))).http_status_as_error(false).build().into()
}

fn get_json(url: &str, key: &str) -> Option<Value> {
    let mut r = agent().get(url);
    if !key.is_empty() {
        r = r.header("Authorization", &format!("Bearer {key}"));
    }
    let mut resp = r.call().ok()?;
    if resp.status() != 200 {
        return None;
    }
    serde_json::from_str(&resp.body_mut().read_to_string().ok()?).ok()
}

/// POST /load or /unload as the server's own page would (JSON, no foreign Origin), with the config's key.
fn control(st: &Shared, path: &str) -> String {
    let (base, key) = {
        let s = st.lock().unwrap();
        (s.base(), s.api_key())
    };
    let a: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(600))).http_status_as_error(false).build().into();
    let mut r = a.post(&format!("{base}{path}")).header("Content-Type", "application/json");
    if !key.is_empty() {
        r = r.header("Authorization", &format!("Bearer {key}"));
    }
    match r.send("{}") {
        Ok(mut resp) => resp.body_mut().read_to_string().unwrap_or_default(),
        Err(e) => format!("{{\"error\": \"{e}\"}}"),
    }
}

/// Every 1.5 s: the process, /health, /metrics -> one status for the window and the menu bar.
fn poll(st: Shared, proxy: EventLoopProxy<Ev>) {
    loop {
        std::thread::sleep(Duration::from_millis(1500));
        let (running, base, key, exited) = {
            let mut s = st.lock().unwrap();
            let mut exited = None;
            if let Some(c) = s.child.as_mut() {
                if let Ok(Some(code)) = c.try_wait() {
                    exited = Some(code.code());
                }
            }
            if let Some(code) = exited {
                s.child = None;
                s.phase = "stopped";
                s.stopping_since = None;
                s.push_log(format!("── the server exited ({}) ──", code.map(|c| c.to_string()).unwrap_or_else(|| "signal".into())));
            }
            if let (Some(t), Some(c)) = (s.stopping_since, s.child.as_mut()) {
                if t.elapsed() > Duration::from_secs(120) {
                    let _ = c.kill();
                }
            }
            (s.child.is_some(), s.base(), s.api_key(), exited.is_some())
        };
        let mut status = json!({"running": running, "exited": exited});
        if running {
            let health = get_json(&format!("{base}/health"), "");
            let metrics = health.as_ref().and_then(|_| get_json(&format!("{base}/metrics"), &key));
            if let Some(h) = &health {
                status["model"] = h["model"].clone();
                status["loaded"] = h["loaded"].clone();
                status["max_context"] = h["max_context"].clone();
            }
            if let Some(m) = &metrics {
                let (l, hw) = (&m["live"], &m["hardware"]);
                for k in ["state", "tok_s", "prompt_read", "prompt_total", "generated", "max_tokens", "queued", "phase", "elapsed_s"] {
                    status[k] = l[k].clone();
                }
                for k in ["gpu_mem_used", "gpu_mem_total", "gpu_temp", "gpu_power", "gpu_util", "cpu", "ram_used", "ram_total"] {
                    status[k] = hw[k].clone();
                }
                status["requests"] = m["totals"]["requests"].clone();
                status["last"] = m["requests"].get(0).cloned().unwrap_or(Value::Null);
            }
            let phase = {
                let mut s = st.lock().unwrap();
                if s.phase != "stopping" {
                    s.phase = match (&health, status["state"].as_str()) {
                        (None, _) => "starting",
                        (Some(_), Some("reading" | "generating")) => "busy",
                        (Some(_), Some("unloaded")) => "unloaded",
                        (Some(_), _) => "ready",
                    };
                }
                s.last = status.clone();
                s.phase
            };
            status["phase"] = json!(phase);
        } else {
            let mut s = st.lock().unwrap();
            if s.phase != "stopped" {
                s.phase = "stopped";
            }
            status["phase"] = json!("stopped");
        }
        if proxy.send_event(Ev::Status(status)).is_err() {
            return;
        }
    }
}

/// A round dot for the menu bar, coloured by state.
fn dot(rgb: (u8, u8, u8)) -> Icon {
    let n = 36u32;
    let mut px = vec![0u8; (n * n * 4) as usize];
    let c = (n as f32 - 1.0) / 2.0;
    for y in 0..n {
        for x in 0..n {
            let d = ((x as f32 - c).powi(2) + (y as f32 - c).powi(2)).sqrt();
            let a = (c - 2.0 - d).clamp(0.0, 1.0); // antialiased edge
            let i = ((y * n + x) * 4) as usize;
            px[i..i + 4].copy_from_slice(&[rgb.0, rgb.1, rgb.2, (a * 255.0) as u8]);
        }
    }
    Icon::from_rgba(px, n, n).expect("icon")
}

fn phase_color(phase: &str) -> (u8, u8, u8) {
    match phase {
        "ready" => (52, 199, 89),
        "busy" => (10, 132, 255),
        "starting" | "stopping" => (255, 159, 10),
        "unloaded" => (191, 90, 242),
        _ => (142, 142, 147),
    }
}

fn js_call(wv: &WebView, f: &str, v: &Value) {
    let _ = wv.evaluate_script(&format!("{f}({v})"));
}

fn snapshot(st: &Shared) -> Value {
    let s = st.lock().unwrap();
    let list: Vec<Value> = configs(&s.root)
        .iter()
        .map(|p| {
            let c = read_config(p);
            json!({"path": p, "name": p.file_name().unwrap_or_default().to_string_lossy(), "model": c["model_name"], "port": c["port"], "problem": problem(&c)})
        })
        .collect();
    let settings = load_settings();
    json!({"root": s.root, "configs": list, "config": s.config, "port": s.port, "phase": s.phase,
        "running": s.child.is_some(), "log": s.log.iter().cloned().collect::<Vec<_>>(), "status": s.last,
        "key": s.api_key(), "autostart": settings["autostart"].as_bool().unwrap_or(false), "server": server_exe(&s.root).is_file()})
}

/// The memory check of the selected config, off the UI thread; `alert`: also say it as a problem (a chat failed).
fn check_memory(st: &Shared, proxy: &EventLoopProxy<Ev>, alert: Option<String>) {
    let (cfg, running) = {
        let s = st.lock().unwrap();
        (s.config.clone(), s.child.is_some())
    };
    let Some(cfg) = cfg else { return };
    let p = proxy.clone();
    std::thread::spawn(move || {
        let m = memory(&cfg, running && alert.is_none());
        let _ = p.send_event(match alert {
            Some(why) => Ev::Alert(json!({"why": why, "memory": m})),
            None => Ev::Memory(m),
        });
    });
}

/// A line of the server's output that means the model did not fit: Metal ran out of memory, or the engine ended
/// while loading.
fn trouble(line: &str) -> Option<&'static str> {
    if line.contains("ran out of memory") || line.contains("Insufficient Memory") || line.contains("llama_process failed (-3)") {
        Some("The GPU ran out of memory while answering.")
    } else if line.contains("the engine exited before it was ready") {
        Some("The engine stopped while loading the model.")
    } else {
        None
    }
}

fn rect(x: f64, y: f64, w: f64, h: f64) -> wry::Rect {
    wry::Rect { position: wry::dpi::LogicalPosition::new(x, y).into(), size: wry::dpi::LogicalSize::new(w, h).into() }
}

fn query_escape(s: &str) -> String {
    s.bytes().map(|b| if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) { (b as char).to_string() } else { format!("%{b:02X}") }).collect()
}

fn open(args: &[&str]) {
    let _ = Command::new("open").args(args).spawn();
}

fn main() {
    let root = root();
    let settings = load_settings();
    let config = settings["config"].as_str().map(PathBuf::from).filter(|p| p.is_file()).or_else(|| configs(&root).into_iter().next());
    let config = config.filter(|c| runnable(&read_config(c))).or_else(|| configs(&root).into_iter().next());
    let port = config.as_deref().map(config_port).unwrap_or(8090);
    let st: Shared = Arc::new(Mutex::new(State { root, config, port, child: None, phase: "stopped", stopping_since: None, log: VecDeque::new(), last: json!({}) }));

    // SIGTERM / Ctrl+C on the app = Quit (the server is stopped its own way, not orphaned); blocked before any thread
    // starts so only the waiter gets them (a spawned server starts with an empty mask)
    let sigs = unsafe {
        let mut set: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut set);
        libc::sigaddset(&mut set, libc::SIGTERM);
        libc::sigaddset(&mut set, libc::SIGINT);
        libc::pthread_sigmask(libc::SIG_BLOCK, &set, std::ptr::null_mut());
        set
    };
    let event_loop = EventLoopBuilder::<Ev>::with_user_event().build();
    let proxy = event_loop.create_proxy();
    std::thread::spawn({
        let p = proxy.clone();
        move || {
            let mut sig = 0;
            unsafe { libc::sigwait(&sigs, &mut sig) };
            let _ = p.send_event(Ev::Quit);
        }
    });
    {
        let p = proxy.clone();
        MenuEvent::set_event_handler(Some(move |e| {
            let _ = p.send_event(Ev::Menu(e));
        }));
    }

    // the app's menu: Strata (About, Quit = stop the server first), Edit (copy and paste in the windows), Window
    let app_menu = Menu::new();
    let quit = MenuItem::new("Quit Strata", true, Some(Accelerator::new(Modifiers::META, Code::KeyQ)));
    let close_win = MenuItem::new("Close Window", true, Some(Accelerator::new(Modifiers::META, Code::KeyW)));
    let strata = Submenu::with_items("Strata", true, &[&PredefinedMenuItem::about(None, None), &PredefinedMenuItem::separator(), &PredefinedMenuItem::hide(None), &PredefinedMenuItem::separator(), &quit]).unwrap();
    let edit = Submenu::with_items("Edit", true, &[&PredefinedMenuItem::undo(None), &PredefinedMenuItem::redo(None), &PredefinedMenuItem::separator(), &PredefinedMenuItem::cut(None), &PredefinedMenuItem::copy(None), &PredefinedMenuItem::paste(None), &PredefinedMenuItem::select_all(None)]).unwrap();
    let window_menu = Submenu::with_items("Window", true, &[&PredefinedMenuItem::minimize(None), &close_win]).unwrap();
    app_menu.append_items(&[&strata, &edit, &window_menu]).unwrap();
    app_menu.init_for_nsapp();

    let control_win = WindowBuilder::new().with_title("Strata").with_inner_size(tao::dpi::LogicalSize::new(1280.0, 900.0)).build(&event_loop).expect("window");
    // two views in the window: the control bar, and the server's own page (Chat, Monitor, Firewall, About) under it.
    // The bar covers the whole window while the server is down, its log is open or there is a problem to show; the
    // page is hidden then.  The page is made last, so it is the top view whenever it shows: it never depends on the
    // bar view keeping to its 44 px.
    let full = std::cell::Cell::new(true);
    let control_view = {
        let p = proxy.clone();
        WebViewBuilder::new()
            .with_html(PANEL)
            .with_bounds(rect(0.0, 0.0, 1280.0, 900.0))
            .with_ipc_handler(move |req| {
                if let Ok(v) = serde_json::from_str::<Value>(req.body()) {
                    let _ = p.send_event(Ev::Ipc(v));
                }
            })
            .build_as_child(&control_win)
            .expect("webview")
    };
    let page = {
        let p = proxy.clone();
        WebViewBuilder::new()
            .with_initialization_script(KEY_SCRIPT)
            .with_bounds(rect(0.0, BAR, 1280.0, 900.0 - BAR))
            .with_visible(false)
            .with_on_page_load_handler(move |e, url| {
                let event = match e {
                    wry::PageLoadEvent::Started => "started",
                    wry::PageLoadEvent::Finished => "finished",
                };
                let _ = p.send_event(Ev::Page(json!({"event": event, "url": url})));
            })
            .build_as_child(&control_win)
            .expect("webview")
    };
    // the page's loads: a load that never finishes (10 s) or finishes empty is said, with Open in browser and Retry
    let (navs, loaded) = (Arc::new(std::sync::atomic::AtomicU64::new(0)), Arc::new(std::sync::atomic::AtomicU64::new(0)));

    // the menu bar
    let m_show = MenuItem::new("Show Strata", true, None);
    let m_chat = MenuItem::new("Open Chat", true, None);
    let m_start = MenuItem::new("Start Server", true, None);
    let m_stop = MenuItem::new("Stop Server", true, None);
    let m_quit = MenuItem::new("Quit Strata", true, None);
    let mut tray_menu = Some(Menu::with_items(&[&m_show, &m_chat, &PredefinedMenuItem::separator(), &m_start, &m_stop, &PredefinedMenuItem::separator(), &m_quit]).unwrap());
    let mut tray: Option<TrayIcon> = None;
    let mut tray_phase = String::new();

    std::thread::spawn({
        let (st, p) = (st.clone(), proxy.clone());
        move || poll(st, p)
    });
    if settings["autostart"].as_bool().unwrap_or(false) {
        let _ = start(&st, &proxy);
    }

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        let layout = |full: bool| {
            let sz = control_win.inner_size().to_logical::<f64>(control_win.scale_factor());
            let _ = control_view.set_bounds(rect(0.0, 0.0, sz.width, if full { sz.height } else { BAR }));
            let _ = page.set_bounds(rect(0.0, BAR, sz.width, (sz.height - BAR).max(0.0)));
            let _ = page.set_visible(!full);
        };
        let act = |cmd: &str, v: &Value| {
            let note = |msg: &str| js_call(&control_view, "note", &json!(msg));
            match cmd {
                "init" => {
                    js_call(&control_view, "init", &snapshot(&st));
                    check_memory(&st, &proxy, None);
                }
                "memory" => check_memory(&st, &proxy, None),
                "set_context" => {
                    // a new context (and prefill batch) in the config; a running server restarts with it
                    let Some(cfg) = st.lock().unwrap().config.clone() else { return };
                    let (ctx, batch) = (v["ctx"].as_u64().unwrap_or(0).max(2048), v["batch512"].as_bool().unwrap_or(false).then_some(512));
                    let running = st.lock().unwrap().child.is_some();
                    let start_after = running || v["start"].as_bool().unwrap_or(false);
                    let (st2, p2) = (st.clone(), proxy.clone());
                    std::thread::spawn(move || {
                        if running {
                            stop_and_wait(&st2);
                        }
                        let msg = match set_context(&cfg, ctx, batch) {
                            Ok(()) => format!("── context set to {ctx}{} in {} ──", batch.map(|b| format!(", prefill batch {b}")).unwrap_or_default(), cfg.file_name().unwrap_or_default().to_string_lossy()),
                            Err(e) => format!("── the config could not be changed: {e} ──"),
                        };
                        let _ = p2.send_event(Ev::Log(msg));
                        if start_after {
                            if let Err(e) = start(&st2, &p2) {
                                let _ = p2.send_event(Ev::Log(format!("── {e} ──")));
                            }
                        }
                        check_memory(&st2, &p2, None);
                    });
                }
                "select" => {
                    let mut s = st.lock().unwrap();
                    if let Some(c) = v["config"].as_str() {
                        s.config = Some(PathBuf::from(c));
                        s.port = config_port(Path::new(c));
                    }
                    let mut set = load_settings();
                    set["config"] = json!(s.config);
                    save_settings(&set);
                    drop(s);
                    js_call(&control_view, "init", &snapshot(&st));
                    check_memory(&st, &proxy, None);
                }
                "autostart" => {
                    let mut set = load_settings();
                    set["autostart"] = json!(v["on"].as_bool().unwrap_or(false));
                    save_settings(&set);
                }
                "start" => {
                    if let Err(e) = start(&st, &proxy) {
                        note(&e);
                    }
                }
                "stop" => stop(&st),
                "restart" => {
                    let (st2, p2) = (st.clone(), proxy.clone());
                    std::thread::spawn(move || {
                        stop_and_wait(&st2);
                        if let Err(e) = start(&st2, &p2) {
                            let _ = p2.send_event(Ev::Log(format!("── {e} ──")));
                        }
                    });
                }
                "load" | "unload" => {
                    let (st2, p2, path) = (st.clone(), proxy.clone(), format!("/{cmd}"));
                    std::thread::spawn(move || {
                        let r = control(&st2, &path);
                        let _ = p2.send_event(Ev::Log(format!("── {path}: {r} ──")));
                    });
                }
                "layout" => {
                    full.set(v["full"].as_bool().unwrap_or(true));
                    layout(full.get());
                }
                "page" => {
                    // the server's page, signed in with the config's key
                    let (base, key) = {
                        let s = st.lock().unwrap();
                        (s.base(), s.api_key())
                    };
                    let url = format!("{base}/?strata-key={}#{}", query_escape(&key), v["tab"].as_str().unwrap_or("monitor"));
                    let n = navs.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                    if let Err(e) = page.load_url(&url) {
                        let _ = proxy.send_event(Ev::Page(json!({"event": "failed", "url": base, "why": e.to_string()})));
                    }
                    let (p2, loaded2) = (proxy.clone(), loaded.clone());
                    std::thread::spawn(move || {
                        std::thread::sleep(Duration::from_secs(10));
                        if loaded2.load(std::sync::atomic::Ordering::SeqCst) < n {
                            let _ = p2.send_event(Ev::Page(json!({"event": "timeout", "url": base})));
                        }
                    });
                }
                "chat" => {
                    control_win.set_visible(true);
                    control_win.set_focus();
                    let _ = page.evaluate_script("location.hash = '#chat'");
                }
                "browser" => open(&[&format!("{}/", st.lock().unwrap().base())]),
                "edit_config" => {
                    if let Some(c) = st.lock().unwrap().config.clone() {
                        open(&["-t", &c.to_string_lossy()]);
                    }
                }
                "reveal" => {
                    if let Some(c) = st.lock().unwrap().config.clone() {
                        open(&["-R", &c.to_string_lossy()]);
                    }
                }
                "engine_log" => {
                    let log = st.lock().unwrap().config.as_deref().map(read_config).and_then(|c| c["log"].as_str().map(String::from));
                    match log {
                        Some(l) => open(&["-a", "Console", &l]),
                        None => note("this config writes no engine log (no \"log\" key)"),
                    }
                }
                "copy_key" => {
                    let key = st.lock().unwrap().api_key();
                    if let Ok(mut c) = Command::new("pbcopy").stdin(Stdio::piped()).spawn() {
                        use std::io::Write;
                        let _ = c.stdin.take().map(|mut i| i.write_all(key.as_bytes()));
                        let _ = c.wait();
                        note(if key.is_empty() { "this config has no API key" } else { "API key copied" });
                    }
                }
                _ => {}
            }
        };
        match event {
            Event::NewEvents(tao::event::StartCause::Init) => {
                // the menu-bar icon is made once the app has started (macOS wants the run loop first)
                tray = TrayIconBuilder::new().with_menu(Box::new(tray_menu.take().unwrap())).with_icon(dot(phase_color("stopped"))).with_tooltip("Strata: stopped").build().ok();
                control_win.set_focus();
            }
            Event::UserEvent(Ev::Ipc(v)) => {
                let cmd = v["cmd"].as_str().unwrap_or("").to_string();
                act(&cmd, &v);
            }
            Event::UserEvent(Ev::Log(line)) => {
                st.lock().unwrap().push_log(line.clone());
                js_call(&control_view, "log", &json!(line));
                if let Some(why) = trouble(&line) {
                    check_memory(&st, &proxy, Some(why.into()));
                }
            }
            Event::UserEvent(Ev::Memory(m)) => js_call(&control_view, "memory", &m),
            Event::UserEvent(Ev::Page(e)) => {
                if e["event"] == "finished" {
                    loaded.store(navs.load(std::sync::atomic::Ordering::SeqCst), std::sync::atomic::Ordering::SeqCst);
                    layout(full.get()); // the bounds again, now that the page has content (a first layout can come too early)
                    // a page that loaded but shows nothing (a blocked or empty answer) counts as not loaded
                    let p = proxy.clone();
                    let _ = page.evaluate_script_with_callback("(document.body && document.body.innerText.trim().length) || 0", move |chars| {
                        let _ = p.send_event(Ev::Page(json!({"event": "checked", "chars": chars.trim().parse::<u64>().unwrap_or(0)})));
                    });
                }
                js_call(&control_view, "pageEvent", &e);
            }
            Event::UserEvent(Ev::Alert(a)) => js_call(&control_view, "trouble", &a),
            Event::UserEvent(Ev::Status(s)) => {
                let phase = s["phase"].as_str().unwrap_or("stopped").to_string();
                js_call(&control_view, "status", &s);
                if let Some(t) = &tray {
                    if phase != tray_phase {
                        let _ = t.set_icon(Some(dot(phase_color(&phase))));
                        let _ = t.set_tooltip(Some(format!("Strata: {phase}{}", s["model"].as_str().map(|m| format!(" - {m}")).unwrap_or_default())));
                        tray_phase = phase.clone();
                    }
                    let rate = s["tok_s"].as_f64().filter(|r| *r > 0.0 && phase == "busy");
                    t.set_title(rate.map(|r| format!("{r:.0} t/s")));
                }
                let running = s["running"].as_bool().unwrap_or(false);
                m_start.set_enabled(!running);
                m_stop.set_enabled(running);
            }
            Event::UserEvent(Ev::Menu(e)) => {
                let id = e.id();
                if id == m_show.id() {
                    control_win.set_visible(true);
                    control_win.set_focus();
                } else if id == m_chat.id() {
                    act("chat", &json!({}));
                } else if id == m_start.id() {
                    act("start", &json!({}));
                } else if id == m_stop.id() {
                    act("stop", &json!({}));
                } else if id == m_quit.id() || id == quit.id() {
                    let _ = proxy.send_event(Ev::Quit);
                } else if id == close_win.id() {
                    control_win.set_visible(false);
                }
            }
            Event::UserEvent(Ev::Quit) => {
                if let Some(t) = &tray {
                    t.set_title(Some("stopping…"));
                }
                control_win.set_visible(false);
                stop_and_wait(&st); // the engine's memory is free before the app is gone
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent { event: WindowEvent::Resized(_), .. } => layout(full.get()),
            Event::WindowEvent { window_id, event: WindowEvent::CloseRequested, .. } => {
                // closing a window hides it: the app stays in the menu bar with the server running
                if window_id == control_win.id() {
                    control_win.set_visible(false);
                }
            }
            _ => {}
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrites_context_keeping_the_config() {
        let p = std::env::temp_dir().join(format!("strata-app-test-{}.json", std::process::id()));
        std::fs::write(&p, "{\n \"exe\": \"x\",\n \"args\": [\"--gguf\", \"m.gguf\", \"--max-context\", \"131072\", \"--kv\", \"f16\"],\n \"port\": 8090\n}\n").unwrap();
        set_context(&p, 8192, Some(512)).unwrap();
        let text = std::fs::read_to_string(&p).unwrap();
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["args"], json!(["--gguf", "m.gguf", "--max-context", "8192", "--kv", "f16", "--prefill-batch", "512"]));
        assert!(text.starts_with("{\n \"exe\""), "key order and 1-space indent kept: {text}");
        set_context(&p, 4096, Some(256)).unwrap(); // a second change replaces, never appends twice
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
        assert_eq!(v["args"].as_array().unwrap().iter().filter(|a| *a == "--prefill-batch").count(), 1);
        assert_eq!(v["args"][3], "4096");
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn spots_out_of_memory_lines() {
        assert!(trouble("[strata] the GPU ran out of memory (Metal) at a context of 8192 tokens: ...").is_some());
        assert!(trouble("error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)").is_some());
        assert!(trouble("[strata] 2026-10-29 10:20:54 the engine exited before it was ready").is_some());
        assert!(trouble("[strata] 200 POST /v1/chat/completions").is_none());
        assert!(problem(&json!({"exe": "/nope/strata-metal", "args": []})).unwrap().contains("make build"));
    }
}
