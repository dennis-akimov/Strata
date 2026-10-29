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

/// The run configs setup wrote (strata-<model>.json), newest first.
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
        .filter(|p| runnable(&read_config(p)))
        .map(|p| (std::fs::metadata(&p).and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH), p))
        .collect();
    v.sort_by(|a, b| b.0.cmp(&a.0));
    v.into_iter().map(|(_, p)| p).collect()
}

/// A config the server can run: it names an engine and every model file it points at is on disk.
fn runnable(c: &Value) -> bool {
    let args: Vec<&str> = c["args"].as_array().map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
    c.get("exe").is_some() && args.windows(2).filter(|w| matches!(w[0], "--gguf" | "--model" | "--pack")).all(|w| Path::new(w[1]).exists())
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
            json!({"path": p, "name": p.file_name().unwrap_or_default().to_string_lossy(), "model": c["model_name"], "port": c["port"]})
        })
        .collect();
    let settings = load_settings();
    json!({"root": s.root, "configs": list, "config": s.config, "port": s.port, "phase": s.phase,
        "running": s.child.is_some(), "log": s.log.iter().cloned().collect::<Vec<_>>(), "status": s.last,
        "key": s.api_key(), "autostart": settings["autostart"].as_bool().unwrap_or(false), "server": server_exe(&s.root).is_file()})
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
    // two views in the window: the server's own page (Chat, Monitor, About) under the control bar; the bar covers the
    // whole window while the server is down or its log is open
    let page = WebViewBuilder::new().with_initialization_script(KEY_SCRIPT).with_bounds(rect(0.0, BAR, 1280.0, 900.0 - BAR)).with_visible(false).build_as_child(&control_win).expect("webview");
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
                "init" => js_call(&control_view, "init", &snapshot(&st)),
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
                    let _ = page.load_url(&format!("{base}/?strata-key={}#{}", query_escape(&key), v["tab"].as_str().unwrap_or("monitor")));
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
            }
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
