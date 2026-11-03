//! Strata.app: the server under one hand.  A control window (pick the model's run config and port, start / stop /
//! restart the server, load / unload the model, live status, the server's log) above the server's own page (Chat, Monitor, About), and a
//! menu-bar icon that shows the state (grey stopped, orange starting, green ready, blue generating).  The server is
//! the app's child: quitting the app stops it the server's own way (SIGTERM: QUIT to the engine, which frees its
//! memory), a stray Strata server on the port is replaced (--replace).

use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
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

mod browser_mcp;
mod deploy;
mod google;
mod mapfile;
mod native_input;
mod table;

const PANEL: &str = include_str!("panel.html");
const LOG_KEEP: usize = 2000;
const BAR: f64 = 44.0;
// in the server's page: the config's API key (handed over as ?strata-key=) goes where Chat and Monitor read it, and
// leaves the address before the page's own scripts run
const KEY_SCRIPT: &str = r#"{ const u = new URL(location.href), k = u.searchParams.get("strata-key"); if (k !== null) { try { localStorage.setItem("strata.apikey", JSON.stringify(k)); sessionStorage.setItem("strata.monitor.key", k); } catch (e) {} u.searchParams.delete("strata-key"); history.replaceState(null, "", u); } }"#;

enum Ev {
    Status(Value),
    Memory(Value),
    Apps(Value),
    System(Value),
    Alert(Value),
    Page(Value),
    Log(String),
    Ipc(Value),
    Menu(MenuEvent),
    /// the browser pane: its page's address (each load), and a page's text for the chat ("Send page to chat")
    Web(String),
    WebText(String),
    /// a browser tool's call from the model (browser_mcp), and an element described for browser_type's Allow
    Tool(browser_mcp::Job),
    /// a search's results page, read a moment after it loaded (Google fills it in after its load event)
    /// the page read for a waiting call: Google's results (true) or a page browser_open / browser_back loaded
    PageRead(std::sync::mpsc::Sender<Result<String, String>>, bool),
    Described(browser_mcp::Job, String),
    /// the picker bridge (browser_mcp): op `String` (id u64) was sent to the open picker, or there is none
    FrameSent(browser_mcp::Job, String, u64, String),
    /// its answer looked for again (the u32-th time)
    FrameCollect(browser_mcp::Job, String, u64, u32),
    /// a Google tool's script (google.rs) looked for its value (id, the u32-th time)
    AsyncCollect(browser_mcp::Job, u64, u32),
    /// a click or typing done with the mouse and keyboard (native_input): the element found by a script first
    Native(browser_mcp::Job, &'static str, String),
    /// a Google tool's script past its time: told to change nothing more
    StopScript(u64),
    Quit,
}

/// The browser pane's User-Agent: Safari's own.  A web view's default leaves out "Version/... Safari/...", and Google
/// (Gmail, Drive, sign-in) answers that with "this browser is no longer supported".  Safari keeps the macOS version in
/// it at 10_15_7 (frozen); the Safari version is the installed one.
fn safari_user_agent(version: &str) -> String {
    format!("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/{version} Safari/605.1.15")
}

fn safari_version() -> String {
    let out = Command::new("defaults").args(["read", "/Applications/Safari.app/Contents/Info.plist", "CFBundleShortVersionString"]).output();
    out.ok().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).filter(|v| !v.is_empty() && v.chars().all(|c| c.is_ascii_digit() || c == '.')).unwrap_or_else(|| "26.0".into())
}

/// The browser pane's address field: an address as typed, a bare host (has a dot, no space) over https, anything else
/// a Google search.
fn web_url(input: &str) -> String {
    let t = input.trim();
    if t.starts_with("http://") || t.starts_with("https://") || t.starts_with("about:") {
        return t.to_string();
    }
    if !t.is_empty() && !t.contains(char::is_whitespace) && t.contains('.') {
        return format!("https://{t}");
    }
    let q: String = form_urlencoded_q(t);
    format!("https://www.google.com/search?q={q}")
}

fn form_urlencoded_q(s: &str) -> String {
    s.bytes().map(|b| match b {
        b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (b as char).to_string(),
        b' ' => "+".into(),
        _ => format!("%{b:02X}"),
    }).collect()
}

/// What "Send page to chat" reads from the browser pane: the title, the address and the text people see (20,000
/// characters at most: a long page is cut, and said to be).
const WEB_READ: &str = "JSON.stringify({title: document.title, url: location.href, text: (document.body ? document.body.innerText : '').slice(0, 20000), cut: (document.body ? document.body.innerText.length : 0) > 20000})";

const WEB_HTML: &str = include_str!("web.html");
const WEB_BAR: f64 = 36.0;

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

/// The config's engine arguments for --plan, with "parallel": N as --batch N the way the server adds it (at most 8):
/// each batch slot holds a whole context's cache, so three agents need three caches.
fn plan_args(c: &Value) -> Vec<String> {
    let mut args: Vec<String> = c["args"].as_array().map(|a| a.iter().filter_map(Value::as_str).filter(|a| *a != "--serve").map(String::from).collect()).unwrap_or_default();
    if let Some(n) = c["parallel"].as_i64().filter(|n| *n > 1) {
        if !args.iter().any(|a| a == "--batch" || a == "--slots") {
            args.extend(["--batch".into(), n.min(8).to_string()]);
        }
    }
    args
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
    let mut cmd = Command::new(&exe);
    cmd.arg("--plan").args(plan_args(&c)).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null());
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

/// The apps using the most memory now, the ones a person can quit to give the model room: each app's processes added
/// up (Chrome's helpers count as Chrome), from top's MEM (it counts compressed memory, unlike ps's RSS) and ps's paths.
/// macOS's own (/System) and Strata are left out.  ([{name, bundle, pid, bytes}], largest first, 150 MB or more;
/// Strata's own processes added up: this app, its server and engine).
fn memory_apps() -> (Vec<Value>, u64) {
    let me = unsafe { libc::getuid() }.to_string();
    let ps = Command::new("/bin/ps").args(["-axo", "pid=,uid=,comm="]).output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
    let top = Command::new("/usr/bin/top").args(["-l", "1", "-stats", "pid,mem"]).output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
    let unit = |m: &str| -> u64 {
        let m = m.trim_end_matches(['+', '-']);
        let (n, u) = m.split_at(m.find(|c: char| c.is_ascii_alphabetic()).unwrap_or(m.len()));
        let n: f64 = n.parse().unwrap_or(0.0);
        (n * match u { "K" => 1024.0, "M" => 1048576.0, "G" => 1073741824.0, _ => 1.0 }) as u64
    };
    let mem: HashMap<u32, u64> = top.lines().filter_map(|l| {
        let mut f = l.split_whitespace();
        Some((f.next()?.parse().ok()?, unit(f.next()?)))
    }).collect();
    let mut apps: HashMap<String, (u64, u32)> = HashMap::new();
    let mut strata = 0;
    for l in ps.lines() {
        // "  296   501 /Applications/X.app/Contents/MacOS/X": columns padded with spaces, the path may hold spaces too
        let Some((pid, rest)) = l.trim_start().split_once(' ') else { continue };
        let Some((uid, path)) = rest.trim_start().split_once(' ') else { continue };
        let path = path.trim();
        let Ok(pid) = pid.parse::<u32>() else { continue };
        if uid != me || path.starts_with("/System/") || path.starts_with("/usr/libexec/") || path.starts_with("/usr/sbin/") {
            continue;
        }
        // an app's processes group under its bundle; any other process (a service, a node server) under its own path,
        // listed but never quit from here: ending a service can do harm a person does not see (an LLM server, a database)
        let bundle = match path.find(".app/") {
            Some(i) => &path[..i + 4],
            None => path,
        };
        let name = Path::new(bundle).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        if bundle.ends_with("/Strata.app") || name.starts_with("strata-") || pid == std::process::id() {
            strata += mem.get(&pid).copied().unwrap_or(0);
            continue;
        }
        let e = apps.entry(bundle.to_string()).or_insert((0, 0));
        e.0 += mem.get(&pid).copied().unwrap_or(0);
        if (path == bundle || path.starts_with(&format!("{bundle}/Contents/MacOS/"))) && (e.1 == 0 || pid < e.1) {
            e.1 = pid; // the app's own process (not a helper): the one quitting reaches
        }
    }
    let mut v: Vec<(String, u64, u32)> = apps
        .into_iter()
        .filter(|(k, (b, pid))| if k.ends_with(".app") { *b >= 150 << 20 && *pid > 0 } else { *b >= 300 << 20 })
        .map(|(k, (b, p))| (k, b, p))
        .collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    let list = v.into_iter().take(8).map(|(bundle, bytes, pid)| {
        let app = bundle.ends_with(".app");
        let name = Path::new(&bundle).file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        json!({"name": name, "bundle": bundle, "pid": pid, "bytes": bytes, "app": app})
    }).collect();
    (list, strata)
}

/// The whole Mac's memory as Activity Monitor counts it: app (anonymous pages less purgeable), wired, compressed (the
/// compressor's own pages), the rest free or file cache; pressure is 100 - kern.memorystatus_level; swap in use.
fn sys_memory() -> Value {
    let vm = Command::new("/usr/bin/vm_stat").output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
    let sc = Command::new("/usr/sbin/sysctl").args(["-n", "kern.memorystatus_level", "hw.memsize", "vm.swapusage"]).output()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
    sys_memory_of(&vm, &sc)
}

fn sys_memory_of(vm: &str, sc: &str) -> Value {
    let page = vm.split("page size of ").nth(1).and_then(|r| r.split_whitespace().next()).and_then(|n| n.parse::<u64>().ok()).unwrap_or(16384);
    let pages = |name: &str| vm.lines().find(|l| l.starts_with(name)).and_then(|l| l.rsplit(':').next()).and_then(|n| n.trim().trim_end_matches('.').parse::<u64>().ok()).unwrap_or(0);
    let mut l = sc.lines();
    let level: Option<u64> = l.next().and_then(|x| x.trim().parse().ok());
    let total: u64 = l.next().and_then(|x| x.trim().parse().ok()).unwrap_or(0);
    // "total = 17408.00M  used = 15980.56M  free = 1427.44M  (encrypted)"
    let swap = |k: &str| l.clone().next().and_then(|x| x.split(k).nth(1)).and_then(|r| r.trim_start().split('M').next())
        .and_then(|n| n.trim().parse::<f64>().ok()).map(|m| (m * 1048576.0) as u64).unwrap_or(0);
    let app = pages("Anonymous pages").saturating_sub(pages("Pages purgeable")) * page;
    let wired = pages("Pages wired down") * page;
    let compressed = pages("Pages occupied by compressor") * page;
    json!({"total": total, "app": app, "wired": wired, "compressed": compressed,
        "free": total.saturating_sub(app + wired + compressed), "pressure": level.map(|v| 100u64.saturating_sub(v)),
        "swap_used": swap("used ="), "swap_total": swap("total =")})
}

/// Quit an app the way the Dock does (it may ask to save); `force`: SIGKILL its processes (unsaved work is lost).
/// Some(still running) after up to 8 s.
fn quit_app(bundle: &str, pid: u32, force: bool) -> bool {
    if !bundle.ends_with(".app") {
        return true; // a service or a command-line process: never ended from here
    }
    if force {
        let _ = Command::new("/usr/bin/pkill").args(["-KILL", "-f", &format!("^{}/", regex_escape(bundle))]).status();
    } else {
        let id = Command::new("/usr/bin/defaults").args(["read", &format!("{bundle}/Contents/Info"), "CFBundleIdentifier"]).output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
        let script = if id.is_empty() { format!("tell application \"{bundle}\" to quit") } else { format!("tell application id \"{id}\" to quit") };
        let _ = Command::new("/usr/bin/osascript").args(["-e", &script]).status();
    }
    for _ in 0..16 {
        if unsafe { libc::kill(pid as i32, 0) } != 0 {
            return false;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    true
}

fn regex_escape(s: &str) -> String {
    s.chars().map(|c| if "\\.+*?()|[]{}^$".contains(c) { format!("\\{c}") } else { c.to_string() }).collect()
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
const CONTEXTS: [u64; 12] = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];

/// What a config needs against what this Mac has: the GPU's share must fit Metal's working set, and everything must
/// fit the memory free now (when the server is not running: a running engine already holds its share).  For each
/// context size the need follows the engine's per-token cost; a smaller --prefill-batch shrinks the work buffers
/// (their size is the batch x the vocabulary's logits).
fn memory(cfg_path: &Path, running: bool) -> Value {
    // Apple's on-device model: macOS runs it (its own memory, outside the GPU limit Strata plans for), at a context
    // Apple sets (4,096 tokens on M1/M2, 8,192 on M3 and later)
    if read_config(cfg_path)["engine"] == "afm" {
        return json!({"config": cfg_path, "afm": true, "state": "green", "running": running, "options": [],
                      "advice": "Apple's on-device model (FoundationModels): macOS runs it in its own memory, no download"});
    }
    let p = match plan(cfg_path) {
        Ok(p) => p,
        Err(e) => return json!({"config": cfg_path, "error": e}),
    };
    let n = |v: &Value| v.as_f64().unwrap_or(0.0);
    let (g, h) = (&p["gpu"], &p["host"]);
    let ctx = n(&p["context"]);
    let per_token = n(&p["per_token"]);
    let (limit, wired_mb, suggest_mb) = gpu_limit(n(&g["limit"]));
    // Metal maps the whole model file: the part llama.cpp counts on the CPU side (token embeddings) is in the GPU's
    // share too (measured on an M1: MTL0_Mapped 11,536 MiB = 10,949 + 587)
    let gpu_need = n(&g["model"]) + n(&h["model"]) + n(&g["context"]) + n(&g["compute"]);
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
            // the GPU's share alone (what fit_fix needs: memory free now changes once macOS drops its caches), at this
            // batch or a smaller one: 512, then 256 (the work buffers scale with it)
            let batch = [None, Some(512.0), Some(256.0)].into_iter().find(|b| match b {
                None => gpu <= limit,
                Some(b) => pb > *b && gpu - (n(&g["compute"]) + n(&h["compute"])) * (1.0 - b / pb) <= limit,
            });
            json!({"ctx": c, "need": if ok512 { total - saved_512 } else { total }, "fits": ok || ok512, "batch512": ok512,
                   "gpu_fits": batch.is_some(), "gpu_batch": batch.flatten().map(|b| b as u64)})
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
        "state": state, "advice": advice, "options": options, "recommended": best.map(|i| options[i].clone()),
        "wired_mb": wired_mb, "suggest_limit_mb": if suggest_mb > 0 && gpu_need > limit { json!(suggest_mb) } else { Value::Null },
        "power": power()})
}

/// What the GPU may really hold: iogpu.wired_limit_mb when it is set; else macOS's default, which on Macs up to 36 GB
/// is about two thirds of the memory, below Metal's recommendedMaxWorkingSetSize (a 16 GB M1: 12.4 GiB recommended,
/// ran out at 11.7).  (limit bytes, the sysctl's MB, a value to suggest: the memory less 3 GB or 15% for macOS)
fn gpu_limit(recommended: f64) -> (f64, u64, u64) {
    let sysctl = |k: &str| Command::new("/usr/sbin/sysctl").args(["-n", k]).output().ok().and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse::<u64>().ok()).unwrap_or(0);
    let (wired, ram) = (sysctl("iogpu.wired_limit_mb"), sysctl("hw.memsize"));
    let ram_mb = ram >> 20;
    let suggest = ram_mb.saturating_sub((ram_mb * 15 / 100).max(3072));
    // 1.25 GB of the limit stays for the others the GPU holds memory for (the window server's buffers, Metal itself,
    // and what Metal holds beyond llama.cpp's own count): a 16 GB M1 at 13,312 MB had IOGPU "FAILED To Free" and
    // WindowServer frames fail at 12.76 GB planned (8K, batch 512), none at 12.53 (8K, batch 256)
    const OTHERS: f64 = 1.25 * 1073741824.0;
    if wired > 0 {
        return (wired as f64 * 1048576.0 - OTHERS, wired, if suggest > wired { suggest } else { 0 });
    }
    let default = if ram <= 36 << 30 { (ram as f64 * 2.0 / 3.0).min(recommended) } else { recommended };
    (default - OTHERS, 0, suggest)
}

/// The Mac's energy mode (pmset): Low Power slows the GPU; High Power (MacBook Pro with Max, 2024+ Pro, Mac mini Pro,
/// Mac Studio 2026: their pmset has a `powermode` line, 0 automatic / 1 low / 2 high) lets its fans hold the clock.
fn power() -> Value {
    let g = Command::new("/usr/bin/pmset").arg("-g").output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
    let batt = Command::new("/usr/bin/pmset").args(["-g", "batt"]).output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
    power_from(&g, &batt)
}

/// pmset -g and pmset -g batt -> {low, high_capable, high, battery}.  A Mac with High Power Mode lists `powermode`
/// (0 automatic, 1 low, 2 high); the others only `lowpowermode`.
fn power_from(g: &str, batt: &str) -> Value {
    let field = |k: &str| g.lines().find_map(|l| { let mut f = l.split_whitespace(); (f.next() == Some(k)).then(|| f.next().and_then(|v| v.parse::<u8>().ok())).flatten() });
    json!({"low": field("lowpowermode") == Some(1) || field("powermode") == Some(1), "high_capable": field("powermode").is_some(),
           "high": field("powermode") == Some(2), "battery": batt.contains("'Battery Power'")})
}

// ------------------------------------------------------------------ the Mac's own memory users, kept off on request
//
// Three toggles in the stopped screen's System card (settings app.json: keep_ai_off, keep_still_wallpaper; the GPU
// limit lives in a LaunchDaemon, not in the app): what macOS runs that a model would rather have the memory of.

const AI_PROCS: [&str; 2] = ["TGOnDeviceInferenceProviderService", "Siri AI"];
const AERIAL_PROC: &str = "WallpaperAerialsExtension";
const STILL_WALLPAPER: &str = "/System/Library/Desktop Pictures/Solid Colors/Black.png";
const GPU_DAEMON: &str = "/Library/LaunchDaemons/us.tottle.strata.gpu-limit.plist";
const GPU_LABEL: &str = "us.tottle.strata.gpu-limit";

/// (pids, resident bytes) of this user's processes with exactly this name.
fn procs(name: &str) -> (Vec<i32>, u64) {
    let out = Command::new("/bin/ps").args(["-axo", "pid=,uid=,rss=,comm="]).output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
    let me = unsafe { libc::getuid() }.to_string();
    let (mut pids, mut bytes) = (vec![], 0u64);
    for l in out.lines() {
        let mut f = l.split_whitespace();
        let (Some(pid), Some(uid), Some(rss)) = (f.next(), f.next(), f.next()) else { continue };
        let comm = f.collect::<Vec<_>>().join(" ");
        if uid == me && Path::new(&comm).file_name().is_some_and(|n| n == name) {
            pids.push(pid.parse().unwrap_or(0));
            bytes += rss.parse::<u64>().unwrap_or(0) * 1024;
        }
    }
    (pids, bytes)
}

fn end(name: &str) {
    for pid in procs(name).0.into_iter().filter(|p| *p > 0) {
        unsafe { libc::kill(pid, libc::SIGTERM) };
    }
}

/// What the System card shows: each toggle's state and the memory its processes hold now.
fn system_state() -> Value {
    let settings = load_settings();
    let (ai, ai_b) = AI_PROCS.iter().map(|n| procs(n)).fold((0usize, 0u64), |(c, b), (p, x)| (c + p.len(), b + x));
    let (aer, aer_b) = procs(AERIAL_PROC);
    let siri = Command::new("/usr/bin/defaults").args(["read", "com.apple.assistant.support", "Assistant Enabled"]).output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "1").unwrap_or(false);
    let wired = Command::new("/usr/sbin/sysctl").args(["-n", "iogpu.wired_limit_mb"]).output().ok().and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse::<u64>().ok()).unwrap_or(0);
    let daemon = std::fs::read_to_string(GPU_DAEMON).ok();
    let daemon_mb = daemon.as_ref().and_then(|t| t.split("iogpu.wired_limit_mb=").nth(1).and_then(|r| r.split('<').next()?.trim().parse::<u64>().ok()));
    // a LaunchDaemon from before the kernel's cap was set with it: turned on again, it is rewritten (sound needs it)
    let daemon_old = daemon.as_ref().is_some_and(|t| !t.contains("vm.global_user_wire_limit"));
    let wire_cap = Command::new("/usr/sbin/sysctl").args(["-n", "vm.global_user_wire_limit"]).output().ok().and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse::<u64>().ok()).unwrap_or(0);
    let (_, _, suggest) = gpu_limit(f64::MAX);
    json!({"keep_ai_off": settings["keep_ai_off"] == true, "ai_running": ai, "ai_bytes": ai_b, "siri": siri,
           "keep_still_wallpaper": settings["keep_still_wallpaper"] == true, "aerial_running": aer.len(), "aerial_bytes": aer_b,
           "gpu_daemon_mb": if daemon_old { None } else { daemon_mb }, "gpu_daemon_old_mb": if daemon_old { daemon_mb } else { None },
           "wire_cap": wire_cap, "wired_mb": wired, "suggest_mb": suggest, "power": power()})
}

fn still_wallpaper() {
    let script = format!("tell application \"System Events\" to tell every desktop to set picture to POSIX file \"{STILL_WALLPAPER}\"");
    let _ = Command::new("/usr/bin/osascript").args(["-e", &script]).status();
    std::thread::sleep(Duration::from_secs(2));
    end(AERIAL_PROC);
}

/// Apply a toggle.  Apple Intelligence's own switch is the user's (System Settings opens at it); Siri's preference
/// and the on-device model's processes this app handles.  The GPU limit: a LaunchDaemon that sets it at every boot.
/// The kernel's cap on memory programs may pin (vm.global_user_wire_limit) for a GPU limit of `mb`: the GPU's share
/// counts against it, and macOS's default (the memory less ~4 GB) is below a raised GPU limit on small Macs: a 16 GB
/// M1 at 13,312 MB had 13.9 GB pinned against a 13.06 GB cap, and coreaudiod could not pin 80 KB for the speakers
/// (mlock failed: no sound at all while the model was loaded).  The GPU's limit + 1.5 GB, at most the memory less 2 GiB.
fn wire_limit_for(mb: u64, memsize: u64) -> u64 {
    (mb * 1048576 + 1_500_000_000).min(memsize.saturating_sub(2 << 30))
}

/// The two settings for a GPU limit of `mb`, for one `sysctl` call: the GPU's, and the kernel's cap above it.
fn wire_settings(mb: u64) -> String {
    let memsize = Command::new("/usr/sbin/sysctl").args(["-n", "hw.memsize"]).output().ok()
        .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse::<u64>().ok()).unwrap_or(0);
    let default = memsize.saturating_sub(Command::new("/usr/sbin/sysctl").args(["-n", "vm.global_no_user_wire_amount"]).output().ok()
        .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse::<u64>().ok()).unwrap_or(0));
    let wire = wire_limit_for(mb, memsize).max(default);
    format!("iogpu.wired_limit_mb={mb} vm.global_user_wire_limit={wire}")
}

fn system_toggle(what: &str, on: bool, mb: u64) -> Result<(), String> {
    let mut set = load_settings();
    match what {
        "ai" => {
            set["keep_ai_off"] = json!(on);
            save_settings(&set);
            if on {
                let _ = Command::new("/usr/bin/defaults").args(["write", "com.apple.assistant.support", "Assistant Enabled", "-bool", "false"]).status();
                for n in AI_PROCS {
                    end(n);
                }
                let _ = Command::new("/usr/bin/open").arg("x-apple.systempreferences:com.apple.Siri-Settings.extension").status();
            } else {
                let _ = Command::new("/usr/bin/defaults").args(["write", "com.apple.assistant.support", "Assistant Enabled", "-bool", "true"]).status();
            }
        }
        "wallpaper" => {
            set["keep_still_wallpaper"] = json!(on);
            save_settings(&set);
            if on {
                still_wallpaper();
            } else {
                let _ = Command::new("/usr/bin/open").arg("x-apple.systempreferences:com.apple.Wallpaper-Settings.extension").status();
            }
        }
        "power" => {
            // battery and power adapter both (-a): High Power Mode where the Mac has it, else Low Power Mode off
            let capable = power()["high_capable"] == true;
            let shell = match (capable, on) {
                (true, true) => "pmset -a powermode 2",
                (true, false) => "pmset -a powermode 0",
                (false, true) => "pmset -a lowpowermode 0",
                (false, false) => return Ok(()), // nothing to undo: Low Power Mode stays as the user sets it
            };
            if !admin(shell) {
                return Err("not changed (the password dialog was cancelled)".into());
            }
        }
        "gpu" => {
            let shell = if on {
                let plist = format!(
                    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n\
<plist version=\"1.0\"><dict><key>Label</key><string>{GPU_LABEL}</string><key>ProgramArguments</key><array><string>/usr/sbin/sysctl</string>\
{}</array><key>RunAtLoad</key><true/></dict></plist>\n",
                    wire_settings(mb).split(' ').map(|a| format!("<string>{a}</string>")).collect::<String>());
                let tmp = std::env::temp_dir().join("us.tottle.strata.gpu-limit.plist");
                std::fs::write(&tmp, plist).map_err(|e| e.to_string())?;
                format!("launchctl bootout system/{GPU_LABEL} 2>/dev/null; install -m 644 -o root -g wheel '{}' {GPU_DAEMON} && launchctl bootstrap system {GPU_DAEMON}", tmp.display())
            } else {
                // macOS's defaults: the GPU's own, and the kernel's cap of the memory less its reserve
                format!("launchctl bootout system/{GPU_LABEL} 2>/dev/null; rm -f {GPU_DAEMON}; sysctl iogpu.wired_limit_mb=0 \
                         vm.global_user_wire_limit=$(( $(sysctl -n hw.memsize) - $(sysctl -n vm.global_no_user_wire_amount) ))")
            };
            if !admin(&shell) {
                return Err("not changed (the password dialog was cancelled)".into());
            }
        }
        _ => return Err(format!("unknown toggle {what}")),
    }
    Ok(())
}

/// While the app runs: what the toggles keep off stays off (macOS starts the on-device model again on demand, and an
/// update can bring the aerial wallpaper back).  Every 30 s; quiet when nothing is on.
fn keep_off() {
    loop {
        let set = load_settings();
        if set["keep_ai_off"] == true {
            for n in AI_PROCS {
                end(n);
            }
        }
        if set["keep_still_wallpaper"] == true && !procs(AERIAL_PROC).0.is_empty() {
            still_wallpaper();
        }
        std::thread::sleep(Duration::from_secs(30));
    }
}

/// A system setting through macOS's own administrator dialog (osascript "with administrator privileges").
fn admin(shell: &str) -> bool {
    let script = format!("do shell script \"{}\" with administrator privileges", shell.replace('\\', "\\\\").replace('"', "\\\""));
    Command::new("/usr/bin/osascript").args(["-e", &script]).status().map(|s| s.success()).unwrap_or(false)
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
    write_config(cfg_path, &text, &v)
}

/// How many requests the server runs at once ("parallel": 2..8 batch slots, one engine and one copy of the weights);
/// 1 takes the key out.  A running server restarts with it.
fn set_parallel(cfg_path: &Path, n: u64) -> Result<(), String> {
    let text = std::fs::read_to_string(cfg_path).map_err(|e| e.to_string())?;
    let mut v: Value = serde_json::from_str(text.trim_start_matches('\u{feff}')).map_err(|e| e.to_string())?;
    let o = v.as_object_mut().ok_or("the config is not a JSON object")?;
    if n > 1 {
        o.insert("parallel".into(), json!(n.min(8)));
    } else {
        o.remove("parallel");
    }
    write_config(cfg_path, &text, &v)
}

/// The engine's conversation cache in RAM (`--conversation-cache-mib N` in "args"): conversations another one displaced
/// come back without being read again; 0 writes "0" (off), -1 takes the flag out (the server sizes it for this Mac).
/// A running server restarts with it.
fn set_cache_mib(cfg_path: &Path, mib: i64) -> Result<(), String> {
    let text = std::fs::read_to_string(cfg_path).map_err(|e| e.to_string())?;
    let mut v: Value = serde_json::from_str(text.trim_start_matches('\u{feff}')).map_err(|e| e.to_string())?;
    let args = v["args"].as_array_mut().ok_or("the config has no \"args\"")?;
    if let Some(i) = args.iter().position(|a| a == "--conversation-cache-mib") {
        args.drain(i..(i + 2).min(args.len()));
    }
    if mib >= 0 {
        args.extend([json!("--conversation-cache-mib"), json!(mib.to_string())]);
    }
    write_config(cfg_path, &text, &v)
}

/// The shared K/V pool (docs/CONTEXT_POOL.md): `--kv-unified --kv-pool N` in "args" (N tokens all agents draw from,
/// each up to its --max-context); 0 takes both out (each agent its own context).  A running server restarts with it.
fn set_pool(cfg_path: &Path, tokens: u64) -> Result<(), String> {
    let text = std::fs::read_to_string(cfg_path).map_err(|e| e.to_string())?;
    let mut v: Value = serde_json::from_str(text.trim_start_matches('\u{feff}')).map_err(|e| e.to_string())?;
    let args = v["args"].as_array_mut().ok_or("the config has no \"args\"")?;
    if let Some(i) = args.iter().position(|a| a == "--kv-pool") {
        args.drain(i..(i + 2).min(args.len()));
    }
    args.retain(|a| a != "--kv-unified");
    if tokens > 0 {
        args.extend([json!("--kv-unified"), json!("--kv-pool"), json!(tokens.to_string())]);
    }
    write_config(cfg_path, &text, &v)
}

/// The config's shared pool in tokens (0: none) and its per-agent context
fn pool_of(c: &Value) -> (u64, u64) {
    let a = c["args"].as_array().map(Vec::as_slice).unwrap_or(&[]);
    let val = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).and_then(Value::as_str).and_then(|n| n.parse().ok());
    (if a.iter().any(|x| x == "--kv-unified") { val("--kv-pool").unwrap_or(0) } else { 0 }, val("--max-context").unwrap_or(0))
}

/// The config's conversation cache budget in MiB (0: off, -1: none set, the server's automatic size).
fn cache_mib(c: &Value) -> i64 {
    let a = c["args"].as_array().map(Vec::as_slice).unwrap_or(&[]);
    a.iter().position(|x| x == "--conversation-cache-mib").and_then(|i| a.get(i + 1)).and_then(Value::as_str).and_then(|n| n.parse().ok()).unwrap_or(-1)
}

/// The config written back with the indent it had (its other keys as they were).
fn write_config(cfg_path: &Path, text: &str, v: &Value) -> Result<(), String> {
    let indent = text.lines().nth(1).map(|l| l.len() - l.trim_start().len()).filter(|n| *n > 0).unwrap_or(1);
    let mut out = vec![];
    let fmt = serde_json::ser::PrettyFormatter::with_indent(&b"        "[..indent.min(8)]);
    let mut ser = serde_json::Serializer::with_formatter(&mut out, fmt);
    serde::Serialize::serialize(v, &mut ser).map_err(|e| e.to_string())?;
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
    /// the model's browser (browser_mcp): its endpoint's port and key, for the server this app starts
    browser: Option<(u16, String)>,
    /// started by `--deploy` for blue-green (deploy.rs): the first server start takes the port over (--handoff) from
    /// the old app's server instead of stopping it (--replace)
    handoff: bool,
}

impl State {
    fn api_key(&self) -> String {
        let cfg = self.config.as_deref().map(read_config).unwrap_or(json!({}));
        let k = match &cfg["api_key"] {
            Value::String(s) => s.split(',').next().unwrap_or("").trim().to_string(),
            Value::Array(a) => a.first().and_then(Value::as_str).unwrap_or("").to_string(),
            _ => String::new(),
        };
        if k.is_empty() { env_api_key(&cfg) } else { k }
    }

    fn base(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    fn push_log(&mut self, line: String) {
        eprintln!("strata-app: {line}"); // the log a run from Terminal (or `open --stderr FILE`) keeps
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
/// A config too big for the GPU's share (memory()'s "red") -> the largest context that fits, and the prefill batch it
/// needs.  Run as it is, the model leaves the window server too little of the GPU's memory: the screen glitches
/// (WindowServer: kIOGPUCommandBufferCallbackErrorOutOfMemory) and the answers slow down (a 16 GB M1 at 16K: 10.6-15
/// tok/s with 0.34 GB left over).
fn fit_fix(m: &Value) -> Option<(u64, Option<u64>)> {
    if m["state"] != "red" {
        return None;
    }
    let r = m["options"].as_array()?.iter().rev().find(|o| o["gpu_fits"] == true)?;
    Some((r["ctx"].as_u64()?, r["gpu_batch"].as_u64()))
}

fn start(st: &Shared, proxy: &EventLoopProxy<Ev>) -> Result<(), String> {
    let cfg = st.lock().unwrap().config.clone().ok_or("pick a model config first")?;
    if let Some((ctx, batch)) = fit_fix(&memory(&cfg, false)) {
        set_context(&cfg, ctx, batch)?;
        st.lock().unwrap().push_log(format!("[app] the config was too big for the GPU's memory: context {ctx}{} so the screen keeps its share", batch.map(|b| format!(", prefill batch {b}")).unwrap_or_default()));
    }
    let mut s = st.lock().unwrap();
    if s.child.is_some() {
        return Err("the server is already running".into());
    }
    let exe = server_exe(&s.root);
    if !exe.is_file() {
        return Err(format!("{} is missing: build it with `make server` in {}", exe.display(), s.root.display()));
    }
    let mut cmd = Command::new(&exe);
    cmd
        .args(["--engine", "strata", "--config"])
        .arg(&cfg)
        .args(["--port", &s.port.to_string(), if s.handoff { "--handoff" } else { "--replace" }])
        .current_dir(&s.root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some((bport, bkey)) = &s.browser {
        // the server adds the model's browser tools (MCP server "browser", read-only, not saved)
        cmd.env("STRATA_BROWSER_MCP_URL", format!("http://127.0.0.1:{bport}/mcp")).env("STRATA_BROWSER_MCP_KEY", bkey);
    }
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
    s.handoff = false;
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
                status["conversation_cache"] = m["conversation_cache"].clone();
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

/// API_KEY from the Strata folder's .env (the config's "cwd"): the server accepts it too, so this one sends it when
/// the config has none (server/src/dotenv.rs reads the same file; quotes and an inline " #" comment are dropped).
fn env_api_key(cfg: &Value) -> String {
    let Some(dir) = cfg["cwd"].as_str() else { return String::new() };
    let text = std::fs::read_to_string(Path::new(dir).join(".env")).unwrap_or_default();
    text.lines()
        .filter_map(|l| l.trim().strip_prefix("export ").unwrap_or(l.trim()).strip_prefix("API_KEY="))
        .last()
        .map(|v| v.split(" #").next().unwrap_or("").trim().trim_matches(|c| c == '"' || c == '\'').to_string())
        .unwrap_or_default()
}

/// An address without its scheme, "www." and a last "/": how a read's links and browser_open's address are compared.
fn link_key(u: &str) -> String {
    let u = u.trim();
    let u = u.split_once("://").map(|(_, r)| r).unwrap_or(u);
    let u = u.strip_prefix("www.").unwrap_or(u);
    u.split('#').next().unwrap_or(u).trim_end_matches('/').to_lowercase()
}

/// The site of an address (its host without "www."): what browser_open's check compares.
fn site_of(u: &str) -> String {
    link_key(u).split('/').next().unwrap_or("").split(':').next().unwrap_or("").to_string()
}

/// After a search, an address opens when its site was among the links or the site names of the pages the model read
/// (a page on bbc.com once bbc.com showed up; Google hides most result addresses behind redirects, so the exact page
/// cannot be required: the model knew bbc.com/weather/2988507, Paris's, and was refused).
fn link_seen(seen: &std::collections::HashSet<String>, url: &str) -> bool {
    if seen.contains(&link_key(url)) {
        return true;
    }
    // the site, or a site it belongs to (weather.bbc.com once bbc.com showed up), down to two labels
    let site = site_of(url);
    let labels: Vec<&str> = site.split('.').collect();
    (0..labels.len().saturating_sub(1)).any(|i| seen.contains(&format!("site:{}", labels[i..].join("."))))
}

fn remember_links(seen: &Mutex<std::collections::HashSet<String>>, v: &Value) {
    let mut s = seen.lock().unwrap();
    for h in v["hrefs"].as_array().into_iter().flatten().filter_map(Value::as_str) {
        s.insert(link_key(h));
        s.insert(format!("site:{}", site_of(h)));
    }
    for h in v["sites"].as_array().into_iter().flatten().filter_map(Value::as_str) {
        s.insert(format!("site:{}", site_of(h)));
    }
}

/// A script's value from evaluate_script_with_callback: JSON-encoded (a string holding the JSON the script built).
fn js_value(r: &str) -> Value {
    serde_json::from_str::<String>(r).ok().and_then(|s| serde_json::from_str(&s).ok()).or_else(|| serde_json::from_str(r).ok()).unwrap_or(Value::Null)
}

/// Run `js` in the browser pane and answer the model's call with `f` of its value (the page's {error} as an error).
fn eval_reply(web: &WebView, js: &str, reply: std::sync::mpsc::Sender<Result<String, String>>, f: impl Fn(Value) -> Result<String, String> + Send + 'static) {
    let _ = web.evaluate_script_with_callback(js, move |r| {
        let v = js_value(&r);
        let out = if let Some(e) = v["error"].as_str() {
            Err(e.to_string())
        } else if v.is_null() {
            Err("the page did not answer (still loading?): try again in a moment".into())
        } else {
            f(v)
        };
        let _ = reply.send(out);
    });
}

/// Run an action's `js` (a click, a scroll, a submit), then answer with the page as it is `ms` later: an answer the model
/// would otherwise spend a second call on (8 calls bought 4 page views: click then read, scroll then read).  A page the
/// action loads answers sooner, when it is a load waiter too (the first answer is the one the call takes).
fn act_then_read(web: &WebView, js: &str, reply: std::sync::mpsc::Sender<Result<String, String>>, proxy: &tao::event_loop::EventLoopProxy<Ev>, ms: u64) {
    let p = Mutex::new(proxy.clone());
    let _ = web.evaluate_script_with_callback(js, move |r| {
        let v = js_value(&r);
        if let Some(e) = v["error"].as_str() {
            let _ = reply.send(Err(e.to_string()));
            return;
        }
        let (reply, p) = (reply.clone(), p.lock().unwrap().clone());
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(ms));
            let _ = p.send_event(Ev::PageRead(reply, false));
        });
    });
}

/// Op `op` to the picker open over My Maps (browser_mcp's bridge); the answer comes as Ev::FrameSent.
fn frame_send(web: &WebView, proxy: &EventLoopProxy<Ev>, token: &str, id: u64, job: browser_mcp::Job, op: &str, args: Value) {
    let (p, slot, op2) = (Mutex::new(proxy.clone()), Mutex::new(Some(job)), op.to_string());
    let _ = web.evaluate_script_with_callback(&browser_mcp::frame_send_js(token, op, &args, id), move |r| {
        if let Some(j) = slot.lock().unwrap().take() {
            let _ = p.lock().unwrap().send_event(Ev::FrameSent(j, op2.clone(), id, r));
        }
    });
}

/// maps_export's files: the KML, a GeoJSON copy, and Lat / Lng into the table of the same name (when there is one).
fn save_map_export(args: &Value, kml: &str, title: &str, page: &Value) -> Result<String, String> {
    let kml_path = std::path::PathBuf::from(args["__map"].as_str().unwrap_or(""));
    let key = args["key"].as_str().filter(|k| !k.is_empty()).unwrap_or("Link");
    let mut places = mapfile::parse_kml(kml);
    // My Maps' KML has no points for places it placed by address: the page's, by key, into the places and the file
    let pointed = mapfile::add_points(&mut places, &mapfile::page_places(page), key);
    let kml = mapfile::kml_with_points(kml, &places, key);
    let without = places.iter().filter(|p| p.lat.is_none()).count();
    if places.is_empty() {
        return Err(format!("maps_export: the map {title} has no places to save"));
    }
    if let Some(d) = kml_path.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    std::fs::write(&kml_path, &kml).map_err(|e| format!("{}: {e}", kml_path.display()))?;
    let gj = kml_path.with_extension("geojson");
    std::fs::write(&gj, serde_json::to_string_pretty(&mapfile::geojson(&places)).unwrap_or_default()).map_err(|e| format!("{}: {e}", gj.display()))?;
    let table_path = std::path::PathBuf::from(args["__table"].as_str().unwrap_or(""));
    let mut tbl = table::load(&table_path)?;
    let filled = if tbl.is_empty() {
        format!("no local table {} to give coordinates to", table_path.display())
    } else {
        let (n, missed) = mapfile::fill_lat_lng(&mut tbl, &places, key)?;
        table::save(&table_path, &tbl)?;
        format!("Lat / Lng written for {n} rows of {} by {key}{}", table_path.display(), if missed > 0 { format!(" ({missed} places matched no row)") } else { String::new() })
    };
    Ok(format!("saved {} places of {title} ({pointed} points from the map's page{}): {} and {}; {filled}. table_sync puts the new columns in the sheet.",
               places.len(), if without > 0 { format!(", {without} with none") } else { String::new() }, kml_path.display(), gj.display()))
}

/// The page read `ms` from now as the call's answer (what a click or typing led to).
fn read_later(reply: std::sync::mpsc::Sender<Result<String, String>>, proxy: &EventLoopProxy<Ev>, ms: u64) {
    let p = Mutex::new(proxy.clone());
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(ms));
        let _ = p.lock().unwrap().send_event(Ev::PageRead(reply, false));
    });
}

fn say_web(line: &str) {
    eprintln!("strata-app: {line}");
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
            json!({"path": p, "name": p.file_name().unwrap_or_default().to_string_lossy(), "model": c["model_name"], "port": c["port"], "problem": problem(&c),
                "parallel": c["parallel"].as_u64().filter(|n| *n > 1).map(|n| n.min(8)).unwrap_or(1), "cache_mib": cache_mib(&c),
                "pool": pool_of(&c).0, "ctx": pool_of(&c).1})
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

/// The first n characters of a text (for the log).
pub fn short(t: &str, n: usize) -> String {
    t.chars().take(n).collect()
}

/// How long an Allow is good for: less than browser_mcp's 150 s wait for an answer.
const ALLOW_WITHIN_S: u64 = 140;

fn main() {
    let root = root();
    let settings = load_settings();
    let config = settings["config"].as_str().map(PathBuf::from).filter(|p| p.is_file()).or_else(|| configs(&root).into_iter().next());
    let config = config.filter(|c| runnable(&read_config(c))).or_else(|| configs(&root).into_iter().next());
    let port = config.as_deref().map(config_port).unwrap_or(8090);
    // make app-run: `Strata --deploy` puts this build in place of the running app (deploy.rs), with no window
    let argv: Vec<String> = std::env::args().collect();
    if argv.iter().any(|a| a == "--deploy") {
        std::process::exit(deploy::run(&root, config.as_deref()));
    }
    let handoff = argv.iter().any(|a| a == "--handoff");
    let st: Shared = Arc::new(Mutex::new(State { root, config, port, child: None, phase: "stopped", stopping_since: None, log: VecDeque::new(), last: json!({}), browser: None, handoff }));

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
    // the model's browser: its MCP endpoint (after the signals are blocked: its threads must not take them)
    let web_stopped = Arc::new(std::sync::atomic::AtomicBool::new(false));
    {
        let mut b = [0u8; 16];
        let _ = std::fs::File::open("/dev/urandom").and_then(|mut f| std::io::Read::read_exact(&mut f, &mut b));
        let key: String = b.iter().map(|x| format!("{x:02x}")).collect();
        let p = Mutex::new(proxy.clone());
        match browser_mcp::serve(key.clone(), web_stopped.clone(), move |job| {
            let _ = p.lock().unwrap().send_event(Ev::Tool(job));
        }) {
            Ok(bport) => st.lock().unwrap().browser = Some((bport, key)),
            Err(e) => eprintln!("strata-app: the model's browser did not start: {e}"),
        }
    }
    let load_waiters: std::cell::RefCell<Vec<std::sync::mpsc::Sender<Result<String, String>>>> = Default::default();
    // each with when it was asked: an Allow after the caller stopped waiting (150 s) runs nothing (it would act with no
    // one to hear the answer)
    let approvals: std::cell::RefCell<HashMap<u64, (std::time::Instant, browser_mcp::Job)>> = Default::default();
    let search_waiters: std::cell::RefCell<Vec<std::sync::mpsc::Sender<Result<String, String>>>> = Default::default();
    // the addresses the model's reads showed (all links of the pages it read), and whether it searched: after a
    // search, browser_open takes only those (LFM2.5 guessed addresses and got 404s, round after round)
    let seen_links: std::sync::Arc<Mutex<std::collections::HashSet<String>>> = Default::default();
    // the last page text the model got (a search's results, a read): the same again is said in one line, not resent
    let last_read: std::sync::Arc<Mutex<String>> = Default::default();
    let searched: std::cell::Cell<Option<Instant>> = std::cell::Cell::new(None); // the last search (the check holds 10 min)
    let next_ask = std::cell::Cell::new(0u64);
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
        let pi = proxy.clone();
        WebViewBuilder::new()
            .with_initialization_script(KEY_SCRIPT)
            .with_bounds(rect(0.0, BAR, 1280.0, 900.0 - BAR))
            .with_visible(false)
            // the chat's globe button: the page may open and close the browser pane, and nothing else (it shows the
            // server's content: no other app command is taken from it)
            .with_ipc_handler(move |req| {
                if let Ok(v) = serde_json::from_str::<Value>(req.body()) {
                    if v["cmd"] == "web" {
                        let _ = pi.send_event(Ev::Ipc(json!({"cmd": "web"})));
                    }
                    // a link clicked in the chat: http(s) only, into the browser pane
                    if v["cmd"] == "web_open" {
                        if let Some(u) = v["url"].as_str().filter(|u| u.starts_with("https://") || u.starts_with("http://")) {
                            let _ = pi.send_event(Ev::Ipc(json!({"cmd": "web_open", "url": u})));
                        }
                    }
                }
            })
            .with_on_page_load_handler(move |e, url| {
                let event = match e {
                    wry::PageLoadEvent::Started => "started",
                    wry::PageLoadEvent::Finished => "finished",
                };
                eprintln!("strata-app: page {event} {url}"); // a trace a run from Terminal shows (no window needed)
                let _ = p.send_event(Ev::Page(json!({"event": event, "url": url})));
            })
            .build_as_child(&control_win)
            .expect("webview")
    };
    // the browser pane (the bar's Web button): a strip of controls, and a web view, right of the page; hidden until opened
    let web_open = std::cell::Cell::new(false);
    let web_bar = {
        let p = proxy.clone();
        WebViewBuilder::new()
            .with_html(WEB_HTML)
            .with_bounds(rect(640.0, BAR, 640.0, WEB_BAR))
            .with_visible(false)
            .with_ipc_handler(move |req| {
                if let Ok(v) = serde_json::from_str::<Value>(req.body()) {
                    let _ = p.send_event(Ev::Ipc(v));
                }
            })
            .build_as_child(&control_win)
            .expect("webview")
    };
    // the picker bridge's token (browser_mcp::frame_bridge_js): made at each start, known to this app and the script
    let frame_token: String = {
        let mut b = [0u8; 16];
        let _ = std::fs::File::open("/dev/urandom").and_then(|mut f| std::io::Read::read_exact(&mut f, &mut b));
        b.iter().map(|x| format!("{x:02x}")).collect()
    };
    let next_frame_op = std::cell::Cell::new(0u64);
    // Google tools waiting for their page to load (re-run once it has)
    let job_waiters: std::cell::RefCell<Vec<browser_mcp::Job>> = Default::default();
    // navigations the app knows of besides a tool's own load (the strip's address, back, forward, reload; a chat
    // link): a sheets_create whose load overlapped one is not sure the sheet it landed on is the one it made
    let nav_epoch = std::cell::Cell::new(0u64);
    let web = {
        let (p, p2) = (proxy.clone(), proxy.clone());
        WebViewBuilder::new()
            .with_url("about:blank")
            .with_initialization_script_for_main_only(browser_mcp::frame_bridge_js(&frame_token), false)
            .with_user_agent(safari_user_agent(&safari_version()))
            // a link for a new window (Gmail's Drive, target=_blank): this pane loads it (no second window to show)
            .with_new_window_req_handler(move |url, _| {
                let _ = p2.send_event(Ev::Ipc(json!({"cmd": "web_open", "url": url})));
                wry::NewWindowResponse::Deny
            })
            .with_bounds(rect(640.0, BAR + WEB_BAR, 640.0, 900.0 - BAR - WEB_BAR))
            .with_visible(false)
            .with_on_page_load_handler(move |e, url| {
                if matches!(e, wry::PageLoadEvent::Finished) {
                    let _ = p.send_event(Ev::Web(url));
                }
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
    std::thread::spawn(keep_off); // what the System card's toggles keep off
    if handoff || settings["autostart"].as_bool().unwrap_or(false) {
        let _ = start(&st, &proxy);
    }

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        let layout = |full: bool| {
            let sz = control_win.inner_size().to_logical::<f64>(control_win.scale_factor());
            let _ = control_view.set_bounds(rect(0.0, 0.0, sz.width, if full { sz.height } else { BAR }));
            // the browser open: the page on the left half, the browser (its strip, then the web view) on the right
            let split = web_open.get() && !full;
            let left = if split { (sz.width / 2.0).round() } else { sz.width };
            let _ = page.set_bounds(rect(0.0, BAR, left, (sz.height - BAR).max(0.0)));
            let _ = page.set_visible(!full);
            let _ = web_bar.set_bounds(rect(left, BAR, sz.width - left, WEB_BAR));
            let _ = web.set_bounds(rect(left, BAR + WEB_BAR, sz.width - left, (sz.height - BAR - WEB_BAR).max(0.0)));
            let _ = web_bar.set_visible(split);
            let _ = web.set_visible(split);
        };
        let act = |cmd: &str, v: &Value| {
            let note = |msg: &str| js_call(&control_view, "note", &json!(msg));
            match cmd {
                "init" => {
                    js_call(&control_view, "init", &snapshot(&st));
                    check_memory(&st, &proxy, None);
                }
                "memory" => check_memory(&st, &proxy, None),
                "raise_limit" | "power_high" | "low_power_off" => {
                    let shell = match cmd {
                        "raise_limit" => format!("sysctl {}", wire_settings(v["mb"].as_u64().unwrap_or(0))),
                        "power_high" => "pmset -a powermode 2".to_string(),
                        _ => "pmset -a lowpowermode 0".to_string(),
                    };
                    let (p, st2, restart) = (proxy.clone(), st.clone(), cmd == "raise_limit");
                    std::thread::spawn(move || {
                        let ok = admin(&shell);
                        let _ = p.send_event(Ev::Log(format!("── {}: {} ──", shell, if ok { "done" } else { "not changed" })));
                        if ok && restart && st2.lock().unwrap().child.is_some() {
                            stop_and_wait(&st2); // the engine reads the limit when it loads
                            if let Err(e) = start(&st2, &p) {
                                let _ = p.send_event(Ev::Log(format!("── {e} ──")));
                            }
                        }
                        check_memory(&st2, &p, None);
                    });
                }
                "system" => {
                    let p = proxy.clone();
                    std::thread::spawn(move || {
                        let _ = p.send_event(Ev::System(system_state()));
                    });
                }
                "system_toggle" => {
                    let (p, st2) = (proxy.clone(), st.clone());
                    let (what, on, mb) = (v["what"].as_str().unwrap_or("").to_string(), v["on"] == true, v["mb"].as_u64().unwrap_or(0));
                    std::thread::spawn(move || {
                        let r = system_toggle(&what, on, mb);
                        let _ = p.send_event(Ev::Log(format!("── {what} {}: {} ──", if on { "on" } else { "off" }, r.clone().err().unwrap_or("done".into()))));
                        std::thread::sleep(Duration::from_secs(2));
                        let _ = p.send_event(Ev::System(system_state()));
                        check_memory(&st2, &p, None);
                    });
                }
                "apps" => {
                    let p = proxy.clone();
                    std::thread::spawn(move || {
                        let (apps, strata) = memory_apps();
                        let _ = p.send_event(Ev::Apps(json!({"apps": apps, "strata": strata, "sys": sys_memory()})));
                    });
                }
                "quit_app" => {
                    let (p, st2) = (proxy.clone(), st.clone());
                    let (bundle, pid, force) = (v["bundle"].as_str().unwrap_or("").to_string(), v["pid"].as_u64().unwrap_or(0) as u32, v["force"] == true);
                    std::thread::spawn(move || {
                        let still = !bundle.is_empty() && pid > 0 && quit_app(&bundle, pid, force);
                        let (apps, strata) = memory_apps();
                        let _ = p.send_event(Ev::Apps(json!({"apps": apps, "strata": strata, "sys": sys_memory(), "still": if still { json!(bundle) } else { Value::Null }})));
                        check_memory(&st2, &p, None);
                    });
                }
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
                "set_parallel" => {
                    // requests at once in the config; a running server restarts with it (as a new context does)
                    let Some(cfg) = st.lock().unwrap().config.clone() else { return };
                    let n = v["n"].as_u64().unwrap_or(1).clamp(1, 8);
                    let running = st.lock().unwrap().child.is_some();
                    let (st2, p2) = (st.clone(), proxy.clone());
                    std::thread::spawn(move || {
                        if running {
                            stop_and_wait(&st2);
                        }
                        let msg = match set_parallel(&cfg, n) {
                            Ok(()) => format!("── {n} request{} at once in {} ──", if n == 1 { "" } else { "s" }, cfg.file_name().unwrap_or_default().to_string_lossy()),
                            Err(e) => format!("── the config could not be changed: {e} ──"),
                        };
                        let _ = p2.send_event(Ev::Log(msg));
                        if running {
                            if let Err(e) = start(&st2, &p2) {
                                let _ = p2.send_event(Ev::Log(format!("── {e} ──")));
                            }
                        }
                        check_memory(&st2, &p2, None);
                    });
                }
                "set_pool" => {
                    // the shared K/V pool in the config; a running server restarts with it
                    let Some(cfg) = st.lock().unwrap().config.clone() else { return };
                    let tokens = v["tokens"].as_u64().unwrap_or(0);
                    let running = st.lock().unwrap().child.is_some();
                    let (st2, p2) = (st.clone(), proxy.clone());
                    std::thread::spawn(move || {
                        if running {
                            stop_and_wait(&st2);
                        }
                        let name = cfg.file_name().unwrap_or_default().to_string_lossy().into_owned();
                        let msg = match set_pool(&cfg, tokens) {
                            Ok(()) if tokens > 0 => format!("── one K/V pool of {}K tokens for all agents in {name} ──", tokens / 1024),
                            Ok(()) => format!("── each agent its own context in {name} ──"),
                            Err(e) => format!("── the config could not be changed: {e} ──"),
                        };
                        let _ = p2.send_event(Ev::Log(msg));
                        if running {
                            if let Err(e) = start(&st2, &p2) {
                                let _ = p2.send_event(Ev::Log(format!("── {e} ──")));
                            }
                        }
                        check_memory(&st2, &p2, None);
                    });
                }
                "set_cache" => {
                    // the conversation cache's RAM budget in the config; a running server restarts with it
                    let Some(cfg) = st.lock().unwrap().config.clone() else { return };
                    let mib = v["mib"].as_i64().unwrap_or(-1);
                    let running = st.lock().unwrap().child.is_some();
                    let (st2, p2) = (st.clone(), proxy.clone());
                    std::thread::spawn(move || {
                        if running {
                            stop_and_wait(&st2);
                        }
                        let msg = match set_cache_mib(&cfg, mib) {
                            Ok(()) if mib > 0 => format!("── conversation cache {} GB in RAM in {} ──", mib / 1024, cfg.file_name().unwrap_or_default().to_string_lossy()),
                            Ok(()) if mib < 0 => format!("── conversation cache automatic (sized for this Mac) in {} ──", cfg.file_name().unwrap_or_default().to_string_lossy()),
                            Ok(()) => format!("── conversation cache off in {} ──", cfg.file_name().unwrap_or_default().to_string_lossy()),
                            Err(e) => format!("── the config could not be changed: {e} ──"),
                        };
                        let _ = p2.send_event(Ev::Log(msg));
                        if running {
                            if let Err(e) = start(&st2, &p2) {
                                let _ = p2.send_event(Ev::Log(format!("── {e} ──")));
                            }
                        }
                        check_memory(&st2, &p2, None);
                    });
                }
                "select" => {
                    let mut s = st.lock().unwrap();
                    let changed = v["config"].as_str().map(PathBuf::from).filter(|c| s.config.as_ref() != Some(c));
                    if let Some(c) = &changed {
                        s.config = Some(c.clone());
                        s.port = config_port(c);
                    }
                    let mut set = load_settings();
                    set["config"] = json!(s.config);
                    save_settings(&set);
                    // another model picked while one runs: stop it and start the new one (the list stays open then)
                    let switch = changed.is_some() && s.child.is_some();
                    drop(s);
                    js_call(&control_view, "init", &snapshot(&st));
                    check_memory(&st, &proxy, None);
                    if switch {
                        let (st2, p2) = (st.clone(), proxy.clone());
                        std::thread::spawn(move || {
                            stop_and_wait(&st2);
                            if let Err(e) = start(&st2, &p2) {
                                let _ = p2.send_event(Ev::Log(format!("── {e} ──")));
                            }
                        });
                    }
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
                "web" => {
                    web_open.set(!web_open.get());
                    if web_open.get() && web.url().map(|u| u == "about:blank").unwrap_or(true) {
                        nav_epoch.set(nav_epoch.get() + 1);
                        let _ = web.load_url("https://www.google.com/");
                    }
                    layout(full.get());
                    js_call(&control_view, "webOpen", &json!(web_open.get()));
                    js_call(&page, "strataWeb", &json!(web_open.get()));
                }
                "web_open" => {
                    nav_epoch.set(nav_epoch.get() + 1);
                    if !web_open.get() {
                        web_open.set(true);
                        layout(full.get());
                        js_call(&control_view, "webOpen", &json!(true));
                        js_call(&page, "strataWeb", &json!(true));
                    }
                    let _ = web.load_url(v["url"].as_str().unwrap_or("about:blank"));
                }
                "web_go" => {
                    nav_epoch.set(nav_epoch.get() + 1);
                    let _ = web.load_url(&web_url(v["url"].as_str().unwrap_or("")));
                }
                "web_back" => { nav_epoch.set(nav_epoch.get() + 1); let _ = web.evaluate_script("history.back()"); }
                "web_fwd" => { nav_epoch.set(nav_epoch.get() + 1); let _ = web.evaluate_script("history.forward()"); }
                "web_reload" => { nav_epoch.set(nav_epoch.get() + 1); let _ = web.evaluate_script("location.reload()"); }
                "web_allow" => {
                    // the person's answer to browser_type's question
                    let asked = approvals.borrow_mut().remove(&v["id"].as_u64().unwrap_or(0));
                    eprintln!("strata-app: ask {} answered {} ({})", v["id"], if v["yes"] == true { "Allow" } else { "No" },
                        asked.as_ref().map_or("not waiting".to_string(), |(at, j)| format!("{} after {} s{}", j.tool, at.elapsed().as_secs(), if at.elapsed().as_secs() >= ALLOW_WITHIN_S { ": too late, not run" } else { "" })));
                    if let Some((_, job)) = asked.filter(|(at, _)| at.elapsed() < Duration::from_secs(ALLOW_WITHIN_S)) {
                        if google::ask(&job.tool, &job.args).is_some() {
                            let mut job = job;
                            if v["yes"] == true {
                                job.args["__allowed"] = json!(true);
                                let _ = proxy.send_event(Ev::Tool(job));
                            } else {
                                let _ = job.reply.send(Err("the user said no: ask them in the chat".into()));
                            }
                        } else if v["yes"] == true && job.args["__frame"] == true {
                            let args = json!([job.args["element"].as_i64().unwrap_or(0), job.args["text"].as_str().unwrap_or(""), job.args["submit"] == true]);
                            next_frame_op.set(next_frame_op.get() + 1);
                            frame_send(&web, &proxy, &frame_token, next_frame_op.get(), job, "type", args);
                        } else if v["yes"] == true {
                            // the field focused, its text selected; then typed with the keyboard (Ev::Native)
                            let (p, slot) = (Mutex::new(proxy.clone()), Mutex::new(Some(job)));
                            let n = slot.lock().unwrap().as_ref().map_or(0, |j| j.args["element"].as_i64().unwrap_or(0));
                            let _ = web.evaluate_script_with_callback(&browser_mcp::focus_js(n), move |r| {
                                if let Some(j) = slot.lock().unwrap().take() {
                                    let _ = p.lock().unwrap().send_event(Ev::Native(j, "type", r));
                                }
                            });
                        } else {
                            let _ = job.reply.send(Err("the user said no to typing that: ask them in the chat".into()));
                        }
                    }
                }
                "web_stop" => {
                    let on = v["on"] == true;
                    web_stopped.store(on, std::sync::atomic::Ordering::SeqCst);
                    if on {
                        for (_, (_, job)) in approvals.borrow_mut().drain() {
                            let _ = job.reply.send(Err("the user stopped the model's browsing".into()));
                        }
                    }
                    js_call(&web_bar, "stopped", &json!(on));
                }
                "web_send" => {
                    let p = proxy.clone();
                    let _ = web.evaluate_script_with_callback(WEB_READ, move |r| {
                        let _ = p.send_event(Ev::WebText(r));
                    });
                }
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
            Event::UserEvent(Ev::Web(url)) => {
                js_call(&web_bar, "setUrl", &json!(url));
                if url.contains("google.") && (url.contains("/search") || url.contains("/sorry/")) {
                    // the results page read compactly (Google gives this web view a plain page: its answers and the
                    // results' snippets read well as text); a "prove you are a person" page is said as such
                    let blocked = url.contains("/sorry/");
                    for w in search_waiters.borrow_mut().drain(..) {
                        if blocked {
                            let _ = w.send(Err("Google asks to prove this is a person (the browser shows it): the user can do that, then search again".into()));
                        } else {
                            let p = Mutex::new(proxy.clone());
                            std::thread::spawn(move || {
                                std::thread::sleep(Duration::from_millis(800));
                                let _ = p.lock().unwrap().send_event(Ev::PageRead(w, true));
                            });
                        }
                    }
                }
                for mut j in job_waiters.borrow_mut().drain(..) {
                    // where its own load landed: the tool acts there or not at all (a navigation in between is not it)
                    // ... unless another navigation the app knows of came between (then it lands nowhere sure)
                    j.args["__landed"] = if j.args["__epoch"].as_u64() == Some(nav_epoch.get()) { json!(url.clone()) } else { Value::Null };
                    let p = Mutex::new(proxy.clone());
                    std::thread::spawn(move || {
                        std::thread::sleep(Duration::from_millis(1500));
                        let _ = p.lock().unwrap().send_event(Ev::Tool(j));
                    });
                }
                // the page itself, not "opened, now read it": a small model skipped the read and answered from memory
                // (six opens, no read, no sources); a moment first, for what its scripts draw
                for w in load_waiters.borrow_mut().drain(..) {
                    let p = Mutex::new(proxy.clone());
                    std::thread::spawn(move || {
                        std::thread::sleep(Duration::from_millis(800));
                        let _ = p.lock().unwrap().send_event(Ev::PageRead(w, false));
                    });
                }
            }
            Event::UserEvent(Ev::PageRead(w, search)) => {
                let (seen, last) = (seen_links.clone(), last_read.clone());
                eval_reply(&web, &browser_mcp::read_js(1), w, move |v| {
                    remember_links(&seen, &v);
                    *last.lock().unwrap() = browser_mcp::page_text(&v);
                    Ok(browser_mcp::loaded_text(&v, search))
                });
            }
            Event::UserEvent(Ev::Tool(job)) => {
                // the model's browser tools work in the pane people see: it opens for them
                if !web_open.get() {
                    web_open.set(true);
                    layout(full.get());
                    js_call(&control_view, "webOpen", &json!(true));
                    js_call(&page, "strataWeb", &json!(true));
                }
                js_call(&web_bar, "busy", &json!(job.tool));
                let n = job.args["element"].as_i64().unwrap_or(0);
                // a load that never finishes still answers (the model reads what is there)
                let wait_load = |reply: std::sync::mpsc::Sender<Result<String, String>>, secs: u64| {
                    let late = reply.clone();
                    std::thread::spawn(move || {
                        std::thread::sleep(Duration::from_secs(secs));
                        let _ = late.send(Ok(format!("still loading after {secs} s: browser_read it in a moment")));
                    });
                    load_waiters.borrow_mut().push(reply);
                };
                let tool = job.tool.clone();
                let in_picker = job.args.get("__top").is_none() && web.url().map(|u| browser_mcp::picker_host(&u)).unwrap_or(false);
                match tool.as_str() {
                    // a table on this Mac (table.rs): no browser
                    t if t.starts_with("table_") && t != "table_sync" => {
                        let dir = st.lock().unwrap().root.join("../Strata-data/sheets");
                        let done = table::local_tool(&dir, t, &job.args);
                        // a change to a table with a sheet goes on to the sheet: a model added a row and imported the
                        // stale sheet into the map, table_sync never called (an error there leaves the file changed:
                        // the upsert again by Link changes nothing)
                        let name = job.args["name"].as_str().unwrap_or("").to_string();
                        match (&done, table::sheet_of(&dir, &name)) {
                            (Ok(said), Some(sheet)) if matches!(t, "table_upsert" | "table_delete") => {
                                let mut job = job;
                                job.tool = "table_sync".into();
                                let mut args = json!({"name": name, "url": sheet, "__note": format!("{said}\nThen synced to its sheet: ")});
                                if let Some(k) = job.args.get("key").filter(|k| k.is_string()) {
                                    args["key"] = k.clone();
                                }
                                if t == "table_upsert" {
                                    args["__rows"] = job.args["rows"].clone();
                                }
                                // the person's columns of the last sync (an empty list too: none of theirs)
                                if let Some(m) = table::mine_of(&dir, &name) {
                                    args["mine"] = json!(m);
                                }
                                job.args = args;
                                let _ = proxy.send_event(Ev::Tool(job));
                            }
                            _ => { let _ = job.reply.send(done); }
                        }
                    }
                    // Google Sheets / My Maps (google.rs): in their page, loaded first when the pane is elsewhere
                    t if google::is_google(t) => {
                        let mut job = job;
                        let here = web.url().unwrap_or_default();
                        // maps_export: where its files go, set by the app
                        if t == "maps_export" && job.args.get("__table").is_none() {
                            let data = st.lock().unwrap().root.join("../Strata-data");
                            let name = job.args["name"].as_str().unwrap_or("").to_string();
                            match (table::path_of(&data.join("maps"), &name), table::path_of(&data.join("sheets"), &name)) {
                                (Ok(m), Ok(tb)) => {
                                    job.args["__map"] = json!(m.with_extension("kml").to_string_lossy());
                                    job.args["__table"] = json!(tb.to_string_lossy());
                                }
                                (Err(e), _) | (_, Err(e)) => { let _ = job.reply.send(Err(e)); return; }
                            }
                        }
                        // a sheet with a local table (table_sync wrote the link): a change goes to the table, not the
                        // sheet (the next sync would undo it); a read says which table it is
                        if t.starts_with("sheets_") && t != "sheets_create" {
                            let dir = st.lock().unwrap().root.join("../Strata-data/sheets");
                            let url = job.args["url"].as_str().filter(|u| !u.is_empty()).unwrap_or(&here).to_string();
                            match table::sheet_guard(&dir, t, google::sheet_base(&url).as_deref()) {
                                Err(e) => { let _ = job.reply.send(Err(e)); return; }
                                Ok(Some(note)) => job.args["__note"] = json!(note),
                                Ok(None) => {}
                            }
                        }
                        // table_sync: its file, set by the app (a caller's __ keys are removed)
                        if t == "table_sync" && job.args.get("__table").is_none() {
                            let dir = st.lock().unwrap().root.join("../Strata-data/sheets");
                            match table::path_of(&dir, job.args["name"].as_str().unwrap_or("")).and_then(|p| table::load(&p).and_then(|t| table::fits_a_to_z(&t)).map(|_| p)) {
                                Ok(p) => job.args["__table"] = json!(p.to_string_lossy()),
                                Err(e) => { let _ = job.reply.send(Err(e)); return; }
                            }
                        }
                        // no url: the open sheet's own address (its ?gid= tab with it), not tab 0 by default
                        if t.starts_with("sheets_") && t != "sheets_create" && job.args["url"].as_str().is_none_or(str::is_empty) && google::sheet_base(&here).is_some() {
                            job.args["url"] = json!(here.clone());
                        }
                        match google::page_for(t, &job.args, &here) {
                            Err(e) => { let _ = job.reply.send(Err(e)); }
                            // exactly true, set by the Allow below (callers' __ keys are removed: browser_mcp::caller_args)
                            Ok(_) if google::ask(t, &job.args).is_some() && job.args["__allowed"] != true => {
                                next_ask.set(next_ask.get() + 1);
                                js_call(&web_bar, "ask", &json!({"id": next_ask.get(), "say": google::ask(t, &job.args)}));
                                eprintln!("strata-app: ask {}: {}", next_ask.get(), google::ask(t, &job.args).unwrap_or_default());
                                approvals.borrow_mut().insert(next_ask.get(), (std::time::Instant::now(), job));
                            }
                            // a page that comes by itself (the new map's editor, after Create): looked for each second
                            Ok((load, prefix)) if load.is_empty() && !google::on_page(&prefix, &here) => {
                                let tries = job.args["__tries"].as_u64().unwrap_or(0);
                                if tries >= 20 {
                                    let _ = job.reply.send(Err(format!("{t}: the page did not come ({prefix}): browser_read to see where the browser is")));
                                } else {
                                    job.args["__tries"] = json!(tries + 1);
                                    let p = Mutex::new(proxy.clone());
                                    std::thread::spawn(move || {
                                        std::thread::sleep(Duration::from_millis(1000));
                                        let _ = p.lock().unwrap().send_event(Ev::Tool(job));
                                    });
                                }
                            }
                            Ok((load, prefix)) if job.args.get("__loaded").is_none() && !google::on_page(&prefix, &here) => {
                                job.args["__loaded"] = json!(true);
                                // every load counts, this one too: its epoch is the one after its own (a browser_open
                                // overlapping a create was not counted: found by a codex review)
                                nav_epoch.set(nav_epoch.get() + 1);
                                job.args["__epoch"] = json!(nav_epoch.get());
                                // the document open before: a "new" sheet that is this one is not new
                                job.args["__before"] = json!(here.split('/').nth(5).unwrap_or(""));
                                let _ = web.load_url(&load);
                                job_waiters.borrow_mut().push(job);
                            }
                            Ok((_, prefix)) if !google::arrived(t, &prefix, &here) => {
                                let _ = job.reply.send(Err(format!("{t} was not run: the browser is at {here}, not the spreadsheet it was for (a sign-in, or another page while it loaded); browser_read it")));
                            }
                            Ok(_) => {
                                let js = if t.starts_with("maps_") { google::maps_js(t, &job.args, &frame_token) } else { google::sheets_js(t, &job.args) };
                                match js {
                                    Err(e) => { let _ = job.reply.send(Err(e)); }
                                    Ok(body) => {
                                        next_frame_op.set(next_frame_op.get() + 1);
                                        let id = next_frame_op.get();
                                        // one stop key and one watchdog per call, through all its phases and pages (a
                                        // second phase once outlived the first phase's watchdog: found by a codex review)
                                        let key = match job.args["__job"].as_u64() {
                                            Some(k) => k,
                                            None => {
                                                job.args["__job"] = json!(id);
                                                // whatever happens to the page, the call is answered (a lost answer
                                                // hung it 150 s); the script is told to stop before its next change
                                                let (late, tool, stop) = (job.reply.clone(), t.to_string(), Mutex::new(proxy.clone()));
                                                std::thread::spawn(move || {
                                                    std::thread::sleep(Duration::from_secs(75));
                                                    let _ = stop.lock().unwrap().send_event(Ev::StopScript(id));
                                                    let _ = late.send(Err(format!("{tool} did not answer in 75 s and was stopped; it may have done part of it: read the sheet or map before trying again")));
                                                });
                                                id
                                            }
                                        };
                                        let _ = web.evaluate_script(&google::async_js(id, key, &body));
                                        let p = Mutex::new(proxy.clone());
                                        std::thread::spawn(move || {
                                            std::thread::sleep(Duration::from_millis(1000));
                                            let _ = p.lock().unwrap().send_event(Ev::AsyncCollect(job, id, 0));
                                        });
                                    }
                                }
                            }
                        }
                    }
                    // My Maps' Import: the picker frame acts while it is open (its answer: Ev::FrameSent)
                    "browser_read" | "browser_click" | "browser_type" if in_picker => {
                        let (op, args) = match tool.as_str() {
                            "browser_read" => ("read", json!([job.args["part"].as_i64().unwrap_or(1), browser_mcp::PART])),
                            "browser_click" => ("click", json!([n])),
                            _ => ("describe", json!([n])),
                        };
                        next_frame_op.set(next_frame_op.get() + 1);
                        frame_send(&web, &proxy, &frame_token, next_frame_op.get(), job, op, args);
                    }
                    "browser_open" => {
                        let url = web_url(job.args["url"].as_str().unwrap_or(""));
                        if searched.get().is_some_and(|t| t.elapsed() < Duration::from_secs(600)) && !link_seen(&seen_links.lock().unwrap(), &url) {
                            // said as a result, not an error: the browser held a guess back on purpose (the chat showed red errors)
                            let _ = job.reply.send(Ok(format!("not opened: {} is a site none of the results or pages you read showed (guessed addresses are usually 404s). Use browser_click with a result's number, or browser_search again.", site_of(&url))));
                        } else {
                            nav_epoch.set(nav_epoch.get() + 1);
                            let _ = web.load_url(&url);
                            wait_load(job.reply, 20);
                        }
                    }
                    // a developer's probes of a page (STRATA_BROWSER_EVAL=1 at the app's start; never in the model's list)
                    t if t.starts_with("dev_") && std::env::var_os("STRATA_BROWSER_EVAL").is_some() => {
                        let a = &job.args;
                        let s = |k: &str| a[k].as_str().unwrap_or("").to_string();
                        match t {
                            "dev_eval" => {
                                let r = job.reply;
                                let _ = web.evaluate_script_with_callback(&s("js"), move |v| { let _ = r.send(Ok(v)); });
                            }
                            _ => { let _ = job.reply.send(Err(format!("no dev tool {t}"))); }
                        }
                    }
                    "browser_back" => {
                        nav_epoch.set(nav_epoch.get() + 1);
                        let _ = web.evaluate_script("history.back()");
                        wait_load(job.reply, 8);
                    }
                    "browser_read" => {
                        let (seen, last) = (seen_links.clone(), last_read.clone());
                        eval_reply(&web, &browser_mcp::read_js(job.args["part"].as_i64().unwrap_or(1)), job.reply, move |v| {
                            remember_links(&seen, &v);
                            Ok(browser_mcp::once(&mut last.lock().unwrap(), browser_mcp::page_text(&v)))
                        });
                    }
                    "browser_search" => {
                        searched.set(Some(Instant::now()));
                        // Google's results, read when the page has loaded (no full page read for a search)
                        nav_epoch.set(nav_epoch.get() + 1);
                        let _ = web.load_url(&web_url(&format!("{} ", job.args["query"].as_str().unwrap_or("")).replace('.', " ")));
                        let late = job.reply.clone();
                        std::thread::spawn(move || {
                            std::thread::sleep(Duration::from_secs(15));
                            let _ = late.send(Ok("the results did not load in 15 s: browser_read the page".into()));
                        });
                        search_waiters.borrow_mut().push(job.reply);
                    }
                    "browser_click" => {
                        nav_epoch.set(nav_epoch.get() + 1);   // a link it clicks may load a page
                        // where the element is, then a real click there (Ev::Native)
                        let (p, slot) = (Mutex::new(proxy.clone()), Mutex::new(Some(job)));
                        let _ = web.evaluate_script_with_callback(&browser_mcp::target_js(n), move |r| {
                            if let Some(j) = slot.lock().unwrap().take() {
                                let _ = p.lock().unwrap().send_event(Ev::Native(j, "click", r));
                            }
                        });
                    }
                    "browser_scroll" => {
                        let js = format!("window.scrollBy(0, {}innerHeight * 0.8); JSON.stringify({{y: Math.round(scrollY), height: document.documentElement.scrollHeight}})",
                                         if job.args["direction"] == "up" { "-" } else { "" });
                        act_then_read(&web, &js, job.reply, &proxy, 800);
                    }
                    "browser_type" => {
                        let (p, slot) = (proxy.clone(), Mutex::new(Some(job)));
                        let _ = web.evaluate_script_with_callback(&browser_mcp::describe_js(n), move |r| {
                            if let Some(j) = slot.lock().unwrap().take() {
                                let _ = p.send_event(Ev::Described(j, r));
                            }
                        });
                    }
                    other => {
                        let _ = job.reply.send(Err(format!("no browser tool {other}")));
                    }
                }
            }
            Event::UserEvent(Ev::FrameSent(mut job, op, id, r)) => {
                if js_value(&r)["sent"] == true {
                    let p = Mutex::new(proxy.clone());
                    std::thread::spawn(move || {
                        std::thread::sleep(Duration::from_millis(if op == "read" || op == "describe" { 300 } else { 1800 }));
                        let _ = p.lock().unwrap().send_event(Ev::FrameCollect(job, op, id, 0));
                    });
                } else {
                    // no picker open (it closed): the page acts, as for any page
                    job.args["__top"] = json!(true);
                    let _ = proxy.send_event(Ev::Tool(job));
                }
            }
            Event::UserEvent(Ev::StopScript(id)) => {
                let _ = web.evaluate_script(&format!("(window.__strataStop = window.__strataStop || {{}})[{id}] = true"));
            }
            Event::UserEvent(Ev::Native(job, what, r)) => {
                let v = js_value(&r);
                let n = job.args["element"].as_i64().unwrap_or(0);
                if let Some(e) = v["error"].as_str() {
                    let _ = job.reply.send(Err(e.to_string()));
                } else if what == "click" && v["hidden"] == true {
                    // nowhere a mouse can reach it (no size, off screen): clicked by the page's script instead
                    load_waiters.borrow_mut().push(job.reply.clone());
                    act_then_read(&web, &browser_mcp::click_js(n), job.reply, &proxy, 2000);
                } else if what == "click" {
                    if let Some(c) = v["cover"].as_str() {
                        say_web(&format!("[web] the click on [{n}] lands on \"{c}\" (on top of it)"));
                    }
                    match native_input::click(&web, v["x"].as_f64().unwrap_or(0.0), v["y"].as_f64().unwrap_or(0.0)) {
                        Err(e) => { let _ = job.reply.send(Err(e)); }
                        Ok(()) => {
                            load_waiters.borrow_mut().push(job.reply.clone());
                            read_later(job.reply, &proxy, 2000);
                        }
                    }
                } else {
                    let text = job.args["text"].as_str().unwrap_or("").to_string();
                    let submit = job.args["submit"] == true;
                    // only into the field that took the focus (else the text went where the focus was before)
                    let typed = if v["focused"] != true {
                        Err(format!("field [{n}] did not take the focus (disabled or covered?): browser_read the page again"))
                    } else {
                        native_input::type_text(&web, &text).and_then(|_| if submit { native_input::press(&web, "enter", &[]) } else { Ok(()) })
                    };
                    match typed {
                        Err(e) => { let _ = job.reply.send(Err(e)); }
                        Ok(()) if submit => {
                            load_waiters.borrow_mut().push(job.reply.clone());
                            read_later(job.reply, &proxy, 2500);
                        }
                        Ok(()) => read_later(job.reply, &proxy, 600),
                    }
                }
            }
            Event::UserEvent(Ev::AsyncCollect(job, id, tries)) => {
                let (p, slot) = (Mutex::new(proxy.clone()), Mutex::new(Some(job)));
                let _ = web.evaluate_script_with_callback(&google::async_collect_js(id), move |r| {
                    let Some(job) = slot.lock().unwrap().take() else { return };
                    let v = js_value(&r);
                    let p = p.lock().unwrap().clone();
                    if let Some(to) = v["gone"].as_str() {
                        // the page was replaced under the script: once more from the start where it went, when that
                        // is safe; else said (a create or an append is never repeated blindly)
                        let mut job = job;
                        if job.tool == "maps_create" && job.args["__phase"] != 2 {
                            // phase 1's Create took the page before its answer was read: the new map's editor next
                            job.args["__phase"] = json!(2);
                            let _ = p.send_event(Ev::Tool(job));
                        } else if google::rerun_safe(&job.tool, to) && job.args.get("__rerun").is_none() {
                            job.args["__rerun"] = json!(true);
                            std::thread::spawn(move || {
                                std::thread::sleep(Duration::from_millis(1500));
                                let _ = p.send_event(Ev::Tool(job));
                            });
                        } else {
                            let _ = job.reply.send(Err(format!("{}: the page changed while it ran (now {to}); browser_read it, and check before doing it again", job.tool)));
                        }
                    } else if v.is_null() {
                        if tries >= 120 {
                            // stopped as it is said to have failed (the 75 s watchdog came later, and the work went on)
                            let _ = p.send_event(Ev::StopScript(job.args["__job"].as_u64().unwrap_or(id)));
                            let _ = job.reply.send(Err(format!("{} did not finish in 60 s and was stopped; it may have done part of it: read the sheet or map before trying again", job.tool)));
                        } else {
                            std::thread::spawn(move || {
                                std::thread::sleep(Duration::from_millis(500));
                                let _ = p.send_event(Ev::AsyncCollect(job, id, tries + 1));
                            });
                        }
                    } else if job.tool == "table_sync" && job.args["__phase"] != 2 && v["ok"].is_string() {
                        // step 1 read the sheet: the person's columns into the file, then step 2 puts the file in it
                        let mut job = job;
                        let pulled: Value = serde_json::from_str(v["ok"].as_str().unwrap_or("")).unwrap_or_default();
                        let path = std::path::PathBuf::from(job.args["__table"].as_str().unwrap_or(""));
                        let key = job.args["key"].as_str().filter(|k| !k.is_empty()).unwrap_or("Link").to_string();
                        let mine: Vec<String> = job.args["mine"].as_array().map(|m| m.iter().filter_map(Value::as_str).map(String::from).collect())
                            .unwrap_or_else(|| vec!["Checkbox".into(), "Confirmation".into()]);
                        let step = table::load(&path).and_then(|mut t| {
                            if t.is_empty() { return Err(format!("no table {} yet: table_upsert with columns makes it", path.display())); }
                            let (took, added) = table::merge(&mut t, pulled["pulled"].as_str().unwrap_or(""), &key, &mine, &table::tombstones(&path))?;
                            // after a table_upsert (the app synced for it): what it set wins over the sheet's old
                            // value in those rows and columns, the person's edits elsewhere are kept
                            if let Some(rows) = job.args["__rows"].as_array() {
                                table::upsert(&mut t, rows, &key, None)?;
                            }
                            table::save(&path, &t)?;
                            Ok((t, took, added))
                        });
                        match step {
                            Err(e) => { let _ = job.reply.send(Err(format!("table_sync: {e}"))); }
                            Ok((t, took, added)) => {
                                job.args["__push"] = json!(table::sync_push_js(&t, pulled["rows"].as_u64().unwrap_or(0) as usize));
                                // the tab it read is the tab it writes and the one saved with the link (a bare address
                                // took whichever tab was open: a later sync could write another)
                                let url = job.args["url"].as_str().unwrap_or("").to_string();
                                if let (None, Some(g), Some(b)) = (google::gid_of(&url), pulled["gid"].as_str(), google::sheet_base(&url)) {
                                    job.args["url"] = json!(format!("{b}/edit#gid={g}"));
                                }
                                job.args["__took"] = json!(format!("{took} of your cells and {added} rows you added came from the sheet into the file; "));
                                job.args["__phase"] = json!(2);
                                let _ = p.send_event(Ev::Tool(job));
                            }
                        }
                    } else if v["ok"] == "__next" {
                        // the script's click took the page away: the next phase where the page goes
                        let mut job = job;
                        job.args["__phase"] = json!(2);
                        let _ = p.send_event(Ev::Tool(job));
                    } else if let Some(e) = v["error"].as_str() {
                        let _ = job.reply.send(Err(format!("{}: {e}", job.tool)));
                    } else if job.tool == "maps_export" && v["ok"].is_string() {
                        let got: Value = serde_json::from_str(v["ok"].as_str().unwrap_or("")).unwrap_or_default();
                        let _ = job.reply.send(save_map_export(&job.args, got["kml"].as_str().unwrap_or(""), got["title"].as_str().unwrap_or(""), &got["places"]));
                    } else if job.tool == "table_sync" {
                        // the table knows its sheet now: changes to the sheet are sent to the table from here on
                        let path = std::path::PathBuf::from(job.args["__table"].as_str().unwrap_or(""));
                        let url = job.args["url"].as_str().unwrap_or("");
                        // the tab too: a sync for the table goes to that tab, not to the one open
                        let mine: Vec<String> = job.args["mine"].as_array().map(|m| m.iter().filter_map(Value::as_str).map(String::from).collect())
                            .unwrap_or_else(|| vec!["Checkbox".into(), "Confirmation".into()]);
                        let linked = google::sheet_base(url).map(|b| table::set_sheet(&path, &format!("{b}{}", google::gid_of(url).map(|g| format!("#gid={g}")).unwrap_or_default()), &mine));
                        let note = if let Some(Err(e)) = linked { format!(" (the link to the sheet was not saved: {e})") } else { String::new() };
                        let _ = job.reply.send(Ok(format!("{}{}{}{note}", job.args["__note"].as_str().unwrap_or(""), job.args["__took"].as_str().unwrap_or(""), v["ok"].as_str().unwrap_or(""))));
                    } else {
                        let _ = job.reply.send(Ok(format!("{}{}", job.args["__note"].as_str().unwrap_or(""), google::answer(&job.tool, &job.args, &v["ok"]))));
                    }
                });
            }
            Event::UserEvent(Ev::FrameCollect(job, op, id, tries)) => {
                let (p, slot) = (Mutex::new(proxy.clone()), Mutex::new(Some((job, op))));
                let _ = web.evaluate_script_with_callback(&browser_mcp::frame_collect_js(id), move |r| {
                    let Some((mut job, op)) = slot.lock().unwrap().take() else { return };
                    let v = js_value(&r);
                    let p = p.lock().unwrap().clone();
                    if v.is_null() {
                        if tries >= 20 {
                            let _ = job.reply.send(Err("the file picker did not answer: browser_read the page".into()));
                        } else {
                            std::thread::spawn(move || {
                                std::thread::sleep(Duration::from_millis(300));
                                let _ = p.send_event(Ev::FrameCollect(job, op, id, tries + 1));
                            });
                        }
                    } else if op == "describe" {
                        // the typing's Allow, as on a page; the typing then goes to the picker
                        job.args["__frame"] = json!(true);
                        let _ = p.send_event(Ev::Described(job, v.to_string()));
                    } else {
                        let page = if op == "read" { v.clone() } else { v["read"].clone() };
                        let _ = job.reply.send(match v["error"].as_str().or(v["acted"]["error"].as_str()) {
                            Some(e) => Err(e.to_string()),
                            None => Ok(format!("[Google's file picker, open over the page: browser_read / click / type act in it]\n\n{}", browser_mcp::page_text(&page))),
                        });
                    }
                });
            }
            Event::UserEvent(Ev::Described(job, r)) => {
                let v = js_value(&r);
                if let Some(e) = v["error"].as_str() {
                    let _ = job.reply.send(Err(e.to_string()));
                } else if v["password"] == true {
                    let _ = job.reply.send(Err("a password field: the model never types there; the user signs in themselves".into()));
                } else {
                    // asked in the browser's strip; the answer comes back as "web_allow"
                    next_ask.set(next_ask.get() + 1);
                    js_call(&web_bar, "ask", &json!({"id": next_ask.get(), "text": job.args["text"], "field": v["field"], "host": v["host"], "submit": job.args["submit"] == true}));
                    approvals.borrow_mut().insert(next_ask.get(), (std::time::Instant::now(), job));
                }
            }
            Event::UserEvent(Ev::WebText(r)) => {
                // the script's value comes JSON-encoded (a string holding the JSON it built)
                let v = js_value(&r);
                if v.is_object() {
                    let title = v["title"].as_str().filter(|t| !t.is_empty()).unwrap_or("web page");
                    let text = format!("{}\n{}\n\n{}{}", title, v["url"].as_str().unwrap_or(""), v["text"].as_str().unwrap_or(""),
                                       if v["cut"] == true { "\n\n[the page goes on: cut at 20,000 characters]" } else { "" });
                    let name: String = title.chars().filter(|c| !"/\\:".contains(*c)).take(60).collect();
                    js_call(&page, "strataAttach", &json!([format!("{name}.txt"), text]));
                }
            }
            Event::UserEvent(Ev::Log(line)) => {
                st.lock().unwrap().push_log(line.clone());
                js_call(&control_view, "log", &json!(line));
                if let Some(why) = trouble(&line) {
                    check_memory(&st, &proxy, Some(why.into()));
                }
            }
            Event::UserEvent(Ev::Memory(m)) => js_call(&control_view, "memory", &m),
            Event::UserEvent(Ev::Apps(a)) => js_call(&control_view, "apps", &a),
            Event::UserEvent(Ev::System(v)) => js_call(&control_view, "system", &v),
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
            // a quit from outside (the Dock, logging out, AppleScript's quit) ends the loop without Ev::Quit: the
            // server would outlive the app, holding the model's memory
            Event::LoopDestroyed => stop_and_wait(&st),
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

    #[test]
    fn lists_the_apps_using_memory() {
        let (apps, _) = memory_apps();
        for w in apps.windows(2) {
            assert!(w[0]["bytes"].as_u64() >= w[1]["bytes"].as_u64());
        }
        assert!(apps.iter().all(|a| !a["bundle"].as_str().unwrap().starts_with("/System/") && a["name"] != "Strata"));
        eprintln!("{}", apps.iter().map(|a| format!("{} {:.1} GB", a["name"].as_str().unwrap(), a["bytes"].as_f64().unwrap() / 1e9)).collect::<Vec<_>>().join(", "));
    }

    #[test]
    fn parallel_is_written_into_the_config_and_taken_out_at_one() {
        let d = std::env::temp_dir().join(format!("strata-par-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("strata-x.json");
        std::fs::write(&p, "{\n \"args\": [\"--gguf\", \"m\"],\n \"api_key\": \"k\"\n}\n").unwrap();
        set_parallel(&p, 3).unwrap();
        assert_eq!(read_config(&p)["parallel"], 3);
        assert!(std::fs::read_to_string(&p).unwrap().contains("\n \"api_key\": \"k\""));   // the rest and its indent kept
        set_parallel(&p, 12).unwrap();
        assert_eq!(read_config(&p)["parallel"], 8);                                  // the engine's most
        set_parallel(&p, 1).unwrap();
        assert!(read_config(&p).get("parallel").is_none());                          // one at a time: no key at all
        assert_eq!(plan_args(&read_config(&p)), ["--gguf", "m"]);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn the_conversation_cache_is_set_in_args_zero_is_off_and_none_is_automatic() {
        let d = std::env::temp_dir().join(format!("strata-cache-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("strata-x.json");
        std::fs::write(&p, "{\n \"args\": [\"--gguf\", \"m\", \"--conversation-cache-mib\", \"4096\", \"--kv\", \"f16\"]\n}\n").unwrap();
        assert_eq!(cache_mib(&read_config(&p)), 4096);
        set_cache_mib(&p, 16384).unwrap();                                   // replaced, never added twice
        assert_eq!(read_config(&p)["args"], json!(["--gguf", "m", "--kv", "f16", "--conversation-cache-mib", "16384"]));
        assert_eq!(cache_mib(&read_config(&p)), 16384);
        set_pool(&p, 294912).unwrap();                                       // a pool: both flags, once
        set_pool(&p, 393216).unwrap();
        assert_eq!(pool_of(&read_config(&p)).0, 393216);
        assert_eq!(read_config(&p)["args"].as_array().unwrap().iter().filter(|a| *a == "--kv-unified").count(), 1);
        set_pool(&p, 0).unwrap();
        assert_eq!(pool_of(&read_config(&p)).0, 0);
        assert!(!read_config(&p)["args"].as_array().unwrap().iter().any(|a| a == "--kv-unified" || a == "--kv-pool"));
        set_cache_mib(&p, 0).unwrap();                                       // off: "0", or the server would size one
        assert_eq!(read_config(&p)["args"], json!(["--gguf", "m", "--kv", "f16", "--conversation-cache-mib", "0"]));
        assert_eq!(cache_mib(&read_config(&p)), 0);
        set_cache_mib(&p, -1).unwrap();                                      // automatic: no flag at all
        assert_eq!(read_config(&p)["args"], json!(["--gguf", "m", "--kv", "f16"]));
        assert_eq!(cache_mib(&read_config(&p)), -1);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn the_memory_check_counts_parallel_slots() {
        let c = |v: Value| plan_args(&v);
        assert_eq!(c(json!({"args": ["--serve", "--gguf", "m"], "parallel": 3})), ["--gguf", "m", "--batch", "3"]);
        assert_eq!(c(json!({"args": ["--gguf", "m"], "parallel": 20}))[3], "8");                 // the engine's most
        assert_eq!(c(json!({"args": ["--gguf", "m", "--batch", "2"], "parallel": 3})), ["--gguf", "m", "--batch", "2"]);
        for p in [json!(1), json!(false), Value::Null, json!("3")] {                                // one request at a time
            assert_eq!(c(json!({"args": ["--gguf", "m"], "parallel": p})), ["--gguf", "m"]);
        }
    }

    #[test]
    fn reads_the_mac_memory_like_activity_monitor() {
        let vm = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:     1460547.\nPages wired down:   534861.\n\
                  Pages purgeable:   17820.\nFile-backed pages:  2135266.\nAnonymous pages:  2898072.\n\
                  Pages stored in compressor:  4012441.\nPages occupied by compressor:  1274808.\n";
        let m = sys_memory_of(vm, "77\n137438953472\ntotal = 17408.00M  used = 15980.56M  free = 1427.44M  (encrypted)\n");
        assert_eq!(m["app"], (2898072u64 - 17820) * 16384);              // anonymous less purgeable, not "stored in"
        assert_eq!(m["wired"], 534861u64 * 16384);
        assert_eq!(m["compressed"], 1274808u64 * 16384);                // the compressor's own pages
        assert_eq!(m["pressure"], 23);
        assert_eq!(m["free"], 137438953472u64 - (2898072 - 17820 + 534861 + 1274808) * 16384);
        assert_eq!(m["swap_used"], (15980.56f64 * 1048576.0) as u64);
        assert_eq!(m["swap_total"], 17408u64 * 1048576);
        let none = sys_memory_of("", "");                               // sysctl or vm_stat missing: nothing made up
        assert_eq!(none["pressure"], Value::Null);
        assert_eq!(none["free"], 0);
        let live = sys_memory();
        assert!(live["total"].as_u64().unwrap() > 0 && live["pressure"].as_u64().unwrap() <= 100, "{live}");
    }

    #[test]
    fn reads_the_system_state() {
        let v = system_state();
        for k in ["keep_ai_off", "ai_running", "siri", "keep_still_wallpaper", "aerial_running", "wired_mb", "suggest_mb"] {
            assert!(v.get(k).is_some(), "{k} missing: {v}");
        }
        eprintln!("{v}");
    }

    #[test]
    fn a_config_too_big_is_fitted_before_it_starts() {
        let opts = |o: Value| json!({"state": "red", "options": o});
        let fit = json!([{"ctx": 4096, "gpu_fits": true}, {"ctx": 8192, "gpu_fits": true}, {"ctx": 16384, "gpu_fits": false}]);
        assert_eq!(fit_fix(&json!({"state": "green", "options": fit.clone()})), None);
        assert_eq!(fit_fix(&json!({"state": "amber", "options": fit.clone()})), None);
        // the largest the GPU holds, even when none fits the memory free right now ("fits" false everywhere)
        assert_eq!(fit_fix(&opts(fit)), Some((8192, None)));
        assert_eq!(fit_fix(&opts(json!([{"ctx": 4096, "gpu_fits": true, "gpu_batch": 512}]))), Some((4096, Some(512))));
        assert_eq!(fit_fix(&opts(json!([{"ctx": 8192, "gpu_fits": true, "gpu_batch": 256}]))), Some((8192, Some(256))));
        assert_eq!(fit_fix(&opts(json!([{"ctx": 4096, "gpu_fits": false}]))), None);
    }

    #[test]
    fn the_wire_cap_follows_the_gpu_limit() {
        let gb16 = 17_179_869_184;
        // the 16 GB M1 at 13,312 MB: 15.03 GB (2 GiB stay unpinned), above macOS's 13.06 GB
        assert_eq!(wire_limit_for(13312, gb16), gb16 - (2 << 30));
        // a 64 GB Mac at 48 GB: the GPU's 48 GiB + 1.5 GB
        assert_eq!(wire_limit_for(49152, 68_719_476_736), 49152 * 1048576 + 1_500_000_000);
    }

    #[test]
    fn addresses_compare_without_their_dressing() {
        for (a, b) in [("https://www.example.com/a/", "http://example.com/a"), ("HTTPS://Example.com/A#top", "example.com/a"),
                       ("https://example.com/a?x=1", "example.com/a?x=1")] {
            assert_eq!(link_key(a), link_key(b), "{a} / {b}");
        }
        assert_ne!(link_key("https://example.com/a?x=1"), link_key("https://example.com/a?x=2")); // a query is the page
        assert_eq!(site_of("example.com"), "example.com");
        assert_eq!(site_of("https://www.sub.example.co.uk/p?q#h"), "sub.example.co.uk");
        assert_eq!(site_of(""), "");
    }

    #[test]
    fn after_a_search_only_seen_addresses_open() {
        let seen: Mutex<std::collections::HashSet<String>> = Default::default();
        // what Google's plain results page gives: redirect links, and the sites in the text
        remember_links(&seen, &json!({"hrefs": ["https://www.google.com/url?q=CAESXgHrOzAV", "https://en.wikipedia.org/wiki/Paris#Climate"],
                                      "sites": ["www.bbc.com", "theweathernetwork.com"]}));
        let s = seen.lock().unwrap();
        assert!(link_seen(&s, "https://www.bbc.com/weather/2988507"));     // its site showed: opened (was refused)
        assert!(link_seen(&s, "http://theweathernetwork.com/en/city/fr/paris"));
        assert!(link_seen(&s, "https://en.wikipedia.org/wiki/Paris"));     // an anchor: the same page
        assert!(!link_seen(&s, "https://www.weatherbug.com/weather-forecast/paris")); // a site no page showed
        assert!(!link_seen(&s, "https://www.espn.com/racing/results"));
        assert_eq!(site_of("https://www.BBC.com:443/x"), "bbc.com");
        assert!(link_seen(&s, "https://weather.bbc.com/paris"));           // a part of a site that showed
        assert!(!link_seen(&s, "https://bbc.co.uk/weather"));              // another site, however alike
        assert!(!link_seen(&s, "https://com/x"));                          // never a bare top-level domain
    }

    #[test]
    fn the_browser_says_it_is_safari() {
        assert_eq!(safari_user_agent("27.0.1"), "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/27.0.1 Safari/605.1.15");
        let v = safari_version();
        assert!(!v.is_empty() && v.split('.').all(|n| n.parse::<u32>().is_ok()), "{v}");
    }

    #[test]
    fn the_address_field_takes_addresses_hosts_and_searches() {
        assert_eq!(web_url("https://example.com/a?b=1"), "https://example.com/a?b=1");
        assert_eq!(web_url("  news.ycombinator.com "), "https://news.ycombinator.com");
        assert_eq!(web_url("rust tokio select"), "https://www.google.com/search?q=rust+tokio+select");
        assert_eq!(web_url("C++ & Rust?"), "https://www.google.com/search?q=C%2B%2B+%26+Rust%3F");
    }

    #[test]
    fn reads_the_power_mode() {
        let m5 = power_from(" lowpowermode         0\n powermode            2\n", "Now drawing from 'AC Power'");
        assert_eq!(m5, json!({"low": false, "high_capable": true, "high": true, "battery": false}));
        let m1 = power_from(" lowpowermode         1\n", "Now drawing from 'Battery Power'");
        assert_eq!(m1, json!({"low": true, "high_capable": false, "high": false, "battery": true}));
        let auto = power_from(" powermode            0\n", "");
        assert_eq!(auto["high_capable"], true);
        assert_eq!(auto["high"], false);
    }
}
