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

/// The apps using the most memory now, the ones a person can quit to give the model room: each app's processes added
/// up (Chrome's helpers count as Chrome), from top's MEM (it counts compressed memory, unlike ps's RSS) and ps's paths.
/// macOS's own (/System) and Strata are left out.  [{name, bundle, pid, bytes}], largest first, 150 MB or more.
fn memory_apps() -> Vec<Value> {
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
    v.into_iter().take(8).map(|(bundle, bytes, pid)| {
        let app = bundle.ends_with(".app");
        let name = Path::new(&bundle).file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        json!({"name": name, "bundle": bundle, "pid": pid, "bytes": bytes, "app": app})
    }).collect()
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
            // the GPU's share alone (what fit_fix needs: memory free now changes once macOS drops its caches)
            let gpu512 = gpu > limit && saved_512 > 0.0 && gpu - saved_512 <= limit;
            json!({"ctx": c, "need": if ok512 { total - saved_512 } else { total }, "fits": ok || ok512, "batch512": ok512,
                   "gpu_fits": gpu <= limit || gpu512, "gpu_batch512": gpu512})
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
    // 1 GB of the limit stays for the others the GPU holds memory for (the window server's buffers, Metal itself): a
    // 16 GB M1 at a 13 GB limit ran out at 12.8 GiB and answered at 11.7
    const OTHERS: f64 = 1073741824.0;
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
    let daemon_mb = std::fs::read_to_string(GPU_DAEMON).ok().and_then(|t| t.split("iogpu.wired_limit_mb=").nth(1).and_then(|r| r.split('<').next()?.trim().parse::<u64>().ok()));
    let (_, _, suggest) = gpu_limit(f64::MAX);
    json!({"keep_ai_off": settings["keep_ai_off"] == true, "ai_running": ai, "ai_bytes": ai_b, "siri": siri,
           "keep_still_wallpaper": settings["keep_still_wallpaper"] == true, "aerial_running": aer.len(), "aerial_bytes": aer_b,
           "gpu_daemon_mb": daemon_mb, "wired_mb": wired, "suggest_mb": suggest, "power": power()})
}

fn still_wallpaper() {
    let script = format!("tell application \"System Events\" to tell every desktop to set picture to POSIX file \"{STILL_WALLPAPER}\"");
    let _ = Command::new("/usr/bin/osascript").args(["-e", &script]).status();
    std::thread::sleep(Duration::from_secs(2));
    end(AERIAL_PROC);
}

/// Apply a toggle.  Apple Intelligence's own switch is the user's (System Settings opens at it); Siri's preference
/// and the on-device model's processes this app handles.  The GPU limit: a LaunchDaemon that sets it at every boot.
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
<string>iogpu.wired_limit_mb={mb}</string></array><key>RunAtLoad</key><true/></dict></plist>\n");
                let tmp = std::env::temp_dir().join("us.tottle.strata.gpu-limit.plist");
                std::fs::write(&tmp, plist).map_err(|e| e.to_string())?;
                format!("launchctl bootout system/{GPU_LABEL} 2>/dev/null; install -m 644 -o root -g wheel '{}' {GPU_DAEMON} && launchctl bootstrap system {GPU_DAEMON}", tmp.display())
            } else {
                format!("launchctl bootout system/{GPU_LABEL} 2>/dev/null; rm -f {GPU_DAEMON}; sysctl iogpu.wired_limit_mb=0")
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
/// A config too big for the GPU's share (memory()'s "red") -> the largest context that fits, and the prefill batch it
/// needs.  Run as it is, the model leaves the window server too little of the GPU's memory: the screen glitches
/// (WindowServer: kIOGPUCommandBufferCallbackErrorOutOfMemory) and the answers slow down (a 16 GB M1 at 16K: 10.6-15
/// tok/s with 0.34 GB left over).
fn fit_fix(m: &Value) -> Option<(u64, Option<u64>)> {
    if m["state"] != "red" {
        return None;
    }
    let r = m["options"].as_array()?.iter().rev().find(|o| o["gpu_fits"] == true)?;
    Some((r["ctx"].as_u64()?, (r["gpu_batch512"] == true).then_some(512)))
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
                eprintln!("strata-app: page {event} {url}"); // a trace a run from Terminal shows (no window needed)
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
    std::thread::spawn(keep_off); // what the System card's toggles keep off
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
                "raise_limit" | "power_high" | "low_power_off" => {
                    let shell = match cmd {
                        "raise_limit" => format!("sysctl iogpu.wired_limit_mb={}", v["mb"].as_u64().unwrap_or(0)),
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
                        let _ = p.send_event(Ev::Apps(json!({"apps": memory_apps()})));
                    });
                }
                "quit_app" => {
                    let (p, st2) = (proxy.clone(), st.clone());
                    let (bundle, pid, force) = (v["bundle"].as_str().unwrap_or("").to_string(), v["pid"].as_u64().unwrap_or(0) as u32, v["force"] == true);
                    std::thread::spawn(move || {
                        let still = !bundle.is_empty() && pid > 0 && quit_app(&bundle, pid, force);
                        let _ = p.send_event(Ev::Apps(json!({"apps": memory_apps(), "still": if still { json!(bundle) } else { Value::Null }})));
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
        let apps = memory_apps();
        for w in apps.windows(2) {
            assert!(w[0]["bytes"].as_u64() >= w[1]["bytes"].as_u64());
        }
        assert!(apps.iter().all(|a| !a["bundle"].as_str().unwrap().starts_with("/System/") && a["name"] != "Strata"));
        eprintln!("{}", apps.iter().map(|a| format!("{} {:.1} GB", a["name"].as_str().unwrap(), a["bytes"].as_f64().unwrap() / 1e9)).collect::<Vec<_>>().join(", "));
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
        assert_eq!(fit_fix(&opts(json!([{"ctx": 4096, "gpu_fits": true, "gpu_batch512": true}]))), Some((4096, Some(512))));
        assert_eq!(fit_fix(&opts(json!([{"ctx": 4096, "gpu_fits": false}]))), None);
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
