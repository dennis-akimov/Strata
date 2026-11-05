//! `Strata --deploy` (make app-run): put the build just made in place without cutting off a request, and with one
//! Strata.app at a time.  The running app switches its server (SIGUSR1, main.rs switch_server, nginx's HUP way): the
//! new server starts with --handoff beside the old one (one port, SO_REUSEPORT: macOS gives the newest listener the new
//! connections), loads its model, takes the port and has the old server finish its requests and stop (SIGUSR2).  When
//! the old server cannot share its port or a second engine does not fit in memory: after the model is idle, a stop
//! and a start.  When the app's own code changed (its build's fingerprint, written at its start, differs from the one
//! on disk): after the model is idle, the app quits and the new one opens.  docs/BLUE_GREEN.md.

use serde_json::Value;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

/// a request is being read or answered, or waits for its turn
pub fn busy(m: &Value) -> bool {
    let l = &m["live"];
    let n = |k: &str| l[k].as_i64().unwrap_or(0);
    matches!(l["state"].as_str(), Some("reading" | "generating")) || n("running") > 0 || n("queued") > 0 || n("waiting") > 0
}

/// a second engine beside the first: its context and work buffers (the weights are the same mapped file), with room
/// to spare in both RAM and the GPU's share of it
pub fn fits(m: &Value, model_bytes: u64) -> bool {
    let (e, h) = (&m["engine"], &m["hardware"]);
    let u = |v: &Value| v.as_u64().unwrap_or(0);
    let need = u(&e["gpu_used"]).saturating_sub(model_bytes) as f64 * 1.25;
    let ram_free = u(&h["ram_total"]).saturating_sub(u(&h["ram_used"])) as f64;
    let gpu_free = u(&h["gpu_mem_total"]).saturating_sub(u(&h["gpu_mem_used"])) as f64;
    need > 0.0 && need <= ram_free && need <= gpu_free
}

/// Can a socket bind beside the server's: only when that one set SO_REUSEPORT too.  Bound, never listening, closed:
/// it takes no connection.
fn can_share(host: &str, port: u16) -> bool {
    let ip: std::net::Ipv4Addr = if host.is_empty() || host == "0.0.0.0" { std::net::Ipv4Addr::UNSPECIFIED } else { host.parse().unwrap_or(std::net::Ipv4Addr::LOCALHOST) };
    unsafe {
        let fd = libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0);
        if fd < 0 {
            return false;
        }
        let one: libc::c_int = 1;
        let sz = std::mem::size_of::<libc::c_int>() as libc::socklen_t;
        libc::setsockopt(fd, libc::SOL_SOCKET, libc::SO_REUSEADDR, &one as *const _ as *const libc::c_void, sz);
        libc::setsockopt(fd, libc::SOL_SOCKET, libc::SO_REUSEPORT, &one as *const _ as *const libc::c_void, sz);
        let mut a: libc::sockaddr_in = std::mem::zeroed();
        a.sin_len = std::mem::size_of::<libc::sockaddr_in>() as u8;
        a.sin_family = libc::AF_INET as libc::sa_family_t;
        a.sin_port = port.to_be();
        a.sin_addr.s_addr = u32::from(ip).to_be();
        let ok = libc::bind(fd, &a as *const _ as *const libc::sockaddr, std::mem::size_of::<libc::sockaddr_in>() as libc::socklen_t) == 0;
        libc::close(fd);
        ok
    }
}

fn pids(cmd: &str, args: &[&str]) -> Vec<i32> {
    Command::new(cmd).args(args).output().map(|o| String::from_utf8_lossy(&o.stdout).split_whitespace().filter_map(|p| p.parse().ok()).collect()).unwrap_or_default()
}

/// An exclusive lock on a file in Strata's folder, held while the returned file is open (the kernel drops it when the
/// process ends): `wait` blocks until it is free, else None when another process holds it
pub fn lock(name: &str, wait: bool) -> Option<std::fs::File> {
    use std::os::fd::AsRawFd;
    let p = running_path().with_file_name(name);
    let _ = std::fs::create_dir_all(p.parent()?);
    let f = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(p).ok()?;
    let op = if wait { libc::LOCK_EX } else { libc::LOCK_EX | libc::LOCK_NB };
    (unsafe { libc::flock(f.as_raw_fd(), op) } == 0).then_some(f)
}

fn alive(pid: i32) -> bool {
    unsafe { libc::kill(pid, 0) == 0 }
}

/// the model's file and the other shards of a split one
fn model_bytes(cfg: &Value) -> u64 {
    let args: Vec<&str> = cfg["args"].as_array().map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
    let Some(g) = args.iter().position(|a| *a == "--gguf").and_then(|i| args.get(i + 1)).map(Path::new) else { return 0 };
    let name = g.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let prefix = name.split_once("-00001-of-").map(|(p, _)| format!("{p}-0")).unwrap_or_else(|| name.to_string());
    std::fs::read_dir(g.parent().unwrap_or(Path::new("."))).into_iter().flatten().flatten()
        .filter(|e| e.file_name().to_str().is_some_and(|n| n == name || (n.starts_with(&prefix) && n.ends_with(".gguf"))))
        .filter_map(|e| e.metadata().ok()).map(|m| m.len()).sum()
}

/// FNV-1a over a file: whether the app on disk is the build that runs (not a secret: a fingerprint)
fn fingerprint(p: &Path) -> Option<u64> {
    let b = std::fs::read(p).ok()?;
    Some(b.iter().fold(0xcbf29ce484222325u64, |h, x| (h ^ *x as u64).wrapping_mul(0x100000001b3)))
}

fn running_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    Path::new(&home).join("Library/Application Support/Strata/running.json")
}

/// The request file a deploy writes before SIGUSR1 ("handoff" or "restart"), and the app's answer
pub fn switch_path() -> std::path::PathBuf {
    std::env::temp_dir().join("strata-app-switch")
}
pub fn switch_done_path() -> std::path::PathBuf {
    std::env::temp_dir().join("strata-app-switch.done")
}

/// At the app's start: its pid and its build's fingerprint, so a deploy knows it can switch servers in it (an app
/// from before this ends on SIGUSR1: it is never sent one) and whether its own code changed
pub fn record_running() {
    let exe = std::env::current_exe().ok();
    let fp = exe.as_deref().and_then(fingerprint);
    let p = running_path();
    let _ = std::fs::create_dir_all(p.parent().unwrap());
    let _ = std::fs::write(p, serde_json::json!({"pid": std::process::id(), "fingerprint": fp}).to_string());
}

/// The running app that can switch its server, when its build is the one on disk: its pid
fn switchable(old: &[i32], bundle: &Path) -> Option<i32> {
    let r: Value = serde_json::from_str(&std::fs::read_to_string(running_path()).ok()?).ok()?;
    switchable_from(&r, old, bundle)
}

fn switchable_from(r: &Value, old: &[i32], bundle: &Path) -> Option<i32> {
    let pid = r["pid"].as_i64()? as i32;
    let same = r["fingerprint"].as_u64().is_some() && r["fingerprint"].as_u64() == fingerprint(&bundle.join("Contents/MacOS/Strata"));
    (old.contains(&pid) && same).then_some(pid)
}

fn say(m: &str) {
    println!("strata-deploy: {m}");
}

/// How the running app switches to the new server: beside the old one when it can share the port and a second
/// engine fits, else (after waiting until idle) a stop and a start
pub fn mode(can_share: bool, fits: bool) -> &'static str {
    if can_share && fits { "handoff" } else { "restart" }
}

/// The client engines (docs/CLIENT_ENGINES.md) answer too, within 5 min: the main server's answer alone said a deploy
/// was done while Zed's engine had refused to start (2026-11-05)
fn clients_up(metrics: &dyn Fn() -> Option<Value>, limit: Duration, pause: Duration) -> i32 {
    let t = Instant::now();
    loop {
        let down: Vec<String> = metrics().and_then(|m| m["client_engines"].as_array().cloned()).unwrap_or_default().iter()
            .filter(|c| c["up"] != true).map(|c| c["name"].as_str().unwrap_or("?").to_string()).collect();
        if down.is_empty() {
            return 0;
        }
        if t.elapsed() >= limit {
            say(&format!("client engine not answering after {:.0} s: {} (its log: strata-<model>-<name>.log)", limit.as_secs_f64(), down.join(", ")));
            return 1;
        }
        std::thread::sleep(pause);
    }
}

/// make app-run: the decision, then its steps; the exit code says whether the new server runs.  One Strata.app at a
/// time: the running app switches its server (SIGUSR1, main.rs switch_server); only when the app's own code changed
/// does it restart, after the model is idle.
pub fn run(root: &Path, config: Option<&Path>) -> i32 {
    // one deploy at a time: two at once each quit the old app and opened a new one (2026-11-04 23:03: two Strata.app,
    // two servers sharing port 8090 and two engines in memory all night)
    let _one = match lock("deploy.lock", false) {
        Some(f) => f,
        None => {
            say("another deploy runs: waiting for it to finish");
            let Some(f) = lock("deploy.lock", true) else { return 1 };
            f
        }
    };
    let me = std::process::id() as i32;
    // an app a blue-green deploy started runs with --handoff: it is the old app the next deploy replaces too
    let old: Vec<i32> = pids("/usr/bin/pgrep", &["-f", "Strata.app/Contents/MacOS/Strata($| --handoff)"]).into_iter().filter(|p| *p != me).collect();
    let bundle = root.join("Strata.app");
    // -n always: right after the old app quits, LaunchServices may still count it and only "activate" it (2026-11-03)
    let open = || Command::new("/usr/bin/open").arg("-n").arg(&bundle).status().map(|s| s.success()).unwrap_or(false);
    let quit_old = || {
        for p in &old {
            unsafe { libc::kill(*p, libc::SIGTERM) };
        }
        let t = Instant::now();
        while old.iter().any(|p| alive(*p)) && t.elapsed() < Duration::from_secs(150) {
            std::thread::sleep(Duration::from_millis(500));
        }
    };
    let cfg = config.map(crate::read_config).unwrap_or(Value::Null);
    let port = cfg["port"].as_u64().map(|p| p as u16).unwrap_or(8090);
    let host = cfg["host"].as_str().unwrap_or("127.0.0.1").to_string();
    let key = match &cfg["api_key"] {
        Value::String(s) => s.split(',').next().unwrap_or("").trim().to_string(),
        Value::Array(a) => a.first().and_then(Value::as_str).unwrap_or("").to_string(),
        _ => String::new(),
    };
    let key = if key.is_empty() { crate::env_api_key(&cfg) } else { key };
    let metrics = || crate::get_json(&format!("http://127.0.0.1:{port}/metrics"), &key);
    // an engine the old app did not start (another config, a test) makes the new server refuse to start ("another
    // Strata engine is running"): say so before anything is stopped
    // every process under the old app: its server, the engine, and a client's own server and engine (docs/CLIENT_ENGINES.md)
    let mut ours: Vec<i32> = old.clone();
    let mut i = 0;
    while i < ours.len() {
        let kids = pids("/usr/bin/pgrep", &["-P", &ours[i].to_string()]);
        ours.extend(kids);
        i += 1;
    }
    let foreign: Vec<i32> = pids("/usr/bin/pgrep", &["-f", "strata-metal --serve"]).into_iter().filter(|p| !ours.contains(p)).collect();
    if !foreign.is_empty() {
        say(&format!("another Strata engine runs (pid {foreign:?}): the new app's server would refuse to start beside it. Stop it first; nothing was changed"));
        return 1;
    }
    // the new server answers: the deploy is done only then
    let up = || {
        let t = Instant::now();
        while t.elapsed() < Duration::from_secs(600) {
            if crate::get_json(&format!("http://127.0.0.1:{port}/health"), &key).is_some_and(|h| h["loaded"] == true) {
                say(&format!("the new server answers on port {port} ({:.0} s)", t.elapsed().as_secs_f64()));
                return clients_up(&metrics, Duration::from_secs(300), Duration::from_secs(3));
            }
            std::thread::sleep(Duration::from_secs(2));
        }
        say(&format!("no server answers on port {port} after 10 min: see Strata.app's log"));
        1
    };
    let wait_idle = |why: &str| {
        let t = Instant::now();
        let mut last = Instant::now();
        while metrics().as_ref().is_some_and(busy) {
            if last.elapsed() > Duration::from_secs(30) {
                say(&format!("waiting for the model to be idle ({:.0} s so far; {why})", t.elapsed().as_secs_f64()));
                last = Instant::now();
            }
            std::thread::sleep(Duration::from_secs(2));
        }
    };
    if old.is_empty() {
        say("no Strata.app runs: opening it");
        return if open() { up() } else { 1 };
    }
    let m = metrics();
    let share = can_share(&host, port);
    let fit = m.as_ref().is_some_and(|m| fits(m, model_bytes(&cfg)));
    let Some(pid) = switchable(&old, &bundle) else {
        // the app's own code changed (or it predates switching servers in it): a restart, one app at a time
        say(&format!("the app itself changed: it restarts (old app {old:?}, port {port}, {})", if m.as_ref().is_some_and(busy) { "busy" } else { "idle" }));
        wait_idle("the app restarts when no request runs");
        quit_old();
        return if open() { up() } else { 1 };
    };
    let mode = mode(share, fit);
    say(&format!("switching the server in the running app (pid {pid}): {mode} (port {port}, {}, sharable {share}, a second engine fits {fit})",
                 if m.as_ref().is_some_and(busy) { "busy" } else { "idle" }));
    if mode == "restart" {
        wait_idle(if share { "a second engine would not fit in memory" } else { "the running server cannot share its port: it predates blue-green" });
    }
    let _ = std::fs::remove_file(switch_done_path());
    if std::fs::write(switch_path(), mode).is_err() || unsafe { libc::kill(pid, libc::SIGUSR1) } != 0 {
        say("could not reach the running app; nothing was changed");
        return 1;
    }
    let t = Instant::now();
    let mut last = Instant::now();
    loop {
        if let Ok(r) = std::fs::read_to_string(switch_done_path()).map(|t| serde_json::from_str::<Value>(&t).unwrap_or(Value::Null)) {
            let _ = std::fs::remove_file(switch_done_path());
            say(r["message"].as_str().unwrap_or("done"));
            return if r["ok"] == true { up() } else { 1 };
        }
        if !alive(pid) || t.elapsed() > Duration::from_secs(1900) {
            say("the app did not finish the switch: its log says why");
            return 1;
        }
        if last.elapsed() > Duration::from_secs(30) {
            say(&format!("the old server is finishing its requests ({:.0} s)", t.elapsed().as_secs_f64()));
            last = Instant::now();
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_switch_follows_the_machine() {
        assert_eq!(mode(true, true), "handoff");
        assert_eq!(mode(false, true), "restart"); // an old server: no SO_REUSEPORT
        assert_eq!(mode(true, false), "restart"); // a second engine does not fit
        assert!(super::busy(&json!({"live": {"state": "reading"}})) && super::busy(&json!({"live": {"queued": 1}})));
        assert!(!super::busy(&json!({"live": {"state": "idle", "queued": 0, "running": 0, "waiting": 0}})));
    }

    #[test]
    fn a_running_app_switches_only_when_its_build_is_on_disk() {
        let d = std::env::temp_dir().join(format!("strata-switchable-{}", std::process::id()));
        std::fs::create_dir_all(d.join("Contents/MacOS")).unwrap();
        std::fs::write(d.join("Contents/MacOS/Strata"), b"build one").unwrap();
        let fp = fingerprint(&d.join("Contents/MacOS/Strata")).unwrap();
        let rec = |pid: i32, fp: Option<u64>| serde_json::json!({"pid": pid, "fingerprint": fp});
        assert_eq!(switchable_from(&rec(42, Some(fp)), &[42], &d), Some(42));
        assert_eq!(switchable_from(&rec(42, Some(fp)), &[7], &d), None);       // that app is gone
        assert_eq!(switchable_from(&rec(42, Some(fp ^ 1)), &[42], &d), None);  // the app's code changed
        assert_eq!(switchable_from(&rec(42, None), &[42], &d), None);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_second_engine_fits_when_its_contexts_do() {
        const G: u64 = 1 << 30;
        let m = |gpu_used: u64, ram_free: u64, gpu_free: u64| json!({"engine": {"gpu_used": gpu_used * G},
            "hardware": {"ram_total": 128 * G, "ram_used": (128 - ram_free) * G, "gpu_mem_total": 108 * G, "gpu_mem_used": (108 - gpu_free) * G}});
        assert!(fits(&m(31, 40, 50), 22 * G));        // Qwen3.6: 9 GB of contexts beside its 22 GB of weights
        assert!(!fits(&m(31, 8, 50), 22 * G));        // not in RAM
        assert!(!fits(&m(100, 40, 50), 22 * G));      // 78 GB of contexts
    }

    #[test]
    fn a_port_held_without_reuseport_cannot_be_shared() {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        assert!(!can_share("127.0.0.1", l.local_addr().unwrap().port()));
        let free = { let t = std::net::TcpListener::bind("127.0.0.1:0").unwrap(); t.local_addr().unwrap().port() };
        assert!(can_share("127.0.0.1", free));
    }

    #[test]
    fn the_deploy_waits_for_every_client_engine() {
        let z = Duration::ZERO;
        let m = |v: Value| move || Some(v.clone());
        assert_eq!(clients_up(&m(json!({"client_engines": [{"name": "zed", "up": true}]})), z, z), 0);
        assert_eq!(clients_up(&m(json!({})), z, z), 0);                       // no client engines
        assert_eq!(clients_up(&m(json!({"client_engines": [{"name": "zed", "up": false}]})), z, z), 1);
        // down at first, then up
        let n = std::cell::Cell::new(0);
        let later = || { n.set(n.get() + 1); Some(json!({"client_engines": [{"name": "zed", "up": n.get() > 2}]})) };
        assert_eq!(clients_up(&later, Duration::from_secs(5), z), 0);
    }

    #[test]
    fn one_deploy_and_one_app_at_a_time() {
        let name = format!("test-{}.lock", std::process::id());
        let held = lock(&name, false).expect("a free lock");
        assert!(lock(&name, false).is_none(), "a second holder while the first holds it");
        drop(held);
        assert!(lock(&name, false).is_some(), "free again once the holder is gone");
        let _ = std::fs::remove_file(running_path().with_file_name(&name));
    }
}
