//! `Strata --deploy` (make app-run): put the Strata.app just built in place of the running one without cutting off a
//! request.  Idle (or nothing running): the old app quits and the new one opens, as before.  Busy: blue-green - the new
//! app starts its server with --handoff beside the old one (one port, SO_REUSEPORT: macOS gives the newest listener the
//! new connections); that server loads its model, takes the port and has the old server finish its requests and stop
//! (SIGUSR2); then the old app quits.  When the old server cannot share its port (a server from before 2026-11-03) or
//! a second engine does not fit in memory, it waits until the model is idle instead.  docs/OBSERVABILITY.md's sibling:
//! the server half is strata-server --handoff.

use serde_json::Value;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

#[derive(Debug, PartialEq)]
pub enum Plan {
    /// no Strata.app runs: just open the new one
    Open,
    /// the model is idle (or its server does not answer): the old app quits, the new one opens
    Restart,
    /// busy: blue-green
    Handoff,
    /// busy and a handoff cannot be done: wait until it is idle, then restart
    WaitIdle,
}

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

pub fn plan(old_app: bool, metrics: Option<&Value>, can_share: bool, fits: bool) -> Plan {
    match metrics {
        _ if !old_app => Plan::Open,
        Some(m) if busy(m) => if can_share && fits { Plan::Handoff } else { Plan::WaitIdle },
        _ => Plan::Restart,
    }
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

fn say(m: &str) {
    println!("strata-deploy: {m}");
}

/// make app-run: the decision, then its steps; the exit code says whether the new app runs
pub fn run(root: &Path, config: Option<&Path>) -> i32 {
    let me = std::process::id() as i32;
    let old: Vec<i32> = pids("/usr/bin/pgrep", &["-f", "Strata.app/Contents/MacOS/Strata$"]).into_iter().filter(|p| *p != me).collect();
    let bundle = root.join("Strata.app");
    // -n always: right after the old app quits, LaunchServices may still count it and only "activate" it (2026-11-03)
    let open = |extra: &[&str]| {
        let mut c = Command::new("/usr/bin/open");
        c.arg("-n").arg(&bundle);
        if !extra.is_empty() {
            c.arg("--args").args(extra);
        }
        c.status().map(|s| s.success()).unwrap_or(false)
    };
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
    let ours: Vec<i32> = old.iter().flat_map(|p| pids("/usr/bin/pgrep", &["-P", &p.to_string()])).flat_map(|s| { let mut v = pids("/usr/bin/pgrep", &["-P", &s.to_string()]); v.push(s); v }).collect();
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
                return 0;
            }
            std::thread::sleep(Duration::from_secs(2));
        }
        say(&format!("no server answers on port {port} after 10 min: see Strata.app's log"));
        1
    };
    let m = metrics();
    let share = can_share(&host, port);
    let fit = m.as_ref().is_some_and(|m| fits(m, model_bytes(&cfg)));
    let p = plan(!old.is_empty(), m.as_ref(), share, fit);
    say(&format!("{p:?} (old app {old:?}, port {port}, {}, sharable {share}, a second engine fits {fit})",
                 if m.as_ref().is_some_and(busy) { "busy" } else { "idle" }));
    match p {
        Plan::Open => return if open(&[]) { up() } else { 1 },
        Plan::WaitIdle => {
            let t = Instant::now();
            let mut last = Instant::now();
            while metrics().as_ref().is_some_and(busy) {
                if last.elapsed() > Duration::from_secs(30) {
                    say(&format!("waiting for the model to be idle ({:.0} s so far; {})", t.elapsed().as_secs_f64(),
                                 if share { "a second engine would not fit in memory" } else { "the running server cannot share its port: it predates blue-green" }));
                    last = Instant::now();
                }
                std::thread::sleep(Duration::from_secs(2));
            }
        }
        Plan::Handoff => {
            let blue = pids("/usr/sbin/lsof", &["-nP", "-t", &format!("-iTCP:{port}"), "-sTCP:LISTEN"]);
            if !open(&["--handoff"]) {
                say("could not open the new Strata.app; the old one keeps running");
                return 1;
            }
            say(&format!("the new app's server loads its model, then takes port {port}; pid {blue:?} finishes its requests"));
            let t = Instant::now();
            let mut last = Instant::now();
            while blue.iter().any(|p| alive(*p)) {
                // a new server that never takes the port (it failed to load) leaves the old one serving: say so, keep it
                if t.elapsed() > Duration::from_secs(1800) {
                    say("the old server still runs after 30 min: the new app's log says why; nothing was stopped");
                    return 1;
                }
                if last.elapsed() > Duration::from_secs(30) {
                    say(&format!("old server {blue:?} still finishing ({:.0} s)", t.elapsed().as_secs_f64()));
                    last = Instant::now();
                }
                std::thread::sleep(Duration::from_millis(500));
            }
            say(&format!("handed over in {:.0} s; the old app quits", t.elapsed().as_secs_f64()));
            quit_old();
            return 0;
        }
        Plan::Restart => {}
    }
    quit_old();
    if open(&[]) { up() } else { 1 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_plan_follows_the_model_and_the_machine() {
        let idle = json!({"live": {"state": "idle", "queued": 0, "running": 0, "waiting": 0}});
        let busy = json!({"live": {"state": "idle", "running": 2}});
        assert_eq!(plan(false, Some(&busy), true, true), Plan::Open);
        assert_eq!(plan(true, Some(&idle), true, true), Plan::Restart);
        assert_eq!(plan(true, None, true, true), Plan::Restart); // its server does not answer
        assert_eq!(plan(true, Some(&busy), true, true), Plan::Handoff);
        assert_eq!(plan(true, Some(&busy), false, true), Plan::WaitIdle); // an old server: no SO_REUSEPORT
        assert_eq!(plan(true, Some(&busy), true, false), Plan::WaitIdle);
        assert!(super::busy(&json!({"live": {"state": "reading"}})) && super::busy(&json!({"live": {"queued": 1}})));
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
}
