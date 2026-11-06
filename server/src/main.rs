//! strata-server: Strata's OpenAI- and Anthropic-compatible server over the resident engine (the Rust port of
//! serve/server.py, Mac/Metal).
//!
//!     strata-server --engine mock --port 8095                                   (a scripted engine, for clients)
//!     strata-server --engine strata --config strata-<model>.json --port 8090    (the real engine, resident)

use clap::Parser;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use strata_server::engine::{py_g, AfmEngine, Engine, MockEngine, Spawn, StrataEngine, ENGINE_SILENCE_S};
use strata_server::py::{self, get, truthy};
use strata_server::service::{clean_shared_defaults, sampling_defaults_from_config, Service, Settings, REPEAT_STOP_TOKENS};
use strata_server::tokenizer::{ByteTokenizer, Tokenize};
use strata_server::{http, mcp, net, root, say};

#[derive(Parser)]
#[command(about = "Strata's server: OpenAI and Anthropic endpoints over the resident engine")]
struct Args {
    #[arg(long, default_value = "mock", value_parser = ["mock", "strata"])]
    engine: String,
    /// strata engine config (JSON: exe, args, cwd, tokenizer, model_name), written by setup.py
    #[arg(long)]
    config: Option<String>,
    /// the address to listen on: 127.0.0.1 = this PC only (the default), 0.0.0.0 = also other devices on your network
    #[arg(long)]
    host: Option<String>,
    /// the mock engine's answer; given more than once, requests get them in turn and the last one repeats
    #[arg(long)]
    script: Vec<String>,
    #[arg(long, default_value_t = 8095)]
    port: u16,
    /// if another Strata server holds the port, stop it and start this one
    #[arg(long)]
    replace: bool,
    /// blue-green (make app-run while the model is busy): share the port with the Strata server that holds it, load the
    /// model first, then take every new connection and have that one finish its requests and stop (SIGUSR2)
    #[arg(long, conflicts_with = "replace")]
    handoff: bool,
    #[arg(long)]
    gpu: Option<String>,
    /// pack tokenizer directory (falls back to a byte tokenizer if absent)
    #[arg(long)]
    tokenizer: Option<String>,
    /// open the local page in the browser once the model is ready
    #[arg(long)]
    open: bool,
    /// clamp max_tokens to the remaining context instead of rejecting the request
    #[arg(long)]
    fit_max_tokens: bool,
    /// require this key on /v1/* (Authorization: Bearer ... or x-api-key); several: key1,key2
    #[arg(long)]
    api_key: Option<String>,
    /// a JSON file with MCP servers in Claude Desktop's format
    #[arg(long)]
    mcp_config: Option<String>,
    /// start the API unloaded (the engine starts with the first request)
    #[arg(long)]
    lazy: bool,
    /// the API request monitor at /api-monitor (the last 100 requests in memory)
    #[arg(long)]
    api_monitor: bool,
    /// unload the model after this many seconds without requests
    #[arg(long, value_name = "SECONDS")]
    idle_unload: Option<f64>,
    /// load an unloaded model only when this much VRAM is free, else answer 503
    #[arg(long)]
    min_free_vram_mib: Option<i64>,
    /// a command run before the model is loaded again
    #[arg(long)]
    before_load: Option<String>,
    /// enable POST /slots/0?action=save|restore {"filename": NAME}
    #[arg(long, value_name = "DIR")]
    slot_save_path: Option<String>,
    /// data-parallel replicas (several GPUs: not on a Mac)
    #[arg(long)]
    replicas: Option<i64>,
}

fn die(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(2)
}

fn exit_msg(msg: &str) -> ! {
    eprintln!("{}", strata_server::engine::stamp(msg));
    std::process::exit(1)
}

/// The config's "system_prompt": a text file (a relative path: from the config's folder), e.g. "prompts/lfm2.5-8b.md"
fn system_prompt(cfg: &Value, config: Option<&str>) -> Result<Option<String>, String> {
    let p = match get(cfg, "system_prompt") {
        Value::Null => return Ok(None),
        Value::String(p) if !p.trim().is_empty() => p,
        v => return Err(format!("[strata] config \"system_prompt\" must be a file's path, not {}", py::py_repr(v))),
    };
    let dir = config.and_then(|c| Path::new(c).parent()).unwrap_or(Path::new("."));
    let text = std::fs::read_to_string(dir.join(p)).map_err(|e| format!("[strata] config \"system_prompt\" {p}: {e}"))?;
    let text = text.trim();
    say(&format!("system prompt: {p} ({} characters) ahead of each request's system text", text.len()));
    Ok((!text.is_empty()).then(|| text.to_string()))
}

/// #1641: "--reasoning-effort high" in a config's args is the config's "reasoning_effort".
fn lift_reasoning_effort_arg(cfg: &mut Map<String, Value>) {
    let Some(args) = cfg.get("args").and_then(Value::as_array).cloned() else { return };
    let (mut out, mut value, mut i) = (vec![], None, 0);
    while i < args.len() {
        let a = &args[i];
        if a == "--reasoning-effort" && i + 1 < args.len() {
            value = Some(args[i + 1].clone());
            i += 2;
        } else if let Some(v) = a.as_str().and_then(|s| s.strip_prefix("--reasoning-effort=")) {
            value = Some(json!(v));
            i += 1;
        } else {
            out.push(a.clone());
            i += 1;
        }
    }
    let Some(v) = value else { return };
    cfg.insert("args".into(), Value::Array(out));
    if cfg.get("reasoning_effort").map(Value::is_null).unwrap_or(true) {
        cfg.insert("reasoning_effort".into(), v);
    }
    say(&format!("'--reasoning-effort' in the config's args is not an engine option: taken out of them. The thinking level is the config's 'reasoning_effort' key (outside 'args'); it is used as {}", py::py_repr(&cfg["reasoning_effort"])));
}

fn str_args(cfg: &Value) -> Vec<String> {
    get(cfg, "args").as_array().map(|a| a.iter().map(py::py_str).collect()).unwrap_or_default()
}

/// The engine's arguments: the config's, plus "parallel" (--batch), "vram_elastic" and the learned expert profile.
fn engine_args(cfg: &Value) -> Vec<String> {
    let mut args = str_args(cfg);
    if get(cfg, "vram_elastic") == &json!(true) && !args.iter().any(|a| a == "--vram-elastic") {
        args.push("--vram-elastic".into());
        if let Some(seg) = get(cfg, "vram_segment_mib").as_i64().filter(|s| *s > 0) {
            if !args.iter().any(|a| a == "--vram-segment-mib") {
                args.extend(["--vram-segment-mib".into(), seg.to_string()]);
            }
        }
    }
    // #465 (opt-in): "parallel": N
    if !args.iter().any(|a| a == "--batch" || a == "--slots") {
        match get(cfg, "parallel") {
            Value::Null | Value::Bool(false) => {}
            Value::Number(n) if n.is_i64() => {
                let n = n.as_i64().unwrap();
                if n > 1 {
                    if n > 8 && !args.iter().any(|a| a == "--batch-mtp") && matches!(std::env::var("STRATA_BATCH_MTP").as_deref(), Err(_) | Ok("" | "0")) {
                        say(&format!("\"parallel\": {n} - the engine runs at most 8 requests together; it will use 8"));
                    }
                    args.extend(["--batch".into(), n.to_string()]);
                }
            }
            v => say(&format!("\"parallel\" must be a whole number of requests (2..8), not {}: ignored", py::py_repr(v))),
        }
    }
    // #477 (opt-in): "expert_profile_save"
    if let Some(save) = get(cfg, "expert_profile_save").as_str().filter(|s| !py::strip(s).is_empty()) {
        if !args.iter().any(|a| a == "--expert-profile-save") {
            args.extend(["--expert-profile-save".into(), save.to_string()]);
            if let Some(every) = get(cfg, "expert_profile_save_every").as_f64().filter(|e| *e >= 0.0 && get(cfg, "expert_profile_save_every").is_number()) {
                args.extend(["--expert-profile-save-every".into(), py::py_str(&get(cfg, "expert_profile_save_every").clone())]);
                let _ = every;
            }
            let n = args.len();
            if let Some(i) = args[..n - 1].iter().position(|a| a == "--expert-profile") {
                let here = get(cfg, "cwd").as_str().unwrap_or(".").to_string();
                let abs = |p: &str| if Path::new(p).is_absolute() { PathBuf::from(p) } else { Path::new(&here).join(p) };
                let learned = profile_shape(&abs(save));
                if learned.is_some() && learned == profile_shape(&abs(&args[i + 1])) {
                    args[i + 1] = save.to_string();
                }
            }
        }
    }
    args
}

/// An expert profile's (layers, experts per layer), from its header; None when it is no whole profile.
fn profile_shape(path: &Path) -> Option<(u32, u32)> {
    let data = std::fs::read(path).ok()?;
    if data.len() < 24 || &data[..4] != b"STRP" {
        return None;
    }
    let w = |i: usize| u32::from_le_bytes(data[4 + 4 * i..8 + 4 * i].try_into().unwrap());
    let (nl, ne, n) = (w(1), w(2), w(4));
    if data.len() as u64 >= 24 + 4 * n as u64 { Some((nl, ne)) } else { None }
}

/// #458 (opt-in): "effort_position": "end" -> --tail-role-token <id of "system">, when the engine has it.
fn effort_end_args(cfg: &Value, exe: &str, tok: &dyn Tokenize) -> Result<Option<Vec<String>>, String> {
    let pos = get(cfg, "effort_position").as_str().unwrap_or("start");
    if !get(cfg, "effort_position").is_null() && pos != "start" && pos != "end" {
        return Err(format!("\"effort_position\" must be \"start\" (the default) or \"end\", not {}", py::py_repr(get(cfg, "effort_position"))));
    }
    if pos == "start" {
        return Ok(None);
    }
    let known = std::fs::read(exe).map(|b| b.windows(17).any(|w| w == b"--tail-role-token")).unwrap_or(false);
    let role = tok.encode("system", true);
    if !known || role.len() != 1 {
        say("effort_position \"end\" needs engine 0.1.39 or newer (--tail-role-token): the reasoning effort stays at the top of the prompt");
        return Ok(None);
    }
    say("effort_position end: a request's non-default reasoning effort goes right before the answer, so changing it keeps the cached conversation");
    Ok(Some(vec!["--tail-role-token".into(), role[0].to_string()]))
}

/// The engine's environment: the server's, the config's "env", its "lib_dirs" first on the library path.
fn child_env(cfg: &Value) -> HashMap<String, String> {
    let mut env: HashMap<String, String> = std::env::vars().collect();
    for (k, v) in get(cfg, "env").as_object().into_iter().flatten() {
        env.insert(k.clone(), py::py_str(v));
    }
    let dirs: Vec<String> = get(cfg, "lib_dirs").as_array().into_iter().flatten().map(py::py_str).filter(|d| Path::new(d).is_dir()).collect();
    if !dirs.is_empty() {
        let var = "LD_LIBRARY_PATH";
        let mut all = dirs.join(":");
        if let Some(old) = env.get(var).filter(|o| !o.is_empty()) {
            all = format!("{all}:{old}");
        }
        env.insert(var.into(), all);
    }
    env
}

/// The id /v1/models reports when the config names none: the GGUF's file name, lower case, without the shard suffix.
fn model_name_from_args(args: &[String]) -> String {
    if let Some(i) = args[..args.len().saturating_sub(1)].iter().position(|a| a == "--gguf") {
        let name = Path::new(&args[i + 1]).file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
        let re = regex::Regex::new(r"(?i)(-\d{5}-of-\d{5})?\.gguf$").unwrap();
        let name = re.replace(&name, "").to_string();
        if !name.is_empty() {
            return name.to_lowercase();
        }
    }
    "qwen3.8-flash-next".into()
}

/// The conversation cache when the config sets no --conversation-cache-mib: by the RAM left beside the model's file.
/// Measured (strata-bench, 2026-11-03): 16 GB kept 4 Qwen3.6 agents' conversations on the 128 GB M5 Max; on the 16 GB
/// M1, 2 GB brought 1-4 agents' 3,000-token conversations back (99%) for LFM2.5-8B / -24B and GPT-OSS-20B, follow-ups
/// in 0.4-2 s instead of 4-32 s.  A model with a big state per token (Sheared-LLaMA: 1 GB a conversation) needs 4 GB
/// for one agent's and is slower with 2-4 agents either way.
// ponytail: by RAM only; sizing by the model's state per token (the engine knows it) when a model needs it
fn auto_cache_mib(ram_gib: f64, model_gib: f64) -> u64 {
    match ram_gib - model_gib {
        f if f >= 64.0 => 16384,
        f if f >= 24.0 => 8192,
        f if f >= 8.0 => 4096,
        f if f >= 4.0 => 2048,
        _ => 0,
    }
}

/// The model's size on disk: its --gguf, with the other shards of a split file (NAME-00001-of-00003.gguf)
fn model_gib(args: &[String]) -> Option<f64> {
    let g = Path::new(args.get(args.iter().position(|a| a == "--gguf")? + 1)?);
    let name = g.file_name()?.to_str()?;
    let prefix = name.split_once("-00001-of-").map(|(p, _)| format!("{p}-0")).unwrap_or_else(|| name.to_string());
    let bytes: u64 = std::fs::read_dir(g.parent()?).ok()?.flatten()
        .filter(|e| e.file_name().to_str().is_some_and(|n| n == name || (n.starts_with(&prefix) && n.ends_with(".gguf"))))
        .filter_map(|e| e.metadata().ok()).map(|m| m.len()).sum();
    Some(bytes as f64 / 1073741824.0)
}

/// --conversation-cache-mib added by auto_cache_mib when the args have none ("0" in the config: off)
fn add_auto_cache(args: &mut Vec<String>) {
    if args.iter().any(|a| a == "--conversation-cache-mib") {
        return;
    }
    let (_, total) = strata_server::telemetry::ram();
    let (Some(total), Some(model)) = (total, model_gib(args)) else { return };
    let ram = total as f64 / 1073741824.0;
    let mib = auto_cache_mib(ram, model);
    say(&format!("conversation cache: {} (automatic: {ram:.0} GB of RAM, the model {model:.1} GB; \"--conversation-cache-mib\" in the config's args sets it, 0 = off)",
                 if mib > 0 { format!("{} GB", mib / 1024) } else { "off".into() }));
    if mib > 0 {
        args.extend(["--conversation-cache-mib".into(), mib.to_string()]);
    }
}

/// Where a draining server leaves its agents' conversations for the one that took its port (docs/BLUE_GREEN.md)
fn handoff_dir(port: u16) -> PathBuf {
    std::env::temp_dir().join(format!("strata-handoff-{port}"))
}

/// The server's listener, SO_REUSEPORT on: a newer server (--handoff) can bind beside it, and macOS gives the newest
/// listener the new connections (measured).  Two servers on one port by mistake stay impossible: port_free's plain
/// bind, before this, fails while one listens.
async fn listen(addr: &str) -> std::io::Result<tokio::net::TcpListener> {
    let sa = tokio::net::lookup_host(addr).await?.next().ok_or_else(|| std::io::Error::other(format!("{addr}: no address")))?;
    let s = if sa.is_ipv4() { tokio::net::TcpSocket::new_v4()? } else { tokio::net::TcpSocket::new_v6()? };
    s.set_reuseaddr(true)?;
    s.set_reuseport(true)?;
    s.bind(sa)?;
    s.listen(1024)
}

fn warn_budget_over_ram(args: &[String]) {
    let Some(i) = args.iter().position(|a| a == "--resident-budget-gib") else { return };
    let Some(budget) = args.get(i + 1).and_then(|v| v.parse::<f64>().ok()) else { return };
    let (used, total) = strata_server::telemetry::ram();
    let (Some(used), Some(total)) = (used, total) else { return };
    let (total, free) = (total as f64 / 1073741824.0, (total - used) as f64 / 1073741824.0);
    if budget <= free && budget <= total - 6.0 {
        return;
    }
    say(&format!("WARNING: the RAM budget for experts is {} GiB (--resident-budget-gib), but this PC has {free:.0} GiB free of {total:.0} GiB: what does not fit goes to swap or is read from disk, which can make decoding very slow. Lower it (setup sizes it for this PC) or remove the argument to let the engine choose.", py_g(budget)));
}

fn warn_tight_ram(arena_mib: Option<i64>) {
    let Some(a) = arena_mib.filter(|a| *a > 0) else { return };
    let (_, total) = strata_server::telemetry::ram();
    let Some(total) = total else { return };
    let left = total as f64 / 1073741824.0 - a as f64 / 1024.0;
    if left < 6.0 {
        say(&format!("WARNING: RAM is tight - the model's experts take {:.1} GB of this PC's {:.0} GB, leaving {left:.1} GB for everything else. Linux may stop the engine in the middle of an answer. Close other programs, or run START-HERE --setup and pick a smaller size (Q2_0 / IQ2_XS).", a as f64 / 1024.0, total as f64 / 1073741824.0));
    }
}

/// The processes listening on `port` as (pid, command line).
fn port_holders(port: u16) -> Vec<(i32, String)> {
    let Ok(o) = std::process::Command::new("lsof").args(["-nP", "-t", &format!("-iTCP:{port}"), "-sTCP:LISTEN"]).output() else { return vec![] };
    let mut pids: Vec<i32> = String::from_utf8_lossy(&o.stdout).split_whitespace().filter_map(|p| p.parse().ok()).collect();
    pids.dedup();
    let found = pids.into_iter()
        .filter_map(|p| {
            let o = std::process::Command::new("ps").args(["-o", "ppid=,command=", "-p", &p.to_string()]).output().ok()?;
            let line = String::from_utf8_lossy(&o.stdout).trim().to_string();
            let (ppid, cmd) = line.split_once(char::is_whitespace)?;   // gone since lsof: not a holder
            Some((p, ppid.parse().unwrap_or(0), cmd.trim().to_string()))
        })
        .collect();
    servers_holding(found)
}

/// The holders that are not a Strata server's own child: a child between fork and exec holds a copy of its server's
/// socket ("(lsof)", "(ioreg)"), for milliseconds, and for seconds once `make build` replaced the binary under the
/// running server (macOS checks the new file at every spawn).  Counted as another program, it stopped the Zed engine
/// in a deploy (2026-11-05: "port 8190 is already in use ... (held by pid 59122: )") and Zed's requests were refused.
fn servers_holding(found: Vec<(i32, i32, String)>) -> Vec<(i32, String)> {
    let servers: Vec<i32> = found.iter().filter(|(_, _, c)| is_strata_server(c)).map(|(p, _, _)| *p).collect();
    found.into_iter().filter(|(_, ppid, c)| is_strata_server(c) || !servers.contains(ppid)).map(|(p, _, c)| (p, c)).collect()
}

/// The port check again while the port is in use and no process listens on it (a server stopping), up to `tries` times
fn port_settled(mut check: impl FnMut() -> std::io::Result<()>, mut nobody: impl FnMut() -> bool, tries: usize, pause: std::time::Duration) -> std::io::Result<()> {
    let mut r = check();
    for _ in 0..tries {
        match &r {
            Err(e) if e.kind() == std::io::ErrorKind::AddrInUse && nobody() => {
                std::thread::sleep(pause);
                r = check();
            }
            _ => break,
        }
    }
    r
}

fn is_strata_server(cmd: &str) -> bool {
    let c = cmd.replace('\\', "/");
    c.contains("serve/server.py") || c.contains("strata-server")
}

/// Stop a Strata server so its port is free: SIGCONT, SIGTERM, then after 30 s SIGKILL for it and its children.
fn stop_server(pid: i32, port: u16) -> bool {
    let free = || std::net::TcpStream::connect_timeout(&format!("127.0.0.1:{port}").parse().unwrap(), std::time::Duration::from_secs(1)).is_err();
    let kids: Vec<i32> = std::process::Command::new("pgrep").args(["-P", &pid.to_string()]).output().map(|o| String::from_utf8_lossy(&o.stdout).split_whitespace().filter_map(|p| p.parse().ok()).collect()).unwrap_or_default();
    unsafe {
        libc::kill(pid, libc::SIGCONT);
        libc::kill(pid, libc::SIGTERM);
    }
    let t = std::time::Instant::now();
    while t.elapsed().as_secs() < 30 {
        if free() && unsafe { libc::kill(pid, 0) } != 0 {
            return true;
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    for p in std::iter::once(pid).chain(kids) {
        unsafe { libc::kill(p, libc::SIGKILL) };
    }
    std::thread::sleep(std::time::Duration::from_secs(1));
    free()
}

/// The pids of Strata engines already running (strata-metal: a model in memory), of any user.
fn other_engines() -> Vec<String> {
    engines_in(&std::process::Command::new("ps").args(["-axo", "pid=,args="]).output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default())
}

/// ps -axo pid=,args= -> the pids whose program is strata-metal (a path, spaces allowed, or a bare name) holding a
/// model: `--plan` only measures one (no weights read; Strata.app runs it for its memory panel while a server starts).
fn engines_in(ps: &str) -> Vec<String> {
    ps.lines()
        .filter_map(|l| {
            let (pid, args) = l.trim_start().split_once(' ')?;
            let args = args.trim();
            let at = args.match_indices("strata-metal").map(|(i, _)| i + "strata-metal".len()).find(|&e| {
                let st = e - "strata-metal".len();
                (st == 0 || args[..st].ends_with('/')) && matches!(args[e..].chars().next(), None | Some(' '))
            })?;
            (!args[at..].split_whitespace().any(|a| a == "--plan")).then(|| pid.to_string())
        })
        .collect()
}

/// Whether this server can listen on host:port.  On a wildcard host the loopback address is tried too: the socket's
/// SO_REUSEADDR lets 0.0.0.0:8080 bind beside another program's 127.0.0.1:8080 (a Docker port), and then the app and
/// every local client reach that program instead of this server.
fn port_free(host: &str, port: u16) -> std::io::Result<()> {
    let addr = format!("{}:{}", if host.contains(':') { format!("[{host}]") } else { host.to_string() }, port);
    std::net::TcpListener::bind(&addr)?;
    // and the other way round: macOS lets 127.0.0.1:port bind while another server holds 0.0.0.0:port (a server
    // on "host": "0.0.0.0" then hid from --replace, and its engine made this one refuse to start)
    let other = match host {
        "0.0.0.0" => "127.0.0.1",
        "::" => "::1",
        h if h.contains(':') => "::",
        _ => "0.0.0.0",
    };
    match std::net::TcpListener::bind((other, port)) {
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => Err(e),
        _ => Ok(()), // no address of that family here: nothing to shadow
    }
}

fn listen_problem(host: &str, port: u16, e: &std::io::Error) -> String {
    match e.kind() {
        std::io::ErrorKind::AddrInUse => format!("port {port} is already in use - is Strata (or another server) already running? Close it, or start this one with a different --port"),
        std::io::ErrorKind::AddrNotAvailable => format!("cannot listen on {host}:{port}: {e}. {host} is not an address of this PC: check \"host\" in the model's strata-<model>.json or --host"),
        std::io::ErrorKind::PermissionDenied => format!("cannot listen on {host}:{port}: {e}. a firewall rule forbids it: start this one with a different --port"),
        _ => format!("cannot listen on {host}:{port}: {e}. check --host and --port"),
    }
}

fn main() {
    let a = Args::parse();
    let mut cfg: Map<String, Value> = match &a.config {
        Some(p) => match std::fs::read_to_string(p).map_err(|e| e.to_string()).and_then(|t| serde_json::from_str::<Value>(t.trim_start_matches('\u{feff}')).map_err(|e| e.to_string())) {
            Ok(Value::Object(o)) => o,
            Ok(_) => die(&format!("strata-server: error: {p} is not a JSON object")),
            Err(e) => die(&format!("strata-server: error: {p}: {e}")),
        },
        None => Map::new(),
    };
    lift_reasoning_effort_arg(&mut cfg);
    if a.replicas.is_some() || get(&Value::Object(cfg.clone()), "replicas").as_i64().map(|r| r > 1).unwrap_or(false) || get(&Value::Object(cfg.clone()), "replicas").is_array() {
        exit_msg("[strata] config replicas: data-parallel replicas need several GPUs; this server runs on a Mac's one GPU");
    }
    if let Some(g) = &a.gpu {
        cfg.insert("gpu".into(), g.trim().parse::<i64>().map(Value::from).unwrap_or(json!(g)));
    }
    let cfgv = Value::Object(cfg.clone());
    // "trace": true - STRATA_TRACE=1 for this server and its engine (Strata.app starts the server, so no shell sets it):
    // which slot each request gets and why, and why the engine reads a prompt again or not
    if get(&cfgv, "trace") == &json!(true) && std::env::var_os("STRATA_TRACE").is_none() {
        // SAFETY: still one thread here - the runtime, the engine and the firewall sampler start below
        unsafe { std::env::set_var("STRATA_TRACE", "1") };
        say("trace on (\"trace\": true): slot choices here, the engine's reuse in this config's log");
    }
    // a client's own engine (docs/CLIENT_ENGINES.md): this server routes by User-Agent ("clients"), or it is one
    // (STRATA_PARENT_PID: the main server runs the firewall history, voice, schedules and MCP servers)
    strata_server::clients::set(strata_server::clients::parse(&cfgv, a.port).unwrap_or_else(|e| exit_msg(&format!("[strata] {e}"))));
    let client_engine = strata_server::clients::watch_parent();
    // a client engine's stdout is its log already (clients::start)
    if let (false, Some(l)) = (client_engine, get(&cfgv, "log").as_str()) {
        strata_server::log_to(l);
    }
    strata_server::firewall::start(get(&cfgv, "firewall"), a.engine == "strata" && !client_engine); // history, connection sampler, the kernel's sandbox log
    let host = a.host.clone().or_else(|| get(&cfgv, "host").as_str().filter(|h| !h.is_empty()).map(String::from)).unwrap_or_else(|| "127.0.0.1".into());
    // before the minutes of loading: is the port free?
    let addr = format!("{}:{}", if host.contains(':') { format!("[{host}]") } else { host.clone() }, a.port);
    let mut handoff_from: Vec<i32> = vec![];
    // a server that is stopping holds its port a moment after its listener is gone: in use, held by no one (a client
    // engine started while the previous run's one stopped quit "port 8190 is already in use", 2026-11-05)
    let busy = port_settled(|| port_free(&host, a.port), || port_holders(a.port).is_empty(), 10, std::time::Duration::from_millis(500));
    if let Err(e) = busy {
        let holders = if e.kind() == std::io::ErrorKind::AddrInUse { port_holders(a.port) } else { vec![] };
        let ours: Vec<&(i32, String)> = holders.iter().filter(|(_, c)| is_strata_server(c)).collect();
        if ours.is_empty() || ours.len() != holders.len() {
            let others: Vec<String> = holders.iter().filter(|(_, c)| !is_strata_server(c)).map(|(p, c)| format!("pid {p}: {}", py::chars_prefix(c, 120))).collect();
            die(&format!("strata-server: error: {}{}", listen_problem(&host, a.port, &e), if others.is_empty() { String::new() } else { format!(" (held by {})", others.join("; ")) }));
        }
        let (pid, cmd) = ours[0];
        let config = cmd.split_once("--config").map(|(_, r)| r.split_whitespace().next().unwrap_or("?").trim_matches('"').to_string()).unwrap_or_else(|| "?".into());
        say(&format!("port {} is held by another Strata server: pid {pid}, {}", a.port, Path::new(&config).file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or(config)));
        if a.handoff {
            handoff_from = ours.iter().map(|(p, _)| *p).collect();
            say(&format!("handoff: this server takes port {} over from pid {pid} once its model is loaded", a.port));
        } else {
        if !a.replace {
            use std::io::IsTerminal;
            if !(std::io::stdin().is_terminal() && std::io::stderr().is_terminal()) {
                die(&format!("strata-server: error: {} (--replace stops that Strata server and starts this one)", listen_problem(&host, a.port, &e)));
            }
            eprint!("Stop it and start this one? [y/N] ");
            let mut answer = String::new();
            let _ = std::io::stdin().read_line(&mut answer);
            if !matches!(answer.trim().to_lowercase().as_str(), "y" | "yes") {
                die(&format!("strata-server: error: port {} is still in use; nothing was stopped", a.port));
            }
        }
        say(&format!("stopping pid {pid} ..."));
        if !stop_server(*pid, a.port) {
            die(&format!("strata-server: error: could not free port {} (pid {pid} did not stop)", a.port));
        }
        say(&format!("port {} is free", a.port));
        }
    }
    // "engine": "afm": Apple's on-device model (metal/afm/strata-afm, FoundationModels): text in and out, so bytes for
    // tokens and the conversation as JSON for a prompt
    let afm = a.engine == "strata" && get(&cfgv, "engine") == &json!("afm");
    let tpath = PathBuf::from(get(&cfgv, "tokenizer").as_str().map(String::from).or(a.tokenizer.clone()).unwrap_or_else(|| root().join("pack/full/tokenizer").to_string_lossy().into_owned()));
    if a.engine == "strata" && !afm && !tpath.join("vocab.json").exists() {
        die(&format!("strata-server: error: the model's tokenizer is missing ({}); run setup again", tpath.join("vocab.json").display()));
    }
    let tok: Arc<dyn Tokenize> = if !afm && tpath.join("vocab.json").exists() {
        match strata_server::tokenizer::from_pack_dir(&tpath) {
            Ok(t) => Arc::from(t),
            Err(e) => die(&format!("strata-server: error: the tokenizer in {}: {e}", tpath.display())),
        }
    } else {
        Arc::new(ByteTokenizer)
    };
    // the Strata folder's .env: one more API key and MCP servers, on top of the config (crate::dotenv)
    let dotenv = strata_server::dotenv::load(&root());
    let mut env_mcp = strata_server::dotenv::mcp_servers(&dotenv);
    // Strata.app's browser for the model (app/src/browser_mcp.rs): its endpoint and key, made at each app start
    if client_engine {
        env_mcp.clear();
    }
    if let (false, Ok(u), Ok(k)) = (client_engine, std::env::var("STRATA_BROWSER_MCP_URL"), std::env::var("STRATA_BROWSER_MCP_KEY")) {
        env_mcp.insert("browser".into(), json!({"url": u, "headers": {"Authorization": format!("Bearer {k}")}}));
        // a "/team"'s agents: a browser each, beside the pane people see (team.rs)
        for i in 1..=4 {
            env_mcp.insert(format!("agent{i}"), json!({"url": format!("{u}/{i}"), "headers": {"Authorization": format!("Bearer {k}")}}));
        }
    }
    if !env_mcp.is_empty() {
        say(&format!("MCP servers from .env or Strata.app (its browser): {}", env_mcp.keys().cloned().collect::<Vec<_>>().join(", ")));
    }
    let hub = match mcp::hub_from_config(&cfgv, a.mcp_config.as_deref(), &env_mcp) {
        Ok(Some(h)) => h,
        Ok(None) => mcp::McpHub::new(Default::default(), mcp::settings_from(&cfgv).unwrap_or_else(|e| exit_msg(&e)), &[]),
        Err(e) => exit_msg(&e),
    };
    let mut vision = None;
    let sampling_defaults;
    let engine: Engine;
    let effort_end;
    if afm {
        effort_end = None;
        sampling_defaults = sampling_defaults_from_config(&cfgv).unwrap_or_else(|e| exit_msg(&e));
        let exe = get(&cfgv, "exe").as_str().unwrap_or("").to_string();
        println!("starting Apple's on-device model ...");
        engine = Engine::Afm(Arc::new(AfmEngine::new(&exe).unwrap_or_else(|e| exit_msg(&format!("[strata] {e}")))));
    } else if a.engine == "strata" {
        if cfg.is_empty() {
            die("strata-server: error: --engine strata needs --config");
        }
        // on the network (0.0.0.0, a LAN address) only with a key - before the model loads: anyone who reaches it
        // could use the model, read the request monitor and drive its MCP tools
        let some_key = |v: &Value| match v {
            Value::String(k) => !k.trim().is_empty(),
            Value::Array(a) => !a.is_empty(),
            _ => false,
        };
        let keyed = a.api_key.as_deref().is_some_and(|k| !k.trim().is_empty())
            || std::env::var("STRATA_API_KEY").is_ok_and(|k| !k.trim().is_empty())
            || some_key(get(&cfgv, "api_key"))
            || strata_server::dotenv::get(&dotenv, "API_KEY").is_some_and(|k| !k.trim().is_empty());
        if !keyed && !["127.0.0.1", "localhost", "::1"].contains(&host.as_str()) {
            exit_msg(&format!("[strata] \"host\": \"{host}\" shares this server with other devices, so it needs an API key: \
                Strata.app's More > Share on the network puts one in .env; or API_KEY in .env, \"api_key\" in the run config, or --api-key"));
        }
        let env = child_env(&cfgv);
        sampling_defaults = sampling_defaults_from_config(&cfgv).unwrap_or_else(|e| exit_msg(&e));
        if !sampling_defaults.is_empty() {
            say(&format!("sampling defaults from the config: {}", sampling_defaults.iter().map(|(k, v)| format!("{k}={}", py::py_str(v))).collect::<Vec<_>>().join(", ")));
        }
        let lazy = a.lazy || get(&cfgv, "lazy_load") == &json!(true);
        let cwd = get(&cfgv, "cwd").as_str().map(String::from);
        let abs_in_cwd = |p: &str| if Path::new(p).is_absolute() { p.to_string() } else { std::path::absolute(Path::new(cwd.as_deref().unwrap_or(".")).join(p)).map(|x| x.to_string_lossy().into_owned()).unwrap_or(p.to_string()) };
        if let Some(v) = get(&cfgv, "vision").as_object() {
            println!("{}", if lazy { "vision encoder unloaded; it starts with the model ..." } else { "loading the vision encoder ..." });
            let mut vcfg = v.clone();
            for k in ["exe", "mmproj", "model"] {
                if let Some(s) = vcfg.get(k).and_then(Value::as_str).map(String::from) {
                    vcfg.insert(k.into(), json!(abs_in_cwd(&s)));
                }
            }
            match strata_server::vision::Vision::new(&Value::Object(vcfg), get(&cfgv, "log").as_str().map(String::from), env.clone(), lazy) {
                Ok(v) => vision = Some(Arc::new(v)),
                Err(e) => exit_msg(&format!("[strata] {e}")),
            }
        }
        // two engines need the memory for two models: on a 16 GB Mac the kernel's out-of-memory killer then ends one of
        // them (and systemd the session it ran in: a TUI was lost that way).  --replace has already stopped this port's.
        let others = other_engines();
        // a handoff runs two engines for a while: the weights are the same mapped file, only the contexts are twice
        if !others.is_empty() && handoff_from.is_empty() && std::env::var("STRATA_ALLOW_SECOND_ENGINE").map(|v| v != "1").unwrap_or(true) {
            exit_msg(&format!(
                "[strata] another Strata engine is running (pid {}): two models need the memory of both, and the system ends one when it runs out. \
                 Stop that one first (Ctrl+S in its make tui, Stop in Strata.app, or make stop), or start this one with --replace on its port. \
                 To run both anyway: STRATA_ALLOW_SECOND_ENGINE=1",
                others.join(", ")
            ));
        }
        println!("{}", if lazy { "model unloaded; the first request loads it ..." } else { "loading the model (the first start takes a minute or two) ..." });
        warn_budget_over_ram(&engine_args(&cfgv));
        let exe = abs_in_cwd(get(&cfgv, "exe").as_str().unwrap_or(""));
        let silence = match get(&cfgv, "engine_silence_s") {
            Value::Null => ENGINE_SILENCE_S,
            v if v.is_number() && v.as_f64().unwrap() >= 0.0 => v.as_f64().unwrap(),
            v => exit_msg(&format!("[strata] config \"engine_silence_s\" must be a number of seconds >= 0 (0 = no limit), not {}", py::py_repr(v))),
        };
        effort_end = effort_end_args(&cfgv, &exe, &*tok).unwrap_or_else(|e| exit_msg(&format!("[strata] config {e}")));
        let mut args = engine_args(&cfgv);
        add_auto_cache(&mut args);
        args.extend(effort_end.clone().unwrap_or_default());
        let spawn = Spawn { exe, args, cwd: cwd.clone(), log: get(&cfgv, "log").as_str().map(String::from), env, jail: get(&cfgv, "sandbox") != &json!(false) };
        let e = match StrataEngine::new(spawn, lazy) {
            Ok(e) => e,
            Err(err) => exit_msg(&format!("[strata] {err}")),
        };
        *e.silence_s.lock().unwrap() = silence;
        warn_tight_ram(e.info.lock().unwrap().get("arena_mib").and_then(Value::as_i64));
        engine = Engine::Strata(e);
    } else {
        effort_end = None;
        sampling_defaults = Map::new();
        let scripts = if a.script.is_empty() { vec!["Thinking about it.</think>\n\nHello from the mock engine.".to_string()] } else { a.script.clone() };
        engine = Engine::Mock(Arc::new(MockEngine::new(&*tok, &scripts, 32768, 0.0)));
    }
    let tpl_path = tpath.join("chat_template.jinja");
    let tpl_path = if afm { root().join("server/afm_template.jinja") } else if tpl_path.exists() { tpl_path } else { root().join("server/chat_template.jinja") };
    let template = strata_server::template::ChatTemplate::load(&tpl_path).unwrap_or_else(|e| exit_msg(&format!("[strata] the chat template {}: {e}", tpl_path.display())));
    // ------------------------------------------------------------------ the settings
    let mut set = Settings { sampling_defaults, fit_max_tokens: a.fit_max_tokens || get(&cfgv, "fit_max_tokens") == &json!(true), ..Default::default() };
    set.reasoning_close_retry = get(&cfgv, "reasoning_close_retry") == &json!(true);
    set.system_prompt = system_prompt(&cfgv, a.config.as_deref()).unwrap_or_else(|e| exit_msg(&e));
    if !get(&cfgv, "prevent_sleep").is_null() && !get(&cfgv, "prevent_sleep").is_boolean() {
        exit_msg(&format!("[strata] config \"prevent_sleep\" must be true or false, not {}", py::py_repr(get(&cfgv, "prevent_sleep"))));
    }
    if !get(&cfgv, "reasoning_effort").is_null() {
        match clean_shared_defaults(&json!({"reasoning_effort": get(&cfgv, "reasoning_effort")})) {
            Ok(m) => set.config_effort = m.get("reasoning_effort").and_then(Value::as_str).map(String::from),
            Err(e) => exit_msg(&format!("[strata] config {e}")),
        }
        if let Some(e) = &set.config_effort {
            say(&format!("thinking level: {e} (reasoning_effort in the config; a request, or the Chat settings shared with other apps, can set its own)"));
        }
    }
    set.codex_thread_titles = get(&cfgv, "codex_thread_titles") == &json!(true);
    set.codex_compaction_cache = get(&cfgv, "codex_compaction_cache") == &json!(true);
    let model = get(&cfgv, "model_name").as_str().filter(|m| !m.is_empty()).map(String::from).unwrap_or_else(|| model_name_from_args(&str_args(&cfgv)));
    // #297: aliases
    let aliases = get(&cfgv, "aliases");
    let list: Vec<String> = match aliases {
        Value::Null => vec![],
        Value::String(s) => s.split(',').map(String::from).collect(),
        Value::Array(x) if x.iter().all(Value::is_string) => x.iter().map(|v| v.as_str().unwrap().to_string()).collect(),
        v => exit_msg(&format!("[strata] config aliases={}: expected a list of model names", py::py_repr(v))),
    };
    for x in list.iter().map(|x| py::strip(x).to_string()) {
        if !x.is_empty() && x != model && !set.aliases.contains(&x) {
            set.aliases.push(x);
        }
    }
    if !set.aliases.is_empty() {
        say(&format!("model aliases: {}", set.aliases.join(", ")));
    }
    let env_key = std::env::var("STRATA_API_KEY").ok();
    let argv: Vec<String> = std::env::args().collect();
    let empty_arg = argv.iter().enumerate().any(|(i, x)| (x == "--api-key" && argv.get(i + 1).map(|v| py::strip(v).is_empty()).unwrap_or(false)) || py::strip(x) == "--api-key=");
    if env_key.as_deref().map(|k| py::strip(k).is_empty()).unwrap_or(false) || empty_arg {
        eprintln!("[strata] an API key was given but it is empty: set a key, or leave --api-key / STRATA_API_KEY out");
        std::process::exit(2);
    }
    let key_src = a.api_key.clone().filter(|k| !k.is_empty()).or(env_key).map(Value::from).unwrap_or_else(|| get(&cfgv, "api_key").clone());
    match net::api_key_of(&key_src) {
        Ok(k) => {
            set.api_key = net::key_list(&k);
            // .env's API_KEY: accepted too, next to the config's / the command line's (never in their place)
            if let Some(ek) = strata_server::dotenv::get(&dotenv, "API_KEY").map(|k| py::strip(k).to_string()).filter(|k| !k.is_empty()) {
                if !set.api_key.contains(&ek) {
                    set.api_key.push(ek);
                    say("API key: .env's API_KEY is accepted too");
                }
            }
        }
        Err(e) => {
            eprintln!("[strata] {e}: set a key, or leave --api-key / STRATA_API_KEY / \"api_key\" out");
            std::process::exit(2);
        }
    }
    if set.api_key.is_empty() && cfg.contains_key("api_key") {
        eprintln!("{}", strata_server::engine::stamp("[strata] the api_key in the config is empty: this server has no API key (anyone who can reach it can use it); set one, or leave api_key out"));
    }
    set.cors_origins = net::origins_of(get(&cfgv, "cors_origins"), "cors_origins", true).unwrap_or_else(|e| exit_msg(&e));
    set.trusted_origins = net::origins_of(get(&cfgv, "trusted_origins"), "trusted_origins", false).unwrap_or_else(|e| exit_msg(&e));
    set.allowed_hosts = net::allowed_hosts_of(get(&cfgv, "allowed_hosts"), &std::env::var("STRATA_ALLOWED_HOSTS").unwrap_or_default()).unwrap_or_else(|e| exit_msg(&format!("[strata] config {e}")));
    if !set.allowed_hosts.is_empty() {
        if set.allowed_hosts.iter().any(|h| h == "*") {
            say("Host check off: any name reaches this server (allowed_hosts \"*\")");
        } else {
            say(&format!("also answers to the host names {} (allowed_hosts)", set.allowed_hosts.join(", ")));
        }
    }
    set.api_monitor = a.api_monitor || get(&cfgv, "api_monitor") == &json!(true);
    let image = strata_server::imagegen::ImageGen::from_config(get(&cfgv, "image")).unwrap_or_else(|e| exit_msg(&format!("[strata] config {e}")));
    let video = strata_server::videogen::VideoGen::from_config(get(&cfgv, "video")).unwrap_or_else(|e| exit_msg(&format!("[strata] config {e}")));
    if let Some(v) = &video {
        say(&format!("videos: POST /v1/videos runs sd-cli with {} (a job: poll GET /v1/videos/{{id}}; the chat model is unloaded for one when less than {:.0} GB is free)", Path::new(&v.low).file_name().unwrap().to_string_lossy(), v.need_gb));
    }
    if let Some(i) = &image {
        say(&format!("pictures: POST /v1/images/generations runs strata-image with {} (the chat model is unloaded for one when less than {:.0} GB is free)", Path::new(i.hunyuan.as_ref().or(i.dit.as_ref()).unwrap()).file_name().unwrap().to_string_lossy(), i.need_gb));
    }
    if set.api_monitor {
        say(&format!("API request monitor on (/api-monitor): the last 100 requests' prompts and answers are kept in memory{}", if set.api_key.is_empty() { "; anyone who can reach this server can read them (no API key)" } else { "" }));
    }
    if !set.cors_origins.is_empty() {
        say(&format!("CORS on /v1/* for {}{}", set.cors_origins.join(", "), if set.api_key.is_empty() && set.cors_origins.iter().any(|o| o == "*") { " - WARNING: any web page may use the model (no API key)" } else { "" }));
    }
    set.idle_unload_s = a.idle_unload.unwrap_or_else(|| get(&cfgv, "idle_unload_s").as_f64().unwrap_or(0.0));
    set.min_free_vram_mib = a.min_free_vram_mib.unwrap_or_else(|| get(&cfgv, "min_free_vram_mib").as_i64().unwrap_or(0));
    set.before_load = a.before_load.clone().map(Value::from).or_else(|| Some(get(&cfgv, "before_load").clone()).filter(truthy));
    let slot = a.slot_save_path.clone().map(|p| (p, None)).or_else(|| get(&cfgv, "slot_save_path").as_str().map(|p| (p.to_string(), get(&cfgv, "cwd").as_str().map(String::from))));
    if let Some((p, base)) = slot {
        if py::strip(&p).is_empty() || p.chars().any(|c| (c as u32) < 32) {
            exit_msg("[strata] slot_save_path must be a directory path (a string)");
        }
        let path = match &base {
            Some(b) if !Path::new(&p).is_absolute() => Path::new(b).join(&p),
            _ => PathBuf::from(&p),
        };
        let path = std::path::absolute(&path).unwrap_or(path);
        {
            use std::os::unix::fs::DirBuilderExt;
            let _ = std::fs::DirBuilder::new().recursive(true).mode(0o700).create(&path);
        }
        if !path.is_dir() {
            exit_msg(&format!("[strata] slot_save_path {} is not a directory", path.display()));
        }
        say(&format!("slot save/restore on: {} (session files are kept until deleted: about 1.2 GB per 63K-token conversation; restore only files this server wrote)", path.display()));
        set.slot_save_path = Some(path.to_string_lossy().into_owned());
    }
    let mode = get(&cfgv, "anthropic_thinking").as_str().unwrap_or("model");
    if !get(&cfgv, "anthropic_thinking").is_null() && mode != "model" && mode != "on_request" {
        exit_msg(&format!("[strata] config anthropic_thinking must be \"model\" or \"on_request\", not {}", py::py_repr(get(&cfgv, "anthropic_thinking"))));
    }
    set.anthropic_think_unasked = mode == "model";
    match get(&cfgv, "tool_call_recovery") {
        Value::Null => {}
        Value::Bool(b) => set.tool_call_recovery = *b,
        v => exit_msg(&format!("[strata] config \"tool_call_recovery\" must be true or false, not {}", py::py_repr(v))),
    }
    set.repeat_stop_tokens = match get(&cfgv, "repeat_stop_tokens") {
        Value::Null => REPEAT_STOP_TOKENS,
        v if v.is_i64() && v.as_i64().unwrap() >= 0 => v.as_i64().unwrap(),
        v => exit_msg(&format!("[strata] config \"repeat_stop_tokens\" must be a whole number >= 0 (0 = off), not {}", py::py_repr(v))),
    };
    set.effort_end = effort_end.is_some();
    set.reasoning_budget_tokens = get(&cfgv, "reasoning_budget_tokens").clone();
    set.reasoning_loop_recovery = match get(&cfgv, "reasoning_loop_recovery") {
        Value::Null | Value::Bool(false) => None,
        Value::Bool(true) => Some("recover".into()),
        Value::String(s) if s == "stop" || s == "recover" => Some(s.clone()),
        v => exit_msg(&format!("[strata] config \"reasoning_loop_recovery\" must be false, \"stop\" or \"recover\", not {}", py::py_repr(v))),
    };
    set.config_path = a.config.clone();
    if let Some(c) = &a.config {
        set.shared_path = Some(format!("{}.shared-settings.json", Path::new(c).with_extension("").display()));
    }
    let host_names = net::host_names_for(&host, &set.allowed_hosts, &set.trusted_origins);
    set.host_names = host_names;
    let mut svc = Service::new(engine.clone(), tok, template, model, vision.clone(), set);
    // left unset (null) when the config has none: a request without a budget then keeps room to answer (thinking_cap)
    if !svc.set.reasoning_budget_tokens.is_null() {
        match svc.reasoning_budget(&json!({})) {
            Err(e) => exit_msg(&format!("[strata] config {e}")),
            Ok(Some(b)) => say(&format!("thinking budget: {b} tokens (reasoning_budget_tokens; a request can set its own)")),
            Ok(None) => {}
        }
    }
    if let Some(p) = &svc.set.shared_path {
        if let Some(shared) = std::fs::read_to_string(p).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()).and_then(|v| clean_shared_defaults(&v).ok()) {
            if !shared.is_empty() {
                say(&format!("other apps use the Chat settings: {}", shared.iter().map(|(k, v)| format!("{k}={}", py::py_str(v))).collect::<Vec<_>>().join(", ")));
            }
            *svc.shared.lock().unwrap() = shared;
        }
    }
    let hub = Arc::new(hub);
    svc.mcp = Some(hub.clone());
    svc.image = image.map(Arc::new);
    svc.video = video.map(Arc::new);
    let n_servers = hub.servers.lock().unwrap().len();
    if n_servers > 0 {
        let names: Vec<String> = hub.servers.lock().unwrap().keys().cloned().collect();
        say(&format!("starting {n_servers} MCP server{} for the web app's chat: {}", if n_servers != 1 { "s" } else { "" }, names.join(", ")));
    }
    hub.start();
    let svc = Arc::new(svc);
    svc.start_telemetry();
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().max_blocking_threads(1024).build().expect("runtime");
    let app = http::App::new(svc.clone());
    if !handoff_from.is_empty() {
        // the old server keeps answering until this one can: the model is loaded before the port is taken
        say("handoff: loading the model before taking the port ...");
        if let Err(e) = svc.ensure_loaded() {
            die(&format!("strata-server: error: handoff: the model did not load ({e}); the server on port {} keeps running", a.port));
        }
    }
    let listener = rt.block_on(listen(&addr)).unwrap_or_else(|e| die(&format!("strata-server: error: {}", listen_problem(&host, a.port, &e))));
    let router = axum::Router::new().fallback(http::handle).with_state(app);
    // SIGUSR2 (a newer server took the port over): stop accepting, finish every request in flight, then stop; the
    // agents' conversations go to the new engine as they come free (docs/BLUE_GREEN.md)
    let drained = Arc::new(tokio::sync::Notify::new());
    let handing = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let hdir = handoff_dir(a.port);
    let strata_engine = engine.strata().cloned();
    let mut usr2 = rt.block_on(async { tokio::signal::unix::signal(tokio::signal::unix::SignalKind::user_defined2()) }).expect("signal");
    rt.spawn({
        let (drained, handing, hdir) = (drained.clone(), handing.clone(), hdir.clone());
        async move {
            let (handing, hdir, eng) = (handing.clone(), hdir.clone(), strata_engine.clone());
            let drain = async move {
                usr2.recv().await;
                say("handed over: no new requests here; finishing the ones in flight, then stopping");
                strata_server::clients::stop_restarts();
                handing.store(true, std::sync::atomic::Ordering::SeqCst);
                if let Some(e) = eng {
                    // now, and again each time a slot's request ends: those agents' state goes over one by one
                    std::thread::spawn(move || {
                        let mut busy_before = usize::MAX;
                        while !e.ended.load(std::sync::atomic::Ordering::SeqCst) {
                            let busy = e.slots_view().iter().filter(|s| s["state"] != "idle").count();
                            if busy < busy_before && e.handoff("EXPORT", &hdir).is_err() {
                                return;
                            }
                            busy_before = busy;
                            std::thread::sleep(std::time::Duration::from_millis(300));
                        }
                    });
                }
            };
            // TCP keepalive on every connection: a non-streaming answer is minutes of silence, and a client
            // whose connection died on the way (2026-11-04: deep research in Docker to the M1's LAN address,
            // "connection closed before message completed" 226 s in) left its generation running to the end
            // while its retry ran beside it; the probes keep such paths open and find a dead peer, which
            // ends the request as a hang-up does
            let listener = axum::serve::ListenerExt::tap_io(listener, |tcp| {
                let ka = socket2::TcpKeepalive::new().with_time(std::time::Duration::from_secs(30)).with_interval(std::time::Duration::from_secs(10));
                let _ = socket2::SockRef::from(&*tcp).set_tcp_keepalive(&ka);
            });
            let _ = axum::serve(listener, router.into_make_service_with_connect_info::<std::net::SocketAddr>()).with_graceful_shutdown(drain).await;
            drained.notify_one();
        }
    });
    for pid in &handoff_from {
        say(&format!("handoff: port {} is this server's now; pid {pid} finishes its requests and stops", a.port));
        unsafe { libc::kill(*pid, libc::SIGUSR2) };
    }
    if let (false, Some(e)) = (handoff_from.is_empty(), engine.strata().cloned()) {
        // the old server's agents, as it writes them: into this engine's conversation cache
        let (blue, dir) = (handoff_from.clone(), handoff_dir(a.port));
        std::thread::spawn(move || {
            let mut after = 0;
            while after < 10 {
                if !blue.iter().any(|p| unsafe { libc::kill(*p, 0) } == 0) {
                    after += 1;
                }
                let waiting = std::fs::read_dir(&dir).map(|d| d.flatten().any(|f| f.path().extension().is_some_and(|x| x == "sseq"))).unwrap_or(false);
                if waiting && e.handoff("IMPORT", &dir).is_err() {
                    return;
                }
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        });
    }
    svc.start_idle_unload();
    let here = if ["0.0.0.0", "", "::"].contains(&host.as_str()) { "127.0.0.1".to_string() } else { host.clone() };
    println!("ready: http://{here}:{}/v1  (OpenAI: /v1/chat/completions, Anthropic: /v1/messages, context {} tokens{}{})", a.port, engine.max_context(), if vision.is_some() { ", images on" } else { "" }, if svc.set.api_key.is_empty() { "" } else { ", API key required" });
    if !client_engine {
        strata_server::voice::autostart(a.port, svc.set.api_key.first().map(String::as_str)); // "Hey ...": on again if it was on
        strata_server::schedules::start(a.port, svc.set.api_key.first().cloned()); // scheduled prompts (missed ones run once)
        strata_server::hub::start(a.port, svc.set.api_key.first().cloned()); // the AI-cluster connection (Hub tab), when it is on
    }
    strata_server::clients::spawn_all(a.port, !handoff_from.is_empty());
    let args = str_args(&cfgv);
    let gguf = args[..args.len().saturating_sub(1)].iter().position(|x| x == "--gguf").map(|i| Path::new(&args[i + 1]).file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default()).unwrap_or_default();
    println!("       model: {}{}", svc.model, if gguf.is_empty() { String::new() } else { format!("  ({gguf})") });
    if !svc.set.api_key.is_empty() {
        // .env's key is not printed (the log is no place for it): only that it is accepted
        let from_env = strata_server::dotenv::get(&dotenv, "API_KEY").map(|k| py::strip(k).to_string());
        let shown: Vec<&str> = svc.set.api_key.iter().map(String::as_str).filter(|k| Some(*k) != from_env.as_deref()).collect();
        let env_note = if from_env.as_deref().is_some_and(|k| svc.set.api_key.iter().any(|x| x == k)) { ".env's API_KEY" } else { "" };
        println!("       API key: {}", shown.into_iter().chain((!env_note.is_empty()).then_some(env_note)).collect::<Vec<_>>().join(", "));
    }
    println!("       open http://{here}:{}/ in a browser to chat; close this window to stop the model", a.port);
    if !["127.0.0.1", "localhost", "::1"].contains(&host.as_str()) {
        let ips = net::lan_addresses();
        for ip in &ips {
            println!("       from other devices: http://{ip}:{}/   (API: http://{ip}:{}/v1)", a.port, a.port);
        }
        if ips.is_empty() {
            println!("       from other devices: http://<this PC's IP address>:{}/", a.port);
        }
        if svc.set.api_key.is_empty() {
            println!("       WARNING: no API key - anyone on your network can use this model. Add \"api_key\": \"...\" to the config (clients send it as their API key; the web page asks for it)");
        }
    }
    if a.open && get(&cfgv, "open_browser") != &json!(false) {
        let _ = std::process::Command::new("open").arg(format!("http://{}:{}/", if ["0.0.0.0", ""].contains(&host.as_str()) { "127.0.0.1" } else { host.as_str() }, a.port)).spawn();
    }
    // SIGTERM (docker stop, launchd) and Ctrl+C take the same path: QUIT to the engine, the encoder, the MCP servers
    rt.block_on(async {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("signal");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {},
            _ = term.recv() => {},
            _ = drained.notified() => {},
        }
    });
    println!();
    say("stopping (Ctrl+C again to end the engine at once) ...");
    // ponytail: after a handoff the old clients' engines stop here too, cutting what they still answer; their
    // replacements took their ports already
    strata_server::clients::stop_all();
    if handing.load(std::sync::atomic::Ordering::SeqCst) {
        if let Some(e) = engine.strata() {
            let _ = e.handoff("EXPORT", &handoff_dir(a.port)); // the last ones, before QUIT (the engine reads in order)
        }
    }
    std::thread::spawn(|| {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(async { let _ = tokio::signal::ctrl_c().await; });
        std::process::exit(130);
    });
    if let Some(e) = engine.strata() {
        let _ = e.close();
    }
    if let Some(v) = &vision {
        v.shutdown();
    }
    hub.close();
    if let Some(v) = &svc.video {
        v.shutdown();
    }
    say("stopped");
}

#[cfg(test)]
mod port_holder_tests {
    use super::servers_holding;

    #[test]
    fn a_servers_half_started_child_is_not_another_program() {
        let srv = "/x/engine-metal/strata-server --engine strata --port 8190";
        let found = vec![(10, 1, srv.to_string()), (11, 10, "(lsof)".to_string()), (12, 99, "nginx: master".to_string())];
        assert_eq!(servers_holding(found), vec![(10, srv.to_string()), (12, "nginx: master".to_string())]);
    }

    #[test]
    fn a_port_held_by_no_one_is_checked_again_until_it_is_free() {
        use std::io::{Error, ErrorKind};
        let z = std::time::Duration::ZERO;
        let mut n = 0;
        let r = super::port_settled(|| { n += 1; if n < 3 { Err(Error::from(ErrorKind::AddrInUse)) } else { Ok(()) } }, || true, 10, z);
        assert!(r.is_ok() && n == 3);
        // held by a process: no retry, the caller looks at who holds it
        let mut m = 0;
        let r = super::port_settled(|| { m += 1; Err(Error::from(ErrorKind::AddrInUse)) }, || false, 10, z);
        assert!(r.is_err() && m == 1);
        // never free: gives up after the tries
        let mut k = 0;
        assert!(super::port_settled(|| { k += 1; Err(Error::from(ErrorKind::AddrInUse)) }, || true, 3, z).is_err());
        assert_eq!(k, 4);
    }
}

#[cfg(test)]
mod engine_guard_tests {
    use super::engines_in;

    #[test]
    fn finds_running_engines_only() {
        let ps = "    1 /sbin/launchd\n 2148 /home/dadmin/Strata/engine-metal/strata-metal --serve --gguf m.gguf\n 2141 engine-metal/strata-server --engine strata\n  77 strata-metal\n 99 /usr/bin/strata-metal-helper\n 120 /Users/x/My Apps/Strata/engine-metal/strata-metal --serve\n 17790 /Users/x/Strata/engine-metal/strata-metal --plan --gguf m.gguf --max-context 8192\n 300 vim notes-about-strata-metal.txt\n";
        assert_eq!(engines_in(ps), ["2148", "77", "120"]);
        assert!(engines_in("  1 /sbin/init\n").is_empty());
    }
}

#[cfg(test)]
mod cache_tests {
    use super::{auto_cache_mib, model_gib};

    #[test]
    fn the_cache_follows_the_ram_beside_the_model() {
        assert_eq!(auto_cache_mib(128.0, 22.0), 16384); // M5 Max, Qwen3.6
        assert_eq!(auto_cache_mib(128.0, 86.0), 8192); // M5 Max, GLM-5.3
        assert_eq!(auto_cache_mib(16.0, 1.8), 4096); // M1, Sheared-LLaMA
        assert_eq!(auto_cache_mib(16.0, 8.4), 2048); // M1, LFM2.5-8B (measured: 99% reused with 2 GB)
        assert_eq!(auto_cache_mib(16.0, 11.3), 2048); // M1, GPT-OSS-20B
        assert_eq!(auto_cache_mib(16.0, 13.0), 0); // no room
    }

    #[test]
    fn a_split_model_counts_all_its_shards() {
        let d = std::env::temp_dir().join(format!("strata-shards-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        for (n, len) in [("m-00001-of-00002.gguf", 3), ("m-00002-of-00002.gguf", 5), ("other.gguf", 7)] {
            std::fs::write(d.join(n), vec![0u8; len]).unwrap();
        }
        let gib = |f: &str| model_gib(&["--gguf".into(), d.join(f).to_string_lossy().into_owned()]).unwrap() * 1073741824.0;
        assert_eq!(gib("m-00001-of-00002.gguf").round(), 8.0);
        assert_eq!(gib("other.gguf").round(), 7.0);
        std::fs::remove_dir_all(&d).unwrap();
    }
}

#[cfg(test)]
mod port_tests {
    use super::port_free;

    #[test]
    fn a_wildcard_host_is_refused_when_another_program_has_the_loopback_port() {
        // a Docker port: 127.0.0.1:P held by another program, this server asked for 0.0.0.0:P
        let other = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let p = other.local_addr().unwrap().port();
        assert!(std::net::TcpListener::bind(("0.0.0.0", p)).is_ok(), "the wildcard alone binds beside it (SO_REUSEADDR)");
        assert_eq!(port_free("0.0.0.0", p).unwrap_err().kind(), std::io::ErrorKind::AddrInUse);
        assert_eq!(port_free("127.0.0.1", p).unwrap_err().kind(), std::io::ErrorKind::AddrInUse);
        drop(other);
        assert!(port_free("0.0.0.0", p).is_ok());
    }

    #[test]
    fn a_loopback_host_is_refused_when_another_server_has_the_wildcard_port() {
        // a Strata server on "host": "0.0.0.0", this one asked for 127.0.0.1: --replace must see that one
        let other = std::net::TcpListener::bind("0.0.0.0:0").unwrap();
        let p = other.local_addr().unwrap().port();
        assert_eq!(port_free("127.0.0.1", p).unwrap_err().kind(), std::io::ErrorKind::AddrInUse);
        drop(other);
        assert!(port_free("127.0.0.1", p).is_ok());
    }
}
