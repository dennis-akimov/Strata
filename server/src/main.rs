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
use strata_server::engine::{py_g, Engine, MockEngine, Spawn, StrataEngine, ENGINE_SILENCE_S};
use strata_server::py::{self, get, truthy};
use strata_server::service::{clean_shared_defaults, sampling_defaults_from_config, Service, Settings, REPEAT_STOP_TOKENS};
use strata_server::tokenizer::{ByteTokenizer, Tokenize, Tokenizer};
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
    pids.into_iter()
        .map(|p| {
            let cmd = std::process::Command::new("ps").args(["-o", "command=", "-p", &p.to_string()]).output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
            (p, cmd)
        })
        .collect()
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
    strata_server::firewall::start(get(&cfgv, "firewall"), a.engine == "strata"); // history, connection sampler, the kernel's sandbox log
    let host = a.host.clone().or_else(|| get(&cfgv, "host").as_str().filter(|h| !h.is_empty()).map(String::from)).unwrap_or_else(|| "127.0.0.1".into());
    // before the minutes of loading: is the port free?
    let addr = format!("{}:{}", if host.contains(':') { format!("[{host}]") } else { host.clone() }, a.port);
    if let Err(e) = std::net::TcpListener::bind(&addr) {
        let holders = if e.kind() == std::io::ErrorKind::AddrInUse { port_holders(a.port) } else { vec![] };
        let ours: Vec<&(i32, String)> = holders.iter().filter(|(_, c)| is_strata_server(c)).collect();
        if ours.is_empty() || ours.len() != holders.len() {
            let others: Vec<String> = holders.iter().filter(|(_, c)| !is_strata_server(c)).map(|(p, c)| format!("pid {p}: {}", py::chars_prefix(c, 120))).collect();
            die(&format!("strata-server: error: {}{}", listen_problem(&host, a.port, &e), if others.is_empty() { String::new() } else { format!(" (held by {})", others.join("; ")) }));
        }
        let (pid, cmd) = ours[0];
        let config = cmd.split_once("--config").map(|(_, r)| r.split_whitespace().next().unwrap_or("?").trim_matches('"').to_string()).unwrap_or_else(|| "?".into());
        say(&format!("port {} is held by another Strata server: pid {pid}, {}", a.port, Path::new(&config).file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or(config)));
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
    let tpath = PathBuf::from(get(&cfgv, "tokenizer").as_str().map(String::from).or(a.tokenizer.clone()).unwrap_or_else(|| root().join("pack/full/tokenizer").to_string_lossy().into_owned()));
    if a.engine == "strata" && !tpath.join("vocab.json").exists() {
        die(&format!("strata-server: error: the model's tokenizer is missing ({}); run setup again", tpath.join("vocab.json").display()));
    }
    let tok: Arc<dyn Tokenize> = if tpath.join("vocab.json").exists() {
        match Tokenizer::from_dir(&tpath) {
            Ok(t) => Arc::new(t),
            Err(e) => die(&format!("strata-server: error: the tokenizer in {}: {e}", tpath.display())),
        }
    } else {
        Arc::new(ByteTokenizer)
    };
    let hub = match mcp::hub_from_config(&cfgv, a.mcp_config.as_deref()) {
        Ok(Some(h)) => h,
        Ok(None) => mcp::McpHub::new(Default::default(), mcp::settings_from(&cfgv).unwrap_or_else(|e| exit_msg(&e)), &[]),
        Err(e) => exit_msg(&e),
    };
    let mut vision = None;
    let sampling_defaults;
    let engine: Engine;
    let effort_end;
    if a.engine == "strata" {
        if cfg.is_empty() {
            die("strata-server: error: --engine strata needs --config");
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
    let tpl_path = if tpl_path.exists() { tpl_path } else { root().join("server/chat_template.jinja") };
    let template = strata_server::template::ChatTemplate::load(&tpl_path).unwrap_or_else(|e| exit_msg(&format!("[strata] the chat template {}: {e}", tpl_path.display())));
    // ------------------------------------------------------------------ the settings
    let mut set = Settings { sampling_defaults, fit_max_tokens: a.fit_max_tokens || get(&cfgv, "fit_max_tokens") == &json!(true), ..Default::default() };
    set.reasoning_close_retry = get(&cfgv, "reasoning_close_retry") == &json!(true);
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
        Ok(k) => set.api_key = net::key_list(&k),
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
    if svc.set.reasoning_budget_tokens.is_null() {
        svc.set.reasoning_budget_tokens = json!(0);
    } else {
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
    let listener = rt.block_on(tokio::net::TcpListener::bind(&addr)).unwrap_or_else(|e| die(&format!("strata-server: error: {}", listen_problem(&host, a.port, &e))));
    let router = axum::Router::new().fallback(http::handle).with_state(app);
    rt.spawn(async move {
        let _ = axum::serve(listener, router.into_make_service_with_connect_info::<std::net::SocketAddr>()).await;
    });
    svc.start_idle_unload();
    let here = if ["0.0.0.0", "", "::"].contains(&host.as_str()) { "127.0.0.1".to_string() } else { host.clone() };
    println!("ready: http://{here}:{}/v1  (OpenAI: /v1/chat/completions, Anthropic: /v1/messages, context {} tokens{}{})", a.port, engine.max_context(), if vision.is_some() { ", images on" } else { "" }, if svc.set.api_key.is_empty() { "" } else { ", API key required" });
    let args = str_args(&cfgv);
    let gguf = args[..args.len().saturating_sub(1)].iter().position(|x| x == "--gguf").map(|i| Path::new(&args[i + 1]).file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default()).unwrap_or_default();
    println!("       model: {}{}", svc.model, if gguf.is_empty() { String::new() } else { format!("  ({gguf})") });
    if !svc.set.api_key.is_empty() {
        println!("       API key: {}", svc.set.api_key.join(", "));
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
        }
    });
    println!();
    say("stopping (Ctrl+C again to end the engine at once) ...");
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
