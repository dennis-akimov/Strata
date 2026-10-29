//! The resident engine (serve/server.py's StrataEngine and MockEngine): `strata-metal --serve` reads `GEN <max_new>
//! [key=value ...] <ids>` lines and streams `T <id>` lines, then `DONE ...`.  One request at a time behind the
//! service's FIFO, or several in the engine's batch slots (`BGEN` / `BT` / `BDONE`, the config's "parallel").

use crate::errors::Fail;
use crate::py;
use crate::say;
use crate::sync::{since, Event, Gen, Sem, Stop, Yielder};
use crossbeam_channel::{unbounded, Receiver, RecvTimeoutError, Sender};
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Stdio};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering::SeqCst};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

pub type Line = Option<String>;
pub type Stats = Arc<Mutex<Option<Map<String, Value>>>>;

/// One generation: its token ids (None: a heartbeat while the engine is quiet), and the DONE figures it read.
pub struct Generation {
    pub tokens: Gen<Option<u32>, Fail>,
    pub stats: Stats,
}

/// The engine's own words for a verifier that cannot go on (#267): it must not take another request.
const FATAL_PREFIXES: [&str; 4] = [
    "ERR verify: timed out at layer ",
    "ERR verify batch: timed out at layer ",
    "ERR verify: an earlier window never finished",
    "ERR verify batch: an earlier window never finished",
];
/// <|endoftext|>, <|im_end|>: the engine's default --eos-ids
pub const EOS_IDS: [u32; 2] = [248044, 248046];

pub const ENGINE_SILENCE_S: f64 = 300.0;
const SESSION_WAIT_MAX_S: u64 = 3600;
const PP_CHUNK_MAX: usize = 32768;
const PP_FLOOR_TOK_S: f64 = 50.0;
const PP_SLACK: f64 = 3.0;
const STALL_CPU_EPS_S: f64 = 0.5;
const STALL_IO_EPS_B: u64 = 1 << 20;
const RESTART_RETRY_S: f64 = 15.0;
const ORPHAN_SLOT_S: f64 = 3.0;
const YIELDS_MAX: usize = 2;
const SOLO_AGAIN_MAX: usize = 2;
const SOLO_AGAIN_MIN_LEFT: i64 = 32;
const SLOT_PREFIX_MIN: usize = 512;

/// STRATA_*_S overrides (0 = wait for ever): None when off.
pub fn timeout_env(name: &str, default: f64) -> Option<f64> {
    let v = std::env::var(name).ok().and_then(|v| v.trim().parse::<f64>().ok()).unwrap_or(default);
    if v <= 0.0 { None } else { Some(v) }
}

pub fn engine_ready_s() -> Option<f64> {
    timeout_env("STRATA_ENGINE_READY_S", 900.0)
}

pub fn engine_stall_s() -> Option<f64> {
    timeout_env("STRATA_ENGINE_STALL_S", 90.0)
}

fn btrace(msg: &str) {
    if std::env::var("STRATA_BATCH_TRACE").map(|v| !v.is_empty()).unwrap_or(false) {
        eprintln!("[batch-trace] {msg}");
    }
}

/// #1317: True when two (cpu seconds, io bytes) samples of the engine show no work between them.
pub fn engine_frozen(base: (f64, u64), now: (f64, u64)) -> bool {
    (now.0 - base.0) <= STALL_CPU_EPS_S && now.1.saturating_sub(base.1) <= STALL_IO_EPS_B
}

/// (CPU seconds, I/O bytes) a process has used so far (macOS proc_pid_rusage), None when they cannot be read.
#[allow(deprecated)] // mach_timebase_info: rusage times are mach ticks (125/3 ns on Apple Silicon)
pub fn process_activity(pid: u32) -> Option<(f64, u64)> {
    #[cfg(target_os = "macos")]
    unsafe {
        let mut info: libc::rusage_info_v2 = std::mem::zeroed();
        let r = libc::proc_pid_rusage(pid as i32, libc::RUSAGE_INFO_V2, &mut info as *mut _ as *mut libc::rusage_info_t);
        if r != 0 {
            return None;
        }
        let mut tb = libc::mach_timebase_info { numer: 0, denom: 0 };
        libc::mach_timebase_info(&mut tb);
        let ticks = (info.ri_user_time + info.ri_system_time) as f64;
        let ns = if tb.denom > 0 { ticks * tb.numer as f64 / tb.denom as f64 } else { ticks };
        Some((ns / 1e9, info.ri_diskio_bytesread + info.ri_diskio_byteswritten))
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = pid;
        None
    }
}

/// The sampling keys of one request, in the engine's spelling, after `GEN <max_new>`.
pub fn sampling_keys(s: &Map<String, Value>) -> String {
    let num = |k: &str| -> Option<f64> {
        match s.get(k) {
            Some(Value::Number(n)) => n.as_f64(),
            _ => None,
        }
    };
    let int = |k: &str| -> Option<i64> {
        match s.get(k) {
            Some(Value::Number(n)) if n.is_i64() || n.is_u64() => n.as_i64(),
            _ => None,
        }
    };
    let r = crate::pyjson::float_repr;
    let mut keys = String::new();
    if let Some(t) = num("temperature").filter(|t| *t > 0.0) {
        keys += &format!(" temperature={}", r(t));
    }
    if let Some(tp) = num("top_p").filter(|t| *t < 1.0) {
        keys += &format!(" top_p={}", r(tp));
    }
    if let Some(tk) = int("top_k").filter(|t| *t >= 0) {
        keys += &format!(" top_k={}", if (1..=64).contains(&tk) { tk } else { 64 });
    }
    if let Some(mp) = num("min_p").filter(|m| *m > 0.0 && *m <= 1.0) {
        keys += &format!(" min_p={}", r(mp));
    }
    let rp = num("repetition_penalty").filter(|v| *v != 1.0);
    let pf = num("frequency_penalty").filter(|v| *v != 0.0);
    let pp = num("presence_penalty").filter(|v| *v != 0.0);
    if let Some(v) = rp {
        keys += &format!(" penalty_repeat={}", r(v));
    }
    if let Some(v) = pf {
        keys += &format!(" penalty_freq={}", r(v));
    }
    if let Some(v) = pp {
        keys += &format!(" penalty_present={}", r(v));
    }
    if rp.is_some() || pf.is_some() || pp.is_some() {
        match int("penalty_last_n").filter(|n| *n > 0) {
            Some(n) => keys += &format!(" penalty_last_n={n}"),
            None => keys += " penalty_last_n=64",
        }
    }
    if let Some(seed) = int("seed").filter(|s| *s > 0) {
        keys += &format!(" seed={seed}");
    }
    if let Some(tune) = s.get("strata_tune").and_then(Value::as_object) {
        for k in ["pcie_frac", "spec_min_p"] {
            if let Some(Value::Number(n)) = tune.get(k) {
                let v = n.as_f64().unwrap_or(-1.0);
                if (0.0..=1.0).contains(&v) {
                    keys += &format!(" {k}={}", r(v));
                }
            }
        }
        match tune.get("aux_cpus") {
            Some(Value::Bool(b)) => keys += &format!(" aux_cpus={}", *b as i32),
            Some(Value::Number(n)) if n.as_f64() == Some(0.0) || n.as_f64() == Some(1.0) => {
                keys += &format!(" aux_cpus={}", n.as_f64().unwrap() as i32)
            }
            _ => {}
        }
    }
    if s.get("strata_checkpoint") == Some(&Value::Bool(false)) {
        keys += " ckpt=0";
    }
    if let Some(Value::Number(pn)) = s.get("strata_prefix").and_then(|p| p.get("tokens")) {
        if let Some(pn) = pn.as_i64().filter(|n| *n > 0) {
            keys += &format!(" pin={pn}");
        }
    }
    if let Some(Value::Bool(on)) = s.get("experimental_speed_projection") {
        keys += &format!(" cvec={}", *on as i32);
    }
    keys + &crate::bias::engine_key(s.get("logit_bias"))
}

// ------------------------------------------------------------------------------------------------ start helpers
const DRAFT_HEAD_FAIL: &str = "the draft head does not fit";
const DRAFT_HEAD_HINT: &str = "a smaller draft vocabulary needs less VRAM: --draft-vocab cyrillic (English, code and the Cyrillic script) or --draft-vocab en (English and code, ~215 MiB less than the default). Start once with it - START-HERE.bat --draft-vocab en (Windows) or ./setup.sh --draft-vocab en - and the model keeps it; or a smaller --context in setup.";

fn read_from(path: &str, offset: u64, max: Option<u64>) -> Option<String> {
    let mut f = std::fs::File::open(path).ok()?;
    let size = f.metadata().ok()?.len();
    let start = match max {
        Some(m) => offset.max(size.saturating_sub(m)),
        None => offset,
    };
    f.seek(SeekFrom::Start(start)).ok()?;
    let mut b = vec![];
    f.read_to_end(&mut b).ok()?;
    Some(String::from_utf8_lossy(&b).into_owned())
}

/// #474: what to change when the engine stopped because the MTP draft head did not fit.
pub fn start_failure_hint(log: Option<&str>, offset: u64) -> String {
    let Some(text) = log.and_then(|l| read_from(l, offset, None)) else { return String::new() };
    if !text.contains(DRAFT_HEAD_FAIL) {
        return String::new();
    }
    let said: Vec<String> = text
        .lines()
        .filter(|x| py::strip(x).starts_with("strata mtp: ") && (x.contains("draft head over") || x.contains("hint:")))
        .map(|x| py::strip(x)["strata mtp: ".len()..].to_string())
        .collect();
    format!(". mtp: {DRAFT_HEAD_FAIL}. {}", if said.is_empty() { format!("Hint: {DRAFT_HEAD_HINT}") } else { said.join(" ") })
}

/// #496: the last n lines this start wrote to the engine log.
pub fn start_log_tail(log: Option<&str>, offset: u64, n: usize) -> String {
    let Some(log) = log else { return String::new() };
    let Ok(size) = std::fs::metadata(log).map(|m| m.len()) else { return String::new() };
    let start = offset.max(size.saturating_sub(64 * 1024));
    let Some(text) = read_from(log, start, None) else { return String::new() };
    let lines: Vec<&str> = text.lines().skip(if start > offset { 1 } else { 0 }).collect();
    let lines: Vec<&str> = lines.into_iter().filter(|x| !py::strip(x).is_empty()).map(|x| x.trim_end()).collect();
    let lines = &lines[lines.len().saturating_sub(n)..];
    if lines.is_empty() {
        return String::new();
    }
    format!("\nthe engine log's last lines:\n{}", lines.iter().map(|x| format!("  {x}")).collect::<Vec<_>>().join("\n"))
}

/// #505: what the start does with the experts, by the engine's flags.
fn experts_loading_words(args: &[String], size: &str) -> String {
    let has = |f: &str| args.iter().any(|a| a == f);
    if has("--resident-budget-gib") {
        let i = args.iter().position(|a| a == "--resident-budget-gib").unwrap();
        let n = args.get(i + 1).and_then(|v| v.parse::<f64>().ok()).map(|v| format!("up to {} GiB", py_g(v))).unwrap_or_else(|| "a RAM budget".into());
        return format!("loading the most-used experts into RAM ({n}; the rest are read from the model files as needed) and locking part of them for the GPU.");
    }
    if has("--resident-experts") {
        return format!("loading the experts the GPU does not hold into RAM (of {size}) and locking part of them for the GPU.");
    }
    if has("--mmap-experts") {
        return format!("mapping the experts from the model files ({size}, --mmap-experts): they are not loaded into RAM - the OS file cache reads them as the GPU's expert cache fills and as requests need them.");
    }
    format!("loading the experts into RAM ({size}) and locking part of them for the GPU.")
}

/// Python's `f"{x:g}"`.
pub fn py_g(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() { "-0".into() } else { "0".into() };
    }
    if !x.is_finite() {
        return if x.is_nan() { "nan".into() } else if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let exp = format!("{:.5e}", x.abs()).split_once('e').map(|(_, e)| e.parse::<i32>().unwrap()).unwrap();
    if (-4..6).contains(&exp) {
        let s = format!("{:.*}", (5 - exp).max(0) as usize, x);
        let s = if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s };
        s
    } else {
        let m = format!("{:.5e}", x);
        let (mant, e) = m.split_once('e').unwrap();
        let mant = if mant.contains('.') { mant.trim_end_matches('0').trim_end_matches('.') } else { mant };
        let e: i32 = e.parse().unwrap();
        format!("{mant}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
    }
}

/// While the engine starts, say in the server window what it is doing, from its log.
fn narrate_start(log_path: String, offset: u64, args: Vec<String>, done: Arc<Event>) {
    let mut gb = 0.0;
    if let Some(i) = args.iter().position(|a| a == "--native") {
        if let Some(p) = args.get(i + 1) {
            gb = std::fs::metadata(p).map(|m| m.len() as f64 / 1e9).unwrap_or(0.0);
        }
    }
    let size = if gb >= 1.0 { format!("about {gb:.0} GB") } else { "tens of GB".to_string() };
    let loading = experts_loading_words(&args, &size);
    let t0 = Instant::now();
    let mut last = Instant::now();
    let mut said: HashSet<&str> = HashSet::new();
    let mut tell = |key: &'static str, text: String, last: &mut Instant| {
        if said.insert(key) {
            *last = Instant::now();
            println!("{}", stamp(&text));
        }
    };
    tell("weights", "[strata] starting the engine: reading the model's weights ...".into(), &mut last);
    let mut pos = offset;
    while !done.wait(Some(Duration::from_millis(500))) {
        let chunk = read_from(&log_path, pos, None).unwrap_or_default();
        if let Some(cut) = chunk.rfind('\n').map(|c| c + 1) {
            pos += cut as u64;
            for line in chunk[..cut].lines() {
                if line.contains("PLE on") || line.contains("expert arena:") || line.contains("experts via mmap") {
                    tell("arena", format!("[strata] {loading}\n         YOUR PC CAN BE SLOW OR STOP RESPONDING FOR 1-3 MINUTES NOW - this is normal.\n         Please wait and don't close this window; the browser opens when it is ready."), &mut last);
                } else if line.contains(" loaded ") && line.contains("GiB at") {
                    let rest = py::strip(line.split_once(" loaded ").unwrap().1).to_string();
                    tell("loaded", format!("[strata] experts loaded: {rest} ({:.0} s so far)", since(t0)), &mut last);
                } else if line.contains("expert cache ") && line.contains(" slots, ") && !line.contains("auto") {
                    let n = py::strip(&line.split_once("expert cache ").unwrap().1.split(';').next().unwrap_or("").replace(" slots,", " experts,")).to_string();
                    tell("cache", format!("[strata] filling the GPU's expert cache ({n}) ..."), &mut last);
                } else if line.contains("session is up") {
                    tell("up", "[strata] almost ready ...".into(), &mut last);
                }
            }
        }
        if since(last) > 20.0 {
            last = Instant::now();
            println!("{}", stamp(&format!("[strata] still starting ({:.0} s) - please wait ...", since(t0))));
        }
    }
}

/// "[strata] " lines carry the time (the Python server's Stamped stream).
pub fn stamp(text: &str) -> String {
    let t = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
    text.split('\n')
        .map(|l| match l.strip_prefix("[strata] ") {
            Some(rest) => format!("[strata] {t} {rest}"),
            None => l.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// STRATA_REQUEST_LINES=1: one stdout line per finished request, from the engine's own summary in its log.
fn echo_requests(log_path: String, offset: u64) {
    let re = regex::Regex::new(r"prompt (?P<prompt>\d+) tokens = (?P<reused>\d+) reused \+ \d+(?: of \d+)? read in (?P<read>[\d.]+) ms \((?P<pp>[\d.]+) tok/s\), (?P<gen>\d+) generated in (?P<gen_ms>[\d.]+) ms \((?P<tg>[\d.]+) tok/s\)").unwrap();
    let Ok(f) = std::fs::File::open(&log_path) else { return };
    let mut r = BufReader::new(f);
    if r.seek(SeekFrom::Start(offset)).is_err() {
        return;
    }
    let mut line = String::new();
    loop {
        line.clear();
        match r.read_line(&mut line) {
            Ok(0) | Err(_) => {
                std::thread::sleep(Duration::from_millis(200));
                continue;
            }
            Ok(_) => {}
        }
        if let Some(m) = re.captures(&line) {
            let read_ms: f64 = m["read"].parse().unwrap_or(0.0);
            let gen_ms: f64 = m["gen_ms"].parse().unwrap_or(0.0);
            say(&format!("request prompt {} cached {} output {} prompt_read {:.0} ms total {:.0} ms prefill {} tok/s decode {} tok/s", &m["prompt"], &m["reused"], &m["gen"], read_ms, read_ms + gen_ms, &m["pp"], &m["tg"]));
        }
    }
}

static ECHOING: Mutex<Vec<PathBuf>> = Mutex::new(vec![]);

/// #596: the engine's conversation cache as its log tells it.
pub struct ConvCacheLog {
    key: Option<(String, u64)>,
    pos: u64,
    pub state: Map<String, Value>,
}

impl Default for ConvCacheLog {
    fn default() -> Self {
        let mut c = ConvCacheLog { key: None, pos: 0, state: Map::new() };
        c.reset();
        c
    }
}

impl ConvCacheLog {
    fn reset(&mut self) {
        self.state = json!({"parked": 0, "bytes": 0, "evictions": 0, "parks": 0, "restores": 0, "last_event": null, "last_tokens": null, "last_at": null})
            .as_object()
            .cloned()
            .unwrap();
    }

    pub fn poll(&mut self, path: Option<&str>, start: Option<u64>) -> Map<String, Value> {
        let (Some(path), Some(start)) = (path, start) else { return self.state.clone() };
        if self.key.as_ref() != Some(&(path.to_string(), start)) {
            self.key = Some((path.to_string(), start));
            self.pos = start;
            self.reset();
        }
        let Ok(size) = std::fs::metadata(path).map(|m| m.len()) else { return self.state.clone() };
        if size < self.pos {
            self.pos = 0;
        }
        if size - self.pos > 1 << 20 {
            self.pos = size - (1 << 20);
        }
        let Some(data) = read_from(path, self.pos, None) else { return self.state.clone() };
        let data = &data[..data.len().min((size - self.pos) as usize)];
        let end = data.rfind('\n').map(|e| e + 1).unwrap_or(0);
        self.pos += end as u64;
        let now = crate::sync::now();
        static EVENT: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        static DROPPED: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        let ev = EVENT.get_or_init(|| regex::Regex::new(r"conversation cache: (parked|skipped|restored) (\d+) tokens.*?parked=(\d+) bytes=(\d+)(?: evictions=(\d+))?").unwrap());
        let dr = DROPPED.get_or_init(|| regex::Regex::new(r"conversation cache: dropped \d+ superseded .*?parked=(\d+)").unwrap());
        for line in data[..end].lines() {
            if !line.contains("conversation cache:") {
                continue;
            }
            if let Some(m) = ev.captures(line) {
                let st = &mut self.state;
                st.insert("parked".into(), json!(m[3].parse::<i64>().unwrap_or(0)));
                st.insert("bytes".into(), json!(m[4].parse::<i64>().unwrap_or(0)));
                if let Some(e) = m.get(5) {
                    st.insert("evictions".into(), json!(e.as_str().parse::<i64>().unwrap_or(0)));
                }
                if &m[1] != "skipped" {
                    let k = if &m[1] == "parked" { "parks" } else { "restores" };
                    let n = st.get(k).and_then(Value::as_i64).unwrap_or(0) + 1;
                    st.insert(k.into(), json!(n));
                    st.insert("last_event".into(), json!(&m[1]));
                    st.insert("last_tokens".into(), json!(m[2].parse::<i64>().unwrap_or(0)));
                    st.insert("last_at".into(), json!(now));
                }
                continue;
            }
            if let Some(m) = dr.captures(line) {
                self.state.insert("parked".into(), json!(m[1].parse::<i64>().unwrap_or(0)));
            }
        }
        self.state.clone()
    }
}

// ------------------------------------------------------------------------------------------------ the process
pub struct Spawn {
    pub exe: String,
    pub args: Vec<String>,
    pub cwd: Option<String>,
    pub log: Option<String>,
    pub env: HashMap<String, String>,
    /// no network for the engine (Seatbelt, firewall::command); the run config's "sandbox": false turns it off
    pub jail: bool,
}

struct Proc {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    lines: Option<Receiver<Line>>,
    pump: Option<std::thread::JoinHandle<()>>,
}

#[derive(Default)]
struct Slots {
    busy: Vec<bool>,
    claims: HashSet<usize>,
    releasing: HashSet<usize>,
    orphan_seen: HashMap<usize, Instant>,
    held: Vec<Vec<u32>>,
    used: Vec<f64>,
    live: Vec<Option<Map<String, Value>>>,
    order: Vec<usize>,
    group: Vec<usize>,
    groups: usize,
    gs: usize,
    waiting: usize,
    admitting: usize,
    wait_lens: Vec<(u64, usize)>,
    next_wait_id: u64,
    ctl_epoch: u64,
}

#[derive(Default)]
struct Ctl {
    mode: &'static str,
    result: Option<(String, bool)>,
    yielded: Option<(usize, usize)>,
    erred: bool,
}

#[derive(Default, Clone)]
pub struct Live {
    pub progress: Option<(i64, i64)>,
    pub progress_ms: f64,
    pub reused: i64,
    pub prefill_tok_s_mean: Option<f64>,
}

pub struct StrataEngine {
    pub spawn: Spawn,
    pub session_save_reclaim: bool,
    pub model_path: String,
    pub log_path: Option<String>,
    pub silence_s: Mutex<f64>,
    proc: Mutex<Proc>,
    pub ended: AtomicBool,
    pub unloaded: AtomicBool,
    pub starting: AtomicBool,
    pub max_context: AtomicI64,
    pub known_ctx: AtomicI64,
    pub can_stop: AtomicBool,
    pub info: Mutex<Map<String, Value>>,
    pub live: Mutex<Live>,
    silent_note: Mutex<Option<String>>,
    last_err: Mutex<Option<String>>,
    pub log_start: AtomicU64,
    proc_gen: AtomicU64,
    batch: AtomicUsize,
    slots: Mutex<Slots>,
    slot_cv: Condvar,
    slot_rx: Mutex<Vec<Receiver<Line>>>,
    slot_tx: Arc<Mutex<Vec<Sender<Line>>>>,
    ctl: Sem,
    ctl_state: Mutex<Ctl>,
    pub gpu_busy: Mutex<Option<Box<dyn Fn() -> bool + Send + Sync>>>,
}

impl StrataEngine {
    pub fn new(spawn: Spawn, lazy: bool) -> Result<Arc<Self>, Fail> {
        let args = &spawn.args;
        let paths: HashMap<&str, &str> = args.windows(2).filter(|w| w[0] == "--native" || w[0] == "--pack").map(|w| (w[0].as_str(), w[1].as_str())).collect();
        let model_path = paths.get("--native").or_else(|| paths.get("--pack")).map(|s| s.to_string()).unwrap_or_else(|| "pack/full".into());
        let max_context = args.iter().position(|a| a == "--max-context").and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(4096);
        let mut info = Map::new();
        let version = Path::new(&spawn.exe)
            .parent()
            .map(|p| p.join("BUILD.json"))
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s| serde_json::from_str::<Value>(&s).ok())
            .map(|v| v.get("version").cloned().unwrap_or(Value::Null))
            .unwrap_or(Value::Null);
        info.insert("version".into(), version);
        let e = Arc::new(StrataEngine {
            session_save_reclaim: args.iter().any(|a| a == "--session-save-reclaim"),
            model_path,
            log_path: spawn.log.clone(),
            silence_s: Mutex::new(ENGINE_SILENCE_S),
            proc: Mutex::new(Proc { child: None, stdin: None, lines: None, pump: None }),
            ended: AtomicBool::new(true),
            unloaded: AtomicBool::new(true),
            starting: AtomicBool::new(false),
            max_context: AtomicI64::new(max_context),
            known_ctx: AtomicI64::new(0),
            can_stop: AtomicBool::new(false),
            info: Mutex::new(info),
            live: Mutex::new(Live::default()),
            silent_note: Mutex::new(None),
            last_err: Mutex::new(None),
            log_start: AtomicU64::new(0),
            proc_gen: AtomicU64::new(0),
            batch: AtomicUsize::new(0),
            slots: Mutex::new(Slots::default()),
            slot_cv: Condvar::new(),
            slot_rx: Mutex::new(vec![]),
            slot_tx: Arc::new(Mutex::new(vec![])),
            ctl: Sem::new(),
            ctl_state: Mutex::new(Ctl::default()),
            gpu_busy: Mutex::new(None),
            spawn,
        });
        if !lazy {
            e.start()?;
        }
        Ok(e)
    }

    pub fn batch(&self) -> usize {
        self.batch.load(SeqCst)
    }

    fn open_log(&self) -> Option<std::fs::File> {
        self.spawn.log.as_ref().and_then(|l| std::fs::OpenOptions::new().create(true).append(true).open(l).ok())
    }

    /// The process start (Python's __init__ past `lazy`): spawn, read INFO lines up to READY within ENGINE_READY_S.
    fn start(self: &Arc<Self>) -> Result<(), Fail> {
        self.unloaded.store(false, SeqCst);
        let log = self.spawn.log.clone();
        let loading = Arc::new(Event::new());
        let log_start = log.as_ref().and_then(|l| std::fs::metadata(l).ok()).map(|m| m.len()).unwrap_or(0);
        self.log_start.store(log_start, SeqCst);
        if let Some(l) = &log {
            let (l2, args, done) = (l.clone(), self.spawn.args.clone(), loading.clone());
            std::thread::spawn(move || narrate_start(l2, log_start, args, done));
            if std::env::var("STRATA_REQUEST_LINES").map(|v| !v.is_empty()).unwrap_or(false) {
                let abs = std::fs::canonicalize(l).unwrap_or_else(|_| PathBuf::from(l));
                let mut echo = ECHOING.lock().unwrap();
                if !echo.contains(&abs) {
                    echo.push(abs);
                    let l3 = l.clone();
                    std::thread::spawn(move || echo_requests(l3, log_start));
                }
            }
        }
        let mut logf = self.open_log();
        if let Some(f) = logf.as_mut() {
            let _ = writeln!(f, "[strata] {} engine started: {} --serve {}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S"), self.spawn.exe, self.spawn.args.join(" "));
            let _ = f.flush();
        }
        self.max_context.store(0, SeqCst);
        let mut cmd = crate::firewall::command(&self.spawn.exe, self.spawn.jail);
        cmd.arg("--serve").args(&self.spawn.args).stdin(Stdio::piped()).stdout(Stdio::piped());
        cmd.stderr(match logf {
            Some(f) => Stdio::from(f),
            None => Stdio::null(),
        });
        cmd.env_clear().envs(&self.spawn.env);
        if let Some(cwd) = &self.spawn.cwd {
            cmd.current_dir(cwd);
        }
        let mut child = cmd.spawn().map_err(|e| Fail::Died(format!("the engine could not be started: {e}")))?;
        let name = std::path::Path::new(&self.spawn.exe).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        crate::firewall::protect(&name, child.id() as i32, self.spawn.jail, "engine (the model)");
        let stdout = child.stdout.take().expect("stdout");
        let stdin = child.stdin.take();
        let (tx, rx) = unbounded::<Line>();
        let born = self.proc_gen.load(SeqCst) + 1;
        let me = Arc::downgrade(self);
        let slot_tx = self.slot_tx.clone();
        let pump = std::thread::Builder::new()
            .name("strata-pump".into())
            .spawn(move || pump(stdout, tx, slot_tx, me, born))
            .expect("pump");
        {
            let mut p = self.proc.lock().unwrap();
            p.child = Some(child);
            p.stdin = stdin;
            p.lines = Some(rx.clone());
            p.pump = Some(pump);
        }
        self.proc_gen.store(born, SeqCst);
        let deadline = engine_ready_s().map(|s| Instant::now() + Duration::from_secs_f64(s));
        let mut timed_out = false;
        let mut ctx = 0i64;
        loop {
            let line = match deadline {
                Some(d) => {
                    let left = d.saturating_duration_since(Instant::now());
                    if left.is_zero() {
                        timed_out = true;
                        break;
                    }
                    match rx.recv_timeout(left) {
                        Ok(l) => l,
                        Err(RecvTimeoutError::Timeout) => {
                            timed_out = true;
                            break;
                        }
                        Err(_) => None,
                    }
                }
                None => rx.recv().unwrap_or(None),
            };
            let Some(line) = line else { break };
            if let Some(rest) = line.strip_prefix("INFO ") {
                let mut info = self.info.lock().unwrap();
                for kv in py::split_ws(rest) {
                    let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
                    info.insert(k.into(), parse_info_value(v));
                }
            }
            if line.starts_with("READY") {
                let f = py::split_ws(&line);
                ctx = f.get(1).and_then(|v| v.parse().ok()).unwrap_or(0);
                self.can_stop.store(f[2.min(f.len())..].contains(&"stop"), SeqCst);
                break;
            }
        }
        loading.set();
        if ctx <= 0 {
            let mut p = self.proc.lock().unwrap();
            if let Some(c) = p.child.as_mut() {
                if timed_out {
                    let _ = c.kill();
                }
                wait_timeout(c, 5.0);
            }
            p.stdin = None;
            let why = if timed_out {
                format!("the engine did not report READY within {:.0} s", engine_ready_s().unwrap_or(0.0))
            } else {
                "the engine exited before it was ready".to_string()
            };
            let l = log.as_deref();
            return Err(Fail::Died(format!(
                "{why}{}{}{}",
                l.map(|l| format!(" (see {l})")).unwrap_or_default(),
                start_failure_hint(l, log_start),
                start_log_tail(l, log_start, 20)
            )));
        }
        self.max_context.store(ctx, SeqCst);
        self.known_ctx.store(ctx, SeqCst);
        {
            let mut info = self.info.lock().unwrap();
            if let Some(e) = info.get("engine").filter(|v| py::truthy(v)).cloned() {
                info.insert("version".into(), Value::String(py::py_str(&e)));
            }
        }
        self.ended.store(false, SeqCst);
        // --batch: the slots the engine runs (INFO batch_slots), spread over its pipeline groups
        let args = &self.spawn.args;
        let asked = ["--batch", "--slots"].iter().find_map(|k| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).and_then(|v| v.parse::<usize>().ok())).unwrap_or(0);
        let info = self.info.lock().unwrap().clone();
        let num = |k: &str| info.get(k).and_then(Value::as_i64).unwrap_or(0).max(0) as usize;
        let batch = num("batch_slots");
        if asked > 0 && batch != asked {
            say(&format!("parallel requests: {asked} asked, the engine runs {} (its log says why)", if batch > 0 { batch.to_string() } else { "one at a time".into() }));
        }
        let bg = args.iter().position(|a| a == "--batch-groups").and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| "default".into());
        let mut groups = if bg == "auto" || bg == "default" { num("batch_groups").max(1) } else { bg.parse().unwrap_or(1) };
        if !(batch > 0 && groups > 0 && batch % groups == 0) {
            groups = 1;
        }
        let gs = if batch > 0 { batch / groups } else { 0 };
        let groups_run = if num("batch_groups") > 0 { num("batch_groups") } else { groups };
        {
            let mut s = self.slots.lock().unwrap();
            s.order = (0..gs).flat_map(|t| (0..groups).map(move |g| g * gs + t)).collect();
            s.group = (0..batch).map(|b| if gs > 0 { b / gs } else { 0 }).collect();
            s.groups = groups;
            s.gs = if groups > 1 && groups_run > 1 { gs } else { 0 };
            s.busy = vec![false; batch];
            s.claims.clear();
            s.releasing.clear();
            s.orphan_seen.clear();
            s.held = vec![vec![]; batch];
            s.used = vec![0.0; batch];
            s.live = vec![None; batch];
        }
        let (txs, rxs): (Vec<_>, Vec<_>) = (0..batch).map(|_| unbounded::<Line>()).unzip();
        *self.slot_tx.lock().unwrap() = txs;
        *self.slot_rx.lock().unwrap() = rxs;
        self.batch.store(batch, SeqCst);
        *self.ctl_state.lock().unwrap() = Ctl::default();
        Ok(())
    }

    fn lines(&self) -> Option<Receiver<Line>> {
        self.proc.lock().unwrap().lines.clone()
    }

    pub fn pid(&self) -> Option<u32> {
        self.proc.lock().unwrap().child.as_ref().map(|c| c.id())
    }

    pub fn alive(&self) -> bool {
        let mut p = self.proc.lock().unwrap();
        match p.child.as_mut() {
            None => false,
            Some(c) => !self.ended.load(SeqCst) && matches!(c.try_wait(), Ok(None)),
        }
    }

    /// The process's exit code, waiting up to 5 s for it (None: still running, or no process).
    pub fn exit_code(&self) -> Option<i32> {
        let mut p = self.proc.lock().unwrap();
        let c = p.child.as_mut()?;
        let t = Instant::now();
        loop {
            match c.try_wait() {
                Ok(Some(s)) => return Some(exit_code_of(s)),
                Ok(None) if since(t) < 5.0 => std::thread::sleep(Duration::from_millis(20)),
                _ => return None,
            }
        }
    }

    fn poll(&self) -> Option<i32> {
        let mut p = self.proc.lock().unwrap();
        p.child.as_mut().and_then(|c| c.try_wait().ok().flatten()).map(exit_code_of)
    }

    fn died(&self) -> Fail {
        Fail::Died(format!("the engine stopped unexpectedly (exit code {})", self.exit_code().map(|c| c.to_string()).unwrap_or_else(|| "None".into())))
    }

    /// Why the engine most likely ended, from the end of its log: its own watchdog (issue #29), else RAM.
    pub fn death_note(&self) -> String {
        if let Some(n) = self.silent_note.lock().unwrap().clone() {
            return n;
        }
        let wide = self.log_path.as_deref().and_then(|l| {
            let size = std::fs::metadata(l).ok()?.len();
            read_from(l, size.saturating_sub(262144), None)
        });
        let wide = wide.unwrap_or_default();
        let tail = &wide[wide.len().saturating_sub(4096)..];
        let tail = wide.get(wide.len() - tail.len()..).unwrap_or("");
        for line in wide.lines().rev() {
            if line.contains("issue #29") {
                return format!("The engine stopped itself because it had stopped making progress - a hang it caught. Its log line: {} - please report it at github.com/Niko1221/Strata/issues.", py::strip(line));
            }
        }
        let rc = self.poll();
        if let (Some(rc), Some(err)) = (rc, self.last_err.lock().unwrap().clone()) {
            if rc >= 0 {
                return format!("The engine exited (code {rc}) after it reported: {err} - please report it at github.com/Niko1221/Strata/issues with the log.");
            }
        }
        let last = tail.lines().rev().map(py::strip).find(|x| x.starts_with("strata") || x.starts_with("ERR")).unwrap_or("");
        if let Some(rc) = rc.filter(|rc| *rc >= 0) {
            if !last.is_empty() {
                return format!("The engine exited (code {rc}). Its last log line: {last} - if that does not explain it, please report it at github.com/Niko1221/Strata/issues with the log.");
            }
        }
        "The usual cause is running out of RAM: Linux then ends the biggest program (check: sudo dmesg | grep -i -E 'killed process|out of memory'); Windows slows down instead. Close other programs or use a smaller model (Q2_0 / IQ2_XS).".to_string()
    }

    /// Release GPU/RAM between requests, keeping the spawn config for the next load.
    pub fn unload(&self) -> Result<(), Fail> {
        self.close()?;
        self.unloaded.store(true, SeqCst);
        Ok(())
    }

    /// Start the engine again (the same command) after it died; retried while a start exits before READY.
    pub fn restart(self: &Arc<Self>, tries: usize) -> Result<(), Fail> {
        let info = self.info.lock().unwrap().clone();
        self.starting.store(true, SeqCst);
        let r = (|| {
            self.close()?;
            for i in 0..tries {
                self.ended.store(true, SeqCst);
                match self.start() {
                    Ok(()) => return Ok(()),
                    Err(e) => {
                        if let Some(c) = self.proc.lock().unwrap().child.as_mut() {
                            wait_timeout(c, 60.0);
                        }
                        if i == tries - 1 {
                            return Err(e);
                        }
                        say(&format!("the engine did not start (try {} of {tries}); again in {} s", i + 1, py_g(RESTART_RETRY_S)));
                        std::thread::sleep(Duration::from_secs_f64(RESTART_RETRY_S));
                    }
                }
            }
            Ok(())
        })();
        self.starting.store(false, SeqCst);
        {
            let _g = self.slots.lock().unwrap();
            self.slot_cv.notify_all();
        }
        let mut cur = self.info.lock().unwrap();
        let mut merged = info;
        for (k, v) in cur.iter() {
            merged.insert(k.clone(), v.clone());
        }
        *cur = merged;
        r
    }

    fn parse_done(line: &str) -> Map<String, Value> {
        let f = py::split_ws(line);
        let i = |k: usize| f.get(k).and_then(|v| v.parse::<i64>().ok()).map(Value::from).unwrap_or(Value::Null);
        let fl = |k: usize| f.get(k).and_then(|v| v.parse::<f64>().ok()).map(Value::from).unwrap_or(Value::Null);
        let mut m = Map::new();
        m.insert("generated".into(), i(1));
        m.insert("prompt_tokens".into(), i(2));
        m.insert("prompt_ms".into(), fl(3));
        m.insert("decode_ms".into(), fl(4));
        m.insert("finish".into(), f.get(5).map(|s| Value::from(*s)).unwrap_or(Value::Null));
        if f.len() >= 9 {
            m.insert("drafts_accepted".into(), i(6));
            m.insert("drafts_offered".into(), i(7));
            m.insert("reused".into(), i(8));
        }
        if f.len() >= 11 {
            m.insert("hits".into(), i(9));
            m.insert("lookups".into(), i(10));
        }
        if f.len() >= 14 {
            m.insert("ram_blobs".into(), i(11));
            m.insert("file_blobs".into(), i(12));
            m.insert("file_mb".into(), fl(13));
        }
        if f.len() >= 15 {
            m.insert("prompt_read".into(), i(14));
        }
        if f.len() >= 16 {
            m.insert("offloaded".into(), i(15));
        }
        m
    }

    fn send(&self, text: &str) -> Result<(), Fail> {
        btrace(&format!("send> {}", py::chars_prefix(text, 60)));
        let ok = {
            let mut p = self.proc.lock().unwrap();
            match p.stdin.as_mut() {
                Some(s) => s.write_all(format!("{text}\n").as_bytes()).and_then(|_| s.flush()).is_ok(),
                None => false,
            }
        };
        if ok { Ok(()) } else { Err(self.died()) }
    }

    /// #533: `VRAM <reserve_mib>` between requests; Value(reason) when the engine refused it.
    pub fn vram(&self, reserve_mib: Option<i64>, timeout: f64) -> Result<Map<String, Value>, Fail> {
        if !self.alive() {
            return Err(Fail::Died("the engine is not running".into()));
        }
        self.send(&format!("VRAM{}", reserve_mib.map(|r| format!(" {r}")).unwrap_or_default()))?;
        let lines = self.lines().ok_or_else(|| Fail::Died("the engine ended".into()))?;
        let deadline = Instant::now() + Duration::from_secs_f64(timeout);
        loop {
            let left = deadline.saturating_duration_since(Instant::now()).max(Duration::from_millis(100));
            let line = match lines.recv_timeout(left) {
                Ok(l) => l,
                Err(_) => return Err(Fail::Died("the engine did not answer the VRAM command".into())),
            };
            let Some(line) = line else { return Err(Fail::Died("the engine ended".into())) };
            let line = py::strip(&line).to_string();
            if line.starts_with("ERR") {
                let why = py::strip(line.get(4..).unwrap_or("")).to_string();
                return Err(Fail::Value(if why.is_empty() { "the engine refused the VRAM command".into() } else { why }));
            }
            if let Some(rest) = line.strip_prefix("VRAM ") {
                let mut out = Map::new();
                for kv in py::split_ws(rest) {
                    let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
                    out.insert(k.into(), parse_info_value(v));
                }
                let mut info = self.info.lock().unwrap();
                for k in ["expert_slots", "vram_free_mib"] {
                    if let Some(v) = out.get(k) {
                        info.insert(k.into(), v.clone());
                    }
                }
                info.insert("vram".into(), Value::Object(out.clone()));
                return Ok(out);
            }
            if Instant::now() > deadline {
                return Err(Fail::Died("the engine did not answer the VRAM command".into()));
            }
        }
    }

    /// #481: end an engine that lost step with the server, so the next request starts it again.
    fn silent(&self, what: &str) -> Fail {
        *self.silent_note.lock().unwrap() = Some("The engine and the server lost step (issue #481; a very slow PC can raise \"engine_silence_s\" in the config, 0 = wait forever). If it happens again, please add the end of the engine log to github.com/Niko1221/Strata/issues/481.".into());
        self.ended.store(true, SeqCst);
        if let Some(c) = self.proc.lock().unwrap().child.as_mut() {
            let _ = c.kill();
            wait_timeout(c, 20.0);
        }
        Fail::Silent(format!("{what}; the server ended the engine"))
    }

    /// #1317: while the engine is silent, end it when it also used no CPU and no I/O for stall_s (and the GPU is idle).
    fn stall_check(&self, heard: Instant, base: &mut Option<Option<(f64, u64)>>, next: &mut f64, warned: &mut bool, stall_s: f64) -> Result<(), Fail> {
        let age = since(heard);
        let activity = || self.pid().and_then(process_activity);
        let Some(b) = base else {
            *base = Some(activity());
            return Ok(());
        };
        if age < *next || b.is_none() {
            return Ok(());
        }
        let b = b.expect("base");
        let now = activity();
        let gpu_working = self.gpu_busy.lock().unwrap().as_ref().map(|f| f()).unwrap_or(false);
        if let Some(n) = now {
            if !gpu_working && engine_frozen(b, n) {
                return Err(self.silent(&format!("the engine said nothing for {age:.0} s and used no CPU or disk in that time (frozen; STRATA_ENGINE_STALL_S sets this, 0 = off)")));
            }
            *base = Some(Some(n));
            *next = age + stall_s;
            if !*warned {
                *warned = true;
                say(&format!("the engine has said nothing for {age:.0} s but is still working; it is not ended (the limit is engine_silence_s). If it is stuck, \"engine_silence_s\" can be set lower."));
            }
        } else {
            *next = f64::INFINITY;
        }
        Ok(())
    }

    /// Yields token ids, and None as a heartbeat every 10 s while the engine is quiet.  A consumer that closes it early
    /// (or `cancel`) makes the engine STOP and reads it to its DONE.
    pub fn generate(self: &Arc<Self>, ids: Vec<u32>, max_new: i64, sampling: Map<String, Value>, cancel: Arc<Event>, embeddings: Option<String>) -> Generation {
        let stats: Stats = Arc::new(Mutex::new(None));
        let st = stats.clone();
        let me = self.clone();
        let tokens = Gen::new(move |y: &Yielder<Option<u32>, Fail>| {
            if !me.alive() {
                return Err(Stop::Err(Fail::Died("the engine is unavailable; this request was not sent".into())));
            }
            let info_bias = me.info.lock().unwrap().get("logit_bias").and_then(Value::as_i64);
            crate::bias::validate_request(&sampling, info_bias, me.batch(), None).map_err(Stop::Err)?;
            if me.batch() > 0 {
                return me.generate_batched(y, ids, max_new, sampling, cancel, embeddings, st);
            }
            me.generate_solo(y, ids, max_new, sampling, cancel, embeddings, st)
        });
        Generation { tokens, stats }
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_solo(&self, y: &Yielder<Option<u32>, Fail>, ids: Vec<u32>, max_new: i64, sampling: Map<String, Value>, cancel: Arc<Event>, embeddings: Option<String>, stats: Stats) -> Result<(), Stop<Fail>> {
        *self.live.lock().unwrap() = Live::default();
        let keys = sampling_keys(&sampling);
        let head = match &embeddings {
            Some(e) => format!("GENI {max_new}{keys} {e}"),
            None => format!("GEN {max_new}{keys}"),
        };
        let ids_s: Vec<String> = ids.iter().map(|t| t.to_string()).collect();
        let sent = {
            let mut p = self.proc.lock().unwrap();
            match p.stdin.as_mut() {
                Some(s) => s.write_all(format!("{head} {}\n", ids_s.join(",")).as_bytes()).and_then(|_| s.flush()).is_ok(),
                None => false,
            }
        };
        if !sent {
            return Err(Stop::Err(self.died()));
        }
        let Some(lines) = self.lines() else { return Err(Stop::Err(self.died())) };
        let silence = *self.silence_s.lock().unwrap();
        let mut allow = if silence > 0.0 { silence + ids.len().min(PP_CHUNK_MAX) as f64 / PP_FLOOR_TOK_S } else { 0.0 };
        let mut done = false;
        let mut heard = Instant::now();
        let mut read_to: i64 = 0;
        let mut beat = heard;
        let stall_s = engine_stall_s().unwrap_or(0.0);
        let (mut base, mut next, mut warned): (Option<Option<(f64, u64)>>, f64, bool) = (None, stall_s, false);
        let mut body = || -> Result<(), Stop<Fail>> {
            loop {
                let mut wait = 10.0f64;
                if allow > 0.0 {
                    wait = wait.min(allow - since(heard));
                    if wait <= 0.0 {
                        done = true;
                        return Err(Stop::Err(self.silent(&format!("the engine said nothing for {:.0} s during the request", since(heard)))));
                    }
                }
                let line = match lines.recv_timeout(Duration::from_secs_f64(wait.min(0.5))) {
                    Ok(l) => l,
                    Err(RecvTimeoutError::Disconnected) => None,
                    Err(RecvTimeoutError::Timeout) => {
                        if cancel.is_set() {
                            return Ok(());
                        }
                        if stall_s > 0.0 {
                            if let Err(e) = self.stall_check(heard, &mut base, &mut next, &mut warned, stall_s) {
                                done = true;
                                return Err(Stop::Err(e));
                            }
                        }
                        if allow > 0.0 && since(heard) >= allow {
                            continue;
                        }
                        if since(beat) >= 10.0 {
                            beat = Instant::now();
                            y.yield_(None)?;
                        }
                        continue;
                    }
                };
                let Some(line) = line else {
                    done = true;
                    return Err(Stop::Err(self.died()));
                };
                heard = Instant::now();
                beat = heard;
                base = None;
                next = stall_s;
                warned = false;
                if let Some(t) = line.strip_prefix("T ") {
                    allow = silence;
                    if cancel.is_set() {
                        return Ok(());
                    }
                    if let Ok(t) = py::strip(t).parse::<u32>() {
                        y.yield_(Some(t))?;
                    }
                } else if line.starts_with("PP ") {
                    let f = py::split_ws(&line);
                    if f.len() >= 3 && is_digits(f[1]) && is_digits(f[2]) {
                        let (a, b): (i64, i64) = (f[1].parse().unwrap_or(0), f[2].parse().unwrap_or(0));
                        let mut live = self.live.lock().unwrap();
                        live.progress = Some((a, b));
                        live.progress_ms = f.get(3).and_then(|v| v.parse().ok()).unwrap_or(0.0);
                        live.prefill_tok_s_mean = f.get(4).and_then(|v| v.parse().ok());
                        let rate = live.prefill_tok_s_mean.unwrap_or(0.0);
                        let chunk = a - read_to;
                        read_to = a;
                        if silence > 0.0 && rate > 0.0 && chunk > 0 {
                            allow = silence.max(PP_SLACK * chunk as f64 / rate);
                        }
                    }
                    if cancel.is_set() {
                        return Ok(());
                    }
                    y.yield_(None)?;
                } else if let Some(rest) = line.strip_prefix("RESUME ") {
                    if let Some(n) = py::split_ws(rest).first().and_then(|v| v.parse::<i64>().ok()) {
                        read_to = n;
                        self.live.lock().unwrap().reused = n;
                    }
                } else if line.starts_with("DONE") {
                    *stats.lock().unwrap() = Some(Self::parse_done(&line));
                    done = true;
                    return Ok(());
                } else if line.starts_with("ERR") {
                    done = true;
                    return Err(Stop::Err(Fail::Value(py::strip(line.get(4..).unwrap_or("")).to_string())));
                }
            }
        };
        let r = body();
        if !done {
            // the consumer stopped early: stop the engine, drain to DONE (never an untimed wait: it holds the FIFO)
            if self.can_stop.load(SeqCst) {
                let mut p = self.proc.lock().unwrap();
                if let Some(s) = p.stdin.as_mut() {
                    let _ = s.write_all(b"STOP\n").and_then(|_| s.flush());
                }
            }
            let mut heard = Instant::now();
            loop {
                let line = if allow > 0.0 {
                    let left = allow - since(heard);
                    if left <= 0.0 {
                        return Err(Stop::Err(self.silent(&format!("the engine did not finish the request after it was stopped (STOP) within {allow:.0} s"))));
                    }
                    match lines.recv_timeout(Duration::from_secs_f64(left)) {
                        Ok(l) => l,
                        Err(RecvTimeoutError::Timeout) => {
                            return Err(Stop::Err(self.silent(&format!("the engine did not finish the request after it was stopped (STOP) within {allow:.0} s"))));
                        }
                        Err(_) => None,
                    }
                } else {
                    lines.recv().unwrap_or(None)
                };
                match line {
                    None => break,
                    Some(l) if l.starts_with("ERR") => break,
                    Some(l) if l.starts_with("DONE") => {
                        *stats.lock().unwrap() = Some(Self::parse_done(&l));
                        break;
                    }
                    _ => {
                        if !self.can_stop.load(SeqCst) {
                            heard = Instant::now();
                        }
                    }
                }
            }
        }
        r
    }

    // -------------------------------------------------------------------------------------------- batch slots
    /// After a consumer left early: read the control lines up to the next `until` line (DONE or BADM).
    fn drain_control(&self, until: &str, timeout: f64, born: u64, stats: &Stats) -> Option<String> {
        let lines = self.lines()?;
        let end = Instant::now() + Duration::from_secs_f64(timeout);
        while Instant::now() < end {
            if self.proc_gen.load(SeqCst) != born || !self.alive() {
                return None;
            }
            let wait = end.saturating_duration_since(Instant::now()).min(Duration::from_secs(1)).max(Duration::from_millis(100));
            let line = match lines.recv_timeout(wait) {
                Ok(l) => l,
                Err(RecvTimeoutError::Timeout) => continue,
                Err(_) => None,
            };
            let line = line?;
            if line.starts_with("DONE") {
                *stats.lock().unwrap() = Some(Self::parse_done(&line));
            }
            if line.starts_with(until) || line.starts_with("ERR") {
                return Some(line);
            }
        }
        None
    }

    /// A slot whose consumer left: BSTOP it and free it once the engine says BDONE (in the background).
    fn release_slot_when_done(self: &Arc<Self>, slot: usize, stream: Option<Vec<u32>>) {
        let Some(q) = self.slot_rx.lock().unwrap().get(slot).cloned() else { return };
        let _ = self.send(&format!("BSTOP {slot}"));
        self.slots.lock().unwrap().releasing.insert(slot);
        let me = self.clone();
        let born = self.proc_gen.load(SeqCst);
        std::thread::spawn(move || {
            let end = Instant::now() + Duration::from_secs(600);
            let mut tail = stream.clone().unwrap_or_default();
            while Instant::now() < end {
                match q.recv_timeout(Duration::from_secs(5)) {
                    Err(RecvTimeoutError::Timeout) => continue,
                    Err(_) | Ok(None) => {
                        tail.clear();
                        break;
                    }
                    Ok(Some(line)) => {
                        if line.starts_with("BT ") {
                            match py::split_ws(&line).get(2).and_then(|v| v.parse::<u32>().ok()) {
                                Some(t) => tail.push(t),
                                None => tail.clear(),
                            }
                        }
                        if line.starts_with("BDONE ") {
                            break;
                        }
                    }
                }
            }
            let mut s = me.slots.lock().unwrap();
            if me.proc_gen.load(SeqCst) == born && slot < s.busy.len() {
                s.held[slot] = if stream.is_some() && !tail.is_empty() { tail[..tail.len() - 1].to_vec() } else { vec![] };
                s.busy[slot] = false;
            }
            s.releasing.remove(&slot);
            me.slot_cv.notify_all();
        });
    }

    /// #1603: free the slots whose busy flag has no owner (the caller holds the slot lock).
    fn reap_orphan_slots(&self, s: &mut Slots) -> Vec<usize> {
        let mut freed = vec![];
        for b in 0..s.busy.len() {
            if s.busy[b] && !s.claims.contains(&b) && !s.releasing.contains(&b) {
                let first = *s.orphan_seen.entry(b).or_insert_with(Instant::now);
                if since(first) >= ORPHAN_SLOT_S {
                    s.busy[b] = false;
                    s.orphan_seen.remove(&b);
                    freed.push(b);
                }
            } else {
                s.orphan_seen.remove(&b);
            }
        }
        if !freed.is_empty() {
            say(&format!("parallel: slot(s) {} were marked busy with no request using them (issue #1603); freed, the waiting request goes on", freed.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(", ")));
            self.slot_cv.notify_all();
        }
        freed
    }

    /// Waits for the control lines (one prompt read at a time), yielding heartbeats; false when cancelled.
    fn take_control(&self, y: &Yielder<Option<u32>, Fail>, cancel: &Event, plen: usize, after_epoch: Option<u64>) -> Result<bool, Stop<Fail>> {
        let wid = {
            let mut s = self.slots.lock().unwrap();
            s.waiting += 1;
            s.next_wait_id += 1;
            let id = s.next_wait_id;
            s.wait_lens.push((id, plen));
            id
        };
        let mut beat = Instant::now();
        let r = (|| -> Result<bool, Stop<Fail>> {
            loop {
                if !self.alive() && !self.starting.load(SeqCst) {
                    return Err(Stop::Err(Fail::Died("the engine stopped while this request waited; it was not sent".into())));
                }
                let mut turn = {
                    let s = self.slots.lock().unwrap();
                    after_epoch.is_none() || s.ctl_epoch > after_epoch.unwrap() || s.waiting <= 1
                };
                if turn && self.ctl.acquire_timeout(Duration::from_millis(500)) {
                    if self.alive() {
                        return Ok(true);
                    }
                    self.ctl.release();
                    turn = false;
                }
                if !turn {
                    let s = self.slots.lock().unwrap();
                    let _ = self.slot_cv.wait_timeout(s, Duration::from_millis(500)).unwrap();
                }
                if cancel.is_set() {
                    return Ok(false);
                }
                if since(beat) >= 10.0 {
                    beat = Instant::now();
                    y.yield_(None)?;
                }
            }
        })();
        {
            let mut s = self.slots.lock().unwrap();
            s.waiting = s.waiting.saturating_sub(1);
            s.wait_lens.retain(|(id, _)| *id != wid);
        }
        if matches!(r, Ok(true)) {
            let mut s = self.slots.lock().unwrap();
            s.ctl_epoch += 1;
            self.slot_cv.notify_all();
        }
        r
    }

    fn may_go_solo(&self, left: i64, times: usize, embeddings: bool) -> bool {
        let slot_cache = self.info.lock().unwrap().get("slot_cache").map(py::truthy).unwrap_or(false);
        if embeddings || times >= SOLO_AGAIN_MAX || left < SOLO_AGAIN_MIN_LEFT || !slot_cache || std::env::var("STRATA_PARALLEL_SOLO").ok().as_deref() == Some("0") {
            return false;
        }
        let s = self.slots.lock().unwrap();
        s.busy.iter().filter(|b| **b).count() == 1 && s.waiting == 0
    }

    fn shorter_waiting(&self, plen: usize) -> bool {
        self.slots.lock().unwrap().wait_lens.iter().any(|(_, n)| n * 2 <= plen)
    }

    /// A free slot for `prompt` (the caller holds the slot lock), None when every slot is busy.
    fn pick_slot(s: &Slots, prompt: &[u32]) -> Option<usize> {
        let free: Vec<usize> = s.order.iter().copied().filter(|b| !s.busy[*b]).collect();
        if free.is_empty() {
            return None;
        }
        if s.gs > 0 {
            let gs = s.gs;
            let fills_hole = |b: usize| {
                let g0 = b - b % gs;
                (b + 1..g0 + gs).any(|x| x < s.busy.len() && s.busy[x])
            };
            return free.iter().copied().min_by_key(|b| (!fills_hole(*b), s.order.iter().position(|x| x == b).unwrap_or(0)));
        }
        let held_prefix = |b: usize| {
            let h = &s.held[b];
            if !h.is_empty() && h.len() < prompt.len() && prompt[..h.len()] == h[..] { h.len() } else { 0 }
        };
        // max() keeps the last of equals in Rust, the first in Python: pick the first maximum by hand
        let mut best = free[0];
        for &b in &free {
            if held_prefix(b) > held_prefix(best) {
                best = b;
            }
        }
        if held_prefix(best) >= if s.groups > 1 { SLOT_PREFIX_MIN } else { 1 } {
            return Some(best);
        }
        let mut busy_in: HashMap<usize, usize> = HashMap::new();
        for b in 0..s.busy.len() {
            if s.busy[b] {
                *busy_in.entry(s.group[b]).or_insert(0) += 1;
            }
        }
        let key = |b: usize| -> (usize, bool, f64) { (if s.groups > 1 { busy_in.get(&s.group[b]).copied().unwrap_or(0) } else { 0 }, !s.held[b].is_empty(), s.used[b]) };
        let mut pick = free[0];
        for &b in &free {
            let (a, k) = (key(b), key(pick));
            if a.0 < k.0 || (a.0 == k.0 && (a.1 < k.1 || (a.1 == k.1 && a.2 < k.2))) {
                pick = b;
            }
        }
        Some(pick)
    }

    /// /metrics: each batch slot - idle (with the tokens it holds for a next turn), reading or decoding.
    pub fn slots_view(&self) -> Vec<Value> {
        let now = crate::sync::now();
        let s = self.slots.lock().unwrap();
        (0..s.busy.len())
            .map(|b| match &s.live[b] {
                None => json!({"slot": b, "state": "idle", "held_tokens": s.held[b].len()}),
                Some(r) => {
                    let ft = r.get("first_token").and_then(Value::as_f64);
                    let generated = r.get("generated").and_then(Value::as_f64).unwrap_or(0.0);
                    let started = r.get("started").and_then(Value::as_f64).unwrap_or(now);
                    json!({"slot": b, "state": r.get("state").cloned().unwrap_or(Value::Null), "prompt_tokens": r.get("prompt_tokens").cloned().unwrap_or(Value::Null),
                           "generated": r.get("generated").cloned().unwrap_or(Value::Null), "elapsed_s": round1(now - started),
                           "tok_s": ft.map(|ft| round1(generated / (now - ft).max(1e-6)))})
                }
            })
            .collect()
    }

    pub fn waiting(&self) -> usize {
        self.slots.lock().unwrap().waiting
    }

    pub fn admitting(&self) -> usize {
        self.slots.lock().unwrap().admitting
    }

    fn set_live(&self, slot: usize, f: impl FnOnce(&mut Map<String, Value>)) {
        let mut s = self.slots.lock().unwrap();
        if let Some(Some(l)) = s.live.get_mut(slot) {
            f(l);
        }
    }

    fn free_slot(&self, slot: usize) {
        let mut s = self.slots.lock().unwrap();
        if slot < s.busy.len() {
            s.busy[slot] = false;
        }
        s.claims.remove(&slot);
        self.slot_cv.notify_all();
    }

    /// --batch: alone, the solo path (GEN with drafts); otherwise BGEN into a free slot and its BT lines to BDONE.
    #[allow(clippy::too_many_arguments)]
    fn generate_batched(self: &Arc<Self>, y: &Yielder<Option<u32>, Fail>, ids: Vec<u32>, max_new: i64, sampling: Map<String, Value>, cancel: Arc<Event>, embeddings: Option<String>, stats: Stats) -> Result<(), Stop<Fail>> {
        *self.live.lock().unwrap() = Live::default();
        let keys = sampling_keys(&sampling);
        let mut out: Vec<u32> = vec![];
        let mut prompt: Vec<u32> = ids.clone();
        let mut left = max_new;
        let mut reused0: Option<Value> = None;
        if !self.take_control(y, &cancel, prompt.len(), None)? {
            return Ok(());
        }
        let mut holding = true;
        let mut born = self.proc_gen.load(SeqCst);
        btrace("ctl acquired");
        let mut slot: Option<usize> = None;
        let mut gen0: Option<usize> = None;
        let mut reserved: Option<usize> = None;
        let mut phase = "none";
        let mut stop_sent = false;
        let (mut yields, mut solo_again) = (0usize, 0usize);
        let (mut admit_wait, mut admit_since, mut admit_said) = (false, Instant::now(), false);
        let emb = embeddings.is_some();
        let ids_line = |p: &[u32]| p.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(",");

        let mut body = || -> Result<(), Stop<Fail>> {
            loop {
                if !holding {
                    if !self.take_control(y, &cancel, prompt.len(), None)? {
                        return Ok(());
                    }
                    holding = true;
                    born = self.proc_gen.load(SeqCst);
                }
                let alone = {
                    let s = self.slots.lock().unwrap();
                    !s.busy.iter().any(|b| *b) && s.waiting == 0
                };
                let mut cont;
                if alone && left > 1 {
                    let head = match &embeddings {
                        Some(e) => format!("GENI {left}{keys} {e}"),
                        None => format!("GEN {left}{keys}"),
                    };
                    self.send(&format!("{head} {}", ids_line(&prompt))).map_err(Stop::Err)?;
                    phase = "solo";
                    {
                        let mut c = self.ctl_state.lock().unwrap();
                        c.mode = "solo";
                        c.result = None;
                        c.yielded = None;
                        c.erred = false;
                    }
                    let me = self.clone();
                    let others = move || me.slots.lock().unwrap().waiting > 0;
                    // the control lines: tokens go out as they come; at a heartbeat a shorter waiting request may get
                    // the read's slot (BYIELD)
                    let ctl_r = self.control(y, &cancel, &mut out, Some(&others), &stats, &mut |me2: &Self| {
                        if reserved.is_none() && !emb && yields < YIELDS_MAX && me2.shorter_waiting(prompt.len()) {
                            let mut s = me2.slots.lock().unwrap();
                            if let Some(r) = Self::pick_slot(&s, &prompt) {
                                s.busy[r] = true;
                                s.claims.insert(r);
                                reserved = Some(r);
                            }
                            drop(s);
                            if let Some(r) = reserved {
                                let _ = me2.send(&format!("BYIELD {r}"));
                            }
                        }
                    });
                    ctl_r?;
                    phase = "none";
                    let yielded = self.ctl_state.lock().unwrap().yielded;
                    if let (Some((ys, _)), Some(r)) = (yielded, reserved) {
                        if ys == r {
                            slot = Some(r);
                            reserved = None;
                        }
                    }
                    if let Some(r) = reserved.take() {
                        self.free_slot(r);
                    }
                    if slot.is_none() {
                        let finish = stats.lock().unwrap().as_ref().and_then(|m| m.get("finish").and_then(Value::as_str).map(String::from));
                        left = max_new - out.len() as i64;
                        if cancel.is_set() || left <= 0 || matches!(finish.as_deref(), Some("stop" | "length")) || out.last().map(|t| EOS_IDS.contains(t)).unwrap_or(false) {
                            return Ok(());
                        }
                        if reused0.is_none() {
                            reused0 = stats.lock().unwrap().as_ref().and_then(|m| m.get("reused").cloned());
                        }
                        prompt = ids.iter().copied().chain(out.iter().copied()).collect();
                    }
                }
                loop {
                    if slot.is_none() {
                        loop {
                            {
                                let mut s = self.slots.lock().unwrap();
                                if let Some(b) = Self::pick_slot(&s, &prompt) {
                                    s.busy[b] = true;
                                    s.claims.insert(b);
                                    if admit_wait {
                                        admit_wait = false;
                                        s.admitting = s.admitting.saturating_sub(1);
                                    }
                                    slot = Some(b);
                                    break;
                                }
                                if !admit_wait {
                                    admit_wait = true;
                                    admit_since = Instant::now();
                                    s.admitting += 1;
                                }
                            }
                            if holding {
                                self.ctl.release();
                                holding = false;
                            }
                            {
                                let mut s = self.slots.lock().unwrap();
                                if Self::pick_slot(&s, &prompt).is_none() {
                                    let stall = engine_stall_s().unwrap_or(0.0);
                                    if stall > 0.0 && since(admit_since) >= stall && !admit_said {
                                        admit_said = true;
                                        let busy: Vec<usize> = (0..s.busy.len()).filter(|b| s.busy[*b]).collect();
                                        let mut claims: Vec<_> = s.claims.iter().copied().collect();
                                        claims.sort();
                                        let mut rel: Vec<_> = s.releasing.iter().copied().collect();
                                        rel.sort();
                                        say(&format!("parallel: a request has waited {:.0} s for a free slot (busy: {busy:?}, in use: {claims:?}, ending: {rel:?}) (issue #1603)", since(admit_since)));
                                    }
                                    self.reap_orphan_slots(&mut s);
                                }
                                if Self::pick_slot(&s, &prompt).is_none() {
                                    let _ = self.slot_cv.wait_timeout(s, Duration::from_secs(1)).unwrap();
                                }
                            }
                            if cancel.is_set() {
                                return Ok(());
                            }
                            if !holding {
                                if !self.take_control(y, &cancel, prompt.len(), None)? {
                                    return Ok(());
                                }
                                holding = true;
                                born = self.proc_gen.load(SeqCst);
                            }
                        }
                    }
                    let b = slot.expect("slot");
                    let yielded = self.ctl_state.lock().unwrap().yielded.take();
                    if let Some((_, read)) = yielded {
                        self.slots.lock().unwrap().held[b] = prompt[..read.min(prompt.len())].to_vec();
                        yields += 1;
                        self.ctl.release();
                        holding = false;
                        let epoch = self.slots.lock().unwrap().ctl_epoch;
                        if !self.take_control(y, &cancel, prompt.len(), Some(epoch))? {
                            return Ok(());
                        }
                        holding = true;
                        born = self.proc_gen.load(SeqCst);
                    }
                    if let Some(q) = self.slot_rx.lock().unwrap().get(b) {
                        while q.try_recv().is_ok() {}
                    }
                    let head = match &embeddings {
                        Some(e) => format!("BGENI {b} {left}{keys} {e}"),
                        None => format!("BGEN {b} {left}{keys}"),
                    };
                    {
                        let mut s = self.slots.lock().unwrap();
                        s.live[b] = Some(json!({"slot": b, "state": "reading", "prompt_tokens": prompt.len(), "generated": 0, "started": crate::sync::now(), "first_token": null}).as_object().cloned().unwrap());
                    }
                    self.send(&format!("{head} {}", ids_line(&prompt))).map_err(Stop::Err)?;
                    self.slots.lock().unwrap().held[b] = vec![];
                    phase = "admit";
                    {
                        let mut c = self.ctl_state.lock().unwrap();
                        c.mode = "batch";
                        c.result = None;
                        c.yielded = None;
                        c.erred = false;
                    }
                    let mut asked = false;
                    self.control(y, &cancel, &mut out, None, &stats, &mut |me2: &Self| {
                        if !asked && !emb && yields < YIELDS_MAX && me2.shorter_waiting(prompt.len()) {
                            let _ = me2.send(&format!("BYIELD {b}"));
                            asked = true;
                        }
                    })?;
                    cont = self.ctl_state.lock().unwrap().result.as_ref().map(|r| r.1).unwrap_or(false);
                    phase = if cont { "slot" } else { "none" };
                    let yielded = self.ctl_state.lock().unwrap().yielded;
                    if !cont && yielded.map(|(s2, _)| s2 == b).unwrap_or(false) && !cancel.is_set() {
                        continue;
                    }
                    break;
                }
                self.ctl.release();
                holding = false;
                if !cont {
                    return Ok(());
                }
                let b = slot.expect("slot");
                self.set_live(b, |l| {
                    l.insert("state".into(), json!("decoding"));
                    l.insert("first_token".into(), json!(crate::sync::now()));
                    l.insert("generated".into(), json!(out.len()));
                });
                gen0 = Some(out.len().saturating_sub(1));
                let mut going_solo = false;
                let q = self.slot_rx.lock().unwrap().get(b).cloned();
                let Some(q) = q else { return Err(Stop::Err(self.died())) };
                loop {
                    let line = match q.recv_timeout(Duration::from_secs(10)) {
                        Err(RecvTimeoutError::Timeout) => {
                            if cancel.is_set() && !stop_sent {
                                self.send(&format!("BSTOP {b}")).map_err(Stop::Err)?;
                                stop_sent = true;
                            }
                            y.yield_(None)?;
                            continue;
                        }
                        Err(_) => None,
                        Ok(l) => l,
                    };
                    let Some(line) = line else {
                        phase = "none";
                        return Err(Stop::Err(self.died()));
                    };
                    if line.starts_with("BT ") {
                        let Some(t) = py::split_ws(&line).get(2).and_then(|v| v.parse::<u32>().ok()) else { continue };
                        out.push(t);
                        let n = out.len();
                        self.set_live(b, |l| {
                            l.insert("generated".into(), json!(n));
                        });
                        if cancel.is_set() {
                            if !stop_sent {
                                self.send(&format!("BSTOP {b}")).map_err(Stop::Err)?;
                                stop_sent = true;
                            }
                            continue;
                        }
                        y.yield_(Some(t))?;
                        if !going_solo && !stop_sent && self.may_go_solo(max_new - out.len() as i64, solo_again, emb) {
                            self.send(&format!("BSTOP {b}")).map_err(Stop::Err)?;
                            going_solo = true;
                        }
                    } else if line.starts_with("BDONE ") {
                        phase = "none";
                        let f = py::split_ws(&line);
                        if f.len() >= 5 {
                            let mut st = stats.lock().unwrap();
                            let m = st.get_or_insert_with(Map::new);
                            m.insert("finish".into(), json!(f[3]));
                            m.insert("decode_ms".into(), json!(f[4].parse::<f64>().unwrap_or(0.0)));
                            if is_digits(f[2]) {
                                m.insert("generated".into(), json!(f[2].parse::<i64>().unwrap_or(0)));
                            }
                        }
                        {
                            let g0 = gen0.unwrap_or(0);
                            let mut s = self.slots.lock().unwrap();
                            s.held[b] = if out.len() > g0 { prompt.iter().copied().chain(out[g0..out.len() - 1].iter().copied()).collect() } else { vec![] };
                        }
                        if going_solo && f.get(3) == Some(&"cancel") && !cancel.is_set() && (out.len() as i64) < max_new && !out.last().map(|t| EOS_IDS.contains(t)).unwrap_or(false) {
                            {
                                let mut s = self.slots.lock().unwrap();
                                s.live[b] = None;
                                s.used[b] = crate::sync::now();
                                s.busy[b] = false;
                                s.claims.remove(&b);
                                self.slot_cv.notify_all();
                            }
                            slot = None;
                            gen0 = None;
                            if reused0.is_none() {
                                reused0 = stats.lock().unwrap().as_ref().and_then(|m| m.get("reused").cloned());
                            }
                            prompt = ids.iter().copied().chain(out.iter().copied()).collect();
                            left = max_new - out.len() as i64;
                            solo_again += 1;
                            btrace("back to the solo path");
                            break;
                        }
                        return Ok(());
                    }
                }
            }
        };
        let r = body();
        // a consumer that left early (or an error): keep the engine and this server in step
        btrace(&format!("finally phase {phase} slot {slot:?} holding {holding}"));
        if (phase == "solo" || phase == "admit") && (self.proc_gen.load(SeqCst) != born || !self.alive()) {
            phase = "none";
        }
        if (phase == "solo" || phase == "admit") && self.ctl_state.lock().unwrap().erred {
            phase = "none";
        }
        if phase == "solo" {
            if self.send("STOP").is_ok() {
                self.drain_control("DONE", 300.0, born, &stats);
            }
        } else if phase == "admit" {
            if let Some(line) = self.drain_control("BADM", 300.0, born, &stats) {
                if line.starts_with("BADM ") && py::split_ws(&line).get(2) == Some(&"1") {
                    phase = "slot";
                }
            }
        }
        if let Some(r0) = reused0 {
            if let Some(m) = stats.lock().unwrap().as_mut() {
                m.insert("reused".into(), r0);
            }
        }
        if holding {
            self.ctl.release();
        }
        if admit_wait {
            let mut s = self.slots.lock().unwrap();
            s.admitting = s.admitting.saturating_sub(1);
        }
        if let Some(rv) = reserved {
            self.free_slot(rv);
        }
        if let Some(b) = slot {
            {
                let mut s = self.slots.lock().unwrap();
                if b < s.live.len() {
                    s.live[b] = None;
                    s.used[b] = crate::sync::now();
                }
            }
            if phase == "slot" {
                self.slots.lock().unwrap().held.get_mut(b).map(|h| h.clear());
                let stream = gen0.filter(|g| out.len() > *g).map(|g| prompt.iter().copied().chain(out[g..].iter().copied()).collect());
                self.release_slot_when_done(b, stream);
                self.slots.lock().unwrap().claims.remove(&b);
            } else {
                self.free_slot(b);
            }
        }
        r
    }

    /// Reads the control lines of the request on them (GEN / BGEN): each `T` goes out at once; heartbeats (and PP
    /// lines) run `at_beat` first (the BYIELD decision); returns at DONE (solo) or BADM.  `stop_when()` true sends
    /// STOP once.
    fn control(&self, y: &Yielder<Option<u32>, Fail>, cancel: &Event, out: &mut Vec<u32>, stop_when: Option<&dyn Fn() -> bool>, stats: &Stats, at_beat: &mut dyn FnMut(&Self)) -> Result<(), Stop<Fail>> {
        let Some(lines) = self.lines() else { return Err(Stop::Err(self.died())) };
        let mut stopped = false;
        let mut beat = Instant::now();
        loop {
            let line = match lines.recv_timeout(Duration::from_millis(500)) {
                Ok(l) => l,
                Err(RecvTimeoutError::Disconnected) => None,
                Err(RecvTimeoutError::Timeout) => {
                    if cancel.is_set() && !stopped {
                        self.send("STOP").map_err(Stop::Err)?;
                        stopped = true;
                    }
                    if since(beat) >= 10.0 {
                        beat = Instant::now();
                        at_beat(self);
                        y.yield_(None)?;
                    }
                    continue;
                }
            };
            beat = Instant::now();
            let Some(line) = line else { return Err(Stop::Err(self.died())) };
            if !line.starts_with("PP ") && !line.starts_with("INFO") {
                btrace(&format!("ctl< {} {}", self.ctl_state.lock().unwrap().mode, py::chars_prefix(py::strip(&line), 60)));
            }
            if let Some(t) = line.strip_prefix("T ") {
                if let Ok(t) = py::strip(t).parse::<u32>() {
                    out.push(t);
                    if !stopped && (cancel.is_set() || stop_when.map(|f| f()).unwrap_or(false)) {
                        self.send("STOP").map_err(Stop::Err)?;
                        stopped = true;
                    }
                    y.yield_(Some(t))?;
                }
            } else if line.starts_with("PP ") {
                let f = py::split_ws(&line);
                if f.len() >= 3 && is_digits(f[1]) && is_digits(f[2]) {
                    let mut live = self.live.lock().unwrap();
                    live.progress = Some((f[1].parse().unwrap_or(0), f[2].parse().unwrap_or(0)));
                    live.progress_ms = f.get(3).and_then(|v| v.parse().ok()).unwrap_or(0.0);
                    live.prefill_tok_s_mean = f.get(4).and_then(|v| v.parse().ok());
                }
                if !stopped && cancel.is_set() {
                    self.send("STOP").map_err(Stop::Err)?;
                    stopped = true;
                }
                at_beat(self);
                y.yield_(None)?;
            } else if line.starts_with("DONE") {
                *stats.lock().unwrap() = Some(Self::parse_done(&line));
            } else if line.starts_with("BADM ") {
                let f = py::split_ws(&line);
                self.ctl_state.lock().unwrap().result = Some(("badm".into(), f.len() >= 3 && f[2] == "1"));
                return Ok(());
            } else if let Some(rest) = line.strip_prefix("YIELDED ") {
                let f = py::split_ws(rest);
                if f.len() >= 2 {
                    if let (Ok(a), Ok(b)) = (f[0].parse::<usize>(), f[1].parse::<usize>()) {
                        self.ctl_state.lock().unwrap().yielded = Some((a, b));
                    }
                }
            } else if line.starts_with("ERR") {
                self.ctl_state.lock().unwrap().erred = true;
                return Err(Stop::Err(Fail::Value(py::strip(line.get(4..).unwrap_or("")).to_string())));
            }
            if line.starts_with("DONE") && self.ctl_state.lock().unwrap().mode == "solo" {
                self.ctl_state.lock().unwrap().result = Some(("done".into(), false));
                return Ok(());
            }
        }
    }

    // -------------------------------------------------------------------------------------------- sessions
    /// Disk sessions: `SAVE <path>` / `RESTORE <path>` between requests -> {"tokens", "bytes", "ms"}.
    pub fn session_file(&self, action: &str, path: &str) -> Result<(i64, i64, f64), Fail> {
        if !(action == "save" || action == "restore") || path.contains(['\r', '\n', '\0']) {
            return Err(Fail::Value("invalid session command".into()));
        }
        self.send(&format!("{} {path}", if action == "save" { "SAVE" } else { "RESTORE" }))?;
        let Some(lines) = self.lines() else { return Err(self.died()) };
        let silence = *self.silence_s.lock().unwrap();
        let mut heard = Instant::now();
        let want = if action == "save" { "SAVED" } else { "RESTORED" };
        let out_of_step = |line: &str| -> Fail {
            self.ended.store(true, SeqCst);
            if let Some(c) = self.proc.lock().unwrap().child.as_mut() {
                let _ = c.kill();
                wait_timeout(c, 20.0);
            }
            Fail::Died(format!("the engine answered a session {action} with {}; the server ended the engine", py::py_repr(&json!(py::chars_prefix(py::strip(line), 200)))))
        };
        let count = |s: &str| -> Option<i64> { if is_digits(s) { s.parse().ok() } else { None } };
        let mut last_done: i64 = -1;
        let mut allow = 0.0f64;
        loop {
            let limit = silence.max(allow);
            let line = if silence > 0.0 {
                let left = limit - since(heard);
                let r = if left <= 0.0 { Err(RecvTimeoutError::Timeout) } else { lines.recv_timeout(Duration::from_secs_f64(left)) };
                match r {
                    Ok(l) => l,
                    Err(RecvTimeoutError::Timeout) => {
                        return Err(self.silent(&format!("the engine said nothing for {limit:.0} s during a session {action}{}", if allow > silence { format!(" (in a blocking step allowed {allow:.0} s)") } else { String::new() })));
                    }
                    Err(_) => None,
                }
            } else {
                lines.recv().unwrap_or(None)
            };
            let Some(line) = line else { return Err(self.died()) };
            let f = py::split_ws(&line);
            let head = f.first().copied().unwrap_or("");
            match head {
                "SESSION" => {
                    let ok = f.len() == 3 && count(f[1]).zip(count(f[2])).map(|(d, t)| d <= t && d >= last_done).unwrap_or(false);
                    if !ok {
                        return Err(out_of_step(&line));
                    }
                    last_done = count(f[1]).unwrap();
                    heard = Instant::now();
                    allow = 0.0;
                }
                "SWAIT" => {
                    let ok = f.len() == 3 && !f[1].is_empty() && f[1].chars().all(|c| c.is_ascii_alphabetic()) && count(f[2]).map(|s| s > 0 && s as u64 <= SESSION_WAIT_MAX_S).unwrap_or(false);
                    if !ok {
                        return Err(out_of_step(&line));
                    }
                    heard = Instant::now();
                    allow = count(f[2]).unwrap() as f64;
                }
                "FATAL" => {
                    let why = py::strip(line.get(6..).unwrap_or("")).to_string();
                    return Err(Fail::Died(if why.is_empty() { "the engine ended during a session restore".into() } else { why }));
                }
                "SERR" => {
                    let parts: Vec<&str> = line.trim_end_matches(['\r', '\n']).splitn(4, ' ').collect();
                    if parts.len() != 4 || Fail::session_status(parts[1]).is_none() || !(parts[2] == "0" || parts[2] == "1") {
                        return Err(out_of_step(&line));
                    }
                    return Err(Fail::Session { kind: parts[1].into(), message: py::strip(parts[3]).into(), published: parts[2] == "1" });
                }
                h if h == want => {
                    let ok = f.len() == 4;
                    let (t, b) = (count(f.get(1).copied().unwrap_or("")), count(f.get(2).copied().unwrap_or("")));
                    let ms = f.get(3).and_then(|v| v.parse::<f64>().ok());
                    match (ok, t, b, ms) {
                        (true, Some(t), Some(b), Some(ms)) if ms.is_finite() && ms >= 0.0 => return Ok((t, b, ms)),
                        _ => return Err(out_of_step(&line)),
                    }
                }
                "INFO" | "WARN" => heard = Instant::now(),
                _ => return Err(out_of_step(&line)),
            }
        }
    }

    /// End the engine process: QUIT first, then terminate, then kill, each given 20 s.
    pub fn close(&self) -> Result<(), Fail> {
        let mut p = self.proc.lock().unwrap();
        let Some(child) = p.child.as_mut() else { return Ok(()) };
        if matches!(child.try_wait(), Ok(None)) {
            let quit = p.stdin.as_mut().map(|s| s.write_all(b"QUIT\n").and_then(|_| s.flush()).is_ok()).unwrap_or(false);
            p.stdin = None; // its stdin closed: EOF is QUIT for the engine too
            let child = p.child.as_mut().unwrap();
            if !quit || !wait_timeout(child, 20.0) {
                unsafe {
                    libc::kill(child.id() as i32, libc::SIGTERM);
                }
                if !wait_timeout(child, 20.0) {
                    let _ = child.kill();
                    if !wait_timeout(child, 20.0) {
                        return Err(Fail::Stuck("Strata is still releasing GPU/RAM; retry unloading after it exits".into()));
                    }
                }
            }
        }
        if let Some(h) = p.pump.take() {
            drop(p);
            let t = Instant::now();
            while !h.is_finished() && since(t) < 2.0 {
                std::thread::sleep(Duration::from_millis(10));
            }
            p = self.proc.lock().unwrap();
        }
        p.stdin = None;
        p.child = None;
        p.lines = None;
        self.ended.store(true, SeqCst);
        *self.live.lock().unwrap() = Live::default();
        Ok(())
    }
}

fn pump(stdout: std::process::ChildStdout, tx: Sender<Line>, slot_tx: Arc<Mutex<Vec<Sender<Line>>>>, me: std::sync::Weak<StrataEngine>, born: u64) {
    let mut r = BufReader::new(stdout);
    let mut buf = vec![];
    let mut last: Option<String> = None;
    loop {
        buf.clear();
        match r.read_until(b'\n', &mut buf) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let line = String::from_utf8_lossy(&buf).trim_end_matches(['\n', '\r']).to_string();
        if FATAL_PREFIXES.iter().any(|p| line.starts_with(p)) {
            if let Some(e) = me.upgrade() {
                if e.proc_gen.load(SeqCst) == born {
                    *e.silent_note.lock().unwrap() = Some(format!("Unrecoverable native verification failure: {}", py::strip(&line[4..])));
                    e.ended.store(true, SeqCst);
                }
                if let Some(c) = e.proc.lock().unwrap().child.as_mut() {
                    unsafe {
                        libc::kill(c.id() as i32, libc::SIGTERM);
                    }
                }
            }
            last = None;
            break;
        }
        if line.starts_with("BT ") || line.starts_with("BDONE ") {
            let slot = py::split_ws(&line).get(1).and_then(|v| v.parse::<usize>().ok());
            let txs = slot_tx.lock().unwrap();
            if let Some(s) = slot.and_then(|s| txs.get(s)) {
                let _ = s.send(Some(line.clone()));
                last = Some(line);
                continue;
            }
        }
        let _ = tx.send(Some(line.clone()));
        last = Some(line);
    }
    if let Some(e) = me.upgrade() {
        if e.proc_gen.load(SeqCst) == born {
            e.ended.store(true, SeqCst);
            if let Some(l) = last.filter(|l| l.starts_with("ERR")) {
                *e.last_err.lock().unwrap() = Some(py::strip(&l[3..]).to_string());
            }
        }
    }
    let _ = tx.send(None);
    for s in slot_tx.lock().unwrap().iter() {
        let _ = s.send(None);
    }
}

fn parse_info_value(v: &str) -> Value {
    let digits = v.strip_prefix('-').unwrap_or(v);
    if !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) {
        if let Ok(n) = v.parse::<i64>() {
            return json!(n);
        }
    }
    json!(v)
}

pub fn is_digits(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())
}

pub fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

fn exit_code_of(s: std::process::ExitStatus) -> i32 {
    use std::os::unix::process::ExitStatusExt;
    s.code().unwrap_or_else(|| -s.signal().unwrap_or(0))
}

/// Wait up to `secs` for the child to exit; true once it has.
pub fn wait_timeout(c: &mut Child, secs: f64) -> bool {
    let t = Instant::now();
    loop {
        match c.try_wait() {
            Ok(Some(_)) => return true,
            Ok(None) if since(t) < secs => std::thread::sleep(Duration::from_millis(20)),
            Ok(None) => return false,
            Err(_) => return true,
        }
    }
}

// ------------------------------------------------------------------------------------------------ the mock
/// Replays a scripted completion as token ids, one per step, then the end-of-turn token.  Given several scripts, each
/// request gets the next one and the last one repeats.
pub struct MockEngine {
    pub max_context: i64,
    delay: f64,
    scripts: Vec<Vec<u32>>,
    turns: Mutex<usize>,
    pub last_prompt: Mutex<Vec<u32>>,
}

impl MockEngine {
    pub fn new(tok: &dyn crate::tokenizer::Tokenize, scripts: &[String], max_context: i64, delay_s: f64) -> Self {
        let end = tok.encode("<|im_end|>", true);
        MockEngine {
            max_context,
            delay: delay_s,
            scripts: scripts.iter().map(|s| tok.encode(s, true).into_iter().chain(end.iter().copied()).collect()).collect(),
            turns: Mutex::new(0),
            last_prompt: Mutex::new(vec![]),
        }
    }

    pub fn generate(&self, ids: Vec<u32>, max_new: i64, cancel: Arc<Event>) -> Generation {
        *self.last_prompt.lock().unwrap() = ids;
        let script = if self.scripts.len() > 1 {
            let mut t = self.turns.lock().unwrap();
            let s = self.scripts[(*t).min(self.scripts.len() - 1)].clone();
            *t += 1;
            s
        } else {
            self.scripts[0].clone()
        };
        let delay = self.delay;
        let n = max_new.max(0) as usize;
        let tokens = Gen::new(move |y: &Yielder<Option<u32>, Fail>| {
            for t in script.into_iter().take(n) {
                if cancel.is_set() {
                    return Ok(());
                }
                if delay > 0.0 {
                    std::thread::sleep(Duration::from_secs_f64(delay));
                }
                y.yield_(Some(t))?;
            }
            Ok(())
        });
        Generation { tokens, stats: Arc::new(Mutex::new(None)) }
    }
}

/// The engine the service runs on: the mock (`--engine mock`) or the real one.
#[derive(Clone)]
pub enum Engine {
    Mock(Arc<MockEngine>),
    Strata(Arc<StrataEngine>),
}

impl Engine {
    pub fn generate(&self, ids: Vec<u32>, max_new: i64, sampling: Map<String, Value>, cancel: Arc<Event>, embeddings: Option<String>) -> Generation {
        match self {
            Engine::Mock(m) => m.generate(ids, max_new, cancel),
            Engine::Strata(s) => s.generate(ids, max_new, sampling, cancel, embeddings),
        }
    }
    pub fn strata(&self) -> Option<&Arc<StrataEngine>> {
        match self {
            Engine::Strata(s) => Some(s),
            _ => None,
        }
    }
    /// hasattr(engine, "alive") and engine.alive(); the mock is always loaded
    pub fn loaded(&self) -> bool {
        self.strata().map(|s| s.alive()).unwrap_or(true)
    }
    pub fn max_context(&self) -> i64 {
        match self {
            Engine::Mock(m) => m.max_context,
            Engine::Strata(s) => s.max_context.load(SeqCst),
        }
    }
    pub fn known_ctx(&self) -> i64 {
        self.strata().map(|s| s.known_ctx.load(SeqCst)).unwrap_or(0)
    }
    pub fn starting(&self) -> bool {
        self.strata().map(|s| s.starting.load(SeqCst)).unwrap_or(false)
    }
    pub fn batch(&self) -> usize {
        self.strata().map(|s| s.batch()).unwrap_or(0)
    }
    pub fn info(&self) -> Map<String, Value> {
        self.strata().map(|s| s.info.lock().unwrap().clone()).unwrap_or_default()
    }
    pub fn live(&self) -> Live {
        self.strata().map(|s| s.live.lock().unwrap().clone()).unwrap_or_default()
    }
    pub fn unloaded(&self) -> bool {
        self.strata().map(|s| s.unloaded.load(SeqCst)).unwrap_or(false)
    }
}
