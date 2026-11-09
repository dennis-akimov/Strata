pub mod api;
pub mod bias;
pub mod clients;
pub mod dotenv;
pub mod engine;
pub mod errors;
pub mod firewall;
pub mod frontend;
pub mod http;
pub mod hub;
pub mod imagegen;
pub mod mcp;
pub mod net;
pub mod observe;
pub mod parser;
pub mod prometheus;
pub mod py;
pub mod pyjson;
pub mod responses;
pub mod run;
pub mod runconfig;
pub mod schedules;
pub mod service;
pub mod structured;
pub mod sync;
pub mod team;
pub mod telemetry;
pub mod template;
pub mod tokenizer;
pub mod videogen;
pub mod vision;
pub mod voice;

/// A "[strata] " line on stdout, with the time (the Python server's Stamped stream), and in the run config's log
/// when log_to set one: stdout reaches only Strata.app's window, so "a request too big for its context goes to the
/// main engine" (why Zed waited behind deep research, 2026-11-06) was nowhere to be read afterwards.
pub fn say(msg: &str) {
    let line = engine::stamp(&format!("[strata] {msg}"));
    println!("{line}");
    if let Some(f) = LOG.get() {
        use std::io::Write;
        let _ = writeln!(f.lock().unwrap_or_else(|e| e.into_inner()), "{line}");
    }
}

static LOG: std::sync::OnceLock<std::sync::Mutex<std::fs::File>> = std::sync::OnceLock::new();

/// say() writes to this file too (appended, beside the engine's own lines)
pub fn log_to(path: &str) {
    if let Ok(f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = LOG.set(std::sync::Mutex::new(f));
    }
}

/// Strata's folder (the web app's files, tools/): $STRATA_ROOT, else the folder this crate is in.
pub fn root() -> std::path::PathBuf {
    match std::env::var("STRATA_ROOT") {
        Ok(r) if !r.is_empty() => r.into(),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf(),
    }
}

#[cfg(test)]
mod say_tests {
    #[test]
    fn say_goes_to_the_log_too() {
        let p = std::env::temp_dir().join(format!("strata-say-{}.log", std::process::id()));
        super::log_to(p.to_str().unwrap());
        super::say("a request too big for its context goes to the main engine");
        let text = std::fs::read_to_string(&p).unwrap();
        // other tests say() at the same time, into the same file: find this test's line
        assert!(text.lines().any(|l| l.starts_with("[strata] 2") && l.ends_with("goes to the main engine")), "{text}");
        let _ = std::fs::remove_file(&p);
    }
}
