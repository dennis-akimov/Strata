pub mod api;
pub mod bias;
pub mod engine;
pub mod errors;
pub mod firewall;
pub mod frontend;
pub mod http;
pub mod imagegen;
pub mod mcp;
pub mod net;
pub mod parser;
pub mod prometheus;
pub mod py;
pub mod pyjson;
pub mod responses;
pub mod run;
pub mod runconfig;
pub mod service;
pub mod structured;
pub mod sync;
pub mod telemetry;
pub mod template;
pub mod tokenizer;
pub mod videogen;
pub mod vision;

/// A "[strata] " line on stdout, with the time (the Python server's Stamped stream).
pub fn say(msg: &str) {
    println!("{}", engine::stamp(&format!("[strata] {msg}")));
}

/// Strata's folder (the web app's files, tools/): $STRATA_ROOT, else the folder this crate is in.
pub fn root() -> std::path::PathBuf {
    match std::env::var("STRATA_ROOT") {
        Ok(r) if !r.is_empty() => r.into(),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf(),
    }
}
