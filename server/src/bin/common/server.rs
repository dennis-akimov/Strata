//! Start a run config's strata-server on its port, wait until our own engine has loaded, stop it again
//! (shared by strata-bench and strata-eval).

use serde_json::Value;
use std::fs;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

pub fn get_json(url: &str, key: &str) -> Option<Value> {
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(5))).build().into();
    let r = agent.get(url).header("Authorization", format!("Bearer {key}")).call().ok()?;
    serde_json::from_reader(r.into_body().into_reader()).ok()
}

pub fn start(server: &Path, cfg: &Path, port: u64, log: &Path, root: &Path) -> Child {
    let log = fs::File::create(log).unwrap();
    Command::new(server).args(["--engine", "strata", "--config"]).arg(cfg).args(["--port", &port.to_string(), "--replace"])
        .current_dir(root).stdin(Stdio::null()).stdout(log.try_clone().unwrap()).stderr(log).spawn().expect("start strata-server")
}

/// Waits for /health loaded; Err when the server exits first (e.g. the engine does not fit) or after 10 minutes.
pub fn wait_loaded(child: &mut Child, base: &str, key: &str) -> Result<f64, String> {
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_secs(600) {
        if let Ok(Some(s)) = child.try_wait() {
            return Err(format!("server exited ({s})"));
        }
        // the port may still be the server --replace is stopping: ours only once our child runs its engine
        let ours = Command::new("/usr/bin/pgrep").args(["-P", &child.id().to_string(), "strata-metal"]).output().is_ok_and(|o| o.status.success());
        if ours && get_json(&format!("{base}/health"), key).is_some_and(|h| h["loaded"] == true) {
            return Ok(t0.elapsed().as_secs_f64());
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    Err("not loaded after 600 s".into())
}

/// Stop a server we started; --replace only finds one that already holds the port, so ours is stopped by hand.
pub fn stop(mut c: Child) {
    unsafe { libc::kill(c.id() as i32, libc::SIGTERM) };
    let _ = c.wait();
}
