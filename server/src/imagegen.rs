//! Pictures from a prompt: POST /v1/images/generations (serve/imagegen.py), made by strata-image (Nucleus-Image, or
//! HunyuanImage-3.0 with its own tokenizer) started for one picture and exited after it.  This server is the gate:
//! the picture takes the server's turn, the chat model is unloaded when memory is short, the prompts go to the
//! process as files in a private folder, and a client that leaves (or the deadline) has it stopped.

use crate::py::{self, get};
use crate::pyjson;
use crate::say;
use crate::service::Service;
use crate::sync::Event;
use base64::Engine as _;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Mutex;
use std::time::{Duration, Instant};

const MAX_PROMPT_CHARS: usize = 4000;
const MAX_PIXELS: i64 = 1536 * 1024;
const KILL_AFTER_S: f64 = 5.0;

pub struct Params {
    pub prompt: String,
    pub negative: String,
    pub width: i64,
    pub height: i64,
    pub steps: i64,
    pub cfg: f64,
    pub seed: i64,
}

fn int_of(v: &Value) -> Option<i64> {
    if v.is_i64() || v.is_u64() { v.as_i64() } else { None }
}

/// "WxH" as Python's `(int(v) for v in s.lower().split("x"))`.
pub fn size_of(v: &Value, default: &str) -> Option<(i64, i64)> {
    let s = if v.is_null() { default.to_string() } else { py::py_str(v) }.to_lowercase();
    let parts: Vec<&str> = s.split('x').collect();
    if parts.len() != 2 {
        return None;
    }
    Some((py::strip(parts[0]).parse().ok()?, py::strip(parts[1]).parse().ok()?))
}

pub fn now_seed() -> i64 {
    (crate::sync::now() * 1000.0) as i64 % (1 << 31)
}

/// OpenAI's images request plus steps, guidance_scale, seed and negative_prompt; Err: a 400.
pub fn parse_request(body: &Value, steps_default: i64, cfg_default: f64) -> Result<Params, String> {
    if !body.is_object() {
        return Err("the body must be a JSON object".into());
    }
    let prompt = get(body, "prompt").as_str().filter(|p| !py::strip(p).is_empty()).ok_or("prompt must be a non-empty string")?;
    if prompt.chars().count() > MAX_PROMPT_CHARS {
        return Err(format!("prompt is longer than {MAX_PROMPT_CHARS} characters"));
    }
    if body.get("n").map(|n| n.as_f64() != Some(1.0)).unwrap_or(false) {
        return Err("n must be 1 (one picture at a time)".into());
    }
    if body.get("response_format").map(|r| r.as_str() != Some("b64_json")).unwrap_or(false) {
        return Err("response_format must be b64_json (this server keeps no files to link to)".into());
    }
    let (width, height) = size_of(get(body, "size"), "1024x1024").ok_or(r#"size must look like "1024x1024""#)?;
    if width % 16 != 0 || height % 16 != 0 || !(256 <= width.min(height) && width.max(height) <= 2048) {
        return Err("size: width and height must be multiples of 16 from 256 to 2048".into());
    }
    if width * height > MAX_PIXELS {
        return Err("size: at most 1536x1024 pixels in all (1024x1024, 1536x1024, 1248x1248 ...)".into());
    }
    let steps = match body.get("steps") {
        None => Some(steps_default),
        Some(v) => int_of(v),
    };
    let steps = steps.filter(|s| (1..=100).contains(s)).ok_or("steps must be an integer from 1 to 100")?;
    let cfg = match body.get("guidance_scale") {
        None => Some(cfg_default),
        Some(v) if v.is_number() => v.as_f64(),
        _ => None,
    };
    let cfg = cfg.filter(|c| (0.0..=20.0).contains(c)).ok_or("guidance_scale must be a number from 0 to 20")?;
    let seed = match body.get("seed") {
        None | Some(Value::Null) => now_seed(),
        Some(v) => int_of(v).filter(|s| (0..1 << 31).contains(s)).ok_or("seed must be an integer from 0 to 2^31-1")?,
    };
    let negative = match body.get("negative_prompt") {
        None => String::new(),
        Some(Value::String(s)) if s.chars().count() <= MAX_PROMPT_CHARS => s.clone(),
        _ => return Err(format!("negative_prompt must be a string of at most {MAX_PROMPT_CHARS} characters")),
    };
    Ok(Params { prompt: prompt.to_string(), negative, width, height, steps, cfg, seed })
}

pub fn error(msg: &str, kind: &str) -> Vec<u8> {
    pyjson::dumps(&json!({"error": {"type": kind, "message": msg}})).into_bytes()
}

/// The memory macOS can give now (GB), None when unreadable.
pub fn available_gb() -> Option<f64> {
    let (used, total) = crate::telemetry::ram();
    Some((total? - used?) as f64 / 1e9)
}

pub struct ImageGen {
    pub exe: String,
    pub dit: Option<String>,
    pub encoder: Option<String>,
    pub vae: Option<String>,
    pub hunyuan: Option<String>,
    pub tokenizer: Option<String>,
    pub need_gb: f64,
    pub queue_s: f64,
    pub timeout_s: f64,
    tok: Mutex<Option<crate::tokenizer::Tokenizer>>,
    pub defaults: (i64, f64),
}

// HunyuanImage-3.0-Instruct-Distil's prompt tokens (tools/image/hunyuan_tokens.py)
const H_BOS: u32 = 127958;
const H_ANSWER: u32 = 128025;
const H_BOI: u32 = 128000;
const H_SIZE_1024: u32 = 128037;
const H_TIMESTEP: u32 = 128017;
const H_GUIDANCE: u32 = 130082;
const H_TIMESTEP_R: u32 = 130107;

fn reso() -> Vec<(i64, i64)> {
    let mut r: Vec<(i64, i64)> = (0..9).map(|i| (512, 2048 - 64 * i)).collect();
    r.extend((0..7).map(|i| (576 + 64 * i, 1472 - 64 * i)));
    r.push((1024, 1024));
    r.extend((0..7).map(|i| (1088 + 64 * i, 960 - 64 * i)));
    r.extend((0..9).map(|i| (1536 + 64 * i, 512)));
    r.extend([(1024, 768), (1280, 720), (768, 1024), (720, 1280)]);
    r
}

fn ratio_token(i: usize) -> u32 {
    if i <= 32 { 128044 + i as u32 } else { 130103 + i as u32 - 33 }
}

/// The nearest size the model was trained for: (index, height, width), the first one of the closest ratio.
pub fn resolution(height: i64, width: i64) -> (usize, i64, i64) {
    let r = height as f64 / width as f64;
    let all = reso();
    let mut best = 0;
    for k in 0..all.len() {
        let d = |k: usize| (all[k].0 as f64 / all[k].1 as f64 - r).abs();
        if d(k) < d(best) {
            best = k;
        }
    }
    (best, all[best].0, all[best].1)
}

fn root() -> PathBuf {
    crate::root()
}

impl ImageGen {
    /// The config's "image" entry, or None; a bad entry is an Err at start.
    pub fn from_config(cfg: &Value) -> Result<Option<ImageGen>, String> {
        if cfg.is_null() {
            return Ok(None);
        }
        if !cfg.is_object() {
            return Err(r#""image" must be {"dit": ..., "encoder": ..., "vae": ...} (docs/MACOS.md)"#.into());
        }
        let hunyuan = cfg.get("hunyuan").is_some();
        let keys: &[&str] = if hunyuan { &["hunyuan", "tokenizer", "exe"] } else { &["dit", "encoder", "vae", "exe"] };
        let mut files = std::collections::HashMap::new();
        for k in keys {
            let v = if *k == "exe" && cfg.get("exe").is_none() {
                Some([root().join("engine-metal/strata-image"), root().join("build-metal/metal/strata-image")].into_iter().find(|p| p.is_file()).map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|| "strata-image".into()))
            } else {
                cfg.get(*k).and_then(|v| v.as_str().map(String::from))
            };
            let p = v.map(|v| expand(&v));
            match p.filter(|p| p.is_file()) {
                Some(p) => {
                    files.insert(*k, std::fs::canonicalize(&p).unwrap_or(p).to_string_lossy().into_owned());
                }
                None => return Err(format!("\"image\".{k} must name an existing file{}", if *k == "exe" { " (run setup, or: cmake --build build-metal --target strata-image)" } else { "" })),
            }
        }
        let mut nums = vec![];
        for (k, d) in [("need_gb", if hunyuan { 70.0 } else { 16.0 }), ("queue_s", 600.0), ("timeout_s", 1800.0)] {
            let v = match cfg.get(k) {
                None => Some(d),
                Some(v) if v.is_number() => v.as_f64(),
                _ => None,
            };
            nums.push(v.filter(|v| v.is_finite() && *v > 0.0).ok_or(format!("\"image\".{k} must be a positive number"))?);
        }
        Ok(Some(ImageGen {
            exe: files["exe"].clone(),
            dit: files.get("dit").cloned(),
            encoder: files.get("encoder").cloned(),
            vae: files.get("vae").cloned(),
            hunyuan: files.get("hunyuan").cloned(),
            tokenizer: files.get("tokenizer").cloned(),
            need_gb: nums[0],
            queue_s: nums[1],
            timeout_s: nums[2],
            tok: Mutex::new(None),
            defaults: if hunyuan { (8, 2.5) } else { (30, 4.0) },
        }))
    }

    /// tools/image/hunyuan_tokens.py's file: the size the model makes and the ids before <timestep>.
    pub fn hunyuan_tokens(&self, prompt: &str, height: i64, width: i64) -> Result<String, String> {
        use crate::tokenizer::Tokenize;
        let mut tok = self.tok.lock().unwrap();
        if tok.is_none() {
            *tok = Some(crate::tokenizer::Tokenizer::from_hf_json(Path::new(self.tokenizer.as_deref().unwrap_or("")), "hunyuan").map_err(|e| e.to_string())?);
        }
        let t = tok.as_ref().unwrap();
        let system = std::fs::read_to_string(root().join("tools/image/hunyuan_system_prompt.txt")).map_err(|e| e.to_string())?;
        let (i, h, w) = resolution(height, width);
        let mut ids = vec![H_BOS];
        ids.extend(t.encode(&system, true));
        ids.extend(t.encode("\n\n", false));
        ids.extend(t.encode("User: ", false));
        ids.extend(t.encode(prompt, false));
        ids.extend(t.encode("\n\n", false));
        ids.extend(t.encode("Assistant: ", false));
        ids.extend([H_ANSWER, H_BOI, H_SIZE_1024, ratio_token(i), H_TIMESTEP, H_GUIDANCE, H_TIMESTEP_R]);
        let head = &ids[..ids.len() - 3];
        Ok(format!("{h} {w}\n{}\n", head.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ")))
    }

    pub fn validate(&self, req: &Value) -> Option<(u16, Vec<u8>)> {
        parse_request(req, self.defaults.0, self.defaults.1).err().map(|e| (400, error(&e, "invalid_request_error")))
    }

    /// The picture: (status, retry_after, body), after strata-image has exited.
    pub fn forward(&self, req: &Value, cancel: &Event) -> (u16, Option<String>, Vec<u8>) {
        let p = match parse_request(req, self.defaults.0, self.defaults.1) {
            Ok(p) => p,
            Err(e) => return (400, None, error(&e, "invalid_request_error")),
        };
        let d = match private_dir("strata-image-") {
            Ok(d) => d,
            Err(e) => return (500, None, error(&format!("strata-image could not start ({e})"), "server_error")),
        };
        let r = self.forward_in(&d, &p, cancel);
        let _ = std::fs::remove_dir_all(&d);
        r
    }

    fn forward_in(&self, d: &Path, p: &Params, cancel: &Event) -> (u16, Option<String>, Vec<u8>) {
        let mut args: Vec<String>;
        if let Some(h) = &self.hunyuan {
            let tokens = match self.hunyuan_tokens(&p.prompt, p.height, p.width) {
                Ok(t) => t,
                Err(e) => return (500, None, error(&format!("strata-image could not start ({e})"), "server_error")),
            };
            let _ = std::fs::write(d.join("tokens.txt"), tokens);
            args = vec!["--hunyuan".into(), h.clone(), "--tokens".into(), d.join("tokens.txt").to_string_lossy().into_owned()];
        } else {
            let _ = std::fs::write(d.join("prompt.txt"), &p.prompt);
            let _ = std::fs::write(d.join("negative.txt"), &p.negative);
            let s = |o: &Option<String>| o.clone().unwrap_or_default();
            args = vec!["--encoder".into(), s(&self.encoder), "--dit".into(), s(&self.dit), "--vae".into(), s(&self.vae), "--prompt-file".into(), d.join("prompt.txt").to_string_lossy().into_owned(), "--negative-file".into(), d.join("negative.txt").to_string_lossy().into_owned(), "--width".into(), p.width.to_string(), "--height".into(), p.height.to_string()];
        }
        args.extend(["--steps".into(), p.steps.to_string(), "--cfg".into(), crate::pyjson::float_repr(p.cfg), "--seed".into(), p.seed.to_string(), "--out".into(), d.join("out.png").to_string_lossy().into_owned(), "--watch-stdin".into(), "1".into()]);
        let Ok(errf) = std::fs::File::create(d.join("stderr.txt")) else { return (500, None, error("strata-image could not start (stderr)", "server_error")) };
        use std::os::unix::process::CommandExt;
        let mut child = match crate::firewall::command(&self.exe, true).args(&args).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(errf).process_group(0).spawn() {
            Ok(c) => c,
            Err(e) => return (500, None, error(&format!("strata-image could not start ({e})"), "server_error")),
        };
        let _stdin = child.stdin.take(); // open and unwritten: strata-image ends itself when it closes
        let stdout = child.stdout.take().unwrap();
        let lines = std::sync::Arc::new(Mutex::new(Vec::<Value>::new()));
        let l2 = lines.clone();
        let reader = std::thread::spawn(move || {
            for line in BufReader::new(stdout).split(b'\n').map_while(Result::ok) {
                if let Ok(Value::Object(o)) = serde_json::from_slice::<Value>(&line) {
                    l2.lock().unwrap().push(Value::Object(o));
                }
            }
        });
        let deadline = Instant::now() + Duration::from_secs_f64(self.timeout_s);
        let mut why = None;
        while matches!(child.try_wait(), Ok(None)) {
            if cancel.is_set() || Instant::now() >= deadline {
                why = Some(if cancel.is_set() { "cancelled" } else { "deadline" });
                stop(&mut child);
                break;
            }
            cancel.wait(Some(Duration::from_millis(250)));
        }
        let status = child.wait().ok();
        let t = Instant::now();
        while !reader.is_finished() && t.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(10));
        }
        if let Some(w) = why {
            say(&format!("picture stopped: {}", if w == "cancelled" { "cancelled" } else { "past timeout_s" }));
            return if w == "cancelled" {
                (499, None, error("the picture was cancelled", "cancelled"))
            } else {
                (504, None, error(&format!("strata-image took longer than {:.0} s", self.timeout_s), "server_error"))
            };
        }
        let lines = lines.lock().unwrap().clone();
        let done = lines.iter().find(|e| py::truthy(get(e, "done"))).cloned();
        let code = status.and_then(|s| s.code());
        let out = d.join("out.png");
        if code != Some(0) || done.is_none() || !out.is_file() {
            let msg = lines.iter().find_map(|e| e.get("error").and_then(|x| x.as_str()).map(String::from)).unwrap_or_else(|| {
                let tail = std::fs::read(d.join("stderr.txt")).unwrap_or_default();
                let tail = String::from_utf8_lossy(&tail[tail.len().saturating_sub(600)..]).trim().to_string();
                format!("strata-image ended with status {}{}", code.map(|c| c.to_string()).unwrap_or_else(|| "None".into()), if tail.is_empty() { String::new() } else { format!(": {tail}") })
            });
            say(&format!("picture failed: {msg}"));
            return (500, None, error(&msg, "server_error"));
        }
        let png = std::fs::read(&out).unwrap_or_default();
        let mut info: serde_json::Map<String, Value> = done.unwrap().as_object().cloned().unwrap_or_default();
        info.shift_remove("done");
        info.insert("steps".into(), json!(p.steps));
        info.insert("seed".into(), json!(p.seed));
        say(&format!("picture {}x{}, {} steps: {}", p.width, p.height, p.steps, pyjson::dumps(&Value::Object(info.clone()))));
        let body = json!({"created": crate::sync::now() as i64, "data": [{"b64_json": base64::engine::general_purpose::STANDARD.encode(&png)}], "strata": info});
        (200, None, pyjson::dumps(&body).into_bytes())
    }
}

pub fn expand(p: &str) -> PathBuf {
    match p.strip_prefix("~/") {
        Some(rest) => std::env::var("HOME").map(|h| Path::new(&h).join(rest)).unwrap_or_else(|_| PathBuf::from(p)),
        None => PathBuf::from(p),
    }
}

/// A 0700 temporary folder (tempfile.mkdtemp).
pub fn private_dir(prefix: &str) -> std::io::Result<PathBuf> {
    use std::os::unix::fs::DirBuilderExt;
    let d = std::env::temp_dir().join(format!("{prefix}{}", &uuid::Uuid::new_v4().simple().to_string()[..8]));
    std::fs::DirBuilder::new().mode(0o700).create(&d)?;
    Ok(d)
}

/// SIGTERM to the process group, SIGKILL when it has not exited KILL_AFTER_S later.
pub fn stop(child: &mut std::process::Child) {
    let pg = child.id() as i32;
    unsafe { libc::killpg(pg, libc::SIGTERM) };
    if !crate::engine::wait_timeout(child, KILL_AFTER_S) {
        unsafe { libc::killpg(pg, libc::SIGKILL) };
        let _ = child.wait();
    }
}

/// With the FIFO held and the image gate set: room for a picture.  "enough", "unloaded", "cancelled", or a refusal:
/// "busy", "short", "unsupported".
pub fn make_room(svc: &Service, need_gb: f64, deadline: Instant, cancel: &Event) -> &'static str {
    if available_gb().map(|a| a >= need_gb).unwrap_or(false) {
        return "enough";
    }
    let mut r = "busy";
    while !cancel.is_set() {
        r = svc.unload_held(None, true).unwrap_or("busy");
        if r != "busy" || Instant::now() >= deadline {
            break;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    if cancel.is_set() {
        return "cancelled";
    }
    if r == "busy" || r == "unsupported" {
        return r;
    }
    let until = deadline.min(Instant::now() + Duration::from_secs(15));
    loop {
        match available_gb() {
            None => return if r == "unloaded" { "unloaded" } else { "enough" },
            Some(a) if a >= need_gb => return if r == "unloaded" { "unloaded" } else { "enough" },
            _ => {}
        }
        if Instant::now() >= until || cancel.is_set() {
            return "short";
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hunyuan_sizes_and_requests() {
        assert_eq!(reso().len(), 37);
        assert_eq!(resolution(1024, 1024), (16, 1024, 1024));
        assert_eq!(ratio_token(33), 130103);
        assert!(parse_request(&json!({"prompt": "x", "size": "100x100"}), 30, 4.0).is_err());
        let p = parse_request(&json!({"prompt": "a cat", "size": "1536x1024", "seed": 5}), 30, 4.0).unwrap();
        assert_eq!((p.width, p.height, p.steps, p.seed), (1536, 1024, 30, 5));
    }
}
