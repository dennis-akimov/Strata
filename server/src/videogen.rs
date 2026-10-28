//! Videos from a prompt: OpenAI's video API (serve/videogen.py) made by stable-diffusion.cpp's sd-cli (Wan 2.2 A14B):
//! POST answers 202 with a job to poll; the job takes the server's turn, the chat model is unloaded when memory is
//! short, one process per job, SIGTERM then SIGKILL on cancel or past timeout_s; a finished clip is kept keep_s.

use crate::imagegen::{available_gb, expand, make_room, private_dir, size_of, stop};
use crate::py::{self, get};
use crate::say;
use crate::service::Service;
use crate::sync::Event;
use regex::Regex;
use serde_json::{json, Value};
use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicI64, Ordering::SeqCst};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const MAX_PROMPT_CHARS: usize = 4000;
const MAX_JOBS: usize = 8;
const MAX_PIXELS: i64 = 1280 * 720;
const FPS: i64 = 16;
const MAX_SECONDS: f64 = 10.0;

#[derive(Clone, Copy)]
pub struct Defaults {
    steps: i64,
    high_noise_steps: i64,
    guidance_scale: f64,
}
const PLAIN: Defaults = Defaults { steps: 10, high_noise_steps: 8, guidance_scale: 3.5 };
const LIGHTNING: Defaults = Defaults { steps: 2, high_noise_steps: 2, guidance_scale: 1.0 };

#[derive(Clone)]
pub struct Params {
    prompt: String,
    width: i64,
    height: i64,
    frames: i64,
    steps: i64,
    high_noise_steps: i64,
    cfg: f64,
    seed: i64,
    negative: String,
}

pub fn parse_request(body: &Value, d: Defaults) -> Result<Params, String> {
    if !body.is_object() {
        return Err("the body must be a JSON object".into());
    }
    let prompt = get(body, "prompt").as_str().filter(|p| !py::strip(p).is_empty() && p.chars().count() <= MAX_PROMPT_CHARS).ok_or(format!("prompt must be a non-empty string of at most {MAX_PROMPT_CHARS} characters"))?;
    for text in [get(body, "prompt"), get(body, "negative_prompt")] {
        if text.as_str().map(|t| t.to_lowercase().contains("<lora:")).unwrap_or(false) {
            return Err("a prompt may not contain <lora:...>: the server's config chooses the LoRAs".into());
        }
    }
    let (width, height) = size_of(get(body, "size"), "832x480").ok_or(r#"size must look like "832x480""#)?;
    if width % 16 != 0 || height % 16 != 0 || width.min(height) < 256 || width.max(height) > 1280 || width * height > MAX_PIXELS {
        return Err("size: multiples of 16, each side 256-1280, at most 1280x720 pixels in all".into());
    }
    let seconds = match body.get("seconds") {
        None => 2.0,
        Some(Value::Number(n)) => n.as_f64().unwrap_or(f64::NAN),
        Some(Value::String(s)) => py::strip(s).parse::<f64>().map_err(|_| "seconds must be a number".to_string())?,
        Some(Value::Bool(b)) => *b as i64 as f64,
        Some(_) => return Err("seconds must be a number".into()),
    };
    if !seconds.is_finite() || !(seconds > 0.0 && seconds <= MAX_SECONDS) {
        return Err(format!("seconds must be more than 0 and at most {}", crate::service::py_num(MAX_SECONDS)));
    }
    let frames = ((crate::service::pyround(seconds * FPS as f64 / 4.0, 0)) as i64) * 4 + 1;
    let int = |k: &str, def: i64| -> Result<i64, String> {
        match body.get(k) {
            None => Ok(def),
            Some(v) if v.is_i64() && (1..=60).contains(&v.as_i64().unwrap()) => Ok(v.as_i64().unwrap()),
            _ => Err(format!("{k} must be an integer from 1 to 60")),
        }
    };
    let steps = int("steps", d.steps)?;
    let high = int("high_noise_steps", d.high_noise_steps)?;
    let g = match body.get("guidance_scale") {
        None => Some(d.guidance_scale),
        Some(v) if v.is_number() => v.as_f64(),
        _ => None,
    };
    let g = g.filter(|g| (1.0..=20.0).contains(g)).ok_or("guidance_scale must be a number from 1 to 20")?;
    let seed = match body.get("seed") {
        None => crate::imagegen::now_seed(),
        Some(v) if v.is_i64() && (0..1 << 31).contains(&v.as_i64().unwrap()) => v.as_i64().unwrap(),
        _ => return Err("seed must be an integer from 0 to 2^31-1".into()),
    };
    let negative = match body.get("negative_prompt") {
        None => String::new(),
        Some(Value::String(s)) if s.chars().count() <= MAX_PROMPT_CHARS => s.clone(),
        _ => return Err(format!("negative_prompt must be a string of at most {MAX_PROMPT_CHARS} characters")),
    };
    Ok(Params { prompt: prompt.into(), width, height, frames: frames.max(5), steps, high_noise_steps: high, cfg: g, seed, negative })
}

pub struct Job {
    pub id: String,
    p: Params,
    pub status: Mutex<String>,
    progress: AtomicI64,
    error: Mutex<Option<String>>,
    created_at: i64,
    completed_at: Mutex<Option<i64>>,
    pub cancel: Event,
    proc: Mutex<Option<u32>>,
    pub dir: PathBuf,
    pub file: Mutex<PathBuf>,
}

impl Job {
    pub fn view(&self, model: &str, keep_s: f64) -> Value {
        let status = self.status.lock().unwrap().clone();
        let completed = *self.completed_at.lock().unwrap();
        let err = self.error.lock().unwrap().clone();
        json!({"id": self.id, "object": "video", "model": model, "status": status, "progress": self.progress.load(SeqCst),
            "created_at": self.created_at, "completed_at": completed,
            "expires_at": match completed { Some(c) if status == "completed" => json!(c + keep_s as i64), _ => Value::Null },
            "prompt": self.p.prompt, "size": format!("{}x{}", self.p.width, self.p.height),
            "seconds": crate::engine::py_g((self.p.frames - 1) as f64 / FPS as f64), "remixed_from_video_id": null,
            "error": err.map(|e| json!({"code": "video_failed", "message": e})).unwrap_or(Value::Null)})
    }

    fn fail(&self, msg: &str) {
        *self.status.lock().unwrap() = "failed".into();
        *self.error.lock().unwrap() = Some(msg.into());
        *self.completed_at.lock().unwrap() = Some(crate::sync::now() as i64);
        say(&format!("video {} failed: {msg}", self.id));
    }
}

pub struct VideoGen {
    pub exe: String,
    pub low: String,
    high: String,
    t5: String,
    vae: String,
    loras: Option<(String, String, String)>,
    defaults: Defaults,
    pub need_gb: f64,
    timeout_s: f64,
    pub keep_s: f64,
    queue_s: f64,
    args: Vec<String>,
    jobs: Mutex<indexmap::IndexMap<String, Arc<Job>>>,
    ffmpeg: Option<String>,
    mp4: Mutex<Option<Vec<String>>>,
}

pub const MODEL: &str = "wan2.2-t2v-a14b";

impl VideoGen {
    pub fn from_config(cfg: &Value) -> Result<Option<VideoGen>, String> {
        if cfg.is_null() {
            return Ok(None);
        }
        if !cfg.is_object() {
            return Err(r#""video" must be {"exe": ..., "low": ..., "high": ..., "t5": ..., "vae": ...}"#.into());
        }
        let mut files = vec![];
        for k in ["exe", "low", "high", "t5", "vae"] {
            let p = get(cfg, k).as_str().map(expand).filter(|p| p.is_file()).ok_or(format!("\"video\".{k} must name an existing file"))?;
            files.push(std::fs::canonicalize(&p).unwrap_or(p).to_string_lossy().into_owned());
        }
        let mut nums = vec![];
        for (k, d) in [("need_gb", 40.0), ("timeout_s", 3600.0), ("keep_s", 3600.0), ("queue_s", 3600.0)] {
            let v = match cfg.get(k) {
                None => Some(d),
                Some(v) if v.is_number() => v.as_f64(),
                _ => None,
            };
            nums.push(v.filter(|v| v.is_finite() && *v > 0.0).ok_or(format!("\"video\".{k} must be a positive number"))?);
        }
        let args: Vec<String> = match cfg.get("args") {
            None => vec!["--diffusion-fa".into()],
            Some(Value::Array(a)) if a.iter().all(Value::is_string) => a.iter().map(|x| x.as_str().unwrap().to_string()).collect(),
            _ => return Err(r#""video".args must be a list of strings"#.into()),
        };
        let loras = match cfg.get("loras") {
            None | Some(Value::Null) => None,
            Some(l) => {
                let dir = l.get("dir").and_then(|d| d.as_str()).map(expand).filter(|d| d.is_dir()).ok_or(r#""video".loras must be {"dir": <a folder>, "low": <name>, "high": <name>}"#)?;
                let d = std::fs::canonicalize(&dir).unwrap_or(dir);
                let re = Regex::new(r"^[A-Za-z0-9._-]+$").unwrap();
                let mut names = vec![];
                for k in ["low", "high"] {
                    let n = l.get(k).and_then(|n| n.as_str()).filter(|n| re.is_match(n) && d.join(format!("{n}.safetensors")).is_file()).ok_or(format!("\"video\".loras.{k} must name a .safetensors file in {} (without \".safetensors\")", d.display()))?;
                    names.push(n.to_string());
                }
                Some((d.to_string_lossy().into_owned(), names[0].clone(), names[1].clone()))
            }
        };
        let ffmpeg = std::env::var("PATH").unwrap_or_default().split(':').map(|d| std::path::Path::new(d).join("ffmpeg")).find(|p| p.is_file()).map(|p| p.to_string_lossy().into_owned());
        Ok(Some(VideoGen {
            exe: files[0].clone(),
            low: files[1].clone(),
            high: files[2].clone(),
            t5: files[3].clone(),
            vae: files[4].clone(),
            defaults: if loras.is_some() { LIGHTNING } else { PLAIN },
            loras,
            need_gb: nums[0],
            timeout_s: nums[1],
            keep_s: nums[2],
            queue_s: nums[3],
            args,
            jobs: Mutex::new(indexmap::IndexMap::new()),
            ffmpeg,
            mp4: Mutex::new(None),
        }))
    }

    /// Forget expired clips (their files go too).
    fn sweep(&self) {
        let now = crate::sync::now() as i64;
        let mut jobs = self.jobs.lock().unwrap();
        let old: Vec<String> = jobs.values().filter(|j| j.completed_at.lock().unwrap().map(|c| (now - c) as f64 > self.keep_s).unwrap_or(false)).map(|j| j.id.clone()).collect();
        for id in old {
            if let Some(j) = jobs.shift_remove(&id) {
                let _ = std::fs::remove_dir_all(&j.dir);
            }
        }
    }

    /// (status, body): 202 with the job, or 400 / 429.
    pub fn create(self: &Arc<Self>, svc: Arc<Service>, req: &Value) -> Result<(u16, Value), String> {
        let p = parse_request(req, self.defaults)?;
        self.sweep();
        let job = {
            let mut jobs = self.jobs.lock().unwrap();
            if jobs.len() >= MAX_JOBS {
                return Ok((429, json!({"error": {"type": "rate_limit_error", "message": format!("at most {MAX_JOBS} videos are kept: DELETE /v1/videos/{{id}} the finished ones, or wait for them to expire")}})));
            }
            let dir = private_dir("strata-video-").map_err(|e| e.to_string())?;
            let job = Arc::new(Job {
                id: format!("video_{}", uuid::Uuid::new_v4().simple()),
                p,
                status: Mutex::new("queued".into()),
                progress: AtomicI64::new(0),
                error: Mutex::new(None),
                created_at: crate::sync::now() as i64,
                completed_at: Mutex::new(None),
                cancel: Event::new(),
                proc: Mutex::new(None),
                file: Mutex::new(dir.join("out.webm")),
                dir,
            });
            jobs.insert(job.id.clone(), job.clone());
            job
        };
        let view = job.view(MODEL, self.keep_s);
        let me = self.clone();
        let j = job.clone();
        let spawned = std::thread::Builder::new().name(job.id.clone()).spawn(move || me.run(&svc, &j));
        if let Err(e) = spawned {
            self.jobs.lock().unwrap().shift_remove(&job.id);
            let _ = std::fs::remove_dir_all(&job.dir);
            return Ok((503, json!({"error": {"type": "server_error", "message": format!("the video could not start ({e})")}})));
        }
        Ok((202, view))
    }

    pub fn get(&self, id: &str) -> Option<Arc<Job>> {
        self.sweep();
        self.jobs.lock().unwrap().get(id).cloned()
    }

    pub fn list(&self) -> Vec<Value> {
        self.sweep();
        let mut jobs: Vec<Arc<Job>> = self.jobs.lock().unwrap().values().cloned().collect();
        jobs.sort_by_key(|j| j.created_at);
        jobs.iter().map(|j| j.view(MODEL, self.keep_s)).collect()
    }

    pub fn delete(&self, id: &str) -> bool {
        let Some(job) = self.jobs.lock().unwrap().shift_remove(id) else { return false };
        job.cancel.set();
        let st = job.status.lock().unwrap().clone();
        if st == "completed" || st == "failed" {
            let _ = std::fs::remove_dir_all(&job.dir);
        }
        true
    }

    /// At the server's exit: every job cancelled, its sd-cli stopped, its folder removed.
    pub fn shutdown(&self) {
        let jobs: Vec<Arc<Job>> = self.jobs.lock().unwrap().drain(..).map(|(_, j)| j).collect();
        for j in jobs {
            j.cancel.set();
            if let Some(pid) = *j.proc.lock().unwrap() {
                unsafe { libc::killpg(pid as i32, libc::SIGTERM) };
            }
            let _ = std::fs::remove_dir_all(&j.dir);
        }
    }

    fn run(&self, svc: &Service, job: &Job) {
        self.run_turn(svc, job);
        let st = job.status.lock().unwrap().clone();
        if st == "queued" || st == "in_progress" {
            job.fail("ended without a result");
        }
        let gone = !self.jobs.lock().unwrap().contains_key(&job.id);
        if gone || *job.status.lock().unwrap() == "failed" {
            let _ = std::fs::remove_dir_all(&job.dir);
        }
    }

    fn run_turn(&self, svc: &Service, job: &Job) {
        let deadline = Instant::now() + Duration::from_secs_f64(self.queue_s);
        let mut got = false;
        while !got && !job.cancel.is_set() && Instant::now() < deadline {
            got = svc.fifo.acquire_timeout(Duration::from_secs(1));
        }
        if !got {
            job.fail(&if job.cancel.is_set() { "cancelled before it started".into() } else { format!("no turn within {:.0} s", self.queue_s) });
            return;
        }
        svc.image_gate.set();
        let room = make_room(svc, self.need_gb, Instant::now() + Duration::from_secs(600), &job.cancel);
        match room {
            "cancelled" => job.fail("cancelled before it started"),
            "enough" | "unloaded" => {
                *job.status.lock().unwrap() = "in_progress".into();
                job.progress.store(1, SeqCst);
                self.render(job);
            }
            r => job.fail(&format!("memory: {r} ({} GB free, {:.0} GB needed)", available_gb().map(crate::pyjson::float_repr).unwrap_or_else(|| "None".into()), self.need_gb)),
        }
        svc.image_gate.clear();
        svc.fifo.release();
    }

    fn command(&self, job: &Job) -> Vec<String> {
        let (p, d) = (&job.p, &job.dir);
        let mut prompt = p.prompt.clone();
        let mut lora = vec![];
        if let Some((dir, low, high)) = &self.loras {
            prompt += &format!("<lora:{low}.safetensors:1><lora:|high_noise|{high}.safetensors:1>");
            lora = vec!["--lora-model-dir".into(), dir.clone()];
        }
        let _ = std::fs::write(d.join("prompt.txt"), prompt);
        let _ = std::fs::write(d.join("negative.txt"), &p.negative);
        let fr = crate::pyjson::float_repr;
        let s = |x: &std::path::Path| x.to_string_lossy().into_owned();
        let mut c = vec![self.exe.clone()];
        c.extend(lora);
        c.extend(["-M", "vid_gen", "--diffusion-model"].map(String::from));
        c.push(self.low.clone());
        c.push("--high-noise-diffusion-model".into());
        c.push(self.high.clone());
        c.extend(["--t5xxl".into(), self.t5.clone(), "--vae".into(), self.vae.clone(), "--prompt-file".into(), s(&d.join("prompt.txt")), "--negative-prompt-file".into(), s(&d.join("negative.txt")), "--cfg-scale".into(), fr(p.cfg), "--high-noise-cfg-scale".into(), fr(p.cfg), "--sampling-method".into(), "euler".into(), "--high-noise-sampling-method".into(), "euler".into(), "--steps".into(), p.steps.to_string(), "--high-noise-steps".into(), p.high_noise_steps.to_string(), "-W".into(), p.width.to_string(), "-H".into(), p.height.to_string(), "--video-frames".into(), p.frames.to_string(), "--fps".into(), FPS.to_string(), "--seed".into(), p.seed.to_string(), "--disable-image-metadata".into()]);
        c.extend(self.args.iter().cloned());
        c.push("-o".into());
        c.push(s(&job.file.lock().unwrap()));
        c
    }

    fn render(&self, job: &Job) {
        let total = job.p.steps + job.p.high_noise_steps;
        let t0 = Instant::now();
        let cmd = self.command(job);
        use std::os::unix::process::CommandExt;
        let mut child = match Command::new(&cmd[0]).args(&cmd[1..]).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).process_group(0).spawn() {
            Ok(c) => c,
            Err(e) => return job.fail(&format!("sd-cli could not start ({e})")),
        };
        *job.proc.lock().unwrap() = Some(child.id());
        let log_path = job.dir.join("log.txt");
        let (out, err) = (child.stdout.take().unwrap(), child.stderr.take().unwrap());
        let progress = Arc::new(AtomicI64::new(1));
        let reader = {
            let progress = progress.clone();
            let log_path = log_path.clone();
            std::thread::spawn(move || {
                let step = Regex::new(r"\|\s*(\d+)/(\d+)\s*-\s*[\d.]+\s*(?:s/it|it/s)").unwrap();
                let mut log = std::fs::File::create(&log_path).ok();
                // stdout and stderr together, as stderr=STDOUT
                let (tx, rx) = crossbeam_channel::unbounded::<Vec<u8>>();
                for mut src in [Box::new(out) as Box<dyn Read + Send>, Box::new(err)] {
                    let tx = tx.clone();
                    std::thread::spawn(move || {
                        let mut b = [0u8; 4096];
                        while let Ok(n) = src.read(&mut b) {
                            if n == 0 || tx.send(b[..n].to_vec()).is_err() {
                                break;
                            }
                        }
                    });
                }
                drop(tx);
                let (mut done, mut last): (i64, Option<i64>) = (0, None);
                let mut buf: Vec<u8> = vec![];
                for chunk in rx {
                    if let Some(l) = log.as_mut() {
                        use std::io::Write;
                        let _ = l.write_all(&chunk);
                    }
                    buf.extend(chunk);
                    while let Some(i) = buf.iter().position(|b| *b == b'\r' || *b == b'\n') {
                        let line = String::from_utf8_lossy(&buf[..i]).into_owned();
                        buf.drain(..=i);
                        if let Some(m) = step.captures(&line) {
                            let k: i64 = m[1].parse().unwrap_or(0);
                            if let Some(l) = last {
                                if k < l {
                                    done += l;
                                }
                            }
                            last = Some(k);
                            progress.store((2 + 90 * (done + k) / total.max(1)).min(95), SeqCst);
                        }
                    }
                }
            })
        };
        let mut why = None;
        while matches!(child.try_wait(), Ok(None)) {
            job.progress.store(progress.load(SeqCst), SeqCst);
            if job.cancel.is_set() || t0.elapsed().as_secs_f64() >= self.timeout_s {
                why = Some(if job.cancel.is_set() { "cancelled".to_string() } else { format!("longer than {:.0} s", self.timeout_s) });
                break;
            }
            job.cancel.wait(Some(Duration::from_millis(500)));
        }
        if matches!(child.try_wait(), Ok(None)) {
            stop(&mut child);
        }
        let status = child.wait().ok();
        let _ = reader.join();
        *job.proc.lock().unwrap() = None;
        if let Some(w) = why {
            return job.fail(&w);
        }
        let logb = std::fs::read(&log_path).unwrap_or_default();
        if self.loras.is_some() && String::from_utf8_lossy(&logb).contains("can not found lora") {
            return job.fail("a LoRA file is missing: sd-cli ran without it (a 4-step clip without Lightning is noise)");
        }
        let code = status.and_then(|s| s.code());
        if code != Some(0) || !job.file.lock().unwrap().is_file() {
            let tail = String::from_utf8_lossy(&logb[logb.len().saturating_sub(400)..]).replace('\r', "\n").trim().to_string();
            let tail: String = tail.chars().rev().take(300).collect::<Vec<_>>().into_iter().rev().collect();
            return job.fail(&format!("sd-cli ended with status {}: {tail}", code.map(|c| c.to_string()).unwrap_or_else(|| "None".into())));
        }
        self.to_mp4(job);
        *job.status.lock().unwrap() = "completed".into();
        job.progress.store(100, SeqCst);
        *job.completed_at.lock().unwrap() = Some(crate::sync::now() as i64);
        say(&format!("video {}: {}x{}, {} frames, {:.0} s", job.id, job.p.width, job.p.height, job.p.frames, t0.elapsed().as_secs_f64()));
    }

    /// The clip as H.264 MP4 (QuickTime plays it); on any failure the webm stays.
    fn to_mp4(&self, job: &Job) {
        let Some(ff) = &self.ffmpeg else { return };
        let mp4 = {
            let mut m = self.mp4.lock().unwrap();
            if m.is_none() {
                let enc = Command::new(ff).args(["-hide_banner", "-encoders"]).output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
                *m = Some(if enc.contains(" libx264 ") {
                    ["-c:v", "libx264", "-crf", "18", "-preset", "medium"].map(String::from).to_vec()
                } else if enc.contains(" h264_videotoolbox ") {
                    ["-c:v", "h264_videotoolbox", "-b:v", "6M"].map(String::from).to_vec()
                } else {
                    vec![]
                });
            }
            m.clone().unwrap()
        };
        if mp4.is_empty() {
            return;
        }
        let out = job.dir.join("out.mp4");
        let src = job.file.lock().unwrap().clone();
        let r = Command::new(ff).args(["-nostdin", "-v", "error", "-y", "-i"]).arg(&src).args(&mp4).args(["-pix_fmt", "yuv420p", "-movflags", "+faststart"]).arg(&out).output();
        match r {
            Ok(o) if o.status.success() && out.metadata().map(|m| m.len() > 0).unwrap_or(false) => {
                let _ = std::fs::remove_file(&src);
                *job.file.lock().unwrap() = out;
            }
            Ok(o) => {
                let e = String::from_utf8_lossy(&o.stderr).trim().to_string();
                let tail: String = e.chars().rev().take(200).collect::<Vec<_>>().into_iter().rev().collect();
                say(&format!("video {}: kept as webm (ffmpeg: {tail})", job.id));
            }
            Err(e) => say(&format!("video {}: kept as webm ({e})", job.id)),
        }
    }
}

pub fn job_file(job: &Job) -> PathBuf {
    job.file.lock().unwrap().clone()
}

pub fn status_of(job: &Job) -> String {
    job.status.lock().unwrap().clone()
}
