//! The resident image encoder (serve/server.py's Vision): `strata-vision` reads `ENC <image> <out>` lines and writes
//! each image's embeddings; results are cached by the image's hash (64 kept), so a conversation that sends the same
//! picture every turn encodes it once.

use crate::engine::{timeout_env, wait_timeout};
use crate::py;
use base64::Engine as _;
use sha2::Digest;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

/// an image URL is read up to this (the web app attaches pictures of up to 20 MB)
pub const IMAGE_URL_MAX: usize = 32 << 20;

pub enum ImgSrc {
    Str(String),
    Bytes(Vec<u8>),
}

struct Proc {
    child: Child,
    stdin: ChildStdin,
    lines: crossbeam_channel::Receiver<Option<String>>,
}

pub struct Vision {
    args: Vec<String>,
    log: Option<String>,
    env: std::collections::HashMap<String, String>,
    pub dir: PathBuf,
    proc: Mutex<Option<Proc>>,
    /// key -> (embeddings file, image tokens), most recently used last
    cache: Mutex<indexmap::IndexMap<String, (PathBuf, i64)>>,
}

/// #480: a work directory whose path has no space (the ENC and GENI lines are split at spaces).
fn work_dir() -> PathBuf {
    let bases = [std::env::temp_dir(), std::env::current_dir().unwrap_or_default()];
    for base in bases {
        let d = base.join(format!("strata-vision-{}", &uuid::Uuid::new_v4().simple().to_string()[..8]));
        if std::fs::create_dir(&d).is_ok() {
            if !d.to_string_lossy().contains(' ') {
                return d;
            }
            let _ = std::fs::remove_dir(&d);
        }
    }
    let d = std::env::temp_dir().join(format!("strata-vision-{}", &uuid::Uuid::new_v4().simple().to_string()[..8]));
    let _ = std::fs::create_dir_all(&d);
    d
}

/// #874: where one request's combined embeddings file goes (the encoder's own directory on a Mac: no /dev/shm).
pub fn combined_embeddings_path(dir: &Path, _nbytes: u64) -> PathBuf {
    dir.join(format!("req-{}.sve", &uuid::Uuid::new_v4().simple().to_string()[..12]))
}

/// The file written private to this user (0600).
pub fn write_private(path: &Path, data: &[u8]) -> std::io::Result<()> {
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(path)?;
    f.write_all(data)
}

/// The message when the image encoder answered something other than READY.
fn vision_start_error(line: &str) -> String {
    format!("the vision encoder did not start: {}", py::strip(line))
}

impl Vision {
    pub fn new(cfg: &serde_json::Value, log: Option<String>, env: std::collections::HashMap<String, String>, lazy: bool) -> Result<Self, String> {
        let s = |k: &str| py::py_str(py::get(cfg, k));
        let absolute = |p: String| if Path::new(&p).parent().map(|d| !d.as_os_str().is_empty()).unwrap_or(false) { std::path::absolute(&p).map(|x| x.to_string_lossy().into_owned()).unwrap_or(p) } else { p };
        let mut args = vec![absolute(s("exe")), "--mmproj".into(), absolute(s("mmproj")), "--model".into(), absolute(s("model"))];
        if py::truthy(py::get(cfg, "gpu")) {
            args.push("--gpu".into());
        }
        for (k, flag) in [("threads", "--threads"), ("max_tokens", "--max-tokens"), ("min_tokens", "--min-tokens")] {
            if py::truthy(py::get(cfg, k)) {
                args.push(flag.into());
                args.push(s(k));
            }
        }
        let v = Vision { args, log, env, dir: work_dir(), proc: Mutex::new(None), cache: Mutex::new(indexmap::IndexMap::new()) };
        if !lazy {
            v.start()?;
        }
        Ok(v)
    }

    fn start(&self) -> Result<(), String> {
        let stderr = match &self.log {
            Some(l) => std::fs::OpenOptions::new().create(true).append(true).open(l).map(Stdio::from).unwrap_or(Stdio::null()),
            None => Stdio::null(),
        };
        let mut child = Command::new(&self.args[0])
            .args(&self.args[1..])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(stderr)
            .env_clear()
            .envs(&self.env)
            .current_dir(&self.dir)
            .spawn()
            .map_err(|e| format!("the vision encoder did not start: {e}"))?;
        let stdin = child.stdin.take().expect("stdin");
        let stdout = child.stdout.take().expect("stdout");
        let lines = pump(stdout);
        let ready = timeout_env("STRATA_VISION_READY_S", 300.0);
        let line = read_line(&lines, ready);
        match line {
            None => {
                let _ = child.kill();
                return Err(format!("the vision encoder did not start: it said nothing for {:.0} s", ready.unwrap_or(0.0)));
            }
            Some(l) if !l.starts_with("READY") => {
                let _ = child.kill();
                wait_timeout(&mut child, 5.0);
                return Err(vision_start_error(&l));
            }
            _ => {}
        }
        *self.proc.lock().unwrap() = Some(Proc { child, stdin, lines });
        Ok(())
    }

    pub fn alive(&self) -> bool {
        self.proc.lock().unwrap().as_mut().map(|p| matches!(p.child.try_wait(), Ok(None))).unwrap_or(false)
    }

    /// Stop the encoder (its memory goes back); the encoded images stay cached on disk.
    pub fn unload(&self) {
        self.close();
    }

    pub fn restart(&self) -> Result<(), String> {
        if let Some(mut p) = self.proc.lock().unwrap().take() {
            let _ = p.child.kill();
        }
        self.start()
    }

    pub fn load(source: &str) -> Result<Vec<u8>, String> {
        if source.starts_with("data:") {
            let Some((_, b64)) = source.split_once(',') else { return Err("the image's data: URL could not be read (no comma)".into()) };
            return base64::engine::general_purpose::STANDARD.decode(b64.trim()).map_err(|e| format!("the image's data: URL could not be read ({e})"));
        }
        if source.starts_with("http://") || source.starts_with("https://") {
            return Self::download(source);
        }
        let path = source.strip_prefix("file://").unwrap_or(source);
        let host = if source.starts_with("file://") { path.split(['/', '\\']).next().unwrap_or("") } else { "" };
        let network = path.replace('/', "\\").starts_with("\\\\") || path.starts_with("\\??\\");
        let host_ok = host.is_empty() || host.eq_ignore_ascii_case("localhost") || (host.len() == 2 && host.ends_with(':') && host.chars().next().unwrap().is_ascii_alphabetic());
        if network || !host_ok {
            return Err("an image file must be on this computer: network paths (\\\\host\\share, //host/share, file://host/...) are not read".into());
        }
        if !path.is_empty() && Path::new(path).is_file() {
            return std::fs::read(path).map_err(|e| format!("the image could not be read from {path} ({e})"));
        }
        Err("an image must be a data: URL, an http(s) URL or a local file path".into())
    }

    /// An image URL's bytes, at most IMAGE_URL_MAX of them.
    pub fn download(url: &str) -> Result<Vec<u8>, String> {
        let too_big = format!("the image URL's file is over {} MiB", IMAGE_URL_MAX >> 20);
        let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(60))).build().into();
        let r = agent.get(url).header("User-Agent", "strata").call().map_err(|e| format!("the image URL could not be read: {e}"))?;
        if let Some(n) = r.headers().get("Content-Length").and_then(|v| v.to_str().ok()).and_then(|v| v.parse::<usize>().ok()) {
            if n > IMAGE_URL_MAX {
                return Err(too_big);
            }
        }
        let mut data = vec![];
        use std::io::Read;
        r.into_body().into_reader().take(IMAGE_URL_MAX as u64 + 1).read_to_end(&mut data).map_err(|e| format!("the image URL could not be read: {e}"))?;
        if data.len() > IMAGE_URL_MAX {
            return Err(too_big);
        }
        Ok(data)
    }

    fn exif_orientation(data: &[u8]) -> u32 {
        exif::Reader::new()
            .read_from_container(&mut std::io::Cursor::new(data))
            .ok()
            .and_then(|e| e.get_field(exif::Tag::Orientation, exif::In::PRIMARY).and_then(|f| f.value.get_uint(0)))
            .unwrap_or(1)
    }

    /// JPEG/PNG/BMP/GIF pass through (a JPEG turned by its EXIF Orientation is turned upright first); anything else is
    /// converted to PNG, transparent areas white.
    pub fn normalize(data: Vec<u8>) -> Result<Vec<u8>, String> {
        let jpeg = data.starts_with(&[0xff, 0xd8, 0xff]);
        let orient = if jpeg { Self::exif_orientation(&data) } else { 1 };
        if (jpeg && matches!(orient, 0 | 1)) || data.starts_with(b"\x89PNG\r\n\x1a\n") || data.starts_with(b"BM") || data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
            return Ok(data);
        }
        let mut im = image::load_from_memory(&data).map_err(|e| format!("the image could not be read ({e})"))?;
        let orient = if jpeg { orient } else { Self::exif_orientation(&data) };
        im = match orient {
            2 => im.fliph(),
            3 => im.rotate180(),
            4 => im.flipv(),
            5 => im.rotate90().fliph(),
            6 => im.rotate90(),
            7 => im.rotate270().fliph(),
            8 => im.rotate270(),
            _ => im,
        };
        let rgb = if im.color().has_alpha() {
            let rgba = im.to_rgba8();
            let mut out = image::RgbImage::new(rgba.width(), rgba.height());
            for (x, y, p) in rgba.enumerate_pixels() {
                let a = p[3] as u32;
                let blend = |c: u8| ((c as u32 * a + 255 * (255 - a)) / 255) as u8;
                out.put_pixel(x, y, image::Rgb([blend(p[0]), blend(p[1]), blend(p[2])]));
            }
            out
        } else {
            im.to_rgb8()
        };
        let mut out = std::io::Cursor::new(vec![]);
        rgb.write_to(&mut out, image::ImageFormat::Png).map_err(|e| format!("the image could not be read ({e})"))?;
        Ok(out.into_inner())
    }

    /// encode() for every image of one request (#1072: this request's files are never evicted for a later one).
    pub fn encode_all(&self, sources: &[ImgSrc]) -> Result<Vec<(PathBuf, i64)>, String> {
        let mut out: Vec<(PathBuf, i64)> = vec![];
        for s in sources {
            let keep: Vec<PathBuf> = out.iter().map(|(p, _)| p.clone()).collect();
            out.push(self.encode(s, &keep)?);
        }
        Ok(out)
    }

    /// -> (embeddings file, number of image tokens).
    pub fn encode(&self, source: &ImgSrc, keep: &[PathBuf]) -> Result<(PathBuf, i64), String> {
        let raw = match source {
            ImgSrc::Bytes(b) => b.clone(),
            ImgSrc::Str(s) => Self::load(s)?,
        };
        let data = Self::normalize(raw)?;
        let key = hex::encode(sha2::Sha256::digest(&data))[..32].to_string();
        let mut cache = self.cache.lock().unwrap();
        if let Some(v) = cache.shift_remove(&key) {
            cache.insert(key.clone(), v.clone());
            return Ok(v);
        }
        let img = self.dir.join(format!("{key}.img"));
        let out = self.dir.join(format!("{key}.sve"));
        std::fs::write(&img, &data).map_err(|e| format!("the image could not be read ({e})"))?;
        let line = {
            let mut p = self.proc.lock().unwrap();
            let r = match p.as_mut() {
                None => Err("the image could not be read: the vision encoder stopped".to_string()),
                Some(pr) => {
                    if writeln!(pr.stdin, "ENC {key}.img {key}.sve").and_then(|_| pr.stdin.flush()).is_err() {
                        Err("the image could not be read: the vision encoder stopped".to_string())
                    } else {
                        let t = timeout_env("STRATA_VISION_ENCODE_S", 300.0);
                        match read_line(&pr.lines, t) {
                            Some(l) => Ok(py::strip(&l).to_string()),
                            None => {
                                let _ = pr.child.kill();
                                Err(format!("the image could not be read: it said nothing for {:.0} s", t.unwrap_or(0.0)))
                            }
                        }
                    }
                }
            };
            if r.is_err() {
                *p = None;
            }
            r
        };
        let _ = std::fs::remove_file(&img);
        let line = line?;
        if !line.starts_with("OK") {
            return Err(format!("the image could not be read: {}", if let Some(r) = line.strip_prefix("ERR") { py::strip(r).to_string() } else { "the vision encoder stopped".into() }));
        }
        let n: i64 = py::split_ws(&line).get(1).and_then(|v| v.parse().ok()).unwrap_or(0);
        cache.insert(key.clone(), (out.clone(), n));
        let held: Vec<PathBuf> = keep.iter().cloned().chain(std::iter::once(out.clone())).collect();
        let keys: Vec<String> = cache.keys().cloned().collect();
        for old in keys {
            if cache.len() <= 64 {
                break;
            }
            if !held.contains(&cache[&old].0) {
                if let Some((p, _)) = cache.shift_remove(&old) {
                    let _ = std::fs::remove_file(p);
                }
            }
        }
        Ok((out, n))
    }

    pub fn close(&self) {
        if let Some(mut p) = self.proc.lock().unwrap().take() {
            let ok = writeln!(p.stdin, "QUIT").and_then(|_| p.stdin.flush()).is_ok();
            if !ok || !wait_timeout(&mut p.child, 10.0) {
                let _ = p.child.kill();
                wait_timeout(&mut p.child, 10.0);
            }
        }
    }

    /// #914: close() for good: the encoder's directory goes with the server.
    pub fn shutdown(&self) {
        self.close();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn pump(stdout: ChildStdout) -> crossbeam_channel::Receiver<Option<String>> {
    let (tx, rx) = crossbeam_channel::unbounded();
    std::thread::spawn(move || {
        let mut r = BufReader::new(stdout);
        let mut b = vec![];
        loop {
            b.clear();
            match r.read_until(b'\n', &mut b) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    let _ = tx.send(Some(String::from_utf8_lossy(&b).trim_end_matches(['\n', '\r']).to_string()));
                }
            }
        }
        let _ = tx.send(None);
    });
    rx
}

/// One line within `timeout` (None: forever); None when it said nothing in time (or its output ended: "").
fn read_line(rx: &crossbeam_channel::Receiver<Option<String>>, timeout: Option<f64>) -> Option<String> {
    let got = match timeout {
        Some(t) => rx.recv_timeout(Duration::from_secs_f64(t)).ok()?,
        None => rx.recv().ok()?,
    };
    Some(got.unwrap_or_default())
}
