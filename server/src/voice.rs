//! "Hey Strata" (macOS 26+): engine-metal/strata-voice (metal/voice/strata_voice.swift) as this server's child.  It
//! listens with Apple's on-device dictation, sends what follows the wake phrase to this server's chat and reads the
//! answer aloud; its JSON events (listening, wake, command, answer, error) are kept here for the chat page's voice
//! panel (GET /voice).  It needs no model of Apple's, and no Apple Intelligence.
//!
//! An open chat page keeps the conversation: it checks in (GET /voice?page=1, every 1.5 s while voice is on), and
//! a question heard while it does is the page's to send through its chat (the helper is told {"ask": "page"}; the
//! page sends the answer back with POST /voice {"say": ...}).  With no page, the helper asks the model itself.
use crate::say;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::io::Write;
use std::process::ChildStdin;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

const KEEP: usize = 40;

#[derive(Default)]
struct State {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    events: VecDeque<Value>,
    /// events so far (the page takes each command once: by its number)
    seq: u64,
    page_seen: Option<Instant>,
    /// what the recognizer heard last (the panel shows it: a wake phrase heard as another word is seen there)
    heard: String,
}

/// A chat page checked in this recently: it takes the next question.
const PAGE_FRESH: Duration = Duration::from_secs(4);

fn tell(s: &mut State, v: Value) {
    if let Some(w) = s.stdin.as_mut() {
        let _ = writeln!(w, "{v}").and_then(|_| w.flush());
    }
}

static STATE: LazyLock<Mutex<State>> = LazyLock::new(Default::default);

/// The wake phrase and whether voice is on, kept in the Strata folder (.strata-voice.json) across restarts.
fn phrase_path() -> PathBuf {
    crate::root().join(".strata-voice.json")
}

fn saved() -> Value {
    std::fs::read_to_string(phrase_path()).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()).unwrap_or(json!({}))
}

fn save(key: &str, v: Value) -> Result<(), String> {
    let mut all = saved();
    all[key] = v;
    std::fs::write(phrase_path(), all.to_string()).map_err(|e| e.to_string())
}

pub fn phrase() -> String {
    saved()["phrase"].as_str().map(String::from).filter(|p| !p.trim().is_empty()).unwrap_or_else(|| "hey strata".into())
}

/// At the server's start: voice back on when it was on when the server last ran.
pub fn autostart(port: u16, key: Option<&str>) {
    if saved()["on"] == json!(true) && cfg!(target_os = "macos") && exe().is_file() {
        if let Err(e) = set(true, port, key) {
            say(&format!("voice: {e}"));
        }
    }
}

/// A wake phrase: 1 to 4 words of letters (the last one is what wakes it), at most 40 characters.
pub fn set_phrase(p: &str) -> Result<(), String> {
    let p = p.trim().to_lowercase();
    let words = p.split_whitespace().count();
    if !(1..=4).contains(&words) || p.len() > 40 || !p.chars().all(|c| c.is_alphabetic() || c == ' ' || c == '\'') {
        return Err("a wake phrase is 1 to 4 words of letters, like \"hey strata\" or \"computer\"".into());
    }
    save("phrase", json!(p))
}

fn exe() -> PathBuf {
    crate::root().join("engine-metal/strata-voice")
}

fn push(s: &mut State, mut v: Value) {
    if s.events.len() == KEEP {
        s.events.pop_front();
    }
    s.seq += 1;
    v["n"] = json!(s.seq);
    s.events.push_back(v);
}

/// GET /voice: whether it can run here, whether it listens, and its recent events.  `page`: a chat page asks (it
/// takes the questions while it keeps asking).
pub fn status_for(page: bool) -> Value {
    if page {
        STATE.lock().unwrap().page_seen = Some(Instant::now());
    }
    status()
}

pub fn status() -> Value {
    let mut s = STATE.lock().unwrap();
    let on = s.child.as_mut().is_some_and(|c| matches!(c.try_wait(), Ok(None)));
    if !on {
        s.child = None;
    }
    json!({"available": cfg!(target_os = "macos") && exe().is_file(), "on": on, "phrase": phrase(), "heard": s.heard, "events": s.events})
}

/// POST /voice {"on": true|false}: start or stop listening; the helper asks this server (127.0.0.1:`port`).
pub fn set(on: bool, port: u16, key: Option<&str>) -> Result<Value, String> {
    let _ = save("on", json!(on));
    let mut s = STATE.lock().unwrap();
    s.stdin = None;
    if let Some(mut c) = s.child.take() {
        let _ = c.kill();
        let _ = c.wait();
    }
    if on {
        if !exe().is_file() {
            return Err("engine-metal/strata-voice is missing: make build compiles it on macOS 26 or later".into());
        }
        let mut cmd = Command::new(exe());
        cmd.args(["--port", &port.to_string(), "--phrase", &phrase(), "--stdin-exit", "--verbose"]).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null());
        if let Some(k) = key {
            cmd.env("STRATA_API_KEY", k); // not on the command line, where ps shows it
        }
        let mut child = cmd.spawn().map_err(|e| format!("strata-voice: {e}"))?;
        let out = child.stdout.take().unwrap();
        s.stdin = child.stdin.take();
        std::thread::spawn(move || {
            for line in BufReader::new(out).lines().map_while(Result::ok) {
                let Ok(mut v) = serde_json::from_str::<Value>(&line) else { continue };
                let mut st = STATE.lock().unwrap();
                if v["event"] == "heard" {
                    st.heard = v["text"].as_str().unwrap_or("").to_string();
                    continue;
                }
                if v["event"] == "command" {
                    let page = st.page_seen.is_some_and(|t| t.elapsed() < PAGE_FRESH);
                    let by = if page { "page" } else { "self" };
                    tell(&mut st, json!({"ask": by}));
                    v["by"] = json!(by);
                }
                drop(st);
                match v["event"].as_str() {
                    Some("command") => say(&format!("voice: {}", v["text"].as_str().unwrap_or(""))),
                    Some("error") => say(&format!("voice: {}", v["text"].as_str().unwrap_or(""))),
                    Some("listening") => say(&format!("voice: listening for {:?}", v["phrase"].as_str().unwrap_or(""))),
                    _ => {}
                }
                push(&mut STATE.lock().unwrap(), v);
            }
        });
        s.child = Some(child);
        s.events.clear(); // a new start: the last one's errors are not this one's
        push(&mut s, json!({"event": "starting"}));
    } else {
        push(&mut s, json!({"event": "off"}));
    }
    drop(s);
    Ok(status())
}

/// POST /voice {"permissions": true}: macOS's prompts for the microphone and speech recognition, for the app that
/// started this server (a refused one opens its System Settings page instead).  Waits for the person's answer.
pub fn permissions() -> Value {
    let out = Command::new(exe()).arg("--permissions").stdin(Stdio::null()).output();
    let v = out.ok().and_then(|o| String::from_utf8_lossy(&o.stdout).lines().rev().find_map(|l| serde_json::from_str::<Value>(l).ok()))
        .unwrap_or(json!({"event": "permissions", "error": "strata-voice did not answer"}));
    push(&mut STATE.lock().unwrap(), v.clone());
    v
}

#[cfg(test)]
mod tests {
    #[test]
    fn wake_phrases_are_short_words() {
        for bad in ["", "   ", "one two three four five", "rm -rf /", "hey; strata", &"a".repeat(41)] {
            assert!(super::set_phrase(bad).is_err(), "{bad:?}");
        }
    }
}

/// POST /voice {"say": text}: the chat page's answer to a question it took, read aloud ({"hush": true} stops it).
pub fn speak(text: &str, hush: bool) -> Value {
    let mut s = STATE.lock().unwrap();
    if hush {
        tell(&mut s, json!({"hush": true}));
    } else {
        tell(&mut s, json!({"say": text}));
        push(&mut s, json!({"event": "answer", "text": text, "by": "page"}));
    }
    drop(s);
    status()
}
