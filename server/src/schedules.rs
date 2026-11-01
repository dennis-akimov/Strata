//! Scheduled prompts (in the server, so they run whatever started it: Strata.app, the TUI,
//! `make run`): a prompt sent to this server's own chat at set times, its answers kept (the chat page's calendar
//! button shows them) and said in a macOS notification.  Kept in the Strata folder (.strata-schedules.json).
//!
//! When: once (at a local time), every N minutes (5 or more), or a cron expression; hourly, daily, weekdays, weekly
//! and monthly are written as cron.  Cron: minute hour day-of-month month day-of-week, with *, lists, ranges, */steps,
//! names (mon, jan) and 7 = Sunday; day-of-month and day-of-week both restricted: either matches (POSIX).
//! A run missed while the server was off or the Mac asleep runs once when it is back (launchd's way), not once per
//! missed time.
use crate::say;
use chrono::{DateTime, Datelike, Duration as CDuration, Local, NaiveDateTime, TimeZone, Timelike};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

// ------------------------------------------------------------------ cron
#[derive(Debug, Clone, PartialEq)]
pub struct Cron {
    min: Vec<bool>,
    hour: Vec<bool>,
    dom: Vec<bool>,
    mon: Vec<bool>,
    dow: Vec<bool>,
    dom_any: bool,
    dow_any: bool,
}

fn field(spec: &str, lo: u32, hi: u32, names: &[&str]) -> Result<(Vec<bool>, bool), String> {
    let mut set = vec![false; hi as usize + 1];
    let num = |t: &str| -> Result<u32, String> {
        let t = t.to_lowercase();
        if let Some(i) = names.iter().position(|n| t.starts_with(n)) {
            return Ok(lo + i as u32);
        }
        t.parse::<u32>().map_err(|_| format!("\"{t}\" is not a number"))
    };
    for part in spec.split(',') {
        let (range, step) = match part.split_once('/') {
            Some((r, s)) => (r, s.parse::<u32>().map_err(|_| format!("bad step in \"{part}\""))?.max(1)),
            None => (part, 1),
        };
        let (a, b) = if range == "*" {
            (lo, hi)
        } else if let Some((x, y)) = range.split_once('-') {
            (num(x)?, num(y)?)
        } else {
            let v = num(range)?;
            (v, if part.contains('/') { hi } else { v })
        };
        if a < lo || b > hi || a > b {
            return Err(format!("\"{part}\" is outside {lo}-{hi}"));
        }
        for v in (a..=b).step_by(step as usize) {
            set[v as usize] = true;
        }
    }
    Ok((set, spec == "*"))
}

impl Cron {
    pub fn parse(s: &str) -> Result<Cron, String> {
        let f: Vec<&str> = s.split_whitespace().collect();
        if f.len() != 5 {
            return Err("a cron expression has 5 fields: minute hour day-of-month month day-of-week".into());
        }
        let (min, _) = field(f[0], 0, 59, &[])?;
        let (hour, _) = field(f[1], 0, 23, &[])?;
        let (dom, dom_any) = field(f[2], 1, 31, &[])?;
        let (mon, _) = field(f[3], 1, 12, &["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])?;
        let (mut dow, dow_any) = field(f[4], 0, 7, &["sun", "mon", "tue", "wed", "thu", "fri", "sat"])?;
        if dow[7] {
            dow[0] = true; // 7 = Sunday
        }
        Ok(Cron { min, hour, dom, mon, dow, dom_any, dow_any })
    }

    fn day_ok(&self, t: &NaiveDateTime) -> bool {
        let (d, w) = (self.dom[t.day() as usize], self.dow[t.weekday().num_days_from_sunday() as usize]);
        match (self.dom_any, self.dow_any) {
            (false, false) => d || w, // both restricted: either (POSIX)
            (true, false) => w,
            (false, true) => d,
            (true, true) => true,
        }
    }

    /// The first minute after `t` this matches (within 4 years), in local time.
    pub fn next_after(&self, t: NaiveDateTime) -> Option<NaiveDateTime> {
        let mut c = t.with_second(0)?.with_nanosecond(0)? + CDuration::minutes(1);
        let end = t + CDuration::days(366 * 4);
        while c < end {
            if !self.mon[c.month() as usize] {
                // the next month's first minute
                let (y, m) = if c.month() == 12 { (c.year() + 1, 1) } else { (c.year(), c.month() + 1) };
                c = chrono::NaiveDate::from_ymd_opt(y, m, 1)?.and_hms_opt(0, 0, 0)?;
                continue;
            }
            if !self.day_ok(&c) {
                c = (c.date() + CDuration::days(1)).and_hms_opt(0, 0, 0)?;
                continue;
            }
            if !self.hour[c.hour() as usize] {
                c = c.with_minute(0)? + CDuration::hours(1);
                continue;
            }
            if !self.min[c.minute() as usize] {
                c += CDuration::minutes(1);
                continue;
            }
            return Some(c);
        }
        None
    }
}

// ------------------------------------------------------------------ when
/// A schedule's "when" (as the page sends it) -> once (a local time), every N minutes, or a cron expression.
pub enum When {
    Once(NaiveDateTime),
    Every(i64),
    Cron(Cron),
}

fn hm(v: &Value) -> Result<(u32, u32), String> {
    let t = v["time"].as_str().unwrap_or("");
    let (h, m) = t.split_once(':').ok_or("a time is HH:MM")?;
    let (h, m) = (h.trim().parse::<u32>().map_err(|_| "a time is HH:MM")?, m.trim().parse::<u32>().map_err(|_| "a time is HH:MM")?);
    if h > 23 || m > 59 {
        return Err("a time is HH:MM (00:00 to 23:59)".into());
    }
    Ok((h, m))
}

pub fn when_of(v: &Value) -> Result<When, String> {
    let kind = v["kind"].as_str().unwrap_or("");
    let cron = |s: String| Cron::parse(&s).map(When::Cron);
    match kind {
        "once" => {
            let at = v["at"].as_str().unwrap_or("").replace('T', " ");
            NaiveDateTime::parse_from_str(&at, "%Y-%m-%d %H:%M").map(When::Once).map_err(|_| "once: a date and time, YYYY-MM-DD HH:MM".into())
        }
        "minutes" => {
            let n = v["every"].as_i64().unwrap_or(0);
            if n < 5 { Err("every N minutes: 5 or more".into()) } else { Ok(When::Every(n)) }
        }
        "hourly" => cron(format!("{} * * * *", v["minute"].as_u64().unwrap_or(0).min(59))),
        "daily" => { let (h, m) = hm(v)?; cron(format!("{m} {h} * * *")) }
        "weekdays" => { let (h, m) = hm(v)?; cron(format!("{m} {h} * * 1-5")) }
        "weekly" => { let (h, m) = hm(v)?; cron(format!("{m} {h} * * {}", v["day"].as_u64().unwrap_or(1) % 7)) }
        "monthly" => { let (h, m) = hm(v)?; cron(format!("{m} {h} {} * *", v["day"].as_u64().unwrap_or(1).clamp(1, 31))) }
        "cron" => cron(v["cron"].as_str().unwrap_or("").to_string()),
        _ => Err("when: once, minutes, hourly, daily, weekdays, weekly, monthly or cron".into()),
    }
}

/// When it runs, in words (the list shows it).
pub fn describe(v: &Value) -> String {
    let t = v["time"].as_str().unwrap_or("");
    const DAYS: [&str; 7] = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
    match v["kind"].as_str().unwrap_or("") {
        "once" => format!("once, {}", v["at"].as_str().unwrap_or("").replace('T', " ")),
        "minutes" => format!("every {} minutes", v["every"]),
        "hourly" => format!("hourly at :{:02}", v["minute"].as_u64().unwrap_or(0)),
        "daily" => format!("daily at {t}"),
        "weekdays" => format!("weekdays at {t}"),
        "weekly" => format!("{}s at {t}", DAYS[(v["day"].as_u64().unwrap_or(1) % 7) as usize]),
        "monthly" => format!("monthly on day {} at {t}", v["day"]),
        "cron" => format!("cron {}", v["cron"].as_str().unwrap_or("")),
        _ => "?".into(),
    }
}

/// The next run after `now` (local); None: never again (a once that has passed).
pub fn next_run(w: &When, now: DateTime<Local>, last: Option<DateTime<Local>>) -> Option<DateTime<Local>> {
    let local = |n: NaiveDateTime| Local.from_local_datetime(&n).earliest();
    match w {
        When::Once(at) => local(*at).filter(|t| last.is_none() && *t > now - CDuration::days(7)),
        When::Every(n) => Some(last.map(|l| l + CDuration::minutes(*n)).unwrap_or(now + CDuration::minutes(*n))),
        When::Cron(c) => c.next_after(now.naive_local()).and_then(local),
    }
}

// ------------------------------------------------------------------ the list and its runs
static LIST: LazyLock<Mutex<Vec<Value>>> = LazyLock::new(|| Mutex::new(load()));
static RUNNING: LazyLock<Mutex<Option<String>>> = LazyLock::new(Default::default);
const HISTORY: usize = 10;

fn path() -> PathBuf {
    crate::root().join(".strata-schedules.json")
}

fn load() -> Vec<Value> {
    std::fs::read_to_string(path()).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()).and_then(|v| v["schedules"].as_array().cloned()).unwrap_or_default()
}

fn save(list: &[Value]) {
    let tmp = path().with_extension("json.tmp");
    if std::fs::write(&tmp, serde_json::to_string_pretty(&json!({"schedules": list})).unwrap_or_default()).is_ok() {
        let _ = std::fs::rename(&tmp, path());
    }
}

fn ts(t: Option<DateTime<Local>>) -> Value {
    t.map(|t| json!(t.timestamp())).unwrap_or(Value::Null)
}

/// GET /schedules: each with its "when" in words, its next run, and whether it runs now.
pub fn list() -> Value {
    let running = RUNNING.lock().unwrap().clone();
    let l: Vec<Value> = LIST.lock().unwrap().iter().map(|s| {
        let mut s = s.clone();
        s["when_text"] = json!(describe(&s["when"]));
        s["running"] = json!(running.as_deref() == s["id"].as_str());
        s
    }).collect();
    json!({"schedules": l, "now": Local::now().timestamp()})
}

/// POST /schedules: a new one, or a change to one (its "id"); its next run worked out here.
pub fn put(req: &Value) -> Result<Value, String> {
    let name = req["name"].as_str().unwrap_or("").trim().to_string();
    let prompt = req["prompt"].as_str().unwrap_or("").trim().to_string();
    if name.is_empty() || name.chars().count() > 80 {
        return Err("a schedule needs a name (up to 80 characters)".into());
    }
    if prompt.is_empty() || prompt.len() > 20_000 {
        return Err("a schedule needs a prompt (up to 20,000 characters)".into());
    }
    let w = when_of(&req["when"])?;
    let enabled = req["enabled"].as_bool().unwrap_or(true);
    let next = if enabled { next_run(&w, Local::now(), None) } else { None };
    if enabled && next.is_none() {
        return Err("that time has passed".into());
    }
    let mut l = LIST.lock().unwrap();
    let id = req["id"].as_str().filter(|i| l.iter().any(|s| s["id"] == **i)).map(String::from)
        .unwrap_or_else(|| format!("s{}", Local::now().timestamp_millis()));
    let old = l.iter().find(|s| s["id"] == id).cloned().unwrap_or(json!({"history": [], "created": Local::now().timestamp()}));
    let s = json!({"id": id, "name": name, "prompt": prompt, "when": req["when"], "tools": req["tools"].as_bool().unwrap_or(true),
                   "enabled": enabled, "created": old["created"], "last_run": old["last_run"], "next_run": ts(next), "history": old["history"]});
    match l.iter().position(|x| x["id"] == id) {
        Some(i) => l[i] = s.clone(),
        None => l.push(s.clone()),
    }
    save(&l);
    Ok(s)
}

pub fn delete(id: &str) -> bool {
    let mut l = LIST.lock().unwrap();
    let n = l.len();
    l.retain(|s| s["id"] != id);
    save(&l);
    l.len() < n
}

/// Run one now (the page's Run now; the runner): this server's own chat, the answer kept and notified.
pub fn run(id: &str, port: u16, key: Option<&str>) -> Result<(), String> {
    let s = LIST.lock().unwrap().iter().find(|s| s["id"] == id).cloned().ok_or("no such schedule")?;
    {
        let mut r = RUNNING.lock().unwrap();
        if r.is_some() {
            return Err("another scheduled prompt is running".into());
        }
        *r = Some(id.to_string());
    }
    let name = s["name"].as_str().unwrap_or("").to_string();
    say(&format!("schedule {name:?}: running"));
    let started = std::time::Instant::now();
    let now = Local::now();
    let system = format!("This is the scheduled prompt \"{name}\" ({}), run on {}. Answer it directly and briefly: the answer is shown in a notification and kept for the user.",
                         describe(&s["when"]), now.format("%A %Y-%m-%d %H:%M"));
    let body = json!({"model": "strata", "max_tokens": 2000, "reasoning_effort": "low", "strata_mcp": s["tools"].as_bool().unwrap_or(true),
                      "messages": [{"role": "system", "content": system}, {"role": "user", "content": s["prompt"]}]});
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(1800))).http_status_as_error(false).build().into();
    let mut req = agent.post(&format!("http://127.0.0.1:{port}/v1/chat/completions")).header("Content-Type", "application/json");
    if let Some(k) = key {
        req = req.header("Authorization", &format!("Bearer {k}"));
    }
    let out: Result<String, String> = req.send(body.to_string().as_bytes()).map_err(|e| e.to_string()).and_then(|r| {
        let v: Value = serde_json::from_reader(r.into_body().into_reader()).map_err(|e| e.to_string())?;
        v["choices"][0]["message"]["content"].as_str().filter(|t| !t.trim().is_empty()).map(String::from)
            .ok_or_else(|| v["error"]["message"].as_str().unwrap_or("no answer").to_string())
    });
    *RUNNING.lock().unwrap() = None;
    let secs = started.elapsed().as_secs_f64();
    let (ok, text) = match &out { Ok(t) => (true, t.clone()), Err(e) => (false, e.clone()) };
    say(&format!("schedule {name:?}: {} in {secs:.0} s", if ok { "answered" } else { "failed" }));
    notify(&name, &text);
    let mut l = LIST.lock().unwrap();
    if let Some(x) = l.iter_mut().find(|x| x["id"] == id) {
        let mut h = x["history"].as_array().cloned().unwrap_or_default();
        h.insert(0, json!({"at": now.timestamp(), "secs": (secs * 10.0).round() / 10.0, "ok": ok, "text": text}));
        h.truncate(HISTORY);
        x["history"] = json!(h);
        x["last_run"] = json!(now.timestamp());
        let next = when_of(&x["when"]).ok().and_then(|w| next_run(&w, Local::now(), Some(now)));
        x["next_run"] = ts(next.filter(|_| x["enabled"] == true));
        if next.is_none() {
            x["enabled"] = json!(false); // a once: done
        }
    }
    save(&l);
    out.map(|_| ())
}

/// A macOS notification with the answer's start (osascript: no app bundle needed).
fn notify(name: &str, text: &str) {
    if !cfg!(target_os = "macos") {
        return;
    }
    let body: String = text.replace(['*', '#', '`'], "").split_whitespace().collect::<Vec<_>>().join(" ").chars().take(220).collect();
    let q = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
    let _ = std::process::Command::new("/usr/bin/osascript")
        .args(["-e", &format!("display notification \"{}\" with title \"Strata: {}\"", q(&body), q(name))]).status();
}

/// The runner: every 20 s, the due ones in turn (one at a time).  Missed while off: next_run is in the past, so it
/// runs once and its next run is worked out from now.
pub fn start(port: u16, key: Option<String>) {
    std::thread::spawn(move || loop {
        let now = Local::now().timestamp();
        let due: Vec<String> = LIST.lock().unwrap().iter()
            .filter(|s| s["enabled"] == true && s["next_run"].as_i64().is_some_and(|t| t <= now))
            .filter_map(|s| s["id"].as_str().map(String::from)).collect();
        for id in due {
            let _ = run(&id, port, key.as_deref());
        }
        std::thread::sleep(Duration::from_secs(20));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(s: &str) -> NaiveDateTime {
        NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M").unwrap()
    }

    #[test]
    fn cron_next_runs() {
        let next = |c: &str, from: &str| Cron::parse(c).unwrap().next_after(at(from)).unwrap().format("%Y-%m-%d %H:%M %a").to_string();
        assert_eq!(next("0 8 * * *", "2026-10-30 07:59"), "2026-10-30 08:00 Fri");
        assert_eq!(next("0 8 * * *", "2026-10-30 08:00"), "2026-10-31 08:00 Sat");          // strictly after
        assert_eq!(next("30 9 * * 1-5", "2026-10-30 10:00"), "2026-11-02 09:30 Mon");       // Fri after: Monday
        assert_eq!(next("*/15 * * * *", "2026-10-30 10:07"), "2026-10-30 10:15 Fri");
        assert_eq!(next("0 0 1 * *", "2026-12-15 12:00"), "2027-01-01 00:00 Fri");          // the year turns
        assert_eq!(next("0 12 13 * fri", "2026-10-30 13:00"), "2026-11-06 12:00 Fri");      // either: Friday 6th
        assert_eq!(next("0 9 * * 7", "2026-10-30 10:00"), "2026-11-01 09:00 Sun");          // 7 = Sunday
        assert_eq!(next("0 9 29 feb *", "2026-10-30 10:00"), "2028-02-29 09:00 Tue");       // a leap day
        for bad in ["* * * *", "60 * * * *", "* 24 * * *", "5-1 * * * *", "x * * * *"] {
            assert!(Cron::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn whens_and_their_words() {
        let now = Local.from_local_datetime(&at("2026-10-30 10:00")).unwrap();
        let n = |v: Value| next_run(&when_of(&v).unwrap(), now, None).map(|t| t.format("%Y-%m-%d %H:%M").to_string());
        assert_eq!(n(json!({"kind": "daily", "time": "08:00"})), Some("2026-10-31 08:00".into()));
        assert_eq!(n(json!({"kind": "weekly", "day": 1, "time": "09:00"})), Some("2026-11-02 09:00".into()));
        assert_eq!(n(json!({"kind": "monthly", "day": 1, "time": "07:30"})), Some("2026-11-01 07:30".into()));
        assert_eq!(n(json!({"kind": "minutes", "every": 30})), Some("2026-10-30 10:30".into()));
        assert_eq!(n(json!({"kind": "once", "at": "2026-10-30 18:00"})), Some("2026-10-30 18:00".into()));
        assert!(when_of(&json!({"kind": "minutes", "every": 2})).is_err());
        assert!(when_of(&json!({"kind": "daily", "time": "25:00"})).is_err());
        assert_eq!(describe(&json!({"kind": "weekly", "day": 1, "time": "09:00"})), "Mondays at 09:00");
        assert_eq!(describe(&json!({"kind": "weekdays", "time": "08:15"})), "weekdays at 08:15");
    }
}
