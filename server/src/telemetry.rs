//! Hardware readings for the Monitor tab (serve/telemetry.py, the Apple Silicon half): a thread samples once a second
//! and keeps the last 60 readings of each series.  GPU load and memory from `ioreg` (IOAccelerator
//! PerformanceStatistics) and Metal's working-set limit; GPU power from IOReport's "Energy Model" counters and the
//! chip's temperature from the PMU die sensors over IOHID (both without root, as sudo-free monitors such as macmon read
//! them); CPU and RAM from the Mach host statistics; disk rates from the block storage drivers' statistics.  Anything
//! that cannot be read is null; nothing here can stop the server.

use serde_json::{json, Map, Value};
use std::collections::VecDeque;
use std::ffi::c_void;
#[cfg(target_os = "macos")]
use std::ffi::{CStr, CString};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const HISTORY: usize = 60;
type Cf = *const c_void;

#[cfg(target_os = "macos")]
#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFStringCreateWithCString(alloc: Cf, s: *const i8, enc: u32) -> Cf;
    fn CFStringGetCString(s: Cf, buf: *mut i8, len: isize, enc: u32) -> bool;
    fn CFDictionaryGetValue(d: Cf, k: Cf) -> Cf;
    fn CFArrayGetCount(a: Cf) -> isize;
    fn CFArrayGetValueAtIndex(a: Cf, i: isize) -> Cf;
    fn CFNumberCreate(alloc: Cf, ty: isize, v: *const c_void) -> Cf;
    fn CFDictionaryCreate(alloc: Cf, k: *const Cf, v: *const Cf, n: isize, kcb: *const c_void, vcb: *const c_void) -> Cf;
    fn CFRelease(r: Cf);
    static kCFTypeDictionaryKeyCallBacks: c_void;
    static kCFTypeDictionaryValueCallBacks: c_void;
}

#[cfg(target_os = "macos")]
#[link(name = "IOReport", kind = "dylib")]
unsafe extern "C" {
    fn IOReportCopyChannelsInGroup(g: Cf, sg: Cf, a: u64, b: u64, c: u64) -> Cf;
    fn IOReportCreateSubscription(a: Cf, chans: Cf, out: *mut Cf, b: u64, c: Cf) -> Cf;
    fn IOReportCreateSamples(sub: Cf, ch: Cf, a: Cf) -> Cf;
    fn IOReportCreateSamplesDelta(a: Cf, b: Cf, c: Cf) -> Cf;
    fn IOReportChannelGetChannelName(ch: Cf) -> Cf;
    fn IOReportChannelGetUnitLabel(ch: Cf) -> Cf;
    fn IOReportSimpleGetIntegerValue(ch: Cf, i: i32) -> i64;
}

#[cfg(target_os = "macos")]
#[link(name = "IOKit", kind = "framework")]
unsafe extern "C" {
    fn IOHIDEventSystemClientCreate(a: Cf) -> Cf;
    fn IOHIDEventSystemClientSetMatching(c: Cf, m: Cf) -> i32;
    fn IOHIDEventSystemClientCopyServices(c: Cf) -> Cf;
    fn IOHIDServiceClientCopyProperty(s: Cf, k: Cf) -> Cf;
    fn IOHIDServiceClientCopyEvent(s: Cf, ty: i64, a: i32, b: i64) -> Cf;
    fn IOHIDEventGetFloatValue(e: Cf, f: i32) -> f64;
}

#[cfg(target_os = "macos")]
#[link(name = "Metal", kind = "framework")]
unsafe extern "C" {
    fn MTLCreateSystemDefaultDevice() -> *mut c_void;
}

#[cfg(target_os = "macos")]
#[link(name = "objc")]
unsafe extern "C" {
    fn sel_registerName(name: *const i8) -> *const c_void;
    fn objc_msgSend();
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn mach_host_self() -> u32;
    fn host_statistics64(host: u32, flavor: i32, info: *mut i32, count: *mut u32) -> i32;
    fn host_statistics(host: u32, flavor: i32, info: *mut i32, count: *mut u32) -> i32;
}

const UTF8: u32 = 0x08000100;

#[cfg(target_os = "macos")]
fn cfs(s: &str) -> Cf {
    let c = CString::new(s).unwrap();
    unsafe { CFStringCreateWithCString(std::ptr::null(), c.as_ptr(), UTF8) }
}

#[cfg(target_os = "macos")]
fn cf_str(r: Cf) -> String {
    if r.is_null() {
        return String::new();
    }
    let mut buf = [0i8; 128];
    if unsafe { CFStringGetCString(r, buf.as_mut_ptr(), 128, UTF8) } {
        unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned()
    } else {
        String::new()
    }
}

/// Metal's recommendedMaxWorkingSetSize: the share of the unified memory the GPU may use (0 when it cannot be asked).
#[cfg(target_os = "macos")]
pub fn metal_working_set_bytes() -> u64 {
    unsafe {
        let dev = MTLCreateSystemDefaultDevice();
        if dev.is_null() {
            return 0;
        }
        let sel = sel_registerName(c"recommendedMaxWorkingSetSize".as_ptr());
        let send: extern "C" fn(*mut c_void, *const c_void) -> u64 = std::mem::transmute(objc_msgSend as *const ());
        send(dev, sel)
    }
}

fn sysctl_str(name: &str) -> Option<String> {
    let out = std::process::Command::new("sysctl").args(["-n", name]).output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

#[cfg(target_os = "macos")]
fn sysctl_u64(name: &str) -> Option<u64> {
    let c = CString::new(name).ok()?;
    let mut v: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    let r = unsafe { libc::sysctlbyname(c.as_ptr(), &mut v as *mut u64 as *mut c_void, &mut len, std::ptr::null_mut(), 0) };
    if r != 0 {
        return None;
    }
    Some(if len == 4 { v & 0xffff_ffff } else { v })
}

/// The highest of the chip's die sensors ("PMU tdie<n>"); None without a valid one.
pub fn chip_temp(readings: &[(String, f64)]) -> Option<f64> {
    let re = regex::Regex::new(r"^PMU tdie\d+$").unwrap();
    readings.iter().filter(|(n, v)| re.is_match(n) && v.is_finite() && *v > -20.0 && *v < 150.0).map(|(_, v)| *v).fold(None, |a, v| Some(a.map_or(v, |a: f64| a.max(v))))
}

/// An IOReport energy counter's change over `seconds` -> watts.
pub fn energy_watts(delta: f64, unit: &str, seconds: f64) -> Option<f64> {
    let scale = match unit {
        "mJ" => 1e-3,
        "uJ" | "µJ" => 1e-6,
        "nJ" => 1e-9,
        _ => return None,
    };
    if !delta.is_finite() || !seconds.is_finite() || seconds <= 0.0 || delta < 0.0 {
        return None;
    }
    Some(delta * scale / seconds)
}

#[cfg(target_os = "macos")]
struct MacSensors {
    power_ok: bool,
    temp_ok: bool,
    prev: Option<(usize, Instant)>,
    zero_streak: u32,
    sub: Cf,
    sub_ch: Cf,
    client: Cf,
    k_channels: Cf,
    k_product: Cf,
}

#[cfg(target_os = "macos")]
unsafe impl Send for MacSensors {}

#[cfg(target_os = "macos")]
impl MacSensors {
    fn new() -> Self {
        unsafe {
            let k_channels = cfs("IOReportChannels");
            let k_product = cfs("Product");
            let group = cfs("Energy Model");
            let chans = IOReportCopyChannelsInGroup(group, std::ptr::null(), 0, 0, 0);
            let mut sub_ch: Cf = std::ptr::null();
            let mut sub: Cf = std::ptr::null();
            if !chans.is_null() {
                sub = IOReportCreateSubscription(std::ptr::null(), chans, &mut sub_ch, 0, std::ptr::null());
            }
            let (a, b): (i32, i32) = (0xff00, 5);
            let keys = [cfs("PrimaryUsagePage"), cfs("PrimaryUsage")];
            let vals = [CFNumberCreate(std::ptr::null(), 3, &a as *const i32 as *const c_void), CFNumberCreate(std::ptr::null(), 3, &b as *const i32 as *const c_void)];
            let matching = CFDictionaryCreate(std::ptr::null(), keys.as_ptr(), vals.as_ptr(), 2, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
            for r in keys.iter().chain(vals.iter()) {
                if !r.is_null() {
                    CFRelease(*r);
                }
            }
            let client = IOHIDEventSystemClientCreate(std::ptr::null());
            if !client.is_null() && !matching.is_null() {
                IOHIDEventSystemClientSetMatching(client, matching);
            }
            MacSensors { power_ok: !sub.is_null() && !sub_ch.is_null(), temp_ok: !client.is_null() && !matching.is_null(), prev: None, zero_streak: 0, sub, sub_ch, client, k_channels, k_product }
        }
    }

    fn gpu_watts(&mut self) -> Option<f64> {
        unsafe {
            let sample = IOReportCreateSamples(self.sub, self.sub_ch, std::ptr::null());
            let now = Instant::now();
            if sample.is_null() {
                return None;
            }
            let prev = self.prev.replace((sample as usize, now))?;
            let delta = IOReportCreateSamplesDelta(prev.0 as Cf, sample, std::ptr::null());
            CFRelease(prev.0 as Cf);
            if delta.is_null() {
                return None;
            }
            let chans = CFDictionaryGetValue(delta, self.k_channels);
            let n = if chans.is_null() { 0 } else { CFArrayGetCount(chans) };
            let mut out = None;
            for i in 0..n {
                let ch = CFArrayGetValueAtIndex(chans, i);
                if cf_str(IOReportChannelGetChannelName(ch)) == "GPU Energy" {
                    out = energy_watts(IOReportSimpleGetIntegerValue(ch, 0) as f64, &cf_str(IOReportChannelGetUnitLabel(ch)), (now - prev.1).as_secs_f64());
                    break;
                }
            }
            CFRelease(delta);
            out
        }
    }

    fn temps(&self) -> Vec<(String, f64)> {
        let mut out = vec![];
        unsafe {
            let svcs = IOHIDEventSystemClientCopyServices(self.client);
            if svcs.is_null() {
                return out;
            }
            for i in 0..CFArrayGetCount(svcs) {
                let s = CFArrayGetValueAtIndex(svcs, i);
                let name_ref = IOHIDServiceClientCopyProperty(s, self.k_product);
                let name = cf_str(name_ref);
                if !name_ref.is_null() {
                    CFRelease(name_ref);
                }
                let ev = IOHIDServiceClientCopyEvent(s, 15, 0, 0);
                if !ev.is_null() {
                    out.push((name, IOHIDEventGetFloatValue(ev, 15 << 16)));
                    CFRelease(ev);
                }
            }
            CFRelease(svcs);
        }
        out
    }

    fn read(&mut self, util: Option<f64>) -> (Option<f64>, Option<f64>) {
        let mut watts = if self.power_ok { self.gpu_watts() } else { None };
        let temp = if self.temp_ok { chip_temp(&self.temps()) } else { None };
        let busy = util.map(|u| u >= 10.0).unwrap_or(false);
        if watts == Some(0.0) && busy {
            self.zero_streak += 1;
            if self.zero_streak >= 3 {
                watts = None;
            }
        } else if watts.is_some() {
            self.zero_streak = 0;
        }
        (watts.filter(|w| w.is_finite()), temp.filter(|t| t.is_finite()))
    }
}

/// Off a Mac (the server in a Linux container): no GPU of its own to read; CPU and RAM from /proc.
#[cfg(not(target_os = "macos"))]
mod other {
    pub fn metal_working_set_bytes() -> u64 {
        0
    }
    pub fn sysctl_u64(name: &str) -> Option<u64> {
        match name {
            "hw.logicalcpu" | "hw.physicalcpu" => std::thread::available_parallelism().ok().map(|n| n.get() as u64),
            _ => None,
        }
    }
    pub fn ram() -> (Option<u64>, Option<u64>) {
        let Ok(t) = std::fs::read_to_string("/proc/meminfo") else { return (None, None) };
        let kb = |k: &str| t.lines().find(|l| l.starts_with(k)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<u64>().ok()).map(|v| v * 1024);
        match (kb("MemTotal:"), kb("MemAvailable:")) {
            (Some(total), Some(avail)) => (Some(total - avail), Some(total)),
            _ => (None, None),
        }
    }
    pub fn cpu_ticks() -> Option<(u64, u64)> {
        let t = std::fs::read_to_string("/proc/stat").ok()?;
        let f: Vec<u64> = t.lines().next()?.split_whitespace().skip(1).filter_map(|x| x.parse().ok()).collect();
        Some((f.get(3)? + f.get(4).unwrap_or(&0), f.iter().sum()))
    }
    pub struct MacSensors;
    impl MacSensors {
        pub fn new() -> Self {
            MacSensors
        }
        pub fn read(&mut self, _util: Option<f64>) -> (Option<f64>, Option<f64>) {
            (None, None)
        }
    }
}
#[cfg(not(target_os = "macos"))]
pub use other::{cpu_ticks, metal_working_set_bytes, ram};
#[cfg(not(target_os = "macos"))]
use other::{sysctl_u64, MacSensors};

/// ioreg's PerformanceStatistics line -> (utilization %, bytes in use).
pub fn parse_ioreg(text: &str) -> (Option<i64>, Option<i64>) {
    let u = regex::Regex::new(r#""Device Utilization %"=(\d+)"#).unwrap();
    let m = regex::Regex::new(r#""In use system memory"=(\d+)"#).unwrap();
    (u.captures(text).and_then(|c| c[1].parse().ok()), m.captures(text).and_then(|c| c[1].parse().ok()))
}

fn ioreg(class: &str) -> String {
    if !cfg!(target_os = "macos") {
        return String::new();
    }
    std::process::Command::new("ioreg").args(["-r", "-d", "1", "-c", class]).output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default()
}

/// The free VRAM of the Mac's GPU in MiB (Metal's limit less what the driver has in use), None when unreadable.
pub fn free_vram_mib() -> Option<i64> {
    let total = metal_working_set_bytes() as i64;
    let (_, used) = parse_ioreg(&ioreg("IOAccelerator"));
    if total <= 0 {
        return None;
    }
    Some((total - used?) >> 20)
}

/// (used, total) RAM bytes, psutil's way on macOS: available = free + inactive pages.
#[cfg(target_os = "macos")]
pub fn ram() -> (Option<u64>, Option<u64>) {
    let total = sysctl_u64("hw.memsize");
    let mut info = [0i32; 64];
    let mut count: u32 = 38; // HOST_VM_INFO64_COUNT
    let r = unsafe { host_statistics64(mach_host_self(), 4, info.as_mut_ptr(), &mut count) };
    if r != 0 || total.is_none() {
        return (None, total);
    }
    // vm_statistics64: free_count (u32) at 0, active 1, inactive 2 (natural_t words)
    let page = sysctl_u64("hw.pagesize").unwrap_or(16384);
    let avail = (info[0] as u32 as u64 + info[2] as u32 as u64) * page;
    let total = total.unwrap();
    (Some(total.saturating_sub(avail)), Some(total))
}

/// RAM available in MiB (session_save_retry), None when unreadable.
pub fn available_mib() -> Option<u64> {
    let (used, total) = ram();
    Some((total? - used?) >> 20)
}

#[cfg(target_os = "macos")]
fn cpu_ticks() -> Option<(u64, u64)> {
    let mut info = [0u32; 4];
    let mut count: u32 = 4; // HOST_CPU_LOAD_INFO_COUNT: user, system, idle, nice
    let r = unsafe { host_statistics(mach_host_self(), 3, info.as_mut_ptr() as *mut i32, &mut count) };
    if r != 0 {
        return None;
    }
    let total: u64 = info.iter().map(|x| *x as u64).sum();
    Some((info[2] as u64, total))
}

fn disk_bytes() -> Option<(u64, u64)> {
    let text = ioreg("IOBlockStorageDriver");
    let rd = regex::Regex::new(r#""Bytes \(Read\)"=(\d+)"#).unwrap();
    let wr = regex::Regex::new(r#""Bytes \(Write\)"=(\d+)"#).unwrap();
    let r: u64 = rd.captures_iter(&text).filter_map(|c| c[1].parse::<u64>().ok()).sum();
    let w: u64 = wr.captures_iter(&text).filter_map(|c| c[1].parse::<u64>().ok()).sum();
    if !text.contains("Bytes (Read)") {
        return None;
    }
    Some((r, w))
}

pub type Extra = Box<dyn Fn() -> Map<String, Value> + Send + Sync>;

pub struct Telemetry {
    now: Mutex<Map<String, Value>>,
    hist: Mutex<indexmap::IndexMap<String, VecDeque<Value>>>,
    stat: Map<String, Value>,
    stop: crate::sync::Event,
}

const HIST_KEYS: [&str; 10] = ["gpu_util", "gpu_mem_used", "gpu_temp", "gpu_power", "gpu_pcie_rx_mb", "cpu", "ram_used", "disk_read_mb", "tok_s", "prefill_tok_s_mean"];

impl Telemetry {
    pub fn start(extra: Extra) -> Arc<Self> {
        let total = metal_working_set_bytes();
        let chip = sysctl_str("machdep.cpu.brand_string");
        let stat = json!({
            "gpu_name": if total > 0 { json!(format!("{} GPU", chip.clone().unwrap_or_else(|| "Apple".into()))) } else { Value::Null },
            "gpu_count": 1, "gpu_note": null, "cpu_name": chip,
            "cores": sysctl_u64("hw.physicalcpu"), "threads": sysctl_u64("hw.logicalcpu"), "psutil": true});
        let t = Arc::new(Telemetry { now: Mutex::new(Map::new()), hist: Mutex::new(indexmap::IndexMap::new()), stat: stat.as_object().cloned().unwrap(), stop: crate::sync::Event::new() });
        let me = Arc::downgrade(&t);
        std::thread::Builder::new()
            .name("strata-telemetry".into())
            .spawn(move || {
                let mut sensors = MacSensors::new();
                let mut cpu_prev = cpu_ticks();
                let mut disk_prev: Option<(Instant, u64, u64)> = None;
                loop {
                    let Some(t) = me.upgrade() else { return };
                    if t.stop.is_set() {
                        return;
                    }
                    let mut s = Map::new();
                    if total > 0 {
                        let (util, used) = parse_ioreg(&ioreg("IOAccelerator"));
                        let (power, temp) = sensors.read(util.map(|u| u as f64));
                        s.insert("gpu_util".into(), json!(util));
                        s.insert("gpu_mem_used".into(), json!(used));
                        s.insert("gpu_mem_total".into(), json!(total));
                        s.insert("gpu_unified".into(), json!(true));
                        s.insert("gpu_power".into(), json!(power));
                        s.insert("gpu_temp".into(), json!(temp));
                    }
                    // psutil.cpu_percent(interval=None): the share busy since the previous call
                    let cur = cpu_ticks();
                    let cpu = match (cur, cpu_prev) {
                        (Some(c), Some(p)) if c.1 != p.1 => json!((100.0 * (1.0 - (c.0 - p.0) as f64 / (c.1 - p.1) as f64)).clamp(0.0, 100.0)),
                        (Some(_), Some(_)) => json!(0.0),
                        _ => Value::Null,
                    };
                    cpu_prev = cur;
                    s.insert("cpu".into(), cpu);
                    let (used, tot) = ram();
                    s.insert("ram_used".into(), json!(used));
                    s.insert("ram_total".into(), json!(tot));
                    let (rmb, wmb) = match disk_bytes() {
                        Some((r, w)) => {
                            let now = Instant::now();
                            let out = match disk_prev {
                                Some((t0, r0, w0)) if now > t0 => {
                                    let dt = (now - t0).as_secs_f64();
                                    (json!(r.saturating_sub(r0) as f64 / dt / 1048576.0), json!(w.saturating_sub(w0) as f64 / dt / 1048576.0))
                                }
                                _ => (Value::Null, Value::Null),
                            };
                            disk_prev = Some((now, r, w));
                            out
                        }
                        None => (Value::Null, Value::Null),
                    };
                    s.insert("disk_read_mb".into(), rmb);
                    s.insert("disk_write_mb".into(), wmb);
                    for (k, v) in extra() {
                        s.insert(k, v);
                    }
                    {
                        let mut h = t.hist.lock().unwrap();
                        for k in HIST_KEYS {
                            let v = s.get(k).cloned();
                            let v = match v {
                                Some(Value::Number(n)) if n.is_f64() => json!(crate::service::pyround(n.as_f64().unwrap(), 2)),
                                Some(v) => v,
                                None => Value::Null,
                            };
                            let q = h.entry(k.to_string()).or_default();
                            if q.len() == HISTORY {
                                q.pop_front();
                            }
                            q.push_back(v);
                        }
                    }
                    *t.now.lock().unwrap() = s;
                    drop(t); // re-upgraded each round: the sampler ends with its Telemetry
                    std::thread::sleep(Duration::from_secs(1));
                }
            })
            .expect("telemetry thread");
        t
    }

    pub fn snapshot(&self) -> Value {
        let hist: Map<String, Value> = self.hist.lock().unwrap().iter().map(|(k, v)| (k.clone(), Value::Array(v.iter().cloned().collect()))).collect();
        json!({"now": Value::Object(self.now.lock().unwrap().clone()), "history": hist, "static": Value::Object(self.stat.clone())})
    }

    pub fn close(&self) {
        self.stop.set();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readers_parse_and_convert() {
        assert_eq!(parse_ioreg(r#""PerformanceStatistics" = {"Device Utilization %"=37,"In use system memory"=123}"#), (Some(37), Some(123)));
        assert_eq!(energy_watts(2000.0, "mJ", 2.0), Some(1.0));
        assert_eq!(energy_watts(-1.0, "mJ", 2.0), None);
        assert_eq!(chip_temp(&[("PMU tdie1".into(), 50.0), ("PMU tdie2".into(), 61.5), ("PMU tdev1".into(), 90.0)]), Some(61.5));
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn this_mac_reads() {
        let (used, total) = ram();
        assert!(total.unwrap() > 1 << 30 && used.unwrap() > 0);
        assert!(metal_working_set_bytes() > 0);
    }
}
