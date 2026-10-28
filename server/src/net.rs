//! Who may use the server (serve/server.py): the Host check against DNS rebinding, the Origin check for browser pages,
//! the API keys (several, compared in constant time in their UTF-8 and Latin-1 forms), and the names this server
//! answers to.

use std::collections::HashSet;

pub const LOOPBACK_NAMES: [&str; 3] = ["localhost", "127.0.0.1", "::1"];

/// The name in a Host header (or an origin's host[:port]): "Example.com:8080" -> "example.com", "[::1]:8095" -> "::1";
/// "" when it is malformed.
pub fn host_name(value: &str) -> String {
    let v = crate::py::strip(value).to_lowercase();
    if let Some(rest) = v.strip_prefix('[') {
        let Some((name, after)) = rest.split_once(']') else { return String::new() };
        let ok = after.is_empty() || (after.starts_with(':') && !after[1..].is_empty() && after[1..].chars().all(|c| c.is_ascii_digit()));
        return if ok { name.to_string() } else { String::new() };
    }
    let mut v = v;
    if v.matches(':').count() == 1 {
        let (h, port) = v.split_once(':').unwrap();
        if port.is_empty() || !port.chars().all(|c| c.is_ascii_digit()) {
            return String::new();
        }
        v = h.to_string();
    } else if v.contains(':') {
        return v;
    }
    let v = v.trim_end_matches('.').to_string();
    if !v.is_empty() && v.chars().all(|c| c.is_alphanumeric() || "-._".contains(c)) { v } else { String::new() }
}

fn is_ip(name: &str) -> bool {
    name.parse::<std::net::IpAddr>().is_ok()
}

fn name_in(name: &str, names: &HashSet<String>) -> bool {
    names.contains(name) || names.iter().any(|n| n.starts_with('.') && (name.ends_with(n.as_str()) || name == &n[1..]))
}

/// The config's allowed_hosts plus $STRATA_ALLOWED_HOSTS; Err for a wrong entry.
pub fn allowed_hosts_of(value: &serde_json::Value, env: &str) -> Result<Vec<String>, String> {
    let items: Vec<String> = match value {
        serde_json::Value::Null => vec![],
        serde_json::Value::String(s) if s.is_empty() => vec![],
        serde_json::Value::String(s) => vec![s.clone()],
        serde_json::Value::Array(a) if a.iter().all(|x| x.is_string()) => a.iter().map(|x| x.as_str().unwrap().to_string()).collect(),
        _ => return Err("allowed_hosts: expected a host name or a list of them".into()),
    };
    let mut out: Vec<String> = vec![];
    for raw in items.into_iter().chain(env.split(',').filter(|x| !crate::py::strip(x).is_empty()).map(String::from)) {
        let x = crate::py::strip(&raw).to_lowercase();
        if x == "*" {
            if !out.contains(&x) {
                out.push(x);
            }
            continue;
        }
        let x = x.split_once("://").map(|p| p.1.to_string()).unwrap_or(x);
        let x = x.split('/').next().unwrap_or("").to_string();
        let dot = x.starts_with('.');
        let name = host_name(if dot { &x[1..] } else { &x });
        if name.is_empty() {
            return Err(format!("allowed_hosts: {} is not a host name like strata.example.com", crate::py::py_repr(&serde_json::json!(raw))));
        }
        let n = if dot { format!(".{name}") } else { name };
        if !out.contains(&n) {
            out.push(n);
        }
    }
    Ok(out)
}

/// This PC's IPv4 addresses on its networks, without loopback/link-local: the default route's first.
pub fn lan_addresses() -> Vec<String> {
    let mut first = None;
    if let Ok(s) = std::net::UdpSocket::bind("0.0.0.0:0") {
        if s.connect("10.255.255.255:1").is_ok() {
            first = s.local_addr().ok().map(|a| a.ip().to_string());
        }
    }
    let mut ips: Vec<String> = vec![];
    unsafe {
        let mut ifa: *mut libc::ifaddrs = std::ptr::null_mut();
        if libc::getifaddrs(&mut ifa) == 0 {
            let mut p = ifa;
            while !p.is_null() {
                let a = (*p).ifa_addr;
                if !a.is_null() && (*a).sa_family as i32 == libc::AF_INET {
                    let sin = a as *const libc::sockaddr_in;
                    let ip = std::net::Ipv4Addr::from(u32::from_be((*sin).sin_addr.s_addr));
                    ips.push(ip.to_string());
                }
                p = (*p).ifa_next;
            }
            libc::freeifaddrs(ifa);
        }
    }
    let ok = |ip: &str| !ip.is_empty() && !ip.starts_with("127.") && !ip.starts_with("169.254.") && !ip.starts_with("0.");
    let mut rest: Vec<String> = ips.into_iter().filter(|ip| ok(ip) && Some(ip) != first.as_ref()).collect();
    rest.sort();
    rest.dedup();
    let mut out: Vec<String> = first.filter(|f| ok(f)).into_iter().collect();
    out.extend(rest);
    out
}

pub fn hostname() -> String {
    let mut buf = [0u8; 256];
    unsafe {
        if libc::gethostname(buf.as_mut_ptr() as *mut libc::c_char, buf.len()) != 0 {
            return String::new();
        }
    }
    let end = buf.iter().position(|b| *b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).to_lowercase()
}

/// Every name this server answers to besides an IP address.
pub fn host_names_for(bind_host: &str, allowed_hosts: &[String], trusted_origins: &[String]) -> HashSet<String> {
    let mut names: HashSet<String> = LOOPBACK_NAMES.iter().map(|s| s.to_string()).collect();
    let bind = host_name(bind_host);
    if !bind.is_empty() && bind != "0.0.0.0" && bind != "::" {
        names.insert(bind.clone());
    }
    if !LOOPBACK_NAMES.contains(&bind.as_str()) {
        let pc = hostname();
        if !pc.is_empty() {
            names.insert(format!("{pc}.local"));
            names.insert(pc);
        }
        names.insert("host.docker.internal".into());
        names.extend(lan_addresses());
    }
    names.extend(allowed_hosts.iter().filter(|x| *x != "*").cloned());
    names.extend(trusted_origins.iter().map(|o| host_name(o.split_once("://").map(|p| p.1).unwrap_or(o))).filter(|n| !n.is_empty()));
    names
}

/// DNS rebinding: only the names this server answers to pass (an IP, *.localhost and no Host header pass too).
pub fn host_allowed(host: Option<&str>, names: &HashSet<String>, any_host: bool) -> bool {
    let h = host.unwrap_or("");
    if any_host || crate::py::strip(h).is_empty() {
        return true;
    }
    let name = host_name(h);
    !name.is_empty() && (is_ip(&name) || name == "localhost" || name.ends_with(".localhost") || name_in(&name, names))
}

/// #1344: the keys a server accepts ("k1,k2" or a config list).
pub fn key_list(value: &serde_json::Value) -> Vec<String> {
    match value {
        serde_json::Value::Null => vec![],
        serde_json::Value::String(s) => s.split(',').map(crate::py::strip).filter(|k| !k.is_empty()).map(String::from).collect(),
        serde_json::Value::Array(a) => a.iter().map(crate::py::py_str).filter(|k| !k.is_empty()).collect(),
        v => vec![crate::py::py_str(v)],
    }
}

/// The key as a client can send it (#725): stripped; only blanks is an error, never "no key" (#213).
pub fn api_key_of(value: &serde_json::Value) -> Result<serde_json::Value, String> {
    match value {
        serde_json::Value::Array(a) => {
            let keys: Vec<String> = a.iter().map(|k| crate::py::strip(&crate::py::py_str(k)).to_string()).collect();
            if keys.is_empty() || keys.iter().any(|k| k.is_empty()) {
                return Err("an API key was given but it is empty".into());
            }
            Ok(serde_json::json!(keys))
        }
        serde_json::Value::Null => Ok(serde_json::json!("")),
        v => {
            let key = crate::py::py_str(v);
            if !key.is_empty() && key_list(&serde_json::json!(key)).is_empty() {
                return Err("an API key was given but it is empty".into());
            }
            Ok(serde_json::json!(crate::py::strip(&key)))
        }
    }
}

fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    let mut diff = (a.len() ^ b.len()) as u8 | (a.len() != b.len()) as u8;
    for i in 0..a.len().max(b.len()) {
        diff |= a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0);
    }
    diff == 0
}

/// `given`: the header's raw bytes.  Any key matches, in its UTF-8 or Latin-1 form; every key is compared.
pub fn key_matches(given: &[u8], keys: &[String]) -> bool {
    let mut ok = false;
    for k in keys {
        ok |= ct_eq(given, k.as_bytes());
        if k.chars().all(|c| (c as u32) < 256) {
            let latin: Vec<u8> = k.chars().map(|c| c as u8).collect();
            ok |= ct_eq(given, &latin);
        }
    }
    ok
}

/// A browser page's Origin that may use the model without an API key.
pub fn origin_allowed(origin: &str, host: Option<&str>, names: &HashSet<String>, origins: &[String]) -> bool {
    let origin = crate::py::strip(origin).trim_end_matches('/');
    if origins.iter().any(|o| o == origin || o == "*") {
        return true;
    }
    let lower = origin.to_lowercase();
    let Some((scheme, rest)) = lower.split_once("://") else { return false };
    if scheme.is_empty() {
        return false;
    }
    if scheme != "http" && scheme != "https" {
        return true;
    }
    if let Some(h) = host {
        if rest == crate::py::strip(h).to_lowercase() {
            return true;
        }
    }
    let name = host_name(rest);
    !name.is_empty() && (LOOPBACK_NAMES.contains(&name.as_str()) || name.ends_with(".localhost") || name_in(&name, names))
}

/// #321: a config's origin list; Err for a wrong entry.
pub fn origins_of(value: &serde_json::Value, key: &str, wildcard: bool) -> Result<Vec<String>, String> {
    let items: Vec<String> = match value {
        serde_json::Value::Null => return Ok(vec![]),
        serde_json::Value::String(s) if s.is_empty() => return Ok(vec![]),
        serde_json::Value::Array(a) if a.is_empty() => return Ok(vec![]),
        serde_json::Value::String(s) => vec![s.clone()],
        serde_json::Value::Array(a) if a.iter().all(|x| x.is_string()) => a.iter().map(|x| x.as_str().unwrap().to_string()).collect(),
        _ => return Err(format!("[strata] {key}: expected an origin or a list of origins")),
    };
    let mut out = vec![];
    for x in items {
        let x = crate::py::strip(&x).trim_end_matches('/').to_string();
        if x == "*" && wildcard {
            out.push(x);
            continue;
        }
        let ok = x.split_once("://").map(|(s, r)| (s == "http" || s == "https") && !r.is_empty() && !r.contains('/') && !r.contains('*')).unwrap_or(false);
        if !ok {
            return Err(format!("[strata] {key}: {} is not an origin like https://chat.example.com{}", crate::py::py_repr(&serde_json::json!(x)), if wildcard { "" } else { " (no wildcards here)" }));
        }
        out.push(x);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hosts_origins_and_keys() {
        assert_eq!(host_name("Example.com:8080"), "example.com");
        assert_eq!(host_name("[::1]:8095"), "::1");
        assert_eq!(host_name("evil.com:x"), "");
        let names: HashSet<String> = ["localhost".to_string(), ".example.com".to_string()].into();
        assert!(host_allowed(Some("a.example.com"), &names, false));
        assert!(host_allowed(Some("192.168.1.5:80"), &names, false));
        assert!(!host_allowed(Some("evil.com"), &names, false));
        assert!(origin_allowed("http://localhost:3000", None, &names, &[]));
        assert!(!origin_allowed("https://evil.com", Some("127.0.0.1:8090"), &names, &[]));
        assert!(origin_allowed("chrome-extension://abc", None, &names, &[]));
        assert!(key_matches(b"k2", &["k1".into(), "k2".into()]));
        assert!(key_matches(&[0xe9], &["é".into()]));
        assert!(!key_matches(b"k", &["k1".into()]));
    }
}
