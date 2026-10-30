//! The model's browser: Strata.app serves browser tools as an MCP server (Streamable HTTP, plain JSON replies) on
//! 127.0.0.1 with a key made at each start, and hands both to the server it starts (STRATA_BROWSER_MCP_URL / _KEY:
//! the server adds "browser", read-only, never saved).  The tools drive the browser pane people see (the bar's Web
//! button): the model's browsing happens in front of them, Osaurus's way, with their say: typing waits for Allow in
//! the browser's strip, password fields are refused, and Stop there ends the model's browsing until they let it again.
//!
//! Each call is a Job for the app's event loop (it owns the web view); the loop answers on the job's channel.
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

pub struct Job {
    pub tool: String,
    pub args: Value,
    pub reply: mpsc::Sender<Result<String, String>>,
}

/// The tools, as tools/list gives them.
pub fn tools() -> Value {
    json!([
        {"name": "browser_open", "description": "Open a web page in the browser beside the chat (the user sees it): an address, a site like example.com, or words to search with DuckDuckGo. Then browser_read it.",
         "inputSchema": {"type": "object", "required": ["url"], "properties": {"url": {"type": "string"}}}},
        {"name": "browser_read", "description": "The page in the browser now: its title, address, text, and its links, buttons and fields numbered [n] for browser_click / browser_type.",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "browser_click", "description": "Click element [n] of the last browser_read (a link, a button). Then browser_read again: the page may have changed.",
         "inputSchema": {"type": "object", "required": ["element"], "properties": {"element": {"type": "integer"}}}},
        {"name": "browser_type", "description": "Type text into field [n] of the last browser_read (the user is asked to allow it; never a password). submit: press Enter / send its form.",
         "inputSchema": {"type": "object", "required": ["element", "text"], "properties": {"element": {"type": "integer"}, "text": {"type": "string"}, "submit": {"type": "boolean"}}}},
        {"name": "browser_scroll", "description": "Scroll the page down (or up) by about a screen.",
         "inputSchema": {"type": "object", "properties": {"direction": {"type": "string", "enum": ["down", "up"]}}}},
        {"name": "browser_back", "description": "Go back to the previous page.",
         "inputSchema": {"type": "object", "properties": {}}}
    ])
}

/// Numbers the visible links, buttons and fields (data-strata-id) and returns the page: title, address, text, items.
pub const READ_JS: &str = r#"(() => {
  document.querySelectorAll('[data-strata-id]').forEach((e) => e.removeAttribute('data-strata-id'));
  const els = [...document.querySelectorAll('a[href],button,input:not([type=hidden]),textarea,select,[role=button],[role=link]')].filter((e) => {
    const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0 && getComputedStyle(e).visibility !== 'hidden'; }).slice(0, 150);
  const items = els.map((e, i) => {
    e.setAttribute('data-strata-id', i + 1);
    const t = (e.innerText || e.value || e.getAttribute('aria-label') || e.getAttribute('placeholder') || e.title || e.name || '').trim().replace(/\s+/g, ' ').slice(0, 80);
    const k = e.tagName === 'A' ? 'link' : e.tagName === 'INPUT' ? 'field ' + (e.type || 'text') : e.tagName === 'TEXTAREA' ? 'text box' : e.tagName === 'SELECT' ? 'choice' : 'button';
    return `[${i + 1}] ${k} "${t}"` + (e.tagName === 'A' && e.href ? ' -> ' + e.href.slice(0, 120) : '');
  });
  const text = (document.body ? document.body.innerText : '').replace(/\n{3,}/g, '\n\n');
  return JSON.stringify({title: document.title, url: location.href, text: text.slice(0, 12000), cut: text.length > 12000, elements: items});
})()"#;

/// Element [n]: what it is (for the Allow question), or that it is gone / a password field.
pub fn describe_js(n: i64) -> String {
    format!(r#"(() => {{ const e = document.querySelector('[data-strata-id="{n}"]');
  if (!e) return JSON.stringify({{error: 'no element [{n}] now: browser_read the page again'}});
  const label = (e.labels && e.labels[0] && e.labels[0].innerText) || e.getAttribute('aria-label') || e.getAttribute('placeholder') || e.name || e.tagName.toLowerCase();
  return JSON.stringify({{password: e.type === 'password', field: String(label).trim().slice(0, 80), host: location.host}}); }})()"#)
}

pub fn click_js(n: i64) -> String {
    format!(r#"(() => {{ const e = document.querySelector('[data-strata-id="{n}"]');
  if (!e) return JSON.stringify({{error: 'no element [{n}] now: browser_read the page again'}});
  e.scrollIntoView({{block: 'center'}}); e.click();
  return JSON.stringify({{clicked: (e.innerText || e.value || e.getAttribute('aria-label') || '').trim().slice(0, 80)}}); }})()"#)
}

pub fn type_js(n: i64, text: &str, submit: bool) -> String {
    let t = serde_json::to_string(text).unwrap_or_default();
    format!(r#"(() => {{ const e = document.querySelector('[data-strata-id="{n}"]');
  if (!e) return JSON.stringify({{error: 'no element [{n}] now: browser_read the page again'}});
  if (e.type === 'password') return JSON.stringify({{error: 'a password field: the model never types there'}});
  e.focus();
  const proto = e.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  const d = Object.getOwnPropertyDescriptor(proto, 'value');
  if (d && d.set) d.set.call(e, {t}); else e.value = {t};
  e.dispatchEvent(new Event('input', {{bubbles: true}})); e.dispatchEvent(new Event('change', {{bubbles: true}}));
  if ({submit}) {{ if (e.form && e.form.requestSubmit) e.form.requestSubmit();
    else e.dispatchEvent(new KeyboardEvent('keydown', {{key: 'Enter', code: 'Enter', keyCode: 13, bubbles: true}})); }}
  return JSON.stringify({{typed: true}}); }})()"#)
}

/// browser_read's JSON -> what the model reads.
pub fn page_text(v: &Value) -> String {
    let els: Vec<&str> = v["elements"].as_array().map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
    format!("{}\n{}\n\n{}{}\n\nElements:\n{}", v["title"].as_str().unwrap_or(""), v["url"].as_str().unwrap_or(""), v["text"].as_str().unwrap_or(""),
            if v["cut"] == true { "\n[the page goes on: cut at 12,000 characters; browser_scroll and read again for more]" } else { "" },
            if els.is_empty() { "(none)".to_string() } else { els.join("\n") })
}

/// The MCP endpoint: 127.0.0.1:<port>/mcp, with Authorization: Bearer <key>.  Returns the port.
pub fn serve(key: String, stopped: Arc<AtomicBool>, to_loop: impl Fn(Job) + Send + Sync + 'static) -> std::io::Result<u16> {
    let l = TcpListener::bind("127.0.0.1:0")?;
    let port = l.local_addr()?.port();
    let to_loop = Arc::new(to_loop);
    std::thread::spawn(move || {
        for s in l.incoming().flatten() {
            let (key, stopped, to_loop) = (key.clone(), stopped.clone(), to_loop.clone());
            std::thread::spawn(move || {
                let _ = handle(s, &key, &stopped, &*to_loop);
            });
        }
    });
    Ok(port)
}

fn handle(mut s: TcpStream, key: &str, stopped: &AtomicBool, to_loop: &dyn Fn(Job)) -> std::io::Result<()> {
    s.set_read_timeout(Some(Duration::from_secs(30)))?;
    let mut r = BufReader::new(s.try_clone()?);
    let mut line = String::new();
    r.read_line(&mut line)?;
    let (mut len, mut auth) = (0usize, String::new());
    loop {
        let mut h = String::new();
        if r.read_line(&mut h)? == 0 || h.trim().is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            match k.trim().to_ascii_lowercase().as_str() {
                "content-length" => len = v.trim().parse().unwrap_or(0),
                "authorization" => auth = v.trim().to_string(),
                _ => {}
            }
        }
    }
    let mut body = vec![0u8; len.min(1 << 20)];
    r.read_exact(&mut body)?;
    let respond = |s: &mut TcpStream, code: &str, body: &str| {
        let _ = write!(s, "HTTP/1.1 {code}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
    };
    if !line.starts_with("POST /mcp") {
        respond(&mut s, "404 Not Found", "{}");
        return Ok(());
    }
    if auth != format!("Bearer {key}") {
        respond(&mut s, "401 Unauthorized", r#"{"error":"a key is needed"}"#);
        return Ok(());
    }
    let m: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    let Some(id) = m.get("id").cloned() else {
        respond(&mut s, "202 Accepted", "");                  // a notification
        return Ok(());
    };
    let result = match m["method"].as_str().unwrap_or("") {
        "initialize" => json!({"protocolVersion": m["params"]["protocolVersion"].as_str().unwrap_or("2025-06-18"),
                               "capabilities": {"tools": {}}, "serverInfo": {"name": "strata-browser", "version": "1"}}),
        "tools/list" => json!({"tools": tools()}),
        "ping" => json!({}),
        "tools/call" => {
            let out = if stopped.load(Ordering::SeqCst) {
                Err("the user stopped the model's browsing (the browser's Stop): ask them, do not retry".to_string())
            } else {
                let (tx, rx) = mpsc::channel();
                to_loop(Job { tool: m["params"]["name"].as_str().unwrap_or("").to_string(), args: m["params"]["arguments"].clone(), reply: tx });
                // a typed text waits for the person's Allow: up to 2 minutes
                rx.recv_timeout(Duration::from_secs(150)).unwrap_or_else(|_| Err("the browser did not answer in time".into()))
            };
            match out {
                Ok(t) => json!({"content": [{"type": "text", "text": t}]}),
                Err(e) => json!({"content": [{"type": "text", "text": e}], "isError": true}),
            }
        }
        other => {
            let e = json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": format!("method not found: {other}")}});
            respond(&mut s, "200 OK", &e.to_string());
            return Ok(());
        }
    };
    respond(&mut s, "200 OK", &json!({"jsonrpc": "2.0", "id": id, "result": result}).to_string());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_endpoint_speaks_mcp_with_its_key() {
        let stopped = Arc::new(AtomicBool::new(false));
        let port = serve("k1".into(), stopped.clone(), |job: Job| {
            let _ = job.reply.send(Ok(format!("ran {}", job.tool)));
        }).unwrap();
        let post = |auth: &str, body: &str| -> (String, String) {
            let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
            write!(s, "POST /mcp HTTP/1.1\r\nHost: x\r\nAuthorization: {auth}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
            let mut out = String::new();
            s.read_to_string(&mut out).unwrap();
            let (head, body) = out.split_once("\r\n\r\n").unwrap();
            (head.lines().next().unwrap().to_string(), body.to_string())
        };
        assert!(post("Bearer nope", "{}").0.contains("401"));
        let (_, b) = post("Bearer k1", r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#);
        let names: Vec<String> = serde_json::from_str::<Value>(&b).unwrap()["result"]["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap().into()).collect();
        assert_eq!(names, ["browser_open", "browser_read", "browser_click", "browser_type", "browser_scroll", "browser_back"]);
        let (_, b) = post("Bearer k1", r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"browser_read","arguments":{}}}"#);
        assert_eq!(serde_json::from_str::<Value>(&b).unwrap()["result"]["content"][0]["text"], "ran browser_read");
        assert!(post("Bearer k1", r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#).0.contains("202"));
        stopped.store(true, Ordering::SeqCst);                 // the browser's Stop: every call refused
        let (_, b) = post("Bearer k1", r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"browser_read","arguments":{}}}"#);
        assert_eq!(serde_json::from_str::<Value>(&b).unwrap()["result"]["isError"], true);
    }

    #[test]
    fn the_page_reads_as_text_with_numbered_elements() {
        let v = json!({"title": "T", "url": "https://x/", "text": "hello", "cut": false, "elements": ["[1] link \"a\" -> https://x/a"]});
        assert_eq!(page_text(&v), "T\nhttps://x/\n\nhello\n\nElements:\n[1] link \"a\" -> https://x/a");
        assert!(type_js(3, "a\"b", true).contains(r#""a\"b""#));
    }
}
