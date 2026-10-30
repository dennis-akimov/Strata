//! The model's browser: Strata.app serves browser tools as an MCP server (Streamable HTTP, plain JSON replies) on
//! 127.0.0.1 with a key made at each start, and hands both to the server it starts (STRATA_BROWSER_MCP_URL / _KEY:
//! the server adds "browser", read-only, never saved).  The tools drive the browser pane people see (the bar's Web
//! button): the model's browsing happens in front of them, with their say: typing waits for Allow in
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
        {"name": "browser_search", "description": "Search Google in the browser beside the chat: its results page, compactly (answers Google shows, the results' titles and snippets, their links numbered for browser_click). Only for what needs the web (news, weather, prices, a site, facts you are unsure of), never for what you already know.",
         "inputSchema": {"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}}},
        {"name": "browser_open", "description": "Open a web page in the browser beside the chat (the user sees it): an address the user gave, or one from a page you read. After a browser_search only addresses from the results or pages you read are opened (guessed ones were 404s): use browser_click on a result's number. Then browser_read it.",
         "inputSchema": {"type": "object", "required": ["url"], "properties": {"url": {"type": "string"}}}},
        {"name": "browser_read", "description": "The page in the browser now (when you answer from it, link it: [its title](its address)), compactly: its main text (in parts of 3,500 characters: part 2, 3... for more) and up to 40 links, buttons and fields numbered [n] for browser_click / browser_type.",
         "inputSchema": {"type": "object", "properties": {"part": {"type": "integer", "description": "which part of the text (default 1)"}}}},
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

/// The text read at once: a page's main text in parts this long.  A read was ~5,300 tokens (12,000 characters and up to
/// 150 elements, menus and footers included): ~25 s of prompt reading each on a 16 GB M1 (~250 tok/s); now ~1,500.
pub const PART: usize = 3500;

/// The page compactly (the accessibility-tree browsers' lesson: only what can be acted on, named by its text): the main
/// part's text (main / article / role=main, without nav, header, footer, aside), part `part` of it, and up to 40 visible
/// links, buttons and fields in it (a search field anywhere), numbered (data-strata-id) for click / type.
pub fn read_js(part: i64) -> String {
    format!(r#"((part, PART) => {{
  document.querySelectorAll('[data-strata-id]').forEach((e) => e.removeAttribute('data-strata-id'));
  let root = document.querySelector('main, article, [role=main], #content, #main') || document.body;
  if (!root) return JSON.stringify({{error: 'the page has no content yet: try again in a moment'}});
  let skip = 'nav, header, footer, aside, script, style, noscript, svg, [role=navigation], [role=banner], [role=contentinfo], [aria-hidden=true]';
  // the main text without its menus: hidden for the moment innerText takes (the same task: nothing is painted)
  const grab = () => {{
    const hid = [...root.querySelectorAll(skip)], was = hid.map((e) => e.style.display);
    hid.forEach((e) => e.style.display = 'none');
    const t = (root.innerText || '').replace(/[ \t]+/g, ' ').replace(/\n\s*\n+/g, '\n\n').trim();
    hid.forEach((e, i) => e.style.display = was[i]);
    return t;
  }};
  let text = grab();
  // an empty or tiny landmark (a promo card first in the page): the whole body; content a dialog marked
  // aria-hidden (a cookie or sign-up box open over it): read it anyway
  if (text.length < 300 && root !== document.body) {{ root = document.body; text = grab(); }}
  if (text.length < 300) {{ skip = skip.replace(', [aria-hidden=true]', ''); text = grab(); }}
  const parts = Math.max(1, Math.ceil(text.length / PART)), p = Math.min(Math.max(1, part), parts);
  const sel = 'a[href],button,input:not([type=hidden]),textarea,select,[role=button],[role=link]';
  const cand = [...new Set([...root.querySelectorAll(sel), ...document.querySelectorAll('input[type=search],input[name=q],textarea[name=q]')])];
  const items = [], seen = new Set(), hrefs = [];
  for (const e of cand) {{
    if (items.length >= 40) break;
    if (e.closest(skip) && !e.matches('input,textarea')) continue;
    const r = e.getBoundingClientRect(); if (!(r.width > 0 && r.height > 0)) continue;
    const name = (e.innerText || e.value || e.getAttribute('aria-label') || e.getAttribute('placeholder') || e.title || e.name || '').trim().replace(/\s+/g, ' ').slice(0, 60);
    const kind = e.tagName === 'A' ? 'link' : e.tagName === 'INPUT' ? 'field ' + (e.type || 'text') : e.tagName === 'TEXTAREA' ? 'text box' : e.tagName === 'SELECT' ? 'choice' : 'button';
    const key = kind + name + (e.href || '');
    if ((!name && kind === 'link') || seen.has(key)) continue;
    seen.add(key);
    e.setAttribute('data-strata-id', items.length + 1);
    if (e.tagName === 'A' && e.href) hrefs.push(e.href);
    items.push(`[${{items.length + 1}}] ${{kind}} "${{name}}"` + (e.tagName === 'A' && e.href ? ' -> ' + e.href.replace(/^https?:\/\//, '').slice(0, 80) : ''));
  }}
  // every link on the page, and the sites the text names ("www.bbc.com › weather": how Google's plain results show
  // them), for browser_open's check: its site must be one a page read showed.  Google's links are redirects
  // (/url?q=<the address>, or an opaque token): the address, when there is one, too.
  document.querySelectorAll('a[href]').forEach((a) => {{
    if (hrefs.length >= 400 || !a.href.startsWith('http')) return;
    hrefs.push(a.href);
    try {{ const u = new URL(a.href), q = u.searchParams.get('q') || u.searchParams.get('url');
      if (u.pathname === '/url' && q && /^https?:/.test(q)) hrefs.push(q); }} catch (e) {{}}
  }});
  const sites = [...new Set((text.match(/\b(?:[a-z0-9-]+\.)+(?:com|org|net|gov|edu|int|info|io|co|ai|app|dev|news|tv|me|us|uk|ca|au|de|fr|es|it|nl|eu|jp|ru|ch|be|at|se|no|dk|fi|pl|pt|ie|in|br|mx|nz)\b/gi) || []).map((h) => h.toLowerCase()))].slice(0, 200);
  const d = new Date(), today = d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0') + '-' + String(d.getDate()).padStart(2, '0');
  return JSON.stringify({{title: document.title, url: location.href, text: text.slice((p - 1) * PART, p * PART), part: p, parts, elements: items, hrefs, sites, today}});
}})({part}, {PART})"#)
}

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
    let (p, n) = (v["part"].as_i64().unwrap_or(1), v["parts"].as_i64().unwrap_or(1));
    let text = v["text"].as_str().unwrap_or("");
    let text = if text.is_empty() { "(no text on this page: a bot check, or still loading. browser_read it once more, or use another result)" } else { text };
    format!("{}\n{}\n\n{}{}\n\nElements:\n{}", v["title"].as_str().unwrap_or(""), v["url"].as_str().unwrap_or(""), text,
            if n > 1 { format!("\n[part {p} of {n}: browser_read with part {} for more]", (p + 1).min(n)) } else { String::new() },
            if els.is_empty() { "(none)".to_string() } else { els.join("\n") })
}

/// A read the model already has (the search's results read again, the same part twice): one line instead of the
/// page again; a round of context and prefill saved.
pub fn once(last: &mut String, text: String) -> String {
    if *last == text {
        let head: Vec<&str> = text.lines().take(2).collect();
        return format!("unchanged: the same text as your last read of this page ({}). Answer from it, use browser_read with another part, or open another page.", head.join(" "));
    }
    *last = text.clone();
    text
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
        assert_eq!(names, ["browser_search", "browser_open", "browser_read", "browser_click", "browser_type", "browser_scroll", "browser_back"]);
        let (_, b) = post("Bearer k1", r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"browser_read","arguments":{}}}"#);
        assert_eq!(serde_json::from_str::<Value>(&b).unwrap()["result"]["content"][0]["text"], "ran browser_read");
        assert!(post("Bearer k1", r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#).0.contains("202"));
        stopped.store(true, Ordering::SeqCst);                 // the browser's Stop: every call refused
        let (_, b) = post("Bearer k1", r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"browser_read","arguments":{}}}"#);
        assert_eq!(serde_json::from_str::<Value>(&b).unwrap()["result"]["isError"], true);
    }

    #[test]
    fn the_page_reads_as_text_with_numbered_elements() {
        let v = json!({"title": "T", "url": "https://x/", "text": "hello", "part": 1, "parts": 1, "elements": ["[1] link \"a\" -> x/a"]});
        assert_eq!(page_text(&v), "T\nhttps://x/\n\nhello\n\nElements:\n[1] link \"a\" -> x/a");
        let v = json!({"title": "T", "url": "u", "text": "t", "part": 1, "parts": 3, "elements": []});
        assert!(page_text(&v).contains("[part 1 of 3: browser_read with part 2 for more]"));
        assert!(read_js(2).ends_with("})(2, 3500)"));
        assert!(type_js(3, "a\"b", true).contains(r#""a\"b""#));
        assert!(page_text(&json!({"title": "T", "url": "u", "text": ""})).contains("(no text on this page"));
    }

    #[test]
    fn a_read_the_model_has_is_one_line() {
        let mut last = String::new();
        assert_eq!(once(&mut last, "T\nu\n\nbody".into()), "T\nu\n\nbody");
        let again = once(&mut last, "T\nu\n\nbody".into());
        assert!(again.starts_with("unchanged") && again.contains("(T u)"), "{again}");
        assert_eq!(once(&mut last, "T\nu\n\nother".into()), "T\nu\n\nother");
    }
}

/// The page scripts in a real browser: headless Chrome (the same JavaScript engine family the web view runs is not
/// needed: these scripts use only DOM APIs both have), driven over the DevTools protocol.  Skipped when there is no
/// Chrome (CHROME=<path> to name one).
#[cfg(test)]
mod in_a_browser {
    use super::*;
    use std::process::{Child, Command, Stdio};

    struct Browser {
        child: Child,
        ws: tungstenite::WebSocket<tungstenite::stream::MaybeTlsStream<TcpStream>>,
        id: u64,
        dir: std::path::PathBuf,
    }

    impl Drop for Browser {
        fn drop(&mut self) {
            let _ = self.child.kill();
            let _ = self.child.wait();
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    impl Browser {
        fn start() -> Option<Browser> {
            let exe = std::env::var("CHROME").unwrap_or_else(|_| "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome".into());
            if !std::path::Path::new(&exe).exists() {
                eprintln!("no Chrome at {exe}: the page scripts' browser tests are skipped");
                return None;
            }
            let dir = std::env::temp_dir().join(format!("strata-js-test-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
            std::fs::create_dir_all(&dir).ok()?;
            let child = Command::new(&exe).args(["--headless=new", "--remote-debugging-port=0", "--no-first-run", "--no-default-browser-check"])
                .arg(format!("--user-data-dir={}", dir.display())).arg("about:blank").stdout(Stdio::null()).stderr(Stdio::null()).spawn().ok()?;
            // Chrome says its port in DevToolsActivePort
            let mut port = None;
            for _ in 0..100 {
                if let Some(p) = std::fs::read_to_string(dir.join("DevToolsActivePort")).ok().and_then(|t| t.lines().next()?.parse::<u16>().ok()) {
                    port = Some(p);
                    break;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            let port = port?;
            let mut s = TcpStream::connect(("127.0.0.1", port)).ok()?;
            write!(s, "PUT /json/new?about:blank HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n").ok()?;
            // Chrome keeps the connection open: read its headers, then Content-Length bytes (a timeout, not a hang)
            s.set_read_timeout(Some(Duration::from_secs(10))).ok()?;
            let mut r = BufReader::new(s);
            let mut len = 0usize;
            loop {
                let mut h = String::new();
                if r.read_line(&mut h).ok()? == 0 || h.trim().is_empty() {
                    break;
                }
                if let Some((k, v)) = h.split_once(':') {
                    if k.eq_ignore_ascii_case("content-length") {
                        len = v.trim().parse().unwrap_or(0);
                    }
                }
            }
            let mut body = vec![0u8; len];
            r.read_exact(&mut body).ok()?;
            let v: Value = serde_json::from_slice(&body).ok()?;
            let (ws, _) = tungstenite::connect(v["webSocketDebuggerUrl"].as_str()?).ok()?;
            if let tungstenite::stream::MaybeTlsStream::Plain(t) = ws.get_ref() {
                let _ = t.set_read_timeout(Some(Duration::from_secs(20))); // a hang fails the test, not the run
            }
            Some(Browser { child, ws, id: 0, dir })
        }

        fn cdp(&mut self, method: &str, params: Value) -> Value {
            self.id += 1;
            self.ws.send(tungstenite::Message::Text(json!({"id": self.id, "method": method, "params": params}).to_string().into())).unwrap();
            loop {
                match self.ws.read() {
                    Ok(tungstenite::Message::Text(t)) => {
                        let v: Value = serde_json::from_str(&t).unwrap_or(Value::Null);
                        if v["id"] == self.id {
                            return v["result"].clone();
                        }
                    }
                    Ok(_) => {}
                    Err(e) => panic!("{method}: no answer from Chrome ({e})"),
                }
            }
        }

        /// A page from this HTML (a data: URL), loaded.
        fn page(&mut self, html: &str) {
            let url = format!("data:text/html;charset=utf-8,{}", html.bytes().map(|b| if b.is_ascii_alphanumeric() { (b as char).to_string() } else { format!("%{b:02X}") }).collect::<String>());
            self.cdp("Page.navigate", json!({"url": url}));
            for _ in 0..50 {
                if self.eval("document.readyState") == "complete" {
                    return;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }

        fn eval(&mut self, js: &str) -> Value {
            self.cdp("Runtime.evaluate", json!({"expression": js, "returnByValue": true}))["result"]["value"].clone()
        }

        /// A page script's JSON string, parsed (as the app's js_value does).
        fn json(&mut self, js: &str) -> Value {
            serde_json::from_str(self.eval(js).as_str().unwrap_or("null")).unwrap_or(Value::Null)
        }
    }

    const RESULTS: &str = r#"<html><head><title>paris weather - Google Search</title></head><body>
<div><a href="https://www.google.com/search?q=paris+weather&tbm=nws">News</a> <a href="https://www.google.com/search?q=paris+weather&tbm=isch">Images</a></div>
<div id="main">
<div>Paris, France / Weather 51°F Clear</div>
<div><a href="https://www.google.com/url?q=CAESXgHrOzAVqG3cU4ZYSa1NAsS98&sa=U">Paris - BBC Weather</a><div>www.bbc.com › weather</div><div>Today, light rain.</div></div>
<div><a href="https://www.google.com/url?q=https://www.theweathernetwork.com/en/city/fr/paris&sa=U">Paris 7 Days Weather</a><div>www.theweathernetwork.com › city</div></div>
<input name="q" value="paris weather">
</div></body></html>"#;

    const ARTICLE: &str = r#"<html><head><title>Rayleigh scattering</title></head><body>
<header><a href="/">Home</a><a href="/login">Log in</a></header><nav><a href="/menu">Menu item</a> navigation words</nav>
<main><h1>Rayleigh scattering</h1><p>The sky is blue because short wavelengths scatter more.</p>
<p>PARTSTART %LONG% PARTEND</p><a href="https://en.wikipedia.org/wiki/Light">Light</a><button>Show more</button></main>
<aside>Sidebar words</aside><footer>Footer words <a href="/privacy">Privacy</a></footer></body></html>"#;

    // an events page: a promo card is the first <article>, and the open cookie dialog marked the content aria-hidden
    const DIALOG: &str = r#"<html><body><nav>Menu words</nav><article>Ad</article><div aria-hidden="true">
<h1>Events today</h1><p>Halloween bar crawl, 7 PM, Gaslamp Quarter. Ghost Light Masquerade, 7:30 PM. %LONG%</p>
<a href="https://x.test/e/1">Bar crawl</a></div><div role="dialog" aria-modal="true">We use cookies <button>Accept</button></div></body></html>"#;

    const FORM: &str = r#"<html><body><main><form id="f" onsubmit="window.sent = (window.sent || 0) + 1; return false;">
<label for="u">Email</label><input id="u" name="email"><input type="password" name="pw" aria-label="Password">
<textarea name="note" placeholder="Note"></textarea><button type="button" onclick="window.clicked = 'yes'">Go</button>
</form></main><script>document.getElementById('u').addEventListener('input', () => window.inputs = (window.inputs || 0) + 1)</script></body></html>"#;

    #[test]
    fn page_scripts_read_click_and_type_in_a_browser() {
        let Some(mut b) = Browser::start() else { return };

        // Google's plain results: redirect links unwrapped when they hold the address, the sites the text names
        b.page(RESULTS);
        let v = b.json(&read_js(1));
        let hrefs: Vec<String> = v["hrefs"].as_array().unwrap().iter().map(|h| h.as_str().unwrap().to_string()).collect();
        assert!(hrefs.iter().any(|h| h == "https://www.theweathernetwork.com/en/city/fr/paris"), "{hrefs:?}");
        let sites: Vec<&str> = v["sites"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
        assert!(sites.contains(&"www.bbc.com") && sites.contains(&"www.theweathernetwork.com"), "{sites:?}");
        assert!(v["text"].as_str().unwrap().contains("51°F Clear"));
        assert!(v["elements"].as_array().unwrap().iter().any(|e| e.as_str().unwrap().contains("field text \"paris weather\"")));
        assert!(v["today"].as_str().unwrap().len() == 10);
        assert!(page_text(&v).contains("Elements:"));

        // an article: its main part only (no header, nav, aside, footer), in parts, elements numbered on the page
        let long = "word ".repeat(1500); // 7,500 characters: 3 parts
        b.page(&ARTICLE.replace("%LONG%", &long));
        let v = b.json(&read_js(1));
        let t = v["text"].as_str().unwrap();
        assert!(t.starts_with("Rayleigh scattering") && t.contains("short wavelengths"), "{t:.120}");
        for gone in ["navigation words", "Sidebar words", "Footer words", "Log in"] {
            assert!(!t.contains(gone), "{gone} in the text");
        }
        assert_eq!((v["part"].as_i64(), v["parts"].as_i64()), (Some(1), Some(3)));
        let names: Vec<&str> = v["elements"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
        assert_eq!(names, ["[1] link \"Light\" -> en.wikipedia.org/wiki/Light", "[2] button \"Show more\""]);
        assert_eq!(b.eval("document.querySelectorAll('[data-strata-id]').length"), json!(2));
        let v3 = b.json(&read_js(3));
        assert!(v3["text"].as_str().unwrap().contains("PARTEND") && v3["part"] == 3);
        assert!(b.json(&read_js(9))["part"] == 3, "a part past the end reads the last");
        let c = b.json(&click_js(2));
        assert_eq!(c["clicked"], "Show more");
        assert!(b.json(&click_js(7))["error"].as_str().unwrap().contains("no element [7]"));

        // the content behind a small landmark and an open dialog: still read, not an empty page
        b.page(&DIALOG.replace("%LONG%", &"more ".repeat(60)));
        let v = b.json(&read_js(1));
        let t = v["text"].as_str().unwrap();
        assert!(t.contains("Halloween bar crawl") && t.contains("Accept") && !t.contains("Menu words"), "{t:.200}");
        assert!(v["elements"].as_array().unwrap().iter().any(|e| e.as_str().unwrap().contains("Bar crawl")));
        // an empty <main> (the content beside it): the body, still without its menus
        b.page(&format!("<html><body><header>Site menu</header><main></main><div><h1>Calendar</h1><p>{}</p></div></body></html>", "Concert tonight. ".repeat(30)));
        let t = b.json(&read_js(1))["text"].as_str().unwrap().to_string();
        assert!(t.starts_with("Calendar") && !t.contains("Site menu"), "{t:.120}");

        // a form: what a field is (for the Allow), a password refused, typing fires input, submit sends the form
        b.page(FORM);
        b.json(&read_js(1));
        let field = |b: &mut Browser, name: &str| -> i64 {
            b.json(&read_js(1))["elements"].as_array().unwrap().iter().filter_map(Value::as_str)
                .find(|e| e.contains(name)).and_then(|e| e[1..e.find(']').unwrap()].parse().ok()).unwrap()
        };
        let email = field(&mut b, "field text");
        let pw = field(&mut b, "field password");
        let d = b.json(&describe_js(email));
        assert_eq!((d["password"].clone(), d["field"].clone()), (json!(false), json!("Email")));
        assert_eq!(b.json(&describe_js(pw))["password"], true);
        assert!(b.json(&type_js(pw, "secret", false))["error"].as_str().unwrap().contains("password"));
        assert_eq!(b.eval("document.querySelector('[name=pw]').value"), json!(""));
        assert_eq!(b.json(&type_js(email, "a\"b@example.com", true))["typed"], true);
        assert_eq!(b.eval("document.getElementById('u').value"), json!("a\"b@example.com"));
        assert_eq!(b.eval("window.inputs"), json!(1));
        assert_eq!(b.eval("window.sent"), json!(1));
        let go = field(&mut b, "button \"Go\"");
        b.json(&click_js(go));
        assert_eq!(b.eval("window.clicked"), json!("yes"));
    }
}
