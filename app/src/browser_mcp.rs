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

#[derive(Clone)]
pub struct Job {
    pub tool: String,
    pub args: Value,
    pub reply: mpsc::Sender<Result<String, String>>,
}

/// A caller's arguments without the app's own job flags (__allowed, __loaded, __phase, __frame, ...): only the app
/// sets those (a call with "__allowed" of its own skipped the trash's Allow: found by a codex review).
pub fn caller_args(args: &Value) -> Value {
    match args.as_object() {
        Some(m) => Value::Object(m.iter().filter(|(k, _)| !k.starts_with("__")).map(|(k, v)| (k.clone(), v.clone())).collect()),
        None => json!({}),
    }
}

/// The tools, as tools/list gives them.
pub fn tools() -> Value {
    let mut t = json!([
        {"name": "browser_search", "description": "Search Google in the browser beside the chat: its results page, compactly (answers Google shows, the results' titles and snippets, their links numbered for browser_click). Only for what needs the web (news, weather, prices, a site, facts you are unsure of), never for what you already know.",
         "inputSchema": {"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}}},
        {"name": "browser_open", "description": "Open a web page in the browser beside the chat (the user sees it): an address the user gave, or one from a page you read. After a browser_search only addresses from the results or pages you read are opened (guessed ones were 404s): use browser_click on a result's number. It answers with the page's text (browser_read for its other parts).",
         "inputSchema": {"type": "object", "required": ["url"], "properties": {"url": {"type": "string"}}}},
        {"name": "browser_read", "description": "The page in the browser now (when you answer from it, link it: [its title](its address)), compactly: its main text (in parts of 3,500 characters: part 2, 3... for more) and up to 40 links, buttons and fields numbered [n] for browser_click / browser_type.",
         "inputSchema": {"type": "object", "properties": {"part": {"type": "integer", "description": "which part of the text (default 1)"}}}},
        {"name": "browser_click", "description": "Click element [n] of the last page you read (a link, a button), with the mouse as a person would. It answers with the page after the click. On a site, get around by its search field and links (browser_type into its search box with submit, then browser_click a result) rather than by making up addresses.",
         "inputSchema": {"type": "object", "required": ["element"], "properties": {"element": {"type": "integer"}}}},
        {"name": "browser_type", "description": "Type text into field [n] of the last browser_read (the user is asked to allow it; never a password). submit: press Enter / send its form (it then answers with the page that follows).",
         "inputSchema": {"type": "object", "required": ["element", "text"], "properties": {"element": {"type": "integer"}, "text": {"type": "string"}, "submit": {"type": "boolean"}}}},
        {"name": "browser_scroll", "description": "Scroll the page down (or up) by about a screen, for content it loads as you scroll; it answers with the page then (for a long page's text, browser_read its next part instead).",
         "inputSchema": {"type": "object", "properties": {"direction": {"type": "string", "enum": ["down", "up"]}}}},
        {"name": "browser_back", "description": "Go back to the previous page. It answers with that page's text.",
         "inputSchema": {"type": "object", "properties": {}}}
    ]);
    // Google Sheets and My Maps through the same pane (google.rs)
    if let (Some(all), Some(g)) = (t.as_array_mut(), crate::google::tools().as_array()) {
        all.extend(g.iter().cloned());
    }
    t
}

/// The text read at once: a page's main text in parts this long.  A read was ~5,300 tokens (12,000 characters and up to
/// 150 elements, menus and footers included): ~25 s of prompt reading each on a 16 GB M1 (~250 tok/s); now ~1,500.
pub const PART: usize = 3500;

/// The page compactly (the accessibility-tree browsers' lesson: only what can be acted on, named by its text): the main
/// part's text (main / article / role=main, without nav, header, footer, aside), part `part` of it, and up to 40 visible
/// links, buttons and fields in it (a search field anywhere), numbered (data-strata-id) for click / type.
pub fn read_js(part: i64) -> String {
    format!("({READ_FN})({part}, {PART})")
}

/// The page read as a function of (part, PART): what read_js runs in the page, and the picker bridge in its frame.
pub const READ_FN: &str = r#"(part, PART) => {
  document.querySelectorAll('[data-strata-id]').forEach((e) => e.removeAttribute('data-strata-id'));
  let root = document.querySelector('main, article, [role=main], #content, #main') || document.body;
  if (!root) return JSON.stringify({error: 'the page has no content yet: try again in a moment'});
  let skip = 'nav, header, footer, aside, script, style, noscript, svg, [role=navigation], [role=banner], [role=contentinfo], [aria-hidden=true]';
  // the main text without its menus: hidden for the moment innerText takes (the same task: nothing is painted)
  const grab = () => {
    const hid = [...root.querySelectorAll(skip)], was = hid.map((e) => e.style.display);
    hid.forEach((e) => e.style.display = 'none');
    const t = (root.innerText || '').replace(/[ \t]+/g, ' ').replace(/\n\s*\n+/g, '\n\n').trim();
    hid.forEach((e, i) => e.style.display = was[i]);
    return t;
  };
  let text = grab();
  // an empty or tiny landmark (a promo card first in the page): the whole body; content a dialog marked
  // aria-hidden (a cookie or sign-up box open over it): read it anyway
  if (text.length < 300 && root !== document.body) { root = document.body; text = grab(); }
  if (text.length < 300) { skip = skip.replace(', [aria-hidden=true]', ''); text = grab(); }
  const parts = Math.max(1, Math.ceil(text.length / PART)), p = Math.min(Math.max(1, part), parts);
  const sel = 'a[href],button,input:not([type=hidden]),textarea,select,[role=button],[role=link],[role=option],[role=gridcell],[role=menuitem],[role=tab],[role=checkbox],[role=radio]';
  const cand = [...new Set([...root.querySelectorAll(sel), ...document.querySelectorAll('input[type=search],input[name=q],textarea[name=q]')])];
  const items = [], seen = new Set(), hrefs = [];
  for (const e of cand) {
    if (items.length >= 40) break;
    if (e.closest(skip) && !e.matches('input,textarea')) continue;
    const r = e.getBoundingClientRect(); if (!(r.width > 0 && r.height > 0)) continue;
    // a label elsewhere (aria-labelledby): My Maps' column boxes are named only so (unnamed, all but one were dropped as duplicates)
    const by = e.getAttribute('aria-labelledby'), lab = by ? by.split(/\s+/).map((id) => (document.getElementById(id) || {}).textContent || '').join(' ') : '';
    const name = (e.innerText || e.value || e.getAttribute('aria-label') || lab || e.getAttribute('placeholder') || e.title || e.name || '').trim().replace(/\s+/g, ' ').slice(0, 60);
    const role = e.getAttribute('role'), on = e.getAttribute('aria-checked') === 'true' || e.getAttribute('aria-selected') === 'true';
    const kind = e.tagName === 'A' ? 'link' : e.tagName === 'INPUT' ? 'field ' + (e.type || 'text') : e.tagName === 'TEXTAREA' ? 'text box' : e.tagName === 'SELECT' ? 'choice'
      : ['option', 'gridcell', 'menuitem', 'tab', 'checkbox', 'radio'].includes(role) ? role + (on ? ' (on)' : '') : 'button';
    const key = kind + name + (e.href || '');
    if ((!name && kind === 'link') || seen.has(key)) continue;
    seen.add(key);
    e.setAttribute('data-strata-id', items.length + 1);
    if (e.tagName === 'A' && e.href) hrefs.push(e.href);
    items.push(`[${items.length + 1}] ${kind} "${name}"` + (e.tagName === 'A' && e.href ? ' -> ' + e.href.replace(/^https?:\/\//, '').slice(0, 80) : ''));
  }
  // every link on the page, and the sites the text names ("www.bbc.com › weather": how Google's plain results show
  // them), for browser_open's check: its site must be one a page read showed.  Google's links are redirects
  // (/url?q=<the address>, or an opaque token): the address, when there is one, too.
  document.querySelectorAll('a[href]').forEach((a) => {
    if (hrefs.length >= 400 || !a.href.startsWith('http')) return;
    hrefs.push(a.href);
    try { const u = new URL(a.href), q = u.searchParams.get('q') || u.searchParams.get('url');
      if (u.pathname === '/url' && q && /^https?:/.test(q)) hrefs.push(q); } catch (e) {}
  });
  const sites = [...new Set((text.match(/\b(?:[a-z0-9-]+\.)+(?:com|org|net|gov|edu|int|info|io|co|ai|app|dev|news|tv|me|us|uk|ca|au|de|fr|es|it|nl|eu|jp|ru|ch|be|at|se|no|dk|fi|pl|pt|ie|in|br|mx|nz)\b/gi) || []).map((h) => h.toLowerCase()))].slice(0, 200);
  const d = new Date(), today = d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0') + '-' + String(d.getDate()).padStart(2, '0');
  return JSON.stringify({title: document.title, url: location.href, text: text.slice((p - 1) * PART, p * PART), part: p, parts, elements: items, hrefs, sites, today});
}"#;

/// Element [n]: what it is (for the Allow question), or that it is gone / a password field.
pub fn describe_js(n: i64) -> String {
    format!("({DESCRIBE_FN})({n})")
}

pub const DESCRIBE_FN: &str = r#"(n) => { const e = document.querySelector('[data-strata-id="' + n + '"]');
  if (!e) return JSON.stringify({error: 'no element [' + n + '] now: browser_read the page again'});
  const label = (e.labels && e.labels[0] && e.labels[0].innerText) || e.getAttribute('aria-label') || e.getAttribute('placeholder') || e.name || e.tagName.toLowerCase();
  return JSON.stringify({password: e.type === 'password', field: String(label).trim().slice(0, 80), host: location.host}); }"#;

pub fn click_js(n: i64) -> String {
    format!("({CLICK_FN})({n})")
}

pub const CLICK_FN: &str = r#"(n) => { const e = document.querySelector('[data-strata-id="' + n + '"]');
  if (!e) return JSON.stringify({error: 'no element [' + n + '] now: browser_read the page again'});
  e.scrollIntoView({block: 'center'});
  // a mouse's press and release first: Google's menus and buttons act on those (a bare click() opened nothing)
  const r = e.getBoundingClientRect();
  for (const t of ['mousedown', 'mouseup']) e.dispatchEvent(new MouseEvent(t, {bubbles: true, cancelable: true, view: window, clientX: r.x + r.width / 2, clientY: r.y + r.height / 2, button: 0}));
  e.click();
  return JSON.stringify({clicked: (e.innerText || e.value || e.getAttribute('aria-label') || '').trim().slice(0, 80)}); }"#;

/// Element [n] brought into view: its centre (CSS pixels from the page's top left), for a real click there; {cover}
/// when something else is on top (a cookie banner): that is what a mouse would hit.
pub fn target_js(n: i64) -> String {
    format!("({TARGET_FN})({n})")
}

pub const TARGET_FN: &str = r#"(n) => { const e = document.querySelector('[data-strata-id="' + n + '"]');
  if (!e) return JSON.stringify({error: 'no element [' + n + '] now: browser_read the page again'});
  e.scrollIntoView({block: 'center', inline: 'center'});
  const r = e.getBoundingClientRect(), x = r.x + r.width / 2, y = r.y + r.height / 2;
  if (!(r.width > 0 && r.height > 0) || x < 0 || y < 0 || x > innerWidth || y > innerHeight) return JSON.stringify({hidden: true});
  const top = document.elementFromPoint(x, y);
  const cover = top && top !== e && !e.contains(top) && !top.contains(e) ? (top.innerText || top.getAttribute('aria-label') || top.tagName).trim().slice(0, 60) : null;
  return JSON.stringify({x, y, cover, name: (e.innerText || e.value || e.getAttribute('aria-label') || '').trim().slice(0, 80)}); }"#;

/// Field [n] focused with its text selected, for the typing that follows (it replaces what was there).
pub fn focus_js(n: i64) -> String {
    format!(r#"(() => {{ const e = document.querySelector('[data-strata-id="{n}"]');
  if (!e) return JSON.stringify({{error: 'no element [{n}] now: browser_read the page again'}});
  if (e.type === 'password') return JSON.stringify({{error: 'a password field: the model never types there'}});
  e.scrollIntoView({{block: 'center'}}); e.focus();
  if (e.select) e.select(); else if (e.isContentEditable) document.execCommand('selectAll');
  return JSON.stringify({{focused: document.activeElement === e}}); }})()"#)
}

/// The words of a site's "prove you are a person" page (PerimeterX's Press & Hold on Zillow, Cloudflare, captchas).
pub fn person_check(title: &str, text: &str) -> bool {
    let t = format!("{title}\n{}", text.chars().take(3000).collect::<String>()).to_lowercase();
    ["press & hold", "press and hold", "verify you are human", "verify you are a human", "verify you're human", "are you a robot",
     "captcha", "unusual traffic", "access to this page has been denied", "just a moment...", "performing security verification"]
        .iter().any(|w| t.contains(w))
}

// the page's own typing, kept for the tests (the app types with the keyboard: native_input; the picker bridge: TYPE_FN)
#[cfg_attr(not(test), allow(dead_code))]
pub fn type_js(n: i64, text: &str, submit: bool) -> String {
    format!("({TYPE_FN})({n}, {}, {submit})", serde_json::to_string(text).unwrap_or_default())
}

pub const TYPE_FN: &str = r#"(n, t, submit) => { const e = document.querySelector('[data-strata-id="' + n + '"]');
  if (!e) return JSON.stringify({error: 'no element [' + n + '] now: browser_read the page again'});
  if (e.type === 'password') return JSON.stringify({error: 'a password field: the model never types there'});
  e.focus();
  const proto = e.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  const d = Object.getOwnPropertyDescriptor(proto, 'value');
  if (d && d.set) d.set.call(e, t); else e.value = t;
  e.dispatchEvent(new Event('input', {bubbles: true})); e.dispatchEvent(new Event('change', {bubbles: true}));
  if (submit) { if (e.form && e.form.requestSubmit) e.form.requestSubmit();
    else e.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', code: 'Enter', keyCode: 13, bubbles: true})); }
  return JSON.stringify({typed: true}); }"#;

// ------------------------------------------------------------------ Google's file picker (My Maps' Import)
// My Maps (www.google.com/maps/d/) opens Google's file picker as a frame of docs.google.com: another origin, out of the
// page's scripts' reach.  A script in every frame (the web view's initialization script) wakes only there and runs the
// read / click / type above for a message from My Maps' page that carries this start's token; the reply goes back the
// same way.  While the picker is open, browser_read / click / type act in it.

/// Where the picker opens over the page (My Maps).
pub fn picker_host(url: &str) -> bool {
    url.starts_with("https://www.google.com/maps/d/")
}

/// The script for every frame: the picker's end of the bridge.  An action answers with the picker read a moment after.
pub fn frame_bridge_js(token: &str) -> String {
    format!(r#"(() => {{
  if (location.origin !== 'https://docs.google.com' || !location.pathname.startsWith('/picker')) return;
  const TOKEN = '{token}', PART = {PART};
  const read = {READ_FN}, ops = {{read, describe: {DESCRIBE_FN}, click: {CLICK_FN}, type: {TYPE_FN}}};
  addEventListener('message', (e) => {{
    const d = e.data;
    if (e.origin !== 'https://www.google.com' || !d || d.strata !== TOKEN || !ops[d.op]) return;
    const reply = (out) => e.source.postMessage({{strataReply: TOKEN, id: d.id, out}}, e.origin);
    let out;
    try {{ out = ops[d.op](...d.args); }} catch (err) {{ out = JSON.stringify({{error: String(err)}}); }}
    if (d.op === 'click' || d.op === 'type') setTimeout(() => reply(JSON.stringify({{acted: JSON.parse(out), read: JSON.parse(read(1, PART))}})), 1500);
    else reply(out);
  }});
}})();"#)
}

/// My Maps' end: the op to the open picker ({{sent}}), or {{none}} when no picker is open (the page acts instead).
pub fn frame_send_js(token: &str, op: &str, args: &Value, id: u64) -> String {
    format!(r#"(() => {{
  if (!location.href.startsWith('https://www.google.com/maps/d/')) return JSON.stringify({{none: true}});
  const f = [...document.querySelectorAll('iframe')].find((f) => (f.src || '').startsWith('https://docs.google.com/picker') && f.getBoundingClientRect().width > 0);
  if (!f) return JSON.stringify({{none: true}});
  if (!window.__strataFrame) {{
    window.__strataFrame = {{}};
    addEventListener('message', (e) => {{ if (e.origin === 'https://docs.google.com' && e.data && e.data.strataReply === '{token}') window.__strataFrame[e.data.id] = e.data.out; }});
  }}
  f.contentWindow.postMessage({{strata: '{token}', op: {op}, args: {args}, id: {id}}}, 'https://docs.google.com');
  return JSON.stringify({{sent: true}});
}})()"#, op = serde_json::to_string(op).unwrap_or_default())
}

/// The picker's answer to op `id`, once it came (null before).
pub fn frame_collect_js(id: u64) -> String {
    format!("(() => {{ const o = window.__strataFrame && window.__strataFrame[{id}]; return o === undefined ? 'null' : o; }})()")
}

/// browser_read's JSON -> what the model reads.
pub fn page_text(v: &Value) -> String {
    let check = if person_check(v["title"].as_str().unwrap_or(""), v["text"].as_str().unwrap_or("")) {
        "[The site shows a check that a person is here. Do not try to pass it: ask the user to do it in the browser beside the chat, then browser_read again.]\n"
    } else {
        ""
    };
    let els: Vec<&str> = v["elements"].as_array().map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
    let (p, n) = (v["part"].as_i64().unwrap_or(1), v["parts"].as_i64().unwrap_or(1));
    let text = v["text"].as_str().unwrap_or("");
    let text = if text.is_empty() { "(no text on this page: a bot check, or still loading. browser_read it once more, or use another result)" } else { text };
    format!("{check}{}\n{}\n\n{}{}\n\nElements:\n{}", v["title"].as_str().unwrap_or(""), v["url"].as_str().unwrap_or(""), text,
            if n > 1 { format!("\n[part {p} of {n}: browser_read with part {} for more]", (p + 1).min(n)) } else { String::new() },
            if els.is_empty() { "(none)".to_string() } else { els.join("\n") })
}

/// What browser_search, browser_open and browser_back answer with: the page read, under a note on using it.
pub fn loaded_text(v: &Value, search: bool) -> String {
    let today = v["today"].as_str().unwrap_or("?");
    let note = if search {
        format!("[Read this first. Today is {today}: prefer the newest results, and put the year in a query when it matters. Answer the user from these results when they hold the answer (Google's own answer is at the top), and link each source you use as [title](address). Only if they do not: browser_click a result's number.]")
    } else {
        format!("[Today is {today}. Answer only from what this page says, and link it as [title](address).]")
    };
    format!("{note}\n\n{}", page_text(v))
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
                to_loop(Job { tool: m["params"]["name"].as_str().unwrap_or("").to_string(), args: caller_args(&m["params"]["arguments"]), reply: tx });
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
        assert_eq!(names, ["browser_search", "browser_open", "browser_read", "browser_click", "browser_type", "browser_scroll", "browser_back",
                           "sheets_create", "sheets_read", "sheets_append", "sheets_write", "sheets_delete_rows", "sheets_clear", "sheets_trash",
                           "maps_create", "maps_import_sheet", "maps_trash"]);
        let (_, b) = post("Bearer k1", r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"browser_read","arguments":{}}}"#);
        assert_eq!(serde_json::from_str::<Value>(&b).unwrap()["result"]["content"][0]["text"], "ran browser_read");
        assert!(post("Bearer k1", r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#).0.contains("202"));
        assert_eq!(caller_args(&json!({"url": "u", "__allowed": true, "__phase": 2})), json!({"url": "u"}), "the app's own flags never come from a caller");
        assert_eq!(caller_args(&Value::Null), json!({}));
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
        // a "prove you are a person" page is said as such, for the user to answer (never passed by the model)
        let check = page_text(&json!({"title": "Access to this page has been denied", "url": "https://www.zillow.com/", "text": "Press & Hold to confirm you are a human (and not a bot)."}));
        assert!(check.starts_with("[The site shows a check that a person is here. Do not try to pass it"), "{check}");
        assert!(!page_text(&json!({"title": "Studios", "url": "u", "text": "Studio apartments in San Diego"})).starts_with("[The site"));
        assert!(person_check("Just a moment...", "") && person_check("", "Our systems have detected unusual traffic") && !person_check("Zillow", "3 studios"));
    }

    #[test]
    fn an_opened_page_answers_with_its_text() {
        let v = json!({"title": "Events | Reader", "url": "https://r.test/events", "text": "Jazz at 8", "today": "2026-10-30", "elements": []});
        let t = loaded_text(&v, false);
        assert!(t.starts_with("[Today is 2026-10-30. Answer only from what this page says") && t.contains("Events | Reader\nhttps://r.test/events\n\nJazz at 8"), "{t}");
        assert!(loaded_text(&v, true).starts_with("[Read this first. Today is 2026-10-30"));
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

    /// One headless Chrome at a time: two at once (cargo runs tests in parallel) reset each other's DevTools socket
    /// ("Connection reset without closing handshake") in 3 of 10 runs; one at a time, 10 of 10 passed.
    static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct Browser {
        _turn: std::sync::MutexGuard<'static, ()>,
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
            // a test that failed (panicked) while holding the turn still hands it on
            let turn = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
            let dir = std::env::temp_dir().join(format!("strata-js-test-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
            std::fs::create_dir_all(&dir).ok()?;
            // a fresh profile, no network chatter or updates in the background: the same browser every run
            let child = Command::new(&exe).args(["--headless=new", "--remote-debugging-port=0", "--no-first-run", "--no-default-browser-check",
                                                  "--disable-background-networking", "--disable-component-update", "--disable-extensions", "--disable-sync"])
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
            Some(Browser { _turn: turn, child, ws, id: 0, dir })
        }

        fn cdp(&mut self, method: &str, params: Value) -> Value {
            self.id += 1;
            self.ws.send(tungstenite::Message::Text(json!({"id": self.id, "method": method, "params": params}).to_string().into())).unwrap();
            loop {
                match self.ws.read() {
                    Ok(tungstenite::Message::Text(t)) => {
                        let v: Value = serde_json::from_str(&t).unwrap_or(Value::Null);
                        if v["id"] == self.id {
                            // a refused command fails here, by name (it was a silent null)
                            if let Some(e) = v.get("error") {
                                panic!("{method}: Chrome refused it: {e}");
                            }
                            return v["result"].clone();
                        }
                    }
                    Ok(_) => {}
                    Err(e) => panic!("{method}: no answer from Chrome ({e})"),
                }
            }
        }

        /// A page from this HTML (a data: URL), loaded.
        /// The page from a local HTTP server: a real origin, where the address's #fragment fires hashchange (a data:
        /// page never does: the fake sheet's selection did not move).  One page, any path; framed by Content-Length
        /// and Connection: close (RFC 9112), so Chrome never waits for more.
        fn page_http(&mut self, html: &'static str) {
            self.page_http_at(html, "/spreadsheets/d/fake/edit");
        }

        fn page_http_at(&mut self, html: &'static str, path: &str) {
            let l = TcpListener::bind("127.0.0.1:0").unwrap();
            let port = l.local_addr().unwrap().port();
            std::thread::spawn(move || {
                for mut s in l.incoming().flatten() {
                    let mut r = BufReader::new(s.try_clone().unwrap());
                    let mut line = String::new();
                    while r.read_line(&mut line).map(|n| n > 0 && line != "\r\n").unwrap_or(false) {
                        line.clear();
                    }
                    let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{html}", html.len());
                }
            });
            self.go(&format!("http://127.0.0.1:{port}{path}"));
        }

        fn page(&mut self, html: &str) {
            let url = format!("data:text/html;charset=utf-8,{}", html.bytes().map(|b| if b.is_ascii_alphanumeric() { (b as char).to_string() } else { format!("%{b:02X}") }).collect::<String>());
            self.go(&url);
        }

        fn go(&mut self, url: &str) {
            // the old page marked: "complete" from it is not the new one's (that race returned early)
            self.eval("window.__strataOld = true");
            self.cdp("Page.navigate", json!({"url": url}));
            let deadline = std::time::Instant::now() + Duration::from_secs(10);
            loop {
                if self.eval("!window.__strataOld && document.readyState === 'complete'") == true {
                    return;
                }
                assert!(std::time::Instant::now() < deadline, "the page did not load in 10 s (readyState {})", self.eval("document.readyState"));
                std::thread::sleep(Duration::from_millis(20));
            }
        }

        /// A script's value; a script that throws fails the test with its message (it was a silent null).
        fn eval(&mut self, js: &str) -> Value {
            let r = self.cdp("Runtime.evaluate", json!({"expression": js, "returnByValue": true}));
            if let Some(e) = r.get("exceptionDetails") {
                panic!("the script threw: {}", e["exception"]["description"].as_str().or(e["text"].as_str()).unwrap_or("?"));
            }
            r["result"]["value"].clone()
        }

        /// An async script's value: Ok(value), or Err(its error's message) when it rejects (a tool's own error).
        fn run(&mut self, js: &str) -> Result<Value, String> {
            let r = self.cdp("Runtime.evaluate", json!({"expression": js, "returnByValue": true, "awaitPromise": true, "timeout": 30000}));
            match r.get("exceptionDetails") {
                Some(e) => Err(e["exception"]["description"].as_str().or(e["text"].as_str()).unwrap_or("?").lines().next().unwrap_or("").trim_start_matches("Error: ").to_string()),
                None => Ok(r["result"]["value"].clone()),
            }
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


    /// Google Sheets as the tools found it live (2026-10-30), small: the cell editor (#waffle-rich-text-editor) takes a
    /// paste as tab-separated cells at the selection and a copy gives the selection back; #range= in the address selects
    /// (a range past the last row is ignored, as Google ignores it: the Name Box keeps the old one); Delete clears the
    /// selection; the File / Edit / Insert menus act on mouse events; the title saves on change.  window.grid is the truth.
    const FAKE_SHEET: &str = r#"<html><head><title>Untitled spreadsheet - Google Sheets</title></head><body>
<input class="docs-title-input" value="Untitled spreadsheet"><input id="t-name-box" value="A1">
<div id="docs-file-menu">File</div><div id="docs-edit-menu">Edit</div><div id="docs-insert-menu">Insert</div>
<div class="goog-menu" id="menu"></div><div id="waffle-rich-text-editor" contenteditable="true">.</div>
<script>
let rows = 30; const COLS = 26;
window.grid = Array.from({length: rows}, () => Array(COLS).fill('')); window.boxes = new Set(); window.trashed = false;
let sel = {r1: 0, c1: 0, r2: 0, c2: 0};
const col = (s) => [...s].reduce((n, ch) => n * 26 + ch.charCodeAt(0) - 64, 0) - 1;
const parse = (range) => {
  const [a, b = a] = range.split(':');
  const p = (x, end) => { const m = x.match(/^([A-Z]+)(\d*)$/); return m && {c: col(m[1]), r: m[2] ? +m[2] - 1 : (end ? rows - 1 : 0)}; };
  const s = p(a, false), e = p(b, true);
  return s && e && e.r < rows && e.c < COLS && s.r <= e.r ? {r1: s.r, c1: s.c, r2: e.r, c2: e.c} : null;
};
addEventListener('hashchange', () => {
  const q = new URLSearchParams(location.hash.slice(1)), r = q.get('range') || 'A1', p = parse(r); tabs.push(q.get('gid'));
  if (!p) return;                                  // past the sheet: ignored, without a word
  sel = p; const [a, b] = r.split(':'); document.querySelector('#t-name-box').value = b === a ? a : r;
});
const ed = document.querySelector('#waffle-rich-text-editor');
window.pastes = []; window.tabs = [];
ed.addEventListener('paste', (e) => { e.preventDefault(); pastes.push(e.clipboardData.getData('text/plain'));
  e.clipboardData.getData('text/plain').split('\n').forEach((line, i) => line.split('\t').forEach((v, j) => {
    if (sel.r1 + i < rows && sel.c1 + j < COLS) grid[sel.r1 + i][sel.c1 + j] = v; })); });
ed.addEventListener('copy', (e) => { e.preventDefault(); const out = [];
  for (let r = sel.r1; r <= sel.r2; r++) { const row = []; for (let c = sel.c1; c <= sel.c2; c++) row.push(grid[r][c]); out.push(row.join('\t')); }
  e.clipboardData.setData('text/plain', out.join('\n')); });
ed.addEventListener('keydown', (e) => { if (e.key !== 'Delete') return;
  for (let r = sel.r1; r <= sel.r2; r++) for (let c = sel.c1; c <= sel.c2; c++) grid[r][c] = boxes.has(r + ',' + c) ? 'FALSE' : ''; });
document.querySelector('.docs-title-input').addEventListener('change', (e) => { document.title = e.target.value + ' - Google Sheets'; });
const menu = document.querySelector('#menu');
const item = (text, act) => { const d = document.createElement('div'); d.className = 'goog-menuitem'; d.innerText = text; d.addEventListener('click', act); menu.appendChild(d); };
const open = (items) => { menu.innerHTML = ''; items.forEach(([t, a]) => item(t, a)); };
document.querySelector('#docs-insert-menu').addEventListener('mousedown', () => open([['Checkbox', () => {
  for (let r = sel.r1; r <= sel.r2; r++) { boxes.add(r + ',' + sel.c1); if (grid[r][sel.c1] !== 'TRUE') grid[r][sel.c1] = 'FALSE'; } menu.innerHTML = ''; }]]));
document.querySelector('#docs-edit-menu').addEventListener('mousedown', () => open([['Delete►', () => {
  const n = sel.r2 - sel.r1 + 1, t = n === 1 ? 'Row ' + (sel.r1 + 1) : 'Rows ' + (sel.r1 + 1) + ' - ' + (sel.r2 + 1);
  item(t, () => { grid.splice(sel.r1, n); rows -= n; menu.innerHTML = ''; }); }]]));
document.querySelector('#docs-file-menu').addEventListener('mousedown', () => open([['Move to trash', () => { window.trashed = true; menu.innerHTML = ''; }]]));
</script></body></html>"#;

    /// A Google tool's script (google.rs) run in the page as the app runs it; its answer as the model reads it.
    fn tool(b: &mut Browser, name: &str, args: Value) -> Result<String, String> {
        let js = crate::google::sheets_js(name, &args)?;
        b.run(&format!("(async () => {{ window.__strataPace = 0.02; {js} }})()")).map(|v| crate::google::answer(name, &args, &v))
    }

    fn grid(b: &mut Browser, rows: usize) -> Vec<Vec<String>> {
        serde_json::from_value(b.eval(&format!("grid.slice(0, {rows}).map((r) => r.slice(0, 10))"))).unwrap()
    }

    #[test]
    fn sheets_tools_on_a_fake_sheet() {
        let Some(mut b) = Browser::start() else { return };
        b.page_http(FAKE_SHEET);
        let cols = ["Checkbox", "Rating", "Address", "Price", "Sq feet", "Time", "Link", "Description", "Availability", "Confirmation"];
        // not a new sheet (something in it already): sheets_create changes nothing
        b.eval("grid[3][2] = 'someone else'");
        let e = tool(&mut b, "sheets_create", json!({"title": "studios", "columns": cols, "__landed": "http://127.0.0.1/spreadsheets/d/fake/edit"})).unwrap_err();
        assert!(e.starts_with("the browser is on a spreadsheet that is not new"), "{e}");
        assert_eq!(b.eval("document.title"), "Untitled spreadsheet - Google Sheets", "not renamed");
        b.eval("grid[3][2] = ''");
        // the document open before the load (the pane never left it), or one with two tabs: not new either
        assert!(tool(&mut b, "sheets_create", json!({"title": "studios", "columns": cols, "__before": "fake", "__landed": "http://127.0.0.1/spreadsheets/d/fake/edit"})).unwrap_err().starts_with("the browser is on a spreadsheet that is not new"));
        // the pane went to another sheet after the load landed: not that one
        assert!(tool(&mut b, "sheets_create", json!({"title": "studios", "columns": cols, "__landed": "http://127.0.0.1/spreadsheets/d/other/edit"})).unwrap_err().starts_with("the browser is on a spreadsheet that is not new"));
        // a title that only begins like the default: not new
        b.eval("document.querySelector('.docs-title-input').value = 'Untitled spreadsheet archive'");
        assert!(tool(&mut b, "sheets_create", json!({"title": "studios", "columns": cols, "__landed": "http://127.0.0.1/spreadsheets/d/fake/edit"})).unwrap_err().starts_with("the browser is on a spreadsheet that is not new"));
        b.eval("document.querySelector('.docs-title-input').value = 'Untitled spreadsheet'");
        b.eval("for (let i = 0; i < 2; i++) { const t = document.createElement('div'); t.className = 'docs-sheet-tab'; document.body.appendChild(t); }");
        assert!(tool(&mut b, "sheets_create", json!({"title": "studios", "columns": cols, "__landed": "http://127.0.0.1/spreadsheets/d/fake/edit"})).unwrap_err().starts_with("the browser is on a spreadsheet that is not new"));
        b.eval("document.querySelectorAll('.docs-sheet-tab').forEach((t) => t.remove())");
        assert_eq!(b.eval("document.title"), "Untitled spreadsheet - Google Sheets", "not renamed");
        // sheets.new's load landed here (the app records the address it landed on)
        let landed = json!("http://127.0.0.1/spreadsheets/d/fake/edit");
        let out = tool(&mut b, "sheets_create", json!({"title": "studios", "columns": cols, "__landed": landed})).unwrap();
        assert!(out.starts_with("created studios (") && out.contains("10 columns"), "{out}");
        assert_eq!(b.eval("document.title"), "studios - Google Sheets");
        assert_eq!(grid(&mut b, 1)[0], cols);

        // rows by column name land under their names; FALSE first cells become tick boxes in these rows only
        let out = tool(&mut b, "sheets_append", json!({"rows": [
            {"Checkbox": "FALSE", "Rating": "4.0 (122)", "Address": "1440 Columbia St", "Price": "$2,916"},
            {"checkbox": "FALSE", "address": "801 A St", "description": "The Rey: a\ntower"}]})).unwrap();
        assert!(out.contains("added 2 rows at rows 2-3") && out.contains("filling Checkbox, Rating, Address, Price, Description") && out.contains("(Checkbox as tick boxes)"), "{out}");
        let g = grid(&mut b, 4);
        assert_eq!(g[1], ["FALSE", "4.0 (122)", "1440 Columbia St", "$2,916", "", "", "", "", "", ""]);
        assert_eq!((g[2][2].as_str(), g[2][7].as_str(), g[2][1].as_str()), ("801 A St", "The Rey: a tower", ""));
        assert_eq!(b.eval("[...boxes].sort().join(' ')"), "1,0 2,0");

        // the failure seen in the chat: [name, address, price] by position. Refused, the sheet as it was
        let before = grid(&mut b, 5);
        let e = tool(&mut b, "sheets_append", json!({"rows": [["The Artisan", "1601 Broadway", "$2,411+"]]})).unwrap_err();
        assert!(e.starts_with("a row of 3 cells for 10 columns: give each row as {column: value}; the columns are: Checkbox, Rating"), "{e}");
        assert!(tool(&mut b, "sheets_append", json!({"rows": [{"Name": "The Artisan"}]})).unwrap_err().starts_with("no column \"Name\""));
        assert_eq!(grid(&mut b, 5), before);

        // an update changes the named cells only, each pasted alone (the whole row pasted back turned formulas
        // into values)
        b.eval("pastes.length = 0");
        let out = tool(&mut b, "sheets_write", json!({"row": 3, "values": {"Price": "$2,101", "Rating": "4.3"}})).unwrap();
        assert_eq!(b.eval("JSON.stringify(pastes)"), r#"["4.3","$2,101"]"#, "two pastes, in column order (not adjacent)");
        assert!(out.starts_with("updated row 3 of studios"), "{out}");
        assert_eq!(grid(&mut b, 3)[2][..4], ["FALSE", "4.3", "801 A St", "$2,101"]);
        assert_eq!(grid(&mut b, 3)[2][7], "The Rey: a tower");
        // an "update" of a row past the data is refused, with the way to add rows
        let e = tool(&mut b, "sheets_write", json!({"row": 9, "values": {"Address": "new"}})).unwrap_err();
        assert!(e.starts_with("row 9 is past the last row with data (3): to add rows use sheets_append"), "{e}");
        // adjacent columns in one paste; the cells beside them stay
        b.eval("pastes.length = 0; grid[2][5] = 'keep'");
        tool(&mut b, "sheets_write", json!({"row": 3, "values": {"Address": "801 A St", "Price": "$2,101", "Sq feet": "470"}})).unwrap();
        assert_eq!(b.eval("JSON.stringify(pastes)"), r#"["801 A St\t$2,101\t470"]"#);
        assert_eq!(grid(&mut b, 3)[2][5], "keep");
        // a value set to "" clears that cell, the rest stay
        tool(&mut b, "sheets_write", json!({"row": 3, "values": {"Rating": ""}})).unwrap();
        assert_eq!(grid(&mut b, 3)[2][..4], ["FALSE", "", "801 A St", "$2,101"]);

        // the read: rows numbered as in the sheet, empty ones at the end left out
        let out = tool(&mut b, "sheets_read", json!({})).unwrap();
        assert!(out.contains("row 1: Checkbox | Rating | Address") && out.contains("row 2: FALSE | 4.0 (122) | 1440 Columbia St | $2,916") && !out.contains("row 4:"), "{out}");
        assert!(tool(&mut b, "sheets_read", json!({"range": "A1:Z5000"})).unwrap_err().contains("is not in the sheet"), "a range past the last row is an error, not the old selection");

        // delete a row: the rows under move up, and the next append goes under the last filled row (it once wrote
        // over row 2: the range it counted rows in was past the sheet and ignored)
        tool(&mut b, "sheets_delete_rows", json!({"from": 2})).unwrap();
        assert_eq!(grid(&mut b, 2)[1][2], "801 A St");
        let out = tool(&mut b, "sheets_append", json!({"rows": [{"Checkbox": "FALSE", "Address": "702 Broadway"}]})).unwrap();
        assert!(out.contains("at rows 3-3"), "{out}");
        assert_eq!((grid(&mut b, 3)[1][2].as_str(), grid(&mut b, 3)[2][2].as_str()), ("801 A St", "702 Broadway"));

        // a row with no FALSE / TRUE first: no tick box made
        let boxes = b.eval("boxes.size");
        let out = tool(&mut b, "sheets_append", json!({"rows": [{"Address": "9 B St"}]})).unwrap();
        assert!(out.contains("at rows 4-4") && !out.contains("tick boxes"), "{out}");
        assert_eq!(b.eval("boxes.size"), boxes);

        // a value TRUE outside column A is data: the next append goes under it, not over it
        tool(&mut b, "sheets_append", json!({"rows": [{"Availability": "TRUE"}]})).unwrap();
        let out = tool(&mut b, "sheets_append", json!({"rows": [{"Address": "5 After St"}]})).unwrap();
        let rows: Vec<Vec<String>> = grid(&mut b, 8);
        let at = rows.iter().position(|r| r[8] == "TRUE").unwrap();
        assert_eq!(rows[at + 1][2], "5 After St", "{out}");
        // a column name that two columns share (any case) is refused, not the first taken
        b.eval("grid[0][9] = 'availability'");
        assert!(tool(&mut b, "sheets_write", json!({"row": 2, "values": {"Availability": "x"}})).unwrap_err().contains("names 2 columns (I, J)"));
        b.eval("grid[0][9] = 'Confirmation'");

        // the tab the address names is the one selected (a call for tab B once acted on the open tab A)
        b.eval("tabs.length = 0");
        tool(&mut b, "sheets_read", json!({"url": "https://docs.google.com/spreadsheets/d/fakefakefake1/edit#gid=55"})).unwrap();
        assert!(b.eval("tabs.every((g) => g === '55') && tabs.length > 0") == true, "{}", b.eval("JSON.stringify(tabs)"));
        // a script the app stopped (past its time) changes nothing more
        let before = grid(&mut b, 6);
        let js = crate::google::sheets_js("sheets_append", &json!({"rows": [{"Address": "1 Stop St"}]})).unwrap();
        let e = b.run(&format!("(async () => {{ window.__strataPace = 0.02; const __ID = 9; window.__strataStop = {{9: true}}; {js} }})()")).unwrap_err();
        assert!(e.starts_with("stopped: it ran past its time"), "{e}");
        assert_eq!(grid(&mut b, 6), before);
        // a stop that comes after the selection, before the paste: the paste does not happen
        let r = b.eval(&format!("(() => {{ const __ID = 3; {} ; window.__strataStop = {{3: true}}; try {{ clip('paste', 'late'); return 'pasted'; }} catch (e) {{ return e.message; }} }})()", crate::google::PRELUDE));
        assert!(r.as_str().unwrap().starts_with("stopped:"), "{r}");
        assert!(!b.eval("pastes.includes('late')").as_bool().unwrap());
        // nor a menu click (File > Move to trash is one)
        let r = b.eval(&format!("(() => {{ const __ID = 4; {} ; window.__strataStop = {{4: true}}; try {{ mouse(document.querySelector('#docs-file-menu')); return 'clicked'; }} catch (e) {{ return e.message; }} }})()", crate::google::PRELUDE));
        assert!(r.as_str().unwrap().starts_with("stopped:"), "{r}");
        assert_eq!(b.eval("document.querySelector('#menu').children.length"), 0, "the menu did not open");
        b.eval("window.__strataStop = {}");

        tool(&mut b, "sheets_clear", json!({"range": "D2:D2"})).unwrap();
        assert_eq!(grid(&mut b, 2)[1][3], "");
        assert!(tool(&mut b, "sheets_trash", json!({})).unwrap().starts_with("moved to the trash: studios"));
        assert_eq!(b.eval("trashed"), true);

        // no url, an address with ?gid=9 (no #fragment): that tab, not tab 0
        b.page_http_at(FAKE_SHEET, "/spreadsheets/d/fake/edit?gid=9");
        b.eval("tabs.length = 0");
        tool(&mut b, "sheets_read", json!({})).unwrap();
        assert!(b.eval("tabs.length > 0 && tabs.every((g) => g === '9')") == true, "{}", b.eval("JSON.stringify(tabs)"));

        // maps_import_sheet on another map than the one asked for (the pane went elsewhere while it loaded): nothing
        b.page_http_at(r#"<html><body><div>Add layer</div><div>Import</div></body></html>"#, "/maps/d/edit?mid=OTHER");
        let js = crate::google::maps_js("maps_import_sheet", &json!({"map_url": "https://www.google.com/maps/d/edit?mid=WANT", "sheet_title": "s", "location_column": "Address", "title_column": "Name"}), "t").unwrap();
        let e = b.run(&format!("(async () => {{ {js} }})()")).unwrap_err();
        assert_eq!(e, "the browser is at another map (OTHER), not WANT: nothing imported");
    }

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

        // what browser_open answers with, from a real page: the note, then title and address on their own lines (the
        // server's Sources list reads those two lines); a page with no text says so in the line the server skips
        b.page("<html><head><title>Jazz tonight</title></head><body><main><p>Quartet at 8 PM.</p></main></body></html>");
        let t = loaded_text(&b.json(&read_js(1)), false);
        let (note, body) = t.split_once("\n\n").unwrap();
        assert!(note.starts_with("[Today is 20") && note.ends_with("link it as [title](address).]"), "{note}");
        let lines: Vec<&str> = body.lines().take(4).collect();
        assert_eq!((lines[0], lines[1].starts_with("data:") || lines[1].starts_with("http") || lines[1].starts_with("about:"), lines[3]), ("Jazz tonight", true, "Quartet at 8 PM."), "{body}");
        b.page("<html><head><title>Just a moment</title></head><body></body></html>");
        assert!(loaded_text(&b.json(&read_js(1)), false).contains("\n(no text on this page"));

        // My Maps' column boxes and the picker's files: named by their label elsewhere and by their role, none dropped
        b.page(r#"<html><body><main><p>Choose columns to position your placemarks.</p>
<style>[role=checkbox]{display:inline-block;width:14px;height:14px}</style><span role="checkbox" aria-checked="false" aria-labelledby="l0"></span><span id="l0">Checkbox</span>
<span role="checkbox" aria-checked="true" aria-labelledby="l1"></span><span id="l1">Address</span>
<div role="option" aria-selected="true">studios-sandiego-downtown-2026</div>
<div role="button" id="menu" onmousedown="window.opened = (window.opened || 0) + 1">Open the map menu</div></main></body></html>"#);
        let v = b.json(&read_js(1));
        let els: Vec<&str> = v["elements"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
        assert!(els.iter().any(|e| e.ends_with("checkbox \"Checkbox\"")) && els.iter().any(|e| e.ends_with("checkbox (on) \"Address\"")), "{els:?}");
        assert!(els.iter().any(|e| e.ends_with("option (on) \"studios-sandiego-downtown-2026\"")), "{els:?}");
        // a click presses and releases the mouse first: Google's menus open on those
        let m = els.iter().find(|e| e.contains("Open the map menu")).map(|e| e[1..e.find(']').unwrap()].parse::<i64>().unwrap()).unwrap();
        b.json(&click_js(m));
        assert_eq!(b.eval("window.opened"), json!(1));

        // Google Sheets' rows by column name (google.rs place): each value under its own column, never shifted
        let place = |b: &mut Browser, rows: &str| -> Value {
            b.json(&format!("(() => {{ {} ; try {{ return JSON.stringify(place(['Checkbox', 'Rating', 'Address', 'Price'], {rows})); }} catch (e) {{ return JSON.stringify({{error: e.message}}); }} }})()", crate::google::PRELUDE))
        };
        assert_eq!(place(&mut b, r#"[{"address": "1 A St", "Price": "$2,100\n/mo", "Checkbox": "FALSE"}]"#), json!([["FALSE", "", "1 A St", "$2,100 /mo"]]));
        // the failure seen: three positional cells (name, address, price) for four columns are refused, not shifted
        let e = place(&mut b, r#"[["The Rey", "801 A St", "$2,101"]]"#);
        assert_eq!(e["error"], "a row of 3 cells for 4 columns: give each row as {column: value}; the columns are: Checkbox, Rating, Address, Price");
        assert_eq!(place(&mut b, r#"[["TRUE", "4.3", "801 A St", "$2,101"]]"#), json!([["TRUE", "4.3", "801 A St", "$2,101"]]));
        assert_eq!(place(&mut b, r#"[{"Name": "The Rey"}]"#)["error"], "no column \"Name\"; the columns are: Checkbox, Rating, Address, Price");
        assert!(place(&mut b, "[]")["error"].as_str().unwrap().starts_with("rows:"));
        assert_eq!(b.json(&format!("(() => {{ {} ; return JSON.stringify([letter(0), letter(9), letter(25), letter(26), letter(27)]); }})()", crate::google::PRELUDE)), json!(["A", "J", "Z", "AA", "AB"]));

        // where a real click goes: the element brought into view, its centre; what covers it; none for the hidden
        b.page(r#"<html><body style="margin:0"><div style="height:2000px"></div><button id="far" style="width:100px;height:40px">Far</button>
<button id="under" style="position:absolute;left:0;top:0;width:100px;height:40px">Under</button>
<div style="position:absolute;left:0;top:0;width:300px;height:60px;background:#fff">Accept cookies</div>
<button style="display:none">Hidden</button><input id="q" value="old words"></body></html>"#);
        let els: Vec<String> = b.json(&read_js(1))["elements"].as_array().unwrap().iter().map(|e| e.as_str().unwrap().to_string()).collect();
        let num = |name: &str| els.iter().find(|e| e.contains(name)).map(|e| e[1..e.find(']').unwrap()].parse::<i64>().unwrap()).unwrap();
        let t = b.json(&target_js(num("Far")));
        let (x, y) = (t["x"].as_f64().unwrap(), t["y"].as_f64().unwrap());
        assert!(x > 0.0 && y > 0.0 && y < b.eval("innerHeight").as_f64().unwrap() && t["cover"].is_null(), "scrolled into view, nothing on top: {t}");
        assert_eq!(b.eval(&format!("document.elementFromPoint({x}, {y}).id")), "far", "the centre is the element");
        b.eval("scrollTo(0, 0)");
        assert_eq!(b.json(&target_js(num("Under")))["cover"], "Accept cookies");
        b.eval("document.querySelector('[style*=\"display:none\"]').setAttribute('data-strata-id', '99')");
        assert_eq!(b.json(&target_js(99))["hidden"], true);
        assert!(b.json(&target_js(77))["error"].as_str().unwrap().contains("no element [77]"));
        // typing replaces the field's text: it is focused with its text selected
        assert_eq!(b.json(&focus_js(num("old words")))["focused"], true);
        assert_eq!(b.eval("[document.activeElement.id, document.activeElement.selectionStart, document.activeElement.selectionEnd].join(' ')"), "q 0 9");

        // My Maps' import: a column by its header name or its letter (the model gave A, C, B, D for Address)
        let pick = |b: &mut Browser, want: &str| -> i64 {
            b.eval(&format!("(() => {{ {} ; return pickColumn(['Checkbox', 'Rating', 'Address', 'Price', 'Sq feet', 'Time', 'Link', 'Description'], {}); }})()",
                            crate::google::MAPS_PRELUDE, serde_json::to_string(want).unwrap())).as_i64().unwrap()
        };
        assert_eq!((pick(&mut b, "Address"), pick(&mut b, "address "), pick(&mut b, "C"), pick(&mut b, "h"), pick(&mut b, "Description")), (2, 2, 2, 7, 7));
        assert_eq!((pick(&mut b, "Z"), pick(&mut b, "Name"), pick(&mut b, ""), pick(&mut b, "AB")), (-1, -1, -1, -1), "no such column: -1, so the error lists them");

        // maps_create with a title the list already has: that map's address (its card's data-mid), no second map
        b.page(r#"<html><body><div role="button" onclick="window.created = true">add_to_drive</div>
<div data-mid="dummyId">Share via Facebook</div><div data-mid="dummyId" data-name="studios-sandiego-downtown-2026">a template</div>
<div class="card" data-mid="1JYYmapid" data-name="studios-sandiego-downtown-2026"><div class="t">studios-sandiego-downtown-2026</div><div aria-label="Open the map menu" data-mid="dummyId">⋮</div></div></body></html>"#);
        let js = crate::google::maps_js("maps_create", &json!({"title": "studios-sandiego-downtown-2026"}), "t").unwrap();
        let out = b.run(&format!("(async () => {{ {js} }})()")).unwrap();
        assert_eq!(out, "a map titled studios-sandiego-downtown-2026 already exists, so none was made: https://www.google.com/maps/d/edit?mid=1JYYmapid. To put places on it: maps_import_sheet (each import adds a layer).");
        assert_eq!(b.eval("window.created === undefined"), true, "Create was not pressed");
        // the title on the page but on no card (a tooltip, the trash page): an error, never a second map
        b.page(r#"<html><body><div role="button" onclick="window.created = true">add_to_drive</div><div>studios-sandiego-downtown-2026</div></body></html>"#);
        let e = b.run(&format!("(async () => {{ {js} }})()")).unwrap_err();
        assert!(e.starts_with("a map titled studios-sandiego-downtown-2026 seems to exist, but its address was not found"), "{e}");
        assert_eq!(b.eval("window.created === undefined"), true, "Create was not pressed");

        // maps_import_sheet into a trashed map: My Maps sends it home with "in your Trash"; said so, with the maps there are
        b.page(r#"<html><body><div>The map you tried to access is in your Trash. To open this map, take it out of your Trash.</div>
<div data-mid="1JYYmapid" data-name="studios-sandiego-downtown-2026"><div>studios-sandiego-downtown-2026</div><div>Oct 30, 2026</div>
<div data-mid="dummyId">Share via Facebook</div></div><div data-mid="dummyId" data-name="Untitled map" style="display:none">template</div></body></html>"#);
        let js = crate::google::maps_js("maps_import_sheet", &json!({"sheet_title": "s", "location_column": "Address", "title_column": "Description"}), "t").unwrap();
        let e = b.run(&format!("(async () => {{ {js} }})()")).unwrap_err();
        assert_eq!(e, "that map is in the trash (gone: use another); the maps in My Maps: studios-sandiego-downtown-2026 (https://www.google.com/maps/d/edit?mid=1JYYmapid). maps_create with a title gives the address of the map with that title");

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
