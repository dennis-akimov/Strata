//! Google Sheets and My Maps for the model, through the browser pane signed in as the person (no Google API: their
//! choice).  Each tool is a script run in the Sheets or My Maps page.  These ways in were tried live on 2026-10-30:
//! - a cell range is selected by the address's fragment (#gid=0&range=C3);
//! - Sheets takes a script's paste event on its cell editor (#waffle-rich-text-editor): tab-separated rows become cells;
//! - its copy event hands back the selection as tab-separated text (empty rows kept, values as shown): the read;
//! - a Delete key event clears the selection; its menus answer mouse events (Insert > Checkbox, File > Move to trash);
//! - the title saves on its input's change and blur; My Maps' title saves in its dialog (the title, then Save).
//! A script returns a Promise; the app keeps its value in window.__strataAsync[id] and looks for it (async_js).
use serde_json::{json, Value};

/// The tools, as tools/list gives them (after the browser's).
pub fn tools() -> Value {
    let rows = json!({"type": "array", "items": {"type": "object", "additionalProperties": {"type": "string"}}, "description": "one object per row, its values by column name (the header row's names, as sheets_read shows them), e.g. [{\"Address\": \"123 Main St\", \"Price\": \"$2,100\"}]; a column left out stays empty"});
    let url = json!({"type": "string", "description": "the spreadsheet's address (docs.google.com/spreadsheets/d/...); default: the one open in the browser"});
    json!([
        {"name": "sheets_create", "description": "Create a Google Sheets spreadsheet in the user's Google Drive (the browser beside the chat, signed in as them): its title and its header row. Answers with its address.",
         "inputSchema": {"type": "object", "required": ["title", "columns"], "properties": {"title": {"type": "string"}, "columns": {"type": "array", "items": {"type": "string"}}}}},
        {"name": "sheets_read", "description": "Read a Google Sheets spreadsheet: its rows, numbered as in the sheet (row 1 is the header), cells as shown.",
         "inputSchema": {"type": "object", "properties": {"url": url, "range": {"type": "string", "description": "e.g. A1:J50 (default: A:Z, the whole sheet up to its last filled row; 300 rows at a time)"}}}},
        {"name": "sheets_append", "description": "Add rows under the last filled row of a Google Sheets spreadsheet, each value under its column's name (read the sheet first to see them). When every row's first-column value is FALSE or TRUE, those cells become tick boxes.",
         "inputSchema": {"type": "object", "required": ["rows"], "properties": {"url": url, "rows": rows}}},
        {"name": "sheets_write", "description": "Change cells of one row of a Google Sheets spreadsheet by column name (update): the other cells of the row stay as they are.",
         "inputSchema": {"type": "object", "required": ["row", "values"], "properties": {"url": url, "row": {"type": "integer", "description": "the row's number as sheets_read shows it (2 or more)"},
             "values": {"type": "object", "additionalProperties": {"type": "string"}, "description": "{column name: new value}"}}}},
        {"name": "sheets_delete_rows", "description": "Delete rows from..to of a Google Sheets spreadsheet (the rows under move up); row 1, the header, stays.",
         "inputSchema": {"type": "object", "required": ["from"], "properties": {"url": url, "from": {"type": "integer"}, "to": {"type": "integer"}}}},
        {"name": "sheets_clear", "description": "Empty the cells of a range of a Google Sheets spreadsheet (e.g. A5:J5 for row 5).",
         "inputSchema": {"type": "object", "required": ["range"], "properties": {"url": url, "range": {"type": "string"}}}},
        {"name": "sheets_trash", "description": "Move a Google Sheets spreadsheet to the trash (the user is asked to allow it).",
         "inputSchema": {"type": "object", "properties": {"url": url}}},
        {"name": "maps_create", "description": "Create a Google My Maps map in the user's Google Drive with this title. Answers with its address. To put places on it: maps_import_sheet.",
         "inputSchema": {"type": "object", "required": ["title"], "properties": {"title": {"type": "string"}, "description": {"type": "string"}}}},
        {"name": "maps_import_sheet", "description": "Put a Google Sheets spreadsheet's rows on a Google My Maps map as places (a new layer named after the sheet): location_column holds the addresses, title_column names each place; every column is kept in each place's details.",
         "inputSchema": {"type": "object", "required": ["map_url", "sheet_title", "location_column", "title_column"], "properties": {"map_url": {"type": "string", "description": "the map's address (maps_create answers with it)"}, "sheet_title": {"type": "string"},
             "location_column": {"type": "string", "description": "the column with the addresses: its name in the header row (e.g. Address) or its letter (C)"},
             "title_column": {"type": "string", "description": "the column that names each place: its name (e.g. Description) or its letter"}}}},
        {"name": "maps_trash", "description": "Move a Google My Maps map (by its title) to the trash (the user is asked to allow it).",
         "inputSchema": {"type": "object", "required": ["title"], "properties": {"title": {"type": "string"}}}}
    ])
}

/// Is `tool` one of these?
pub fn is_google(tool: &str) -> bool {
    tool.starts_with("sheets_") || tool.starts_with("maps_")
}

/// The page `tool` works in: (the address to load when the pane is elsewhere, the prefix that says it is there).
pub fn page_for(tool: &str, args: &Value, here: &str) -> Result<(String, String), String> {
    match tool {
        "sheets_create" => Ok(("https://sheets.new".into(), "\u{0}".into())),   // always a new one
        "maps_create" if args["__phase"] == 2 => Ok((String::new(), MAPS_EDIT.into())),
        "maps_create" | "maps_trash" => Ok((MAPS_LIST.into(), MAPS_LIST.into())),
        "maps_import_sheet" => {
            let url = args["map_url"].as_str().unwrap_or("");
            let mid = url.split("mid=").nth(1).map(|m| m.split('&').next().unwrap_or("")).filter(|m| !m.is_empty() && m.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'));
            let mid = mid.ok_or_else(|| format!("map_url {url:?}: a My Maps address with mid= (maps_create answers with it)"))?;
            // always loaded fresh: an import after a cancelled one, on the same page, never finished
            Ok((format!("{MAPS_EDIT}?mid={mid}"), "\u{0}".into()))
        }
        _ => {
            let url = args["url"].as_str().filter(|u| !u.is_empty()).map(String::from).unwrap_or_else(|| here.to_string());
            let base = sheet_base(&url).ok_or_else(|| format!("{url} is not a Google Sheets address: give the spreadsheet's url (sheets_create answers with it)"))?;
            Ok((format!("{base}/edit"), base))
        }
    }
}

pub const MAPS_LIST: &str = "https://www.google.com/maps/d/";
pub const MAPS_EDIT: &str = "https://www.google.com/maps/d/edit";

/// Is the pane where page_for's prefix says (My Maps' list is not its editor, which starts the same)?
pub fn on_page(prefix: &str, here: &str) -> bool {
    here.starts_with(prefix) && (prefix != MAPS_LIST || !here.starts_with(MAPS_EDIT) && !here.starts_with("https://www.google.com/maps/d/viewer"))
}

/// The tools that wait for the person's Allow in the browser's strip, and what it asks.
pub fn ask(tool: &str, args: &Value) -> Option<String> {
    match tool {
        "sheets_trash" => Some(format!("The model wants to move {} to the trash.", args["url"].as_str().unwrap_or("the spreadsheet open in the browser"))),
        "maps_trash" => Some(format!("The model wants to move the map \u{201c}{}\u{201d} to the trash.", args["title"].as_str().unwrap_or(""))),
        _ => None,
    }
}

/// https://docs.google.com/spreadsheets/d/<id> of a spreadsheet's address.
pub fn sheet_base(url: &str) -> Option<String> {
    let rest = url.strip_prefix("https://docs.google.com/spreadsheets/d/")?;
    let id: String = rest.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_').collect();
    (id.len() > 10).then(|| format!("https://docs.google.com/spreadsheets/d/{id}"))
}

/// A cell address or range the fragment can take (A1, B5, A2:J40): nothing else goes into the script.
pub fn range_ok(r: &str) -> bool {
    let parts: Vec<&str> = r.split(':').collect();
    let cell = |p: &str, whole_column: bool| {
        let letters = p.chars().take_while(|c| c.is_ascii_uppercase()).count();
        let digits = &p[letters..];
        (1..=3).contains(&letters) && digits.len() <= 7 && digits.chars().all(|c| c.is_ascii_digit()) && (whole_column || !digits.is_empty())
    };
    // A1, A2:J40, or whole columns A:Z (both ends letters only)
    let columns = parts.len() == 2 && parts.iter().all(|p| p.chars().all(|c| c.is_ascii_uppercase()));
    !r.is_empty() && parts.len() <= 2 && parts.iter().all(|p| cell(p, columns))
}

/// Rows as the paste takes them: cells by tabs, rows by newlines (a tab or newline in a cell would split it: a space).
pub fn tsv(rows: &Value) -> Result<String, String> {
    let rows = rows.as_array().ok_or("rows: a list of rows, each a list of cells")?;
    let cell = |v: &Value| -> String {
        let s = match v { Value::String(s) => s.clone(), Value::Null => String::new(), other => other.to_string() };
        s.split(['\t', '\r', '\n']).filter(|p| !p.is_empty()).collect::<Vec<_>>().join(" ")
    };
    Ok(rows.iter().map(|r| match r.as_array() {
        Some(cells) => cells.iter().map(cell).collect::<Vec<_>>().join("\t"),
        None => cell(r),
    }).collect::<Vec<_>>().join("\n"))
}

/// Rows read at once (a model's context holds about this many short rows).
const MAX_ROWS: usize = 300;

/// A cell with something in it: a tick box alone (TRUE / FALSE) does not fill a row (200 unticked rows are no data).
fn filled(c: &str) -> bool {
    let c = c.trim();
    !c.is_empty() && c != "FALSE" && c != "TRUE"
}

/// What the copy gave (tab-separated) -> rows numbered from `first`, empty rows and columns at the end left out.
pub fn numbered(text: &str, first: usize) -> String {
    let rows: Vec<Vec<&str>> = text.split('\n').map(|l| l.trim_end_matches('\r').split('\t').collect()).collect();
    let width = rows.iter().map(|r| r.iter().rposition(|c| !c.trim().is_empty()).map_or(0, |i| i + 1)).max().unwrap_or(0);
    let last = rows.iter().rposition(|r| r.iter().any(|c| filled(c)));
    let Some(last) = last else { return "(empty)".into() };
    let shown = (last + 1).min(MAX_ROWS);
    let mut out = rows[..shown].iter().enumerate().map(|(i, r)| {
        let cells: Vec<&str> = (0..width).map(|j| r.get(j).copied().unwrap_or("").trim()).collect();
        format!("row {}: {}", first + i, cells.join(" | "))
    }).collect::<Vec<_>>().join("\n");
    if last + 1 > shown {
        out.push_str(&format!("\n[{} more rows: sheets_read with range A{}:Z{}]", last + 1 - shown, first + shown, first + shown + MAX_ROWS - 1));
    }
    out
}

/// The first row of a range ("A2:J40" -> 2).
pub fn first_row(range: &str) -> usize {
    range.split(':').next().unwrap_or("A1").trim_start_matches(|c: char| c.is_ascii_uppercase()).parse().unwrap_or(1)
}

/// Sheets' helpers, before each body.
pub const PRELUDE: &str = r#"
// window.__strataPace: tests run the same waits shorter (a fake sheet answers at once)
const sleep = (ms) => new Promise((r) => setTimeout(r, ms * (window.__strataPace || 1)));
const editor = () => document.querySelector('#waffle-rich-text-editor');
const ready = async () => { for (let i = 0; i < 40 && !editor(); i++) await sleep(250); if (!editor()) throw new Error('the spreadsheet did not open (signed in to Google in the browser?)'); };
const go = async (range) => {
  const gid = new URLSearchParams(location.hash.slice(1)).get('gid') || '0';
  location.hash = 'gid=' + gid + '&range=' + range; await sleep(800); editor().focus(); await sleep(100);
  // a range past the sheet's last row or column is ignored without a word (the selection stays where it was: an
  // append once counted the rows of A1 alone and wrote over row 2): the Name Box says what is selected
  const nb = document.querySelector('#t-name-box'), [a, b] = range.split(':');
  const shown = b === a ? a : range;     // the Name Box says A4 for A4:A4
  if (nb && nb.value !== shown) throw new Error('range ' + range + ' is not in the sheet (it has fewer rows or columns); selected: ' + nb.value);
};
const clip = (type, text) => { const dt = new DataTransfer(); if (text !== undefined) dt.setData('text/plain', text);
  editor().dispatchEvent(new ClipboardEvent(type, {clipboardData: dt, bubbles: true, cancelable: true})); return dt.getData('text/plain'); };
const del = () => editor().dispatchEvent(new KeyboardEvent('keydown', {key: 'Delete', code: 'Delete', keyCode: 46, which: 46, bubbles: true, cancelable: true}));
const mouse = (el) => { const r = el.getBoundingClientRect();
  for (const t of ['mouseover', 'mousedown', 'mouseup', 'click']) el.dispatchEvent(new MouseEvent(t, {bubbles: true, cancelable: true, view: window, clientX: r.x + r.width / 2, clientY: r.y + r.height / 2, button: 0})); };
const menu = async (id, item) => { const m = document.querySelector(id); if (!m) throw new Error('no menu ' + id);
  mouse(m); await sleep(400);
  const it = [...document.querySelectorAll('.goog-menu .goog-menuitem')].find((e) => e.offsetParent && e.innerText.trim().split('\n')[0].trim().startsWith(item));
  if (!it) { mouse(m); throw new Error('no menu item ' + item); } mouse(it); await sleep(600); };
// rows by column name -> cells in the header's order.  Positional rows went wrong: a model wrote [name, address, price]
// from B5, under Rating, Address and Price, and each row then held halves of two buildings.  A row as a list must
// fill every column; a name that is no column is refused (with the columns), never put somewhere.
const place = (header, rows) => {
  const cols = header.map((h) => String(h).trim()), key = (k) => String(k).trim().toLowerCase();
  const cell = (v) => (v === null || v === undefined ? '' : String(v)).replace(/[\t\r\n]+/g, ' ').trim();
  const list = cols.join(', ');
  if (!Array.isArray(rows) || !rows.length) throw new Error('rows: a list of {column: value}; the columns are: ' + list);
  return rows.map((r) => {
    if (Array.isArray(r)) {
      if (r.length !== cols.length) throw new Error('a row of ' + r.length + ' cells for ' + cols.length + ' columns: give each row as {column: value}; the columns are: ' + list);
      return r.map(cell);
    }
    if (!r || typeof r !== 'object') throw new Error('each row: {column: value}; the columns are: ' + list);
    const out = cols.map(() => '');
    for (const [k, v] of Object.entries(r)) {
      const i = cols.findIndex((c) => key(c) === key(k));
      if (i < 0) throw new Error('no column "' + k + '"; the columns are: ' + list);
      out[i] = cell(v);
    }
    return out;
  });
};
const letter = (i) => { let s = ''; for (i++; i > 0; i = Math.floor((i - 1) / 26)) s = String.fromCharCode(65 + (i - 1) % 26) + s; return s; };
// the header row's names (row 1 up to its last named column)
const header = async () => { await go('A1:Z1'); const h = clip('copy').split('\n')[0].split('\t').map((c) => c.trim());
  while (h.length && !h[h.length - 1]) h.pop(); if (!h.length) throw new Error('row 1 has no column names: sheets_create gives a sheet its header'); return h; };
// a cell with something in it: an unticked (or ticked) box alone does not fill a row
const filled = (c) => c.trim() && c.trim() !== 'FALSE' && c.trim() !== 'TRUE';
const title = () => (document.querySelector('.docs-title-input') || {}).value || document.title;
const where = () => title() + ' (' + location.href.split('#')[0].replace(/\?.*$/, '') + ')';
"#;

/// The script for a Sheets tool (run where page_for says, once it has loaded).
pub fn sheets_js(tool: &str, args: &Value) -> Result<String, String> {
    let s = |k: &str| serde_json::to_string(args[k].as_str().unwrap_or("")).unwrap_or_default();
    let range = |k: &str, default: &str| -> Result<String, String> {
        let r = args[k].as_str().filter(|r| !r.is_empty()).unwrap_or(default).to_ascii_uppercase();
        if range_ok(&r) { Ok(r) } else { Err(format!("{k} {r:?}: a cell like B5 or a range like A2:J40")) }
    };
    let body = match tool {
        "sheets_create" => {
            let cols = tsv(&json!([args["columns"].clone()]))?;
            let n = args["columns"].as_array().map_or(0, |c| c.len());
            format!(r#"await ready(); await sleep(1500);
  const t = document.querySelector('.docs-title-input');
  t.focus(); t.value = {title}; t.dispatchEvent(new Event('input', {{bubbles: true}})); t.dispatchEvent(new Event('change', {{bubbles: true}})); t.blur();
  await sleep(1000); await go('A1'); clip('paste', {cols}); await sleep(800);
  await go('A1');
  return 'created ' + where() + ' with the header row (' + {n} + ' columns). Add rows with sheets_append (url: this address; a first cell FALSE or TRUE makes it a tick box).';"#,
                    title = s("title"), cols = serde_json::to_string(&cols).unwrap_or_default())
        }
        "sheets_read" => {
            let r = range("range", "A:Z")?;
            format!(r#"await ready(); await go('{r}'); const t = clip('copy'); await go('A1'); return JSON.stringify({{where: where(), text: t}});"#)
        }
        "sheets_append" => {
            let rows = serde_json::to_string(&args["rows"]).unwrap_or_default();
            format!(r#"await ready(); const h = await header(); const rows = place(h, {rows}), n = rows.length;
  await go('A:Z');
  const have = clip('copy').split('\n'); let last = 0;
  have.forEach((l, i) => {{ if (l.split('\t').some(filled)) last = i + 1; }});
  await go('A' + (last + 1)); clip('paste', rows.map((r) => r.join('\t')).join('\n')); await sleep(800);
  // first cells all FALSE / TRUE: tick boxes, in these rows only (pre-filled boxes under the data were rows to
  // My Maps' import: 181 rows with no address, and "trouble finding those locations")
  const boxes = rows.every((r) => r[0] === 'FALSE' || r[0] === 'TRUE');
  if (boxes) {{ await go('A' + (last + 1) + ':A' + (last + n)); await menu('#docs-insert-menu', 'Checkbox'); }}
  await go('A1');
  const used = h.filter((c, i) => rows.some((r) => r[i]));
  return 'added ' + n + ' rows at rows ' + (last + 1) + '-' + (last + n) + ' of ' + where() + ', filling ' + used.join(', ') + (boxes ? ' (' + h[0] + ' as tick boxes)' : '');"#)
        }
        "sheets_write" => {
            let row = args["row"].as_u64().unwrap_or(0);
            if row < 2 {
                return Err("row: the row's number, 2 or more (row 1 is the header; sheets_read shows the numbers)".into());
            }
            let values = serde_json::to_string(&args["values"]).unwrap_or_default();
            format!(r#"await ready(); const h = await header(); const end = letter(h.length - 1);
  const [want] = place(h, [{values}]);
  await go('A{row}:' + end + '{row}');
  const now = clip('copy').split('\n')[0].split('\t');
  const merged = h.map((c, i) => (want[i] !== '' || Object.keys({values}).some((k) => k.trim().toLowerCase() === c.toLowerCase())) ? want[i] : (now[i] || '').trim());
  await go('A{row}'); clip('paste', merged.join('\t')); await sleep(800); await go('A1');
  return 'updated row {row} of ' + where() + ': ' + Object.keys({values}).join(', ');"#)
        }
        "sheets_delete_rows" => {
            let (from, to) = (args["from"].as_u64().unwrap_or(0), args["to"].as_u64().or(args["from"].as_u64()).unwrap_or(0));
            if from < 2 || to < from {
                return Err("from / to: row numbers, 2 or more (row 1 is the header), from <= to".into());
            }
            let item = if from == to { format!("Row {from}") } else { format!("Rows {from} - {to}") };
            format!(r#"await ready(); await go('A{from}:A{to}'); await menu('#docs-edit-menu', 'Delete');
  // the submenu is open now: its item, clicked where it is (the menu's button again would close it)
  const it = [...document.querySelectorAll('.goog-menu .goog-menuitem')].find((e) => e.offsetParent && e.innerText.trim().startsWith('{item}'));
  if (!it) throw new Error('no menu item {item}'); mouse(it); await sleep(800);
  await go('A1'); return 'deleted {item} (the rows under moved up) in ' + where();"#)
        }
        "sheets_clear" => {
            let r = range("range", "")?;
            format!(r#"await ready(); await go('{r}'); del(); await sleep(600); await go('A1'); return 'cleared {r} in ' + where();"#)
        }
        "sheets_trash" => r#"await ready(); const w = where(); await menu('#docs-file-menu', 'Move to trash'); await sleep(1500); return 'moved to the trash: ' + w;"#.to_string(),
        other => return Err(format!("no tool {other}")),
    };
    Ok(format!("{PRELUDE}\n{body}"))
}

/// My Maps' helpers, before each body.
pub const MAPS_PRELUDE: &str = r#"
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const byText = (sel, t) => [...document.querySelectorAll(sel)].find((e) => e.offsetParent && (e.innerText || e.textContent || '').trim() === t);
const mouse = (el) => { const r = el.getBoundingClientRect();
  for (const t of ['mousedown', 'mouseup', 'click']) el.dispatchEvent(new MouseEvent(t, {bubbles: true, cancelable: true, view: window, clientX: r.x + r.width / 2, clientY: r.y + r.height / 2, button: 0})); };
const set = (el, v) => { const p = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  el.focus(); Object.getOwnPropertyDescriptor(p, 'value').set.call(el, v); el.dispatchEvent(new Event('input', {bubbles: true})); el.dispatchEvent(new Event('change', {bubbles: true})); };
const wait = async (f, what) => { for (let i = 0; i < 60; i++) { const v = f(); if (v) return v; await sleep(250); } throw new Error(what); };
// the deepest visible element with exactly this text (My Maps' Import is a div around other elements: no leaf)
// a column by its header name (any case) or its letter (C: the third); -1 when neither.  A model gave A, C and B for
// Address and Description, three times, and heard only "no column A"
const pickColumn = (names, want) => { const w = String(want || '').trim(), i = names.findIndex((n) => n.toLowerCase() === w.toLowerCase());
  if (i >= 0) return i;
  if (/^[A-Za-z]{1,2}$/.test(w)) { const k = [...w.toUpperCase()].reduce((n, ch) => n * 26 + ch.charCodeAt(0) - 64, 0) - 1; if (k < names.length) return k; }
  return -1; };
const leaf = (t) => [...document.querySelectorAll('*')].find((e) => e.offsetParent && e.textContent.trim() === t && ![...e.children].some((c) => c.textContent.trim() === t));
"#;

/// A My Maps tool's script.  maps_create is two pages: the list's Create (phase 1: the page goes), then the new map's
/// title in its editor (phase 2).  maps_trash: the map's card in the list, its menu, Move to Trash, the dialog's yes.
pub fn maps_js(tool: &str, args: &Value, token: &str) -> Result<String, String> {
    let s = |k: &str| serde_json::to_string(args[k].as_str().unwrap_or("")).unwrap_or_default();
    let body = match tool {
        "maps_create" if args["__phase"] == 2 => format!(r#"
const t = await wait(() => leaf('Untitled map'), 'no title on the new map');
mouse(t);
const input = await wait(() => [...document.querySelectorAll('input')].find((e) => e.offsetParent && e.placeholder === 'Untitled map'), 'no title dialog');
set(input, {title});
const d = [...document.querySelectorAll('textarea')].find((e) => e.offsetParent);
if (d && {desc}) set(d, {desc});
mouse(await wait(() => byText('button', 'Save'), 'no Save'));
await sleep(1500);
return 'created the map ' + document.title + ' (' + location.href.split('&')[0] + '). To put places on it: maps_import_sheet.';"#, title = s("title"), desc = s("description")),
        // a map with this title already: its address, not a second map with the same name (a model that lost track
        // repeats a create: idempotent, as RFC 9110 9.2.2 means it).  The list's card holds the map's id (data-mid);
        // clicking the card instead took the page away before the answer was read
        "maps_create" => format!(r#"
await wait(() => byText('button,[role=button]', 'add_to_drive'), 'no Create button in My Maps (signed in to Google?)');
const name = {title}, have = leaf(name);
if (have) {{
  // its card: data-mid with data-name (menus carry data-mid too, one a "dummyId")
  const card = have.closest('[data-mid][data-name]'), mid = card && card.getAttribute('data-mid');
  if (mid) return 'a map titled ' + name + ' already exists, so none was made: https://www.google.com/maps/d/edit?mid=' + mid + '. To put places on it: maps_import_sheet (each import adds a layer).';
}}
mouse(byText('button,[role=button]', 'add_to_drive'));
mouse(await wait(() => byText('button,[role=button],div', 'CREATE'), 'no CREATE in the dialog'));
return '__next';"#, title = s("title")),
        "maps_trash" => format!(r#"
const name = {title};
const t = await wait(() => leaf(name), 'no map titled ' + name + ' in My Maps');
let card = t;
for (let i = 0; i < 12 && card && !card.querySelector('[aria-label="Open the map menu"]'); i++) card = card.parentElement;
if (!card) throw new Error('no menu for the map ' + name);
mouse(card.querySelector('[aria-label="Open the map menu"]'));
mouse(await wait(() => [...document.querySelectorAll('[role=menuitem]')].find((e) => e.offsetParent && e.innerText.trim() === 'Move to Trash'), 'no Move to Trash'));
mouse(await wait(() => byText('[role=button],button', 'MOVE TO TRASH'), 'no confirmation'));
await sleep(2000);
return 'moved the map ' + name + ' to the trash';"#, title = s("title")),
        "maps_import_sheet" => format!(r#"
// the picker is another origin's frame: its end of the bridge (browser_mcp::frame_bridge_js) acts for these messages
const TOKEN = '{token}';
let seq = 0;
const picker = (op, args) => new Promise((ok, no) => {{
  const f = [...document.querySelectorAll('iframe')].find((f) => (f.src || '').startsWith('https://docs.google.com/picker') && f.getBoundingClientRect().width > 0);
  if (!f) return no(new Error('the file picker is not open'));
  const id = 'imp' + (++seq) + Math.random();
  const on = (e) => {{ if (e.origin === 'https://docs.google.com' && e.data && e.data.strataReply === TOKEN && e.data.id === id) {{ removeEventListener('message', on); ok(JSON.parse(e.data.out)); }} }};
  addEventListener('message', on);
  f.contentWindow.postMessage({{strata: TOKEN, op, args, id}}, 'https://docs.google.com');
  setTimeout(() => {{ removeEventListener('message', on); no(new Error('the file picker did not answer')); }}, 10000);
}});
const num = (page, kind, name) => {{ const l = (page.elements || []).find((e) => e.startsWith('[') && e.includes('] ' + kind) && e.includes('"' + name + '"'));
  return l ? +l.slice(1, l.indexOf(']')) : 0; }};
const sheet = {sheet}, loc = {loc}, ttl = {ttl};
// the map open?  A trashed one sends My Maps home ("The map you tried to access is in your Trash"): said so, with the
// maps there are (a model kept importing into a trashed map 8 times, hearing only "no Add layer")
await wait(() => leaf('Add layer') || leaf('Import') || !location.pathname.startsWith('/maps/d/edit'), 'the map page did not load');
if (!location.pathname.startsWith('/maps/d/edit')) {{
  const said = (document.body.innerText.match(/[^\n]*(Trash|exist|permission|access)[^\n]*/) || [''])[0].trim().slice(0, 160);
  // the cards (data-mid with data-name): the share and map menus carry data-mid too, one with a "dummyId"
  const maps = [...new Set([...document.querySelectorAll('[data-mid][data-name]')].map((c) => c.getAttribute('data-name') + ' (https://www.google.com/maps/d/edit?mid=' + c.getAttribute('data-mid') + ')'))];
  throw new Error((/Trash/.test(said) ? 'that map is in the trash (gone: use another)' : 'that map did not open (' + (said || location.href) + ')') + '; the maps in My Maps: ' + (maps.join('; ') || 'none') + '. maps_create with a title gives the address of the map with that title');
}}
let imp = leaf('Import');
if (!imp) {{ mouse(await wait(() => leaf('Add layer'), 'no Add layer')); imp = await wait(() => leaf('Import'), 'no Import on the new layer'); }}
mouse(imp);
await wait(() => [...document.querySelectorAll('iframe')].some((f) => (f.src || '').startsWith('https://docs.google.com/picker') && f.getBoundingClientRect().width > 0), 'the file picker did not open');
let page; for (let i = 0; i < 20; i++) {{ await sleep(500); try {{ page = await picker('read', [1, 3500]); if (num(page, 'option', sheet)) break; }} catch (e) {{}} }}
let n = page ? num(page, 'option', sheet) : 0;
if (!n && page) {{ const q = num(page, 'field text', 'Search in Drive or paste URL'); if (q) {{ page = (await picker('type', [q, sheet, true])).read; await sleep(1500); page = await picker('read', [1, 3500]); n = num(page, 'option', sheet); }} }}
if (!n) throw new Error('no spreadsheet titled ' + sheet + ' in the file picker');
page = (await picker('click', [n])).read;
const ins = num(page, 'button', 'Insert') || num(page, 'button', 'Select');
if (!ins) throw new Error('no Insert in the file picker');
await picker('click', [ins]);
// the column step: as soon as it shows its columns (no waiting out a name that is not there)
const labels = await wait(() => {{ const l = [...document.querySelectorAll('[id^=upload-checkbox-label-]')].filter((e) => e.offsetParent); return l.length ? l : null; }}, 'My Maps did not ask which column places the markers');
const names = labels.map((e) => e.textContent.trim());
const li = pickColumn(names, loc), ti = pickColumn(names, ttl);
if (li < 0 || ti < 0) {{
  const c = byText('button,[role=button]', 'Cancel'); if (c) mouse(c);   // the half-done import is not left open
  throw new Error('no column ' + (li < 0 ? loc : ttl) + ' in ' + sheet + '; its columns are: ' + names.join(', ') + ' (give a name or a letter: C is the third)');
}}
const box = document.querySelector('[aria-labelledby="' + labels[li].id + '"]');
// one location column: any other ticked (My Maps ticks a likely one itself) is unticked
for (const o of labels) {{ const b = document.querySelector('[aria-labelledby="' + o.id + '"]'); if (b !== box && b.getAttribute('aria-checked') === 'true') mouse(b); }}
if (box.getAttribute('aria-checked') !== 'true') mouse(box);
mouse(await wait(() => byText('button,[role=button]', 'Continue'), 'no Continue'));
const radio = await wait(() => [...document.querySelectorAll('[role=radio]')].find((e) => e.offsetParent && (e.innerText || '').trim() === names[ti]), 'no column ' + names[ti] + ' for the titles');
mouse(radio);
mouse(await wait(() => byText('button,[role=button]', 'Finish'), 'no Finish'));
// done: the new layer (named after the sheet) in the map's list, or My Maps' "trouble finding those locations"
const done = await wait(() => /trouble finding those locations/.test(document.body.innerText) ? 'fail' : (leaf(sheet) ? 'ok' : ''), 'the import did not finish');
if (done === 'fail') throw new Error('My Maps could not place the rows by ' + loc + ' (rows with nothing in that column?)');
await sleep(2000);
const t = document.body.innerText, all = t.match(/All items\s*\((\d+)\)/), bad = t.match(/(\d+) rows? couldn.t be shown/);
return 'imported ' + sheet + ' into ' + document.title + ' as a layer' + (all ? ' of ' + all[1] + ' places' : '') + ', placed by ' + names[li] + ', titled by ' + names[ti]
  + (bad ? '; ' + bad[1] + ' row(s) could not be placed (My Maps found no place for their ' + loc + ': fix them in the sheet, then import again)' : '')
  + ' (' + location.href.split('&')[0] + ')';"#,
                    sheet = s("sheet_title"), loc = s("location_column"), ttl = s("title_column")),
        other => return Err(format!("no tool {other}")),
    };
    Ok(format!("{MAPS_PRELUDE}\n{body}"))
}

/// The page script for a body that awaits: its value (or error) kept in window.__strataAsync[id], for async_collect_js.
pub fn async_js(id: u64, body: &str) -> String {
    format!(r#"(() => {{ window.__strataAsync = window.__strataAsync || {{}}; window.__strataAsync[{id}] = null;
  (async () => {{ {body} }})().then((v) => {{ window.__strataAsync[{id}] = JSON.stringify({{ok: v}}); }},
    (e) => {{ window.__strataAsync[{id}] = JSON.stringify({{error: String((e && e.message) || e)}}); }});
  return 'started'; }})()"#)
}

/// The script's value; {gone} when the page it ran in has been replaced (a redirect took it away).
pub fn async_collect_js(id: u64) -> String {
    format!("(() => {{ if (!window.__strataAsync) return JSON.stringify({{gone: location.href}}); const o = window.__strataAsync[{id}]; return o ? o : 'null'; }})()")
}

/// Is `tool` safe to run again from the start when its page was replaced under it (it had changed nothing yet)?  A
/// trashed map's address loads, then My Maps sends it home: the import's script went with the first page.
pub fn rerun_safe(tool: &str) -> bool {
    matches!(tool, "sheets_read" | "maps_import_sheet")
}

/// The tool's answer from the script's value.
pub fn answer(tool: &str, args: &Value, ok: &Value) -> String {
    let text = ok.as_str().unwrap_or("").to_string();
    if tool == "sheets_read" {
        let v: Value = serde_json::from_str(&text).unwrap_or_default();
        let range = args["range"].as_str().filter(|r| !r.is_empty()).unwrap_or("A:Z").to_ascii_uppercase();
        return format!("{}\n\n{}", v["where"].as_str().unwrap_or(""), numbered(v["text"].as_str().unwrap_or(""), first_row(&range)));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_and_addresses() {
        for ok in ["A1", "B5", "A2:J40", "AA10:AB12", "A:Z", "B:B"] {
            assert!(range_ok(ok), "{ok}");
        }
        for bad in ["", "5", "A", "a1", "A1:B2:C3", "A1');alert(1);('", "ABCD1", "A:", ":Z", "A1:Z"] {
            assert!(!range_ok(bad), "{bad}");
        }
        assert_eq!(sheet_base("https://docs.google.com/spreadsheets/d/1xsvX_nD7BEB6Xft/edit?gid=0#gid=0").as_deref(), Some("https://docs.google.com/spreadsheets/d/1xsvX_nD7BEB6Xft"));
        assert_eq!(sheet_base("https://evil.test/spreadsheets/d/1xsvX_nD7BEB6Xft"), None);
        assert!(page_for("sheets_read", &json!({}), "https://example.com/").is_err());
        assert_eq!(page_for("sheets_read", &json!({}), "https://docs.google.com/spreadsheets/d/1xsvX_nD7BEB6Xft/edit#gid=0").unwrap().0, "https://docs.google.com/spreadsheets/d/1xsvX_nD7BEB6Xft/edit");
    }

    #[test]
    fn rows_paste_and_read_back() {
        assert_eq!(tsv(&json!([["a", "b\tc"], ["x\ny", 4.5, null]])).unwrap(), "a\tb c\nx y\t4.5\t");
        assert!(tsv(&json!("no")).is_err());
        let copied = "Checkbox\tRating\tAddress\t\nFALSE\t4.5\t123 Main St\t\n\t\t\t\nFALSE\t\t9 B St\t\n\t\t\t";
        assert_eq!(numbered(copied, 1), "row 1: Checkbox | Rating | Address\nrow 2: FALSE | 4.5 | 123 Main St\nrow 3:  |  | \nrow 4: FALSE |  | 9 B St");
        assert_eq!(numbered("\t\n\t", 1), "(empty)");
        // unticked boxes under the data are not rows
        assert_eq!(numbered("Checkbox\tName\nFALSE\tA\nFALSE\t\nFALSE\t", 1), "row 1: Checkbox | Name\nrow 2: FALSE | A");
        assert_eq!(first_row("A12:J40"), 12);
        assert_eq!(first_row("C3"), 3);
        assert_eq!(first_row("A:Z"), 1);
        let many: String = (1..=400).map(|i| format!("r{i}\tx")).collect::<Vec<_>>().join("\n");
        let out = numbered(&many, 1);
        assert!(out.contains("row 300: r300 | x") && !out.contains("row 301:") && out.ends_with("[100 more rows: sheets_read with range A301:Z600]"), "{}", &out[out.len() - 80..]);
    }

    #[test]
    fn my_maps_pages() {
        assert!(on_page(MAPS_LIST, "https://www.google.com/maps/d/"));
        assert!(!on_page(MAPS_LIST, "https://www.google.com/maps/d/edit?mid=1"), "the editor is not the list");
        assert!(on_page(MAPS_EDIT, "https://www.google.com/maps/d/edit?mid=1"));
        assert_eq!(page_for("maps_create", &json!({"__phase": 2}), "").unwrap().1, MAPS_EDIT);
        let create = maps_js("maps_create", &json!({"title": "studios"}), "t").unwrap();
        assert!(create.contains("return '__next'") && create.contains(r#"const name = "studios", have = leaf(name);"#) && create.contains("already exists, so none was made"));
        assert!(maps_js("maps_create", &json!({"__phase": 2, "title": "a\"b"}), "t").unwrap().contains(r#"set(input, "a\"b")"#));
        let (load, prefix) = page_for("maps_import_sheet", &json!({"map_url": "https://www.google.com/maps/d/edit?mid=1JYY-x_q&ll=1"}), "").unwrap();
        assert_eq!(load, "https://www.google.com/maps/d/edit?mid=1JYY-x_q");
        assert!(!on_page(&prefix, &load), "the map is loaded fresh for each import");
        assert!(page_for("maps_import_sheet", &json!({"map_url": "https://www.google.com/maps/d/edit?mid=a');x('"}), "").is_err());
        assert!(maps_js("maps_import_sheet", &json!({"sheet_title": "s", "location_column": "Address", "title_column": "Description"}), "tok1").unwrap().contains("const TOKEN = 'tok1'"));
        assert!(ask("maps_trash", &json!({"title": "x"})).unwrap().contains("\u{201c}x\u{201d}"));
        assert_eq!(ask("sheets_read", &json!({})), None);
        assert!(rerun_safe("maps_import_sheet") && rerun_safe("sheets_read"));
        assert!(!rerun_safe("sheets_append") && !rerun_safe("maps_create"), "a create or append is never run twice");
        assert!(async_collect_js(7).contains("gone: location.href"));
    }

    #[test]
    fn rows_by_number_and_name() {
        assert!(sheets_js("sheets_write", &json!({"row": 1, "values": {"A": "x"}})).is_err(), "the header stays");
        assert!(sheets_js("sheets_write", &json!({"row": 5, "values": {"Price": "$1"}})).unwrap().contains("go('A5:' + end + '5')"));
        assert!(sheets_js("sheets_delete_rows", &json!({"from": 1})).is_err(), "the header stays");
        assert!(sheets_js("sheets_delete_rows", &json!({"from": 5, "to": 7})).unwrap().contains("Rows 5 - 7"));
        assert!(sheets_js("sheets_delete_rows", &json!({"from": 5})).unwrap().contains("'Row 5'"));
    }

    #[test]
    fn scripts_take_only_checked_ranges() {
        assert!(sheets_js("sheets_clear", &json!({"range": "A1');x('"})).is_err());
        let js = sheets_js("sheets_append", &json!({"rows": [{"Name": "it's \"q\""}]})).unwrap();
        assert!(js.contains(r#"[{"Name":"it's \"q\""}]"#));
        assert!(!sheets_js("sheets_create", &json!({"title": "t", "columns": ["A", "B"]})).unwrap().contains("Checkbox"), "no tick boxes before there are rows");
    }
}
