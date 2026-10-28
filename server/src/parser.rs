//! The model's streamed text -> reasoning, content and tool-call events (serve/frontend.py's OutputParser,
//! HarmonyParser and GlmParser).  A tag split across deltas is held back until whole, so clients never see `<tool_`
//! or `</thi`.  Offsets are byte offsets; every tag is ASCII, so a held suffix is the same in bytes and characters.

use crate::py::{self, find, find0, lstrip, strip};
use crate::pyjson;
use serde_json::{Map, Value};

pub const THINK_END: &str = "</think>";
pub const CALL_START: &str = "<tool_call>";
pub const CALL_END: &str = "</tool_call>";
pub const PARAM_END: &str = "</parameter>";
pub const PARAM_START: &str = "<parameter=";
pub const FUNC_START: &str = "<function=";
pub const FUNC_END: &str = "</function>";
/// the most of a `<tool_call>` inside the reasoning that is held while waiting for its end (#804)
pub const RCALL_MAX: usize = 32768;

#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Map<String, Value>,
    pub id: String,
}

pub fn call_id() -> String {
    format!("call_{}", &uuid::Uuid::new_v4().simple().to_string()[..24])
}

impl ToolCall {
    pub fn new(name: impl Into<String>, arguments: Map<String, Value>) -> Self {
        ToolCall { name: name.into(), arguments, id: call_id() }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Reasoning,
    Content,
    ToolCall,
    ToolStart,
    ToolArgs,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Reasoning => "reasoning",
            Kind::Content => "content",
            Kind::ToolCall => "tool_call",
            Kind::ToolStart => "tool_start",
            Kind::ToolArgs => "tool_args",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub kind: Kind,
    pub text: String,
    pub call: Option<ToolCall>,
}

impl Event {
    pub fn text(kind: Kind, text: impl Into<String>) -> Self {
        Event { kind, text: text.into(), call: None }
    }
    pub fn call(kind: Kind, call: ToolCall) -> Self {
        Event { kind, text: String::new(), call: Some(call) }
    }
}

/// `json.loads(s)` for a parameter value: None when it is no JSON.
fn json_loads(s: &str) -> Option<Value> {
    serde_json::from_str(s).ok()
}

/// Where a parameter value in `text` ends: the first `</parameter>` followed (after whitespace) by the next
/// `<parameter=` or `</function>`.  -1: none yet; -2: a candidate whose follower has not arrived (`final` accepts it).
pub fn param_end(text: &str, fin: bool) -> isize {
    let mut at = find0(text, PARAM_END);
    while at >= 0 {
        let after = lstrip(&text[at as usize + PARAM_END.len()..]);
        if after.starts_with(PARAM_START) || after.starts_with(FUNC_END) {
            return at;
        }
        if after.is_empty() || PARAM_START.starts_with(after) || FUNC_END.starts_with(after) {
            return if fin { at } else { -2 };
        }
        at = find(text, PARAM_END, at as usize + 1);
    }
    -1
}

/// Where a tool call's body ends (#210): the `</tool_call>` after the call's own `</function>`.  -1: not complete.
pub fn call_end(text: &str) -> isize {
    let s = lstrip(text);
    let mut pos = text.len() - s.len();
    if !s.starts_with(FUNC_START) {
        return if !FUNC_START.starts_with(s) { find0(text, CALL_END) } else { -1 };
    }
    let gt = find(text, ">", pos);
    if gt < 0 {
        return -1;
    }
    pos = gt as usize + 1;
    loop {
        let rest = &text[pos..];
        let s = lstrip(rest);
        pos += rest.len() - s.len();
        if s.starts_with(PARAM_START) {
            let gt = find(text, ">", pos);
            if gt < 0 {
                return -1;
            }
            let end = param_end(&text[gt as usize + 1..], false);
            if end < 0 {
                return -1;
            }
            pos = gt as usize + 1 + end as usize + PARAM_END.len();
        } else if s.starts_with(FUNC_END) {
            let rest = &text[pos + FUNC_END.len()..];
            let s = lstrip(rest);
            if s.starts_with(CALL_END) {
                return (pos + FUNC_END.len() + rest.len() - s.len()) as isize;
            }
            return if CALL_END.starts_with(s) { -1 } else { find(text, CALL_END, pos) };
        } else if s.is_empty() || PARAM_START.starts_with(s) || FUNC_END.starts_with(s) {
            return -1;
        } else {
            return find(text, CALL_END, pos);
        }
    }
}

/// Where a `<function=NAME>...</function>` at the start of `text` ends (just past `</function>`); -1: not complete.
pub fn function_end(text: &str) -> isize {
    let s = lstrip(text);
    let mut pos = text.len() - s.len();
    if !s.starts_with(FUNC_START) {
        return -1;
    }
    let gt = find(text, ">", pos);
    if gt < 0 {
        return -1;
    }
    pos = gt as usize + 1;
    loop {
        let rest = &text[pos..];
        let s = lstrip(rest);
        pos += rest.len() - s.len();
        if s.starts_with(PARAM_START) {
            let gt = find(text, ">", pos);
            if gt < 0 {
                return -1;
            }
            let end = param_end(&text[gt as usize + 1..], false);
            if end < 0 {
                return -1;
            }
            pos = gt as usize + 1 + end as usize + PARAM_END.len();
        } else if s.starts_with(FUNC_END) {
            return (pos + FUNC_END.len()) as isize;
        } else {
            return -1;
        }
    }
}

/// A client's `tool_choice` in either API's shape -> ("auto" | "none" | "required" | "named" | "unknown", name).
pub fn tool_choice_of(tc: &Value) -> (&'static str, Option<String>) {
    match tc {
        Value::Null => ("auto", None),
        Value::String(s) => match s.as_str() {
            "auto" => ("auto", None),
            "none" => ("none", None),
            "required" => ("required", None),
            _ => ("unknown", None),
        },
        Value::Object(_) => {
            let kind = py::get(tc, "type").as_str().unwrap_or("");
            match kind {
                "auto" => ("auto", None),
                "none" => ("none", None),
                "any" | "required" => ("required", None),
                "function" | "tool" => {
                    let fname = py::get(py::get(tc, "function"), "name");
                    let name = if py::truthy(fname) { fname } else { py::get(tc, "name") };
                    match name.as_str() {
                        Some(n) if !n.is_empty() => ("named", Some(n.to_string())),
                        _ => ("unknown", None),
                    }
                }
                _ => ("unknown", None),
            }
        }
        _ => ("unknown", None),
    }
}

/// `tool_choice` -> the text that opens the call the reply must make, or None (the model decides).
pub fn forced_call(tc: &Value, tools: Option<&[Value]>) -> Option<String> {
    let (mut kind, mut name) = tool_choice_of(tc);
    let mut names: Vec<String> = vec![];
    for t in tools.unwrap_or(&[]) {
        if t.is_object() {
            let n = py::get(t, "name").as_str().map(String::from).unwrap_or_default();
            if !names.contains(&n) {
                names.push(n);
            }
        }
    }
    if kind == "auto" || kind == "none" {
        return None;
    }
    if kind == "required" && names.len() == 1 {
        kind = "named";
        name = Some(names[0].clone());
    }
    if kind == "required" && !names.is_empty() {
        return Some(format!("{CALL_START}\n<function="));
    }
    if kind == "named" {
        if let Some(n) = name.filter(|n| names.contains(n)) {
            return Some(format!("{CALL_START}\n<function={n}>\n"));
        }
    }
    let shown: String = pyjson::dumps(tc).chars().take(200).collect();
    crate::say(&format!("tool_choice {shown} is not supported here (or names no tool of the request): the model decides, as with \"auto\""));
    None
}

/// `{"name": N, "arguments": {...}}` (or "parameters", or the arguments as a JSON string) of a declared tool.
pub fn json_tool_call(body: &str, schemas: &Map<String, Value>) -> Option<ToolCall> {
    let obj = json_loads(body)?;
    let name = obj.as_object()?.get("name")?.as_str()?.to_string();
    if !schemas.contains_key(&name) {
        return None;
    }
    let o = obj.as_object()?;
    let mut args = o.get("arguments").or_else(|| o.get("parameters")).cloned().unwrap_or(Value::Object(Map::new()));
    if let Value::String(s) = &args {
        args = json_loads(s)?;
    }
    if args.is_null() {
        args = Value::Object(Map::new());
    }
    match args {
        Value::Object(m) => Some(ToolCall::new(name, m)),
        _ => None,
    }
}

/// `<function=NAME>\n<parameter=P>\nVALUE\n</parameter>...</function>` -> ToolCall.  Values are JSON-decoded unless
/// the tool's schema says the parameter is a string.
pub fn parse_tool_call(body: &str, schema: Option<&Value>) -> Result<ToolCall, String> {
    let body = strip(body);
    let Some(gt) = body.find('>').filter(|_| body.starts_with(FUNC_START)) else {
        return Err(format!("malformed tool call: {}", py::chars_prefix(body, 80)));
    };
    let name = strip(&body[FUNC_START.len()..gt]).to_string();
    let mut rest = &body[gt + 1..];
    let empty = Value::Null;
    let props = schema.map(|s| py::get(py::get(s, "parameters"), "properties")).unwrap_or(&empty);
    let mut args = Map::new();
    while let Some(i) = rest.find(PARAM_START) {
        rest = &rest[i + PARAM_START.len()..];
        let gt = rest.find('>').ok_or_else(|| "substring not found".to_string())?;
        let pname = rest[..gt].to_string();
        rest = &rest[gt + 1..];
        let end = param_end(rest, true);
        let mut value = if end >= 0 { &rest[..end as usize] } else { rest };
        rest = if end >= 0 { &rest[end as usize + PARAM_END.len()..] } else { "" };
        value = value.strip_prefix('\n').unwrap_or(value);
        value = value.strip_suffix('\n').unwrap_or(value);
        let declared = py::get(py::get(props, &pname), "type");
        let v = if declared.as_str() == Some("string") {
            Value::String(value.to_string())
        } else {
            json_loads(value).unwrap_or_else(|| Value::String(value.to_string()))
        };
        args.insert(pname, v);
    }
    Ok(ToolCall::new(name, args))
}

/// The arguments a client's history (or a harmony call) carries -> the mapping the template renders.
pub fn tool_arguments(raw: &Value) -> Map<String, Value> {
    match raw {
        Value::Null => Map::new(),
        Value::String(s) if strip(s).is_empty() => Map::new(),
        Value::Object(m) => m.clone(),
        Value::String(s) => match json_loads(s) {
            Some(Value::Object(m)) => m,
            _ => Map::from_iter([("arguments".to_string(), Value::String(s.clone()))]),
        },
        other => Map::from_iter([("arguments".to_string(), Value::String(pyjson::dumps_u(other)))]),
    }
}

/// Length of the longest suffix of `text` that is a proper prefix of one of `tags`.
fn hold(text: &str, tags: &[&str]) -> usize {
    let mut best = 0;
    for tag in tags {
        for n in 1..tag.len() {
            if text.ends_with(&tag[..n]) {
                best = best.max(n);
            }
        }
    }
    best
}

/// Everything after the first `n` characters (`s[n:]`).
fn skip_chars(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[i..],
        None => "",
    }
}

/// `body.strip()[len("<function="):].split(">", 1)[0].strip()`
fn call_name(body: &str) -> String {
    let s = skip_chars(strip(body), FUNC_START.len());
    strip(s.split('>').next().unwrap_or("")).to_string()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PState {
    Reasoning,
    Content,
    Call,
    Json,
    Rcall,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Scan {
    Name,
    Between,
    Str,
    Raw,
    Done,
}

/// What the server reads of any parser.
pub trait Parse: Send {
    fn feed(&mut self, delta: &str) -> Result<Vec<Event>, String>;
    fn finish(&mut self, reason: Option<&str>) -> Result<Vec<Event>, String>;
    /// "reasoning" | "content" | "call" | "json" | "rcall" | "header"
    fn state(&self) -> &'static str;
    /// What is held back (the server's "clean point" is an empty buffer).
    fn buf(&self) -> String;
    fn pending(&self) -> usize;
    fn rescued(&self) -> usize;
    fn refused(&self) -> usize;
}

/// Incremental parser of the model's text (Qwen's `<think>` and `<tool_call><function=...>` form).
pub struct OutputParser {
    state: PState,
    pub buf: String,
    lead: bool,
    schemas: Map<String, Value>,
    stream_tools: bool,
    /// #804/#1058: calls found inside the reasoning, (raw text, call) until the turn shows they were acts
    pending: Vec<(String, Option<ToolCall>)>,
    pub rescued: usize,
    pub refused: usize,
    fence: String,
    line: String,
    ticks: usize,
    recover: bool,
    bare: bool,
    batch: bool,
    sp: usize,
    ss: Scan,
    scall: Option<ToolCall>,
    sfirst: bool,
    sval_started: bool,
    sdeclared: Map<String, Value>,
}

impl OutputParser {
    pub fn new(thinking: bool, tools: Option<&[Value]>, stream_tools: bool, recover: bool) -> Self {
        let mut schemas = Map::new();
        for t in tools.unwrap_or(&[]) {
            let name = py::get(t, "name");
            // `{t.get("name"): t}`: a tool without a name is keyed None; the JSON key "None" stands in
            let key = match name {
                Value::String(s) => s.clone(),
                other => py::py_str(other),
            };
            schemas.insert(key, t.clone());
        }
        OutputParser {
            state: if thinking { PState::Reasoning } else { PState::Content },
            buf: String::new(),
            lead: false,
            schemas,
            stream_tools,
            pending: vec![],
            rescued: 0,
            refused: 0,
            fence: String::new(),
            line: String::new(),
            ticks: 0,
            recover,
            bare: false,
            batch: false,
            sp: 0,
            ss: Scan::Name,
            scall: None,
            sfirst: true,
            sval_started: false,
            sdeclared: Map::new(),
        }
    }

    fn snap(&self) -> (String, String, usize) {
        (self.fence.clone(), self.line.clone(), self.ticks)
    }

    fn restore(&mut self, s: (String, String, usize)) {
        (self.fence, self.line, self.ticks) = s;
    }

    /// self.buf[p] would open a call: at the start of a line, outside a code fence and inline code.
    fn ok_at(&mut self, p: usize) -> bool {
        let snap = self.snap();
        let head = self.buf[..p].to_string();
        self.track(&head);
        let ok = self.opener_ok();
        self.restore(snap);
        ok
    }

    /// recover: what a `<tool_call>` followed by `after` opens: "call", "param", "json", "wait" or "text".
    fn follower(&self, after: &str) -> &'static str {
        if after.starts_with(FUNC_START) {
            return "call";
        }
        if after.starts_with(PARAM_START) {
            return match after.find('>') {
                None => if after.contains('\n') { "text" } else { "wait" },
                Some(gt) => if self.schemas.contains_key(&after[PARAM_START.len()..gt]) { "param" } else { "text" },
            };
        }
        if after.starts_with('{') {
            return if !self.schemas.is_empty() { "json" } else { "text" };
        }
        if after.is_empty() || FUNC_START.starts_with(after) || PARAM_START.starts_with(after) {
            return "wait";
        }
        "text"
    }

    /// recover: the first `<function=` in self.buf that opens a call without the wrapper -> (position, decided).
    fn bare_opener(&mut self) -> (isize, bool) {
        let mut p = find0(&self.buf, FUNC_START);
        while p >= 0 {
            let pu = p as usize;
            if self.ok_at(pu) {
                let gt = find(&self.buf, ">", pu);
                let name_end = if gt >= 0 { gt as usize } else { self.buf.len() };
                let name = &self.buf[pu + FUNC_START.len()..name_end.max(pu + FUNC_START.len())];
                if gt < 0 && !name.chars().any(py::is_space) {
                    return (p, false);
                }
                if gt >= 0 && self.schemas.contains_key(name) {
                    return (p, true);
                }
            }
            p = find(&self.buf, FUNC_START, pu + 1);
        }
        (-1, false)
    }

    /// Follow the reasoning text that has gone out: the open code fence, the current line, the paragraph's backticks.
    fn track(&mut self, text: &str) -> String {
        if !text.contains('\n') {
            self.line.push_str(text);
            return text.to_string();
        }
        let parts: Vec<&str> = text.split('\n').collect();
        let last = parts.len() - 1;
        for (k, part) in parts.iter().enumerate() {
            if k < last {
                let line = std::mem::take(&mut self.line) + part;
                let s = lstrip(&line);
                if !self.fence.is_empty() {
                    if s.starts_with(&self.fence.repeat(3)) {
                        self.fence.clear();
                    }
                } else if {
                    let head = py::chars_prefix(s, 3);
                    (head == "```" || head == "~~~") && {
                        let c0: String = s.chars().take(1).collect::<String>().repeat(3);
                        !skip_chars(s, 3).contains(&c0)
                    }
                } {
                    self.fence = s.chars().take(1).collect();
                } else if s.is_empty() {
                    self.ticks = 0;
                } else {
                    self.ticks += line.matches('`').count();
                }
            } else {
                self.line.push_str(part);
            }
        }
        text.to_string()
    }

    fn opener_ok(&self) -> bool {
        self.fence.is_empty() && strip(&self.line).is_empty() && self.ticks % 2 == 0
    }

    fn in_code(&self) -> bool {
        !self.fence.is_empty() || (self.ticks + self.line.matches('`').count()) % 2 == 1
    }

    /// #537: the `</think>` at self.buf[i] is quoted while thinking (after a quote or backtick, not followed by a line
    /// break).  None while the character after it has not arrived.
    fn quoted_think_end(&self, i: usize) -> Option<bool> {
        let before = if i > 0 { py::char_before(&self.buf, i) } else { self.line.char_indices().next_back().map(|(j, _)| &self.line[j..]).unwrap_or("") };
        if !matches!(before, "\"" | "'" | "`") {
            return Some(false);
        }
        let after = &self.buf[i + THINK_END.len()..];
        let c = after.chars().next()?;
        Some(c != '\r' && c != '\n')
    }

    fn release(&mut self, deliver: bool) -> Vec<Event> {
        let mut out = vec![];
        for (raw, call) in std::mem::take(&mut self.pending) {
            match call {
                Some(c) if deliver => {
                    out.push(Event::call(Kind::ToolCall, c));
                    self.rescued += 1;
                }
                c => {
                    if c.is_some() {
                        self.refused += 1;
                    }
                    if !raw.is_empty() {
                        let t = self.track(&raw);
                        out.push(Event::text(Kind::Reasoning, t));
                    }
                }
            }
        }
        out
    }

    fn reset_scan(&mut self) {
        self.sp = 0;
        self.ss = Scan::Name;
        self.scall = None;
        self.sfirst = true;
        self.sval_started = false;
        self.sdeclared = Map::new();
    }

    /// Advance the streaming view of the call body in self.buf (stream_tools).
    fn scan(&mut self) -> Vec<Event> {
        let mut out = vec![];
        macro_rules! args {
            ($s:expr) => {{
                let s: String = $s;
                if !s.is_empty() {
                    out.push(Event { kind: Kind::ToolArgs, text: s, call: self.scall.clone() });
                }
            }};
        }
        loop {
            let rest = self.buf[self.sp..].to_string();
            match self.ss {
                Scan::Name => {
                    let a = find0(&rest, FUNC_START);
                    let b = if a >= 0 { find(&rest, ">", a as usize + 10) } else { -1 };
                    if b < 0 {
                        return out;
                    }
                    let name = rest[a as usize + 10..b as usize].to_string();
                    let call = ToolCall::new(name.clone(), Map::new());
                    let props = py::get(py::get(self.schemas.get(&name).unwrap_or(&Value::Null), "parameters"), "properties");
                    self.sdeclared = props.as_object().map(|p| p.iter().map(|(k, v)| (k.clone(), py::get(v, "type").clone())).collect()).unwrap_or_default();
                    self.scall = Some(call.clone());
                    out.push(Event::call(Kind::ToolStart, call));
                    args!("{".to_string());
                    self.sp += b as usize + 1;
                    self.ss = Scan::Between;
                }
                Scan::Between => {
                    let stripped = lstrip(&rest);
                    self.sp += rest.len() - stripped.len();
                    if stripped.starts_with(PARAM_START) {
                        let Some(b) = stripped.find('>') else { return out };
                        let pname = stripped[11..b].to_string();
                        args!(format!("{}{}:", if self.sfirst { "" } else { "," }, pyjson::dumps(&Value::String(pname.clone()))));
                        self.sfirst = false;
                        self.sp += b + 1;
                        if self.sdeclared.get(&pname).and_then(|v| v.as_str()) == Some("string") {
                            args!("\"".to_string());
                            self.ss = Scan::Str;
                            self.sval_started = false;
                        } else {
                            self.ss = Scan::Raw;
                        }
                    } else if stripped.starts_with(FUNC_END) {
                        args!("}".to_string());
                        self.sp += FUNC_END.len();
                        self.ss = Scan::Done;
                    } else {
                        return out;
                    }
                }
                Scan::Str => {
                    let mut rest = rest.as_str();
                    if !self.sval_started {
                        if rest.is_empty() {
                            return out;
                        }
                        if rest.starts_with('\n') {
                            self.sp += 1;
                            rest = &rest[1..];
                        }
                        self.sval_started = true;
                    }
                    let end = param_end(rest, false);
                    if end >= 0 {
                        let mut value = &rest[..end as usize];
                        value = value.strip_suffix('\n').unwrap_or(value);
                        let d = pyjson::dumps(&Value::String(value.to_string()));
                        args!(format!("{}\"", &d[1..d.len() - 1]));
                        self.sp += end as usize + PARAM_END.len();
                        self.ss = Scan::Between;
                        continue;
                    }
                    let mut safe = if end == -2 { find0(rest, PARAM_END) } else { (rest.len() - hold(rest, &[PARAM_END])) as isize };
                    if safe > 0 && rest.as_bytes()[safe as usize - 1] == b'\n' {
                        safe -= 1;
                    }
                    if safe > 0 {
                        let d = pyjson::dumps(&Value::String(rest[..safe as usize].to_string()));
                        args!(d[1..d.len() - 1].to_string());
                        self.sp += safe as usize;
                    }
                    return out;
                }
                Scan::Raw => {
                    let end = param_end(&rest, false);
                    if end < 0 {
                        return out;
                    }
                    let mut value = &rest[..end as usize];
                    value = value.strip_prefix('\n').unwrap_or(value);
                    value = value.strip_suffix('\n').unwrap_or(value);
                    let v = json_loads(value).unwrap_or_else(|| Value::String(value.to_string()));
                    args!(pyjson::dumps_u(&v));
                    self.sp += end as usize + PARAM_END.len();
                    self.ss = Scan::Between;
                }
                Scan::Done => return out,
            }
        }
    }

    /// A streamed call that ended without a clean </function>: close its JSON so clients can still parse it.
    fn close_scan(&mut self) -> Vec<Event> {
        if self.scall.is_none() || self.ss == Scan::Done {
            return vec![];
        }
        let mut tail = String::new();
        if self.ss == Scan::Str {
            tail.push('"');
        } else if self.ss == Scan::Raw {
            tail += &pyjson::dumps(&Value::String(self.buf[self.sp..].trim_matches('\n').to_string()));
        }
        tail.push('}');
        self.ss = Scan::Done;
        vec![Event { kind: Kind::ToolArgs, text: tail, call: self.scall.clone() }]
    }

    fn finish_call(&mut self, body: &str) -> Result<Vec<Event>, String> {
        let name = call_name(body);
        let bare = std::mem::replace(&mut self.bare, false);
        let mut call = match parse_tool_call(body, self.schemas.get(&name)) {
            Ok(c) => c,
            Err(e) => {
                if !self.recover {
                    return Err(e);
                }
                self.reset_scan();
                self.state = PState::Content;
                let text = if bare { body.to_string() } else { format!("{CALL_START}{body}{CALL_END}") };
                let t = self.track(&text);
                return Ok(vec![Event::text(Kind::Content, t)]);
            }
        };
        if let Some(s) = &self.scall {
            call.id = s.id.clone();
        }
        self.reset_scan();
        self.state = PState::Content;
        self.lead = true;
        Ok(vec![Event::call(Kind::ToolCall, call)])
    }

    fn emit(&mut self, out: &mut Vec<Event>, kind: Kind, text: &str) {
        let t = self.track(text);
        out.push(Event::text(kind, t));
    }

    pub fn feed_inner(&mut self, delta: &str) -> Result<Vec<Event>, String> {
        self.buf.push_str(delta);
        let mut out: Vec<Event> = vec![];
        loop {
            match self.state {
                PState::Rcall => {
                    let body = self.buf[CALL_START.len()..].to_string();
                    let (end, think) = (call_end(&body), find0(&body, THINK_END));
                    if end >= 0 && (think < 0 || think >= end) {
                        let e = end as usize;
                        let name = call_name(&body[..e]);
                        let call = if self.schemas.contains_key(&name) { parse_tool_call(&body[..e], self.schemas.get(&name)).ok() } else { None };
                        let raw = self.buf[..CALL_START.len() + e + CALL_END.len()].to_string();
                        if call.is_some() || !self.pending.is_empty() {
                            self.pending.push((raw, call));
                        } else {
                            self.emit(&mut out, Kind::Reasoning, &raw);
                        }
                        self.buf = body[e + CALL_END.len()..].to_string();
                        self.state = PState::Reasoning;
                    } else if think >= 0 {
                        out.extend(self.release(false));
                        let head = self.buf[..CALL_START.len() + think as usize].to_string();
                        self.emit(&mut out, Kind::Reasoning, &head);
                        self.buf = body[think as usize..].to_string();
                        self.state = PState::Reasoning;
                    } else if body.len() > RCALL_MAX {
                        out.extend(self.release(false));
                        let all = std::mem::take(&mut self.buf);
                        self.emit(&mut out, Kind::Reasoning, &all);
                        self.state = PState::Reasoning;
                    } else {
                        return Ok(out);
                    }
                }
                PState::Reasoning => {
                    if !self.pending.is_empty() {
                        let stripped = lstrip(&self.buf).to_string();
                        if stripped.len() < self.buf.len() {
                            let ws = self.buf[..self.buf.len() - stripped.len()].to_string();
                            self.pending.last_mut().expect("pending").0.push_str(&ws);
                            self.buf = stripped;
                        }
                        if self.buf.is_empty() {
                            return Ok(out);
                        }
                        if self.buf.starts_with(CALL_START) {
                            self.state = PState::Rcall;
                        } else if self.buf.starts_with(THINK_END) {
                            out.extend(self.release(true));
                        } else if CALL_START.starts_with(self.buf.as_str()) || THINK_END.starts_with(self.buf.as_str()) {
                            return Ok(out);
                        } else {
                            out.extend(self.release(false));
                        }
                        continue;
                    }
                    let i = find0(&self.buf, THINK_END);
                    let tool = if !self.schemas.is_empty() { find0(&self.buf, CALL_START) } else { -1 };
                    if tool >= 0 && (i < 0 || tool < i) {
                        let t = tool as usize;
                        if t > 0 {
                            let head = self.buf[..t].to_string();
                            self.emit(&mut out, Kind::Reasoning, &head);
                        }
                        if self.opener_ok() {
                            self.buf = self.buf[t..].to_string();
                            self.state = PState::Rcall;
                        } else {
                            self.emit(&mut out, Kind::Reasoning, CALL_START);
                            self.buf = self.buf[t + CALL_START.len()..].to_string();
                        }
                        continue;
                    }
                    if i < 0 {
                        let keep = hold(&self.buf, if !self.schemas.is_empty() { &[THINK_END, CALL_START] } else { &[THINK_END] });
                        if self.buf.len() > keep {
                            let n = self.buf.len() - keep;
                            let head = self.buf[..n].to_string();
                            self.emit(&mut out, Kind::Reasoning, &head);
                            self.buf = self.buf[n..].to_string();
                        }
                        return Ok(out);
                    }
                    let iu = i as usize;
                    match self.quoted_think_end(iu) {
                        None => {
                            if iu > 0 {
                                let head = self.buf[..iu].to_string();
                                self.emit(&mut out, Kind::Reasoning, &head);
                                self.buf = self.buf[iu..].to_string();
                            }
                            return Ok(out);
                        }
                        Some(true) => {
                            let head = self.buf[..iu + THINK_END.len()].to_string();
                            self.emit(&mut out, Kind::Reasoning, &head);
                            self.buf = self.buf[iu + THINK_END.len()..].to_string();
                            continue;
                        }
                        Some(false) => {}
                    }
                    if iu > 0 {
                        let head = self.buf[..iu].to_string();
                        self.emit(&mut out, Kind::Reasoning, &head);
                    }
                    self.buf = self.buf[iu + THINK_END.len()..].to_string();
                    self.state = PState::Content;
                    self.lead = true;
                    self.fence.clear();
                    self.line.clear();
                    self.ticks = 0;
                }
                PState::Content => {
                    if self.lead {
                        let stripped = self.buf.trim_start_matches('\n').to_string();
                        if stripped.is_empty() {
                            self.buf.clear();
                            return Ok(out);
                        }
                        self.buf = stripped;
                        self.lead = false;
                    }
                    let i = find0(&self.buf, CALL_START);
                    let (b, decided) = if self.recover && !self.schemas.is_empty() { self.bare_opener() } else { (-1, false) };
                    if b >= 0 && (i < 0 || b < i) {
                        let b = b as usize;
                        let mut j = b;
                        while j > 0 && self.buf.as_bytes()[j - 1] == b'\n' {
                            j -= 1;
                        }
                        if j > 0 && !strip(&self.buf[..j]).is_empty() {
                            let head = self.buf[..j].to_string();
                            self.emit(&mut out, Kind::Content, &head);
                        }
                        if !decided {
                            self.buf = self.buf[j..].to_string();
                            return Ok(out);
                        }
                        self.buf = self.buf[b..].to_string();
                        self.state = PState::Call;
                        self.bare = true;
                        continue;
                    }
                    if i < 0 {
                        let mut j = self.buf.len() - hold(&self.buf, &[CALL_START]);
                        if self.recover && !self.schemas.is_empty() {
                            let k = self.buf.rfind('\n').map(|x| x + 1).unwrap_or(0);
                            let tail = &self.buf[k..];
                            if !tail.is_empty() && FUNC_START.starts_with(tail) && self.ok_at(k) {
                                j = j.min(k);
                            }
                        }
                        while j > 0 && self.buf.as_bytes()[j - 1] == b'\n' {
                            j -= 1;
                        }
                        if j > 0 {
                            let head = self.buf[..j].to_string();
                            self.emit(&mut out, Kind::Content, &head);
                            self.buf = self.buf[j..].to_string();
                        }
                        return Ok(out);
                    }
                    let iu = i as usize;
                    let snap = self.snap();
                    let head = self.buf[..iu].to_string();
                    self.track(&head);
                    let in_code = self.in_code();
                    self.restore(snap);
                    if in_code {
                        let head = self.buf[..iu + CALL_START.len()].to_string();
                        self.emit(&mut out, Kind::Content, &head);
                        self.buf = self.buf[iu + CALL_START.len()..].to_string();
                        continue;
                    }
                    let mut after = lstrip(&self.buf[iu + CALL_START.len()..]).to_string();
                    let kind = if self.recover { Some(self.follower(&after)) } else { None };
                    if kind == Some("param") || kind == Some("json") {
                        if iu > 0 && !strip(&self.buf[..iu]).is_empty() {
                            let head = self.buf[..iu].trim_end_matches('\n').to_string();
                            self.emit(&mut out, Kind::Content, &head);
                        }
                        let mut rest = self.buf[iu + CALL_START.len()..].to_string();
                        if kind == Some("param") {
                            let at = rest.len() - after.len();
                            rest = format!("{}{}{}", &rest[..at], FUNC_START, &after[PARAM_START.len()..]);
                        }
                        self.buf = rest;
                        self.state = if kind == Some("param") { PState::Call } else { PState::Json };
                        continue;
                    }
                    if kind == Some("wait") {
                        after.clear();
                    }
                    if !after.is_empty() && !after.starts_with(FUNC_START) && !FUNC_START.starts_with(after.as_str()) {
                        let head = self.buf[..iu + CALL_START.len()].to_string();
                        self.emit(&mut out, Kind::Content, &head);
                        self.buf = self.buf[iu + CALL_START.len()..].to_string();
                        continue;
                    }
                    if !after.starts_with(FUNC_START) {
                        let mut j = iu;
                        while j > 0 && self.buf.as_bytes()[j - 1] == b'\n' {
                            j -= 1;
                        }
                        if j > 0 {
                            let head = self.buf[..j].to_string();
                            self.emit(&mut out, Kind::Content, &head);
                            self.buf = self.buf[j..].to_string();
                        }
                        return Ok(out);
                    }
                    if iu > 0 && !strip(&self.buf[..iu]).is_empty() {
                        let head = self.buf[..iu].trim_end_matches('\n').to_string();
                        self.emit(&mut out, Kind::Content, &head);
                    }
                    self.buf = self.buf[iu + CALL_START.len()..].to_string();
                    self.state = PState::Call;
                }
                PState::Json => {
                    let i = find0(&self.buf, CALL_END);
                    if i < 0 {
                        return Ok(out);
                    }
                    let iu = i as usize;
                    let call = json_tool_call(strip(&self.buf[..iu]), &self.schemas);
                    let ok = call.is_some();
                    match call {
                        None => {
                            let text = format!("{CALL_START}{}", &self.buf[..iu + CALL_END.len()]);
                            self.emit(&mut out, Kind::Content, &text);
                        }
                        Some(c) => out.push(Event::call(Kind::ToolCall, c)),
                    }
                    self.buf = self.buf[iu + CALL_END.len()..].to_string();
                    self.state = PState::Content;
                    self.lead = ok;
                }
                PState::Call => {
                    let mut drop: Option<usize> = None;
                    let mut i: isize;
                    if self.bare {
                        i = function_end(&self.buf);
                        if i >= 0 {
                            let tail = &self.buf[i as usize..];
                            let after = lstrip(tail);
                            if after.starts_with(CALL_END) {
                                drop = Some(tail.len() - after.len() + CALL_END.len());
                            } else if !after.is_empty() && !CALL_END.starts_with(after) {
                                drop = Some(0);
                            }
                        }
                    } else {
                        i = call_end(&self.buf);
                        if i >= 0 {
                            drop = Some(CALL_END.len());
                        }
                        if self.recover {
                            let f = function_end(&self.buf);
                            let mut nxt = if f >= 0 { lstrip(&self.buf[f as usize..]).to_string() } else { String::new() };
                            if nxt.starts_with(PARAM_START) && self.follower(&nxt) == "param" {
                                let at = self.buf.len() - nxt.len();
                                self.buf = format!("{}{}{}", &self.buf[..at], FUNC_START, &nxt[PARAM_START.len()..]);
                                nxt = self.buf[at..].to_string();
                            }
                            if nxt.starts_with(FUNC_START) {
                                i = f;
                                drop = Some(0);
                                self.batch = true;
                            }
                        }
                    }
                    if self.stream_tools {
                        if i >= 0 {
                            let whole = self.buf.clone();
                            self.buf.truncate(i as usize);
                            let mut s = self.scan();
                            s.extend(self.close_scan());
                            out.extend(s);
                            self.buf = whole;
                        } else {
                            let s = self.scan();
                            out.extend(s);
                        }
                    }
                    let (true, Some(d)) = (i >= 0, drop) else { return Ok(out) };
                    let iu = i as usize;
                    let body = self.buf[..iu].to_string();
                    self.buf = self.buf[iu + d..].to_string();
                    let batch = std::mem::replace(&mut self.batch, false);
                    let evs = self.finish_call(&body)?;
                    let last_is_call = evs.last().map(|e| e.kind == Kind::ToolCall).unwrap_or(false);
                    out.extend(evs);
                    if batch && last_is_call {
                        self.state = PState::Call;
                        self.lead = false;
                    }
                }
            }
        }
    }

    pub fn finish_inner(&mut self, reason: Option<&str>) -> Result<Vec<Event>, String> {
        let mut out = vec![];
        if !self.pending.is_empty() {
            let deliver = matches!(reason, None | Some("stop")) && self.state == PState::Reasoning && self.buf.is_empty();
            out.extend(self.release(deliver));
        }
        if self.state == PState::Call && self.bare && function_end(&self.buf) >= 0 {
            let i = function_end(&self.buf) as usize;
            let body = self.buf[..i].to_string();
            let rest = strip(&self.buf[i..]).to_string();
            self.buf.clear();
            out.extend(self.finish_call(&body)?);
            if !rest.is_empty() && !CALL_END.starts_with(rest.as_str()) {
                self.emit(&mut out, Kind::Content, &rest);
            }
            return Ok(out);
        }
        if self.state == PState::Call && self.stream_tools && self.scall.is_some() {
            out.extend(self.scan());
            if self.ss == Scan::Done {
                out.push(Event::call(Kind::ToolCall, self.scall.clone().expect("scall")));
            }
            self.buf.clear();
            self.reset_scan();
            return Ok(out);
        }
        if self.state == PState::Reasoning && self.buf == THINK_END {
            self.buf.clear();
            self.state = PState::Content;
        }
        if !self.buf.is_empty() {
            let kind = match self.state {
                PState::Reasoning | PState::Rcall => Kind::Reasoning,
                _ => Kind::Content,
            };
            let text = if matches!(self.state, PState::Call | PState::Json) && !self.bare {
                format!("{CALL_START}{}", self.buf)
            } else {
                self.buf.clone()
            };
            out.push(Event::text(kind, text));
            self.buf.clear();
        }
        Ok(out)
    }

    fn state_name(&self) -> &'static str {
        match self.state {
            PState::Reasoning => "reasoning",
            PState::Content => "content",
            PState::Call => "call",
            PState::Json => "json",
            PState::Rcall => "rcall",
        }
    }
}

impl Parse for OutputParser {
    fn feed(&mut self, delta: &str) -> Result<Vec<Event>, String> {
        self.feed_inner(delta)
    }
    fn finish(&mut self, reason: Option<&str>) -> Result<Vec<Event>, String> {
        self.finish_inner(reason)
    }
    fn state(&self) -> &'static str {
        self.state_name()
    }
    fn buf(&self) -> String {
        self.buf.clone()
    }
    fn pending(&self) -> usize {
        self.pending.len()
    }
    fn rescued(&self) -> usize {
        self.rescued
    }
    fn refused(&self) -> usize {
        self.refused
    }
}

// ------------------------------------------------------------------ GPT-OSS: OpenAI's harmony format
pub const H_START: &str = "<|start|>";
pub const H_END: &str = "<|end|>";
pub const H_MESSAGE: &str = "<|message|>";
pub const H_CHANNEL: &str = "<|channel|>";
pub const H_RETURN: &str = "<|return|>";
pub const H_CALL: &str = "<|call|>";
const H_MARKERS: [&str; 7] = [H_START, H_END, H_MESSAGE, H_CHANNEL, H_RETURN, H_CALL, "<|constrain|>"];

#[derive(Clone, Copy, PartialEq, Eq)]
enum HState {
    Header,
    Reasoning,
    Content,
    Call,
}

/// Harmony: messages of a header, `<|message|>`, a body and `<|end|>` / `<|return|>` / `<|call|>`; `analysis` is the
/// reasoning, `final` the answer, `to=functions.NAME` a call with JSON arguments, anything else answer text.
pub struct HarmonyParser {
    state: HState,
    pub buf: String,
    call_name: String,
    call_body: String,
}

impl HarmonyParser {
    pub fn new() -> Self {
        HarmonyParser { state: HState::Header, buf: String::new(), call_name: String::new(), call_body: String::new() }
    }

    fn held(text: &str) -> usize {
        let max = H_MARKERS.iter().map(|m| m.len()).max().unwrap_or(0) - 1;
        for k in (1..=text.len().min(max)).rev() {
            let at = text.len() - k;
            if text.is_char_boundary(at) && H_MARKERS.iter().any(|m| m.starts_with(&text[at..])) {
                return k;
            }
        }
        0
    }

    fn body(&mut self, text: &str) -> Vec<Event> {
        if text.is_empty() {
            return vec![];
        }
        match self.state {
            HState::Call => {
                self.call_body.push_str(text);
                vec![]
            }
            HState::Reasoning => vec![Event::text(Kind::Reasoning, text)],
            _ => vec![Event::text(Kind::Content, text)],
        }
    }

    fn end_message(&mut self) -> Vec<Event> {
        let mut out = vec![];
        if self.state == HState::Call {
            let args = tool_arguments(&Value::String(strip(&self.call_body).to_string()));
            out.push(Event::call(Kind::ToolCall, ToolCall::new(self.call_name.clone(), args)));
        }
        self.state = HState::Header;
        self.call_name.clear();
        self.call_body.clear();
        out
    }

    /// `to=([^\s<]+)` in the header (Python's \s: with U+001C..U+001F)
    fn recipient(head: &str) -> Option<&str> {
        let i = head.find("to=")?;
        let rest = &head[i + 3..];
        let end = rest.find(|c: char| py::is_space(c) || c == '<').unwrap_or(rest.len());
        if end == 0 {
            // `to=` followed by nothing usable: search on (re.search finds the next match)
            return Self::recipient(rest);
        }
        Some(&rest[..end])
    }
}

impl Default for HarmonyParser {
    fn default() -> Self {
        Self::new()
    }
}

impl Parse for HarmonyParser {
    fn feed(&mut self, delta: &str) -> Result<Vec<Event>, String> {
        self.buf.push_str(delta);
        let mut out = vec![];
        loop {
            if self.state == HState::Header {
                let Some(i) = self.buf.find(H_MESSAGE) else { return Ok(out) };
                let head = self.buf[..i].to_string();
                self.buf = self.buf[i + H_MESSAGE.len()..].to_string();
                let channel = match head.split_once(H_CHANNEL) {
                    Some((_, after)) => py::split_ws(after).first().map(|w| w.split("<|").next().unwrap_or("")).unwrap_or("").to_string(),
                    None => String::new(),
                };
                match Self::recipient(&head) {
                    Some(to) if to.starts_with("functions.") => {
                        self.state = HState::Call;
                        self.call_name = to["functions.".len()..].to_string();
                    }
                    _ => self.state = if channel == "analysis" { HState::Reasoning } else { HState::Content },
                }
                continue;
            }
            let ends = [H_END, H_RETURN, H_CALL].iter().filter_map(|m| self.buf.find(m).map(|i| (i, *m))).min();
            if let Some((i, m)) = ends {
                let head = self.buf[..i].to_string();
                out.extend(self.body(&head));
                self.buf = self.buf[i + m.len()..].to_string();
                out.extend(self.end_message());
                continue;
            }
            let keep = Self::held(&self.buf);
            let n = self.buf.len() - keep;
            let head = self.buf[..n].to_string();
            out.extend(self.body(&head));
            self.buf = self.buf[n..].to_string();
            return Ok(out);
        }
    }

    fn finish(&mut self, reason: Option<&str>) -> Result<Vec<Event>, String> {
        let rest = std::mem::take(&mut self.buf);
        let mut out = vec![];
        match self.state {
            HState::Call => {
                if matches!(reason, None | Some("stop")) {
                    self.call_body.push_str(&rest);
                    return Ok(self.end_message());
                }
                out.push(Event::text(Kind::Content, format!("{}{}", self.call_body, rest)));
            }
            HState::Header => {}
            s => {
                if !rest.is_empty() {
                    out.push(Event::text(if s == HState::Reasoning { Kind::Reasoning } else { Kind::Content }, rest));
                }
            }
        }
        self.state = HState::Header;
        Ok(out)
    }

    fn state(&self) -> &'static str {
        match self.state {
            HState::Header => "header",
            HState::Reasoning => "reasoning",
            HState::Content => "content",
            HState::Call => "call",
        }
    }
    fn buf(&self) -> String {
        self.buf.clone()
    }
    fn pending(&self) -> usize {
        0
    }
    fn rescued(&self) -> usize {
        0
    }
    fn refused(&self) -> usize {
        0
    }
}

// ------------------------------------------------------------------ GLM-5.3-Flash: its tool-call form
/// `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>` -> OutputParser's form; anything else
/// is returned as it is.
pub fn glm_call_to_qwen(call: &str) -> String {
    let body = if call.ends_with(CALL_END) && call.len() >= CALL_START.len() + CALL_END.len() {
        &call[CALL_START.len()..call.len() - CALL_END.len()]
    } else {
        call.get(CALL_START.len()..).unwrap_or("")
    };
    let name = strip(body.split("<arg_key>").next().unwrap_or(""));
    if name.is_empty() || name.contains('<') || name.contains('\n') {
        return call.to_string();
    }
    static ARG: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re = ARG.get_or_init(|| regex::Regex::new(r"(?s)<arg_key>(.*?)</arg_key>[\s\x1c-\x1f]*<arg_value>(.*?)</arg_value>").unwrap());
    let params: String = re.captures_iter(body).map(|c| format!("<parameter={}>\n{}\n</parameter>\n", strip(&c[1]), &c[2])).collect();
    format!("{CALL_START}\n<function={name}>\n{params}</function>\n{CALL_END}")
}

/// GLM: Qwen's reasoning and text, and its own call form, held whole and handed to OutputParser in OutputParser's form.
pub struct GlmParser {
    inner: OutputParser,
    hold: String,
}

impl GlmParser {
    pub fn new(thinking: bool, tools: Option<&[Value]>, recover: bool) -> Self {
        GlmParser { inner: OutputParser::new(thinking, tools, false, recover), hold: String::new() }
    }
}

impl Parse for GlmParser {
    fn feed(&mut self, delta: &str) -> Result<Vec<Event>, String> {
        self.hold.push_str(delta);
        let mut out = vec![];
        while !self.hold.is_empty() {
            let Some(i) = self.hold.find(CALL_START) else {
                let max = self.hold.len().min(CALL_START.len() - 1);
                let keep = (1..=max).rev().find(|&k| {
                    let at = self.hold.len() - k;
                    self.hold.is_char_boundary(at) && CALL_START.starts_with(&self.hold[at..])
                }).unwrap_or(0);
                let n = self.hold.len() - keep;
                let head = self.hold[..n].to_string();
                out.extend(self.inner.feed_inner(&head)?);
                self.hold = self.hold[n..].to_string();
                return Ok(out);
            };
            if i > 0 {
                let head = self.hold[..i].to_string();
                out.extend(self.inner.feed_inner(&head)?);
                self.hold = self.hold[i..].to_string();
            }
            let Some(j) = self.hold.find(CALL_END) else { return Ok(out) };
            let call = self.hold[..j + CALL_END.len()].to_string();
            self.hold = self.hold[j + CALL_END.len()..].to_string();
            out.extend(self.inner.feed_inner(&glm_call_to_qwen(&call))?);
        }
        Ok(out)
    }

    fn finish(&mut self, reason: Option<&str>) -> Result<Vec<Event>, String> {
        let mut out = vec![];
        if !self.hold.is_empty() {
            let h = std::mem::take(&mut self.hold);
            let text = if h.starts_with(CALL_START) && matches!(reason, None | Some("stop")) { glm_call_to_qwen(&h) } else { h };
            out.extend(self.inner.feed_inner(&text)?);
        }
        out.extend(self.inner.finish_inner(reason)?);
        Ok(out)
    }

    fn state(&self) -> &'static str {
        self.inner.state_name()
    }
    fn buf(&self) -> String {
        format!("{}{}", self.hold, self.inner.buf)
    }
    fn pending(&self) -> usize {
        self.inner.pending.len()
    }
    fn rescued(&self) -> usize {
        self.inner.rescued
    }
    fn refused(&self) -> usize {
        self.inner.refused
    }
}
