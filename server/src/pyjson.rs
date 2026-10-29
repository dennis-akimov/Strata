//! Python's `json.dumps`, byte for byte: the prompt (a template's `tojson` of tool arguments) and every body the
//! Python server wrote come out of it, and a different separator or float spelling is a different prompt.

use serde_json::Value;
use std::fmt::Write;

#[derive(Clone, Copy)]
pub struct Opts<'a> {
    pub ensure_ascii: bool,
    pub indent: Option<usize>,
    /// (item, key) separators; None: Python's default for the indent (", " / ": ", or "," / ": " with an indent)
    pub separators: Option<(&'a str, &'a str)>,
    pub sort_keys: bool,
}

impl Default for Opts<'_> {
    fn default() -> Self {
        Opts { ensure_ascii: true, indent: None, separators: None, sort_keys: false }
    }
}

/// `json.dumps(v)`
pub fn dumps(v: &Value) -> String {
    dumps_with(v, Opts::default())
}

/// `json.dumps(v, ensure_ascii=False)`
pub fn dumps_u(v: &Value) -> String {
    dumps_with(v, Opts { ensure_ascii: false, ..Opts::default() })
}

/// `json.dumps(v, ensure_ascii=False, separators=(",", ":"))`
pub fn dumps_compact(v: &Value) -> String {
    dumps_with(v, Opts { ensure_ascii: false, separators: Some((",", ":")), ..Opts::default() })
}

pub fn dumps_with(v: &Value, o: Opts) -> String {
    let mut out = String::new();
    let (item, key) = o.separators.unwrap_or(if o.indent.is_some() { (",", ": ") } else { (", ", ": ") });
    write_value(&mut out, v, &o, item, key, 0);
    out
}

fn newline(out: &mut String, o: &Opts, level: usize) {
    if let Some(n) = o.indent {
        out.push('\n');
        for _ in 0..n * level {
            out.push(' ');
        }
    }
}

fn write_value(out: &mut String, v: &Value, o: &Opts, item: &str, key: &str, level: usize) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                let _ = write!(out, "{i}");
            } else if let Some(u) = n.as_u64() {
                let _ = write!(out, "{u}");
            } else {
                out.push_str(&float_repr(n.as_f64().unwrap_or(f64::NAN)));
            }
        }
        Value::String(s) => write_str(out, s, o.ensure_ascii),
        Value::Array(a) => {
            if a.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(item);
                }
                newline(out, o, level + 1);
                write_value(out, x, o, item, key, level + 1);
            }
            newline(out, o, level);
            out.push(']');
        }
        Value::Object(m) => {
            if m.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            let mut entries: Vec<(&String, &Value)> = m.iter().collect();
            if o.sort_keys {
                entries.sort_by(|a, b| a.0.cmp(b.0));
            }
            for (i, (k, x)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(item);
                }
                newline(out, o, level + 1);
                write_str(out, k, o.ensure_ascii);
                out.push_str(key);
                write_value(out, x, o, item, key, level + 1);
            }
            newline(out, o, level);
            out.push('}');
        }
    }
}

/// A JSON string as Python writes it: `\"` `\\` `\n` `\r` `\t` `\b` `\f`, other controls `\u00XX`; with ensure_ascii
/// everything past 0x7F as `\uXXXX` (a surrogate pair past the BMP), lowercase hex.
pub fn write_str(out: &mut String, s: &str, ensure_ascii: bool) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c if ensure_ascii && (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    let _ = write!(out, "\\u{:04x}", u);
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `repr(float)`: the shortest digits that read back as the same float, positional for 1e-4 <= |x| < 1e16
/// ("1.0", "0.0001", "28.9"), else scientific with a signed two-digit-minimum exponent ("1e-05", "1.5e+16").
pub fn float_repr(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    // Rust's `{:e}` is the shortest round-trip spelling too: "2.89e1", "1e-5"
    let sci = format!("{:e}", f.abs());
    let (mant, exp) = sci.split_once('e').expect("exponent");
    let exp: i32 = exp.parse().expect("exponent");
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let sign = if f < 0.0 { "-" } else { "" };
    if (-4..16).contains(&exp) {
        let n = digits.len() as i32;
        let s = if exp < 0 {
            format!("0.{}{}", "0".repeat((-exp - 1) as usize), digits)
        } else if exp + 1 >= n {
            format!("{}{}.0", digits, "0".repeat((exp + 1 - n) as usize))
        } else {
            format!("{}.{}", &digits[..(exp + 1) as usize], &digits[(exp + 1) as usize..])
        };
        format!("{sign}{s}")
    } else {
        let m = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits };
        format!("{sign}{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn floats_read_as_python_writes_them() {
        for (f, s) in [(1.0, "1.0"), (28.9, "28.9"), (0.0001, "0.0001"), (0.00001, "1e-05"), (1e16, "1e+16"),
                       (1.5e16, "1.5e+16"), (123456789012345.6, "123456789012345.6"), (-0.5, "-0.5"), (1e22, "1e+22"),
                       (0.1 + 0.2, "0.30000000000000004"), (100.0, "100.0"), (2.5e-7, "2.5e-07")] {
            assert_eq!(float_repr(f), s, "{f}");
        }
    }

    #[test]
    fn strings_objects_and_separators() {
        let v = json!({"a": "é\n\u{1}😀", "b": [1, 2.0, null, true], "c": {}});
        assert_eq!(dumps(&v), "{\"a\": \"\\u00e9\\n\\u0001\\ud83d\\ude00\", \"b\": [1, 2.0, null, true], \"c\": {}}");
        assert_eq!(dumps_u(&v), "{\"a\": \"é\\n\\u0001😀\", \"b\": [1, 2.0, null, true], \"c\": {}}");
        assert_eq!(dumps_compact(&json!({"x": [1, {"y": 2}]})), r#"{"x":[1,{"y":2}]}"#);
        let ind = dumps_with(&json!({"k": [1], "e": []}), Opts { indent: Some(1), ..Opts::default() });
        assert_eq!(ind, "{\n \"k\": [\n  1\n ],\n \"e\": []\n}");
    }
}
