//! Python's string and value semantics where the server's behaviour depends on them: `str.strip()` and `isspace()`
//! (Unicode White_Space plus U+001C..U+001F, which Rust's `trim` leaves), truthiness of JSON values, `dict.get`.

use serde_json::Value;

pub fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

pub fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

pub fn lstrip(s: &str) -> &str {
    s.trim_start_matches(is_space)
}

pub fn rstrip(s: &str) -> &str {
    s.trim_end_matches(is_space)
}

/// `str.split()` with no separator.
pub fn split_ws(s: &str) -> Vec<&str> {
    s.split(is_space).filter(|x| !x.is_empty()).collect()
}

/// `s.find(sub, start)` in bytes, -1 when absent (start past the end is -1 too, as Python's).
pub fn find(s: &str, sub: &str, start: usize) -> isize {
    if start > s.len() {
        return -1;
    }
    s[start..].find(sub).map(|i| (i + start) as isize).unwrap_or(-1)
}

pub fn find0(s: &str, sub: &str) -> isize {
    find(s, sub, 0)
}

/// Python truthiness of a JSON value.
pub fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// `v.get(key)` on a dict (Null for a missing key or a value that is no dict).
pub fn get<'a>(v: &'a Value, key: &str) -> &'a Value {
    static NULL: Value = Value::Null;
    v.as_object().and_then(|o| o.get(key)).unwrap_or(&NULL)
}

pub fn get_str<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    get(v, key).as_str()
}

/// The previous character before byte offset `i` (`s[i-1]` in Python), as a string ("" at the start).
pub fn char_before(s: &str, i: usize) -> &str {
    match s[..i].char_indices().next_back() {
        Some((j, _)) => &s[j..i],
        None => "",
    }
}

/// `s[:n]` in characters.
pub fn chars_prefix(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// The byte offsets of every character boundary of `s` past 0 (the lengths of its non-empty prefixes).
pub fn prefix_ends(s: &str) -> impl Iterator<Item = usize> + '_ {
    s.char_indices().skip(1).map(|(i, _)| i).chain(std::iter::once(s.len()))
}

/// The type name Python says for a JSON value (`type(x).__name__`).
pub fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) if n.is_f64() => "float",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// Python's `str(v)` / f-string of a JSON value (repr-like for containers: `{'a': 1}`).
pub fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        _ => py_repr(v),
    }
}

pub fn py_repr(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True".into() } else { "False".into() },
        Value::Number(n) => match (n.as_i64(), n.as_u64()) {
            (Some(i), _) => i.to_string(),
            (_, Some(u)) => u.to_string(),
            _ => {
                let f = n.as_f64().unwrap_or(f64::NAN);
                if f.is_nan() { "nan".into() } else if f.is_infinite() { if f > 0.0 { "inf".into() } else { "-inf".into() } } else { crate::pyjson::float_repr(f) }
            }
        },
        Value::String(s) => {
            let q = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
            let mut out = String::from(q);
            for c in s.chars() {
                match c {
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c if c == q => { out.push('\\'); out.push(c) }
                    c if (c as u32) < 0x20 || c as u32 == 0x7f => out.push_str(&format!("\\x{:02x}", c as u32)),
                    c => out.push(c),
                }
            }
            out.push(q);
            out
        }
        Value::Array(a) => format!("[{}]", a.iter().map(py_repr).collect::<Vec<_>>().join(", ")),
        Value::Object(o) => format!("{{{}}}", o.iter().map(|(k, x)| format!("{}: {}", py_repr(&Value::String(k.clone())), py_repr(x))).collect::<Vec<_>>().join(", ")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn python_whitespace_and_repr() {
        assert_eq!(strip("\u{1c} a \u{a0}"), "a");
        assert_eq!(split_ws(" a\u{1f}b  c "), vec!["a", "b", "c"]);
        assert_eq!(py_repr(&json!({"a": [1, 2.5, null, true, "it's"]})), "{'a': [1, 2.5, None, True, \"it's\"]}");
        assert_eq!(char_before("aé", 3), "é");
        assert_eq!(find("abcabc", "c", 3), 5);
    }
}
