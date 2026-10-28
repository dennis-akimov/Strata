//! JSON response formats at the HTTP boundary (serve/structured.py): prompt once, validate before delivery.  The
//! engine has no grammar decoder; a failed generation is an error, never retried or passed off as valid output.

use crate::errors::Fail;
use crate::py::get;
use crate::pyjson;
use serde_json::{json, Value};

pub enum Validator {
    /// json_object: one JSON object
    ObjectOnly,
    Schema(Box<jsonschema::Validator>),
}

/// True when every value `node` accepts is a JSON object.
fn only_objects(node: &Value, root: &Value, refs: &mut Vec<String>) -> bool {
    let Some(o) = node.as_object() else { return false };
    if o.get("type").and_then(Value::as_str) == Some("object") {
        return true;
    }
    if let Some(r) = o.get("$ref").and_then(Value::as_str) {
        if r.starts_with('#') && !refs.iter().any(|x| x == r) {
            let mut target = root;
            for part in r[1..].split('/').skip(1) {
                let part = part.replace("~1", "/").replace("~0", "~");
                match target.as_object().and_then(|t| t.get(&part)) {
                    Some(t) => target = t,
                    None => return false,
                }
            }
            refs.push(r.to_string());
            let ok = only_objects(target, root, refs);
            refs.pop();
            return ok;
        }
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(b) = o.get(key).and_then(Value::as_array) {
            if !b.is_empty() && b.iter().all(|x| only_objects(x, root, refs)) {
                return true;
            }
        }
    }
    o.get("allOf").and_then(Value::as_array).map(|b| b.iter().any(|x| only_objects(x, root, refs))).unwrap_or(false)
}

fn check_refs(node: &Value) -> Result<(), Fail> {
    match node {
        Value::Object(o) => {
            for (k, v) in o {
                if (k == "$ref" || k == "$dynamicRef") && v.as_str().map(|s| !s.starts_with('#')).unwrap_or(false) {
                    return Err(Fail::Value("response_format supports only local schema references (#...)".into()));
                }
                check_refs(v)?;
            }
        }
        Value::Array(a) => {
            for v in a {
                check_refs(v)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// (messages with the format's instruction, the validator) - or the messages as they are for none / text.
pub fn prepare_format(rf: &Value, messages: Vec<Value>, with_tools: bool) -> Result<(Vec<Value>, Option<Validator>), Fail> {
    if rf.is_null() {
        return Ok((messages, None));
    }
    if !rf.is_object() {
        return Err(Fail::Value("response_format must be an object".into()));
    }
    let (schema, validator) = match get(rf, "type").as_str() {
        Some("text") => return Ok((messages, None)),
        Some("json_object") => (json!({"type": "object"}), None),
        Some("json_schema") => {
            let spec = get(rf, "json_schema");
            if !spec.is_object() || !get(spec, "schema").is_object() {
                return Err(Fail::Value("response_format.json_schema needs a schema object".into()));
            }
            if !get(spec, "name").as_str().map(|n| !n.is_empty()).unwrap_or(false) {
                return Err(Fail::Value("response_format.json_schema needs a name".into()));
            }
            if spec.get("strict").map(|s| !s.is_boolean()).unwrap_or(false) {
                return Err(Fail::Value("response_format.json_schema.strict must be boolean".into()));
            }
            let schema = spec["schema"].clone();
            if !only_objects(&schema, &schema, &mut vec![]) {
                return Err(Fail::Value("response_format schema must accept only JSON objects at its root (type object, or anyOf/oneOf of object schemas)".into()));
            }
            (schema, Some(()))
        }
        _ => return Err(Fail::Value("response_format.type must be text, json_object or json_schema".into())),
    };
    check_refs(&schema)?;
    let validator = match validator {
        None => Validator::ObjectOnly,
        Some(()) => Validator::Schema(Box::new(jsonschema::validator_for(&schema).map_err(|e| Fail::Value(format!("invalid response_format schema: {e}")))?)),
    };
    let mut directive = format!("OUTPUT FORMAT REQUIREMENT: Return exactly one JSON object matching the JSON Schema below. No Markdown, headings, code fences, commentary, or text outside the JSON. Use every required field, correct types, and only allowed fields. Put all requested writing inside the appropriate JSON string fields.\nJSON Schema:\n{}", pyjson::dumps_compact(&schema));
    if with_tools {
        directive += "\nThis applies only to your final answer. To use a tool, call it as usual; the JSON object is what you write once you are done with the tools.";
    }
    let mut messages = messages;
    let first_system = messages.first().map(|m| get(m, "role").as_str() == Some("system")).unwrap_or(false);
    if first_system {
        let m = messages[0].as_object_mut().unwrap();
        let content = m.get("content").cloned().unwrap_or(Value::Null);
        let c = match content {
            Value::Array(mut a) => {
                a.push(json!({"type": "text", "text": directive}));
                Value::Array(a)
            }
            other => {
                let s = if crate::py::truthy(&other) { crate::py::py_str(&other) } else { String::new() };
                json!(format!("{s}\n\n{directive}"))
            }
        };
        m.insert("content".into(), c);
    } else {
        messages.insert(0, json!({"role": "system", "content": directive}));
    }
    Ok((messages, Some(validator)))
}

/// The JSON object out of the text: inside a code fence, from the first `{` to its matching `}`.
pub fn extract_json(text: &str) -> String {
    let mut s = crate::py::strip(text).to_string();
    if s.is_empty() {
        return s;
    }
    if s.starts_with("```") {
        if let Some(first) = s.find('\n') {
            if let Some(end) = s[first + 1..].find("```") {
                s = crate::py::strip(&s[first + 1..first + 1 + end]).to_string();
            }
        }
    }
    let Some(start) = s.find('{') else { return s };
    let (mut depth, mut in_str, mut esc) = (0i64, false, false);
    for (i, ch) in s[start..].char_indices() {
        if in_str {
            if esc {
                esc = false;
            } else if ch == '\\' {
                esc = true;
            } else if ch == '"' {
                in_str = false;
            }
        } else if ch == '"' {
            in_str = true;
        } else if ch == '{' {
            depth += 1;
        } else if ch == '}' {
            depth -= 1;
            if depth == 0 {
                return s[start..start + i + 1].to_string();
            }
        }
    }
    s[start..].to_string()
}

/// json.loads with duplicate keys refused (object_pairs_hook).
fn strict_parse(text: &str) -> Result<Value, String> {
    struct V;
    impl<'de> serde::de::Visitor<'de> for V {
        type Value = Value;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("JSON")
        }
        fn visit_bool<E>(self, v: bool) -> Result<Value, E> {
            Ok(Value::Bool(v))
        }
        fn visit_i64<E>(self, v: i64) -> Result<Value, E> {
            Ok(json!(v))
        }
        fn visit_u64<E>(self, v: u64) -> Result<Value, E> {
            Ok(json!(v))
        }
        fn visit_f64<E>(self, v: f64) -> Result<Value, E> {
            Ok(json!(v))
        }
        fn visit_str<E>(self, v: &str) -> Result<Value, E> {
            Ok(json!(v))
        }
        fn visit_string<E>(self, v: String) -> Result<Value, E> {
            Ok(Value::String(v))
        }
        fn visit_unit<E>(self) -> Result<Value, E> {
            Ok(Value::Null)
        }
        fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut a: A) -> Result<Value, A::Error> {
            let mut out = vec![];
            while let Some(x) = a.next_element_seed(Seed)? {
                out.push(x);
            }
            Ok(Value::Array(out))
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(self, mut a: A) -> Result<Value, A::Error> {
            let mut out = serde_json::Map::new();
            while let Some(k) = a.next_key::<String>()? {
                if out.contains_key(&k) {
                    return Err(serde::de::Error::custom(format!("duplicate JSON key: {k}")));
                }
                let v = a.next_value_seed(Seed)?;
                out.insert(k, v);
            }
            Ok(Value::Object(out))
        }
    }
    struct Seed;
    impl<'de> serde::de::DeserializeSeed<'de> for Seed {
        type Value = Value;
        fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> Result<Value, D::Error> {
            d.deserialize_any(V)
        }
    }
    let mut de = serde_json::Deserializer::from_str(text);
    let v = serde::de::DeserializeSeed::deserialize(Seed, &mut de).map_err(|e| e.to_string())?;
    de.end().map_err(|e| e.to_string())?;
    Ok(v)
}

/// The answer as one canonical JSON object, or StructuredOutputError.
pub fn validated_json(text: &str, validator: &Validator, finish: &str) -> Result<String, Fail> {
    if finish != "stop" {
        return Err(Fail::Structured(format!("structured output was incomplete (finish_reason={finish}); increase the output budget")));
    }
    let value = strict_parse(&extract_json(text)).map_err(|e| Fail::Structured(format!("model did not return valid JSON: {e}")))?;
    let canonical = pyjson::dumps_compact(&value);
    let err = match validator {
        Validator::ObjectOnly => (!value.is_object()).then(|| ("/".to_string(), "the answer is not a JSON object".to_string())),
        Validator::Schema(v) => v.iter_errors(&value).next().map(|e| {
            let p = e.instance_path().to_string();
            (if p.is_empty() { "/".into() } else { p }, e.to_string())
        }),
    };
    if let Some((path, msg)) = err {
        return Err(Fail::Structured(format!("model output failed the JSON Schema at {path}: {msg}")));
    }
    Ok(canonical)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_validate_and_canonicalize() {
        let (m, v) = prepare_format(&json!({"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}}}), vec![json!({"role": "user", "content": "q"})], false).unwrap();
        assert_eq!(m.len(), 2);
        let v = v.unwrap();
        assert_eq!(validated_json("```json\n{\"a\": 1, \"b\": \"é\"}\n```", &v, "stop").unwrap(), "{\"a\":1,\"b\":\"é\"}");
        assert!(validated_json("{\"b\": 1}", &v, "stop").is_err());
        assert!(validated_json("{\"a\": 1, \"a\": 2}", &v, "stop").is_err());
        assert!(validated_json("{\"a\": 1}", &v, "length").is_err());
        assert!(prepare_format(&json!({"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "array"}}}), vec![], false).is_err());
    }
}
