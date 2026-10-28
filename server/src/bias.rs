//! Request-scoped token biases (serve/logit_bias.py): normalized before HTTP streaming starts.

use crate::engine::py_g;
use crate::errors::Fail;
use serde_json::{Map, Value};
use std::collections::BTreeMap;

fn bad<T>(m: &str) -> Result<T, Fail> {
    Err(Fail::Value(format!("logit_bias: {m}")))
}

/// {token id: bias} from an object or a list of [token_id, bias] pairs; ValueError (a 400) for anything else.
pub fn normalize(value: Option<&Value>, vocab_size: Option<usize>) -> Result<BTreeMap<u64, f64>, Fail> {
    let (entries, pairs): (Vec<(Value, Value)>, bool) = match value {
        None | Some(Value::Null) => return Ok(BTreeMap::new()),
        Some(Value::Object(o)) => (o.iter().map(|(k, v)| (Value::String(k.clone()), v.clone())).collect(), false),
        Some(Value::Array(a)) => {
            let mut out = vec![];
            for e in a {
                match e.as_array() {
                    Some(p) if p.len() == 2 => out.push((p[0].clone(), p[1].clone())),
                    _ => return bad("each entry must be [token_id, bias]"),
                }
            }
            (out, true)
        }
        _ => return bad("expected an object or a list of [token_id, bias] pairs"),
    };
    let mut out = BTreeMap::new();
    let mut seen = std::collections::HashSet::new();
    for (token, bias) in entries {
        let t = match &token {
            Value::String(s) if !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()) => s.parse::<u128>().unwrap_or(u128::MAX),
            Value::Number(n) if n.is_u64() => n.as_u64().unwrap() as u128,
            _ => return bad("token IDs must be non-negative integers"),
        };
        if t > 2147483647 || vocab_size.map(|v| t >= v as u128).unwrap_or(false) {
            return bad("token ID outside the model vocabulary");
        }
        let t = t as u64;
        if !seen.insert(t) {
            return bad("duplicate token ID");
        }
        let b = match bias {
            Value::Bool(false) if pairs => -100.0,
            Value::Number(n) => n.as_f64().unwrap_or(f64::NAN),
            _ => return bad("biases must be finite numbers between -100 and 100; false is a pair-list ban"),
        };
        if !(-100.0..=100.0).contains(&b) || !b.is_finite() {
            return bad("biases must be finite numbers between -100 and 100; false is a pair-list ban");
        }
        out.insert(t, b);
    }
    if let Some(v) = vocab_size {
        if out.values().filter(|b| **b == -100.0).count() == v {
            return bad("cannot ban the entire vocabulary");
        }
    }
    Ok(out)
}

/// ` logit_bias=12:-100,13:5` for the GEN line ("" for none).
pub fn engine_key(value: Option<&Value>) -> String {
    match normalize(value, None) {
        Ok(m) if !m.is_empty() => format!(" logit_bias={}", m.iter().map(|(i, b)| format!("{i}:{}", py_g(*b))).collect::<Vec<_>>().join(",")),
        _ => String::new(),
    }
}

/// Fail closed for old engines and continuous batching.  `engine_logit_bias`: the engine's INFO logit_bias.
pub fn validate_request(req: &Map<String, Value>, engine_logit_bias: Option<i64>, batch: usize, vocab_size: Option<usize>) -> Result<BTreeMap<u64, f64>, Fail> {
    let n = normalize(req.get("logit_bias"), vocab_size)?;
    if !n.is_empty() {
        if engine_logit_bias != Some(1) {
            return bad("this engine does not support token biases; use a supporting CUDA/HIP build");
        }
        if batch > 0 {
            return bad("continuous batching is not supported; run without --batch");
        }
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn biases_normalize_as_the_reference_does() {
        assert_eq!(engine_key(Some(&json!({"12": -100, "3": 5.5}))), " logit_bias=3:5.5,12:-100");
        assert_eq!(engine_key(Some(&json!([[7, false]]))), " logit_bias=7:-100");
        assert!(normalize(Some(&json!({"x": 1})), None).is_err());
        assert!(normalize(Some(&json!([[1, 1], [1, 2]])), None).is_err());
        assert!(normalize(Some(&json!({"1": true})), None).is_err());
    }
}
