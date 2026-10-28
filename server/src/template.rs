//! The model's own Jinja chat template, rendered with the settings serve/frontend.py's ChatTemplate gives jinja2:
//! trim_blocks, lstrip_blocks, loop controls, `tojson` = Python's json.dumps (ensure_ascii off), `raise_exception`
//! (a 400 with the template's message) and `strftime_now` (GPT-OSS dates its system message with it).

use crate::frontend::{has_image, text_of, BadRequest};
use crate::parser::{glm_call_to_qwen, parse_tool_call, ToolCall, CALL_END, CALL_START};
use crate::py::{self, get};
use crate::pyjson;
use minijinja::value::{Kwargs, Value as MjValue};
use minijinja::{Environment, Error, ErrorKind};
use serde_json::{json, Map, Value};
use std::path::Path;

const RAISED: &str = "\u{0}raise_exception:";

/// What a render can fail with: the template refused the request (raise_exception: a 400), or it broke.
#[derive(Debug, Clone, thiserror::Error)]
pub enum TemplateError {
    #[error("{0}")]
    Request(String),
    #[error("{0}")]
    Broken(String),
}

impl From<TemplateError> for BadRequest {
    fn from(e: TemplateError) -> Self {
        BadRequest(e.to_string())
    }
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Caps {
    pub supports_tools: bool,
    pub supports_tool_calls: bool,
    pub supports_system_role: bool,
    pub supports_parallel_tool_calls: bool,
    pub supports_preserve_reasoning: bool,
}

pub struct ChatTemplate {
    env: Environment<'static>,
    pub source: String,
    pub caps: Caps,
}

fn tojson(v: MjValue, kwargs: Kwargs) -> Result<String, Error> {
    let ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    let indent: Option<usize> = kwargs.get("indent")?;
    let sort_keys: Option<bool> = kwargs.get("sort_keys")?;
    let separators: Option<Vec<String>> = kwargs.get("separators")?;
    kwargs.assert_all_used()?;
    let j: Value = serde_json::to_value(&v).map_err(|e| Error::new(ErrorKind::InvalidOperation, e.to_string()))?;
    let seps = separators.as_ref().map(|s| (s.first().map(String::as_str).unwrap_or(", "), s.get(1).map(String::as_str).unwrap_or(": ")));
    Ok(pyjson::dumps_with(&j, pyjson::Opts { ensure_ascii: ensure_ascii.unwrap_or(false), indent, separators: seps, sort_keys: sort_keys.unwrap_or(false) }))
}

/// Python's strftime spellings that chrono writes differently (none of the templates' formats need more).
fn strftime(fmt: &str) -> String {
    chrono::Local::now().format(fmt).to_string()
}

impl ChatTemplate {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        Self::from_source(std::fs::read_to_string(path)?)
    }

    pub fn from_source(source: String) -> anyhow::Result<Self> {
        let mut env = Environment::new();
        env.set_syntax(minijinja::syntax::SyntaxConfig::builder().trim_blocks(true).lstrip_blocks(true).build()?);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_filter("tojson", tojson);
        env.add_function("raise_exception", |msg: String| -> Result<String, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, format!("{RAISED}{msg}")))
        });
        env.add_function("strftime_now", |fmt: String| strftime(&fmt));
        env.add_template_owned("chat", source.clone())?;
        let mut t = ChatTemplate { env, source, caps: Caps::default() };
        t.caps = t.detect_caps();
        Ok(t)
    }

    /// The prompt for `messages` (completed empty assistant turns left out, #843; the last message kept).
    pub fn render(&self, messages: &[Value], tools: Option<&[Value]>, add_generation_prompt: bool, kwargs: &Map<String, Value>) -> Result<String, TemplateError> {
        let keep_empty = std::env::var("STRATA_KEEP_EMPTY_TURNS").ok().as_deref() == Some("1");
        let n = messages.len();
        let messages: Vec<&Value> = messages
            .iter()
            .enumerate()
            .filter(|(i, m)| {
                *i == n - 1
                    || keep_empty
                    || !(m.is_object()
                        && get(m, "role").as_str() == Some("assistant")
                        && py::strip(&text_of(get(m, "content"))).is_empty()
                        && !has_image(get(m, "content"))
                        && !py::truthy(get(m, "tool_calls")))
            })
            .map(|(_, m)| m)
            .collect();
        let tools: Option<Vec<Value>> = tools.map(|ts| {
            if ts.is_empty() || !self.source.contains("tool.function") {
                ts.to_vec()
            } else {
                ts.iter().map(|t| if t.as_object().map(|o| o.contains_key("function")).unwrap_or(false) { t.clone() } else { json!({"type": "function", "function": t}) }).collect()
            }
        });
        let mut ctx = Map::new();
        ctx.insert("messages".into(), Value::Array(messages.into_iter().cloned().collect()));
        ctx.insert("tools".into(), tools.map(Value::Array).unwrap_or(Value::Null));
        ctx.insert("add_generation_prompt".into(), Value::Bool(add_generation_prompt));
        for (k, v) in kwargs {
            ctx.insert(k.clone(), v.clone());
        }
        let tmpl = self.env.get_template("chat").map_err(|e| TemplateError::Broken(e.to_string()))?;
        tmpl.render(MjValue::from(minijinja::value::Serde(Value::Object(ctx)))).map_err(|e| {
            let mut cur: Option<&(dyn std::error::Error + 'static)> = Some(&e);
            while let Some(err) = cur {
                if let Some(mj) = err.downcast_ref::<Error>() {
                    if let Some(d) = mj.detail() {
                        if let Some(msg) = d.strip_prefix(RAISED) {
                            return TemplateError::Request(msg.to_string());
                        }
                    }
                }
                cur = err.source();
            }
            TemplateError::Broken(e.to_string())
        })
    }

    fn probe(&self, messages: &[Value], tools: Option<&[Value]>) -> String {
        self.render(messages, tools, true, &Map::new()).unwrap_or_default()
    }

    /// llama.cpp's capability names, checked against this template and the server's tool-call format.
    fn detect_caps(&self) -> Caps {
        let user = json!({"role": "user", "content": "strata_caps_user"});
        let tools: Vec<Value> = (0..2)
            .map(|i| json!({"name": format!("strata_caps_call_{i}"), "description": "strata_caps_description",
                            "parameters": {"type": "object", "properties": {"arg": {"type": "string"}}}}))
            .collect();
        let tool_prompt = self.probe(&[user.clone()], Some(&tools));
        let harmony = self.source.contains("<|channel|>");
        let glm = self.source.contains("<arg_key>");
        let calls_supported = |count: usize| -> bool {
            let calls: Vec<Value> = (0..count).map(|i| json!({"name": format!("strata_caps_call_{i}"), "arguments": {"arg": format!("strata_caps_arg_{i}")}})).collect();
            let mut messages = vec![user.clone(), json!({"role": "assistant", "content": "", "tool_calls": calls.iter().map(|c| json!({"function": c})).collect::<Vec<_>>()})];
            let replies: Vec<String> = (0..count).map(|i| format!("strata_caps_result_{i}")).collect();
            messages.extend(replies.iter().map(|r| json!({"role": "tool", "content": r})));
            messages.push(user.clone());
            let prompt = self.probe(&messages, Some(&tools));
            let parsed: Option<Vec<ToolCall>> = if harmony {
                let re = regex::Regex::new(r"(?s)to=functions\.(strata_caps_call_\d+)<\|channel\|>commentary json<\|message\|>(\{.*?\})<\|call\|>").unwrap();
                re.captures_iter(&prompt)
                    .map(|c| serde_json::from_str::<Value>(&c[2]).ok().and_then(|v| v.as_object().cloned()).map(|a| ToolCall::new(&c[1], a)))
                    .collect()
            } else if glm {
                let re = regex::Regex::new(r"(?s)<tool_call>strata_caps_call_\d+<arg_key>.*?</tool_call>").unwrap();
                re.find_iter(&prompt)
                    .map(|m| {
                        let q = glm_call_to_qwen(m.as_str());
                        let inner = q.get(CALL_START.len()..q.len().saturating_sub(CALL_END.len())).unwrap_or("");
                        parse_tool_call(inner, None).ok()
                    })
                    .collect()
            } else {
                let re = regex::Regex::new(r"(?s)<tool_call>[\s\x1c-\x1f]*(<function=strata_caps_call_\d+>.*?</function>)[\s\x1c-\x1f]*</tool_call>").unwrap();
                re.captures_iter(&prompt).map(|c| parse_tool_call(&c[1], None).ok()).collect()
            };
            let Some(parsed) = parsed else { return false };
            let got: Vec<Value> = parsed.iter().map(|c| json!({"name": c.name, "arguments": Value::Object(c.arguments.clone())})).collect();
            got == calls && replies.iter().all(|r| prompt.contains(r.as_str()))
        };
        let marks: &[&str] = if harmony { &["namespace functions"] } else if glm { &["<tools>", "<arg_key>"] } else { &["<tool_call>", "<function="] };
        let history = [user.clone(), json!({"role": "assistant", "content": "strata_caps_answer", "reasoning_content": "strata_caps_reasoning"}), user.clone()];
        Caps {
            supports_tools: ["strata_caps_call_0", "strata_caps_description"].iter().chain(marks).all(|s| tool_prompt.contains(s)),
            supports_tool_calls: calls_supported(1),
            supports_system_role: self.probe(&[json!({"role": "system", "content": "strata_caps_system"}), user.clone()], None).contains("strata_caps_system"),
            supports_parallel_tool_calls: calls_supported(2),
            supports_preserve_reasoning: self.probe(&history, None).contains("strata_caps_reasoning"),
        }
    }
}
