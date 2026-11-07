//! The model's own Jinja chat template, rendered with the settings serve/frontend.py's ChatTemplate gives jinja2:
//! trim_blocks, lstrip_blocks, loop controls, `tojson` = Python's json.dumps (ensure_ascii off), `raise_exception`
//! (a 400 with the template's message) and `strftime_now` (GPT-OSS dates its system message with it).

use crate::frontend::{has_image, text_of, BadRequest};
use crate::parser::{glm_call_to_qwen, parse_tool_call, PyLit, ToolCall, CALL_END, CALL_START};
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
    /// The template knows thinking (`<think>`, `enable_thinking`, GPT-OSS's channels); without, all output is answer.
    /// Not in /props: the reference's caps have no such key
    #[serde(skip)]
    pub thinks: bool,
}

/// An answer that starts with an empty thought: Qwen3's template ends the prompt with it for enable_thinking=false.
pub const EMPTY_THINK: &str = "<think>\n\n</think>\n\n";

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

/// transformers' own tags that jinja2 alone does not know: `{% generation %}...{% endgeneration %}` marks the assistant's
/// tokens for training masks and renders its body unchanged (LiquidAI's LFM2.5 templates use it).  Rewritten to
/// `{% if true %}...{% endif %}`, the whitespace dashes kept.
fn hf_tags(source: &str) -> String {
    let re = regex::Regex::new(r"\{%(-?)\s*(end)?generation\s*(-?)%\}").expect("regex");
    re.replace_all(source, |c: &regex::Captures| format!("{{%{} {} {}%}}", &c[1], if c.get(2).is_some() { "endif" } else { "if true" }, &c[3])).into_owned()
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
        // jinja2: None is not iterable (minijinja's says it is); Qwen3-Coder's template then takes `tools | length` of none
        env.add_test("iterable", |v: MjValue| !v.is_none() && v.try_iter().is_ok());
        env.add_function("raise_exception", |msg: String| -> Result<String, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, format!("{RAISED}{msg}")))
        });
        env.add_function("strftime_now", |fmt: String| strftime(&fmt));
        env.add_template_owned("chat", hf_tags(&source))?;
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
        let off = add_generation_prompt && kwargs.get("enable_thinking") == Some(&Value::Bool(false));
        tmpl.render(MjValue::from(minijinja::value::Serde(Value::Object(ctx))))
            .map(|p| if off && self.thinks_without_switch() { p + EMPTY_THINK } else { p })
            .map_err(|e| {
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

    /// The template writes `<think>` but has no `enable_thinking` to turn it off (LFM2.5): "thinking off" was passed in
    /// and ignored, the model thought anyway - ~200 tokens per answer, and response_format answers cut off at max_tokens
    /// came back 502 structured_output_failed.  Such a template gets what Qwen3's writes for enable_thinking=false.
    fn thinks_without_switch(&self) -> bool {
        // GLM 5.3 writes <think> too, but turns it with reasoning_effort: that is a switch
        self.source.contains("<think>") && !self.source.contains("enable_thinking") && !self.source.contains("reasoning_effort")
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
        let lfm = self.source.contains("<|tool_call_start|>");
        // Hermes' JSON calls (Qwen3-4B-Instruct-2507, Qwen2.5): `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`
        let hermes = !harmony && !glm && !lfm && self.source.contains("<tool_call>") && !self.source.contains("<function=");
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
            } else if lfm {
                let re = regex::Regex::new(r"(?s)<\|tool_call_start\|>(.*?)<\|tool_call_end\|>").unwrap();
                re.captures_iter(&prompt)
                    .map(|c| PyLit::new(&c[1]).calls())
                    .collect::<Option<Vec<_>>>()
                    .map(|v| v.into_iter().flatten().map(|(n, a)| ToolCall::new(n, a)).collect())
            } else if hermes {
                // the system prompt's own example (`{"name": <function-name>, ...}`) is no JSON: skipped, not a failure
                let re = regex::Regex::new(r"(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>").unwrap();
                Some(re.captures_iter(&prompt)
                    .filter_map(|c| {
                        let v: Value = serde_json::from_str(&c[1]).ok()?;
                        let args = match &v["arguments"] {
                            Value::String(a) => serde_json::from_str::<Value>(a).ok()?,
                            a => a.clone(),
                        };
                        Some(ToolCall::new(v["name"].as_str()?, args.as_object()?.clone()))
                    })
                    .collect())
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
        let marks: &[&str] = if harmony { &["namespace functions"] } else if lfm { &["List of tools:"] } else if hermes { &["<tool_call>"] } else if glm { &["<tools>", "<arg_key>"] } else { &["<tool_call>", "<function="] };
        let history = [user.clone(), json!({"role": "assistant", "content": "strata_caps_answer", "reasoning_content": "strata_caps_reasoning"}), user.clone()];
        Caps {
            supports_tools: ["strata_caps_call_0", "strata_caps_description"].iter().chain(marks).all(|s| tool_prompt.contains(s)),
            supports_tool_calls: calls_supported(1),
            supports_system_role: self.probe(&[json!({"role": "system", "content": "strata_caps_system"}), user.clone()], None).contains("strata_caps_system"),
            supports_parallel_tool_calls: calls_supported(2),
            supports_preserve_reasoning: self.probe(&history, None).contains("strata_caps_reasoning"),
            // ponytail: a word in the source, not a probe.  "thinking" (enable_thinking, keep_past_thinking: every thinking
            // model's template here), not "think": Qwen3-4B-Instruct-2507's only strips old `</think>` from history, and
            // its answers then all went to reasoning_content (its tool calls never ran)
            thinks: self.source.contains("thinking") || harmony,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_template_without_thinking_marks_a_model_that_does_not_think() {
        let plain = ChatTemplate::from_source("{% for m in messages %}User: {{ m.content }}\n{% endfor %}Assistant:".into()).unwrap();
        assert!(!plain.caps.thinks);
        let cleanup = ChatTemplate::from_source("{% for m in messages %}{% if '</think>' in m.content %}{{ m.content.split('</think>')|last }}{% else %}{{ m.content }}{% endif %}{% endfor %}".into()).unwrap();
        assert!(!cleanup.caps.thinks);                     // Qwen3-4B-Instruct-2507: </think> only in history cleanup
        assert!(ChatTemplate::load(&crate::root().join("server/chat_template.jinja")).unwrap().caps.thinks);
    }

    #[test]
    fn lfm_tool_calls_round_trip_through_its_template() {
        // LiquidAI's LFM2.5 template's call rendering and tool list, the rest cut down
        let src = r#"{%- macro fa(v) -%}{%- if v is string -%}{{ "'" + v + "'" }}{%- elif v is mapping -%}{{ v | tojson }}{%- else -%}{{ v | string }}{%- endif -%}{%- endmacro -%}
{%- if tools -%}<|im_start|>system
List of tools: [{%- for t in tools -%}{{ t | tojson }}{% if not loop.last %}, {% endif %}{%- endfor -%}]<|im_end|>
{% endif -%}
{%- for m in messages -%}<|im_start|>{{ m.role }}
{%- if m.tool_calls is defined -%}{%- set ns = namespace(c=[]) -%}{%- for tc in m.tool_calls -%}{%- set a = namespace(s=[]) -%}{%- for k, v in tc.function.arguments.items() -%}{%- set a.s = a.s + [k + "=" + fa(v)] -%}{%- endfor -%}{%- set ns.c = ns.c + [tc.function.name + "(" + (a.s | join(", ")) + ")"] -%}{%- endfor -%}{{ "<|tool_call_start|>[" + (ns.c | join(", ")) + "]<|tool_call_end|>" }}{%- endif -%}
{{ m.content }}<|im_end|>
{% endfor -%}"#;
        let t = ChatTemplate::from_source(src.into()).unwrap();
        assert!(t.caps.supports_tools && t.caps.supports_tool_calls && t.caps.supports_parallel_tool_calls, "{:?}", t.caps);
    }

    #[test]
    fn hermes_json_tool_calls_round_trip_through_its_template() {
        // Qwen3-4B-Instruct-2507's tool list and call rendering, the rest cut down
        let src = r#"{%- if tools %}<|im_start|>system
# Tools
<tools>{%- for tool in tools %}
{{ tool | tojson }}{%- endfor %}
</tools>
For each function call, return a json object within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
{% endif %}
{%- for m in messages %}<|im_start|>{{ m.role }}
{{ m.content }}
{%- if m.tool_calls %}{%- for tc in m.tool_calls %}{%- set tc = tc.function if tc.function else tc %}
<tool_call>
{"name": "{{ tc.name }}", "arguments": {{ tc.arguments | tojson if tc.arguments is not string else tc.arguments }}}
</tool_call>{%- endfor %}{%- endif %}<|im_end|>
{% endfor %}"#;
        let t = ChatTemplate::from_source(src.into()).unwrap();
        assert!(t.caps.supports_tools && t.caps.supports_tool_calls && t.caps.supports_parallel_tool_calls, "{:?}", t.caps);
        // the plain base-model template stays without tools
        let plain = ChatTemplate::from_source("{% for m in messages %}User: {{ m.content }}\n{% endfor %}Assistant:".into()).unwrap();
        assert!(!plain.caps.supports_tools && !plain.caps.supports_tool_calls);
    }

    #[test]
    fn generation_tags_render_their_body() {
        assert_eq!(hf_tags("a{%- generation -%} b {%- endgeneration -%}c{% generation %}d{% endgeneration %}"),
                   "a{%- if true -%} b {%- endif -%}c{% if true %}d{% endif %}");
        let t = ChatTemplate::from_source("{%- for m in messages %}<{{ m.role }}>{%- generation -%} {{ m.content }} {%- endgeneration -%}</>{% endfor %}".into()).unwrap();
        let p = t.render(&[json!({"role": "user", "content": "hi"}), json!({"role": "assistant", "content": "yo"})], None, false, &Map::new()).unwrap();
        assert_eq!(p, "<user>hi</><assistant>yo</>");
    }

    #[test]
    fn none_is_not_iterable_as_in_jinja2() {
        // Qwen3-Coder's template: no tools sent -> tools is none -> must skip the `| length` branch
        let t = ChatTemplate::from_source("{%- if tools is iterable and tools | length > 0 %}T{% endif %}{{ messages[0].content }}{{ 'ab' is iterable }}{{ 3 is iterable }}".into()).unwrap();
        assert_eq!(t.render(&[json!({"role": "user", "content": "hi"})], None, false, &Map::new()).unwrap(), "hiTrueFalse");
        assert_eq!(t.render(&[json!({"role": "user", "content": "hi"})], Some(&[json!({"name": "f"})]), false, &Map::new()).unwrap(), "ThiTrueFalse");
    }
}

#[cfg(test)]
mod empty_think_tests {
    use super::*;

    fn render(src: &str, kwargs: Value) -> String {
        let t = ChatTemplate::from_source(src.to_string()).unwrap();
        let msgs = [json!({"role": "user", "content": "hi"})];
        t.render(&msgs, None, true, kwargs.as_object().unwrap()).unwrap()
    }

    // LFM2.5's shape: writes past thoughts as <think>..</think>, knows no enable_thinking
    const NO_SWITCH: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{% if m.thinking %}<think>{{ m.thinking }}</think>{% endif %}{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
    // Qwen3's shape: the template turns thinking off itself
    const SWITCH: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}{% endif %}";

    #[test]
    fn thinking_off_reaches_a_template_with_no_switch_for_it() {
        let off = render(NO_SWITCH, json!({"enable_thinking": false}));
        assert!(off.ends_with(&format!("<|im_start|>assistant\n{EMPTY_THINK}")), "{off:?}");
        // not asked, or asked on: the model decides, as before
        assert!(render(NO_SWITCH, json!({})).ends_with("<|im_start|>assistant\n"));
        assert!(render(NO_SWITCH, json!({"enable_thinking": true})).ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn the_real_lfm25_template_gets_the_empty_thought() {
        let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/lfm2.5-chat_template.jinja")).unwrap();
        let off = render(&src, json!({"enable_thinking": false}));
        assert!(off.ends_with(&format!("<|im_start|>assistant\n{EMPTY_THINK}")), "{off:?}");
        assert!(render(&src, json!({})).ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn a_template_with_its_own_switch_is_left_to_it() {
        let off = render(SWITCH, json!({"enable_thinking": false}));
        assert_eq!(off.matches("</think>").count(), 1, "{off:?}");
    }
}
