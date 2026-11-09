//! Model quality score: code that has to pass its tests, tool calls that have to match, instructions that have to be
//! followed.  Fixed samples of public benchmarks, graded by code, so two models (or two quants) compare on the same tasks.
//!
//!     strata-eval CONFIG.json [CONFIG2.json ...] [--n 15] [--data ../Strata-data/eval] [--out bench/results/DATE-eval]
//!                 [--effort none|low|medium|high] [--parallel 4] [--restore CONFIG]
//!     strata-eval --selftest [--n 15]       graders against the benchmarks' own answers (no model)
//!
//! Areas (n tasks each):
//!   py    HumanEval+ (evalplus, Apache-2.0): the function, run against its base + plus tests (python3 + numpy)
//!   rs    MultiPL-E HumanEval in Rust (nuprl, MIT): compiled with rustc, its asserts run
//!   cpp   MultiPL-E HumanEval in C++: compiled with clang++ -std=c++17, its asserts run
//!   tools BFCL v3 simple / multiple / parallel (gorilla-llm, Apache-2.0): the API's tool_calls against the
//!         possible answers, BFCL's AST rules (name, no extra arguments, each value one of the accepted ones)
//!   ifeval IFEval (google, Apache-2.0): every rule of the prompt followed (prompt-level strict)
//! Score: code = mean(py, rs, cpp); overall = mean(code, tools, ifeval).  Each config is started on its port with
//! --parallel slots (4; fewer for a model that does not fit) and a 32K context, tasks run that many at a time, greedy (temperature 0).  At the end --restore (default: the first config)
//! is started again.  --effort goes to the API's reasoning_effort: none (the default) turns thinking off where the
//! template can (GPT-OSS always reasons a little), so a run takes minutes, not an hour.  Rows: tasks.jsonl (every answer and why it failed) and scores.jsonl; a table on stdout.
//
// ponytail: generated code runs as you, in a temp dir with a timeout, not in a sandbox (as evalplus/MultiPL-E do);
// IFEval's sentence count and capital words use regexes instead of nltk's punkt/word_tokenize, and its two case rules
// skip langdetect's "is English" part; prompts with language:response_language are left out.

#[path = "common/server.rs"]
mod server;

use regex::{Regex, RegexBuilder};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

const HF: &str = "https://huggingface.co/datasets";
const HE_URL: &str = "evalplus/humanevalplus/resolve/d32357cf319e50e9c8d8dab5ea876c72b0fd321b/test.jsonl";
const HE_SHA: &str = "908377f1daf28dcb36846db73a5662b2e05a9907407c2696c89ad9d3b0b04492";
const IFEVAL_URL: &str = "google/IFEval/resolve/966cd89545d6b6acfd7638bc708b98261ca58e84/ifeval_input_data.jsonl";
const IFEVAL_SHA: &str = "6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2";
const BFCL_REV: &str = "gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/61fc0608cfd831fcfbbaa676ebdfef0ed963eeda";
const BFCL_SHA: [(&str, &str, &str); 3] = [("simple", "fbc37b2ad252bf9af985582e0e07b456173fe627d957491472ea9cef5fb83158", "2911a2bc00df82c4f999ffa64fedb0164cb88e96212ffa5972087eb91fd496ee"), ("multiple", "aef168155ebd74b7ac2401198b201343bc7d16d7a3d7e0d4e6d8ee82c6969b2a", "244e00ce9395df948bcafc7bee64e8f9c87ef70887587d83cae45b13699f3047"), ("parallel", "fad375d68295776efb63e5aa331dd884f8635e7bb50e05908113c74f14281c3c", "3af54bf04e5b7640b03a9b6a82440f52474feb24b398f94435fd2b155c7727de")];
/// MultiPL-E is parquet on the Hub: its rows come from the datasets-server API, pinned by the hash of (name, prompt, tests).
const MPLE: [(&str, &str); 2] = [("rs", "ba736ada6ede9bd02dc9c50c1d49f3e80b3ee6abcc6b7a1d4dfa1253c4bd9dca"), ("cpp", "06b20060636f3994acb2d919089b04010714c6c4c8e2e0210920542b26723ba0")];
const AREAS: [&str; 5] = ["py", "rs", "cpp", "tools", "ifeval"];

fn sha256(b: &[u8]) -> String {
    hex::encode(Sha256::digest(b))
}

fn http() -> ureq::Agent {
    ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(900))).http_status_as_error(false).build().into()
}

/// One file of a pinned Hub revision, kept in `data` and checked by its sha-256.
fn get(data: &Path, url: &str, sha: &str) -> Vec<u8> {
    let path = data.join(url.replace('/', "_"));
    if let Ok(b) = fs::read(&path)
        && sha256(&b) == sha
    {
        return b;
    }
    let mut b = vec![];
    http().get(format!("{HF}/{url}")).call().unwrap_or_else(|e| panic!("{url}: {e}")).into_body().into_reader().read_to_end(&mut b).unwrap();
    let got = sha256(&b);
    assert!(got == sha, "{url}: sha-256 {got}, pinned {sha}");
    fs::write(&path, &b).unwrap();
    b
}

fn jsonl(b: &[u8]) -> Vec<Value> {
    String::from_utf8_lossy(b).lines().filter(|l| !l.trim().is_empty()).map(|l| serde_json::from_str(l).unwrap()).collect()
}

fn multipl_e(data: &Path, lang: &str, sha: &str) -> Vec<Value> {
    let path = data.join(format!("multipl-e-humaneval-{lang}.json"));
    if let Ok(b) = fs::read(&path) {
        return serde_json::from_slice(&b).unwrap();
    }
    let mut rows: Vec<Value> = vec![];
    loop {
        let url = format!("https://datasets-server.huggingface.co/rows?dataset=nuprl/MultiPL-E&config=humaneval-{lang}&split=test&offset={}&length=100", rows.len());
        let v: Value = serde_json::from_reader(http().get(&url).call().unwrap().into_body().into_reader()).unwrap();
        let page = v["rows"].as_array().unwrap();
        rows.extend(page.iter().map(|r| json!({"name": r["row"]["name"], "prompt": r["row"]["prompt"], "tests": r["row"]["tests"]})));
        if page.len() < 100 {
            break;
        }
    }
    rows.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    let got = sha256(rows.iter().map(|r| format!("{}\0{}\0{}\0", r["name"], r["prompt"], r["tests"])).collect::<String>().as_bytes());
    assert!(got == sha, "MultiPL-E humaneval-{lang}: rows hash {got}, pinned {sha}");
    fs::write(&path, serde_json::to_vec(&rows).unwrap()).unwrap();
    rows
}

/// n items spread evenly over xs (the same ones every run).
fn pick<T: Clone>(xs: &[T], n: usize) -> Vec<T> {
    let n = n.min(xs.len());
    (0..n).map(|k| xs[k * xs.len() / n].clone()).collect()
}

fn he_number(s: &str) -> Option<u64> {
    s.trim_start_matches("HumanEval").trim_start_matches(['/', '_']).split('_').next()?.parse().ok()
}

/// BFCL's schemas are Python-flavoured: dict/float/tuple/any -> JSON Schema; dots in names are not allowed by the API.
fn schema(v: &Value) -> Value {
    match v {
        Value::Object(m) => Value::Object(m.iter().map(|(k, x)| {
            let x = match (k.as_str(), x.as_str()) {
                ("type", Some("dict")) => json!("object"),
                ("type", Some("float")) => json!("number"),
                ("type", Some("tuple")) => json!("array"),
                ("type", Some("any")) => json!("string"),
                _ => schema(x),
            };
            (k.clone(), x)
        }).collect()),
        Value::Array(a) => Value::Array(a.iter().map(schema).collect()),
        x => x.clone(),
    }
}

fn tasks(data: &Path, n: usize) -> Vec<Value> {
    fs::create_dir_all(data).unwrap();
    let mut out = vec![];
    // code: the same HumanEval problems in all three languages
    let he = jsonl(&get(data, HE_URL, HE_SHA));
    let [rs, cpp] = MPLE.map(|(l, s)| multipl_e(data, l, s));
    let has = |rows: &[Value], k: u64| rows.iter().find(|r| he_number(r["name"].as_str().unwrap()) == Some(k)).cloned();
    let common: Vec<u64> = he.iter().filter_map(|h| he_number(h["task_id"].as_str()?)).filter(|&k| has(&rs, k).is_some() && has(&cpp, k).is_some()).collect();
    for k in pick(&common, n) {
        let h = he.iter().find(|h| he_number(h["task_id"].as_str().unwrap()) == Some(k)).unwrap();
        out.push(json!({"id": format!("py/{k}"), "area": "py", "prompt_code": h["prompt"], "entry_point": h["entry_point"],
                        "test": h["test"], "canonical": h["canonical_solution"]}));
        for (area, rows) in [("rs", &rs), ("cpp", &cpp)] {
            let r = has(rows, k).unwrap();
            out.push(json!({"id": format!("{area}/{k}"), "area": area, "prompt_code": r["prompt"], "tests": r["tests"]}));
        }
    }
    // tools: a third each of simple, multiple, parallel
    for (cat, qs, as_) in BFCL_SHA {
        let q = jsonl(&get(data, &format!("{BFCL_REV}/BFCL_v3_{cat}.json"), qs));
        let a = jsonl(&get(data, &format!("{BFCL_REV}/possible_answer/BFCL_v3_{cat}.json"), as_));
        for (q, a) in pick(&q.into_iter().zip(a).collect::<Vec<_>>(), n.div_ceil(3)) {
            assert_eq!(q["id"], a["id"]);
            let tools: Vec<Value> = q["function"].as_array().unwrap().iter().map(|f| json!({"type": "function", "function": {
                "name": f["name"].as_str().unwrap().replace('.', "_"), "description": f["description"], "parameters": schema(&f["parameters"])}})).collect();
            out.push(json!({"id": format!("tools/{}", q["id"].as_str().unwrap()), "area": "tools", "messages": q["question"][0],
                            "tools": tools, "ground_truth": a["ground_truth"]}));
        }
    }
    // instruction following: language detection is not graded here
    let ie: Vec<Value> = jsonl(&get(data, IFEVAL_URL, IFEVAL_SHA)).into_iter()
        .filter(|e| !e["instruction_id_list"].as_array().unwrap().iter().any(|i| i == "language:response_language")).collect();
    for e in pick(&ie, n) {
        out.push(json!({"id": format!("ifeval/{}", e["key"]), "area": "ifeval", "prompt": e["prompt"],
                        "instruction_id_list": e["instruction_id_list"], "kwargs": e["kwargs"]}));
    }
    out
}

// ---------- asking the model ----------

const LANG: [(&str, &str, &str); 3] = [("py", "Python", "python"), ("rs", "Rust", "rust"), ("cpp", "C++", "cpp")];

fn messages(t: &Value) -> Value {
    match t["area"].as_str().unwrap() {
        "tools" => t["messages"].clone(),
        "ifeval" => json!([{"role": "user", "content": t["prompt"]}]),
        area => {
            let (_, name, tag) = LANG.iter().find(|l| l.0 == area).unwrap();
            json!([{"role": "user", "content": format!(
                "Complete this {name} function. Reply with the complete function, with the imports it needs, in one ```{tag} code block. \
                 Do not write a main function or tests.\n\n```{tag}\n{}\n```", t["prompt_code"].as_str().unwrap().trim_end())}])
        }
    }
}

struct Answer {
    content: String,
    calls: Vec<(String, Value)>,
    out_tokens: u64,
    secs: f64,
    err: Option<String>,
}

fn ask(base: &str, key: &str, model: &str, effort: &str, t: &Value) -> Answer {
    let mut body = json!({"model": model, "messages": messages(t), "max_tokens": 4096, "temperature": 0, "stream": false,
                          "reasoning_effort": effort});
    if t["area"] == "tools" {
        body["tools"] = t["tools"].clone();
    }
    let t0 = Instant::now();
    let mut a = Answer { content: String::new(), calls: vec![], out_tokens: 0, secs: 0.0, err: None };
    let r = http().post(format!("{base}/v1/chat/completions")).header("Authorization", format!("Bearer {key}"))
        .header("Content-Type", "application/json").send(body.to_string().as_bytes());
    a.secs = t0.elapsed().as_secs_f64();
    let v: Value = match r {
        Err(e) => {
            a.err = Some(e.to_string());
            return a;
        }
        Ok(r) => {
            let mut s = String::new();
            let _ = r.into_body().into_reader().read_to_string(&mut s);
            serde_json::from_str(&s).unwrap_or_else(|_| json!({"error": s.chars().take(300).collect::<String>()}))
        }
    };
    if v.get("error").is_some() || v["choices"][0].is_null() {
        a.err = Some(v["error"].to_string().chars().take(300).collect());
        return a;
    }
    let m = &v["choices"][0]["message"];
    a.content = m["content"].as_str().unwrap_or("").to_string();
    a.out_tokens = v["usage"]["completion_tokens"].as_u64().unwrap_or(0);
    for c in m["tool_calls"].as_array().into_iter().flatten() {
        let args = &c["function"]["arguments"];
        let args = match args.as_str() {
            Some(s) => serde_json::from_str(s).unwrap_or(Value::String(s.into())),
            None => args.clone(),
        };
        a.calls.push((c["function"]["name"].as_str().unwrap_or("").to_string(), args));
    }
    a
}

// ---------- grading: code ----------

/// The code block of the answer: the first one tagged with the language, else the first one, else the whole text.
fn code_block(text: &str, tag: &str) -> String {
    // fences at the start of a line only: a doc comment's own /// ``` example is not one
    let re = Regex::new(r"(?ms)^```([A-Za-z0-9+#]*)[^\n]*\n(.*?)^```").unwrap();
    let blocks: Vec<(String, String)> = re.captures_iter(text).map(|c| (c[1].to_lowercase(), c[2].to_string())).collect();
    let alias = |t: &str| t == tag || (tag == "cpp" && t == "c++") || (tag == "rust" && t == "rs") || (tag == "python" && t == "py");
    blocks.iter().find(|b| alias(&b.0)).or(blocks.first()).map(|b| b.1.clone()).unwrap_or_else(|| text.to_string())
}

/// Runs a command with a time limit; Ok(()) on exit 0, Err(the reason) otherwise.
fn run(cmd: &mut Command, secs: u64) -> Result<(), String> {
    let mut c = cmd.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::piped()).spawn().map_err(|e| e.to_string())?;
    let t0 = Instant::now();
    loop {
        if let Some(s) = c.try_wait().map_err(|e| e.to_string())? {
            if s.success() {
                return Ok(());
            }
            let mut e = String::new();
            let _ = c.stderr.take().unwrap().read_to_string(&mut e);
            return Err(format!("{s}: {}", e.lines().filter(|l| !l.trim().is_empty()).take(6).collect::<Vec<_>>().join(" | ")));
        }
        if t0.elapsed() > Duration::from_secs(secs) {
            let _ = c.kill();
            let _ = c.wait();
            return Err(format!("timeout {secs} s"));
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// The function name the prompt ends with (fn NAME( / NAME( for C++).
fn fn_name(prompt: &str, area: &str) -> String {
    let re = if area == "rs" { Regex::new(r"fn\s+(\w+)\s*[<(]").unwrap() } else { Regex::new(r"(\w+)\s*\(").unwrap() };
    re.captures_iter(prompt).last().map(|c| c[1].to_string()).unwrap_or_default()
}

fn grade_code(t: &Value, answer: &str, dir: &Path) -> Result<(), String> {
    let area = t["area"].as_str().unwrap();
    let prompt = t["prompt_code"].as_str().unwrap();
    let tag = LANG.iter().find(|l| l.0 == area).unwrap().2;
    let code = code_block(answer, tag);
    fs::create_dir_all(dir).unwrap();
    // a body without its signature goes after the prompt; a whole function replaces it (the prompt's imports kept)
    let lines = |pre: &[&str]| prompt.lines().filter(|l| pre.iter().any(|p| l.trim_start().starts_with(p))).collect::<Vec<_>>().join("\n");
    match area {
        "py" => {
            let ep = t["entry_point"].as_str().unwrap();
            let body = if code.contains(&format!("def {ep}")) { format!("{}\n{code}", lines(&["import ", "from "])) } else { format!("{prompt}{code}") };
            fs::write(dir.join("t.py"), format!("{body}\n\n{}\n\ncheck({ep})\n", t["test"].as_str().unwrap())).unwrap();
            run(Command::new("python3").arg("-I").arg("t.py").current_dir(dir), 60)
        }
        _ => {
            let tests = t["tests"].as_str().unwrap();
            let name = fn_name(prompt, area);
            let whole = Regex::new(&format!(r"\b{}\s*[<(]", regex::escape(&name))).unwrap().is_match(&code);
            let body = if whole {
                let head = if area == "rs" { lines(&["use "]) } else { lines(&["#include"]) };
                format!("{head}\n{code}\n{}", tests.trim_start().trim_start_matches('}'))
            } else {
                format!("{prompt}{code}\n{tests}")
            };
            if area == "rs" {
                fs::write(dir.join("t.rs"), body).unwrap();
                run(Command::new("rustc").args(["--edition", "2021", "-A", "warnings", "-o", "t", "t.rs"]).current_dir(dir), 120)?;
            } else {
                // macOS's libc++ has no <bits/stdc++.h>, which every MultiPL-E C++ prompt includes
                fs::create_dir_all(dir.join("bits")).unwrap();
                fs::write(dir.join("bits/stdc++.h"), STDCPP).unwrap();
                fs::write(dir.join("t.cpp"), body).unwrap();
                run(Command::new("clang++").args(["-std=c++17", "-w", "-I", ".", "-o", "t", "t.cpp"]).current_dir(dir), 120)?;
            }
            run(Command::new("./t").current_dir(dir), 30)
        }
    }
}

const STDCPP: &str = "#include <algorithm>\n#include <array>\n#include <bitset>\n#include <cassert>\n#include <cctype>\n#include <climits>\n\
#include <cmath>\n#include <cstdint>\n#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n#include <deque>\n#include <functional>\n\
#include <iomanip>\n#include <iostream>\n#include <iterator>\n#include <limits>\n#include <list>\n#include <map>\n#include <numeric>\n\
#include <optional>\n#include <queue>\n#include <set>\n#include <sstream>\n#include <stack>\n#include <string>\n#include <tuple>\n\
#include <unordered_map>\n#include <unordered_set>\n#include <utility>\n#include <variant>\n#include <vector>\n";

// ---------- grading: tool calls (BFCL's AST rules) ----------

/// BFCL's standardize_string: no spaces or ,./-_*^, lower case, ' as ".
fn norm(s: &str) -> String {
    s.chars().filter(|c| !" ,./-_*^".contains(*c)).collect::<String>().to_lowercase().replace('\'', "\"")
}

fn value_ok(v: &Value, accepted: &Value) -> bool {
    match (v, accepted) {
        (Value::String(a), Value::String(b)) => norm(a) == norm(b),
        (Value::Number(a), Value::Number(b)) => (a.as_f64().unwrap() - b.as_f64().unwrap()).abs() < 1e-9,
        (Value::Array(a), Value::Array(b)) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| value_ok(x, y)),
        // nested dicts: each key's accepted values are a list again ("" = may be absent)
        (Value::Object(a), Value::Object(b)) => a.keys().all(|k| b.contains_key(k)) && b.iter().all(|(k, acc)| match a.get(k) {
            None => acc.as_array().is_some_and(|l| l.contains(&json!(""))),
            Some(x) => acc.as_array().map_or_else(|| value_ok(x, acc), |l| l.iter().any(|y| value_ok(x, y))),
        }),
        (a, b) => a == b,
    }
}

fn call_ok(call: &(String, Value), gt: &Value) -> bool {
    let (gname, gparams) = gt.as_object().unwrap().iter().next().unwrap();
    let Some(args) = call.1.as_object() else { return false };
    call.0 == gname.replace('.', "_")
        && args.keys().all(|k| gparams.get(k).is_some())
        && gparams.as_object().unwrap().iter().all(|(k, acc)| {
            let acc = acc.as_array().unwrap();
            match args.get(k) {
                None => acc.contains(&json!("")),
                Some(v) => acc.iter().any(|a| a != "" && value_ok(v, a)),
            }
        })
}

/// Every expected call matched by a different call of the model, in any order, and no call left over.
fn grade_tools(calls: &[(String, Value)], gt: &[Value]) -> Result<(), String> {
    fn assign(calls: &[(String, Value)], gt: &[Value], used: &mut Vec<bool>) -> bool {
        let Some((g, rest)) = gt.split_first() else { return true };
        (0..calls.len()).any(|i| {
            if used[i] || !call_ok(&calls[i], g) {
                return false;
            }
            used[i] = true;
            let ok = assign(calls, rest, used);
            used[i] = ok;
            ok
        })
    }
    if calls.len() != gt.len() {
        return Err(format!("{} calls, {} expected", calls.len(), gt.len()));
    }
    if assign(calls, gt, &mut vec![false; calls.len()]) { Ok(()) } else { Err("a call does not match".into()) }
}

// ---------- grading: IFEval ----------

fn re(p: &str) -> Regex {
    Regex::new(p).unwrap()
}
fn re_i(p: &str) -> Regex {
    RegexBuilder::new(p).case_insensitive(true).build().unwrap()
}
/// Python's str.isupper()/islower(): at least one cased character and none of the other case.
fn py_case(s: &str, upper: bool) -> bool {
    let cased = s.chars().any(|c| c.is_lowercase() || c.is_uppercase());
    cased && !s.chars().any(|c| if upper { c.is_lowercase() } else { c.is_uppercase() })
}
fn cmp(n: usize, rel: &Value, thr: &Value) -> bool {
    let t = thr.as_u64().unwrap() as usize;
    if rel == "less than" { n < t } else { n >= t }
}
fn sentences(v: &str) -> usize {
    re(r"[.!?]+(\s+|$)").split(v).filter(|s| !s.trim().is_empty()).count()
}

fn ifeval_rule(id: &str, k: &Value, v: &str, prompt: &str) -> bool {
    let s = |n: &str| k[n].as_str().unwrap_or("");
    match id {
        "punctuation:no_comma" => !v.contains(','),
        "length_constraints:number_words" => cmp(re(r"\w+").find_iter(v).count(), &k["relation"], &k["num_words"]),
        "length_constraints:number_sentences" => cmp(sentences(v), &k["relation"], &k["num_sentences"]),
        "length_constraints:number_paragraphs" => {
            let p: Vec<&str> = re(r"\s?\*\*\*\s?").split(v).collect();
            let last = p.len() - 1;
            if p.iter().enumerate().any(|(i, x)| x.trim().is_empty() && i != 0 && i != last) {
                return false;
            }
            p.iter().filter(|x| !x.trim().is_empty()).count() == k["num_paragraphs"].as_u64().unwrap() as usize
        }
        "length_constraints:nth_paragraph_first_word" => {
            let p: Vec<&str> = v.split("\n\n").collect();
            let count = p.iter().filter(|x| !x.trim().is_empty()).count();
            let nth = k["nth_paragraph"].as_u64().unwrap() as usize;
            if nth > count || nth == 0 || p[nth - 1].trim().is_empty() {
                return false;
            }
            let w = p[nth - 1].split_whitespace().next().unwrap_or("").trim_start_matches(['\'', '"']);
            let first: String = w.chars().take_while(|c| !".,?!'\"".contains(*c)).collect::<String>().to_lowercase();
            count == k["num_paragraphs"].as_u64().unwrap() as usize && first == s("first_word")
        }
        "keywords:forbidden_words" => k["forbidden_words"].as_array().unwrap().iter()
            .all(|w| !re_i(&format!(r"\b{}\b", regex::escape(w.as_str().unwrap()))).is_match(v)),
        "keywords:existence" => k["keywords"].as_array().unwrap().iter().all(|w| re_i(&regex::escape(w.as_str().unwrap())).is_match(v)),
        "keywords:frequency" => cmp(re_i(&regex::escape(s("keyword"))).find_iter(v).count(), &k["relation"], &k["frequency"]),
        "keywords:letter_frequency" => {
            let n = v.to_lowercase().matches(s("letter").to_lowercase().as_str()).count();
            cmp(n, &k["let_relation"], &k["let_frequency"])
        }
        "detectable_format:number_highlighted_sections" => {
            let single = re(r"\*[^\n*]*\*").find_iter(v).filter(|m| !m.as_str().trim_matches('*').trim().is_empty()).count();
            let double = re(r"\*\*[^\n*]*\*\*").find_iter(v).filter(|m| !m.as_str()[2..m.len() - 2].trim().is_empty()).count();
            single + double >= k["num_highlights"].as_u64().unwrap() as usize
        }
        "detectable_format:number_bullet_lists" => {
            let n = re(r"(?m)^\s*\*[^*].*$").find_iter(v).count() + re(r"(?m)^\s*-.*$").find_iter(v).count();
            n == k["num_bullets"].as_u64().unwrap() as usize
        }
        "detectable_format:multiple_sections" => {
            re(&format!(r"\s?{}\s?\d+\s?", regex::escape(s("section_spliter")))).split(v).count() > k["num_sections"].as_u64().unwrap() as usize
        }
        "detectable_format:title" => re(r"<<[^\n]+>>").find_iter(v).any(|m| !m.as_str().trim_start_matches('<').trim_end_matches('>').trim().is_empty()),
        "detectable_format:json_format" => {
            let t = v.trim();
            let t = ["```json", "```Json", "```JSON", "```"].iter().find_map(|p| t.strip_prefix(p)).unwrap_or(t);
            serde_json::from_str::<Value>(t.strip_suffix("```").unwrap_or(t).trim()).is_ok()
        }
        "detectable_format:constrained_response" => ["My answer is yes.", "My answer is no.", "My answer is maybe."].iter().any(|r| v.trim().contains(r)),
        "detectable_content:number_placeholders" => re(r"\[.*?\]").find_iter(v).count() >= k["num_placeholders"].as_u64().unwrap() as usize,
        "detectable_content:postscript" => {
            let p = match s("postscript_marker") {
                "P.P.S" => r"(?m)\s*p\.\s?p\.\s?s.*$".to_string(),
                "P.S." => r"(?m)\s*p\.\s?s\..*$".to_string(),
                m => format!(r"(?m)\s*{}.*$", regex::escape(&m.to_lowercase())),
            };
            re(&p).is_match(&v.to_lowercase())
        }
        "combination:repeat_prompt" => v.trim().to_lowercase().starts_with(&s("prompt_to_repeat").trim().to_lowercase()),
        "combination:two_responses" => {
            let r: Vec<&str> = v.split("******").collect();
            let last = r.len() - 1;
            if r.iter().enumerate().any(|(i, x)| x.trim().is_empty() && i != 0 && i != last) {
                return false;
            }
            let ok: Vec<&str> = r.into_iter().filter(|x| !x.trim().is_empty()).collect();
            ok.len() == 2 && ok[0].trim() != ok[1].trim()
        }
        "startend:end_checker" => v.trim().trim_matches('"').to_lowercase().ends_with(&s("end_phrase").trim().to_lowercase()),
        "startend:quotation" => {
            let t = v.trim();
            t.len() > 1 && t.starts_with('"') && t.ends_with('"')
        }
        "change_case:english_capital" => py_case(v, true),
        "change_case:english_lowercase" => py_case(v, false),
        "change_case:capital_word_frequency" => {
            let n = re(r"\w+").find_iter(v).filter(|w| py_case(w.as_str(), true)).count();
            cmp(n, &k["capital_relation"], &k["capital_frequency"])
        }
        _ => panic!("IFEval rule {id} not graded (prompt: {})", prompt.chars().take(80).collect::<String>()),
    }
}

fn grade_ifeval(t: &Value, v: &str) -> Result<(), String> {
    let failed: Vec<&str> = t["instruction_id_list"].as_array().unwrap().iter().zip(t["kwargs"].as_array().unwrap())
        .map(|(id, k)| (id.as_str().unwrap(), k))
        .filter(|(id, k)| !ifeval_rule(id, k, v, t["prompt"].as_str().unwrap()))
        .map(|(id, _)| id)
        .collect();
    if failed.is_empty() { Ok(()) } else { Err(failed.join(", ")) }
}

fn grade(t: &Value, a: &Answer, dir: &Path) -> Result<(), String> {
    if let Some(e) = &a.err {
        return Err(format!("request: {e}"));
    }
    match t["area"].as_str().unwrap() {
        "tools" => grade_tools(&a.calls, t["ground_truth"].as_array().unwrap()),
        "ifeval" => grade_ifeval(t, &a.content),
        _ => grade_code(t, &a.content, dir),
    }
}

// ---------- the run ----------

fn arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(String::as_str)
}

fn set_arg(args: &mut Vec<Value>, name: &str, val: &str) {
    match args.iter().position(|a| a == name) {
        Some(i) => args[i + 1] = json!(val),
        None => args.extend([json!(name), json!(val)]),
    }
}

fn scores(rows: &[(Value, bool)]) -> Map<String, Value> {
    let mut m = Map::new();
    for area in AREAS {
        let r: Vec<_> = rows.iter().filter(|(t, _)| t["area"] == area).collect();
        m.insert(area.into(), json!(100.0 * r.iter().filter(|x| x.1).count() as f64 / r.len().max(1) as f64));
    }
    let f = |k: &str| m[k].as_f64().unwrap();
    let code = (f("py") + f("rs") + f("cpp")) / 3.0;
    let overall = (code + f("tools") + f("ifeval")) / 3.0;
    m.insert("code".into(), json!(code));
    m.insert("overall".into(), json!(overall));
    m
}

fn selftest(ts: &[Value], scratch: &Path) {
    // the benchmarks' own answers must pass their graders: HumanEval+'s canonical solutions, BFCL's first accepted values
    let mut bad = 0;
    for t in ts {
        let r = match t["area"].as_str().unwrap() {
            "py" => grade_code(t, &format!("```python\n{}{}\n```", t["prompt_code"].as_str().unwrap(), t["canonical"].as_str().unwrap()),
                               &scratch.join(t["id"].as_str().unwrap().replace('/', "-"))),
            "tools" => {
                let calls: Vec<(String, Value)> = t["ground_truth"].as_array().unwrap().iter().map(|g| {
                    let (name, ps) = g.as_object().unwrap().iter().next().unwrap();
                    let args = ps.as_object().unwrap().iter().filter_map(|(k, acc)| acc.as_array().unwrap().iter().find(|a| *a != "").map(|a| (k.clone(), a.clone()))).collect();
                    (name.replace('.', "_"), Value::Object(args))
                }).collect();
                grade_tools(&calls, t["ground_truth"].as_array().unwrap())
            }
            _ => continue,
        };
        println!("{:<22} {}", t["id"].as_str().unwrap(), r.as_ref().err().map_or("ok".into(), |e| format!("FAIL {e}")));
        bad += r.is_err() as usize;
    }
    assert_eq!(bad, 0, "{bad} reference answers failed their graders");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let n: usize = arg(&args, "--n").map_or(15, |x| x.parse().expect("--n"));
    let effort = arg(&args, "--effort").unwrap_or("none");
    let slots: usize = arg(&args, "--parallel").map_or(4, |x| x.parse().expect("--parallel"));
    let configs: Vec<PathBuf> = {
        let mut v = vec![];
        let mut it = args.iter();
        while let Some(a) = it.next() {
            if a.starts_with("--") {
                if a != "--selftest" {
                    it.next();
                }
            } else {
                v.push(PathBuf::from(a));
            }
        }
        v
    };
    let root = std::env::current_dir().unwrap();
    let data = arg(&args, "--data").map_or_else(|| root.join("../Strata-data/eval"), PathBuf::from);
    let ts = tasks(&data, n);
    let scratch = std::env::temp_dir().join(format!("strata-eval-{}", std::process::id()));
    if args.iter().any(|a| a == "--selftest") {
        selftest(&ts, &scratch);
        let _ = fs::remove_dir_all(&scratch);
        println!("selftest: {} tasks, graders pass the reference answers", ts.len());
        return;
    }
    assert!(!configs.is_empty(), "usage: strata-eval CONFIG.json [CONFIG2.json ...] [--n 15] [--restore CONFIG]");
    let out = arg(&args, "--out").map_or_else(|| root.join(format!("bench/results/{}-eval", chrono::Local::now().format("%Y-%m-%d"))), PathBuf::from);
    fs::create_dir_all(&out).unwrap();
    let mut task_rows = fs::OpenOptions::new().create(true).append(true).open(out.join("tasks.jsonl")).unwrap();
    let mut score_rows = fs::OpenOptions::new().create(true).append(true).open(out.join("scores.jsonl")).unwrap();
    let mut table = vec![];
    let mut prev = None;
    let mut port = 8090;
    let mut server_exe = PathBuf::new();
    for cfg_path in &configs {
        let mut cfg: Value = serde_json::from_str(&fs::read_to_string(cfg_path).unwrap()).unwrap();
        let key = server::api_key(&cfg);
        let model = cfg["model_name"].as_str().unwrap_or("").to_string();
        port = cfg["port"].as_u64().unwrap_or(8090);
        server_exe = Path::new(cfg["exe"].as_str().unwrap()).with_file_name("strata-server");
        let cwd = PathBuf::from(cfg["cwd"].as_str().unwrap());
        set_arg(cfg["args"].as_array_mut().unwrap(), "--max-context", "32768");
        cfg["parallel"] = json!(slots);
        cfg["log"] = json!(out.join("engine.log"));
        let name = cfg_path.file_stem().unwrap().to_string_lossy().trim_start_matches("strata-").to_string();
        let cpath = out.join("config.json");
        fs::write(&cpath, serde_json::to_string_pretty(&cfg).unwrap()).unwrap();
        if let Some(c) = prev.take() {
            server::stop(c);
        }
        let base = format!("http://127.0.0.1:{port}");
        let mut child = server::start(&server_exe, &cpath, port, &out.join(format!("server-{name}.log")), &cwd);
        let t0 = Instant::now();
        if let Err(e) = server::wait_loaded(&mut child, &base, &key) {
            println!("{name}: {e}");
            prev = Some(child);
            continue;
        }
        let next = AtomicUsize::new(0);
        let rows = Mutex::new(vec![]);
        std::thread::scope(|s| {
            for _ in 0..slots {
                s.spawn(|| loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(t) = ts.get(i) else { break };
                    let a = ask(&base, &key, &model, effort, t);
                    let dir = scratch.join(format!("{name}-{}", t["id"].as_str().unwrap().replace('/', "-")));
                    let r = grade(t, &a, &dir);
                    let _ = fs::remove_dir_all(&dir);
                    let row = json!({"model": name, "id": t["id"], "area": t["area"], "pass": r.is_ok(), "why": r.as_ref().err(),
                                     "secs": a.secs, "out_tokens": a.out_tokens, "content": a.content,
                                     "tool_calls": a.calls.iter().map(|(n, x)| json!({"name": n, "arguments": x})).collect::<Vec<_>>()});
                    rows.lock().unwrap().push((t.clone(), r.is_ok(), row));
                });
            }
        });
        let rows = rows.into_inner().unwrap();
        for (_, _, row) in &rows {
            writeln!(task_rows, "{row}").unwrap();
        }
        let mut sc = scores(&rows.iter().map(|(t, ok, _)| (t.clone(), *ok)).collect::<Vec<_>>());
        sc.insert("model".into(), json!(name));
        sc.insert("n".into(), json!(n));
        sc.insert("effort".into(), json!(effort));
        sc.insert("wall_s".into(), json!(t0.elapsed().as_secs_f64()));
        sc.insert("out_tokens".into(), json!(rows.iter().map(|r| r.2["out_tokens"].as_u64().unwrap_or(0)).sum::<u64>()));
        writeln!(score_rows, "{}", Value::Object(sc.clone())).unwrap();
        println!("{name}: overall {:.0} (code {:.0}, tools {:.0}, ifeval {:.0}) in {:.0} s", sc["overall"].as_f64().unwrap(),
                 sc["code"].as_f64().unwrap(), sc["tools"].as_f64().unwrap(), sc["ifeval"].as_f64().unwrap(), sc["wall_s"].as_f64().unwrap());
        table.push(sc);
        prev = Some(child);
    }
    if let Some(c) = prev.take() {
        server::stop(c);
    }
    let _ = fs::remove_dir_all(&scratch);
    table.sort_by(|a, b| b["overall"].as_f64().partial_cmp(&a["overall"].as_f64()).unwrap());
    println!("\n{:<28} {:>7} {:>5} {:>4} {:>4} {:>4} {:>6} {:>7} {:>6}", "model", "overall", "code", "py", "rs", "cpp", "tools", "ifeval", "wall_s");
    for r in &table {
        let f = |k: &str| r[k].as_f64().unwrap_or(f64::NAN);
        println!("{:<28} {:>7.0} {:>5.0} {:>4.0} {:>4.0} {:>4.0} {:>6.0} {:>7.0} {:>6.0}", r["model"].as_str().unwrap(), f("overall"), f("code"),
                 f("py"), f("rs"), f("cpp"), f("tools"), f("ifeval"), f("wall_s"));
    }
    let restore = arg(&args, "--restore").map_or_else(|| configs[0].clone(), PathBuf::from);
    let cwd = serde_json::from_str::<Value>(&fs::read_to_string(&restore).unwrap()).unwrap()["cwd"].as_str().map(PathBuf::from).unwrap_or(root);
    let _ = server::start(&server_exe, &restore, port, &out.join("server-restore.log"), &cwd);
    println!("results: {}\nrestored {} on :{port}", out.display(), restore.display());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ifeval_rules_follow_the_reference_checkers() {
        let k = json!({"relation": "at least", "num_words": 3, "num_highlights": 2, "num_bullets": 2, "postscript_marker": "P.S.",
                       "num_paragraphs": 2, "section_spliter": "SECTION", "num_sections": 2, "end_phrase": "Any other questions?"});
        let r = |id: &str, v: &str| ifeval_rule(id, &k, v, "");
        assert!(r("length_constraints:number_words", "one two three") && !r("length_constraints:number_words", "one two"));
        assert!(r("detectable_format:number_highlighted_sections", "*a* and **b**") && !r("detectable_format:number_highlighted_sections", "*a* and ** **"));
        assert!(r("detectable_format:number_bullet_lists", "* one\n- two\n**bold**") && !r("detectable_format:number_bullet_lists", "* one"));
        assert!(r("detectable_content:postscript", "hi\nP.S. bye") && !r("detectable_content:postscript", "hi"));
        assert!(r("length_constraints:number_paragraphs", "a\n***\nb") && !r("length_constraints:number_paragraphs", "a\n***\n\n***\nb"));
        assert!(r("detectable_format:multiple_sections", "SECTION 1 a SECTION 2 b") && !r("detectable_format:multiple_sections", "SECTION 1 a"));
        assert!(r("startend:end_checker", "ok. Any other questions?\"") && !r("startend:end_checker", "Any other questions? ok"));
        assert!(r("detectable_format:json_format", "```json\n{\"a\": 1}\n```") && !r("detectable_format:json_format", "{a: 1}"));
        assert!(r("change_case:english_lowercase", "all lower, 123") && !r("change_case:english_lowercase", "Not lower"));
        assert!(r("punctuation:no_comma", "no commas") && !r("punctuation:no_comma", "a, b"));
        assert_eq!(sentences("One. Two! Three? 3.5 is a number"), 4);
    }

    #[test]
    fn tool_calls_match_in_any_order_and_reject_extras() {
        let gt = vec![json!({"spotify.play": {"artist": ["Taylor Swift"], "duration": [20]}}), json!({"spotify.play": {"artist": ["Maroon 5"], "duration": [15]}})];
        let c = |a: &str, d: f64| ("spotify_play".to_string(), json!({"artist": a, "duration": d}));
        assert!(grade_tools(&[c("maroon 5", 15.0), c("Taylor Swift", 20.0)], &gt).is_ok());
        assert!(grade_tools(&[c("Maroon 5", 15.0), c("Maroon 5", 15.0)], &gt).is_err());
        assert!(grade_tools(&[c("Taylor Swift", 20.0)], &gt).is_err());
        let opt = [json!({"area": {"base": [10], "unit": ["units", ""]}})];
        assert!(grade_tools(&[("area".into(), json!({"base": 10}))], &opt).is_ok());
        assert!(grade_tools(&[("area".into(), json!({"base": 10, "color": "red"}))], &opt).is_err());
    }

    #[test]
    fn code_block_prefers_the_language_tag() {
        assert_eq!(code_block("x\n```text\nno\n```\n```rust\nfn a() {}\n```", "rust"), "fn a() {}\n");
        assert_eq!(code_block("```\nplain\n```", "cpp"), "plain\n");
        assert_eq!(code_block("```rust\n/// ```\n/// f();\n/// ```\nfn f() {}\n```", "rust"), "/// ```\n/// f();\n/// ```\nfn f() {}\n");
        assert_eq!(fn_name("/// doc\nfn has_close(numbers: Vec<f64>) -> bool {\n", "rs"), "has_close");
        assert_eq!(fn_name("#include<assert.h>\nbool has_close(std::vector<float> n) {\n", "cpp"), "has_close");
    }
}
