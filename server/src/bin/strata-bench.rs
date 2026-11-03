//! Agents x context x conversation-cache matrix: decode and prefill tok/s of N concurrent agents.
//!
//!     strata-bench strata-qwen3.6-35b.json [--agents 3,4] [--ctx 16384,32768,65536,131072,262144]
//!                  [--cache-gb 0,8,16,32] [--depth 8000] [--out bench/results/DATE-agents-matrix] [--restore CONFIG]
//!
//! Each cell restarts the config's server (--replace on its port) with --batch = agents, --max-context = ctx (per
//! slot) and --conversation-cache-mib = cache.  Every agent then runs, all agents in lockstep:
//!   1. cold: its own ~depth-token prompt (repository files), prefilled from zero;
//!   2. side: a short second conversation per agent, which takes every slot away from the big ones;
//!   3. warm: the big conversation again plus one turn: restored from the conversation cache, or re-prefilled at 0.
//! Rows go to results.jsonl and a table on stdout.  At the end the original config is started again on its port.
// ponytail: fixed depth, not a fill of the whole window; pass --depth to go deeper (prefill time grows with it).

#[path = "common/server.rs"]
mod server;

use serde_json::{Value, json};
use server::{get_json, start, stop, wait_loaded};
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Child;
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

const MAX_TOKENS: u64 = 128;

#[derive(Default, Clone, Debug)]
struct Turn {
    prompt: u64,
    cached: u64,
    out: u64,
    ttft: f64,
    decode_tps: f64,
    err: Option<String>,
}

fn arg_list(args: &[String], name: &str, default: &str) -> Vec<u64> {
    let v = args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(String::as_str).unwrap_or(default);
    v.split(',').map(|x| x.trim().parse().unwrap_or_else(|_| panic!("{name}: not a number: {x}"))).collect()
}

fn set_arg(args: &mut Vec<Value>, name: &str, val: String) {
    match args.iter().position(|a| a == name) {
        Some(i) => args[i + 1] = json!(val),
        None => args.extend([json!(name), json!(val)]),
    }
}

/// Text from the repository's own sources, enough for `chars`.
fn corpus(root: &Path, chars: usize) -> String {
    let mut files = vec![];
    for dir in ["src", "docs", "metal", "server/src"] {
        let mut stack = vec![root.join(dir)];
        while let Some(d) = stack.pop() {
            let Ok(rd) = fs::read_dir(&d) else { continue };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if matches!(p.extension().and_then(|x| x.to_str()), Some("cpp" | "hpp" | "rs" | "md" | "metal")) {
                    files.push(p);
                }
            }
        }
    }
    files.sort();
    let mut s = String::new();
    for p in files {
        if s.len() >= chars {
            break;
        }
        if let Ok(t) = fs::read_to_string(&p) {
            s += &format!("\n=== {} ===\n{t}", p.strip_prefix(root).unwrap_or(&p).display());
        }
    }
    assert!(s.len() >= chars, "repository text is only {} chars, {chars} needed", s.len());
    s.chars().take(chars).collect()
}

/// One streamed chat turn: TTFT and decode tok/s measured here, token counts from the server's usage.
fn chat(base: &str, key: &str, model: &str, messages: &Value) -> Turn {
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(3600))).http_status_as_error(false).build().into();
    let body = json!({"model": model, "messages": messages, "max_tokens": MAX_TOKENS, "stream": true,
                      "stream_options": {"include_usage": true}});
    let t0 = Instant::now();
    let r = match agent.post(format!("{base}/v1/chat/completions")).header("Authorization", format!("Bearer {key}"))
        .header("Content-Type", "application/json").send(body.to_string().as_bytes()) {
        Ok(r) => r,
        Err(e) => return Turn { err: Some(e.to_string()), ..Default::default() },
    };
    if r.status().as_u16() >= 400 {
        let mut d = String::new();
        let _ = r.into_body().into_reader().take(400).read_to_string(&mut d);
        return Turn { err: Some(format!("HTTP {d}")), ..Default::default() };
    }
    let (mut first, mut last, mut t) = (None, t0, Turn::default());
    for line in BufReader::new(r.into_body().into_reader()).lines() {
        let Ok(line) = line else { break };
        let Some(d) = line.strip_prefix("data: ") else { continue };
        if d == "[DONE]" {
            break;
        }
        let Ok(v) = serde_json::from_str::<Value>(d) else { continue };
        let delta = &v["choices"][0]["delta"];
        let has = |k: &str| delta[k].as_str().is_some_and(|s| !s.is_empty());
        if has("content") || has("reasoning_content") || has("reasoning") {
            first.get_or_insert_with(Instant::now);
            last = Instant::now();
        }
        if let Some(u) = v.get("usage").filter(|u| u.is_object()) {
            t.prompt = u["prompt_tokens"].as_u64().unwrap_or(0);
            t.out = u["completion_tokens"].as_u64().unwrap_or(0);
            t.cached = u["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0);
        }
    }
    match first {
        Some(f) => {
            t.ttft = (f - t0).as_secs_f64();
            let dt = (last - f).as_secs_f64();
            t.decode_tps = if dt > 0.0 && t.out > 1 { (t.out - 1) as f64 / dt } else { 0.0 };
        }
        None => t.err = Some("no tokens".into()),
    }
    t
}

fn sum_tps(ts: &[Turn]) -> f64 {
    ts.iter().map(|t| t.decode_tps).sum()
}
fn mean(ts: &[Turn], f: impl Fn(&Turn) -> f64) -> f64 {
    ts.iter().map(f).sum::<f64>() / ts.len().max(1) as f64
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cfg_path = PathBuf::from(args.first().filter(|a| !a.starts_with("--")).expect("usage: strata-bench CONFIG.json [options]"));
    let base_cfg: Value = serde_json::from_str(&fs::read_to_string(&cfg_path).unwrap()).unwrap();
    let agents_l = arg_list(&args, "--agents", "3,4");
    let ctx_l = arg_list(&args, "--ctx", "16384,32768,65536,131072,262144");
    let cache_l = arg_list(&args, "--cache-gb", "0,8,16,32");
    let depth = arg_list(&args, "--depth", "8000")[0];
    let root = PathBuf::from(base_cfg["cwd"].as_str().unwrap());
    let out = args.iter().position(|a| a == "--out").map(|i| PathBuf::from(&args[i + 1]))
        .unwrap_or_else(|| root.join(format!("bench/results/{}-agents-matrix", chrono::Local::now().format("%Y-%m-%d"))));
    fs::create_dir_all(&out).unwrap();
    let server = Path::new(base_cfg["exe"].as_str().unwrap()).with_file_name("strata-server");
    let (port, key, model) = (base_cfg["port"].as_u64().unwrap_or(8090), base_cfg["api_key"].as_str().unwrap_or("").to_string(),
                              base_cfg["model_name"].as_str().unwrap_or("").to_string());
    let base = format!("http://127.0.0.1:{port}");
    // ~3 chars per token for code; the real count comes back in usage.prompt_tokens
    let big_text = Arc::new(corpus(&root, depth as usize * 3));
    let side_text = Arc::new(big_text.chars().rev().take(1500 * 3).collect::<String>());
    let mut rows = fs::OpenOptions::new().create(true).append(true).open(out.join("results.jsonl")).unwrap();

    println!("{:>6} {:>7} {:>5} | {:>6} {:>8} {:>8} | {:>8} {:>8} {:>7} {:>8} | {:>6} {:>6}  {}",
             "agents", "ctx", "cache", "load_s", "pf_tok/s", "cold_tps", "warm_ttft", "warm_tps", "cached%", "side_tps", "gpu_GB", "wall_s", "err");
    let mut prev: Option<Child> = None;
    for &a in &agents_l {
        for &ctx in &ctx_l {
            for &gb in &cache_l {
                let mut cfg = base_cfg.clone();
                let cargs = cfg["args"].as_array_mut().unwrap();
                set_arg(cargs, "--max-context", ctx.to_string());
                set_arg(cargs, "--conversation-cache-mib", (gb * 1024).to_string());
                cfg["parallel"] = json!(a);
                cfg["log"] = json!(out.join("engine.log"));
                let cell = format!("a{a}-c{}k-g{gb}", ctx / 1024);
                let cpath = out.join("config.json");
                fs::write(&cpath, serde_json::to_string_pretty(&cfg).unwrap()).unwrap();
                // stop our previous server ourselves: --replace only finds one that already holds the port
                if let Some(c) = prev.take() {
                    stop(c);
                }
                let mut child = start(&server, &cpath, port, &out.join(format!("server-{cell}.log")), &root);
                let mut row = json!({"agents": a, "ctx": ctx, "cache_gb": gb, "depth": depth});
                let t_cell = Instant::now();
                match wait_loaded(&mut child, &base, &key) {
                    Err(e) => row["error"] = json!(e),
                    Ok(load) => {
                        row["load_s"] = json!(load);
                        let bar = Arc::new(Barrier::new(a as usize));
                        let hs: Vec<_> = (0..a).map(|i| {
                            let (bar, big, side, base, key, model) = (bar.clone(), big_text.clone(), side_text.clone(), base.clone(), key.clone(), model.clone());
                            let tag = format!("{cell} agent {i} {}", uuid::Uuid::new_v4());
                            std::thread::spawn(move || {
                                let mut msgs = json!([{"role": "system", "content": format!("Session {tag}. You are a coding agent.")},
                                    {"role": "user", "content": format!("Read this code and explain what it does.\n{big}")}]);
                                let cold = chat(&base, &key, &model, &msgs);
                                bar.wait();
                                let side = chat(&base, &key, &model, &json!([{"role": "system", "content": format!("Side {tag}.")},
                                    {"role": "user", "content": format!("Summarize:\n{side}")}]));
                                bar.wait();
                                let m = msgs.as_array_mut().unwrap();
                                m.push(json!({"role": "assistant", "content": "I have read the code."}));
                                m.push(json!({"role": "user", "content": "Now list its main functions."}));
                                let warm = chat(&base, &key, &model, &msgs);
                                (cold, side, warm)
                            })
                        }).collect();
                        let res: Vec<_> = hs.into_iter().map(|h| h.join().unwrap()).collect();
                        let (cold, side, warm): (Vec<Turn>, Vec<Turn>, Vec<Turn>) =
                            res.into_iter().fold((vec![], vec![], vec![]), |mut acc, (c, s, w)| { acc.0.push(c); acc.1.push(s); acc.2.push(w); acc });
                        let errs: Vec<_> = cold.iter().chain(&side).chain(&warm).filter_map(|t| t.err.clone()).collect();
                        // prefill tok/s of the cold round: every agent's prompt over the slowest first token
                        let pf = cold.iter().map(|t| t.prompt as f64).sum::<f64>() / cold.iter().map(|t| t.ttft).fold(0.0, f64::max).max(1e-9);
                        let warm_prompt: u64 = warm.iter().map(|t| t.prompt).sum();
                        let gpu = get_json(&format!("{base}/metrics"), &key).and_then(|m| m["hardware"]["gpu_mem_used"].as_f64()).unwrap_or(0.0) / 1e9;
                        row.as_object_mut().unwrap().extend([
                            ("prompt_tokens".into(), json!(mean(&cold, |t| t.prompt as f64))),
                            ("prefill_tok_s".into(), json!(pf)),
                            ("cold_ttft_s".into(), json!(mean(&cold, |t| t.ttft))),
                            ("cold_decode_tok_s".into(), json!(sum_tps(&cold))),
                            ("side_decode_tok_s".into(), json!(sum_tps(&side))),
                            ("warm_ttft_s".into(), json!(mean(&warm, |t| t.ttft))),
                            ("warm_decode_tok_s".into(), json!(sum_tps(&warm))),
                            ("warm_cached_pct".into(), json!(100.0 * warm.iter().map(|t| t.cached).sum::<u64>() as f64 / warm_prompt.max(1) as f64)),
                            ("per_agent_decode_tok_s".into(), json!(mean(&cold, |t| t.decode_tps))),
                            ("gpu_mem_gb".into(), json!(gpu)),
                        ]);
                        if !errs.is_empty() {
                            row["error"] = json!(errs.join("; ").chars().take(300).collect::<String>());
                        }
                    }
                }
                row["wall_s"] = json!(t_cell.elapsed().as_secs_f64());
                prev = Some(child);
                writeln!(rows, "{row}").unwrap();
                let f = |k: &str| row[k].as_f64().unwrap_or(f64::NAN);
                println!("{a:>6} {:>6}k {:>4}G | {:>6.1} {:>8.0} {:>8.1} | {:>8.2}s {:>8.1} {:>6.0}% {:>8.1} | {:>6.1} {:>6.0}  {}",
                         ctx / 1024, gb, f("load_s"), f("prefill_tok_s"), f("cold_decode_tok_s"), f("warm_ttft_s"), f("warm_decode_tok_s"),
                         f("warm_cached_pct"), f("side_decode_tok_s"), f("gpu_mem_gb"), f("wall_s"), row["error"].as_str().unwrap_or(""));
            }
        }
    }
    // put the original model back on its port, detached: it outlives this process
    if let Some(c) = prev.take() {
        stop(c);
    }
    let restore = args.iter().position(|a| a == "--restore").map(|i| PathBuf::from(&args[i + 1])).unwrap_or(cfg_path);
    let _ = start(&server, &restore, port, &out.join("server-restore.log"), &root);
    println!("results: {}\nrestored {} on :{port}", out.join("results.jsonl").display(), restore.display());
}
