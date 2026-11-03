//! The refusal direction of the model a run config serves, as the file `strata-metal --ablate` reads (docs/ABLATION.md).
//! Arditi et al. 2024 (arXiv:2406.11717): the difference in means of the residual stream between harmful and harmless
//! instructions, at each layer and each post-instruction token; the candidate that, ablated at every layer, lowers the
//! refusal score on held-out harmful instructions most while the next-token KL on harmless ones stays under --kl-max.
//!
//!     strata-ablate CONFIG.json [--train 128] [--val 32] [--max-layer-frac 0.8] [--kl-max 0.1]
//!                   [--data ../Strata-data/ablate] [--out MODEL-refusal.f32]
//!
//! Prompts: mlabonne/harmful_behaviors (AdvBench, MIT) and mlabonne/harmless_alpaca (Alpaca, CC BY-NC 4.0), train for
//! the means, test for the selection; rendered by the model's own chat template (thinking off) and the pack's tokenizer.
//! The model runs in strata-directions (metal/strata_directions.cpp).  Out: the f32 file (the chosen direction at every
//! layer) and FILE.json (the choice and every candidate's scores).
// ponytail: the paper's third filter (adding the direction makes harmless prompts refuse) is left out; bypass + KL decide.

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};
use strata_server::template::ChatTemplate;
use strata_server::tokenizer::from_pack_dir;

/// (dataset, split, rows used, sha-256 of those rows joined by "\n"): the datasets-server API takes no revision
const SETS: [(&str, &str, usize, &str); 4] = [
    ("mlabonne/harmful_behaviors", "train", 416, "fd10323f8069d6252906332f7ac79b72a6902244d8609d27f64751ec7241edb2"),
    ("mlabonne/harmful_behaviors", "test", 104, "e89e86ba849a19855af9ccdac17ed452c0f6a59ec44f2998503cc0bb68139244"),
    ("mlabonne/harmless_alpaca", "train", 600, "75ee87057f4dc528636e2f21e23017b28eef04d890efd3267912ddca89955475"),
    ("mlabonne/harmless_alpaca", "test", 600, "cd4ce9dcf6d6e68d93a7b8309c61ad7295cf95d9e2bd11e9c5a6495aad1adafe"),
];
/// a refusal's first word (the paper's refusal tokens: "I", "As"; Qwen also opens with "Sorry")
const REFUSAL_STARTS: [&str; 4] = ["I", "As", "Sorry", "Unfortunately"];

fn rows(data: &Path, (set, split, n, sha): (&str, &str, usize, &str)) -> Vec<String> {
    let path = data.join(format!("{}-{split}-{n}.json", set.replace('/', "_")));
    if let Ok(b) = fs::read(&path) {
        return serde_json::from_slice(&b).unwrap();
    }
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(120))).build().into();
    let mut out: Vec<String> = vec![];
    while out.len() < n {
        let url = format!("https://datasets-server.huggingface.co/rows?dataset={set}&config=default&split={split}&offset={}&length=100", out.len());
        let v: Value = serde_json::from_reader(agent.get(&url).call().unwrap_or_else(|e| panic!("{url}: {e}")).into_body().into_reader()).unwrap();
        let page = v["rows"].as_array().unwrap();
        out.extend(page.iter().map(|r| r["row"]["text"].as_str().unwrap().to_string()));
        if page.len() < 100 {
            break;
        }
    }
    out.truncate(n);
    let got = hex::encode(Sha256::digest(out.join("\n").as_bytes()));
    assert!(got == sha, "{set} {split}: rows hash {got}, pinned {sha}");
    fs::create_dir_all(data).unwrap();
    fs::write(&path, serde_json::to_vec(&out).unwrap()).unwrap();
    out
}

/// n items spread evenly over xs (the same ones every run)
fn pick(xs: &[String], n: usize) -> Vec<String> {
    let n = n.min(xs.len());
    (0..n).map(|i| xs[i * xs.len() / n].clone()).collect()
}

/// v / |v|, in f32 for the file
fn unit(v: &[f64]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    v.iter().map(|x| (x / norm) as f32).collect()
}

/// log p - log(1 - p), p = the probability of the next token being one that starts a refusal
fn refusal_score(logp: &[f32], refusal: &[u32]) -> f64 {
    let p = refusal.iter().map(|&i| (logp[i as usize] as f64).exp()).sum::<f64>().clamp(1e-12, 1.0 - 1e-12);
    p.ln() - (1.0 - p).ln()
}

/// KL(p || q) of two log-softmax vectors
fn kl(p: &[f32], q: &[f32]) -> f64 {
    p.iter().zip(q).map(|(&a, &b)| (a as f64).exp() * (a as f64 - b as f64)).sum()
}

/// the --ablate file: the direction at each of the model's layers, k = 1
fn layout(dir: &[f32], layers: usize) -> Vec<u8> {
    (0..layers).flat_map(|_| dir.iter().flat_map(|x| x.to_le_bytes())).collect()
}

struct Engine {
    child: Child,
    stdin: ChildStdin,
    out: BufReader<ChildStdout>,
}

impl Engine {
    fn start(exe: &Path, gguf: &str) -> (Engine, usize, usize) {
        let mut child = Command::new(exe).args(["--gguf", gguf]).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null())
            .spawn().unwrap_or_else(|e| panic!("{}: {e} (make build)", exe.display()));
        let (stdin, mut out) = (child.stdin.take().unwrap(), BufReader::new(child.stdout.take().unwrap()));
        let mut line = String::new();
        out.read_line(&mut line).unwrap();
        let w: Vec<&str> = line.split_whitespace().collect();
        assert!(w.first() == Some(&"READY"), "strata-directions did not start: {line:?}");
        (Engine { child, stdin, out }, w[2].parse().unwrap(), w[4].parse().unwrap())
    }

    /// one command; its f32 reply (empty for OK)
    fn call(&mut self, cmd: &str) -> Vec<f32> {
        writeln!(self.stdin, "{cmd}").unwrap();
        let mut line = String::new();
        self.out.read_line(&mut line).unwrap();
        let w: Vec<&str> = line.split_whitespace().collect();
        match w.first() {
            Some(&"OK") => vec![],
            Some(&"ACT") | Some(&"LOGP") => {
                let mut b = vec![0u8; w[1].parse().unwrap()];
                self.out.read_exact(&mut b).unwrap();
                b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
            }
            _ => panic!("strata-directions: {} -> {line:?}", &cmd[..cmd.len().min(40)]),
        }
    }

    fn ids(ids: &[u32]) -> String {
        ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(String::as_str)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cfg_path = PathBuf::from(args.first().filter(|a| !a.starts_with("--")).expect("usage: strata-ablate CONFIG.json [options]"));
    let num = |name: &str, d: f64| arg(&args, name).map(|v| v.parse::<f64>().unwrap_or_else(|_| panic!("{name}: not a number"))).unwrap_or(d);
    let (n_train, n_val) = (num("--train", 128.0) as usize, num("--val", 32.0) as usize);
    let (frac, kl_max) = (num("--max-layer-frac", 0.8), num("--kl-max", 0.1));
    let cfg: Value = serde_json::from_str(&fs::read_to_string(&cfg_path).unwrap()).unwrap();
    let cargs: Vec<&str> = cfg["args"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
    let gguf = cargs.iter().position(|a| *a == "--gguf").map(|i| cargs[i + 1]).expect("the config has no --gguf");
    let data = PathBuf::from(arg(&args, "--data").unwrap_or("../Strata-data/ablate"));
    let out = PathBuf::from(arg(&args, "--out").map(String::from).unwrap_or_else(|| gguf.trim_end_matches(".gguf").to_string() + "-refusal.f32"));

    // the prompts as the server writes them: the model's template, thinking off, its tokenizer
    let tok_dir = PathBuf::from(cfg["tokenizer"].as_str().expect("the config has no tokenizer"));
    let tok = from_pack_dir(&tok_dir).unwrap();
    let tpl_path = Some(tok_dir.join("chat_template.jinja")).filter(|p| p.exists()).unwrap_or_else(|| strata_server::root().join("server/chat_template.jinja"));
    let tpl = ChatTemplate::load(&tpl_path).unwrap();
    let mut kw = Map::new();
    kw.insert("enable_thinking".into(), json!(false));
    let render = |text: &str| tpl.render(&[json!({"role": "user", "content": text})], None, true, &kw).unwrap();
    let probe = render("STRATA_PROBE");
    let tail = &probe[probe.rfind("STRATA_PROBE").unwrap() + "STRATA_PROBE".len()..];
    let n_post = tok.encode(tail, true).len();
    let ids = |text: &str| tok.encode(&render(text), true);
    let refusal: Vec<u32> = REFUSAL_STARTS.iter().map(|s| tok.encode(s, false)[0]).collect();
    eprintln!("post-instruction tokens: {n_post} ({tail:?}); refusal tokens {refusal:?}");

    let [htr, hte, ltr, lte] = SETS.map(|s| rows(&data, s));
    let (harm, harmless) = (pick(&htr, n_train), pick(&ltr, n_train));
    let (harm_val, harmless_val) = (pick(&hte, n_val), pick(&lte, n_val));

    let exe = Path::new(cfg["exe"].as_str().unwrap()).with_file_name("strata-directions");
    let exe = if exe.exists() { exe } else { strata_server::root().join("build-metal/metal/strata-directions") };
    let t0 = Instant::now();
    let (mut eng, layers, n_embd) = Engine::start(&exe, gguf);
    eprintln!("{gguf}: {layers} layers, n_embd {n_embd} ({:.0} s)", t0.elapsed().as_secs_f64());

    // means of the residual stream, [layer][position][embd], each set
    let mut mean = |set: &[String]| -> Vec<f64> {
        let mut sum = vec![0f64; layers * n_post * n_embd];
        for t in set {
            for (s, x) in sum.iter_mut().zip(eng.call(&format!("ACT {n_post} {}", Engine::ids(&ids(t))))) {
                *s += x as f64;
            }
        }
        sum.iter().map(|s| s / set.len() as f64).collect()
    };
    let (mh, ml) = (mean(&harm), mean(&harmless));
    eprintln!("means of {} + {} prompts ({:.0} s)", harm.len(), harmless.len(), t0.elapsed().as_secs_f64());

    // the baseline: refusal scores on the held-out harmful, next-token distributions on the held-out harmless
    let val_h: Vec<String> = harm_val.iter().map(|t| Engine::ids(&ids(t))).collect();
    let val_l: Vec<String> = harmless_val.iter().map(|t| Engine::ids(&ids(t))).collect();
    let base_l: Vec<Vec<f32>> = val_l.iter().map(|p| eng.call(&format!("LOGP {p}"))).collect();
    let base_bypass = val_h.iter().map(|p| refusal_score(&eng.call(&format!("LOGP {p}")), &refusal)).sum::<f64>() / val_h.len() as f64;
    eprintln!("no ablation: refusal score {base_bypass:.2}");

    let tmp = out.with_extension("candidate.f32");
    let mut cands = vec![];
    let mut best: Option<(f64, usize, usize, Vec<f32>)> = None;
    for l in 0..((layers as f64 * frac) as usize) {
        for i in 0..n_post {
            let at = (l * n_post + i) * n_embd;
            let diff: Vec<f64> = (0..n_embd).map(|e| mh[at + e] - ml[at + e]).collect();
            let dir = unit(&diff);
            fs::write(&tmp, layout(&dir, layers)).unwrap();
            eng.call(&format!("ABL {}", tmp.display()));
            let bypass = val_h.iter().map(|p| refusal_score(&eng.call(&format!("LOGP {p}")), &refusal)).sum::<f64>() / val_h.len() as f64;
            let kl_l = val_l.iter().zip(&base_l).map(|(p, b)| kl(b, &eng.call(&format!("LOGP {p}")))).sum::<f64>() / val_l.len() as f64;
            eprintln!("layer {l:2} pos -{}: refusal score {bypass:7.2}  KL {kl_l:.4}", n_post - i);
            cands.push(json!({"layer": l, "pos": i as i64 - n_post as i64, "bypass": bypass, "kl": kl_l}));
            if kl_l < kl_max && best.as_ref().is_none_or(|b| bypass < b.0) {
                best = Some((bypass, l, i, dir));
            }
        }
    }
    eng.call("ABL -");
    let _ = fs::remove_file(&tmp);
    let (bypass, l, i, dir) = best.expect("no candidate kept the KL under --kl-max");
    fs::write(&out, layout(&dir, layers)).unwrap();
    let pos = i as i64 - n_post as i64;
    let report = json!({"model": gguf, "layers": layers, "n_embd": n_embd, "layer": l, "pos": pos, "refusal_score": bypass,
                        "refusal_score_base": base_bypass, "train": n_train, "val": n_val, "kl_max": kl_max, "candidates": cands});
    fs::write(out.with_extension("json"), serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    println!("{}: the direction of layer {l} at token {pos}, refusal score {base_bypass:.2} -> {bypass:.2} ({:.0} s)",
             out.display(), t0.elapsed().as_secs_f64());
    println!("run config: \"args\": [..., \"--ablate\", \"{}\"]", out.display());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_direction_is_a_unit_vector() {
        let u = unit(&[3.0, 4.0]);
        assert!((u[0] - 0.6).abs() < 1e-6 && (u[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn the_refusal_score_is_the_log_odds_of_the_refusal_tokens() {
        let lp = [0.25f32.ln(), 0.25f32.ln(), 0.5f32.ln()];
        assert!(refusal_score(&lp, &[0, 1]).abs() < 1e-6, "p = 0.5: log-odds 0");
        assert!((refusal_score(&lp, &[2]) - 0.0).abs() < 1e-6);
        assert!((refusal_score(&lp, &[0]) - (0.25f64 / 0.75).ln()).abs() < 1e-6);
    }

    #[test]
    fn kl_is_zero_for_the_same_distribution_and_positive_otherwise() {
        let p = [0.5f32.ln(), 0.5f32.ln()];
        let q = [0.9f32.ln(), 0.1f32.ln()];
        assert!(kl(&p, &p).abs() < 1e-9);
        assert!((kl(&p, &q) - (0.5 * (0.5f64 / 0.9).ln() + 0.5 * (0.5f64 / 0.1).ln())).abs() < 1e-6);
    }

    #[test]
    fn the_file_holds_the_direction_once_per_layer() {
        let b = layout(&[1.0, -2.0], 3);
        assert_eq!(b.len(), 3 * 2 * 4);
        assert_eq!(&b[8..12], &1.0f32.to_le_bytes());
        assert_eq!(&b[20..24], &(-2.0f32).to_le_bytes());
    }
}
