//! The pack's byte-level BPE and SentencePiece tokenizers (tools/strata_tokenizer.py, the reference of record) and the
//! byte stand-in the mock engine uses.  Offsets are byte offsets of the text (Python's were character offsets: the same spans).

use aho_corasick::{AhoCorasick, MatchKind};
use anyhow::{anyhow, bail, Context, Result};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::path::Path;
use std::sync::Mutex;

// The pre-tokenizer patterns, as strata_tokenizer.py ships them (see its notes for where each comes from).
const QWEN35: &str = concat!(
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])",
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+",
    r"|\p{N}",
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+"
);
const GPT4O: &str = concat!(
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+"
);
const GLM4: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
    r"|[^\r\n\p{L}\p{N}]?\p{L}+",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+"
);
const HUNYUAN: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
    r"|[^\r\n\p{L}\p{N}]?\p{L}+",
    r"|\p{N}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+"
);

/// pre -> (pattern, ignore_merges: a piece that is a whole token is that token, as tiktoken does)
fn pre_of(pre: &str) -> Option<(&'static str, bool)> {
    Some(match pre {
        "qwen35" => (QWEN35, false),
        "gpt-4o" => (GPT4O, true),
        "glm4" => (GLM4, true),
        "lfm2" => (GLM4, true), // LiquidAI's LFM2 / LFM2.5: their tokenizer.json's pattern is glm4's
        "qwen2" => (HUNYUAN, false), // Qwen2 / Qwen3 / Qwen3-Coder: hunyuan's pattern, no ignore_merges
        "hunyuan" => (HUNYUAN, true),
        _ => return None,
    })
}

/// The GPT-2 byte encoder: 256 bytes -> 256 printable characters, reversibly.
fn bytes_to_unicode() -> [char; 256] {
    let mut bs: Vec<u32> = (0x21..0x7f).chain(0xa1..0xad).chain(0xae..0x100).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut out = ['\0'; 256];
    for (b, c) in bs.iter().zip(cs) {
        out[*b as usize] = char::from_u32(c).expect("char");
    }
    out
}

/// What the server needs of a tokenizer (the pack's BPE, or the mock's bytes).
pub trait Tokenize: Send + Sync {
    /// `parse_special` matches the CONTROL literals (`<|im_end|>`); USER_DEFINED ones (`<think>`) match always.
    /// `plain`: byte spans of `text` that stay ordinary text even where they spell a literal (#537).
    fn encode_plain(&self, text: &str, parse_special: bool, plain: &[(usize, usize)]) -> Vec<u32>;
    fn encode(&self, text: &str, parse_special: bool) -> Vec<u32> {
        self.encode_plain(text, parse_special, &[])
    }
    /// The raw bytes of one token (a multi-byte character can be split across tokens).
    fn token_bytes(&self, id: u32) -> Vec<u8>;
    fn decode(&self, ids: &[u32]) -> String {
        let raw: Vec<u8> = ids.iter().flat_map(|&i| self.token_bytes(i)).collect();
        String::from_utf8_lossy(&raw).into_owned()
    }
    /// The CONTROL literals (type 3), in vocabulary order.
    fn control_tokens(&self) -> Vec<String>;
    fn vocab_size(&self) -> usize;
}

pub struct Tokenizer {
    pub tokens: Vec<String>,
    pub pre: String,
    ids: HashMap<String, u32>,
    /// (left, right) -> (rank, merged id)
    merges: HashMap<(u32, u32), (u32, u32)>,
    re: fancy_regex::Regex,
    ignore_merges: bool,
    special: HashMap<String, u32>,
    control: Vec<String>,
    always_re: Option<(AhoCorasick, Vec<u32>)>,
    special_re: Option<(AhoCorasick, Vec<u32>)>,
    byte_ids: [Option<u32>; 256],
    byte_of: HashMap<char, u8>,
    piece_ids: Mutex<HashMap<String, Vec<u32>>>,
}

const PIECE_CACHE_MAX: usize = 200_000;

fn alt(literals: &[(&String, u32)]) -> Option<(AhoCorasick, Vec<u32>)> {
    if literals.is_empty() {
        return None;
    }
    // longest literal first, then leftmost-first: the regex alternation the reference builds
    let mut v: Vec<(&String, u32)> = literals.to_vec();
    v.sort_by_key(|(s, _)| Reverse(s.chars().count()));
    let ac = AhoCorasick::builder().match_kind(MatchKind::LeftmostFirst).build(v.iter().map(|(s, _)| s.as_str())).ok()?;
    Some((ac, v.iter().map(|(_, i)| *i).collect()))
}

impl Tokenizer {
    pub fn new(tokens: Vec<String>, merges: &[String], types: Option<&[i64]>, pre: &str) -> Result<Self> {
        let mut ids = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            ids.insert(t.clone(), i as u32);
        }
        if ids.len() != tokens.len() {
            bail!("vocabulary has duplicate tokens: {} entries, {} unique", tokens.len(), ids.len());
        }
        let mut mm = HashMap::with_capacity(merges.len());
        for (i, m) in merges.iter().enumerate() {
            let parts: Vec<&str> = m.split(' ').collect();
            if parts.len() != 2 {
                bail!("merge {i} is not a pair: {m:?}");
            }
            let (Some(&l), Some(&r)) = (ids.get(parts[0]), ids.get(parts[1])) else {
                bail!("merge {i} names a token outside the vocabulary: {m:?}");
            };
            let merged = format!("{}{}", parts[0], parts[1]);
            // a merge whose result is no token can never be applied by the reference either (it raises then)
            let id = ids.get(&merged).copied().unwrap_or(u32::MAX);
            mm.insert((l, r), (i as u32, id));
        }
        let (pat, ignore_merges) = pre_of(pre).ok_or_else(|| anyhow!("no pre-tokenizer for {pre:?} (known: qwen35, qwen2, gpt-4o, glm4, lfm2, hunyuan)"))?;
        let re = fancy_regex::Regex::new(pat).context("pre-tokenizer pattern")?;
        let mut special = HashMap::new();
        let mut always = vec![];
        let mut control = vec![];
        let mut both = vec![];
        if let Some(types) = types {
            for (i, &ty) in types.iter().enumerate() {
                if ty == 3 || ty == 4 {
                    special.insert(tokens[i].clone(), i as u32);
                }
            }
            for (i, &ty) in types.iter().enumerate() {
                if ty == 3 || ty == 4 {
                    both.push((&tokens[i], i as u32));
                }
                if ty == 4 {
                    always.push((&tokens[i], i as u32));
                }
                if ty == 3 {
                    control.push(tokens[i].clone());
                }
            }
        }
        let b2u = bytes_to_unicode();
        let mut byte_ids = [None; 256];
        let mut byte_of = HashMap::new();
        for (b, c) in b2u.iter().enumerate() {
            byte_ids[b] = ids.get(&c.to_string()).copied();
            byte_of.insert(*c, b as u8);
        }
        let always_re = alt(&always);
        let special_re = alt(&both);
        Ok(Tokenizer {
            pre: pre.to_string(),
            ids,
            merges: mm,
            re,
            ignore_merges,
            special,
            control,
            always_re,
            special_re,
            byte_ids,
            byte_of,
            piece_ids: Mutex::new(HashMap::new()),
            tokens,
        })
    }

    /// The pack's `tokenizer/` directory: vocab.json, merges.txt, token_type.json, tokenizer.json (`pre`).
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let vocab: HashMap<String, usize> =
            serde_json::from_str(&std::fs::read_to_string(dir.join("vocab.json")).context("vocab.json")?)?;
        let mut tokens = vec![String::new(); vocab.len()];
        for (t, i) in vocab {
            *tokens.get_mut(i).ok_or_else(|| anyhow!("vocab id {i} out of range"))? = t;
        }
        let merges: Vec<String> =
            std::fs::read_to_string(dir.join("merges.txt")).context("merges.txt")?.split('\n').map(String::from).collect();
        let types: Vec<i64> = serde_json::from_str(&std::fs::read_to_string(dir.join("token_type.json")).context("token_type.json")?)?;
        let pre = std::fs::read_to_string(dir.join("tokenizer.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|v| v["pre"].as_str().map(String::from))
            .unwrap_or_else(|| "qwen35".into());
        Self::new(tokens, &merges, Some(&types), &pre)
    }

    /// A Hugging Face tokenizer.json (byte-level BPE); its added tokens are CONTROL (type 3).
    pub fn from_hf_json(path: &Path, pre: &str) -> Result<Self> {
        let t: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).context("tokenizer.json")?)?;
        let m = &t["model"];
        if m["type"] != "BPE" || m["byte_fallback"].as_bool().unwrap_or(false) {
            bail!("expected a byte-level BPE tokenizer.json, got {}", m["type"]);
        }
        let vocab = m["vocab"].as_object().ok_or_else(|| anyhow!("no vocab"))?;
        let added = t["added_tokens"].as_array().cloned().unwrap_or_default();
        let n = vocab.values().filter_map(|v| v.as_u64()).chain(added.iter().filter_map(|a| a["id"].as_u64())).max().unwrap_or(0) as usize + 1;
        let mut tokens = vec![String::new(); n];
        let mut types = vec![1i64; n];
        for (tok, i) in vocab {
            tokens[i.as_u64().unwrap_or(0) as usize] = tok.clone();
        }
        for a in &added {
            let i = a["id"].as_u64().unwrap_or(0) as usize;
            tokens[i] = a["content"].as_str().unwrap_or("").to_string();
            types[i] = 3;
        }
        if tokens.iter().any(|t| t.is_empty()) {
            bail!("tokenizer.json leaves ids without a token");
        }
        let merges: Vec<String> = m["merges"].as_array().cloned().unwrap_or_default().iter().map(|x| match x {
            serde_json::Value::String(s) => s.clone(),
            v => v.as_array().map(|a| a.iter().filter_map(|y| y.as_str()).collect::<Vec<_>>().join(" ")).unwrap_or_default(),
        }).collect();
        Self::new(tokens, &merges, Some(&types), pre)
    }

    pub fn id_of(&self, token: &str) -> Option<u32> {
        self.ids.get(token).copied()
    }

    /// Merge the piece's symbols by lowest rank first (leftmost among equals) - _bpe_heap's order, which is _bpe's.
    fn bpe(&self, piece: &[u8]) -> Result<Vec<u32>> {
        let mut parts: Vec<u32> = Vec::with_capacity(piece.len());
        for &b in piece {
            parts.push(self.byte_ids[b as usize].ok_or_else(|| anyhow!("byte {b:#x} has no token"))?);
        }
        let n = parts.len();
        if n < 2 {
            return Ok(parts);
        }
        let mut alive = vec![true; n];
        let mut nxt: Vec<isize> = (1..=n as isize).collect();
        nxt[n - 1] = -1;
        let mut prv: Vec<isize> = (-1..n as isize - 1).collect();
        // (rank, position, left, right): popped lowest rank, then lowest position
        let mut heap = BinaryHeap::new();
        for i in 0..n - 1 {
            if let Some(&(r, _)) = self.merges.get(&(parts[i], parts[i + 1])) {
                heap.push(Reverse((r, i, parts[i], parts[i + 1])));
            }
        }
        while let Some(Reverse((_, i, left, right))) = heap.pop() {
            let j = nxt[i];
            if !alive[i] || parts[i] != left || j < 0 || parts[j as usize] != right {
                continue; // stale: one of its symbols has merged since
            }
            let j = j as usize;
            let merged = self.merges[&(left, right)].1;
            if merged == u32::MAX {
                bail!("BPE produced a token outside the vocabulary");
            }
            parts[i] = merged;
            alive[j] = false;
            let k = nxt[j];
            nxt[i] = k;
            if k >= 0 {
                let k = k as usize;
                prv[k] = i as isize;
                if let Some(&(r2, _)) = self.merges.get(&(parts[i], parts[k])) {
                    heap.push(Reverse((r2, i, parts[i], parts[k])));
                }
            }
            let p = prv[i];
            if p >= 0 {
                let p = p as usize;
                if let Some(&(r2, _)) = self.merges.get(&(parts[p], parts[i])) {
                    heap.push(Reverse((r2, p, parts[p], parts[i])));
                }
            }
        }
        Ok(parts.into_iter().zip(alive).filter(|(_, a)| *a).map(|(p, _)| p).collect())
    }

    fn encode_piece(&self, piece: &str, out: &mut Vec<u32>) {
        if let Some(got) = self.piece_ids.lock().unwrap().get(piece) {
            out.extend_from_slice(got);
            return;
        }
        let whole = if self.ignore_merges {
            let mapped: String = piece.bytes().map(|b| BYTE_TO_UNICODE.with(|t| t[b as usize])).collect();
            self.ids.get(&mapped).copied()
        } else {
            None
        };
        let got = match whole {
            Some(id) => vec![id],
            // ponytail: a piece whose merge lands outside the vocabulary encodes as its bytes (the reference raises)
            None => self.bpe(piece.as_bytes()).unwrap_or_else(|_| piece.bytes().filter_map(|b| self.byte_ids[b as usize]).collect()),
        };
        out.extend_from_slice(&got);
        let mut cache = self.piece_ids.lock().unwrap();
        if cache.len() < PIECE_CACHE_MAX {
            cache.insert(piece.to_string(), got);
        }
    }

    fn encode_ordinary(&self, text: &str, out: &mut Vec<u32>) {
        for m in self.re.find_iter(text) {
            match m {
                Ok(m) => self.encode_piece(m.as_str(), out),
                // backtrack limit: the rest as one piece, never a lost character
                Err(_) => {
                    self.encode_piece(text, out);
                    return;
                }
            }
        }
    }
}

thread_local! {
    static BYTE_TO_UNICODE: [char; 256] = bytes_to_unicode();
}

impl Tokenize for Tokenizer {
    fn encode_plain(&self, text: &str, parse_special: bool, plain: &[(usize, usize)]) -> Vec<u32> {
        let mut out = Vec::new();
        let Some((ac, ids)) = (if parse_special { &self.special_re } else { &self.always_re }) else {
            self.encode_ordinary(text, &mut out);
            return out;
        };
        let mut spans: Vec<(usize, usize)> = plain.to_vec();
        spans.sort();
        let mut upcoming = spans.into_iter().peekable();
        let mut plain_end = 0;
        let mut pos = 0;
        for m in ac.find_iter(text) {
            let start = m.start();
            while let Some(&(a, b)) = upcoming.peek() {
                if a > start {
                    break;
                }
                plain_end = plain_end.max(b);
                upcoming.next();
            }
            if start < plain_end {
                continue;
            }
            if start > pos {
                self.encode_ordinary(&text[pos..start], &mut out);
            }
            out.push(ids[m.pattern().as_usize()]);
            pos = m.end();
        }
        if pos < text.len() {
            self.encode_ordinary(&text[pos..], &mut out);
        }
        out
    }

    fn token_bytes(&self, id: u32) -> Vec<u8> {
        self.tokens.get(id as usize).map(|t| t.chars().filter_map(|c| self.byte_of.get(&c).copied()).collect()).unwrap_or_default()
    }

    fn control_tokens(&self) -> Vec<String> {
        self.control.clone()
    }

    fn vocab_size(&self) -> usize {
        self.tokens.len()
    }
}

impl Tokenizer {
    pub fn special_id(&self, s: &str) -> Option<u32> {
        self.special.get(s).copied()
    }
}

/// The pack's `tokenizer/` directory as whichever tokenizer its tokenizer.json names: `llama` (SentencePiece) or BPE.
pub fn from_pack_dir(dir: &Path) -> Result<Box<dyn Tokenize>> {
    let cfg: serde_json::Value =
        std::fs::read_to_string(dir.join("tokenizer.json")).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
    Ok(if cfg["model"] == "llama" { Box::new(SpmTokenizer::from_dir(dir)?) } else { Box::new(Tokenizer::from_dir(dir)?) })
}

const SPACE: char = '\u{2581}'; // SentencePiece's visible space

/// SentencePiece as llama.cpp runs it (llm_tokenizer_spm_session): characters merged pairwise, highest score first
/// (leftmost among equals), while the pair spells a token; a final symbol that is no token becomes its `<0xXX>` bytes.
/// A text fragment at the start or right after a special token gets one leading space (add_space_prefix).
pub struct SpmTokenizer {
    tokens: Vec<String>,
    scores: Vec<f64>,
    types: Vec<i64>,
    ids: HashMap<String, u32>,
    add_space_prefix: bool,
    byte_ids: [u32; 256],
    control: Vec<String>,
    always_re: Option<(AhoCorasick, Vec<u32>)>,
    special_re: Option<(AhoCorasick, Vec<u32>)>,
}

/// A candidate merge: popped highest score first, then leftmost (llm_bigram_spm::comparator).
#[derive(PartialEq)]
struct Bigram {
    score: f64,
    left: usize,
    right: usize,
    size: usize,
}

impl Eq for Bigram {}

impl Ord for Bigram {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        // partial_cmp, not total_cmp: -0.0 == 0.0 as C++'s `<` has it
        self.score.partial_cmp(&o.score).unwrap_or(std::cmp::Ordering::Equal).then(o.left.cmp(&self.left))
    }
}

impl PartialOrd for Bigram {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}

impl SpmTokenizer {
    pub fn new(tokens: Vec<String>, scores: Vec<f64>, types: Vec<i64>, add_space_prefix: bool, unk: u32) -> Result<Self> {
        if scores.len() != tokens.len() || types.len() != tokens.len() {
            bail!("{} tokens, {} scores, {} token types", tokens.len(), scores.len(), types.len());
        }
        let ids: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        if ids.len() != tokens.len() {
            bail!("vocabulary has duplicate tokens: {} entries, {} unique", tokens.len(), ids.len());
        }
        let mut byte_ids = [unk; 256];
        for (b, id) in byte_ids.iter_mut().enumerate() {
            if let Some(&i) = ids.get(&format!("<0x{b:02X}>")).or_else(|| ids.get(&(b as u8 as char).to_string())) {
                *id = i;
            }
        }
        // llama.cpp's special cache: CONTROL (3) and UNKNOWN (2) need parse_special, USER_DEFINED (4) always matches
        let both: Vec<(&String, u32)> = (0..tokens.len()).filter(|&i| matches!(types[i], 2..=4)).map(|i| (&tokens[i], i as u32)).collect();
        let always: Vec<(&String, u32)> = both.iter().copied().filter(|&(_, i)| types[i as usize] == 4).collect();
        let control = (0..tokens.len()).filter(|&i| types[i] == 3).map(|i| tokens[i].clone()).collect();
        let (always_re, special_re) = (alt(&always), alt(&both));
        Ok(SpmTokenizer { tokens, scores, types, ids, add_space_prefix, byte_ids, control, always_re, special_re })
    }

    /// The pack's `tokenizer/` directory: vocab.json, scores.json, token_type.json, tokenizer.json (add_space_prefix).
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let vocab: HashMap<String, usize> =
            serde_json::from_str(&std::fs::read_to_string(dir.join("vocab.json")).context("vocab.json")?)?;
        let mut tokens = vec![String::new(); vocab.len()];
        for (t, i) in vocab {
            *tokens.get_mut(i).ok_or_else(|| anyhow!("vocab id {i} out of range"))? = t;
        }
        let scores = serde_json::from_str(&std::fs::read_to_string(dir.join("scores.json")).context("scores.json")?)?;
        let types = serde_json::from_str(&std::fs::read_to_string(dir.join("token_type.json")).context("token_type.json")?)?;
        let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer.json")).context("tokenizer.json")?)?;
        let unk = cfg["special_ids"]["tokenizer.ggml.unknown_token_id"].as_u64().unwrap_or(0) as u32;
        Self::new(tokens, scores, types, cfg["add_space_prefix"].as_bool().unwrap_or(true), unk)
    }

    /// One text fragment (spaces already U+2581), as llm_tokenizer_spm_session::tokenize.
    fn spm(&self, text: &str, out: &mut Vec<u32>) {
        let start: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
        let n = start.len();
        if n == 0 {
            return;
        }
        let mut size: Vec<usize> = (0..n).map(|i| start.get(i + 1).unwrap_or(&text.len()) - start[i]).collect(); // bytes; 0 = merged away
        let mut nxt: Vec<isize> = (1..=n as isize).collect();
        nxt[n - 1] = -1;
        let mut prv: Vec<isize> = (-1..n as isize - 1).collect();
        let mut heap = BinaryHeap::new();
        let add = |heap: &mut BinaryHeap<Bigram>, size: &[usize], l: isize, r: isize| {
            if l < 0 || r < 0 {
                return;
            }
            let (l, r) = (l as usize, r as usize);
            if let Some(&i) = self.ids.get(&text[start[l]..start[l] + size[l] + size[r]]) {
                heap.push(Bigram { score: self.scores[i as usize], left: l, right: r, size: size[l] + size[r] });
            }
        };
        for i in 1..n {
            add(&mut heap, &size, i as isize - 1, i as isize);
        }
        while let Some(b) = heap.pop() {
            if size[b.left] == 0 || size[b.right] == 0 || size[b.left] + size[b.right] != b.size {
                continue; // stale: one of its symbols has merged since
            }
            size[b.left] += size[b.right];
            size[b.right] = 0;
            nxt[b.left] = nxt[b.right];
            if nxt[b.right] >= 0 {
                prv[nxt[b.right] as usize] = b.left as isize;
            }
            add(&mut heap, &size, prv[b.left], b.left as isize);
            add(&mut heap, &size, b.left as isize, nxt[b.left]);
        }
        let mut i = 0isize;
        while i >= 0 {
            let piece = &text[start[i as usize]..start[i as usize] + size[i as usize]];
            // a merged symbol always spells a token, so llama.cpp's rev_merge split-back never runs: bytes it is
            match self.ids.get(piece) {
                Some(&id) => out.push(id),
                None => out.extend(piece.bytes().map(|b| self.byte_ids[b as usize])),
            }
            i = nxt[i as usize];
        }
    }

    fn ordinary(&self, text: &str, prev_special: bool, out: &mut Vec<u32>) {
        let prefix = if self.add_space_prefix && prev_special { " " } else { "" };
        self.spm(&format!("{prefix}{text}").replace(' ', &SPACE.to_string()), out);
    }
}

impl Tokenize for SpmTokenizer {
    fn encode_plain(&self, text: &str, parse_special: bool, plain: &[(usize, usize)]) -> Vec<u32> {
        let mut out = Vec::new();
        let (mut pos, mut prev_special) = (0, true);
        if let Some((ac, ids)) = if parse_special { &self.special_re } else { &self.always_re } {
            for m in ac.find_iter(text) {
                if plain.iter().any(|&(a, b)| a <= m.start() && m.start() < b) {
                    continue;
                }
                if m.start() > pos {
                    self.ordinary(&text[pos..m.start()], prev_special, &mut out);
                }
                out.push(ids[m.pattern().as_usize()]);
                (pos, prev_special) = (m.end(), true);
            }
        }
        if pos < text.len() {
            self.ordinary(&text[pos..], prev_special, &mut out);
        }
        out
    }

    fn token_bytes(&self, id: u32) -> Vec<u8> {
        let Some(t) = self.tokens.get(id as usize) else { return vec![] };
        match self.types[id as usize] {
            6 => u8::from_str_radix(t.get(3..5).unwrap_or(""), 16).map(|b| vec![b]).unwrap_or_default(), // BYTE: <0xXX>
            1 => t.replace(SPACE, " ").into_bytes(),
            _ => t.clone().into_bytes(),
        }
    }

    fn control_tokens(&self) -> Vec<String> {
        self.control.clone()
    }

    fn vocab_size(&self) -> usize {
        self.tokens.len()
    }
}

/// Tiny stand-in tokenizer for tests without the pack: one id per UTF-8 byte, specials as ids >= 256.
pub struct ByteTokenizer;

pub const BYTE_SPECIALS: [&str; 6] =
    ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"];

impl Tokenize for ByteTokenizer {
    fn encode_plain(&self, text: &str, parse_special: bool, plain: &[(usize, usize)]) -> Vec<u32> {
        let mut out = vec![];
        let mut i = 0;
        'outer: while i < text.len() {
            if parse_special && !plain.iter().any(|&(a, b)| a <= i && i < b) {
                for (k, s) in BYTE_SPECIALS.iter().enumerate() {
                    if text[i..].starts_with(s) {
                        out.push(256 + k as u32);
                        i += s.len();
                        continue 'outer;
                    }
                }
            }
            let c = text[i..].chars().next().expect("char");
            out.extend(c.to_string().bytes().map(u32::from));
            i += c.len_utf8();
        }
        out
    }

    fn token_bytes(&self, id: u32) -> Vec<u8> {
        if id >= crate::engine::CHUNK_BASE {
            crate::engine::chunk_bytes(id) // a piece of text Apple's model streamed (engine::AfmEngine)
        } else if id >= 256 {
            BYTE_SPECIALS.get(id as usize - 256).map(|s| s.as_bytes().to_vec()).unwrap_or_default()
        } else {
            vec![id as u8]
        }
    }

    fn control_tokens(&self) -> Vec<String> {
        BYTE_SPECIALS.iter().map(|s| s.to_string()).collect()
    }

    fn vocab_size(&self) -> usize {
        256 + BYTE_SPECIALS.len()
    }
}

/// Incremental decode: each token's bytes through an incremental UTF-8 decoder that emits the complete characters,
/// holds a character split across tokens, and turns invalid bytes into U+FFFD as a whole lossy decode does.
#[derive(Default)]
pub struct Detokenizer {
    held: Vec<u8>,
}

impl Detokenizer {
    pub fn pending(&self) -> bool {
        !self.held.is_empty()
    }

    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.held.extend_from_slice(bytes);
        let mut out = String::new();
        let mut rest: &[u8] = &self.held;
        loop {
            match std::str::from_utf8(rest) {
                Ok(s) => {
                    out.push_str(s);
                    rest = &[];
                    break;
                }
                Err(e) => {
                    let ok = e.valid_up_to();
                    out.push_str(std::str::from_utf8(&rest[..ok]).expect("valid"));
                    match e.error_len() {
                        Some(n) => {
                            out.push('\u{fffd}');
                            rest = &rest[ok + n..];
                        }
                        None => {
                            rest = &rest[ok..]; // an incomplete character at the end: held for the next token
                            break;
                        }
                    }
                }
            }
        }
        self.held = rest.to_vec();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detokenizer_holds_a_split_character_and_replaces_bad_bytes() {
        let mut d = Detokenizer::default();
        let e = "é".as_bytes();
        assert_eq!(d.push(&e[..1]), "");
        assert!(d.pending());
        assert_eq!(d.push(&e[1..]), "é");
        assert_eq!(d.push(&[0xff, b'a']), "\u{fffd}a");
    }

    #[test]
    fn sentencepiece_merges_by_score_with_a_space_prefix_and_byte_fallback() {
        let toks = ["<unk>", "<s>", "</s>", "<0x0A>", "\u{2581}", "a", "b", "\u{2581}a", "ab"];
        let scores = vec![0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -2.0, -3.0];
        let types = vec![2, 3, 3, 6, 1, 1, 1, 1, 1];
        let t = SpmTokenizer::new(toks.iter().map(|s| s.to_string()).collect(), scores, types, true, 0).unwrap();
        // "\u{2581}a" (-2) outscores "ab" (-3): taken first, so "ab" never forms
        assert_eq!(t.encode("ab", false), vec![7, 6]);
        assert_eq!(t.decode(&[7, 6]), " ab");
        // a special token, then a fragment that gets its own space prefix; "\n" is no token: its byte
        assert_eq!(t.encode("<s>\n", true), vec![1, 4, 3]);
        assert_eq!(t.decode(&[4, 3]), " \n");
        // without parse_special "<s>" is text: bytes with no <0xXX> token are <unk>
        assert_eq!(t.encode("<s>", false), vec![4, 0, 0, 0]);
    }

    #[test]
    fn byte_tokenizer_round_trips() {
        let t = ByteTokenizer;
        let ids = t.encode("a<|im_end|>é", true);
        assert_eq!(ids, vec![97, 257, 0xc3, 0xa9]);
        assert_eq!(t.decode(&ids), "a<|im_end|>é");
        assert_eq!(t.encode("<|im_end|>", false).len(), 10);
    }
}
