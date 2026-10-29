//! Service::run (serve/server.py's Service._run): one request's events from the engine's tokens - the turn, the
//! parser, stop strings, the thinking budget (#123), a forced tool call, the repeat guard (#606), reasoning-loop
//! recovery (#728), the close-retry (#1053) - and its record in the history and totals.  Python's generator is a push
//! here: `emit` hands each item out, and its Err(Closed) is the consumer leaving (GeneratorExit).

use crate::errors::Fail;
use crate::parser::{Event, GlmParser, HarmonyParser, Kind, OutputParser, Parse, THINK_END};
use crate::py::{self, truthy};
use crate::say;
use crate::service::*;
use crate::sync::{now, Closed, Event as Flag, SemGuard, Stop};
use crate::tokenizer::Detokenizer;
use serde_json::{json, Map, Value};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub enum Item {
    Ping,
    Event(Event),
    Mcp(Value),
    Done(Map<String, Value>),
}

pub type Emit<'a> = &'a mut dyn FnMut(Item) -> Result<(), Closed>;

/// Where this request's status lives: the service's own (one at a time) or its entry among the parallel ones.
#[derive(Clone, Copy)]
enum St {
    Main,
    Par(u64),
}

impl Service {
    fn with_st<R>(&self, st: St, f: impl FnOnce(&mut Map<String, Value>, &mut VecDeque<(f64, i64)>) -> R) -> R {
        let mut s = self.state.lock().unwrap();
        let s = &mut *s;
        match st {
            St::Main => f(&mut s.status, &mut s.rate),
            St::Par(id) => {
                let e = s.live_reqs.entry(id).or_insert_with(|| (Map::new(), VecDeque::new()));
                f(&mut e.0, &mut e.1)
            }
        }
    }

    fn note(&self, n: i64, evs: &[Event], st: St) {
        self.with_st(st, |s, rate| {
            s.insert("generated".into(), json!(n));
            if s.get("first_token").map(Value::is_null).unwrap_or(true) {
                s.insert("first_token".into(), json!(now()));
            }
            if rate.len() == 32 {
                rate.pop_front();
            }
            rate.push_back((now(), n));
            for ev in evs {
                match ev.kind {
                    Kind::Reasoning => {
                        s.insert("phase".into(), json!("thinking"));
                    }
                    Kind::Content => {
                        s.insert("phase".into(), json!("answering"));
                    }
                    Kind::ToolStart => {
                        let name = ev.call.as_ref().map(|c| c.name.clone()).unwrap_or_default();
                        s.insert("phase".into(), json!(format!("writing a tool call: {name}")));
                        s.insert("tool".into(), json!(name));
                    }
                    Kind::ToolCall => {
                        s.insert("phase".into(), json!("tool call complete"));
                    }
                    Kind::ToolArgs => {}
                }
                let tail = format!("{}{}", s.get("tail").and_then(Value::as_str).unwrap_or(""), ev.text);
                let chars: Vec<char> = tail.chars().collect();
                s.insert("tail".into(), json!(chars[chars.len().saturating_sub(600)..].iter().collect::<String>()));
            }
        });
    }

    fn progress(&self, last_print: f64, st: St) -> f64 {
        let t = now();
        if t - last_print < 1.0 {
            return last_print;
        }
        let s = self.with_st(st, |s, _| s.clone());
        let el = t - s.get("started").and_then(Value::as_f64).unwrap_or(t);
        if s.get("first_token").map(Value::is_null).unwrap_or(true) {
            let done = match self.engine.live().progress {
                Some((a, b)) if b != 0 => format!("{} of {}", crate::mcp::thousands(a), crate::mcp::thousands(b)),
                _ => crate::mcp::thousands(s.get("prompt_tokens").and_then(Value::as_i64).unwrap_or(0)),
            };
            say(&format!("reading the prompt: {done} tokens, {el:.0} s so far"));
        } else {
            let generated = s.get("generated").and_then(Value::as_f64).unwrap_or(0.0);
            let rate = generated / (t - s["first_token"].as_f64().unwrap_or(t)).max(1e-6);
            let phase = s.get("phase").and_then(Value::as_str).unwrap_or("");
            let mt = py::py_str(s.get("max_tokens").unwrap_or(&Value::Null));
            let limit = match s.get("thinking_budget").filter(|b| truthy(b)) {
                Some(b) if phase == "thinking" => format!("budget {} (max {mt})", py::py_str(b)),
                _ => format!("max {mt}"),
            };
            say(&format!("{phase}: {} of {limit} tokens, {rate:.1} tok/s, {el:.0} s", generated as i64));
        }
        t
    }

    /// The request's events, then Item::Done({"finish", "completion_tokens", ...}).
    #[allow(clippy::too_many_arguments)]
    pub fn run(&self, ids: &[u32], thinking: bool, tools: Option<&[Value]>, max_new: i64, sampling: &Map<String, Value>, cancel: &Arc<Flag>, force: Option<String>, emit: Emit) -> Result<(), Stop<Fail>> {
        // the config's sampling, then the Chat settings shared with apps; the request's own fields win
        let mut sampling = sampling.clone();
        let shared = self.shared.lock().unwrap().clone();
        if !self.set.sampling_defaults.is_empty() || !shared.is_empty() {
            let mut merged = self.set.sampling_defaults.clone();
            for (k, v) in shared {
                merged.insert(k, v);
            }
            for (k, v) in &sampling {
                if !v.is_null() {
                    merged.insert(k.clone(), v.clone());
                }
            }
            sampling = merged;
        }
        let samp_v = Value::Object(sampling.clone());
        let mut budget = if thinking { self.reasoning_budget(&samp_v).map_err(Stop::Err)? } else { None };
        if let Some(b) = budget {
            if max_new > 0 {
                let reserve = (ANSWER_RESERVE_MIN.max(max_new / 4)).min(max_new / 2);
                if b > max_new - reserve {
                    say(&format!("thinking capped at {} of max_tokens {max_new} to leave room for the answer (budget {b})", max_new - reserve));
                    budget = Some((max_new - reserve).max(1));
                }
            }
        }
        let mut parser: Box<dyn Parse> = if self.harmony {
            Box::new(HarmonyParser::new())
        } else if self.glm {
            Box::new(GlmParser::new(thinking, tools, self.set.tool_call_recovery))
        } else {
            Box::new(OutputParser::new(thinking, tools, true, self.set.tool_call_recovery))
        };
        let mut detok = Detokenizer::default();
        let mut n: i64 = 0;
        let mut finish = "length".to_string();
        let (mut run_tok, mut run_len, mut repeated): (Option<u32>, i64, bool) = (None, 0, false);
        let mut thinking_n: i64 = 0;
        let stop_list = crate::service::stop_strings(&samp_v).map_err(Stop::Err)?;
        let mut stops = if stop_list.is_empty() { None } else { Some(StopMatcher::new(stop_list)) };
        let mut force = force;
        let mut opening: Vec<Event> = vec![];
        if let Some(f) = force.clone() {
            if !thinking {
                opening = parser.feed(&f).map_err(|e| Stop::Err(Fail::Value(e)))?;
                force = None;
            }
        }
        let mut tail = String::new();
        let (mut answered, mut close_retried) = (false, false);
        let mut timings: Option<Value> = None;
        let mut recovery_count: i64 = 0;
        let mut reasoning_text = String::new();
        let mut repeat_coverage = 0.0f64;
        let (mut looped, mut next_loop_check) = (false, LOOP_CHECK_EVERY);
        let mut raw_ids: Vec<u32> = vec![];
        let emb = EMBEDDINGS.with(|e| e.borrow_mut().take());
        let mut segments: Vec<Map<String, Value>> = vec![];
        let trace = trace();
        let waiting = Instant::now();
        let par = self.engine.batch() > 0;
        let rid = self.next_req.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let st = if par { St::Par(rid) } else { St::Main };
        let queued = |d: i64| {
            let mut s = self.state.lock().unwrap();
            let q = s.status.get("queued").and_then(Value::as_i64).unwrap_or(0) + d;
            s.status.insert("queued".into(), json!(q));
        };
        queued(1);

        let cut = |stops: &mut Option<StopMatcher>, evs: Vec<Event>| -> Vec<Event> {
            for c in evs.iter().filter(|e| e.kind == Kind::ToolCall).filter_map(|e| e.call.as_ref()) {
                if !self.mcp.as_ref().is_some_and(|h| h.routes().contains_key(&c.name)) {
                    crate::firewall::model_call(&c.name, &c.arguments); // Strata's own tool calls are noted where they run
                }
            }
            let Some(sm) = stops.as_mut() else { return evs };
            let mut out = vec![];
            for ev in evs {
                if sm.hit.is_some() {
                    break;
                }
                if ev.kind == Kind::Content {
                    let text = sm.push(&ev.text);
                    if !text.is_empty() {
                        out.push(Event::text(Kind::Content, text));
                    }
                    continue;
                }
                let held = sm.flush();
                if !held.is_empty() {
                    out.push(Event::text(Kind::Content, held));
                }
                out.push(ev);
            }
            out
        };

        // ------------------------------------------------------------------------------------ the turn
        let mut turn: Option<SemGuard> = None;
        if !par {
            // #1619: a keep-alive every QUEUE_BEAT_S while waiting for the single turn; a client that left gives it up
            loop {
                if self.fifo.acquire_timeout(Duration::from_secs_f64(QUEUE_BEAT_S)) {
                    turn = Some(SemGuard(&self.fifo));
                    break;
                }
                if cancel.is_set() {
                    break;
                }
                if emit(Item::Ping).is_err() {
                    queued(-1);
                    return Err(Stop::Closed);
                }
            }
            if turn.is_none() {
                queued(-1);
                let done = json!({"finish": "cancel", "completion_tokens": 0, "reused": 0, "timings": null, "reasoning_tokens": 0, "reasoning_recoveries": 0});
                emit(Item::Done(done.as_object().cloned().unwrap()))?;
                return Ok(());
            }
        }
        let body = (|| -> Result<(), Stop<Fail>> {
            {
                let mut s = self.state.lock().unwrap();
                if let Some(t) = &trace {
                    let mut t = t.lock().unwrap();
                    add_f(&mut t, "queue_s", pyround(waiting.elapsed().as_secs_f64(), 3));
                    t.insert("state".into(), json!("generating"));
                }
                if !par {
                    let q = s.status.get("queued").and_then(Value::as_i64).unwrap_or(0) - 1;
                    s.status.insert("queued".into(), json!(q));
                }
            }
            let mut entering = false;
            let load = (|| -> Result<(), Fail> {
                if !par {
                    return self.ensure_loaded();
                }
                loop {
                    {
                        self.state.lock().unwrap().entering += 1;
                        entering = true;
                    }
                    if !self.image_gate.is_set() && self.loaded() && !self.vision.as_ref().map(|v| !v.alive()).unwrap_or(false) {
                        return Ok(());
                    }
                    {
                        self.state.lock().unwrap().entering -= 1;
                        entering = false;
                    }
                    self.fifo.acquire();
                    let _g = SemGuard(&self.fifo);
                    self.ensure_loaded()?;
                }
            })();
            if let Err(e) = load {
                if par {
                    let mut s = self.state.lock().unwrap();
                    let q = s.status.get("queued").and_then(Value::as_i64).unwrap_or(0) - 1;
                    s.status.insert("queued".into(), json!(q));
                    if entering {
                        s.entering -= 1;
                    }
                }
                return Err(Stop::Err(e));
            }
            {
                let mut s = self.state.lock().unwrap();
                if par {
                    let q = s.status.get("queued").and_then(Value::as_i64).unwrap_or(0) - 1;
                    s.status.insert("queued".into(), json!(q));
                    s.entering -= 1;
                }
            }
            self.with_st(st, |s, rate| {
                for (k, v) in [("busy", json!(true)), ("phase", json!("reading the prompt")), ("prompt_tokens", json!(ids.len())), ("generated", json!(0)), ("started", json!(now())), ("first_token", Value::Null), ("tool", Value::Null), ("tail", json!("")), ("max_tokens", json!(max_new)), ("reasoning_recoveries", json!(0)), ("thinking_budget", json!(budget))] {
                    s.insert(k.into(), v);
                }
                rate.clear();
            });
            {
                let mut s = self.state.lock().unwrap();
                if let St::Par(id) = st {
                    let newest = s.live_reqs.get(&id).map(|e| e.0.clone()).unwrap_or_default();
                    for (k, v) in newest {
                        s.status.insert(k, v);
                    }
                }
                s.last_request_at = Some(now());
            }
            let mut last_print = now();
            let mut prompt = ids.to_vec();
            let mut thought: i64 = 0;
            for ev in std::mem::take(&mut opening) {
                emit(Item::Event(ev))?;
            }
            loop {
                let mut g = self.engine.generate(prompt.clone(), max_new - n, sampling.clone(), cancel.clone(), emb.as_ref().map(|p| p.to_string_lossy().into_owned()));
                let mut recover_prompt: Option<Vec<u32>> = None;
                let mut seg: Vec<u32> = vec![];
                let (mut wrap, mut leaving, mut opens) = (false, false, false);
                let mut last_out = Instant::now();
                let tokens = (|| -> Result<(), Stop<Fail>> {
                    loop {
                        let t = match g.tokens.next() {
                            Ok(Some(t)) => t,
                            Ok(None) => return Ok(()),
                            Err(e) => return Err(Stop::Err(e)),
                        };
                        let Some(t) = t else {
                            last_print = self.progress(last_print, st);
                            last_out = Instant::now();
                            emit(Item::Ping)?;
                            continue;
                        };
                        n += 1;
                        if let Some(tr) = &trace {
                            let mut tr = tr.lock().unwrap();
                            if tr.get("first_token_s").map(Value::is_null).unwrap_or(true) {
                                let c = tr.get("_clock").and_then(Value::as_f64).unwrap_or(0.0);
                                tr.insert("first_token_s".into(), json!(pyround(clock() - c, 3)));
                            }
                        }
                        if self.stop_ids.contains(&t) {
                            if force.is_some() && parser.state() == "reasoning" && parser.buf() == THINK_END {
                                opens = true;
                                return Ok(());
                            }
                            finish = "stop".into();
                            raw_ids.push(t);
                            return Ok(());
                        }
                        raw_ids.push(t);
                        seg.push(t);
                        if matches!(parser.state(), "reasoning" | "rcall") {
                            thinking_n += 1;
                        }
                        run_len = if Some(t) == run_tok { run_len + 1 } else { 1 };
                        run_tok = Some(t);
                        if self.set.repeat_stop_tokens != 0 && run_len >= self.set.repeat_stop_tokens {
                            repeated = true;
                            return Ok(());
                        }
                        let piece = detok.push(&self.tok.token_bytes(t));
                        let both = format!("{tail}{piece}");
                        tail = both.chars().rev().take(2).collect::<Vec<_>>().into_iter().rev().collect();
                        let evs = cut(&mut stops, parser.feed(&piece).map_err(|e| Stop::Err(Fail::Value(e)))?);
                        self.note(n, &evs, st);
                        last_print = self.progress(last_print, st);
                        let any = !evs.is_empty();
                        let starts_call = evs.iter().any(|e| matches!(e.kind, Kind::ToolStart | Kind::ToolCall));
                        for ev in evs {
                            if self.set.reasoning_loop_recovery.is_some() && ev.kind == Kind::Reasoning {
                                reasoning_text.push_str(&ev.text);
                            }
                            if matches!(ev.kind, Kind::Content | Kind::ToolStart | Kind::ToolCall) {
                                answered = true;
                            }
                            emit(Item::Event(ev))?;
                        }
                        if any {
                            last_out = Instant::now();
                        } else if last_out.elapsed().as_secs_f64() >= KEEPALIVE_S {
                            last_out = Instant::now();
                            emit(Item::Ping)?;
                        }
                        if stops.as_ref().map(|s| s.hit.is_some()).unwrap_or(false) {
                            finish = "stop".into();
                            return Ok(());
                        }
                        if let Some(mode) = &self.set.reasoning_loop_recovery {
                            if recovery_count == 0 && parser.state() == "reasoning" && n >= next_loop_check && parser.buf().is_empty() && parser.pending() == 0 && !detok.pending() {
                                next_loop_check = n + LOOP_CHECK_EVERY;
                                repeat_coverage = reasoning_repeat_coverage(&reasoning_text);
                                if repeat_coverage >= LOOP_COVERAGE {
                                    if mode == "stop" {
                                        looped = true;
                                        return Ok(());
                                    }
                                    if emb.is_none() {
                                        recover_prompt = focused_recovery_prompt(&*self.tok, ids, &raw_ids);
                                        if recover_prompt.is_some() {
                                            return Ok(());
                                        }
                                    }
                                }
                            }
                        }
                        if let Some(b) = budget {
                            if parser.state() == "reasoning" {
                                thought += 1;
                                if thought >= b && parser.buf().is_empty() && !detok.pending() {
                                    wrap = true;
                                    return Ok(());
                                }
                            }
                        }
                        if force.is_some() {
                            if starts_call {
                                force = None;
                            } else if parser.state() == "content" && parser.buf().is_empty() && !detok.pending() {
                                opens = true;
                                return Ok(());
                            }
                        }
                    }
                })();
                if let Err(e) = &tokens {
                    match e {
                        Stop::Closed => leaving = true,
                        Stop::Err(f) if f.is_died() => {
                            finish = "error".into();
                            self.say_died(f);
                        }
                        Stop::Err(f) => {
                            finish = "error".into();
                            say(&format!("the engine reported an error: {f}"));
                        }
                    }
                }
                // STOP + drain to THIS request's DONE while still holding the turn
                let mut closing: Option<Fail> = None;
                if let Err(e) = g.tokens.close() {
                    if matches!(e, Fail::Silent(_)) {
                        finish = "error".into();
                        self.say_died(&e);
                        if !leaving && !cancel.is_set() {
                            closing = Some(e);
                        }
                    }
                }
                if let Some(s) = g.stats.lock().unwrap().take() {
                    segments.push(s);
                }
                tokens?;
                if let Some(e) = closing {
                    return Err(Stop::Err(e));
                }
                if let Some(rp) = recover_prompt.take() {
                    if !cancel.is_set() && n < max_new {
                        recovery_count += 1;
                        let num = |k: &str| sampling.get(k).and_then(Value::as_f64).unwrap_or(0.0);
                        let want = (num("temperature").max(1.0), num("presence_penalty").max(1.5));
                        if want != (num("temperature"), num("presence_penalty")) {
                            sampling.insert("temperature".into(), json!(want.0));
                            sampling.insert("presence_penalty".into(), json!(want.1));
                        }
                        prompt = rp;
                        self.with_st(st, |s, _| {
                            s.insert("reasoning_recoveries".into(), json!(recovery_count));
                        });
                        if let Some(t) = &trace {
                            let mut t = t.lock().unwrap();
                            t.insert("reasoning_recoveries".into(), json!(recovery_count));
                            t.insert("reasoning_repeat_coverage".into(), json!(pyround(repeat_coverage, 3)));
                        }
                        say(&format!("repeated reasoning detected at {n} tokens (coverage={repeat_coverage:.3}); resuming the same output with the low-effort instruction (reasoning_loop_recovery)"));
                        continue;
                    }
                }
                if self.set.reasoning_close_retry && thinking && finish == "stop" && !close_retried && !answered && !wrap && !opens && parser.state() == "reasoning" && parser.buf() != THINK_END && stops.as_ref().map(|s| s.hit.is_none()).unwrap_or(true) && !cancel.is_set() {
                    close_retried = true;
                    let extra = self.tok.encode(if self.harmony { HARMONY_CLOSE } else { REASONING_CLOSE }, true);
                    if max_new - n - extra.len() as i64 >= 1 {
                        say("the reply ended inside its thinking with no answer: closing the thinking once and continuing (reasoning_close_retry)");
                        for &t in &extra {
                            n += 1;
                            raw_ids.push(t);
                            if matches!(parser.state(), "reasoning" | "rcall") {
                                thinking_n += 1;
                            }
                            let evs = cut(&mut stops, parser.feed(&detok.push(&self.tok.token_bytes(t))).map_err(|e| Stop::Err(Fail::Value(e)))?);
                            self.note(n, &evs, st);
                            for ev in evs {
                                emit(Item::Event(ev))?;
                            }
                        }
                        prompt = [prompt.as_slice(), seg.as_slice(), extra.as_slice()].concat();
                        finish = "length".into();
                        continue;
                    }
                }
                if !(wrap || opens) || cancel.is_set() {
                    break;
                }
                // #123: close the thinking the way the model would and let it answer; a forced call opens the same way
                let text = if wrap {
                    budget = None;
                    self.with_st(st, |s, _| {
                        s.insert("thinking_budget".into(), Value::Null);
                    });
                    format!("{}{}", if self.harmony { harmony_wrap_up() } else { REASONING_WRAP_UP.to_string() }, force.clone().unwrap_or_default())
                } else {
                    let nl = tail.len() - tail.trim_end_matches('\n').len();
                    format!("{}{}", "\n".repeat(2usize.saturating_sub(nl)), force.clone().unwrap_or_default())
                };
                force = None;
                let extra = self.tok.encode(&text, true);
                if max_new - n - (extra.len() as i64) < 1 {
                    break;
                }
                if wrap {
                    say(&format!("thinking budget reached ({thought} tokens): wrapping up the thinking"));
                }
                for &t in &extra {
                    n += 1;
                    raw_ids.push(t);
                    if matches!(parser.state(), "reasoning" | "rcall") {
                        thinking_n += 1;
                    }
                    let evs = cut(&mut stops, parser.feed(&detok.push(&self.tok.token_bytes(t))).map_err(|e| Stop::Err(Fail::Value(e)))?);
                    self.note(n, &evs, st);
                    for ev in evs {
                        emit(Item::Event(ev))?;
                    }
                }
                if stops.as_ref().map(|s| s.hit.is_some()).unwrap_or(false) {
                    finish = "stop".into();
                    break;
                }
                prompt = [prompt.as_slice(), seg.as_slice(), extra.as_slice()].concat();
            }
            if cancel.is_set() {
                finish = "cancel".into();
            } else if looped {
                say(&format!("the thinking repeated the same passages (coverage={repeat_coverage:.3}) at {n} tokens: ended as \"length\" (reasoning_loop_recovery: \"stop\" in strata-<model>.json; remove it to turn this off)"));
            } else if repeated {
                let shown = py::py_repr(&json!(self.tok.decode(&[run_tok.unwrap_or(0)])));
                say(&format!("the reply repeated one token ({shown}) {run_len} times in a row: ended as \"length\" (repeat_stop_tokens in strata-<model>.json; 0 turns this off). If a new request with a short prompt does the same, restart the server and report it (#606)"));
            }
            Ok(())
        })();
        if matches!(body, Err(Stop::Closed)) {
            finish = "disconnect".into();
        }
        // #266: settle this request's status, history and totals while still holding the turn
        self.settle(st, ids.len() as i64, n, &finish, &sampling, &segments, recovery_count, repeat_coverage, cancel, &*parser, &raw_ids, &mut timings);
        drop(turn);
        if let Some(e) = &emb {
            let _ = std::fs::remove_file(e);
        }
        body?;
        let fin = parser.finish(Some(&finish)).map_err(|e| Stop::Err(Fail::Value(e)))?;
        for ev in cut(&mut stops, fin) {
            emit(Item::Event(ev))?;
        }
        if parser.rescued() > 0 || parser.refused() > 0 {
            {
                let mut s = self.state.lock().unwrap();
                let v = s.totals.get("tool_calls_from_reasoning").and_then(Value::as_i64).unwrap_or(0) + parser.rescued() as i64;
                s.totals.insert("tool_calls_from_reasoning".into(), json!(v));
            }
            say(&format!("tool calls inside the thinking: {} read as calls, {} kept as reasoning (quoted, or the turn did not end on a stop)", parser.rescued(), parser.refused()));
        }
        if let Some(sm) = stops.as_mut() {
            if sm.hit.is_none() && !sm.held.is_empty() {
                emit(Item::Event(Event::text(Kind::Content, sm.flush())))?;
            }
        }
        let reused = timings.as_ref().and_then(|t| t.get("cache_n").cloned()).unwrap_or(json!(0));
        let mut done = json!({"finish": finish, "completion_tokens": n, "reused": reused, "timings": timings, "reasoning_tokens": thinking_n, "reasoning_recoveries": recovery_count}).as_object().cloned().unwrap();
        if let Some(h) = stops.as_ref().and_then(|s| s.hit.clone()) {
            done.insert("stop_sequence".into(), json!(h));
        }
        emit(Item::Done(done))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn settle(&self, st: St, prompt_len: i64, n: i64, finish: &str, sampling: &Map<String, Value>, segments: &[Map<String, Value>], recovery_count: i64, repeat_coverage: f64, cancel: &Flag, parser: &dyn Parse, raw_ids: &[u32], timings: &mut Option<Value>) {
        let busy = self.with_st(st, |s, _| truthy(s.get("busy").unwrap_or(&Value::Null)));
        if busy {
            let last = request_stats(segments);
            let s = self.with_st(st, |s, _| s.clone());
            let t = now();
            let started = s.get("started").and_then(Value::as_f64).unwrap_or(t);
            let cvec = self.engine.info().get("cvec").cloned().unwrap_or(json!(0));
            let loaded = !matches!(py::py_str(&cvec).as_str(), "0" | "" | "None");
            let g = |k: &str| last.get(k).cloned().unwrap_or(Value::Null);
            let gi = |k: &str| last.get(k).and_then(Value::as_i64);
            let hit_rate = match (gi("hits"), gi("lookups")) {
                (Some(h), Some(l)) if l != 0 => json!(pyround(h as f64 / l as f64, 3)),
                _ => Value::Null,
            };
            let routed = gi("lookups").unwrap_or(0) + gi("offloaded").unwrap_or(0);
            let pcie_share = match gi("offloaded") {
                Some(o) if routed != 0 => json!(pyround(o as f64 / routed as f64, 3)),
                _ => Value::Null,
            };
            let seen = prompt_tokens_seen(prompt_len, &last);
            let decode_ms = last.get("decode_ms").and_then(Value::as_f64);
            let gen_e = gi("generated");
            let decode_tok_s = match (gen_e, decode_ms) {
                (Some(g2), Some(d)) if n != 0 && g2 != 0 && d != 0.0 => json!(pyround(g2 as f64 / (d / 1000.0), 1)),
                _ => Value::Null,
            };
            let rec = json!({
                "projection": if loaded { json!(sampling.get("experimental_speed_projection") != Some(&Value::Bool(false))) } else { Value::Null },
                "time": started, "duration_s": pyround(t - started, 1), "finish": finish,
                "prompt_tokens": seen, "reused": g("reused"), "output_tokens": n,
                "prompt_total": prompt_len, "prompt_read": g("prompt_read"),
                "reasoning_recoveries": recovery_count, "reasoning_repeat_coverage": pyround(repeat_coverage, 3),
                "engine_generated": g("generated"), "prompt_ms": g("prompt_ms"), "decode_ms": g("decode_ms"),
                "decode_tok_s": decode_tok_s, "hit_rate": hit_rate, "pcie_share": pcie_share,
                "ram_blobs": g("ram_blobs"), "file_blobs": g("file_blobs"), "file_mb": g("file_mb"),
                "drafts_offered": g("drafts_offered"), "drafts_accepted": g("drafts_accepted")});
            {
                let mut stt = self.state.lock().unwrap();
                stt.history.push_back(rec);
                while stt.history.len() > 500 {
                    stt.history.pop_front();
                }
                let tt = &mut stt.totals;
                let addi = |tt: &mut Map<String, Value>, k: &str, d: i64| {
                    let v = tt.get(k).and_then(Value::as_i64).unwrap_or(0) + d;
                    tt.insert(k.into(), json!(v));
                };
                addi(tt, "requests", 1);
                addi(tt, "prompt_tokens", seen);
                addi(tt, "reused", gi("reused").unwrap_or(0));
                addi(tt, "output_tokens", n);
                add_f(tt, "prompt_ms", last.get("prompt_ms").and_then(Value::as_f64).unwrap_or(0.0));
                add_f(tt, "decode_ms", decode_ms.unwrap_or(0.0));
                addi(tt, "drafts_offered", gi("drafts_offered").unwrap_or(0));
                addi(tt, "drafts_accepted", gi("drafts_accepted").unwrap_or(0));
            }
            let ft0 = s.get("first_token").and_then(Value::as_f64);
            self.latencies.observe(ft0.map(|f| f - started), n, decode_ms.unwrap_or(0.0) / 1000.0, now() - started);
            if !segments.is_empty() {
                *timings = request_timings(seen, n, &last);
                self.state.lock().unwrap().last_timings = timings.as_ref().map(|t| {
                    let mut m = t.as_object().cloned().unwrap();
                    m.insert("at".into(), json!(now() as i64));
                    Value::Object(m)
                });
            }
            self.state.lock().unwrap().last_request_at = Some(now());
            let t2 = now();
            let el = t2 - started;
            let rate = ft0.map(|f| n as f64 / (t2 - f).max(1e-6)).unwrap_or(0.0);
            let mut hit_msg = hit_rate.as_f64().map(|h| format!(", expert cache {:.1}% hit", h * 100.0)).unwrap_or_default();
            if let (false, Some(p)) = (hit_msg.is_empty(), pcie_share.as_f64().filter(|p| *p != 0.0)) {
                hit_msg += &format!(" (+{:.1}% of the routed experts over PCIe)", p * 100.0);
            }
            say(&format!("done: {n} tokens in {el:.0} s ({rate:.1} tok/s) ({finish}, cancel={}){hit_msg}", if cancel.is_set() { "True" } else { "False" }));
            if finish == "length" && matches!(parser.state(), "reasoning" | "rcall") {
                say("the reply reached max tokens while still thinking, so it has no answer: a thinking budget (reasoning_budget_tokens, in the request or in strata-<model>.json for every request) leaves room to answer");
            }
            if std::env::var("STRATA_DEBUG").map(|v| !v.is_empty()).unwrap_or(false) && !raw_ids.is_empty() {
                say(&format!("raw: {}", py::py_repr(&json!(self.tok.decode(raw_ids)))));
            }
        }
        self.with_st(st, |s, _| {
            s.insert("busy".into(), json!(false));
            s.shift_remove("tail");
            s.shift_remove("tool");
        });
        if let St::Par(id) = st {
            let mut s = self.state.lock().unwrap();
            s.live_reqs.shift_remove(&id);
            let newest = s.live_reqs.values().last().map(|e| e.0.clone());
            match newest {
                Some(m) => {
                    for (k, v) in m {
                        s.status.insert(k, v);
                    }
                }
                None => {
                    s.status.insert("busy".into(), json!(false));
                    s.status.shift_remove("tail");
                    s.status.shift_remove("tool");
                }
            }
        }
    }
}
