//! A router in front of Strata servers with different contexts (docs/CONTEXT_POOL.md, option B): each request goes to
//! the smallest backend whose context holds its prompt and answer, so a short agent does not hold a long agent's room.
//!
//!     strata-router --port 8100 --backend http://127.0.0.1:8101=32768 --backend http://127.0.0.1:8102=65536 ...
//!
//! The prompt's size is estimated from its text (3 bytes a token: generous for English and code, so a request is
//! never sent where it would not fit); the answer's room is its max_tokens (4096 when unset).  Anything that is not a
//! generation (models, health, metrics, the web page) goes to the largest backend.  Responses stream through as they
//! come.
// ponytail: no stickiness - a conversation that outgrows its backend moves to a larger one and is read there again;
// the backends' state handoff (EXPORT/IMPORT) could carry it over.

use axum::body::{Body, Bytes};
use axum::extract::Request;
use axum::http::{HeaderName, HeaderValue, StatusCode};
use axum::response::Response;
use serde_json::Value;
use std::io::Read;
use std::sync::Arc;
use std::time::Duration;

const BYTES_PER_TOKEN: u64 = 3;
const DEFAULT_ANSWER: u64 = 4096;

/// Tokens a request needs: its text's bytes / 3 (every string in it: messages, system, tools) plus its answer's room
fn needed(body: &Value) -> u64 {
    fn bytes(v: &Value) -> u64 {
        match v {
            Value::String(s) => s.len() as u64,
            Value::Array(a) => a.iter().map(bytes).sum(),
            Value::Object(o) => o.iter().filter(|(k, _)| !matches!(k.as_str(), "max_tokens" | "max_completion_tokens" | "max_output_tokens")).map(|(_, v)| bytes(v)).sum(),
            _ => 0,
        }
    }
    let answer = ["max_tokens", "max_completion_tokens", "max_output_tokens"].iter().find_map(|k| body[*k].as_u64()).unwrap_or(DEFAULT_ANSWER);
    bytes(body) / BYTES_PER_TOKEN + answer
}

/// The smallest backend that holds `need` tokens; the largest when none does (it answers why it cannot)
fn pick(contexts: &[u64], need: u64) -> usize {
    let largest = (0..contexts.len()).max_by_key(|&i| contexts[i]).unwrap_or(0);
    (0..contexts.len()).filter(|&i| contexts[i] >= need).min_by_key(|&i| contexts[i]).unwrap_or(largest)
}

struct Router {
    backends: Vec<(String, u64)>,
    agent: ureq::Agent,
}

async fn route(router: Arc<Router>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let Ok(body) = axum::body::to_bytes(body, 256 << 20).await else {
        return Response::builder().status(StatusCode::PAYLOAD_TOO_LARGE).body(Body::empty()).unwrap();
    };
    let path = parts.uri.path_and_query().map(|p| p.as_str().to_string()).unwrap_or_else(|| "/".into());
    let contexts: Vec<u64> = router.backends.iter().map(|b| b.1).collect();
    let generation = parts.method == axum::http::Method::POST && ["/v1/chat/completions", "/v1/messages", "/v1/completions", "/v1/responses"].iter().any(|p| path.starts_with(p));
    let i = if generation {
        serde_json::from_slice::<Value>(&body).map(|v| pick(&contexts, needed(&v))).unwrap_or_else(|_| pick(&contexts, u64::MAX))
    } else {
        pick(&contexts, u64::MAX)
    };
    let url = format!("{}{path}", router.backends[i].0);
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(32);
    let (head_tx, head_rx) = tokio::sync::oneshot::channel();
    let agent = router.agent.clone();
    let method = parts.method.clone();
    let headers = parts.headers.clone();
    // ureq is blocking: the upstream call and its body run on a blocking thread, the body handed over chunk by chunk
    tokio::task::spawn_blocking(move || {
        let mut b = ureq::http::Request::builder().method(method.as_str()).uri(&url);
        for (k, v) in headers.iter().filter(|(k, _)| !matches!(k.as_str(), "host" | "content-length" | "connection" | "transfer-encoding")) {
            b = b.header(k.as_str(), v.as_bytes());
        }
        let r = match b.body(body.to_vec()).map_err(|e| e.to_string()).and_then(|r| agent.run(r).map_err(|e| e.to_string())) {
            Ok(r) => r,
            Err(e) => {
                let _ = head_tx.send(Err(e));
                return;
            }
        };
        let status = r.status().as_u16();
        let hs: Vec<(String, Vec<u8>)> = r.headers().iter().map(|(k, v)| (k.as_str().to_string(), v.as_bytes().to_vec())).collect();
        let _ = head_tx.send(Ok((status, hs)));
        let mut rd = r.into_body().into_reader();
        let mut buf = vec![0u8; 16384];
        loop {
            match rd.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    if tx.blocking_send(Ok(Bytes::copy_from_slice(&buf[..n]))).is_err() {
                        break; // the client went away
                    }
                }
                Err(e) => {
                    let _ = tx.blocking_send(Err(e));
                    break;
                }
            }
        }
    });
    match head_rx.await {
        Ok(Ok((status, hs))) => {
            let mut resp = Response::builder().status(status);
            for (k, v) in hs.iter().filter(|(k, _)| !matches!(k.as_str(), "content-length" | "transfer-encoding" | "connection")) {
                if let (Ok(k), Ok(v)) = (HeaderName::try_from(k.as_str()), HeaderValue::from_bytes(v)) {
                    resp = resp.header(k, v);
                }
            }
            resp.header("x-strata-backend", router.backends[i].0.as_str()).body(Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx))).unwrap()
        }
        failed => {
            let why = match failed { Ok(Err(e)) => e, _ => "no answer".into() };
            let msg = serde_json::json!({"error": {"type": "server_error", "message": format!("backend {}: {why}", router.backends[i].0)}});
            Response::builder().status(StatusCode::BAD_GATEWAY).header("content-type", "application/json").body(Body::from(msg.to_string())).unwrap()
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let port: u16 = args.iter().position(|a| a == "--port").and_then(|i| args.get(i + 1)).and_then(|p| p.parse().ok()).unwrap_or(8100);
    let backends: Vec<(String, u64)> = args.iter().enumerate().filter(|(_, a)| *a == "--backend").filter_map(|(i, _)| args.get(i + 1))
        .map(|b| {
            let (u, c) = b.rsplit_once('=').expect("--backend URL=CONTEXT");
            (u.trim_end_matches('/').to_string(), c.parse().expect("--backend URL=CONTEXT: a number of tokens"))
        })
        .collect();
    assert!(!backends.is_empty(), "usage: strata-router --port 8100 --backend http://127.0.0.1:8101=32768 [--backend ...]");
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(3600))).http_status_as_error(false).build().into();
    let router = Arc::new(Router { backends, agent });
    println!("strata-router on :{port}: {}", router.backends.iter().map(|(u, c)| format!("{u} ({c})")).collect::<Vec<_>>().join(", "));
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await.expect("port");
        let app = axum::Router::new().fallback(move |req: Request| route(router.clone(), req));
        axum::serve(listener, app).await.unwrap();
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_request_needs_its_text_and_its_answer() {
        let r = json!({"model": "m", "max_tokens": 100, "messages": [{"role": "user", "content": "x".repeat(3000)}]});
        assert_eq!(needed(&r), (3000 + 1 + 4) / 3 + 100); // every string (the text, "m", "user") and the answer
        assert_eq!(needed(&json!({"messages": []})), DEFAULT_ANSWER);
    }

    #[test]
    fn the_smallest_backend_that_fits_takes_it() {
        let c = [32768, 65536, 131072];
        assert_eq!(pick(&c, 20_000), 0);
        assert_eq!(pick(&c, 40_000), 1);
        assert_eq!(pick(&c, 100_000), 2);
        assert_eq!(pick(&c, 500_000), 2); // none fits: the largest says why
        assert_eq!(pick(&[65536, 32768], 1000), 1);
    }
}
