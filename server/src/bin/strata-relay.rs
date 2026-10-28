//! strata-relay: the engine's line protocol over TCP, for a server that runs where the engine cannot (a Linux
//! container on a Mac has no Metal).  The server starts `strata-relay --serve <args>` as if it were the engine; the
//! relay hands the arguments to `strata-relay listen` on the host, which starts the real engine with them and pipes
//! its stdin/stdout through, its stderr into the server's log file (the same path on both sides: the folder is shared).
//!
//!     host:      strata-relay listen --engine engine-metal/strata-metal [--addr 127.0.0.1:7071]
//!     container: STRATA_ENGINE_RELAY=host.docker.internal:7071 strata-server ...   (the server starts the client)
//!
//! Both sides read the shared secret from STRATA_RELAY_TOKEN.  The host starts only the engine named on its own command
//! line, one at a time; a client gives arguments, never a program.  The engine ends as it does under the server: QUIT,
//! or EOF when the connection closes.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn token() -> String {
    std::env::var("STRATA_RELAY_TOKEN").unwrap_or_default()
}

fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    let mut d = (a.len() != b.len()) as u8;
    for i in 0..a.len().max(b.len()) {
        d |= a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0);
    }
    d == 0
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("listen") => listen(&args[1..]),
        Some("--serve") => client(&args),
        _ => {
            eprintln!("usage: strata-relay listen --engine PATH [--addr 127.0.0.1:7071]   (host)\n       strata-relay --serve <engine args>   (started by the server; STRATA_ENGINE_RELAY=host:port)");
            std::process::exit(2);
        }
    }
}

fn client(args: &[String]) {
    let addr = std::env::var("STRATA_ENGINE_RELAY").unwrap_or_default();
    let mut sock = match TcpStream::connect(&addr) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("strata-relay: cannot reach the engine relay at {addr}: {e} (start it on the host: make -C server relay)");
            std::process::exit(1);
        }
    };
    // the server's log (our stderr) is the engine's log too: the host appends the engine's stderr to the same file
    let log = std::fs::read_link("/proc/self/fd/2").ok().map(|p| p.to_string_lossy().into_owned()).filter(|p| p.starts_with('/'));
    let cwd = std::env::current_dir().ok().map(|p| p.to_string_lossy().into_owned());
    let hello = serde_json::json!({"token": token(), "args": args, "cwd": cwd, "log": log});
    if writeln!(sock, "{hello}").is_err() {
        std::process::exit(1);
    }
    let mut up = sock.try_clone().expect("socket");
    std::thread::spawn(move || {
        let _ = std::io::copy(&mut std::io::stdin().lock(), &mut up);
        let _ = up.shutdown(std::net::Shutdown::Write); // the server closed the engine's stdin: EOF is QUIT
    });
    let mut out = std::io::stdout().lock();
    let mut buf = [0u8; 65536];
    loop {
        match sock.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                if out.write_all(&buf[..n]).and_then(|_| out.flush()).is_err() {
                    break;
                }
            }
        }
    }
}

fn listen(args: &[String]) {
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let Some(engine) = opt("--engine") else {
        eprintln!("strata-relay listen: --engine PATH (the engine to start, e.g. engine-metal/strata-metal)");
        std::process::exit(2);
    };
    let engine = std::path::absolute(&engine).unwrap_or_else(|_| engine.clone().into());
    let addr = opt("--addr").unwrap_or_else(|| "127.0.0.1:7071".into());
    let secret = token();
    if secret.len() < 16 {
        eprintln!("strata-relay listen: set STRATA_RELAY_TOKEN (16+ characters; server/.env has one)");
        std::process::exit(2);
    }
    let l = TcpListener::bind(&addr).unwrap_or_else(|e| {
        eprintln!("strata-relay: cannot listen on {addr}: {e}");
        std::process::exit(1)
    });
    println!("strata-relay: {} for containers at {addr}", engine.display());
    let busy = Arc::new(AtomicBool::new(false));
    for conn in l.incoming().flatten() {
        let (busy, secret, engine) = (busy.clone(), secret.clone(), engine.clone());
        std::thread::spawn(move || session(conn, &busy, &secret, &engine));
    }
}

fn session(conn: TcpStream, busy: &AtomicBool, secret: &str, engine: &std::path::Path) {
    let peer = conn.peer_addr().map(|a| a.to_string()).unwrap_or_default();
    let mut rd = BufReader::new(conn.try_clone().expect("socket"));
    let mut line = String::new();
    if rd.read_line(&mut line).is_err() {
        return;
    }
    let mut w = conn;
    let hello: serde_json::Value = serde_json::from_str(&line).unwrap_or_default();
    if !ct_eq(hello["token"].as_str().unwrap_or("").as_bytes(), secret.as_bytes()) {
        let _ = writeln!(w, "ERR strata-relay: wrong token");
        println!("strata-relay: refused {peer}: wrong token");
        return;
    }
    if busy.swap(true, Ordering::SeqCst) {
        let _ = writeln!(w, "ERR strata-relay: an engine already runs through this relay");
        return;
    }
    let args: Vec<String> = hello["args"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default();
    let mut cmd = Command::new(engine);
    cmd.args(&args).stdin(Stdio::piped()).stdout(Stdio::piped());
    match hello["log"].as_str().and_then(|p| std::fs::OpenOptions::new().create(true).append(true).open(p).ok()) {
        Some(f) => cmd.stderr(Stdio::from(f)),
        None => cmd.stderr(Stdio::inherit()),
    };
    if let Some(cwd) = hello["cwd"].as_str().filter(|c| std::path::Path::new(c).is_dir()) {
        cmd.current_dir(cwd);
    }
    println!("strata-relay: {peer} starts {} {}", engine.display(), args.join(" "));
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            let _ = writeln!(w, "ERR strata-relay: the engine could not start: {e}");
            busy.store(false, Ordering::SeqCst);
            return;
        }
    };
    let mut stdin = child.stdin.take().expect("stdin");
    let mut stdout = child.stdout.take().expect("stdout");
    let mut w2 = w.try_clone().expect("socket");
    let down = std::thread::spawn(move || {
        let _ = std::io::copy(&mut stdout, &mut w2);
        let _ = w2.shutdown(std::net::Shutdown::Both);
    });
    let _ = std::io::copy(&mut rd, &mut stdin); // until the server's side closes: then the engine reads EOF (QUIT)
    drop(stdin);
    let _ = down.join();
    let status = child.wait();
    let _ = w.shutdown(std::net::Shutdown::Both);
    println!("strata-relay: the engine for {peer} ended ({})", status.map(|s| s.to_string()).unwrap_or_default());
    busy.store(false, Ordering::SeqCst);
}
