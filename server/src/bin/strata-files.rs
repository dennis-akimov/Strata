//! strata-files: a read-only filesystem MCP server over stdio, in place of `npx @modelcontextprotocol/server-filesystem`
//! (no Node, no npm registry call at every start).  Same tool names, input schemas and answers as that server's
//! 1.0.0 for its read tools (read_text_file, read_file, read_multiple_files, list_directory, directory_tree,
//! search_files, get_file_info, list_allowed_directories); it has no write tools at all.
//!
//!     strata-files DIR [DIR ...]      (a config: "files": {"command": ".../engine-metal/strata-files", "args": [DIR]})
//!
//! Access: every path is made absolute (~ expanded; a relative one against the allowed directories), checked against
//! the allowed directories component by component, then resolved (symlinks) and checked again, so a link inside an
//! allowed directory that points out of it is refused.  Stdio framing: one JSON-RPC 2.0 message per line (MCP stdio).

use chrono::{DateTime, Local};
use globset::{Glob, GlobBuilder, GlobMatcher};
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Component, Path, PathBuf};

const NAME: &str = "strata-files";
const PROTOCOLS: [&str; 3] = ["2025-06-18", "2025-03-26", "2024-11-05"];

fn expand_home(p: &str) -> PathBuf {
    match p.strip_prefix('~') {
        Some(rest) if rest.is_empty() || rest.starts_with('/') => PathBuf::from(std::env::var("HOME").unwrap_or_default() + rest),
        _ => PathBuf::from(p),
    }
}

/// path.resolve: absolute, `.` and `..` folded without touching the disk.
fn lexical(p: &Path) -> PathBuf {
    let base = if p.is_absolute() { PathBuf::new() } else { std::env::current_dir().unwrap_or_default() };
    let mut out = PathBuf::from("/");
    for c in base.join(p).components() {
        match c {
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(n) => out.push(n),
            _ => {}
        }
    }
    out
}

struct Fs {
    allowed: Vec<PathBuf>,
}

impl Fs {
    /// The directories from the command line, each also in its resolved form when that differs (/tmp and /private/tmp).
    fn new(args: &[String]) -> Fs {
        let mut allowed = vec![];
        for a in args {
            let abs = lexical(&expand_home(a));
            let real = std::fs::canonicalize(&abs).unwrap_or_else(|_| abs.clone());
            for d in if real != abs { vec![abs, real] } else { vec![real] } {
                if d.is_dir() {
                    allowed.push(d);
                } else {
                    eprintln!("Warning: Cannot access directory {}, skipping", d.display());
                }
            }
        }
        Fs { allowed }
    }

    fn inside(&self, p: &Path) -> bool {
        !p.as_os_str().as_encoded_bytes().contains(&0) && self.allowed.iter().any(|d| p.starts_with(d))
    }

    fn dirs(&self) -> String {
        self.allowed.iter().map(|d| d.display().to_string()).collect::<Vec<_>>().join(", ")
    }

    fn validate(&self, requested: &str) -> Result<PathBuf, String> {
        let expanded = expand_home(requested);
        let abs = if expanded.is_absolute() {
            lexical(&expanded)
        } else {
            // a relative path: the first allowed directory it exists under, else the first one
            self.allowed.iter().map(|d| lexical(&d.join(&expanded))).find(|c| c.exists()).unwrap_or_else(|| lexical(&self.allowed.first().cloned().unwrap_or_default().join(&expanded)))
        };
        if !self.inside(&abs) {
            return Err(format!("Access denied - path outside allowed directories: {} not in {}", abs.display(), self.dirs()));
        }
        match std::fs::canonicalize(&abs) {
            Ok(real) if self.inside(&real) => Ok(real),
            Ok(real) => Err(format!("Access denied - symlink target outside allowed directories: {} not in {}", real.display(), self.dirs())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(format!("ENOENT: no such file or directory, open '{}'", abs.display())),
            Err(e) => Err(e.to_string()),
        }
    }

    fn call(&self, tool: &str, a: &Value) -> Result<String, String> {
        let s = |k: &str| a[k].as_str().ok_or_else(|| format!("Invalid arguments: \"{k}\" must be a string"));
        let n = |k: &str| a[k].as_f64().filter(|v| *v > 0.0).map(|v| v as usize);
        let list = |k: &str| a[k].as_array().map(|v| v.iter().filter_map(Value::as_str).map(String::from).collect::<Vec<_>>()).unwrap_or_default();
        match tool {
            "read_text_file" | "read_file" => {
                let p = self.validate(s("path")?)?;
                let (head, tail) = (n("head"), n("tail"));
                if head.is_some() && tail.is_some() {
                    return Err("Cannot specify both head and tail parameters simultaneously".into());
                }
                // ponytail: reads the whole file even for head/tail; stream it if huge logs become a use
                let text = read_text(&p)?;
                Ok(match (head, tail) {
                    (_, Some(t)) => {
                        let text = text.replace("\r\n", "\n");
                        let lines: Vec<&str> = text.split('\n').collect();
                        lines[lines.len().saturating_sub(t)..].join("\n")
                    }
                    (Some(h), _) => {
                        let mut lines: Vec<&str> = text.split('\n').collect();
                        if lines.last() == Some(&"") {
                            lines.pop();
                        }
                        lines.into_iter().take(h).collect::<Vec<_>>().join("\n")
                    }
                    _ => text,
                })
            }
            "read_multiple_files" => {
                let paths = list("paths");
                if paths.is_empty() {
                    return Err("Invalid arguments: \"paths\" must hold at least one path".into());
                }
                Ok(paths
                    .iter()
                    .map(|f| match self.validate(f).and_then(|p| read_text(&p)) {
                        Ok(c) => format!("{f}:\n{c}\n"),
                        Err(e) => format!("{f}: Error - {e}"),
                    })
                    .collect::<Vec<_>>()
                    .join("\n---\n"))
            }
            "list_directory" => {
                let p = self.validate(s("path")?)?;
                Ok(entries(&p)?.iter().map(|(name, dir)| format!("{} {name}", if *dir { "[DIR]" } else { "[FILE]" })).collect::<Vec<_>>().join("\n"))
            }
            "directory_tree" => {
                let root = s("path")?.to_string();
                let ex = list("excludePatterns");
                let ex: Vec<Vec<GlobMatcher>> = ex
                    .iter()
                    .map(|p| if p.contains('*') { vec![p.clone()] } else { vec![p.clone(), format!("**/{p}"), format!("**/{p}/**")] })
                    .map(|ps| ps.iter().filter_map(|p| glob(p)).collect())
                    .collect();
                let tree = self.tree(Path::new(&root), Path::new(&root), &ex)?;
                Ok(serde_json::to_string_pretty(&tree).unwrap_or_default())
            }
            "search_files" => {
                let root = self.validate(s("path")?)?;
                let pat = glob(s("pattern")?).ok_or("Invalid glob pattern")?;
                let ex: Vec<GlobMatcher> = list("excludePatterns").iter().filter_map(|p| glob(p)).collect();
                let mut out = vec![];
                self.search(&root, &root, &pat, &ex, &mut out)?;
                Ok(if out.is_empty() { "No matches found".into() } else { out.join("\n") })
            }
            "get_file_info" => {
                let p = self.validate(s("path")?)?;
                let m = std::fs::metadata(&p).map_err(|e| e.to_string())?;
                let t = |s: i64, ns: i64| date(DateTime::from_timestamp(s, ns as u32).map(|d| d.with_timezone(&Local)));
                Ok([
                    format!("size: {}", m.len()),
                    format!("created: {}", date(m.created().ok().map(DateTime::<Local>::from))),
                    format!("modified: {}", t(m.mtime(), m.mtime_nsec())),
                    format!("accessed: {}", t(m.atime(), m.atime_nsec())),
                    format!("isDirectory: {}", m.is_dir()),
                    format!("isFile: {}", m.is_file()),
                    format!("permissions: {:o}", m.permissions().mode() & 0o777),
                ]
                .join("\n"))
            }
            "list_allowed_directories" => Ok(format!("Allowed directories:\n{}", self.allowed.iter().map(|d| d.display().to_string()).collect::<Vec<_>>().join("\n"))),
            _ => Err(format!("Tool {tool} not found")),
        }
    }

    fn tree(&self, root: &Path, cur: &Path, ex: &[Vec<GlobMatcher>]) -> Result<Value, String> {
        let real = self.validate(&cur.to_string_lossy())?;
        let mut out = vec![];
        for (name, dir) in entries(&real)? {
            let rel = rel_path(root, &cur.join(&name));
            if ex.iter().any(|ms| ms.iter().any(|m| m.is_match(&rel))) {
                continue;
            }
            let mut e = json!({"name": name, "type": if dir { "directory" } else { "file" }});
            if dir {
                e["children"] = self.tree(root, &cur.join(&name), ex)?;
            }
            out.push(e);
        }
        Ok(Value::Array(out))
    }

    fn search(&self, root: &Path, cur: &Path, pat: &GlobMatcher, ex: &[GlobMatcher], out: &mut Vec<String>) -> Result<(), String> {
        for (name, dir) in entries(cur)? {
            let full = cur.join(&name);
            if self.validate(&full.to_string_lossy()).is_err() {
                continue;
            }
            let rel = rel_path(root, &full);
            if ex.iter().any(|m| m.is_match(&rel)) {
                continue;
            }
            if pat.is_match(&rel) {
                out.push(full.display().to_string());
            }
            if dir {
                let _ = self.search(root, &full, pat, ex, out); // an unreadable folder is skipped, as the reference does
            }
        }
        Ok(())
    }
}

/// minimatch(..., {dot: true}): `*` stays inside one path segment, dotfiles match.
fn glob(p: &str) -> Option<GlobMatcher> {
    GlobBuilder::new(p).literal_separator(true).build().or_else(|_| Glob::new(&globset::escape(p))).ok().map(|g| g.compile_matcher())
}

fn rel_path(root: &Path, p: &Path) -> String {
    let (r, p) = (lexical(root), lexical(p));
    p.strip_prefix(&r).map(|x| x.to_string_lossy().into_owned()).unwrap_or_else(|_| p.to_string_lossy().into_owned())
}

/// readdir with file types, sorted by name as libuv's scandir does (a symlink is not a directory, as Node's Dirent says).
fn entries(p: &Path) -> Result<Vec<(String, bool)>, String> {
    let rd = std::fs::read_dir(p).map_err(|e| format!("{e}: '{}'", p.display()))?;
    let mut v: Vec<(String, bool)> = rd.flatten().map(|e| (e.file_name().to_string_lossy().into_owned(), e.file_type().map(|t| t.is_dir()).unwrap_or(false))).collect();
    v.sort();
    Ok(v)
}

/// fs.readFile(p, "utf-8"): bad bytes become U+FFFD.
fn read_text(p: &Path) -> Result<String, String> {
    if p.is_dir() {
        return Err(format!("EISDIR: illegal operation on a directory, read '{}'", p.display()));
    }
    std::fs::read(p).map(|b| String::from_utf8_lossy(&b).into_owned()).map_err(|e| e.to_string())
}

/// a JS Date's toString(): "Tue Oct 28 2026 18:31:32 GMT-0700"
fn date(d: Option<DateTime<Local>>) -> String {
    d.map(|d| d.format("%a %b %d %Y %H:%M:%S GMT%z").to_string()).unwrap_or_else(|| "Invalid Date".into())
}

fn tools() -> Value {
    // the schemas exactly as the reference server's zod -> JSON Schema 2020-12 gives them
    let obj = |props: Value, req: &[&str]| {
        let mut o = json!({"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "properties": props});
        if !req.is_empty() {
            o["required"] = json!(req);
        }
        o
    };
    let mut out = obj(json!({"content": {"type": "string"}}), &["content"]);
    out["additionalProperties"] = json!(false);
    let ro = json!({"readOnlyHint": true, "openWorldHint": false});
    let path = json!({"type": "string"});
    let read_props = json!({"path": path, "tail": {"type": "number", "description": "If provided, returns only the last N lines of the file"}, "head": {"type": "number", "description": "If provided, returns only the first N lines of the file"}});
    let ex = json!({"type": "array", "items": {"type": "string"}, "default": []});
    let t = |name: &str, title: &str, desc: &str, schema: Value| json!({"name": name, "title": title, "description": desc, "inputSchema": schema, "outputSchema": out, "annotations": ro});
    json!([
        t("read_file", "Read File (Deprecated)", "Read the complete contents of a file as text. DEPRECATED: Use read_text_file instead.", obj(read_props.clone(), &["path"])),
        t("read_text_file", "Read Text File", "Read the complete contents of a file from the file system as text. Handles various text encodings and provides detailed error messages if the file cannot be read. Use this tool when you need to examine the contents of a single file. Use the 'head' parameter to read only the first N lines of a file, or the 'tail' parameter to read only the last N lines of a file. Operates on the file as text regardless of extension. Only works within allowed directories.", obj(read_props, &["path"])),
        t("read_multiple_files", "Read Multiple Files", "Read the contents of multiple files simultaneously. This is more efficient than reading files one by one when you need to analyze or compare multiple files. Each file's content is returned with its path as a reference. Failed reads for individual files won't stop the entire operation. Only works within allowed directories.", obj(json!({"paths": {"type": "array", "items": {"type": "string"}, "minItems": 1, "description": "Array of file paths to read. Each path must be a string pointing to a valid file within allowed directories."}}), &["paths"])),
        t("list_directory", "List Directory", "Get a detailed listing of all files and directories in a specified path. Results clearly distinguish between files and directories with [FILE] and [DIR] prefixes. This tool is essential for understanding directory structure and finding specific files within a directory. Only works within allowed directories.", obj(json!({"path": path}), &["path"])),
        t("directory_tree", "Directory Tree", "Get a recursive tree view of files and directories as a JSON structure. Each entry includes 'name', 'type' (file/directory), and 'children' for directories. Files have no children array, while directories always have a children array (which may be empty). The output is formatted with 2-space indentation for readability. Only works within allowed directories.", obj(json!({"path": path, "excludePatterns": ex}), &["path"])),
        t("search_files", "Search Files", "Recursively search for files and directories matching a pattern. The patterns should be glob-style patterns that match paths relative to the working directory. Use pattern like '*.ext' to match files in current directory, and '**/*.ext' to match files in all subdirectories. Returns full paths to all matching items. Great for finding files when you don't know their exact location. Only searches within allowed directories.", obj(json!({"path": path, "pattern": {"type": "string"}, "excludePatterns": ex}), &["path", "pattern"])),
        t("get_file_info", "Get File Info", "Retrieve detailed metadata about a file or directory. Returns comprehensive information including size, creation time, last modified time, permissions, and type. This tool is perfect for understanding file characteristics without reading the actual content. Only works within allowed directories.", obj(json!({"path": path}), &["path"])),
        t("list_allowed_directories", "List Allowed Directories", "Returns the list of directories that this server is allowed to access. Subdirectories within these allowed directories are also accessible. Use this to understand which directories and their nested paths are available before trying to access files.", obj(json!({}), &[])),
    ])
}

/// One JSON-RPC message in, at most one out (None for a notification).
fn handle(fs: &Fs, msg: &Value) -> Option<Value> {
    let id = msg.get("id")?.clone();
    let method = msg["method"].as_str().unwrap_or("");
    let result = match method {
        "initialize" => {
            let asked = msg["params"]["protocolVersion"].as_str().unwrap_or("");
            let proto = if PROTOCOLS.contains(&asked) { asked } else { PROTOCOLS[0] };
            Ok(json!({"protocolVersion": proto, "capabilities": {"tools": {"listChanged": false}}, "serverInfo": {"name": NAME, "version": env!("CARGO_PKG_VERSION")}}))
        }
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({"tools": tools()})),
        "tools/call" => {
            let p = &msg["params"];
            Ok(match fs.call(p["name"].as_str().unwrap_or(""), &p["arguments"]) {
                Ok(text) => json!({"content": [{"type": "text", "text": text}], "structuredContent": {"content": text}}),
                Err(e) => json!({"content": [{"type": "text", "text": e}], "isError": true}),
            })
        }
        _ => Err(json!({"code": -32601, "message": format!("Method not found: {method}")})),
    };
    Some(match result {
        Ok(r) => json!({"jsonrpc": "2.0", "id": id, "result": r}),
        Err(e) => json!({"jsonrpc": "2.0", "id": id, "error": e}),
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("Usage: strata-files DIR [DIR ...]  (the folders the model may read)");
        std::process::exit(1);
    }
    let fs = Fs::new(&args);
    if fs.allowed.is_empty() {
        eprintln!("Error: None of the specified directories are accessible");
        std::process::exit(1);
    }
    eprintln!("strata-files: read-only MCP server on stdio for {}", fs.dirs());
    let (stdin, mut stdout) = (std::io::stdin(), std::io::stdout());
    for line in stdin.lock().lines().map_while(Result::ok) {
        if line.trim().is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<Value>(&line) {
            Ok(Value::Array(batch)) => {
                let r: Vec<Value> = batch.iter().filter_map(|m| handle(&fs, m)).collect();
                (!r.is_empty()).then(|| Value::Array(r))
            }
            Ok(m) => handle(&fs, &m),
            Err(e) => Some(json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32700, "message": format!("Parse error: {e}")}})),
        };
        if let Some(r) = reply {
            if writeln!(stdout, "{r}").and_then(|_| stdout.flush()).is_err() {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup(tag: &str) -> (PathBuf, Fs) {
        let base = std::env::temp_dir().join(format!("strata-files-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("ok/sub/.hid")).unwrap();
        std::fs::create_dir_all(base.join("secret")).unwrap();
        std::fs::write(base.join("ok/a.txt"), "one\ntwo\nthree\n").unwrap();
        std::fs::write(base.join("ok/sub/b.md"), "b").unwrap();
        std::fs::write(base.join("ok/sub/.hid/c.md"), "c").unwrap();
        std::fs::write(base.join("secret/key"), "nope").unwrap();
        std::os::unix::fs::symlink(base.join("secret/key"), base.join("ok/link")).unwrap();
        let fs = Fs::new(&[base.join("ok").to_string_lossy().into_owned()]);
        (base, fs)
    }

    #[test]
    fn reads_and_refuses() {
        let (base, fs) = setup("read");
        let ok = base.join("ok");
        let p = |s: &str| ok.join(s).to_string_lossy().into_owned();
        assert_eq!(fs.call("read_text_file", &json!({"path": p("a.txt")})).unwrap(), "one\ntwo\nthree\n");
        assert_eq!(fs.call("read_text_file", &json!({"path": p("a.txt"), "head": 2})).unwrap(), "one\ntwo");
        assert_eq!(fs.call("read_text_file", &json!({"path": p("a.txt"), "tail": 2})).unwrap(), "three\n"); // as the reference: last 2 of [.., three, ""]
        assert_eq!(fs.call("read_text_file", &json!({"path": "a.txt"})).unwrap(), "one\ntwo\nthree\n"); // relative: against the allowed dir
        // outside, by .. and by a symlink
        assert!(fs.call("read_text_file", &json!({"path": p("../secret/key")})).unwrap_err().contains("outside allowed directories"));
        assert!(fs.call("read_text_file", &json!({"path": p("link")})).unwrap_err().contains("symlink target outside"));
        assert!(fs.call("read_text_file", &json!({"path": base.join("okay").to_string_lossy()})).is_err()); // /ok is not a prefix of /okay
        let multi = fs.call("read_multiple_files", &json!({"paths": [p("sub/b.md"), p("../secret/key")]})).unwrap();
        assert!(multi.starts_with(&format!("{}:\nb\n\n---\n", p("sub/b.md"))) && multi.contains("Error - Access denied"));
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn lists_trees_and_searches() {
        let (base, fs) = setup("list");
        let ok = base.join("ok").to_string_lossy().into_owned();
        let mut ls: Vec<String> = fs.call("list_directory", &json!({"path": ok})).unwrap().lines().map(String::from).collect();
        ls.sort();
        assert_eq!(ls, ["[DIR] sub", "[FILE] a.txt", "[FILE] link"]);
        let found = fs.call("search_files", &json!({"path": ok, "pattern": "**/*.md"})).unwrap();
        assert!(found.contains("sub/b.md") && found.contains(".hid/c.md"), "{found}"); // dot: true
        assert_eq!(fs.call("search_files", &json!({"path": ok, "pattern": "*.md"})).unwrap(), "No matches found"); // * stays in one segment
        let none = fs.call("search_files", &json!({"path": ok, "pattern": "**/*.md", "excludePatterns": ["**/.hid/**"]})).unwrap();
        assert!(!none.contains("c.md"), "{none}");
        let tree: Value = serde_json::from_str(&fs.call("directory_tree", &json!({"path": ok, "excludePatterns": [".hid"]})).unwrap()).unwrap();
        let sub = tree.as_array().unwrap().iter().find(|e| e["name"] == "sub").unwrap();
        assert_eq!(sub["children"], json!([{"name": "b.md", "type": "file"}]));
        assert!(fs.call("get_file_info", &json!({"path": format!("{ok}/a.txt")})).unwrap().starts_with("size: 14\ncreated: "));
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn speaks_mcp() {
        let (base, fs) = setup("mcp");
        let init = handle(&fs, &json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}})).unwrap();
        assert_eq!(init["result"]["protocolVersion"], "2025-06-18");
        assert!(handle(&fs, &json!({"jsonrpc": "2.0", "method": "notifications/initialized"})).is_none());
        assert_eq!(handle(&fs, &json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})).unwrap()["result"]["tools"].as_array().unwrap().len(), 8);
        let bad = handle(&fs, &json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "read_text_file", "arguments": {"path": "/etc/passwd"}}})).unwrap();
        assert_eq!(bad["result"]["isError"], true);
        assert_eq!(handle(&fs, &json!({"jsonrpc": "2.0", "id": 4, "method": "nope"})).unwrap()["error"]["code"], -32601);
        let _ = std::fs::remove_dir_all(base);
    }
}
