//! The Strata folder's `.env`, a convenience on top of the run config, never in place of it (nothing here is
//! written back, and the config / the page / the command line keep their own):
//!
//! - `API_KEY=...`: one more key /v1 (and the page's endpoints) accept, next to the config's "api_key", --api-key and
//!   STRATA_API_KEY (several keys are accepted at once).
//! - `MCP_<NAME>_URL=...`, `MCP_<NAME>_API_KEY=...` [, `MCP_<NAME>_HEADER=X-API-Key`]: an MCP server named <name>
//!   (lowercase), with `Authorization: Bearer <key>` (or the key in that header); one of the same name in the config,
//!   --mcp-config or added on the page wins.  Read-only on the page, never saved to the config.
//!
//! The format (it has no specification; the subset dotenv parsers agree on): `KEY=value` lines, `#` comment lines, an
//! optional `export `, ` #` starting a comment after an unquoted value, '...' literal, "..." with \n \" \\.
//! ponytail: no multi-line values, no ${VAR} expansion; add them when a value needs one.
use serde_json::{json, Map, Value};
use std::path::Path;

pub fn parse(text: &str) -> Vec<(String, String)> {
    let mut out = vec![];
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((k, v)) = line.split_once('=') else { continue };
        let k = k.trim();
        if k.is_empty() || !k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            continue;
        }
        let v = v.trim();
        let value = if let Some(r) = v.strip_prefix('\'') {
            r.split_once('\'').map(|(a, _)| a.to_string()).unwrap_or_else(|| r.to_string())
        } else if let Some(r) = v.strip_prefix('"') {
            let mut s = String::new();
            let mut it = r.chars();
            while let Some(c) = it.next() {
                match c {
                    '"' => break,
                    '\\' => match it.next() {
                        Some('n') => s.push('\n'),
                        Some(o) => s.push(o),
                        None => {}
                    },
                    c => s.push(c),
                }
            }
            s
        } else {
            v.split(" #").next().unwrap_or("").trim().to_string()
        };
        out.push((k.to_string(), value));
    }
    out
}

/// `<root>/.env` (none: nothing); a file other users may read is said once (it holds keys).
pub fn load(root: &Path) -> Vec<(String, String)> {
    let p = root.join(".env");
    let Ok(text) = std::fs::read_to_string(&p) else { return vec![] };
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if std::fs::metadata(&p).map(|m| m.permissions().mode() & 0o077 != 0).unwrap_or(false) {
            crate::say(&format!("{} can be read by other users of this computer: chmod 600 {}", p.display(), p.display()));
        }
    }
    parse(&text)
}

pub fn get<'a>(vars: &'a [(String, String)], key: &str) -> Option<&'a str> {
    vars.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v.as_str()).filter(|v| !v.is_empty())
}

/// MCP_<NAME>_URL (+ _API_KEY, _HEADER) -> {"<name>": {"url", "headers"}}, as the config's "mcp_servers" holds them.
pub fn mcp_servers(vars: &[(String, String)]) -> Map<String, Value> {
    let mut out = Map::new();
    for (k, url) in vars {
        let Some(name) = k.strip_prefix("MCP_").and_then(|r| r.strip_suffix("_URL")).filter(|n| !n.is_empty()) else { continue };
        if url.is_empty() {
            continue;
        }
        let mut entry = json!({"url": url});
        if let Some(key) = get(vars, &format!("MCP_{name}_API_KEY")) {
            let header = get(vars, &format!("MCP_{name}_HEADER")).unwrap_or("Authorization");
            let value = if header.eq_ignore_ascii_case("authorization") { format!("Bearer {key}") } else { key.to_string() };
            entry["headers"] = json!({header: value});
        }
        out.insert(name.to_lowercase(), entry);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_common_subset() {
        let v = parse("# keys\nAPI_KEY=sk-1 # mine\nexport A='x #y'\nB=\"a\\nb \\\"q\\\"\" # c\n bad line\n=x\nEMPTY=\nC = spaced \n");
        assert_eq!(get(&v, "API_KEY"), Some("sk-1"));
        assert_eq!(get(&v, "A"), Some("x #y"));
        assert_eq!(get(&v, "B"), Some("a\nb \"q\""));
        assert_eq!(get(&v, "C"), Some("spaced"));
        assert_eq!(get(&v, "EMPTY"), None);
        assert_eq!(v.len(), 5);
    }

    #[test]
    fn mcp_servers_from_names() {
        let v = parse("MCP_BIBLE_URL=http://127.0.0.1:3460/mcp\nMCP_BIBLE_API_KEY=k1\nMCP_BIBLE_HEADER=X-API-Key\nMCP_GH_URL=https://x/mcp\nMCP_GH_API_KEY=k2\nMCP__URL=x\n");
        let m = mcp_servers(&v);
        assert_eq!(m["bible"], json!({"url": "http://127.0.0.1:3460/mcp", "headers": {"X-API-Key": "k1"}}));
        assert_eq!(m["gh"], json!({"url": "https://x/mcp", "headers": {"Authorization": "Bearer k2"}}));
        assert_eq!(m.len(), 2);
    }
}
