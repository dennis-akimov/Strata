//! A table kept as a CSV file in Strata-data/sheets/ (the master copy), and synced to a Google Sheet: the model edits
//! the file by a key column (Link), quickly and without a browser, and table_sync puts it in the sheet in one paste,
//! after taking the person's own columns (tick boxes, Confirmation) from the sheet back into the file.
//! CSV as RFC 4180 section 2 has it: records by CRLF (LF read too), fields by commas, a field with a comma, a quote or
//! a line break in double quotes, a quote inside one doubled.
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

pub type Table = Vec<Vec<String>>;

/// RFC 4180 text -> rows (the header is row 0).  Lenient where it costs nothing: LF alone ends a record too.
pub fn parse(text: &str) -> Result<Table, String> {
    let (mut rows, mut row, mut field) = (Vec::new(), Vec::new(), String::new());
    let mut chars = text.trim_start_matches('\u{feff}').chars().peekable();
    let (mut quoted, mut at_start) = (false, true);
    while let Some(c) = chars.next() {
        if quoted {
            match c {
                '"' if chars.peek() == Some(&'"') => {
                    chars.next();
                    field.push('"');
                }
                '"' => quoted = false,
                _ => field.push(c),
            }
            continue;
        }
        match c {
            '"' if at_start => {
                quoted = true;
                at_start = false;
            }
            ',' => {
                row.push(std::mem::take(&mut field));
                at_start = true;
            }
            '\r' if chars.peek() == Some(&'\n') => {}
            '\n' | '\r' => {
                row.push(std::mem::take(&mut field));
                rows.push(std::mem::take(&mut row));
                at_start = true;
            }
            _ => {
                field.push(c);
                at_start = false;
            }
        }
    }
    if quoted {
        return Err("the CSV ends inside a quoted field (a quote not closed)".into());
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        rows.push(row);
    }
    Ok(rows)
}

/// Rows -> RFC 4180 text (CRLF, a field quoted when it has a comma, a quote or a line break).
pub fn write(rows: &Table) -> String {
    let field = |f: &str| if f.contains([',', '"', '\n', '\r']) { format!("\"{}\"", f.replace('"', "\"\"")) } else { f.to_string() };
    rows.iter().map(|r| r.iter().map(|f| field(f)).collect::<Vec<_>>().join(",") + "\r\n").collect()
}

/// A table's file: a name only (letters, digits, space . _ - ( )), in `dir`; never a path the model makes up.
pub fn path_of(dir: &Path, name: &str) -> Result<PathBuf, String> {
    let n = name.trim().trim_end_matches(".csv");
    let ok = !n.is_empty() && n.len() <= 120 && !n.starts_with('.') && n.chars().all(|c| c.is_alphanumeric() || " ._-()".contains(c));
    if !ok {
        return Err(format!("{name:?}: a table's name is letters, digits, spaces and . _ - ( ) only"));
    }
    Ok(dir.join(format!("{n}.csv")))
}

fn col(header: &[String], name: &str) -> Option<usize> {
    header.iter().position(|h| h.trim().eq_ignore_ascii_case(name.trim()))
}

/// Add or change rows by the key column: a row whose key is in the table updates the columns given (the others stay),
/// a new key is a new row.  A table that does not exist yet takes `columns` as its header.  (changed, added)
pub fn upsert(table: &mut Table, rows: &[Value], key: &str, columns: Option<&[String]>) -> Result<(usize, usize), String> {
    if table.is_empty() {
        let cols = columns.filter(|c| !c.is_empty()).ok_or("a new table needs columns (its header)")?;
        table.push(cols.to_vec());
    }
    let header = table[0].clone();
    let k = col(&header, key).ok_or_else(|| format!("no key column {key:?}; the columns are: {}", header.join(", ")))?;
    let (mut changed, mut added) = (0, 0);
    for r in rows {
        let obj = r.as_object().ok_or_else(|| format!("each row: {{column: value}}; the columns are: {}", header.join(", ")))?;
        let mut vals = vec![];
        for (name, v) in obj {
            let i = col(&header, name).ok_or_else(|| format!("no column {name:?}; the columns are: {}", header.join(", ")))?;
            vals.push((i, match v { Value::String(s) => s.trim().to_string(), Value::Null => String::new(), o => o.to_string() }));
        }
        let kv = vals.iter().find(|(i, _)| *i == k).map(|(_, v)| v.clone()).filter(|v| !v.is_empty())
            .ok_or_else(|| format!("each row needs its {key:?} (the key)"))?;
        match table.iter().skip(1).position(|row| row.get(k).is_some_and(|x| x.trim() == kv)) {
            Some(p) => {
                let row = &mut table[p + 1];
                row.resize(header.len(), String::new());
                for (i, v) in vals {
                    row[i] = v;
                }
                changed += 1;
            }
            None => {
                let mut row = vec![String::new(); header.len()];
                for (i, v) in vals {
                    row[i] = v;
                }
                table.push(row);
                added += 1;
            }
        }
    }
    Ok((changed, added))
}

/// Remove the rows with these keys; how many went.
pub fn delete(table: &mut Table, keys: &[String], key: &str) -> Result<usize, String> {
    let header = table.first().ok_or("the table is empty")?.clone();
    let k = col(&header, key).ok_or_else(|| format!("no key column {key:?}; the columns are: {}", header.join(", ")))?;
    let before = table.len();
    let mut i = 1;
    while i < table.len() {
        if table[i].get(k).is_some_and(|x| keys.iter().any(|w| w.trim() == x.trim())) { table.remove(i); } else { i += 1; }
    }
    Ok(before - table.len())
}

/// The sheet's rows (as its copy gave them, tab-separated) into the table: the person's columns (`mine`) of rows with
/// the same key come from the sheet; a row only in the sheet (one the person added) is added.  The rest is the file's.
/// `deleted`: keys the file's rows were removed by (table_delete): such a row still in the sheet is not taken back.
pub fn merge(table: &mut Table, sheet_tsv: &str, key: &str, mine: &[String], deleted: &[String]) -> Result<(usize, usize), String> {
    let sheet: Table = sheet_tsv.split('\n').map(|l| l.trim_end_matches('\r').split('\t').map(|c| c.trim().to_string()).collect()).collect();
    let Some(sh) = sheet.first().filter(|h| h.iter().any(|c| !c.is_empty())) else { return Ok((0, 0)) };
    let header = table.first().ok_or("the table is empty")?.clone();
    let (Some(k), Some(sk)) = (col(&header, key), col(sh, key)) else {
        return Err(format!("the sheet's header has no {key:?}: is it the sheet of this table?"));
    };
    let (mut took, mut added) = (0, 0);
    for srow in sheet.iter().skip(1).filter(|r| r.iter().any(|c| !c.is_empty() && c != "FALSE")) {
        let Some(kv) = srow.get(sk).filter(|v| !v.is_empty()) else { continue };
        match table.iter().skip(1).position(|r| r.get(k).is_some_and(|x| x.trim() == kv)) {
            Some(p) => {
                let row = &mut table[p + 1];
                row.resize(header.len(), String::new());
                for m in mine {
                    if let (Some(i), Some(si)) = (col(&header, m), col(sh, m)) {
                        let v = srow.get(si).cloned().unwrap_or_default();
                        if row[i] != v {
                            row[i] = v;
                            took += 1;
                        }
                    }
                }
            }
            None if deleted.iter().any(|d| d.trim() == kv) => {}   // removed from the file: the push removes it here too
            None => {
                table.push(header.iter().map(|h| col(sh, h).and_then(|si| srow.get(si).cloned()).unwrap_or_default()).collect());
                added += 1;
            }
        }
    }
    Ok((took, added))
}

/// The table as the paste takes it: tab-separated, a tab or line break in a cell a space (the file keeps it).
pub fn tsv(table: &Table) -> String {
    let width = table.first().map_or(0, Vec::len);
    table.iter().map(|r| (0..width).map(|i| r.get(i).map_or("", |s| s.as_str()).replace(['\t', '\r', '\n'], " ")).collect::<Vec<_>>().join("\t"))
        .collect::<Vec<_>>().join("\n")
}

/// The keys of rows table_delete removed (`<file>.deleted`, one per line), so a sync does not take them back.
pub fn tombstones(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path.with_extension("csv.deleted")).unwrap_or_default().lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from).collect()
}

/// The Google Sheet a table is synced to (`<file>.sheet`: its address, written by table_sync).
pub fn set_sheet(path: &Path, sheet_base: &str) -> Result<(), String> {
    std::fs::write(path.with_extension("csv.sheet"), format!("{sheet_base}\n")).map_err(|e| format!("{}: {e}", path.display()))
}

/// The table whose sheet this is (by the sheet's base address), if one was synced to it.
pub fn table_for_sheet(dir: &Path, sheet_base: &str) -> Option<String> {
    std::fs::read_dir(dir).ok()?.flatten().find_map(|e| {
        let p = e.path();
        let name = p.file_name()?.to_str()?.strip_suffix(".csv.sheet")?.to_string();
        (std::fs::read_to_string(&p).ok()?.trim() == sheet_base).then_some(name)
    })
}

/// The tables there are (their names), for an answer that names them.
pub fn names(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir).into_iter().flatten().flatten()
        .filter_map(|e| e.file_name().to_str()?.strip_suffix(".csv").map(String::from)).collect();
    v.sort();
    v
}

/// A change to a sheet that has a local table: not made in the sheet (the file is the master copy, and the next sync
/// would undo it), but said with the call to make instead, as a 308 answer names the new address (RFC 9110 15.4.9).
pub fn redirect(tool: &str, table: &str) -> String {
    let instead = match tool {
        "sheets_delete_rows" | "sheets_clear" => format!("table_delete with name \"{table}\" and the rows' Link values"),
        _ => format!("table_upsert with name \"{table}\" and rows as {{column: value}} with their Link"),
    };
    format!("{tool} was not run: this sheet's master copy is the local table \"{table}\" (a change made only in the sheet is overwritten by the next sync). Instead: {instead}, then table_sync with name \"{table}\" and this sheet's url.")
}

/// Read a table's file (missing: empty).
pub fn load(path: &Path) -> Result<Table, String> {
    match std::fs::read_to_string(path) {
        Ok(t) => parse(&t),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(vec![]),
        Err(e) => Err(format!("{}: {e}", path.display())),
    }
}

/// Write it whole, through a temporary file renamed over it (a crash never leaves half a table).
pub fn save(path: &Path, table: &Table) -> Result<(), String> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let tmp = path.with_extension("csv.tmp");
    std::fs::write(&tmp, write(table)).and_then(|_| std::fs::rename(&tmp, path)).map_err(|e| format!("{}: {e}", path.display()))
}

/// The local tools (no browser): table_read, table_upsert, table_delete.  The answer for the model.
pub fn local_tool(dir: &Path, tool: &str, args: &Value) -> Result<String, String> {
    let name = args["name"].as_str().unwrap_or("");
    let path = path_of(dir, name)?;
    let key = args["key"].as_str().filter(|k| !k.is_empty()).unwrap_or("Link");
    let mut table = load(&path)?;
    match tool {
        "table_read" => {
            if table.is_empty() {
                return Err(format!("no table {name:?} yet (tables: {}): table_upsert with columns makes one", names(dir).join(", ")));
            }
            Ok(format!("{} ({} rows, key {key})\n\n{}", path.display(), table.len() - 1, crate::google::numbered(&tsv(&table), 1)))
        }
        "table_upsert" => {
            let rows = args["rows"].as_array().ok_or("rows: a list of {column: value}")?;
            let cols: Option<Vec<String>> = args["columns"].as_array().map(|c| c.iter().filter_map(Value::as_str).map(String::from).collect());
            let (changed, added) = upsert(&mut table, rows, key, cols.as_deref())?;
            save(&path, &table)?;
            Ok(format!("{name}: {changed} rows changed, {added} added by {key} ({} rows now). table_sync puts it in the Google Sheet.", table.len() - 1))
        }
        "table_delete" => {
            let keys: Vec<String> = args["keys"].as_array().ok_or("keys: the key values of the rows to remove")?.iter().filter_map(Value::as_str).map(String::from).collect();
            let n = delete(&mut table, &keys, key)?;
            save(&path, &table)?;
            let mut gone = tombstones(&path);
            for k in keys.iter().map(|k| k.trim().to_string()) {
                if !gone.contains(&k) {
                    gone.push(k);
                }
            }
            std::fs::write(path.with_extension("csv.deleted"), gone.join("\n") + "\n").map_err(|e| format!("{}: {e}", path.display()))?;
            Ok(format!("{name}: {n} rows removed ({} rows now)", table.len().saturating_sub(1)))
        }
        other => Err(format!("no tool {other}")),
    }
}

/// The tools, as tools/list gives them.
pub fn tools() -> Value {
    let name = json!({"type": "string", "description": "the table's name (its file in Strata-data/sheets/, e.g. studios-sandiego-downtown-2026)"});
    json!([
        {"name": "table_read", "description": "Read a table kept as a CSV file on this Mac (the master copy of a Google Sheet): its rows numbered, row 1 the header. Use the table tools, not sheets_append / sheets_write, for a sheet that has a table: change rows with table_upsert / table_delete, then table_sync.",
         "inputSchema": {"type": "object", "required": ["name"], "properties": {"name": name}}},
        {"name": "table_upsert", "description": "Add or change rows of a local table by its key column (default Link): a row with a key already there changes only the columns given; a new key is a new row. A new table needs columns (its header). Fast: no browser. Then table_sync to put it in the Google Sheet.",
         "inputSchema": {"type": "object", "required": ["name", "rows"], "properties": {"name": name,
             "rows": {"type": "array", "items": {"type": "object", "additionalProperties": {"type": "string"}}, "description": "one {column: value} per row, each with its key"},
             "columns": {"type": "array", "items": {"type": "string"}, "description": "the header, for a new table"},
             "key": {"type": "string", "description": "the key column (default Link)"}}}},
        {"name": "table_delete", "description": "Remove rows of a local table by their key values.",
         "inputSchema": {"type": "object", "required": ["name", "keys"], "properties": {"name": name, "keys": {"type": "array", "items": {"type": "string"}}, "key": {"type": "string"}}}},
        {"name": "table_sync", "description": "Put a local table in its Google Sheet (one paste of the whole table, rows past it removed, a Checkbox column as tick boxes), after taking the person's own columns (default Checkbox, Confirmation) and any rows they added from the sheet into the file, so their edits are kept.",
         "inputSchema": {"type": "object", "required": ["name", "url"], "properties": {"name": name, "url": {"type": "string", "description": "the Google Sheet's address"},
             "mine": {"type": "array", "items": {"type": "string"}, "description": "the columns the person edits in the sheet (default Checkbox, Confirmation)"},
             "key": {"type": "string"}}}}
    ])
}

/// table_sync's second step: the merged table into the sheet: `old` rows were there (with the header).
pub fn sync_push_js(table: &Table, old: usize) -> String {
    let rows = serde_json::to_string(&tsv(table)).unwrap_or_default();
    let n = table.len();
    let box_col = table.first().and_then(|h| col(h, "Checkbox"));
    let boxes = box_col.map(|i| format!("await go(letter({i}) + '2:' + letter({i}) + '{n}'); await menu('#docs-insert-menu', 'Checkbox');")).unwrap_or_default();
    let extra = if old > n {
        let item = if old == n + 1 { format!("Row {old}") } else { format!("Rows {} - {old}", n + 1) };
        format!(r#"await go('A{a}:A{old}'); await menu('#docs-edit-menu', 'Delete');
  const it = [...document.querySelectorAll('.goog-menu .goog-menuitem')].find((e) => e.offsetParent && e.innerText.trim().startsWith('{item}'));
  if (!it) throw new Error('no menu item {item}'); mouse(it); await sleep(800);"#, a = n + 1)
    } else {
        String::new()
    };
    format!(r#"await ready(); await go('A1'); clip('paste', {rows}); await sleep(1200);
  {extra}
  {boxes}
  await go('A1');
  return 'put {rows_n} rows in ' + where() + '{removed}';"#,
            rows_n = n.saturating_sub(1),
            removed = if old > n { format!(" (rows {}-{old} that the table no longer has removed)", n + 1) } else { String::new() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_as_rfc_4180_writes_it() {
        // section 2's own examples: CRLF records, a last record without one, quoted fields with commas, line breaks
        // and doubled quotes
        assert_eq!(parse("aaa,bbb,ccc\r\nzzz,yyy,xxx\r\n").unwrap(), [["aaa", "bbb", "ccc"], ["zzz", "yyy", "xxx"]]);
        assert_eq!(parse("aaa,bbb,ccc\r\nzzz,yyy,xxx").unwrap(), [["aaa", "bbb", "ccc"], ["zzz", "yyy", "xxx"]]);
        assert_eq!(parse("\"aaa\",\"b\r\nbb\",\"ccc\"\r\nzzz,yyy,xxx").unwrap(), [vec!["aaa", "b\r\nbb", "ccc"], vec!["zzz", "yyy", "xxx"]]);
        assert_eq!(parse("\"aaa\",\"b\"\"bb\",\"ccc\"").unwrap(), [["aaa", "b\"bb", "ccc"]]);
        assert_eq!(parse("a,,c\n,\n").unwrap(), [vec!["a", "", "c"], vec!["", ""]], "empty fields, LF alone");
        assert!(parse("a,\"b").is_err(), "an unclosed quote");
        // and back: what needs quotes gets them, the rest stays plain
        let rows: Table = vec![vec!["Price".into(), "Description".into()], vec!["$2,411".into(), "The \"Rey\"\nlofts".into()]];
        assert_eq!(write(&rows), "Price,Description\r\n\"$2,411\",\"The \"\"Rey\"\"\nlofts\"\r\n");
        assert_eq!(parse(&write(&rows)).unwrap(), rows);
    }

    #[test]
    fn names_stay_in_the_folder() {
        let d = Path::new("/x/sheets");
        assert_eq!(path_of(d, "studios-sandiego-downtown-2026 (3 more)").unwrap(), d.join("studios-sandiego-downtown-2026 (3 more).csv"));
        assert_eq!(path_of(d, "t.csv").unwrap(), d.join("t.csv"));
        for bad in ["../etc/passwd", "a/b", "", ".hidden", "x\0y"] {
            assert!(path_of(d, bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn upsert_by_link_and_delete() {
        let cols: Vec<String> = ["Checkbox", "Address", "Price", "Link"].iter().map(|s| s.to_string()).collect();
        let mut t: Table = vec![];
        let r = upsert(&mut t, &[json!({"Link": "https://a", "Address": "1 A St", "Checkbox": "FALSE"})], "Link", Some(&cols)).unwrap();
        assert_eq!(r, (0, 1));
        // the same link: only the given column changes
        assert_eq!(upsert(&mut t, &[json!({"link": "https://a", "Price": "$2,100"}), json!({"Link": "https://b", "Address": "2 B St"})], "Link", None).unwrap(), (1, 1));
        assert_eq!(t[1], ["FALSE", "1 A St", "$2,100", "https://a"]);
        assert_eq!(t[2], ["", "2 B St", "", "https://b"]);
        assert!(upsert(&mut t, &[json!({"Address": "no key"})], "Link", None).unwrap_err().contains("needs its \"Link\""));
        assert!(upsert(&mut t, &[json!({"Link": "x", "Name": "y"})], "Link", None).unwrap_err().starts_with("no column \"Name\""));
        assert!(upsert(&mut vec![], &[json!({"Link": "x"})], "Link", None).unwrap_err().contains("needs columns"));
        assert_eq!(delete(&mut t, &["https://a".into()], "Link").unwrap(), 1);
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn the_persons_columns_come_from_the_sheet() {
        let mut t: Table = vec![
            vec!["Checkbox".into(), "Address".into(), "Link".into(), "Confirmation".into()],
            vec!["FALSE".into(), "1 A St (file)".into(), "https://a".into(), "".into()],
        ];
        // in the sheet: the person ticked a, wrote a confirmation, edited the address (not theirs), added a row
        let sheet = "Checkbox\tAddress\tLink\tConfirmation\nTRUE\t1 A St (edited)\thttps://a\tcalled Mon\nFALSE\t9 Z St\thttps://z\t\nFALSE\t\t\t";
        let mine: Vec<String> = vec!["Checkbox".into(), "Confirmation".into()];
        assert_eq!(merge(&mut t.clone(), sheet, "Link", &mine, &["https://z".into()]).unwrap(), (2, 0), "a row the file deleted is not taken back");
        assert_eq!(merge(&mut t, sheet, "Link", &mine, &[]).unwrap(), (2, 1));
        assert_eq!(t[1], ["TRUE", "1 A St (file)", "https://a", "called Mon"], "their columns from the sheet, the rest from the file");
        assert_eq!(t[2], ["FALSE", "9 Z St", "https://z", ""], "a row they added is kept");
        assert_eq!(t.len(), 3, "an empty tick-box row is no row");
        assert!(merge(&mut t, "Name\tPrice\nx\t1", "Link", &mine, &[]).unwrap_err().contains("has no \"Link\""));
        assert_eq!(merge(&mut t, "", "Link", &mine, &[]).unwrap(), (0, 0), "an empty sheet takes nothing");
    }

    #[test]
    fn the_push_pastes_once_and_trims() {
        let t: Table = vec![vec!["Checkbox".into(), "Link".into()], vec!["FALSE".into(), "https://a\tb".into()]];
        assert_eq!(tsv(&t), "Checkbox\tLink\nFALSE\thttps://a b");
        let js = sync_push_js(&t, 5);
        assert!(js.contains("clip('paste', \"Checkbox\\tLink\\nFALSE\\thttps://a b\")"));
        assert!(js.contains("'Rows 3 - 5'") && js.contains("go(letter(0) + '2:' + letter(0) + '2')"), "{js}");
        assert!(!sync_push_js(&t, 2).contains("Delete"), "nothing to remove");
    }

    #[test]
    fn a_sheet_with_a_table_sends_changes_to_the_table() {
        let dir = std::env::temp_dir().join(format!("strata-sheetlink-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let base = "https://docs.google.com/spreadsheets/d/abcdefghijk";
        assert_eq!(table_for_sheet(&dir, base), None);
        save(&dir.join("studios.csv"), &vec![vec!["Link".into()]]).unwrap();
        set_sheet(&dir.join("studios.csv"), base).unwrap();
        assert_eq!(table_for_sheet(&dir, base).as_deref(), Some("studios"));
        assert_eq!(table_for_sheet(&dir, "https://docs.google.com/spreadsheets/d/other"), None);
        assert_eq!(names(&dir), ["studios"]);
        let r = redirect("sheets_append", "studios");
        assert!(r.starts_with("sheets_append was not run: this sheet's master copy is the local table \"studios\"") && r.contains("table_upsert with name \"studios\"") && r.contains("then table_sync"), "{r}");
        assert!(redirect("sheets_delete_rows", "studios").contains("table_delete"));
        assert!(local_tool(&dir, "table_read", &json!({"name": "nope"})).unwrap_err().contains("(tables: studios)"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn local_tools_keep_the_file_and_the_deleted_keys() {
        let dir = std::env::temp_dir().join(format!("strata-table-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let up = |rows: Value, cols: Value| local_tool(&dir, "table_upsert", &json!({"name": "t", "rows": rows, "columns": cols}));
        assert!(up(json!([{"Link": "https://a", "Price": "$1"}]), json!(["Price", "Link"])).unwrap().contains("0 rows changed, 1 added"));
        assert!(up(json!([{"Link": "https://b", "Price": "$2"}]), Value::Null).unwrap().contains("(2 rows now)"));
        assert_eq!(std::fs::read_to_string(dir.join("t.csv")).unwrap(), "Price,Link\r\n$1,https://a\r\n$2,https://b\r\n");
        assert!(local_tool(&dir, "table_delete", &json!({"name": "t", "keys": ["https://a"]})).unwrap().contains("1 rows removed"));
        assert_eq!(tombstones(&dir.join("t.csv")), ["https://a"]);
        assert!(local_tool(&dir, "table_read", &json!({"name": "t"})).unwrap().contains("row 2: $2 | https://b"));
        assert!(local_tool(&dir, "table_read", &json!({"name": "../x"})).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
