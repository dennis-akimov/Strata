// strata-reminders: macOS Reminders as an MCP server (stdio, JSON-RPC 2.0, a message a line) for Strata's model.
// Tools: reminders_lists, reminders_list, reminders_add, reminders_complete.  EventKit asks once for Reminders access
// (for Strata.app, which started the server that started this).  Turned on in the chat page's MCP section.
// Build: swiftc -O -parse-as-library strata_reminders.swift -o strata-reminders \
//          -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker Info.plist
// --selftest: the date parsing's checks.
import EventKit
import Foundation

func send(_ o: [String: Any]) {
    guard let d = try? JSONSerialization.data(withJSONObject: o) else { return }
    FileHandle.standardOutput.write(d + Data([10]))
}

let tools: [[String: Any]] = [
    ["name": "reminders_lists", "description": "The user's Reminders lists (macOS Reminders): their names and how many open reminders each has.",
     "inputSchema": ["type": "object", "properties": [:] as [String: Any]]],
    ["name": "reminders_list", "description": "Reminders from the user's macOS Reminders: the open ones by default, from one list or all.",
     "inputSchema": ["type": "object", "properties": [
        "list": ["type": "string", "description": "a list's name (all lists when left out)"],
        "include_completed": ["type": "boolean", "description": "completed ones too (default false)"],
        "limit": ["type": "integer", "description": "at most this many (default 50)"]]]],
    ["name": "reminders_add", "description": "Add a reminder to the user's macOS Reminders.",
     "inputSchema": ["type": "object", "required": ["title"], "properties": [
        "title": ["type": "string"], "notes": ["type": "string"],
        "due": ["type": "string", "description": "YYYY-MM-DD, or YYYY-MM-DD HH:MM (local time; with a time it also alerts then)"],
        "list": ["type": "string", "description": "a list's name (the default list when left out)"]]]],
    ["name": "reminders_complete", "description": "Mark a reminder done (its id from reminders_list).",
     "inputSchema": ["type": "object", "required": ["id"], "properties": ["id": ["type": "string"]]]],
]

/// "2026-10-31" or "2026-10-31 09:30" (or with a T) -> the reminder's due date parts (a time: an alert too).
func dueParts(_ s: String) -> DateComponents? {
    let t = s.trimmingCharacters(in: .whitespaces).replacingOccurrences(of: "T", with: " ")
    let parts = t.split(separator: " ")
    let d = parts.first?.split(separator: "-").compactMap { Int($0) } ?? []
    guard d.count == 3, (1...12).contains(d[1]), (1...31).contains(d[2]) else { return nil }
    var c = DateComponents(calendar: Calendar.current, timeZone: TimeZone.current, year: d[0], month: d[1], day: d[2])
    if parts.count > 1 {
        let hm = parts[1].split(separator: ":").compactMap { Int($0) }
        guard hm.count >= 2, (0...23).contains(hm[0]), (0...59).contains(hm[1]) else { return nil }
        c.hour = hm[0]; c.minute = hm[1]
    }
    return c
}

func dueText(_ c: DateComponents?) -> String? {
    guard let c, let y = c.year, let m = c.month, let d = c.day else { return nil }
    let day = String(format: "%04d-%02d-%02d", y, m, d)
    if let h = c.hour, let mi = c.minute { return day + String(format: " %02d:%02d", h, mi) }
    return day
}

actor Store {
    let store = EKEventStore()
    var granted = false

    func access() async throws {
        if granted { return }
        granted = try await store.requestFullAccessToReminders()
        if !granted {
            throw NSError(domain: "strata-reminders", code: 1, userInfo: [NSLocalizedDescriptionKey:
                "no access to Reminders: System Settings > Privacy & Security > Reminders > Strata (then ask again)"])
        }
    }

    func lists() -> [EKCalendar] { store.calendars(for: .reminder) }

    func list(named n: String?) throws -> [EKCalendar]? {
        guard let n, !n.isEmpty else { return nil }
        let m = lists().filter { $0.title.caseInsensitiveCompare(n) == .orderedSame }
        if m.isEmpty { throw NSError(domain: "strata-reminders", code: 2, userInfo: [NSLocalizedDescriptionKey: "no list called \"\(n)\"; the lists: \(lists().map(\.title).joined(separator: ", "))"]) }
        return m
    }

    func fetch(_ p: NSPredicate) async -> [EKReminder] {
        await withCheckedContinuation { c in store.fetchReminders(matching: p) { c.resume(returning: $0 ?? []) } }
    }

    func call(_ name: String, _ a: [String: Any]) async throws -> Any {
        try await access()
        switch name {
        case "reminders_lists":
            var out: [[String: Any]] = []
            for l in lists() {
                let open = await fetch(store.predicateForIncompleteReminders(withDueDateStarting: nil, ending: nil, calendars: [l]))
                out.append(["name": l.title, "open": open.count])
            }
            return out
        case "reminders_list":
            let cals = try list(named: a["list"] as? String)
            let all = (a["include_completed"] as? Bool) == true
            let p = all ? store.predicateForReminders(in: cals) : store.predicateForIncompleteReminders(withDueDateStarting: nil, ending: nil, calendars: cals)
            let limit = max(1, (a["limit"] as? Int) ?? 50)
            let rs = await fetch(p).sorted { (dueText($0.dueDateComponents) ?? "9999") < (dueText($1.dueDateComponents) ?? "9999") }
            return rs.prefix(limit).map { r -> [String: Any] in
                var o: [String: Any] = ["id": r.calendarItemIdentifier, "title": r.title ?? "", "list": r.calendar?.title ?? "", "done": r.isCompleted]
                if let d = dueText(r.dueDateComponents) { o["due"] = d }
                if let n = r.notes, !n.isEmpty { o["notes"] = n }
                return o
            }
        case "reminders_add":
            guard let title = (a["title"] as? String)?.trimmingCharacters(in: .whitespaces), !title.isEmpty else {
                throw NSError(domain: "strata-reminders", code: 3, userInfo: [NSLocalizedDescriptionKey: "a reminder needs a title"])
            }
            let r = EKReminder(eventStore: store)
            r.title = title
            r.notes = a["notes"] as? String
            r.calendar = try list(named: a["list"] as? String)?.first ?? store.defaultCalendarForNewReminders()
            if let due = a["due"] as? String, !due.isEmpty {
                guard let c = dueParts(due) else { throw NSError(domain: "strata-reminders", code: 4, userInfo: [NSLocalizedDescriptionKey: "due: YYYY-MM-DD or YYYY-MM-DD HH:MM, not \"\(due)\""]) }
                r.dueDateComponents = c
                if c.hour != nil, let date = Calendar.current.date(from: c) { r.addAlarm(EKAlarm(absoluteDate: date)) }
            }
            try store.save(r, commit: true)
            return ["added": title, "id": r.calendarItemIdentifier, "list": r.calendar?.title ?? "", "due": dueText(r.dueDateComponents) as Any? ?? NSNull()]
        case "reminders_complete":
            guard let id = a["id"] as? String, let r = store.calendarItem(withIdentifier: id) as? EKReminder else {
                throw NSError(domain: "strata-reminders", code: 5, userInfo: [NSLocalizedDescriptionKey: "no reminder with that id (reminders_list gives them)"])
            }
            r.isCompleted = true
            try store.save(r, commit: true)
            return ["completed": r.title ?? ""]
        default:
            throw NSError(domain: "strata-reminders", code: 6, userInfo: [NSLocalizedDescriptionKey: "unknown tool \(name)"])
        }
    }
}

@main
struct StrataReminders {
    static func main() async {
        if CommandLine.arguments.contains("--selftest") {
            assert(dueText(dueParts("2026-10-31")) == "2026-10-31")
            assert(dueText(dueParts("2026-10-31 09:05")) == "2026-10-31 09:05")
            assert(dueText(dueParts("2026-10-31T18:30")) == "2026-10-31 18:30")
            assert(dueParts("31/10/2026") == nil && dueParts("2026-13-01") == nil && dueParts("2026-10-31 25:00") == nil)
            print("selftest ok")
            return
        }
        let store = Store()
        while let line = readLine() {
            guard let m = (try? JSONSerialization.jsonObject(with: Data(line.utf8))) as? [String: Any] else { continue }
            let id = m["id"], method = m["method"] as? String ?? ""
            guard let id else { continue }                        // a notification: nothing to answer
            switch method {
            case "initialize":
                let v = (m["params"] as? [String: Any])?["protocolVersion"] as? String ?? "2025-06-18"
                send(["jsonrpc": "2.0", "id": id, "result": ["protocolVersion": v, "capabilities": ["tools": [:] as [String: Any]],
                                                              "serverInfo": ["name": "strata-reminders", "version": "1"]]])
            case "tools/list":
                send(["jsonrpc": "2.0", "id": id, "result": ["tools": tools]])
            case "tools/call":
                let p = m["params"] as? [String: Any] ?? [:]
                do {
                    let r = try await store.call(p["name"] as? String ?? "", p["arguments"] as? [String: Any] ?? [:])
                    let text = String(data: (try? JSONSerialization.data(withJSONObject: r, options: [.sortedKeys])) ?? Data(), encoding: .utf8) ?? ""
                    send(["jsonrpc": "2.0", "id": id, "result": ["content": [["type": "text", "text": text]]]])
                } catch {
                    send(["jsonrpc": "2.0", "id": id, "result": ["content": [["type": "text", "text": error.localizedDescription]], "isError": true]])
                }
            case "ping":
                send(["jsonrpc": "2.0", "id": id, "result": [:] as [String: Any]])
            default:
                send(["jsonrpc": "2.0", "id": id, "error": ["code": -32601, "message": "method not found: \(method)"]])
            }
        }
    }
}
