// strata-afm: Apple's on-device model (FoundationModels, macOS 26+; AFM 3 Core on macOS 27) for strata-server's
// "engine": "afm" configs.  stdin: one JSON line a request ({"conversation": {"messages": [...]}, "max_tokens",
// "temperature"}); stdout: {"ready": true, "context": N} first, then per request {"delta": "..."} lines and
// {"done": true}, or {"error": "..."}.  Build: swiftc -O -parse-as-library strata_afm.swift -o strata-afm
import Foundation
import FoundationModels

func emit(_ o: [String: Any]) {
    guard let d = try? JSONSerialization.data(withJSONObject: o) else { return }
    FileHandle.standardOutput.write(d + Data([10]))
}

/// A message's text: a string, or the text parts of an OpenAI content list.
func text(_ c: Any?) -> String {
    if let s = c as? String { return s }
    if let parts = c as? [[String: Any]] { return parts.compactMap { $0["text"] as? String }.joined(separator: "\n") }
    return ""
}

@main
struct StrataAFM {
    static func main() async {
        let model = SystemLanguageModel.default
        guard case .available = model.availability else {
            emit(["error": "not available on this Mac (\(model.availability)): it needs Apple Intelligence on (System Settings > Apple Intelligence & Siri)"])
            exit(1)
        }
        var context = 4096
        if #available(macOS 27.0, *) { context = model.contextSize }
        emit(["ready": true, "context": context])
        while let line = readLine() {
            guard let req = (try? JSONSerialization.jsonObject(with: Data(line.utf8))) as? [String: Any] else {
                emit(["error": "a request that is not JSON"])
                continue
            }
            let msgs = (req["conversation"] as? [String: Any])?["messages"] as? [[String: Any]] ?? []
            // the system messages and the turns before the last one become the session's instructions; the last
            // message is the prompt (ponytail: the history as text, not a rebuilt Transcript)
            var system: [String] = [], turns: [String] = []
            for m in msgs.dropLast() {
                let role = m["role"] as? String ?? "user", t = text(m["content"])
                if role == "system" || role == "developer" { system.append(t) }
                else { turns.append("\(role == "assistant" ? "Assistant" : role == "tool" ? "Tool result" : "User"): \(t)") }
            }
            var instructions = system.joined(separator: "\n\n")
            if !turns.isEmpty { instructions += (instructions.isEmpty ? "" : "\n\n") + "The conversation so far:\n" + turns.joined(separator: "\n") }
            let prompt = text(msgs.last?["content"])
            let session = instructions.isEmpty ? LanguageModelSession(model: model) : LanguageModelSession(model: model, instructions: instructions)
            var opts = GenerationOptions()
            if let t = req["temperature"] as? Double { opts.temperature = t }
            if let n = req["max_tokens"] as? Int, n > 0 { opts.maximumResponseTokens = n }
            do {
                var sent = ""
                for try await snap in session.streamResponse(to: prompt, options: opts) {
                    let full = snap.content
                    if full.hasPrefix(sent) {
                        let d = String(full.dropFirst(sent.count))
                        if !d.isEmpty { emit(["delta": d]) }
                    } else {
                        emit(["delta": full]) // a snapshot that is not an extension of the last (none seen): sent whole
                    }
                    sent = full
                }
                emit(["done": true])
            } catch {
                emit(["error": "\(error)"])
            }
        }
    }
}
