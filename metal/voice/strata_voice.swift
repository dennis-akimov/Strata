// strata-voice: "Hey Strata" for Strata (macOS 26+).  Listens with Apple's on-device dictation (SpeechAnalyzer +
// DictationTranscriber: no network, no Apple Intelligence), and after the wake phrase sends what you say to the
// Strata server's chat, then reads the answer aloud with macOS's voice (`say`).  One JSON line an event on stdout
// (Strata.app shows them in its log): listening, wake, command, answer, error.
//        --permissions: ask for the microphone and speech recognition (the page's button), --stdin-exit: end with stdin
// Usage: strata-voice --port 8090 [--key KEY] [--phrase "hey strata"] [--voice NAME] [--locale en-US]
//        [--file AUDIO --once]: a recording instead of the microphone, exit after the first answer (a test);
//        --verbose: every transcript too;
//        --selftest: the wake phrase matcher's and the voice text's checks
// Build: swiftc -O -parse-as-library strata_voice.swift -o strata-voice \
//          -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker Info.plist   (the microphone's prompt)
@preconcurrency import AVFoundation
import Foundation
import Speech
import AppKit

nonisolated(unsafe) var micKeep: Mic?   // the capture session lives as long as the helper

func emit(_ o: [String: Any]) {
    guard let d = try? JSONSerialization.data(withJSONObject: o) else { return }
    FileHandle.standardOutput.write(d + Data([10]))
}

func arg(_ name: String, _ def: String) -> String {
    let a = CommandLine.arguments
    if let i = a.firstIndex(of: name), i + 1 < a.count, !a[i + 1].hasPrefix("--") { return a[i + 1] }
    return def
}

/// Lowercase words without punctuation: what the wake phrase is matched in.
func words(_ s: String) -> [String] {
    s.lowercased().split { !$0.isLetter && !$0.isNumber && $0 != "'" }.map(String.init)
}

/// Where the wake phrase ends in `ws` (the index after it), or nil.  Its last word ("strata") is the anchor, heard a
/// little off too ("strada", "stratas": one letter changed, added or missing), within the first words of what was
/// said: "Hey Strata", "Strata, ...", and "Play Strata" (how the recognizer heard a Mac voice's "Hey Strata"), but
/// not "strata" in the middle of a sentence.
func wakeEnd(_ ws: [String], _ phrase: [String]) -> Int? {
    guard let anchor = phrase.last else { return nil }
    func close(_ a: String, _ b: String) -> Bool {
        if a == b || a.hasPrefix(b) && a.count <= b.count + 2 { return true }
        let (x, y) = (Array(a), Array(b))
        var d = Array(0...y.count)
        for i in 1...max(x.count, 1) where !x.isEmpty {
            var prev = d[0]; d[0] = i
            for j in 1...max(y.count, 1) where !y.isEmpty {
                let t = d[j]
                d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (x[i - 1] == y[j - 1] ? 0 : 1))
                prev = t
            }
        }
        return d[y.count] <= (y.count >= 6 ? 2 : 1)      // a long word may be heard two letters off ("Haydn")
    }
    let head = Array(ws.prefix(phrase.count + 1))
    if let i = head.prefix(phrase.count).firstIndex(where: { close($0, anchor) }) { return i + 1 }
    // a word heard as two ("heyden" as "hey den")
    for i in 0..<max(0, min(phrase.count, head.count - 1)) where close(head[i] + head[i + 1], anchor) { return i + 2 }
    return nil
}

/// Markdown to plain sentences for the voice.
func plain(_ s: String) -> String {
    var t = s.replacingOccurrences(of: "```[\\s\\S]*?```", with: " (code shown in the chat) ", options: .regularExpression)
    for (p, r) in [("\\*\\*|__|`|#+ ", ""), ("\\[([^\\]]+)\\]\\([^)]+\\)", "$1"), ("(?m)^[ \\t]*[-*] ", "")] {
        t = t.replacingOccurrences(of: p, with: r, options: .regularExpression)
    }
    return t
}

/// The microphone through AVCaptureSession (input only).  AVAudioEngine also starts an output stream, and for this
/// helper (a child of Strata.app's server, no window) coreaudiod refused to start it: 'what', 2003329396.
final class Mic: NSObject, AVCaptureAudioDataOutputSampleBufferDelegate, @unchecked Sendable {
    let session = AVCaptureSession()
    let output = AVCaptureAudioDataOutput()
    let queue = DispatchQueue(label: "strata-voice.mic")
    var onBuffer: ((AVAudioPCMBuffer) -> Void)?

    func start() throws {
        guard let device = AVCaptureDevice.default(for: .audio) else { throw NSError(domain: "strata-voice", code: 1, userInfo: [NSLocalizedDescriptionKey: "no microphone"]) }
        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else { throw NSError(domain: "strata-voice", code: 2, userInfo: [NSLocalizedDescriptionKey: "the microphone cannot be opened"]) }
        session.addInput(input)
        // mono float32 at 48 kHz (what the converter to the recognizer's format starts from)
        output.audioSettings = [AVFormatIDKey: kAudioFormatLinearPCM, AVLinearPCMBitDepthKey: 32, AVLinearPCMIsFloatKey: true,
                                AVLinearPCMIsNonInterleaved: true, AVNumberOfChannelsKey: 1, AVSampleRateKey: 48000]
        output.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(output) else { throw NSError(domain: "strata-voice", code: 3, userInfo: [NSLocalizedDescriptionKey: "no audio output for the capture"]) }
        session.addOutput(output)
        session.startRunning()
    }

    func captureOutput(_ o: AVCaptureOutput, didOutput sb: CMSampleBuffer, from c: AVCaptureConnection) {
        guard let desc = CMSampleBufferGetFormatDescription(sb), let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(desc),
              let format = AVAudioFormat(streamDescription: asbd) else { return }
        let n = AVAudioFrameCount(CMSampleBufferGetNumSamples(sb))
        guard n > 0, let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: n) else { return }
        buf.frameLength = n
        guard CMSampleBufferCopyPCMDataIntoAudioBufferList(sb, at: 0, frameCount: Int32(n), into: buf.mutableAudioBufferList) == noErr else { return }
        onBuffer?(buf)
    }
}

actor Voice {
    let port: String, key: String, phrase: [String], voice: String
    var history: [[String: String]] = []
    var capturing = false, command = "", lastHeard = Date(), started = Date(), speaking: Process?
    var baseline = 0   // words of the current utterance before the wake phrase's end
    var open = false   // the recognizer's current utterance has not had its final result yet
    var consumed = false  // that utterance's question was sent: the rest of it (its final result) is no new wake
    var directive: String?  // strata-server's word on a question: "page" (the open chat sends it) or "self"

    func direct(_ d: String) { directive = d }

    init(port: String, key: String, phrase: String, voice: String) {
        self.port = port; self.key = key; self.phrase = words(phrase); self.voice = voice
    }

    /// A recognizer result: the current utterance's text so far (`final`: the utterance ended).
    func heard(_ text: String, final: Bool) {
        open = !final
        if consumed {
            if final { consumed = false }
            return
        }
        let ws = words(text)
        if !capturing {
            guard let end = wakeEnd(ws, phrase) else { return }
            speaking?.terminate(); speaking = nil         // a new question interrupts the answer
            capturing = true; started = Date(); lastHeard = Date()
            NSSound(named: "Tink")?.play()
            emit(["event": "wake"])
            command = ws[end...].joined(separator: " ")
            baseline = 0
            if final { baseline = -1 }                     // the rest comes in the next utterances
            return
        }
        lastHeard = Date()
        // the utterance with the wake phrase: what follows it; later utterances: all of it
        let part = baseline < 0 ? text : (wakeEnd(ws, phrase).map { ws[$0...].joined(separator: " ") } ?? text)
        if baseline < 0 {
            command = (command + " " + part).trimmingCharacters(in: .whitespaces)
        } else {
            command = part
            if final { baseline = -1 }
        }
    }

    /// Called every 0.3 s: a pause after the question (or a long one) sends it.
    func tick() async {
        guard capturing else { return }
        let quiet = Date().timeIntervalSince(lastHeard), total = Date().timeIntervalSince(started)
        let q = command.trimmingCharacters(in: .whitespaces)
        if (q.isEmpty && quiet < 6) || (!q.isEmpty && quiet < 1.4 && total < 30) { return }
        capturing = false
        guard !q.isEmpty else { emit(["event": "wake-timeout"]); return }
        NSSound(named: "Pop")?.play()
        consumed = open
        directive = nil
        emit(["event": "command", "text": q])
        // started by strata-server: it says whether an open chat page takes the question (the chat keeps it, and
        // sends the answer back to be read: {"say": ...}); no word in 1.5 s, or a page-less run: ask here
        if CommandLine.arguments.contains("--stdin-exit") {
            for _ in 0..<15 where directive == nil { try? await Task.sleep(nanoseconds: 100_000_000) }
            if directive == "page" { return }
        }
        await ask(q)
    }

    func ask(_ q: String) async {
        history.append(["role": "user", "content": q])
        history = Array(history.suffix(8))
        let msgs: [[String: String]] = [["role": "system", "content": "You answer by voice: a few short, plain sentences, no lists, no markdown, no code unless asked."]] + history
        var req = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        req.httpMethod = "POST"
        req.timeoutInterval = 600
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if !key.isEmpty { req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization") }
        req.httpBody = try? JSONSerialization.data(withJSONObject: ["model": "strata", "messages": msgs, "max_tokens": 600, "reasoning_effort": "low"])
        do {
            let (data, _) = try await URLSession.shared.data(for: req)
            let j = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            guard let msg = ((j?["choices"] as? [[String: Any]])?.first?["message"] as? [String: Any]),
                  let text = msg["content"] as? String, !text.isEmpty else {
                let e = ((j?["error"] as? [String: Any])?["message"] as? String) ?? "no answer"
                emit(["event": "error", "text": e]); say("Sorry, \(e)"); return
            }
            history.append(["role": "assistant", "content": text])
            emit(["event": "answer", "text": text])
            say(plain(text))
            if CommandLine.arguments.contains("--once") {
                speaking?.waitUntilExit()
                exit(0)
            }
        } catch {
            emit(["event": "error", "text": "the Strata server did not answer: \(error.localizedDescription)"])
            say("The Strata server did not answer.")
        }
    }

    func hush() {
        speaking?.terminate()
        speaking = nil
    }

    func say(_ text: String) {
        speaking?.terminate()
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/say")
        p.arguments = (voice.isEmpty ? [] : ["-v", voice]) + [text]
        try? p.run()
        speaking = p
        // a reading that outlives its words (macOS's audio output broken: coreaudiod's 'what') is ended and said, so
        // stuck `say`s do not pile up; about 2.5 words a second, and 20 s more
        let limit = 20 + Double(text.split(separator: " ").count) / 2.5
        Task {
            try? await Task.sleep(nanoseconds: UInt64(limit * 1e9))
            if p.isRunning {
                p.terminate()
                emit(["event": "error", "text": "the speakers did not play the answer: macOS's sound output is stuck (sudo killall coreaudiod, or restart the Mac)"])
            }
        }
    }

    /// While the answer is read aloud the microphone hears it: no wake phrase is taken from it.
    var isSpeaking: Bool { speaking?.isRunning ?? false }
}

func selftest() {
    let p = words("hey strata")
    assert(wakeEnd(words("Hey Strata, what is 2+2?"), p) == 2)
    assert(wakeEnd(words("hey strada what time"), p) == 2)               // one letter off
    assert(wakeEnd(words("hey stratas weather"), p) == 2)                // a letter more
    assert(wakeEnd(words("Play Strata what is the capital"), p) == 2)    // "Hey" misheard
    assert(wakeEnd(words("Strata, what time is it"), p) == 1)
    assert(wakeEnd(words("hey there"), p) == nil)
    assert(wakeEnd(words("the rock strata here are old"), p) == nil)     // mid-sentence: not a wake
    let h = words("heyden")
    for (said, end) in [("Heyden, what time is it", 1), ("Hayden what time", 1), ("Haydn, the weather", 1), ("hey den what time", 2)] {
        assert(wakeEnd(words(said), h) == end, said)
    }
    assert(wakeEnd(words("I read Haydn yesterday"), h) == nil)         // not the first word: no wake
    assert(plain("**Paris** is `the` capital.\n- one\n```py\nx=1\n```") == "Paris is the capital.\none\n (code shown in the chat) ")
    print("selftest ok")
}

@main
struct StrataVoice {
    /// Core Audio wants the main run loop running while the microphone starts (a plain async main gave it none:
    /// AVAudioEngine.start failed with 'what', 2003329396): the work runs in a task, the main thread runs the loop.
    static func main() {
        Task { await run(); exit(0) }
        RunLoop.main.run()
    }

    static func run() async {
        if CommandLine.arguments.contains("--selftest") { selftest(); return }
        if CommandLine.arguments.contains("--permissions") {
            // the "Allow microphone & speech" button: macOS asks (for Strata.app, the process that started this one);
            // once refused it does not ask again, so its System Settings page opens instead
            let mic = await AVCaptureDevice.requestAccess(for: .audio)
            let speech = await withCheckedContinuation { c in SFSpeechRecognizer.requestAuthorization { c.resume(returning: $0 == .authorized) } }
            if !mic, let u = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone") { NSWorkspace.shared.open(u) }
            emit(["event": "permissions", "microphone": mic, "speech": speech])
            return
        }
        let voice = Voice(port: arg("--port", "8090"), key: arg("--key", ProcessInfo.processInfo.environment["STRATA_API_KEY"] ?? ""),
                          phrase: arg("--phrase", "hey strata"), voice: arg("--voice", ""))
        if CommandLine.arguments.contains("--stdin-exit") {
            // started by strata-server: a JSON line a word from it ({"ask": "page"|"self"}, {"say": text},
            // {"hush": true}); its end closes this pipe, and this helper ends with it
            Thread.detachNewThread {
                while let line = readLine() {
                    guard let v = (try? JSONSerialization.jsonObject(with: Data(line.utf8))) as? [String: Any] else { continue }
                    Task {
                        if let d = v["ask"] as? String { await voice.direct(d) }
                        if let t = v["say"] as? String, !t.isEmpty { await voice.say(plain(t)) }
                        if v["hush"] as? Bool == true { await voice.hush() }
                    }
                }
                exit(0)
            }
        }
        let file = arg("--file", "")
        let locale = Locale(identifier: arg("--locale", Locale.current.identifier))

        let micOK = file.isEmpty ? await AVCaptureDevice.requestAccess(for: .audio) : true
        guard micOK else {
            emit(["event": "error", "permission": "microphone", "text": "no microphone access: press Allow microphone, or System Settings > Privacy & Security > Microphone > Strata"]); exit(1)
        }
        // a test from a file asks nothing (no prompt can show over ssh, and the request then never returns)
        let speechOK = !file.isEmpty ? SFSpeechRecognizer.authorizationStatus() == .authorized
            : await withCheckedContinuation { c in SFSpeechRecognizer.requestAuthorization { c.resume(returning: $0 == .authorized) } }
        emit(["event": "starting", "speech_permission": speechOK])
        // SpeechAnalyzer's dictation runs without it (tried: a refused one transcribes); asked so the prompt is said once
        guard let loc = await DictationTranscriber.supportedLocale(equivalentTo: locale) else {
            emit(["event": "error", "text": "Apple's dictation has no model for \(locale.identifier)"]); exit(1)
        }
        let transcriber = DictationTranscriber(locale: loc, contentHints: [], transcriptionOptions: [.punctuation],
                                               reportingOptions: [.volatileResults, .frequentFinalization], attributeOptions: [])
        do {
            if let install = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
                emit(["event": "downloading", "text": "Apple's dictation model for \(loc.identifier)"])
                try await install.downloadAndInstall()
            }
        } catch {
            emit(["event": "error", "text": "the dictation model did not install: \(error.localizedDescription)"]); exit(1)
        }

        let audioFile = file.isEmpty ? nil : try? AVAudioFile(forReading: URL(fileURLWithPath: file))
        let micFormat = audioFile?.processingFormat
            ?? AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 48000, channels: 1, interleaved: false)!
        guard let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber], considering: micFormat),
              let converter = AVAudioConverter(from: micFormat, to: format) else {
            emit(["event": "error", "text": "no audio format for speech recognition"]); exit(1)
        }
        let analyzer = SpeechAnalyzer(modules: [transcriber], options: SpeechAnalyzer.Options(priority: .userInitiated, modelRetention: .lingering))
        let context = AnalysisContext()
        context.contextualStrings[.general] = [arg("--phrase", "hey strata"), "Strata"]
        try? await analyzer.setContext(context)
        do { try await analyzer.prepareToAnalyze(in: format) } catch {
            emit(["event": "error", "text": "speech recognition did not start: \(error.localizedDescription)"]); exit(1)
        }
        let (inputs, feed) = AsyncStream.makeStream(of: AnalyzerInput.self)
        let convert: (AVAudioPCMBuffer) -> Void = { buffer in
            let ratio = format.sampleRate / micFormat.sampleRate
            guard let out = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 32) else { return }
            var given = false
            var err: NSError?
            converter.convert(to: out, error: &err) { _, status in
                if given { status.pointee = .noDataNow; return nil }
                given = true; status.pointee = .haveData; return buffer
            }
            if out.frameLength > 0 { feed.yield(AnalyzerInput(buffer: out)) }
        }
        try? await analyzer.start(inputSequence: inputs)
        if let f = audioFile {
            // the recording, then 3 s of silence (the pause that ends the question)
            while let b = AVAudioPCMBuffer(pcmFormat: f.processingFormat, frameCapacity: 4096), (try? f.read(into: b)) != nil, b.frameLength > 0 {
                convert(b)
            }
            if let silence = AVAudioPCMBuffer(pcmFormat: f.processingFormat, frameCapacity: AVAudioFrameCount(f.processingFormat.sampleRate * 3)) {
                silence.frameLength = silence.frameCapacity
                convert(silence)
            }
        } else {
            let mic = Mic()
            mic.onBuffer = { b in
                if b.format == micFormat { convert(b) }       // the settings above: always mono float32 48 kHz
            }
            do { try mic.start() } catch {
                emit(["event": "error", "text": "the microphone did not start: \(error.localizedDescription)",
                      "mic_permission": AVCaptureDevice.authorizationStatus(for: .audio).rawValue]); exit(1)
            }
            micKeep = mic
        }
        emit(["event": "listening", "phrase": arg("--phrase", "hey strata"), "locale": loc.identifier])

        Task {
            while true {
                try? await Task.sleep(nanoseconds: 300_000_000)
                await voice.tick()
            }
        }
        do {
            let verbose = CommandLine.arguments.contains("--verbose")
            for try await r in transcriber.results {
                if verbose { emit(["event": "heard", "text": String(r.text.characters), "final": r.isFinal]) }
                if await voice.isSpeaking { continue }
                await voice.heard(String(r.text.characters), final: r.isFinal)
            }
        } catch {
            emit(["event": "error", "text": "speech recognition stopped: \(error.localizedDescription)"]); exit(1)
        }
    }
}
