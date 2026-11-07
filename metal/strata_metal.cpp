// strata-metal: Strata's engine line protocol (serve/server.py <-> engine) over llama.cpp's Metal backend.
//
// macOS / Apple Silicon only.  The server starts it as `strata-metal --serve <args>` and talks to it over stdin/stdout
// exactly as it talks to the CUDA engine (src/program/generate.cpp): INFO/READY at start, then GEN, GENI, STOP, SAVE,
// RESTORE, VRAM and QUIT.  A protocol adapter, not a port of the CUDA engine: that engine's kernels and expert
// cache are CUDA code, and llama.cpp already runs this model on Metal (docs/MACOS.md).
//
// The model is the original GGUF (no Strata pack): llama.cpp's qwen4exp architecture, every tensor in unified memory.
//
// Conversation reuse.  The gated-delta-net layers keep a recurrent state that cannot be cut back to any position, so a
// request whose ids leave the held ones goes back to a checkpoint: the recurrent part of the state (the attention KV
// is cut with seq_rm), taken while reading a prompt just before each <|im_start|> - the message boundaries a chat
// template's re-render leaves intact.  Never a state that does not match: the held ids are compared by their content
// (a token id, or for an image's <|image_pad|> cells the picture's hash and the row), no checkpoint -> the whole prompt.
//
// Images (GENI): strata-vision's records, placed at the prompt's <|image_pad|> runs with the CUDA engine's M-RoPE
// positions (t = p, h = p + y, w = p + x; the text after an image goes on at p + max(nx, ny)).
//
// Speculative decoding (--mtp <MTP-only GGUF>, or --mtp self for a GGUF that carries its own MTP layer): the checkpoint's own MTP head drafts --spec tokens, the trunk checks
// them in one pass (llama.cpp's draft-mtp, #29761); the output is the trunk's, drafts only decide how many tokens a
// pass yields.  metal/mtp_gguf.py makes the file from the checkpoint.
// --eagle3 <gguf> and --dflash <gguf> draft with a separate small model instead (DFlash: a block of tokens in one pass
// from 5 of the target's layers; docs/MACOS.md, "GPT-OSS 120B with a DFlash draft").
//
// Batch slots (--batch N, the server's "parallel"): BGEN admissions into sequences 1..N, decoded together a token a
// window between commands (BT / BDONE); BSTOP ends one.  Not here: the expert cache (VRAM), BYIELD (never gives way).
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "json-schema-to-grammar.h"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <set>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifndef STRATA_VERSION
#define STRATA_VERSION "0"
#endif
#ifndef STRATA_LLAMA_COMMIT
#define STRATA_LLAMA_COMMIT "unknown"
#endif

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
const bool g_phases = std::getenv("STRATA_PHASES") != nullptr;   // per-phase timings of the MTP loop, to stderr

// ---------------------------------------------------------------------------------------------- stdout / stdin
// The protocol owns the real stdout.  llama.cpp's common library prints plain LOG() lines to stdout, so main() moves fd 1
// onto stderr and the protocol writes to its own copy of the original stdout: no library line can reach the server.
FILE* g_proto = stdout;

void out(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void out(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(g_proto, fmt, ap);
    va_end(ap);
    std::fflush(g_proto);
}

// STOP has to reach a running request, so stdin is read on its own thread.  A GEN line clears the stop flag before
// it is queued: a STOP the server sent for an earlier request (after that one's DONE) cannot cancel the next.
std::atomic<bool> g_stop{false};
// VIP 1 / VIP 0 (the server: a client with an engine of its own is being answered, docs/CLIENT_ENGINES.md): this
// engine gives the GPU way between its passes until VIP 0, and reads prompts in pieces only meanwhile (a piece at a
// time costs Qwen3.6 a third of its prompt speed: 17 -> 27 s for 34K tokens).  ponytail: at most 30 s a wait, so a lost VIP 0 never hangs it
std::atomic<bool> g_vip{false};
void vip_wait() {
    const auto t = Clock::now();
    while (g_vip && ms_since(t) < 30000) std::this_thread::sleep_for(std::chrono::milliseconds(2));
}
// STRATA_TRACE=1: why a prompt is read again or not (each checkpoint taken or skipped, each rollback) on stderr, the
// run config's log.  Off by default: debug-level lines (RFC 5424 severity 7); llama.cpp hides its own "full prompt
// re-processing" line the same way, at trace verbosity.
static const bool g_trace = [] { const char* v = std::getenv("STRATA_TRACE"); return v && *v && std::strcmp(v, "0") != 0; }();
#define TRACE(...) do { if (g_trace) { std::fprintf(stderr, "strata-metal: trace " __VA_ARGS__); std::fputc('\n', stderr); } } while (0)
// a compute pass failed (llama.cpp's -3: on a Mac, Metal ran out of memory): its backend stays in that error state and
// every later pass fails at once ("recreate the backend to recover"; 2026-11-05: each request answered 503 in 0 s until
// the server was stopped), so the engine ends after this request and the server starts a fresh one at the next
static bool g_gpu_failed = false;
std::unique_ptr<std::atomic<bool>[]> g_bstop;   // BSTOP <slot>: that slot ends at its next window
int g_slots = 0;
std::mutex g_mu;
std::condition_variable g_cv;
std::deque<std::string> g_lines;

void read_stdin() {
    std::string line;
    int c;
    for (;;) {
        line.clear();
        while ((c = std::fgetc(stdin)) != EOF && c != '\n') line.push_back((char) c);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (c == EOF && line.empty()) line = "QUIT";      // the server is gone
        if (line == "STOP") { g_stop = true; continue; }
        if (line.rfind("VIP ", 0) == 0) { g_vip = line.size() > 4 && line[4] == '1'; continue; }
        if (line.rfind("BSTOP ", 0) == 0) {
            const int b = std::atoi(line.c_str() + 6);
            if (b >= 0 && b < g_slots) g_bstop[(size_t) b] = true;
            continue;
        }
        // BYIELD <slot>: a long prompt read may give way to a shorter waiting one (#656).  This engine reads a prompt
        // to its end; the server treats that as "it did not give way".
        if (line.rfind("BYIELD", 0) == 0) continue;
        if (line.rfind("GEN", 0) == 0 || line.rfind("BGEN", 0) == 0) g_stop = false;
        { std::lock_guard<std::mutex> lk(g_mu); g_lines.push_back(line); }
        g_cv.notify_one();
        if (c == EOF) return;
    }
}

// The next command; with `wait` false only one that is there already (batch slots decode in the meantime).
bool next_line(std::string& s, bool wait = true) {
    std::unique_lock<std::mutex> lk(g_mu);
    if (wait) g_cv.wait(lk, [] { return !g_lines.empty(); });
    if (g_lines.empty()) return false;
    s = std::move(g_lines.front());
    g_lines.pop_front();
    return true;
}

// ---------------------------------------------------------------------------------------------- options
llama_token g_image_pad = 248056;   // qwen4exp.ple.image_token_id; --image-pad-id for another model (the tests)
struct Options {
    std::string gguf, mtp;
    std::string eagle3;                                // --eagle3 <gguf>: an EAGLE3 draft model instead of --mtp
    std::string dflash;                                // --dflash <gguf>: a DFlash draft model (a block per pass)
    int64_t max_context = 32768;
    std::string kv = "int8";
    std::vector<llama_token> eos = {248044, 248046};   // <|endoftext|>, <|im_end|>: the CUDA engine's --eos-ids
    int n_batch = 2048;
    int checkpoints = 8;
    int checkpoint_tail = 0;                           // --checkpoint-tail N: the prompt's last N tokens in small chunks
    int prefill_share = 512;                           // --prefill-share N: while slots decode, a prompt in N-token pieces, windows between (0: whole)
    double decode_share = 0.25;                        // --decode-share F: after each piece, windows for F x the piece's time
    int64_t cache_mib = 0;                             // --conversation-cache-mib N: displaced conversations kept in RAM
    int cache_slots = 16;                              // --conversation-cache-slots K: at most K of them
    int64_t cache_min_tokens = 0;                      // --conversation-cache-min-tokens N: none shorter than N
    int threads = 0;
    int spec = 3;                                      // drafted tokens per pass with --mtp, --eagle3 or --dflash
    int batch = 0;                                     // --batch N: batch slots (the server's "parallel")
    int poll = -1;                                     // --poll N: ggml's CPU busy-wait level (llama.cpp default 50)
    // --ablate <file>: directions projected out of every layer's output while the model runs (directional ablation;
    // the GGUF is not changed; metal/patches/core).  Raw float32, layers x k x n_embd, each layer's k rows orthogonal
    // and of length <= 1 (a row of length sqrt(w) removes w of its direction: strata-ablate's per-layer weights), a
    // layer of zeros left alone.  --ablate-layers: the file's layer count (default: the model's); k follows from it.
    // --ablate-mid: also right after each layer's attention, before its experts read the stream (metal/patches/core/0004)
    std::string ablate;
    int ablate_layers = 0;
    float ablate_scale = 1.0f;
    bool ablate_mid = false;
    // --kv-unified: one K/V pool shared by all sequences (llama.cpp's kv_unified) instead of a fixed share each, so an
    // agent can take up to --max-context while the others use less (docs/CONTEXT_POOL.md); --kv-pool N: the pool's
    // cells (default --max-context x sequences)
    bool kv_unified = false;
    int64_t kv_pool = 0;
    bool plan = false;                                 // --plan: print the memory this model needs (JSON) and exit
};

bool parse_ids(const char* s, std::vector<llama_token>& v) {
    v.clear();
    while (*s) {
        char* e = nullptr;
        long x = std::strtol(s, &e, 10);
        if (e == s || x < 0) return false;
        v.push_back((llama_token) x);
        if (*e && *e != ',') return false;
        s = *e == ',' ? e + 1 : e;
    }
    return !v.empty();
}

bool parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "strata-metal: %s needs a value\n", name); std::exit(2); }
            return argv[++i];
        };
        if (a == "--serve") continue;
        else if (a == "--plan") o.plan = true;
        else if (a == "--gguf") o.gguf = val("--gguf");
        else if (a == "--mtp") o.mtp = val("--mtp");
        else if (a == "--eagle3") o.eagle3 = val("--eagle3");
        else if (a == "--dflash") o.dflash = val("--dflash");
        else if (a == "--spec") o.spec = std::atoi(val("--spec"));
        else if (a == "--batch" || a == "--slots") o.batch = std::max(0, std::min(16, std::atoi(val("--batch"))));
        else if (a == "--max-context") o.max_context = std::atoll(val("--max-context"));
        else if (a == "--kv") o.kv = val("--kv");
        else if (a == "--prefill-batch") o.n_batch = std::atoi(val("--prefill-batch"));
        else if (a == "--checkpoints") o.checkpoints = std::max(0, std::atoi(val("--checkpoints")));
        else if (a == "--prefill-share") o.prefill_share = std::max(0, std::atoi(val("--prefill-share")));
        else if (a == "--decode-share") o.decode_share = std::max(0.0, std::atof(val("--decode-share")));
        else if (a == "--checkpoint-tail") o.checkpoint_tail = std::max(0, std::atoi(val("--checkpoint-tail")));
        else if (a == "--conversation-cache-mib") o.cache_mib = std::max(0LL, std::atoll(val("--conversation-cache-mib")));
        else if (a == "--conversation-cache-slots") o.cache_slots = std::max(1, std::atoi(val("--conversation-cache-slots")));
        else if (a == "--conversation-cache-min-tokens") o.cache_min_tokens = std::max(0LL, std::atoll(val("--conversation-cache-min-tokens")));
        else if (a == "--threads") o.threads = std::atoi(val("--threads"));
        else if (a == "--poll") o.poll = std::max(0, std::min(100, std::atoi(val("--poll"))));
        else if (a == "--image-pad-id") g_image_pad = (llama_token) std::atoi(val("--image-pad-id"));
        else if (a == "--ablate") o.ablate = val("--ablate");
        else if (a == "--ablate-layers") o.ablate_layers = std::atoi(val("--ablate-layers"));
        else if (a == "--ablate-scale") o.ablate_scale = (float) std::atof(val("--ablate-scale"));
        else if (a == "--ablate-mid") o.ablate_mid = true;
        else if (a == "--kv-unified") o.kv_unified = true;
        else if (a == "--kv-pool") o.kv_pool = std::atoll(val("--kv-pool"));
        else if (a == "--eos-ids") {
            if (!parse_ids(val("--eos-ids"), o.eos)) { std::fprintf(stderr, "strata-metal: bad --eos-ids\n"); return false; }
        } else if (a.rfind("--", 0) == 0) {
            // a flag of the CUDA engine (a shared config): said once, not fatal.  Its value, if any, is skipped too.
            std::fprintf(stderr, "strata-metal: %s is not used by the Metal engine (ignored)\n", a.c_str());
            if (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) ++i;
        }
    }
    if (o.gguf.empty()) { std::fprintf(stderr, "strata-metal: --gguf <the model's first GGUF shard> is required\n"); return false; }
    if (o.max_context < 512) { std::fprintf(stderr, "strata-metal: --max-context must be 512 or more\n"); return false; }
    if (!o.mtp.empty() + !o.eagle3.empty() + !o.dflash.empty() > 1) {
        std::fprintf(stderr, "strata-metal: --mtp, --eagle3 and --dflash: only one of them\n");
        return false;
    }
    if (o.spec < 1 || o.spec > 16) { std::fprintf(stderr, "strata-metal: --spec takes 1 to 16 drafted tokens\n"); return false; }
    if (o.kv != "int8" && o.kv != "q4_0" && o.kv != "k8v4" && o.kv != "f16") {
        std::fprintf(stderr, "strata-metal: --kv takes int8, q4_0, k8v4 or f16\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------- images
constexpr uint64_t kImageBit = 1ull << 63;

uint64_t fnv1a(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const auto* b = static_cast<const unsigned char*>(p);
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

// What one request's prompt is: the ids, each cell's content key (an id, or an image row), its M-RoPE position
// (t, h, w; text has t = h = w) and its embedding row (images only), and the position the output goes on at.
struct Prompt {
    std::vector<llama_token> ids;
    std::vector<uint64_t> keys;
    std::vector<std::array<llama_pos, 3>> pos;
    std::vector<const float*> rows;
    std::vector<float> data;
    llama_pos next = 0;
};

// The CUDA engine's GENI placement (src/program/generate.cpp): "" or why the request is refused.  `mrope`: 2-D image
// positions (qwen4exp); a model without M-RoPE gives the rows ordinary consecutive positions, as llama.cpp's mtmd does.
std::string place_images(Prompt& P, const std::string& path, int n_embd, bool mrope) {
    struct Img { int64_t n, nx, ny; size_t off; uint64_t hash; };
    std::vector<Img> imgs;
    if (!path.empty()) {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return "cannot open " + path;
        std::string err;
        for (;;) {
            int32_t hdr[5];
            const size_t got = std::fread(hdr, sizeof(int32_t), 5, f);
            if (got == 0) break;
            if (got != 5 || hdr[0] != 0x31455653 || hdr[1] < 1 || hdr[2] < 1 || hdr[3] < 1 ||
                (int64_t) hdr[2] * hdr[3] != hdr[1] || hdr[4] != n_embd) {
                err = "bad embeddings file (expected strata-vision records of width " + std::to_string(n_embd) + ")";
                break;
            }
            const size_t off = P.data.size(), cnt = (size_t) hdr[1] * (size_t) hdr[4];
            P.data.resize(off + cnt);
            if (std::fread(P.data.data() + off, sizeof(float), cnt, f) != cnt) { err = "short embeddings file"; break; }
            const int64_t grid[3] = {hdr[1], hdr[2], hdr[3]};
            imgs.push_back({hdr[1], hdr[2], hdr[3], off,
                            fnv1a(P.data.data() + off, cnt * sizeof(float), fnv1a(grid, sizeof grid))});
        }
        std::fclose(f);
        if (!err.empty()) return err;
    }
    const size_t n = P.ids.size();
    P.keys.resize(n);
    P.pos.resize(n);
    P.rows.assign(n, nullptr);
    llama_pos p = 0;
    size_t i = 0, k = 0;
    while (i < n) {
        if (P.ids[i] != g_image_pad || imgs.empty()) {
            P.keys[i] = (uint64_t) (uint32_t) P.ids[i];
            P.pos[i] = {p, p, p};
            ++p;
            ++i;
            continue;
        }
        if (k >= imgs.size()) return "the prompt has more images than the embeddings file";
        const Img& im = imgs[k++];
        for (int64_t j = 0; j < im.n; ++j)
            if (i + (size_t) j >= n || P.ids[i + (size_t) j] != g_image_pad)
                return "image " + std::to_string(k) + " has " + std::to_string((long long) im.n) +
                       " rows but fewer <|image_pad|> tokens";
        for (int64_t j = 0; j < im.n; ++j) {
            const llama_pos y = (llama_pos) (j / im.nx), x = (llama_pos) (j % im.nx);
            P.keys[i + (size_t) j] = kImageBit | ((im.hash + (uint64_t) j * 0x9E3779B97F4A7C15ull) >> 1);
            const llama_pos q = mrope ? p : p + (llama_pos) j;
            P.pos[i + (size_t) j] = mrope ? std::array<llama_pos, 3>{p, p + y, p + x} : std::array<llama_pos, 3>{q, q, q};
            P.rows[i + (size_t) j] = P.data.data() + im.off + (size_t) j * (size_t) n_embd;
        }
        i += (size_t) im.n;
        p += mrope ? (llama_pos) std::max(im.nx, im.ny) : (llama_pos) im.n;
    }
    if (k != imgs.size()) return "the embeddings file has more images than the prompt";
    if (!imgs.empty() && P.ids.back() == g_image_pad) return "the prompt cannot end in an image";
    P.next = p;
    return "";
}

// ---------------------------------------------------------------------------------------------- the engine
struct Checkpoint {
    size_t n;                                 // cells the state covers: live[0, n)
    llama_pos pos;                            // the position of cell n (the first one cut)
    std::vector<uint8_t> tgt, dft, spec;      // recurrent part of the trunk, the draft context's, the drafter's
    bool inside = false;                      // inside a message (a chunk's end), not at its first token
};

// What one sequence of the context holds: seq 0 is the solo path's (GEN, with the MTP drafts), seq b + 1 batch slot b's.
struct Seq {
    std::vector<llama_token> live;            // the cells it holds
    std::vector<uint64_t> keys;               // ... and their content keys
    std::deque<Checkpoint> ckpts;             // ascending n
};

// The conversation cache (--conversation-cache-mib): a sequence's whole state (its K/V, the recurrent part and its
// checkpoints) kept in RAM when another conversation takes the sequence, given back when a prompt starts with it
// again.  The same exact content keys as the live sequences decide a match, so nothing unrelated is ever restored.
struct Parked {
    Seq seq;
    std::vector<uint8_t> state;               // llama_state_seq_get_data: the sequence as it was
    size_t bytes = 0;                         // the state, the checkpoints and the keys
    uint64_t used = 0;                        // the park clock when it was parked (the oldest goes first)
};

// A batch slot decoding between requests (after its BGEN admission): one token a window, its own sampler.
struct Slot {
    bool active = false;
    common_sampler* smpl = nullptr;
    llama_token pending = 0;                  // produced and sent, not yet fed
    llama_pos pos = 0;                        // its position
    int64_t produced = 0, max_new = 0;
    Clock::time_point t0;
};

struct Engine {
    Options o;
    common_params params;
    common_init_result_ptr init;
    common_speculative_init_result_ptr spec_init;
    common_speculative* spec = nullptr;
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    llama_context* ctx_dft = nullptr;
    const llama_vocab* vocab = nullptr;
    llama_token im_start = -1;
    int n_embd = 0;
    std::vector<Seq> seqs;
    std::vector<Slot> slots;
    std::vector<Parked> parked;
    uint64_t park_clock = 0;
    long evictions = 0;

    llama_memory_t mem() const { return llama_get_memory(ctx); }

    // --plan: the memory this model needs with these options, measured by llama.cpp without reading the weights
    // (no_alloc: the model's tensors and the context's buffers are laid out, not filled), as one JSON line on stdout.
    // The GPU's share (weights, KV cache, compute buffers) is what must fit Metal's working set (the device's
    // "total"); measured at two context sizes, the difference per token gives the largest context that fits.  The
    // draft model of --mtp/--eagle3/--dflash is not counted.  Setup, the app and the server's errors read it.
    bool plan(common_params& p, std::string& err) {
        const llama_model_params mp = common_model_params_to_llama(p);
        uint32_t n_train = 0;
        auto measure = [&](int64_t n_ctx, common_device_memory_data& gpu, common_device_memory_data& host) {
            p.n_ctx = (int32_t) (n_ctx * p.n_parallel);
            const llama_context_params cp = common_context_params_to_llama(p);
            std::vector<ggml_backend_dev_t> devs;
            uint32_t ngl = 0, nex = 0;
            const auto d = common_get_device_memory_data(p.model.path.c_str(), &mp, &cp, devs, ngl, n_train, nex, GGML_LOG_LEVEL_ERROR);
            gpu = d.size() > 1 ? d[0] : common_device_memory_data{};
            host = d.back();
        };
        try {
            common_device_memory_data g{}, h{}, g2{}, h2{};
            const int64_t c = o.max_context, c2 = c > 4096 ? 4096 : 8192;
            measure(c, g, h);
            measure(c2, g2, h2);
            const auto need = [](const common_device_memory_data& d) { return (int64_t) (d.model + d.context + d.compute); };
            const double per_token = (double) (need(g) - need(g2)) / (double) (c - c2);
            int64_t fits = per_token > 0 ? c + (int64_t) ((double) (g.total - need(g)) / per_token) : (need(g) <= g.total ? (int64_t) n_train : 0);
            fits = std::max<int64_t>(0, std::min<int64_t>(fits, n_train > 0 ? (int64_t) n_train : fits)) / 1024 * 1024;
            if (need(g) - (int64_t) g.context - (int64_t) g.compute > g.total) fits = 0;   // the weights alone do not fit
            out("{\"plan\": 1, \"context\": %lld, \"n_ctx_train\": %u, \"kv\": \"%s\", \"prefill_batch\": %d, "
                "\"gpu\": {\"model\": %zu, \"context\": %zu, \"compute\": %zu, \"limit\": %lld, \"free\": %lld}, "
                "\"host\": {\"model\": %zu, \"context\": %zu, \"compute\": %zu}, \"per_token\": %.1f, \"max_context\": %lld}\n",
                (long long) c, n_train, o.kv.c_str(), o.n_batch, g.model, g.context, g.compute, (long long) g.total,
                (long long) g.free, h.model, h.context, h.compute, per_token, (long long) fits);
            return true;
        } catch (const std::exception& e) {
            err = std::string("could not measure the model: ") + e.what();
            return false;
        }
    }

    bool load(std::string& err) {
        common_params& p = params;
        p.model.path = o.gguf;
        p.n_parallel = 1 + o.batch;               // seq 0 and one per batch slot, each with the whole context
        // default own cells per sequence.  "A copied sequence rolled back read wrong in the unified cache" (once
        // measured) was batching's arithmetic: two sequences decoded together differ from one alone in the last bits in
        // either mode, and the tests' made-up prompts tie their top tokens (docs/CONTEXT_POOL.md, 2026-11-03)
        p.kv_unified = o.kv_unified;
        p.n_ctx = (int32_t) (o.kv_unified && o.kv_pool > 0 ? o.kv_pool : o.max_context * p.n_parallel);
        p.n_batch = o.n_batch;
        p.n_ubatch = std::min(o.n_batch, 2048);   // 2048: prompts +9% on an M5 Max (llama-bench pp4096, 268 -> 293 tok/s)
        p.n_gpu_layers = 999;                     // everything in unified memory
        p.fit_params = false;
        p.warmup = false;
        p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        p.cache_type_k = o.kv == "f16" ? GGML_TYPE_F16 : o.kv == "q4_0" ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0;
        p.cache_type_v = o.kv == "f16" ? GGML_TYPE_F16 : o.kv == "int8" ? GGML_TYPE_Q8_0 : GGML_TYPE_Q4_0;
        if (o.threads > 0) p.cpuparams.n_threads = p.cpuparams_batch.n_threads = o.threads;
        if (o.poll >= 0) p.cpuparams.poll = p.cpuparams_batch.poll = (uint32_t) o.poll;
        postprocess_cpu_params(p.cpuparams, nullptr);       // what llama.cpp's argument parser does (thread counts)
        postprocess_cpu_params(p.cpuparams_batch, &p.cpuparams);
        postprocess_cpu_params(p.speculative.draft.cpuparams, &p.cpuparams);
        postprocess_cpu_params(p.speculative.draft.cpuparams_batch, &p.cpuparams_batch);
        const std::string& draft_path = !o.mtp.empty() ? o.mtp : !o.eagle3.empty() ? o.eagle3 : o.dflash;
        if (!draft_path.empty()) {
            // the draft: the model's MTP head, an EAGLE3 model (it reads 3 of the target's layers) or a DFlash model
            // (it reads 5 and drafts a whole block in one pass; metal/patches/core/0002 for gpt-oss-120b-DFlash's
            // attention biases); llama.cpp's common_speculative drives all three through the same begin / process /
            // draft / accept calls.  DFlash writes the target's features at their own positions, so unlike EAGLE3 it
            // has no lagging cell for rollback() to read again
            p.speculative.types = {!o.mtp.empty()    ? COMMON_SPECULATIVE_TYPE_DRAFT_MTP
                                   : !o.eagle3.empty() ? COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3
                                                       : COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH};
            // --mtp self (or the model's own file): the MTP layer inside the model's GGUF.  Without a draft path
            // llama.cpp builds the MTP context on the model already loaded; with one it would load the file again
            const bool own_mtp = !o.mtp.empty() && (o.mtp == "self" || o.mtp == o.gguf);
            if (!own_mtp) p.speculative.draft.mparams.path = draft_path;
            p.speculative.draft.n_max = o.spec;
            p.speculative.draft.n_gpu_layers = 999;
            p.speculative.draft.cache_type_k = p.cache_type_k;
            p.speculative.draft.cache_type_v = p.cache_type_v;
            const auto lim = common_speculative_get_output_limits(p.n_batch, p.n_parallel, common_speculative_n_max(&p.speculative));
            p.n_outputs_max = lim.total;
            p.n_outputs_max_per_seq = lim.per_seq;
        }
        if (o.plan) return plan(p, err);
        init = common_init_from_params(p);
        model = init ? init->model() : nullptr;
        ctx = init ? init->context() : nullptr;
        if (!model) { err = "could not load " + o.gguf; return false; }
        if (!ctx) { err = "could not create the context (too long a --max-context for this Mac's memory?)"; return false; }
        vocab = llama_model_get_vocab(model);
        n_embd = llama_model_n_embd(model);
        if (!o.ablate.empty() && !ablate(err)) return false;
        if (!draft_path.empty()) {
            common_params pd = common_base_params_to_speculative(p);
            spec_init = common_speculative_init_from_params(pd, model, ctx);
            ctx_dft = spec_init ? spec_init->context() : nullptr;
            if (!ctx_dft) { err = "could not load the draft model " + draft_path; return false; }
            p.speculative.draft.ctx_tgt = ctx;
            p.speculative.draft.ctx_dft = ctx_dft;
            spec = common_speculative_init(p.speculative, 1);   // the solo path's sequence (0) drafts
            if (!spec) { err = "could not start speculative decoding with " + draft_path; return false; }
        }
        seqs.resize((size_t) p.n_parallel);
        slots.resize((size_t) o.batch);
        llama_token t[4];
        // a message's first token, where the checkpoints go: Qwen's <|im_start|>, GPT-OSS's <|start|>, GLM's <|user|>
        for (const char* s : {"<|im_start|>", "<|start|>", "<|user|>"})   // Qwen, GPT-OSS (harmony), GLM
            if (im_start < 0 && llama_tokenize(vocab, s, (int32_t) std::strlen(s), t, 4, false, true) == 1) im_start = t[0];
        return true;
    }

    // --ablate: read the directions, check them, and hand them to llama.cpp (the target only: the MTP draft head is
    // left alone, a draft only has to be a good guess)
    bool ablate(std::string& err) {
        std::FILE* f = std::fopen(o.ablate.c_str(), "rb");
        if (!f) { err = "cannot read --ablate " + o.ablate; return false; }
        std::vector<float> v;
        float buf[4096];
        for (size_t n; (n = std::fread(buf, sizeof(float), 4096, f)) > 0;) v.insert(v.end(), buf, buf + n);
        const bool tail = std::fgetc(f) != EOF;               // a size that is not whole floats
        std::fclose(f);
        const int layers = o.ablate_layers > 0 ? o.ablate_layers : llama_model_n_layer(model);
        const size_t per = (size_t) layers * (size_t) n_embd;
        if (tail || v.empty() || per == 0 || v.size() % per != 0) {
            err = "--ablate " + o.ablate + ": " + std::to_string(v.size()) + " floats are not " + std::to_string(layers) +
                  " layers x k x " + std::to_string(n_embd) + " (set --ablate-layers to the file's layer count)";
            return false;
        }
        const int k = (int) (v.size() / per);
        // each steered layer's directions orthogonal and no longer than 1: x - sum (x . v) v then removes |v|^2 of each
        // (a weight), never more; a file of another layout (a wrong --ablate-layers, another model's width) fails here
        // instead of steering at random
        int steered = 0;
        for (int l = 0; l < layers; ++l) {
            const float* L = v.data() + (size_t) l * k * n_embd;
            if (std::all_of(L, L + (size_t) k * n_embd, [](float x) { return x == 0.0f; })) continue;
            ++steered;
            for (int a = 0; a < k; ++a)
                for (int b = a; b < k; ++b) {
                    double d = 0.0;
                    for (int e = 0; e < n_embd; ++e) d += (double) L[(size_t) a * n_embd + e] * L[(size_t) b * n_embd + e];
                    if (a == b ? (d <= 0.0 || d > 1.0 + 1e-3) : std::fabs(d) > 1e-3) {
                        err = "--ablate " + o.ablate + ": layer " + std::to_string(l) + "'s directions are not orthogonal of length <= 1 (v" +
                              std::to_string(a) + " . v" + std::to_string(b) + " = " + std::to_string(d) + ")";
                        return false;
                    }
                }
        }
        if (llama_set_adapter_ablation(ctx, v.data(), v.size(), n_embd, k, o.ablate_scale) != 0) {
            err = "llama.cpp refused --ablate " + o.ablate;
            return false;
        }
        if (o.ablate_mid) llama_set_adapter_ablation_mid(ctx, true);
        std::fprintf(stderr, "strata-metal: --ablate %s: %d of %d layers steered, %d directions each, scale %.2f%s\n",
                     o.ablate.c_str(), steered, layers, k, o.ablate_scale, o.ablate_mid ? ", after the attention too" : "");
        return true;
    }

    std::vector<uint8_t> state(llama_context* c, llama_seq_id sq) {
        std::vector<uint8_t> d(llama_state_seq_get_size_ext(c, sq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
        if (!d.empty() && llama_state_seq_get_data_ext(c, d.data(), d.size(), sq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0)
            d.clear();
        return d;
    }

    // At a message boundary, or `inside` one at a chunk's end: a new question after the same long message (a document,
    // a tool's result) goes back there, not to the message's start.  --checkpoints of each kind are kept, counted apart,
    // so a long message's inside ones never push out the boundary before it.
    void checkpoint(llama_seq_id sq, size_t n, llama_pos pos, bool inside) {
        if (o.checkpoints == 0) return;
        Checkpoint c{n, pos, state(ctx, sq), {}, {}, inside};
        if (c.tgt.empty()) {
            TRACE("seq %d: no checkpoint at %zu: the model keeps no partial state to save", sq, n);
            return;
        }
        TRACE("seq %d: checkpoint at %zu (pos %d, %s, %zu bytes)", sq, n, (int) pos, inside ? "inside a message" : "boundary", c.tgt.size());
        if (sq == 0 && ctx_dft) c.dft = state(ctx_dft, 0);
        if (sq == 0 && spec) common_speculative_get_state(spec, 0, c.spec);
        auto& ck = seqs[(size_t) sq].ckpts;
        while (!ck.empty() && ck.back().n >= n) ck.pop_back();
        ck.push_back(std::move(c));
        if (std::count_if(ck.begin(), ck.end(), [&](const Checkpoint& x) { return x.inside == inside; }) > o.checkpoints)
            ck.erase(std::find_if(ck.begin(), ck.end(), [&](const Checkpoint& x) { return x.inside == inside; }));
    }

    void clear(llama_seq_id sq) {
        llama_memory_seq_rm(mem(), sq, -1, -1);
        if (sq == 0 && ctx_dft) llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, -1, -1);
        seqs[(size_t) sq] = Seq{};
    }

    // Back to at most p held cells: the latest checkpoint at or below p, else nothing held.
    void rollback(llama_seq_id sq, size_t p) {
        Seq& S = seqs[(size_t) sq];
        while (!S.ckpts.empty() && S.ckpts.back().n > p) S.ckpts.pop_back();
        if (S.ckpts.empty()) TRACE("seq %d: back to %zu of %zu held: no checkpoint at or below it, reading the prompt again", sq, p, S.live.size());
        if (!S.ckpts.empty()) {
            const Checkpoint& c = S.ckpts.back();
            bool ok = llama_state_seq_set_data_ext(ctx, c.tgt.data(), c.tgt.size(), sq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) &&
                      llama_memory_seq_rm(mem(), sq, c.pos, -1);
            if (ok && sq == 0 && ctx_dft) {
                ok = (c.dft.empty() || llama_state_seq_set_data_ext(ctx_dft, c.dft.data(), c.dft.size(), 0,
                                                                    LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) &&
                     llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, c.pos, -1);
                if (ok && !c.spec.empty()) common_speculative_set_state(spec, 0, c.spec);
            }
            // EAGLE3's draft context lags one cell (its last pair waits for the next token), and llama.cpp keeps that
            // boundary in a checkpoint only for a recurrent target: for another (GPT-OSS) the restored draft would
            // jump a position.  One cell less of both, read again, gives EAGLE3 the boundary from fresh features.
            size_t n = c.n;
            if (ok && sq == 0 && !o.eagle3.empty() && c.spec.empty() && n > 0) {
                ok = llama_memory_seq_rm(mem(), sq, c.pos - 1, -1) &&
                     llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, c.pos - 1, -1);
                --n;
            }
            if (ok) {
                TRACE("seq %d: back to %zu of %zu held: restored the checkpoint at %zu", sq, p, S.live.size(), n);
                S.live.resize(n);
                S.keys.resize(n);
                return;
            }
            std::fprintf(stderr, "strata-metal: checkpoint at %zu could not be restored; reading the prompt again\n", c.n);
        }
        // plain K/V (no recurrent part, so no checkpoints): the cells past p cut, nothing to restore.  A recurrent
        // model's memory refuses a partial cut; a picture's cells have positions of their own; EAGLE3 lags a cell
        const bool text = std::none_of(S.live.begin(), S.live.begin() + (long) std::min(p, S.live.size()), [](llama_token t) { return t == g_image_pad; });
        if (S.ckpts.empty() && text && o.eagle3.empty() && p > 0 && llama_memory_seq_rm(mem(), sq, (llama_pos) p, -1) &&
            !(sq == 0 && ctx_dft && !llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, (llama_pos) p, -1))) {
            TRACE("seq %d: back to %zu of %zu held: cut (plain K/V)", sq, p, S.live.size());
            S.live.resize(p);
            S.keys.resize(p);
            return;
        }
        clear(sq);
    }

    size_t parked_bytes() const {
        size_t b = 0;
        for (const auto& x : parked) b += x.bytes;
        return b;
    }

    static size_t prefix_of(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
        size_t q = 0;
        while (q < a.size() && q < b.size() && a[q] == b[q]) ++q;
        return q;
    }

    // Before `sq` loses what it holds to another conversation (keeping `keep` cells of it): into the RAM cache, the
    // oldest parked ones evicted to make room.  Only when at least half of it goes - a new question after the same
    // document keeps most of it live and parks nothing.
    void park(llama_seq_id sq, size_t keep) {
        Seq& S = seqs[(size_t) sq];
        const size_t n = S.live.size();
        if (o.cache_mib == 0 || n == 0 || keep * 2 >= n) return;
        const size_t budget = (size_t) o.cache_mib << 20;
        if ((int64_t) n < o.cache_min_tokens) {
            out_cache("skipped", n, "below the minimum");
            return;
        }
        for (auto& x : parked)                // already there with as much or more of it
            if (prefix_of(x.seq.keys, S.keys) == n) { x.used = ++park_clock; return; }
        const size_t before = parked.size();  // and what this one supersedes goes
        parked.erase(std::remove_if(parked.begin(), parked.end(), [&](const Parked& x) { return prefix_of(x.seq.keys, S.keys) == x.seq.keys.size(); }), parked.end());
        if (parked.size() != before) TRACE("conversation cache: dropped %zu superseded by seq %d", before - parked.size(), sq);
        Parked e;
        const auto t0 = std::chrono::steady_clock::now();
        e.state.resize(llama_state_seq_get_size(ctx, sq));
        if (e.state.empty() || llama_state_seq_get_data(ctx, e.state.data(), e.state.size(), sq) == 0) return;
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        e.seq = S;                            // a copy: the caller decides what the sequence holds next
        for (auto& c : e.seq.ckpts) { c.dft.clear(); c.spec.clear(); }   // the drafter starts from new tokens anyway
        e.bytes = e.state.size() + e.seq.keys.size() * (sizeof(uint64_t) + sizeof(llama_token));
        for (const auto& c : e.seq.ckpts) e.bytes += c.tgt.size();
        if (e.bytes > budget) {
            out_cache("skipped", n, "larger than the budget");
            return;
        }
        while (!parked.empty() && ((int) parked.size() >= o.cache_slots || parked_bytes() + e.bytes > budget)) {
            parked.erase(std::min_element(parked.begin(), parked.end(), [](const Parked& a, const Parked& b) { return a.used < b.used; }));
            ++evictions;
        }
        e.used = ++park_clock;
        parked.push_back(std::move(e));
        out_cache("parked", n, sq, ms);
    }

    // The parked conversation that starts this prompt furthest, when it gives more than `p` cells: moved into `sq`
    // (what `sq` held is parked first).  The cells it gives, or `p` when none does.
    size_t restore(llama_seq_id sq, const std::vector<uint64_t>& keys, size_t p) {
        size_t best = parked.size(), q = p;
        for (size_t i = 0; i < parked.size(); ++i) {
            const size_t m = prefix_of(parked[i].seq.keys, keys);
            if (m > q + 16) { q = m; best = i; }
        }
        if (best == parked.size()) return p;
        Parked e = std::move(parked[best]);
        parked.erase(parked.begin() + (long) best);
        park(sq, p);
        llama_memory_seq_rm(mem(), sq, -1, -1);
        const auto t0 = std::chrono::steady_clock::now();
        if (llama_state_seq_set_data(ctx, e.state.data(), e.state.size(), sq) == 0) {
            clear(sq);
            std::fprintf(stderr, "strata-metal: conversation cache: could not restore %zu tokens; reading the prompt again\n", e.seq.live.size());
            return 0;
        }
        seqs[(size_t) sq] = std::move(e.seq);
        if (sq == 0 && ctx_dft) llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, -1, -1);
        out_cache("restored", seqs[(size_t) sq].live.size(), sq, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        return q;
    }

    // the CUDA engine's log line, which the server reads into /metrics ("conversation_cache"); ms: the state's copy
    void out_cache(const char* what, size_t n, llama_seq_id sq, double ms) {
        std::fprintf(stderr, "strata-metal: conversation cache: %s %zu tokens (seq %d) parked=%zu bytes=%zu evictions=%ld ms=%.1f\n", what, n, sq,
                     parked.size(), parked_bytes(), evictions, ms);
    }
    void out_cache(const char* what, size_t n, const char* why) {
        std::fprintf(stderr, "strata-metal: conversation cache: %s %zu tokens (%s) parked=%zu bytes=%zu\n", what, n, why, parked.size(), parked_bytes());
    }

    // Another sequence's whole state into `dst` (the solo request that a second request turned into a batch slot, and
    // back): the prompt it already read is not read again.  false: dst is left empty.
    bool copy_seq(llama_seq_id src, llama_seq_id dst) {
        park(dst, prefix_of(seqs[(size_t) dst].keys, seqs[(size_t) src].keys));
        clear(dst);
        if (o.kv_unified) {
            // one pool: the cells shared by both sequences (llama.cpp's own copy: nothing is written)
            llama_memory_seq_cp(mem(), src, dst, -1, -1);
            seqs[(size_t) dst] = seqs[(size_t) src];
            if (dst == 0) {
                for (auto& c : seqs[0].ckpts) { c.dft.clear(); c.spec.clear(); }
                if (ctx_dft) llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, -1, -1);
            }
            return true;
        }
        std::vector<uint8_t> buf(llama_state_seq_get_size(ctx, src));
        if (buf.empty() || llama_state_seq_get_data(ctx, buf.data(), buf.size(), src) == 0 ||
            llama_state_seq_set_data(ctx, buf.data(), buf.size(), dst) == 0) {
            clear(dst);
            return false;
        }
        seqs[(size_t) dst] = seqs[(size_t) src];
        if (dst == 0) {                           // the drafter knows none of it: it starts from the new tokens
            for (auto& c : seqs[0].ckpts) { c.dft.clear(); c.spec.clear(); }
            if (ctx_dft) llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, -1, -1);
        }
        return true;
    }

    // Cells [a, b) of P into sequence sq (and the drafter), the last with logits when `logits`.  false: err says why.
    // A batch holds one kind of entry (token ids or embedding rows), so text and a picture go in as separate batches.
    // `drafter`: the MTP drafter follows too (seq 0 only).  Not for a prompt with pictures: llama.cpp's MTP batch pairs
    // each cell with the next one's token, and a text cell before a picture's rows makes a batch it refuses ("entries
    // with both a token id and an embedding cannot be mixed").  The drafter then drafts without that prompt: weaker
    // drafts, the same answer.
    bool feed(llama_seq_id sq, const Prompt& P, size_t a, size_t b, bool logits, std::string& err, bool drafter = true) {
        Seq& S = seqs[(size_t) sq];
        for (size_t s = a; s < b;) {
            const bool image = P.rows[s] != nullptr;
            size_t e = s + 1;
            while (e < b && (P.rows[e] != nullptr) == image) ++e;
            common_batch batch(ctx);
            for (size_t i = s; i < e; ++i) {
                const bool lg = logits && i + 1 == b;
                if (image) {
                    const llama_pos p4[4] = {P.pos[i][0], P.pos[i][1], P.pos[i][2], 0};
                    batch.add_embd({P.rows[i], 1, (size_t) n_embd}, p4, sq, lg);
                } else {
                    batch.add(P.ids[i], P.pos[i][0], sq, lg);
                }
            }
            const int rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
            if (rc != 0) { g_gpu_failed |= rc == -3; err = "llama_process failed (" + std::to_string(rc) + ")"; return false; }
            if (spec && sq == 0 && drafter && !common_speculative_process(spec, batch)) {
                err = "the draft model could not follow the batch";
                return false;
            }
            S.live.insert(S.live.end(), P.ids.begin() + (long) s, P.ids.begin() + (long) e);
            S.keys.insert(S.keys.end(), P.keys.begin() + (long) s, P.keys.begin() + (long) e);
            s = e;
        }
        return true;
    }

    // grammar=<hex of GBNF> and grammar_trigger=<token id>: a lazy grammar, applied from that token on (the server's
    // tool-call grammar: only the tools' own names and arguments; thinking and text stay free).  A grammar that does
    // not parse is an error, here or in common_sampler_init (it throws): `err` says it, the sampler is null.
    common_sampler* sampler(const std::string& line_keys, std::string* err = nullptr) {
        common_params_sampling s = params.sampling;
        std::string grammar, schema;
        long long trigger = -1;
        s.temp = 0.0f;                        // absent keys: greedy, as the CUDA engine
        s.top_k = 20;
        s.top_p = 1.0f;
        s.min_p = 0.0f;
        s.penalty_last_n = 64;
        size_t i = 0;
        while (i < line_keys.size()) {
            size_t j = line_keys.find(' ', i);
            if (j == std::string::npos) j = line_keys.size();
            std::string kv = line_keys.substr(i, j - i);
            i = j + 1;
            size_t eq = kv.find('=');
            if (eq == std::string::npos) continue;
            std::string k = kv.substr(0, eq);
            const char* v = kv.c_str() + eq + 1;
            if (k == "temperature") s.temp = std::strtof(v, nullptr);
            else if (k == "top_p") s.top_p = std::strtof(v, nullptr);
            else if (k == "top_k") s.top_k = std::atoi(v);
            else if (k == "min_p") s.min_p = std::strtof(v, nullptr);
            else if (k == "penalty_repeat") s.penalty_repeat = std::strtof(v, nullptr);
            else if (k == "penalty_freq") s.penalty_freq = std::strtof(v, nullptr);
            else if (k == "penalty_present") s.penalty_present = std::strtof(v, nullptr);
            else if (k == "penalty_last_n") s.penalty_last_n = std::atoi(v);
            else if (k == "seed") s.seed = (uint32_t) std::strtoull(v, nullptr, 10);
            else if (k == "grammar_trigger") trigger = std::strtoll(v, nullptr, 10);
            else if (k == "logit_bias") {                    // <id>:<bias>,... (the server's bias::engine_key); -100: never
                const llama_token n_vocab = llama_vocab_n_tokens(vocab);
                for (const char* q = v; *q;) {
                    char* e = nullptr;
                    const long id = std::strtol(q, &e, 10);
                    if (e == q || *e != ':' || id < 0 || id >= n_vocab) {
                        if (err) *err = "logit_bias: expected <token id>:<bias>,... within the vocabulary";
                        return nullptr;
                    }
                    const float b = std::strtof(e + 1, &e);
                    s.logit_bias.push_back({(llama_token) id, b <= -100.0f ? -INFINITY : b});
                    q = *e == ',' ? e + 1 : e;
                }
            }
            else if (k == "grammar" || k == "json_schema") {   // hex: the line is split at spaces
                const std::string h(v);
                if (h.size() % 2 || h.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos) {
                    if (err) *err = k + ": not hex";
                    return nullptr;
                }
                std::string& to = k == "grammar" ? grammar : schema;
                for (size_t q = 0; q < h.size(); q += 2) to += (char) std::stoi(h.substr(q, 2), nullptr, 16);
            }
        }
        // json_schema=<hex of a JSON schema>: llama.cpp's converter makes it the grammar (a response_format the reply
        // can only meet), lazy from grammar_trigger as any grammar
        if (!schema.empty()) {
            if (!grammar.empty()) {
                if (err) *err = "grammar and json_schema: give one";
                return nullptr;
            }
            try {
                grammar = json_schema_to_grammar(common_json::parse(schema));
            } catch (const std::exception& e) {
                if (err) *err = std::string("json_schema: ") + e.what();
                return nullptr;
            }
            // lazy (from the thinking's end token): llama.cpp feeds the grammar its trigger token too, so the schema's
            // JSON comes after that token and the blank lines after it (else "empty grammar stack after accepting
            // </think>" ended the engine).  ponytail: the converter's own "root" renamed; a schema whose $ref points at
            // "#" itself would need its references renamed too
            if (trigger >= 0) {
                const size_t at = grammar.rfind("root ::=", 0) == 0 ? 0 : grammar.find("\nroot ::=");
                if (at == std::string::npos) {
                    if (err) *err = "json_schema: the grammar has no root rule";
                    return nullptr;
                }
                grammar.replace(at == 0 ? 0 : at + 1, 4, "json-root");
                grammar = "root ::= <[" + std::to_string(trigger) + "]> [ \\t\\n]{0,4} json-root\n" + grammar;   // bounded: a run of blanks never ends the reply
            }
            TRACE("json_schema: %zu bytes of grammar, %s", grammar.size(), trigger >= 0 ? "after the trigger token" : "from the first token");
        }
        if (s.top_k <= 0) s.top_k = 64;
        if (!grammar.empty()) {
            if (trigger >= llama_vocab_n_tokens(vocab)) {
                if (err) *err = "grammar_trigger " + std::to_string(trigger) + " is outside the vocabulary";
                return nullptr;
            }
            s.grammar = common_grammar(COMMON_GRAMMAR_TYPE_USER, grammar);
            s.grammar_lazy = trigger >= 0;
            if (trigger >= 0) s.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN, "", (llama_token) trigger});
        }
        try {
            common_sampler* smpl = common_sampler_init(model, s);
            if (!smpl && err) *err = "the sampler did not start (grammar?)";
            return smpl;
        } catch (const std::exception& e) {
            if (err) *err = std::string("grammar: ") + e.what();
            return nullptr;
        }
    }

    bool decoding() const {
        return std::any_of(slots.begin(), slots.end(), [](const Slot& x) { return x.active; });
    }

    //   GEN <max_new> [key=value ...] <id,id,...>   |   GENI <max_new> [key=value ...] <embeddings file> <id,id,...>
    // slot >= 0: BGEN <slot> ... / BGENI <slot> ... (the line passed here without "B" and the slot): the prompt and its
    // first token as a GEN 1, then BADM <slot> <1: it goes on in the slot's windows | 0: done>.
    void gen(const std::string& line, bool geni, int slot = -1, int64_t slot_max_new = 0) {
        const llama_seq_id sq = slot + 1;
        Seq& S = seqs[(size_t) sq];
        const size_t sp = line.rfind(' ');
        const size_t head = geni ? 5 : 4;
        Prompt P;
        char* endp = nullptr;
        const long long max_new = std::strtoll(line.c_str() + head, &endp, 10);
        auto refuse = [&](const std::string& why) {
            out("ERR %s\n", why.c_str());
        };
        if (sp == std::string::npos || sp < head || max_new < 0 || !parse_ids(line.c_str() + sp + 1, P.ids))
            return refuse("expected: GEN <max_new> <id,id,...> or GENI <max_new> <file> <id,id,...>");
        std::string keys_txt(static_cast<const char*>(endp), line.c_str() + sp), file;
        if (geni) {                                   // the embeddings file: the first word without an '='
            std::string rest;
            size_t i = 0;
            while (i < keys_txt.size()) {
                size_t j = keys_txt.find(' ', i);
                if (j == std::string::npos) j = keys_txt.size();
                const std::string w = keys_txt.substr(i, j - i);
                i = j + 1;
                if (w.empty()) continue;
                if (file.empty() && w.find('=') == std::string::npos) file = w;
                else rest += " " + w;
            }
            keys_txt = rest;
            if (file.empty()) return refuse("expected: GENI <max_new> <file> <id,id,...>");
        }
        {                                             // a grammar that does not parse: refused before the prompt is read
            std::string gerr;
            const bool constrained = keys_txt.find(" grammar=") != std::string::npos || keys_txt.find(" json_schema=") != std::string::npos;
            common_sampler* probe = constrained ? sampler(keys_txt, &gerr) : nullptr;
            if (probe) common_sampler_free(probe);
            if (!gerr.empty()) return refuse(gerr);
        }
        const size_t n = P.ids.size();
        if ((int64_t) n >= o.max_context)
            return refuse("the prompt (" + std::to_string(n) + " tokens) does not fit the context (" +
                          std::to_string(o.max_context) + ")");
        const llama_token n_vocab = llama_vocab_n_tokens(vocab);
        for (llama_token t : P.ids)
            if (t >= n_vocab) return refuse("token id " + std::to_string(t) + " is outside the vocabulary");
        const auto rope = llama_model_rope_type(model);
        if (const std::string e = place_images(P, file, n_embd, rope == LLAMA_ROPE_TYPE_MROPE || rope == LLAMA_ROPE_TYPE_IMROPE);
            !e.empty())
            return refuse(e);

        // reuse: the held cells that start the prompt - this sequence's, or an idle one's that holds more of it (copied
        // over); the last prompt token is always read again (the first pass)
        auto prefix = [&](const Seq& x) {
            size_t q = 0;
            while (q < x.keys.size() && q < n && x.keys[q] == P.keys[q]) ++q;
            return q;
        };
        size_t p = prefix(S);
        for (size_t other = 0; other < seqs.size(); ++other) {
            const bool busy = other > 0 && slots[other - 1].active;
            if ((llama_seq_id) other == sq || busy) continue;
            const size_t q = prefix(seqs[other]);
            if (q > p + 16 && copy_seq((llama_seq_id) other, sq)) {
                TRACE("seq %d: copied idle seq %zu, which holds %zu of the prompt's start (this one %zu)", sq, other, q, p);
                p = q;
            }
        }
        if (o.cache_mib > 0) p = restore(sq, P.keys, p);
        TRACE("seq %d: prompt %zu, the same start %zu of %zu held, %zu checkpoints", sq, n, p, S.live.size(), S.ckpts.size());
        p = std::min(p, n - 1);
        if (p < S.live.size()) {
            park(sq, p);
            rollback(sq, p);
        }
        const size_t resume = S.live.size();
        out("RESUME %zu\n", resume);

        std::string err;
        const auto t0 = Clock::now();
        const char* finish = "length";
        bool cancelled = false;
        const bool drafts = spec && sq == 0;
        // the prompt but its last token, in chunks of n_batch, each also ending at a message boundary (a checkpoint at
        // each chunk's end)
        size_t at = resume;
        while (at + 1 < n) {
            size_t end = std::min(n - 1, at + (size_t) o.n_batch);
            // --checkpoint-tail: the bulk in whole batches, the last N tokens in chunks of 256, each ending at a checkpoint:
            // a new question after the same document resumes within 256 tokens of where it differs (measured on Qwen3.6, a
            // 3,042-token prompt: 0.83 s to read a new question at batch 2048, 0.32 s at 256), and only the tail reads slower
            // ponytail: a fixed 256; a knob when another model wants other chunks
            if (o.checkpoint_tail > 0) {
                const size_t tail0 = n - 1 > (size_t) o.checkpoint_tail ? n - 1 - (size_t) o.checkpoint_tail : 0;
                end = at < tail0 ? std::min(end, tail0) : std::min(end, at + 256);
            }
            size_t b = at + 1;
            while (b < end && P.ids[b] != im_start) ++b;
            const bool boundary = b < end;
            if (boundary) end = b;
            // while other slots decode, the chunk in --prefill-share pieces, and after each the decoding slots' windows
            // for --decode-share of the piece's time (at least one): their agents go on writing instead of waiting for
            // the whole prompt (Qwen3-Coder, a 34K prompt read beside a writing agent: 0 tokens in 79 s before).  Text
            // only: an image's rows go in whole.
            for (size_t s0 = at; s0 < end;) {
                const size_t s1 = !geni && o.prefill_share > 0 && (decoding() || g_vip) ? std::min(end, s0 + (size_t) o.prefill_share) : end;
                vip_wait();
                const auto tp = Clock::now();
                if (!feed(sq, P, s0, s1, false, err, !geni)) { clear(sq); return refuse(err); }
                s0 = s1;
                if (!decoding()) continue;
                llama_synchronize(ctx);
                const auto tw = Clock::now();
                const double budget = ms_since(tp) * o.decode_share;
                do window(); while (decoding() && ms_since(tw) < budget);
            }
            at = end;
            // only where seq_rm from the cut's position keeps every held cell (an M-RoPE image's rows share one
            // position), and from the cell before it too (rollback's EAGLE3 rule cuts one more); a boundary is the cut's
            // own token (a chunk can end right before an <|im_start|>)
            // ponytail: the M-RoPE and EAGLE3 cases have no test model here (LFM2-350M's images are not M-RoPE)
            if (P.pos[at][0] > P.pos[at - 1][0] && (at < 2 || P.pos[at - 1][0] > P.pos[at - 2][0]))
                checkpoint(sq, at, P.pos[at][0], P.ids[at] != im_start);
            else
                TRACE("seq %d: no checkpoint at %zu: an image's rows share its position", sq, at);
            const double ms = ms_since(t0);
            out("PP %zu %zu %.0f %.1f\n", at, n, ms, (at - resume) / std::max(1e-3, ms / 1000.0));
            if (g_stop) { cancelled = true; finish = "cancel"; break; }
        }

        // decode: each pass reads the pending token (the prompt's last, then the last one produced) and the drafts
        const int64_t limit = slot >= 0 ? 1 : max_new;      // an admission produces the first token only
        int64_t produced = 0, drafted = 0, accepted = 0;
        double prompt_ms = ms_since(t0);
        const auto t1 = Clock::now();
        double ph[4] = {0, 0, 0, 0};                         // STRATA_PHASES: draft, target, catch-up, sample
        int ph_n = 0;
        common_sampler* smpl = nullptr;
        llama_token pending = P.ids[n - 1];
        llama_pos pending_pos = P.pos[n - 1][0];
        uint64_t pending_key = P.keys[n - 1];
        if (!cancelled) {
            smpl = sampler(keys_txt);
            if (drafts) common_speculative_begin(spec, 0, std::vector<llama_token>(P.ids.begin(), P.ids.end() - 1));
            llama_pos next = P.next;                         // where the output goes on
            std::vector<llama_token> draft;
            bool first = true, done = false;
            while (!done) {
                vip_wait();
                draft.clear();
                const int64_t room = std::min<int64_t>(limit - produced - 1, o.max_context - (int64_t) S.live.size() - 2);
                if (drafts && room > 0) {
                    common_speculative_get_draft_params(spec, 0) = {
                        /* .drafting = */ true, /* .n_max = */ (int32_t) std::min<int64_t>(room, o.spec),
                        /* .pos0 = */ pending_pos, /* .id_last = */ pending, /* .prompt = */ &S.live, /* .result = */ &draft};
                    const auto tp = Clock::now();
                    common_speculative_draft(spec);
                    if (g_phases) ph[0] += ms_since(tp);
                    // drafting wrote the drafts into the draft context: out again, the verify pass feeds it the
                    // trunk's own (speculative-simple does the same)
                    llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, pending_pos, -1);
                }
                common_batch batch(ctx);
                batch.add(pending, pending_pos, sq, true);
                for (size_t i = 0; i < draft.size(); ++i) batch.add(draft[i], next + (llama_pos) i, sq, true);
                const auto tv = Clock::now();
                const int rc_tgt = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
                const bool fail_tgt = rc_tgt != 0;
                g_gpu_failed |= rc_tgt == -3;
                if (g_phases && !fail_tgt) { llama_synchronize(ctx); ph[1] += ms_since(tv); }
                const auto tc = Clock::now();
                if (fail_tgt || (drafts && !common_speculative_process(spec, batch))) {
                    common_sampler_free(smpl);
                    clear(sq);
                    return refuse("the decode pass failed");
                }
                if (g_phases && ctx_dft) { llama_synchronize(ctx_dft); ph[2] += ms_since(tc); }
                if (first) { prompt_ms = ms_since(t0); first = false; }
                const auto ts = Clock::now();
                std::vector<llama_token> ids;
                try {                                 // a grammar's error ends this request, never the engine
                    ids = common_sampler_sample_and_accept_n(smpl, ctx, draft);
                } catch (const std::exception& e) {
                    common_sampler_free(smpl);
                    clear(sq);
                    return refuse(std::string("sampling: ") + e.what());
                }
                if (g_phases) { ph[3] += ms_since(ts); ++ph_n; }
                drafted += (int64_t) draft.size();
                accepted += (int64_t) ids.size() - 1;
                if (drafts) common_speculative_accept(spec, 0, (uint16_t) (ids.size() - 1));
                // the pending token and the accepted drafts are in the memory now; ids.back() is the next pending one
                S.live.push_back(pending);
                S.keys.push_back(pending_key);
                size_t take = 0;
                for (llama_token t : ids) {
                    out("T %d\n", t);
                    ++produced;
                    ++take;
                    if (std::find(o.eos.begin(), o.eos.end(), t) != o.eos.end() || llama_vocab_is_eog(vocab, t)) {
                        finish = "stop";
                        done = true;
                        break;
                    }
                    if (produced >= limit) { finish = "length"; done = true; break; }
                }
                for (size_t i = 0; i + 1 < take; ++i) { S.live.push_back(ids[i]); S.keys.push_back((uint64_t) (uint32_t) ids[i]); }
                const llama_pos keep = next + (llama_pos) take - 1;     // positions from here on were not emitted
                // the rejected drafts out again (the recurrent state keeps --spec snapshots for this, n_rs_seq).
                // With no speculation configured (GPT-OSS) nothing ever sits past `keep`, and seq_rm scans every
                // cache cell (131,072 a token at 128K): skip it then.  With drafts or MTP it runs as before (the
                // recurrent state's snapshots are rolled back through it).
                if (drafts &&
                    (!llama_memory_seq_rm(mem(), sq, keep, -1) ||
                     (drafts && !llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, keep, -1)))) {
                    common_sampler_free(smpl);
                    clear(sq);
                    return refuse("the rejected drafts could not be taken back out of the state");
                }
                pending = ids[take - 1];
                pending_key = (uint64_t) (uint32_t) pending;
                pending_pos = keep;
                next = keep + 1;
                if (!done && g_stop) { finish = "cancel"; done = true; }
                if (!done && (int64_t) S.live.size() + 2 >= o.max_context) { finish = "length"; done = true; }
            }
            if (slot < 0) {                                  // the solo path's sampler ends with its request
                common_sampler_free(smpl);
                smpl = nullptr;
            }
        }
        // the last token produced is not fed: the sequence holds the prompt and every token but the last, as the CUDA
        // engine's do (the server's slot_held) - the next prompt carries it, and is read from there with no rollback
        const size_t read_n = cancelled ? at - resume : n - resume;
        // DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <drafts accepted> <drafts offered> <reused>
        //      <hits> <lookups> <RAM blobs> <file blobs> <file MB> <prompt tokens read> <offloaded>
        if (g_phases && ph_n > 0)      // STRATA_PHASES=1: ms per verify pass, to stderr (the target and catch-up synced)
            std::fprintf(stderr, "PHASES passes %d draft %.2f target %.2f catchup %.2f sample %.2f ms/pass\n", ph_n,
                         ph[0] / ph_n, ph[1] / ph_n, ph[2] / ph_n, ph[3] / ph_n);
        out("DONE %lld %zu %.1f %.1f %s %lld %lld %zu 0 0 0 0 0.0 %zu 0\n", (long long) produced, n, prompt_ms,
            ms_since(t1), finish, (long long) accepted, (long long) drafted, resume, read_n);
        if (slot >= 0) {
            // the slot goes on when its first token is not the end and it may write more; it holds the prompt, and from
            // here every token it produces but the last (the server's slot_held)
            const bool cont = !cancelled && !g_stop && produced == 1 && slot_max_new > 1 && std::strcmp(finish, "length") == 0;
            if (cont) {
                Slot& X = slots[(size_t) slot];
                X = Slot{true, smpl, pending, pending_pos, 1, slot_max_new, Clock::now()};
                g_bstop[(size_t) slot] = false;
            } else if (smpl) {
                common_sampler_free(smpl);
            }
            out("BADM %d %d\n", slot, cont ? 1 : 0);
        }
    }

    void slot_done(int b, const char* finish) {
        Slot& X = slots[(size_t) b];
        out("BDONE %d %lld %s %.1f\n", b, (long long) X.produced, finish, ms_since(X.t0));
        common_sampler_free(X.smpl);
        X = Slot{};
    }

    // One batch window: every decoding slot's pending token in one pass, a token each (BT <slot> <id>), BDONE at its end.
    void window() {
        vip_wait();
        std::vector<int> who;
        for (int b = 0; b < (int) slots.size(); ++b) {
            if (!slots[(size_t) b].active) continue;
            if (g_bstop[(size_t) b]) { slot_done(b, "cancel"); continue; }
            who.push_back(b);
        }
        if (who.empty()) return;
        llama_batch batch = llama_batch_init((int32_t) who.size(), 0, 1);
        for (int b : who) {
            const int32_t i = batch.n_tokens++;
            batch.token[i] = slots[(size_t) b].pending;
            batch.pos[i] = slots[(size_t) b].pos;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = b + 1;
            batch.logits[i] = true;
        }
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            g_gpu_failed |= rc == -3;
            std::fprintf(stderr, "strata-metal: a batch window failed; its slots end\n");
            for (int b : who) { clear(b + 1); slot_done(b, "cancel"); }
            return;
        }
        for (size_t k = 0; k < who.size(); ++k) {
            const int b = who[k];
            Slot& X = slots[(size_t) b];
            Seq& S = seqs[(size_t) b + 1];
            // row k is this slot's (not sample_and_accept_n(smpl, ctx, std::vector<int>{k}, {}): llama_tokens is a
            // std::vector<int32_t> too, so that picks the overload that takes {k} as a one-token draft and samples row 0)
            llama_token t;
            try {                                     // a grammar's error ends this slot, never the engine
                t = common_sampler_sample(X.smpl, ctx, (int) k);
                common_sampler_accept(X.smpl, t, true);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "strata-metal: slot %d: sampling: %s; the slot ends\n", b, e.what());
                clear(b + 1);
                slot_done(b, "cancel");
                continue;
            }
            S.live.push_back(X.pending);
            S.keys.push_back((uint64_t) (uint32_t) X.pending);
            X.pending = t;
            X.pos += 1;
            ++X.produced;
            out("BT %d %d\n", b, t);
            if (std::find(o.eos.begin(), o.eos.end(), t) != o.eos.end() || llama_vocab_is_eog(vocab, t)) slot_done(b, "stop");
            else if (X.produced >= X.max_new || (int64_t) S.live.size() + 2 >= o.max_context) slot_done(b, "length");
        }
    }

    // Blue-green (docs/BLUE_GREEN.md): the conversations this engine holds - the parked ones and every slot not in a
    // request - written for the new engine (EXPORT <dir>), and that engine taking them into its conversation cache
    // (IMPORT <dir>).  A file each, named by its content: "SSEQ2", n (u64), n keys (u64), n tokens (i32), the state's
    // size (u64) and bytes (llama_state_seq_get_data), then its checkpoints - a follow-up that differs at the end goes
    // back to one instead of reading the prompt again: count (u64), each n (u64), pos (i32), inside (u8), size (u64)
    // and bytes (the drafter's parts left out, as park does).  Written under a temporary name, then renamed.  No stdout reply:
    // the server sends these beside requests in flight; the log says what was done.
    std::set<uint64_t> exported;

    static uint64_t keys_hash(const std::vector<uint64_t>& k) {
        uint64_t h = 1469598103934665603ULL;
        for (uint64_t x : k) h = (h ^ x) * 1099511628211ULL;
        return h ^ k.size();
    }

    bool write_conv(const std::string& dir, const Seq& S, const std::vector<uint8_t>& state) {
        const uint64_t h = keys_hash(S.keys);
        if (S.live.empty() || state.empty() || S.keys.size() != S.live.size() || !exported.insert(h).second) return false;
        char name[32];
        std::snprintf(name, sizeof name, "%016llx.sseq", (unsigned long long) h);
        const std::string path = dir + "/" + name, tmp = path + ".part";
        std::ofstream f(tmp, std::ios::binary);
        const uint64_t n = S.live.size(), sz = state.size();
        f.write("SSEQ2", 5);
        f.write((const char*) &n, sizeof n);
        f.write((const char*) S.keys.data(), (std::streamsize) (n * sizeof(uint64_t)));
        f.write((const char*) S.live.data(), (std::streamsize) (n * sizeof(llama_token)));
        f.write((const char*) &sz, sizeof sz);
        f.write((const char*) state.data(), (std::streamsize) sz);
        const uint64_t nc = S.ckpts.size();
        f.write((const char*) &nc, sizeof nc);
        for (const auto& c : S.ckpts) {
            const uint64_t cn = c.n, cs = c.tgt.size();
            const int32_t cp = c.pos;
            const uint8_t in = c.inside;
            f.write((const char*) &cn, sizeof cn);
            f.write((const char*) &cp, sizeof cp);
            f.write((const char*) &in, sizeof in);
            f.write((const char*) &cs, sizeof cs);
            f.write((const char*) c.tgt.data(), (std::streamsize) cs);
        }
        f.close();
        if (!f || std::rename(tmp.c_str(), path.c_str()) != 0) {
            std::remove(tmp.c_str());
            exported.erase(h);
            return false;
        }
        return true;
    }

    void export_all(const std::string& dir) {
        const auto t0 = Clock::now();
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        size_t n = 0, bytes = 0;
        for (const auto& x : parked)
            if (write_conv(dir, x.seq, x.state)) { ++n; bytes += x.state.size(); }
        for (size_t sq = 0; sq < seqs.size(); ++sq) {
            if (sq >= 1 && sq - 1 < slots.size() && slots[sq - 1].active) continue;   // in a request: when it ends
            const Seq& S = seqs[sq];
            if (S.live.empty() || exported.count(keys_hash(S.keys))) continue;
            std::vector<uint8_t> st(llama_state_seq_get_size(ctx, (llama_seq_id) sq));
            if (st.empty() || llama_state_seq_get_data(ctx, st.data(), st.size(), (llama_seq_id) sq) == 0) continue;
            if (write_conv(dir, S, st)) { ++n; bytes += st.size(); }
        }
        std::fprintf(stderr, "strata-metal: handoff: exported %zu conversations (%zu bytes) to %s in %.0f ms\n", n, bytes, dir.c_str(), ms_since(t0));
    }

    void import_all(const std::string& dir) {
        const auto t0 = Clock::now();
        if (o.cache_mib == 0) {
            std::fprintf(stderr, "strata-metal: handoff: no conversation cache here, nothing imported from %s\n", dir.c_str());
            return;
        }
        const size_t budget = (size_t) o.cache_mib << 20;
        size_t n = 0;
        std::error_code ec;
        for (const auto& ent : std::filesystem::directory_iterator(dir, ec)) {
            if (ent.path().extension() != ".sseq") continue;
            std::ifstream f(ent.path(), std::ios::binary);
            char magic[5];
            uint64_t len = 0, sz = 0;
            Parked e;
            bool ok = f.read(magic, 5) && std::memcmp(magic, "SSEQ2", 5) == 0 && f.read((char*) &len, sizeof len) && len > 0 && len <= (uint64_t) o.max_context;
            if (ok) {
                e.seq.keys.resize(len);
                e.seq.live.resize(len);
                ok = f.read((char*) e.seq.keys.data(), (std::streamsize) (len * sizeof(uint64_t))) &&
                     f.read((char*) e.seq.live.data(), (std::streamsize) (len * sizeof(llama_token))) &&
                     f.read((char*) &sz, sizeof sz) && sz > 0 && sz <= budget;
            }
            if (ok) {
                e.state.resize(sz);
                ok = (bool) f.read((char*) e.state.data(), (std::streamsize) sz);
            }
            uint64_t nc = 0;
            if (ok) ok = f.read((char*) &nc, sizeof nc) && nc <= 64;
            for (uint64_t i = 0; ok && i < nc; ++i) {
                uint64_t cn = 0, cs = 0;
                int32_t cp = 0;
                uint8_t in = 0;
                ok = f.read((char*) &cn, sizeof cn) && f.read((char*) &cp, sizeof cp) && f.read((char*) &in, sizeof in) &&
                     f.read((char*) &cs, sizeof cs) && cn <= len && cs <= budget;
                if (!ok) break;
                Checkpoint c{(size_t) cn, (llama_pos) cp, std::vector<uint8_t>(cs), {}, {}, in != 0};
                ok = (bool) f.read((char*) c.tgt.data(), (std::streamsize) cs);
                if (ok) e.seq.ckpts.push_back(std::move(c));
            }
            f.close();
            std::filesystem::remove(ent.path(), ec);
            if (!ok) continue;
            // as park(): one with as much or more of it already here wins; what it supersedes goes; the oldest make room
            bool have = false;
            for (auto& x : parked)
                if (prefix_of(x.seq.keys, e.seq.keys) == len) { have = true; break; }
            if (have) continue;
            parked.erase(std::remove_if(parked.begin(), parked.end(), [&](const Parked& x) { return prefix_of(x.seq.keys, e.seq.keys) == x.seq.keys.size(); }), parked.end());
            e.bytes = e.state.size() + len * (sizeof(uint64_t) + sizeof(llama_token));
            for (const auto& c : e.seq.ckpts) e.bytes += c.tgt.size();
            while (!parked.empty() && ((int) parked.size() >= o.cache_slots || parked_bytes() + e.bytes > budget)) {
                parked.erase(std::min_element(parked.begin(), parked.end(), [](const Parked& a, const Parked& b) { return a.used < b.used; }));
                ++evictions;
            }
            e.used = ++park_clock;
            parked.push_back(std::move(e));
            out_cache("parked", (size_t) len, -1, 0.0);
            ++n;
        }
        std::fprintf(stderr, "strata-metal: handoff: imported %zu conversations from %s in %.0f ms\n", n, dir.c_str(), ms_since(t0));
    }

    void save(const std::string& path) {
        const auto t0 = Clock::now();
        const Seq& S = seqs[0];
        const size_t bytes = llama_state_seq_save_file(ctx, path.c_str(), 0, S.live.data(), S.live.size());
        if (bytes == 0) { out("SERR io 0 could not write %s\n", path.c_str()); return; }
        out("SAVED %zu %zu %.1f\n", S.live.size(), bytes, ms_since(t0));
    }

    void restore(const std::string& path) {
        const auto t0 = Clock::now();
        std::vector<llama_token> toks((size_t) o.max_context);
        size_t n = 0;
        clear(0);
        const size_t bytes = llama_state_seq_load_file(ctx, path.c_str(), 0, toks.data(), toks.size(), &n);
        if (bytes == 0) {
            clear(0);
            out("SERR invalid 0 could not read a session for this model and context from %s\n", path.c_str());
            return;
        }
        Seq& S = seqs[0];
        S.live.assign(toks.begin(), toks.begin() + (long) n);
        // a picture's cells cannot be told apart from another picture's after a restore: they match nothing, so a
        // prompt is reused up to its first image at most.  The drafter starts empty: drafting resumes from new tokens.
        S.keys.resize(n);
        for (size_t i = 0; i < n; ++i) S.keys[i] = S.live[i] == g_image_pad ? kImageBit : (uint64_t) (uint32_t) S.live[i];
        out("RESTORED %zu %zu %.1f\n", n, bytes, ms_since(t0));
    }
};

}  // namespace

int main(int argc, char** argv) {
    // the protocol keeps the original stdout; anything a library prints to fd 1 goes to the log (stderr)
    const int proto_fd = dup(STDOUT_FILENO);
    if (proto_fd < 0 || dup2(STDERR_FILENO, STDOUT_FILENO) < 0 || !(g_proto = fdopen(proto_fd, "w"))) {
        std::fprintf(stderr, "strata-metal: cannot set up stdout\n");
        return 1;
    }
    Engine e;
    if (!parse_args(argc, argv, e.o)) return 2;
    llama_backend_init();
    // llama.cpp's debug lines only under STRATA_TRACE: its default logger prints them all, and a lazy grammar writes
    // one per token while it waits for its trigger (the M1's log: 309,663 of them in 12 hours)
    if (!g_trace)
        llama_log_set([](ggml_log_level level, const char* text, void*) {
            static bool debug = false;
            if (level != GGML_LOG_LEVEL_CONT) debug = level == GGML_LOG_LEVEL_DEBUG;
            if (!debug) std::fputs(text, stderr);
        }, nullptr);
    std::string err;
    if (!e.load(err)) {
        out("ERR %s\n", err.c_str());
        return 1;
    }
    if (e.o.plan) return 0;
    // the GPU's memory once the model is in it (Metal: its working set; Vulkan: the heap, Honeykrisp's HK_SYSMEM on
    // Linux): what the server's Monitor shows where the OS has no GPU counters of its own (Asahi Linux)
    size_t gpu_free = 0, gpu_total = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const auto t = ggml_backend_dev_type(d);
        if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            ggml_backend_dev_memory(d, &gpu_free, &gpu_total);
            break;
        }
    }
    // a Vulkan driver without the memory-budget extension (Honeykrisp) reports the heap's free as its total: the model's
    // own weights are then the closest measure of what it holds
    size_t gpu_used = gpu_total - gpu_free;
    if (gpu_used == 0 && e.model) gpu_used = (size_t) llama_model_size(e.model);
    // the conversation cache is plain RAM beside what the GPU holds: at most the memory less that and macOS's share
    // (15%, 3 GB at least - the app's rule for iogpu.wired_limit_mb), or a full cache swaps everything else out (a
    // 16 GB M1 with LFM2.5-8B: 8 GB asked, 5 GB fit)
    if (e.o.cache_mib > 0) {
        const int64_t ram = (int64_t) sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE) >> 20;
        const int64_t held = (int64_t) std::max(gpu_used, e.model ? (size_t) llama_model_size(e.model) : 0) >> 20;
        const int64_t cap = std::max<int64_t>(0, (ram - held - std::max<int64_t>(ram * 15 / 100, 3072)) / 256 * 256);
        if (ram > 0 && e.o.cache_mib > cap) {
            std::fprintf(stderr, "strata-metal: conversation cache: %lld MiB asked, %lld MiB fit (%lld MiB of RAM, %lld held by the model)\n",
                         (long long) e.o.cache_mib, (long long) cap, (long long) ram, (long long) held);
            e.o.cache_mib = cap;
        }
    }
#ifdef __APPLE__
    const char* backend = "metal";
#else
    const char* backend = "vulkan";
#endif
    out("INFO engine=" STRATA_VERSION "-metal backend=%s llama=" STRATA_LLAMA_COMMIT " context=%lld kv=%s "
        "batch_slots=%d checkpoints=%d im_start=%d n_embd=%d mtp=%d spec=%d gpu_used=%zu gpu_total=%zu grammar=1 json_schema=1 logit_bias=2 "
        "conversation_cache_mib=%lld conversation_cache_slots=%d\n", backend,
        (long long) e.o.max_context, e.o.kv.c_str(), e.o.batch, e.o.checkpoints, e.im_start, e.n_embd, e.spec ? 1 : 0,
        e.spec ? e.o.spec : 0, gpu_used, gpu_total, (long long) e.o.cache_mib, e.o.cache_mib > 0 ? e.o.cache_slots : 0);
    out("READY %lld stop\n", (long long) e.o.max_context);
    g_slots = e.o.batch;
    g_bstop.reset(new std::atomic<bool>[(size_t) std::max(1, g_slots)]());
    std::thread(read_stdin).detach();
    for (;;) {
        std::string line;
        if (g_gpu_failed) {
            std::fprintf(stderr, "strata-metal: the GPU's compute failed (out of memory?): the engine ends, the server starts a fresh one\n");
            break;
        }
        if (!next_line(line, !e.decoding())) {         // slots decoding and no command: one batch window
            e.window();
            continue;
        }
        if (line == "QUIT") break;
        if (line.rfind("BGEN ", 0) == 0 || line.rfind("BGENI ", 0) == 0) {
            // BGEN <slot> <max_new> ...: read as a GEN 1 in that slot's sequence, then BADM
            const bool img = line.rfind("BGENI ", 0) == 0;
            char* e1 = nullptr;
            const long b = std::strtol(line.c_str() + (img ? 6 : 5), &e1, 10);
            char* e2 = nullptr;
            const long long mn = std::strtoll(e1, &e2, 10);
            if (b < 0 || b >= e.o.batch || e.slots[(size_t) b].active || mn < 1 || e2 == e1) {
                // the log says which (the server takes another slot when this one is still running)
                std::fprintf(stderr, "strata-metal: BGEN refused: slot %ld (of %d)%s, max_new %lld\n", b, e.o.batch,
                             b >= 0 && b < e.o.batch && e.slots[(size_t) b].active ? " still running" : "", mn);
                out("ERR BGEN: no such free slot (--batch %d) or a bad max_new\n", e.o.batch);
                continue;
            }
            e.gen(std::string(img ? "GENI 1" : "GEN 1") + e2, img, (int) b, mn);
        } else if (line.rfind("GENI ", 0) == 0) e.gen(line, true);
        else if (line.rfind("GEN ", 0) == 0) e.gen(line, false);
        else if (line.rfind("SAVE ", 0) == 0) e.save(line.substr(5));
        else if (line.rfind("EXPORT ", 0) == 0) e.export_all(line.substr(7));
        else if (line.rfind("IMPORT ", 0) == 0) e.import_all(line.substr(7));
        else if (line.rfind("RESTORE ", 0) == 0) e.restore(line.substr(8));
        else if (line.rfind("VRAM", 0) == 0) out("ERR the Metal engine has no expert cache to resize\n");
        else if (!line.empty()) out("ERR unknown command: %.40s\n", line.c_str());
    }
    if (e.spec) common_speculative_free(e.spec);
    e.spec_init.reset();
    e.init.reset();
    llama_backend_free();
    return g_gpu_failed ? 3 : 0;
}
