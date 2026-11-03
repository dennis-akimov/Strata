// strata-directions: what server/src/bin/strata-ablate needs from a model to find its refusal direction (docs/ABLATION.md).
// It loads the GGUF once and answers one command per stdin line; a reply is a header line on stdout, then raw f32 data:
//   ACT <n> <id,id,...>   the residual stream after each layer (l_out) at the prompt's last n tokens:
//                         "ACT <bytes>\n" + layers x n x n_embd f32
//   LOGP <id,id,...>      the log-softmax over the vocabulary at the prompt's last token: "LOGP <bytes>\n" + n_vocab f32
//   ABL <path> | ABL -    project the file's directions out of every layer (as strata-metal --ablate), or stop: "OK"
// Each prompt starts from an empty memory.  Errors: "ERR <message>" (the command is skipped, the process keeps going).
//
//     strata-directions --gguf MODEL.gguf [--max-context 4096]

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct Capture {
    int n_layer = 0, n_embd = 0, n_last = 0, n_tokens = 0;
    bool on = false;
    std::vector<float> out;   // layers x n_last x n_embd
};

// llama.cpp's eval callback: ask for the l_out-<layer> tensors, then copy their last n_last rows
static bool cb_eval(ggml_tensor* t, bool ask, void* user) {
    auto* c = (Capture*) user;
    int il = -1;
    if (!c->on || std::sscanf(t->name, "l_out-%d", &il) != 1 || il < 0 || il >= c->n_layer) return false;
    if (ask) return true;
    // every token's row, or (the last layer) only the rows of the tokens that output: the last n_last
    if (t->type != GGML_TYPE_F32 || t->ne[0] != c->n_embd || (t->ne[1] != c->n_tokens && t->ne[1] != c->n_last)) return true;
    const size_t row = (size_t) c->n_embd * sizeof(float);
    ggml_backend_tensor_get(t, c->out.data() + (size_t) il * c->n_last * c->n_embd,
                            (size_t) (t->ne[1] - c->n_last) * t->nb[1], (size_t) c->n_last * row);
    return true;
}

static std::vector<llama_token> ids_of(const std::string& s) {
    std::vector<llama_token> v;
    std::stringstream ss(s);
    for (std::string x; std::getline(ss, x, ',');)
        if (!x.empty()) v.push_back((llama_token) std::strtol(x.c_str(), nullptr, 10));
    return v;
}

static void reply(const char* head, const std::vector<float>& data) {
    std::printf("%s %zu\n", head, data.size() * sizeof(float));
    std::fwrite(data.data(), sizeof(float), data.size(), stdout);
    std::fflush(stdout);
}

static void err(const std::string& m) {
    std::printf("ERR %s\n", m.c_str());
    std::fflush(stdout);
}

int main(int argc, char** argv) {
    std::string gguf;
    int n_ctx = 4096;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--gguf")) gguf = argv[i + 1];
        else if (!std::strcmp(argv[i], "--max-context")) n_ctx = std::atoi(argv[i + 1]);
    }
    if (gguf.empty()) {
        std::fprintf(stderr, "usage: strata-directions --gguf MODEL.gguf [--max-context 4096]\n");
        return 2;
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model* model = llama_model_load_from_file(gguf.c_str(), mp);
    if (!model) {
        std::fprintf(stderr, "strata-directions: could not load %s\n", gguf.c_str());
        return 1;
    }
    Capture cap;
    cap.n_layer = llama_model_n_layer(model);
    cap.n_embd = llama_model_n_embd(model);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = cp.n_batch = cp.n_ubatch = (uint32_t) n_ctx;   // a prompt in one batch: l_out holds all its tokens
    cp.n_seq_max = 1;
    cp.cb_eval = cb_eval;
    cp.cb_eval_user_data = &cap;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        std::fprintf(stderr, "strata-directions: could not create the context\n");
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::printf("READY layers %d n_embd %d n_vocab %d\n", cap.n_layer, cap.n_embd, n_vocab);
    std::fflush(stdout);

    for (std::string line; std::getline(std::cin, line);) {
        std::istringstream in(line);
        std::string cmd;
        in >> cmd;
        if (cmd == "ABL") {
            std::string path;
            in >> path;
            if (path == "-") {
                llama_set_adapter_ablation(ctx, nullptr, 0, cap.n_embd, 1, 1.0f);
                std::printf("OK\n");
                std::fflush(stdout);
                continue;
            }
            std::vector<float> v;
            if (FILE* f = std::fopen(path.c_str(), "rb")) {
                float buf[4096];
                for (size_t n; (n = std::fread(buf, sizeof(float), 4096, f)) > 0;) v.insert(v.end(), buf, buf + n);
                std::fclose(f);
            }
            const size_t per = (size_t) cap.n_layer * cap.n_embd;
            if (v.empty() || v.size() % per != 0) { err("ABL " + path + ": not layers x k x n_embd floats"); continue; }
            if (llama_set_adapter_ablation(ctx, v.data(), v.size(), cap.n_embd, (int) (v.size() / per), 1.0f) != 0) {
                err("llama.cpp refused " + path);
                continue;
            }
            std::printf("OK\n");
            std::fflush(stdout);
            continue;
        }
        if (cmd != "ACT" && cmd != "LOGP") { err("unknown command " + cmd); continue; }
        int n_last = 1;
        if (cmd == "ACT") in >> n_last;
        std::string ids_s;
        in >> ids_s;
        std::vector<llama_token> ids = ids_of(ids_s);
        if (ids.empty() || (int) ids.size() > n_ctx || n_last < 1 || n_last > (int) ids.size()) {
            err("bad prompt: " + std::to_string(ids.size()) + " tokens, last " + std::to_string(n_last));
            continue;
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        cap.on = cmd == "ACT";
        cap.n_last = n_last;
        cap.n_tokens = (int) ids.size();
        if (cap.on) cap.out.assign((size_t) cap.n_layer * n_last * cap.n_embd, NAN);
        llama_batch b = llama_batch_init((int32_t) ids.size(), 0, 1);
        for (size_t i = 0; i < ids.size(); i++) {
            b.token[i] = ids[i];
            b.pos[i] = (llama_pos) i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = (int) i >= cap.n_tokens - n_last;   // outputs at the last n: the last layer keeps those rows
        }
        b.n_tokens = (int32_t) ids.size();
        const int rc = llama_decode(ctx, b);
        llama_batch_free(b);
        if (rc != 0) {
            err("decode failed");
            continue;
        }
        if (cap.on) {
            reply("ACT", cap.out);
            continue;
        }
        const float* lg = llama_get_logits_ith(ctx, -1);
        float mx = -INFINITY;
        for (int i = 0; i < n_vocab; i++) mx = std::max(mx, lg[i]);
        double z = 0.0;
        for (int i = 0; i < n_vocab; i++) z += std::exp((double) lg[i] - mx);
        const float lz = (float) (mx + std::log(z));
        std::vector<float> lp(lg, lg + n_vocab);
        for (float& x : lp) x -= lz;
        reply("LOGP", lp);
    }
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
