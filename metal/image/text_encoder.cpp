// metal/image/text_encoder.cpp - see text_encoder.h.  The hidden state is taken from llama.cpp's graph by its name
// ("l_out-28") through the scheduler's eval callback, and the graph stops there: layers 29..35 and the head never run.
#include "text_encoder.h"

#include "llama.h"

#include <cstring>
#include <memory>

namespace strata_image {
namespace {

constexpr int kLayer = 28;                        // hidden_states[-8] of 37 (the embeddings + 36 layers)
constexpr int kMaxTokens = 16384;
constexpr const char* kSystem = "You are an image generation assistant.";

struct ModelFree { void operator()(llama_model* m) const { llama_model_free(m); } };
struct ContextFree { void operator()(llama_context* c) const { llama_free(c); } };

struct Capture {
    std::string name = "l_out-" + std::to_string(kLayer);
    std::vector<float>* out = nullptr;
    bool got = false;
};

bool on_tensor(ggml_tensor* t, bool ask, void* ud) {
    auto* c = static_cast<Capture*>(ud);
    if (std::strcmp(t->name, c->name.c_str()) != 0) return ask ? false : true;
    if (ask) return true;
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) return false;
    c->out->resize(static_cast<size_t>(ggml_nelements(t)));
    ggml_backend_tensor_get(t, c->out->data(), 0, ggml_nbytes(t));
    c->got = true;
    return false;                                 // stop: nothing after this layer is needed
}

}  // namespace

std::vector<TextEncoding> encode_prompts(const std::string& gguf, const std::vector<std::string>& prompts,
                                         std::string& err) {
    llama_backend_init();
    llama_log_set([](ggml_log_level lvl, const char* text, void*) { if (lvl >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr); },
                  nullptr);
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    std::unique_ptr<llama_model, ModelFree> model(llama_model_load_from_file(gguf.c_str(), mp));
    if (!model) { err = "cannot load the text encoder " + gguf; return {}; }
    if (llama_model_n_layer(model.get()) <= kLayer) { err = gguf + " has too few layers for a Qwen3-VL-8B"; return {}; }
    const llama_vocab* vocab = llama_model_get_vocab(model.get());

    std::vector<TextEncoding> out;
    for (const std::string& p : prompts) {
        const std::string text = "<|im_start|>system\n" + std::string(kSystem) + "<|im_end|>\n<|im_start|>user\n" + p +
                                 "<|im_end|>\n<|im_start|>assistant\n";
        const int n = -llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), nullptr, 0, false, true);
        // serve/imagegen.py's 4000 characters are at most 4 tokens each (byte-level BPE): never past this
        if (n <= 0 || n > kMaxTokens) {
            err = "the prompt is too long (" + std::to_string(n) + " tokens, at most " + std::to_string(kMaxTokens) + ")";
            return {};
        }
        std::vector<llama_token> tok(static_cast<size_t>(n));
        llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), tok.data(), n, false, true);

        TextEncoding enc;
        Capture cap;
        cap.out = &enc.h;
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = cp.n_batch = cp.n_ubatch = static_cast<uint32_t>((n + 255) / 256 * 256);
        cp.n_seq_max = 1;
        cp.cb_eval = on_tensor;
        cp.cb_eval_user_data = &cap;
        std::unique_ptr<llama_context, ContextFree> ctx(llama_init_from_model(model.get(), cp));
        if (!ctx) { err = "cannot make a context for the text encoder"; return {}; }
        if (llama_decode(ctx.get(), llama_batch_get_one(tok.data(), n)) != 0 || !cap.got ||
            enc.h.size() != static_cast<size_t>(n) * llama_model_n_embd(model.get())) {
            err = "the text encoder did not produce layer " + std::to_string(kLayer);
            return {};
        }
        enc.T = n;
        out.push_back(std::move(enc));
    }
    return out;
}

}  // namespace strata_image
