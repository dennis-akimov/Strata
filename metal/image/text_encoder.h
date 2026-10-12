// metal/image/text_encoder.h - Nucleus-Image's text encoder: Qwen3-VL-8B on Strata's llama.cpp, the prompt's hidden
// states at layer 28 (the port's hidden_states[-8]: 36 layers, before the final norm).
#pragma once

#include <string>
#include <vector>

namespace strata_image {

struct TextEncoding {
    std::vector<float> h;   // [T, 4096], row-major
    int T = 0;
};

// Loads the encoder GGUF, encodes every prompt (ChatML with the port's system prompt) and frees the model before it
// returns, so the DiT gets the memory.  An empty result with `err` set on failure.
std::vector<TextEncoding> encode_prompts(const std::string& gguf, const std::vector<std::string>& prompts,
                                         std::string& err);

}  // namespace strata_image
