// metal/image/gguf_weights.cpp - see gguf_weights.h
#include "gguf_weights.h"

#include "ggml-backend-impl.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>

namespace strata_image {

Weights::~Weights() {
    mapped_.reset();                                   // the Metal buffer wraps the mapping: it goes first
    if (map_) ::munmap(map_, map_len_);
}

bool Weights::load(const std::string& path, const std::string& arch, ggml_backend_t backend, std::string& err) {
    arch_ = arch;
    ggml_context* meta_raw = nullptr;
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = &meta_raw;
    g_.reset(gguf_init_from_file(path.c_str(), params));
    meta_.reset(meta_raw);
    if (!g_ || !meta_) { err = "cannot read " + path + " as GGUF"; return false; }
    if (str("general.architecture") != arch) {
        err = path + " is not a " + arch + " GGUF (tools/image/nucleus_gguf.py makes one)";
        return false;
    }

    const int fd = ::open(path.c_str(), O_RDONLY);
    struct stat st{};
    if (fd < 0 || fstat(fd, &st) != 0) { if (fd >= 0) ::close(fd); err = "cannot open " + path; return false; }
    map_len_ = static_cast<size_t>(st.st_size);
    void* m = ::mmap(nullptr, map_len_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);                                       // the mapping keeps the file
    if (m == MAP_FAILED) { err = "cannot map " + path; return false; }
    map_ = m;

    const int64_t n = gguf_get_n_tensors(g_.get());
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * static_cast<size_t>(n + 1);
    ip.no_alloc = true;
    ctx_.reset(ggml_init(ip));
    owned_ctx_.reset(ggml_init(ip));                   // the converted ones: allocated, not mapped
    const uint64_t data_off = gguf_get_data_offset(g_.get());
    std::vector<std::pair<ggml_tensor*, uint64_t>> in_file, converted;
    size_t max_tensor = 0;
    for (int64_t i = 0; i < n; ++i) {
        const char* name = gguf_get_tensor_name(g_.get(), i);
        ggml_tensor* mt = ggml_get_tensor(meta_.get(), name);
        const uint64_t off = gguf_get_tensor_offset(g_.get(), i), nb = ggml_nbytes(mt);
        // overflow-safe bounds (OpenSSF: "offset + len" can wrap before the comparison)
        if (data_off > map_len_ || off > map_len_ - data_off || nb > map_len_ - data_off - off) {
            err = path + ": tensor " + name + " lies past the end of the file";
            return false;
        }
        const bool to_f32 = ggml_n_dims(mt) == 1 && mt->type == GGML_TYPE_BF16;
        ggml_tensor* t = ggml_new_tensor(to_f32 ? owned_ctx_.get() : ctx_.get(), to_f32 ? GGML_TYPE_F32 : mt->type,
                                         GGML_MAX_DIMS, mt->ne);
        ggml_set_name(t, name);
        by_name_[name] = t;
        (to_f32 ? converted : in_file).emplace_back(t, data_off + off);
        max_tensor = std::max(max_tensor, static_cast<size_t>(nb));
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_dev_props props{};
    ggml_backend_dev_get_props(dev, &props);
    if (!props.caps.buffer_from_host_ptr) { err = "the GPU backend cannot map a file"; return false; }
    mapped_.reset(ggml_backend_dev_buffer_from_host_ptr(dev, map_, map_len_, max_tensor));
    // Metal wraps a mapping it could not make (newBufferWithBytesNoCopy: nil) in a buffer whose context is null
    if (!mapped_ || !mapped_->context) { err = "the GPU cannot map " + path + " (see the log above)"; return false; }
    for (auto& [t, off] : in_file)
        if (ggml_backend_tensor_alloc(mapped_.get(), t, static_cast<char*>(map_) + off) != GGML_STATUS_SUCCESS) {
            err = path + ": cannot place " + ggml_get_name(t);
            return false;
        }
    if (!converted.empty()) {
        owned_.reset(ggml_backend_alloc_ctx_tensors(owned_ctx_.get(), backend));
        if (!owned_) { err = "not enough GPU memory for " + path; return false; }
        std::vector<float> f;
        for (auto& [t, off] : converted) {
            f.resize(static_cast<size_t>(ggml_nelements(t)));
            const auto* src = reinterpret_cast<const uint16_t*>(static_cast<const char*>(map_) + off);
            for (size_t j = 0; j < f.size(); ++j) {
                uint16_t b;
                std::memcpy(&b, src + j, sizeof b);
                const uint32_t bits = static_cast<uint32_t>(b) << 16;           // bf16 -> f32 exactly
                std::memcpy(&f[j], &bits, sizeof(float));
            }
            ggml_backend_tensor_set(t, f.data(), 0, ggml_nbytes(t));
        }
    }
    return true;
}

ggml_tensor* Weights::get(const std::string& name) const {
    const auto it = by_name_.find(name);
    if (it != by_name_.end()) return it->second;
    std::fprintf(stderr, "strata-image: the model has no tensor %s\n", name.c_str());
    std::abort();
}

uint32_t Weights::u32(const std::string& key, uint32_t def) const {
    const int64_t k = gguf_find_key(g_.get(), (arch_ + "." + key).c_str());
    return k < 0 || gguf_get_kv_type(g_.get(), k) != GGUF_TYPE_UINT32 ? def : gguf_get_val_u32(g_.get(), k);
}

float Weights::f32(const std::string& key, float def) const {
    const int64_t k = gguf_find_key(g_.get(), (arch_ + "." + key).c_str());
    return k < 0 || gguf_get_kv_type(g_.get(), k) != GGUF_TYPE_FLOAT32 ? def : gguf_get_val_f32(g_.get(), k);
}

std::string Weights::str(const std::string& key) const {
    const int64_t k = gguf_find_key(g_.get(), key.c_str());
    return k < 0 || gguf_get_kv_type(g_.get(), k) != GGUF_TYPE_STRING ? std::string() : gguf_get_val_str(g_.get(), k);
}

std::vector<float> Weights::floats(const std::string& key) const {
    std::vector<float> out;
    const int64_t k = gguf_find_key(g_.get(), (arch_ + "." + key).c_str());
    if (k < 0 || gguf_get_kv_type(g_.get(), k) != GGUF_TYPE_ARRAY) return out;
    const size_t n = gguf_get_arr_n(g_.get(), k);
    const enum gguf_type t = gguf_get_arr_type(g_.get(), k);
    const void* data = gguf_get_arr_data(g_.get(), k);
    for (size_t i = 0; i < n; ++i) {
        if (t == GGUF_TYPE_FLOAT32) out.push_back(static_cast<const float*>(data)[i]);
        else if (t == GGUF_TYPE_INT32) out.push_back(static_cast<float>(static_cast<const int32_t*>(data)[i]));
        else if (t == GGUF_TYPE_UINT32) out.push_back(static_cast<float>(static_cast<const uint32_t*>(data)[i]));
        else return {};
    }
    return out;
}

}  // namespace strata_image
