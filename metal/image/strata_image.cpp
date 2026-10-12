// metal/image/strata_image.cpp - strata-image: Strata's image engine on ggml/Metal (Nucleus-Image).  One picture per
// process (serve/imagegen.py starts it): its exit is what gives the memory back.
//
//   strata-image --encoder qwen3vl-8b-text.gguf --dit nucleus-dit.gguf --vae nucleus-vae.gguf
//                --prompt-file p.txt [--negative-file n.txt] --width 1024 --height 1024 --steps 20 --cfg 4 --seed 42
//                --out picture.png [--noise z.npy] [--latent-out z.npy] [--watch-stdin 1]
//
// --watch-stdin 1: the process ends when its stdin closes (serve/imagegen.py keeps it open: a server that is gone does
// not leave a picture running).
// stdout gets one JSON object a line: {"phase": "text"|"dit"|"decode"}, {"step": i, "steps": n, "s": seconds} and at
// the end {"done": true, ...timings}; an error is {"error": "..."} and exit status 1.  The text encoder runs first and is
// freed, then the DiT (memory-mapped: its pages are the file's), then the VAE.
//
// The checks against the port:
//
//   strata-image --dit nucleus-dit-bf16.gguf --check <dump> [--given] [--each]
//   strata-image --encoder <gguf> --check-text <dir>
//   strata-image --vae <gguf> --check-vae <dir>
//
// <dump> is tools/image/dump_ref.py's folder (x.npy, txt.npy, case.json and the port's every block); the outputs land
// next to them as *_cpp.npy and tools/image/check_dit.py compares the two.  --given takes the RoPE tables and the
// timestep embedding from the dump too, so the blocks are checked apart from the host-side conditioning.
#include "nucleus_dit.h"
#include "nucleus_vae.h"
#include "text_encoder.h"

#include "ggml-backend.h"

#include <zlib.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

// .npy: little-endian float32, C order, any shape (only the element count is checked against the caller's)
bool read_npy(const std::string& path, std::vector<float>& out, std::vector<int64_t>& shape) {
    std::ifstream f(path, std::ios::binary);
    char magic[6];
    if (!f.read(magic, 6) || std::memcmp(magic, "\x93NUMPY", 6) != 0) return false;
    unsigned char ver[2];
    if (!f.read(reinterpret_cast<char*>(ver), 2)) return false;
    uint32_t hlen = 0;
    if (ver[0] == 1) {
        unsigned char b[2];
        if (!f.read(reinterpret_cast<char*>(b), 2)) return false;
        hlen = static_cast<uint32_t>(b[0] | (b[1] << 8));
    } else {
        unsigned char b[4];
        if (!f.read(reinterpret_cast<char*>(b), 4)) return false;
        hlen = static_cast<uint32_t>(b[0] | (b[1] << 8) | (b[2] << 16) | (static_cast<uint32_t>(b[3]) << 24));
    }
    if (hlen > (1u << 20)) return false;
    std::string header(hlen, '\0');
    if (!f.read(header.data(), hlen)) return false;
    if (header.find("'<f4'") == std::string::npos || header.find("'fortran_order': False") == std::string::npos) return false;
    const size_t a = header.find('('), b = header.find(')');
    if (a == std::string::npos || b == std::string::npos || b < a) return false;
    shape.clear();
    std::stringstream ss(header.substr(a + 1, b - a - 1));
    std::string item;
    int64_t n = 1;
    while (std::getline(ss, item, ',')) {
        if (item.find_first_not_of(" ") == std::string::npos) continue;
        int64_t d = -1;
        try { d = std::stoll(item); } catch (const std::exception&) { return false; }
        if (d < 0 || (d > 0 && n > (int64_t{1} << 40) / d)) return false;
        shape.push_back(d);
        n *= d;
    }
    out.resize(static_cast<size_t>(n));
    return static_cast<bool>(f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(n * 4)));
}

bool write_npy(const std::string& path, const std::vector<float>& v, const std::vector<int64_t>& shape) {
    std::string dims;
    for (const int64_t d : shape) dims += std::to_string(d) + ",";
    if (shape.size() > 1) dims.pop_back();
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + dims + "), }";
    while ((10 + header.size() + 1) % 64) header += ' ';
    header += '\n';
    std::ofstream f(path, std::ios::binary);
    const uint16_t hl = static_cast<uint16_t>(header.size());
    f.write("\x93NUMPY\x01\x00", 8);
    f.put(static_cast<char>(hl & 0xff));
    f.put(static_cast<char>(hl >> 8));
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
    return static_cast<bool>(f);
}

// case.json is written by our own dump script: four numbers, read by key
double json_number(const std::string& text, const std::string& key) {
    const size_t k = text.find("\"" + key + "\"");
    if (k == std::string::npos) return 0.0;
    const size_t c = text.find(':', k);
    return c == std::string::npos ? 0.0 : std::stod(text.substr(c + 1));
}

int fail(const std::string& msg) {
    std::fprintf(stderr, "strata-image: %s\n", msg.c_str());
    return 1;
}

std::string json_str(const std::string& v) {
    std::string o = "\"";
    for (const char ch : v) {
        const auto c = static_cast<unsigned char>(ch);
        if (c == '"' || c == '\\') { o += '\\'; o += ch; }
        else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += ch;
    }
    return o + "\"";
}

// The protocol's error line (stdout) and exit status
int fail_line(const std::string& msg) {
    std::printf("{\"error\": %s}\n", json_str(msg).c_str());
    std::fflush(stdout);
    return fail(msg);
}

// RGB, 8 bits, no interlace: IHDR, one zlib IDAT, IEND
bool write_png(const std::string& path, const std::vector<uint8_t>& rgb, int w, int h) {
    std::vector<uint8_t> raw;
    raw.reserve(static_cast<size_t>(h) * (static_cast<size_t>(w) * 3 + 1));
    for (int y = 0; y < h; ++y) {
        raw.push_back(0);                                             // filter: none
        raw.insert(raw.end(), rgb.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(y) * w * 3),
                   rgb.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(y + 1) * w * 3));
    }
    uLongf zn = compressBound(static_cast<uLong>(raw.size()));
    std::vector<uint8_t> z(zn);
    if (compress2(z.data(), &zn, raw.data(), static_cast<uLong>(raw.size()), 6) != Z_OK) return false;
    z.resize(zn);
    std::ofstream f(path, std::ios::binary);
    auto be = [](uint32_t v) {                                        // big-endian, as PNG stores numbers
        return std::array<uint8_t, 4>{static_cast<uint8_t>(v >> 24), static_cast<uint8_t>(v >> 16),
                                      static_cast<uint8_t>(v >> 8), static_cast<uint8_t>(v)};
    };
    auto be32 = [&](uint32_t v) { const auto b = be(v); f.write(reinterpret_cast<const char*>(b.data()), 4); };
    auto chunk = [&](const char* type, const uint8_t* data, size_t n) {
        be32(static_cast<uint32_t>(n));
        f.write(type, 4);
        if (n) f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));
        uLong crc = crc32(0, reinterpret_cast<const Bytef*>(type), 4);
        if (n) crc = crc32(crc, data, static_cast<uInt>(n));
        be32(static_cast<uint32_t>(crc));
    };
    f.write("\x89PNG\r\n\x1a\n", 8);
    const auto bw = be(static_cast<uint32_t>(w)), bh = be(static_cast<uint32_t>(h));
    const std::array<uint8_t, 13> ihdr{bw[0], bw[1], bw[2], bw[3], bh[0], bh[1], bh[2], bh[3], 8, 2, 0, 0, 0};  // 8-bit RGB
    chunk("IHDR", ihdr.data(), ihdr.size());
    chunk("IDAT", z.data(), z.size());
    chunk("IEND", nullptr, 0);
    return static_cast<bool>(f);
}

// Gaussian noise from a seed: splitmix64 and Box-Muller (the same numbers on every machine; not MLX's)
std::vector<float> noise(uint64_t seed, size_t n) {
    auto next = [&seed]() {
        uint64_t z = (seed += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    };
    auto unit = [&]() { return (static_cast<double>(next() >> 11) + 0.5) * (1.0 / 9007199254740992.0); };   // (0, 1)
    std::vector<float> v(n);
    for (size_t i = 0; i < n; i += 2) {
        const double r = std::sqrt(-2.0 * std::log(unit())), a = 6.283185307179586 * unit();
        v[i] = static_cast<float>(r * std::cos(a));
        if (i + 1 < n) v[i + 1] = static_cast<float>(r * std::sin(a));
    }
    return v;
}

double now_s() { return static_cast<double>(ggml_time_us()) / 1e6; }

bool read_text(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    out.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return true;
}

struct Request {
    std::string encoder, dit, vae, prompt_file, negative_file, out, noise_file, latent_out;
    int width = 1024, height = 1024, steps = 20;
    double cfg = 4.0;
    uint64_t seed = 0;
    bool watch_stdin = false;
};

int generate(const Request& r) {
    const auto say = [](const std::string& line) { std::printf("%s\n", line.c_str()); std::fflush(stdout); };
    if (r.width % 16 || r.height % 16 || std::min(r.width, r.height) < 256 || std::max(r.width, r.height) > 2048)
        return fail_line("width and height must be multiples of 16 from 256 to 2048");
    if (r.width * r.height > 1536 * 1024) return fail_line("at most 1536x1024 pixels in all (serve/imagegen.py: MAX_PIXELS)");
    if (r.steps < 1 || r.steps > 100) return fail_line("steps must be from 1 to 100");
    if (!(r.cfg >= 0.0 && r.cfg <= 20.0)) return fail_line("cfg must be from 0 to 20");
    std::string prompt, negative;
    if (!read_text(r.prompt_file, prompt) || prompt.empty()) return fail_line("cannot read the prompt file");
    if (!r.negative_file.empty() && !read_text(r.negative_file, negative)) return fail_line("cannot read the negative prompt file");
    const bool cfg = r.cfg > 1.0;
    const double t0 = now_s();
    if (r.watch_stdin)
        std::thread([] { while (std::fgetc(stdin) != EOF) {} std::_Exit(3); }).detach();

    // 1. the text: both prompts, then the encoder is gone
    say("{\"phase\": \"text\"}");
    std::string err;
    std::vector<std::string> prompts{prompt};
    if (cfg) prompts.push_back(negative);
    const auto texts = strata_image::encode_prompts(r.encoder, prompts, err);
    if (texts.empty()) return fail_line(err);
    const double t_text = now_s() - t0;

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) return fail_line("no GPU backend (Metal) in this build");
    strata_image::Backend backend(ggml_backend_dev_init(dev, nullptr));
    if (!backend) return fail_line("could not start the Metal backend");

    // 2. the noise, as [lh, lw, 16], and its tokens [S, 64] ([C, ph, pw] per token: diffusers' _pack_latents)
    const int lh = r.height / 8, lw = r.width / 8, gh = lh / 2, gw = lw / 2, S = gh * gw;
    std::vector<float> lat;
    if (!r.noise_file.empty()) {
        std::vector<int64_t> s;
        if (!read_npy(r.noise_file, lat, s) || s != std::vector<int64_t>{lh, lw, 16}) return fail_line("--noise must be [h/8, w/8, 16]");
    } else {
        lat = noise(r.seed, static_cast<size_t>(lh) * lw * 16);
    }
    std::vector<float> tok(static_cast<size_t>(S) * 64);
    auto at = [&](int y, int x, int c) -> size_t { return (static_cast<size_t>(y) * lw + x) * 16 + c; };
    for (int gy = 0; gy < gh; ++gy)
        for (int gx = 0; gx < gw; ++gx)
            for (int c = 0; c < 16; ++c)
                for (int py = 0; py < 2; ++py)
                    for (int px = 0; px < 2; ++px)
                        tok[(static_cast<size_t>(gy) * gw + gx) * 64 + c * 4 + py * 2 + px] = lat[at(gy * 2 + py, gx * 2 + px, c)];

    // 3. the DiT: Euler on sigma = linspace(1, 1/steps, steps) then 0; CFG with the norm rescale; the velocity negated
    say("{\"phase\": \"dit\"}");
    double t_dit = 0.0, t_load = now_s(), dit_gb = 0.0;
    {
        strata_image::NucleusDiT dit;
        if (!dit.load(r.dit, backend.get(), err)) return fail_line(err);
        t_load = now_s() - t_load;
        std::vector<strata_image::Text> branches{{&texts[0].h, texts[0].T}};
        if (cfg) branches.push_back({&texts[1].h, texts[1].T});
        const float g = static_cast<float>(r.cfg);
        for (int i = 0; i < r.steps; ++i) {
            const double ts = now_s();
            const float sig = r.steps == 1 ? 1.0f : static_cast<float>(1.0 - (1.0 - 1.0 / r.steps) * i / (r.steps - 1));
            const float nxt = i + 1 < r.steps ? static_cast<float>(1.0 - (1.0 - 1.0 / r.steps) * (i + 1) / (r.steps - 1)) : 0.0f;
            // the port passes timesteps (sigma * 1000) through float32 and divides by 1000 again
            const float t = static_cast<float>(static_cast<float>(sig * 1000.0f) / 1000.0);
            auto ys = dit.forward_branches(tok, gh, gw, branches, t, err);
            if (ys.empty()) return fail_line(err);
            std::vector<float>& pred = ys[0];
            for (int k = 0; k < S; ++k) {
                float* p = &pred[static_cast<size_t>(k) * 64];
                if (cfg) {
                    const float* n = &ys[1][static_cast<size_t>(k) * 64];
                    double cn = 0.0, mn = 0.0;
                    for (int j = 0; j < 64; ++j) {
                        cn += static_cast<double>(p[j]) * p[j];
                        p[j] = n[j] + g * (p[j] - n[j]);
                        mn += static_cast<double>(p[j]) * p[j];
                    }
                    const float sc = static_cast<float>(std::sqrt(cn + 1e-8) / std::sqrt(mn + 1e-8));
                    for (int j = 0; j < 64; ++j) p[j] *= sc;
                }
                float* x = &tok[static_cast<size_t>(k) * 64];
                for (int j = 0; j < 64; ++j) x[j] += (nxt - sig) * -p[j];
            }
            t_dit += now_s() - ts;
            dit_gb = std::max(dit_gb, dit.compute_bytes() / 1e9);
            char line[96];
            std::snprintf(line, sizeof line, "{\"step\": %d, \"steps\": %d, \"s\": %.3f}", i + 1, r.steps, now_s() - ts);
            say(line);
        }
    }

    // 4. the picture: tokens -> latent, z * std + mean, the VAE, [-1, 1] -> 0..255
    say("{\"phase\": \"decode\"}");
    const double td = now_s();
    for (int gy = 0; gy < gh; ++gy)
        for (int gx = 0; gx < gw; ++gx)
            for (int c = 0; c < 16; ++c)
                for (int py = 0; py < 2; ++py)
                    for (int px = 0; px < 2; ++px)
                        lat[at(gy * 2 + py, gx * 2 + px, c)] = tok[(static_cast<size_t>(gy) * gw + gx) * 64 + c * 4 + py * 2 + px];
    if (!r.latent_out.empty()) write_npy(r.latent_out, lat, {lh, lw, 16});
    std::vector<uint8_t> rgb;
    {
        strata_image::NucleusVAE vae;
        if (!vae.load(r.vae, backend.get(), err)) return fail_line(err);
        for (size_t i = 0; i < lat.size(); ++i) lat[i] = lat[i] * vae.std_dev()[i % 16] + vae.mean()[i % 16];
        const std::vector<float> img = vae.decode(lat, lh, lw, err);
        if (img.empty()) return fail_line(err);
        rgb.resize(img.size());
        // the port: clip to [-1, 1], (v + 1) / 2 * 255, then a cast (truncation) to uint8
        for (size_t i = 0; i < img.size(); ++i)
            rgb[i] = static_cast<uint8_t>((std::clamp(img[i], -1.0f, 1.0f) + 1.0f) / 2.0f * 255.0f);
    }
    if (!write_png(r.out, rgb, r.width, r.height)) return fail_line("cannot write " + r.out);
    char line[256];
    std::snprintf(line, sizeof line, "{\"done\": true, \"text_s\": %.1f, \"load_s\": %.1f, \"denoise_s\": %.1f, "
                  "\"decode_s\": %.1f, \"total_s\": %.1f, \"dit_activations_gb\": %.2f}", t_text, t_load, t_dit,
                  now_s() - td, now_s() - t0, dit_gb);
    say(line);
    return 0;
}

}  // namespace

// --encoder <gguf> --check-text <dir>: <dir>/prompts.txt (one per line) to <dir>/emb<i>_cpp.npy
int check_text(const std::string& enc, const std::string& dir) {
    std::ifstream f(dir + "/prompts.txt");
    std::vector<std::string> prompts;
    for (std::string line; std::getline(f, line);) prompts.push_back(line);
    if (prompts.empty()) return fail("no prompts in " + dir + "/prompts.txt");
    std::string err;
    const auto t0 = ggml_time_us();
    const auto embs = strata_image::encode_prompts(enc, prompts, err);
    if (embs.empty()) return fail(err);
    std::fprintf(stderr, "strata-image: %zu prompts encoded in %.1f ms (with the load)\n", embs.size(),
                 (ggml_time_us() - t0) / 1000.0);
    for (size_t i = 0; i < embs.size(); ++i)
        write_npy(dir + "/emb" + std::to_string(i) + "_cpp.npy", embs[i].h, {embs[i].T, 4096});
    return 0;
}

// --vae <gguf> --check-vae <dir>: <dir>/z.npy [h, w, 16] to <dir>/img_cpp.npy [8h, 8w, 3]
int check_vae(const std::string& path, const std::string& dir) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) return fail("no GPU backend (Metal) in this build");
    strata_image::Backend backend(ggml_backend_dev_init(dev, nullptr));
    strata_image::NucleusVAE vae;
    std::string err;
    if (!backend || !vae.load(path, backend.get(), err)) return fail(err.empty() ? "no Metal backend" : err);
    std::vector<float> z;
    std::vector<int64_t> s;
    if (!read_npy(dir + "/z.npy", z, s) || s.size() != 3 || s[2] != 16) return fail("cannot read " + dir + "/z.npy [h, w, 16]");
    std::vector<float> img;
    for (int run = 0; run < 2; ++run) {
        const auto t0 = ggml_time_us();
        img = vae.decode(z, static_cast<int>(s[0]), static_cast<int>(s[1]), err);
        if (img.empty()) return fail(err);
        std::fprintf(stderr, "strata-image: VAE decode %lldx%lld, %s: %.1f ms\n", static_cast<long long>(s[0] * 8),
                     static_cast<long long>(s[1] * 8), run == 0 ? "first" : "warm", (ggml_time_us() - t0) / 1000.0);
    }
    write_npy(dir + "/img_cpp.npy", img, {s[0] * 8, s[1] * 8, 3});
    return 0;
}

// --bench <dit.gguf> <size> <reps>: the DiT alone, both CFG branches on random inputs; the fastest of `reps` steps
int bench(const std::string& path, int size, int reps) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev || size < 256 || size % 16 || reps < 1) return fail("usage: strata-image --bench <dit.gguf> <size> <reps> (Metal)");
    strata_image::Backend backend(ggml_backend_dev_init(dev, nullptr));
    strata_image::NucleusDiT dit;
    std::string err;
    if (!dit.load(path, backend.get(), err)) return fail(err);
    const int g = size / 16;
    const std::vector<float> x = noise(1, static_cast<size_t>(g) * g * 64), a = noise(2, 23 * 4096), b = noise(3, 20 * 4096);
    double best = 1e9;
    for (int i = 0; i <= reps; ++i) {
        const double t0 = now_s();
        if (dit.forward_branches(x, g, g, {{&a, 23}, {&b, 20}}, 0.5f, err).empty()) return fail(err);
        if (i > 0) best = std::min(best, now_s() - t0);     // the first compiles the kernels
    }
    std::printf("%s %dx%d: %.3f s a step (best of %d), activations %.2f GB\n", path.c_str(), size, size, best, reps,
                dit.compute_bytes() / 1e9);
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 5 && std::string(argv[1]) == "--bench") return bench(argv[2], std::atoi(argv[3]), std::atoi(argv[4]));
    if (argc > 1 && std::string(argv[1]) == "--encoder" && !(argc == 5 && std::string(argv[3]) == "--check-text")) {
        Request r;
        for (int i = 1; i + 1 < argc; i += 2) {
            const std::string a = argv[i], v = argv[i + 1];
            try {
                if (a == "--encoder") r.encoder = v;
                else if (a == "--dit") r.dit = v;
                else if (a == "--vae") r.vae = v;
                else if (a == "--prompt-file") r.prompt_file = v;
                else if (a == "--negative-file") r.negative_file = v;
                else if (a == "--out") r.out = v;
                else if (a == "--noise") r.noise_file = v;
                else if (a == "--latent-out") r.latent_out = v;
                else if (a == "--width") r.width = std::stoi(v);
                else if (a == "--height") r.height = std::stoi(v);
                else if (a == "--steps") r.steps = std::stoi(v);
                else if (a == "--cfg") r.cfg = std::stod(v);
                else if (a == "--seed") r.seed = std::stoull(v);
                else if (a == "--watch-stdin") r.watch_stdin = v == "1";
                else return fail_line("unknown option " + a);
            } catch (const std::exception&) {
                return fail_line("bad value for " + a);
            }
        }
        if (argc % 2 == 0) return fail_line("every option takes a value");
        if (r.encoder.empty() || r.dit.empty() || r.vae.empty() || r.prompt_file.empty() || r.out.empty())
            return fail_line("--encoder, --dit, --vae, --prompt-file and --out are required");
        return generate(r);
    }
    if (argc == 5 && std::string(argv[1]) == "--vae" && std::string(argv[3]) == "--check-vae")
        return check_vae(argv[2], argv[4]);
    if (argc == 5 && std::string(argv[1]) == "--encoder" && std::string(argv[3]) == "--check-text")
        return check_text(argv[2], argv[4]);
    std::string dit_path, check_dir;
    bool given = false, each = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--dit" && i + 1 < argc) dit_path = argv[++i];
        else if (a == "--check" && i + 1 < argc) check_dir = argv[++i];
        else if (a == "--given") given = true;
        else if (a == "--each") each = true;
        else return fail("usage: strata-image --dit <nucleus-dit.gguf> --check <dump> [--given]");
    }
    if (dit_path.empty() || check_dir.empty()) return fail("usage: strata-image --dit <nucleus-dit.gguf> --check <dump> [--given]");

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) return fail("no GPU backend (Metal) in this build");
    strata_image::Backend backend(ggml_backend_dev_init(dev, nullptr));
    if (!backend) return fail("could not start the Metal backend");

    strata_image::NucleusDiT dit;
    std::string err;
    if (!dit.load(dit_path, backend.get(), err)) return fail(err);

    std::ifstream cf(check_dir + "/case.json");
    const std::string case_text((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
    const int gh = static_cast<int>(json_number(case_text, "grid_h")), gw = static_cast<int>(json_number(case_text, "grid_w"));
    const int T = static_cast<int>(json_number(case_text, "T"));
    const float t = static_cast<float>(json_number(case_text, "t"));
    std::vector<float> x, txt;
    std::vector<int64_t> sx, st;
    if (!read_npy(check_dir + "/x.npy", x, sx) || !read_npy(check_dir + "/txt.npy", txt, st))
        return fail("cannot read x.npy / txt.npy in " + check_dir);

    strata_image::Conditioning cond = dit.conditioning(gh, gw, T, t);
    const int hd2 = dit.hp().head_dim / 2;
    write_npy(check_dir + "/img_cos_cpp.npy", cond.img_cos, {gh * gw, hd2});
    write_npy(check_dir + "/img_sin_cpp.npy", cond.img_sin, {gh * gw, hd2});
    write_npy(check_dir + "/txt_cos_cpp.npy", cond.txt_cos, {T, hd2});
    write_npy(check_dir + "/txt_sin_cpp.npy", cond.txt_sin, {T, hd2});
    if (given) {
        std::vector<int64_t> s;
        if (!read_npy(check_dir + "/img_cos.npy", cond.img_cos, s) || !read_npy(check_dir + "/img_sin.npy", cond.img_sin, s) ||
            !read_npy(check_dir + "/txt_cos.npy", cond.txt_cos, s) || !read_npy(check_dir + "/txt_sin.npy", cond.txt_sin, s))
            return fail("cannot read the dump's RoPE tables");
    }

    if (each) {           // every block alone: the port's previous block in, so an error is that block's own
        std::vector<float> h, ref;
        std::vector<int64_t> s;
        for (int l = 0; l < dit.hp().layers; ++l) {
            char prev[64];
            if (l == 0) std::snprintf(prev, sizeof prev, "/after_img_in.npy");
            else std::snprintf(prev, sizeof prev, "/block%02d.npy", l - 1);
            if (!read_npy(check_dir + prev, h, s)) return fail(std::string("cannot read ") + prev);
            std::vector<float> y = dit.block(l, h, gh, gw, txt, T, t, &cond, err);
            if (y.empty()) return fail(err);
            char name[64];
            std::snprintf(name, sizeof name, "/each%02d_cpp.npy", l);
            write_npy(check_dir + name, y, {gh * gw, dit.hp().hidden});
        }
        return 0;
    }
    std::vector<std::vector<float>> blocks;
    std::vector<float> y;
    for (int run = 0; run < 2; ++run) {          // the second is warm (the first compiles Metal kernels)
        const auto t0 = ggml_time_us();
        y = dit.forward(x, gh, gw, txt, T, t, &cond, run == 0 ? &blocks : nullptr, err);
        if (y.empty()) return fail(err);
        std::fprintf(stderr, "strata-image: DiT forward %dx%d tokens, %d text, %s: %.1f ms\n", gh, gw, T,
                     run == 0 ? "first" : "warm", (ggml_time_us() - t0) / 1000.0);
    }
    write_npy(check_dir + "/out_cpp.npy", y, {gh * gw, dit.hp().in_channels});
    for (size_t i = 0; i < blocks.size(); ++i) {
        char name[64];
        std::snprintf(name, sizeof name, "/block%02zu_cpp.npy", i);
        write_npy(check_dir + name, blocks[i], {gh * gw, dit.hp().hidden});
    }
    return 0;
}
