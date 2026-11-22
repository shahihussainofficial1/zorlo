#include "pack.h"

#include "gguf.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace zorlo {

static uint32_t get_u32(gguf_context * g, const std::string & key) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) throw std::runtime_error("missing GGUF key " + key);
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(g, id);
        case GGUF_TYPE_INT32:  return (uint32_t) gguf_get_val_i32(g, id);
        default: throw std::runtime_error("unexpected type for GGUF key " + key);
    }
}

static float get_f32(gguf_context * g, const std::string & key) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) throw std::runtime_error("missing GGUF key " + key);
    return gguf_get_val_f32(g, id);
}

static float opt_f32(gguf_context * g, const std::string & key, float def) {
    const int64_t id = gguf_find_key(g, key.c_str());
    return id < 0 ? def : gguf_get_val_f32(g, id);
}

static uint32_t opt_u32(gguf_context * g, const std::string & key, uint32_t def) {
    return gguf_find_key(g, key.c_str()) < 0 ? def : get_u32(g, key);
}

static std::string opt_str(gguf_context * g, const std::string & key, const std::string & def) {
    const int64_t id = gguf_find_key(g, key.c_str());
    return id < 0 ? def : gguf_get_val_str(g, id);
}

static ExpertPart parse_part(const nlohmann::json & j) {
    return { (ggml_type) j.at("type_id").get<int>(), j.at("ne")[0].get<int64_t>(), j.at("ne")[1].get<int64_t>(),
             j.at("offset").get<size_t>(), j.at("bytes").get<size_t>() };
}

Pack::Pack(const std::string & dir) {
    const fs::path d = fs::u8path(dir);
    std::ifstream mf(d / "manifest.json");
    if (!mf) throw std::runtime_error("no manifest.json in " + dir);
    const nlohmann::json m = nlohmann::json::parse(mf);
    // zoro-pack-1: the same layout, written before the project was renamed to Zorlo
    if (m.at("format") != "zorlo-pack-1" && m.at("format") != "zoro-pack-1") throw std::runtime_error("unknown pack format");

    auto parse_layers = [&](const nlohmann::json & arr, std::vector<ExpertLayout> & out) {
        for (const auto & jl : arr) {
            ExpertLayout L;
            L.offset = jl.at("offset").get<uint64_t>();
            L.stride = jl.at("stride").get<size_t>();
            for (const auto & p : jl.at("parts")) {
                const std::string kind = p.at("kind");
                (kind == "gate" ? L.gate : kind == "up" ? L.up : L.down) = parse_part(p);
            }
            if ((size_t) jl.at("layer").get<int>() != out.size()) throw std::runtime_error("layers out of order");
            out.push_back(L);
            max_stride_ = std::max(max_stride_, L.stride);
        }
    };
    parse_layers(m.at("layers"), layers_);
    experts_path_ = (d / m.at("experts").get<std::string>()).string();
    // the low-bit copy is optional: a pulled pack may come without it (zorlo pull without --lowbit)
    if (m.contains("lowbit") && fs::exists(d / fs::u8path(m.at("lowbit").at("experts").get<std::string>()))) {
        parse_layers(m.at("lowbit").at("layers"), lo_layers_);
        lo_experts_path_ = (d / m.at("lowbit").at("experts").get<std::string>()).string();
    }
    core_path_ = (d / m.at("core").get<std::string>()).string();

    gguf_ = gguf_init_from_file(core_path_.c_str(), { /*no_alloc*/ true, &ctx_ });
    if (!gguf_) throw std::runtime_error("cannot load " + core_path_);

    // load every tensor into one buffer, except the token embedding table: a token needs only its own
    // row, so it stays on disk (embed() reads rows). Tied output heads read it whole, so it stays then.
    const size_t data_off = gguf_get_data_offset(gguf_);
    const bool embd_on_disk = ggml_get_tensor(ctx_, "output.weight") != nullptr;
    core_file_.open(fs::u8path(core_path_), std::ios::binary);
    size_t total = 0;
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf_); ++i) total += gguf_get_tensor_size(gguf_, i) + 64;
    data_.resize(total);
    uint8_t * p = data_.data();
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf_); ++i) {
        ggml_tensor * t = ggml_get_tensor(ctx_, gguf_get_tensor_name(gguf_, i));
        const size_t n = gguf_get_tensor_size(gguf_, i);
        if (embd_on_disk && strcmp(t->name, "token_embd.weight") == 0) {
            embd_ = t;
            embd_off_ = data_off + gguf_get_tensor_offset(gguf_, i);
            continue;
        }
        p = (uint8_t *) (((uintptr_t) p + 63) & ~(uintptr_t) 63);
        core_file_.seekg((std::streamoff) (data_off + gguf_get_tensor_offset(gguf_, i)));
        core_file_.read((char *) p, (std::streamsize) n);
        if ((size_t) core_file_.gcount() != n) throw std::runtime_error("short read in " + core_path_);
        t->data = p;
        p += n;
        core_bytes_ += n;
    }

    const std::string a = gguf_get_val_str(gguf_, gguf_find_key(gguf_, "general.architecture"));
    hp_.arch          = a;
    hp_.n_layer       = (int) get_u32(gguf_, a + ".block_count");
    hp_.n_embd        = (int) get_u32(gguf_, a + ".embedding_length");
    hp_.n_head        = (int) get_u32(gguf_, a + ".attention.head_count");
    hp_.n_head_kv     = (int) get_u32(gguf_, a + ".attention.head_count_kv");
    hp_.head_dim      = (int) get_u32(gguf_, a + ".attention.key_length");
    hp_.n_expert      = (int) get_u32(gguf_, a + ".expert_count");
    hp_.n_expert_used = (int) get_u32(gguf_, a + ".expert_used_count");
    hp_.n_ff_exp      = (int) get_u32(gguf_, a + ".expert_feed_forward_length");
    hp_.n_ctx_train   = (int) get_u32(gguf_, a + ".context_length");
    hp_.rms_eps       = get_f32(gguf_, a + ".attention.layer_norm_rms_epsilon");
    hp_.rope_base     = get_f32(gguf_, a + ".rope.freq_base");
    hp_.n_vocab       = (int) core("token_embd.weight")->ne[1];

    // same derivation as llama.cpp (llama-context.cpp), so RoPE matches bit for bit
    const std::string scaling = opt_str(gguf_, a + ".rope.scaling.type", "none");
    const float factor = opt_f32(gguf_, a + ".rope.scaling.factor", 0.0f);
    if (scaling != "none" && factor > 0) hp_.rope_freq_scale = 1.0f / factor;
    hp_.n_ctx_orig = (int) opt_u32(gguf_, a + ".rope.scaling.original_context_length", hp_.n_ctx_train);
    hp_.yarn_beta_fast = opt_f32(gguf_, a + ".rope.scaling.yarn_beta_fast", 32.0f);
    hp_.yarn_beta_slow = opt_f32(gguf_, a + ".rope.scaling.yarn_beta_slow", 1.0f);
    if (scaling == "yarn") {
        hp_.rope_ext_factor = 1.0f;
        auto mscale = [](float s, float m) { return s <= 1.0f ? 1.0f : 0.1f * m * logf(s) + 1.0f; };
        const float f = 1.0f / hp_.rope_freq_scale;
        const float log_mul = opt_f32(gguf_, a + ".rope.scaling.yarn_log_multiplier", 0.0f);
        hp_.rope_attn_factor = log_mul != 0.0f ? mscale(f, 1.0f) / mscale(f, log_mul) : mscale(f, 1.0f);
        hp_.rope_attn_factor *= 1.0f / (1.0f + 0.1f * logf(f));
    }
    hp_.n_swa = (int) opt_u32(gguf_, a + ".attention.sliding_window", 0);
    hp_.swa_pattern = (int) opt_u32(gguf_, a + ".attention.sliding_window_pattern", hp_.n_swa ? 2 : 1);
    hp_.expert_weights_scale = opt_f32(gguf_, a + ".expert_weights_scale", 0.0f);
    if ((int) layers_.size() != hp_.n_layer) throw std::runtime_error("manifest layer count != block_count");
}

Pack::~Pack() {
    if (gguf_) gguf_free(gguf_);
    if (ctx_) ggml_free(ctx_);
}

ggml_tensor * Pack::core(const std::string & name) const {
    ggml_tensor * t = ggml_get_tensor(ctx_, name.c_str());
    if (!t) throw std::runtime_error("missing tensor " + name);
    return t;
}

void Pack::embed(const int32_t * ids, int n, float * out) const {
    ggml_tensor * t = embd_ ? embd_ : core("token_embd.weight");
    const size_t row = ggml_row_size(t->type, t->ne[0]);
    const auto to_float = ggml_get_type_traits(t->type)->to_float;  // what ggml's get_rows uses
    std::vector<uint8_t> buf(row);
    for (int i = 0; i < n; ++i) {
        if (ids[i] < 0 || ids[i] >= t->ne[1]) throw std::runtime_error("token id out of range");
        const uint8_t * src = (const uint8_t *) t->data + (size_t) ids[i] * row;
        if (embd_) {
            core_file_.seekg((std::streamoff) (embd_off_ + (uint64_t) ids[i] * row));
            core_file_.read((char *) buf.data(), (std::streamsize) row);
            src = buf.data();
        }
        if (to_float) to_float(src, out + (size_t) i * t->ne[0], t->ne[0]);
        else memcpy(out + (size_t) i * t->ne[0], src, row);  // f32 table
    }
}

ggml_tensor * Pack::core_opt(const char * fmt, int il) const {
    char buf[128];
    snprintf(buf, sizeof(buf), fmt, il);
    return ggml_get_tensor(ctx_, buf);
}

ggml_tensor * Pack::core(const char * fmt, int il) const {
    char buf[128];
    snprintf(buf, sizeof(buf), fmt, il);
    return core(std::string(buf));
}

}  // namespace zorlo
