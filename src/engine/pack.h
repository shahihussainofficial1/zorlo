#pragma once

// A Zorlo pack (made by zorlo-pack): core.gguf resident in RAM, experts.bin streamed from disk.

#include "ggml.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

struct gguf_context;

namespace zorlo {

struct ExpertPart {
    ggml_type type;
    int64_t   ne0, ne1;  // matrix of one expert: ne0 = row length, ne1 = rows
    size_t    offset;    // inside the blob
    size_t    bytes;
};

// one layer's experts in experts.bin: expert e is the blob at offset + e * stride
struct ExpertLayout {
    uint64_t   offset;
    size_t     stride;
    ExpertPart gate, up, down;
};

struct HParams {
    std::string arch;
    int   n_layer, n_embd, n_head, n_head_kv, head_dim;
    int   n_expert, n_expert_used, n_ff_exp;
    int   n_ctx_train, n_vocab;
    float rms_eps, rope_base;
    // RoPE scaling (YaRN when rope_ext_factor != 0), as llama.cpp derives it from the GGUF
    float rope_freq_scale = 1, rope_ext_factor = 0, rope_attn_factor = 1, yarn_beta_fast = 32, yarn_beta_slow = 1;
    int   n_ctx_orig = 0;
    // sliding-window attention on layers with il % swa_pattern < swa_pattern - 1 (n_swa = 0: none)
    int   n_swa = 0, swa_pattern = 1;
    float expert_weights_scale = 0;  // 0 or 1: unscaled

    bool is_swa(int il) const { return n_swa > 0 && il % swa_pattern < swa_pattern - 1; }
};

class Pack {
public:
    explicit Pack(const std::string & dir);
    ~Pack();
    Pack(const Pack &) = delete;
    Pack & operator=(const Pack &) = delete;

    const HParams & hp() const { return hp_; }
    const ExpertLayout & layer(int il) const { return layers_.at(il); }
    // optional low-bit copy of every expert (zorlo-pack --lowbit)
    bool has_lowbit() const { return !lo_layers_.empty(); }
    const ExpertLayout & lo_layer(int il) const { return lo_layers_.at(il); }
    const std::string & lo_experts_path() const { return lo_experts_path_; }
    size_t max_stride() const { return max_stride_; }
    size_t core_bytes() const { return core_bytes_; }
    const std::string & experts_path() const { return experts_path_; }
    const std::string & core_path() const { return core_path_; }

    // resident tensor by GGUF name; throws if missing
    ggml_tensor * core(const std::string & name) const;
    ggml_tensor * core(const char * fmt, int il) const;
    // nullptr if the model has no such tensor (biases, norms some architectures lack)
    ggml_tensor * core_opt(const char * fmt, int il = -1) const;
    // token embeddings as f32 [n][n_embd]; rows come from disk unless the table is resident
    void embed(const int32_t * ids, int n, float * out) const;

private:
    HParams hp_{};
    std::vector<ExpertLayout> layers_, lo_layers_;
    size_t max_stride_ = 0, core_bytes_ = 0;
    std::string experts_path_, core_path_, lo_experts_path_;
    gguf_context * gguf_ = nullptr;
    ggml_context * ctx_ = nullptr;  // tensor metadata; data lives in data_
    std::vector<uint8_t> data_;
    mutable std::ifstream core_file_;
    ggml_tensor * embd_ = nullptr;  // set when the embedding table stays on disk
    uint64_t embd_off_ = 0;
};

}  // namespace zorlo
