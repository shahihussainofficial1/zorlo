#pragma once

// MoE forward pass (qwen3moe, gpt-oss) with experts streamed through ExpertCache.
//
// Per layer: one ggml graph for attention + routing (all weights resident), then the selected
// experts are fetched from the cache and computed in a second graph. The ops mirror llama.cpp's
// graphs for these architectures (flash attention, f16 KV cache, same routing and activation,
// expert outputs summed in top-k order) so greedy output matches llama.cpp.

#include "expert_cache.h"
#include "pack.h"

#include <cstdint>
#include <cstdio>
#include <vector>

struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;
struct ggml_threadpool;
struct ggml_backend;

namespace zorlo {

class MoeModel {
public:
    // prefetch_extra < 0 disables prefetch; otherwise top-(k + extra) predicted experts of the next layer.
    // n_seq independent sequences (slots), each with its own n_ctx KV cache
    MoeModel(const Pack & pack, ExpertCache & cache, int n_threads, int n_ctx, int prefetch_extra = 0, int n_seq = 1);
    ~MoeModel();
    MoeModel(const MoeModel &) = delete;
    MoeModel & operator=(const MoeModel &) = delete;

    // opt-in approximations (all off = output identical to llama.cpp); see README for measured loss
    struct Lossy {
        float skip_below = 0;     // drop routed experts whose renormalized weight is below this
        float prefer_cached = 0;  // swap an uncached pick for a cached expert with prob >= (1 - this) * its prob
        float expert_mass = 0;    // keep a token's top experts until their weights reach this share, drop the rest
        bool  lowbit = false;     // misses outside the hot set read the pack's low-bit copy
        bool  routing() const { return skip_below > 0 || prefer_cached > 0 || expert_mass > 0; }
    };
    Lossy lossy;  // what eval() uses; eval_batch takes it per sequence

    // importance matrix for requantizing experts (zorlo imatrix): when set, every expert evaluation adds the
    // squares of its inputs, as [layer][expert][n_embd (gate/up input), n_ff (down input)]
    std::vector<float> * imatrix = nullptr;
    size_t imatrix_size() const { return (size_t) hp_.n_layer * hp_.n_expert * (hp_.n_embd + hp_.n_ff_exp); }

    // runs `tokens` at positions n_past() .. n_past()+n-1 of slot 0 and returns the logits of the last one
    const std::vector<float> & eval(const std::vector<int32_t> & tokens);
    // several slots in one step: attention per sequence, each expert read once for all their tokens. Every
    // sequence gets its own matmuls, so its results do not depend on the rest of the batch. Returns, per entry
    // of `seqs` in order, [n_vocab] logits of its last token, or of all its tokens with all_logits
    struct Seq {
        int slot;
        std::vector<int32_t> tokens;
        bool all_logits = false;  // also: computed exactly as if its tokens came one at a time (a draft check)
        Lossy lossy;
    };
    const std::vector<float> & eval_batch(const std::vector<Seq> & seqs);
    void reset(int slot = 0) { n_past_[slot] = 0; }
    // keep the first n cached positions (later cells are masked out, then overwritten)
    void truncate(int n, int slot = 0) { if (n < n_past_[slot]) n_past_[slot] = n; }
    int n_past(int slot = 0) const { return n_past_[slot]; }
    int n_ctx() const { return n_ctx_; }
    int n_seq() const { return (int) n_past_.size(); }

    // KV cells [0, n) exactly as stored: per layer K then V, each n cells of f16 [head_dim][n_head_kv]
    size_t kv_bytes(int n) const { return 2ull * hp_.n_layer * hp_.head_dim * hp_.n_head_kv * sizeof(uint16_t) * n; }
    void save_kv(std::FILE * f, int n, int slot = 0) const;
    // read the first n cells of a block of `stored` cells written by save_kv; the slot then holds n positions
    void load_kv(std::FILE * f, int n, int stored, int slot = 0);


    struct Timing {
        double attn_s = 0, experts_s = 0, head_s = 0;
        uint64_t pred_total = 0, pred_hit = 0;  // next-layer prediction recall
        uint64_t routed = 0, skipped = 0, swapped = 0;  // lossy switch activity
    };
    Timing timing;

private:
    ggml_context * graph_ctx();
    void compute(ggml_cgraph * gf);
    void attention_and_route(int il);
    void experts(int il, int T);
    void apply_lossy(int il, int T);
    void compute_experts(int il, int T, const std::vector<int> & which, const std::vector<int> & slot_of,
                         const std::vector<std::vector<int>> & uses);

    const Pack & pack_;
    ExpertCache & cache_;
    HParams hp_;
    int n_threads_, n_ctx_, prefetch_extra_;
    std::vector<int> n_past_;  // per slot
    bool oss_ = false;  // gpt-oss: biases, attention sinks, sliding window, softmax-of-top-k routing

    ggml_backend * backend_ = nullptr;  // CPU backend from the ggml registry
    ggml_threadpool * threadpool_ = nullptr;
    void (*threadpool_free_)(ggml_threadpool *) = nullptr;
    std::vector<uint8_t> arena_;
    ggml_context * kv_ctx_ = nullptr;
    std::vector<ggml_tensor *> k_cache_, v_cache_;  // [slot * n_layer + layer]

    // one sequence of the current batch: its rows in the host buffers below, and its attention masks
    struct Run {
        int slot, row, T, n_kv;
        bool like_decode;  // a draft check: same math as evaluating its tokens one at a time
        Lossy lossy;
        std::vector<uint16_t> mask, mask_swa;  // f16 [T][n_kv]: causal, causal + sliding window
    };
    std::vector<Run> runs_;

    // host-side state for the current batch
    std::vector<float>    x_;       // residual stream [T][n_embd]
    std::vector<float>    hn_;      // ffn_norm output [T][n_embd]
    std::vector<int32_t>  sel_;     // selected experts [T][k]
    std::vector<float>    w_;       // their weights [T][k]
    std::vector<int32_t>  pred_;    // predicted experts of the next layer (decode only)
    std::vector<float>    probs_;   // router probabilities of all experts, decode only (prefer_cached)
    std::vector<float>    eout_;    // expert outputs [T][k][n_embd]
    std::vector<float>    logits_;  // [seq][n_vocab]
};

}  // namespace zorlo
