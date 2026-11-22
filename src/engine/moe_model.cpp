#include "moe_model.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace zorlo {

static constexpr int KV_PAD = 256;           // llama.cpp pads the attended KV range to 256 cells
static constexpr int EXPERTS_PER_GRAPH = 64;  // cap on experts computed in one graph (prefill)

using clk = std::chrono::steady_clock;
static double since(clk::time_point t0) { return std::chrono::duration<double>(clk::now() - t0).count(); }

MoeModel::MoeModel(const Pack & pack, ExpertCache & cache, int n_threads, int n_ctx, int prefetch_extra, int n_seq)
    : pack_(pack), cache_(cache), hp_(pack.hp()), n_threads_(n_threads), prefetch_extra_(prefetch_extra),
      n_past_(std::max(1, n_seq), 0) {
    oss_ = hp_.arch == "gpt-oss";
    if (!oss_ && hp_.arch != "qwen3moe") throw std::runtime_error("unsupported model arch " + hp_.arch + " (qwen3moe, gpt-oss)");
    n_ctx_ = (n_ctx + KV_PAD - 1) / KV_PAD * KV_PAD;

    // CPU backend through the registry, so the engine also works when ggml-cpu is a runtime-loaded DLL
    // (one per CPU variant); in a static build it is registered already and nothing is loaded
    if (!ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev || !(backend_ = ggml_backend_dev_init(dev, nullptr))) throw std::runtime_error("no ggml CPU backend found");
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto proc = [&](const char * name) {
        void * p = ggml_backend_reg_get_proc_address(reg, name);
        if (!p) throw std::runtime_error(std::string("ggml CPU backend lacks ") + name);
        return p;
    };
    using threadpool_new_t = ggml_threadpool * (*)(ggml_threadpool_params *);
    using set_threadpool_t = void (*)(ggml_backend_t, ggml_threadpool *);
    threadpool_free_ = (void (*)(ggml_threadpool *)) proc("ggml_threadpool_free");
    ggml_threadpool_params tp = ggml_threadpool_params_default(n_threads_);
    threadpool_ = ((threadpool_new_t) proc("ggml_threadpool_new"))(&tp);
    ((ggml_backend_set_n_threads_t) proc("ggml_backend_set_n_threads"))(backend_, n_threads_);
    ((set_threadpool_t) proc("ggml_backend_cpu_set_threadpool"))(backend_, threadpool_);

    // f16 KV cache, zeroed: cells beyond the current position are masked but still read by flash attention
    // shortcut: sliding-window layers get a full-length cache and a window mask, which costs
    // n_ctx cells instead of n_swa per such layer; use a ring buffer if long contexts run short of RAM
    const size_t n_kv_tensors = 2ull * hp_.n_layer * n_past_.size();
    kv_ctx_ = ggml_init({ kv_bytes(n_ctx_) * n_past_.size() + n_kv_tensors * ggml_tensor_overhead() + 1024, nullptr, false });
    for (size_t il = 0; il < hp_.n_layer * n_past_.size(); ++il) {
        k_cache_.push_back(ggml_new_tensor_3d(kv_ctx_, GGML_TYPE_F16, hp_.head_dim, hp_.n_head_kv, n_ctx_));
        v_cache_.push_back(ggml_new_tensor_3d(kv_ctx_, GGML_TYPE_F16, hp_.head_dim, hp_.n_head_kv, n_ctx_));
        memset(k_cache_.back()->data, 0, ggml_nbytes(k_cache_.back()));
        memset(v_cache_.back()->data, 0, ggml_nbytes(v_cache_.back()));
    }
}

MoeModel::~MoeModel() {
    if (kv_ctx_) ggml_free(kv_ctx_);
    if (backend_) ggml_backend_free(backend_);
    if (threadpool_) threadpool_free_(threadpool_);
}

void MoeModel::save_kv(std::FILE * f, int n, int slot) const {
    if (n > n_past_[slot]) throw std::runtime_error("save_kv: only " + std::to_string(n_past_[slot]) + " cells cached");
    const size_t cell = ggml_nbytes(k_cache_[0]) / n_ctx_;
    for (int il = 0; il < hp_.n_layer; ++il) {
        const int i = slot * hp_.n_layer + il;
        for (const ggml_tensor * t : { k_cache_[i], v_cache_[i] }) {
            if (std::fwrite(t->data, cell, n, f) != (size_t) n) throw std::runtime_error("save_kv: write failed");
        }
    }
}

void MoeModel::load_kv(std::FILE * f, int n, int stored, int slot) {
    if (n > stored || n > n_ctx_) throw std::runtime_error("load_kv: bad cell count");
    const size_t cell = ggml_nbytes(k_cache_[0]) / n_ctx_;
    n_past_[slot] = 0;  // nothing valid until every layer is read
    for (int il = 0; il < hp_.n_layer; ++il) {
        const int i = slot * hp_.n_layer + il;
        for (ggml_tensor * t : { k_cache_[i], v_cache_[i] }) {
            if (std::fread(t->data, cell, n, f) != (size_t) n || _fseeki64(f, (int64_t) cell * (stored - n), SEEK_CUR) != 0) {
                throw std::runtime_error("load_kv: short read");
            }
        }
    }
    n_past_[slot] = n;
}

ggml_context * MoeModel::graph_ctx() {
    return ggml_init({ arena_.size(), arena_.data(), false });
}

void MoeModel::compute(ggml_cgraph * gf) {
    // same plan as before: n_threads_ on threadpool_, work buffer kept by the backend
    if (ggml_backend_graph_compute(backend_, gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("ggml graph compute failed");
}

static void copy_out(const ggml_tensor * t, float * dst) {
    // t is [ne0, ne1] f32, possibly strided along ne1
    for (int64_t j = 0; j < t->ne[1]; ++j) {
        memcpy(dst + j * t->ne[0], (const char *) t->data + j * t->nb[1], t->ne[0] * sizeof(float));
    }
}

void MoeModel::attention_and_route(int il) {
    const int E = hp_.n_embd, H = hp_.n_head, HKV = hp_.n_head_kv, D = hp_.head_dim, K = hp_.n_expert_used;
    const float eps = hp_.rms_eps;
    const bool solo_decode = runs_.size() == 1 && runs_[0].T == 1;  // single-sequence decode: lossy + stats
    std::vector<int32_t> pred_next;

  for (const Run & r : runs_) {  // each sequence attends to its own KV: one graph per sequence
    const int T = r.T, n_kv = r.n_kv, n_past = n_past_[r.slot];
    ggml_context * ctx = graph_ctx();
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 256, false);

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, T);
    memcpy(x->data, x_.data() + (size_t) r.row * E, (size_t) E * T * sizeof(float));
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    for (int t = 0; t < T; ++t) ((int32_t *) pos->data)[t] = n_past + t;
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, T);
    memcpy(mask->data, (hp_.is_swa(il) ? r.mask_swa : r.mask).data(), (size_t) n_kv * T * sizeof(uint16_t));

    // W x (+ b): gpt-oss has biases on attention and router, qwen3moe has none
    auto linear = [&](const char * w, const char * b, int l, ggml_tensor * in) {
        ggml_tensor * y = ggml_mul_mat(ctx, pack_.core(w, l), in);
        if (ggml_tensor * bias = pack_.core_opt(b, l)) y = ggml_add(ctx, y, bias);
        return y;
    };
    auto ffn_norm = [&](int l) {
        ggml_tensor * n = pack_.core_opt("blk.%d.ffn_norm.weight", l);
        return n ? n : pack_.core("blk.%d.post_attention_norm.weight", l);
    };
    auto rope = [&](ggml_tensor * t) {
        return ggml_rope_ext(ctx, t, pos, nullptr, D, GGML_ROPE_TYPE_NEOX, hp_.n_ctx_orig, hp_.rope_base,
                             hp_.rope_freq_scale, hp_.rope_ext_factor, hp_.rope_attn_factor, hp_.yarn_beta_fast,
                             hp_.yarn_beta_slow);
    };

    ggml_tensor * cur = ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), pack_.core("blk.%d.attn_norm.weight", il));

    ggml_tensor * q = ggml_reshape_3d(ctx, linear("blk.%d.attn_q.weight", "blk.%d.attn_q.bias", il, cur), D, H, T);
    ggml_tensor * k = ggml_reshape_3d(ctx, linear("blk.%d.attn_k.weight", "blk.%d.attn_k.bias", il, cur), D, HKV, T);
    ggml_tensor * v = ggml_reshape_3d(ctx, linear("blk.%d.attn_v.weight", "blk.%d.attn_v.bias", il, cur), D, HKV, T);

    if (ggml_tensor * qn = pack_.core_opt("blk.%d.attn_q_norm.weight", il)) q = ggml_mul(ctx, ggml_rms_norm(ctx, q, eps), qn);
    q = rope(q);
    if (ggml_tensor * kn = pack_.core_opt("blk.%d.attn_k_norm.weight", il)) k = ggml_mul(ctx, ggml_rms_norm(ctx, k, eps), kn);
    k = rope(k);

    // write this batch into the KV cache before attention reads it (graph nodes run in order)
    ggml_tensor * kc = k_cache_[r.slot * hp_.n_layer + il], * vc = v_cache_[r.slot * hp_.n_layer + il];
    ggml_build_forward_expand(gf, ggml_cpy(ctx, k, ggml_view_3d(ctx, kc, D, HKV, T, kc->nb[1], kc->nb[2], (size_t) n_past * kc->nb[2])));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, v, ggml_view_3d(ctx, vc, D, HKV, T, vc->nb[1], vc->nb[2], (size_t) n_past * vc->nb[2])));

    ggml_tensor * kall = ggml_permute(ctx, ggml_view_3d(ctx, kc, D, HKV, n_kv, kc->nb[1], kc->nb[2], 0), 0, 2, 1, 3);
    ggml_tensor * vall = ggml_permute(ctx, ggml_view_3d(ctx, vc, D, HKV, n_kv, vc->nb[1], vc->nb[2], 0), 0, 2, 1, 3);
    ggml_tensor * att = ggml_flash_attn_ext(ctx, ggml_permute(ctx, q, 0, 2, 1, 3), kall, vall, mask,
                                            1.0f / sqrtf((float) D), 0.0f, 0.0f);
    if (ggml_tensor * sinks = pack_.core_opt("blk.%d.attn_sinks.weight", il)) ggml_flash_attn_ext_add_sinks(att, sinks);
    ggml_flash_attn_ext_set_n_kv_max(att, 0);
    ggml_prec_set_acc(att, GGML_PREC_F32);
    att = ggml_reshape_2d(ctx, att, (int64_t) D * H, T);
    ggml_tensor * ffn_inp = ggml_add(ctx, linear("blk.%d.attn_output.weight", "blk.%d.attn_output.bias", il, att), x);

    ggml_tensor * hn = ggml_mul(ctx, ggml_rms_norm(ctx, ffn_inp, eps), ffn_norm(il));

    // the router is F32, and llamafile's F32 GEMM takes over from 2 columns with a different summation:
    // a draft check routes one token per matmul so it matches decoding them one at a time
    ggml_tensor * logits = nullptr;
    for (int t = 0; t < (r.like_decode ? T : 1); ++t) {
        ggml_tensor * in = r.like_decode ? ggml_view_2d(ctx, hn, E, 1, hn->nb[1], t * hn->nb[1]) : hn;
        ggml_tensor * l = linear("blk.%d.ffn_gate_inp.weight", "blk.%d.ffn_gate_inp.bias", il, in);
        logits = logits ? ggml_concat(ctx, logits, l, 1) : l;
    }
    ggml_tensor * probs = nullptr, * sel, * w;
    if (oss_) {
        // top-k on the raw logits, softmax over just the k picked (llama.cpp's SOFTMAX_WEIGHT gating)
        sel = ggml_argsort_top_k(ctx, logits, K);
        w = ggml_get_rows(ctx, ggml_reshape_3d(ctx, logits, 1, hp_.n_expert, T), sel);
        w = ggml_soft_max(ctx, ggml_reshape_2d(ctx, w, K, T));
        if (solo_decode && runs_[0].lossy.prefer_cached > 0) probs = ggml_soft_max(ctx, logits);
    } else {
        // softmax over all experts, top-k, weights renormalized to sum 1
        probs = ggml_soft_max(ctx, logits);
        sel = ggml_argsort_top_k(ctx, probs, K);
        w = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, hp_.n_expert, T), sel);
        w = ggml_reshape_2d(ctx, w, K, T);
        w = ggml_div(ctx, w, ggml_clamp(ctx, ggml_sum_rows(ctx, w), 6.103515625e-5f, INFINITY));
    }
    if (hp_.expert_weights_scale != 0.0f && hp_.expert_weights_scale != 1.0f) w = ggml_scale(ctx, w, hp_.expert_weights_scale);

    // decode: guess the next layer's experts by running its router on this layer's post-attention
    // state (adjacent layers' hidden states are close), so their reads can start early
    ggml_tensor * pred = nullptr;
    if (T == 1 && prefetch_extra_ >= 0 && il + 1 < hp_.n_layer) {
        ggml_tensor * hnext = ggml_mul(ctx, ggml_rms_norm(ctx, ffn_inp, eps), ffn_norm(il + 1));
        ggml_tensor * lnext = linear("blk.%d.ffn_gate_inp.weight", "blk.%d.ffn_gate_inp.bias", il + 1, hnext);
        pred = ggml_argsort_top_k(ctx, lnext, std::min(hp_.n_expert, K + prefetch_extra_));
    }

    for (ggml_tensor * t : { ffn_inp, hn, sel, w }) ggml_build_forward_expand(gf, t);
    if (pred) ggml_build_forward_expand(gf, pred);
    if (probs) ggml_build_forward_expand(gf, probs);
    compute(gf);
    if (solo_decode && runs_[0].lossy.prefer_cached > 0) probs_.assign((const float *) probs->data, (const float *) probs->data + hp_.n_expert);

    copy_out(ffn_inp, x_.data() + (size_t) r.row * E);
    copy_out(hn, hn_.data() + (size_t) r.row * E);
    copy_out(w, w_.data() + (size_t) r.row * K);
    for (int t = 0; t < T; ++t) {
        memcpy(&sel_[(size_t) (r.row + t) * K], (const char *) sel->data + t * sel->nb[1], K * sizeof(int32_t));
    }
    if (pred) {  // union over the batch's decoding sequences
        for (int64_t i = 0; i < pred->ne[0]; ++i) {
            const int32_t e = ((const int32_t *) pred->data)[i];
            if (std::find(pred_next.begin(), pred_next.end(), e) == pred_next.end()) pred_next.push_back(e);
        }
    }
    ggml_free(ctx);
  }

    // recall of the prediction made for this layer
    if (solo_decode && !pred_.empty()) {
        timing.pred_total += K;
        for (int i = 0; i < K; ++i) {
            timing.pred_hit += std::find(pred_.begin(), pred_.end(), sel_[i]) != pred_.end();
        }
    }
    pred_ = std::move(pred_next);
}

void MoeModel::compute_experts(int il, int T, const std::vector<int> & which, const std::vector<int> & slot_of,
                         const std::vector<std::vector<int>> & uses) {
    const int E = hp_.n_embd, K = hp_.n_expert_used;

    ggml_context * ctx = graph_ctx();
    ggml_context * wctx = ggml_init({ 3 * which.size() * ggml_tensor_overhead() + 1024, nullptr, true });
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16 * which.size() * runs_.size() + 16, false);

    ggml_tensor * hn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, T);
    memcpy(hn->data, hn_.data(), (size_t) E * T * sizeof(float));

    auto weight = [&](const ExpertPart & p, const uint8_t * blob) {
        ggml_tensor * t = ggml_new_tensor_2d(wctx, p.type, p.ne0, p.ne1);
        t->data = (void *) (blob + p.offset);
        return t;
    };
    // one matmul per sequence: ggml picks its kernel by column count (tiled from 8 on), so a sequence's
    // math must not depend on what else is in the batch; the expert is still read once
    auto run_of = [&](int t) {
        int r = 0;
        while (t >= runs_[r].row + runs_[r].T) ++r;
        return r;
    };
    std::vector<std::pair<ggml_tensor *, std::vector<int>>> ys;  // output, its (token, top-k slot) entries
    std::vector<ggml_tensor *> acts;  // down-projection input of each, for the importance matrix
    for (int c : which) {
        const uint8_t * b = cache_.data(slot_of[c]);
        const ExpertLayout & L = cache_.is_lo(slot_of[c]) ? pack_.lo_layer(il) : pack_.layer(il);
        ggml_tensor * wg = weight(L.gate, b), * wu = weight(L.up, b), * wd = weight(L.down, b);
        std::vector<std::vector<int>> parts;  // uses[c] is in token order, so each sequence's entries are adjacent
        for (int i = 0, last = -1; i < (int) uses[c].size(); ++i) {
            const int r = run_of(uses[c][i] / K);
            if (r != last) parts.emplace_back(), last = r;
            parts.back().push_back(uses[c][i]);
        }
        for (auto & u : parts) {
            ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) u.size());
            for (size_t j = 0; j < u.size(); ++j) ((int32_t *) ids->data)[j] = u[j] / K;
            ggml_tensor * xe = ggml_get_rows(ctx, hn, ids);
            ggml_tensor * g = ggml_mul_mat(ctx, wg, xe);
            ggml_tensor * up = ggml_mul_mat(ctx, wu, xe);
            ggml_tensor * y;
            if (oss_) {
                // per-expert biases stay resident: column e of [n, n_expert]; clamped OpenAI SwiGLU
                const int e = sel_[u[0]];
                auto bias = [&](const char * fmt) {
                    ggml_tensor * t = pack_.core(fmt, il);
                    return ggml_view_1d(ctx, t, t->ne[0], e * t->nb[1]);
                };
                up = ggml_add(ctx, up, bias("blk.%d.ffn_up_exps.bias"));
                g = ggml_add(ctx, g, bias("blk.%d.ffn_gate_exps.bias"));
                acts.push_back(ggml_swiglu_oai(ctx, g, up, 1.702f, 7.0f));
                y = ggml_mul_mat(ctx, wd, acts.back());
                y = ggml_add(ctx, y, bias("blk.%d.ffn_down_exps.bias"));
            } else {
                acts.push_back(ggml_swiglu_split(ctx, g, up));
                y = ggml_mul_mat(ctx, wd, acts.back());
            }
            ggml_build_forward_expand(gf, y);
            ys.emplace_back(y, std::move(u));
        }
    }
    compute(gf);

    if (imatrix) {  // squared inputs per expert: gate/up see hn, down sees the activation
        const int F = hp_.n_ff_exp;
        for (size_t i = 0; i < ys.size(); ++i) {
            const auto & u = ys[i].second;
            float * dst = imatrix->data() + ((size_t) il * hp_.n_expert + sel_[u[0]]) * (E + F);
            for (size_t j = 0; j < u.size(); ++j) {
                const float * x = &hn_[(size_t) (u[j] / K) * E];
                const float * a = (const float *) ((const char *) acts[i]->data + j * acts[i]->nb[1]);
                for (int k = 0; k < E; ++k) dst[k] += x[k] * x[k];
                for (int k = 0; k < F; ++k) dst[E + k] += a[k] * a[k];
            }
        }
    }
    for (const auto & [y, u] : ys) {
        for (size_t j = 0; j < u.size(); ++j) {
            memcpy(&eout_[(size_t) u[j] * E], (const char *) y->data + j * y->nb[1], E * sizeof(float));
        }
    }
    ggml_free(wctx);
    ggml_free(ctx);
}

void MoeModel::apply_lossy(int il, int T) {
    const int K = hp_.n_expert_used;
    timing.routed += (uint64_t) T * K;
    const Lossy & solo = runs_[0].lossy;
    if (T == 1 && solo.prefer_cached > 0) {
        bool changed = false;
        for (int i = 0; i < K; ++i) {
            if (cache_.resident(il, sel_[i])) continue;
            int best = -1;
            for (int c = 0; c < hp_.n_expert; ++c) {
                if (probs_[c] < (1.0f - solo.prefer_cached) * probs_[sel_[i]]) continue;
                if (std::find(sel_.begin(), sel_.begin() + K, c) != sel_.begin() + K) continue;
                if ((best < 0 || probs_[c] > probs_[best]) && cache_.resident(il, c)) best = c;
            }
            if (best >= 0) {
                sel_[i] = best;
                changed = true;
                ++timing.swapped;
            }
        }
        if (changed) {  // weights = probs of the new picks, renormalized like the router does
            float sum = 0;
            for (int i = 0; i < K; ++i) sum += probs_[sel_[i]];
            for (int i = 0; i < K; ++i) w_[i] = probs_[sel_[i]] / sum;
        }
    }
    for (const Run & r : runs_) {
        if (r.lossy.skip_below <= 0 && r.lossy.expert_mass <= 0) continue;
        for (int t = r.row; t < r.row + r.T; ++t) {
            int32_t * s = &sel_[(size_t) t * K];
            float * w = &w_[(size_t) t * K];
            float sum = 0, kept = 0;
            for (int i = 0; i < K; ++i) {  // picks are in descending weight order; the top one always stays
                const bool mass_done = r.lossy.expert_mass > 0 && kept >= r.lossy.expert_mass;
                if (i > 0 && (w[i] < r.lossy.skip_below || mass_done)) {
                    s[i] = -1;
                    ++timing.skipped;
                } else {
                    sum += w[i];
                }
                kept += w[i];
            }
            for (int i = 0; i < K; ++i) w[i] = s[i] < 0 ? 0.0f : w[i] / sum;
        }
    }
}
void MoeModel::experts(int il, int T) {
    const int E = hp_.n_embd, K = hp_.n_expert_used;
    bool lo_ok = true;  // low-bit copies only if every sequence in the batch accepts them
    bool routing = false;
    for (const Run & r : runs_) lo_ok = lo_ok && r.lossy.lowbit, routing = routing || r.lossy.routing();
    if (routing) apply_lossy(il, T);

    // group (token, top-k slot) pairs by expert, experts in order of first use
    std::vector<int> order;
    std::vector<std::vector<int>> uses;  // per entry of `order`: list of t*K + slot
    std::unordered_map<int, int> idx;
    for (int i = 0; i < T * K; ++i) {
        if (sel_[i] < 0) continue;  // dropped by skip_below
        auto it = idx.find(sel_[i]);
        if (it == idx.end()) {
            it = idx.emplace(sel_[i], (int) order.size()).first;
            order.push_back(sel_[i]);
            uses.emplace_back();
        }
        uses[it->second].push_back(i);
    }

    // request a chunk of experts (all of them at decode), then compute each batch as its reads land;
    // reads run on the cache's I/O threads, so compute overlaps the remaining reads
    std::vector<int> slot_of(order.size());
    const int chunk = cache_.n_slots();
    for (size_t c0 = 0; c0 < order.size(); c0 += chunk) {
        const size_t c1 = std::min(order.size(), c0 + (size_t) chunk);
        std::unordered_map<int, int> by_slot;
        std::vector<int> pending;
        for (size_t c = c0; c < c1; ++c) {
            slot_of[c] = cache_.request(il, order[c], lo_ok);
            by_slot[slot_of[c]] = (int) c;
            pending.push_back(slot_of[c]);
        }
        // prefetch after this layer's demand reads are queued, so they go first
        if (c1 == order.size() && !pred_.empty()) {
            cache_.cancel_prefetch();
            for (int32_t e : pred_) cache_.prefetch(il + 1, e, lo_ok);
        }
        while (!pending.empty()) {
            std::vector<int> which;
            for (int s : cache_.wait_any(pending)) which.push_back(by_slot[s]);
            for (size_t b = 0; b < which.size(); b += EXPERTS_PER_GRAPH) {
                const std::vector<int> part(which.begin() + b, which.begin() + std::min(which.size(), b + EXPERTS_PER_GRAPH));
                compute_experts(il, T, part, slot_of, uses);
            }
        }
        cache_.release_all();
    }

    // x = sum_k(expert_k * w_k) + ffn_inp, multiplied then added in top-k order like llama.cpp
    std::vector<float> acc(E);
    for (int t = 0; t < T; ++t) {
        const float * y0 = &eout_[(size_t) t * K * E];
        const float w0 = w_[(size_t) t * K];
        for (int i = 0; i < E; ++i) acc[i] = y0[i] * w0;
        for (int s = 1; s < K; ++s) {
            if (sel_[(size_t) t * K + s] < 0) continue;  // dropped (top pick s=0 never is)
            const float * ys = &eout_[((size_t) t * K + s) * E];
            const float ws = w_[(size_t) t * K + s];
            for (int i = 0; i < E; ++i) {
                const float p = ys[i] * ws;
                acc[i] = acc[i] + p;
            }
        }
        float * x = &x_[(size_t) t * E];
        for (int i = 0; i < E; ++i) x[i] = acc[i] + x[i];
    }
}

const std::vector<float> & MoeModel::eval(const std::vector<int32_t> & tokens) {
    return eval_batch({ { 0, tokens, false, lossy } });
}

const std::vector<float> & MoeModel::eval_batch(const std::vector<Seq> & seqs) {
    const int E = hp_.n_embd, K = hp_.n_expert_used, S = (int) seqs.size();
    if (S == 0) throw std::runtime_error("eval: no sequences");

    // causal mask per sequence: query t (position n_past+t) sees cells 0..n_past+t; sliding-window layers
    // also hide cells n_swa or more positions back (llama.cpp's SWA_TYPE_STANDARD)
    const uint16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    runs_.resize(S);
    int T = 0, max_kv = 0, n_logit_rows = 0;
    for (int s = 0; s < S; ++s) {
        const Seq & q = seqs[s];
        const int t_s = (int) q.tokens.size(), np = n_past_.at(q.slot);
        if (t_s == 0) throw std::runtime_error("eval: no tokens");
        if (np + t_s > n_ctx_) throw std::runtime_error("context full");
        for (int o = 0; o < s; ++o) {
            if (seqs[o].slot == q.slot) throw std::runtime_error("eval: slot twice in one batch");
        }
        Run & r = runs_[s];
        r.slot = q.slot;
        r.row = T;
        r.T = t_s;
        r.like_decode = q.all_logits && t_s > 1;
        r.lossy = q.lossy;
        r.n_kv = std::min(n_ctx_, std::max(KV_PAD, (np + t_s + KV_PAD - 1) / KV_PAD * KV_PAD));
        r.mask.resize((size_t) t_s * r.n_kv);
        for (int t = 0; t < t_s; ++t) {
            for (int j = 0; j < r.n_kv; ++j) r.mask[(size_t) t * r.n_kv + j] = j <= np + t ? zero : ninf;
        }
        r.mask_swa = r.mask;
        for (int t = 0; t < t_s && hp_.n_swa > 0; ++t) {
            for (int j = 0; j < r.n_kv; ++j) {
                if (np + t - j >= hp_.n_swa) r.mask_swa[(size_t) t * r.n_kv + j] = ninf;
            }
        }
        T += t_s;
        max_kv = std::max(max_kv, r.n_kv);
        n_logit_rows += q.all_logits ? t_s : 1;
    }

    const size_t need = (8u << 20) + (size_t) T * (512u << 10) + (size_t) max_kv * T * 2 + (size_t) hp_.n_vocab * 4 * n_logit_rows +
                        ggml_graph_overhead_custom(16 * EXPERTS_PER_GRAPH + 16, false);
    if (arena_.size() < need) arena_.resize(need);
    x_.resize((size_t) T * E);
    hn_.resize((size_t) T * E);
    sel_.resize((size_t) T * K);
    w_.resize((size_t) T * K);
    eout_.resize((size_t) T * K * E);

    for (const Run & r : runs_) pack_.embed(seqs[&r - runs_.data()].tokens.data(), r.T, x_.data() + (size_t) r.row * E);

    for (int il = 0; il < hp_.n_layer; ++il) {
        auto t0 = clk::now();
        attention_and_route(il);
        timing.attn_s += since(t0);
        t0 = clk::now();
        experts(il, T);
        timing.experts_s += since(t0);
    }

    {  // output head per sequence (own matmul, as in compute_experts): its last token, or all with all_logits
        const auto t0 = clk::now();
        ggml_context * ctx = graph_ctx();
        ggml_cgraph * gf = ggml_new_graph(ctx);
        std::vector<ggml_tensor *> outs;
        size_t n_out = 0;
        for (int s = 0; s < S; ++s) {
            const Run & r = runs_[s];
            const int n = seqs[s].all_logits ? r.T : 1;
            ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, n);
            memcpy(x->data, &x_[(size_t) (r.row + r.T - n) * E], (size_t) n * E * sizeof(float));
            ggml_tensor * cur = ggml_mul(ctx, ggml_rms_norm(ctx, x, hp_.rms_eps), pack_.core("output_norm.weight"));
            cur = ggml_mul_mat(ctx, pack_.core("output.weight"), cur);
            ggml_build_forward_expand(gf, cur);
            outs.push_back(cur);
            n_out += n;
        }
        compute(gf);
        logits_.resize((size_t) hp_.n_vocab * n_out);
        float * dst = logits_.data();
        for (ggml_tensor * cur : outs) {
            copy_out(cur, dst);
            dst += (size_t) hp_.n_vocab * cur->ne[1];
        }
        ggml_free(ctx);
        timing.head_s += since(t0);
    }

    for (const Run & r : runs_) n_past_[r.slot] += r.T;
    return logits_;
}

}  // namespace zorlo
