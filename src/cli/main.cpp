// zorlo: run big MoE models with experts streamed from SSD.
//
//   zorlo run    <pack> -p "prompt" [-n 256] [options]
//   zorlo serve  <pack> [--host 127.0.0.1] [--port 8080] [options]   chat UI at /, OpenAI-compatible API at /v1
//   zorlo golden <pack> <golden.jsonl> [--prompts N] [options]
//   zorlo imatrix <pack> <text file>                                  <pack>/imatrix.bin for zorlo-pack --imatrix
//   zorlo start                                                      menu: pick/download/update a model, serve, open chat
//   zorlo pull   <owner/name>[@rev] [out_dir] [--lowbit]             download or update a pack from Hugging Face
//   zorlo pack   <model.gguf> <out_dir>                              GGUF -> Zorlo pack (same as zorlo-pack)
//   zorlo selftest
//
// options: -t threads (4)  -c context (4096)  --cache-mb N (0 = auto from free RAM, re-checked while
//          running)  --io-threads N (4)  --protect F (hot-set share, default by budget)  --prefetch N
//          (extra predicted experts per layer, default 0; -1 disables prefetch)  --kv-disk-mb N (prompt KV
//          kept under <pack>/kv so a later prompt with the same start skips its prefill; 4096, 0 = off)
//          -np N (requests served at once, each with its own context-sized KV cache; default 1)
//          --draft N (exact speculative decoding: check up to N tokens copied from earlier text per step; 6, 0 = off)
// lossy (off by default; measured in README): --mode exact|balanced|fast (presets; a request may send "mode")
//          --skip-below W  --prefer-cached D  --expert-mass M  --lowbit (override the preset)
//
// `golden` replays reference runs from zorlo-golden (llama.cpp) with the reference tokens fed back
// (teacher forcing) and reports, per prompt, how many greedy picks match and the largest logit gap.

#include "engine/expert_cache.h"
#include "engine/pack.h"
#include "engine/moe_model.h"

#include "chat.h"
#include "cli/agent.h"
#include "cli/pull.h"
#include "httplib.h"
#include "ui_html.h"
#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <string>
#include <vector>

using json = nlohmann::json;
using clk = std::chrono::steady_clock;
static double since(clk::time_point t0) { return std::chrono::duration<double>(clk::now() - t0).count(); }

static constexpr size_t KEEP_FREE = 512ull << 20;  // RAM auto mode leaves for the OS and other apps
static constexpr int REBALANCE_EVERY = 16;         // tokens between free-RAM checks in auto mode
static constexpr size_t PREFILL_CHUNK = 512;       // prompt tokens per step (llama.cpp's ubatch), so others keep decoding
static constexpr int MAX_DRAFT = 6;  // verify <= 7 tokens: from 8 columns ggml switches to its tiled matmul, which rounds differently

struct Args {
    std::string pack, golden, prompt = "Explain what a Mixture-of-Experts language model is.", host = "127.0.0.1";
    int n_gen = 256, threads = 4, n_ctx = 4096, max_prompts = 1 << 30, io_threads = 4, prefetch = 0, port = 8080, n_parallel = 1, draft = MAX_DRAFT;
    float protect = -1.0f, skip_below = 0, prefer_cached = 0, expert_mass = 0;
    std::string mode = "exact";
    bool lowbit = false;
    size_t cache_mb = 0, kv_disk_mb = 4096;
};

static Args parse(int argc, char ** argv, int first) {
    Args a;
    std::vector<std::string> pos;
    for (int i = first; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + s);
            return argv[++i];
        };
        if (s == "-p") a.prompt = next();
        else if (s == "-n") a.n_gen = std::stoi(next());
        else if (s == "-t") a.threads = std::stoi(next());
        else if (s == "-c") a.n_ctx = std::stoi(next());
        else if (s == "--cache-mb") a.cache_mb = std::stoull(next());
        else if (s == "--prompts") a.max_prompts = std::stoi(next());
        else if (s == "--io-threads") a.io_threads = std::stoi(next());
        else if (s == "--protect") a.protect = std::stof(next());
        else if (s == "--prefetch") a.prefetch = std::stoi(next());
        else if (s == "--host") a.host = next();
        else if (s == "--port") a.port = std::stoi(next());
        else if (s == "--skip-below") a.skip_below = std::stof(next());
        else if (s == "--prefer-cached") a.prefer_cached = std::stof(next());
        else if (s == "--lowbit") a.lowbit = true;
        else if (s == "--expert-mass") a.expert_mass = std::stof(next());
        else if (s == "--mode") a.mode = next();
        else if (s == "--kv-disk-mb") a.kv_disk_mb = std::stoull(next());
        else if (s == "-np") a.n_parallel = std::max(1, std::stoi(next()));
        else if (s == "--draft") a.draft = std::clamp(std::stoi(next()), 0, MAX_DRAFT);
        else pos.push_back(s);
    }
    if (!pos.empty()) a.pack = pos[0];
    if (pos.size() > 1) a.golden = pos[1];
    return a;
}

static int argmax(const std::vector<float> & v) {
    return (int) (std::max_element(v.begin(), v.end()) - v.begin());
}

// length of the longest prefix of s that does not end inside a UTF-8 sequence
static size_t utf8_complete(const std::string & s) {
    for (size_t i = s.size(), k = 1; i > 0 && k <= 4; --i, ++k) {
        const unsigned char c = (unsigned char) s[i - 1];
        if ((c & 0xC0) == 0x80) continue;  // continuation byte
        const size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        return need <= k ? s.size() : i - 1;
    }
    return s.size();
}

// lossy presets, measured in README (Phase 11); a request can pick one with "mode"
static zorlo::MoeModel::Lossy mode_lossy(const std::string & m) {
    if (m == "exact") return {};
    if (m == "balanced") return { 0, 0, 0.9f, false };
    if (m == "fast") return { 0, 0, 0.9f, true };
    throw std::runtime_error("mode must be exact, balanced or fast");
}

// the --mode preset, with any explicit lossy switch overriding it
static zorlo::MoeModel::Lossy lossy_of(const Args & a) {
    zorlo::MoeModel::Lossy l = mode_lossy(a.mode);
    if (a.skip_below > 0) l.skip_below = a.skip_below;
    if (a.prefer_cached > 0) l.prefer_cached = a.prefer_cached;
    if (a.expert_mass > 0) l.expert_mass = a.expert_mass;
    l.lowbit = l.lowbit || a.lowbit;
    return l;
}

// expert cache: fixed size, or auto-sized from free RAM and allowed to grow to every expert
static std::unique_ptr<zorlo::ExpertCache> make_cache(const zorlo::Pack & pack, const Args & a) {
    if (a.cache_mb) {
        auto c = std::make_unique<zorlo::ExpertCache>(pack, a.cache_mb << 20, a.io_threads, a.protect);
        c->set_lowbit(pack.has_lowbit() || lossy_of(a).lowbit);  // throws when asked for and missing
        return c;
    }
    const size_t avail = zorlo::available_ram_bytes();
    const size_t budget = avail > KEEP_FREE ? avail - KEEP_FREE : 0;
    auto c = std::make_unique<zorlo::ExpertCache>(pack, budget, a.io_threads, a.protect, SIZE_MAX);
    c->set_lowbit(pack.has_lowbit() || lossy_of(a).lowbit);
    if (budget < c->bytes()) {
        fprintf(stderr, "warning: only %.0f MB of RAM free; running with the minimum expert cache (%.0f MB)\n",
                avail / 1048576.0, c->bytes() / 1048576.0);
    }
    return c;
}

static void print_stats(const zorlo::ExpertCache & cache, const zorlo::MoeModel & m) {
    const auto s = cache.stats();
    const uint64_t n = s.hits + s.late + s.misses;
    fprintf(stderr, "  cache: %d slots (%.0f%% hot-protected), hit %.1f%% (+%.1f%% caught in flight), read %.2f GB\n",
            cache.n_slots(), 100.0 * cache.protect(), n ? 100.0 * s.hits / n : 0.0, n ? 100.0 * s.late / n : 0.0,
            s.bytes_read / 1e9);
    fprintf(stderr, "  prefetch: %llu issued, %llu used (%.0f%%), next-layer prediction recall %.0f%%\n",
            (unsigned long long) s.prefetch_issued, (unsigned long long) s.prefetch_used,
            s.prefetch_issued ? 100.0 * s.prefetch_used / s.prefetch_issued : 0.0,
            m.timing.pred_total ? 100.0 * m.timing.pred_hit / m.timing.pred_total : 0.0);
    if (s.lo_reads || s.upgrades) {
        fprintf(stderr, "  lowbit: %llu reads used the low-bit copy, %llu re-read in full after turning hot\n",
                (unsigned long long) s.lo_reads, (unsigned long long) s.upgrades);
    }
    if (m.timing.routed) {
        fprintf(stderr, "  lossy: %.1f%% of routed experts skipped, %.1f%% swapped for cached ones\n",
                100.0 * m.timing.skipped / m.timing.routed, 100.0 * m.timing.swapped / m.timing.routed);
    }
    fprintf(stderr, "  time: attention+routing %.1f s, experts %.1f s (of which waiting on reads %.1f s), head %.1f s\n",
            m.timing.attn_s, m.timing.experts_s, s.wait_s, m.timing.head_s);
}

struct Sampling {
    float temperature = 0.0f, top_p = 1.0f;
    int top_k = 0;
    uint32_t seed = LLAMA_DEFAULT_SEED;
};

struct GenStats {
    int n_prompt = 0, n_reused = 0, n_disk = 0, n_gen = 0, n_drafted = 0, n_accepted = 0;
    double prefill_s = 0, gen_s = 0;
    std::string finish = "length";
};

// model + tokenizer + the tokens whose KV is cached, so a follow-up prompt only evaluates what is new
class Session {
public:
    explicit Session(const Args & a) : args_(a), pack_(a.pack) {
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.vocab_only = true;
        vm_ = llama_model_load_from_file(pack_.core_path().c_str(), mp);
        if (!vm_) throw std::runtime_error("cannot load vocabulary from core.gguf");
        vocab_ = llama_model_get_vocab(vm_);
        n_vocab_ = llama_vocab_n_tokens(vocab_);
        tmpls_ = common_chat_templates_init(vm_, "");
        cache_ = make_cache(pack_, a);
        model_ = std::make_unique<zorlo::MoeModel>(pack_, *cache_, a.threads, a.n_ctx, a.prefetch, a.n_parallel);
        model_->lossy = lossy_of(a);
        cached_.resize(model_->n_seq());
        load_profile();
        if (a.kv_disk_mb) scan_kv();
        engine_ = std::thread([this] { engine(); });
        fprintf(stderr, "core %.2f GB resident, expert cache %d slots (%.0f MB, %s), context %d\n",
                pack_.core_bytes() / 1e9, cache_->n_slots(), cache_->bytes() / 1048576.0,
                a.cache_mb ? "fixed" : "auto", model_->n_ctx());
    }
    ~Session() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        engine_.join();
        model_.reset();
        cache_.reset();
        llama_model_free(vm_);
        llama_backend_free();
    }

    // the model's Jinja chat template (tools, thinking) and output parser, as llama-server uses them
    common_chat_session chat(const common_chat_templates_inputs & in) const { return common_chat_session(tmpls_.get(), vocab_, in); }
    const common_chat_templates * templates() const { return tmpls_.get(); }

    std::vector<int32_t> tokenize(const std::string & text) const {
        const int nt = -llama_tokenize(vocab_, text.c_str(), (int32_t) text.size(), nullptr, 0, true, true);
        std::vector<int32_t> toks(nt);
        llama_tokenize(vocab_, text.c_str(), (int32_t) text.size(), toks.data(), nt, true, true);
        return toks;
    }

    // emit(chunk) gets UTF-8-complete text with its token ids and returns false to stop (client gone); an empty
    // chunk is a keep-alive while the prompt is read or the request waits for a slot.
    // generation also stops before any of `stops` appears, which is never emitted. Thread-safe: requests
    // queue for a free slot, then share one engine thread that steps all running ones together; emit runs there
    GenStats generate(const std::vector<int32_t> & prompt, int max_tokens, const Sampling & sp,
                      const std::vector<std::string> & stops, const std::function<bool(const common_chat_input &)> & emit,
                      const zorlo::MoeModel::Lossy * lossy = nullptr) {
        if (prompt.empty()) throw std::runtime_error("empty prompt");
        if ((int) prompt.size() >= model_->n_ctx()) throw std::runtime_error("prompt longer than the context");
        Job j{ prompt, max_tokens, sp, stops, emit };
        j.lossy = lossy ? *lossy : model_->lossy;  // default: the server's --mode
        std::unique_lock<std::mutex> lk(mu_);
        queue_.push_back(&j);
        cv_.notify_all();
        cv_.wait(lk, [&] { return j.done; });
        if (j.err) std::rethrow_exception(j.err);
        return j.st;
    }

  private:
    struct Job {
        std::vector<int32_t> prompt;
        int max_tokens;
        Sampling sp;
        std::vector<std::string> stops;
        std::function<bool(const common_chat_input &)> emit;
        GenStats st;
        std::exception_ptr err;
        bool done = false;
        // engine thread only
        int slot = -1;
        size_t fed = 0;  // prompt tokens in the slot's KV
        int next = -1;   // sampled token, evaluated in the next step
        llama_sampler * smpl = nullptr;
        size_t hold = 0;  // trailing bytes kept back because they may start a stop string
        common_chat_input pending;
        clk::time_point t0;
        std::vector<int32_t> draft;  // tokens checked after `next` in this step
        float p_accept = 0.7f;       // running share of draft tokens accepted
        zorlo::MoeModel::Lossy lossy;
    };

    void engine() {
        std::vector<Job *> active;
        int steps = 0;
        std::unique_lock<std::mutex> lk(mu_);
        for (;;) {
            cv_.wait(lk, [&] { return stop_ || !queue_.empty() || !active.empty(); });
            if (stop_) return;  // shortcut: requests still running at shutdown never return
            while (!queue_.empty() && (int) active.size() < model_->n_seq()) {
                active.push_back(queue_.front());
                queue_.pop_front();
                lk.unlock();
                rebalance();
                start(*active.back(), active);
                lk.lock();
            }
            lk.unlock();
            if (++steps % REBALANCE_EVERY == 0) rebalance();
            step(active);
            lk.lock();
            for (size_t i = 0; i < active.size();) {
                if (active[i]->done) active.erase(active.begin() + i);
                else ++i;
            }
            // keep-alive for requests still waiting for a slot (Claude Code drops a stream silent for 5 min);
            // only this thread touches a job before it is done, so emit can run without the lock
            const std::vector<Job *> waiting(queue_.begin(), queue_.end());
            lk.unlock();
            std::vector<Job *> gone;
            for (Job * w : waiting) {
                if (since(w->t0) < 15) continue;
                w->t0 = clk::now();
                if (!w->emit({})) gone.push_back(w);
            }
            lk.lock();
            for (Job * w : gone) {
                queue_.erase(std::find(queue_.begin(), queue_.end(), w));
                w->st.finish = "stop";
                w->done = true;
            }
            cv_.notify_all();
        }
    }

    // put a new job in the free slot that already holds the longest start of its prompt
    void start(Job & j, const std::vector<Job *> & active) {
        size_t best = 0;
        for (int s = 0; s < model_->n_seq(); ++s) {
            if (std::any_of(active.begin(), active.end(), [&](const Job * o) { return o->slot == s; })) continue;
            const size_t n = lcp(cached_[s], j.prompt);
            if (j.slot < 0 || n > best) j.slot = s, best = n;
        }
        // reuse the cached prefix (in RAM, or a longer one saved on disk); at least one token must be
        // evaluated to get logits
        j.st.n_prompt = (int) j.prompt.size();
        size_t common = std::min(best, j.prompt.size() - 1);
        common = load_saved_kv(j.prompt, common, j.st, j.slot);
        model_->truncate((int) common, j.slot);
        cached_[j.slot].resize(common);
        j.st.n_reused = (int) common;
        j.fed = common;
        if (j.sp.temperature > 0) {
            j.smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
            if (j.sp.top_k > 0) llama_sampler_chain_add(j.smpl, llama_sampler_init_top_k(j.sp.top_k));
            if (j.sp.top_p < 1.0f) llama_sampler_chain_add(j.smpl, llama_sampler_init_top_p(j.sp.top_p, 1));
            llama_sampler_chain_add(j.smpl, llama_sampler_init_temp(j.sp.temperature));
            llama_sampler_chain_add(j.smpl, llama_sampler_init_dist(j.sp.seed));
        }
        for (const auto & s : j.stops) j.hold = std::max(j.hold, s.size() > 0 ? s.size() - 1 : 0);
        j.t0 = clk::now();
    }

    // one batched eval: the next prompt chunk of each prefilling job; for each decoding one its last sampled
    // token, plus a draft of what may follow (checked in the same eval, see verify)
    void step(const std::vector<Job *> & active) {
        std::vector<zorlo::MoeModel::Seq> seqs;
        for (Job * j : active) {
            if (j->fed < j->prompt.size()) {
                const size_t n = std::min(PREFILL_CHUNK, j->prompt.size() - j->fed);
                seqs.push_back({ j->slot, { j->prompt.begin() + j->fed, j->prompt.begin() + j->fed + n }, false, j->lossy });
            } else {
                j->draft = make_draft(*j);
                seqs.push_back({ j->slot, { j->next }, !j->draft.empty(), j->lossy });
                seqs.back().tokens.insert(seqs.back().tokens.end(), j->draft.begin(), j->draft.end());
            }
        }
        try {
            const std::vector<float> & logits = model_->eval_batch(seqs);
            const float * lg = logits.data();
            for (size_t i = 0; i < active.size(); ++i) {
                Job & j = *active[i];
                const size_t rows = seqs[i].all_logits ? seqs[i].tokens.size() : 1;
                auto & cached = cached_[j.slot];
                if (j.fed < j.prompt.size()) {
                    cached.insert(cached.end(), seqs[i].tokens.begin(), seqs[i].tokens.end());
                    j.fed += seqs[i].tokens.size();
                    if (j.fed == j.prompt.size()) {
                        j.st.prefill_s = since(j.t0);
                        j.t0 = clk::now();
                        if (!next_token(j, lg)) finish(j);
                    } else if (!j.emit({})) {  // keep-alive between prompt chunks; false: the client left
                        j.st.finish = "stop";
                        finish(j);
                    }
                } else {
                    verify(j, lg);
                }
                lg += rows * n_vocab_;
            }
        } catch (...) {
            for (Job * j : active) {
                if (j->done) continue;
                j->err = std::current_exception();
                finish(*j);
            }
        }
    }

    // prompt lookup: find the last 3 (else 2) tokens earlier in the prompt + output and draft what followed
    // them. Agents repeat a lot (tool-call JSON, code they just read), so this is often right, and checking
    // the draft costs one eval. Its length weighs the cost of evaluating w tokens at once against the
    // acceptance rate seen so far; the cost is measured (step time vs one token, Qwen3-30B-A3B, 1.5 GB cache)
    // shortcut: costs fixed from the dev PC (disk-bound); time steps online if other machines differ a lot
    std::vector<int32_t> make_draft(Job & j) const {
        static constexpr float cost[MAX_DRAFT + 2] = { 0, 1.0f, 1.92f, 2.46f, 2.99f, 3.74f, 4.12f, 5.81f };
        const int room = std::min({ args_.draft, j.max_tokens - j.st.n_gen - 1, model_->n_ctx() - model_->n_past(j.slot) - 1 });
        if (room <= 0) return {};
        static const bool always = getenv("ZORLO_DRAFT_ALWAYS") != nullptr;  // test hook: check a draft every step
        if (always) {
            const auto & c = cached_[j.slot];
            return { c.end() - std::min<size_t>(room, c.size()), c.end() };
        }
        int best = 0;
        float best_gain = 1.0f;
        for (int d = 1; d <= room; ++d) {
            const float gain = (1.0f - std::pow(j.p_accept, (float) d + 1)) / (1.0f - j.p_accept) / cost[d + 1];
            if (gain > best_gain) best = d, best_gain = gain;
        }
        if (best == 0) {
            j.p_accept = std::min(0.95f, j.p_accept + 0.02f);  // drift back up so drafting is tried again later
            return {};
        }
        // the earlier position whose preceding tokens match the end of the text the longest (most recent on ties);
        // short matches like "\n    " continue too many ways, so at least 3
        std::vector<int32_t> h = cached_[j.slot];
        h.push_back(j.next);
        const int n = (int) h.size();
        int at = -1, len = 2;
        for (int i = n - 1; i > 0; --i) {  // continuation would start at h[i]
            int l = 0;
            while (l < 16 && i - 1 - l >= 0 && h[i - 1 - l] == h[n - 1 - l]) ++l;
            if (l > len) at = i, len = l;
        }
        if (at < 0) return {};
        return { h.begin() + at, h.begin() + std::min(n, at + best) };
    }

    // rows of lg: after j.next, then after each draft token. Sampling every row with the job's own sampler
    // and keeping drafts only while they equal the sampled token gives exactly the tokens (and sampler
    // state) of decoding one at a time; the KV of rejected drafts is dropped
    void verify(Job & j, const float * lg) {
        auto & cached = cached_[j.slot];
        cached.push_back(j.next);
        size_t m = 0;
        bool more = true;
        for (size_t r = 0;; ++r) {
            if (!(more = next_token(j, lg + r * n_vocab_))) break;
            if (r < j.draft.size() && j.next == j.draft[r]) {
                cached.push_back(j.draft[r]);
                ++m;
                continue;
            }
            break;
        }
        model_->truncate((int) cached.size(), j.slot);
        if (!j.draft.empty()) {
            j.st.n_drafted += (int) j.draft.size();
            j.st.n_accepted += (int) m;
            for (size_t r = 0; r < std::min(m + 1, j.draft.size()); ++r) j.p_accept = 0.8f * j.p_accept + 0.2f * (r < m);
        }
        if (!more) finish(j);
    }

    // sample, emit; false when the job is complete
    bool next_token(Job & j, const float * logits) {
        GenStats & st = j.st;
        int tok;
        if (j.smpl) {
            std::vector<llama_token_data> cand(n_vocab_);
            for (int i = 0; i < n_vocab_; ++i) cand[i] = { i, logits[i], 0.0f };
            llama_token_data_array arr = { cand.data(), cand.size(), -1, false };
            llama_sampler_apply(j.smpl, &arr);
            tok = arr.data[arr.selected].id;
        } else {
            tok = (int) (std::max_element(logits, logits + n_vocab_) - logits);
        }
        if (llama_vocab_is_eog(vocab_, tok)) {
            st.finish = "stop";
            return false;
        }
        ++st.n_gen;
        char piece[256];
        const int len = llama_token_to_piece(vocab_, tok, piece, sizeof(piece), 0, true);
        if (len > 0) j.pending.append(std::string(piece, len), tok);
        size_t cut = std::string::npos;
        for (const auto & s : j.stops) {
            if (!s.empty()) cut = std::min(cut, j.pending.text.find(s));
        }
        if (cut != std::string::npos) {
            j.pending.truncate(cut);
            st.finish = "stop";
            return false;
        }
        const size_t ok = utf8_complete(j.pending.text.substr(0, j.pending.size() > j.hold ? j.pending.size() - j.hold : 0));
        if (ok && !j.emit(j.pending.substr(0, ok))) {
            st.finish = "stop";
            return false;
        }
        j.pending = j.pending.substr(ok);
        if (st.n_gen >= j.max_tokens || model_->n_past(j.slot) >= model_->n_ctx()) return false;
        j.next = tok;
        return true;
    }

    void finish(Job & j) {
        if (!j.err && !j.pending.empty()) j.emit(j.pending);
        j.st.gen_s = since(j.t0);
        if (j.smpl) llama_sampler_free(j.smpl);
        if (!j.err) save_kv(j.prompt, j.slot);
        save_profile();
        std::lock_guard<std::mutex> lk(mu_);
        j.done = true;
    }

  public:
    // --- saved state under the pack: prompt KV (<pack>/kv/*.zkv) and expert use counts (<pack>/hot.bin)

    static size_t lcp(const std::vector<int32_t> & a, const std::vector<int32_t> & b) {
        size_t n = 0;
        while (n < a.size() && n < b.size() && a[n] == b[n]) ++n;
        return n;
    }

    std::filesystem::path pack_dir() const { return std::filesystem::u8path(args_.pack); }
    std::filesystem::path kv_dir() const { return pack_dir() / "kv"; }

    // .zkv: magic, token count, KV bytes per token (rejects files from another model), tokens, save_kv() cells
    static constexpr uint32_t KV_MAGIC = 0x31564B5A;  // "ZKV1"

    bool read_kv_header(std::FILE * f, std::vector<int32_t> & toks) const {
        uint32_t magic = 0, n = 0;
        uint64_t per_token = 0;
        if (fread(&magic, 4, 1, f) != 1 || fread(&n, 4, 1, f) != 1 || fread(&per_token, 8, 1, f) != 1) return false;
        if (magic != KV_MAGIC || per_token != model_->kv_bytes(1) || (int) n > model_->n_ctx()) return false;
        toks.resize(n);
        return fread(toks.data(), sizeof(int32_t), n, f) == n;
    }

    void scan_kv() {
        std::error_code ec;
        std::filesystem::create_directories(kv_dir(), ec);
        std::vector<std::pair<std::filesystem::file_time_type, KvFile>> found;
        for (const auto & e : std::filesystem::directory_iterator(kv_dir(), ec)) {
            if (e.path().extension() != ".zkv") continue;
            KvFile k{ e.path(), {} };
            if (std::FILE * f = _wfopen(e.path().c_str(), L"rb")) {
                if (read_kv_header(f, k.tokens)) found.emplace_back(e.last_write_time(), std::move(k));
                fclose(f);
            }
        }
        std::sort(found.begin(), found.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
        for (auto & [t, k] : found) kv_files_.push_back(std::move(k));
        if (!kv_files_.empty()) fprintf(stderr, "%zu saved prompt KVs in %s\n", kv_files_.size(), kv_dir().u8string().c_str());
    }

    // returns the new reusable prefix length: `have`, or more if a saved prompt starts the same way
    size_t load_saved_kv(const std::vector<int32_t> & prompt, size_t have, GenStats & st, int slot) {
        size_t best = have;
        int src = -1;
        for (int i = 0; i < (int) kv_files_.size(); ++i) {
            const size_t n = std::min(lcp(kv_files_[i].tokens, prompt), prompt.size() - 1);
            if (n > best) best = n, src = i;
        }
        if (src < 0) return have;
        KvFile k = kv_files_[src];
        kv_files_.erase(kv_files_.begin() + src);
        const auto t0 = clk::now();
        std::FILE * f = _wfopen(k.path.c_str(), L"rb");
        try {
            std::vector<int32_t> toks;
            if (!f || !read_kv_header(f, toks) || toks != k.tokens) throw std::runtime_error("unreadable");
            model_->load_kv(f, (int) best, (int) toks.size(), slot);
        } catch (const std::exception & e) {  // the RAM KV may be half overwritten: start over
            if (f) fclose(f);
            fprintf(stderr, "[kv] dropping %s: %s\n", k.path.u8string().c_str(), e.what());
            std::error_code ec;
            std::filesystem::remove(k.path, ec);
            model_->reset(slot);
            cached_[slot].clear();
            return 0;
        }
        fclose(f);
        kv_files_.push_back(std::move(k));  // most recently used last
        cached_[slot].assign(prompt.begin(), prompt.begin() + best);
        st.n_disk = (int) best;
        fprintf(stderr, "[kv] read %zu prompt tokens from disk in %.2f s\n", best, since(t0));
        return best;
    }

    // keep this prompt's KV (cells 0..n-1; generation only wrote after them) unless a saved one covers it
    void save_kv(const std::vector<int32_t> & prompt, int slot) {
        if (!args_.kv_disk_mb || model_->n_past(slot) < (int) prompt.size() || lcp(cached_[slot], prompt) < prompt.size()) return;
        for (const auto & k : kv_files_) {
            if (lcp(k.tokens, prompt) == prompt.size()) return;
        }
        uint64_t h = 1469598103934665603ull;  // FNV-1a of the tokens names the file
        for (int32_t t : prompt) h = (h ^ (uint32_t) t) * 1099511628211ull;
        char name[32];
        snprintf(name, sizeof(name), "%016llx.zkv", (unsigned long long) h);
        const auto path = kv_dir() / name, tmp = kv_dir() / (std::string(name) + ".tmp");
        std::error_code ec;
        try {
            std::filesystem::create_directories(kv_dir(), ec);
            std::FILE * f = _wfopen(tmp.c_str(), L"wb");
            if (!f) throw std::runtime_error("cannot create file");
            const uint32_t hdr[2] = { KV_MAGIC, (uint32_t) prompt.size() };
            const uint64_t per_token = model_->kv_bytes(1);
            bool ok = fwrite(hdr, 4, 2, f) == 2 && fwrite(&per_token, 8, 1, f) == 1 &&
                      fwrite(prompt.data(), sizeof(int32_t), prompt.size(), f) == prompt.size();
            try {
                if (ok) model_->save_kv(f, (int) prompt.size(), slot);
            } catch (...) {
                ok = false;
            }
            ok = fclose(f) == 0 && ok;
            if (!ok) throw std::runtime_error("write failed (disk full?)");
            std::filesystem::rename(tmp, path);  // a crash mid-write never leaves a broken .zkv
        } catch (const std::exception & e) {
            std::filesystem::remove(tmp, ec);
            fprintf(stderr, "[kv] not saved: %s\n", e.what());
            return;
        }
        // drop saved prompts this one extends (a conversation's earlier turns), then the oldest over budget
        for (size_t i = 0; i < kv_files_.size();) {
            if (lcp(kv_files_[i].tokens, prompt) == kv_files_[i].tokens.size() && kv_files_[i].path != path) {
                std::filesystem::remove(kv_files_[i].path, ec);
                kv_files_.erase(kv_files_.begin() + i);
            } else {
                ++i;
            }
        }
        kv_files_.push_back({ path, prompt });
        uint64_t total = 0;
        for (const auto & k : kv_files_) total += std::filesystem::file_size(k.path, ec);
        while (kv_files_.size() > 1 && total > (uint64_t) args_.kv_disk_mb << 20) {
            total -= std::min(total, (uint64_t) std::filesystem::file_size(kv_files_.front().path, ec));
            std::filesystem::remove(kv_files_.front().path, ec);
            kv_files_.erase(kv_files_.begin());
        }
    }

    void load_profile() {
        std::ifstream in(pack_dir() / "hot.bin", std::ios::binary);
        if (!in) return;
        std::vector<float> c((size_t) pack_.hp().n_layer * pack_.hp().n_expert);
        if (in.read((char *) c.data(), c.size() * sizeof(float)) && in.peek() == EOF) cache_->set_profile(c);
    }

    void save_profile() const {
        const std::vector<float> c = cache_->profile();
        const auto path = pack_dir() / "hot.bin", tmp = pack_dir() / "hot.bin.tmp";
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            out.write((const char *) c.data(), c.size() * sizeof(float));
            if (!out) return;
        }
        std::error_code ec;
        std::filesystem::rename(tmp, path, ec);
    }

    // auto mode: follow free RAM, so other apps opening or closing grows or shrinks the expert cache
    void rebalance() {
        if (args_.cache_mb) return;
        const size_t want = zorlo::available_ram_bytes() + cache_->resident_bytes();
        const size_t target = want > KEEP_FREE ? want - KEEP_FREE : 0;
        const long long diff = (long long) (target / cache_->slot_bytes()) - cache_->n_slots();
        if (std::llabs(diff) < 16) return;  // ignore small wobble
        const int before = cache_->n_slots();
        cache_->resize(target);
        if (cache_->n_slots() != before) {
            fprintf(stderr, "[budget] free RAM %.0f MB: expert cache %d -> %d slots (%.0f MB)\n",
                    zorlo::available_ram_bytes() / 1048576.0, before, cache_->n_slots(), cache_->bytes() / 1048576.0);
        }
    }

    const zorlo::ExpertCache & cache() const { return *cache_; }
    const zorlo::MoeModel & model() const { return *model_; }
    int n_ctx() const { return model_->n_ctx(); }
    std::string model_id() const { return std::filesystem::u8path(args_.pack).filename().u8string(); }

private:
    Args args_;
    zorlo::Pack pack_;
    llama_model * vm_ = nullptr;
    const llama_vocab * vocab_ = nullptr;
    int n_vocab_ = 0;
    std::unique_ptr<zorlo::ExpertCache> cache_;
    std::unique_ptr<zorlo::MoeModel> model_;
    std::vector<std::vector<int32_t>> cached_;  // per slot: the tokens whose KV it holds
    common_chat_templates_ptr tmpls_;
    struct KvFile {
        std::filesystem::path path;
        std::vector<int32_t> tokens;
    };
    std::vector<KvFile> kv_files_;  // saved prompt KVs, least recently used first (engine thread)
    std::mutex mu_;                 // guards queue_, stop_ and Job::done
    std::condition_variable cv_;
    std::deque<Job *> queue_;       // waiting for a free slot
    bool stop_ = false;
    std::thread engine_;
};

// importance matrix for zorlo-pack --imatrix: squared expert inputs over a text file, in 512-token windows
// (each from an empty context, like llama-imatrix), written to <pack>/imatrix.bin
static int cmd_imatrix(const Args & a) {
    zorlo::Pack pack(a.pack);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * vm = llama_model_load_from_file(pack.core_path().c_str(), mp);
    if (!vm) throw std::runtime_error("cannot load vocabulary from core.gguf");
    std::ifstream in(std::filesystem::u8path(a.golden), std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + a.golden);
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<int32_t> toks(text.size() + 16);
    const int nt = llama_tokenize(llama_model_get_vocab(vm), text.c_str(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), false, false);
    if (nt < 0) throw std::runtime_error("tokenize failed");
    toks.resize(nt);
    llama_model_free(vm);

    auto cache = make_cache(pack, a);
    zorlo::MoeModel model(pack, *cache, a.threads, 512, a.prefetch);
    std::vector<float> imat(model.imatrix_size());
    model.imatrix = &imat;
    const auto t0 = clk::now();
    for (size_t i = 0; i < toks.size(); i += 512) {
        model.reset();
        model.eval({ toks.begin() + i, toks.begin() + std::min(toks.size(), i + 512) });
        fprintf(stderr, "\rimatrix: %zu / %zu tokens, %.0f s", std::min(toks.size(), i + 512), toks.size(), since(t0));
    }
    const zorlo::HParams & hp = pack.hp();
    const int32_t hdr[4] = { hp.n_layer, hp.n_expert, hp.n_embd, hp.n_ff_exp };
    const auto path = std::filesystem::u8path(a.pack) / "imatrix.bin";
    std::ofstream out(path, std::ios::binary);
    out.write((const char *) hdr, sizeof(hdr));
    out.write((const char *) imat.data(), (std::streamsize) (imat.size() * sizeof(float)));
    if (!out) throw std::runtime_error("cannot write " + path.u8string());
    fprintf(stderr, "\nwrote %s\n", path.u8string().c_str());
    return 0;
}

static int cmd_golden(const Args & a) {
    zorlo::Pack pack(a.pack);
    auto cache = make_cache(pack, a);
    zorlo::MoeModel model(pack, *cache, a.threads, a.n_ctx, a.prefetch);
    model.lossy = lossy_of(a);
    fprintf(stderr, "core %.2f GB resident, expert cache %d slots (%.0f MB)\n", pack.core_bytes() / 1e9,
            cache->n_slots(), cache->bytes() / 1048576.0);

    std::ifstream f(a.golden);
    if (!f) throw std::runtime_error("cannot open " + a.golden);
    std::string line;
    int n_prompts = 0, all_steps = 0, all_match = 0, exact_prompts = 0;
    double worst = 0, decode_s = 0, nll = 0;
    int decode_tokens = 0;
    while (std::getline(f, line) && n_prompts < a.max_prompts) {
        if (line.empty()) continue;
        const auto j = json::parse(line);
        const auto ptoks = j.at("prompt_tokens").get<std::vector<int32_t>>();
        const auto gen = j.at("gen").get<std::vector<int32_t>>();
        const auto & top = j.at("top");

        model.reset();
        int match = 0, first_miss = -1;
        double max_gap = 0;
        std::vector<float> logits = model.eval(ptoks);
        const auto t0 = clk::now();
        for (size_t s = 0; s < gen.size(); ++s) {
            const int ours = argmax(logits);
            if (ours == gen[s]) ++match;
            else if (first_miss < 0) first_miss = (int) s;
            // -log p(reference token): the exact engine gives the baseline, approximations raise it
            const float mx = logits[ours];
            double z = 0;
            for (float l : logits) z += std::exp((double) l - mx);
            nll += mx + std::log(z) - logits[gen[s]];
            for (const auto & pr : top[s]) {
                max_gap = std::max(max_gap, (double) std::fabs(logits[pr[0].get<int>()] - pr[1].get<float>()));
            }
            if (s + 1 < gen.size()) logits = model.eval({ gen[s] });  // feed the reference token
        }
        decode_s += since(t0);
        decode_tokens += (int) gen.size() - 1;
        ++n_prompts;
        all_steps += (int) gen.size();
        all_match += match;
        exact_prompts += match == (int) gen.size();
        worst = std::max(worst, max_gap);
        printf("prompt %2d: %3zu prompt tokens, greedy match %d/%zu%s, max |logit diff| on reference top-5 %.3g\n",
               j.at("prompt").get<int>(), ptoks.size(), match, gen.size(),
               first_miss >= 0 ? (" (first miss at step " + std::to_string(first_miss) + ")").c_str() : "", max_gap);
        fflush(stdout);
    }
    printf("\n%d/%d prompts identical, %d/%d steps match, worst logit diff %.3g, decode %.2f tok/s\n",
           exact_prompts, n_prompts, all_match, all_steps, worst, decode_tokens / std::max(decode_s, 1e-9));
    printf("mean NLL of reference tokens %.4f (perplexity %.4f)\n", nll / std::max(all_steps, 1),
           std::exp(nll / std::max(all_steps, 1)));
    print_stats(*cache, model);
    return exact_prompts == n_prompts ? 0 : 2;
}

static int cmd_run(const Args & a) {
    Session s(a);
    common_chat_templates_inputs in;
    in.messages.push_back({});
    in.messages[0].role = "user";
    in.messages[0].content = a.prompt;
    const auto toks = s.tokenize(s.chat(in).prompt().str());
    const GenStats st = s.generate(toks, a.n_gen, Sampling(), {}, [](const common_chat_input & c) {
        const std::string & t = c.text;
        fwrite(t.data(), 1, t.size(), stdout);
        fflush(stdout);
        return true;
    });
    printf("\n");
    fprintf(stderr, "\nprefill %d tokens in %.1f s (%.1f tok/s), generated %d tokens in %.1f s (%.2f tok/s), %d of %d drafts accepted\n",
            st.n_prompt, st.prefill_s, st.n_prompt / std::max(st.prefill_s, 1e-9), st.n_gen, st.gen_s,
            st.n_gen / std::max(st.gen_s, 1e-9), st.n_accepted, st.n_drafted);
    print_stats(s.cache(), s.model());
    return 0;
}

static json error_json(const std::string & msg) {
    return { { "error", { { "message", msg }, { "type", "invalid_request_error" } } } };
}

static json message_json(const common_chat_msg & m) {
    json j = json::parse(m.to_json_oaicompat().dump());
    j["role"] = "assistant";
    return j;
}

// one streamed change of the parsed message as an OpenAI delta (as llama-server's server-chat.cpp)
static json delta_json(const common_chat_msg_diff & d) {
    json delta = json::object();
    if (!d.reasoning_content_delta.empty()) delta["reasoning_content"] = d.reasoning_content_delta;
    if (!d.content_delta.empty()) delta["content"] = d.content_delta;
    if (d.tool_call_index != std::string::npos) {
        json call = { { "index", d.tool_call_index } };
        if (!d.tool_call_delta.id.empty()) {
            call["id"] = d.tool_call_delta.id;
            call["type"] = "function";
        }
        json fn = json::object();
        if (!d.tool_call_delta.name.empty()) fn["name"] = d.tool_call_delta.name;
        if (!d.tool_call_delta.arguments.empty()) fn["arguments"] = d.tool_call_delta.arguments;
        if (!fn.empty()) call["function"] = fn;
        delta["tool_calls"] = json::array({ call });
    }
    return delta;
}

static json anthropic_error(const std::string & msg) {
    return { { "type", "error" }, { "error", { { "type", "invalid_request_error" }, { "message", msg } } } };
}

// Claude Code puts "x-anthropic-billing-header: ...cch=XXXXX;" first in its system prompt with a new hash per
// request; blank the hash (as llama-server does) so the prompt start stays the same and its KV is reused
static void normalize_billing_header(std::string & s) {
    if (s.rfind("x-anthropic-billing-header:", 0) != 0) return;
    const size_t i = s.find("cch=");
    if (i != std::string::npos && i + 9 < s.size() && s[i + 9] == ';') s.replace(i + 4, 5, "fffff");
}

// Anthropic Messages request -> the OpenAI chat body the handler takes, mapped like llama-server's
// server_chat_convert_anthropic_to_oai (text only). ordered_json keeps the key order the template prints
static std::string anthropic_to_oai(const std::string & text) {
    using oj = nlohmann::ordered_json;
    const oj b = oj::parse(text);
    auto text_of = [](const oj & c) {  // a string, or the text of its text blocks
        if (c.is_string()) return c.get<std::string>();
        std::string t;
        if (c.is_array()) {
            for (const auto & blk : c) {
                if (blk.value("type", "") == "text") t += blk.value("text", "");
            }
        }
        return t;
    };
    oj msgs = oj::array();
    if (b.contains("system")) {
        std::string sys;
        if (b["system"].is_string()) {
            sys = b["system"].get<std::string>();
            normalize_billing_header(sys);
        } else {
            for (const auto & blk : b["system"]) {
                if (blk.value("type", "") != "text") continue;
                std::string t = blk.value("text", "");
                normalize_billing_header(t);
                sys += t;
            }
        }
        msgs.push_back({ { "role", "system" }, { "content", sys } });
    }
    for (const auto & m : b.at("messages")) {
        const std::string role = m.at("role");
        const oj & c = m.at("content");
        if (c.is_string()) {
            msgs.push_back({ { "role", role }, { "content", c } });
            continue;
        }
        std::string txt, thinking;
        oj calls = oj::array(), results = oj::array();
        for (const auto & blk : c) {
            const std::string ty = blk.value("type", "");
            if (ty == "text") {
                txt += blk.value("text", "");
            } else if (ty == "thinking") {
                thinking += blk.value("thinking", "");
            } else if (ty == "tool_use") {
                calls.push_back({ { "id", blk.value("id", "") }, { "type", "function" },
                                  { "function", { { "name", blk.value("name", "") }, { "arguments", blk.value("input", oj::object()).dump() } } } });
            } else if (ty == "tool_result") {
                results.push_back({ { "role", "tool" }, { "tool_call_id", blk.value("tool_use_id", "") },
                                    { "content", blk.contains("content") ? text_of(blk["content"]) : "" } });
            }
        }
        if (!txt.empty() || !calls.empty() || !thinking.empty()) {
            oj nm = { { "role", role }, { "content", txt } };
            if (!calls.empty()) nm["tool_calls"] = calls;
            if (!thinking.empty()) nm["reasoning_content"] = thinking;
            msgs.push_back(nm);
        }
        for (const auto & r : results) msgs.push_back(r);
    }
    oj out = { { "messages", msgs } };
    if (b.contains("tools")) {
        oj tools = oj::array();
        for (const auto & t : b["tools"]) {
            if (t.contains("type") && t["type"] != "custom" && !t.contains("input_schema")) continue;  // server tools (web search, ...)
            tools.push_back({ { "type", "function" }, { "function", { { "name", t.value("name", "") }, { "description", t.value("description", "") },
                                                                       { "parameters", t.value("input_schema", oj::object()) } } } });
        }
        if (!tools.empty()) out["tools"] = tools;
    }
    if (b.contains("tool_choice") && b["tool_choice"].is_object()) {
        const std::string ty = b["tool_choice"].value("type", "auto");
        out["tool_choice"] = ty == "any" || ty == "tool" ? "required" : ty == "none" ? "none" : "auto";
    }
    for (const char * k : { "model", "max_tokens", "temperature", "top_p", "top_k", "stream", "mode" }) {
        if (b.contains(k)) out[k] = b[k];
    }
    if (b.contains("stop_sequences")) out["stop"] = b["stop_sequences"];
    return out.dump();
}

// Anthropic content blocks of a parsed reply: thinking, text, then tool_use (as llama-server)
static json anthropic_content(const common_chat_msg & m) {
    json blocks = json::array();
    if (!m.reasoning_content.empty()) blocks.push_back({ { "type", "thinking" }, { "thinking", m.reasoning_content }, { "signature", "" } });
    if (!m.content.empty()) blocks.push_back({ { "type", "text" }, { "text", m.content } });
    for (const auto & tc : m.tool_calls) {
        json input = json::parse(tc.arguments, nullptr, false);
        blocks.push_back({ { "type", "tool_use" }, { "id", tc.id }, { "name", tc.name },
                           { "input", input.is_discarded() ? json::object() : input } });
    }
    return blocks;
}

static std::string anthropic_stop_reason(const common_chat_msg & m, const GenStats & st) {
    if (!m.tool_calls.empty()) return "tool_use";
    return st.finish == "length" ? "max_tokens" : "end_turn";
}

// Anthropic SSE from successive parses of the reply: one content block open at a time (thinking, text, then
// one per tool call), each fed what grew since the last update. A tool block starts once its name is complete
// (its arguments began, a later call exists, or the reply ended), as llama-server holds names back
struct AnthropicStream {
    httplib::DataSink & sink;
    int index = -1;
    std::string open;  // "", "thinking", "text", "tool"
    size_t sent_think = 0, sent_text = 0, sent_args = 0, tools = 0;  // tools = tool blocks started

    bool event(const std::string & type, const json & data) {
        const std::string s = "event: " + type + "\ndata: " + data.dump() + "\n\n";
        return sink.write(s.data(), s.size());
    }
    bool close() {
        if (open.empty()) return true;
        open.clear();
        return event("content_block_stop", { { "type", "content_block_stop" }, { "index", index } });
    }
    bool start(const std::string & kind, const json & block) {
        if (!close()) return false;
        open = kind;
        return event("content_block_start", { { "type", "content_block_start" }, { "index", ++index }, { "content_block", block } });
    }
    bool delta(const json & d) { return event("content_block_delta", { { "type", "content_block_delta" }, { "index", index }, { "delta", d } }); }

    bool update(const common_chat_msg & m, bool last) {
        if (m.reasoning_content.size() > sent_think) {
            if (open != "thinking" && !start("thinking", { { "type", "thinking" }, { "thinking", "" } })) return false;
            if (!delta({ { "type", "thinking_delta" }, { "thinking", m.reasoning_content.substr(sent_think) } })) return false;
            sent_think = m.reasoning_content.size();
        }
        if (m.content.size() > sent_text) {
            if (open != "text" && !start("text", { { "type", "text" }, { "text", "" } })) return false;
            if (!delta({ { "type", "text_delta" }, { "text", m.content.substr(sent_text) } })) return false;
            sent_text = m.content.size();
        }
        for (size_t i = tools ? tools - 1 : 0; i < m.tool_calls.size(); ++i) {
            const auto & tc = m.tool_calls[i];
            if (i == tools) {  // not started yet
                if (tc.name.empty() || (tc.arguments.empty() && i + 1 == m.tool_calls.size() && !last)) break;
                if (!start("tool", { { "type", "tool_use" }, { "id", tc.id }, { "name", tc.name }, { "input", json::object() } })) return false;
                ++tools;
                sent_args = 0;
            }
            if (tc.arguments.size() > sent_args) {
                if (!delta({ { "type", "input_json_delta" }, { "partial_json", tc.arguments.substr(sent_args) } })) return false;
                sent_args = tc.arguments.size();
            }
        }
        return last ? close() : true;
    }
};

// Windows lets a second program listen on a port opened with SO_REUSEADDR (httplib's default), and then requests
// go to either copy. Zorlo takes its port exclusively, and checks it the same way before loading anything.
#ifdef _WIN32
static void exclusive_port(socket_t s) {
    BOOL on = TRUE;
    setsockopt(s, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, (const char *) &on, sizeof on);
}
static bool port_in_use(const std::string & host, int port) {
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET) return false;
    exclusive_port(s);
    sockaddr_in sa = {};
    sa.sin_family = AF_INET;
    sa.sin_port = htons((u_short) port);
    inet_pton(AF_INET, host == "localhost" ? "127.0.0.1" : host.c_str(), &sa.sin_addr);
    const bool busy = bind(s, (const sockaddr *) &sa, sizeof sa) != 0;
    closesocket(s);
    return busy;
}
#else
static void exclusive_port(socket_t) {}
static bool port_in_use(const std::string &, int) { return false; }  // listen() reports it on other systems
#endif

static int cmd_serve(const Args & a) {
    // the port first: a second copy must say so at once, not after loading the model (minutes, and its memory)
    if (port_in_use(a.host, a.port))
        throw std::runtime_error("port " + std::to_string(a.port) + " is in use. Zorlo is probably already running: open http://127.0.0.1:" +
                                 std::to_string(a.port) + " in your browser, or stop every copy with: Stop-Process -Name zorlo");
    Session s(a);
    std::atomic<uint64_t> next_id{ 1 };
    httplib::Server svr;
    svr.set_socket_options(exclusive_port);

    zorlo_agent_register(svr, a.host, a.port, ZORLO_UI_HTML);  // "/" (agent app), /ui, /preview, /agent/* (src/cli/agent.cpp)
    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });
    svr.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        const json j = { { "object", "list" },
                         { "data", json::array({ { { "id", s.model_id() }, { "object", "model" }, { "owned_by", "zorlo" } } }) } };
        res.set_content(j.dump(), "application/json");
    });
    // one chat request with an OpenAI-shaped body; `anthropic` answers in the Messages API format instead
    auto complete = [&](const std::string & body_text, httplib::Response & res, bool anthropic) {
        std::vector<int32_t> prompt;
        Sampling sp;
        int max_tokens = 0;
        bool stream = false;
        auto chat = std::make_shared<common_chat_session>();
        std::vector<std::string> stops;
        zorlo::MoeModel::Lossy lossy = s.model().lossy;
        std::string model = s.model_id();
        try {
            const json body = json::parse(body_text);
            // same request semantics as llama-server --jinja (tools/server/server-common.cpp)
            // common_json keeps the client's key order, which the template prints (nlohmann::json sorts keys)
            const common_json cbody = common_json::parse(body_text);
            common_chat_templates_inputs in;
            in.messages = common_chat_msgs_parse_oaicompat(cbody.at("messages"));
            if (in.messages.empty()) throw std::runtime_error("messages is empty");
            if (cbody.contains("tools")) in.tools = common_chat_tools_parse_oaicompat(cbody.at("tools"));
            in.tool_choice = common_chat_tool_choice_parse_oaicompat(body.value("tool_choice", std::string("auto")));
            in.parallel_tool_calls = body.value("parallel_tool_calls", common_chat_templates_get_caps(s.templates())["supports_parallel_tool_calls"]);
            in.reasoning_format = common_reasoning_format_from_name(body.value("reasoning_format", std::string("deepseek")));
            in.enable_thinking = common_chat_templates_support_enable_thinking(s.templates());
            const json kwargs = body.value("chat_template_kwargs", json::object());
            for (const auto & [k, v] : kwargs.items()) in.chat_template_kwargs[k] = v.dump();
            if (in.chat_template_kwargs.count("enable_thinking")) in.enable_thinking = in.chat_template_kwargs["enable_thinking"] == "true";
            *chat = s.chat(in);
            prompt = s.tokenize(chat->prompt().str());
            stops = chat->additional_stops();
            if (body.contains("stop")) {
                const auto & st = body.at("stop");
                if (st.is_string()) stops.push_back(st.get<std::string>());
                else for (const auto & x : st) stops.push_back(x.get<std::string>());
            }
            if ((int) prompt.size() >= s.n_ctx()) {
                throw std::runtime_error("prompt is " + std::to_string(prompt.size()) + " tokens, the context is " +
                                         std::to_string(s.n_ctx()) + " (zorlo serve -c)");
            }
            const int room = s.n_ctx() - (int) prompt.size();
            max_tokens = std::clamp(body.value("max_completion_tokens", body.value("max_tokens", room)), 1, room);
            sp.temperature = body.value("temperature", 0.6f);  // Qwen3's recommended sampling
            sp.top_p = body.value("top_p", 0.95f);
            sp.top_k = body.value("top_k", 20);
            if (body.contains("seed")) sp.seed = body.at("seed").get<uint32_t>();
            stream = body.value("stream", false);
            if (body.contains("mode")) lossy = mode_lossy(body.at("mode").get<std::string>());  // else the server default
            model = anthropic ? body.value("model", model) : model;
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content((anthropic ? anthropic_error(e.what()) : error_json(e.what())).dump(), "application/json");
            return;
        }
        const std::string n = std::to_string(next_id++);
        const std::string id = (anthropic ? "msg_" : "chatcmpl-") + n;
        const long long created = (long long) std::time(nullptr);

        // tool-call ids stay fixed per index while the message is parsed again on every chunk
        auto ids = std::make_shared<std::vector<std::string>>();
        auto parsed = [chat, ids, n, anthropic](const common_chat_input & c, bool last) {
            common_chat_msg m = last ? chat->finish(c) : chat->feed(c);
            m.set_tool_call_ids(*ids, [&] { return (anthropic ? "toolu_" : "call_") + n + "_" + std::to_string(ids->size()); });
            return m;
        };
        auto log = [id](const GenStats & st) {
            fprintf(stderr, "request %s: %d prompt tokens (%d reused, %d from disk), prefill %.1f s, %d generated (%d of %d drafts accepted), %.2f tok/s\n", id.c_str(),
                    st.n_prompt, st.n_reused, st.n_disk, st.prefill_s, st.n_gen, st.n_accepted, st.n_drafted, st.n_gen / std::max(st.gen_s, 1e-9));
        };
        auto finish_reason = [](const common_chat_msg & m, const GenStats & st) {
            return m.tool_calls.empty() ? st.finish : std::string("tool_calls");
        };

        if (!stream) {
            common_chat_input out;
            const GenStats st = s.generate(prompt, max_tokens, sp, stops, [&](const common_chat_input & c) {
                out.append(c);
                return true;
            }, &lossy);
            const common_chat_msg msg = parsed(out, true);
            const json j = anthropic
                ? json{ { "id", id }, { "type", "message" }, { "role", "assistant" }, { "model", model },
                        { "content", anthropic_content(msg) }, { "stop_reason", anthropic_stop_reason(msg, st) }, { "stop_sequence", nullptr },
                        { "usage", { { "input_tokens", st.n_prompt }, { "output_tokens", st.n_gen } } } }
                : json{ { "id", id }, { "object", "chat.completion" }, { "created", created }, { "model", model },
                        { "choices", json::array({ { { "index", 0 },
                                                     { "message", message_json(msg) },
                                                     { "finish_reason", finish_reason(msg, st) } } }) },
                        { "usage", { { "prompt_tokens", st.n_prompt }, { "completion_tokens", st.n_gen },
                                     { "total_tokens", st.n_prompt + st.n_gen } } } };
            res.set_content(j.dump(), "application/json");
            log(st);
            return;
        }

        if (anthropic) {
            res.set_chunked_content_provider("text/event-stream", [&s, prompt, max_tokens, sp, stops, lossy, chat, parsed,
                                                                   log, id, model](size_t, httplib::DataSink & sink) {
                AnthropicStream as{ sink };
                if (!as.event("message_start", { { "type", "message_start" },
                                                 { "message", { { "id", id }, { "type", "message" }, { "role", "assistant" }, { "model", model },
                                                                { "content", json::array() }, { "stop_reason", nullptr }, { "stop_sequence", nullptr },
                                                                { "usage", { { "input_tokens", prompt.size() }, { "output_tokens", 0 } } } } } })) {
                    return false;
                }
                GenStats st;
                common_chat_msg msg;
                try {
                    st = s.generate(prompt, max_tokens, sp, stops, [&](const common_chat_input & c) {
                        return c.empty() ? as.event("ping", { { "type", "ping" } }) : as.update(parsed(c, false), false);
                    }, &lossy);
                    msg = parsed({}, true);
                    as.update(msg, true);
                } catch (const std::exception & e) {
                    as.event("error", anthropic_error(e.what()));
                    sink.done();
                    return false;
                }
                as.event("message_delta", { { "type", "message_delta" },
                                            { "delta", { { "stop_reason", anthropic_stop_reason(msg, st) }, { "stop_sequence", nullptr } } },
                                            { "usage", { { "output_tokens", st.n_gen } } } });
                as.event("message_stop", { { "type", "message_stop" } });
                sink.done();
                log(st);
                return true;
            });
            return;
        }

        res.set_chunked_content_provider("text/event-stream", [&s, prompt, max_tokens, sp, stops, lossy, chat, parsed,
                                                               finish_reason, log, id, created, model](size_t, httplib::DataSink & sink) {
            auto chunk = [&](const json & delta, const json & finish) {
                const json j = { { "id", id }, { "object", "chat.completion.chunk" }, { "created", created },
                                 { "model", model },
                                 { "choices", json::array({ { { "index", 0 }, { "delta", delta }, { "finish_reason", finish } } }) } };
                const std::string line = "data: " + j.dump() + "\n\n";
                return sink.write(line.data(), line.size());
            };
            if (!chunk({ { "role", "assistant" } }, nullptr)) return false;
            common_chat_msg prev = chat->msg();
            auto send = [&](const common_chat_msg & m) {
                for (const auto & d : common_chat_msg_diff::compute_diffs(prev, m)) {
                    if (!chunk(delta_json(d), nullptr)) return false;
                }
                prev = m;
                return true;
            };
            GenStats st;
            // when the model's output doesn't fit the template's tool-call grammar, the parser throws; then the raw text
            // is sent as content instead, so the client can still read (or repair) the tool call
            std::string raw;
            bool broken = false;
            try {
                st = s.generate(prompt, max_tokens, sp, stops, [&](const common_chat_input & c) {
                    if (c.empty()) return sink.write(": ping\n\n", 8);  // SSE comment: keep-alive
                    raw += c.text;
                    if (broken) return true;
                    try { return send(parsed(c, false)); } catch (const std::exception & pe) {
                        fprintf(stderr, "request %s: output not in the tool-call format (%s); sending it as text\n", id.c_str(), pe.what());
                        broken = true;
                        return true;
                    }
                }, &lossy);
                if (!broken) {
                    try { send(parsed({}, true)); } catch (const std::exception & pe) {
                        fprintf(stderr, "request %s: output not in the tool-call format (%s); sending it as text\n", id.c_str(), pe.what());
                        broken = true;
                    }
                }
                if (broken) {
                    const size_t k = raw.find("<tool_call>");
                    std::string tail = k != std::string::npos ? raw.substr(k)
                                       : raw.rfind(prev.content, 0) == 0 ? raw.substr(prev.content.size()) : raw;
                    if (!tail.empty() && !chunk({ { "content", (prev.content.empty() ? "" : "\n") + tail } }, nullptr)) return false;
                }
            } catch (const std::exception & e) {
                const std::string line = "data: " + error_json(e.what()).dump() + "\n\n";
                sink.write(line.data(), line.size());
                sink.done();
                return false;
            }
            chunk(json::object(), finish_reason(prev, st));
            sink.write("data: [DONE]\n\n", 14);
            sink.done();
            log(st);
            return true;
        });
    };

    svr.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & res) { complete(req.body, res, false); });
    // Anthropic Messages API (Claude Code: ANTHROPIC_BASE_URL=http://host:port)
    svr.Post("/v1/messages", [&](const httplib::Request & req, httplib::Response & res) {
        std::string oai;
        try {
            oai = anthropic_to_oai(req.body);
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(anthropic_error(e.what()).dump(), "application/json");
            return;
        }
        complete(oai, res, true);
    });
    svr.Post("/v1/messages/count_tokens", [&](const httplib::Request & req, httplib::Response & res) {
        try {
            const common_json cbody = common_json::parse(anthropic_to_oai(req.body));
            common_chat_templates_inputs in;
            in.messages = common_chat_msgs_parse_oaicompat(cbody.at("messages"));
            if (cbody.contains("tools")) in.tools = common_chat_tools_parse_oaicompat(cbody.at("tools"));
            in.enable_thinking = common_chat_templates_support_enable_thinking(s.templates());
            res.set_content(json{ { "input_tokens", s.tokenize(s.chat(in).prompt().str()).size() } }.dump(), "application/json");
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(anthropic_error(e.what()).dump(), "application/json");
        }
    });

    fprintf(stderr, "serving %s on http://%s:%d (chat UI at /, OpenAI /v1/chat/completions, Anthropic /v1/messages)\n", s.model_id().c_str(),
            a.host.c_str(), a.port);
    if (!svr.listen(a.host, a.port))
        throw std::runtime_error("cannot listen on " + a.host + ":" + std::to_string(a.port) +
                                 ". Is Zorlo already running? Use that one at http://127.0.0.1:" + std::to_string(a.port) +
                                 ", or stop every copy with:  Stop-Process -Name zorlo");
    return 0;
}

static int cmd_selftest() {
    int bad = 0;
    auto check = [&](bool ok, const char * what) {
        if (!ok) fprintf(stderr, "FAIL: %s\n", what);
        bad += !ok;
    };
    check(utf8_complete("abc") == 3, "ascii is complete");
    check(utf8_complete("a\xE2\x82") == 1, "cut 3-byte sequence is held back");
    check(utf8_complete("a\xE2\x82\xAC") == 4, "full 3-byte sequence passes");
    check(utf8_complete("\xF0\x9F\x98") == 0, "cut 4-byte sequence is held back");
    check(utf8_complete("") == 0, "empty");

    // Anthropic -> OpenAI: billing hash blanked, tool results after the turn, tool_use as tool_calls
    const json oai = json::parse(anthropic_to_oai(R"({"system":"x-anthropic-billing-header: v=1; cch=ab12c; rest",
        "messages":[{"role":"user","content":"hi"},
                    {"role":"assistant","content":[{"type":"text","text":"ok"},{"type":"tool_use","id":"t1","name":"f","input":{"a":1}}]},
                    {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":[{"type":"text","text":"42"}]}]}],
        "tools":[{"name":"f","description":"d","input_schema":{"type":"object"}}],"stop_sequences":["X"]})"));
    const auto & m = oai["messages"];
    check(m[0]["content"] == "x-anthropic-billing-header: v=1; cch=fffff; rest", "billing hash blanked");
    check(m[2]["tool_calls"][0]["function"]["arguments"] == R"({"a":1})" && m[3]["role"] == "tool" && m[3]["content"] == "42",
          "tool_use and tool_result mapped");
    check(oai["tools"][0]["function"]["name"] == "f" && oai["stop"][0] == "X", "tools and stop sequences mapped");
    fprintf(stderr, bad ? "selftest: %d failures\n" : "selftest: ok\n", bad);
    return bad ? 1 : 0;
}

int zorlo_pack_main(int argc, char ** argv);   // src/pack/pack.cpp
int zorlo_pull_main(int argc, char ** argv);   // src/cli/pull.cpp
int zorlo_start_main(int argc, char ** argv);  // src/cli/start.cpp
int zorlo_path_main(int argc, char ** argv);   // src/cli/start.cpp

// `zorlo serve` for zorlo start: argv as on the command line ("zorlo", "serve", <pack>, options...)
int zorlo_serve_main(int argc, char ** argv) {
    try {
        return cmd_serve(parse(argc, argv, 2));
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}

#ifdef _WIN32
// Ctrl+C / Ctrl+Break / closing the window: stop at once (the model's memory is the OS's to take back). Switching the
// handling on explicitly also covers a Zorlo started by a program that told its children to ignore Ctrl+C.
static BOOL WINAPI on_console_stop(DWORD ev) {
    if (ev == CTRL_C_EVENT || ev == CTRL_BREAK_EVENT || ev == CTRL_CLOSE_EVENT) {
        fprintf(stderr, "\n  Stopping Zorlo.\n");
        fflush(stderr);
        _exit(0);
    }
    return FALSE;
}
#endif

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
#ifdef _WIN32
    SetConsoleCtrlHandler(nullptr, FALSE);
    SetConsoleCtrlHandler(on_console_stop, TRUE);
#endif
    if (argc >= 2 && std::string(argv[1]) == "selftest") return cmd_selftest();
    if (argc >= 2 && std::string(argv[1]) == "pull") return zorlo_pull_main(argc - 1, argv + 1);
    if (argc >= 2 && std::string(argv[1]) == "start") return zorlo_start_main(argc - 1, argv + 1);
    if (argc >= 2 && std::string(argv[1]) == "path") return zorlo_path_main(argc - 1, argv + 1);
    if (argc >= 2 && std::string(argv[1]) == "pack") {
        std::vector<char *> pa(argv + 1, argv + argc);
        char name[] = "zorlo pack";
        pa[0] = name;
        return zorlo_pack_main((int) pa.size(), pa.data());
    }
    if (argc < 3) {
        fprintf(stderr,
                "usage:\n  zorlo run    <pack> -p \"prompt\" [-n 256] [options]\n"
                "  zorlo serve  <pack> [--host 127.0.0.1] [--port 8080] [options]   (chat UI at /)\n"
                "  zorlo golden <pack> <golden.jsonl> [--prompts N] [options]\n"
                "  zorlo start  (menu: pick, download or update a model, serve it, open the chat page)\n"
                "  zorlo path on | off  (whether `zorlo` works from any folder; on after the first start)\n"
                "  zorlo pull   <owner/name> [out_dir] [--lowbit]   (download or update a pack from Hugging Face)\n"
                "  zorlo pack   <model.gguf> <out_dir>   (GGUF -> Zorlo pack, run once per model)\n"
                "  zorlo selftest\n"
                "options: -t 4  -c 4096  --cache-mb 0 (auto)  --io-threads 4  --protect F  --prefetch 0 (-1 = off)\n"
                "         --kv-disk-mb 4096 (saved prompt KV, 0 = off)  -np 1 (parallel requests)\n"
                "lossy:   --mode exact|balanced|fast  --skip-below W  --prefer-cached D  --expert-mass M\n"
                "         --lowbit (needs a pack made with --lowbit)\n");
        return 1;
    }
    try {
        const std::string cmd = argv[1];
        Args a = parse(argc, argv, 2);
        // a model's short name works too: `zorlo serve qwen3-30b-a3b` = <zorlo folder>\models\qwen3-30b-a3b
        if (!a.pack.empty() && !std::filesystem::exists(std::filesystem::u8path(a.pack))) {
            const auto m = std::filesystem::u8path(zorlo_models_dir()) / std::filesystem::u8path(a.pack);
            if (std::filesystem::exists(m / "manifest.json")) a.pack = m.u8string();
        }
        if (cmd == "run") return cmd_run(a);
        if (cmd == "serve") return cmd_serve(a);
        if (cmd == "golden") return cmd_golden(a);
        if (cmd == "imatrix") return cmd_imatrix(a);
        fprintf(stderr, "unknown command %s\n", cmd.c_str());
        return 1;
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
