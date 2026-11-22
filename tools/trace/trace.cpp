// zorlo-trace: record which experts a MoE model routes each generated token to.
//
// Runs greedy chat generation with llama.cpp over a prompt file and, through the scheduler eval
// callback, captures the "ffn_moe_topk-<layer>" tensors. Output is a .npy int16 array
// [tokens, layers, top_k] for decode steps only (one row per generated token), the input
// tools/cachesim.py expects.
//
//   zorlo-trace -m model.gguf -f prompts.txt -o trace.npy [-n 200] [-t threads] [-c ctx]
//
// Prompts are separated by blank lines. Each prompt starts from an empty KV cache.

#include "ggml-backend.h"
#include "llama.h"

#include <chrono>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct trace_state {
    int n_layer = 0;
    int top_k   = 0;
    std::vector<int32_t> row;   // [n_layer * top_k] for the current decode step
    std::vector<bool>    seen;  // which layers reported in the current step
};

static const char PREFIX[] = "ffn_moe_topk-";

static bool on_eval(ggml_tensor * t, bool ask, void * user_data) {
    const bool want = strncmp(t->name, PREFIX, sizeof(PREFIX) - 1) == 0;
    if (ask) {
        return want;
    }
    if (!want) {
        return true;
    }
    auto * s = (trace_state *) user_data;
    const int layer = atoi(t->name + sizeof(PREFIX) - 1);
    const int k = (int) t->ne[0];
    const int64_t n_tok = t->ne[1];
    if (t->type != GGML_TYPE_I32 || layer < 0 || layer >= s->n_layer) {
        return true;
    }
    if (s->top_k == 0) {
        s->top_k = k;
        s->row.assign((size_t) s->n_layer * k, 0);
    }
    // keep the last token of the batch: during decode the batch is exactly one token
    std::vector<int32_t> buf((size_t) k);
    const size_t off = (size_t) (n_tok - 1) * t->nb[1];
    if (ggml_backend_buffer_is_host(t->buffer)) {
        memcpy(buf.data(), (const char *) t->data + off, k * sizeof(int32_t));
    } else {
        ggml_backend_tensor_get(t, buf.data(), off, k * sizeof(int32_t));
    }
    memcpy(&s->row[(size_t) layer * k], buf.data(), k * sizeof(int32_t));
    s->seen[layer] = true;
    return true;
}

static void write_npy(const std::string & path, const std::vector<int16_t> & data, size_t T, int L, int K) {
    std::ostringstream h;
    h << "{'descr': '<i2', 'fortran_order': False, 'shape': (" << T << ", " << L << ", " << K << "), }";
    std::string header = h.str();
    const size_t total = 10 + header.size() + 1;
    header.append((64 - total % 64) % 64, ' ');
    header += '\n';
    std::ofstream f(path, std::ios::binary);
    f.write("\x93NUMPY\x01\x00", 8);
    const uint16_t hl = (uint16_t) header.size();
    f.write((const char *) &hl, 2);
    f.write(header.data(), header.size());
    f.write((const char *) data.data(), data.size() * sizeof(int16_t));
}

static std::vector<std::string> read_prompts(const std::string & path) {
    std::ifstream f(path);
    std::vector<std::string> out;
    std::string line, cur;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            if (!cur.empty()) {
                out.push_back(cur);
            }
            cur.clear();
        } else {
            cur += cur.empty() ? line : "\n" + line;
        }
    }
    if (!cur.empty()) {
        out.push_back(cur);
    }
    return out;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    std::string model_path, prompts_path, out_path = "trace.npy";
    int n_gen = 200, n_threads = 0, n_ctx = 2048;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "-m") model_path = argv[i + 1];
        else if (a == "-f") prompts_path = argv[i + 1];
        else if (a == "-o") out_path = argv[i + 1];
        else if (a == "-n") n_gen = atoi(argv[i + 1]);
        else if (a == "-t") n_threads = atoi(argv[i + 1]);
        else if (a == "-c") n_ctx = atoi(argv[i + 1]);
    }
    if (model_path.empty() || prompts_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf -f prompts.txt [-o trace.npy] [-n 200] [-t threads] [-c 2048]\n", argv[0]);
        return 1;
    }
    const auto prompts = read_prompts(prompts_path);
    if (prompts.empty()) {
        fprintf(stderr, "no prompts in %s\n", prompts_path.c_str());
        return 1;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", model_path.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    trace_state st;
    st.n_layer = llama_model_n_layer(model);
    st.seen.assign(st.n_layer, false);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_ctx;
    cp.cb_eval = on_eval;
    cp.cb_eval_user_data = &st;
    if (n_threads > 0) {
        cp.n_threads = cp.n_threads_batch = n_threads;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    const char * tmpl = llama_model_chat_template(model, nullptr);

    std::vector<int16_t> rows;
    size_t n_rows = 0, n_incomplete = 0;
    double decode_s = 0;

    for (size_t p = 0; p < prompts.size(); ++p) {
        llama_memory_clear(llama_get_memory(ctx), true);

        std::string text = prompts[p];
        if (tmpl) {
            llama_chat_message msg = { "user", prompts[p].c_str() };
            std::vector<char> buf(text.size() * 2 + 512);
            int n = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), (int32_t) buf.size());
            if (n > (int) buf.size()) {
                buf.resize(n);
                n = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), (int32_t) buf.size());
            }
            if (n > 0) {
                text.assign(buf.data(), n);
            }
        }
        const int n_prompt = -llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, true, true);
        std::vector<llama_token> toks(n_prompt);
        llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(), n_prompt, true, true);
        if (n_prompt + n_gen > n_ctx) {
            fprintf(stderr, "prompt %zu too long for -c %d, skipped\n", p, n_ctx);
            continue;
        }

        llama_batch_ext_clear(batch);
        for (int i = 0; i < n_prompt; ++i) {
            const int32_t idx = llama_batch_ext_add_token(batch, 0, toks[i]);
            const llama_pos pos = i;
            llama_batch_ext_set_pos(batch, idx, &pos);
        }
        llama_batch_ext_set_output_logits(batch, n_prompt - 1, true);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
            fprintf(stderr, "prefill failed on prompt %zu\n", p);
            return 1;
        }

        int made = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (llama_pos pos = n_prompt; made < n_gen; ++pos, ++made) {
            llama_token tok = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, tok)) {
                break;
            }
            llama_batch_ext_clear(batch);
            const int32_t idx = llama_batch_ext_add_token(batch, 0, tok);
            llama_batch_ext_set_pos(batch, idx, &pos);
            llama_batch_ext_set_output_logits(batch, 0, true);
            st.seen.assign(st.n_layer, false);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
                fprintf(stderr, "decode failed on prompt %zu\n", p);
                return 1;
            }
            bool complete = st.top_k > 0;
            for (bool b : st.seen) {
                complete = complete && b;
            }
            if (!complete) {
                ++n_incomplete;
                continue;
            }
            for (int32_t e : st.row) {
                rows.push_back((int16_t) e);
            }
            ++n_rows;
        }
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        decode_s += dt;
        fprintf(stderr, "prompt %zu/%zu: %d prompt tokens, %d generated, %.2f tok/s\n",
                p + 1, prompts.size(), n_prompt, made, made / (dt > 0 ? dt : 1));
        write_npy(out_path, rows, n_rows, st.n_layer, st.top_k);  // checkpoint after every prompt
    }

    fprintf(stderr, "wrote %s: [%zu, %d, %d], %zu steps without a full routing row, decode %.2f tok/s\n",
            out_path.c_str(), n_rows, st.n_layer, st.top_k, n_incomplete, n_rows / (decode_s > 0 ? decode_s : 1));

    llama_batch_ext_free(batch);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return n_rows ? 0 : 1;
}
