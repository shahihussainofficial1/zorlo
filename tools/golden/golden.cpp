// zorlo-golden: reference greedy outputs from llama.cpp, for checking the Zorlo engine.
//
// For each prompt (chat template applied, prompts separated by blank lines) it runs greedy decoding
// for exactly -n tokens (not stopping at end-of-generation, so runs stay comparable) and writes one
// JSON line: prompt token ids, generated ids, and the top-5 (id, logit) of every step.
//
// Kernels are pinned to what Zorlo uses: flash attention on, weight repacking off (repacked Q4_K uses
// different kernels), f16 KV cache.
//
//   zorlo-golden -m model.gguf -f prompts.txt -o golden.jsonl [-n 64] [-t 4] [-c 2048] [-p max_prompts]

#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

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
    std::string model_path, prompts_path, out_path = "golden.jsonl";
    int n_gen = 64, n_threads = 4, n_ctx = 2048, max_prompts = 1 << 30;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "-m") model_path = argv[i + 1];
        else if (a == "-f") prompts_path = argv[i + 1];
        else if (a == "-o") out_path = argv[i + 1];
        else if (a == "-n") n_gen = atoi(argv[i + 1]);
        else if (a == "-t") n_threads = atoi(argv[i + 1]);
        else if (a == "-c") n_ctx = atoi(argv[i + 1]);
        else if (a == "-p") max_prompts = atoi(argv[i + 1]);
    }
    if (model_path.empty() || prompts_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf -f prompts.txt [-o golden.jsonl] [-n 64] [-t 4] [-c 2048] [-p max_prompts]\n", argv[0]);
        return 1;
    }
    auto prompts = read_prompts(prompts_path);
    if ((int) prompts.size() > max_prompts) {
        prompts.resize(max_prompts);
    }

    llama_backend_init();
    if (ggml_backend_reg_count() == 0) ggml_backend_load_all();  // GGML_BACKEND_DL build: CPU backend is a DLL
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_extra_bufts = false;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", model_path.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_ctx;
    cp.n_threads = cp.n_threads_batch = n_threads;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }
    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    const char * tmpl = llama_model_chat_template(model, nullptr);
    FILE * out = fopen(out_path.c_str(), "w");

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

        llama_batch_ext_clear(batch);
        for (int i = 0; i < n_prompt; ++i) {
            const int32_t idx = llama_batch_ext_add_token(batch, 0, toks[i]);
            const llama_pos pos = i;
            llama_batch_ext_set_pos(batch, idx, &pos);
        }
        llama_batch_ext_set_output_logits(batch, n_prompt - 1, true);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
            fprintf(stderr, "prefill failed\n");
            return 1;
        }

        std::string gen = "[", top = "[";
        std::vector<int> order(n_vocab);
        for (int s = 0; s < n_gen; ++s) {
            const float * logits = llama_get_logits_ith(ctx, -1);
            for (int i = 0; i < n_vocab; ++i) order[i] = i;
            std::partial_sort(order.begin(), order.begin() + 5, order.end(),
                              [&](int a, int b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); });
            const llama_token tok = order[0];
            char buf[256];
            top += s ? ",[" : "[";
            for (int j = 0; j < 5; ++j) {
                snprintf(buf, sizeof(buf), "%s[%d,%.9g]", j ? "," : "", order[j], logits[order[j]]);
                top += buf;
            }
            top += "]";
            gen += (s ? "," : "") + std::to_string(tok);

            if (s + 1 < n_gen) {
                llama_batch_ext_clear(batch);
                const int32_t idx = llama_batch_ext_add_token(batch, 0, tok);
                const llama_pos pos = n_prompt + s;
                llama_batch_ext_set_pos(batch, idx, &pos);
                llama_batch_ext_set_output_logits(batch, 0, true);
                if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
                    fprintf(stderr, "decode failed\n");
                    return 1;
                }
            }
        }
        std::string ptoks = "[";
        for (int i = 0; i < n_prompt; ++i) ptoks += (i ? "," : "") + std::to_string(toks[i]);
        fprintf(out, "{\"prompt\": %zu, \"prompt_tokens\": %s], \"gen\": %s], \"top\": %s]}\n",
                p, ptoks.c_str(), gen.c_str(), top.c_str());
        fflush(out);
        fprintf(stderr, "prompt %zu/%zu done (%d prompt tokens)\n", p + 1, prompts.size(), n_prompt);
    }
    fclose(out);
    llama_batch_ext_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
