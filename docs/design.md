# Design and measurements

How the engine works, what each part changed, and how it was measured. Numbers are from the
reference PC unless stated: a 4-core laptop CPU with AVX-512 VNNI, 8 GB RAM and one NVMe SSD.

## The disk is the limit

Reference PC: a 4-core laptop processor with AVX-512 VNNI, 8 GB of memory and an NVMe SSD; 2.75 MB random unbuffered reads:

| Queue depth | MB/s | reads/s | no-cache ceiling, Qwen3-30B-A3B (384 experts/token) |
|---|---|---|---|
| 1 | 1851 | 673 | 1.75 tok/s |
| 2 | 2963 | 1077 | 2.81 tok/s |
| 4-16 | ~2940 | ~1070 | ~2.8 tok/s |

The SSD saturates at queue depth 2, so the engine needs only 2-4 reads in flight. Every expert cache
hit raises this ceiling (50% hit rate -> ~5.6 tok/s, if compute keeps up).

### Qwen3-30B-A3B Q4_K_M (Phase 0)

- llama.cpp `7170cad`, mmap from SSD, 4 threads, ~1.5 GB RAM free: **1.43 tok/s** decode (`llama-bench tg32`).
- Pack: `core.gguf` 1.00 GB (incl. 175 MB embedding table that can stay on disk) + `experts.bin` 17.55 GB;
  expert blob 2.92 MB (layers with Q6_K down) or 2.53 MB (all Q4_K), ~1.05 GB per token with no cache.
- Routing trace: 20 mixed prompts x 200 greedy tokens = 4000 tokens (`traces/qwen3-30b-a3b.npy`).
  Each token reuses 42.6% of the previous token's experts per layer (random: 6.25%); the top 16 of 128
  experts per layer carry 37% of traffic; two unrelated prompts share only 37% of their top-16 sets.

Simulated expert cache on that trace (`tools/cachesim.py`, 2.72 MB/expert, 2950 MB/s, disk time only;
CPU compute caps the reference PC at very roughly 8-12 tok/s):

| Cache | best policy | hit rate | MB read/token | disk-bound tok/s |
|---|---|---|---|---|
| 0.75 GB | static hot set | 15.8% | 879 | ~3.4 |
| 1.5 GB | per-layer LRU | 45.9% | 565 | ~5.2 |
| 3 GB | per-layer LRU | 67.5% | 340 | ~8.7 |
| 5 GB | LFU | 86.1% | 145 | ~20 |

A recency cache only works once it holds at least one token's working set (48 layers x 8 experts ~ 1.05 GB);
below that a fixed hot set is better. The engine should switch policy by budget.

### Phase 1: exact streaming engine

`zorlo golden` vs `zorlo-golden` (llama.cpp, flash attention, no weight repacking), 20 prompts x 64 greedy
tokens with reference tokens fed back: **20/20 prompts identical, 1280/1280 steps match**, largest logit
difference 5.7e-6 (the 9-digit precision of the reference file).

Speed, same run, 256 MB cache (87 slots, below one token's working set, so 0% hits), one read at a time:

| | decode tok/s | notes |
|---|---|---|
| llama.cpp mmap | 1.43 | |
| Zorlo Phase 1 | **1.80** | 1.0 GB resident; reads 2.17 GB/s at queue depth 1 |

Time split: expert reads ~86%, attention+routing ~4%, expert math + head ~10%. Next: parallel reads
(SSD gives 2.95 GB/s at QD 2+), a cache above 1.1 GB, and prefetch.

```
zorlo run models\qwen3-30b-a3b.zorlo -p "your prompt" -n 256 --cache-mb 1024
zorlo golden models\qwen3-30b-a3b.zorlo golden\qwen3-30b-a3b.jsonl
```

### Phase 2: parallel reads, learned cache, prefetch

- 4 I/O threads, each with its own unbuffered handle; demand reads ahead of prefetches; a layer's experts
  are computed batch by batch as their reads land.
- Cache: LRU whose eviction skips a protected hot set (top decayed use counts). Protected share by budget,
  chosen on the real trace: 90% below one token's working set, 50% up to 2x, 75% above
  (simulated hit rate at 0.75 GB: 28.7% vs 0% for LRU and 15.8% for a fixed hot set).
- Prefetch: the next layer's router applied to this layer's post-attention state predicts its experts
  (recall 88% for top-8, 95% for top-10); reads start while this layer computes.

Output is still identical to llama.cpp on all 20 golden prompts. Speed on the reference PC (decode, tok/s):

| Run | 256 MB cache | 1.2 GB cache |
|---|---|---|
| llama.cpp mmap | 1.43 | 1.43 |
| Phase 1 (sync reads, LRU) | 1.80 | - |
| Phase 2, golden prompts 1-3 (64 tokens each) | 2.94 | 3.40 |
| Phase 2, `zorlo run` 256 tokens | **2.98** | **3.47** |

Prefetch off vs on at 1.2 GB: 3.03 vs 3.39 tok/s; 0, 2 or 4 extra predicted experts all give ~3.4, so the
default is 0 (86% of prefetches used vs 60% at 4). Prefill runs ~9.5 tok/s. The SSD now runs at its limit
(~2.9 GB/s), so further speed comes from fewer bytes per token: a larger cache (more RAM) or the Phase 4
lossy options.

### Phase 3: live RAM budget, OpenAI-compatible server

- `zorlo serve`: OpenAI chat completions over cpp-httplib (vendored by llama.cpp), sampling via llama.cpp's
  sampler chain. A conversation's earlier turns stay in the KV cache: a follow-up only evaluates the new
  tokens (test: 44 of 69 prompt tokens reused).
- Budget: cache target = free RAM + RAM the cache already holds - 512 MB, checked every 16 tokens; the
  arena reserves address space for every expert and commits/decommits slots as the target moves.
  Test: another process took 1.5 GB mid-generation; the cache went 288 -> 159-222 slots, then back to 349
  when it was released; free RAM stayed near 512 MB and decoding held 3.36 tok/s.
- Output still identical to llama.cpp (golden prompts 1-3 with the auto-sized cache, 2.82 tok/s at 198 MB).

Not done (by design for now): parallel requests (they queue). Tool calls, stop strings and
`reasoning_content` came in Phase 6.

### Phase 4: opt-in lossy speed switches

All off by default (output then stays identical to llama.cpp). Quality is measured against the exact
model on the golden set: greedy picks that still match, and the rise in mean NLL of the exact model's tokens
(teacher-forced; 0 = no drift). 10 prompts x 64 tokens, 256 MB cache (the 4 GB-laptop case):

| Switch | decode tok/s | same greedy token | NLL rise | what it does |
|---|---|---|---|---|
| none (exact) | 2.93 | 100% | 0 | |
| `--skip-below 0.05` | 2.99 | 98.8% | +0.006 | drop routed experts with weight < 0.05 (2.8% of them), renormalize |
| `--skip-below 0.07` | **3.27** | 96.9% | +0.030 | drops 12.4% |
| `--skip-below 0.10` | 4.35 | 93.8% | +0.171 | drops 40.8% |
| `--prefer-cached 0.3` | 3.01 | 98.6% | +0.011 | use a cached expert whose router prob is within 30% of an uncached pick |
| `--prefer-cached 0.5` | 3.03 | 97.7% | +0.014 | |
| `--lowbit` | **4.89** | 92.7% | +0.205 | misses read a Q2_K copy (1.48 vs 2.5-2.9 MB); hot experts stay full |
| `--lowbit --skip-below 0.05` | 4.90 | 92.0% | +0.218 | |

Reading it: `--skip-below 0.07` is the cheap win (+12% speed, small drift). `--lowbit` is the big one
(+67%) at a cost similar to running a 2-bit model for the cold experts; the Q2_K copy is requantized from
Q4_K without an importance matrix, so a better 2-bit source would cost less. `--prefer-cached` barely helps
at this cache size (few experts resident) and is kept for larger caches.

`--lowbit` needs the low-bit copy in the pack (once, ~14 min here, +9.5 GB on disk):

```
zorlo-pack Qwen3-30B-A3B-Q4_K_M.gguf models\qwen3-30b-a3b.zorlo --lowbit q2_K
```

### Phase 5: embedding table on disk

The token embedding table now stays on disk (a token reads its own row), which saves 175 MB of RAM on
Qwen3-30B-A3B: 0.82 GB stays resident.

### Phase 6: agent API (tool calls), same as llama-server

`zorlo serve` now renders prompts and parses replies with llama.cpp's own chat code (`common/chat.cpp`, linked
as `llama-common`): the model's Jinja template with `tools`, Qwen3 `<tool_call>` parsing into OpenAI
`tool_calls` (streamed as deltas too), `<think>` returned as `reasoning_content`, `stop` strings, and
`chat_template_kwargs` (e.g. `{"enable_thinking": false}`). Request fields follow `llama-server --jinja`:
`tools`, `tool_choice`, `parallel_tool_calls`, `reasoning_format`. Prompt rendering is per request, so decode
speed is unchanged.

Test against `llama-server` (same commit, `-fa on --no-repack`), greedy, Qwen3-30B-A3B, non-streamed and
streamed: a tool call, a reply to a tool result, and thinking off with a stop string. **3/3 identical**
(same prompt token counts, same `tool_calls`, `reasoning_content`, `content` and `finish_reason`).

Not done: grammar-constrained sampling, so `tool_choice: "required"` and `response_format` are not enforced
(llama-server constrains them with a grammar).

### Phase 7: saved prompt KV and hot-expert profile

Agents resend a long system prompt + tool list with every request (and every subagent shares it), and
prefill runs at only ~30 tok/s. Now each prompt's KV is written to `<pack>\kv\*.zkv` (raw f16 cells, ~96 KB
per token on Qwen3-30B-A3B) and a later prompt that starts the same way reads the longest matching prefix
from disk instead of recomputing it, also after a restart. Saved prompts that a newer one extends (earlier
turns of a conversation) are dropped; the oldest go once the folder passes `--kv-disk-mb` (default 4096,
0 = off). The expert use counts are saved to `<pack>\hot.bin` after each request; at startup they restore
the hot set and the most used experts are read into the cache in the background.

Test (`zorlo serve`, greedy, 2367-token agent prompt: 60-rule system prompt + 2 tools, two different user
tasks, then the server restarted and both asked again):

| | prefill | output |
|---|---|---|
| first request, cold | 69.0 s | tool call |
| second task, same system prompt (KV in RAM, as before) | 4.3 s | tool call |
| after restart, each task (KV read from disk in 0.12-0.14 s) | **0.4 s** | **identical to the cold run** |

### Phase 8: several requests at once (`-np N`)

`zorlo serve -np N` runs up to N requests together (an agent and its subagents) instead of queueing them.
Each slot has its own context-sized KV cache (~400 MB at `-c 4096` on Qwen3-30B-A3B). One engine thread
steps all running requests together: attention runs per sequence on its own KV, and the experts run
once over all of the batch's tokens, so an expert two requests both need is read once. Prompts are fed
512 tokens per step (llama.cpp's ubatch), so a long prompt doesn't stop the others from decoding. A new
request takes the free slot whose KV already holds the longest start of its prompt (then the disk KV of
Phase 7). The default stays `-np 1`.

Output is unchanged by batching. Since Phase 10, each sequence gets its own expert and output-head matmuls,
because ggml picks its kernel by column count. So a sequence's math doesn't depend on what else is in the
batch. `zorlo golden` still gives 3/3 prompts and 192/192 steps identical to
llama.cpp. With `-np 3` and requests sent together, the tool call and the stop-string cases match
llama-server, and both 2367-token agent prompts (prefilled in 512-token chunks next to each other) match
their solo cold runs. The tool-result case gives the same text batched and solo when run cold. It differs
from the llama-server reference, whose prompt was evaluated as a 204-token cached prefix + 52 new tokens.
llama.cpp's own output depends on how a prompt is split into batches.

Throughput on the reference PC (3 stories x 96 tokens, greedy, fixed 1.5 GB expert cache):

| | total tok/s |
|---|---|
| `-np 1`, one after another | 3.36 |
| `-np 2`, two at a time | 3.30 |
| `-np 3`, all three at once | 3.41 |

So batching makes requests run side by side, but it doesn't raise total speed here. Two different
sequences mostly overlap on the popular experts, and the cache already holds those. What they don't share
has to be read anyway, so disk bytes per token stay the same. The gain is responsiveness for agents: a
subagent's short request no longer waits for another request to finish.

### Phase 9: AVX-512 VNNI prefill, still exact

Prefill is CPU-bound. In a 2162-token prefill (1.5 GB cache) only ~1-3 s go to waiting on reads: the 53 GB
of expert reads overlap with compute. `-t 8` gives the same as `-t 4`. For 8 or more columns,
llama.cpp runs K-quant matmuls in its "tiled" kernel (`ggml-cpu/tiled`), which has an AVX2 and an
AVX-512 VNNI variant. MSVC has no `/arch` switch for VNNI, so `GGML_NATIVE` built the tiled kernel's AVX2
variant, even on the reference CPU (the rest of ggml-cpu was built with `/arch:AVX512`).

- CMake now checks the build machine with CPUID and, if it has AVX-512 VNNI, defines `__AVX512VNNI__` for
  ggml-cpu.
- `patches/ggml-tiled-vnni-same-rounding.patch` makes the VNNI variant round like the AVX2 one: separate
  multiply, subtract and add when `__FMA__` is not defined, as on MSVC. The integer parts are exact in
  both, so the outputs match bit for bit. CMake applies the patch to the submodule if it isn't applied yet.

| 2162-token prefill | total | attention + routing | experts | tok/s |
|---|---|---|---|---|
| AVX2 (before) | 70.7 s | 26.8 s | 43.6 s | 30.6 |
| AVX-512 VNNI | **51.9 s** | 19.6 s | 32.0 s | **41.6** (+36%) |

The 2367-token agent prompt (Phase 7 test, cold) went from 69.0 s to 55.8 s. Exactness: `zorlo golden` on
all 20 prompts is **20/20 identical, 1280/1280 steps**, worst logit diff 5.7e-6 (as before). Both agent
prompts give byte-identical replies to the AVX2 build. Decode (1 column) still uses ggml's AVX2
`vec_dot`, so it is unchanged.

Tried and dropped:

- ggml's repacked weights (`CPU_REPACK`, llama.cpp's default) are 15% faster on prefill but not exact:
  golden went to 2/3 prompts, max logit diff 5.6, from the q4_K 8x8 kernel.
- A hand-written VNNI `vec_dot` was bit-identical to `vec_dot`, but `ggml_mul_mat` uses the tiled kernel
  for 8+ columns, so it differed from llama.cpp there, and it was slower than ggml's tiled loop.

### Phase 10: exact speculative decoding from the text itself (`--draft 6`, default)

Agents copy a lot: tool-call JSON, file contents they just read, code they rewrite. Each decode step now
looks for the longest earlier place (3 to 16 tokens) where the prompt + output so far ended the same way,
and drafts up to 6 of the tokens that followed. The draft goes into the same eval as the last sampled
token. Every row is then sampled with the request's own sampler, and drafts are kept only while they
equal the sampled token. Rejected drafts' KV is dropped. So the output is exactly what one-at-a-time
decoding gives, also with temperature > 0: same tokens, same sampler state.

What keeps it exact: ggml changes its kernel with the number of columns. K-quant matmuls switch to the
tiled kernel at 8 columns, and the F32 router goes to llamafile's GEMM at 2. So a check is at most 7
tokens, and its router runs one token per matmul.

Whether to draft is a cost decision. A check of w tokens costs more than one step (measured on the reference PC,
1.5 GB cache):

| tokens checked | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| step time vs 1 token | 1.0 | 1.92 | 2.46 | 2.99 | 3.74 | 4.12 | 5.81 |

Two tokens cost nearly two steps here, because consecutive tokens share few experts and decode is
disk-bound. So drafting only pays at very high acceptance (>90%). The draft length is chosen from this table
and the request's running acceptance rate. At low acceptance it doesn't draft (it retries now and then).

Test (`zorlo serve`, 1.5 GB cache, `--draft 0` vs `--draft 6`; decode tok/s from the server log):

| request | drafts accepted | output | decode tok/s |
|---|---|---|---|
| rewrite a 55-line Python file, one name changed (greedy) | 243 / 243 | identical | 2.70 -> **3.26** |
| same, temperature 0.6, seed 7 | 243 / 243 | identical | 2.75 -> **3.33** |
| three tool calls | 29 / 42 | identical | 2.04 -> 2.18 |
| short story | 0 / 3 | identical | (no change beyond noise) |

Exactness under stress: with `ZORLO_DRAFT_ALWAYS=1` (test hook: draft 6 tokens every step, almost all
wrong), the same four requests ran about 3,200 checks and gave identical output. `zorlo golden` stays at
20/20 prompts identical. Run-to-run noise on the reference PC is large (the same request varies by ~±20% with
disk/cache state), so small differences in the table are not significant.

### Phase 11: speed modes (`--mode exact|balanced|fast`), adaptive expert count, imatrix low-bit copy

Opt-in only: the default stays `exact`, identical to llama.cpp. What's new:

- **Adaptive expert count** (`--expert-mass M`): a token keeps its top experts until their router weights add
  up to M, and drops the rest (the top one always stays). Confident tokens use fewer experts, unsure ones
  keep all 8.
- **imatrix low-bit copy:** `zorlo imatrix <pack> <text>` runs a text through the exact model and records, per
  expert, how strongly each input is used. `zorlo-pack ... --lowbit q2_K --imatrix <pack>/imatrix.bin
  --lowbit-only` then rewrites only `experts.lo.bin`, weighting the 2-bit rounding by that importance, as
  llama.cpp's imatrix quantization does. Here: ~30k tokens of README, code and llama.cpp docs (not the
  golden prompts); 20 min for the matrix, 14 min for the copy.
- **Modes:** `--mode` sets the server default, and a request can send `"mode": "exact" | "balanced" |
  "fast"`. Explicit switches (`--skip-below`, `--expert-mass`, `--lowbit`, ...) override the preset.
  Requests with different modes can share a batch. Low-bit copies are read only when every request in
  the batch allows them. An exact request that finds a low-bit copy in the cache gets it re-read in full,
  even if that read is still in flight.

Quality vs speed (golden: 10 prompts x 64 tokens, 256 MB cache as in Phase 4; NLL rise = drift of the exact
model's tokens):

| setting | same greedy token | NLL rise | decode tok/s |
|---|---|---|---|
| **exact** | 100% | 0 | 2.25 |
| `--expert-mass 0.95` | 98.8% | +0.004 | ~2.2 |
| **balanced** = `--expert-mass 0.9` | 98.0% | +0.032 | 2.53 (+12%) |
| `--expert-mass 0.85` | 96.4% | +0.055 | 2.55 |
| `--expert-mass 0.8` | 95.6% | +0.081 | 2.64 |
| `--skip-below 0.07` (Phase 4) | 96.9% | +0.030 | 2.48 |
| `--lowbit`, plain Q2_K (Phase 4) | 92.8% | +0.200 | 4.89 (Phase 4) |
| `--lowbit`, imatrix Q2_K | 93.9% | +0.092 | 4.88 |
| **fast** = imatrix `--lowbit` + `--expert-mass 0.9` | 94.7% | +0.104 | **5.47 (2.4x)** |

Reading it: the imatrix copy halves the drift of the 2-bit experts (+0.200 -> +0.092) at the same speed.
`balanced` matches `--skip-below 0.07` on drift and keeps more tokens the same. `fast` is 2.4x the exact
speed at about the drift the old `--lowbit` had at 2.2x.

Server check (1.5 GB cache): the same tool-call request in `fast`, then `balanced`, then `exact`. All three
returned a valid `get_weather {"city": "Paris"}` call (5.47 / 3.13 / 2.70 tok/s). The `exact` one, run after
`fast` had filled the cache with low-bit copies, is identical to llama-server.

On disk: `experts.lo.bin` is now the imatrix copy. The old one is kept as `experts.lo.plain.bin` (9.5 GB, safe
to delete), and `experts.lo.imat.bin` is a hard link to the new one.

### Phase 13: one build for every x86 CPU, same output on all

The release build (`-DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON`) ships one `ggml-cpu-*.dll` per CPU type
and loads the best one at start. The engine reaches ggml-cpu only through ggml's backend registry, so it works
with either build.

Exactness is relative to **llama.cpp with AVX2 float order**. ggml's AVX-512 code computes float ops
(f32/f16 matmul incl. the f32 router, softmax, silu/exp, flash attention) in a different order than its AVX2
code; quantized matmuls already matched. Small router differences flipped expert choices, so llama.cpp's own
output differed between CPUs (12/20 golden prompts diverged). `patches/ggml-cpu-float-avx2-order.patch` adds
`GGML_CPU_X86_FLOAT_AVX2`, which CMake defines for every ggml-cpu target: AVX-512 builds then use the AVX2
code for float math only (quantized and VNNI kernels unchanged). Every CPU type now gives the same output.

`golden/qwen3-30b-a3b.jsonl` was regenerated with llama.cpp's AVX2 build. **The old golden file used the
AVX-512 float order** (`GGML_NATIVE` on the reference CPU), so the exactness numbers in earlier phases were measured
against that order. With the patch: haswell and icelake DLLs each give **20/20 prompts, 1280/1280 steps**,
worst logit diff 7.6e-6; attention+routing time is unchanged (41.1 s vs 41.5 s over the 20 prompts).

