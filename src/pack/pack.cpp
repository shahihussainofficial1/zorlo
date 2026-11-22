// zorlo-pack: turn a MoE GGUF into a Zorlo pack directory.
//
//   core.gguf      every non-expert tensor + all metadata (stays resident in RAM)
//   experts.bin    for each layer, for each expert: one blob = gate | up | down, each part padded to
//                  64 bytes, the blob padded to 4096 bytes, so one expert = one aligned unbuffered read
//   manifest.json  where everything is
//
// Bytes are copied as-is (no requantization). Q4_K_M mixes types between layers, so the blob layout is
// recorded per layer.
//
//   zorlo pack model.gguf out_dir [--verify N]   (zorlo-pack is the same program on its own)

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <regex>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

static constexpr size_t BLOB_ALIGN = 4096;
static constexpr size_t PART_ALIGN = 64;
static const char * KINDS[3] = { "gate", "up", "down" };

static size_t pad(size_t n, size_t a) { return (n + a - 1) / a * a; }

struct part {
    int64_t     tensor_id;
    ggml_type   type;
    int64_t     ne0, ne1;   // per-expert matrix: ne0 = row length, ne1 = rows
    size_t      bytes;      // per expert
    size_t      offset;     // inside the blob
};

struct layer_info {
    int    il;
    part   parts[3];
    size_t stride;          // padded blob size
    size_t offset;          // of expert 0 in experts.bin
};

static void die(const char * msg, const std::string & arg = "") {
    fprintf(stderr, "zorlo-pack: %s%s\n", msg, arg.c_str());
    exit(1);
}

static void read_at(std::ifstream & f, uint64_t off, void * dst, size_t n) {
    f.seekg((std::streamoff) off);
    f.read((char *) dst, (std::streamsize) n);
    if ((size_t) f.gcount() != n) {
        die("short read from source GGUF");
    }
}

// fills `blob` (stride bytes) with expert e of layer L from the source GGUF
static void build_blob(std::ifstream & src, size_t data_off, gguf_context * g, const layer_info & L, int64_t e,
                       std::vector<uint8_t> & blob) {
    std::fill(blob.begin(), blob.end(), 0);
    for (const part & p : L.parts) {
        read_at(src, data_off + gguf_get_tensor_offset(g, p.tensor_id) + (uint64_t) e * p.bytes,
                blob.data() + p.offset, p.bytes);
    }
}

int zorlo_pack_main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.gguf out_dir [--verify N] [--lowbit q2_K [--imatrix FILE] [--lowbit-only]]\n"
                        "  --imatrix: importance matrix from `zorlo imatrix`, weights the low-bit requantization\n"
                        "  --lowbit-only: only rewrite experts.lo.bin of an existing pack in out_dir\n", argv[0]);
        return 1;
    }
    const std::string src_path = argv[1];
    const fs::path out = argv[2];
    int n_verify = 64;
    ggml_type lowbit = GGML_TYPE_COUNT;  // none
    std::string imatrix_path;
    bool lowbit_only = false;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&] {
            if (i + 1 >= argc) die("missing value for ", a);
            return std::string(argv[++i]);
        };
        if (a == "--verify") n_verify = atoi(val().c_str());
        else if (a == "--imatrix") imatrix_path = val();
        else if (a == "--lowbit-only") lowbit_only = true;
        else if (a == "--lowbit") {
            const std::string v = val();
            for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
                if (ggml_type_name((ggml_type) t) && v == ggml_type_name((ggml_type) t)) lowbit = (ggml_type) t;
            }
            if (lowbit == GGML_TYPE_COUNT) die("unknown --lowbit type ", v);
        } else {
            die("unknown option ", a);
        }
    }
    if (lowbit_only && lowbit == GGML_TYPE_COUNT) die("--lowbit-only needs --lowbit", "");
    if (lowbit_only && !fs::exists(out / "manifest.json")) die("--lowbit-only needs an existing pack in ", out.string());

    ggml_context * meta = nullptr;
    gguf_context * g = gguf_init_from_file(src_path.c_str(), { /*no_alloc*/ true, &meta });
    if (!g) die("cannot read GGUF ", src_path);
    const size_t data_off = gguf_get_data_offset(g);

    // find blk.N.ffn_{gate,up,down}_exps.weight
    const std::regex exps(R"(^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$)");
    std::map<int, layer_info> layers;
    std::vector<int64_t> core_ids;
    int64_t n_expert = 0;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        std::cmatch m;
        if (!std::regex_match(name, m, exps)) {
            // per-expert biases (gpt-oss) are small and stay resident; fused weights are not supported
            if (strstr(name, "_exps") && !strstr(name, ".bias")) die("unsupported expert tensor (fused?): ", name);
            core_ids.push_back(i);
            continue;
        }
        const ggml_tensor * t = ggml_get_tensor(meta, name);
        if (ggml_n_dims(t) != 3) die("expert tensor is not 3-D: ", name);
        if (n_expert == 0) n_expert = t->ne[2];
        if (t->ne[2] != n_expert) die("expert count differs: ", name);
        if (t->nb[2] * t->ne[2] != ggml_nbytes(t)) die("expert tensor not expert-major: ", name);
        const int il = std::stoi(m[1].str());
        const int k = m[2] == "gate" ? 0 : m[2] == "up" ? 1 : 2;
        layer_info & L = layers[il];
        L.il = il;
        L.parts[k] = { i, t->type, t->ne[0], t->ne[1], t->nb[2], 0 };
    }
    if (layers.empty()) die("no blk.N.ffn_{gate,up,down}_exps.weight tensors found");

    // layout of an experts file; returns its size
    auto layout = [&](std::map<int, layer_info> & ls) {
        size_t cur = 0;
        for (auto & [il, L] : ls) {
            size_t o = 0;
            for (int k = 0; k < 3; ++k) {
                if (L.parts[k].bytes == 0) die("layer is missing an expert tensor: blk.", std::to_string(il));
                L.parts[k].offset = o;
                o = pad(o + L.parts[k].bytes, PART_ALIGN);
            }
            L.stride = pad(o, BLOB_ALIGN);
            L.offset = cur;
            cur += L.stride * n_expert;
        }
        return cur;
    };
    const size_t cursor = layout(layers);

    // optional low-bit copy of every expert (read instead of the full one on a cache miss)
    std::map<int, layer_info> lo_layers = layers;
    size_t lo_bytes = 0;
    if (lowbit != GGML_TYPE_COUNT) {
        for (auto & [il, L] : lo_layers) {
            for (part & p : L.parts) {
                if (p.ne0 % ggml_blck_size(lowbit)) die("row length not a multiple of the --lowbit block size");
                p.type = lowbit;
                p.bytes = ggml_row_size(lowbit, p.ne0) * p.ne1;
            }
        }
        lo_bytes = layout(lo_layers);
    } else {
        lo_layers.clear();
    }

    fs::create_directories(out);
    std::ifstream src(src_path, std::ios::binary);
    if (!src) die("cannot open ", src_path);

    // experts.bin (kept as it is with --lowbit-only)
    if (!lowbit_only) {
        std::ofstream f(out / "experts.bin", std::ios::binary);
        std::vector<uint8_t> blob;
        for (auto & [il, L] : layers) {
            blob.resize(L.stride);
            for (int64_t e = 0; e < n_expert; ++e) {
                build_blob(src, data_off, g, L, e, blob);
                f.write((const char *) blob.data(), (std::streamsize) blob.size());
            }
            fprintf(stderr, "\rexperts: layer %d/%zu", il + 1, layers.size());
        }
        fprintf(stderr, "\n");
        if (!f) die("write failed: experts.bin");
    }

    // experts.lo.bin: dequantize each expert and requantize it, one layer at a time on all cores
    if (!lo_layers.empty()) {
        // importance matrix (zorlo imatrix): per layer and expert, squared inputs of gate/up (n_embd) and down (n_ff)
        std::vector<float> imat;
        int32_t ih[4] = {};
        if (!imatrix_path.empty()) {
            std::ifstream im(imatrix_path, std::ios::binary);
            im.read((char *) ih, sizeof(ih));
            const auto & L0 = layers.begin()->second;
            if (!im || ih[1] != n_expert || ih[2] != L0.parts[0].ne0 || ih[3] != L0.parts[2].ne0) die("imatrix does not fit this model: ", imatrix_path);
            imat.resize((size_t) ih[0] * ih[1] * (ih[2] + ih[3]));
            if (!im.read((char *) imat.data(), (std::streamsize) (imat.size() * sizeof(float)))) die("short imatrix: ", imatrix_path);
        }
        auto weights = [&](int il, int64_t e, int k) -> const float * {  // nullptr: no data, plain quantization
            if (imat.empty() || il >= ih[0]) return nullptr;
            const float * w = imat.data() + ((size_t) il * ih[1] + e) * (ih[2] + ih[3]) + (k < 2 ? 0 : ih[2]);
            const int n = k < 2 ? ih[2] : ih[3];
            return std::any_of(w, w + n, [](float v) { return v > 0; }) ? w : nullptr;  // never routed to: no data
        };
        std::ifstream hi_f(out / "experts.bin", std::ios::binary);
        std::ofstream f(out / "experts.lo.bin", std::ios::binary);
        const unsigned nt = std::max(1u, std::thread::hardware_concurrency());
        for (auto & [il, L] : layers) {
            const layer_info & LL = lo_layers.at(il);
            std::vector<uint8_t> hi(L.stride * n_expert), lo(LL.stride * n_expert, 0);
            read_at(hi_f, L.offset, hi.data(), hi.size());
            std::atomic<int64_t> next{ 0 };
            std::vector<std::thread> ts;
            for (unsigned t = 0; t < nt; ++t) {
                ts.emplace_back([&] {
                    std::vector<float> f32;
                    for (int64_t e; (e = next++) < n_expert;) {
                        for (int k = 0; k < 3; ++k) {
                            const part & p = L.parts[k], & q = LL.parts[k];
                            f32.resize((size_t) p.ne0 * p.ne1);
                            ggml_get_type_traits(p.type)->to_float(hi.data() + e * L.stride + p.offset, f32.data(), p.ne0 * p.ne1);
                            const size_t n = ggml_quantize_chunk(q.type, f32.data(), lo.data() + e * LL.stride + q.offset, 0,
                                                                 q.ne1, q.ne0, weights(il, e, k));
                            if (n != q.bytes) die("unexpected quantized size");
                        }
                    }
                });
            }
            for (auto & t : ts) t.join();
            f.write((const char *) lo.data(), (std::streamsize) lo.size());
            fprintf(stderr, "\rlow-bit experts: layer %d/%zu", il + 1, layers.size());
        }
        fprintf(stderr, "\n");
        if (!f) die("write failed: experts.lo.bin");
    }

    if (lowbit_only) {  // core.gguf, experts.bin and the manifest (same low-bit layout) stay as they are
        fprintf(stderr, "rewrote %s\n", (out / "experts.lo.bin").string().c_str());
        return 0;
    }

    // core.gguf: header via gguf, then tensor bytes copied in the order of the new context
    {
        gguf_context * c = gguf_init_empty();
        gguf_set_kv(c, g);
        for (int64_t id : core_ids) {
            gguf_add_tensor(c, ggml_get_tensor(meta, gguf_get_tensor_name(g, id)));
        }
        const fs::path cp = out / "core.gguf";
        if (!gguf_write_to_file(c, cp.string().c_str(), /*only_meta*/ true)) die("cannot write core.gguf");
        const size_t meta_size = gguf_get_meta_size(c);
        std::fstream f(cp, std::ios::binary | std::ios::in | std::ios::out);
        std::vector<char> buf;
        for (size_t j = 0; j < core_ids.size(); ++j) {
            const size_t n = gguf_get_tensor_size(g, core_ids[j]);
            buf.resize(n);
            read_at(src, data_off + gguf_get_tensor_offset(g, core_ids[j]), buf.data(), n);
            f.seekp((std::streamoff) (meta_size + gguf_get_tensor_offset(c, (int64_t) j)));
            f.write(buf.data(), (std::streamsize) n);
        }
        // pad the last tensor so the file size matches what gguf readers expect
        const int64_t last = (int64_t) core_ids.size() - 1;
        const size_t end = meta_size + gguf_get_tensor_offset(c, last) +
                           pad(gguf_get_tensor_size(c, last), gguf_get_alignment(c));
        f.seekp(0, std::ios::end);
        const size_t have = (size_t) f.tellp();
        if (have < end) {
            std::vector<char> z(end - have, 0);
            f.write(z.data(), (std::streamsize) z.size());
        }
        if (!f) die("write failed: core.gguf");
        gguf_free(c);
    }

    // manifest.json
    {
        char arch[128] = "unknown";
        const int64_t ka = gguf_find_key(g, "general.architecture");
        if (ka >= 0) snprintf(arch, sizeof(arch), "%s", gguf_get_val_str(g, ka));
        std::ofstream m(out / "manifest.json");
        m << "{\n  \"format\": \"zorlo-pack-1\",\n  \"source\": \"" << fs::path(src_path).filename().string()
          << "\",\n  \"arch\": \"" << arch << "\",\n  \"n_expert\": " << n_expert
          << ",\n  \"blob_align\": " << BLOB_ALIGN << ",\n  \"core\": \"core.gguf\",\n  \"experts\": \"experts.bin\""
          << ",\n  \"experts_bytes\": " << cursor << ",\n  \"layers\": [\n";
        auto write_layers = [&](const std::map<int, layer_info> & ls) {
            size_t n = 0;
            for (auto & [il, L] : ls) {
                m << "    {\"layer\": " << il << ", \"offset\": " << L.offset << ", \"stride\": " << L.stride << ", \"parts\": [";
                for (int k = 0; k < 3; ++k) {
                    const part & p = L.parts[k];
                    m << (k ? ", " : "") << "{\"kind\": \"" << KINDS[k] << "\", \"type\": \"" << ggml_type_name(p.type)
                      << "\", \"type_id\": " << (int) p.type << ", \"offset\": " << p.offset << ", \"bytes\": " << p.bytes
                      << ", \"ne\": [" << p.ne0 << ", " << p.ne1 << "]}";
                }
                m << "]}" << (++n < ls.size() ? ",\n" : "\n");
            }
        };
        write_layers(layers);
        m << "  ]";
        if (!lo_layers.empty()) {
            m << ",\n  \"lowbit\": {\"experts\": \"experts.lo.bin\", \"experts_bytes\": " << lo_bytes << ", \"layers\": [\n";
            write_layers(lo_layers);
            m << "  ]}";
        }
        m << "\n}\n";
    }

    // verify: random experts read back from experts.bin must equal the source slices
    {
        std::ifstream f(out / "experts.bin", std::ios::binary);
        std::mt19937_64 rng(42);
        std::vector<int> ils;
        for (auto & [il, L] : layers) ils.push_back(il);
        std::vector<uint8_t> want, got;
        for (int i = 0; i < n_verify; ++i) {
            const layer_info & L = layers[ils[rng() % ils.size()]];
            const int64_t e = (int64_t) (rng() % (uint64_t) n_expert);
            want.resize(L.stride);
            got.resize(L.stride);
            build_blob(src, data_off, g, L, e, want);
            read_at(f, L.offset + (uint64_t) e * L.stride, got.data(), L.stride);
            if (want != got) die("verify failed at layer ", std::to_string(L.il) + " expert " + std::to_string(e));
        }
    }

    size_t core_bytes = 0;
    for (int64_t id : core_ids) core_bytes += gguf_get_tensor_size(g, id);
    const layer_info & L0 = layers.begin()->second;
    fprintf(stderr, "packed %zu layers x %" PRId64 " experts -> %s\n", layers.size(), n_expert, out.string().c_str());
    fprintf(stderr, "  core tensors : %zu, %.2f GB\n", core_ids.size(), core_bytes / 1e9);
    fprintf(stderr, "  experts.bin  : %.2f GB, blob %.2f MB (layer %d: %s/%s/%s)\n", cursor / 1e9, L0.stride / 1048576.0,
            L0.il, ggml_type_name(L0.parts[0].type), ggml_type_name(L0.parts[1].type), ggml_type_name(L0.parts[2].type));
    fprintf(stderr, "  verified %d random experts byte-for-byte\n", n_verify);
    if (!lo_layers.empty()) {
        fprintf(stderr, "  experts.lo.bin: %.2f GB, blob %.2f MB (%s)\n", lo_bytes / 1e9,
                lo_layers.begin()->second.stride / 1048576.0, ggml_type_name(lowbit));
    }
    gguf_free(g);
    ggml_free(meta);
    return 0;
}

#ifndef ZORLO_PACK_IN_CLI
int main(int argc, char ** argv) { return zorlo_pack_main(argc, argv); }
#endif
