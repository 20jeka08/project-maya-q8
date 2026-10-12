// src/program/glm_dsa_moe_bench.cpp - the GLM-5.3 routed-expert CPU path on its own: how many GB/s of expert weights
// the CPU dots once the weights sit in RAM.  No GPU.  For each thread count it runs the decode's shape - 8 experts
// of a layer, gate/up rows then down rows - over every MoE layer, with a FIXED expert set per layer (warmed first,
// so the page cache holds it) and reports ms per token-equivalent (all MoE layers) and GB/s.
//
//   glm_dsa_moe_bench --model <any shard>.gguf [--threads 8,16,24,32,48,64,96] [--iters 3] [--k 8]
#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/gguf_split.hpp"
#include "strata/core/glm_dsa_model.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace kc = strata::kernels::cpu;

struct Layer {
    kc::NativeFmt f;
    const uint8_t *gate, *up, *down;
    size_t gu_e, d_e;
    std::vector<int> ids;
};

int main(int argc, char** argv) {
    std::string model, threads_arg = "8,16,24,32,48,64,96";
    int iters = 3, k = 8;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--model") model = argv[i + 1];
        else if (a == "--threads") threads_arg = argv[i + 1];
        else if (a == "--iters") iters = std::atoi(argv[i + 1]);
        else if (a == "--k") k = std::atoi(argv[i + 1]);
    }
    std::vector<std::unique_ptr<strata::GgufFile>> files;
    for (const auto& p : strata::gguf_split_paths(model)) files.emplace_back(new strata::GgufFile(p));
    strata::core::GlmDsaGeometry g;
    std::string err;
    if (!strata::core::GlmDsaGeometry::from_gguf(*files[0], g, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    const auto find = [&](const std::string& n, const strata::GgufFile*& f) -> const strata::TensorInfo* {
        for (auto& x : files)
            if (auto* t = x->find(n)) {
                f = x.get();
                return t;
            }
        return nullptr;
    };
    std::mt19937 rng(1234);
    std::vector<Layer> layers;
    size_t bytes = 0;
    for (int il = g.dense_lead; il < g.n_layers; ++il) {
        const std::string L = "blk." + std::to_string(il) + ".";
        const strata::GgufFile *fg, *fu, *fd;
        auto* tg = find(L + "ffn_gate_exps.weight", fg);
        auto* tu = find(L + "ffn_up_exps.weight", fu);
        auto* td = find(L + "ffn_down_exps.weight", fd);
        Layer ly;
        if (!kc::native_fmt((int) tg->type, (int) td->type, g.n_embd, g.n_ff_exp, ly.f, err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        ly.gate = fg->tensor_data(*tg);
        ly.up = fu->tensor_data(*tu);
        ly.down = fd->tensor_data(*td);
        ly.gu_e = ly.f.gu_row * (size_t) g.n_ff_exp;
        ly.d_e = ly.f.d_row * (size_t) g.n_embd;
        std::vector<int> all(g.n_expert);
        for (int e = 0; e < g.n_expert; ++e) all[e] = e;
        std::shuffle(all.begin(), all.end(), rng);
        ly.ids.assign(all.begin(), all.begin() + k);
        bytes += (size_t) k * (2 * ly.gu_e + ly.d_e);
        layers.push_back(ly);
    }
    std::printf("%zu MoE layers, %d experts each, %.2f GB per token-equivalent\n", layers.size(), k, bytes / 1e9);

    std::vector<float> x(g.n_embd), ff((size_t) k * g.n_ff_exp), dn((size_t) k * g.n_embd);
    for (auto& v : x) v = std::uniform_real_distribution<float>(-1, 1)(rng);
    std::vector<uint8_t> act(kc::kNativeActBytes), hq((size_t) k * kc::kNativeHBytes);

    std::vector<int> tlist;
    for (size_t s = 0, e; s < threads_arg.size(); s = e + 1) {
        e = threads_arg.find(',', s);
        if (e == std::string::npos) e = threads_arg.size();
        tlist.push_back(std::atoi(threads_arg.substr(s, e - s).c_str()));
    }
    bool warmed = false;
    for (int nt : tlist) {
        strata::core::GlmCpuPool pool(nt);
        const auto token = [&]() {
            for (const Layer& L : layers) {
                kc::native_quant_act(L.f, x.data(), act.data());
                const void* a[1] = {act.data()};
                constexpr int RG = 64, RD = 128;
                const int gc = g.n_ff_exp / RG, dc = g.n_embd / RD;
                pool.run(k * gc, [&](int t) {
                    const int i = t / gc, c = t % gc, e = L.ids[i];
                    float* o[1] = {ff.data() + (size_t) i * g.n_ff_exp};
                    kc::native_gu_rows_split(L.f, L.gate + L.gu_e * e, L.up + L.gu_e * e, a, 1, o, c * RG, (c + 1) * RG);
                });
                for (int i = 0; i < k; ++i)
                    kc::native_quant_h(L.f, ff.data() + (size_t) i * g.n_ff_exp, hq.data() + (size_t) i * kc::kNativeHBytes);
                pool.run(k * dc, [&](int t) {
                    const int i = t / dc, c = t % dc, e = L.ids[i];
                    const void* h[1] = {hq.data() + (size_t) i * kc::kNativeHBytes};
                    float* o[1] = {dn.data() + (size_t) i * g.n_embd};
                    kc::native_down_rows_split(L.f, L.down + L.d_e * e, h, 1, o, c * RD, (c + 1) * RD);
                });
            }
        };
        if (!warmed) {
            const auto t0 = std::chrono::steady_clock::now();
            token();
            std::printf("warm-up pass (page cache fill): %.0f ms\n",
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            warmed = true;
        }
        double best = 1e30;
        for (int it = 0; it < iters; ++it) {
            const auto t0 = std::chrono::steady_clock::now();
            token();
            best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        std::printf("threads %3d: %7.1f ms per token, %6.1f GB/s\n", nt, best, bytes / 1e6 / best);
        std::fflush(stdout);
    }
    return 0;
}
