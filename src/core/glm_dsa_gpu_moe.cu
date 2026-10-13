// src/core/glm_dsa_gpu_moe.cu - the batched path's routed experts on the GPUs (a long prompt's chunks).
//
// On the CPU a chunk's experts cost ~37 ms a token: every chunk of 512+ tokens routes to nearly all 256 experts of
// every layer, and the CPU multiplies them at its int8 rate.  The GPU multiplies them ~50x faster (llama.cpp's MMQ,
// strata/prefill/moe_mmq.hpp) - what it pays instead is bringing the weights over PCIe, ~435 GB a chunk whatever its
// length.  So the chunk is made long (STRATA_GLM_CHUNK, 2048 when the GPU experts are on) and the transfer is made
// cheap and wide:
//   * the shards' memory maps are registered with CUDA (cudaHostRegister, read-only, portable) once and kept: the DMA
//     engines read the page cache directly, no staging copy - up to the driver's cap (~224 GB here); the rest is
//     copied from pageable memory.  Measured (glm_dsa_pcie_bench, EPYC 7K62, PCIe 4.0 x16 on both cards): 28.4 GB/s
//     into the RTX 5090, 26.1 into the RTX 3090, 51.8 into both at once (~9 GB/s while pages first lock; pageable
//     14.3 / 12.5);
//   * every GPU is a LANE with its own PCIe link (STRATA_GLM_GPU_LANES: "auto" = every device, or a list): a layer's
//     expert groups (STRATA_GLM_GPU_GROUP, 16) alternate between the lanes, each lane fetching and multiplying its
//     own on its own host thread (a pageable copy blocks its caller), double buffered against its copy stream;
//   * each expert's rows: activations gathered + q8_1 (mmq::quantize), gate/up (one MMQ product over [gate | up]),
//     SwiGLU, q8_1, down (one MMQ product) written straight into the (token, slot) row; per lane a kernel sums its
//     slots per token in slot order with the router's weights (a lane's other slots are zero); another lane's sum
//     comes to the main device by cudaMemcpyPeer and is added there.
// Windows (WDDM) caps the system memory the GPUs may map: registered shards count against it, and past ~80 GB with
// two cards the 3090's MMQ allocations failed ("out of memory"), so only STRATA_GLM_GPU_REG_GB (80) is registered.
// Measured, 3268-token prompt, both cards: 46.7 s with 79 GB registered, 47.8 s with 98 GB (the rest of the
// experts, pageable, set the pace); one card: 53.6 s; the CPU path 117 s.
#include "strata/core/glm_dsa_model.hpp"

#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <sstream>
#include <thread>

namespace strata::core {
namespace {

#define GM_CUDA(call)                                                                                   \
    do {                                                                                                \
        const cudaError_t e_ = (call);                                                                  \
        if (e_ != cudaSuccess) {                                                                        \
            err = std::string("cuda: ") + #call + ": " + cudaGetErrorString(e_);                        \
            return false;                                                                               \
        }                                                                                               \
    } while (0)

/// out[t, :] = sum_k rw[t, k] * dn[(t*k_used + k), :] - in slot order, as the CPU path sums them.
__global__ void moe_combine_kernel(const float* __restrict__ dn, const float* __restrict__ rw, float* __restrict__ out,
                                   int T, int k_used, int E) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) T * E) return;
    const int t = (int) (i / E), e = (int) (i % E);
    float acc = 0.0f;
    for (int k = 0; k < k_used; ++k) acc += rw[t * k_used + k] * dn[((size_t) t * k_used + k) * E + e];
    out[i] = acc;
}

__global__ void add_into_kernel(float* __restrict__ y, const float* __restrict__ a, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a[i];
}

}  // namespace

/// One GPU's share of the experts: its streams, double buffer, MMQ context and row buffers.
struct GmLane {
    int dev = 0;
    bool main = false;              ///< the model's own device (its inputs and output are already there)
    cudaStream_t ks = nullptr;      ///< compute (the main lane: the legacy stream the batched path runs on)
    cudaStream_t cs = nullptr;      ///< copies
    cudaEvent_t ready[2] = {}, free_[2] = {};
    void* w[2] = {};
    strata::prefill::mmq::Context* ctx = nullptr;
    int *bounds = nullptr, *row_tok = nullptr, *row_slot = nullptr, *iota = nullptr;
    void *xq = nullptr, *hq = nullptr;
    float *gu = nullptr, *h = nullptr, *dn = nullptr;
    float *x = nullptr, *rw = nullptr, *out = nullptr;   ///< a second lane's own copies of the inputs / its sum
    std::vector<void*> allocs;
};

struct GmState {
    std::vector<GmLane> lanes;
    int tmax = 0, group = 16;
    size_t stride = 0;
    std::vector<float*> sums;       ///< every other lane's sum, brought to the main device
};

bool GlmDsaModel::gpu_moe_enabled() const {
    static const bool on = !(std::getenv("STRATA_GLM_GPU_MOE") && std::atoi(std::getenv("STRATA_GLM_GPU_MOE")) == 0) &&
                           strata::prefill::mmq::built();
    return on;
}

void GlmDsaModel::gpu_moe_release() {
    for (void* b : gm_regs_) cudaHostUnregister(b);
    gm_regs_.clear();
    GmState* st = (GmState*) gm_;
    if (!st) return;
    for (GmLane& L : st->lanes) {
        cudaSetDevice(L.dev);
        delete L.ctx;
        for (int b = 0; b < 2; ++b) {
            if (L.ready[b]) cudaEventDestroy(L.ready[b]);
            if (L.free_[b]) cudaEventDestroy(L.free_[b]);
        }
        if (L.cs) cudaStreamDestroy(L.cs);
            if (L.ks && !L.main) cudaStreamDestroy(L.ks);
        for (void* p : L.allocs) cudaFree(p);
    }
    cudaSetDevice(dev_);
    for (float* q : st->sums) cudaFree(q);
    delete st;
    gm_ = nullptr;
    cudaSetDevice(dev_);
}

bool GlmDsaModel::gpu_moe_alloc(int Tmax, std::string& err) {
    namespace mmq = strata::prefill::mmq;
    const GlmDsaGeometry& g = g_;
    GmState* st = (GmState*) gm_;
    if (st && st->tmax >= Tmax) return true;
    const int64_t rows = (int64_t) Tmax * g.n_exp_used;
    const int64_t F = g.n_ff_exp, E = g.n_embd;
    if (!st) {
        st = new GmState();
        gm_ = st;
        if (const char* s = std::getenv("STRATA_GLM_GPU_GROUP")) st->group = std::max(1, std::atoi(s));
        size_t stride = 0, unit = 256;
        for (const ExpertLayer& L : experts_)
            if (L.fmt) {
                if (!mmq::supported(L.gu_type) || !mmq::supported(L.d_type)) {
                    err = "glm-dsa: MMQ has no kernel for this layer's expert types";
                    return false;
                }
                stride = std::max(stride, 2 * L.gu_expert + L.d_expert);
                unit = std::lcm(unit, (size_t) ggml_type_size((ggml_type) L.gu_type));
                unit = std::lcm(unit, (size_t) ggml_type_size((ggml_type) L.d_type));
            }
        // MMQ steps from expert to expert in whole quant blocks: the stride must be a multiple of every block size the
        // experts use (Q4_K 144, Q5_K 176, Q6_K 210 bytes here) - 22,806,528 bytes is not, and every expert after the
        // first of a launch was read from a shifted offset (NaN).  Rounded up to that common multiple (and 256).
        st->stride = (stride + unit - 1) / unit * unit;
        // the lanes: the model's device first, then the others (STRATA_GLM_GPU_LANES: "auto" or a device list)
        int n_dev = 0;
        cudaGetDeviceCount(&n_dev);
        std::vector<int> devs = {dev_};
        const char* ls = std::getenv("STRATA_GLM_GPU_LANES");
        if (ls && std::string(ls) != "auto") {
            devs.clear();
            std::stringstream ss(ls);
            std::string tok;
            while (std::getline(ss, tok, ','))
                if (!tok.empty()) devs.push_back(std::atoi(tok.c_str()));
            if (devs.empty() || devs[0] != dev_) devs.insert(devs.begin(), dev_);
        } else {
            for (int d = 0; d < n_dev; ++d)
                if (d != dev_) devs.push_back(d);
        }
        for (int d : devs) {
            GmLane L;
            L.dev = d;
            L.main = d == dev_;
            st->lanes.push_back(L);
        }
    }
    // (re)size every lane for Tmax tokens
    for (GmLane& L : st->lanes) {
        GM_CUDA(cudaSetDevice(L.dev));
        for (void* p : L.allocs) cudaFree(p);
        L.allocs.clear();
        const auto alloc = [&](size_t n) -> void* {
            void* p = nullptr;
            if (cudaMalloc(&p, (n + 255) & ~(size_t) 255) != cudaSuccess) return nullptr;
            L.allocs.push_back(p);
            return p;
        };
        if (!L.ctx) {
            L.ctx = new mmq::Context();
            GM_CUDA(cudaStreamCreateWithFlags(&L.cs, cudaStreamNonBlocking));
            if (L.main) L.ks = nullptr;   // the legacy stream: ordered with the batched path's own work
            else GM_CUDA(cudaStreamCreateWithFlags(&L.ks, cudaStreamNonBlocking));
            for (int b = 0; b < 2; ++b) {
                GM_CUDA(cudaEventCreateWithFlags(&L.ready[b], cudaEventDisableTiming));
                GM_CUDA(cudaEventCreateWithFlags(&L.free_[b], cudaEventDisableTiming));
            }
        }
        bool ok = true;
        for (int b = 0; b < 2; ++b) ok = ok && (L.w[b] = alloc((size_t) st->group * st->stride)) != nullptr;
        ok = ok && (L.bounds = (int*) alloc((size_t) (g.n_expert + 1) * sizeof(int))) != nullptr;
        ok = ok && (L.row_tok = (int*) alloc(rows * sizeof(int))) != nullptr;
        ok = ok && (L.row_slot = (int*) alloc(rows * sizeof(int))) != nullptr;
        ok = ok && (L.iota = (int*) alloc(rows * sizeof(int))) != nullptr;
        ok = ok && (L.xq = alloc(mmq::q8_bytes(rows, E))) != nullptr;
        ok = ok && (L.hq = alloc(mmq::q8_bytes(rows, F))) != nullptr;
        ok = ok && (L.gu = (float*) alloc((size_t) rows * 2 * F * sizeof(float))) != nullptr;
        ok = ok && (L.h = (float*) alloc((size_t) rows * F * sizeof(float))) != nullptr;
        ok = ok && (L.dn = (float*) alloc((size_t) rows * E * sizeof(float))) != nullptr;
        if (!L.main) {
            ok = ok && (L.x = (float*) alloc((size_t) Tmax * E * sizeof(float))) != nullptr;
            ok = ok && (L.rw = (float*) alloc((size_t) Tmax * g.n_exp_used * sizeof(float))) != nullptr;
            ok = ok && (L.out = (float*) alloc((size_t) Tmax * E * sizeof(float))) != nullptr;
        }
        if (!ok) {
            err = "glm-dsa: out of device memory for the GPU experts on device " + std::to_string(L.dev) +
                  " (STRATA_GLM_CHUNK smaller, or STRATA_GLM_GPU_MOE=0)";
            return false;
        }
        mmq::iota(L.iota, rows, L.ks);
        GM_CUDA(cudaStreamSynchronize(L.ks));
    }
    GM_CUDA(cudaSetDevice(dev_));
    // every other lane's sum comes to the main device (cudaMemcpyPeer: the driver routes it, no pinned buffer of
    // ours - with the shards registered, a pinned buffer's copies failed "out of memory")
    for (float* p : st->sums) cudaFree(p);
    st->sums.assign(st->lanes.size() > 1 ? st->lanes.size() - 1 : 0, nullptr);
    for (float*& p : st->sums) GM_CUDA(cudaMalloc(&p, (size_t) Tmax * E * sizeof(float)));
    st->tmax = Tmax;
    std::string names;
    for (const GmLane& L : st->lanes) names += (names.empty() ? "" : ", ") + std::to_string(L.dev);
    std::fprintf(stderr, "glm-dsa: GPU experts for long prompts on device(s) %s: groups of %d experts\n", names.c_str(),
                 st->group);
    return true;
}

bool GlmDsaModel::moe_gpu_batch(int il, int T, const float* XN, const float* RW, const int* ids, float* out_dev,
                                std::string& err) {
    namespace mmq = strata::prefill::mmq;
    const GlmDsaGeometry& g = g_;
    if (!gpu_moe_alloc(T, err)) return false;
    GmState* st = (GmState*) gm_;
    const ExpertLayer& L = experts_[(size_t) il];
    const int K = g.n_exp_used, E = g.n_embd, F = g.n_ff_exp, NE = g.n_expert;
    const int n_lanes = (int) st->lanes.size();

    // the shards' mappings registered once (read-only, portable), kept: the copies read the page cache directly.
    // Whole shards, not tensors - neighbouring tensors share pages, and a page cannot be registered twice.  The pages
    // are locked as the copies first touch them; the driver refuses past ~224 GB (those shards stay pageable).
    if (!gm_registered_) {
        gm_registered_ = true;
        // a budget (STRATA_GLM_GPU_REG_GB, 80) well under what WDDM lets the GPUs map: near it, other mappings fail
        double budget = 80;
        if (const char* s = std::getenv("STRATA_GLM_GPU_REG_GB")) budget = std::atof(s);
        double used = 0;
        int left = 0;
        for (const auto& fp : files_) {
            if (fp->tensors().empty()) continue;
            const strata::TensorInfo& t0 = fp->tensors()[0];
            uint8_t* base = (uint8_t*) fp->tensor_data(t0) - fp->data_start() - t0.offset;
            const double gb = fp->file_size() / 1e9;
            if (used + gb <= budget &&
                cudaHostRegister(base, fp->file_size(), cudaHostRegisterReadOnly | cudaHostRegisterPortable) ==
                    cudaSuccess) {
                gm_regs_.push_back(base);
                used += gb;
            } else {
                cudaGetLastError();
                ++left;
            }
        }
        std::fprintf(stderr, "glm-dsa: %zu shard mappings (%.0f GB) registered for DMA, %d left pageable\n",
                     gm_regs_.size(), used, left);
    }

    // the chunk's (token, slot) rows sorted by expert
    std::vector<int> bounds((size_t) NE + 1, 0), row_tok((size_t) T * K), row_slot((size_t) T * K);
    for (int j = 0; j < T * K; ++j) ++bounds[(size_t) ids[j] + 1];
    for (int e = 0; e < NE; ++e) bounds[(size_t) e + 1] += bounds[(size_t) e];
    {
        std::vector<int> fill(bounds.begin(), bounds.end() - 1);
        for (int j = 0; j < T * K; ++j) {
            const int r = fill[(size_t) ids[j]]++;
            row_tok[(size_t) r] = j / K;
            row_slot[(size_t) r] = j;
        }
    }
    // the groups: at most `group` consecutive ids, starting at an id with rows; dealt to the lanes in turn
    std::vector<std::pair<int, int>> groups;
    for (int e = 0; e < NE;) {
        while (e < NE && bounds[(size_t) e + 1] == bounds[(size_t) e]) ++e;
        if (e >= NE) break;
        const int e0 = e;
        e = std::min(NE, e0 + st->group);
        groups.push_back({e0, e});
    }
    const size_t stride = st->stride, guh = L.gu_expert, db = L.d_expert;
    const size_t qblk = mmq::q8_block_bytes();
    std::vector<std::string> errs((size_t) n_lanes);
    const auto run_lane = [&](int li) -> bool {
        std::string& err = errs[(size_t) li];
        GmLane& ln = st->lanes[(size_t) li];
        GM_CUDA(cudaSetDevice(ln.dev));
        cudaStream_t s = ln.ks, cs = ln.cs;
        const float* x = ln.main ? XN : ln.x;
        const float* rw = ln.main ? RW : ln.rw;
        float* out = ln.main ? out_dev : ln.out;
        if (!ln.main) {   // the activations and the router weights from the main device
            GM_CUDA(cudaMemcpyPeerAsync(ln.x, ln.dev, XN, dev_, (size_t) T * E * sizeof(float), s));
            GM_CUDA(cudaMemcpyPeerAsync(ln.rw, ln.dev, RW, dev_, (size_t) T * K * sizeof(float), s));
        }
        GM_CUDA(cudaMemcpyAsync(ln.bounds, bounds.data(), bounds.size() * sizeof(int), cudaMemcpyHostToDevice, s));
        GM_CUDA(cudaMemcpyAsync(ln.row_tok, row_tok.data(), row_tok.size() * sizeof(int), cudaMemcpyHostToDevice, s));
        GM_CUDA(cudaMemcpyAsync(ln.row_slot, row_slot.data(), row_slot.size() * sizeof(int), cudaMemcpyHostToDevice, s));
        if (n_lanes > 1)   // the slots another lane multiplies stay zero in this lane's sum
            GM_CUDA(cudaMemsetAsync(ln.dn, 0, (size_t) T * K * E * sizeof(float), s));
        std::vector<size_t> mine;
        for (size_t gi = (size_t) li; gi < groups.size(); gi += (size_t) n_lanes) mine.push_back(gi);
        const auto issue_copy = [&](size_t k) -> bool {
            const int b = (int) (k & 1);
            uint8_t* buf = (uint8_t*) ln.w[b];
            GM_CUDA(cudaStreamWaitEvent(cs, ln.free_[b], 0));   // its previous group has multiplied
            const auto& gr = groups[mine[k]];
            int slot = 0;
            for (int e = gr.first; e < gr.second; ++e, ++slot) {
                if (bounds[(size_t) e + 1] == bounds[(size_t) e]) continue;   // no rows: its slot stays unread
                uint8_t* dst = buf + (size_t) slot * stride;
                GM_CUDA(cudaMemcpyAsync(dst, L.gate + guh * (size_t) e, guh, cudaMemcpyHostToDevice, cs));
                GM_CUDA(cudaMemcpyAsync(dst + guh, L.up + guh * (size_t) e, guh, cudaMemcpyHostToDevice, cs));
                GM_CUDA(cudaMemcpyAsync(dst + 2 * guh, L.down + db * (size_t) e, db, cudaMemcpyHostToDevice, cs));
            }
            GM_CUDA(cudaEventRecord(ln.ready[b], cs));
            return true;
        };
        for (int b = 0; b < 2; ++b) GM_CUDA(cudaEventRecord(ln.free_[b], s));
        if (!mine.empty() && !issue_copy(0)) return false;
        for (size_t k = 0; k < mine.size(); ++k) {
            if (k + 1 < mine.size() && !issue_copy(k + 1)) return false;
            const int b = (int) (k & 1);
            GM_CUDA(cudaStreamWaitEvent(s, ln.ready[b], 0));
            const int e0 = groups[mine[k]].first, e1 = groups[mine[k]].second;
            const int a = bounds[(size_t) e0], nrows = bounds[(size_t) e1] - a;
            int max_rows = 0;
            for (int e = e0; e < e1; ++e) max_rows = std::max(max_rows, bounds[(size_t) e + 1] - bounds[(size_t) e]);
            const uint8_t* buf = (const uint8_t*) ln.w[b];
            mmq::quantize(x, ln.row_tok + a, ln.xq, L.gu_type, E, E, nrows, s);
            mmq::Product gu;
            gu.w = buf;
            gu.type = L.gu_type;
            gu.w_rows = 2 * F;
            gu.w_cols = E;
            gu.expert_bytes = stride;
            gu.n = e1 - e0;
            gu.xq = (const uint8_t*) ln.xq - (size_t) a * qblk;
            gu.bounds = ln.bounds + e0;
            gu.ids = ln.iota;
            gu.total_rows = nrows;
            gu.max_rows = max_rows;
            gu.dst = ln.gu - (size_t) a * 2 * F;
            gu.ld_dst = 2 * F;
            ln.ctx->run(gu, s);
            mmq::swiglu(ln.gu, ln.h, nrows, F, false, s);
            mmq::quantize(ln.h, nullptr, ln.hq, L.d_type, F, F, nrows, s);
            mmq::Product dn;
            dn.w = buf + 2 * guh;
            dn.type = L.d_type;
            dn.w_rows = E;
            dn.w_cols = F;
            dn.expert_bytes = stride;
            dn.n = e1 - e0;
            dn.xq = (const uint8_t*) ln.hq - (size_t) a * qblk;
            dn.bounds = ln.bounds + e0;
            dn.ids = ln.row_slot;   // each row's result goes to its (token, slot) row
            dn.total_rows = nrows;
            dn.max_rows = max_rows;
            dn.dst = ln.dn;
            dn.ld_dst = E;
            ln.ctx->run(dn, s);
            GM_CUDA(cudaEventRecord(ln.free_[b], s));
        }
        const int64_t n_out = (int64_t) T * E;
        const unsigned n_blocks = (unsigned) ((n_out + 255) / 256);
        moe_combine_kernel<<<n_blocks, 256, 0, s>>>(ln.dn, rw, out, T, K, E);
        GM_CUDA(cudaGetLastError());
        if (!ln.main)   // this lane's sum to the main device
            GM_CUDA(cudaMemcpyPeerAsync(st->sums[(size_t) li - 1], dev_, out, ln.dev, (size_t) n_out * sizeof(float), s));
        GM_CUDA(cudaStreamSynchronize(s));
        return true;
    };
    bool ok = true;
    if (n_lanes == 1) {
        ok = run_lane(0);
    } else {
        std::vector<std::thread> th;
        std::vector<char> res((size_t) n_lanes, 0);
        for (int li = 0; li < n_lanes; ++li) th.emplace_back([&, li] { res[(size_t) li] = run_lane(li) ? 1 : 0; });
        for (auto& t : th) t.join();
        for (int li = 0; li < n_lanes; ++li) ok = ok && res[(size_t) li];
    }
    GM_CUDA(cudaSetDevice(dev_));
    if (!ok) {
        for (const auto& e : errs)
            if (!e.empty()) err = e;
        return false;
    }
    // the other lanes' sums into the main lane's
    const int64_t n_out = (int64_t) T * E;
    for (int li = 1; li < n_lanes; ++li) {
        add_into_kernel<<<(unsigned) ((n_out + 255) / 256), 256>>>(out_dev, st->sums[(size_t) li - 1], n_out);
        GM_CUDA(cudaGetLastError());
    }
    return true;
}

}  // namespace strata::core
