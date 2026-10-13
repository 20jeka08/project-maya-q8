// src/core/glm_dsa_gpu_moe.cu - the batched path's routed experts on the GPU (a long prompt's chunks).
//
// On the CPU a chunk's experts cost ~37 ms a token: every chunk of 512+ tokens routes to nearly all 256 experts of
// every layer, and the CPU multiplies them at its int8 rate.  The GPU multiplies them ~50x faster (llama.cpp's MMQ,
// strata/prefill/moe_mmq.hpp) - what it pays instead is bringing the weights over PCIe, ~435 GB a chunk whatever its
// length.  So the chunk is made long (STRATA_GLM_CHUNK, 2048 when the GPU experts are on) and the transfer is made cheap:
//   * the expert tensors of the shards' memory maps are registered with CUDA (cudaHostRegister, read-only) once and
//     kept: the DMA engine reads the page cache directly, no staging copy.  Measured (glm_dsa_pcie_bench, EPYC 7K62,
//     PCIe 4.0 x16): 28.4 GB/s into the RTX 5090 once the pages are locked (~9 GB/s the first time they are touched);
//   * a layer's experts go up in groups (STRATA_GLM_GPU_GROUP, 16), double buffered: group g+1's copy runs on its
//     own stream while group g multiplies;
//   * each expert's rows: activations gathered + q8_1 (mmq::quantize), gate/up (one MMQ product over [gate | up]),
//     SwiGLU, q8_1, down (one MMQ product) written straight into the (token, slot) row; a last kernel sums each
//     token's slots in slot order with the router's weights.
#include "strata/core/glm_dsa_model.hpp"

#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <numeric>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

bool gm_sync() {
    static const bool on = std::getenv("STRATA_GLM_GPU_SYNC") && std::atoi(std::getenv("STRATA_GLM_GPU_SYNC"));
    return on;
}

}  // namespace

bool GlmDsaModel::gpu_moe_enabled() const {
    static const bool on = !(std::getenv("STRATA_GLM_GPU_MOE") && std::atoi(std::getenv("STRATA_GLM_GPU_MOE")) == 0) &&
                           strata::prefill::mmq::built();
    return on;
}

void GlmDsaModel::gpu_moe_release() {
    for (void* b : gm_regs_) cudaHostUnregister(b);
    gm_regs_.clear();
    delete (strata::prefill::mmq::Context*) gm_ctx_;
    gm_ctx_ = nullptr;
    for (int b = 0; b < 2; ++b) {
        if (gm_ready_[b]) cudaEventDestroy((cudaEvent_t) gm_ready_[b]);
        if (gm_free_[b]) cudaEventDestroy((cudaEvent_t) gm_free_[b]);
    }
    if (gm_copy_) cudaStreamDestroy((cudaStream_t) gm_copy_);
}

bool GlmDsaModel::gpu_moe_alloc(int Tmax, std::string& err) {
    namespace mmq = strata::prefill::mmq;
    const GlmDsaGeometry& g = g_;
    if (gm_tmax_ >= Tmax) return true;
    if (gm_tmax_ == 0) {
        gm_group_ = 16;
        if (const char* s = std::getenv("STRATA_GLM_GPU_GROUP")) gm_group_ = std::max(1, std::atoi(s));
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
        stride = (stride + unit - 1) / unit * unit;
        gm_stride_ = stride;
        for (int b = 0; b < 2; ++b) {
            GM_CUDA(cudaMalloc(&gm_w_[b], (size_t) gm_group_ * stride));
            allocs_.push_back(gm_w_[b]);
            cudaEvent_t e1, e2;
            GM_CUDA(cudaEventCreateWithFlags(&e1, cudaEventDisableTiming));
            GM_CUDA(cudaEventCreateWithFlags(&e2, cudaEventDisableTiming));
            gm_ready_[b] = e1;
            gm_free_[b] = e2;
        }
        cudaStream_t cs;
        GM_CUDA(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
        gm_copy_ = cs;
        gm_ctx_ = new mmq::Context();
        GM_CUDA(cudaMalloc(&gm_bounds_, (size_t) (g.n_expert + 1) * sizeof(int)));
        allocs_.push_back(gm_bounds_);
    }
    const int64_t rows = (int64_t) Tmax * g.n_exp_used;
    const int64_t F = g.n_ff_exp, E = g.n_embd;
    // the per-row buffers, for the most rows one group can have (all of the chunk's)
    void* p = nullptr;
    const size_t need = rows * sizeof(int) * 3 + mmq::q8_bytes(rows, E) + mmq::q8_bytes(rows, F) +
                        (size_t) rows * (2 * F + F + E) * sizeof(float) + 1024;
    GM_CUDA(cudaMalloc(&p, need));
    allocs_.push_back(p);
    uint8_t* c = (uint8_t*) p;
    const auto take = [&](size_t n) {
        uint8_t* r = c;
        c += (n + 255) & ~(size_t) 255;
        return r;
    };
    gm_row_tok_ = (int*) take(rows * sizeof(int));
    gm_row_slot_ = (int*) take(rows * sizeof(int));
    gm_iota_ = (int*) take(rows * sizeof(int));
    gm_xq_ = take(mmq::q8_bytes(rows, E));
    gm_hq_ = take(mmq::q8_bytes(rows, F));
    gm_gu_ = (float*) take((size_t) rows * 2 * F * sizeof(float));
    gm_h_ = (float*) take((size_t) rows * F * sizeof(float));
    gm_dn_ = (float*) take((size_t) rows * E * sizeof(float));
    mmq::iota(gm_iota_, rows, nullptr);
    gm_tmax_ = Tmax;
    std::fprintf(stderr, "glm-dsa: GPU experts for long prompts: groups of %d experts, %.2f GB of buffers\n", gm_group_,
                 (2.0 * gm_group_ * gm_stride_ + need) / 1e9);
    return true;
}

bool GlmDsaModel::moe_gpu_batch(int il, int T, const float* XN, const float* RW, const int* ids, float* out_dev,
                                std::string& err) {
    namespace mmq = strata::prefill::mmq;
    const GlmDsaGeometry& g = g_;
    if (!gpu_moe_alloc(T, err)) return false;
    const ExpertLayer& L = experts_[(size_t) il];
    const int K = g.n_exp_used, E = g.n_embd, F = g.n_ff_exp, NE = g.n_expert;
    cudaStream_t s = nullptr;   // the batched path's stream (legacy)
    cudaStream_t cs = (cudaStream_t) gm_copy_;

    // the shards' mappings registered once (read-only), kept: the copies read the page cache directly.  Whole
    // shards, not tensors - neighbouring tensors share pages, and a page cannot be registered twice.  The pages
    // are locked as the copies first touch them.
    if (!gm_registered_) {
        gm_registered_ = true;
        for (const auto& fp : files_) {
            if (fp->tensors().empty()) continue;
            const strata::TensorInfo& t0 = fp->tensors()[0];
            uint8_t* base = (uint8_t*) fp->tensor_data(t0) - fp->data_start() - t0.offset;
            if (cudaHostRegister(base, fp->file_size(), cudaHostRegisterReadOnly | cudaHostRegisterPortable) == cudaSuccess)
                gm_regs_.push_back(base);
            else {
                cudaGetLastError();
                std::fprintf(stderr, "glm-dsa: a shard could not be registered - its experts are copied from pageable memory");
                std::fputc(10, stderr);
            }
        }
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
    GM_CUDA(cudaMemcpy(gm_bounds_, bounds.data(), bounds.size() * sizeof(int), cudaMemcpyHostToDevice));
    GM_CUDA(cudaMemcpy(gm_row_tok_, row_tok.data(), row_tok.size() * sizeof(int), cudaMemcpyHostToDevice));
    GM_CUDA(cudaMemcpy(gm_row_slot_, row_slot.data(), row_slot.size() * sizeof(int), cudaMemcpyHostToDevice));

    // the groups: runs of up to gm_group_ experts that have rows
    std::vector<std::pair<int, int>> groups;   // [e0, e1)
    for (int e = 0; e < NE;) {
        while (e < NE && bounds[(size_t) e + 1] == bounds[(size_t) e]) ++e;
        if (e >= NE) break;
        // at most gm_group_ consecutive ids: each id has its slot in the buffer (an id without rows keeps its slot,
        // unread), so the span - not the count of experts with rows - is what the buffer must hold
        const int e0 = e;
        e = std::min(NE, e0 + gm_group_);
        groups.push_back({e0, e});
    }
    const size_t stride = gm_stride_, guh = L.gu_expert, db = L.d_expert;
    const size_t qblk = mmq::q8_block_bytes();
    mmq::Context* ctx = (mmq::Context*) gm_ctx_;
    const auto issue_copy = [&](size_t gi) -> bool {
        const int b = (int) (gi & 1);
        uint8_t* buf = (uint8_t*) gm_w_[b];
        GM_CUDA(cudaStreamWaitEvent(cs, (cudaEvent_t) gm_free_[b], 0));   // its previous group has multiplied
        int slot = 0;
        for (int e = groups[gi].first; e < groups[gi].second; ++e, ++slot) {
            if (bounds[(size_t) e + 1] == bounds[(size_t) e]) continue;   // no rows: its slot stays unread
            uint8_t* dst = buf + (size_t) slot * stride;
            GM_CUDA(cudaMemcpyAsync(dst, L.gate + guh * (size_t) e, guh, cudaMemcpyHostToDevice, cs));
            GM_CUDA(cudaMemcpyAsync(dst + guh, L.up + guh * (size_t) e, guh, cudaMemcpyHostToDevice, cs));
            GM_CUDA(cudaMemcpyAsync(dst + 2 * guh, L.down + db * (size_t) e, db, cudaMemcpyHostToDevice, cs));
        }
        GM_CUDA(cudaEventRecord((cudaEvent_t) gm_ready_[b], cs));
        if (gm_sync()) GM_CUDA(cudaDeviceSynchronize());
        return true;
    };
    // both buffers start free
    for (int b = 0; b < 2; ++b) GM_CUDA(cudaEventRecord((cudaEvent_t) gm_free_[b], s));
    if (!groups.empty() && !issue_copy(0)) return false;
    for (size_t gi = 0; gi < groups.size(); ++gi) {
        if (gi + 1 < groups.size() && !issue_copy(gi + 1)) return false;
        const int b = (int) (gi & 1);
        GM_CUDA(cudaStreamWaitEvent(s, (cudaEvent_t) gm_ready_[b], 0));
        const int e0 = groups[gi].first, e1 = groups[gi].second;
        const int a = bounds[(size_t) e0], nrows = bounds[(size_t) e1] - a;
        int max_rows = 0;
        for (int e = e0; e < e1; ++e) max_rows = std::max(max_rows, bounds[(size_t) e + 1] - bounds[(size_t) e]);
        const uint8_t* buf = (const uint8_t*) gm_w_[b];
        // stride is the same for every layer's buffer: the experts sit `stride` apart, the down rows 2*guh in
        mmq::quantize(XN, gm_row_tok_ + a, gm_xq_, L.gu_type, E, E, nrows, s);
        mmq::Product gu;
        gu.w = buf;
        gu.type = L.gu_type;
        gu.w_rows = 2 * F;
        gu.w_cols = E;
        gu.expert_bytes = stride;
        gu.n = e1 - e0;
        gu.xq = (const uint8_t*) gm_xq_ - (size_t) a * qblk;
        gu.bounds = gm_bounds_ + e0;
        gu.ids = gm_iota_;
        gu.total_rows = nrows;
        gu.max_rows = max_rows;
        gu.dst = gm_gu_ - (size_t) a * 2 * F;
        gu.ld_dst = 2 * F;
        ctx->run(gu, s);
        mmq::swiglu(gm_gu_, gm_h_, nrows, F, false, s);
        mmq::quantize(gm_h_, nullptr, gm_hq_, L.d_type, F, F, nrows, s);
        mmq::Product dn;
        dn.w = buf + 2 * guh;
        dn.type = L.d_type;
        dn.w_rows = E;
        dn.w_cols = F;
        dn.expert_bytes = stride;
        dn.n = e1 - e0;
        dn.xq = (const uint8_t*) gm_hq_ - (size_t) a * qblk;
        dn.bounds = gm_bounds_ + e0;
        dn.ids = gm_row_slot_;   // each row's result goes to its (token, slot) row
        dn.total_rows = nrows;
        dn.max_rows = max_rows;
        dn.dst = gm_dn_;
        dn.ld_dst = E;
        ctx->run(dn, s);
        GM_CUDA(cudaEventRecord((cudaEvent_t) gm_free_[b], s));
        if (gm_sync()) GM_CUDA(cudaDeviceSynchronize());
    }
    const int64_t n_out = (int64_t) T * E;
    const unsigned n_blocks = (unsigned) ((n_out + 255) / 256);
    moe_combine_kernel<<<n_blocks, 256, 0, s>>>(gm_dn_, RW, out_dev, T, K, E);
    GM_CUDA(cudaGetLastError());
    return true;
}

}  // namespace strata::core
