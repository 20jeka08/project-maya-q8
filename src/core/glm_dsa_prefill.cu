// src/core/glm_dsa_prefill.cu - the GLM-5.3 runner's batched path: a prompt read in chunks of T tokens.
//
// The same math as GlmDsaModel::step (glm_dsa_model.hpp), arranged for T tokens at once:
//   * every projection is one cuBLAS FP32 GEMM, Y (T x n_out) = X (T x n_in) . W^T, with the Q8_0 weight expanded
//     to F32 on the device first (one weight at a time, into one scratch);
//   * the absorbed MLA attention is single-stream (MQA): the scores of all T*n_head query rows against the cache
//     are two GEMMs (latent part + rope part), a masked softmax, and two more GEMMs for the context and v;
//   * the lightning indexer's scores are a GEMM too; a query that sees more than top_k cells keeps the top_k
//     (chosen on the host), the rest see every visible cell - the decode path's rule;
//   * the routed experts run on the CPU grouped by expert: an expert chosen by several of the chunk's tokens has
//     its rows read once and dotted against all of them (native_gu_rows_split / native_down_rows_split, nt > 1).
#include "strata/core/glm_dsa_model.hpp"

#include "glm_dsa_kv.cuh"

#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/glm_ffn.hpp"
#include "strata/kernels/router_sigmoid.hpp"

#include "ggml.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>

namespace strata::core {
namespace {

constexpr int TB = 256;
int nb(int64_t n) { return (int) ((n + TB - 1) / TB); }

#define PF_CUDA(call)                                                                                   \
    do {                                                                                                \
        const cudaError_t e_ = (call);                                                                  \
        if (e_ != cudaSuccess) {                                                                        \
            err = std::string("cuda: ") + #call + ": " + cudaGetErrorString(e_);                        \
            return false;                                                                               \
        }                                                                                               \
    } while (0)
#define PF_BLAS(call)                                                                                   \
    do {                                                                                                \
        const cublasStatus_t s_ = (call);                                                               \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                                              \
            err = std::string("cublas: ") + #call + " failed (" + std::to_string((int) s_) + ")";        \
            return false;                                                                               \
        }                                                                                               \
    } while (0)

/// Q8_0 -> F32: blocks of {f16 d, int8 q[32]}, 34 bytes.
__global__ void dequant_q8_0_kernel(const uint8_t* __restrict__ w, float* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint8_t* b = w + (i >> 5) * 34;
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    y[i] = d * (float) reinterpret_cast<const int8_t*>(b + 2)[i & 31];
}

__device__ float bsum(float v, float* red) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    float s = 0.0f;
    for (int i = 0; i < nw; ++i) s += red[i];
    return s;
}

__device__ float bmax(float v, float* red) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    float m = -INFINITY;
    for (int i = 0; i < nw; ++i) m = fmaxf(m, red[i]);
    return m;
}

/// Row t: y = x / rms(x) * w over `dim`; rows at x + t*ldx, y + t*ldy (in place is fine).
__global__ void rms_rows_kernel(const float* x, const float* __restrict__ w, float* y, int dim, int ldx, int ldy,
                                float eps) {
    __shared__ float red[32];
    const float* xr = x + (size_t) blockIdx.x * ldx;
    float* yr = y + (size_t) blockIdx.x * ldy;
    float ss = 0.0f;
    for (int e = threadIdx.x; e < dim; e += blockDim.x) ss += xr[e] * xr[e];
    const float inv = rsqrtf(bsum(ss, red) / (float) dim + eps);
    for (int e = threadIdx.x; e < dim; e += blockDim.x) yr[e] = xr[e] * inv * w[e];
}

__global__ void ln_rows_kernel(float* x, const float* __restrict__ w, const float* __restrict__ b, int dim, int ld,
                               float eps) {
    __shared__ float red[32];
    float* xr = x + (size_t) blockIdx.x * ld;
    float s = 0.0f;
    for (int e = threadIdx.x; e < dim; e += blockDim.x) s += xr[e];
    const float mu = bsum(s, red) / (float) dim;
    float v = 0.0f;
    for (int e = threadIdx.x; e < dim; e += blockDim.x) v += (xr[e] - mu) * (xr[e] - mu);
    const float inv = rsqrtf(bsum(v, red) / (float) dim + eps);
    __syncthreads();
    for (int e = threadIdx.x; e < dim; e += blockDim.x) xr[e] = (xr[e] - mu) * inv * w[e] + b[e];
}

/// ROPE_TYPE_NORM on T tokens x n_vec vectors: the n_rot slice at x + t*tok_stride + v*vec_stride + off, position
/// pos0 + t (the step path's rope_norm_kernel, per token).
__global__ void rope_rows_kernel(float* x, int T, int n_vec, int vec_stride, int64_t tok_stride, int off, int n_rot,
                                 int pos0, float theta_scale) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int half = n_rot / 2;
    if (idx >= (int64_t) T * n_vec * half) return;
    const int i = (int) (idx % half);
    const int v = (int) ((idx / half) % n_vec);
    const int t = (int) (idx / ((int64_t) half * n_vec));
    const float theta = (float) (pos0 + t) * powf(theta_scale, (float) i);
    const float c = cosf(theta), s = sinf(theta);
    float* p = x + (size_t) t * tok_stride + (size_t) v * vec_stride + off + 2 * i;
    const float x0 = p[0], x1 = p[1];
    p[0] = x0 * c - x1 * s;
    p[1] = x0 * s + x1 * c;
}

/// Row r of S (query t = t0 + r / n_head) over n_ctx cells: softmax over the cells it attends - s <= pos0 + t, and
/// mask[t, s] when a mask is given - with every other cell set to 0.
__global__ void attn_softmax_kernel(float* S, int n_ctx, int t0, int pos0, int n_head, const uint8_t* mask,
                                    int64_t mask_ld) {
    __shared__ float red[32];
    const int r = blockIdx.x;
    const int t = t0 + r / n_head;
    const int vis = pos0 + t;
    float* row = S + (size_t) r * n_ctx;
    const uint8_t* m = mask ? mask + (size_t) t * mask_ld : nullptr;
    float mx = -INFINITY;
    for (int s = threadIdx.x; s < n_ctx; s += blockDim.x)
        if (s <= vis && (!m || m[s])) mx = fmaxf(mx, row[s]);
    mx = bmax(mx, red);
    float sum = 0.0f;
    for (int s = threadIdx.x; s < n_ctx; s += blockDim.x) {
        const bool ok = s <= vis && (!m || m[s]);
        const float e = ok ? expf(row[s] - mx) : 0.0f;
        row[s] = e;
        sum += e;
    }
    const float inv = 1.0f / bsum(sum, red);
    for (int s = threadIdx.x; s < n_ctx; s += blockDim.x) row[s] *= inv;
}

/// The indexer's score of query t (t0 + local row) at cell s: sum_h relu(IS[(t,h), s]) * w[t, h]; cells past the
/// query's position are -inf.  IS rows are (local t, h) at stride n_ctx; out rows at stride out_ld.
__global__ void idx_combine_kernel(const float* __restrict__ IS, const float* __restrict__ w, float* __restrict__ out,
                                   int Tsub, int heads, int n_ctx, int t0, int pos0, int64_t out_ld, int out_t0) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t) Tsub * n_ctx) return;
    const int s = (int) (idx % n_ctx);
    const int tl = (int) (idx / n_ctx);
    const int t = t0 + tl;
    float acc = 0.0f;
    for (int h = 0; h < heads; ++h)
        acc += fmaxf(IS[((size_t) tl * heads + h) * n_ctx + s], 0.0f) * w[(size_t) t * heads + h];
    out[(size_t) (t - out_t0) * out_ld + s] = s <= pos0 + t ? acc : -INFINITY;
}

__global__ void add_rows_kernel(float* __restrict__ y, const float* __restrict__ a, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a[i];
}

__global__ void scale_rows_kernel(float* __restrict__ y, float s, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] *= s;
}

const strata::kernels::cpu::NativeFmt& fmt_of(void* p) { return *(const strata::kernels::cpu::NativeFmt*) p; }

}  // namespace

bool GlmDsaModel::pf_alloc(std::string& err) {
    if (pf_) return true;
    const GlmDsaGeometry& g = g_;
    // chunks of 512 tokens; 2048 when the experts may run on the GPU (its weight transfer is per chunk: 3268-token
    // prompt, RTX 5090: 1024 -> 83 s, 2048 -> 54 s)
    int T = gpu_moe_enabled() ? 2048 : 512;
    if (const char* s = std::getenv("STRATA_GLM_CHUNK")) T = std::max(1, std::atoi(s));
    pf_chunk_ = T;
    cublasHandle_t h = nullptr;
    PF_BLAS(cublasCreate(&h));
    cublas_ = h;
    PF_BLAS(cublasSetStream(h, 0));
    const size_t per_tok = (size_t) 5 * g.n_embd + g.q_lora + g.cache_row() + (size_t) g.n_head * g.qk_head() +
                           (size_t) g.n_head * g.kv_lora + (size_t) g.n_head * g.v_head + (size_t) 3 * g.n_ff_dense +
                           (size_t) g.idx_heads * g.idx_key + 64 + (size_t) g.n_expert + 2 * 64;
    PF_CUDA(cudaMalloc(&pf_, per_tok * (size_t) T * sizeof(float)));
    allocs_.push_back(pf_);
    // the largest weight any GEMM expands
    size_t big = 0;
    for (const auto& kv : w_)
        if (kv.second.type != 0 && kv.first != "output.weight")
            big = std::max(big, (size_t) (kv.second.n_in * kv.second.n_out));
    deq_floats_ = big;
    PF_CUDA(cudaMalloc(&deq_, big * sizeof(float)));
    allocs_.push_back(deq_);
    // blocks of queries sized to STRATA_GLM_ATT_MB (256) MB each:
    //  * the indexer's scores (idx_heads x max_ctx a query) - pf_sub_ queries a block;
    //  * dense attention over at most top_k cells (n_head x top_k a query) - pf_dsub_;
    //  * sparse attention: each query's top_k cells gathered (top_k x 576) + scores - pf_ssub_.
    size_t att_mb = 256;
    if (const char* s = std::getenv("STRATA_GLM_ATT_MB")) att_mb = (size_t) std::max(16, std::atoi(s));
    const size_t att_floats = att_mb << 18;
    const size_t K = (size_t) g.idx_top_k, CR = (size_t) g.cache_row();
    pf_sub_ = (int) std::max<size_t>(1, std::min<size_t>((size_t) T, att_floats / ((size_t) g.idx_heads * max_ctx_)));
    pf_dsub_ = (int) std::max<size_t>(1, std::min<size_t>((size_t) T, att_floats / ((size_t) g.n_head * K)));
    pf_ssub_ = (int) std::max<size_t>(1, std::min<size_t>((size_t) T, att_floats / (K * CR + (size_t) g.n_head * (K + CR))));
    const size_t att_n = std::max((size_t) pf_sub_ * g.idx_heads * max_ctx_, (size_t) pf_dsub_ * g.n_head * K);
    PF_CUDA(cudaMalloc(&att_, att_n * sizeof(float)));
    allocs_.push_back(att_);
    PF_CUDA(cudaMalloc(&pf_score_, (size_t) pf_sub_ * (size_t) max_ctx_ * sizeof(float)));
    allocs_.push_back(pf_score_);
    PF_CUDA(cudaMalloc(&sg_, (size_t) pf_ssub_ * K * CR * sizeof(float)));
    allocs_.push_back(sg_);
    PF_CUDA(cudaMalloc(&ss_, (size_t) pf_ssub_ * g.n_head * K * sizeof(float)));
    allocs_.push_back(ss_);
    PF_CUDA(cudaMalloc(&qc_, (size_t) pf_ssub_ * g.n_head * CR * sizeof(float)));
    allocs_.push_back(qc_);
    PF_CUDA(cudaMalloc(&d_sel_, (size_t) T * K * sizeof(int)));
    allocs_.push_back(d_sel_);
    PF_CUDA(cudaMalloc(&d_cnt_, (size_t) T * sizeof(int)));
    allocs_.push_back(d_cnt_);
    PF_CUDA(cudaMalloc(&iq16_, (size_t) T * g.idx_heads * g.idx_key * sizeof(__half)));
    allocs_.push_back(iq16_);
    const size_t row_floats = 0;
    (void) row_floats;
    std::fprintf(stderr, "glm-dsa prefill: chunks of %d tokens; query blocks: indexer %d, dense %d, sparse %d\n", T,
                 pf_sub_, pf_dsub_, pf_ssub_);
    return true;
}

bool GlmDsaModel::gemm(const DevW& w, const float* X, float* Y, int T, std::string& err) {
    const float* Wf = (const float*) w.d;
    if (w.type == 8) {
        const int64_t n = w.n_in * w.n_out;
        dequant_q8_0_kernel<<<(unsigned) ((n + TB - 1) / TB), TB>>>((const uint8_t*) w.d, deq_, n);
        PF_CUDA(cudaGetLastError());
        Wf = deq_;
    } else if (w.type != 0) {
        err = "glm-dsa prefill: weight type " + std::to_string(w.type) + " has no device expansion yet (Q8_0, F32)";
        return false;
    }
    const float one = 1.0f, zero = 0.0f;
    PF_BLAS(cublasSgemm((cublasHandle_t) cublas_, CUBLAS_OP_T, CUBLAS_OP_N, (int) w.n_out, T, (int) w.n_in, &one, Wf,
                        (int) w.n_in, X, (int) w.n_in, &zero, Y, (int) w.n_out));
    return true;
}

bool GlmDsaModel::moe_cpu_batch(int il, int T, const int* ids, const float* rw, std::string& err) {
    namespace kc = strata::kernels::cpu;
    const GlmDsaGeometry& g = g_;
    const ExpertLayer& L = experts_[(size_t) il];
    const kc::NativeFmt& f = fmt_of(L.fmt);
    const int k = g.n_exp_used, E = g.n_embd, F = g.n_ff_exp;
    const size_t AB = kc::kNativeActBytes, HB = kc::kNativeHBytes;
    pool_->run(T, [&](int t) { kc::native_quant_act(f, hb_x_.data() + (size_t) t * E, hb_act_.data() + t * AB); });

    // slots (t*k + i) grouped by expert
    std::vector<std::vector<int>> by((size_t) g.n_expert);
    for (int j = 0; j < T * k; ++j) {
        const int e = ids[j];
        if (e < 0 || e >= g.n_expert) {
            err = "glm-dsa prefill: router chose expert " + std::to_string(e);
            return false;
        }
        by[(size_t) e].push_back(j);
    }
    struct Group {
        int e;
        std::vector<const void*> act, hq;
        std::vector<float*> ff, dn;
    };
    std::vector<Group> groups;
    for (int e = 0; e < g.n_expert; ++e) {
        if (by[(size_t) e].empty()) continue;
        Group gr;
        gr.e = e;
        for (int j : by[(size_t) e]) {
            gr.act.push_back(hb_act_.data() + (size_t) (j / k) * AB);
            gr.hq.push_back(hb_hq_.data() + (size_t) j * HB);
            gr.ff.push_back(hb_ff_.data() + (size_t) j * F);
            gr.dn.push_back(hb_dn_.data() + (size_t) j * E);
        }
        groups.push_back(std::move(gr));
    }
    constexpr int RG = 64, RD = 128;
    const int gc = F / RG, dc = E / RD;
    pool_->run((int) groups.size() * gc, [&](int task) {
        const Group& gr = groups[(size_t) (task / gc)];
        const int c = task % gc;
        kc::native_gu_rows_split(f, L.gate + L.gu_expert * (size_t) gr.e, L.up + L.gu_expert * (size_t) gr.e,
                                 gr.act.data(), (int) gr.act.size(), gr.ff.data(), c * RG, (c + 1) * RG);
    });
    pool_->run(T * k, [&](int j) { kc::native_quant_h(f, hb_ff_.data() + (size_t) j * F, hb_hq_.data() + j * HB); });
    pool_->run((int) groups.size() * dc, [&](int task) {
        const Group& gr = groups[(size_t) (task / dc)];
        const int c = task % dc;
        kc::native_down_rows_split(f, L.down + L.d_expert * (size_t) gr.e, gr.hq.data(), (int) gr.hq.size(),
                                   gr.dn.data(), c * RD, (c + 1) * RD);
    });
    // per token, the weighted sum in slot order (the step path's order)
    pool_->run(T, [&](int t) {
        float* out = hb_moe_.data() + (size_t) t * E;
        for (int j = 0; j < E; ++j) {
            float acc = 0.0f;
            for (int i = 0; i < k; ++i) acc += rw[t * k + i] * hb_dn_[((size_t) t * k + i) * E + j];
            out[j] = acc;
        }
    });
    return true;
}

bool GlmDsaModel::decode_index_scores(const void* ik16, const float* iq, const float* iw, float* score, int n_vis,
                                      std::string& err) {
    if (!pf_alloc(err)) return false;
    const float one = 1.0f, zero = 0.0f;
    const int IH = g_.idx_heads, IK = g_.idx_key;
    kv::to_f16(iq, (__half*) iq16_, (int64_t) IH * IK, nullptr);
    PF_BLAS(cublasGemmEx((cublasHandle_t) cublas_, CUBLAS_OP_T, CUBLAS_OP_N, n_vis, IH, IK, &one, ik16, CUDA_R_16F, IK,
                         iq16_, CUDA_R_16F, IK, &zero, att_, CUDA_R_32F, n_vis, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT));
    idx_combine_kernel<<<nb(n_vis), TB>>>(att_, iw, score, 1, IH, n_vis, 0, n_vis - 1, n_vis, 0);
    PF_CUDA(cudaGetLastError());
    return true;
}

bool GlmDsaModel::decode_attend(const float* cache, int p, const float* q, float* qabs, float* o, const float* vb,
                                bool masked, std::string& err) {
    if (!pf_alloc(err)) return false;
    const GlmDsaGeometry& g = g_;
    cublasHandle_t cb = (cublasHandle_t) cublas_;
    const int H = g.n_head, KL = g.kv_lora, CR = g.cache_row(), QH = g.qk_head(), n = p + 1;
    const float one = 1.0f, zero = 0.0f, kq = 1.0f / std::sqrt((float) QH);
    PF_BLAS(cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, n, H, KL, &kq, cache, CR, qabs, KL, &zero, att_, n));
    PF_BLAS(cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, n, H, g.qk_rope, &kq, cache + KL, CR, q + g.qk_nope, QH, &one, att_, n));
    attn_softmax_kernel<<<H, TB>>>(att_, n, 0, p, H, nullptr, max_ctx_);
    PF_BLAS(cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, KL, H, n, &one, cache, CR, att_, n, &zero, qabs, KL));
    PF_BLAS(cublasSgemmStridedBatched(cb, CUBLAS_OP_T, CUBLAS_OP_N, g.v_head, 1, KL, &one, vb, KL,
                                      (long long) KL * g.v_head, qabs, H * KL, KL, &zero, o, H * g.v_head, g.v_head, H));
    PF_CUDA(cudaGetLastError());
    return true;
}

bool GlmDsaModel::prefill(const int32_t* toks, int64_t n, std::string& err, bool head,
                          const std::function<void(int64_t, int64_t)>& progress) {
    if (n <= 0) return true;
    bind(0);
    if (!pf_alloc(err)) return false;
    if (pos_ + n > max_ctx_) {
        err = "glm-dsa: the prompt does not fit the context (" + std::to_string(pos_ + n) + " > " +
              std::to_string(max_ctx_) + ")";
        return false;
    }
    for (int64_t i = 0; i < n; i += pf_chunk_) {
        const int T = (int) std::min<int64_t>(pf_chunk_, n - i);
        if (!prefill_chunk(toks + i, T, head && i + T == n, err)) return false;
        if (progress) progress(i + T, n);
    }
    return true;
}

bool GlmDsaModel::prefill_chunk(const int32_t* toks, int T, bool head, std::string& err) {
    const GlmDsaGeometry& g = g_;
    bind(0);
    const auto t_start = std::chrono::steady_clock::now();
    double cpu_ms = 0;
    cublasHandle_t cb = (cublasHandle_t) cublas_;
    const int E = g.n_embd, H = g.n_head, QH = g.qk_head(), KL = g.kv_lora, CR = g.cache_row();
    const int pos0 = (int) pos_;
    const int64_t TE = (int64_t) T * E;
    for (int t = 0; t < T; ++t)
        if (toks[t] < 0 || toks[t] >= (int32_t) tok_embd_->shape[1]) {
            err = "glm-dsa: token id " + std::to_string(toks[t]) + " out of range";
            return false;
        }

    // chunk activations
    float *X, *XN, *TMP, *MOE, *SH, *QR, *KV, *Q, *QA, *O, *GA, *UP, *HH, *IQ, *IW, *RL, *RW;
    int *IDS;
    const auto carve = [&]() {
        X = pf_;
        XN = X + TE;
        TMP = XN + TE;
        MOE = TMP + TE;
        SH = MOE + TE;
        QR = SH + TE;
        KV = QR + (size_t) T * g.q_lora;
        Q = KV + (size_t) T * CR;
        QA = Q + (size_t) T * H * QH;
        O = QA + (size_t) T * H * KL;
        GA = O + (size_t) T * H * g.v_head;
        UP = GA + (size_t) T * g.n_ff_dense;
        HH = UP + (size_t) T * g.n_ff_dense;
        IQ = HH + (size_t) T * g.n_ff_dense;
        IW = IQ + (size_t) T * g.idx_heads * g.idx_key;
        RL = IW + (size_t) T * 64;
        RW = RL + (size_t) T * g.n_expert;
        IDS = (int*) (RW + (size_t) T * 64);
    };
    carve();

    // host buffers
    namespace kc = strata::kernels::cpu;
    if ((int) hb_ids_.size() < T * g.n_exp_used) {
        hb_x_.resize((size_t) T * E);
        hb_moe_.resize((size_t) T * E);
        hb_ff_.resize((size_t) T * g.n_exp_used * g.n_ff_exp);
        hb_dn_.resize((size_t) T * g.n_exp_used * E);
        hb_act_.resize((size_t) T * kc::kNativeActBytes);
        hb_hq_.resize((size_t) T * g.n_exp_used * kc::kNativeHBytes);
        hb_ids_.resize((size_t) T * g.n_exp_used);
        hb_rw_.resize((size_t) T * g.n_exp_used);
    }

    // embeddings (host dequant from the mmap)
    {
        const size_t row = ggml_row_size((ggml_type) tok_embd_->type, E);
        const auto to_float = ggml_get_type_traits((ggml_type) tok_embd_->type)->to_float;
        const uint8_t* base = tok_embd_file_->tensor_data(*tok_embd_);
        for (int t = 0; t < T; ++t) to_float(base + row * (size_t) toks[t], hb_x_.data() + (size_t) t * E, E);
        PF_CUDA(cudaMemcpy(X, hb_x_.data(), (size_t) TE * sizeof(float), cudaMemcpyHostToDevice));
    }

    const float theta_scale = std::pow(g.rope_base, -2.0f / (float) g.qk_rope);
    const float kq_scale = 1.0f / std::sqrt((float) QH);
    const float one = 1.0f, zero = 0.0f;
    const int n_end = pos0 + T;               // cells visible to the chunk's last query
    const bool any_sparse = n_end > g.idx_top_k;
    bool sparse = false;                      // the current selection is a mask (else: every visible cell)

    for (int il = 0; il < g.n_layers; ++il) {
        if (part_of(il) != cur_) {   // the layer split: the chunk's residual to the next device, its selections too
            const float* X_old = X;
            const int *sel_old = d_sel_, *cnt_old = d_cnt_;
            PF_CUDA(cudaDeviceSynchronize());
            bind(part_of(il));
            if (!pf_alloc(err)) return false;
            carve();
            cb = (cublasHandle_t) cublas_;
            if (!hand_over(X_old, X, T, err)) return false;
            if (sparse) {
                PF_CUDA(cudaMemcpy(d_sel_, sel_old, (size_t) T * g.idx_top_k * sizeof(int), cudaMemcpyDefault));
                PF_CUDA(cudaMemcpy(d_cnt_, cnt_old, (size_t) T * sizeof(int), cudaMemcpyDefault));
            }
        }
        // ---- attention inputs
        rms_rows_kernel<<<T, TB>>>(X, (const float*) W(il, "attn_norm.weight").d, XN, E, E, E, g.norm_eps);
        if (!gemm(W(il, "attn_q_a.weight"), XN, QR, T, err)) return false;
        rms_rows_kernel<<<T, TB>>>(QR, (const float*) W(il, "attn_q_a_norm.weight").d, QR, g.q_lora, g.q_lora,
                                   g.q_lora, g.norm_eps);
        // kv rows [c | kpe], normed and roped, then quantized into the cache rows pos0..pos0+T-1
        if (!gemm(W(il, "attn_kv_a_mqa.weight"), XN, KV, T, err)) return false;
        rms_rows_kernel<<<T, TB>>>(KV, (const float*) W(il, "attn_kv_a_norm.weight").d, KV, KL, CR, CR, g.norm_eps);
        rope_rows_kernel<<<nb((int64_t) T * g.qk_rope / 2), TB>>>(KV, T, 1, CR, CR, KL, g.qk_rope, pos0, theta_scale);
        kv::store_q8(KV, T, CR, CR, kv_q(il), (__half*) kv_s(il), pos0, nullptr, nullptr);

        // ---- the indexer (full layers): keys into their cache; the selection when a query sees > top_k cells
        if (g.indexer_full[(size_t) il]) {
            float* IKC = TMP;   // (free until the out-projection)
            if (!gemm(W(il, "indexer.attn_k.weight"), XN, IKC, T, err)) return false;
            ln_rows_kernel<<<T, 128>>>(IKC, (const float*) W(il, "indexer.k_norm.weight").d,
                                       (const float*) W(il, "indexer.k_norm.bias").d, g.idx_key, g.idx_key,
                                       g.idx_norm_eps);
            rope_rows_kernel<<<nb((int64_t) T * g.qk_rope / 2), TB>>>(IKC, T, 1, g.idx_key, g.idx_key, 0, g.qk_rope,
                                                                      pos0, theta_scale);
            kv::store_f16(IKC, T, g.idx_key, g.idx_key, (__half*) ik_keys(il), pos0, nullptr, nullptr);
            sparse = any_sparse;
            if (any_sparse) {
                const int IH = g.idx_heads, IK = g.idx_key, K = g.idx_top_k;
                if (!gemm(W(il, "indexer.attn_q_b.weight"), QR, IQ, T, err)) return false;
                rope_rows_kernel<<<nb((int64_t) T * IH * g.qk_rope / 2), TB>>>(IQ, T, IH, IK, (int64_t) IH * IK, 0,
                                                                               g.qk_rope, pos0, theta_scale);
                kv::to_f16(IQ, (__half*) iq16_, (int64_t) T * IH * IK, nullptr);
                if (!gemm(W(il, "indexer.proj.weight"), XN, IW, T, err)) return false;
                scale_rows_kernel<<<nb((int64_t) T * IH), TB>>>(IW, 1.0f / std::sqrt((float) (IK * IH)),
                                                                (int64_t) T * IH);
                // each query's selection: a query that sees <= top_k cells sees them all; else the top_k of its scores
                // (ties to the lower cell), ascending, padded with -1 - on the device; STRATA_GLM_TOPK_HOST=1 the host
                static const bool topk_host =
                    std::getenv("STRATA_GLM_TOPK_HOST") && std::atoi(std::getenv("STRATA_GLM_TOPK_HOST"));
                h_sel_.assign((size_t) T * K, -1);
                h_cnt_.assign((size_t) T, 0);
                hb_score_.resize((size_t) pf_sub_ * max_ctx_);
                const void* ik16 = ik_keys(il);
                for (int t0 = 0; t0 < T; t0 += pf_sub_) {
                    const int ts = std::min(pf_sub_, T - t0);
                    const int nc = pos0 + t0 + ts;
                    PF_BLAS(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, nc, ts * IH, IK, &one, ik16, CUDA_R_16F, IK,
                                         (const __half*) iq16_ + (size_t) t0 * IH * IK, CUDA_R_16F, IK, &zero, att_,
                                         CUDA_R_32F, nc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                    idx_combine_kernel<<<nb((int64_t) ts * nc), TB>>>(att_, IW, pf_score_, ts, IH, nc, t0, pos0,
                                                                      max_ctx_, t0);
                    if (!topk_host) {   // the selection on the device (glm_dsa_kv.cuh topk_rows)
                        kv::topk_rows(pf_score_, max_ctx_, ts, 0, pos0 + t0 + 1, K, d_sel_ + (size_t) t0 * K, K,
                                      d_cnt_ + t0, nullptr);
                        PF_CUDA(cudaGetLastError());
                        continue;
                    }
                    PF_CUDA(cudaMemcpy2D(hb_score_.data(), (size_t) max_ctx_ * sizeof(float), pf_score_,
                                         (size_t) max_ctx_ * sizeof(float), (size_t) nc * sizeof(float), (size_t) ts,
                                         cudaMemcpyDeviceToHost));
                    pool_->run(ts, [&](int tl) {
                        const int t = t0 + tl, vis = pos0 + t + 1;
                        int* sel = h_sel_.data() + (size_t) t * K;
                        if (vis <= K) {
                            for (int i = 0; i < vis; ++i) sel[i] = i;
                            h_cnt_[(size_t) t] = vis;
                            return;
                        }
                        const float* sc = hb_score_.data() + (size_t) tl * max_ctx_;
                        std::vector<int> c((size_t) vis);
                        std::iota(c.begin(), c.end(), 0);
                        std::nth_element(c.begin(), c.begin() + K, c.end(), [&](int a, int b) {
                            return sc[a] > sc[b] || (sc[a] == sc[b] && a < b);
                        });
                        std::sort(c.begin(), c.begin() + K);
                        std::copy(c.begin(), c.begin() + K, sel);
                        h_cnt_[(size_t) t] = K;
                    });
                }
                if (topk_host) {
                    PF_CUDA(cudaMemcpy(d_sel_, h_sel_.data(), h_sel_.size() * sizeof(int), cudaMemcpyHostToDevice));
                    PF_CUDA(cudaMemcpy(d_cnt_, h_cnt_.data(), h_cnt_.size() * sizeof(int), cudaMemcpyHostToDevice));
                }
            }
        }

        // ---- q, absorbed
        if (!gemm(W(il, "attn_q_b.weight"), QR, Q, T, err)) return false;
        rope_rows_kernel<<<nb((int64_t) T * H * g.qk_rope / 2), TB>>>(Q, T, H, QH, (int64_t) H * QH, g.qk_nope,
                                                                      g.qk_rope, pos0, theta_scale);
        // QA[t, h, :] = kb_h^T q_nope[t, h, :]   (kb {qk_nope, kv_lora, n_head})
        PF_BLAS(cublasSgemmStridedBatched(cb, CUBLAS_OP_T, CUBLAS_OP_N, KL, T, g.qk_nope, &one,
                                          (const float*) W(il, "attn_k_b.weight").d, g.qk_nope,
                                          (long long) g.qk_nope * KL, Q, H * QH, QH, &zero, QA, H * KL, KL, H));

        // ---- attention; the context overwrites QA row by row
        if (!sparse) {
            // every query sees <= top_k cells: the visible rows dequantized once, then GEMMs over them
            const int n = pos0 + T;
            kv::gather_q8(kv_q(il), (const __half*) kv_s(il), CR, nullptr, 0, n, kvg_, nullptr);
            for (int t0 = 0; t0 < T; t0 += pf_dsub_) {
                const int ts = std::min(pf_dsub_, T - t0);
                const int nc = pos0 + t0 + ts;     // the cells the block's last query sees
                const int rows = ts * H;
                float* QAb = QA + (size_t) t0 * H * KL;
                PF_BLAS(cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, nc, rows, KL, &kq_scale, kvg_, CR, QAb, KL, &zero,
                                    att_, nc));
                PF_BLAS(cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, nc, rows, g.qk_rope, &kq_scale, kvg_ + KL, CR,
                                    Q + (size_t) t0 * H * QH + g.qk_nope, QH, &one, att_, nc));
                attn_softmax_kernel<<<rows, TB>>>(att_, nc, t0, pos0, H, nullptr, max_ctx_);
                PF_BLAS(cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, KL, rows, nc, &one, kvg_, CR, att_, nc, &zero, QAb,
                                    KL));
            }
        } else {
            // each query's own (up to) top_k cells gathered, then batched GEMMs per query: scores, softmax, context
            const int K = g.idx_top_k;
            for (int t0 = 0; t0 < T; t0 += pf_ssub_) {
                const int ts = std::min(pf_ssub_, T - t0);
                kv::gather_q8(kv_q(il), (const __half*) kv_s(il), CR, d_sel_ + (size_t) t0 * K, 0, ts * K, sg_,
                              nullptr);
                kv::concat_q(QA + (size_t) t0 * H * KL, Q + (size_t) t0 * H * QH, qc_, ts, H, KL, g.qk_nope,
                             g.qk_rope, nullptr);
                PF_BLAS(cublasSgemmStridedBatched(cb, CUBLAS_OP_T, CUBLAS_OP_N, K, H, CR, &kq_scale, sg_, CR,
                                                  (long long) K * CR, qc_, CR, (long long) H * CR, &zero, ss_, K,
                                                  (long long) H * K, ts));
                kv::softmax_counted(ss_, d_cnt_ + t0, ts, H, K, nullptr);
                PF_BLAS(cublasSgemmStridedBatched(cb, CUBLAS_OP_N, CUBLAS_OP_N, KL, H, K, &one, sg_, CR,
                                                  (long long) K * CR, ss_, K, (long long) H * K, &zero,
                                                  QA + (size_t) t0 * H * KL, KL, (long long) H * KL, ts));
            }
        }
        // O[t, h, :] = vb_h^T ctx[t, h, :]   (vb {kv_lora, v_head, n_head})
        PF_BLAS(cublasSgemmStridedBatched(cb, CUBLAS_OP_T, CUBLAS_OP_N, g.v_head, T, KL, &one,
                                          (const float*) W(il, "attn_v_b.weight").d, KL, (long long) KL * g.v_head, QA,
                                          H * KL, KL, &zero, O, H * g.v_head, g.v_head, H));
        if (!gemm(W(il, "attn_output.weight"), O, TMP, T, err)) return false;
        add_rows_kernel<<<nb(TE), TB>>>(X, TMP, TE);
        capture(il, 0, X + (size_t) (T - 1) * E);

        // ---- the FFN
        rms_rows_kernel<<<T, TB>>>(X, (const float*) W(il, "ffn_norm.weight").d, XN, E, E, E, g.norm_eps);
        if (il < g.dense_lead) {
            if (!gemm(W(il, "ffn_gate.weight"), XN, GA, T, err) || !gemm(W(il, "ffn_up.weight"), XN, UP, T, err))
                return false;
            strata::kernels::glm_swiglu_clamp(GA, UP, INFINITY, (int64_t) T * g.n_ff_dense, HH, nullptr);
            if (!gemm(W(il, "ffn_down.weight"), HH, TMP, T, err)) return false;
            add_rows_kernel<<<nb(TE), TB>>>(X, TMP, TE);
        } else {
            if (!gemm(W(il, "ffn_gate_inp.weight"), XN, RL, T, err)) return false;
            strata::kernels::router_sigmoid(RL, (const float*) W(il, "exp_probs_b.bias").d, T, g.n_expert,
                                            g.n_exp_used, g.w_scale, g.norm_w != 0, IDS, RW, nullptr);
            const size_t kk = (size_t) T * g.n_exp_used;
            PF_CUDA(cudaMemcpy(hb_ids_.data(), IDS, kk * sizeof(int), cudaMemcpyDeviceToHost));
            PF_CUDA(cudaMemcpy(hb_rw_.data(), RW, kk * sizeof(float), cudaMemcpyDeviceToHost));
            route_log(il, hb_ids_.data(), T);
            // a long chunk's experts on the GPU (STRATA_GLM_GPU_MOE_MIN tokens, 512: below it the CPU is as fast -
            // the GPU's cost is the weight transfer, the same for any chunk length)
            static const int gpu_min = std::getenv("STRATA_GLM_GPU_MOE_MIN") ? std::atoi(std::getenv("STRATA_GLM_GPU_MOE_MIN")) : 512;
            if (gpu_moe_enabled() && T >= gpu_min) {
                const int FS = g.n_ff_exp * g.n_shared;
                if (!gemm(W(il, "ffn_gate_shexp.weight"), XN, GA, T, err) ||
                    !gemm(W(il, "ffn_up_shexp.weight"), XN, UP, T, err))
                    return false;
                strata::kernels::glm_swiglu_clamp(GA, UP, INFINITY, (int64_t) T * FS, HH, nullptr);
                if (!gemm(W(il, "ffn_down_shexp.weight"), HH, SH, T, err)) return false;
                if (!moe_gpu_batch(il, T, XN, RW, hb_ids_.data(), MOE, err)) return false;
                // STRATA_GLM_GPU_MOE_CHECK=1: the CPU path on the same input, the relative difference printed
                static const bool chk = std::getenv("STRATA_GLM_GPU_MOE_CHECK") && std::atoi(std::getenv("STRATA_GLM_GPU_MOE_CHECK"));
                if (chk) {
                    PF_CUDA(cudaMemcpy(hb_x_.data(), XN, (size_t) TE * sizeof(float), cudaMemcpyDeviceToHost));
                    if (!moe_cpu_batch(il, T, hb_ids_.data(), hb_rw_.data(), err)) return false;
                    std::vector<float> gpu((size_t) TE);
                    PF_CUDA(cudaMemcpy(gpu.data(), MOE, (size_t) TE * sizeof(float), cudaMemcpyDeviceToHost));
                    double d = 0, n = 0, d0 = 0, n0 = 0;
                    for (int64_t i = 0; i < TE; ++i) {
                        d += (double) (gpu[i] - hb_moe_[i]) * (gpu[i] - hb_moe_[i]);
                        n += (double) hb_moe_[i] * hb_moe_[i];
                        if (i < E) d0 += (double) (gpu[i] - hb_moe_[i]) * (gpu[i] - hb_moe_[i]), n0 += (double) hb_moe_[i] * hb_moe_[i];
                    }
                    std::fprintf(stderr, "gpu-moe check L%d: rel diff %.3e (token 0: %.3e), |cpu| %.3e, gpu[0..3] %g %g %g cpu %g %g %g", il,
                                 std::sqrt(d / std::max(n, 1e-30)), std::sqrt(d0 / std::max(n0, 1e-30)), std::sqrt(n), gpu[0], gpu[1], gpu[2],
                                 hb_moe_[0], hb_moe_[1], hb_moe_[2]);
                    std::fputc(10, stderr);
                }
                add_rows_kernel<<<nb(TE), TB>>>(MOE, SH, TE);
                add_rows_kernel<<<nb(TE), TB>>>(X, MOE, TE);
                PF_CUDA(cudaGetLastError());
                capture(il, 1, X + (size_t) (T - 1) * E);
                continue;
            }
            PF_CUDA(cudaMemcpy(hb_x_.data(), XN, (size_t) TE * sizeof(float), cudaMemcpyDeviceToHost));
            capture_ids(il, hb_ids_.data() + (size_t) (T - 1) * g.n_exp_used, hb_rw_.data() + (size_t) (T - 1) * g.n_exp_used);
            // the shared expert on the device while the CPU runs the routed ones
            const int FS = g.n_ff_exp * g.n_shared;
            if (!gemm(W(il, "ffn_gate_shexp.weight"), XN, GA, T, err) ||
                !gemm(W(il, "ffn_up_shexp.weight"), XN, UP, T, err))
                return false;
            strata::kernels::glm_swiglu_clamp(GA, UP, INFINITY, (int64_t) T * FS, HH, nullptr);
            if (!gemm(W(il, "ffn_down_shexp.weight"), HH, SH, T, err)) return false;
            const auto c0 = std::chrono::steady_clock::now();
            if (!moe_cpu_batch(il, T, hb_ids_.data(), hb_rw_.data(), err)) return false;
            cpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
            PF_CUDA(cudaMemcpy(MOE, hb_moe_.data(), (size_t) TE * sizeof(float), cudaMemcpyHostToDevice));
            add_rows_kernel<<<nb(TE), TB>>>(MOE, SH, TE);
            add_rows_kernel<<<nb(TE), TB>>>(X, MOE, TE);
        }
        PF_CUDA(cudaGetLastError());
        capture(il, 1, X + (size_t) (T - 1) * E);
    }

    if (head) {
        float* xl = X + (size_t) (T - 1) * E;
        rms_rows_kernel<<<1, TB>>>(xl, (const float*) w_.at("output_norm.weight").d, XN, E, E, E, g.norm_eps);
        if (!gemv(w_.at("output.weight"), XN, d_logits_buf_, err)) return false;
        d_logits_ = d_logits_buf_;
    }
    PF_CUDA(cudaDeviceSynchronize());
    pos_ += T;
    const double total =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    t_total_ += total;
    t_cpu_ += cpu_ms;
    t_dev_ += total - cpu_ms;
    return true;
}

}  // namespace strata::core
