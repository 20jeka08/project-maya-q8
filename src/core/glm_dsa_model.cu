// src/core/glm_dsa_model.cu - the GLM-5.3 ("glm-dsa") runner.  See glm_dsa_model.hpp for the math.
#include "strata/core/glm_dsa_model.hpp"

#include "strata/artifact/gguf_split.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/glm_ffn.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/router_sigmoid.hpp"
#include "strata/kernels/sampler.hpp"
#include "glm_dsa_kv.cuh"

#include "ggml.h"

#include <cublas_v2.h>
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

constexpr int T_ = 256;   // threads per block for the elementwise/row kernels
int nblk(int64_t n) { return (int) ((n + T_ - 1) / T_); }

#define GD_CHECK(call)                                                                                  \
    do {                                                                                                \
        const cudaError_t e_ = (call);                                                                  \
        if (e_ != cudaSuccess) {                                                                        \
            err = std::string("cuda: ") + #call + ": " + cudaGetErrorString(e_);                        \
            return false;                                                                               \
        }                                                                                               \
    } while (0)

// ---- kernels -------------------------------------------------------------------------------------------------

/// y[r] = w[r, :] . x, warp per row (F32 weights in ggml order: row r at w + r*in).
__global__ void gemv_f32_kernel(const float* __restrict__ w, const float* __restrict__ x, float* __restrict__ y,
                                int out, int in) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= out) return;
    const float* wr = w + (size_t) row * in;
    float acc = 0.0f;
    for (int k = lane; k < in; k += 32) acc += wr[k] * x[k];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) y[row] = acc;
}

__device__ float block_sum(float v, float* red) {
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

__device__ float block_max(float v, float* red) {
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

/// y = x / rms(x) * w over `dim` (one block); in place is fine.
__global__ void rms_norm_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y,
                                int dim, float eps) {
    __shared__ float red[32];
    float ss = 0.0f;
    for (int e = threadIdx.x; e < dim; e += blockDim.x) ss += x[e] * x[e];
    ss = block_sum(ss, red);
    const float inv = rsqrtf(ss / (float) dim + eps);
    for (int e = threadIdx.x; e < dim; e += blockDim.x) y[e] = x[e] * inv * w[e];
}

/// y = (x - mean) / sqrt(var + eps) * w + b over `dim` (one block); in place is fine.
__global__ void layer_norm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                  const float* __restrict__ b, float* __restrict__ y, int dim, float eps) {
    __shared__ float red[32];
    float s = 0.0f;
    for (int e = threadIdx.x; e < dim; e += blockDim.x) s += x[e];
    const float mu = block_sum(s, red) / (float) dim;
    float v = 0.0f;
    for (int e = threadIdx.x; e < dim; e += blockDim.x) v += (x[e] - mu) * (x[e] - mu);
    const float inv = rsqrtf(block_sum(v, red) / (float) dim + eps);
    for (int e = threadIdx.x; e < dim; e += blockDim.x) y[e] = (x[e] - mu) * inv * w[e] + b[e];
}

/// ggml's ROPE_TYPE_NORM on n_vec vectors: pairs (2i, 2i+1) of the n_rot-wide slice at x + v*stride + off.
/// theta = pos * theta_scale^i, theta_scale = base^(-2/n_rot) - ggml-cuda's rope_norm with ext_factor 0.
__global__ void rope_norm_kernel(float* __restrict__ x, int n_vec, int stride, int off, int n_rot, int pos,
                                 float theta_scale) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int half = n_rot / 2;
    if (idx >= n_vec * half) return;
    const int v = idx / half, i = idx % half;
    const float theta = (float) pos * powf(theta_scale, (float) i);
    const float c = cosf(theta), s = sinf(theta);
    float* p = x + (size_t) v * stride + off + 2 * i;
    const float x0 = p[0], x1 = p[1];
    p[0] = x0 * c - x1 * s;
    p[1] = x0 * s + x1 * c;
}

/// qabs[h, c] = sum_e kb[e, c, h] * q[h*qk_head + e], e < qk_nope.  kb is GGUF {qk_nope, kv_lora, n_head}.
__global__ void q_absorb_kernel(const float* __restrict__ kb, const float* __restrict__ q, float* __restrict__ qabs,
                                int qk_nope, int qk_head, int kv_lora, int n_head) {
    const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wid >= kv_lora * n_head) return;
    const int h = wid / kv_lora, c = wid % kv_lora;
    const float* w = kb + (size_t) qk_nope * c + (size_t) qk_nope * kv_lora * h;
    const float* qh = q + (size_t) qk_head * h;
    float acc = 0.0f;
    for (int e = lane; e < qk_nope; e += 32) acc += w[e] * qh[e];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) qabs[(size_t) kv_lora * h + c] = acc;
}

/// One block per head.  Scores over the n_sel cells (cells == null: cells 0..n_sel-1), softmax, the latent context,
/// then v through vb (GGUF {kv_lora, v_head, n_head}).  Cache rows are [c (kv_lora) | kpe (qk_rope)].
__global__ void mla_attn_kernel(const float* __restrict__ qabs, const float* __restrict__ q,
                                const float* __restrict__ cache, const int* __restrict__ cells, int n_sel,
                                const float* __restrict__ vb, float* __restrict__ out, int kv_lora, int qk_nope,
                                int qk_rope, int v_head, float scale) {
    extern __shared__ float sm[];
    const int h = blockIdx.x;
    const int row = kv_lora + qk_rope;
    float* s_q = sm;                 // row
    float* s_ctx = s_q + row;        // kv_lora
    float* red = s_ctx + kv_lora;    // 32
    float* s_p = red + 32;           // n_sel
    for (int e = threadIdx.x; e < kv_lora; e += blockDim.x) s_q[e] = qabs[(size_t) kv_lora * h + e];
    for (int e = threadIdx.x; e < qk_rope; e += blockDim.x)
        s_q[kv_lora + e] = q[(size_t) (qk_nope + qk_rope) * h + qk_nope + e];
    __syncthreads();

    float mx = -INFINITY;
    for (int s = threadIdx.x; s < n_sel; s += blockDim.x) {
        const int cell = cells ? cells[s] : s;
        const float* cr = cache + (size_t) row * cell;
        float d = 0.0f;
        for (int e = 0; e < row; ++e) d += s_q[e] * cr[e];
        d *= scale;
        s_p[s] = d;
        mx = fmaxf(mx, d);
    }
    mx = block_max(mx, red);
    float sum = 0.0f;
    for (int s = threadIdx.x; s < n_sel; s += blockDim.x) {
        const float e = expf(s_p[s] - mx);
        s_p[s] = e;
        sum += e;
    }
    const float inv = 1.0f / block_sum(sum, red);
    __syncthreads();

    for (int e = threadIdx.x; e < kv_lora; e += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < n_sel; ++s) {
            const int cell = cells ? cells[s] : s;
            acc += s_p[s] * cache[(size_t) row * cell + e];
        }
        s_ctx[e] = acc * inv;
    }
    __syncthreads();

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    for (int v = warp; v < v_head; v += nw) {
        const float* w = vb + (size_t) kv_lora * v + (size_t) kv_lora * v_head * h;
        float acc = 0.0f;
        for (int c = lane; c < kv_lora; c += 32) acc += w[c] * s_ctx[c];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) out[(size_t) v_head * h + v] = acc;
    }
}

/// score[s] = sum_h relu(iq_h . ik_s) * w_h for s < n (one thread per cell; iq and w staged in shared memory).
__global__ void indexer_score_kernel(const float* __restrict__ iq, const float* __restrict__ ik,
                                     const float* __restrict__ w, float* __restrict__ score, int n, int heads,
                                     int key) {
    extern __shared__ float sm[];
    float* s_q = sm;            // heads*key
    float* s_w = sm + heads * key;
    for (int i = threadIdx.x; i < heads * key; i += blockDim.x) s_q[i] = iq[i];
    for (int i = threadIdx.x; i < heads; i += blockDim.x) s_w[i] = w[i];
    __syncthreads();
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) return;
    const float* k = ik + (size_t) key * s;
    float acc = 0.0f;
    for (int h = 0; h < heads; ++h) {
        float d = 0.0f;
        for (int e = 0; e < key; ++e) d += s_q[h * key + e] * k[e];
        acc += fmaxf(d, 0.0f) * s_w[h];
    }
    score[s] = acc;
}

__global__ void add_kernel(float* __restrict__ y, const float* __restrict__ a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a[i];
}

__global__ void scale_kernel(float* __restrict__ y, float s, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] *= s;
}

/// rope_norm_kernel with the position read from the device (a captured graph serves every token).
__global__ void rope_dev_kernel(float* __restrict__ x, int n_vec, int stride, int off, int n_rot,
                                const int* __restrict__ pos, float theta_scale) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int half = n_rot / 2;
    if (idx >= n_vec * half) return;
    const int v = idx / half, i = idx % half;
    const float theta = (float) (*pos) * powf(theta_scale, (float) i);
    const float c = cosf(theta), s = sinf(theta);
    float* p = x + (size_t) v * stride + off + 2 * i;
    const float x0 = p[0], x1 = p[1];
    p[0] = x0 * c - x1 * s;
    p[1] = x0 * s + x1 * c;
}

/// base[pos * row + i] = src[i], the position from the device.
__global__ void store_row_kernel(float* __restrict__ base, int row, const float* __restrict__ src,
                                 const int* __restrict__ pos) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < row) base[(size_t) (*pos) * row + i] = src[i];
}

const strata::kernels::cpu::NativeFmt& fmt_of(void* p) { return *(const strata::kernels::cpu::NativeFmt*) p; }

}  // namespace

// ================================ geometry ================================

bool GlmDsaGeometry::from_gguf(const strata::GgufFile& g, GlmDsaGeometry& out, std::string& err) {
    const strata::MetaValue* arch = g.get("general.architecture");
    if (!arch || arch->s != "glm-dsa") {
        err = "not a glm-dsa model (general.architecture = '" + std::string(arch ? arch->s : "?") + "')";
        return false;
    }
    const std::string p = "glm-dsa.";
    const auto num = [&](const char* key, double def) -> double {
        const strata::MetaValue* v = g.get(p + key);
        if (!v) return def;
        if (v->type == strata::MetaType::ARRAY) return v->items.empty() ? def : v->items[0].num();
        return v->is_num() ? v->num() : def;
    };
    const auto need = [&](const char* key) -> bool {
        if (g.get(p + key)) return true;
        err = "glm-dsa: missing metadata key " + p + key;
        return false;
    };
    for (const char* k : {"embedding_length", "block_count", "attention.head_count", "attention.q_lora_rank",
                          "attention.kv_lora_rank", "attention.key_length_mla", "attention.value_length_mla",
                          "rope.dimension_count", "expert_count", "expert_used_count", "expert_feed_forward_length",
                          "attention.indexer.head_count", "attention.indexer.key_length", "attention.indexer.top_k"})
        if (!need(k)) return false;

    out.n_embd = (int) num("embedding_length", out.n_embd);
    out.nextn = (int) num("nextn_predict_layers", 0);
    out.n_layers = (int) num("block_count", 0) - out.nextn;
    out.n_head = (int) num("attention.head_count", out.n_head);
    out.q_lora = (int) num("attention.q_lora_rank", out.q_lora);
    out.kv_lora = (int) num("attention.kv_lora_rank", out.kv_lora);
    out.qk_rope = (int) num("rope.dimension_count", out.qk_rope);
    out.qk_nope = (int) num("attention.key_length_mla", 256) - out.qk_rope;
    out.v_head = (int) num("attention.value_length_mla", out.v_head);
    out.idx_heads = (int) num("attention.indexer.head_count", out.idx_heads);
    out.idx_key = (int) num("attention.indexer.key_length", out.idx_key);
    out.idx_top_k = (int) num("attention.indexer.top_k", out.idx_top_k);
    out.n_expert = (int) num("expert_count", out.n_expert);
    out.n_exp_used = (int) num("expert_used_count", out.n_exp_used);
    out.n_ff_exp = (int) num("expert_feed_forward_length", out.n_ff_exp);
    out.n_shared = (int) num("expert_shared_count", out.n_shared);
    out.n_ff_dense = (int) num("feed_forward_length", out.n_ff_dense);
    out.dense_lead = (int) num("leading_dense_block_count", out.dense_lead);
    out.w_scale = (float) num("expert_weights_scale", 1.0);
    out.norm_w = (int) num("expert_weights_norm", 0);
    out.rope_base = (float) num("rope.freq_base", 10000.0);
    out.norm_eps = (float) num("attention.layer_norm_rms_epsilon", 1e-6);
    if (const strata::MetaValue* v = g.get("tokenizer.ggml.tokens"))
        if (v->type == strata::MetaType::ARRAY) out.n_vocab = (int) v->count;
    out.n_vocab = (int) num("vocab_size", out.n_vocab);

    const double gating = num("expert_gating_func", 2);
    if (gating != 2) {   // llama.cpp: 0 (none) also means sigmoid for glm-dsa
        if (gating != 0) {
            err = "glm-dsa: expert_gating_func " + std::to_string((int) gating) + " (only sigmoid is implemented)";
            return false;
        }
    }
    if (g.get(p + "rope.scaling.type") || g.get(p + "rope.scaling.factor")) {
        err = "glm-dsa: rope scaling (YaRN) metadata present - not implemented in this runner";
        return false;
    }

    // the indexer types: GLM 5/5.1 (n_ctx_train < 1M) had a full indexer on every layer; GLM-5.2+ computes it on
    // layers 0, 1 and then every 4th from 2, and the others reuse it (llama.cpp GLM_5_2_DEFAULT_INDEXER_TYPES).
    // An explicit attention.indexer.types array overrides both.
    out.indexer_full.assign((size_t) out.n_layers, 1);
    const double n_ctx_train = num("context_length", 0);
    if (n_ctx_train >= 1048576)
        for (int il = 0; il < out.n_layers; ++il) out.indexer_full[(size_t) il] = (il < 2 || (il - 2) % 4 == 0);
    if (const strata::MetaValue* v = g.get(p + "attention.indexer.types")) {
        if (v->type != strata::MetaType::ARRAY || v->items.size() < (size_t) out.n_layers) {
            err = "glm-dsa: attention.indexer.types is present but not readable (the reader keeps 64 array items)";
            return false;
        }
        for (int il = 0; il < out.n_layers; ++il) out.indexer_full[(size_t) il] = v->items[(size_t) il].u != 0;
    }
    if (!out.indexer_full.empty() && !out.indexer_full[0]) {
        err = "glm-dsa: layer 0 must carry a full indexer";
        return false;
    }
    if (out.n_layers <= 0 || out.qk_nope <= 0 || out.idx_key < out.qk_rope || out.n_embd % 256) {
        err = "glm-dsa: implausible geometry";
        return false;
    }
    return true;
}

// ================================ the CPU pool ================================

GlmCpuPool::GlmCpuPool(int n_threads) {
    for (int i = 1; i < n_threads; ++i) workers_.emplace_back([this] { worker(); });
}

GlmCpuPool::~GlmCpuPool() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : workers_) t.join();
}

void GlmCpuPool::drain() {
    for (;;) {
        const int i = next_.fetch_add(1);
        if (i >= n_) break;
        (*fn_)(i);
        finished_.fetch_add(1);
    }
}

void GlmCpuPool::worker() {
    uint64_t seen = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return stop_ || epoch_ != seen; });
            if (stop_) return;
            seen = epoch_;
            ++busy_;
        }
        drain();
        {
            std::lock_guard<std::mutex> lk(mu_);
            --busy_;
        }
        done_cv_.notify_all();
    }
}

void GlmCpuPool::run(int n, const std::function<void(int)>& fn) {
    if (n <= 0) return;
    {
        std::lock_guard<std::mutex> lk(mu_);
        fn_ = &fn;
        n_ = n;
        next_ = 0;
        finished_ = 0;
        ++epoch_;
    }
    cv_.notify_all();
    drain();
    std::unique_lock<std::mutex> lk(mu_);
    // every task finished AND no worker is still inside drain() (a late worker could otherwise read next_ of the
    // NEXT run's reset)
    done_cv_.wait(lk, [&] { return finished_.load() >= n_ && busy_ == 0; });
}

// ================================ the model ================================

GlmDsaModel::~GlmDsaModel() {
    if (!allocs_.empty()) cudaSetDevice(dev_);
    for (auto* v : {&gA_, &gB_, &gS_, &gC_})
        for (void* e : *v)
            if (e) cudaGraphExecDestroy((cudaGraphExec_t) e);
    if (gstream_) cudaStreamDestroy((cudaStream_t) gstream_);
    if (pin_) cudaFreeHost(pin_);
    gpu_moe_release();
    if (cublas_) cublasDestroy((cublasHandle_t) cublas_);
    for (void* p : allocs_) cudaFree(p);
    for (auto& e : experts_) delete (strata::kernels::cpu::NativeFmt*) e.fmt;
}

const strata::TensorInfo* GlmDsaModel::find(const std::string& name, const strata::GgufFile** file) const {
    for (const auto& f : files_)
        if (const strata::TensorInfo* t = f->find(name)) {
            if (file) *file = f.get();
            return t;
        }
    return nullptr;
}

bool GlmDsaModel::upload(const std::string& name, bool keep_quant, std::string& err) {
    const strata::GgufFile* f = nullptr;
    const strata::TensorInfo* t = find(name, &f);
    if (!t) {
        err = "glm-dsa: tensor " + name + " not found";
        return false;
    }
    DevW w;
    w.n_in = (int64_t) t->shape[0];
    w.n_out = (int64_t) (t->elements() / t->shape[0]);
    const uint8_t* src = f->tensor_data(*t);
    if (keep_quant && t->type != 0) {
        if (!strata::kernels::native_mmvq_supported((int) t->type)) {
            err = "glm-dsa: " + name + " has type " + t->type_name() + ", which the device MMVQ does not take";
            return false;
        }
        const size_t bytes = strata::kernels::native_mmvq_weight_bytes((int) t->type, (int) w.n_in, (int) w.n_out);
        GD_CHECK(cudaMalloc(&w.d, bytes));
        allocs_.push_back(w.d);
        GD_CHECK(cudaMemcpy(w.d, src, bytes, cudaMemcpyHostToDevice));
        w.type = (int) t->type;
    } else {
        std::vector<float> host((size_t) t->elements());
        if (t->type == 0) {
            std::memcpy(host.data(), src, host.size() * sizeof(float));
        } else {
            const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) t->type);
            if (!tt || !tt->to_float) {
                err = "glm-dsa: no dequantizer for " + name;
                return false;
            }
            tt->to_float(src, host.data(), (int64_t) host.size());
        }
        GD_CHECK(cudaMalloc(&w.d, host.size() * sizeof(float)));
        allocs_.push_back(w.d);
        GD_CHECK(cudaMemcpy(w.d, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
        w.type = 0;
    }
    w_[name] = w;
    return true;
}

const GlmDsaModel::DevW& GlmDsaModel::W(int il, const char* suffix) const {
    return w_.at("blk." + std::to_string(il) + "." + suffix);
}

bool GlmDsaModel::load(const std::string& gguf_path, int64_t max_ctx, int dev, std::string& err,
                       const std::string& split) {
    // 1. the shards
    std::vector<std::string> paths;
    try {
        paths = strata::gguf_split_paths(gguf_path);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    try {
        for (const auto& p : paths) files_.emplace_back(new strata::GgufFile(p));
    } catch (const std::exception& e) {
        err = std::string("glm-dsa: ") + e.what();
        return false;
    }
    if (!GlmDsaGeometry::from_gguf(*files_[0], g_, err)) return false;
    const GlmDsaGeometry& g = g_;
    max_ctx_ = max_ctx;

    // 2. the devices: `dev` (else the one with the most memory) first, then the others by memory
    int n_dev = 0;
    GD_CHECK(cudaGetDeviceCount(&n_dev));
    // spin while waiting for a device, on every device, before any context exists: decode synchronises once
    // per MoE layer, and with two contexts CUDA's automatic choice turned to blocking waits whose wake-up cost
    // ~0.8 ms each on Windows - 77 -> ~140 ms of device time a token with the layers split over two cards.
    // STRATA_GLM_SPIN=0 keeps the automatic choice.
    if (!(std::getenv("STRATA_GLM_SPIN") && std::atoi(std::getenv("STRATA_GLM_SPIN")) == 0))
        for (int d = 0; d < n_dev; ++d) {
            cudaSetDevice(d);
            cudaSetDeviceFlags(cudaDeviceScheduleSpin);
            cudaGetLastError();   // (a context that already exists keeps its flags)
        }
    std::vector<std::pair<size_t, int>> by_mem;
    for (int d = 0; d < n_dev; ++d) {
        cudaDeviceProp pr{};
        if (cudaGetDeviceProperties(&pr, d) == cudaSuccess) by_mem.push_back({pr.totalGlobalMem, d});
    }
    std::sort(by_mem.begin(), by_mem.end(), [](auto a, auto b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    if (dev < 0) dev = by_mem.empty() ? -1 : by_mem[0].second;
    if (dev < 0 || dev >= n_dev) {
        err = "glm-dsa: no such CUDA device " + std::to_string(dev);
        return false;
    }
    dev_ = dev;
    std::vector<int> devs = {dev};
    for (const auto& bm : by_mem)
        if (bm.second != dev) devs.push_back(bm.second);

    // 3. what each layer costs on a device: its weights and its caches for max_ctx tokens
    const char* quant[] = {"attn_q_a.weight", "attn_q_b.weight", "attn_kv_a_mqa.weight", "attn_output.weight",
                           "indexer.attn_q_b.weight", "indexer.attn_k.weight"};
    const char* f32[] = {"attn_norm.weight", "attn_q_a_norm.weight", "attn_kv_a_norm.weight", "ffn_norm.weight",
                         "attn_k_b.weight", "attn_v_b.weight", "indexer.k_norm.weight", "indexer.k_norm.bias",
                         "indexer.proj.weight"};
    const char* dense[] = {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"};
    const char* shexp[] = {"ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"};
    const char* router[] = {"ffn_gate_inp.weight", "exp_probs_b.bias"};
    const auto tensor_bytes = [&](const std::string& name, bool quantized) -> size_t {
        const strata::TensorInfo* t = find(name);
        if (!t) return 0;
        if (quantized && t->type != 0)
            return ggml_row_size((ggml_type) t->type, (int64_t) t->shape[0]) * (size_t) (t->elements() / t->shape[0]);
        return (size_t) t->elements() * sizeof(float);
    };
    const size_t kv_row = (size_t) g.cache_row() + (size_t) (g.cache_row() / kv::kBlock) * sizeof(__half);
    std::vector<size_t> layer_bytes((size_t) g.n_layers, 0);
    for (int il = 0; il < g.n_layers; ++il) {
        const std::string L = "blk." + std::to_string(il) + ".";
        size_t b = 0;
        for (const char* t : quant) b += tensor_bytes(L + t, true);
        for (const char* t : f32) b += tensor_bytes(L + t, false);
        if (il < g.dense_lead)
            for (const char* t : dense) b += tensor_bytes(L + t, true);
        else {
            for (const char* t : shexp) b += tensor_bytes(L + t, true);
            for (const char* t : router) b += tensor_bytes(L + t, false);
        }
        b += (size_t) max_ctx * kv_row;
        if (g.indexer_full[(size_t) il]) b += (size_t) max_ctx * g.idx_key * sizeof(__half);
        layer_bytes[(size_t) il] = b;
    }
    const size_t head_bytes = tensor_bytes("output.weight", true) + tensor_bytes("output_norm.weight", false);
    // per device, beside its layers: the batched path's scratch, the long-prompt expert lane, cuBLAS, graphs
    double reserve_gb = 5.0;
    if (const char* r = std::getenv("STRATA_GLM_RESERVE_GB")) reserve_gb = std::atof(r);
    const size_t reserve = (size_t) (reserve_gb * 1e9);

    // the split: "0" one device, "<layer>" two parts, "auto"/"" as many layers as fit on each device in turn
    std::string sp = split;
    if (const char* e = std::getenv("STRATA_GLM_SPLIT")) sp = e;
    std::vector<int> starts = {0};
    if (sp == "0") {
        devs.resize(1);
    } else if (!sp.empty() && sp != "auto") {
        const int at = std::atoi(sp.c_str());
        if (at <= 0 || at >= g.n_layers || devs.size() < 2) {
            err = "glm-dsa: --layer-split " + sp + " needs a layer in 1.." + std::to_string(g.n_layers - 1) + " and two devices";
            return false;
        }
        starts.push_back(at);
        devs.resize(2);
    } else {
        int il = 0;
        for (size_t di = 0; di < devs.size() && il < g.n_layers; ++di) {
            size_t fr = 0, tot = 0;
            GD_CHECK(cudaSetDevice(devs[di]));
            GD_CHECK(cudaMemGetInfo(&fr, &tot));
            const size_t room = fr > reserve ? fr - reserve : 0;
            size_t used = 0;
            const int first = il;
            while (il < g.n_layers) {
                const size_t need = layer_bytes[(size_t) il] + (il + 1 == g.n_layers ? head_bytes : 0);
                if (used + need > room) break;
                used += need;
                ++il;
            }
            if (il == g.n_layers) {
                devs.resize(di + 1);
                break;
            }
            if (di + 1 < devs.size()) {
                if (il == first) {
                    err = "glm-dsa: device " + std::to_string(devs[di]) + " cannot hold even one layer";
                    return false;
                }
                starts.push_back(il);
            }
        }
        if (il < g.n_layers) {
            double need = 0;
            for (int l = il; l < g.n_layers; ++l) need += layer_bytes[(size_t) l] / 1e9;
            err = "glm-dsa: " + std::to_string((long long) max_ctx) + " tokens of context do not fit the GPUs (" +
                  std::to_string(need) + " GB more needed) - a smaller --ctx / --max-context";
            return false;
        }
    }
    parts_.clear();
    layer_part_.assign((size_t) g.n_layers, 0);
    for (size_t pi = 0; pi < starts.size(); ++pi) {
        Part p;
        p.dev = devs[pi];
        p.l0 = starts[pi];
        p.l1 = pi + 1 < starts.size() ? starts[pi + 1] : g.n_layers;
        for (int il = p.l0; il < p.l1; ++il) layer_part_[(size_t) il] = (int) pi;
        parts_.push_back(p);
    }
    for (const Part& p : parts_) {
        double gb = 0;
        for (int il = p.l0; il < p.l1; ++il) gb += layer_bytes[(size_t) il] / 1e9;
        std::fprintf(stderr, "glm-dsa: device %d holds layers %d-%d (%.1f GB with their caches for %lld tokens)\n",
                     p.dev, p.l0, p.l1 - 1, gb, (long long) max_ctx);
    }

    // 4. the trunk's non-expert weights, each layer's on its part's device; the head on the last part's
    for (int il = 0; il < g.n_layers; ++il) {
        GD_CHECK(cudaSetDevice(parts_[(size_t) part_of(il)].dev));
        const std::string L = "blk." + std::to_string(il) + ".";
        for (const char* t : quant)
            if (!upload(L + t, true, err)) return false;
        for (const char* t : f32)
            if (!upload(L + t, false, err)) return false;
        if (il < g.dense_lead) {
            for (const char* t : dense)
                if (!upload(L + t, true, err)) return false;
        } else {
            for (const char* t : shexp)
                if (!upload(L + t, true, err)) return false;
            for (const char* t : router)
                if (!upload(L + t, false, err)) return false;
        }
        if ((il + 1) % 10 == 0 || il + 1 == g.n_layers) {
            std::fprintf(stderr, "glm-dsa: layer %d/%d uploaded\r", il + 1, g.n_layers);
            std::fflush(stderr);
        }
    }
    std::fprintf(stderr, "\n");
    GD_CHECK(cudaSetDevice(parts_.back().dev));
    if (!upload("output_norm.weight", false, err) || !upload("output.weight", true, err)) return false;
    tok_embd_ = find("token_embd.weight", &tok_embd_file_);
    if (!tok_embd_ || !ggml_get_type_traits((ggml_type) tok_embd_->type)->to_float) {
        err = "glm-dsa: token_embd.weight missing or of an undequantizable type";
        return false;
    }

    // 4. the routed experts: mmap views, one NativeFmt per layer
    // the draft (MTP) layer, index n_layers, loaded only with STRATA_GLM_MTP=1: measured on this box (EPYC 7K62,
    // RTX 5090, UD-Q4_K_XL) its drafts are accepted 67-77% of the time, but a verify window reads the union of two
    // tokens' experts from RAM (1.65x the bytes a token, measured over a story's routes), so speculative decoding
    // ran 240 ms a token against 190 without it - decode here is bound by RAM bandwidth, not by passes
    mtp_ = g.nextn > 0 && std::getenv("STRATA_GLM_MTP") && std::atoi(std::getenv("STRATA_GLM_MTP")) == 1 &&
           find("blk." + std::to_string(g.n_layers) + ".nextn.eh_proj.weight") != nullptr;
    const int n_moe_end = g.n_layers + (mtp_ ? 1 : 0);
    experts_.assign((size_t) n_moe_end, ExpertLayer{});
    for (int il = g.dense_lead; il < n_moe_end; ++il) {
        const std::string L = "blk." + std::to_string(il) + ".";
        const strata::GgufFile *fg = nullptr, *fu = nullptr, *fd = nullptr;
        const strata::TensorInfo* tg = find(L + "ffn_gate_exps.weight", &fg);
        const strata::TensorInfo* tu = find(L + "ffn_up_exps.weight", &fu);
        const strata::TensorInfo* td = find(L + "ffn_down_exps.weight", &fd);
        if (!tg || !tu || !td) {
            err = "glm-dsa: layer " + std::to_string(il) + " lacks its routed expert tensors";
            return false;
        }
        if (tg->type != tu->type) {
            err = "glm-dsa: layer " + std::to_string(il) + " gate and up experts differ in type";
            return false;
        }
        if (tg->shape.size() != 3 || (int) tg->shape[0] != g.n_embd || (int) tg->shape[1] != g.n_ff_exp ||
            (int) tg->shape[2] != g.n_expert || (int) td->shape[0] != g.n_ff_exp || (int) td->shape[1] != g.n_embd) {
            err = "glm-dsa: layer " + std::to_string(il) + " expert shapes differ from the metadata";
            return false;
        }
        auto* f = new strata::kernels::cpu::NativeFmt();
        if (!strata::kernels::cpu::native_fmt((int) tg->type, (int) td->type, g.n_embd, g.n_ff_exp, *f, err)) {
            delete f;
            return false;
        }
        ExpertLayer& e = experts_[(size_t) il];
        e.fmt = f;
        e.gu_type = (int) tg->type;
        e.d_type = (int) td->type;
        e.gate = fg->tensor_data(*tg);
        e.up = fu->tensor_data(*tu);
        e.down = fd->tensor_data(*td);
        e.gu_expert = f->gu_row * (size_t) g.n_ff_exp;
        e.d_expert = f->d_row * (size_t) g.n_embd;
    }

    // 5. the caches and the scratch, per part
    ik_slot_.assign((size_t) g.n_layers, -1);
    int64_t n_full = 0;
    for (int il = 0; il < g.n_layers; ++il)
        if (g.indexer_full[(size_t) il]) ik_slot_[(size_t) il] = n_full++;
    kvq_l_.assign((size_t) g.n_layers + 1, nullptr);
    kvs_l_.assign((size_t) g.n_layers + 1, nullptr);
    ik_l_.assign((size_t) g.n_layers + 1, nullptr);
    const size_t scratch = (size_t) 16 * g.n_embd + (size_t) 4 * g.n_head * g.qk_head() +
                           (size_t) 2 * g.n_head * g.kv_lora + (size_t) 4 * g.n_ff_dense +
                           (size_t) g.idx_heads * g.idx_key + (size_t) max_ctx + (size_t) g.n_vocab + 4096;
    const int max_in = std::max({g.n_embd, g.n_head * g.v_head, g.n_ff_dense, g.q_lora, g.n_ff_exp});
    for (Part& p : parts_) {
        GD_CHECK(cudaSetDevice(p.dev));
        // the attention cache in int8 (glm_dsa_kv.cuh): 612 bytes a token a layer instead of 2304; the indexer keys FP16
        const size_t n_l = (size_t) (p.l1 - p.l0);
        int8_t* q = nullptr;
        __half* sc = nullptr;
        GD_CHECK(cudaMalloc(&q, n_l * (size_t) max_ctx * g.cache_row()));
        allocs_.push_back(q);
        GD_CHECK(cudaMalloc(&sc, n_l * (size_t) max_ctx * (g.cache_row() / kv::kBlock) * sizeof(__half)));
        allocs_.push_back(sc);
        int n_fl = 0;
        for (int il = p.l0; il < p.l1; ++il) n_fl += g.indexer_full[(size_t) il] ? 1 : 0;
        __half* ik = nullptr;
        GD_CHECK(cudaMalloc(&ik, std::max<size_t>(1, (size_t) n_fl * (size_t) max_ctx * g.idx_key) * sizeof(__half)));
        allocs_.push_back(ik);
        int fi = 0;
        for (int il = p.l0; il < p.l1; ++il) {
            const size_t k = (size_t) (il - p.l0);
            kvq_l_[(size_t) il] = q + k * (size_t) max_ctx * g.cache_row();
            kvs_l_[(size_t) il] = sc + k * (size_t) max_ctx * (g.cache_row() / kv::kBlock);
            if (g.indexer_full[(size_t) il]) ik_l_[(size_t) il] = ik + (size_t) (fi++) * (size_t) max_ctx * g.idx_key;
        }
        GD_CHECK(cudaMalloc(&p.kvg, (size_t) std::max(g.idx_top_k, 1) * g.cache_row() * sizeof(float)));
        allocs_.push_back(p.kvg);
        GD_CHECK(cudaMalloc(&p.ikv, (size_t) g.idx_key * sizeof(float)));
        allocs_.push_back(p.ikv);
        GD_CHECK(cudaMalloc(&p.sc, scratch * sizeof(float)));
        allocs_.push_back(p.sc);
        GD_CHECK(cudaMalloc(&p.q8, strata::kernels::native_q8_1_bytes(max_in, 8)));   // (up to 8 columns: verify)
        allocs_.push_back(p.q8);
        GD_CHECK(cudaMalloc(&p.d_cells, (size_t) g.idx_top_k * sizeof(int)));
        allocs_.push_back(p.d_cells);
    }
    // the draft (MTP) layer on the last part, beside the head it shares: its weights, its int8 cache, its residual
    if (mtp_) {
        GD_CHECK(cudaSetDevice(parts_.back().dev));
        const std::string L = "blk." + std::to_string(g.n_layers) + ".";
        for (const char* t : quant)
            if (std::string(t).rfind("indexer.", 0) != 0 && !upload(L + t, true, err)) return false;
        for (const char* t : f32)
            if (std::string(t).rfind("indexer.", 0) != 0 && !upload(L + t, false, err)) return false;
        for (const char* t : shexp)
            if (!upload(L + t, true, err)) return false;
        for (const char* t : router)
            if (!upload(L + t, false, err)) return false;
        if (!upload(L + "nextn.eh_proj.weight", true, err) || !upload(L + "nextn.enorm.weight", false, err) ||
            !upload(L + "nextn.hnorm.weight", false, err))
            return false;
        // (shared_head_norm is optional in the format: the trunk's output_norm stands in)
        mtp_head_norm_ = find(L + "nextn.shared_head_norm.weight") ? L + "nextn.shared_head_norm.weight"
                                                                    : std::string("output_norm.weight");
        if (mtp_head_norm_ != "output_norm.weight" && !upload(mtp_head_norm_, false, err)) return false;
        int8_t* q = nullptr;
        __half* sc = nullptr;
        GD_CHECK(cudaMalloc(&q, (size_t) max_ctx * g.cache_row()));
        allocs_.push_back(q);
        GD_CHECK(cudaMalloc(&sc, (size_t) max_ctx * (g.cache_row() / kv::kBlock) * sizeof(__half)));
        allocs_.push_back(sc);
        kvq_l_[(size_t) g.n_layers] = q;
        kvs_l_[(size_t) g.n_layers] = sc;
        GD_CHECK(cudaMalloc(&h_last_, (size_t) g.n_embd * sizeof(float)));
        allocs_.push_back(h_last_);
        GD_CHECK(cudaMalloc(&mtp_buf_, (size_t) (2 * g.n_embd + 2 * g.n_embd + g.n_vocab) * sizeof(float)));
        allocs_.push_back(mtp_buf_);
        GD_CHECK(cudaMalloc(&logits_rows_, (size_t) 8 * g.n_vocab * sizeof(float)));
        allocs_.push_back(logits_rows_);
        GD_CHECK(cudaMalloc(&h_rows_, (size_t) 8 * g.n_embd * sizeof(float)));
        allocs_.push_back(h_rows_);
        std::fprintf(stderr, "glm-dsa: the MTP draft layer (blk.%d) loaded on device %d\n", g.n_layers,
                     parts_.back().dev);
    }
    // the head's buffers on the last part
    GD_CHECK(cudaSetDevice(parts_.back().dev));
    GD_CHECK(cudaMalloc(&d_tok_, sizeof(int)));
    allocs_.push_back(d_tok_);
    GD_CHECK(cudaMalloc(&d_logits_buf_, (size_t) g.n_vocab * sizeof(float)));
    allocs_.push_back(d_logits_buf_);
    cur_ = -1;
    for (int i = (int) parts_.size() - 1; i >= 0; --i) bind(i);   // (every part's members loaded once, then part 0)

    int threads = (int) std::thread::hardware_concurrency() / 2;
    if (const char* s = std::getenv("STRATA_GLM_THREADS")) threads = std::atoi(s);
    pool_.reset(new GlmCpuPool(std::max(1, threads)));
    h_x_.resize((size_t) g.n_embd);
    h_moe_.resize((size_t) g.n_embd);
    h_ff_.resize((size_t) g.n_exp_used * g.n_ff_exp);
    h_dn_.resize((size_t) g.n_exp_used * g.n_embd);
    h_act_.resize(strata::kernels::cpu::kNativeActBytes);
    h_hq_.resize((size_t) g.n_exp_used * strata::kernels::cpu::kNativeHBytes);
    h_embd_.resize((size_t) g.n_embd);

    // 6. warm the routed experts: every page read once, in parallel.  Decode touches ~600 random experts a token;
    //    a page that is not resident costs a disk read, and one resident in the file cache but not yet mapped
    //    into this process costs a soft fault - measured 340-1500 ms a token against 138 ms once all of them are
    //    mapped (glm_dsa_moe_bench, EPYC 7K62 8x DDR4-2933: 99.6 GB/s).  STRATA_GLM_WARM=0 skips it.
    const char* warm_env = std::getenv("STRATA_GLM_WARM");
    if (!warm_env || std::atoi(warm_env) != 0) {
        struct Span {
            const uint8_t* p;
            size_t n;
        };
        std::vector<Span> spans;
        constexpr size_t CH = (size_t) 64 << 20;
        size_t total = 0;
        for (const ExpertLayer& e : experts_) {
            if (!e.fmt) continue;
            const size_t gb = e.gu_expert * (size_t) g.n_expert, db = e.d_expert * (size_t) g.n_expert;
            const Span whole[3] = {{e.gate, gb}, {e.up, gb}, {e.down, db}};
            for (const Span& w : whole)
                for (size_t o = 0; o < w.n; o += CH) spans.push_back({w.p + o, std::min(CH, w.n - o)});
            total += 2 * gb + db;
        }
        std::atomic<uint64_t> sink{0}, done{0};
        std::atomic<int> shown{0};
        const auto w0 = std::chrono::steady_clock::now();
        pool_->run((int) spans.size(), [&](int i) {
            uint64_t s = 0;
            for (size_t o = 0; o < spans[(size_t) i].n; o += 4096) s += spans[(size_t) i].p[o];
            sink += s;
            const uint64_t d = done += spans[(size_t) i].n;
            const int pct = (int) (d * 10 / total);
            int prev = shown.load();
            while (pct > prev && !shown.compare_exchange_weak(prev, pct)) {}
            if (pct > prev) {
                const double sec =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
                std::fprintf(stderr, "glm-dsa: experts warm %d%% (%.0f of %.0f GB, %.0f s)\n", pct * 10, d / 1e9,
                             total / 1e9, sec);
            }
        });
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
        std::fprintf(stderr, "glm-dsa: %.0f GB of experts mapped in %.0f s (%.2f GB/s)%s\n", total / 1e9, sec,
                     total / 1e9 / sec, sink.load() == 1 ? " " : "");
    }
    pos_ = 0;
    return true;
}

bool GlmDsaModel::gemv(const DevW& w, const float* x, float* y, std::string& err) {
    // STRATA_GLM_F32_GEMV=1 (a check, not a mode): the quantized weights expanded and multiplied in FP32 exactly as
    // the batched path does, so the two paths can be compared without the Q8_1 activation rounding between them
    static const bool f32_gemv = std::getenv("STRATA_GLM_F32_GEMV") && std::atoi(std::getenv("STRATA_GLM_F32_GEMV"));
    if (f32_gemv && w.type != 0 && &w != &w_.at("output.weight")) {
        if (!pf_alloc(err)) return false;
        return gemm(w, x, y, 1, err);
    }
    if (w.type == 0) {
        gemv_f32_kernel<<<nblk(w.n_out * 32), T_>>>((const float*) w.d, x, y, (int) w.n_out, (int) w.n_in);
    } else {
        strata::kernels::quantize_q8_1_rows(x, 1, w.n_in, q8_, cudaStreamLegacy);
        strata::kernels::native_mmvq(w.type, w.d, q8_, y, (int) w.n_in, (int) w.n_out, 1, cudaStreamLegacy);
    }
    GD_CHECK(cudaGetLastError());
    return true;
}

bool GlmDsaModel::moe_cpu(int il, const std::vector<int>& ids, const std::vector<float>& rw, std::string& err) {
    namespace kc = strata::kernels::cpu;
    const GlmDsaGeometry& g = g_;
    const ExpertLayer& L = experts_[(size_t) il];
    const kc::NativeFmt& f = fmt_of(L.fmt);
    const int k = (int) ids.size();
    kc::native_quant_act(f, moe_x_ ? moe_x_ : h_x_.data(), h_act_.data());
    const void* act[1] = {h_act_.data()};

    // gate/up: k experts x (n_ff / RG) row chunks
    constexpr int RG = 64;
    const int gu_chunks = g.n_ff_exp / RG;
    pool_->run(k * gu_chunks, [&](int task) {
        const int i = task / gu_chunks, c = task % gu_chunks;
        const int e = ids[(size_t) i];
        float* ff[1] = {h_ff_.data() + (size_t) i * g.n_ff_exp};
        kc::native_gu_rows_split(f, L.gate + L.gu_expert * (size_t) e, L.up + L.gu_expert * (size_t) e, act, 1, ff,
                                 c * RG, (c + 1) * RG);
    });
    for (int i = 0; i < k; ++i)
        kc::native_quant_h(f, h_ff_.data() + (size_t) i * g.n_ff_exp, h_hq_.data() + (size_t) i * kc::kNativeHBytes);

    constexpr int RD = 128;
    const int d_chunks = g.n_embd / RD;
    pool_->run(k * d_chunks, [&](int task) {
        const int i = task / d_chunks, c = task % d_chunks;
        const int e = ids[(size_t) i];
        const void* hq[1] = {h_hq_.data() + (size_t) i * kc::kNativeHBytes};
        float* dn[1] = {h_dn_.data() + (size_t) i * g.n_embd};
        kc::native_down_rows_split(f, L.down + L.d_expert * (size_t) e, hq, 1, dn, c * RD, (c + 1) * RD);
    });
    // the weighted sum, in expert order (llama.cpp sums the k weighted experts the same way)
    for (int j = 0; j < g.n_embd; ++j) {
        float acc = 0.0f;
        for (int i = 0; i < k; ++i) acc += rw[(size_t) i] * h_dn_[(size_t) i * g.n_embd + j];
        (moe_y_ ? moe_y_ : h_moe_.data())[(size_t) j] = acc;
    }
    (void) err;
    return true;
}

bool GlmDsaModel::step(int32_t token, float* logits, std::string& err, bool head) {
    const GlmDsaGeometry& g = g_;
    if (pos_ >= max_ctx_) {
        err = "glm-dsa: context full (" + std::to_string(max_ctx_) + " tokens)";
        return false;
    }
    if (token < 0 || token >= (int32_t) tok_embd_->shape[1]) {
        err = "glm-dsa: token id " + std::to_string(token) + " out of range";
        return false;
    }
    // the graph decode unless STRATA_GLM_GRAPHS=0 (and not under the checks that read every op: the profiler,
    // the seam capture)
    static const bool graphs = !(std::getenv("STRATA_GLM_GRAPHS") && std::atoi(std::getenv("STRATA_GLM_GRAPHS")) == 0) &&
                               !(std::getenv("STRATA_GLM_PROFILE") && std::atoi(std::getenv("STRATA_GLM_PROFILE"))) &&
                               !(std::getenv("STRATA_GLM_F32_GEMV") && std::atoi(std::getenv("STRATA_GLM_F32_GEMV")));
    if (graphs && !capture_) return step_graph(token, logits, err, head);
    bind(0);
    const auto t0 = std::chrono::steady_clock::now();
    double cpu_ms = 0;
    const int p = (int) pos_;
    // decode attention/indexer on cuBLAS once the visible context passes STRATA_GLM_BLAS_MIN cells (1024): measured on
    // the RTX 5090, 3.3K cells 270 -> 242 ms a token; at ~60 cells the per-head kernels win (222 vs 249 ms).
    // STRATA_GLM_DECODE_BLAS=0 keeps the per-head kernels everywhere.
    static const bool blas_on = !(std::getenv("STRATA_GLM_DECODE_BLAS") && std::atoi(std::getenv("STRATA_GLM_DECODE_BLAS")) == 0);
    static const int blas_min = std::getenv("STRATA_GLM_BLAS_MIN") ? std::atoi(std::getenv("STRATA_GLM_BLAS_MIN")) : 1024;
    const bool dec_blas = blas_on && p + 1 > blas_min;
    // STRATA_GLM_PROFILE=1: the device synchronised at each section's end, the time added to its bucket
    static const bool profile = std::getenv("STRATA_GLM_PROFILE") && std::atoi(std::getenv("STRATA_GLM_PROFILE"));
    auto mark = std::chrono::steady_clock::now();
    const auto prof = [&](int s) {
        if (!profile) return;
        cudaDeviceSynchronize();
        const auto n = std::chrono::steady_clock::now();
        prof_acc_[s] += std::chrono::duration<double, std::milli>(n - mark).count();
        mark = n;
    };

    // scratch carve-up
    float *x, *xn, *tmp, *moe, *sh, *router, *rw, *qr, *kvpe, *q, *qabs, *o, *gate, *up, *hh, *iq, *iw, *score, *d_logits;
    int *ids;
    const auto carve = [&]() {
        x = sc_;
        xn = x + g.n_embd;
        tmp = xn + g.n_embd;
        moe = tmp + g.n_embd;
        sh = moe + g.n_embd;
        router = sh + g.n_embd;
        rw = router + 4 * g.n_embd;
        ids = (int*) (rw + 64);
        qr = (float*) (ids + 64);
        kvpe = qr + g.q_lora;
        q = kvpe + g.cache_row();
        qabs = q + (size_t) g.n_head * g.qk_head();
        o = qabs + (size_t) g.n_head * g.kv_lora;
        gate = o + (size_t) g.n_head * g.v_head;
        up = gate + g.n_ff_dense;
        hh = up + g.n_ff_dense;
        iq = hh + g.n_ff_dense;
        iw = iq + (size_t) g.idx_heads * g.idx_key;
        score = iw + 64;
        d_logits = d_logits_buf_;
    };
    carve();

    // the embedding row, dequantized on the host from the mmap
    {
        const size_t row = ggml_row_size((ggml_type) tok_embd_->type, g.n_embd);
        ggml_get_type_traits((ggml_type) tok_embd_->type)
            ->to_float(tok_embd_file_->tensor_data(*tok_embd_) + row * (size_t) token, h_embd_.data(), g.n_embd);
        GD_CHECK(cudaMemcpy(x, h_embd_.data(), (size_t) g.n_embd * sizeof(float), cudaMemcpyHostToDevice));
    }

    const float theta_scale = std::pow(g.rope_base, -2.0f / (float) g.qk_rope);
    const float kq_scale = 1.0f / std::sqrt((float) g.qk_head());
    const int n_vis = p + 1;
    int n_sel = n_vis;
    bool sparse = false;   // the selection is a subset (d_cells_) rather than every visible cell

    prof(7);
    for (int il = 0; il < g.n_layers; ++il) {
        if (part_of(il) != cur_) {   // the layer split: the residual to the next device, its selection too
            const float* x_old = x;
            const int* cells_old = d_cells_;
            GD_CHECK(cudaDeviceSynchronize());
            bind(part_of(il));
            carve();
            if (!hand_over(x_old, x, 1, err)) return false;
            if (sparse)
                GD_CHECK(cudaMemcpy(d_cells_, cells_old, (size_t) g.idx_top_k * sizeof(int), cudaMemcpyDefault));
        }
        // ---- attention
        rms_norm_kernel<<<1, T_>>>(x, (const float*) W(il, "attn_norm.weight").d, xn, g.n_embd, g.norm_eps);
        if (!gemv(W(il, "attn_q_a.weight"), xn, qr, err)) return false;
        rms_norm_kernel<<<1, T_>>>(qr, (const float*) W(il, "attn_q_a_norm.weight").d, qr, g.q_lora, g.norm_eps);
        if (!gemv(W(il, "attn_kv_a_mqa.weight"), xn, kvpe, err)) return false;
        rms_norm_kernel<<<1, T_>>>(kvpe, (const float*) W(il, "attn_kv_a_norm.weight").d, kvpe, g.kv_lora,
                                   g.norm_eps);
        rope_norm_kernel<<<nblk(g.qk_rope / 2), T_>>>(kvpe, 1, g.cache_row(), g.kv_lora, g.qk_rope, p, theta_scale);
        kv::store_q8(kvpe, 1, g.cache_row(), g.cache_row(), kv_q(il), (__half*) kv_s(il), p, nullptr, nullptr);

        // the lightning indexer: keys are cached on full layers; the selection matters only past top_k cells
        prof(0);
        if (g.indexer_full[(size_t) il]) {
            if (!gemv(W(il, "indexer.attn_k.weight"), xn, ikv_, err)) return false;
            layer_norm_kernel<<<1, 128>>>(ikv_, (const float*) W(il, "indexer.k_norm.weight").d,
                                          (const float*) W(il, "indexer.k_norm.bias").d, ikv_, g.idx_key,
                                          g.idx_norm_eps);
            rope_norm_kernel<<<nblk(g.qk_rope / 2), T_>>>(ikv_, 1, g.idx_key, 0, g.qk_rope, p, theta_scale);
            kv::store_f16(ikv_, 1, g.idx_key, g.idx_key, (__half*) ik_keys(il), p, nullptr, nullptr);
            if (n_vis > g.idx_top_k) {
                if (!gemv(W(il, "indexer.attn_q_b.weight"), qr, iq, err)) return false;
                rope_norm_kernel<<<nblk(g.idx_heads * g.qk_rope / 2), T_>>>(iq, g.idx_heads, g.idx_key, 0, g.qk_rope,
                                                                            p, theta_scale);
                if (!gemv(W(il, "indexer.proj.weight"), xn, iw, err)) return false;
                scale_kernel<<<1, T_>>>(iw, 1.0f / std::sqrt((float) (g.idx_key * g.idx_heads)), g.idx_heads);
                if (!select_cells(il, n_vis, iq, iw, score, err)) return false;
                n_sel = g.idx_top_k;
                sparse = true;
            } else {
                n_sel = n_vis;
                sparse = false;
            }
        }

        prof(1);
        if (!gemv(W(il, "attn_q_b.weight"), qr, q, err)) return false;
        rope_norm_kernel<<<nblk(g.n_head * g.qk_rope / 2), T_>>>(q, g.n_head, g.qk_head(), g.qk_nope, g.qk_rope, p,
                                                                 theta_scale);
        prof(8);
        q_absorb_kernel<<<nblk((int64_t) g.n_head * g.kv_lora * 32), T_>>>(
            (const float*) W(il, "attn_k_b.weight").d, q, qabs, g.qk_nope, g.qk_head(), g.kv_lora, g.n_head);
        prof(9);
        if (!attend_one(il, n_sel, sparse, dec_blas, q, qabs, o, nullptr, err)) return false;
        GD_CHECK(cudaGetLastError());
        prof(2);
        if (!gemv(W(il, "attn_output.weight"), o, tmp, err)) return false;
        add_kernel<<<nblk(g.n_embd), T_>>>(x, tmp, g.n_embd);
        capture(il, 0, x);

        // ---- the FFN
        prof(3);
        rms_norm_kernel<<<1, T_>>>(x, (const float*) W(il, "ffn_norm.weight").d, xn, g.n_embd, g.norm_eps);
        if (il < g.dense_lead) {
            if (!gemv(W(il, "ffn_gate.weight"), xn, gate, err) || !gemv(W(il, "ffn_up.weight"), xn, up, err))
                return false;
            strata::kernels::glm_swiglu_clamp(gate, up, INFINITY, g.n_ff_dense, hh, cudaStreamLegacy);
            if (!gemv(W(il, "ffn_down.weight"), hh, tmp, err)) return false;
            add_kernel<<<nblk(g.n_embd), T_>>>(x, tmp, g.n_embd);
        } else {
            if (!gemv(W(il, "ffn_gate_inp.weight"), xn, router, err)) return false;
            strata::kernels::router_sigmoid(router, (const float*) W(il, "exp_probs_b.bias").d, 1, g.n_expert,
                                            g.n_exp_used, g.w_scale, g.norm_w != 0, ids, rw, cudaStreamLegacy);
            std::vector<int> h_ids((size_t) g.n_exp_used);
            std::vector<float> h_rw((size_t) g.n_exp_used);
            prof(10);
            GD_CHECK(cudaMemcpy(h_ids.data(), ids, h_ids.size() * sizeof(int), cudaMemcpyDeviceToHost));
            GD_CHECK(cudaMemcpy(h_rw.data(), rw, h_rw.size() * sizeof(float), cudaMemcpyDeviceToHost));
            GD_CHECK(cudaMemcpy(h_x_.data(), xn, (size_t) g.n_embd * sizeof(float), cudaMemcpyDeviceToHost));
            capture_ids(il, h_ids.data(), h_rw.data());
            // the shared expert runs on the device while the CPU does the routed ones
            prof(11);
            if (!gemv(W(il, "ffn_gate_shexp.weight"), xn, gate, err) ||
                !gemv(W(il, "ffn_up_shexp.weight"), xn, up, err))
                return false;
            strata::kernels::glm_swiglu_clamp(gate, up, INFINITY, (int64_t) g.n_ff_exp * g.n_shared, hh,
                                              cudaStreamLegacy);
            if (!gemv(W(il, "ffn_down_shexp.weight"), hh, sh, err)) return false;
            prof(4);
            const auto c0 = std::chrono::steady_clock::now();
            if (!moe_cpu(il, h_ids, h_rw, err)) return false;
            cpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
            prof(5);
            GD_CHECK(cudaMemcpy(moe, h_moe_.data(), (size_t) g.n_embd * sizeof(float), cudaMemcpyHostToDevice));
            add_kernel<<<nblk(g.n_embd), T_>>>(moe, sh, g.n_embd);   // moe_out + shexp, then the residual
            add_kernel<<<nblk(g.n_embd), T_>>>(x, moe, g.n_embd);
        }
        prof(6);
        capture(il, 1, x);
    }

    if (logits || head) {
        rms_norm_kernel<<<1, T_>>>(x, (const float*) w_.at("output_norm.weight").d, xn, g.n_embd, g.norm_eps);
        if (h_last_) GD_CHECK(cudaMemcpy(h_last_, xn, (size_t) g.n_embd * sizeof(float), cudaMemcpyDeviceToDevice));
        if (!gemv(w_.at("output.weight"), xn, d_logits, err)) return false;
        d_logits_ = d_logits;
        if (logits)
            GD_CHECK(cudaMemcpy(logits, d_logits, (size_t) g.n_vocab * sizeof(float), cudaMemcpyDeviceToHost));
    }
    prof(7);
    GD_CHECK(cudaDeviceSynchronize());
    ++pos_;
    if (profile && ++prof_n_ % 16 == 0) {
        static const char* names[12] = {"attn-in", "indexer", "attend", "out-proj", "shexp", "cpu-experts", "moe-tail", "embd+head", "q_b+rope", "absorb", "router", "d2h"};
        std::fprintf(stderr, "glm-dsa profile (ms/token over %d):", prof_n_);
        for (int i = 0; i < 12; ++i) std::fprintf(stderr, " %s %.1f", names[i], prof_acc_[i] / prof_n_);
        std::fprintf(stderr, "\n");
    }
    const double total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    t_total_ += total;
    t_cpu_ += cpu_ms;
    t_dev_ += total - cpu_ms;
    return true;
}

int8_t* GlmDsaModel::kv_q(int il) const { return kvq_l_[(size_t) il]; }
void* GlmDsaModel::kv_s(int il) const { return kvs_l_[(size_t) il]; }
void* GlmDsaModel::ik_keys(int il) const { return ik_l_[(size_t) il]; }

void GlmDsaModel::bind(int i) {
    if (i == cur_) {
        cudaSetDevice(cur_dev_);
        return;
    }
    if (cur_ >= 0) {   // stash what the current part allocated since it was bound
        Part& o = parts_[(size_t) cur_];
        o.sc = sc_, o.q8 = q8_, o.d_cells = d_cells_, o.kvg = kvg_, o.ikv = ikv_, o.cublas = cublas_;
        o.pf = pf_, o.deq = deq_, o.att = att_, o.pf_score = pf_score_, o.sg = sg_, o.ss = ss_, o.qc = qc_;
        o.d_sel = d_sel_, o.d_cnt = d_cnt_, o.iq16 = iq16_, o.gstream = gstream_, o.pin = pin_, o.d_pos = d_pos_;
        o.dq_ctx = dq_ctx_, o.dq_xq = dq_xq_, o.dq_bounds = dq_bounds_, o.dq_iota = dq_iota_;
    }
    const Part& p = parts_[(size_t) i];
    sc_ = p.sc, q8_ = p.q8, d_cells_ = p.d_cells, kvg_ = p.kvg, ikv_ = p.ikv, cublas_ = p.cublas;
    pf_ = p.pf, deq_ = p.deq, att_ = p.att, pf_score_ = p.pf_score, sg_ = p.sg, ss_ = p.ss, qc_ = p.qc;
    d_sel_ = p.d_sel, d_cnt_ = p.d_cnt, iq16_ = p.iq16, gstream_ = p.gstream, pin_ = p.pin, d_pos_ = p.d_pos;
    dq_ctx_ = p.dq_ctx, dq_xq_ = p.dq_xq, dq_bounds_ = p.dq_bounds, dq_iota_ = p.dq_iota;
    cur_ = i;
    cur_dev_ = p.dev;
    cudaSetDevice(cur_dev_);
}

/// The residual into the next part: everything queued on the old device finishes, then a peer copy (the driver
/// routes it through the host when the cards have no peer access).
bool GlmDsaModel::hand_over(const float* x_from, float* x_to, int64_t rows, std::string& err) {
    (void) err;
    return cudaMemcpy(x_to, x_from, (size_t) rows * g_.n_embd * sizeof(float), cudaMemcpyDefault) == cudaSuccess;
}

/// The indexer's selection for one decode token: its scores over the n_vis cached keys, the top_k of them chosen on
/// the host (ties to the lower cell, as llama.cpp's top_k), ascending, into d_cells_.
bool GlmDsaModel::select_cells(int il, int n_vis, const float* iq, const float* iw, float* score, std::string& err) {
    const GlmDsaGeometry& g = g_;
    if (!decode_index_scores(ik_keys(il), iq, iw, score, n_vis, err)) return false;
    // on the device (glm_dsa_kv.cuh topk_rows); STRATA_GLM_TOPK_HOST=1: the host's nth_element, the reference
    static const bool on_host = std::getenv("STRATA_GLM_TOPK_HOST") && std::atoi(std::getenv("STRATA_GLM_TOPK_HOST"));
    if (!on_host) {
        kv::topk_rows(score, 0, 1, n_vis, 0, g.idx_top_k, d_cells_, 0, nullptr, nullptr);
        GD_CHECK(cudaGetLastError());
        return true;
    }
    h_score_.resize((size_t) n_vis);
    GD_CHECK(cudaMemcpy(h_score_.data(), score, (size_t) n_vis * sizeof(float), cudaMemcpyDeviceToHost));
    cells_.resize((size_t) n_vis);
    std::iota(cells_.begin(), cells_.end(), 0);
    std::nth_element(cells_.begin(), cells_.begin() + g.idx_top_k, cells_.end(), [&](int a, int b) {
        return h_score_[(size_t) a] > h_score_[(size_t) b] || (h_score_[(size_t) a] == h_score_[(size_t) b] && a < b);
    });
    cells_.resize((size_t) g.idx_top_k);
    std::sort(cells_.begin(), cells_.end());
    GD_CHECK(cudaMemcpy(d_cells_, cells_.data(), cells_.size() * sizeof(int), cudaMemcpyHostToDevice));
    return true;
}

/// One decode token's attention over its n_sel cells of layer il (d_cells_ when sparse, else 0..n_sel-1): the
/// per-head int8 kernel for a short dense context, else the cells dequantized into kvg_ and the cuBLAS path.
bool GlmDsaModel::attend_one(int il, int n_sel, bool sparse, bool blas, const float* q, float* qabs, float* o,
                             void* stream, std::string& err) {
    const GlmDsaGeometry& g = g_;
    const float* vb = (const float*) W(il, "attn_v_b.weight").d;
    if (!blas && !sparse) {
        kv::attend_small_q8(qabs, q, kv_q(il), (const __half*) kv_s(il), nullptr, n_sel, vb, o, g.n_head, g.kv_lora,
                            g.qk_nope, g.qk_rope, g.v_head, 1.0f / std::sqrt((float) g.qk_head()),
                            (cudaStream_t) stream);
    } else {
        // (the legacy stream orders after `stream`'s earlier work: both are blocking streams)
        kv::gather_q8(kv_q(il), (const __half*) kv_s(il), g.cache_row(), sparse ? d_cells_ : nullptr, 0, n_sel, kvg_,
                      nullptr);
        if (!decode_attend(kvg_, n_sel - 1, q, qabs, o, vb, false, err)) return false;
    }
    GD_CHECK(cudaGetLastError());
    return true;
}

/// The draft (MTP) layer at position pos() for `token`, as llama.cpp's glm-dsa graph_mtp builds it:
///   x = eh_proj . [enorm * rms(embed(token)) | hnorm * rms(h)]   (h: the trunk's post-norm hidden state)
///   then one decoder block - MLA attention over the MTP layer's own cache (dense: the indexer is not used), the
///   sigmoid-routed MoE with the shared expert - then shared_head_norm and the shared head.  Returns its argmax.
int GlmDsaModel::mtp_draft(int32_t token, std::string& err, int64_t at) {
    const GlmDsaGeometry& g = g_;
    if (!mtp_) {
        err = "glm-dsa: the MTP layer is not loaded";
        return -1;
    }
    const int il = g.n_layers, p = (int) (at >= 0 ? at : pos_), E = g.n_embd;
    bind((int) parts_.size() - 1);
    // the step path's scratch (its residual is not needed once the trunk's head has run) and the MTP's own
    float* xn = sc_ + E;
    float* tmp = xn + E;
    float* moe = tmp + E;
    float* sh = moe + E;
    float* router = sh + E;
    float* rw = router + 4 * E;
    int* ids = (int*) (rw + 64);
    float* qr = (float*) (ids + 64);
    float* kvpe = qr + g.q_lora;
    float* q = kvpe + g.cache_row();
    float* qabs = q + (size_t) g.n_head * g.qk_head();
    float* o = qabs + (size_t) g.n_head * g.kv_lora;
    float* gate = o + (size_t) g.n_head * g.v_head;
    float* up = gate + g.n_ff_dense;
    float* hh = up + g.n_ff_dense;
    float* x = mtp_buf_;
    float* cat = x + E;
    float* lg = cat + 3 * E;
    const auto fail = [&](const std::string& e) {
        err = e;
        return -1;
    };
#define MTP_CK(call)                                                                                    \
    do {                                                                                                \
        const cudaError_t e_ = (call);                                                                  \
        if (e_ != cudaSuccess) return fail(std::string("cuda: ") + #call + ": " + cudaGetErrorString(e_)); \
    } while (0)
    {
        const size_t row = ggml_row_size((ggml_type) tok_embd_->type, E);
        ggml_get_type_traits((ggml_type) tok_embd_->type)
            ->to_float(tok_embd_file_->tensor_data(*tok_embd_) + row * (size_t) token, h_embd_.data(), E);
        MTP_CK(cudaMemcpy(cat, h_embd_.data(), (size_t) E * sizeof(float), cudaMemcpyHostToDevice));
    }
    rms_norm_kernel<<<1, T_>>>(cat, (const float*) W(il, "nextn.enorm.weight").d, cat, E, g.norm_eps);
    rms_norm_kernel<<<1, T_>>>(h_last_, (const float*) W(il, "nextn.hnorm.weight").d, cat + E, E, g.norm_eps);
    if (!gemv(W(il, "nextn.eh_proj.weight"), cat, x, err)) return -1;

    const float theta_scale = std::pow(g.rope_base, -2.0f / (float) g.qk_rope);
    static const int blas_min = std::getenv("STRATA_GLM_BLAS_MIN") ? std::atoi(std::getenv("STRATA_GLM_BLAS_MIN")) : 1024;
    rms_norm_kernel<<<1, T_>>>(x, (const float*) W(il, "attn_norm.weight").d, xn, E, g.norm_eps);
    if (!gemv(W(il, "attn_q_a.weight"), xn, qr, err)) return -1;
    rms_norm_kernel<<<1, T_>>>(qr, (const float*) W(il, "attn_q_a_norm.weight").d, qr, g.q_lora, g.norm_eps);
    if (!gemv(W(il, "attn_kv_a_mqa.weight"), xn, kvpe, err)) return -1;
    rms_norm_kernel<<<1, T_>>>(kvpe, (const float*) W(il, "attn_kv_a_norm.weight").d, kvpe, g.kv_lora, g.norm_eps);
    rope_norm_kernel<<<nblk(g.qk_rope / 2), T_>>>(kvpe, 1, g.cache_row(), g.kv_lora, g.qk_rope, p, theta_scale);
    kv::store_q8(kvpe, 1, g.cache_row(), g.cache_row(), kv_q(il), (__half*) kv_s(il), p, nullptr, nullptr);
    if (!gemv(W(il, "attn_q_b.weight"), qr, q, err)) return -1;
    rope_norm_kernel<<<nblk(g.n_head * g.qk_rope / 2), T_>>>(q, g.n_head, g.qk_head(), g.qk_nope, g.qk_rope, p,
                                                             theta_scale);
    q_absorb_kernel<<<nblk((int64_t) g.n_head * g.kv_lora * 32), T_>>>((const float*) W(il, "attn_k_b.weight").d, q,
                                                                      qabs, g.qk_nope, g.qk_head(), g.kv_lora, g.n_head);
    if (!attend_one(il, p + 1, false, p + 1 > blas_min, q, qabs, o, nullptr, err)) return -1;
    if (!gemv(W(il, "attn_output.weight"), o, tmp, err)) return -1;
    add_kernel<<<nblk(E), T_>>>(x, tmp, E);

    rms_norm_kernel<<<1, T_>>>(x, (const float*) W(il, "ffn_norm.weight").d, xn, E, g.norm_eps);
    if (!gemv(W(il, "ffn_gate_inp.weight"), xn, router, err)) return -1;
    strata::kernels::router_sigmoid(router, (const float*) W(il, "exp_probs_b.bias").d, 1, g.n_expert, g.n_exp_used,
                                    g.w_scale, g.norm_w != 0, ids, rw, cudaStreamLegacy);
    std::vector<int> h_ids((size_t) g.n_exp_used);
    std::vector<float> h_rw((size_t) g.n_exp_used);
    MTP_CK(cudaMemcpy(h_ids.data(), ids, h_ids.size() * sizeof(int), cudaMemcpyDeviceToHost));
    MTP_CK(cudaMemcpy(h_rw.data(), rw, h_rw.size() * sizeof(float), cudaMemcpyDeviceToHost));
    MTP_CK(cudaMemcpy(h_x_.data(), xn, (size_t) E * sizeof(float), cudaMemcpyDeviceToHost));
    if (!gemv(W(il, "ffn_gate_shexp.weight"), xn, gate, err) || !gemv(W(il, "ffn_up_shexp.weight"), xn, up, err))
        return -1;
    strata::kernels::glm_swiglu_clamp(gate, up, INFINITY, (int64_t) g.n_ff_exp * g.n_shared, hh, cudaStreamLegacy);
    if (!gemv(W(il, "ffn_down_shexp.weight"), hh, sh, err)) return -1;
    if (!moe_cpu(il, h_ids, h_rw, err)) return -1;
    MTP_CK(cudaMemcpy(moe, h_moe_.data(), (size_t) E * sizeof(float), cudaMemcpyHostToDevice));
    add_kernel<<<nblk(E), T_>>>(moe, sh, E);
    add_kernel<<<nblk(E), T_>>>(x, moe, E);

    rms_norm_kernel<<<1, T_>>>(x, (const float*) w_.at(mtp_head_norm_).d, xn, E, g.norm_eps);
    if (!gemv(w_.at("output.weight"), xn, lg, err)) return -1;
    strata::kernels::SamplerParams sp;
    sp.greedy = true;
    strata::kernels::sample_tokens(lg, 1, g.n_vocab, nullptr, 0, sp, d_tok_, cudaStreamLegacy);
    int tok = -1;
    MTP_CK(cudaMemcpy(&tok, d_tok_, sizeof(int), cudaMemcpyDeviceToHost));
#undef MTP_CK
    return tok;
}

bool GlmDsaModel::verify(const int32_t* toks, int T, std::string& err) {
    if (T < 1 || T > 8 || !logits_rows_) {
        err = "glm-dsa: verify takes 1-8 tokens (and the MTP layer loaded)";
        return false;
    }
    if (pos_ + T > max_ctx_) {
        err = "glm-dsa: context full";
        return false;
    }
    bind(0);
    if (!pf_alloc(err)) return false;
    all_rows_ = true;
    const bool ok = prefill_chunk(toks, T, true, err);
    all_rows_ = false;
    return ok;
}

int GlmDsaModel::sample_row(int r, const strata::kernels::SamplerParams& sp, std::string& err) {
    if (cudaSetDevice(parts_.back().dev) != cudaSuccess) {
        err = "glm-dsa: cudaSetDevice failed";
        return -1;
    }
    strata::kernels::SamplerParams p = sp;
    if (p.temperature <= 0.0f) p.greedy = true;
    strata::kernels::sample_tokens(logits_rows_ + (size_t) r * g_.n_vocab, 1, g_.n_vocab, nullptr, 0, p, d_tok_,
                                   cudaStreamLegacy);
    int tok = -1;
    const cudaError_t e = cudaMemcpy(&tok, d_tok_, sizeof(int), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        err = std::string("glm-dsa: sample_row: ") + cudaGetErrorString(e);
        return -1;
    }
    return tok;
}

bool GlmDsaModel::use_h_row(int r, std::string& err) {
    GD_CHECK(cudaSetDevice(parts_.back().dev));
    GD_CHECK(cudaMemcpy(h_last_, h_rows_ + (size_t) r * g_.n_embd, (size_t) g_.n_embd * sizeof(float),
                        cudaMemcpyDeviceToDevice));
    return true;
}

bool GlmDsaModel::read_logits(float* out, std::string& err) {
    if (!d_logits_) {
        err = "glm-dsa: no head has run";
        return false;
    }
    GD_CHECK(cudaSetDevice(parts_.back().dev));
    GD_CHECK(cudaMemcpy(out, d_logits_, (size_t) g_.n_vocab * sizeof(float), cudaMemcpyDeviceToHost));
    return true;
}

int GlmDsaModel::sample(const strata::kernels::SamplerParams& sp, std::string& err) {
    if (!d_logits_) {
        err = "glm-dsa: sample() before any head ran";
        return -1;
    }
    if (cudaSetDevice(parts_.back().dev) != cudaSuccess) {
        err = "glm-dsa: cudaSetDevice failed";
        return -1;
    }
    strata::kernels::SamplerParams p = sp;
    if (p.temperature <= 0.0f) p.greedy = true;
    strata::kernels::sample_tokens(d_logits_, 1, g_.n_vocab, nullptr, 0, p, d_tok_, cudaStreamLegacy);
    int tok = -1;
    const cudaError_t e = cudaMemcpy(&tok, d_tok_, sizeof(int), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        err = std::string("glm-dsa: sample: ") + cudaGetErrorString(e);
        return -1;
    }
    return tok;
}

void GlmDsaModel::take_timing(double& dev_ms, double& cpu_ms, double& total_ms) {
    dev_ms = t_dev_;
    cpu_ms = t_cpu_;
    total_ms = t_total_;
    t_dev_ = t_cpu_ = t_total_ = 0;
}

// ================================ the graph decode ================================
//
// step() issues ~25 launches and 4 synchronous copies a layer; on this Windows box (WDDM) every one of them costs
// host latency the GPU waits through, ~75 ms a token for ~14 ms of weight reads.  step_graph() runs the same
// kernels in the same order, recorded once per layer into CUDA graphs on a stream of its own:
//   A  attention inputs: norms, q_a / kv_a, the cache row, the indexer key (full layers) and its query/weights, q, qabs
//   -  [outside: the indexer selection past top_k cells, then the attention - their sizes change every token]
//   B  out-proj + residual, ffn_norm, then the dense FFN + residual, or the router with its ids/weights and the
//      normed activation copied to pinned host memory
//   S  the shared expert (runs while the CPU does the routed ones)
//   C  the routed result back from pinned memory, + shared, + residual
// The position the kernels need is read from the device (d_pos_), so one recording serves every token.

bool GlmDsaModel::gemv_s(const DevW& w, const float* x, float* y, void* stream, std::string& err) {
    cudaStream_t s = (cudaStream_t) stream;
    if (w.type == 0) {
        gemv_f32_kernel<<<nblk(w.n_out * 32), T_, 0, s>>>((const float*) w.d, x, y, (int) w.n_out, (int) w.n_in);
    } else {
        strata::kernels::quantize_q8_1_rows(x, 1, w.n_in, q8_, s);
        strata::kernels::native_mmvq(w.type, w.d, q8_, y, (int) w.n_in, (int) w.n_out, 1, s);
    }
    GD_CHECK(cudaGetLastError());
    return true;
}

bool GlmDsaModel::build_graphs(std::string& err) {
    const GlmDsaGeometry& g = g_;
    // every part: a stream, pinned words and the device position of its own
    for (int pi = 0; pi < (int) parts_.size(); ++pi) {
        bind(pi);
        cudaStream_t ps = nullptr;
        GD_CHECK(cudaStreamCreate(&ps));
        gstream_ = ps;
        GD_CHECK(cudaHostAlloc((void**) &pin_, (size_t) (128 + 3 * g.n_embd + 64) * sizeof(float), cudaHostAllocPortable));
        GD_CHECK(cudaMalloc(&d_pos_, sizeof(int)));
        allocs_.push_back(d_pos_);
    }
    bind(0);
    cudaStream_t s = (cudaStream_t) gstream_;

    // the step path's scratch carve-up (the same buffers, so the two paths are interchangeable mid-sequence)
    float *x, *xn, *tmp, *moe, *sh, *router, *rw, *qr, *kvpe, *q, *qabs, *o, *gate, *up, *hh, *iq, *iw, *h_rwids, *h_x, *h_moe;
    int *ids;
    const auto carve = [&]() {
        x = sc_;
        xn = x + g.n_embd;
        tmp = xn + g.n_embd;
        moe = tmp + g.n_embd;
        sh = moe + g.n_embd;
        router = sh + g.n_embd;
        rw = router + 4 * g.n_embd;
        ids = (int*) (rw + 64);
        qr = (float*) (ids + 64);
        kvpe = qr + g.q_lora;
        q = kvpe + g.cache_row();
        qabs = q + (size_t) g.n_head * g.qk_head();
        o = qabs + (size_t) g.n_head * g.kv_lora;
        gate = o + (size_t) g.n_head * g.v_head;
        up = gate + g.n_ff_dense;
        hh = up + g.n_ff_dense;
        iq = hh + g.n_ff_dense;
        iw = iq + (size_t) g.idx_heads * g.idx_key;
        h_rwids = pin_;
        h_x = pin_ + 128;
        h_moe = h_x + g.n_embd;
    };
    carve();

    const float theta_scale = std::pow(g.rope_base, -2.0f / (float) g.qk_rope);
    const auto record = [&](void*& out, const std::function<bool()>& body) -> bool {
        GD_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        const bool ok = body();
        cudaGraph_t graph = nullptr;
        const cudaError_t e = cudaStreamEndCapture(s, &graph);
        if (!ok) return false;
        if (e != cudaSuccess) {
            err = std::string("glm-dsa graphs: capture failed: ") + cudaGetErrorString(e);
            return false;
        }
        cudaGraphExec_t ex = nullptr;
        GD_CHECK(cudaGraphInstantiate(&ex, graph, 0));
        cudaGraphDestroy(graph);
        out = ex;
        return true;
    };

    gA_.assign((size_t) g.n_layers, nullptr);
    gB_.assign((size_t) g.n_layers, nullptr);
    gS_.assign((size_t) g.n_layers, nullptr);
    gC_.assign((size_t) g.n_layers, nullptr);
    for (int il = 0; il < g.n_layers; ++il) {
        if (part_of(il) != cur_) {   // the layer split: this layer's part, its stream and buffers
            bind(part_of(il));
            carve();
            s = (cudaStream_t) gstream_;
        }
        const bool full = g.indexer_full[(size_t) il];
        if (!record(gA_[(size_t) il], [&]() -> bool {
                rms_norm_kernel<<<1, T_, 0, s>>>(x, (const float*) W(il, "attn_norm.weight").d, xn, g.n_embd,
                                                 g.norm_eps);
                if (!gemv_s(W(il, "attn_q_a.weight"), xn, qr, s, err)) return false;
                rms_norm_kernel<<<1, T_, 0, s>>>(qr, (const float*) W(il, "attn_q_a_norm.weight").d, qr, g.q_lora,
                                                 g.norm_eps);
                if (!gemv_s(W(il, "attn_kv_a_mqa.weight"), xn, kvpe, s, err)) return false;
                rms_norm_kernel<<<1, T_, 0, s>>>(kvpe, (const float*) W(il, "attn_kv_a_norm.weight").d, kvpe,
                                                 g.kv_lora, g.norm_eps);
                rope_dev_kernel<<<nblk(g.qk_rope / 2), T_, 0, s>>>(kvpe, 1, g.cache_row(), g.kv_lora, g.qk_rope,
                                                                   d_pos_, theta_scale);
                kv::store_q8(kvpe, 1, g.cache_row(), g.cache_row(), kv_q(il), (__half*) kv_s(il), 0, d_pos_, s);
                if (full) {
                    if (!gemv_s(W(il, "indexer.attn_k.weight"), xn, ikv_, s, err)) return false;
                    layer_norm_kernel<<<1, 128, 0, s>>>(ikv_, (const float*) W(il, "indexer.k_norm.weight").d,
                                                        (const float*) W(il, "indexer.k_norm.bias").d, ikv_, g.idx_key,
                                                        g.idx_norm_eps);
                    rope_dev_kernel<<<nblk(g.qk_rope / 2), T_, 0, s>>>(ikv_, 1, g.idx_key, 0, g.qk_rope, d_pos_,
                                                                       theta_scale);
                    kv::store_f16(ikv_, 1, g.idx_key, g.idx_key, (__half*) ik_keys(il), 0, d_pos_, s);
                    // the query side of the selection: cheap (2 small GEMVs), so always computed and the graph
                    // stays one shape; it is read only past top_k cells
                    if (!gemv_s(W(il, "indexer.attn_q_b.weight"), qr, iq, s, err)) return false;
                    rope_dev_kernel<<<nblk(g.idx_heads * g.qk_rope / 2), T_, 0, s>>>(iq, g.idx_heads, g.idx_key, 0,
                                                                                     g.qk_rope, d_pos_, theta_scale);
                    if (!gemv_s(W(il, "indexer.proj.weight"), xn, iw, s, err)) return false;
                    scale_kernel<<<1, T_, 0, s>>>(iw, 1.0f / std::sqrt((float) (g.idx_key * g.idx_heads)),
                                                  g.idx_heads);
                }
                if (!gemv_s(W(il, "attn_q_b.weight"), qr, q, s, err)) return false;
                rope_dev_kernel<<<nblk(g.n_head * g.qk_rope / 2), T_, 0, s>>>(q, g.n_head, g.qk_head(), g.qk_nope,
                                                                              g.qk_rope, d_pos_, theta_scale);
                q_absorb_kernel<<<nblk((int64_t) g.n_head * g.kv_lora * 32), T_, 0, s>>>(
                    (const float*) W(il, "attn_k_b.weight").d, q, qabs, g.qk_nope, g.qk_head(), g.kv_lora, g.n_head);
                return true;
            }))
            return false;
        if (!record(gB_[(size_t) il], [&]() -> bool {
                if (!gemv_s(W(il, "attn_output.weight"), o, tmp, s, err)) return false;
                add_kernel<<<nblk(g.n_embd), T_, 0, s>>>(x, tmp, g.n_embd);
                rms_norm_kernel<<<1, T_, 0, s>>>(x, (const float*) W(il, "ffn_norm.weight").d, xn, g.n_embd,
                                                 g.norm_eps);
                if (il < g.dense_lead) {
                    if (!gemv_s(W(il, "ffn_gate.weight"), xn, gate, s, err) ||
                        !gemv_s(W(il, "ffn_up.weight"), xn, up, s, err))
                        return false;
                    strata::kernels::glm_swiglu_clamp(gate, up, INFINITY, g.n_ff_dense, hh, s);
                    if (!gemv_s(W(il, "ffn_down.weight"), hh, tmp, s, err)) return false;
                    add_kernel<<<nblk(g.n_embd), T_, 0, s>>>(x, tmp, g.n_embd);
                } else {
                    if (!gemv_s(W(il, "ffn_gate_inp.weight"), xn, router, s, err)) return false;
                    strata::kernels::router_sigmoid(router, (const float*) W(il, "exp_probs_b.bias").d, 1,
                                                    g.n_expert, g.n_exp_used, g.w_scale, g.norm_w != 0, ids, rw, s);
                    GD_CHECK(cudaMemcpyAsync(h_rwids, rw, 128 * sizeof(float), cudaMemcpyDeviceToHost, s));
                    GD_CHECK(cudaMemcpyAsync(h_x, xn, (size_t) g.n_embd * sizeof(float), cudaMemcpyDeviceToHost, s));
                }
                return true;
            }))
            return false;
        if (il < g.dense_lead) continue;
        if (!record(gS_[(size_t) il], [&]() -> bool {
                if (!gemv_s(W(il, "ffn_gate_shexp.weight"), xn, gate, s, err) ||
                    !gemv_s(W(il, "ffn_up_shexp.weight"), xn, up, s, err))
                    return false;
                strata::kernels::glm_swiglu_clamp(gate, up, INFINITY, (int64_t) g.n_ff_exp * g.n_shared, hh, s);
                return gemv_s(W(il, "ffn_down_shexp.weight"), hh, sh, s, err);
            }))
            return false;
        if (!record(gC_[(size_t) il], [&]() -> bool {
                GD_CHECK(cudaMemcpyAsync(moe, h_moe, (size_t) g.n_embd * sizeof(float), cudaMemcpyHostToDevice, s));
                add_kernel<<<nblk(g.n_embd), T_, 0, s>>>(moe, sh, g.n_embd);
                add_kernel<<<nblk(g.n_embd), T_, 0, s>>>(x, moe, g.n_embd);
                return true;
            }))
            return false;
    }
    return true;
}

bool GlmDsaModel::step_graph(int32_t token, float* logits, std::string& err, bool head) {
    const GlmDsaGeometry& g = g_;
    bind(0);
    if (!gstream_) {
        const auto b0 = std::chrono::steady_clock::now();
        if (!build_graphs(err)) return false;
        std::fprintf(stderr, "glm-dsa: decode graphs recorded (%d layers) in %.0f ms\n", g.n_layers,
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - b0).count());
    }
    bind(0);
    cudaStream_t s = (cudaStream_t) gstream_;
    const auto t0 = std::chrono::steady_clock::now();
    double cpu_ms = 0;
    const int p = (int) pos_;
    static const bool blas_on = !(std::getenv("STRATA_GLM_DECODE_BLAS") && std::atoi(std::getenv("STRATA_GLM_DECODE_BLAS")) == 0);
    static const int blas_min = std::getenv("STRATA_GLM_BLAS_MIN") ? std::atoi(std::getenv("STRATA_GLM_BLAS_MIN")) : 1024;
    const bool dec_blas = blas_on && p + 1 > blas_min;

    float *x, *xn, *router, *rw, *qr, *kvpe, *q, *qabs, *o, *gate, *iq, *iw, *score, *h_x, *h_moe, *h_emb;
    int *ids, *h_pos;
    const float *h_rw;
    const int *h_ids;
    const auto carve = [&]() {
        x = sc_;
        xn = x + g.n_embd;
        router = xn + 4 * g.n_embd;
        rw = router + 4 * g.n_embd;
        ids = (int*) (rw + 64);
        qr = (float*) (ids + 64);
        kvpe = qr + g.q_lora;
        q = kvpe + g.cache_row();
        qabs = q + (size_t) g.n_head * g.qk_head();
        o = qabs + (size_t) g.n_head * g.kv_lora;
        gate = o + (size_t) g.n_head * g.v_head;
        iq = gate + 3 * (size_t) g.n_ff_dense;
        iw = iq + (size_t) g.idx_heads * g.idx_key;
        score = iw + 64;
        h_rw = pin_;
        h_ids = (const int*) (pin_ + 64);
        h_x = pin_ + 128;
        h_moe = h_x + g.n_embd;
        h_emb = h_moe + g.n_embd;
        h_pos = (int*) (h_emb + g.n_embd);
    };
    carve();

    // the embedding row and the position
    {
        const size_t row = ggml_row_size((ggml_type) tok_embd_->type, g.n_embd);
        ggml_get_type_traits((ggml_type) tok_embd_->type)
            ->to_float(tok_embd_file_->tensor_data(*tok_embd_) + row * (size_t) token, h_emb, g.n_embd);
        // the position on every part's device (the graphs read it there)
        const auto dg0 = std::chrono::steady_clock::now();
        for (int pi = (int) parts_.size() - 1; pi >= 0; --pi) {
            bind(pi);
            carve();
            *h_pos = p;
            GD_CHECK(cudaMemcpyAsync(d_pos_, h_pos, sizeof(int), cudaMemcpyHostToDevice, (cudaStream_t) gstream_));
        }
        diag_pos_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - dg0).count();
        ggml_get_type_traits((ggml_type) tok_embd_->type)
            ->to_float(tok_embd_file_->tensor_data(*tok_embd_) + row * (size_t) token, h_emb, g.n_embd);
        GD_CHECK(cudaMemcpyAsync(x, h_emb, (size_t) g.n_embd * sizeof(float), cudaMemcpyHostToDevice, s));
    }
    const float kq_scale = 1.0f / std::sqrt((float) g.qk_head());
    const int n_vis = p + 1;
    int n_sel = n_vis;
    bool sparse = false;
    std::vector<int> v_ids((size_t) g.n_exp_used);
    std::vector<float> v_rw((size_t) g.n_exp_used);

    for (int il = 0; il < g.n_layers; ++il) {
        if (part_of(il) != cur_) {   // the layer split: the residual to the next device, its selection too
            const float* x_old = x;
            const int* cells_old = d_cells_;
            const auto ds0 = std::chrono::steady_clock::now();
            GD_CHECK(cudaStreamSynchronize(s));
            diag_sync_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ds0).count();
            bind(part_of(il));
            carve();
            s = (cudaStream_t) gstream_;
            const auto dh0 = std::chrono::steady_clock::now();
            if (!hand_over(x_old, x, 1, err)) return false;
            diag_hand_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - dh0).count();
            if (sparse)
                GD_CHECK(cudaMemcpy(d_cells_, cells_old, (size_t) g.idx_top_k * sizeof(int), cudaMemcpyDefault));
        }
        GD_CHECK(cudaGraphLaunch((cudaGraphExec_t) gA_[(size_t) il], s));
        if (g.indexer_full[(size_t) il]) {
            if (n_vis > g.idx_top_k) {
                // the selection: the scores on the device, the top_k on the host (as step does)
                GD_CHECK(cudaStreamSynchronize(s));
                if (!select_cells(il, n_vis, iq, iw, score, err)) return false;
                n_sel = g.idx_top_k;
                sparse = true;
            } else {
                n_sel = n_vis;
                sparse = false;
            }
        }
        // the attention (its size changes every token): outside the graphs
        if (!attend_one(il, n_sel, sparse, dec_blas, q, qabs, o, s, err)) return false;
        GD_CHECK(cudaGraphLaunch((cudaGraphExec_t) gB_[(size_t) il], s));
        if (il < g.dense_lead) continue;
        GD_CHECK(cudaStreamSynchronize(s));   // the route and the activation are in pinned memory now
        for (int i = 0; i < g.n_exp_used; ++i) {
            v_ids[(size_t) i] = h_ids[i];
            v_rw[(size_t) i] = h_rw[i];
        }
        capture_ids(il, v_ids.data(), v_rw.data());
        route_log(il, v_ids.data(), 1);
        GD_CHECK(cudaGraphLaunch((cudaGraphExec_t) gS_[(size_t) il], s));   // the shared expert, meanwhile
        const auto c0 = std::chrono::steady_clock::now();
        moe_x_ = h_x;
        moe_y_ = h_moe;
        const bool ok = moe_cpu(il, v_ids, v_rw, err);
        moe_x_ = nullptr;
        moe_y_ = nullptr;
        if (!ok) return false;
        cpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
        GD_CHECK(cudaGraphLaunch((cudaGraphExec_t) gC_[(size_t) il], s));
    }

    if (logits || head) {
        rms_norm_kernel<<<1, T_, 0, s>>>(x, (const float*) w_.at("output_norm.weight").d, xn, g.n_embd, g.norm_eps);
        if (h_last_)
            GD_CHECK(cudaMemcpyAsync(h_last_, xn, (size_t) g.n_embd * sizeof(float), cudaMemcpyDeviceToDevice, s));
        if (!gemv_s(w_.at("output.weight"), xn, d_logits_buf_, s, err)) return false;
        d_logits_ = d_logits_buf_;
        if (logits)
            GD_CHECK(cudaMemcpyAsync(logits, d_logits_buf_, (size_t) g.n_vocab * sizeof(float), cudaMemcpyDeviceToHost,
                                     s));
    }
    GD_CHECK(cudaStreamSynchronize(s));
    ++pos_;
    if (std::getenv("STRATA_GLM_SPLIT_DIAG") && ++diag_n_ % 8 == 0)
        std::fprintf(stderr, "split diag: pos upload %.2f ms, boundary sync %.2f ms, hand-over %.2f ms (per token)%c", diag_pos_ / diag_n_, diag_sync_ / diag_n_, diag_hand_ / diag_n_, 10);
    const double total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    t_total_ += total;
    t_cpu_ += cpu_ms;
    t_dev_ += total - cpu_ms;
    return true;
}

}  // namespace strata::core
