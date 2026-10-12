// src/core/glm_dsa_model.cu - the GLM-5.3 ("glm-dsa") runner.  See glm_dsa_model.hpp for the math.
#include "strata/core/glm_dsa_model.hpp"

#include "strata/artifact/gguf_split.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/glm_ffn.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/router_sigmoid.hpp"
#include "strata/kernels/sampler.hpp"

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
    if (!allocs_.empty() || cache_) cudaSetDevice(dev_);
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

bool GlmDsaModel::load(const std::string& gguf_path, int64_t max_ctx, int dev, std::string& err) {
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

    // 2. the device: the one with the most memory unless asked
    int n_dev = 0;
    GD_CHECK(cudaGetDeviceCount(&n_dev));
    if (dev < 0) {
        size_t best = 0;
        for (int d = 0; d < n_dev; ++d) {
            cudaDeviceProp pr{};
            if (cudaGetDeviceProperties(&pr, d) == cudaSuccess && pr.totalGlobalMem > best) {
                best = pr.totalGlobalMem;
                dev = d;
            }
        }
    }
    if (dev < 0 || dev >= n_dev) {
        err = "glm-dsa: no such CUDA device " + std::to_string(dev);
        return false;
    }
    dev_ = dev;
    GD_CHECK(cudaSetDevice(dev_));

    // 3. the trunk's non-expert weights
    const char* quant[] = {"attn_q_a.weight", "attn_q_b.weight", "attn_kv_a_mqa.weight", "attn_output.weight",
                           "indexer.attn_q_b.weight", "indexer.attn_k.weight"};
    const char* f32[] = {"attn_norm.weight", "attn_q_a_norm.weight", "attn_kv_a_norm.weight", "ffn_norm.weight",
                         "attn_k_b.weight", "attn_v_b.weight", "indexer.k_norm.weight", "indexer.k_norm.bias",
                         "indexer.proj.weight"};
    const char* dense[] = {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"};
    const char* shexp[] = {"ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"};
    const char* router[] = {"ffn_gate_inp.weight", "exp_probs_b.bias"};
    for (int il = 0; il < g.n_layers; ++il) {
        const std::string L = "blk." + std::to_string(il) + ".";
        for (const char* s : quant)
            if (!upload(L + s, true, err)) return false;
        for (const char* s : f32)
            if (!upload(L + s, false, err)) return false;
        if (il < g.dense_lead) {
            for (const char* s : dense)
                if (!upload(L + s, true, err)) return false;
        } else {
            for (const char* s : shexp)
                if (!upload(L + s, true, err)) return false;
            for (const char* s : router)
                if (!upload(L + s, false, err)) return false;
        }
        if ((il + 1) % 10 == 0 || il + 1 == g.n_layers) {
            std::fprintf(stderr, "glm-dsa: layer %d/%d uploaded\r", il + 1, g.n_layers);
            std::fflush(stderr);
        }
    }
    std::fprintf(stderr, "\n");
    if (!upload("output_norm.weight", false, err) || !upload("output.weight", true, err)) return false;
    tok_embd_ = find("token_embd.weight", &tok_embd_file_);
    if (!tok_embd_ || !ggml_get_type_traits((ggml_type) tok_embd_->type)->to_float) {
        err = "glm-dsa: token_embd.weight missing or of an undequantizable type";
        return false;
    }

    // 4. the routed experts: mmap views, one NativeFmt per layer
    experts_.assign((size_t) g.n_layers, ExpertLayer{});
    for (int il = g.dense_lead; il < g.n_layers; ++il) {
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

    // 5. the caches and the scratch
    const size_t cache_floats = (size_t) g.n_layers * (size_t) max_ctx * (size_t) g.cache_row();
    GD_CHECK(cudaMalloc(&cache_, cache_floats * sizeof(float)));
    allocs_.push_back(cache_);
    ik_slot_.assign((size_t) g.n_layers, -1);
    int64_t n_full = 0;
    for (int il = 0; il < g.n_layers; ++il)
        if (g.indexer_full[(size_t) il]) ik_slot_[(size_t) il] = n_full++;
    GD_CHECK(cudaMalloc(&ikeys_, (size_t) n_full * (size_t) max_ctx * (size_t) g.idx_key * sizeof(float)));
    allocs_.push_back(ikeys_);
    const size_t scratch = (size_t) 16 * g.n_embd + (size_t) 4 * g.n_head * g.qk_head() +
                           (size_t) 2 * g.n_head * g.kv_lora + (size_t) 4 * g.n_ff_dense +
                           (size_t) g.idx_heads * g.idx_key + (size_t) max_ctx + (size_t) g.n_vocab + 4096;
    GD_CHECK(cudaMalloc(&sc_, scratch * sizeof(float)));
    allocs_.push_back(sc_);
    const int max_in = std::max({g.n_embd, g.n_head * g.v_head, g.n_ff_dense, g.q_lora, g.n_ff_exp});
    GD_CHECK(cudaMalloc(&q8_, strata::kernels::native_q8_1_bytes(max_in)));
    allocs_.push_back(q8_);
    GD_CHECK(cudaMalloc(&d_cells_, (size_t) g.idx_top_k * sizeof(int)));
    allocs_.push_back(d_cells_);
    GD_CHECK(cudaMalloc(&d_tok_, sizeof(int)));
    allocs_.push_back(d_tok_);
    GD_CHECK(cudaMalloc(&d_logits_buf_, (size_t) g.n_vocab * sizeof(float)));
    allocs_.push_back(d_logits_buf_);

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
    kc::native_quant_act(f, h_x_.data(), h_act_.data());
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
        h_moe_[(size_t) j] = acc;
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
    GD_CHECK(cudaSetDevice(dev_));
    const auto t0 = std::chrono::steady_clock::now();
    double cpu_ms = 0;
    const int p = (int) pos_;

    // scratch carve-up
    float* x = sc_;
    float* xn = x + g.n_embd;
    float* tmp = xn + g.n_embd;
    float* moe = tmp + g.n_embd;
    float* sh = moe + g.n_embd;
    float* router = sh + g.n_embd;   // n_expert (< 4 * n_embd)
    float* rw = router + 4 * g.n_embd;
    int* ids = (int*) (rw + 64);
    float* qr = (float*) (ids + 64);
    float* kvpe = qr + g.q_lora;
    float* q = kvpe + g.cache_row();
    float* qabs = q + (size_t) g.n_head * g.qk_head();
    float* o = qabs + (size_t) g.n_head * g.kv_lora;
    float* gate = o + (size_t) g.n_head * g.v_head;
    float* up = gate + g.n_ff_dense;
    float* hh = up + g.n_ff_dense;
    float* iq = hh + g.n_ff_dense;
    float* iw = iq + (size_t) g.idx_heads * g.idx_key;
    float* score = iw + 64;
    float* d_logits = d_logits_buf_;

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

    for (int il = 0; il < g.n_layers; ++il) {
        float* cache = cache_ + (size_t) il * (size_t) max_ctx_ * g.cache_row();
        // ---- attention
        rms_norm_kernel<<<1, T_>>>(x, (const float*) W(il, "attn_norm.weight").d, xn, g.n_embd, g.norm_eps);
        if (!gemv(W(il, "attn_q_a.weight"), xn, qr, err)) return false;
        rms_norm_kernel<<<1, T_>>>(qr, (const float*) W(il, "attn_q_a_norm.weight").d, qr, g.q_lora, g.norm_eps);
        if (!gemv(W(il, "attn_kv_a_mqa.weight"), xn, kvpe, err)) return false;
        rms_norm_kernel<<<1, T_>>>(kvpe, (const float*) W(il, "attn_kv_a_norm.weight").d, kvpe, g.kv_lora,
                                   g.norm_eps);
        rope_norm_kernel<<<nblk(g.qk_rope / 2), T_>>>(kvpe, 1, g.cache_row(), g.kv_lora, g.qk_rope, p, theta_scale);
        GD_CHECK(cudaMemcpy(cache + (size_t) p * g.cache_row(), kvpe, (size_t) g.cache_row() * sizeof(float),
                            cudaMemcpyDeviceToDevice));

        // the lightning indexer: keys are cached on full layers; the selection matters only past top_k cells
        if (g.indexer_full[(size_t) il]) {
            float* ik = ikeys_ + (size_t) ik_slot_[(size_t) il] * (size_t) max_ctx_ * g.idx_key;
            float* ikp = ik + (size_t) p * g.idx_key;
            if (!gemv(W(il, "indexer.attn_k.weight"), xn, ikp, err)) return false;
            layer_norm_kernel<<<1, 128>>>(ikp, (const float*) W(il, "indexer.k_norm.weight").d,
                                          (const float*) W(il, "indexer.k_norm.bias").d, ikp, g.idx_key,
                                          g.idx_norm_eps);
            rope_norm_kernel<<<nblk(g.qk_rope / 2), T_>>>(ikp, 1, g.idx_key, 0, g.qk_rope, p, theta_scale);
            if (n_vis > g.idx_top_k) {
                if (!gemv(W(il, "indexer.attn_q_b.weight"), qr, iq, err)) return false;
                rope_norm_kernel<<<nblk(g.idx_heads * g.qk_rope / 2), T_>>>(iq, g.idx_heads, g.idx_key, 0, g.qk_rope,
                                                                            p, theta_scale);
                if (!gemv(W(il, "indexer.proj.weight"), xn, iw, err)) return false;
                scale_kernel<<<1, T_>>>(iw, 1.0f / std::sqrt((float) (g.idx_key * g.idx_heads)), g.idx_heads);
                const size_t smem = (size_t) (g.idx_heads * g.idx_key + g.idx_heads) * sizeof(float);
                indexer_score_kernel<<<nblk(n_vis), T_, smem>>>(iq, ik, iw, score, n_vis, g.idx_heads, g.idx_key);
                h_score_.resize((size_t) n_vis);
                GD_CHECK(cudaMemcpy(h_score_.data(), score, (size_t) n_vis * sizeof(float), cudaMemcpyDeviceToHost));
                cells_.resize((size_t) n_vis);
                std::iota(cells_.begin(), cells_.end(), 0);
                std::nth_element(cells_.begin(), cells_.begin() + g.idx_top_k, cells_.end(), [&](int a, int b) {
                    return h_score_[(size_t) a] > h_score_[(size_t) b] ||
                           (h_score_[(size_t) a] == h_score_[(size_t) b] && a < b);
                });
                cells_.resize((size_t) g.idx_top_k);
                std::sort(cells_.begin(), cells_.end());
                GD_CHECK(cudaMemcpy(d_cells_, cells_.data(), cells_.size() * sizeof(int), cudaMemcpyHostToDevice));
                n_sel = g.idx_top_k;
                sparse = true;
            } else {
                n_sel = n_vis;
                sparse = false;
            }
        }

        if (!gemv(W(il, "attn_q_b.weight"), qr, q, err)) return false;
        rope_norm_kernel<<<nblk(g.n_head * g.qk_rope / 2), T_>>>(q, g.n_head, g.qk_head(), g.qk_nope, g.qk_rope, p,
                                                                 theta_scale);
        q_absorb_kernel<<<nblk((int64_t) g.n_head * g.kv_lora * 32), T_>>>(
            (const float*) W(il, "attn_k_b.weight").d, q, qabs, g.qk_nope, g.qk_head(), g.kv_lora, g.n_head);
        const size_t smem = (size_t) (g.cache_row() + g.kv_lora + 32 + n_sel) * sizeof(float);
        mla_attn_kernel<<<g.n_head, T_, smem>>>(qabs, q, cache, sparse ? d_cells_ : nullptr, n_sel,
                                                (const float*) W(il, "attn_v_b.weight").d, o, g.kv_lora, g.qk_nope,
                                                g.qk_rope, g.v_head, kq_scale);
        GD_CHECK(cudaGetLastError());
        if (!gemv(W(il, "attn_output.weight"), o, tmp, err)) return false;
        add_kernel<<<nblk(g.n_embd), T_>>>(x, tmp, g.n_embd);
        capture(il, 0, x);

        // ---- the FFN
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
            GD_CHECK(cudaMemcpy(h_ids.data(), ids, h_ids.size() * sizeof(int), cudaMemcpyDeviceToHost));
            GD_CHECK(cudaMemcpy(h_rw.data(), rw, h_rw.size() * sizeof(float), cudaMemcpyDeviceToHost));
            GD_CHECK(cudaMemcpy(h_x_.data(), xn, (size_t) g.n_embd * sizeof(float), cudaMemcpyDeviceToHost));
            capture_ids(il, h_ids.data(), h_rw.data());
            // the shared expert runs on the device while the CPU does the routed ones
            if (!gemv(W(il, "ffn_gate_shexp.weight"), xn, gate, err) ||
                !gemv(W(il, "ffn_up_shexp.weight"), xn, up, err))
                return false;
            strata::kernels::glm_swiglu_clamp(gate, up, INFINITY, (int64_t) g.n_ff_exp * g.n_shared, hh,
                                              cudaStreamLegacy);
            if (!gemv(W(il, "ffn_down_shexp.weight"), hh, sh, err)) return false;
            const auto c0 = std::chrono::steady_clock::now();
            if (!moe_cpu(il, h_ids, h_rw, err)) return false;
            cpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
            GD_CHECK(cudaMemcpy(moe, h_moe_.data(), (size_t) g.n_embd * sizeof(float), cudaMemcpyHostToDevice));
            add_kernel<<<nblk(g.n_embd), T_>>>(moe, sh, g.n_embd);   // moe_out + shexp, then the residual
            add_kernel<<<nblk(g.n_embd), T_>>>(x, moe, g.n_embd);
        }
        capture(il, 1, x);
    }

    if (logits || head) {
        rms_norm_kernel<<<1, T_>>>(x, (const float*) w_.at("output_norm.weight").d, xn, g.n_embd, g.norm_eps);
        if (!gemv(w_.at("output.weight"), xn, d_logits, err)) return false;
        d_logits_ = d_logits;
        if (logits)
            GD_CHECK(cudaMemcpy(logits, d_logits, (size_t) g.n_vocab * sizeof(float), cudaMemcpyDeviceToHost));
    }
    GD_CHECK(cudaDeviceSynchronize());
    ++pos_;
    const double total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    t_total_ += total;
    t_cpu_ += cpu_ms;
    t_dev_ += total - cpu_ms;
    return true;
}

bool GlmDsaModel::read_logits(float* out, std::string& err) {
    if (!d_logits_) {
        err = "glm-dsa: no head has run";
        return false;
    }
    GD_CHECK(cudaSetDevice(dev_));
    GD_CHECK(cudaMemcpy(out, d_logits_, (size_t) g_.n_vocab * sizeof(float), cudaMemcpyDeviceToHost));
    return true;
}

int GlmDsaModel::sample(const strata::kernels::SamplerParams& sp, std::string& err) {
    if (!d_logits_) {
        err = "glm-dsa: sample() before any head ran";
        return -1;
    }
    if (cudaSetDevice(dev_) != cudaSuccess) {
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

}  // namespace strata::core
