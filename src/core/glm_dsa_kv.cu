// src/core/glm_dsa_kv.cu - the int8 attention cache's kernels (see glm_dsa_kv.cuh).
#include "glm_dsa_kv.cuh"

#include <cmath>

namespace strata::core::kv {
namespace {

constexpr int TB = 256;

/// one warp per 32-value block: max |x| -> scale, then the int8 codes.
__global__ void store_q8_kernel(const float* __restrict__ src, int T, int src_ld, int row, int8_t* __restrict__ q,
                                __half* __restrict__ s, int pos0, const int* __restrict__ d_pos) {
    const int nblk = row / kBlock;
    const int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (w >= T * nblk) return;
    const int t = w / nblk, b = w % nblk;
    const int pos = (d_pos ? *d_pos : pos0) + t;
    const float x = src[(size_t) t * src_ld + b * kBlock + lane];
    float m = fabsf(x);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    const float d = m / 127.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    q[(size_t) pos * row + b * kBlock + lane] = (int8_t) __float2int_rn(x * id);
    if (lane == 0) s[(size_t) pos * nblk + b] = __float2half(d);
}

__global__ void gather_q8_kernel(const int8_t* __restrict__ q, const __half* __restrict__ s, int row,
                                 const int* __restrict__ cells, int first, int n, float* __restrict__ dst) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) n * row) return;
    const int r = (int) (i / row), e = (int) (i % row);
    const int cell = cells ? cells[r] : first + r;
    if (cell < 0) {
        dst[i] = 0.0f;
        return;
    }
    const int nblk = row / kBlock;
    dst[i] = __half2float(s[(size_t) cell * nblk + e / kBlock]) * (float) q[(size_t) cell * row + e];
}

__global__ void store_f16_kernel(const float* __restrict__ src, int T, int src_ld, int row, __half* __restrict__ dst,
                                 int pos0, const int* __restrict__ d_pos) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) T * row) return;
    const int t = (int) (i / row), e = (int) (i % row);
    const int pos = (d_pos ? *d_pos : pos0) + t;
    dst[(size_t) pos * row + e] = __float2half(src[(size_t) t * src_ld + e]);
}

__global__ void to_f16_kernel(const float* __restrict__ src, __half* __restrict__ dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

__device__ float bsum(float v, float* red) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    float r = 0.0f;
    for (int i = 0; i < nw; ++i) r += red[i];
    return r;
}

__device__ float bmax(float v, float* red) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    float r = -INFINITY;
    for (int i = 0; i < nw; ++i) r = fmaxf(r, red[i]);
    return r;
}

/// One block per head (the F32 cache's mla_attn_kernel with an int8 load): scores, softmax, the latent context,
/// then v through vb.
__global__ void attend_small_q8_kernel(const float* __restrict__ qabs, const float* __restrict__ q,
                                       const int8_t* __restrict__ cq, const __half* __restrict__ cs,
                                       const int* __restrict__ cells, int n_sel, const float* __restrict__ vb,
                                       float* __restrict__ out, int kv_lora, int qk_nope, int qk_rope, int v_head,
                                       float scale) {
    extern __shared__ float sm[];
    const int h = blockIdx.x;
    const int row = kv_lora + qk_rope, nblk = row / kBlock;
    float* s_q = sm;                 // row
    float* s_ctx = s_q + row;        // kv_lora
    float* red = s_ctx + kv_lora;    // 32
    float* s_p = red + 32;           // n_sel
    for (int e = threadIdx.x; e < kv_lora; e += blockDim.x) s_q[e] = qabs[(size_t) kv_lora * h + e];
    for (int e = threadIdx.x; e < qk_rope; e += blockDim.x)
        s_q[kv_lora + e] = q[(size_t) (qk_nope + qk_rope) * h + qk_nope + e];
    __syncthreads();

    float mx = -INFINITY;
    for (int i = threadIdx.x; i < n_sel; i += blockDim.x) {
        const int cell = cells ? cells[i] : i;
        const int8_t* cr = cq + (size_t) row * cell;
        const __half* sr = cs + (size_t) nblk * cell;
        float d = 0.0f;
        for (int b = 0; b < nblk; ++b) {
            float a = 0.0f;
            for (int j = 0; j < kBlock; ++j) a += s_q[b * kBlock + j] * (float) cr[b * kBlock + j];
            d += a * __half2float(sr[b]);
        }
        d *= scale;
        s_p[i] = d;
        mx = fmaxf(mx, d);
    }
    mx = bmax(mx, red);
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n_sel; i += blockDim.x) {
        const float e = expf(s_p[i] - mx);
        s_p[i] = e;
        sum += e;
    }
    const float inv = 1.0f / bsum(sum, red);
    __syncthreads();

    for (int e = threadIdx.x; e < kv_lora; e += blockDim.x) {
        float acc = 0.0f;
        const int b = e / kBlock;
        for (int i = 0; i < n_sel; ++i) {
            const int cell = cells ? cells[i] : i;
            acc += s_p[i] * __half2float(cs[(size_t) nblk * cell + b]) * (float) cq[(size_t) row * cell + e];
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

__global__ void concat_q_kernel(const float* __restrict__ qabs, const float* __restrict__ q, float* __restrict__ qc,
                                int T, int n_head, int kv_lora, int qk_nope, int qk_rope) {
    const int row = kv_lora + qk_rope;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) T * n_head * row) return;
    const int64_t r = i / row;   // (t, h)
    const int e = (int) (i % row);
    qc[i] = e < kv_lora ? qabs[r * kv_lora + e] : q[r * (qk_nope + qk_rope) + qk_nope + (e - kv_lora)];
}

__global__ void softmax_counted_kernel(float* S, const int* __restrict__ cnt, int n_head, int K) {
    __shared__ float red[32];
    const int r = blockIdx.x;                    // (t, h)
    const int n = cnt[r / n_head];
    float* row = S + (size_t) r * K;
    float mx = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) mx = fmaxf(mx, row[i]);
    mx = bmax(mx, red);
    float sum = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        const float e = i < n ? expf(row[i] - mx) : 0.0f;
        row[i] = e;
        sum += e;
    }
    const float inv = 1.0f / bsum(sum, red);
    for (int i = threadIdx.x; i < K; i += blockDim.x) row[i] *= inv;
}

int nb(int64_t n) { return (int) ((n + TB - 1) / TB); }

__global__ void gather_q8_f16_kernel(const int8_t* __restrict__ q, const __half* __restrict__ s, int row,
                                     const int* __restrict__ cells, int n, __half* __restrict__ dst) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) n * row) return;
    const int r = (int) (i / row), e = (int) (i % row);
    const int cell = cells[r];
    if (cell < 0) {
        dst[i] = __float2half(0.0f);
        return;
    }
    const int nblk = row / kBlock;
    dst[i] = __float2half(__half2float(s[(size_t) cell * nblk + e / kBlock]) * (float) q[(size_t) cell * row + e]);
}

__global__ void concat_q_f16_kernel(const float* __restrict__ qabs, const float* __restrict__ q, __half* __restrict__ qc,
                                    int T, int n_head, int kv_lora, int qk_nope, int qk_rope) {
    const int row = kv_lora + qk_rope;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) T * n_head * row) return;
    const int64_t r = i / row;
    const int e = (int) (i % row);
    qc[i] = __float2half(e < kv_lora ? qabs[r * kv_lora + e] : q[r * (qk_nope + qk_rope) + qk_nope + (e - kv_lora)]);
}

/// float -> a key whose unsigned order is the float order
__device__ __forceinline__ uint32_t fkey(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/// exclusive scan of v over the block (blockDim.x <= 1024), the total to *total
__device__ int block_scan(int v, int* sh, int* total) {
    const int t = threadIdx.x;
    sh[t] = v;
    __syncthreads();
    for (int o = 1; o < blockDim.x; o <<= 1) {
        const int a = t >= o ? sh[t - o] : 0;
        __syncthreads();
        sh[t] += a;
        __syncthreads();
    }
    const int incl = sh[t];
    if (t == blockDim.x - 1) *total = incl;
    __syncthreads();
    return incl - v;
}

constexpr int TK = 1024;

__global__ void topk_rows_kernel(const float* __restrict__ scores, int64_t ld, int n_fixed, int vis0, int K,
                                 int* __restrict__ out, int64_t out_ld, int* __restrict__ cnt) {
    __shared__ unsigned hist[256];
    __shared__ int sh[TK];
    __shared__ int s_digit, s_rem, s_total;
    const int r = blockIdx.x, tid = threadIdx.x;
    const int vis = n_fixed > 0 ? n_fixed : vis0 + r;
    const float* sc = scores + (size_t) r * ld;
    int* o = out + (size_t) r * out_ld;
    if (vis <= K) {
        for (int i = tid; i < K; i += blockDim.x) o[i] = i < vis ? i : -1;
        if (cnt && tid == 0) cnt[r] = vis;
        return;
    }
    // the K-th largest key: its bits fixed 8 at a time, from the top
    uint32_t prefix = 0, mask = 0;
    if (tid == 0) s_rem = K;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int b = tid; b < 256; b += blockDim.x) hist[b] = 0;
        __syncthreads();
        for (int i = tid; i < vis; i += blockDim.x) {
            const uint32_t u = fkey(sc[i]);
            if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            int acc = 0, rem = s_rem, d = 0;
            for (int b = 255; b >= 0; --b) {
                if (acc + (int) hist[b] >= rem) {
                    d = b;
                    rem -= acc;   // how many of the threshold's equals are still to take
                    break;
                }
                acc += (int) hist[b];
            }
            s_digit = d;
            s_rem = rem;
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        mask |= 255u << shift;
    }
    const uint32_t thr = prefix;
    const int take_eq = s_rem;   // equals to the threshold to include (the lowest cells first)
    // stable compaction: each thread a contiguous range of cells
    const int per = (vis + blockDim.x - 1) / blockDim.x;
    const int i0 = min(vis, tid * per), i1 = min(vis, i0 + per);
    int eq = 0;
    for (int i = i0; i < i1; ++i) eq += fkey(sc[i]) == thr;
    const int eq_base = block_scan(eq, sh, &s_total);
    int inc = 0, er = eq_base;
    for (int i = i0; i < i1; ++i) {
        const uint32_t u = fkey(sc[i]);
        if (u > thr) ++inc;
        else if (u == thr) inc += (er++ < take_eq);
    }
    const int out_base = block_scan(inc, sh, &s_total);
    int w = out_base;
    er = eq_base;
    for (int i = i0; i < i1; ++i) {
        const uint32_t u = fkey(sc[i]);
        if (u > thr || (u == thr && er++ < take_eq)) o[w++] = i;
    }
    if (cnt && tid == 0) cnt[r] = K;
}

}  // namespace

void gather_q8_f16(const int8_t* q, const __half* s, int row, const int* cells, int n, __half* dst, cudaStream_t st) {
    gather_q8_f16_kernel<<<nb((int64_t) n * row), TB, 0, st>>>(q, s, row, cells, n, dst);
}

void concat_q_f16(const float* qabs, const float* q, __half* qc, int T, int n_head, int kv_lora, int qk_nope,
                  int qk_rope, cudaStream_t st) {
    concat_q_f16_kernel<<<nb((int64_t) T * n_head * (kv_lora + qk_rope)), TB, 0, st>>>(qabs, q, qc, T, n_head,
                                                                                      kv_lora, qk_nope, qk_rope);
}

void topk_rows(const float* scores, int64_t ld, int rows, int n_fixed, int vis0, int K, int* out, int64_t out_ld,
               int* cnt, cudaStream_t st) {
    topk_rows_kernel<<<rows, TK, 0, st>>>(scores, ld, n_fixed, vis0, K, out, out_ld, cnt);
}

void store_q8(const float* src, int T, int src_ld, int row, int8_t* q, __half* s, int pos0, const int* d_pos,
              cudaStream_t st) {
    const int64_t warps = (int64_t) T * (row / kBlock);
    store_q8_kernel<<<nb(warps * 32), TB, 0, st>>>(src, T, src_ld, row, q, s, pos0, d_pos);
}

void gather_q8(const int8_t* q, const __half* s, int row, const int* cells, int first, int n, float* dst,
               cudaStream_t st) {
    gather_q8_kernel<<<nb((int64_t) n * row), TB, 0, st>>>(q, s, row, cells, first, n, dst);
}

void store_f16(const float* src, int T, int src_ld, int row, __half* dst, int pos0, const int* d_pos,
               cudaStream_t st) {
    store_f16_kernel<<<nb((int64_t) T * row), TB, 0, st>>>(src, T, src_ld, row, dst, pos0, d_pos);
}

void to_f16(const float* src, __half* dst, int64_t n, cudaStream_t st) {
    to_f16_kernel<<<nb(n), TB, 0, st>>>(src, dst, n);
}

void attend_small_q8(const float* qabs, const float* q, const int8_t* cq, const __half* cs, const int* cells,
                     int n_sel, const float* vb, float* out, int n_head, int kv_lora, int qk_nope, int qk_rope,
                     int v_head, float scale, cudaStream_t st) {
    const size_t smem = (size_t) (kv_lora + qk_rope + kv_lora + 32 + n_sel) * sizeof(float);
    attend_small_q8_kernel<<<n_head, TB, smem, st>>>(qabs, q, cq, cs, cells, n_sel, vb, out, kv_lora, qk_nope,
                                                     qk_rope, v_head, scale);
}

void concat_q(const float* qabs, const float* q, float* qc, int T, int n_head, int kv_lora, int qk_nope, int qk_rope,
              cudaStream_t st) {
    concat_q_kernel<<<nb((int64_t) T * n_head * (kv_lora + qk_rope)), TB, 0, st>>>(qabs, q, qc, T, n_head, kv_lora,
                                                                                   qk_nope, qk_rope);
}

void softmax_counted(float* S, const int* cnt, int T, int n_head, int K, cudaStream_t st) {
    softmax_counted_kernel<<<T * n_head, TB, 0, st>>>(S, cnt, n_head, K);
}

}  // namespace strata::core::kv
