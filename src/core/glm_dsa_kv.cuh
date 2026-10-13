// src/core/glm_dsa_kv.cuh - the GLM-5.3 runner's attention cache in int8 (internal to src/core/glm_dsa_*.cu).
//
// The MLA cache holds one row of kv_lora + qk_rope = 576 values per token per layer: the normed latent c and the
// roped kpe.  In F32 that is 179.7 KB a token over 78 layers - 70.7 GB at 393,216 tokens.  Here every row is
// quantized like ggml's Q8_0: blocks of 32 values, int8 with one FP16 scale (max |x| / 127), 612 bytes a row
// (46.6 KB a token, 18.3 GB at 393K) - llama.cpp's `-ctk q8_0` arithmetic.  The indexer's keys (128 a token on the
// layers with a full indexer) are FP16, llama.cpp's default cache type.
//
// Layout per layer: q = int8 [max_ctx x 576], s = FP16 [max_ctx x 18].
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace strata::core::kv {

constexpr int kBlock = 32;

/// rows t = 0..T-1 of `src` (stride src_ld floats, `row` values each) -> cache rows pos + t, pos = *d_pos when
/// d_pos is non-null, else pos0.  row must be a multiple of 32.
void store_q8(const float* src, int T, int src_ld, int row, int8_t* q, __half* s, int pos0, const int* d_pos,
              cudaStream_t st);
/// dst[i, :] = cache row (cells ? cells[i] : first + i), i < n, dequantized to F32 (dst rows of `row` floats); a
/// negative cell gives a row of zeros.
void gather_q8(const int8_t* q, const __half* s, int row, const int* cells, int first, int n, float* dst,
               cudaStream_t st);
/// rows -> FP16 cache rows pos + t (as store_q8).
void store_f16(const float* src, int T, int src_ld, int row, __half* dst, int pos0, const int* d_pos,
               cudaStream_t st);
/// dst = (half) src, n values.
void to_f16(const float* src, __half* dst, int64_t n, cudaStream_t st);

/// The decode attention of one token, per head, reading the int8 cache directly (the per-head kernel of the F32
/// cache with a dequantizing load): cells ? the n_sel listed : cells 0..n_sel-1; then v through vb.
void attend_small_q8(const float* qabs, const float* q, const int8_t* cq, const __half* cs, const int* cells,
                     int n_sel, const float* vb, float* out, int n_head, int kv_lora, int qk_nope, int qk_rope,
                     int v_head, float scale, cudaStream_t st);

/// QC[(t, h), :] = [qabs[t, h, 0..kv_lora) | q[t, h, qk_nope..qk_nope+qk_rope)] for T tokens (576 a row).
void concat_q(const float* qabs, const float* q, float* qc, int T, int n_head, int kv_lora, int qk_nope, int qk_rope,
              cudaStream_t st);
/// softmax over the first cnt[t] entries of every row (t, h) of S [T x n_head x K]; the rest set to 0.
void softmax_counted(float* S, const int* cnt, int T, int n_head, int K, cudaStream_t st);

}  // namespace strata::core::kv
