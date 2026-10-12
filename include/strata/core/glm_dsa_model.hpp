// include/strata/core/glm_dsa_model.hpp - a GLM-5.3 ("glm-dsa") model runner: GGUF shards in, logits out.
//
// GLM-5.3 is the DeepSeek-V3.2 shape: every layer is MLA attention (q/kv low-rank, RoPE on a 64-wide slice) with a
// lightning indexer choosing up to top_k cells, then a dense FFN (the first `dense_lead` layers) or a sigmoid-routed
// MoE with one shared expert.  No hyper-connections, no linear attention - unlike glm5-next (Glm5Model).  The spec
// is llama.cpp's src/models/glm-dsa.cpp (vendored in the Strata-GLM-5.3 tree under ref/llama-upstream/).
//
// Per token, per layer (llama.cpp's cb() names in brackets):
//   xn   = rms(x) * attn_norm                                   [attn_norm]
//   qr   = rms(attn_q_a . xn) * attn_q_a_norm                   [qr]
//   kvpe = attn_kv_a_mqa . xn ; c = rms(kvpe[:kv_lora]) * kv_a_norm ; kpe = rope(kvpe[kv_lora:])
//   q    = attn_q_b . qr  -> per head [nope (qk_nope) | pe (qk_rope)] ; q_pe = rope(q_pe)
//   qabs = attn_k_b_h^T q_nope_h                               [q_nope_absorbed]
//   cells: all cells <= p while p < top_k, else the indexer's top_k (full layers compute it, "shared" layers reuse
//          the previous full layer's - GLM-5.2+'s indexer_types)
//   att  = softmax_s((qabs_h . c_s + q_pe_h . kpe_s) / sqrt(qk_nope + qk_rope)) ; o_h = attn_v_b_h (sum_s att c_s)
//   x   += attn_output . o
//   x   += ffn(rms(x) * ffn_norm)       dense SwiGLU, or sigmoid-routed top-k experts * w_scale + shared expert
// head: logits = output . (rms(x) * output_norm)
//
// CORRECTNESS-FIRST SCOPE (milestone 1): one token at a time; the non-expert weights stay quantized in VRAM and are
// dotted by the device MMVQ (exactly the Q8_1 activation path llama.cpp's CUDA backend uses); the routed experts are
// read from the shards' mmaps and dotted on the CPU by ggml-cpu (native_expert); the MTP layer is not loaded.
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace strata::core {

/// Every number the runner depends on, from the GGUF's glm-dsa.* metadata.
struct GlmDsaGeometry {
    int n_embd = 6144, n_layers = 78, nextn = 1, n_head = 64, n_vocab = 154880;
    int q_lora = 2048, kv_lora = 512, qk_nope = 192, qk_rope = 64, v_head = 256;
    int idx_heads = 32, idx_key = 128, idx_top_k = 2048;
    std::vector<int> indexer_full;   ///< per trunk layer: 1 computes the selection, 0 reuses the previous one
    int n_expert = 256, n_exp_used = 8, n_ff_exp = 2048, n_shared = 1, n_ff_dense = 12288, dense_lead = 3;
    float w_scale = 2.5f;
    int norm_w = 1;
    float rope_base = 8000000.0f, norm_eps = 1e-5f;
    float idx_norm_eps = 0.0f;       ///< llama.cpp's LLM_NORM eps is f_norm_eps, never set for glm-dsa (0)

    int qk_head() const { return qk_nope + qk_rope; }   ///< 256: the attention's head size (the kq scale)
    int cache_row() const { return kv_lora + qk_rope; }  ///< 576: one cached cell, [c | kpe]

    static bool from_gguf(const strata::GgufFile& g, GlmDsaGeometry& out, std::string& err);
};

/// A tiny persistent fork-join pool for the CPU expert dots.
class GlmCpuPool {
public:
    explicit GlmCpuPool(int n_threads);
    ~GlmCpuPool();
    /// fn(i) for i in [0, n), spread over the workers and the caller; returns when all are done.
    void run(int n, const std::function<void(int)>& fn);
    int threads() const { return (int) workers_.size() + 1; }

private:
    void worker();
    void drain();
    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    const std::function<void(int)>* fn_ = nullptr;
    int n_ = 0;
    std::atomic<int> next_{0}, finished_{0};
    uint64_t epoch_ = 0;
    int busy_ = 0;
    bool stop_ = false;
};

class GlmDsaModel {
public:
    GlmDsaModel() = default;
    ~GlmDsaModel();
    GlmDsaModel(const GlmDsaModel&) = delete;
    GlmDsaModel& operator=(const GlmDsaModel&) = delete;

    /// Opens every shard of the split GGUF (`gguf_path` is any one of them), uploads the trunk's non-expert weights
    /// to device `dev` (-1: the device with the most memory) and sizes the caches for `max_ctx` tokens.
    bool load(const std::string& gguf_path, int64_t max_ctx, int dev, std::string& err);

    /// Appends one token at the next position; when `logits` is non-null the head runs and its n_vocab floats are
    /// written there (the head is skipped otherwise - a teacher-forced prompt only needs the last one).
    bool step(int32_t token, float* logits, std::string& err);
    /// Forgets the sequence (the caches are overwritten from position 0).
    void reset() { pos_ = 0; }
    int64_t pos() const { return pos_; }
    const GlmDsaGeometry& geometry() const { return g_; }
    int device() const { return dev_; }

    /// Per-step timing (milliseconds, summed since the last call): the device part and the CPU expert part.
    void take_timing(double& dev_ms, double& cpu_ms, double& total_ms);

private:
    struct DevW {   ///< one served weight: quantized rows (type != 0) or F32
        int type = 0;
        void* d = nullptr;
        int64_t n_in = 0, n_out = 0;
    };
    struct ExpertLayer {
        int gu_type = -1, d_type = -1;
        const uint8_t* gate = nullptr;
        const uint8_t* up = nullptr;
        const uint8_t* down = nullptr;
        size_t gu_expert = 0, d_expert = 0;   ///< bytes of one expert's slab in each tensor
        void* fmt = nullptr;                  ///< a NativeFmt, owned
    };

    bool upload(const std::string& name, bool keep_quant, std::string& err);
    const strata::TensorInfo* find(const std::string& name, const strata::GgufFile** file = nullptr) const;
    bool gemv(const DevW& w, const float* x, float* y, std::string& err);
    const DevW& W(int il, const char* suffix) const;
    bool moe_cpu(int il, const std::vector<int>& ids, const std::vector<float>& rw, std::string& err);

    GlmDsaGeometry g_;
    int dev_ = 0;
    int64_t max_ctx_ = 0, pos_ = 0;
    std::vector<std::unique_ptr<strata::GgufFile>> files_;
    std::unordered_map<std::string, DevW> w_;
    std::vector<void*> allocs_;
    std::vector<ExpertLayer> experts_;
    const strata::TensorInfo* tok_embd_ = nullptr;
    const strata::GgufFile* tok_embd_file_ = nullptr;

    // device state
    float* cache_ = nullptr;      ///< per layer: max_ctx x cache_row
    float* ikeys_ = nullptr;      ///< per FULL indexer layer: max_ctx x idx_key
    std::vector<int64_t> ik_slot_;
    float* sc_ = nullptr;         ///< scratch arena
    void* q8_ = nullptr;          ///< the Q8_1 activation scratch
    int* d_cells_ = nullptr;
    std::vector<int> cells_;      ///< the current selection (host copy), reused by shared layers
    std::vector<float> h_score_;

    // host scratch for the CPU experts
    std::unique_ptr<GlmCpuPool> pool_;
    std::vector<float> h_x_, h_moe_, h_ff_, h_dn_;
    std::vector<uint8_t> h_act_, h_hq_;
    std::vector<float> h_embd_;

    double t_dev_ = 0, t_cpu_ = 0, t_total_ = 0;
};

}  // namespace strata::core
