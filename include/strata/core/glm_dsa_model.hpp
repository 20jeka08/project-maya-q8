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
#include "strata/kernels/sampler.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
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
    /// and sizes the caches for `max_ctx` tokens.  `dev` (-1: the device with the most memory) takes the first
    /// layers; `split`: "0" that device alone, "<layer>" the layers from <layer> on the next device, "auto" (or
    /// empty) as many layers as fit on each visible device in turn (the largest first) - the layer split.
    bool load(const std::string& gguf_path, int64_t max_ctx, int dev, std::string& err, const std::string& split = "");

    /// Appends one token at the next position.  The head runs when `logits` is non-null (its n_vocab floats are
    /// written there) or when `head` is set (the logits stay on the device for sample()); otherwise it is skipped -
    /// a teacher-forced prompt only needs the last one.
    bool step(int32_t token, float* logits, std::string& err, bool head = false);
    /// Appends n tokens, batched in chunks of up to STRATA_GLM_CHUNK (512): per layer the dense projections and the
    /// attention are cuBLAS FP32 GEMMs over the chunk (the Q8_0 weights expanded on the device), and the routed
    /// experts run on the CPU grouped by expert, so each selected expert's weights are read once a chunk instead
    /// of once a token.  The head runs for the last token when `head` (logits on the device, for sample()).
    /// `progress(done, n)` after each chunk.
    bool prefill(const int32_t* toks, int64_t n, std::string& err, bool head,
                 const std::function<void(int64_t, int64_t)>& progress = {});
    /// One token from the last head's logits (greedy when sp.greedy or temperature <= 0); -1 with `err` on failure.
    int sample(const strata::kernels::SamplerParams& sp, std::string& err);
    /// A check: record the residual of the last token processed after each layer's attention and after its FFN
    /// (2 x n_layers x n_embd floats, layer-major), by step() and by prefill() alike.
    void set_capture(bool on) {
        capture_ = on;
        if (on) cap_.assign((size_t) 2 * g_.n_layers * g_.n_embd, 0.0f), cap_ids_.assign((size_t) g_.n_layers * 2 * g_.n_exp_used, -1);
    }
    const std::vector<float>& captured_ids() const { return cap_ids_; }   ///< per layer: k ids then k weights
    const std::vector<float>& captured() const { return cap_; }
    /// Copies the last head's n_vocab logits to the host.
    bool read_logits(float* out, std::string& err);
    /// Forgets the sequence (the caches are overwritten from position 0).
    void reset() { pos_ = 0; }
    /// Keeps the first `pos` positions and forgets the rest (the caches are append-only by position, so the cells
    /// past `pos` are simply overwritten) - a prompt that extends the previous conversation reuses its prefix.
    void rewind(int64_t pos) { pos_ = pos < pos_ ? (pos < 0 ? 0 : pos) : pos_; }
    int64_t max_ctx() const { return max_ctx_; }
    int64_t pos() const { return pos_; }
    const GlmDsaGeometry& geometry() const { return g_; }
    int device() const { return dev_; }
    /// The layer split: (device, first layer) of every part.
    std::vector<std::pair<int, int>> parts() const {
        std::vector<std::pair<int, int>> v;
        for (const Part& p : parts_) v.push_back({p.dev, p.l0});
        return v;
    }

    /// Per-step timing (milliseconds, summed since the last call): the device part and the CPU expert part.
    void take_timing(double& dev_ms, double& cpu_ms, double& total_ms);

private:
    /// Everything one device holds for its layers [l0, l1) besides their weights and caches (which are per tensor /
    /// per layer): its scratch, cuBLAS handle, decode graph stream, pinned words, batched-path buffers.  bind(i)
    /// stashes the current part's pointers and loads part i's into the members below, so the per-layer code reads
    /// the members as before.
    struct Part {
        int dev = 0, l0 = 0, l1 = 0;
        float* sc = nullptr;
        void* q8 = nullptr;
        int* d_cells = nullptr;
        float* kvg = nullptr;
        float* ikv = nullptr;
        void* cublas = nullptr;
        float *pf = nullptr, *deq = nullptr, *att = nullptr, *pf_score = nullptr;
        float *sg = nullptr, *ss = nullptr, *qc = nullptr;
        int *d_sel = nullptr, *d_cnt = nullptr;
        void* iq16 = nullptr;
        void* gstream = nullptr;
        float* pin = nullptr;
        int* d_pos = nullptr;
    };
    std::vector<Part> parts_;
    std::vector<int> layer_part_;
    int cur_ = -1, cur_dev_ = 0;
    void bind(int part);
    int part_of(int il) const { return layer_part_[(size_t) il]; }
    /// The residual (rows x n_embd floats at `x` in part `from`'s buffer) into part `to`'s buffer at `x_to`.
    bool hand_over(const float* x_from, float* x_to, int64_t rows, std::string& err);

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
    /// gemv on an explicit stream (graph capture)
    bool gemv_s(const DevW& w, const float* x, float* y, void* stream, std::string& err);
    /// The decode step as per-layer CUDA graphs on a stream of its own (STRATA_GLM_GRAPHS=0: step's per-op path).
    bool step_graph(int32_t token, float* logits, std::string& err, bool head);
    bool build_graphs(std::string& err);
    const DevW& W(int il, const char* suffix) const;
    bool moe_cpu(int il, const std::vector<int>& ids, const std::vector<float>& rw, std::string& err);
    // the int8 cache of layer il (glm_dsa_kv.cuh): codes, FP16 scales; the FP16 indexer keys of a full layer
    int8_t* kv_q(int il) const;
    void* kv_s(int il) const;
    void* ik_keys(int il) const;
    bool select_cells(int il, int n_vis, const float* iq, const float* iw, float* score, std::string& err);
    bool attend_one(int il, int n_sel, bool sparse, bool blas, const float* q, float* qabs, float* o, void* stream,
                    std::string& err);
    // the batched path (src/core/glm_dsa_prefill.cu)
    bool pf_alloc(std::string& err);
    bool prefill_chunk(const int32_t* toks, int T, bool head, std::string& err);
    bool gemm(const DevW& w, const float* X, float* Y, int T, std::string& err);
    bool moe_cpu_batch(int il, int T, const int* ids, const float* rw, std::string& err);
    // the batched path's experts on the GPU (src/core/glm_dsa_gpu_moe.cu)
    bool gpu_moe_enabled() const;
    void gpu_moe_release();
    bool gpu_moe_alloc(int Tmax, std::string& err);
    bool moe_gpu_batch(int il, int T, const float* XN, const float* RW, const int* ids, float* out_dev, std::string& err);
    // decode on the batched path's kernels: the indexer's scores and the attention as GEMMs over every visible
    // cell (all heads at once), the top_k selection as a mask (STRATA_GLM_DECODE_BLAS=0: the per-head kernels)
    bool decode_index_scores(const void* ik16, const float* iq, const float* iw, float* score, int n_vis, std::string& err);
    bool decode_attend(const float* cache, int p, const float* q, float* qabs, float* o, const float* vb, bool masked,
                       std::string& err);

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
    // the attention cache in int8 (src/core/glm_dsa_kv.cuh): per layer int8 [max_ctx x cache_row] and FP16 scales
    // [max_ctx x cache_row/32]; the indexer's keys FP16 [max_ctx x idx_key] per FULL indexer layer
    std::vector<int8_t*> kvq_l_;
    std::vector<void*> kvs_l_, ik_l_;
    float* kvg_ = nullptr;        ///< up to top_k cache rows dequantized (decode past the small kernel; short prefill)
    std::vector<int64_t> ik_slot_;
    float* sc_ = nullptr;         ///< scratch arena
    void* q8_ = nullptr;          ///< the Q8_1 activation scratch
    int* d_cells_ = nullptr;
    float* d_logits_ = nullptr;   ///< the last head's logits (device)
    float* d_logits_buf_ = nullptr;
    int* d_tok_ = nullptr;        ///< the sampler's output (device)
    std::vector<int> cells_;      ///< the current selection (host copy), reused by shared layers
    std::vector<float> h_score_;

    // host scratch for the CPU experts
    std::unique_ptr<GlmCpuPool> pool_;
    std::vector<float> h_x_, h_moe_, h_ff_, h_dn_;
    std::vector<uint8_t> h_act_, h_hq_;
    std::vector<float> h_embd_;

    // the batched path's state (allocated by the first prefill)
    void* cublas_ = nullptr;
    int pf_chunk_ = 0, pf_sub_ = 0;
    float* pf_ = nullptr;          ///< chunk activations
    float* deq_ = nullptr;         ///< one weight expanded to F32
    size_t deq_floats_ = 0;
    float* att_ = nullptr;         ///< attention / indexer score blocks
    float* pf_score_ = nullptr;    ///< the indexer's scores of a block of queries (pf_sub_ x max_ctx)
    int pf_dsub_ = 0, pf_ssub_ = 0;  ///< query blocks: dense attention (<= top_k cells), sparse attention
    float *sg_ = nullptr, *ss_ = nullptr, *qc_ = nullptr;   ///< sparse attention: gathered rows, scores, q
    int *d_sel_ = nullptr, *d_cnt_ = nullptr;   ///< the chunk's selections (chunk x top_k, -1 padded) and counts
    void* iq16_ = nullptr;         ///< the indexer's queries in FP16 (chunk x idx_heads x idx_key)
    std::vector<int> h_sel_, h_cnt_;
    std::vector<float> hb_x_, hb_ff_, hb_dn_, hb_moe_, hb_rw_, hb_score_;
    std::vector<int> hb_ids_;
    std::vector<uint8_t> hb_act_, hb_hq_;

    bool capture_ = false;
    /// STRATA_GLM_ROUTE_LOG=<file>: every routed token's experts appended as text lines 'layer id id ...'
    void route_log(int il, const int* ids, int n_tok) {
        static std::FILE* f = std::getenv("STRATA_GLM_ROUTE_LOG") ? std::fopen(std::getenv("STRATA_GLM_ROUTE_LOG"), "a") : nullptr;
        if (!f) return;
        for (int t = 0; t < n_tok; ++t) {
            std::fprintf(f, "%d", il);
            for (int i = 0; i < g_.n_exp_used; ++i) std::fprintf(f, " %d", ids[t * g_.n_exp_used + i]);
            std::fputc(10, f);
        }
    }
    std::vector<float> cap_;
    std::vector<float> cap_ids_;
    void capture_ids(int il, const int* ids, const float* rw) {
        if (!capture_) return;
        for (int i = 0; i < g_.n_exp_used; ++i) {
            cap_ids_[((size_t) il * 2) * g_.n_exp_used + i] = (float) ids[i];
            cap_ids_[((size_t) il * 2 + 1) * g_.n_exp_used + i] = rw[i];
        }
    }
    void capture(int il, int half, const float* dev_row) {
        if (capture_)
            cudaMemcpy(cap_.data() + ((size_t) il * 2 + half) * g_.n_embd, dev_row, (size_t) g_.n_embd * sizeof(float),
                       cudaMemcpyDeviceToHost);
    }

    double t_dev_ = 0, t_cpu_ = 0, t_total_ = 0;
    double prof_acc_[12] = {};
    // the graph decode: a stream, per layer A (attention inputs), B (out-proj .. router), S (shared expert),
    // C (the routed result back + residual); pinned host words; the position on the device
    void* gstream_ = nullptr;
    void* gm_ = nullptr;            ///< the GPU experts' lanes (GmState, src/core/glm_dsa_gpu_moe.cu)
    bool gm_registered_ = false;
    std::vector<void*> gm_regs_;
    std::vector<void*> gA_, gB_, gS_, gC_;
    float* pin_ = nullptr;          ///< pinned: [rw 64 | ids 64 | x n_embd | moe n_embd | embd n_embd | pos 64]
    int* d_pos_ = nullptr;
    float* ikv_ = nullptr;          ///< the indexer key of the current token before it is stored
    const float* moe_x_ = nullptr;  ///< moe_cpu's input / output when not h_x_ / h_moe_
    float* moe_y_ = nullptr;
    int prof_n_ = 0;
    double diag_pos_ = 0, diag_sync_ = 0, diag_hand_ = 0;
    int diag_n_ = 0;
};

}  // namespace strata::core
