// src/program/glm_dsa_run.cpp - drive the GLM-5.3 (glm-dsa) runner from token ids: teacher-forced log-probs,
// perplexity and greedy continuation, printed as JSON lines so they can be held against llama.cpp's.
//
//   glm_dsa_run --model <any shard>.gguf --tokens 1,2,3 [--tokens-file f] [--ctx 4096] [--dev -1]
//               [--gen 32] [--top 5] [--eval]
//
// --eval prints, for every position, the top-N log-probs and the log-prob of the token that follows (and the
// perplexity at the end); without it only the last prompt position's head runs.  --gen N continues greedily.
#include "strata/core/glm_dsa_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<int32_t> parse_ids(const std::string& s) {
    std::vector<int32_t> out;
    std::string cur;
    for (char c : s) {
        if ((c >= '0' && c <= '9') || c == '-') {
            cur += c;
        } else if (!cur.empty()) {
            out.push_back((int32_t) std::atol(cur.c_str()));
            cur.clear();
        }
    }
    if (!cur.empty()) out.push_back((int32_t) std::atol(cur.c_str()));
    return out;
}

/// log-softmax summary: the top-n (id, logprob) and the log-prob of `want` (when >= 0).
void summarize(const std::vector<float>& logits, int n, int want, std::vector<std::pair<int, float>>& top,
               float& want_lp) {
    float mx = -INFINITY;
    for (float v : logits) mx = std::max(mx, v);
    double sum = 0;
    for (float v : logits) sum += std::exp((double) v - mx);
    const float lse = mx + (float) std::log(sum);
    std::vector<int> idx(logits.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + n, idx.end(),
                      [&](int a, int b) { return logits[(size_t) a] > logits[(size_t) b]; });
    top.clear();
    for (int i = 0; i < n; ++i) top.emplace_back(idx[(size_t) i], logits[(size_t) idx[(size_t) i]] - lse);
    want_lp = want >= 0 ? logits[(size_t) want] - lse : 0.0f;
}

void print_top(const char* kind, int pos, const std::vector<std::pair<int, float>>& top, int next, float next_lp) {
    std::printf("{\"%s\":%d,\"top\":[", kind, pos);
    for (size_t i = 0; i < top.size(); ++i)
        std::printf("%s[%d,%.5f]", i ? "," : "", top[i].first, top[i].second);
    std::printf("]");
    if (next >= 0) std::printf(",\"next\":%d,\"next_lp\":%.5f", next, next_lp);
    std::printf("}\n");
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    std::string model, tokens_arg, tokens_file;
    int64_t ctx = 4096;
    int dev = -1, gen = 0, top_n = 5;
    bool eval = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--model") model = next();
        else if (a == "--tokens") tokens_arg = next();
        else if (a == "--tokens-file") tokens_file = next();
        else if (a == "--ctx") ctx = std::atoll(next().c_str());
        else if (a == "--dev") dev = std::atoi(next().c_str());
        else if (a == "--gen") gen = std::atoi(next().c_str());
        else if (a == "--top") top_n = std::atoi(next().c_str());
        else if (a == "--eval") eval = true;
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (!tokens_file.empty()) {
        std::ifstream f(tokens_file);
        std::stringstream ss;
        ss << f.rdbuf();
        tokens_arg += " " + ss.str();
    }
    const std::vector<int32_t> toks = parse_ids(tokens_arg);
    if (model.empty() || toks.empty()) {
        std::fprintf(stderr, "usage: glm_dsa_run --model <shard>.gguf --tokens 1,2,3 [--eval] [--gen N]\n");
        return 2;
    }

    strata::core::GlmDsaModel m;
    std::string err;
    if (!m.load(model, ctx, dev, err)) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    const auto& g = m.geometry();
    std::fprintf(stderr, "glm-dsa: %d layers, n_embd %d, %d experts top-%d, device %d, ctx %lld\n", g.n_layers,
                 g.n_embd, g.n_expert, g.n_exp_used, m.device(), (long long) ctx);

    std::vector<float> logits((size_t) g.n_vocab);
    std::vector<std::pair<int, float>> top;
    double nll = 0;
    int n_nll = 0;
    for (size_t i = 0; i < toks.size(); ++i) {
        const bool last = i + 1 == toks.size();
        if (!m.step(toks[i], (eval || last) ? logits.data() : nullptr, err)) {
            std::fprintf(stderr, "step %zu failed: %s\n", i, err.c_str());
            return 1;
        }
        double dms, cms, tms;
        m.take_timing(dms, cms, tms);
        std::fprintf(stderr, "pos %zu: %.0f ms (device %.0f, cpu experts %.0f)\n", i, tms, dms, cms);
        if (eval || last) {
            const int nxt = last ? -1 : toks[i + 1];
            float lp = 0;
            summarize(logits, top_n, nxt, top, lp);
            print_top("pos", (int) i, top, nxt, lp);
            if (nxt >= 0) {
                nll -= lp;
                ++n_nll;
            }
        }
    }
    if (eval && n_nll > 0) std::printf("{\"ppl\":%.4f,\"n\":%d}\n", std::exp(nll / n_nll), n_nll);

    for (int k = 0; k < gen; ++k) {
        float lp = 0;
        summarize(logits, top_n, -1, top, lp);
        const int tok = top[0].first;
        print_top("gen", (int) toks.size() + k, top, -1, 0);
        if (!m.step(tok, logits.data(), err)) {
            std::fprintf(stderr, "gen step failed: %s\n", err.c_str());
            return 1;
        }
        double dms, cms, tms;
        m.take_timing(dms, cms, tms);
        std::fprintf(stderr, "gen %d: %.0f ms (device %.0f, cpu experts %.0f)\n", k, tms, dms, cms);
    }
    return 0;
}
