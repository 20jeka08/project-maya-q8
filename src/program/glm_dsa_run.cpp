// src/program/glm_dsa_run.cpp - drive the GLM-5.3 (glm-dsa) runner from token ids: teacher-forced log-probs,
// perplexity and greedy continuation, printed as JSON lines so they can be held against llama.cpp's.
//
//   glm_dsa_run --model <any shard>.gguf --tokens 1,2,3 [--tokens-file f] [--ctx 4096] [--dev -1]
//               [--gen 32] [--top 5] [--eval]
//   glm_dsa_run --model <any shard>.gguf --serve [--max-context 8192]     (Maya's engine protocol, see serve())
//
// --eval prints, for every position, the top-N log-probs and the log-prob of the token that follows (and the
// perplexity at the end); without it only the last prompt position's head runs.  --gen N continues greedily.
#include "strata/core/glm_dsa_model.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <thread>
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

// ---------------------------------------------------------------- --serve: Maya's engine wire contract
// (serve/server.py StrataEngine): READY <max_context> stop, then per request
//   GEN <max_new> [temperature=F top_p=F top_k=N min_p=F seed=N ...] <id,id,...>   (the WHOLE conversation)
// answered by PP <read> <total> while the prompt is read, T <id> per generated token, and
//   DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <finish> 0 0 <reused>
// STOP ends the running request at its next token; QUIT (or the end of input) ends the process.  The cached
// conversation is kept: a GEN whose ids extend it reads only the new tokens.
struct Lines {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> q;
    std::atomic<bool> stop{false}, eof{false};
};

int serve(strata::core::GlmDsaModel& m) {
    const auto& g = m.geometry();
    static const int64_t kEos[] = {154820, 154827, 154829};   // <|endoftext|>, <|user|>, <|observation|>
    Lines in;
    std::thread reader([&] {
        std::string l;
        while (std::getline(std::cin, l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (l == "STOP") {
                in.stop = true;
                continue;
            }
            std::lock_guard<std::mutex> lk(in.mu);
            if (l == "QUIT") in.stop = true;   // a QUIT also cancels the running request
            in.q.push_back(l);
            in.cv.notify_one();
        }
        in.eof = true;
        in.stop = true;
        in.cv.notify_one();
    });
    reader.detach();

    std::printf("INFO engine=glm-dsa-0.1 model=glm-5.3\n");
    std::printf("READY %lld stop\n", (long long) m.max_ctx());
    std::fflush(stdout);
    std::vector<int32_t> hist;   // the tokens the cache holds, positions 0..pos-1
    uint64_t draws = 0;
    for (;;) {
        std::string line;
        {
            std::unique_lock<std::mutex> lk(in.mu);
            in.cv.wait(lk, [&] { return !in.q.empty() || in.eof.load(); });
            if (in.q.empty()) break;
            line = in.q.front();
            in.q.pop_front();
        }
        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;
        if (cmd == "QUIT") break;
        if (cmd.empty()) continue;
        if (cmd != "GEN") {
            std::printf("ERR unsupported command '%s' (GEN, STOP and QUIT)\n", cmd.c_str());
            std::fflush(stdout);
            continue;
        }
        in.stop = false;
        int64_t max_new = 0;
        ss >> max_new;
        strata::kernels::SamplerParams sp;
        sp.greedy = true;   // the server's contract: no temperature key means greedy
        sp.seed = (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count();
        std::string tok, ids;
        while (ss >> tok) {
            const size_t eq = tok.find('=');
            if (eq == std::string::npos) {
                ids = tok;
                continue;
            }
            const std::string k = tok.substr(0, eq), v = tok.substr(eq + 1);
            if (k == "temperature") {
                sp.temperature = (float) std::atof(v.c_str());
                sp.greedy = sp.temperature <= 0.0f;
            } else if (k == "top_p") sp.top_p = (float) std::atof(v.c_str());
            else if (k == "top_k") sp.top_k = std::atoi(v.c_str());
            else if (k == "min_p") sp.min_p = (float) std::atof(v.c_str());
            else if (k == "seed") sp.seed = (uint64_t) std::atoll(v.c_str());
        }
        const std::vector<int32_t> prompt = parse_ids(ids);
        bool bad = prompt.empty() || max_new <= 0;
        for (int32_t t : prompt) bad = bad || t < 0 || t >= g.n_vocab;
        if (bad) {
            std::printf("ERR malformed GEN line\n");
            std::fflush(stdout);
            continue;
        }
        if ((int64_t) prompt.size() >= m.max_ctx()) {
            std::printf("ERR the prompt (%zu tokens) does not fit the context (%lld)\n", prompt.size(),
                        (long long) m.max_ctx());
            std::fflush(stdout);
            continue;
        }

        // the prefix the cache already holds (at least the last prompt token is read again: its head is needed)
        size_t reuse = 0;
        while (reuse < hist.size() && reuse + 1 < prompt.size() && hist[reuse] == prompt[reuse]) ++reuse;
        m.rewind((int64_t) reuse);
        hist.resize(reuse);

        std::string err;
        const char* finish = "length";
        const auto t0 = std::chrono::steady_clock::now();
        bool failed = false;
        for (size_t i = reuse; i < prompt.size(); ++i) {
            const bool last = i + 1 == prompt.size();
            if (!m.step(prompt[i], nullptr, err, last)) {
                failed = true;
                break;
            }
            hist.push_back(prompt[i]);
            if ((i - reuse) % 8 == 7 || last) {
                std::printf("PP %zu %zu\n", i + 1 - reuse, prompt.size() - reuse);
                std::fflush(stdout);
            }
            if (in.stop) {
                finish = "cancel";
                break;
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        int64_t produced = 0;
        while (!failed && std::strcmp(finish, "cancel") != 0) {
            sp.counter = draws++;
            const int t = m.sample(sp, err);
            if (t < 0) {
                failed = true;
                break;
            }
            std::printf("T %d\n", t);
            std::fflush(stdout);
            ++produced;
            if (std::find(std::begin(kEos), std::end(kEos), (int64_t) t) != std::end(kEos)) {
                finish = "stop";
                break;
            }
            if (produced >= max_new) break;
            if (in.stop) {
                finish = "cancel";
                break;
            }
            if (m.pos() >= m.max_ctx()) break;   // the context is full: "length"
            if (!m.step(t, nullptr, err, true)) {
                failed = true;
                break;
            }
            hist.push_back(t);
            if (produced % 8 == 0) {
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
                std::printf("STAT tok_s=%.2f ms_tok=%.2f\n", produced * 1000.0 / ms, ms / produced);
                std::fflush(stdout);
            }
        }
        if (failed) {
            // a failed step leaves the cache unknown: forget the conversation
            m.reset();
            hist.clear();
            std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            continue;
        }
        const auto t2 = std::chrono::steady_clock::now();
        std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu\n", (long long) produced, prompt.size(),
                    std::chrono::duration<double, std::milli>(t1 - t0).count(),
                    std::chrono::duration<double, std::milli>(t2 - t1).count(), finish, reuse);
        std::fflush(stdout);
        double a, b, c;
        m.take_timing(a, b, c);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model, tokens_arg, tokens_file;
    int64_t ctx = 4096;
    int dev = -1, gen = 0, top_n = 5;
    bool eval = false, serve_mode = false;
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
        else if (a == "--serve") serve_mode = true;
        else if (a == "--max-context") ctx = std::atoll(next().c_str());
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
    if (model.empty() || (toks.empty() && !serve_mode)) {
        std::fprintf(stderr, "usage: glm_dsa_run --model <shard>.gguf --tokens 1,2,3 [--eval] [--gen N]\n");
        return 2;
    }

    strata::core::GlmDsaModel m;
    std::string err;
    if (!m.load(model, ctx, dev, err)) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    if (serve_mode) return serve(m);
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
