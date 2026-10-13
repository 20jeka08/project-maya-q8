// src/program/glm_dsa_pcie_bench.cpp - how fast the routed experts of GLM-5.3 can reach a GPU from the model's
// memory map: (a) cudaMemcpy straight from the mapping (pageable), (b) a multi-threaded copy into pinned staging
// then DMA, (c) cudaHostRegister on the mapping itself (read-only) then DMA, and (d) (b) on every GPU at once.
//
//   glm_dsa_pcie_bench --model <any shard>.gguf [--gb 4] [--threads 32]
#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/gguf_split.hpp"
#include "strata/core/glm_dsa_model.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {
double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_));     \
    } while (0)
}  // namespace

int main(int argc, char** argv) {
    std::string model;
    double gb = 4;
    int threads = 32;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--model") model = argv[i + 1];
        else if (a == "--gb") gb = std::atof(argv[i + 1]);
        else if (a == "--threads") threads = std::atoi(argv[i + 1]);
    }
    std::vector<std::unique_ptr<strata::GgufFile>> files;
    for (const auto& p : strata::gguf_split_paths(model)) files.emplace_back(new strata::GgufFile(p));
    if (std::getenv("PIN_GB")) {   // pinned host memory in 8 GiB pieces (Maya's RAM tier): how much, how fast
        const double want = std::atof(std::getenv("PIN_GB"));
        int n_dev = 0;
        cudaGetDeviceCount(&n_dev);
        for (int d = 0; d < n_dev; ++d) cudaSetDevice(d), cudaFree(nullptr);   // both contexts first
        cudaSetDevice(0);
        std::vector<uint8_t*> pieces;
        const size_t piece = (size_t) 8 << 30;
        const double t0 = now_s();
        while (pieces.size() * 8.0 * 1.073741824 < want) {
            uint8_t* p = nullptr;
            const cudaError_t e = cudaHostAlloc((void**) &p, piece, cudaHostAllocPortable);
            if (e != cudaSuccess) {
                std::printf("piece %zu refused: %s\n", pieces.size(), cudaGetErrorString(e));
                cudaGetLastError();
                break;
            }
            pieces.push_back(p);
        }
        std::printf("pinned %.0f GB in %zu pieces in %.1f s\n", pieces.size() * 8 * 1.073741824, pieces.size(),
                    now_s() - t0);
        // touch the pieces (as a copy from the mmap would) then DMA from them to every device at once
        for (auto* p : pieces) std::memset(p, 1, piece);
        std::vector<void*> d(n_dev);
        std::vector<cudaStream_t> st(n_dev);
        for (int k = 0; k < n_dev; ++k) {
            cudaSetDevice(k);
            CK(cudaMalloc(&d[k], 1u << 30));
            CK(cudaStreamCreate(&st[k]));
        }
        const double t1 = now_s();
        size_t moved = 0;
        for (size_t i = 0; i < pieces.size(); ++i) {
            const int k = (int) (i % n_dev);
            cudaSetDevice(k);
            for (size_t o = 0; o < piece; o += 1u << 30) {
                CK(cudaMemcpyAsync(d[k], pieces[i] + o, 1u << 30, cudaMemcpyHostToDevice, st[k]));
                moved += 1u << 30;
            }
        }
        for (int k = 0; k < n_dev; ++k) cudaSetDevice(k), cudaStreamSynchronize(st[k]);
        std::printf("DMA from pinned to all %d devices: %.1f GB/s\n", n_dev, moved / 1e9 / (now_s() - t1));
        // what the GPU experts need besides: plain device allocations on every card
        for (int k = 0; k < n_dev; ++k) {
            cudaSetDevice(k);
            void* q = nullptr;
            const cudaError_t e = cudaMalloc(&q, (size_t) 2 << 30);
            std::printf("device %d: 2 GB cudaMalloc beside it: %s\n", k, cudaGetErrorString(e));
            if (q) cudaFree(q);
        }
        return 0;
    }
    if (std::getenv("REG_ALL")) {   // register every shard's whole mapping; then an async copy from each
        const unsigned flags = std::atoi(std::getenv("REG_ALL"));
        cudaSetDevice(0);
        cudaStream_t cs;
        CK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
        void* d = nullptr;
        CK(cudaMalloc(&d, 64 << 20));
        double total = 0;
        for (size_t i = 0; i < files.size(); ++i) {
            const auto& f = *files[i];
            if (f.tensors().empty()) continue;
            uint8_t* base = (uint8_t*) f.tensor_data(f.tensors()[0]) - f.data_start() - f.tensors()[0].offset;
            const cudaError_t e = cudaHostRegister(base, f.file_size(), flags);
            if (e == cudaSuccess) total += f.file_size() / 1e9;
            std::printf("shard %zu (%.1f GB, base %% 4096 = %zu): register %s (total %.0f GB)\n", i, f.file_size() / 1e9,
                        (size_t) ((uintptr_t) base % 4096), cudaGetErrorString(e), total);
            cudaGetLastError();
            const cudaError_t c = cudaMemcpyAsync(d, f.tensor_data(f.tensors()[0]), 64 << 20, cudaMemcpyHostToDevice, cs);
            const cudaError_t w = cudaStreamSynchronize(cs);
            std::printf("    async copy: %s / %s\n", cudaGetErrorString(c), cudaGetErrorString(w));
            cudaGetLastError();
        }
        return 0;
    }
    // a run of expert tensors in one shard: the largest tensor of shard 5 and its successors
    const strata::GgufFile& f = *files[4];
    const uint8_t* base = f.tensor_data(f.tensors()[0]);
    const size_t bytes = (size_t) (gb * (1ull << 30));
    if (f.data_start() + f.tensors()[0].offset + bytes > f.file_size()) {
        std::fprintf(stderr, "shard too small\n");
        return 1;
    }
    // page it in first (the decode's warm-up does the same): a cold read would measure the disk
    {
        volatile uint64_t s = 0;
        strata::core::GlmCpuPool pool(threads);
        const size_t ch = 64u << 20;
        pool.run((int) ((bytes + ch - 1) / ch), [&](int i) {
            uint64_t a = 0;
            for (size_t o = (size_t) i * ch; o < std::min(bytes, (size_t) (i + 1) * ch); o += 4096) a += base[o];
            s += a;
        });
    }
    int n_dev = 0;
    CK(cudaGetDeviceCount(&n_dev));
    std::vector<void*> dbuf(n_dev);
    for (int d = 0; d < n_dev; ++d) {
        cudaDeviceProp pr{};
        cudaGetDeviceProperties(&pr, d);
        CK(cudaSetDevice(d));
        CK(cudaMalloc(&dbuf[d], 1u << 30));
        std::printf("device %d: %s, PCIe gen %s\n", d, pr.name, "?");
    }
    const size_t chunk = 1u << 30;   // 1 GB per transfer into the 1 GB device buffer (overwritten)

    // (a) pageable
    for (int d = 0; d < n_dev; ++d) {
        CK(cudaSetDevice(d));
        const double t0 = now_s();
        for (size_t o = 0; o < bytes; o += chunk) CK(cudaMemcpy(dbuf[d], base + o, std::min(chunk, bytes - o), cudaMemcpyHostToDevice));
        std::printf("(a) pageable mmap -> device %d: %.1f GB/s\n", d, bytes / 1e9 / (now_s() - t0));
    }

    // (b) multi-threaded copy into pinned staging (double buffered) + DMA
    const size_t stage = 256u << 20;
    std::vector<uint8_t*> pin(2 * n_dev);
    for (auto& p : pin) CK(cudaHostAlloc((void**) &p, stage, cudaHostAllocDefault));
    strata::core::GlmCpuPool pool(threads);
    const auto staged = [&](std::vector<int> devs) {
        std::vector<cudaStream_t> st(n_dev);
        for (int d : devs) {
            CK(cudaSetDevice(d));
            CK(cudaStreamCreate(&st[d]));
        }
        const double t0 = now_s();
        int k = 0;
        for (size_t o = 0; o < bytes; o += stage * devs.size(), ++k) {
            // each device takes its own slice of this round; the copy runs on the pool, all devices at once
            for (size_t j = 0; j < devs.size(); ++j) {
                const int d = devs[j];
                uint8_t* buf = pin[2 * d + (k & 1)];
                CK(cudaSetDevice(d));
                CK(cudaStreamSynchronize(st[d]));   // the buffer used two rounds ago is free again
                const size_t off = o + j * stage;
                if (off >= bytes) continue;
                const size_t n = std::min(stage, bytes - off);
                const size_t sub = n / (size_t) pool.threads() + 1;
                pool.run(pool.threads(), [&](int t) {
                    const size_t a = (size_t) t * sub, b = std::min(n, a + sub);
                    if (a < b) std::memcpy(buf + a, base + off + a, b - a);
                });
                CK(cudaMemcpyAsync((uint8_t*) dbuf[d] + ((k * stage) % chunk), buf, n, cudaMemcpyHostToDevice, st[d]));
            }
        }
        for (int d : devs) {
            CK(cudaSetDevice(d));
            CK(cudaStreamSynchronize(st[d]));
            cudaStreamDestroy(st[d]);
        }
        return bytes / 1e9 / (now_s() - t0);
    };
    for (int d = 0; d < n_dev; ++d) std::printf("(b) staged pinned -> device %d: %.1f GB/s\n", d, staged({d}));
    if (n_dev > 1) {
        std::vector<int> all;
        for (int d = 0; d < n_dev; ++d) all.push_back(d);
        std::printf("(d) staged pinned -> all %d devices at once: %.1f GB/s total\n", n_dev, staged(all));
    }

    // (c) register the mapping itself
    {
        const size_t reg = bytes;
        const double t0 = now_s();
        const cudaError_t e = cudaHostRegister((void*) base, reg, cudaHostRegisterReadOnly | cudaHostRegisterPortable);
        const double tr = now_s() - t0;
        if (e != cudaSuccess) {
            std::printf("(c) cudaHostRegister on the mapping: refused (%s)\n", cudaGetErrorString(e));
            cudaGetLastError();
        } else {
            std::printf("(c) cudaHostRegister of %.1f GB: %.2f s (%.1f GB/s)\n", reg / 1e9, tr, reg / 1e9 / tr);
            // two passes in opposite device orders: the first touch pays the page locking, not the link
            for (int pass = 0; pass < 2; ++pass)
                for (int dd = 0; dd < n_dev; ++dd) {
                    const int d = pass ? n_dev - 1 - dd : dd;
                    CK(cudaSetDevice(d));
                    const double t1 = now_s();
                    for (size_t o = 0; o < reg; o += chunk)
                        CK(cudaMemcpy(dbuf[d], base + o, std::min(chunk, reg - o), cudaMemcpyHostToDevice));
                    std::printf("(c) pass %d registered mmap -> device %d: %.1f GB/s\n", pass, d,
                                reg / 1e9 / (now_s() - t1));
                }
            if (n_dev > 1) {   // every device at once, each its own share
                std::vector<cudaStream_t> st(n_dev);
                const size_t share = reg / n_dev;
                const double t1 = now_s();
                for (int d = 0; d < n_dev; ++d) {
                    CK(cudaSetDevice(d));
                    CK(cudaStreamCreate(&st[d]));
                    for (size_t o = 0; o < share; o += chunk)
                        CK(cudaMemcpyAsync(dbuf[d], base + d * share + o, std::min(chunk, share - o),
                                           cudaMemcpyHostToDevice, st[d]));
                }
                for (int d = 0; d < n_dev; ++d) {
                    CK(cudaSetDevice(d));
                    CK(cudaStreamSynchronize(st[d]));
                }
                std::printf("(c) registered mmap -> all devices at once: %.1f GB/s total\n", reg / 1e9 / (now_s() - t1));
            }
            const double t2 = now_s();
            cudaHostUnregister((void*) base);
            std::printf("(c) unregister: %.2f s\n", now_s() - t2);
        }
    }
    return 0;
}
