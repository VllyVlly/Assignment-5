#include "trace_replay.h"

#include <algorithm>
#include <stdexcept>

#include <vector>
#include <thread>
#include <algorithm>
#include <cstdint>

namespace {

static inline uint64_t trace_replay_cost(const RequestRecord& record) {
    uint64_t cost = 0;
    cost += record.base_cost;
    cost += 2ull * record.retry_penalty;
    cost += record.miss_penalty;
    cost += record.bytes >> 4;
    return cost;
}

} // namespace

void initialize_trace_replay(trace_replay_args& args,
                             size_t record_count,
                             size_t trace_count,
                             uint32_t seed) {
    if (record_count == 0) {
        throw std::invalid_argument(
            "initialize_trace_replay: records must be non-empty.");
    }
    if (trace_count == 0) {
        throw std::invalid_argument(
            "initialize_trace_replay: trace must be non-empty.");
    }

    args.out = 0;
    args.records.resize(record_count);
    args.trace.resize(trace_count);

    uint32_t current = seed;

    for (size_t i = 0; i < args.records.size(); ++i) {
        current = current * 1664525u + 1013904223u;
        const uint32_t r0 = current;
        current = current * 1664525u + 1013904223u;
        const uint32_t r1 = current;

        args.records[i].base_cost = 20u + (r0 & 255u);
        args.records[i].retry_penalty = 1u + ((r0 >> 8) & 31u);
        args.records[i].miss_penalty = 1u + (r1 & 63u);
        args.records[i].bytes = 64u + ((r1 >> 8) & 511u);

        for (int k = 0; k < 24; ++k) {
            args.records[i].padding[k] =
                r0 ^ (r1 + static_cast<uint32_t>(k) * 17u);
        }
    }

    const uint32_t record_count_u32 = static_cast<uint32_t>(args.records.size());
    const uint32_t window_size = std::min<uint32_t>(1024u, record_count_u32);
    const uint32_t window_mask = window_size - 1;
    const uint32_t segment_len = 256u;
    const uint32_t window_count =
        std::max<uint32_t>(1u, record_count_u32 / window_size);

    uint32_t base = 0;
    uint32_t stride = 1;
    for (size_t i = 0; i < args.trace.size(); ++i) {
        if ((i % segment_len) == 0) {
            current = current * 1664525u + 1013904223u;
            base = (current % window_count) * window_size;

            current = current * 1664525u + 1013904223u;
            stride = ((current >> 3) & window_mask) | 1u;
        }

        const uint32_t local =
            static_cast<uint32_t>(i % segment_len) & window_mask;
        args.trace[i] = base + ((local * stride) & window_mask);
    }
}

void naive_trace_replay(uint64_t& out,
                        const std::vector<RequestRecord>& records,
                        const std::vector<uint32_t>& trace) {
    uint64_t total = 0;
    const uint64_t order_mix = 1315423911ull;

    for (size_t i = 0; i < trace.size(); ++i) {
        total = total * order_mix + trace_replay_cost(records[trace[i]]);
    }

    out = total;
}

static inline uint64_t fast_pow_u64(uint64_t base, size_t exp) {
    uint64_t result = 1;

    while (exp > 0) {
        if (exp & 1) {
            result *= base;
        }

        base *= base;
        exp >>= 1;
    }

    return result;
}

void stu_trace_replay(uint64_t& out,
                      const std::vector<RequestRecord>& records,
                      const std::vector<uint32_t>& trace) {
    const uint64_t order_mix = 1315423911ull;
    const size_t n = trace.size();

    if (n == 0) {
        out = 0;
        return;
    }

    constexpr size_t MIN_WORK_PER_THREAD = 4096;

    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;

    size_t num_threads = 1;

    if (n >= MIN_WORK_PER_THREAD) {
        num_threads = std::min<size_t>(
            hw,
            (n + MIN_WORK_PER_THREAD - 1) / MIN_WORK_PER_THREAD
        );
    }

    if (num_threads <= 1) {
        uint64_t total = 0;

        for (size_t i = 0; i < n; ++i) {
            const size_t pf = i + 32;
            if (pf < n) {
                __builtin_prefetch(&records[trace[pf]], 0, 1);
            }

            total = total * order_mix + trace_replay_cost(records[trace[i]]);
        }

        out = total;
        return;
    }

    struct ChunkResult {
        uint64_t hash;
        uint64_t power;
        size_t length;
    };

    std::vector<ChunkResult> partials(num_threads);

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    const size_t chunk_size = (n + num_threads - 1) / num_threads;

    for (size_t tid = 0; tid < num_threads; ++tid) {
        const size_t begin = tid * chunk_size;
        const size_t end = std::min(n, begin + chunk_size);

        if (begin >= end) {
            partials[tid] = {0, 1, 0};
            continue;
        }

        threads.emplace_back([&, tid, begin, end]() {
            uint64_t local = 0;

            for (size_t i = begin; i < end; ++i) {
                const size_t pf = i + 32;

                if (pf < end) {
                    __builtin_prefetch(&records[trace[pf]], 0, 1);
                }

                local = local * order_mix + trace_replay_cost(records[trace[i]]);
            }

            const size_t len = end - begin;

            partials[tid].hash = local;
            partials[tid].power = fast_pow_u64(order_mix, len);
            partials[tid].length = len;
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    uint64_t total = 0;

    for (size_t tid = 0; tid < num_threads; ++tid) {
        if (partials[tid].length == 0) continue;

        total = total * partials[tid].power + partials[tid].hash;
    }

    out = total;
}

void naive_trace_replay_wrapper(void* ctx) {
    auto& args = *static_cast<trace_replay_args*>(ctx);
    naive_trace_replay(args.out, args.records, args.trace);
}

void stu_trace_replay_wrapper(void* ctx) {
    auto& args = *static_cast<trace_replay_args*>(ctx);
    stu_trace_replay(args.out, args.records, args.trace);
}

bool trace_replay_check(void* stu_ctx,
                        void* ref_ctx,
                        lab_test_func naive_func) {
    naive_func(ref_ctx);

    auto& stu_args = *static_cast<trace_replay_args*>(stu_ctx);
    auto& ref_args = *static_cast<trace_replay_args*>(ref_ctx);
    return stu_args.out == ref_args.out;
}
