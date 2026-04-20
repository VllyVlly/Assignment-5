#ifndef TRACE_REPLAY_H
#define TRACE_REPLAY_H

#include "bench.h"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

struct RequestRecord {
    uint32_t base_cost;
    uint32_t retry_penalty;
    uint32_t miss_penalty;
    uint32_t bytes;
    uint32_t padding[24];
};

const std::chrono::nanoseconds BASELINE_TRACE_REPLAY{3400000};

struct trace_replay_args {
    uint64_t out = 0;
    std::vector<RequestRecord> records;
    std::vector<uint32_t> trace;
    // TODO: You may want to add new params at the end...
};

void initialize_trace_replay(trace_replay_args& args,
                             size_t record_count = 1 << 16,
                             size_t trace_count = 1 << 20,
                             uint32_t seed = 12345u);

void naive_trace_replay(uint64_t& out,
                        const std::vector<RequestRecord>& records,
                        const std::vector<uint32_t>& trace);
// TODO: Implement your version, and call it in stu_trace_replay_wrapper
void stu_trace_replay(uint64_t& out,
                      const std::vector<RequestRecord>& records,
                      const std::vector<uint32_t>& trace);

void naive_trace_replay_wrapper(void* ctx);
void stu_trace_replay_wrapper(void* ctx);

bool trace_replay_check(void* stu_ctx,
                        void* ref_ctx,
                        lab_test_func naive_func);

#endif // TRACE_REPLAY_H
