#ifndef BIT_PACKING_H
#define BIT_PACKING_H

#include "bench.h"
#include <chrono>
#include <cstdint>
#include <span>
#include <vector>

const std::chrono::nanoseconds BASELINE_BITWISE{250000};

struct bitwise_args {
    std::vector<std::int8_t> a;
    std::vector<std::int8_t> b;
    std::vector<std::int8_t> result;
    // TODO: You may want to add new params at the end...
    bitwise_args() = default;
};

void naive_bitwise(std::span<std::int8_t> result,
                   std::span<const std::int8_t> a,
                   std::span<const std::int8_t> b);
// TODO: Implement your version, and call it in stu_bitwise_wrapper
void stu_bitwise(std::span<std::int8_t> result, std::span<const std::int8_t> a,
                 std::span<const std::int8_t> b);

void naive_bitwise_wrapper(void *ctx);
void stu_bitwise_wrapper(void *ctx);

void initialize_bitwise(bitwise_args *args, const size_t size,
                                  const std::uint_fast64_t seed);

bool bitwise_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func);

#endif // BIT_PACKING_H
