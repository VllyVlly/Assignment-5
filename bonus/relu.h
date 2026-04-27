#ifndef RELU_H
#define RELU_H

#include "bench.h"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

const std::chrono::nanoseconds BASELINE_RELU{550000};

struct relu_args {
    std::vector<float> data;
    double epsilon;

    explicit relu_args(double epsilon_in = 1e-6) : epsilon{epsilon_in} {}
};

void naive_relu(std::span<float> data);
// TODO: Implement your version, and call it in stu_relu_wrapper
void stu_relu(std::span<float> data);

void naive_relu_wrapper(void *ctx);
void stu_relu_wrapper(void *ctx);

void initialize_relu(relu_args *args, const size_t size,
                     const std::uint_fast64_t seed);

bool relu_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func);

#endif // RELU_H
