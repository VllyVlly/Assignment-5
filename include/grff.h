#ifndef GRFF_H
#define GRFF_H

#include "bench.h"
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

const std::chrono::nanoseconds BASELINE_GRFF{8500000};

struct grff_args {
    // Input Features
    std::vector<float> a_features;
    std::vector<float> b_features;
    std::vector<float> c_features;
    
    // Output
    std::vector<float> f_output;
    
    double epsilon;
    // TODO: You may want to add new params here

    explicit grff_args(double epsilon_in = 1e-5) : epsilon{epsilon_in} {}
};

void naive_grff(grff_args& args);
// TODO: Student Implementation
void stu_grff(grff_args& args);

void naive_grff_wrapper(void *ctx);
void stu_grff_wrapper(void *ctx);

void initialize_grff(grff_args *args, const size_t size, const std::uint_fast64_t seed);
bool grff_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func);

#endif // GRFF_H