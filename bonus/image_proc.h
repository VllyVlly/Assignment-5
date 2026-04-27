#ifndef IMAGE_PROC_H
#define IMAGE_PROC_H

#include "bench.h"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

const std::chrono::nanoseconds BASELINE_IMAGE_PROC{43000000};

struct image_proc_args {
    std::vector<float> r_channel;
    std::vector<float> g_channel;
    std::vector<float> b_channel;
    std::vector<float> output;
    
    float threshold;
    size_t width;
    size_t height;
    double epsilon;
    // TODO: You may want to add new params here
    explicit image_proc_args(double eps = 1e-5) : threshold(0.5f), width(0), height(0), epsilon(eps) {}
};

// Core functions
void naive_image_proc(image_proc_args& args);
// TODO: Student Implementation
void stu_image_proc(image_proc_args& args);

// Benchmarking wrappers
void naive_image_proc_wrapper(void *ctx);
void stu_image_proc_wrapper(void *ctx);

// Utilities
void initialize_image_proc(image_proc_args *args, size_t w, size_t h, uint64_t seed);
bool image_proc_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func);

#endif // IMAGE_PROC_H
