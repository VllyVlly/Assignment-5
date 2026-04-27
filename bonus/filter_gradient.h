#ifndef FILTER_GRADIENT_H
#define FILTER_GRADIENT_H

#include "bench.h"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

inline constexpr std::chrono::nanoseconds BASELINE_FILTER_GRADIENT{25000000};

struct data_struct {
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> c;
    std::vector<float> d;
    std::vector<float> e;
    std::vector<float> f;
    std::vector<float> g;
    std::vector<float> h;
    std::vector<float> i;
};

struct filter_gradient_args {
    data_struct data; 
    // TODO: You may want to add new params at the end...

    std::size_t width;
    std::size_t height;
    float out;
    double epsilon;

    explicit filter_gradient_args(double epsilon_in = 1e-6)
        : width(0), height(0), out(0.0f), epsilon(epsilon_in) {}
};

// TODO: You may need to add a function to convert data structure (not 
// included in time measurement), then implement your version in 
// stu_filter_gradient, whch is called by stu_filter_gradient_wrapper.

void naive_filter_gradient(float& out, const data_struct& data,
                   std::size_t width, std::size_t height);
void stu_filter_gradient(float& out, const data_struct& data,
                   std::size_t width, std::size_t height);

void naive_filter_gradient_wrapper(void* ctx);
void stu_filter_gradient_wrapper(void* ctx);

void initialize_filter_gradient(filter_gradient_args* args,
                        std::size_t width,
                        std::size_t height,
                        std::uint_fast64_t seed);

bool filter_gradient_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func);

#endif // filter_gradient_H