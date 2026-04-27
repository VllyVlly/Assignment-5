#ifndef MATMUL_H
#define MATMUL_H

#include "bench.h"
#include <chrono>
#include <cstdint>
#include <vector>

const std::chrono::nanoseconds BASELINE_MATMUL{88000000};

struct matmul_args {
    std::vector<float> C;
    std::vector<float> A;
    std::vector<float> B;
    int n = 0;
    double epsilon = 1e-3;
    // TODO: You may want to add new params at the end...
};

void initialize_matmul(matmul_args& args,
                       int n = 512,
                       uint32_t seed = 12345u);

// C = A * B. Matrices are N*N flat vectors.
void naive_matmul(std::vector<float>& C,
                  const std::vector<float>& A,
                  const std::vector<float>& B,
                  int n);
// TODO: Implement your version, and call it in stu_matmul_wrapper
void stu_matmul(std::vector<float>& C,
                const std::vector<float>& A,
                const std::vector<float>& B,
                int n);

void naive_matmul_wrapper(void* ctx);
void stu_matmul_wrapper(void* ctx);

bool matmul_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func);

#endif // MATMUL_H
