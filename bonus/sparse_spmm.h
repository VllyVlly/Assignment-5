#ifndef SPARSE_SPMM_H
#define SPARSE_SPMM_H

#include "bench.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

struct CSRMatrix {
    int rows = 0;
    int cols = 0;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<float> values;
};

const std::chrono::nanoseconds BASELINE_SPARSE_SPMM{116000000};

struct sparse_spmm_args {
    std::vector<float> out;
    CSRMatrix csr;
    std::vector<float> dense_t;
    double epsilon = 1e-3;
    // TODO: You may want to add new params at the end...
};

void initialize_spmm(sparse_spmm_args &args,
                     int block_row_count = 512,
                     int block_col_count = 512,
                     int dense_cols = -1,
                     const std::vector<int> &diagonal_offsets =
                         std::vector<int>{},
                     unsigned int seed = 12345u);

// Check the data's validity in CSR format
inline bool validate_csr(const CSRMatrix &csr) {
    if (csr.rows < 0 || csr.cols < 0)
        return false;
    if (static_cast<int>(csr.row_ptr.size()) != csr.rows + 1)
        return false;
    if (csr.row_ptr.empty())
        return csr.rows == 0;
    if (csr.row_ptr[0] != 0)
        return false;

    const int nnz = static_cast<int>(csr.values.size());
    if (static_cast<int>(csr.col_idx.size()) != nnz)
        return false;
    if (csr.row_ptr.back() != nnz)
        return false;

    for (int i = 1; i < static_cast<int>(csr.row_ptr.size()); ++i) {
        if (csr.row_ptr[i] < csr.row_ptr[i - 1])
            return false;
    }
    for (int i = 0; i < nnz; ++i) {
        if (csr.col_idx[i] < 0 || csr.col_idx[i] >= csr.cols)
            return false;
    }
    return true;
}

// Convert CSR to normal row-major 2D format
inline std::vector<float> csr_to_dense_row_major(const CSRMatrix &csr) {
    if (!validate_csr(csr)) {
        throw std::invalid_argument(
            "csr_to_dense_row_major: invalid CSR matrix.");
    }

    std::vector<float> dense(csr.rows * csr.cols, 0.0f);
    for (int r = 0; r < csr.rows; ++r) {
        for (int p = csr.row_ptr[r]; p < csr.row_ptr[r + 1]; ++p) {
            dense[r * csr.cols + csr.col_idx[p]] = csr.values[p];
        }
    }
    return dense;
}

// print sparse matrix for debugging
inline void print_dense_matrix(const CSRMatrix &csr) {
    const std::vector<float> dense = csr_to_dense_row_major(csr);
    std::cout << "Dense matrix (" << csr.rows << "x" << csr.cols << ")\n";
    for (int r = 0; r < csr.rows; ++r) {
        for (int c = 0; c < csr.cols; ++c) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(1)
                      << dense[r * csr.cols + c] << " ";
        }
        std::cout << "\n";
    }
}

void csr_spmm(const CSRMatrix &csr, const std::vector<float> &dense_t,
              std::vector<float> &out);

void naive_sparse_spmm_wrapper(void *ctx);

// TODO: Implement your version (e.g. stu_csr_spmm), and call it in stu_sparse_spmm_wrapper
void stu_sparse_spmm_wrapper(void *ctx);

bool sparse_spmm_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func);

#endif // SPARSE_SPMM_H
