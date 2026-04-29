#include "sparse_spmm.h"

#include <array>
#include <vector>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>

static unsigned int sparse_runtime_salt() {
    static const unsigned int salt = [] {
        std::random_device rd;
        std::seed_seq seq{rd(), rd(), rd(), rd()};
        std::array<unsigned int, 1> values{};
        seq.generate(values.begin(), values.end());
        return values[0];
    }();
    return salt;
}

static CSRMatrix build_sparse_matrix(int block_row_count, int block_col_count,
                                     const std::vector<int> &diagonal_offsets,
                                     unsigned int seed) {
    if (block_row_count <= 0 || block_col_count <= 0) {
        throw std::invalid_argument(
            "initialize_spmm: block counts must be positive.");
    }

    const int min_offset = -(block_row_count - 1);
    const int max_offset = block_col_count - 1;

    std::vector<int> offsets;
    if (diagonal_offsets.empty()) {
        constexpr int kDefaultOffsets[] = {-456, -123, 0, 137, 246};
        for (int offset : kDefaultOffsets) {
            if (offset >= min_offset && offset <= max_offset) {
                offsets.push_back(offset);
            }
        }
        if (offsets.empty()) {
            offsets.push_back(0);
        }
    } else {
        offsets = diagonal_offsets;
        for (size_t i = 1; i < offsets.size(); ++i) {
            if (offsets[i - 1] >= offsets[i]) {
                throw std::invalid_argument(
                    "initialize_spmm: diagonal_offsets must be strictly "
                    "increasing.");
            }
        }
        for (int offset : offsets) {
            if (offset < min_offset || offset > max_offset) {
                throw std::invalid_argument(
                    "initialize_spmm: diagonal_offsets contain an out-of-range "
                    "value.");
            }
        }
    }

    CSRMatrix csr;
    csr.rows = block_row_count * 4;
    csr.cols = block_col_count * 4;

    std::vector<std::vector<int>> row_cols(csr.rows);
    std::vector<std::vector<float>> row_vals(csr.rows);
    const size_t n_offsets = offsets.size();
    for (int r = 0; r < csr.rows; ++r) {
        row_cols[r].reserve(n_offsets * 4);
        row_vals[r].reserve(n_offsets * 4);
    }

    std::seed_seq value_seed{seed, sparse_runtime_salt(), 0x85ebca6bu};
    std::mt19937 rng(value_seed);
    std::uniform_int_distribution<int> value_dist(-10, 10);

    for (int br = 0; br < block_row_count; ++br) {
        for (size_t od = 0; od < n_offsets; ++od) {
            const int bc = br + offsets[od];
            if (bc < 0 || bc >= block_col_count)
                continue;

            for (int lr = 0; lr < 4; ++lr) {
                const int row = br * 4 + lr;
                std::vector<int> &cols = row_cols[row];
                std::vector<float> &vals = row_vals[row];

                for (int lc = 0; lc < 4; ++lc) {
                    const int v = value_dist(rng);
                    if (v == 0)
                        continue;

                    cols.push_back(bc * 4 + lc);
                    vals.push_back(v);
                }
            }
        }
    }

    csr.row_ptr.assign(csr.rows + 1, 0);
    for (int r = 0; r < csr.rows; ++r) {
        const int row_nnz = row_cols[r].size();
        csr.row_ptr[r + 1] = csr.row_ptr[r] + row_nnz;
    }

    const int total_nnz = csr.row_ptr.back();
    csr.col_idx.resize(total_nnz);
    csr.values.resize(total_nnz);

    for (int r = 0; r < csr.rows; ++r) {
        int out = csr.row_ptr[r];
        for (size_t k = 0; k < row_cols[r].size(); ++k) {
            csr.col_idx[out] = row_cols[r][k];
            csr.values[out] = row_vals[r][k];
            ++out;
        }
    }

    return csr;
}

void initialize_spmm(sparse_spmm_args &args, int block_row_count,
                     int block_col_count, int dense_cols,
                     const std::vector<int> &diagonal_offsets,
                     unsigned int seed) {
    args.csr = build_sparse_matrix(block_row_count,
                                   block_col_count,
                                   diagonal_offsets,
                                   seed);

    if (dense_cols <= 0) {
        dense_cols = args.csr.cols;
    }

    const size_t dense_cols_sz = dense_cols;
    const size_t csr_cols_sz = args.csr.cols;
    const size_t csr_rows_sz = args.csr.rows;
    args.dense_t.resize(dense_cols_sz * csr_cols_sz);
    args.out.resize(csr_rows_sz * dense_cols_sz);
    args.epsilon = 1e-3;

    std::seed_seq dense_seed{seed, sparse_runtime_salt(), 0x9e3779b9u};
    std::mt19937 rng(dense_seed);
    std::uniform_real_distribution<float> dense_dist(-1.0f, 1.0f);
    for (size_t i = 0; i < args.dense_t.size(); ++i) {
        args.dense_t[i] = dense_dist(rng);
    }

    std::fill(args.out.begin(), args.out.end(), 0.0f);
    volatile float touch = 0.0f;
    for (size_t i = 0; i < args.dense_t.size(); ++i)
        touch = touch + args.dense_t[i];
    for (size_t i = 0; i < args.out.size(); ++i)
        touch = touch + args.out[i];
    for (size_t i = 0; i < args.csr.values.size(); ++i)
        touch = touch + args.csr.values[i];
    (void)touch;
}

void csr_spmm(const CSRMatrix &csr, const std::vector<float> &dense_t,
              std::vector<float> &out) {
    if (!validate_csr(csr)) {
        throw std::invalid_argument("csr_spmm: invalid CSR matrix.");
    }
    const size_t rows = csr.rows;
    const size_t cols = csr.cols;
    if (rows == 0 || cols == 0) {
        if (!dense_t.empty() || !out.empty()) {
            throw std::invalid_argument(
                "csr_spmm: non-empty dense buffers for empty CSR shape.");
        }
        return;
    }
    if (dense_t.size() % cols != 0) {
        throw std::invalid_argument(
            "csr_spmm: dense_t.size() must be a multiple of csr.cols.");
    }
    const size_t dense_cols = dense_t.size() / cols;
    if (dense_cols == 0) {
        throw std::invalid_argument("csr_spmm: dense_cols must be positive.");
    }
    if (out.size() != rows * dense_cols) {
        throw std::invalid_argument("csr_spmm: out size mismatch.");
    }

    for (int r = 0; r < csr.rows; ++r) {
        float *out_row = &out[r * dense_cols];
        for (size_t n = 0; n < dense_cols; ++n) {
            const float *bt_row = &dense_t[n * cols];
            float acc = 0.0f;
            for (int p = csr.row_ptr[r]; p < csr.row_ptr[r + 1]; ++p) {
                acc += csr.values[p] * bt_row[csr.col_idx[p]];
            }
            out_row[n] = acc;
        }
    }
}

void naive_sparse_spmm_wrapper(void *ctx) {
    auto &args = *static_cast<sparse_spmm_args *>(ctx);
    csr_spmm(args.csr, args.dense_t, args.out);
}

void stu_csr_spmm(const CSRMatrix &csr, const std::vector<float> &dense_t,
                  std::vector<float> &out) {
    if (!validate_csr(csr)) {
        throw std::invalid_argument("stu_csr_spmm: invalid CSR matrix.");
    }

    const size_t rows = csr.rows;
    const size_t cols = csr.cols;

    if (rows == 0 || cols == 0) {
        if (!dense_t.empty() || !out.empty()) {
            throw std::invalid_argument(
                "stu_csr_spmm: non-empty dense buffers for empty CSR shape.");
        }
        return;
    }

    if (dense_t.size() % cols != 0) {
        throw std::invalid_argument(
            "stu_csr_spmm: dense_t.size() must be a multiple of csr.cols.");
    }

    const size_t dense_cols = dense_t.size() / cols;

    if (dense_cols == 0) {
        throw std::invalid_argument("stu_csr_spmm: dense_cols must be positive.");
    }

    if (out.size() != rows * dense_cols) {
        throw std::invalid_argument("stu_csr_spmm: out size mismatch.");
    }

    const float *dense_src = dense_t.data();
    const float *values = csr.values.data();
    const auto *col_idx = csr.col_idx.data();
    const auto *row_ptr = csr.row_ptr.data();
    float *out_data = out.data();

    std::vector<float> dense_by_col(cols * dense_cols);

    for (size_t n = 0; n < dense_cols; ++n) {
        const float *src_row = dense_src + n * cols;
        for (size_t c = 0; c < cols; ++c) {
            dense_by_col[c * dense_cols + n] = src_row[c];
        }
    }

    const float *dense = dense_by_col.data();

    const size_t block8_end = dense_cols - (dense_cols % 8);

    for (size_t r = 0; r < rows; ++r) {
        const size_t row_start = static_cast<size_t>(row_ptr[r]);
        const size_t row_end = static_cast<size_t>(row_ptr[r + 1]);

        float *out_row = out_data + r * dense_cols;

        size_t n = 0;

        // Main optimized path: 8 output columns at once.
        for (; n < block8_end; n += 8) {
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            float acc4 = 0.0f;
            float acc5 = 0.0f;
            float acc6 = 0.0f;
            float acc7 = 0.0f;

            for (size_t p = row_start; p < row_end; ++p) {
                const size_t c = static_cast<size_t>(col_idx[p]);
                const float v = values[p];

                const float *d = dense + c * dense_cols + n;

                acc0 += v * d[0];
                acc1 += v * d[1];
                acc2 += v * d[2];
                acc3 += v * d[3];
                acc4 += v * d[4];
                acc5 += v * d[5];
                acc6 += v * d[6];
                acc7 += v * d[7];
            }

            out_row[n + 0] = acc0;
            out_row[n + 1] = acc1;
            out_row[n + 2] = acc2;
            out_row[n + 3] = acc3;
            out_row[n + 4] = acc4;
            out_row[n + 5] = acc5;
            out_row[n + 6] = acc6;
            out_row[n + 7] = acc7;
        }

        // Leftover columns.
        for (; n < dense_cols; ++n) {
            float acc = 0.0f;

            for (size_t p = row_start; p < row_end; ++p) {
                const size_t c = static_cast<size_t>(col_idx[p]);
                acc += values[p] * dense[c * dense_cols + n];
            }

            out_row[n] = acc;
        }
    }
}

void stu_sparse_spmm_wrapper(void *ctx) {
    auto &args = *static_cast<sparse_spmm_args *>(ctx);
    stu_csr_spmm(args.csr, args.dense_t, args.out);
}

bool sparse_spmm_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);

    auto &stu_args = *static_cast<sparse_spmm_args *>(stu_ctx);
    auto &ref_args = *static_cast<sparse_spmm_args *>(ref_ctx);
    const double eps = ref_args.epsilon;
    if (stu_args.out.size() != ref_args.out.size())
        return false;

    const double atol = 2e-6;
    for (size_t i = 0; i < ref_args.out.size(); ++i) {
        const double r = static_cast<double>(ref_args.out[i]);
        const double s = static_cast<double>(stu_args.out[i]);
        const double err = std::abs(r - s);
        if (err > (atol + eps * std::abs(r))) {
            debug_log("\tDEBUG: sparse_spmm mismatch at {}: ref={} stu={} err={} thr={}\n",
                      i,
                      r,
                      s,
                      err,
                      (atol + eps * std::abs(r)));
            return false;
        }
    }
    return true;
}