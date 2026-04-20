#ifndef BENCH_H
#define BENCH_H

#include <cmath>
#include <chrono>
#include <print>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

//===----------------------------------------------------------------------===//
// In each <KERNEL_NAME>.h and <KERNEL_NAME>.cpp, they should define necessary
// data structure for its corresponding function, its check function as well as
// wrapper functions.

// The args in bench_t::tfunc and bench_t::naiveFunc are not necessarily the
// same, and the check function just needs to check over the specific type of
// output, which is the first argument in stu_params[] and ref_params[].
//
// Note because the args in bench_t::tfunc and bench_t::naiveFunc are not
// necessarily the same, students may need to re-write wrapper functions.
//
// <KERNEL_NAME>.h and <KERNEL_NAME>.cpp also define wrapper functions, which is
// used to unrolling args to call specific kernel (i.e. student's kernel or 
// naive one), and this wrapper function would become bench_t::tfunc or 
// bench_t::naiveFunc.
//===----------------------------------------------------------------------===//

typedef void (*lab_test_func)(void *ctx);

// Validator Function Pointer
//    Takes student's output, reference args (ref_args)
//    and the naive function pointer to generate truth.
// Note: Student and reference contexts may differ, but they must expose
// comparable output fields for validation.
typedef bool (*lab_check_func)(void *stu_ctx, void *ref_ctx,
                               lab_test_func naive_func);

// Benchmark Structure
// stu_input and naive_input are not necessarily the same,
// but the output type (student_dst and ref_dst) must be the same
typedef struct {
    std::string description;

    lab_test_func tfunc;      // Student Implementation
    lab_test_func naiveFunc;  // Naive Implementation
    lab_check_func checkFunc; // Validator Wrapper

    void *args;     // Student benchmark context
    void *ref_args; // Naive benchmark context

    std::chrono::nanoseconds baseline_time;
} bench_t;

#ifdef DEBUG
inline constexpr bool kDebugEnabled = true;
#else
inline constexpr bool kDebugEnabled = false;
#endif

template <typename... Args>
inline void debug_log(std::format_string<Args...> fmt, Args &&...args) {
    if constexpr (kDebugEnabled) {
        std::print(fmt, std::forward<Args>(args)...);
    }
}

template <typename Func>
static inline std::chrono::nanoseconds measure_time(Func &&func) {
    const auto start = std::chrono::steady_clock::now();
    func();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
}

inline double calculate_speedup(std::chrono::nanoseconds measured_time,
                                std::chrono::nanoseconds baseline_time) {
    const auto measured_count = measured_time.count();
    const auto baseline_count = baseline_time.count();
    if (measured_count <= 0 || baseline_count <= 0) {
        throw std::invalid_argument(
            "calculate_speedup: measured_time and baseline_time must be positive.");
    }

    return static_cast<double>(baseline_count) /
           static_cast<double>(measured_count);
}

inline double calculate_speedup(const bench_t &bench,
                                std::chrono::nanoseconds measured_time) {
    return calculate_speedup(measured_time, bench.baseline_time);
}

inline std::vector<double>
calculate_speedups(const std::vector<std::chrono::nanoseconds> &measured_times,
                   const std::vector<bench_t> &benchmarks) {
    if (measured_times.size() != benchmarks.size()) {
        throw std::invalid_argument(
            "calculate_speedups: measured_times and benchmarks size mismatch.");
    }

    std::vector<double> speedups;
    speedups.reserve(benchmarks.size());
    for (size_t i = 0; i < benchmarks.size(); ++i) {
        speedups.push_back(calculate_speedup(benchmarks[i], measured_times[i]));
    }
    return speedups;
}

inline double
calculate_geometric_mean_speedup(const std::vector<double> &speedups) {
    if (speedups.empty()) {
        throw std::invalid_argument(
            "calculate_geometric_mean_speedup: speedups must be non-empty.");
    }

    double log_sum = 0.0;
    for (double speedup : speedups) {
        if (speedup <= 0.0) {
            throw std::invalid_argument(
                "calculate_geometric_mean_speedup: speedups must be positive.");
        }
        log_sum += std::log(speedup);
    }

    return std::exp(log_sum / static_cast<double>(speedups.size()));
}

inline double calculate_geometric_mean_speedup(
    const std::vector<std::chrono::nanoseconds> &measured_times,
    const std::vector<bench_t> &benchmarks) {
    return calculate_geometric_mean_speedup(
        calculate_speedups(measured_times, benchmarks));
}

constexpr int CACHE_SIZE = 1 << 26;
static std::vector<int> cache_buf(CACHE_SIZE / sizeof(int));

static inline void flush_cache() {
    volatile int sink = 0;
    for (size_t i = 0; i < cache_buf.size(); ++i) {
        sink = sink + cache_buf[i];
        cache_buf[i] = sink;
    }
}

#endif // BENCH_H
