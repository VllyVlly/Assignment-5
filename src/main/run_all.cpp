#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <print>
#include <vector>

#include "bench.h"
#include "bitwise.h"
#include "blackscholes.h"
#include "filter_gradient.h"
#include "graph.h"
#include "grff.h"
#include "image_proc.h"
#include "matmul.h"
#include "relu.h"
#include "sparse_spmm.h"
#include "trace_replay.h"

#define GEOMETRIC_MEAN 1

int main() {
    std::cout << "Benchmark setup\n";

    std::uint32_t seed = 12345u;

    blackscholes_args black_args;
    initialize_blackscholes(black_args, 81920, seed);
    std::cout << "\tBlack-Scholes options: " << black_args.spot_price.size()
              << '\n';

    sparse_spmm_args sparse_args;
    initialize_spmm(sparse_args, 512, 512, -1, {}, seed);
    std::cout << "\tSparse A (CSR): " << sparse_args.csr.rows << " x "
              << sparse_args.csr.cols
              << ", nnz=" << sparse_args.csr.values.size() << '\n';

    constexpr size_t relu_size = 1024000;
    relu_args relu_args_naive;
    initialize_relu(&relu_args_naive, relu_size, seed);
    std::println("\tReLU: vector length={}", relu_size);

    constexpr size_t bitwise_size = 1024000;
    bitwise_args bitwise_args_naive;
    initialize_bitwise(&bitwise_args_naive, bitwise_size, seed);
    std::println("\tBitwise: vector length={}", bitwise_size);

    matmul_args matmul_args_naive;
    initialize_matmul(matmul_args_naive, 512, seed);
    std::cout << "\tMatMul: n=" << matmul_args_naive.n << '\n';

    trace_replay_args trace_args_naive;
    initialize_trace_replay(trace_args_naive, 1 << 16, 1 << 20, seed);
    std::cout << "\tTrace Replay: records=" << trace_args_naive.records.size()
              << ", trace_length=" << trace_args_naive.trace.size() << '\n';

    constexpr std::size_t graph_node_count = 1024000;
    constexpr int graph_avg_degree = 8;
    graph_args graph_args_naive;
    initialize_graph(&graph_args_naive, graph_node_count, graph_avg_degree, seed);
    std::cout << "\tGraph: node_count=" << graph_node_count
              << ", avg_degree=" << graph_avg_degree << '\n';

    constexpr std::size_t grff_size = 1024000;
    grff_args grff_args_naive;
    initialize_grff(&grff_args_naive, grff_size, seed);
    std::cout << "\tGRFF: feature size=" << grff_args_naive.a_features.size()
              << '\n';

    constexpr std::size_t image_width = 1024;
    constexpr std::size_t image_height = 1000;
    image_proc_args image_args_naive;
    initialize_image_proc(&image_args_naive, image_width, image_height, seed);
    std::cout << "\tImage Proc: " << image_args_naive.width << " x "
              << image_args_naive.height << '\n';

    const std::size_t WIDTH = 1024;
    const std::size_t HEIGHT = 1024;
    filter_gradient_args filter_gradient_args_ref;
    initialize_filter_gradient(&filter_gradient_args_ref,
                               WIDTH,
                               HEIGHT,
                               seed);
    std::cout << "\tFilter Gradient: " << HEIGHT << " x " << WIDTH << '\n';

    // STU

    blackscholes_args black_args_stu;
    initialize_blackscholes(black_args_stu, 81920, seed);

    sparse_spmm_args sparse_args_stu;
    initialize_spmm(sparse_args_stu, 512, 512, -1, {}, seed);

    relu_args relu_args_stu;
    initialize_relu(&relu_args_stu, relu_size, seed);

    bitwise_args bitwise_args_stu;
    initialize_bitwise(&bitwise_args_stu, bitwise_size, seed);

    matmul_args matmul_args_stu;
    initialize_matmul(matmul_args_stu, 512, seed);

    trace_replay_args trace_args_stu;
    initialize_trace_replay(trace_args_stu, 1 << 16, 1 << 20, seed);
    
    graph_args graph_args_stu;
    initialize_graph(&graph_args_stu, graph_node_count, graph_avg_degree, seed);

    grff_args grff_args_stu;
    initialize_grff(&grff_args_stu, grff_size, seed);

    image_proc_args image_args_stu;
    initialize_image_proc(&image_args_stu, image_width, image_height, seed);

    filter_gradient_args filter_gradient_args_stu;
    initialize_filter_gradient(&filter_gradient_args_stu,
                               WIDTH,
                               HEIGHT,
                               seed);

    // STU END

    std::vector<bench_t> benchmarks = {
        {"Black-Scholes (Naive)",
         stu_BlkSchls_wrapper,
         naive_BlkSchls_wrapper,
         BlkSchls_check,
         &black_args_stu,
         &black_args,
         BASELINE_BLACKSCHOLES},
        {"Sparse SpMM (Naive)",
         stu_sparse_spmm_wrapper,
         naive_sparse_spmm_wrapper,
         sparse_spmm_check,
         &sparse_args_stu,
         &sparse_args,
         BASELINE_SPARSE_SPMM},
        {"ReLU (Naive)",
         stu_relu_wrapper,
         naive_relu_wrapper,
         relu_check,
         &relu_args_stu,
         &relu_args_naive,
         BASELINE_RELU},
        {"Bitwise (Naive)",
         stu_bitwise_wrapper,
         naive_bitwise_wrapper,
         bitwise_check,
         &bitwise_args_stu,
         &bitwise_args_naive,
         BASELINE_BITWISE},
        {"MatMul (Naive)",
         stu_matmul_wrapper,
         naive_matmul_wrapper,
         matmul_check,
         &matmul_args_stu,
         &matmul_args_naive,
         BASELINE_MATMUL},
        {"Trace Replay (Naive)",
         stu_trace_replay_wrapper,
         naive_trace_replay_wrapper,
         trace_replay_check,
         &trace_args_stu,
         &trace_args_naive,
         BASELINE_TRACE_REPLAY},
        {"Graph (Naive)",
         stu_graph_wrapper,
         naive_graph_wrapper,
         graph_check,
         &graph_args_stu,
         &graph_args_naive,
         BASELINE_GRAPH},
        {"GRFF (Naive)",
         stu_grff_wrapper,
         naive_grff_wrapper,
         grff_check,
         &grff_args_stu,
         &grff_args_naive,
         BASELINE_GRFF},
        {"Image Proc (Naive)",
         stu_image_proc_wrapper,
         naive_image_proc_wrapper,
         image_proc_check,
         &image_args_stu,
         &image_args_naive,
         BASELINE_IMAGE_PROC},
        {"Filter Gradient (Naive)",
         stu_filter_gradient_wrapper,
         naive_filter_gradient_wrapper,
         filter_gradient_check,
         &filter_gradient_args_stu,
         &filter_gradient_args_ref,
         BASELINE_FILTER_GRADIENT}};

    std::cout << "\nRunning Benchmarks...\n";
    std::cout
        << "-----------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(25) << "Benchmark" << std::setw(12)
              << "Status" << std::right << std::setw(15) << "Nanoseconds"
              << std::setw(12) << "Speedup" << '\n';
    std::cout
        << "-----------------------------------------------------------------------\n";

#if GEOMETRIC_MEAN
    std::vector<std::chrono::nanoseconds> measured_times;
    measured_times.reserve(benchmarks.size());
    bool gm_enable = true;
#endif

    constexpr int k_best = 20;
    for (const auto &bench : benchmarks) {
        std::chrono::nanoseconds avg_time{0};

        for (int i = 0; i < k_best; ++i) {
            flush_cache();
            const auto elapsed = measure_time([&] { bench.tfunc(bench.args); });
            avg_time += elapsed;
            debug_log("\tDEBUG: {}-th measurement: {} ns\n",
                      i,
                      static_cast<std::uint64_t>(elapsed.count()));
        }
        avg_time /= static_cast<uint64_t>(k_best);

        const bool correct =
            bench.checkFunc(bench.args, bench.ref_args, bench.naiveFunc);

        std::cout << std::left << std::setw(25) << bench.description;
        if (!correct) {
            std::cout << "\033[1;31mFAILED\033[0m" << std::right
                      << std::setw(15) << "N/A" << std::setw(12) << "N/A"
                      << '\n';
            std::cout
                << "  Error: Results do not match naive implementation!\n";
#if GEOMETRIC_MEAN
            gm_enable = false;
#endif
            continue;
        }

        const double speedup = calculate_speedup(bench, avg_time);
        std::cout << "\033[1;32mPASSED\033[0m" << std::right << std::setw(15)
                  << avg_time.count() << " ns" << std::setw(11) << std::fixed
                  << std::setprecision(3) << speedup << "x";
        if (avg_time.count() > bench.baseline_time.count() * 1.1) {
            std::cout << " (SLOWER)";
        }
        std::cout << '\n';
        std::cout.unsetf(std::ios::floatfield);
        std::cout << std::setprecision(6);

#if GEOMETRIC_MEAN
        measured_times.push_back(avg_time);
#endif
    }

#if GEOMETRIC_MEAN
    if (gm_enable && measured_times.size() == benchmarks.size()) {
        const double geometric_mean_speedup =
            calculate_geometric_mean_speedup(measured_times, benchmarks);
        std::println("\nGeometric mean speedup: {:.3f}x",
                     geometric_mean_speedup);
    } else {
        std::println("\nGeometric mean speedup: N/A");
    }
#endif

    return 0;
}
