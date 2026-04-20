# CSC3060 Project 5 - Code Optimization

This repository is the starter code for the code optimization assignment. The main job for students is to replace the placeholder `stu_*` implementations in `src/kernel/*.cpp` with faster versions, keep the results correct, and use the benchmark harness to measure the speedup.

## Repository Layout

```text
CSC3060_Project5_Code_Optimization/
├── CMakeLists.txt          # Build rules for the kernels library and executables
├── README.md               # This guide
├── perfUsage.md            # How to use Linux perf for this project
├── include/                # Shared headers and benchmark interfaces
│   ├── bench.h             # Common benchmark harness and timing/check helpers
│   ├── blackscholes.h      # Benchmark-specific args, baseline, wrappers, checker
│   ├── bitwise.h
│   ├── filter_gradient.h
│   ├── graph.h
│   ├── grff.h
│   ├── image_proc.h
│   ├── matmul.h
│   ├── relu.h
│   ├── sparse_spmm.h
│   └── trace_replay.h
├── src/
│   ├── kernel/             # Naive reference code and student code live here
│   │   ├── blackscholes.cpp
│   │   ├── bitwise.cpp
│   │   ├── filter_gradient.cpp
│   │   ├── graph.cpp
│   │   ├── grff.cpp
│   │   ├── image_proc.cpp
│   │   ├── matmul.cpp
│   │   ├── relu.cpp
│   │   ├── sparse_spmm.cpp
│   │   └── trace_replay.cpp
│   ├── main/
│   │   ├── single_bench.cpp # Run a small benchmark set while tuning/debugging
│   │   └── run_all.cpp      # Run the full suite and report speedups
│   └── res/                 # Images used by perfUsage.md
├── .clang-format           # Optional formatting config
└── .clang-tidy             # Optional lint config
```

What each directory is for:

- `include/`: One header per benchmark. Each header usually defines the argument struct, baseline time, initialization helper, naive implementation declaration, student implementation declaration, wrapper functions, and correctness checker.
- `src/kernel/`: The actual benchmark implementations. This is the directory students will edit most often.
- `src/main/`: The programs that run the benchmarks.
- `src/res/`: Screenshot assets used by `perfUsage.md`. These are not benchmark inputs.

Notes:

- `single_bench.cpp` is the easier place to debug one kernel at a time.
- `run_all.cpp` registers the full benchmark suite and is the better entry point for `perf`.

## How The Benchmark Files Are Organized

Most benchmarks follow the same pattern:

- `include/<name>.h`: declares the benchmark argument struct, `BASELINE_*`, `initialize_*`, `naive_*`, `stu_*`, wrapper functions, and the checker.
- `src/kernel/<name>.cpp`: defines the naive implementation, the placeholder student implementation, wrappers, initialization logic, and result checking.

In other words, students normally edit `src/kernel/<name>.cpp`, and sometimes add fields to the matching `include/<name>.h` argument struct if they need extra precomputed data.

## Baseline Values And Default Input Sizes

The baseline times are defined in the benchmark headers and are used by the benchmark harness when it computes per-benchmark speedup. The table below is sorted by header file name in ascending order. For kernels not yet registered in `run_all.cpp`, the default input size column says so explicitly.

| Header file | Baseline constant | Baseline time (ns) | Default input size in `run_all.cpp` |
| --- | --- | --- | --- |
| `bitwise.h` | `BASELINE_BITWISE` | `250,000` | vector length `1,024,000` |
| `blackscholes.h` | `BASELINE_BLACKSCHOLES` | `4,800,000` | `81,l920` options |
| `filter_gradient.h` | `BASELINE_FILTER_GRADIENT` | `25,000,000` | height x width `1024 x 1024` |
| `graph.h` | `BASELINE_GRAPH` | `5,000,000` | `1,024,000` nodes, average degree `8` |
| `grff.h` | `BASELINE_GRFF` | `8,500,000` | feature size `1,024,000` |
| `image_proc.h` | `BASELINE_IMAGE_PROC` | `43,000,000` | image size `1024 x 1000` |
| `matmul.h` | `BASELINE_MATMUL` | `88,000,000` | matrix size `512 x 512` |
| `relu.h` | `BASELINE_RELU` | `550,000` | vector length `1,024,000` |
| `sparse_spmm.h` | `BASELINE_SPARSE_SPMM` | `116,000,000` | LHS (left hand side) CSR matrix of `2048 x 2048`, the default dense RHS (right hand side) is also `2048 x 2048` |
| `trace_replay.h` | `BASELINE_TRACE_REPLAY` | `3,400,000` | `65,536` records and trace length `1,048,576` |

## What `bench.h` Is For

`include/bench.h` is the shared benchmark harness used by both `single_bench` and `run_all`.

It provides:

- `bench_t`: one record describing a benchmark run.
- `tfunc`: the function wrapper that should call the student implementation.
- `naiveFunc`: the wrapper for the reference implementation.
- `checkFunc`: the validator used after timing.
- `args` and `ref_args`: the benchmark contexts passed to the student and reference wrappers.
- `baseline_time`: the baseline runtime in `std::chrono::nanoseconds`.
- `measure_time(...)`: helper for timing one benchmark run.
- `flush_cache()`: touches a large buffer before each measurement to reduce cache-warm effects between runs.
- `debug_log(...)`: prints debug messages only when `DEBUG` is enabled.
- `calculate_speedup(...)` and `calculate_geometric_mean_speedup(...)`: helpers used by `run_all.cpp`.

Important detail:

- `args` and `ref_args` do not have to point to the same object.
- If a kernel modifies its input in place, keep separate student and reference contexts. `relu` is the clearest example because the input vector is also the output.

For example, when you want to measure a student version, the benchmark entry should look like this idea:

```cpp
{"ReLU",
 stu_relu_wrapper,
 naive_relu_wrapper,
 relu_check,
 &relu_args_student,
 &relu_args_reference,
 BASELINE_RELU}
```

Right now, the starter `run_all.cpp` and `single_bench.cpp` mainly register `naive_*_wrapper` as placeholders. Students should switch `tfunc` to `stu_*_wrapper` when they want to benchmark their optimized code.

## Build

Run all commands from the repository root.

Build the project:

```bash
cmake -S . -B build
cmake --build build -j
```

Run the executables:

```bash
./build/single_bench
./build/run_all
```

Build only one target if you want:

```bash
cmake --build build --target single_bench -j
cmake --build build --target run_all -j
```

## Rebuild After You Change Files

If you only changed source or header files such as:

- `src/kernel/*.cpp`
- `src/main/*.cpp`
- `include/*.h`

then a normal rebuild is enough:

```bash
cmake --build build -j
```

If you changed build settings such as:

- `CMakeLists.txt`
- the compiler
- compile flags
- the build directory you want to use

then re-run CMake configure first:

```bash
cmake -S . -B build
cmake --build build -j
```

If you want a separate debug build, use a different build directory such as `build-debug` instead of overwriting `build`.

## Debugging Options

There are several useful ways to debug this project.

### 1. Enable project debug logs

`bench.h` defines `debug_log(...)`. It only prints when the `DEBUG` macro is enabled.

Use a separate debug build:

```bash
cmake -S . -B build-debug -DCMAKE_CXX_FLAGS="-DDEBUG"
cmake --build build-debug -j
./build-debug/single_bench
```

With `DEBUG` enabled, many checkers print useful mismatch information such as:

- first failing index
- reference value vs student value
- absolute or relative error
- per-run timing inside the benchmark loop

### 2. Use `single_bench` for focused debugging

`single_bench.cpp` is the fastest place to test one benchmark at a time. You can:

- register only the kernel you are currently working on
- reduce problem sizes temporarily
- confirm correctness before profiling the whole suite

### 3. Use symbols with `gdb` or `perf`

`CMakeLists.txt` already adds:

- `-g`
- `-fno-omit-frame-pointer`

This is useful for:

- source-level debugging
- better call stacks in `perf record -g`
- more useful output from `perf report` and `perf annotate`

## `GEOMETRIC_MEAN` In `run_all.cpp`

`src/main/run_all.cpp` defines:

```cpp
#define GEOMETRIC_MEAN 1
```

This macro controls whether the program prints one overall performance summary at the end.

- When `GEOMETRIC_MEAN` is `1`, `run_all` collects the runtime of every benchmark and prints the geometric mean speedup after all benchmarks finish.
- When `GEOMETRIC_MEAN` is `0`, that overall summary is disabled.
- If any benchmark fails correctness checking, the geometric mean is reported as `N/A`.

Why this matters:

- The geometric mean is a better overall score than averaging raw runtimes or averaging speedup numbers directly.
- It gives one single summary number for the whole benchmark suite.
- For the assignment, `run_all` is the place to look when you want the overall performance result across all kernels, not just one kernel.

## Profiling With `perf`

Please read [perfUsage.md](./perfUsage.md). It already contains a practical workflow and example screenshots for:

- `perf stat`
- `perf record`
- `perf report`
- `perf annotate`

A good workflow for this assignment is:

1. Run `run_all` under `perf` to find the hottest benchmark or function.
2. Optimize one kernel.
3. Rebuild.
4. Re-run `run_all`.
5. Use `single_bench` when you want focused measurements for one kernel.

Typical commands:

```bash
perf stat ./build/run_all
perf record -g ./build/run_all
perf report
perf annotate --stdio --source -l -s <function_name>
```

## Common Student Workflow

1. Pick one kernel in `src/kernel/`.
2. Implement or improve the matching `stu_*` function.
3. Make sure the benchmark entry uses `stu_*_wrapper` as `tfunc` and `naive_*_wrapper` as `naiveFunc`.
4. Keep separate `args` and `ref_args` if the kernel changes its inputs in place.
5. Rebuild with `cmake --build build -j`.
6. Check correctness with `single_bench`.
7. Measure end-to-end impact with `run_all` and `perf`.
