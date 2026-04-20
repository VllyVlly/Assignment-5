# A Practical Guide to Using `perf` on Linux

`perf` is a Linux performance analysis tool used to measure and profile the behavior of programs and systems. It can collect both high-level statistics and detailed sampling data, which makes it useful for performance tuning, debugging, and computer architecture analysis.

## What `perf` Can Help You Study

- CPU cycles
- Instructions retired
- Cache behavior
- Branch prediction behavior
- Function-level hotspots
- System-wide performance issues

## Environment on Our Server

On our server, `perf` has already been installed and can be used directly.

```bash
perf --version
```

> Note:
> Restricted access to kernel symbols or kernel profiling scope is normal on this server.
> For our benchmark work, this is not a problem. We only need to analyze the user-space benchmark code, so limited kernel-space visibility does not affect the main results.

## Basic Syntax

```bash
perf <subcommand> [options] [command]
```

## Common Subcommands

| Subcommand | Purpose |
| --- | --- |
| `perf list` | Show available events |
| `perf stat` | Collect summary statistics |
| `perf record` | Record sampling data |
| `perf report` | Explore recorded samples |
| `perf top` | Show live hotspots |
| `perf annotate` | Inspect hot source lines or assembly |

## Listing Available Events

To see all events supported by your system:

```bash
perf list
```

This usually includes:

- Hardware events
- Software events
- Cache events
- Tracepoints
- Architecture-specific PMU events

Example:

```bash
perf list | less
```

## Using `perf stat`

`perf stat` provides summary statistics for a command.

Example:

```bash
perf stat ls
```

Common metrics include:

- `cycles`
- `instructions`
- `cache-references`
- `cache-misses`
- `branches`
- `branch-misses`
- `task-clock`
- `elapsed time`

Useful patterns:

```bash
# Measure a program
perf stat ./my_program [args...]

# Measure selected events
perf stat -e cycles,instructions,cache-misses,branch-misses ./my_program [args...]

# Collect system-wide statistics for 5 seconds
perf stat -a sleep 5
```

If your benchmark takes command-line parameters, place them after `./my_program`. For example, `./my_program <input_size> <num_threads>`.

## Using `perf record`

`perf record` collects sampling data while a program runs.

Example:

```bash
perf record ./my_program [args...]
```

This creates a file named `perf.data`.

More useful variants:

```bash
# Record call graphs
perf record -g ./my_program [args...]

# Record a specific event
perf record -e cycles -g ./my_program [args...]
```

Call graphs are especially useful for understanding which call paths consume the most time.

## Using `perf report`

After recording data, use:

```bash
perf report
```

This opens an interactive report that shows:

- Hot functions
- Symbol names
- Shared libraries
- Sample percentages
- Call relationships

Typical workflow:

```bash
perf record -g ./my_program [args...]
perf report
```

## Profiling an Existing Process

If a process is already running, you can attach `perf` to it.

```bash
# Collect statistics from a running process
perf stat -p <pid> sleep 5

# Record samples from a running process
perf record -p <pid> -g sleep 10
```

Then view the results:

```bash
perf report
```

## Important Events

Some frequently used events are:

| Event | Meaning |
| --- | --- |
| `cycles` | Total CPU cycles |
| `instructions` | Total retired instructions |
| `cache-references` | Cache accesses |
| `cache-misses` | Cache misses |
| `branches` | Branch instructions |
| `branch-misses` | Branch mispredictions |
| `context-switches` | Context switch count |
| `page-faults` | Page fault count |
| `task-clock` | CPU time consumed |

Example:

```bash
perf stat -e cycles,instructions,branches,branch-misses ./my_program [args...]
```


## Cache Analysis

To study cache behavior:

```bash
perf stat -e cache-references,cache-misses ./my_program [args...]
```

A high cache miss rate may indicate:

- Poor locality
- Inefficient data layout
- Large working sets
- Memory-bound behavior

For more detailed analysis, use architecture-specific cache events from `perf list`.

## Branch Prediction Analysis

To analyze branch behavior:

```bash
perf stat -e branches,branch-misses ./my_program [args...]
```

A high branch miss rate may suggest:

- Unpredictable control flow
- Many conditional branches
- Poor branch locality

This is especially useful in performance-sensitive and architecture-focused work.

## Source and Assembly Annotation

To inspect which instructions or source lines are hot:

```bash
perf annotate
```

This can help you understand:

- Which instructions consume the most samples
- Whether the compiler generated efficient code
- Where bottlenecks appear in the instruction stream

You can also annotate a specific function:

```bash
perf annotate <function_name>
```

## Typical Workflow

### 1. Get overall statistics

```bash
perf stat ./my_program [args...]
```

The result would be like:

![overallStat](./src/res/overallPerfStat.png)

![overallStatRes](./src/res/overallPerfRes.png)

### 2. Measure key architecture-related metrics

```bash
perf stat -e cycles,instructions,branch-misses ./my_program [args...]
```

![Example `perf stat` output](./src/res/perfStat.png)

![`perf stat` Result](./src/res/perfStatRes.png)

### 3. Record detailed samples

```bash
perf record -g -F 999 ./my_program [args...]
```

If `perf` reports limited access to kernel symbols or kernel profiling data here, that is expected on our server. We can continue and focus on user-space samples from the benchmark itself.

![Permission note during `perf record`](./src/res/image-2.png)

![Completed `perf record` run](./src/res/image-3.png)

### 4. View hotspots

```bash
perf report
```

![Permission note during `perf report`](./src/res/perfReportPermission.png)
![Interactive `perf report` view](./src/res/perfReportView.png)


### 5. Drill down into a specific function

We can use `perf annotate` to see the details of a certain function we are interested as long as we provide its symbol. But due to the possibility of function overload, its symbol could be so long. We can use the following command to find the complete symbol name firstly:

```bash
nm -C ./single_bench | grep <funcName(>
```

![nm_example](./src/res/nmSymbol.png)


Here the complete name is `csr_spmm(CSRMatrix const&, std::vector<float, std::allocator<float> > const&, std::vector<float, std::allocator<float> >&)`, then we have:


```bash
perf annotate --stdio --source -l -s "csr_spmm(CSRMatrix const&, std::vector<float, std::allocator<float> > const&, std::vector<float, std::allocator<float> >&)"
```

And it shows the location of hottest source code and corresponding assembly code:

![Example `perf annotate` output](./src/res/perfAnnotate.png)

![AsmSrc](./src/res/AsmSrc.png)

## Useful Tips

- Compile with `-g -fno-omit-frame-pointer` for better symbols
- Use realistic workloads
- Run tests multiple times to reduce noise
- Minimize background activity
- Use architecture-specific events when needed
