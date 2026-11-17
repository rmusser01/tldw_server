# Workflow SQLite statement-cache benchmark — TASK-13819

Measured 2026-10-16 against dev `696d7fdeced92f23b72ae416aeddc6ea9fd5f945` plus this task's pool-return fix. Both cache controls use the same pool fix. Production caching remains disabled as established by PR #3252.

## Decision and observed cost

Keep `cached_statements=0`. Every uncached trial returned correct identities/events and retained its pooled connections on both platforms (476,000 operations per platform). Cached concurrent controls returned wrong run/owner identities and raised row/SQLite errors, making their performance comparisons invalid.

The valid serial cases show a measurable preparation cost: approximately 2–5 µs per adapter operation on macOS and 6–15 µs in the Linux container. Paired median latency increases range from 6.5–38.8% and 49.2–64.2%, respectively. These are database microbenchmarks, not end-to-end WebSocket latency measurements.

The benchmark exposed a separate pool bug: returning a borrowed connection was conditional on the pool still being nonempty. Exhaustion therefore discarded borrowed connections. The minimal repair returns every non-primary connection, including when the pool is empty. A deterministic regression borrows all three available connections and verifies that all three return; the old condition loses all three. Every measured trial with the repair retained the initial pool size. This report does not claim a measured before/after throughput gain for that repair.

## Method and reproduction

Activate the project environment, then run:

```bash
python Helper_Scripts/benchmarks/workflows_sqlite_cache.py \
  --iterations 1000 --trials 7 --threads 16 --output /tmp/workflows-cache.json
```

The standard-library harness uses the real `WorkflowsDatabase` adapter in disposable temporary databases: 32 runs with distinct owners and one 256-byte event each. `run` means `get_run`; `poll` means `get_run` followed by `get_events(since=0)`. An operation is one workload iteration, not one SQL statement. Each trial performs 100 warmup iterations, then 1,000 iterations per worker. Cases use 1 or 16 workers and configured pool sizes 0 or 4. Size 4 means the primary connection plus three borrowed connections.

Seven paired trials alternate cache sizes 128 (unsafe historical control) and 0. Only construction-time `sqlite3.connect` settings are patched; workers run after the patch is removed. Parallel workers start together; there is no barrier per read. Each returned run ID, owner, event run ID, and event owner is checked. Errors are counted by class. Cache controls are isolated to the benchmark; there is no runtime server switch.

Latency covers adapter calls, excluding validation. Throughput includes validation and the parallel start barrier; attempted and valid operations are both recorded. Tables show the median of trial medians, median of trial P95s, and median trial throughput. The percentage column is the median of seven paired latency ratios; it need not equal the ratio of the aggregated latency columns. A comparison is suppressed unless every trial is correct and retains its pool. The CLI exits nonzero for any invalid uncached trial.

## Environments and source identity

Runs were sequential on the same 18-CPU arm64 Mac, without concurrent test/benchmark jobs. Linux ran in an existing local Docker backend image with networking disabled, disposable source copy, and read-only input mount. It is a Linux container on the same host, not a separate production Linux machine.

- macOS: Python 3.12.13, SQLite 3.50.4, `macOS-26.5.2-arm64-arm-64bit`, reported CPUs 18, SQLite thread safety 3.
- Linux container: Python 3.12.14, SQLite 3.46.1, `Linux-7.0.14-linuxkit-aarch64-with-glibc2.41`, reported CPUs 18, SQLite thread safety 3.

Both raw reports record identical SHA-256 source fingerprints:

- Benchmark: `84a21262b0a5330f86260e3657567a28dbe4759377b4f94e1d92164f416df927`
- Workflow DB: `f3a8ec457b0fb9b9ffa95286f883c95289e2d683ae4d14c1fec219c78ac4ff89`

Raw trial reports: [macOS](workflows_sqlite_cache_13819_macos.json), [Linux container](workflows_sqlite_cache_13819_linux.json).

## Valid serial comparisons

| Platform | Pool | Workload | Cached median / P95 µs | Uncached median / P95 µs | Paired latency change | Cached / uncached ops/s |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| macOS | 0 | run | 7.83 / 10.29 | 9.71 / 12.92 | +25.4% | 107,441 / 87,836 |
| macOS | 0 | poll | 18.12 / 23.46 | 19.83 / 26.46 | +6.5% | 48,477 / 44,153 |
| macOS | 4 | run | 9.58 / 13.21 | 13.29 / 22.67 | +38.8% | 87,782 / 63,158 |
| macOS | 4 | poll | 21.38 / 41.25 | 26.08 / 46.79 | +22.0% | 39,368 / 32,789 |
| Linux container | 0 | run | 15.79 / 20.79 | 23.52 / 36.92 | +49.2% | 57,519 / 36,708 |
| Linux container | 0 | poll | 22.75 / 45.42 | 37.42 / 81.21 | +62.7% | 35,811 / 22,005 |
| Linux container | 4 | run | 10.38 / 10.88 | 16.33 / 17.58 | +57.5% | 92,530 / 58,137 |
| Linux container | 4 | poll | 15.00 / 16.42 | 24.67 / 29.25 | +64.2% | 62,928 / 38,158 |

## Concurrent correctness and uncached capacity

Each row aggregates seven trials of 16 workers (112,000 operations per cache setting). All cached concurrent comparisons are invalid; their timings in the raw reports must not be interpreted as a valid speed advantage. Uncached rows below have zero mismatches/errors and preserve the pool.

| Platform | Pool | Workload | Cached mismatches | Cached errors | Uncached median / P95 µs | Uncached ops/s |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| macOS | 0 | run | 9,782 | 1 | 2671.00 / 5538.25 | 6,113 |
| macOS | 0 | poll | 1,972 | 166 | 36.54 / 14207.92 | 6,825 |
| macOS | 4 | run | 623 | 0 | 15.67 / 5308.54 | 19,323 |
| macOS | 4 | poll | 2,899 | 457 | 32.54 / 12940.54 | 7,837 |
| Linux container | 0 | run | 10,743 | 7 | 6745.19 / 12923.71 | 2,495 |
| Linux container | 0 | poll | 16,320 | 3,644 | 11550.37 / 18064.67 | 1,419 |
| Linux container | 4 | run | 7,350 | 1 | 4232.83 / 13699.79 | 3,106 |
| Linux container | 4 | poll | 2,831 | 881 | 85.92 / 19120.83 | 3,976 |

## Limits and validation

This read-only synthetic workload excludes concurrent writers, network/HTTP/WebSocket framing, JWT/cookie verification, LLM calls, long event histories, memory profiling, and production traffic. Runtime/SQLite versions differ across platforms, so this is not a controlled OS comparison. Host scheduling and container overhead affect timings; seven trials establish repeatability here, not statistical significance or a service SLO. Rebenchmark on deployment hardware before capacity planning. No policy relaxation or alternative caching architecture is justified by these results.

Regression tests cover CLI reports and invalid arguments, corrupt-row error accounting, pool exhaustion/return, and WebSocket diagnostic code/reason/severity and secret-safe logging. The connected workflow/authentication suite passed 172 tests. Optional native ONNX telemetry crashed one earlier pytest teardown after assertions completed; a private harness disabled that unrelated telemetry and the focused and connected runs exited cleanly. Production dependencies and telemetry behavior were not changed. Hosted PR checks remain the integration gate.

ADR assessment: no new ADR is required; this measures an existing correctness safeguard, repairs connection return, and adds bounded diagnostic observability without changing storage or authorization policy. ADR-059 governs task tracking.
