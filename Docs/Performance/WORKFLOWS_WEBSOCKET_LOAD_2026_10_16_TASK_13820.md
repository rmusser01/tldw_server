# Workflow WebSocket concurrent-load benchmark — TASK-13820

Baseline: latest remote dev `0566130cbed2670efca6bbfa3c309031d2d3bc3b` on 2026-10-16. The fixed source offloads the workflow WebSocket owner lookup and event poll to Starlette's bounded threadpool, finishes normally when the peer disconnects, and cooperatively yields between history events.

## Root cause and behavioral verification

Both synchronous database reads ran directly on the ASGI event loop. A deterministic regression holds each read pending while an unrelated HTTP request tries to complete on the same server loop. Both cases fail before the repair and pass after it. The real database call still runs after release; ownership, exception diagnostics, event ordering, frames and the one-second polling interval are preserved.

The final connected workflow storage, runtime, API, real-TCP harness and WebSocket cookie/JWT authentication suite passed 183 tests, including all three root-cause regressions and all close-code cases. Scoped Ruff, Black and Bandit checks passed with no security findings. Native tests use a private source copy and the CI FastAPI/Starlette versions; optional ONNX telemetry is disabled only in the private runner to avoid an unrelated native teardown crash. No production dependency or telemetry configuration changes.

A separate untimed watchdog sampled the main server thread during remaining stalls. It repeatedly found Uvicorn's ASGI exception handler formatting Loguru traceback locals after a routine peer disconnect. The endpoint logged a safe INFO diagnostic but then re-raised `WebSocketDisconnect`, turning ordinary departure into an unhandled server exception. Returning after the diagnostic fixes that lifecycle error. Three failing-first endpoint regressions cover close codes 1000, 1001 and 1006. This also avoids sending the exception into a traceback logger that may inspect sensitive request locals.

The final harness counts Uvicorn ERROR records by exception class without retaining exception messages or local values; a non-suppressing standard logging filter preserves the production handlers and formatters. A report with server errors is invalid even if clients received every event. The first exploratory ABBA set preceded that accounting and is excluded from final validated comparisons.

A second untimed watchdog sampled the two-repair version during large-history stalls and found the main thread in the workflow send loop, through Uvicorn framing and per-message compression. An awaited send can complete immediately on a writable transport, allowing a history burst to monopolize the loop. A failing-first regression uses real database events, JWT verification and Starlette serialization with an immediate ASGI transport: a callback scheduled at the first event does not run until all 500 events are sent. Yielding with `asyncio.sleep(0)` after each successfully sent event lets unrelated work run while preserving every ordered frame, cursor update, page size and polling interval. All six root-cause regression cases pass together.

## Method and reproduction

Activate the backend environment in a Git checkout, then run:

```bash
python Helper_Scripts/benchmarks/workflows_websocket_load.py \
  --clients 1 16 64 --backlog 0 500 --rounds 30 --trials 3 \
  --output /tmp/workflows-ws-load.json
```

Each invocation copies tracked working source into private temporary storage, creates fresh databases and secrets, and starts the production ASGI app over real loopback TCP. There are no dependency or authentication overrides. Lifespan workers are off; observed server test-mode flags must both be false for a valid report. Authentication controls verify HTTP 403 handshake refusals for missing credentials, a different owner, and a missing run.

Each subscriber has its own run, all owned by one authenticated user. History contains zero or 500 events per run. After every subscriber receives its snapshot and drains history, one independent sequential producer appends a live event to each run through the real database adapter, then waits 137 ms. Thirty rounds span several one-second polling periods. Payloads contain a synthetic, highly compressible 512-byte repeated-character body plus run/index/timing fields; Uvicorn defaults, including compression, are preserved. The harness validates every history/live sequence and run identity, counts errors by class, and exits nonzero for invalid delivery or server shutdown.

Handshake and snapshot latency start immediately before client connection. History drain includes connection and snapshot time. Live-event latency starts immediately before the SQLite append and includes commit, server polling, framing and client receipt. A 10 ms probe measures lag on the server's own event loop; each trial includes connection setup, history drain and live delivery. Trial metrics retain all samples, median, nearest-rank P95/P99 and maximum. Failed/incomplete trials cannot silently become successful samples.

Runs are sequential on the same host, with no competing test or benchmark jobs, in baseline-A / fixed-A / fixed-B / baseline-B order. The leading baseline is the preceding error-aware baseline run; an untimed diagnostic and regression checks occurred before the final fixed runs. Source, harness, runtime and workload settings are matched throughout. Each run has three trials per case, giving six trials per variant and opposite between-variant order. This reduces run-order bias; it does not establish statistical significance. The database grows across trials in fixed case order. Comparisons match each case between variants; these rows do not isolate client count from all other effects. The timing comparison combines all three repairs; their separate performance contributions are not measured. The earlier five-round pilot is excluded from final comparisons.

## Results

Both baseline runs are **invalid service runs**: each logs 486 unhandled `WebSocketDisconnect` exceptions even though every client receives its events. Their timings describe the broken behavior and must not be used as valid service-capacity results. Both repaired runs are valid, with zero Uvicorn errors, zero client errors/mismatches, and exact ordered delivery. Across the four runs, 58,320 live events and 486,000 history frames were validated. All missing-credential, foreign-owner and missing-run controls refused the handshake with HTTP 403.

The repair eliminates 972 spurious ASGI exceptions and protects unrelated loop work during disk reads and history bursts. At 64 clients with no history, median trial P95 handshake latency falls from 1766.50 to 232.29 ms. Live-event P95 remains roughly 0.92–1.00 seconds because polling is unchanged.

**Material tradeoff:** at 64 clients with 500 history events, median trial P95 replay grows from 942.34 to 3170.31 ms. Maximum observed loop lag falls from 558.72 to 135.35 ms, but median trial loop-lag P99 rises from 3.56 to 72.24 ms. Setup timings also vary substantially. This is a lifecycle and scheduling-fairness repair with workload-dependent latency costs, not a general latency improvement. No replay SLO or production capacity target is established here.

Tables show medians of six trial percentiles, not a percentile of pooled samples. `Before / repaired` values are milliseconds. Maximum loop lag is the maximum across all six trials; ranges below are ranges of trial P95s.

| History/run | Clients | Handshake P95 before / repaired | Snapshot P95 before / repaired | Live event P95 before / repaired |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 3.14 / 3.96 | 3.25 / 4.02 | 987.97 / 996.05 |
| 0 | 16 | 340.22 / 56.10 | 340.46 / 56.25 | 924.48 / 956.65 |
| 0 | 64 | 1766.50 / 232.29 | 1767.63 / 232.51 | 963.71 / 967.24 |
| 500 | 1 | 4.00 / 3.10 | 4.05 / 3.12 | 995.81 / 997.38 |
| 500 | 16 | 108.68 / 72.43 | 241.94 / 72.92 | 957.84 / 919.56 |
| 500 | 64 | 715.70 / 1073.16 | 930.60 / 1094.51 | 950.54 / 968.40 |

| History/run | Clients | Loop-lag P99 before / repaired | Maximum loop lag before / repaired |
| ---: | ---: | ---: | ---: |
| 0 | 1 | 1.13 / 1.13 | 2.09 / 7.03 |
| 0 | 16 | 1.36 / 2.17 | 446.28 / 23.64 |
| 0 | 64 | 10.62 / 5.61 | 1584.92 / 295.99 |
| 500 | 1 | 1.14 / 1.15 | 1328.91 / 9.35 |
| 500 | 16 | 1.75 / 2.27 | 177.95 / 16.36 |
| 500 | 64 | 3.56 / 72.24 | 558.72 / 135.35 |

| History/run | Clients | Full history P95 before / repaired | Before trial-P95 range | Repaired trial-P95 range |
| ---: | ---: | ---: | ---: | ---: |
| 500 | 1 | 16.66 / 31.40 | 14.40–1799.95 | 26.83–38.36 |
| 500 | 16 | 255.27 / 758.06 | 216.55–267.89 | 458.22–919.82 |
| 500 | 64 | 942.34 / 3170.31 | 855.61–1012.77 | 2451.20–3480.33 |

The six-trial 64-client/500-event handshake P95 ranges are 641.14–801.49 before and 399.17–1652.57 ms repaired. Do not infer a stable setup speedup in this case.

## Environment and source identity

Native macOS: Python 3.12.13, SQLite 3.50.4, `macOS-26.5.2-arm64-arm-64bit`, 18 reported CPUs; FastAPI 0.142.4, Starlette 1.8.0, Uvicorn 0.35.0, WebSockets 15.0.1. Both observed test-mode flags are false; the shared threadpool has 40 tokens. There is no Linux timing comparison.

- Repaired `tldw_Server_API/app/api/v1/endpoints/workflows.py` SHA-256: `29b6a3c8cb91e61f0bb807898eae3c1cb02688a5ff8f46319ec52cb0494eafce`.
- Repaired `tldw_Server_API/app/core/DB_Management/Workflows_DB.py` SHA-256: `f3a8ec457b0fb9b9ffa95286f883c95289e2d683ae4d14c1fec219c78ac4ff89`.
- Repaired `Helper_Scripts/benchmarks/workflows_websocket_load.py` SHA-256: `6a05ffb80a5dab9333ac3129177747a312e64f1d6984a9db7e2e4008f09e6d76`.
- Baseline route SHA-256: `deb0c44e05136567192ccbbd7f9960fa22a7e89df518bc495f25a3587840d41f`.

Raw reports: [baseline A](workflows_websocket_load_13820_macos_baseline_a.json), [repaired A](workflows_websocket_load_13820_macos_fixed_a.json), [repaired B](workflows_websocket_load_13820_macos_fixed_b.json), [baseline B](workflows_websocket_load_13820_macos_baseline_b.json). Baseline CLI exit 1 is intentional evidence of server errors; repaired CLI exits are 0.


## Limits

This is a macOS loopback test of unique runs with one owner and one sequential producer. It excludes TLS/WAN latency, network-bandwidth saturation, cookie authentication timing, same-run fanout, multiple tenants, PostgreSQL, workflow execution/LLM time, enabled background services and long-duration production soak. Linux real-network smoke coverage belongs to hosted CI; no Linux capacity claim is made. Windows subprocess cleanup is not verified. The shared threadpool has 40 tokens in this environment and can queue readers; offloading does not remove up to one polling interval of detection delay. Rebenchmark on deployment hardware before capacity planning or defining an SLO.

ADR assessment: no new ADR is required. This measures the existing protocol and repairs blocking I/O without changing storage, authentication, cache safety or event framing policy. ADR-059 governs task tracking.
