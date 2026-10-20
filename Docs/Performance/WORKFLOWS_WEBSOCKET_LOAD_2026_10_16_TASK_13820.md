# Workflow WebSocket concurrent-load benchmark — TASK-13820

Final measurement: 2026-10-18, reference dev `42176e5bd45e213e81da81d18091b8d6948a7de7`, candidate `136070c23ee14766099c8e6a6a1be19de211ff78`. The reference includes only the candidate's normal-disconnect return, retaining dev's original read/send behavior. Both variants therefore describe valid service runs without exception-formatting stalls.

## Outcome

The requested concurrent large-history regressions are repaired in this bounded, descriptive comparison. At 64 clients with 500 events per run, median trial replay P95 improves **856.82 → 802.14 ms (6.38%)**, and loop-lag P99 improves **6.34 → 2.65 ms (58.22%)**. Both opposite-order invocation pairs improve these measures. Maximum observed loop lag falls **458.43 → 33.56 ms**. At 16 clients, history replay and loop P99 also improve.

This is not a uniform latency improvement. With no history, handshake P95 increases **3.00 → 8.30 ms**, **48.93 → 61.86 ms**, and **183.51 → 205.66 ms** at 1/16/64 clients. Keeping disk reads off the loop carries worker-handoff costs; do not undo the nonblocking-read fix to remove those costs. Single-client history replay P95 changes **16.11 → 16.76 ms**. Live-event P95 remains approximately one second because polling is unchanged. No production SLO, capacity limit or statistical-significance claim is established.

All **72 trials** are valid, with zero server/client errors, zero mismatches and exact ordered delivery of **58,320 live events and 486,000 history frames**. All missing-credential, foreign-owner and missing-run controls reject the handshake with HTTP 403. Server test-mode flags are false; the existing threadpool has 40 tokens.

## Root causes and repair

Synchronous owner lookup and event reads blocked the ASGI loop. They now use Starlette's existing bounded threadpool. Deterministic regressions hold each real read pending while unrelated HTTP work completes. Authentication and ownership checks still precede history delivery.

Normal peer departure previously re-raised `WebSocketDisconnect` into Uvicorn's exception logger. Returning after the existing safe diagnostic avoids spurious ASGI errors and expensive traceback formatting. Regressions cover close codes 1000/1001/1006. The harness counts server error classes without retaining messages, credentials or traceback locals.

Offloading large SQLite pages exposed repeated SQLite/Python row-transfer and per-payload decode overhead. The SQLite fast path transfers one native JSON page. Bound filtering, ordering and the 500-row limit remain inside the query. Strict JSON checks retain original behavior for JSON5/non-finite values and BLOBs; missing JSON functions, oversized aggregates and aggregate decoder limits fall back to the original per-row path. Other database errors propagate. PostgreSQL, statement-cache safety and connection ownership remain unchanged.

Per-frame `sleep(0)` requeued senders ahead of timer polling and caused replay/P99 regressions. A loop-local shared 1 ms work budget now waits on a timer-backed Event. Senders recheck the deadline after waking, and cancellation remains local to each waiter. Weak ownership releases unused loop state. Event removes the extra callback generation introduced by the rejected shared-Future/shield prototype.

An untimed phase trace then found duplicate metric-label normalization on every frame. `MetricsRegistry.record` already validates and normalizes its labels before constructing a series key; its old key helper repeated that work. The repair constructs the key from those canonical labels. First-pass collision rejection, sensitive-label hashing, raw-label lookup/export normalization and legacy aliases remain intact. No cache, new dependency, global read limiter or schema shortcut was added. Rejected scheduling prototypes and diagnostic profiles do not form acceptance evidence.

## Method and reproduction

Activate the backend environment in a Git checkout, then run each frozen variant:

```bash
source .venv/bin/activate
python Helper_Scripts/benchmarks/workflows_websocket_load.py \
  --clients 1 16 64 --backlog 0 500 --rounds 30 --trials 3 \
  --output /tmp/workflows-ws-load.json
```

Each invocation copies tracked working source, creates fresh disposable databases/secrets and starts the production ASGI app over real loopback TCP. JWT, SQLite, routing and individual frames are real; there are no dependency/authentication overrides. Lifespan workers are off. Each client owns a separate run under one authenticated user. History has zero or 500 events; payloads include a highly compressible 512-byte repeated-character body. Compression defaults are preserved. After all clients drain history, a sequential producer appends through the real database adapter every 137 ms for 30 rounds. Polling remains one second.

Connection/snapshot/history durations start before connection; history includes setup. Live-event timing includes append/commit, polling and receipt. A 10 ms server-loop probe records one observation after each delayed sleep: a long stall is one extreme sample, not many missed observations. Report maximum lag alongside P99. Do not subtract snapshot and replay percentiles to infer a per-client phase duration; their completion-order arrays differ.

The sequence is **reference A / candidate A / candidate B / reference B**, three trials per case per invocation, six trials per variant. All completed trials are retained. The final reference was interrupted at startup by unrelated CPU-bound project tests; that aborted run is excluded before inspecting results. Independent review predeclared completing only reference B in the next quiet window with identical source/runtime/workload. The resumed gap was **48.86 minutes**. Each invocation uses fresh processes/data, so no shared process or database state crosses this gap. This is explicitly an interrupted/resumed ABBA, not a continuous-window experiment.

No competing CPU-bound project workers were observed in retained workload snapshots. Desktop activity was not fully idle: early reference B included a brief Sublime Text burst using one core, which then subsided; the host has 18 logical CPUs. End snapshots include filesystem activity from deleting temporary source copies. A later project test/Git burst appeared after reference B's report completed. These observations and the gap remain limitations, not grounds for a production-capacity claim. Cases run in fixed order as the disposable database grows. Comparisons match each case; they do not isolate client count from every other effect or identify the marginal contribution of each repair.

## Results

Tables show **medians of six trial percentiles**, not pooled-sample percentiles. Values are **reference / candidate**, in milliseconds. The maximum column is the maximum across all six trials.

| History/run | Clients | Handshake P95 | Snapshot P95 | Live event P95 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 3.00 / 8.30 | 3.04 / 8.34 | 982.86 / 974.23 |
| 0 | 16 | 48.93 / 61.86 | 49.04 / 61.93 | 971.97 / 997.95 |
| 0 | 64 | 183.51 / 205.66 | 183.79 / 205.69 | 959.46 / 957.77 |
| 500 | 1 | 4.07 / 3.76 | 4.12 / 3.78 | 976.77 / 972.50 |
| 500 | 16 | 68.16 / 55.16 | 204.01 / 55.20 | 962.65 / 961.62 |
| 500 | 64 | 663.05 / 222.87 | 845.52 / 224.93 | 952.24 / 948.90 |

| History/run | Clients | Full history P95 | Loop-lag P99 | Maximum observed loop lag |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | — / — | 1.60 / 2.12 | 10.89 / 3.27 |
| 0 | 16 | — / — | 1.75 / 2.13 | 13.44 / 13.01 |
| 0 | 64 | — / — | 6.61 / 3.04 | 251.16 / 223.29 |
| 500 | 1 | 16.11 / 16.76 | 2.62 / 2.13 | 13.74 / 3.10 |
| 500 | 16 | 215.16 / 201.32 | 2.69 / 2.14 | 158.12 / 10.71 |
| 500 | 64 | 856.82 / 802.14 | 6.34 / 2.65 | 458.43 / 33.56 |

No-history 64-client setup still has recurring maximum stalls around 220–251 ms across the variants. Retain those outliers; percentile improvements do not prove their removal. Raw reports retain every sample, trial percentile and maximum.

## Runtime, source identity and verification

Native environment: Python 3.12.13, SQLite 3.50.4, macOS 26.5.2 arm64, 18 logical CPUs; FastAPI 0.142.4, Starlette 1.8.0, Uvicorn 0.35.0 and websockets 15.0.1. Private native test infrastructure uses the CI framework versions and disables optional ONNX telemetry only to avoid an unrelated teardown crash. Production dependencies and telemetry configuration are unchanged.

| Source | Reference SHA-256 | Candidate SHA-256 |
| --- | --- | --- |
| `tldw_Server_API/app/api/v1/endpoints/workflows.py` | `cc4f94613fc371e3a5428a16da2390ac9b8e83c738f8b588cecb80efd60af347` | `1e69757f35bec502840227d36b967939183c1c84b174e50d600de24b87074bd6` |
| `tldw_Server_API/app/core/DB_Management/Workflows_DB.py` | `f3a8ec457b0fb9b9ffa95286f883c95289e2d683ae4d14c1fec219c78ac4ff89` | `fa238d3b877fa1bf658a80a4fc78fd4ff441f151c282b3cef16251fec3d83759` |
| `Helper_Scripts/benchmarks/workflows_websocket_load.py` | `6a05ffb80a5dab9333ac3129177747a312e64f1d6984a9db7e2e4008f09e6d76` | `6a05ffb80a5dab9333ac3129177747a312e64f1d6984a9db7e2e4008f09e6d76` |
| `tldw_Server_API/app/core/Metrics/metrics_manager.py` | `76e6a67cd96dab047336489d36daab2d6d5267f3552bd8f78ab4449955db8192` | `50d4d5ad102d4ac3280c6588a94adc78c2ad82bb2a9cf93b47478e6b563576b9` |

The unchanged harness records route/DB/helper hashes. Each final raw report additionally retains `supplemental_source_sha256` for the metrics file. External orchestration asserted all four hashes before and after each invocation; the resumption also verified all three preserved report hashes before appending reference B. The [manifest](workflows_websocket_load_13820_macos_manifest.json) retains expected source hashes, report hashes, timestamps, interruption metadata and bounded host snapshots; expected hashes alone are not independent proof of an observed source.

- **310 connected tests pass** on the measured candidate: workflow storage/runtime/API, streaming/authentication, real-TCP harness and all metrics contracts.
- Failing-first coverage includes pending reads, disconnects, shared scheduling/cancellation/loop cleanup, legacy page/decode/error boundaries and normalization-once behavior with preserved stored labels. Property-based page cases exercise recursive JSON.
- Scoped Ruff/Black and Bandit pass with zero findings/errors; test assertions exclude B101 only. No test or heavy diagnostic workload overlapped retained timing runs.
- Independent code/data review confirmed the requested large-history gate and no actionable production findings. Exact final documentation-head review and hosted checks remain required before merge.

ADR assessment: no new durable architecture rule; storage/auth/cache/framing/metrics policies remain intact. ADR-059 governs tracking. TASK-13820 links the plan and verification.

Raw acceptance reports:

- [baseline-a](workflows_websocket_load_13820_macos_baseline_a.json)
- [fixed-a](workflows_websocket_load_13820_macos_fixed_a.json)
- [fixed-b](workflows_websocket_load_13820_macos_fixed_b.json)
- [baseline-b](workflows_websocket_load_13820_macos_baseline_b.json)

The [original PR-head report](https://github.com/rmusser01/tldw_server/blob/5c4bab8e400119bac964d0b75dc94c2e931b0c75/Docs/Performance/WORKFLOWS_WEBSOCKET_LOAD_2026_10_16_TASK_13820.md) documents the rejected per-frame-yield version and invalid unmodified-dev disconnect-error baseline. Those historical numbers and failed Future/shield/Lock diagnostics are not the final acceptance comparison.

Integration rebase: PR #3256 was rebased onto dev `e1f7770953a78224e43f92976e526ae88b155d2b` after the measured runs. All six PR patches and all four measured source hashes remain unchanged. Upstream adds a lazily imported TTS startup worker; the harness disables lifespan, so that worker is outside the measured path. Independent review confirmed the existing timing evidence remains applicable to this scope. Enabled-worker performance remains unmeasured. Fresh rebased-head tests and hosted checks are separate integration gates.

A subsequent integration rebase onto dev `7c94a2efb1edc938070abcaa868d59e319a95112` also retains every measured source hash and all PR patches. Upstream changes sandbox WebSocket teardown and adds a test-quality CI gate; neither changes the measured workflow path. A CI-only chat test fix scopes synthetic exception state to the endpoint instead of Python’s shared `sys` module. No benchmark runtime change or additional performance claim is introduced.

Prior integration base: dev `29fda0d860c7a0d76dd3d9ea0561a32ac8e48148`. Independent review confirms all nine PR patches and the four measured source hashes remain unchanged. Upstream provider/tool usage, Claims, admin, watchlist and TTS changes do not execute in this lifespan-disabled workflow workload. The measurements remain historical evidence for the unchanged path; they are not a new timing run on this integration base.

Prior integration base: dev `b5b572c01dd6dfea231b0928d6c9d20f64b181dc`. All ten PR patches and four measured source hashes remain unchanged. Independent review confirms upstream resource-governance middleware bypasses WebSockets; audio/media concurrency leases, audio workers and frontend transport are outside this workload. The database parity addition is documentation only. Historical timings remain evidence for the unchanged measured path, without a new timing run or a capacity claim on this integration base.

Latest integration base: dev `a6719bf24f5f124f931c48a2250e103365514c48`. All fourteen PR patches and all four measured source hashes remain unchanged. Independent review confirms upstream workflow execution/retry handling and adapter quota errors are outside this database-row replay workload; chat, TTS and audio workers are also outside the lifespan-disabled measured path. Subsequent CI repairs affect tests and the pytest version bound only. These remain historical timings, without a new timing run or a capacity claim on this integration base.

## Limits and rollback

This synthetic loopback workload does not measure WAN/TLS, PostgreSQL, same-run fanout, many tenants, lifespan/background services, non-compressible payloads or a production soak. Native performance evidence is macOS-only; hosted Linux real-network tests cover behavior, not equivalent capacity. The shared threadpool may queue reads. Setup and single-client costs above remain disclosed. Revert the runtime changes if deployment evidence requires rollback, retaining the harness and regression evidence.
