# Backend Performance Remediation — Batch 0: Baseline Harness Implementation Plan (2026-10-06)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan stage-by-stage. Steps use checkbox (`- [ ]`) syntax for tracking.

**Backlog task:** TASK-13512 · **Index:** [2026-10-06-perf-remediation-coordination-index.md](2026-10-06-perf-remediation-coordination-index.md)

**Goal:** Build the measurement infrastructure (SQLite query-counting test helper + hot-path benchmark scripts) and record pre-fix baseline numbers, so Batches 1-7 can prove their improvements.

**Architecture:** A `tests/perf/` package inside `tldw_Server_API/tests/` providing (a) a trace-callback-based query counter usable as a pytest fixture or standalone context manager, and (b) small benchmark scripts that print one JSON line each. No production code changes in this batch.

**Tech Stack:** pytest, sqlite3 (`Connection.set_trace_callback`), stdlib `time.perf_counter`, `statistics`.

## Global constraints

- Backend only: never modify `apps/**`.
- Activate venv first: `source .venv/bin/activate` (repo root).
- One commit per stage; message references TASK-13512.
- TDD where a behavior is asserted (Stage 1); Stages 2-3 are measurement tooling validated by running them.
- Bandit on touched paths before completing: `python -m bandit -r tldw_Server_API/tests/perf -f json -o /tmp/bandit_perf_b0.json`.
- Benchmarks must be deterministic in *structure* (fixed seed fixtures), not in timing; timing numbers are informational.

---

## Stage 1: Query-counting test helper

**Goal:** A reusable helper that counts SQL statements executed on a SQLite connection, assertable in tests.

**Files:**
- Create: `tldw_Server_API/tests/perf/__init__.py`
- Create: `tldw_Server_API/tests/perf/query_counting.py`
- Test: `tldw_Server_API/tests/perf/test_query_counting.py`

**Interfaces (produced — Batches 2/3/4 depend on these exact names):**
- `count_statements(conn: sqlite3.Connection) -> context manager` yielding a `QueryCounter` with `.statements: list[str]`, `.count: int`, `.count_matching(prefix: str) -> int`.
- `count_connections() -> context manager` yielding a `ConnectionCounter` with `.count` — patches `sqlite3.connect` to count opens (use `unittest.mock.patch`).
- pytest fixture `query_counter` (in `query_counting.py`, registered via `tldw_Server_API/tests/conftest.py` plugin import if the suite supports it; otherwise documented import path).

**Change sketch:**

```python
@contextlib.contextmanager
def count_statements(conn):
    counter = QueryCounter()
    conn.set_trace_callback(counter._on_statement)
    try:
        yield counter
    finally:
        conn.set_trace_callback(None)
```

**Success criteria:** counting 3 executes reports 3; works against a real `ChaChaNotes_DB` SQLite instance.

**Tests:**
- [ ] `test_count_statements_counts_selects` — in-memory DB, 3 SELECTs → `counter.count == 3`.
- [ ] `test_count_statements_cleans_callback` — after context exit, callback unset.
- [ ] `test_count_connections_counts_opens` — 2 `sqlite3.connect(":memory:")` calls → `.count == 2`.
- [ ] `test_counter_filters_by_prefix` — `count_matching("SELECT")` ignores INSERTs.

**Status:** Not Started

---

## Stage 2: Hot-path benchmark scripts

**Goal:** Four standalone scripts under `tldw_Server_API/tests/perf/benchmarks/`, each printing a single JSON line `{"bench": ..., "metric": ..., "value": ..., "unit": ...}` and exiting 0. They must run without network, Docker, or GPU.

**Files:**
- Create: `tldw_Server_API/tests/perf/benchmarks/__init__.py`
- Create: `tldw_Server_API/tests/perf/benchmarks/bench_api_key_verify.py`
- Create: `tldw_Server_API/tests/perf/benchmarks/bench_reranker_instantiation.py`
- Create: `tldw_Server_API/tests/perf/benchmarks/bench_chat_history_assembly.py`
- Create: `tldw_Server_API/tests/perf/benchmarks/bench_jobs_acquire_tick.py`
- Create: `tldw_Server_API/tests/perf/benchmarks/_fixtures.py` (builders reused by later batches' post-fix runs)

**What each measures (pre-fix expectation):**

1. `bench_api_key_verify.py` — generate one `tldw_...` key via the AuthNZ manager helpers, store its KDF hash (210k iterations), then time 5 × `verify_kdf_hash` (successful + failing). Expect ~50-150ms/op successful. Metric: `ms_per_successful_verify`.
2. `bench_reranker_instantiation.py` — time `create_reranker(...)` twice (use `DiversityReranker`, which needs no model download; if flashrank weights are cached locally, add it as a second sample guarded by cache-dir existence). Metric: `ms_second_instantiation` (pre-fix ≈ `ms_first_instantiation`).
3. `bench_chat_history_assembly.py` — build a temp `ChaChaNotes_DB` with 1 conversation × 100 messages + metadata rows; run the *current* per-message loop shape (`get_messages_for_conversation` + `for id: get_message_metadata`) under `count_statements`; report `statement_count` and `ddl_statement_count` (statements starting with `CREATE TABLE`). Metric: `statements_per_100_messages` (pre-fix ≈ 200+ incl. 100 DDL).
4. `bench_jobs_acquire_tick.py` — init a `JobManager` on a temp SQLite DB (empty queue), wrap one `acquire_next_job(domain="chatbooks")` call in `count_connections` + `perf_counter`. Metric: `connections_per_empty_tick` (pre-fix ≈ 5-7) and `ms_per_empty_tick`.

**Success criteria:** all four run green via `python -m tldw_Server_API.tests.perf.benchmarks.bench_api_key_verify` (and the other three) from repo root, no network; support `PERF_BENCH_ITERS` env override for smoke tests.

**Tests:** `tldw_Server_API/tests/perf/test_benchmarks_smoke.py` — imports each module's `main()` and asserts it returns a dict (tiny iteration counts via `PERF_BENCH_ITERS=1`).

**Status:** Not Started

---

## Stage 3: Record baselines

**Goal:** A committed baseline document every later batch updates.

**Files:**
- Create: `Docs/Reviews/PERF_BASELINE_2026_10.md`

**Change:** Table per benchmark: `bench | metric | baseline (2026-10-06) | python/platform | re-measured (date, batch, value)`. Include machine context (python version, `platform.processor()`, macOS version). Run each bench 3×, record median; add a "known-noise" note.

**Success criteria:** file committed with all four benchmarks' baseline rows; the coordination index's "Machine-readable baseline" section still points here.

**Tests:** none (documentation).

**Status:** Not Started

---

## Batch verification

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/perf -v -m "not external_api and not local_llm_service"
python -m tldw_Server_API.tests.perf.benchmarks.bench_api_key_verify
python -m tldw_Server_API.tests.perf.benchmarks.bench_reranker_instantiation
python -m tldw_Server_API.tests.perf.benchmarks.bench_chat_history_assembly
python -m tldw_Server_API.tests.perf.benchmarks.bench_jobs_acquire_tick
python -m bandit -r tldw_Server_API/tests/perf -f json -o /tmp/bandit_perf_b0.json
```

Then update TASK-13512: implementation notes (what landed), verification results, final summary, DOD checkboxes.
