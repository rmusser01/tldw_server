## Stage 1: Distributed Lock Safety
**Goal**: Make migration/config locks safe under long-running work and multi-host Redis outages.
**Success Criteria**: Redis-backed migration locks do not silently downgrade when Redis is explicitly configured; file locks are not broken while the owning process is alive; Redis lock TTL can be renewed during long operations.
**Tests**: `test_distributed_lock.py` coverage for fail-closed Redis fallback, live PID stale handling, and Redis renewal.
**Status**: Complete

## Stage 2: Redis Factory Hardening
**Goal**: Preserve optional Redis behavior while avoiding secret leakage.
**Success Criteria**: Missing Redis package falls back to in-memory clients when allowed; disabled fallback still raises; logs redact credentials in Redis URLs.
**Tests**: `test_redis_factory.py` and metrics tests for missing dependency fallback and redacted warning output.
**Status**: Complete

## Stage 3: Provider Registry and Pool Metrics Robustness
**Goal**: Remove duplicate sync adapter materialization races and keep pool metric collection best-effort.
**Success Criteria**: Concurrent sync callers materialize a provider once; pool metric accessor failures return unavailable metrics without raising.
**Tests**: provider registry concurrency test and pool metrics failure test.
**Status**: Complete

## Stage 4: Circuit Breaker Persistence Semantics and Cleanup
**Goal**: Prevent unsupported rolling-window persistence from giving misleading cross-process behavior and remove unused state.
**Success Criteria**: Persistent store attachment rejects rolling-window configs with a clear error; in-memory rolling-window behavior is unchanged; unused async lock field removed.
**Tests**: circuit breaker persistence/window rejection tests and existing rolling-window tests.
**Status**: Complete

## Stage 5: Verification and Task Finalization
**Goal**: Verify targeted behavior and security checks, then update Backlog task status.
**Success Criteria**: Targeted pytest suite passes; Bandit reports no new findings for touched Infrastructure files; Backlog task includes touched files and verification notes.
**Tests**: `python -m pytest` targeted Infrastructure tests and `python -m bandit -r` on touched files.
**Status**: Complete
