---
id: TASK-13400
title: Four tests fail under pytest -n 8 from cross-test pollution
status: Done
assignee: []
created_date: '2026-09-30 07:02'
updated_date: '2026-10-02 05:24'
labels:
  - tests
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Running RUN_EVALUATIONS=1 RUN_JOBS=1 pytest -n 8 over tests/Resource_Governance, AuthNZ_Unit, Embeddings, lint and Utils deterministically fails these four tests. They pass when run alone, and the failure reproduces on origin/dev 955b1d9626:
- Resource_Governance/test_e2e_tokens_daily_cap.py::test_e2e_embeddings_tokens_daily_cap_denies[rg-memory] and [rg-redis] (assert [] == [False] on cache_scope_sensitive_values)
- Embeddings/test_trace_headers.py::test_health_includes_trace_headers (X-Request-ID missing from the response headers)
- Utils/test_utils_general.py::test_safe_read_file_handles_empty_decodes (ERROR at setup: TypeError: 'DummyFile' object is not iterable; a patched open() leaks into config.py's configparser read)
Each points at shared process state (patched builtins, middleware or env) that an earlier test in the same xdist worker leaves behind. Found while running the broad suites for the RG safety-net PR B (plan 2026-09-29-rg-ingress-safety-net).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The four tests pass under the same -n 8 command
- [x] #2 Each polluting fixture restores the state it patches (monkeypatch or finally)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
More xdist pollution seen in the final-fix run, reproduced at 2d62baa055:
- test_legacy_openai_transient_failure_retains_retry_policy: the system_log_buffer writer thread's 0.05 s flock poll lands in the test's patched global time.sleep.
- 3 test_orchestrator_summary nodes: the redis_client fixture runs flushdb on the shared localhost:6379/0 while other workers are mid-test.

Also: the rg-redis variants in Resource_Governance/test_e2e_domains_headers.py talk to an ambient Redis on 127.0.0.1:6379 when one is running. It had ~665 stale rg* keys from other runs, so a colliding policy id starts the test already rate-limited. These tests should inject InMemoryAsyncRedis, as test_governor_safety_net.py does.

Four tests reproduced and fixed under RUN_EVALUATIONS=1 RUN_JOBS=1 pytest -n 8 over tests/Resource_Governance tests/AuthNZ_Unit tests/Embeddings tests/lint tests/Utils (both required -n 8 runs: 3244 passed / 25 skipped / 2 xfailed, the only failure in each run was the pre-existing, out-of-scope test_backpressure_and_quotas.py::test_tenant_quota_429, which fails standalone too and is unrelated to this task).

1) test_e2e_tokens_daily_cap.py::test_e2e_embeddings_tokens_daily_cap_denies[rg-memory/rg-redis] (assert [] == [False]):
   Polluter: tests/Embeddings/test_embeddings_v5_unit.py sets os.environ["TESTING"]="true" at MODULE import time (before any fixture runs), with the matching restore living only in its module-scoped autouse `cleanup_testing_env` fixture's teardown. Under xdist every worker collects (imports) this module even when none of its own tests get scheduled onto that worker, so the module-level set fires but the fixture (and its teardown) never does, leaking TESTING=true for the rest of that worker. With TESTING truthy, embeddings_v5_production_enhanced.py's `use_synthetic_openai` fast-path bypasses the (mocked) create_embeddings_batch_async entirely, so the test's mock is never called -> cache_scope_sensitive_values stays [].
   Fix: restore both env vars immediately after the import block that needed them (test_embeddings_v5_unit.py), closing the leak window independent of whether this worker ever runs that module's own tests.
   Proof: `pytest tests/Embeddings/test_embeddings_v5_unit.py tests/Resource_Governance/test_e2e_tokens_daily_cap.py::test_e2e_embeddings_tokens_daily_cap_denies -k test_e2e_embeddings_tokens_daily_cap_denies` (the -k deselect simulates "collected, not executed" the same way xdist does) -> FAILED both params before the fix, 2 passed after.

2) test_trace_headers.py::test_health_includes_trace_headers (X-Request-ID missing):
   Polluter: tests/AuthNZ/conftest.py's `reset_singletons` (autouse; becomes global for the whole xdist worker once any file opts in via `pytest_plugins = (authnz_full_fixtures,)` -- e.g. tests/AuthNZ_Unit/test_mcp_hub_repo.py) strips RequestIDMiddleware/HTTPMetricsMiddleware/SecurityHeadersMiddleware from the shared app.user_middleware in its setup with no restore anywhere. Fixed the fixture to save the pre-strip list and put it back after `yield`.
   That restore alone isn't enough for this specific victim, though: because reset_singletons is now global, it ALSO strips those middlewares around trace_headers' own test body (strip-before-yield is by design, for the AuthNZ tests that opted in; trace_headers has no way to opt out of the leaked scope). Hardened the victim to guarantee RequestIDMiddleware is present for its own request, mirroring the existing `_with_rg_middleware` pattern from the Resource_Governance e2e tests.
   Proof: `pytest tests/AuthNZ_Unit/test_mcp_hub_repo.py tests/AuthNZ_Unit/test_mcp_hub_capability_adapter_repo.py tests/Embeddings/test_trace_headers.py` -> FAILED trace_headers before either fix; 18 passed after both.

3) test_utils_general.py::test_safe_read_file_handles_empty_decodes (ERROR at TEARDOWN: TypeError: 'DummyFile' object is not iterable):
   Real polluter is the same reset_singletons global-scope leak as #2: its teardown calls shutdown_audit_service() -> ... -> config.py's lazy settings loader -> configparser.read(), landing on this test's own `monkeypatch.setattr(builtins, "open", ...)` which is still active during that teardown. Hardened the victim (the test's patch was genuinely too broad): scope the override to `Utils.open` (the module's own globals) instead of `builtins.open`, so only Utils.py's bare `open(...)` call is intercepted and configparser's real file read elsewhere in the process is unaffected.
   Proof: `pytest tests/AuthNZ_Unit/test_mcp_hub_repo.py tests/AuthNZ_Unit/test_mcp_hub_capability_adapter_repo.py tests/Embeddings/test_trace_headers.py tests/Utils/test_utils_general.py` -> ERROR at teardown before the fix (TypeError: 'DummyFile' object is not iterable); 0 failures/errors after (1218 passed, 1 skipped).

Additional xdist pollution from the task notes, fixed in the same commit:
- test_legacy_openai_transient_failure_retains_retry_policy: patched `ec.time.sleep`, which is the global `time` module object (not a per-module copy), so the system_log_buffer writer thread's 0.05s flock-poll sleep could land in the test's `sleeps` list from another thread. Replaced the `time` name in ec's own module globals with a small proxy (delegates everything except `.sleep` to the real module), scoping the patch to just this module's call site. Standalone: 1 passed.
- 3 test_orchestrator_summary_endpoint.py nodes (uses the `redis_client` fixture): the fixture ran flushdb on the shared localhost:6379/0 Redis at setup and teardown, wiping another xdist worker's mid-test streams/keys (reproduced: assert 3 == 2 on a stream length). Gave each xdist worker its own logical Redis DB index (0-15) derived from PYTEST_XDIST_WORKER instead of hardcoding /0. Verified via `pytest -n4 tests/Embeddings`: failure gone after the fix (680 passed; 2 unrelated Hypothesis property tests flaked under this machine's heavy concurrent load from other sessions and passed standalone -- pre-existing, not touched).
- test_e2e_domains_headers.py rg-redis variants: talked to whatever real Redis is on 127.0.0.1:6379 (confirmed 57 stale rg* keys present on this machine during this run), risking an already-rate-limited start and cross-worker interleaving. Injected InMemoryAsyncRedis via monkeypatching governor_redis.RedisResourceGovernor, the same way test_governor_safety_net.py's `_gov` already does. Verified: 4 passed despite the ambient stale keys.

Test-only changes across 7 files (tests/AuthNZ/conftest.py, tests/Embeddings/conftest.py, test_embeddings_create_credential_policy.py, test_embeddings_v5_unit.py, test_trace_headers.py, tests/Resource_Governance/test_e2e_domains_headers.py, test_utils_general.py); no app code touched. ruff check: zero new findings introduced (verified file-by-file against origin/dev; test_trace_headers.py's rewrite incidentally fixed 2 pre-existing findings).

Both required `-n 8` runs and the changed-files-alone run are in the implementer's final report. Known out-of-scope flake found during verification: tests/Embeddings/test_backpressure_and_quotas.py::test_tenant_quota_429 fails standalone on this checkout (pre-existing, unrelated to xdist pollution) -- not part of this task's AC, not fixed here.

DoD: test-only change, so Bandit doesn't apply (no app code touched). Known limits: the per-worker Redis DB index wraps at 16 workers, and an explicit TEST_REDIS_URL/EMBEDDINGS_REDIS_URL/REDIS_URL bypasses it (pre-existing precedence).

Qodo review on PR #3085: addressed all 8 findings, commit 7afbbce75e (not pushed yet; coordinator will push when this PR's turn comes).

(5) reset_singletons restored onto a freshly re-imported app.main.app at teardown instead of the object it actually stripped at setup, so a mid-fixture app reload (auth tests do this) would leave the real stripped app untouched. Fixed: capture the app object itself (_stripped_app) at setup, restore onto that same object.

(7) The restore ran after ~16 unguarded teardown calls; any one raising skipped it. Fixed: wrap `yield` in try/finally, restore first inside the finally.

Verified (5) and (7) directly: drove reset_singletons.__wrapped__ as a plain async generator (bypassing pytest), swapped app.main.app for an unrelated object between setup/teardown (restore still landed on the real stripped object -- 5), and made reset_db_pool raise on its 2nd call only, i.e. the teardown one (middleware still restored despite the raise -- 7). Both passed.

(6)/(8) _xdist_worker_db_index() mapped gw0 to DB 0 (the app's default DB) and wrapped at 16 workers. Fixed: workers map to DB 1..15 (1+n); a 16th+ worker (gw15+) now pytest.skip()s with a clear reason instead of reusing an index; outside xdist, DB 1 instead of 0. Explicit TEST_REDIS_URL/EMBEDDINGS_REDIS_URL/REDIS_URL precedence unchanged (short-circuits before the function is called). Verified: gw0->1, gw1->2, gw7->8, gw14->15, no-worker->1, gw15->skipped with the stated reason.

(1)-(4) Added type hints (test_trace_headers.py's middleware helper + its Iterator[None] return, the _ScopedTime proxy in test_embeddings_create_credential_policy.py, the rg_backend fixture's request/monkeypatch params in test_e2e_domains_headers.py) and a docstring on the _InMemoryRedisGovernor test double explaining why it's a subclass rather than an instance patch.

Re-ran: all 5 changed files alone (131 passed) plus test_orchestrator_summary_endpoint.py (9 passed); the tokens_daily_cap victim in the polluted order (2 passed); the AuthNZ_Unit + trace_headers + utils_general polluted-order combo in full (1218 passed, 1 skipped, 0 failed) and the fast 4-file subset of it (35 passed). No regressions.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
