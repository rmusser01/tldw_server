---
id: TASK-13400
title: Four tests fail under pytest -n 8 from cross-test pollution
status: To Do
assignee: []
created_date: '2026-09-30 07:02'
updated_date: '2026-09-30 09:47'
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
- [ ] #1 The four tests pass under the same -n 8 command
- [ ] #2 Each polluting fixture restores the state it patches (monkeypatch or finally)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
More xdist pollution seen in the final-fix run, reproduced at 2d62baa055:
- test_legacy_openai_transient_failure_retains_retry_policy: the system_log_buffer writer thread's 0.05 s flock poll lands in the test's patched global time.sleep.
- 3 test_orchestrator_summary nodes: the redis_client fixture runs flushdb on the shared localhost:6379/0 while other workers are mid-test.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
