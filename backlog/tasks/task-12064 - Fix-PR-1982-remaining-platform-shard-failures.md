---
id: TASK-12064
title: Fix PR 1982 remaining platform shard failures
status: Done
assignee:
- Codex
labels:
- ci
- pr-1982
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and address the remaining PR #1982 CI failures on commit 36014d9c211a5b7fbbf2af0600e8489353b9f88f: platform-sandbox-state-store on Ubuntu Python 3.12 and platform-infrastructure-metrics on Windows Python 3.12. Keep changes minimal, verify locally where feasible, run Bandit for touched Python code, and push fixes to the PR branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Reproduce or verify both PR #1982 shard failures against current logs/code.
- [x] Fix the Windows provider-registry retry-window race without widening production retry behavior.
- [x] Fix sandbox active-run admission so expired claims do not block slots indefinitely.
- [x] Validate focused regressions and the two formerly failing shard scopes locally.
- [x] Run Bandit on touched production scope and document test-only baseline warnings.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Live PR #1982 status for run `28398063376` had two actionable failed shard jobs:
  - `platform-sandbox-state-store` on Ubuntu/Python 3.12: `test_background_execution_respects_max_concurrent_runs` timed out waiting for the first background runner start.
  - `platform-infrastructure-metrics` on Windows/Python 3.12: `test_get_adapter_async_respects_retry_window_after_failure` saw `ProviderStatus.ENABLED` immediately after a failed async materialization because the 20 ms wall-clock retry window had already elapsed on CI.
- Provider-registry fix is test-only: replace real `asyncio.sleep()`/wall-clock timing with a monkeypatched deterministic `time.time()` clock.
- Sandbox root cause: active admission counted `starting`/`running` rows even when their claim lease had expired, and background workers waiting for an active slot did not renew their queued claim. This could leave workers stuck in `_admit_run_starting()` and can also make stale active rows block future runs.
- Sandbox fix:
  - Ignore expired active claims in in-memory, SQLite, and Postgres active-slot counts.
  - Renew queued background run claims while waiting for an active slot.
  - Added regression coverage for expired active claims and queued claim renewal while waiting.
- Bandit:
  - Touched production files only: `/tmp/bandit_task12064_app.json`, 0 results.
  - Full touched scope: `/tmp/bandit_task12064.json`, B101 assert warnings in test files only; no production findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Fixed PR #1982's remaining actionable CI failures by removing the provider-registry wall-clock race and hardening sandbox claim/admission behavior against expired leases.
- Local validation:
  - Focused regressions: 3 passed.
  - Affected files in CI-style mode: 17 passed.
  - Exact `platform-sandbox-state-store` shard scope: 231 passed, 7 skipped.
  - Exact `platform-infrastructure-metrics` shard scope: 391 passed.
  - Production Bandit touched scope: 0 findings.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
