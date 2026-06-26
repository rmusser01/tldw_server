---
id: TASK-12021
title: Address follow-up infrastructure and security review findings
status: Done
created_date: 2026-06-26 18:18
labels:
- backend
- security
- infrastructure
- review-fix
priority: high
references:
- tldw_Server_API/app/core/Infrastructure/distributed_lock.py
- tldw_Server_API/app/core/Security/egress.py
- tldw_Server_API/app/core/Infrastructure/redis_factory.py
- tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py
modified_files:
- tldw_Server_API/app/core/Infrastructure/distributed_lock.py
- tldw_Server_API/app/core/Security/egress.py
- tldw_Server_API/app/core/Infrastructure/redis_factory.py
- tldw_Server_API/app/core/Infrastructure/README.md
- tldw_Server_API/app/api/v1/API_Deps/backpressure.py
- tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py
- tldw_Server_API/tests/Infrastructure/test_distributed_lock.py
- tldw_Server_API/tests/Security/test_egress.py
- tldw_Server_API/tests/Infrastructure/test_redis_factory.py
- tldw_Server_API/tests/Infrastructure/test_redis_factory_metrics.py
- tldw_Server_API/tests/Embeddings/test_backpressure_and_quotas.py
updated_date: 2026-06-26 18:45
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix follow-up code review findings in infrastructure and security hardening: file lock ownership races, bounded DNS resolver behavior, explicit Redis fake fallback policy, Redis migration lock exception handling, and embeddings tenant quota fail-closed behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 FileLock release cannot delete a newer active lock and has focused regression coverage.
- [x] #2 FileLock residual files do not break native lock ownership and stale-file unlinking is not used for active locks.
- [x] #3 acquire_migration_lock preserves exceptions raised inside the protected body while still failing closed on Redis setup errors.
- [x] #4 Egress DNS timeout handling has bounded outstanding resolver work and focused regression coverage.
- [x] #5 Redis factory fake fallback is explicit for production-sensitive callers and missing-package behavior has focused regression coverage.
- [x] #6 Embeddings tenant RPS quota fails closed when Redis quota state is unavailable.
- [x] #7 Focused tests, diff check, py_compile, and Bandit touched-scope scan are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Port the reviewed fixes into a clean branch based on origin/dev. 2. Run focused regression tests and related suites in the clean worktree. 3. Run py_compile, diff hygiene, and Bandit on touched production files. 4. Mark the task Done with verification evidence before opening the PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Clean PR branch created from origin/dev at 6fe09bb972 in .worktrees/task-12021-followup-review-fixes.

Implemented fixes:
- FileLock keeps lock files as durable metadata and no longer unlinks them on normal release.
- FileLock no longer attempts stale-file unlinking on failed acquisition; native platform locks determine ownership.
- acquire_migration_lock already separated Redis setup errors from protected-body exceptions on the clean base; retained and verified that behavior.
- Redis factory fake fallback is disabled by default for async and sync factories; callers must opt in explicitly.
- Egress DNS resolution uses bounded resolver slots so stuck resolver calls cannot create unbounded outstanding work.
- Embeddings tenant RPS quota fails closed with HTTP 503 and Retry-After when Redis quota state is unavailable.

Verification in clean worktree:
- Baseline before patch: `python -m pytest tldw_Server_API/tests/Infrastructure/test_distributed_lock.py tldw_Server_API/tests/Security/test_egress.py tldw_Server_API/tests/Infrastructure/test_redis_factory.py tldw_Server_API/tests/Infrastructure/test_redis_factory_metrics.py tldw_Server_API/tests/test_minimal_deploy.py tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py -q` -> 49 passed.
- After patch: `python -m pytest tldw_Server_API/tests/Infrastructure/test_distributed_lock.py tldw_Server_API/tests/Embeddings/test_backpressure_and_quotas.py tldw_Server_API/tests/Security/test_egress.py tldw_Server_API/tests/Infrastructure/test_redis_factory.py tldw_Server_API/tests/Infrastructure/test_redis_factory_metrics.py tldw_Server_API/tests/test_minimal_deploy.py tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py -q` -> 60 passed.
- `python -m py_compile tldw_Server_API/app/core/Infrastructure/distributed_lock.py tldw_Server_API/app/core/Security/egress.py tldw_Server_API/app/core/Infrastructure/redis_factory.py tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py` -> passed.
- `git diff --check -- ...touched files...` -> passed.
- `python -m bandit -r tldw_Server_API/app/core/Infrastructure/distributed_lock.py tldw_Server_API/app/core/Security/egress.py tldw_Server_API/app/core/Infrastructure/redis_factory.py tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py -f json -o /tmp/bandit_task_12021_clean_pr.json` -> 0 errors, 0 findings.

Documentation: no user-facing docs change needed; this is internal hardening.
Reopened after self-review before merge. Review follow-up items:
- Ingest tenant quota path must convert Redis quota outages into controlled 503 responses after Redis fallback default changed to fail closed.
- Redis factory README must reflect fallback_to_fake=False defaults and explicit opt-in examples.
- Remove unused FileLock instance owner token state while retaining lock-file owner token metadata.
Addressed all self-review findings before continuing:
- Added red regression test for ingest tenant quota Redis outage; confirmed it failed with raw RuntimeError before the fix.
- Added red README contract test for Redis factory fail-closed defaults and explicit fallback opt-in example; confirmed it failed before the docs update.
- Changed ingest quota enforcement to raise controlled HTTP 503 with Retry-After when Redis quota state is unavailable.
- Updated Infrastructure Redis factory README to document fallback_to_fake=False defaults and explicit fallback_to_fake=True examples.
- Removed unused FileLock `_owner_token` instance state while retaining per-acquire token metadata in the lock file.

Review follow-up verification:
- Red run: `python -m pytest tldw_Server_API/tests/Embeddings/test_backpressure_and_quotas.py::test_ingest_tenant_quota_fails_closed_when_redis_unavailable tldw_Server_API/tests/Infrastructure/test_redis_factory.py::test_redis_factory_readme_documents_fail_closed_defaults -q` -> 2 failed for expected reasons.
- Green focused run: same command -> 2 passed.
- Full focused run: `python -m pytest tldw_Server_API/tests/Infrastructure/test_distributed_lock.py tldw_Server_API/tests/Embeddings/test_backpressure_and_quotas.py tldw_Server_API/tests/Security/test_egress.py tldw_Server_API/tests/Infrastructure/test_redis_factory.py tldw_Server_API/tests/Infrastructure/test_redis_factory_metrics.py tldw_Server_API/tests/test_minimal_deploy.py tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py -q` -> 62 passed.
- `python -m py_compile tldw_Server_API/app/core/Infrastructure/distributed_lock.py tldw_Server_API/app/core/Security/egress.py tldw_Server_API/app/core/Infrastructure/redis_factory.py tldw_Server_API/app/api/v1/API_Deps/backpressure.py tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py` -> passed.
- `git diff --check -- ...touched files...` -> passed.
- `python -m bandit -r tldw_Server_API/app/core/Infrastructure/distributed_lock.py tldw_Server_API/app/core/Security/egress.py tldw_Server_API/app/core/Infrastructure/redis_factory.py tldw_Server_API/app/api/v1/API_Deps/backpressure.py tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py -f json -o /tmp/bandit_task_12021_review_followup.json` -> 0 errors, 0 findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created a clean PR branch from origin/dev, ported the TASK-12021 review fixes, and addressed the follow-up self-review findings. The branch keeps file locks from unlinking active successors, preserves migration body exceptions, defaults Redis helper fallbacks to fail-closed behavior unless explicitly enabled, bounds egress DNS resolver work, fails closed for embeddings and ingest tenant quota Redis outages, and updates Redis factory docs to match the new fallback contract. Focused tests, compile, diff hygiene, and Bandit all passed in the clean worktree.
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
