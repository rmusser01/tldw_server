---
id: TASK-9933
title: Harden Infrastructure module review findings
status: Done
assignee: []
created_date: 2026-06-23 18:55
updated_date: 2026-06-24 03:49
labels:
- infrastructure
- backend
- review-fix
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address validated review findings in tldw_Server_API/app/core/Infrastructure. Scope: distributed lock safety, Redis factory fallback/redaction, provider registry sync initialization concurrency, pool metrics robustness, and circuit breaker persistence semantics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Validated Infrastructure review findings are reproduced or explicitly rejected with evidence.
- [x] #2 Accepted fixes include focused regression tests before production code changes.
- [x] #3 Touched Infrastructure code passes targeted pytest checks and Bandit scan.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce validated Infrastructure review findings with focused tests. 2. Patch Infrastructure modules with minimal behavior changes. 3. Run targeted pytest checks, py_compile, and Bandit on touched scope. 4. Record final summary and verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Touched files:
- IMPLEMENTATION_PLAN_infrastructure_hardening_task_9933.md
- tldw_Server_API/app/core/Infrastructure/distributed_lock.py
- tldw_Server_API/app/core/Infrastructure/redis_factory.py
- tldw_Server_API/app/core/Infrastructure/provider_registry.py
- tldw_Server_API/app/core/Infrastructure/pool_metrics.py
- tldw_Server_API/app/core/Infrastructure/circuit_breaker.py
- tldw_Server_API/tests/Infrastructure/test_distributed_lock.py
- tldw_Server_API/tests/Infrastructure/test_redis_factory.py
- tldw_Server_API/tests/Infrastructure/test_provider_registry_base.py
- tldw_Server_API/tests/Infrastructure/test_pool_metrics.py
- tldw_Server_API/tests/Infrastructure/test_circuit_breaker.py

Verification from isolated worktree .worktrees/infrastructure-review-fixes-9933 after rebasing onto origin/dev:
- py_compile passed for edited Infrastructure modules.
- git diff --check passed.
- pytest distributed-lock focused set: 6 passed.
- pytest Redis factory focused set: 5 passed.
- pytest provider registry/pool metrics/circuit breaker focused set: 3 passed.
- Bandit touched Infrastructure files: exit 0, JSON report at /tmp/bandit_infrastructure_task_9933_rebased.json.

Known skips/blockers: full repository pytest was not run; focused Infrastructure regression tests were run for the review fixes. Pytest cleanup is slow in this repo because global app teardown loads full app services.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened Infrastructure review findings: migration locks now fail closed when Redis is explicitly configured but unavailable unless fallback is opted in; file locks no longer break live owners by age; Redis locks support token-checked TTL renewal; Redis factory fallback works when the redis package is missing and redacts URL user-info; sync provider adapter initialization is serialized per provider; pool metrics fail closed to unavailable on broken accessors; persistent circuit breakers reject unsupported rolling-window configs.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Acceptance criteria completed
- [x] #8 Tests or verification recorded
- [x] #9 Bandit run for touched code
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR review follow-up after latest dev rebase:
- Rebased branch onto origin/dev and addressed PR comments on migration-lock fallback scope, Redis URL redaction error handling, Redis factory tests using private internals, and pool metrics broad-exception logging.
- Added regression coverage that Redis-backed acquire_migration_lock propagates caller exceptions, does not rerun caller blocks through file fallback, and still closes the Redis client.
- Moved Redis factory redaction checks behind public create_*_redis_client behavior and warning-output assertions.
- Added debug logging assertion for pool metric accessor failures.
- Added tldw_Server_API/tests/Agent_Client_Protocol/test_acp_hardening_helpers.py to the existing shard coverage baseline because the rebased PR's Shard coverage guard found it newly unshared.

Fresh verification after review follow-up:
- py_compile passed for touched Infrastructure modules.
- git diff --check passed.
- Shard coverage guard passed: new_uncovered=0.
- pytest focused Infrastructure suite: 38 passed, 88 warnings.
- Bandit touched Infrastructure files: exit 0, 0 results, JSON report at /tmp/bandit_infrastructure_task_9933_review_followup.json.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
