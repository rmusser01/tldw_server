---
id: TASK-15498
title: 17 Telegram tests fail with 403 on dev, and no CI shard runs tests/Telegram
status: Done
labels:
- tests
- ci
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-15426 PR D's test-isolation investigation (2026-11-13). Run locally, 17 tests under tldw_Server_API/tests/Telegram/ fail with 403 'Not authorized to manage shared BYOK keys', on PR D's head and on origin/dev alike (not caused by PR D; not investigated). CI does not catch it: every Telegram test file is listed in Helper_Scripts/ci/shard_coverage_baseline.txt (the grandfathered list of test files no CI shard runs), so no required check executes them. Whether the 403 is a product regression (who may manage shared, team or org, BYOK keys from Telegram flows) or a test-setup gap is unknown.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The cause of the 403 is found: product regression (fixed, with a test) or test setup (fixed)
- [x] #2 tests/Telegram runs in a CI shard and its files leave shard_coverage_baseline.txt
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root cause (test setup, not a product regression): the shared-BYOK write path (AuthnzOrgProviderSecretsRepo._lock_authorized_active_parent_scope, required_authority=PLATFORM_ADMIN) authorizes the actor against persisted users/organizations/teams rows, never the principal's claimed roles. The four failing files faked AuthPrincipal(user_id=202/303/404, roles=[admin]) with no rows, so PUT /telegram/admin/bot returned 403. The DB-authoritative actor-row check predates f7af8c9181 (it is already in a7fa919b80); f7af8c9181 and e57fce5ff4 only tightened it. The tests went stale because no shard ran them.
Fix: tests/Telegram/conftest.py fixture back_principal seeds real users/organizations/teams rows via ensure_test_user and returns the principal with the real user_id; four principal_override fixtures use it. The authorization check is untouched; no bypass added. Added a telegram shard to the five ci.yml shard matrices and removed the nine tests/Telegram lines from shard_coverage_baseline.txt.
Owner decisions 2026-11-16: (1) Org/team admins managing their own Telegram bot: no change now, consistent with shared BYOK where only a platform admin may write; nothing filed. (2) The telegram shard stays serial; to parallelize it each xdist worker would need its own DATABASE_URL (the runtime repo uses the shared global pool). (3) No lighter repo-override path: tests keep exercising the real authorization path. (4) ADR-004 Change-summary waiver covers this PR.
Verification: webhook file RED 5 failed/4 passed before the fix; tests/Telegram plus tests/AuthNZ_Unit/test_telegram_admin_write_authority.py 97 passed serially, netlines 0. check_shard_coverage.py OK, tests/Docs 214 passed, tests/CI 1074 passed with the 2 known env-only coverage failures. Bandit skipped: only test and CI config touched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
17 Telegram tests failed because four test files faked an admin principal with no persisted users/orgs/teams rows, which the shared-BYOK write path correctly rejects. Added a back_principal fixture that seeds real rows, added a serial telegram CI shard in all five matrices, and emptied the nine tests/Telegram files from the shard-coverage baseline.
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
