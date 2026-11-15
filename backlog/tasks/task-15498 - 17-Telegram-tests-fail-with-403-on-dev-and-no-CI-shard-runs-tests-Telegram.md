---
id: TASK-15498
title: 17 Telegram tests fail with 403 on dev, and no CI shard runs tests/Telegram
status: To Do
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
- [ ] #1 The cause of the 403 is found: product regression (fixed, with a test) or test setup (fixed)
- [ ] #2 tests/Telegram runs in a CI shard and its files leave shard_coverage_baseline.txt
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
