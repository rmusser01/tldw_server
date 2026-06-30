---
id: TASK-446
title: Fix audio preset Postgres boolean partial indexes
status: Done
labels:
- audio
- postgres
- review-fix
- db-schema
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Validate and fix the PR #1865 follow-up finding that audio preset partial indexes use SQLite integer boolean predicates which can survive SQLite-to-Postgres schema conversion as invalid PostgreSQL SQL. Keep the fix scoped to audio preset schema conversion/bootstrap and add focused regression coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Verify whether the reported PostgreSQL predicate issue is real.
- [x] #2 Audio preset partial indexes use SQLite-compatible boolean literals that remain valid after PostgreSQL schema conversion.
- [x] #3 Regression coverage proves converted audio preset index predicates use `TRUE`/`FALSE`, not integer boolean comparisons.
- [x] #4 Focused schema/audio tests, Bandit, and diff checks are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Confirmed the issue is valid: `_AUDIO_PRESETS_TABLE_SQL` is converted for PostgreSQL bootstrap, and the converter only rewrites a narrow `WHERE deleted = 0` pattern. The composite `WHERE is_default = 1 AND deleted = 0` predicate survived conversion as integer comparisons against PostgreSQL boolean columns. Added a failing schema-conversion regression, then changed the audio preset partial index predicates to `deleted = FALSE` and `is_default = TRUE AND deleted = FALSE`; SQLite accepts these literals and PostgreSQL conversion now preserves valid boolean predicates.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the audio preset partial index predicates introduced by PR #1865 so PostgreSQL bootstrap SQL remains valid after SQLite-to-PostgreSQL conversion. Added regression coverage for the converted audio preset index SQL and verified adjacent audio preset endpoint behavior.

Verification: the new regression failed before the schema fix, then passed. Focused pytest passed 9 tests across schema conversion and audio preset endpoint coverage. Bandit on the touched backend schema file reported 0 findings. `git diff --check` passed.
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
