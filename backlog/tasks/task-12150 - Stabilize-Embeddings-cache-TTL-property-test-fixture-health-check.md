---
id: TASK-12150
title: Stabilize Embeddings cache TTL property test fixture health check
status: Done
assignee: []
created_date: '2026-07-04 18:46'
updated_date: '2026-07-04 19:34'
labels:
  - tests
  - embeddings
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The broad Discord-to-Jobs verification slice stops in the Embeddings property tests because Hypothesis rejects a property test that uses the function-scoped monkeypatch fixture without suppressing the fixture health check. Neighboring fixture-backed property tests in the same file already suppress this check.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The focused cache TTL property test no longer fails Hypothesis health checks.
- [x] #2 The Discord-to-Jobs verification slice progresses past the Embeddings cache TTL property test.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add the same targeted Hypothesis function-scoped fixture health-check suppression to the cache TTL property test.
2. Re-run the focused property test.
3. Re-run the broad Discord-to-Jobs verification slice.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Added the existing Hypothesis function-scoped fixture health-check suppression to the cache TTL property test that uses monkeypatch.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the Embeddings cache TTL property test by adding the same HealthCheck.function_scoped_fixture suppression already used by neighboring fixture-backed property tests. Verification: focused cache TTL property test passed (1 passed); focused touched-scope command passed (44 passed); Discord-to-Jobs slice passed (3247 passed, 156 skipped); git diff --check passed; Bandit on touched tests reported no findings.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Focused property test output captured.
- [x] #2 Broad slice verification output captured.
- [x] #3 Task updated with final summary.
<!-- DOD:END -->
