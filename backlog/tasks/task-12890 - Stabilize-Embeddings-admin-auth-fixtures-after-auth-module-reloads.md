---
id: TASK-12890
title: Stabilize Embeddings admin auth fixtures after auth module reloads
status: Done
assignee: []
created_date: '2026-07-04 19:03'
updated_date: '2026-07-04 19:34'
labels:
  - tests
  - embeddings
  - auth
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Discord-to-Jobs verification slice can fail Embeddings admin endpoint tests with 401 responses because the shared Embeddings admin fixtures rely on stale auth dependency keys and a hardcoded Bearer test-api-key that does not match the deterministic single-user key used by the suite.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Priority bump admin endpoint tests do not return 401 in focused runs.
- [x] #2 The Discord-to-Jobs slice progresses past the priority bump endpoint tests.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update the shared Embeddings admin fixture to override the current auth_deps dependency keys and known endpoint-imported keys used by admin Embeddings routes.
2. Update the local auth_headers fixture to use the active deterministic SINGLE_USER_API_KEY instead of a hardcoded mismatched key.
3. Run focused priority bump tests, the Embeddings prefix, and the broader Discord-to-Jobs slice.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Updated the shared Embeddings admin fixtures to override registered route auth dependency callables and to use the active deterministic single-user key in headers.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed Embeddings admin auth fixture instability by overriding the actual get_request_user and get_auth_principal callables registered on Embeddings/vector-store routes, and by deriving Authorization/X-API-KEY headers from get_settings().SINGLE_USER_API_KEY. Verification: focused priority bump tests passed (2 passed); focused touched-scope command passed (44 passed); Discord-to-Jobs slice passed (3247 passed, 156 skipped); git diff --check passed; Bandit on touched tests reported no findings.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Focused priority tests output captured.
- [x] #2 Broad slice output captured.
- [x] #3 Task updated with final summary.
<!-- DOD:END -->
