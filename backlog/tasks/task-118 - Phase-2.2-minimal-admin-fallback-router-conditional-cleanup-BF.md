---
id: TASK-118
title: Phase 2.2 minimal admin fallback router conditional cleanup BF
status: Done
assignee: []
created_date: '2026-05-07 20:06'
updated_date: '2026-05-08 00:43'
labels:
  - phase-2.2
  - router-groups
  - minimal
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1367'
  - 'https://github.com/rmusser01/tldw_server/pull/1369'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move the minimal-test admin/admin_byok fallback block onto shared optional-router semantics so missing optional targets are skippable without catching every runtime import defect. Preserve the primary admin router preference, admin_byok fallback behavior, prefixes, tags, and minimal-test contract metadata.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Minimal admin route metadata is represented through a shared optional-router-compatible path while preserving prefix /api/v1, admin tag, and primary admin preference.
- [x] #2 Minimal admin_byok fallback metadata is represented through a shared optional-router-compatible path while preserving prefix /api/v1/admin, admin tag, and fallback behavior when the primary admin router is missing.
- [x] #3 Focused contract coverage proves missing admin target skips to admin_byok while runtime import defects propagate instead of being swallowed by broad Exception handling.
- [x] #4 Focused source/contract coverage removes the minimal admin broad except import block while existing minimal metadata contracts still pass.
- [x] #5 Existing router group, main router, and OpenAPI contracts still pass for the touched scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED contract tests for admin/admin_byok fallback skip semantics and runtime defect propagation.
2. Replace the broad try/except import block with shared optional-router resolution semantics while preserving fallback behavior.
3. Run focused, router-group, main-router, OpenAPI, Bandit, and diff hygiene verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Baseline verification before edits: python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -k "minimal_optional_router_specs and admin" -q passed with 1 passed, 164 deselected, 10 warnings.

RED verification: the new runtime-defect test failed before implementation because the broad admin import catch swallowed RuntimeError and fell through to admin_byok.

GREEN verification: focused selector python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -k "admin_fallback or (minimal_optional_router_specs and admin)" -q passed with 3 passed, 164 deselected, 10 warnings. Full router group contracts passed with 167 passed, 30 warnings. Main router contracts passed with 6 passed, 5 warnings. OpenAPI contracts passed with 69 passed, 24 warnings. Bandit on tldw_Server_API/app/api/v1/router_groups/minimal.py reported 0 results and 0 errors. git diff --check passed.

No documentation change required for this internal router-registration cleanup. No blockers or known skips beyond existing test-suite warnings.

PR review follow-up: addressed Gemini/Qodo review comments by adding a defensive empty candidate guard, switching fallback skip handling to candidate_spec.skip_exceptions, restoring debug logging for skipped fallback candidates, and adding docstrings to the new source helper and tests. Post-review validation: focused selector passed with 3 passed, 164 deselected, 10 warnings; full router group contracts passed with 167 passed, 30 warnings; main router contracts passed with 6 passed, 5 warnings; OpenAPI contracts passed with 69 passed, 24 warnings; Bandit reported 0 results and 0 errors; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Moved the minimal admin/admin_byok fallback away from broad try/except imports and onto the shared ImportedRouterSpec resolution path. The primary admin router remains preferred with prefix /api/v1 and admin tags, while admin_byok remains the fallback with prefix /api/v1/admin when the primary optional target is missing. Missing optional modules or router attributes continue to skip/fallback, but runtime import defects now propagate instead of being silently swallowed. Added focused contract coverage for primary preference, fallback behavior, runtime defect propagation, and removal of the broad exception source pattern.
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
