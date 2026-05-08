---
id: TASK-119
title: Phase 2.2 minimal always-included router lazy import cleanup BG
status: Done
assignee: []
created_date: '2026-05-08 02:00'
updated_date: '2026-05-08 02:18'
labels:
  - phase-2.2
  - router-groups
  - minimal
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1370'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue Phase 2.2 router cleanup after PR #1369 by deferring the always-included minimal-test router imports in router_groups/minimal.py while preserving prefixes, tags, and runtime defect behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 iter_minimal_test_router_specs returns RouterSpec entries without importing endpoint modules during spec construction
- [x] #2 health/auth/research/chat/character/workspace minimal router metadata remains unchanged
- [x] #3 runtime import defects for always-included minimal routers propagate instead of being silently skipped
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
PR #1370 review fix pass: verify each review finding against current code, fix only still-valid issues, keep edits minimal, validate with focused pytest/diff/Bandit checks, then reply to and resolve handled GitHub review threads.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Baseline selector passed before edits with 2 passed, 165 deselected, 6 warnings. RED selector failed as expected because iter_minimal_test_router_specs eagerly imported auth and the source still contained direct endpoint imports. GREEN selector passed with 5 passed, 165 deselected, 6 warnings after converting always-included minimal routers to required ImportedRouterSpec factories.

Validation: full router group contracts passed with 170 passed, 30 warnings; main router contracts passed with 6 passed, 5 warnings; OpenAPI contracts passed with 69 passed, 24 warnings; Bandit on tldw_Server_API/app/api/v1/router_groups/minimal.py reported 0 results and 0 errors; git diff --check passed. No documentation change required for this internal router-registration cleanup.

Reopened for PR #1370 review comments. Gemini ModuleType import thread was invalid: from types import ModuleType already exists and the targeted test passes; replied and resolved. CodeRabbit's registry-path coverage comment is valid and will be handled with a minimal test-only update.

PR #1370 review fix validation: focused pytest for test_iter_minimal_test_router_specs_propagates_runtime_import_failures passed with 1 passed, 169 deselected; full router group contract suite passed with 170 passed; Bandit on the touched test file with B101 skipped reported 0 results and 0 errors; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Deferred always-included minimal-test router imports through ImportedRouterSpec while preserving prefixes/tags and required-router failure propagation. Added contract coverage for no eager endpoint imports, source cleanup, and runtime defect propagation.

Review fix: strengthened the required minimal-router runtime defect test to assert empty skip_exceptions and execute through register_router_specs(FastAPI(), (auth_spec,)), covering the registry skip-policy path requested by CodeRabbit.
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
