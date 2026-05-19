---
id: TASK-99
title: Phase 2.2 minimal tail router conditional cleanup AV
status: Done
assignee: []
created_date: '2026-05-07 00:44'
updated_date: '2026-05-07 01:01'
labels:
  - phase2.2
  - router-cleanup
  - minimal
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1345'
  - 'https://github.com/rmusser01/tldw_server/pull/1347'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Convert the minimal-test agent_orchestration setup metrics and authnz_debug optional router blocks from eager try/except handling to lazy ImportedRouterSpec registration with precise optional-missing skip semantics. This continues the Phase 2.2 minimal router cleanup after the privileges/tools pair merged, keeping the remaining tail routers aligned with the lazy optional router contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Minimal agent_orchestration setup metrics and authnz_debug routers are represented as lazy ImportedRouterSpec entries.
- [x] #2 Missing target optional modules are skipped while runtime import defects propagate during registration.
- [x] #3 Existing prefixes tags route keys and default stability behavior are preserved.
- [x] #4 Focused router contract tests cover lazy attr lookup missing optional imports and runtime error propagation for this router family.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add contract coverage for the remaining minimal tail router family. 2. Replace eager broad try/except imports with lazy ImportedRouterSpec entries. 3. Verify router, lifecycle, OpenAPI, Bandit, and diff gates. 4. Publish a dev PR for review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TDD RED selector failed as expected before the implementation: 3 new tail tests failed against eager imports while an existing content-tail test passed.

GREEN and broader validation passed: router groups 119 passed, lifecycle 54 passed, OpenAPI 69 passed, Bandit results 0, git diff --check clean.

No docs update was required for this internal router registration cleanup. No known blockers remain.

Opened PR https://github.com/rmusser01/tldw_server/pull/1347 against dev for this slice.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Converted the minimal agent_orchestration, setup, metrics, and authnz_debug router registrations to lazy ImportedRouterSpec entries. This preserves prefixes and tags while deferring import and router attribute resolution until registration, so missing optional modules remain skippable and runtime import defects still propagate. Validation: router group contracts 119 passed, main lifecycle contracts 54 passed, OpenAPI contracts 69 passed, Bandit on minimal.py found 0 results, and git diff --check was clean.
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
