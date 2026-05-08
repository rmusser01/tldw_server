---
id: TASK-120
title: Phase 2.2 minimal setup router conditional cleanup BH
status: Done
assignee: []
created_date: '2026-05-08 02:27'
updated_date: '2026-05-08 02:35'
labels:
  - phase-2.2
  - router-groups
  - minimal
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1371'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue Phase 2.2 router cleanup after PR #1370 by removing the dead direct minimal-mode setup endpoint import from app/main.py. The setup router is already represented by the minimal router group; main should rely on grouped registration instead of probing the endpoint directly.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 main.py no longer imports endpoints.setup directly for MINIMAL_TEST_APP
- [x] #2 minimal router group continues to expose the setup router spec with the existing /api/v1 prefix and setup tag
- [x] #3 minimal app/router contract tests validate the grouped setup path without the dead main import
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a RED source/contract assertion showing the remaining direct setup endpoint import in main.py is not allowed. 2. Remove the unused minimal-mode setup_router import/exception block from main.py. 3. Run focused router/main contract tests, Bandit on touched production file and test file, and git diff --check before committing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RED: added a main source assertion for the remaining direct endpoints.setup import and verified it failed in test_main_source_delegates_minimal_optional_llm_routers_to_group. GREEN: removed the unused MINIMAL_TEST_APP setup_router import block from app/main.py and tightened the main import contract so direct setup imports raise AssertionError. Validation: focused router contract selector passed with 2 passed; focused main router selector passed with 1 passed; full router group contract suite passed with 170 passed; main router contracts passed with 6 passed; OpenAPI contracts passed with 69 passed; git diff --check passed. Bandit on app/main.py reported 0 results and 0 errors. Broader touched-scope Bandit reported pre-existing test-file B404/B603 findings on subprocess import/call lines outside this patch's changed lines; no new security findings introduced.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the dead direct setup endpoint import from the MINIMAL_TEST_APP branch in app/main.py so setup registration stays delegated to router groups. Added source/import contract coverage that prevents reintroducing the direct setup probe while preserving the existing minimal setup router spec metadata.
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
