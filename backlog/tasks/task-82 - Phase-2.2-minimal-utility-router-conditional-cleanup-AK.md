---
id: TASK-82
title: Phase 2.2 minimal utility router conditional cleanup AK
status: Done
assignee: []
created_date: '2026-05-05 18:15'
updated_date: '2026-05-05 18:21'
labels:
  - phase2.2
  - router-cleanup
  - issue-1116
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1320'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue issue #1116 Phase 2.2 after PR #1320. Convert the next small minimal-test optional utility router registrations from eager broad try/import RouterSpec blocks to ImportedRouterSpec-backed lazy router specs. Scope is limited to web_clipper, skills, translate, and slides in tldw_Server_API/app/api/v1/router_groups/minimal.py. Preserve prefixes, tags, default route_key/default_stable behavior, and existing minimal-test broad skip behavior for import-time failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Web clipper, skills, translate, and slides minimal optional specs defer module import and router attribute lookup until registration
- [x] #2 Existing route metadata is preserved for web clipper, skills, translate, and slides
- [x] #3 Focused router-group tests cover lazy behavior and broad import failure skipping with red-green verification
- [x] #4 Router-group contract tests and touched-source Bandit pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Red-green utility tranche: add focused router-group tests for lazy web_clipper, skills, translate, and slides resolution plus broad runtime import-failure skipping; then replace only those eager minimal optional imports with ImportedRouterSpec entries preserving metadata and skip semantics.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red verification: focused utility router tests failed before production changes because selected routers were imported during spec construction and named lazy specs were absent.

Green verification: focused utility tests, full router group contract tests, main router contract tests, touched-source Bandit, and git diff --check all passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Converted minimal-test web clipper, skills, translate, and slides optional router registrations to ImportedRouterSpec-backed lazy specs. This preserves prefixes, tags, default route_key/default_stable behavior, and broad minimal-test skip semantics with skip_exceptions=(Exception,) while deferring endpoint module imports until router registration. Added regression tests for lazy import/router attribute lookup and RuntimeError skip behavior. Verification: focused red/green utility tests, full test_router_groups_contract.py, test_main_router_contract.py, Bandit on minimal.py with 0 results/0 errors, and git diff --check.
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
