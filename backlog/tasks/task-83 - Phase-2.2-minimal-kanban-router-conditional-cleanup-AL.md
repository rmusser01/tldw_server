---
id: TASK-83
title: Phase 2.2 minimal kanban router conditional cleanup AL
status: Done
assignee: []
created_date: '2026-05-05 18:55'
updated_date: '2026-05-05 19:43'
labels:
  - phase2.2
  - router-cleanup
  - issue-1116
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1322'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue issue #1116 Phase 2.2 after PR #1322. Convert the minimal-test Kanban optional router family in tldw_Server_API/app/api/v1/router_groups/minimal.py from one eager try/import RouterSpec block to ImportedRouterSpec-backed lazy router specs. Scope is limited to kanban_boards, kanban_lists, kanban_cards, kanban_labels, kanban_checklists, kanban_comments, kanban_search, kanban_links, and kanban_workflow. Preserve prefix /api/v1/kanban, tags, route_key=kanban, and default_stable behavior. After PR #1324 review, Kanban lazy resolution now skips only missing optional router failures via ImportError/AttributeError and propagates runtime defects.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Kanban minimal optional specs defer module import and router attribute lookup until registration
- [x] #2 Existing route metadata is preserved for every Kanban router in scope
- [x] #3 Router-group contract tests and touched-source Bandit pass
- [x] #4 Kanban minimal optional specs skip only missing optional router import or attr lookup failures and propagate runtime defects
- [x] #5 Focused router-group tests cover lazy behavior, missing optional import skipping, and runtime import defect propagation with red-green verification
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Review fix plan:
1. Add focused RED coverage showing Kanban lazy import RuntimeError from module import is not swallowed by register_router_specs.
2. Change only the Kanban ImportedRouterSpec skip_exceptions tuple from broad Exception to missing optional router failures.
3. Update existing Kanban metadata/runtime tests and Backlog notes to reflect narrow skip semantics.
4. Run focused Kanban tests, full router group/main router contracts, Bandit on minimal.py, and git diff --check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red verification: focused minimal Kanban router tests failed before production changes because Kanban endpoint modules were imported during spec construction and named lazy specs were absent.

PR #1324 review follow-up: verified Kanban minimal specs used skip_exceptions=(Exception,), which register_router_specs would treat as skippable for any lazy import or attr resolution exception. Added RED coverage proving RuntimeError import defects were swallowed; the focused run failed with DID NOT RAISE RuntimeError.

Green verification: narrowed Kanban skip_exceptions to (ImportError, AttributeError). Focused Kanban tests passed with 3 passed; full router group contract tests passed with 88 passed; main router contract tests passed with 6 passed; Bandit on minimal.py reported 0 results and 0 errors; git diff --check was clean. A broad Ruff check on touched files was not used as a completion gate because it reports unrelated pre-existing style findings across older test sections.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Converted the minimal-test Kanban optional router family to ImportedRouterSpec-backed lazy specs and addressed the PR #1324 review finding by narrowing Kanban skip_exceptions from broad Exception to ImportError/AttributeError. This keeps missing optional router modules or router attributes skippable in the minimal test app while allowing runtime defects during lazy import/attr resolution to surface.

Updated focused contract coverage to assert the narrower skip tuple, keep missing-import skip behavior, and prove RuntimeError import failures propagate through register_router_specs. Verification: focused Kanban router tests passed with 3 passed after first failing red; full test_router_groups_contract.py passed with 88 passed; test_main_router_contract.py passed with 6 passed; Bandit on minimal.py reported 0 results/0 errors; git diff --check passed. Ruff full touched-file check still reports unrelated pre-existing style findings in older test sections, so no broad lint cleanup was included in this review fix.
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
