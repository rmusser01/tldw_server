---
id: TASK-83
title: Phase 2.2 minimal kanban router conditional cleanup AL
status: Done
assignee: []
created_date: '2026-05-05 18:55'
updated_date: '2026-05-05 19:30'
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
Continue issue #1116 Phase 2.2 after PR #1322. Convert the minimal-test Kanban optional router family in tldw_Server_API/app/api/v1/router_groups/minimal.py from one eager broad try/import RouterSpec block to ImportedRouterSpec-backed lazy router specs. Scope is limited to kanban_boards, kanban_lists, kanban_cards, kanban_labels, kanban_checklists, kanban_comments, kanban_search, kanban_links, and kanban_workflow. Preserve prefix /api/v1/kanban, tags, route_key=kanban, default_stable behavior, and existing minimal-test broad skip behavior for import-time failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Kanban minimal optional specs defer module import and router attribute lookup until registration
- [x] #2 Existing route metadata is preserved for every Kanban router in scope
- [x] #3 Focused router-group tests cover lazy behavior and broad import failure skipping with red-green verification
- [x] #4 Router-group contract tests and touched-source Bandit pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Red-green Kanban tranche: add focused router-group tests for lazy minimal Kanban module import/router attribute lookup and broad runtime import-failure skipping; then replace only the eager minimal Kanban try/import block with ImportedRouterSpec entries preserving prefix, tags, route_key, default_stable, and minimal skip context.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red verification: focused minimal Kanban router tests failed before production changes because Kanban endpoint modules were imported during spec construction and named lazy specs were absent.

Green verification: focused Kanban tests, full router group contract tests, main router contract tests, touched-source Bandit, and git diff --check all passed.

Reopened for PR #1324 review feedback. Gemini requested deduplicating Kanban router test data. Qodo flagged broad skip_exceptions=(Exception,); verified against origin/dev that the old minimal-test Kanban block already used one broad except Exception and this task intentionally preserved that compatibility boundary. Address Gemini with a test-only dedupe and respond to Qodo with the compatibility rationale unless the local tests expose a safer narrow behavior.

Review follow-up result: deduplicated Kanban test data through shared module suffix/path data, preserving the intentional minimal-test broad skip behavior. Qodo broad-skip thread will be answered with the base-code/task-contract rationale instead of changing PR semantics in this tranche.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #1324 review follow-up by deduplicating the Kanban router contract test data from one shared module/path table and deriving module names, expected names, selected modules, and debug expectations from it. Preserved skip_exceptions=(Exception,) because origin/dev used a broad except Exception for the minimal-test Kanban block and TASK-83 explicitly scoped this tranche to preserve that compatibility behavior; the broad-skip cleanup should be handled as a separate behavior-changing tranche if desired. Verification: focused Kanban router contract tests passed, full test_router_groups_contract.py passed, test_main_router_contract.py passed, Bandit on minimal.py reported 0 results/0 errors, and git diff --check passed.
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
