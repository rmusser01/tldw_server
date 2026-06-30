---
id: TASK-80
title: Phase 2.2 minimal workflow router conditional cleanup AJ
status: Done
assignee: []
created_date: '2026-05-05 17:48'
labels:
  - phase2.2
  - router-cleanup
  - issue-1116
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1313'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue issue #1116 Phase 2.2 after PR #1313. Convert the minimal-test workflow router registrations from an eager broad try/import RouterSpec block to ImportedRouterSpec-backed lazy router specs. Scope is limited to workflows, chat_workflows, and scheduler_workflows in tldw_Server_API/app/api/v1/router_groups/minimal.py. Preserve prefixes, tags, default route_key behavior, and the existing minimal-test broad skip behavior for import-time failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workflow, chat workflow, and scheduler workflow minimal optional specs defer module import and router attribute lookup until registration
- [x] #2 Existing workflow route metadata is preserved
- [x] #3 Focused router-group tests cover lazy behavior and broad import failure skipping with red-green verification
- [x] #4 Router-group contract tests and touched-source Bandit pass
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started after PR #1313 merge was verified at merge commit 4db3d76fca5790a883f35db8ad93d5890bb61ed1. Worktree: /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/phase2-2-minimal-workflow-router-conditionals-aj. Branch: codex/phase2-2-minimal-workflow-router-conditionals-aj.

Added focused red-green coverage for the minimal workflow router group. The focused tests failed red against the eager broad try/import block: spec construction imported workflows, chat_workflows, and scheduler_workflows through builtins.__import__, and the old eager RouterSpec entries did not expose named workflow specs for registration-time failure testing.

Converted only workflows, chat_workflows, and scheduler_workflows to ImportedRouterSpec-backed lazy specs while preserving empty prefixes, tags, default route_key behavior, default_stable=True, and broad skip_exceptions=(Exception,) behavior in minimal test mode. Verification passed: focused selection workflow_attr_lookup or workflow_runtime_import_failures passed with 2 tests; full test_router_groups_contract.py passed with 83 tests; test_main_router_contract.py passed with 6 tests; Bandit on minimal.py reported 0 results and 0 errors; git diff --check was clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Converted the minimal workflow router registrations to lazy ImportedRouterSpec entries. The tranche defers workflow endpoint imports and router attribute lookup until registration while preserving the previous broad minimal-mode skip behavior for import-time failures. Added focused regression tests for lazy resolution and RuntimeError import-failure skipping.
<!-- SECTION:FINAL_SUMMARY:END -->
