---
id: TASK-73
title: Phase 2.2 minimal experience router conditional cleanup AG
status: Done
assignee: []
created_date: '2026-05-05 14:45'
updated_date: '2026-05-05 14:48'
labels:
  - phase2.2
  - router-cleanup
  - issue-1116
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1305'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue issue #1116 Phase 2.2 by converting the next small set of minimal-test optional single-router experience/support registrations from eager try/import RouterSpec blocks to ImportedRouterSpec-backed lazy router specs. Scope is limited to sharing, personalization, and companion in tldw_Server_API/app/api/v1/router_groups/minimal.py. Preserve existing prefixes, tags, route keys, default_stable values, and current optional-missing skip behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Selected minimal optional experience/support router specs defer module import and router attribute lookup until registration
- [x] #2 Existing route metadata is preserved for sharing personalization and companion
- [x] #3 Focused router-group test covers lazy behavior with red/green verification
- [x] #4 Router-group contract tests and touched-source Bandit pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused lazy-import contract test for sharing, personalization, and companion.
2. Run the focused test red against the current eager try/import blocks.
3. Convert only those three blocks to ImportedRouterSpec while preserving prefix, tags, empty route_key, default_stable=True, and minimal skip context.
4. Rerun the focused test, full router_groups contract file, Bandit on minimal.py, git diff --check, and status check before commit/PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started after PR #1305 merge was verified at merge commit 67d13e1bc6621c3cd37e387eeea06dda657b07a5. Worktree: /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/phase2-2-minimal-experience-router-conditionals-ag. Branch: codex/phase2-2-minimal-experience-router-conditionals-ag.

Implemented the minimal experience/support router tranche. Added a red/green focused contract test proving sharing, personalization, and companion defer router attribute lookup until ImportedRouterSpec resolution. Converted only those three eager try/import blocks while preserving prefixes, tags, empty route_key, default_stable=True, and minimal skip context.

Verification: focused test failed red before implementation because router attributes were accessed during iter_minimal_optional_router_specs; focused test passed after implementation; full test_router_groups_contract.py passed with 76 tests; Bandit on minimal.py reported 0 results and 0 errors; git diff --check was clean.

Additional router gates passed after task finalization: test_main_router_contract.py passed with 6 tests; test_openapi_contracts.py passed with 69 tests.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Converted minimal optional sharing, personalization, and companion router registrations to ImportedRouterSpec-backed lazy specs with focused contract coverage. Preserved existing route metadata and optional skip context. Verification covered red/green focused test, full router_groups contract suite, Bandit, and diff check.

Additional verification: main router contract suite and OpenAPI contract suite passed.
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
