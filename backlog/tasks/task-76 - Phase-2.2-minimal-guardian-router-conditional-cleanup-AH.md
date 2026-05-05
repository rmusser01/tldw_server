---
id: TASK-76
title: Phase 2.2 minimal guardian router conditional cleanup AH
status: Done
assignee: []
created_date: '2026-05-05 16:17'
updated_date: '2026-05-05 16:21'
labels:
  - phase2.2
  - router-cleanup
  - issue-1116
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1307'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue issue #1116 Phase 2.2 after PR #1307. Convert the next small minimal-test optional guardian/safety registrations from eager try/import RouterSpec blocks to ImportedRouterSpec-backed lazy router specs. Scope is limited to family_wizard, guardian_controls, and self_monitoring in tldw_Server_API/app/api/v1/router_groups/minimal.py. Preserve prefixes, tags, route_key behavior, default_stable behavior, and the existing minimal-test broad skip behavior for import-time failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Guardian controls, family wizard, and self-monitoring minimal optional specs defer module import and router attribute lookup until registration
- [x] #2 Existing route metadata is preserved for guardian controls, family wizard, and self-monitoring
- [x] #3 Focused router-group tests cover lazy behavior and broad import failure skipping with red/green verification
- [x] #4 Router-group contract tests and touched-source Bandit pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused contract coverage for the minimal guardian/safety specs.
2. Run the focused selection red against the current eager try/import blocks.
3. Convert only family_wizard, guardian_controls, and self_monitoring to ImportedRouterSpec entries while preserving metadata and skip semantics.
4. Rerun focused tests, full router_groups contract tests, Bandit on minimal.py, git diff --check, and status before commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started after PR #1307 merge was verified at merge commit 6242560e22a4d9fce2b5d74679ce0f467943bd22. Worktree: /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/phase2-2-minimal-guardian-router-conditionals-ah. Branch: codex/phase2-2-minimal-guardian-router-conditionals-ah. Baseline full test_router_groups_contract.py passed with 77 tests before edits.

Implemented the guardian/safety minimal router tranche. Added red/green focused contract coverage proving guardian_controls, family_wizard, and self_monitoring defer module import/router attribute lookup until registration and preserve broad skip_exceptions=(Exception,) behavior for RuntimeError import failures. Converted only those three eager try/import RouterSpec blocks to ImportedRouterSpec-backed lazy specs while preserving prefixes, tags, route_key, and default_stable behavior. Verification: focused selection guardian_safety_attr_lookup or guardian_safety_runtime_import_failures failed red before the source change and passed after implementation; full test_router_groups_contract.py passed with 79 tests; test_main_router_contract.py passed with 6 tests; Bandit on minimal.py reported 0 results and 0 errors; git diff --check was clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Converted minimal optional guardian controls, family wizard, and self-monitoring router registrations to ImportedRouterSpec-backed lazy specs. The change keeps the prior minimal-test broad import-failure skip behavior by explicitly setting skip_exceptions=(Exception,) while deferring endpoint imports until registration. Added focused regression coverage for lazy router resolution and RuntimeError import-failure skipping. Verification covered red/green focused tests, full router_groups contract tests, main router contract tests, Bandit, and diff check.
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
