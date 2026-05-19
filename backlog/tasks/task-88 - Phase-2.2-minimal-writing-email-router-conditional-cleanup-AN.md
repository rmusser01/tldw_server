---
id: TASK-88
title: Phase 2.2 minimal writing/email router conditional cleanup AN
status: Done
assignee: []
created_date: '2026-05-05 21:58'
updated_date: '2026-05-05 22:11'
labels:
  - phase2.2
  - router-cleanup
  - issue-1116
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1327'
  - 'https://github.com/rmusser01/tldw_server/pull/1329'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue issue #1116 Phase 2.2 after PR #1327. Convert the next small minimal-test optional router family in tldw_Server_API/app/api/v1/router_groups/minimal.py from eager try/import RouterSpec blocks to ImportedRouterSpec-backed lazy router specs. Scope is limited to writing, writing_manuscripts, and email. Preserve prefixes, tags, route_key/default_stable behavior, and minimal skip context while using narrow missing-optional skip semantics: ImportError/AttributeError are skippable, runtime import defects propagate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Writing, manuscripts, and email minimal optional specs defer module import and router attribute lookup until registration
- [x] #2 Existing route metadata is preserved for every router in scope
- [x] #3 Focused router-group tests cover lazy behavior, missing optional import skipping, and runtime import defect propagation with red-green verification
- [x] #4 Router-group/main-router/OpenAPI contract tests and touched-source Bandit pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused RED coverage for minimal writing/email router lazy resolution: spec construction does not import writing, writing_manuscripts, or email, metadata is preserved, and router attr lookup happens only during registration. 2. Add RED coverage for missing optional ImportError skip behavior and RuntimeError propagation through register_router_specs. 3. Convert only writing, writing_manuscripts, and email to ImportedRouterSpec entries in minimal.py using skip_exceptions=(ImportError, AttributeError). 4. Run focused tests, full router group/main/OpenAPI contract tests, Bandit on minimal.py, and git diff --check before commit/PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started after PR #1327 merge was verified on origin/dev as 62002bcd8 in the git log. Worktree: local feature worktree for phase2-2-minimal-writing-router-conditionals-an. Branch: codex/phase2-2-minimal-writing-router-conditionals-an. Baseline router group contract tests passed with 91 passed before edits.

RED verification: focused writing/email tests failed before production changes because writing, writing_manuscripts, and email were imported during spec construction and no named lazy specs existed for registration-time skip/propagation assertions.

GREEN verification: converted writing, writing_manuscripts, and email to ImportedRouterSpec-backed lazy specs with skip_exceptions=(ImportError, AttributeError). Focused writing/email tests passed with 3 passed; full test_router_groups_contract.py passed with 94 passed; test_main_router_contract.py passed with 6 passed; test_openapi_contracts.py passed with 69 passed; Bandit on minimal.py reported 0 results and 0 errors; git diff --check was clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Converted the minimal-test writing/email router family, limited to writing, writing_manuscripts, and email, from eager try/import RouterSpec blocks to ImportedRouterSpec-backed lazy specs. The change preserves prefixes, tags, default route_key/default_stable behavior, and minimal skip context while using narrow missing-optional skip semantics: ImportError and AttributeError are skipped, but runtime import defects propagate through register_router_specs.

Added focused contract coverage for lazy module import/router attribute lookup, missing optional import skipping, and RuntimeError propagation. Verification: focused writing/email tests passed with 3 passed after a failing RED run; full router group contract tests passed with 94 passed; main router contract tests passed with 6 passed; OpenAPI contract tests passed with 69 passed; Bandit on minimal.py reported 0 results and 0 errors; git diff --check passed.

PR opened: https://github.com/rmusser01/tldw_server/pull/1329
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
