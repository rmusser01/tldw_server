---
id: TASK-19.1
title: 'Address PR #1245 review comments'
status: Done
assignee: []
created_date: '2026-05-03 22:45'
updated_date: '2026-05-03 22:50'
labels:
  - phase-2
  - router-groups
  - review-fix
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1245'
parent_task_id: TASK-19
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow up on PR #1245 review feedback for the Phase 2.2 chat router conditional cleanup. Verify and fix the chat router import crash regression, ambiguous skip logging for lazy router resolution failures, test helper fake-module leakage risk, and overlong test asserts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Unexpected import-time exceptions from covered core chat router modules are caught and skip only the affected chat router spec with a router-specific log name.
- [x] #2 RouterSpec resolution failure logs use a router-specific name when available and preserve the existing route_key fallback.
- [x] #3 Router contract test helpers do not mutate pre-imported real endpoint modules when installing fake routers.
- [x] #4 Focused router contract tests, Bandit touched-source scope, and git diff --check pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red check: python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -k 'fake_router_module_does_not_mutate or crashing_chat_import or logs_spec_name' -q failed on the expected fake-module mutation, unhandled chat_loop RuntimeError, and missing RouterSpec name support. Green checks: focused review selection passed with 3 passed; full router contract passed with 41 passed; main router plus OpenAPI contracts passed with 75 passed. Security and hygiene: Bandit touched source scope reported 0 results and 0 errors in /tmp/bandit_pr1245_review_fixes.json; git diff --check passed; newly-added line-length scan found no added lines over 100 characters.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #1245 review feedback by preserving catch-and-skip behavior for covered core chat router import crashes, carrying ImportedRouterSpec.log_name into RouterSpec diagnostics, using that name for lazy router resolution skip logs, preventing fake-router test helpers from mutating real pre-imported modules, and wrapping the reviewed overlong assertions.
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
