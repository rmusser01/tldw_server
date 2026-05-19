---
id: TASK-21
title: Phase 2.2 admin router conditional cleanup C
status: Done
assignee: []
created_date: '2026-05-03 23:07'
updated_date: '2026-05-03 23:45'
labels:
  - phase-2
  - router-groups
  - refactor
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move only covered single-router admin specs onto the shared lazy ImportedRouterSpec helper. Preserve public route prefixes, tags, route keys, default stability, skip behavior, and minimal app behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Selected admin RouterSpec metadata is preserved.
- [x] #2 Router attribute lookup for moved admin specs is lazy through RouterSpec resolution.
- [x] #3 Full router group contract and adjacent main/OpenAPI contract tests pass.
- [x] #4 Bandit touched-source scope and git diff --check pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused regression test proving selected admin router attribute lookup is deferred until RouterSpec resolution. 2. Run the focused selection red on current code. 3. Replace only the covered single-router admin imports with ImportedRouterSpec plus append_imported_router_spec. 4. Run focused and full router contract tests plus adjacent main/OpenAPI contract tests. 5. Run Bandit on touched router source and git diff --check. 6. Commit the narrow tranche and update the task record.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-05-03 PR #1246 review follow-up: hardened test_iter_admin_router_specs_defers_selected_router_attr_lookup so sandbox must be stubbed in sys.modules before iter_admin_router_specs() runs. Added an importlib guard that fails if the test would import the real sandbox endpoint module during spec registration.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Moved the covered single-router admin registrations to the shared lazy `ImportedRouterSpec` helper while preserving prefixes, tags, route keys, default stability, order, and per-router skip logging. Added regression coverage proving selected admin router attributes are not resolved until `RouterSpec.router` is used, then verified the full router group contract, adjacent main/OpenAPI contract tests, Bandit touched-source scope, and whitespace checks.
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

## Notes

<!-- SECTION:NOTES:BEGIN -->
- Red check: `python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -k "admin_router_specs_defers_selected" -q` failed before implementation because selected admin router attributes were resolved during spec construction.
- Green focused check: `python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -k "admin_router_specs_defers_selected or admin_router_specs_populates_expected" -q` passed with `2 passed`.
- Green full/adjacent checks: `python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -q` passed with `42 passed`; `python -m pytest tldw_Server_API/tests/Services/test_main_router_contract.py -q` passed with `6 passed`; `python -m pytest tldw_Server_API/tests/Services/test_openapi_contracts.py -q` passed with `69 passed`.
- Security and hygiene: `python -m bandit -r tldw_Server_API/app/api/v1/router_groups/admin.py -f json -o /tmp/bandit_phase2_2_admin_router_conditionals_c.json` reported `0 results` and `0 errors`; `git diff --check` passed.
- Documentation: no user-facing docs required for this internal router registration refactor.
- Known skips or blockers: none.
<!-- SECTION:NOTES:END -->
