---
id: TASK-24
title: Phase 2.2 workflow router conditional cleanup F
status: Done
assignee: []
created_date: '2026-05-04 01:52'
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
Move only the covered workflow router specs onto the shared lazy ImportedRouterSpec helper. Preserve public route metadata, route ordering, and the explicit pytest runtime force-include behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workflows, chat workflows, and scheduler workflows RouterSpec metadata is preserved.
- [x] #2 Router attribute lookup for moved workflow specs is lazy through RouterSpec resolution.
- [x] #3 Full router group contract and adjacent main/OpenAPI contract tests pass.
- [x] #4 Bandit touched-source scope and git diff --check pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused regression test proving workflow router attribute lookup is deferred until RouterSpec resolution.
2. Run the focused selection red on current code.
3. Replace only the workflow import block with ImportedRouterSpec plus append_imported_router_spec while preserving explicit pytest route-key behavior.
4. Run focused and full router contract tests plus adjacent main/OpenAPI contract tests.
5. Run Bandit on touched router source and git diff --check.
6. Commit the narrow tranche and update the issue.
<!-- SECTION:PLAN:END -->

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
- Red check: `python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -k "workflow_router_attr_lookup" -q` failed before implementation because workflow router attributes were resolved during spec construction.
- Green focused check: `python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -k "workflow_router_attr_lookup" -q` passed with `1 passed`.
- Green full/adjacent checks: `python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -q` passed with `47 passed`; `python -m pytest tldw_Server_API/tests/Services/test_main_router_contract.py -q` passed with `6 passed`; `python -m pytest tldw_Server_API/tests/Services/test_openapi_contracts.py -q` passed with `69 passed`.
- Security and hygiene: `python -m bandit -r tldw_Server_API/app/api/v1/router_groups/content.py -f json -o /tmp/bandit_phase2_2_workflow_router_conditionals_f.json` reported `0 results` and `0 errors`; `git diff --check` passed.
- Documentation: no user-facing docs required for this internal router registration refactor.
- Known skips or blockers: workflow route keys still intentionally become empty in explicit pytest runtime to preserve force-inclusion behavior for unit coverage.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Moved the covered `workflows`, `chat-workflows`, and `scheduler` registrations to the shared lazy `ImportedRouterSpec` helper while preserving route metadata, default-stable flags, ordering, and explicit pytest force-include behavior. Added regression coverage proving workflow router attributes are not resolved until the selected `RouterSpec` objects are used, then verified the full router group contract, adjacent main/OpenAPI contract tests, Bandit touched-source scope, and whitespace checks.
<!-- SECTION:FINAL_SUMMARY:END -->
