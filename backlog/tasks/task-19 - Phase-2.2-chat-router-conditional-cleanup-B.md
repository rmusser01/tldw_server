---
id: TASK-19
title: Phase 2.2 chat router conditional cleanup B
status: Done
assignee: []
created_date: '2026-05-03 22:03'
labels:
  - phase-2
  - router-groups
  - refactor
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
documentation:
  - Docs/superpowers/specs/2026-05-03-phase2-followup-stack-design.md
  - >-
    Docs/superpowers/plans/2026-05-03-phase2-followup-stack-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move only the covered core chat router specs onto the shared conditional router helper introduced by the Phase 2.2 A tranche. Preserve public route paths, tags, route keys, route enablement behavior, and minimal-app behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Core chat, chat_loop, and conversations_alias RouterSpec metadata is preserved.
- [x] #2 Router attribute lookup for the moved chat specs is lazy through RouterSpec resolution.
- [x] #3 Minimal test app chat behavior remains unchanged.
- [x] #4 Focused router contract tests, Bandit touched-source scope, and git diff --check pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused regression test proving core chat router attribute lookup is deferred until RouterSpec resolution. 2. Run the focused selection red on current code. 3. Replace the three covered open-coded core chat imports with ImportedRouterSpec plus append_imported_router_spec. 4. Run focused and full router contract tests. 5. Run Bandit on touched router source and git diff --check. 6. Commit the narrow tranche and update the task record.
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
- Red check: `python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -k "core_router_specs_defers_chat" -q` failed before the implementation because core chat router attributes were resolved while building specs.
- Green checks: `python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -q` passed with `38 passed`; `python -m pytest tldw_Server_API/tests/Services/test_main_router_contract.py -q` passed with `6 passed`; `python -m pytest tldw_Server_API/tests/Services/test_openapi_contracts.py -q` passed with `69 passed`.
- Security and hygiene: `python -m bandit -r tldw_Server_API/app/api/v1/router_groups/core.py -f json -o /tmp/bandit_phase2_2_chat_router_conditionals_b.json` reported `0 results` and `0 errors`; `git diff --check` passed.
- Documentation: no user-facing docs required for this internal router registration refactor.
- Known skips or blockers: none.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Moved the covered core chat, chat loop, and conversations alias router registrations from eager local imports to the shared lazy `ImportedRouterSpec` helper while preserving prefixes, tags, and route keys. Added a regression test that proves these chat router attributes are not resolved until the `RouterSpec.router` accessor is used, and verified the focused router contracts, minimal main-router contract, OpenAPI contracts, Bandit touched-source scope, and whitespace checks.
<!-- SECTION:FINAL_SUMMARY:END -->
