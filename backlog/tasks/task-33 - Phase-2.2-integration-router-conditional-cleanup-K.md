---
id: TASK-33
title: Phase 2.2 integration router conditional cleanup K
status: Done
assignee: []
created_date: '2026-05-04 04:05'
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
Move only the covered integration-style content router specs onto the shared lazy ImportedRouterSpec helper. Preserve public route metadata and route ordering for Slack, Discord, Telegram, and Meetings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Slack, Discord, Telegram, and Meetings RouterSpec metadata is preserved.
- [x] #2 Router module import and router attribute lookup for moved integration specs is lazy through RouterSpec resolution.
- [x] #3 Full router group contract and adjacent main/OpenAPI contract tests pass.
- [x] #4 Bandit touched-source scope and git diff --check pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused regression test proving integration router module import and attribute lookup are deferred until RouterSpec resolution.
2. Run the focused selection red on current code.
3. Replace only the Slack, Discord, Telegram, and Meetings import blocks with ImportedRouterSpec plus append_imported_router_spec while preserving metadata.
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
- Red check: `python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -k "integration_router_attr_lookup" -q` failed before implementation because Slack, Discord, Telegram, and Meetings router attributes were resolved during spec construction.
- Green focused check: `python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -k "integration_router_attr_lookup" -q` passed with `1 passed`.
- Green full/adjacent checks: `python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -q` passed with `51 passed`; `python -m pytest tldw_Server_API/tests/Services/test_main_router_contract.py -q` passed with `6 passed`; `python -m pytest tldw_Server_API/tests/Services/test_openapi_contracts.py -q` passed with `69 passed`.
- Security and hygiene: `python -m bandit -r tldw_Server_API/app/api/v1/router_groups/content.py -f json -o /tmp/bandit_phase2_2_integration_router_conditionals_k.json` reported `0 results` and `0 errors`; `git diff --check` passed.
- Rebased onto `fe21a86f61` after #1256 merged; kept both the output-router lazy test from #1256 and this tranche's integration-router lazy test. Rebased verification: `python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -q` passed with `52 passed`; `python -m pytest tldw_Server_API/tests/Services/test_main_router_contract.py -q` passed with `6 passed`; `python -m pytest tldw_Server_API/tests/Services/test_openapi_contracts.py -q` passed with `69 passed`; `python -m bandit -r tldw_Server_API/app/api/v1/router_groups/content.py -f json -o /tmp/bandit_phase2_2_integration_router_conditionals_k_rebased.json` reported `0 results` and `0 errors`; `git diff --check HEAD` passed.
- Documentation: no user-facing docs required for this internal router registration refactor.
- Known skips or blockers: none.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Moved the covered Slack, Discord, Telegram, and Meetings registrations to the shared lazy ImportedRouterSpec helper while preserving route prefixes, tags, route keys, default-stable policy, and ordering. Added regression coverage proving selected integration router modules and router attributes are not resolved until the selected RouterSpec objects are used, then verified the full router group contract, adjacent main/OpenAPI contract tests, Bandit touched-source scope, and whitespace checks.
<!-- SECTION:FINAL_SUMMARY:END -->
