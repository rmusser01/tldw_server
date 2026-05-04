---
id: TASK-39
title: Phase 2.2 workspace character router conditional cleanup N
status: Done
assignee: []
created_date: '2026-05-04 06:00'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue #1116 Phase 2.2 by deferring workspace and character-family content router imports from iter_content_router_specs while preserving existing route metadata and optional-import behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workspace, character chat sessions, character memory, characters, and character messages router specs defer router attribute lookup until registration/resolution.
- [x] #2 Existing prefix, tags, route_key, and default_stable behavior for workspace and character-family routes remain unchanged.
- [x] #3 Focused/full router contract tests, main router/OpenAPI contracts, Bandit touched source scan, and git diff hygiene are run before commit.
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

## Verification

<!-- SECTION:VERIFICATION:BEGIN -->
- Red: `python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -k workspace_character_router_attr_lookup -q`
  - Failed before implementation because all five fake router attributes were read during `iter_content_router_specs()`.
- Green: `python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -k workspace_character_router_attr_lookup -q`
  - `1 passed, 54 deselected`
- Full router group contract: `python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -q`
  - `55 passed`
- Main router contract: `python -m pytest tldw_Server_API/tests/Services/test_main_router_contract.py -q`
  - `6 passed`
- OpenAPI contracts: `python -m pytest tldw_Server_API/tests/Services/test_openapi_contracts.py -q`
  - `69 passed`
- Bandit: `python -m bandit -r tldw_Server_API/app/api/v1/router_groups/content.py -f json -o /tmp/bandit_phase2_2_character_workspace_router_conditionals_n.json`
  - `result_count: 0`
- Diff hygiene: `git diff --check`
  - Passed with no output.
<!-- SECTION:VERIFICATION:END -->

## Summary

<!-- SECTION:SUMMARY:BEGIN -->
Workspace and character-family content routers now use lazy `ImportedRouterSpec`
definitions. This preserves the existing route metadata while deferring endpoint
module imports and `router` attribute lookup until router registration/resolution.
<!-- SECTION:SUMMARY:END -->
