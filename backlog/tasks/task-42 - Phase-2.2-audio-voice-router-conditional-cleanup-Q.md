---
id: TASK-42
title: Phase 2.2 audio voice router conditional cleanup Q
status: Done
assignee: []
created_date: '2026-05-04 19:25'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue #1116 Phase 2.2 by deferring audiobook and voice-assistant content
router imports from `iter_content_router_specs()` while preserving route
metadata and optional-import behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Audiobooks, voice assistant REST, and voice assistant WebSocket router specs defer module import and router attribute lookup until registration/resolution.
- [x] #2 Existing prefix, tags, route_key, default_stable, and `ws_router` attribute behavior remain unchanged.
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
- Red: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -k audio_voice_router_attr_lookup -q`
  - Failed before implementation because audiobook `router`, voice `router`, and voice `ws_router` were read during `iter_content_router_specs()`.
- Green: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -k audio_voice_router_attr_lookup -q`
  - `1 passed, 57 deselected`
- Full router group contract: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -q`
  - `58 passed`
- Main router contract: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Services/test_main_router_contract.py -q`
  - `6 passed`
- OpenAPI contracts: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Services/test_openapi_contracts.py -q`
  - `69 passed`
- Bandit: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/router_groups/content.py -f json -o /tmp/bandit_phase2_2_audio_voice_router_conditionals_q.json`
  - `result_count: 0`
- Diff hygiene: `git diff --check`
  - Passed with no output.
<!-- SECTION:VERIFICATION:END -->

## Summary

<!-- SECTION:SUMMARY:BEGIN -->
Audiobook and voice-assistant content routers now use lazy `ImportedRouterSpec`
definitions, including the voice WebSocket router's `ws_router` attribute. Route
metadata remains unchanged while imports move to registration/resolution time.
<!-- SECTION:SUMMARY:END -->
