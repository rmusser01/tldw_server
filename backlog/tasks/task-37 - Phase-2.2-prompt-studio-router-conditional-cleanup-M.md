---
id: TASK-37
title: Phase 2.2 prompt studio router conditional cleanup M
status: Done
assignee: []
created_date: '2026-05-04 05:40'
updated_date: '2026-05-04 05:47'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue #1116 Phase 2.2 by deferring the remaining Prompt Studio content router imports from iter_content_router_specs while preserving existing route metadata and optional-import behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Prompt Studio projects, prompts, test cases, optimization, status, evaluations, and websocket router specs defer router attribute lookup until registration/resolution.
- [x] #2 Existing prefix, tags, route_key, and default_stable behavior for Prompt Studio routes remain unchanged.
- [x] #3 Focused/full router contract tests, main router/OpenAPI contracts, Bandit touched source scan, and git diff hygiene are run before commit.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red/green focused test: python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -k "prompt_studio_router_attr_lookup" -q failed before implementation on eager router attr lookup, then passed after converting Prompt Studio specs to lazy imported specs.

Verification: router group contract 54 passed; main router contract 6 passed; OpenAPI contracts 69 passed; Bandit source scan content.py returned 0 results/0 errors; git diff --check clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Converted the Prompt Studio content router block to lazy ImportedRouterSpec entries and added covered contract coverage proving router attribute lookup is deferred until resolution while preserving existing route metadata.

PR: https://github.com/rmusser01/tldw_server/pull/1262
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
