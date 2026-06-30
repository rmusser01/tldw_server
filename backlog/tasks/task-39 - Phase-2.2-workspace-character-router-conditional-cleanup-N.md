---
id: TASK-39
title: Phase 2.2 workspace character router conditional cleanup N
status: Done
assignee: []
created_date: '2026-05-04 06:00'
updated_date: '2026-05-04 15:38'
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PR #1263 review follow-up: split the overlong access_count dict-comprehension in the workspace/character router laziness test. Verification rerun: focused workspace_character_router_attr_lookup 1 passed; full router_groups_contract 55 passed; git diff --check clean.

PR #1265 review follow-up: Qodo flagged that the workspace/character laziness test asserts import calls in exact router-definition order. Verified this is incidental to the lazy-import contract; change the assertion to be order-insensitive while preserving duplicate detection.
<!-- SECTION:NOTES:END -->

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
- PR #1263 review follow-up:
  - Split the overlong `access_count` dict-comprehension flagged by Qodo.
  - Reran focused `workspace_character_router_attr_lookup`: `1 passed, 54 deselected`.
  - Reran full router group contract: `55 passed`.
  - Reran `git diff --check`: passed with no output.
- PR #1265 review follow-up:
  - Replaced the ordered `import_calls` list assertion with a `Counter` comparison so duplicate or missing imports still fail without requiring router-definition order.
  - Reran focused `workspace_character_router_attr_lookup`: `1 passed, 54 deselected`.
  - Reran full router group contract: `55 passed`.
  - Reran `python -m ruff check --select I001 tldw_Server_API/tests/Services/test_router_groups_contract.py`: passed.
  - Reran Bandit on touched source: `python -m bandit -r tldw_Server_API/app/api/v1/router_groups/content.py -f json -o /tmp/bandit_pr1265_content_review.json`, `result_count: 0`.
  - Reran `git diff --check`: passed with no output.
  - Rebasing onto current `origin/dev` skipped the already-applied original router-import commit and left the PR with only review follow-up commits.
  - After rebase, reran focused `workspace_character_router_attr_lookup`: `1 passed, 56 deselected`.
  - After rebase, reran full router group contract: `57 passed`.
  - After rebase, reran main router contract: `6 passed`.
  - After rebase, reran OpenAPI contracts: `69 passed`.
  - After rebase, reran `python -m ruff check --select I001 tldw_Server_API/tests/Services/test_router_groups_contract.py`: passed.
  - After rebase, reran Bandit on touched source: `python -m bandit -r tldw_Server_API/app/api/v1/router_groups/content.py -f json -o /tmp/bandit_pr1265_content_review_rebased.json`, `result_count: 0`.
  - After rebase, reran `git diff --check origin/dev..HEAD`: passed with no output.
<!-- SECTION:VERIFICATION:END -->

## Summary

<!-- SECTION:SUMMARY:BEGIN -->
Workspace and character-family content routers now use lazy `ImportedRouterSpec`
definitions. This preserves the existing route metadata while deferring endpoint
module imports and `router` attribute lookup until router registration/resolution.
<!-- SECTION:SUMMARY:END -->
