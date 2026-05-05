---
id: TASK-53
title: Phase 2.2 minimal LLM router conditional cleanup U
status: Done
assignee:
  - codex
created_date: '2026-05-05 01:56'
updated_date: '2026-05-05 02:03'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1279'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue #1116 Phase 2.2 with an independent minimal-test router tranche while PR #1279 settles. Scope is limited to the minimal optional Llama.cpp/messages try block in tldw_Server_API/app/api/v1/router_groups/minimal.py; broader minimal optional routers remain separate follow-up slices.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Minimal optional Llama.cpp and messages router specs defer module import and router/public_router attribute lookup until registration or resolution
- [x] #2 Existing prefixes and tags for minimal Llama.cpp/messages routers remain unchanged
- [x] #3 Focused red/green minimal-router laziness coverage, full router contract tests, main router/OpenAPI contracts, Bandit touched source scan, and git diff hygiene are run before commit
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused contract coverage showing iter_minimal_optional_router_specs does not import or touch router/public_router attrs for Llama.cpp and messages during spec construction. 2. Run the focused test red against origin/dev behavior. 3. Replace only the minimal optional Llama.cpp/messages try block with ImportedRouterSpec entries. 4. Re-run focused/full router contracts, main router contracts, OpenAPI contracts, Bandit on minimal.py, and diff hygiene. 5. Commit and decide whether to push/open a PR after checking #1279 state.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification: focused red failed on origin/dev with eager Llama.cpp/messages attr lookup; focused green passed after patch (1 passed, 61 deselected); full router groups passed (62 passed); main router contract passed (6 passed); OpenAPI contracts passed (69 passed); Bandit on minimal.py reported 0 results and 0 errors; git diff --check clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Converted only the minimal optional Llama.cpp/messages grouped eager import block to four ImportedRouterSpec entries, preserving prefixes/tags while deferring module imports and router/public_router attr lookup until registration/resolution.
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
