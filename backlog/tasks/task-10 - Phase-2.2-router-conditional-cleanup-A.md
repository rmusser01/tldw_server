---
id: TASK-10
title: Phase 2.2 router conditional cleanup A
status: Done
assignee: []
created_date: '2026-05-03 19:00'
updated_date: '2026-05-03 20:09'
labels:
  - phase-2
  - issue-1116
  - router-groups
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Conservative Phase 2.2 follow-up tranche for #1116. Characterize sandbox/ACP router conditional specs first, then extract or narrow repeated conditional router construction only where tests cover the result. Preserve route paths, prefixes, tags, route keys, lazy import behavior, and minimal test app behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sandbox and/or ACP router conditional specs preserve prefix, tags, route keys, default stability, and lazy import behavior.
- [x] #2 Repeated conditional import logic is extracted only where characterization tests cover the result.
- [x] #3 Minimal test app behavior remains unchanged.
- [x] #4 Focused router contract tests, OpenAPI route contract tests if applicable, Bandit, and git diff --check pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect existing router group conditional specs and tests for sandbox and ACP route families.
2. Add characterization tests for the selected smallest router family before refactoring.
3. Run focused router contract tests to prove baseline behavior.
4. Extract the smallest helper only if it removes repeated conditional RouterSpec construction without changing route metadata or eager imports; otherwise make the smallest local cleanup and stop.
5. Rerun focused router/main/OpenAPI tests as applicable, Bandit on touched router source, and git diff --check before commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification completed:
- RED: test_append_imported_router_spec_preserves_metadata failed before implementation with missing router_groups.conditional module.
- GREEN/focused: router_groups_contract -k "append_imported_router_spec or acp or ACP or sandbox" passed 3 selected tests.
- Full/adjacent: router_groups_contract 33 passed; main_router_contract 6 passed; openapi_phase4_contract 5 passed.
- Bandit router group scope: 0 findings in /tmp/bandit_phase2_2_router_conditionals_a.json.
- git diff --check passed.
- Review follow-up:
  - RED: `test_append_imported_router_spec_defers_router_attr_lookup_until_resolution` failed with eager router attribute access.
  - GREEN/focused: `test_router_groups_contract.py -k "append_imported_router_spec or populates_llm_specs"` passed 3 selected tests.
  - Full/adjacent: `test_router_groups_contract.py -q` passed 34 tests.
  - Bandit review-follow-up scope: 0 findings in `/tmp/bandit_pr1242_router_groups.json`.

PR opened: https://github.com/rmusser01/tldw_server/pull/1242
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a shared conditional router import helper and used it for the covered ACP and sandbox RouterSpec families in core, admin, and minimal router groups. The change preserves existing prefixes, tags, route keys, default stability, and skip logging while removing repeated optional import-to-RouterSpec blocks.

Review follow-up: `append_imported_router_spec()` now preserves that metadata while deferring router attribute lookup until `RouterSpec.resolve_router()`, and `minimal.py` now reuses the shared helper for the covered `llm_providers` and `mlx` optional imports instead of keeping a duplicate local helper.
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
