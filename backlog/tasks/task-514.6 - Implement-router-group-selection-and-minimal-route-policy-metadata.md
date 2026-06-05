---
id: TASK-514.6
title: Implement router group selection and minimal route policy metadata
status: Done
parent_task_id: TASK-514
documentation:
- tldw_Server_API/tests/Services/test_router_groups_contract.py
modified_files:
- tldw_Server_API/app/api/v1/router_groups/selection.py
- tldw_Server_API/app/api/v1/router_groups/minimal.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the currently failing router_groups contract tests by adding the missing router spec selection helper and preserving route policy metadata for minimal router specs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Minimal-test router specs preserve canonical route policy metadata.
- [x] Router spec selection helper supports explicit metadata overrides without importing endpoint modules.
- [x] Previously failing router_groups contract tests pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added a reusable router spec selection helper with explicit per-field overrides, then changed the minimal-test router group to select the always-included routers from canonical core/content specs. The minimal app now preserves route_key/default_stable metadata while overriding skip_exceptions to the minimal required value.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the router group contract failures by preserving route policy metadata for minimal-test router specs and adding the missing select_router_specs_by_name helper. Verification: focused failing router slice passed (7 passed), full router contract suite passed (173 passed), broader notes+router backend slice passed (348 passed), and Bandit reported zero findings for the touched router group modules.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant: not needed for internal router helper.
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented: none.
<!-- DOD:END -->
