---
id: TASK-12025
title: Finalize RPG privilege catalog and route snapshot
status: Done
created_date: 2026-06-25 04:10
labels:
- rpg
- ttrpg
- backend
- authnz
- implementation
priority: high
references:
- TASK-12018
- TASK-12024
documentation:
- Docs/superpowers/plans/2026-06-25-rpg-campaign-session-runtime-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-06-25-rpg-campaign-session-runtime-implementation-plan.md
- tldw_Server_API/Config_Files/privilege_catalog.yaml
- tldw_Server_API/tests/fixtures/privilege_route_registry_snapshot.json
updated_date: 2026-06-25 04:12
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete the privilege-registration follow-up for the RPG REST slice: add the generic RPG token-scope catalog entry if needed, regenerate the privilege route registry snapshot, update the implementation plan, and verify the privilege snapshot/catalog tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Generic rpg token scope is represented in the privilege catalog or explicitly documented as unnecessary
- [x] #2 Privilege route registry snapshot includes RPG REST routes
- [x] #3 Privilege catalog/snapshot tests pass
- [x] #4 Bandit/diff checks are recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add the generic `rpg` token-scope catalog entry because RPG endpoints use `TokenScopeGuard("rpg", ...)`.
2. Regenerate the privilege route registry snapshot with the project helper.
3. Verify the live route registry snapshot, endpoint scope catalog sync, and RPG API tests.
4. Record the non-Python Bandit skip and commit the follow-up slice.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added the generic `rpg` token scope to the privilege catalog and regenerated `privilege_route_registry_snapshot.json`. The generated snapshot now includes RPG routes under the generic `rpg` scope and concrete endpoint scopes (`rpg.rules.read`, `rpg.campaigns.manage`, `rpg.sessions.manage`, `rpg.proposals.review`). The helper also refreshed two stale chat route descriptions in the generated fixture; the live snapshot test passes with those updates. Verification: `test_endpoint_scope_catalog_sync.py`, `test_privilege_registry_snapshot_matches_live_app`, and `test_rpg_api.py` passed together (7 passed); git diff --check passed. Bandit is not applicable for this slice because only YAML, JSON, Backlog, and plan files changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Finalized RPG privilege registration by adding the generic `rpg` token scope and regenerating the privilege route registry snapshot so the new RPG REST routes are represented in the generated AuthNZ fixture.
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
