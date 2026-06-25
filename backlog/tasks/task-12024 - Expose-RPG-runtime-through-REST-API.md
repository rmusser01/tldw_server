---
id: TASK-12024
title: Expose RPG runtime through REST API
status: Done
created_date: 2026-06-25 03:53
labels:
- rpg
- ttrpg
- backend
- api
- implementation
priority: high
references:
- TASK-12018
- TASK-12019
- TASK-12020
- TASK-12021
- TASK-12022
- TASK-12023
documentation:
- Docs/superpowers/plans/2026-06-25-rpg-campaign-session-runtime-implementation-plan.md
updated_date: 2026-06-25 04:09
modified_files:
- Docs/superpowers/plans/2026-06-25-rpg-campaign-session-runtime-implementation-plan.md
- tldw_Server_API/app/api/v1/schemas/rpg_schemas.py
- tldw_Server_API/app/api/v1/endpoints/rpg.py
- tldw_Server_API/app/api/v1/router_groups/content.py
- tldw_Server_API/app/api/v1/router_groups/minimal.py
- tldw_Server_API/Config_Files/privilege_catalog.yaml
- tldw_Server_API/tests/RPG/test_rpg_api.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the RPG REST API slice from the reviewed plan: Pydantic schemas, endpoints for adapters/campaign/session/event/proposal basics, router registration, privilege catalog entries, and focused API tests. Include minimal-test router registration so API tests hit the router under the repo's test app mode.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Adapters endpoint lists bundled adapters
- [x] #2 Campaign/session creation works through REST with Idempotency-Key
- [x] #3 User event recording commits through REST and returns committed events
- [x] #4 RPG router is registered in normal and minimal test router groups
- [x] #5 Privilege catalog includes all RPG endpoint_id values
- [x] #6 Focused API and privilege catalog tests pass
- [x] #7 Bandit/diff checks are recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write failing API tests for adapter listing and campaign/session/user-event workflow.
2. Add RPG Pydantic schemas and endpoint handlers using RPGService.
3. Register router in content and minimal router groups; add privilege catalog entries for RPG endpoint IDs.
4. Run API tests, privilege catalog sync, adjacent RPG service tests, compileall, Bandit, and diff checks.
5. Record modified files and final notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented RPG REST schemas/endpoints, registered the router in normal content routing and minimal-test routing, and added RPG privilege catalog entries. TDD RED confirmed before implementation: focused RPG API tests failed with 404 for /api/v1/rpg/rules/adapters and /api/v1/rpg/campaigns. Verification: focused API tests passed (2 passed); API plus service tests passed (7 passed); focused RPG plus privilege catalog suite passed (38 passed); privilege catalog sync passed; compileall passed; Bandit reported 0 results; git diff --check passed.
Post-review fixes applied after subagent review: clarified the implementation plan that Task 6 is the backed REST basics slice rather than the full target route matrix; added a focused RPG catalog guard for dynamic endpoint scope constants; added API negative tests for missing Idempotency-Key and stale expected event sequence; and set RPG TokenScopeGuard dependencies to count scoped-token calls. Verification after review: focused API/service/catalog suite passed (11 passed); full focused RPG plus privilege catalog suite passed (41 passed); compileall passed; Bandit reported 0 results; git diff --check passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Exposed the RPG runtime through REST for adapter discovery, campaign creation, session creation, event recording, and proposal apply/reject basics. The router is available in both normal and minimal test app registration, and RPG endpoint IDs are represented in the privilege catalog.
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
