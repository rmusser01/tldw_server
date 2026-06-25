---
id: TASK-12026
title: Add RPG rules lookup and session context builder
status: Done
created_date: 2026-06-25 04:14
labels:
- rpg
- ttrpg
- backend
- api
- implementation
priority: high
references:
- TASK-12018
- TASK-12024
- TASK-12025
documentation:
- Docs/superpowers/plans/2026-06-25-rpg-campaign-session-runtime-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-06-25-rpg-campaign-session-runtime-implementation-plan.md
- tldw_Server_API/app/core/RPG/models.py
- tldw_Server_API/app/core/RPG/rules/content_packs.py
- tldw_Server_API/app/core/RPG/rules/lookup.py
- tldw_Server_API/app/core/RPG/context.py
- tldw_Server_API/app/core/RPG/service.py
- tldw_Server_API/app/api/v1/schemas/rpg_schemas.py
- tldw_Server_API/app/api/v1/endpoints/rpg.py
- tldw_Server_API/tests/RPG/test_rpg_rules_context.py
- tldw_Server_API/tests/RPG/test_rpg_api.py
- tldw_Server_API/tests/fixtures/privilege_route_registry_snapshot.json
updated_date: 2026-06-25 04:22
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the rules lookup and bounded session context slice for the RPG runtime, including citation-only bundled lookup behavior, context text/diagnostics, service methods, REST schemas/endpoints for rules lookup and context building, tests, and plan updates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Rules lookup returns citation metadata without bundled PF2E prose
- [x] #2 Session context builder includes session/snapshot/rules data and respects max character budget
- [x] #3 RPGService exposes lookup_rules and build_context using owner-scoped session state
- [x] #4 REST rules/context endpoints work with focused API tests
- [x] #5 Focused rules/context/API tests, compileall, Bandit, and diff checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write failing core and API tests for citation-only rules lookup, bounded session context, service methods, and REST endpoints.
2. Implement citation/result dataclasses and lookup service with citation-only bundled results.
3. Implement a bounded, incremental context builder and wire owner-scoped service methods.
4. Add typed REST schemas and session-scoped lookup/context endpoints; regenerate the privilege route registry snapshot.
5. Run focused RPG/API/privilege tests, compileall, Bandit, and diff checks, then commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented citation-only rules lookup and bounded context building. Post-review fixes included: context assembly now stops incrementally at budget instead of slicing large output, REST responses use typed citation/item/diagnostic schemas, REST `max_chars` requires 1000..24000, service callers are clamped into that range, lookup diagnostics mark `result_mode="citation_index"`, and the privilege route registry snapshot was regenerated to include the new rules/context routes. TDD RED confirmed before implementation: focused tests failed with `ModuleNotFoundError: No module named 'tldw_Server_API.app.core.RPG.context'`. Verification: focused rules/context + RPG API tests passed (11 passed); full focused RPG plus privilege catalog/snapshot suite passed (48 passed); compileall passed; Bandit reported 0 results; git diff --check passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added citation-only rules lookup, bounded session context generation, owner-scoped service methods, and session-scoped REST endpoints for RPG rules lookup/context. The API now returns typed citation/context diagnostics, and the AuthNZ route snapshot includes the new routes.
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
