---
id: TASK-530.4
title: Implement Skills server-backed filters and sorting
status: Done
labels:
- skills
- webui
- ux
- power-user
priority: high
parent_task_id: TASK-530
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue TASK-530 power-user Skills remediation after TASK-2342. Add a focused, reviewable slice for server-backed filters and sorting on /skills.

Scope:
- Extend the Skills list backend contract with safe filter/sort parameters beyond q.
- Apply filters and sort before pagination in the ChaChaNotes skill registry query path.
- Keep q search and filtered totals consistent with TASK-2342.
- Add typed frontend client params with camelCase-to-snake_case query serialization.
- Wire the Skills manager table/query state to server-backed filters and sort controls.

Out of scope:
- Dense view.
- Metadata column picker beyond controls needed for filter/sort visibility.
- Bulk export/actions.
- Safe operations, dry-run execution, import review, delete/version semantics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 GET /api/v1/skills accepts optional context, user_invocable, has_tools, model, sort, and order parameters in addition to q/limit/offset.
- [x] #2 Skills registry filtering and sorting are applied before pagination, and total reflects the filtered result count.
- [x] #3 Sort fields and directions are whitelisted before SQL interpolation; user-controlled filter values remain parameterized.
- [x] #4 The frontend API client exposes typed Skills list params and serializes camelCase filter/sort options to backend query parameters.
- [x] #5 The Skills manager uses server-backed filter and sort query state, resets pagination when filters change, and preserves existing debounced search behavior.
- [x] #6 Focused backend, client, and Skills manager tests cover filter/sort behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/Plans/IMPLEMENTATION_PLAN_skills_filters_sorting_TASK_530_4.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added backend Skills list params for context, explicit visibility, tool presence, model, sort field, and sort direction.
- Added a shared skill registry filter builder so list and count queries apply the same predicates before pagination.
- Kept dynamic sort SQL limited to whitelisted columns/directions; all user-provided filter values remain parameterized.
- Extended the UI client with typed `SkillsListParams` and centralized camelCase-to-snake_case serialization.
- Added compact Skills manager controls for mode, visibility, tools, and model filtering; table sorting now requests server-backed ordering and resets pagination.
- Added accessible names for icon-only row action buttons while preserving the existing compact table layout.
- Review fixes: debounced model filtering, corrected filtered-empty state behavior, switched registry validation to `InputError`, replaced dynamic sort clause construction with prebuilt constants, reused `SkillContext` for execution mode, and aligned frontend sort/order type names with backend aliases.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented server-backed Skills filters and sorting across the API, registry query layer, typed UI client, and `/skills` manager. The page now supports mode, visibility, tools, model filtering, and sortable name/mode table columns without client-side page-only filtering.

Verification:
- `python -m pytest tldw_Server_API/tests/Skills/unit/test_skill_registry_queries.py tldw_Server_API/tests/Skills/unit/test_skills_service.py tldw_Server_API/tests/Skills/integration/test_skills_api.py -q` - 100 passed, 6 warnings.
- `bunx vitest run src/services/__tests__/tldw-api-client.boundary-slices.test.ts src/components/Option/Skills/__tests__/Manager.test.tsx` - 26 passed after review fixes.
- `python -m bandit -r tldw_Server_API/app/api/v1/endpoints/skills.py tldw_Server_API/app/api/v1/schemas/skills_schemas.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/Skills/skills_service.py -f json -o /tmp/bandit_skills_filters_sorting_TASK_530_4.json` - exit 0; remaining warnings are existing `nosec` comments in the large DB module.
- `git diff --check` - passed.

Known verification caveat:
- `bunx tsc --noEmit -p tsconfig.json` was attempted as an extra guard. The default run OOMed; the 8GB retry reached existing package-level failures in Notes tests, background response result handling, and voice-cloning ArrayBuffer typing. Focused Vitest coverage is the UI gate for this slice.
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
