---
id: TASK-2342
title: Implement Skills power-user server-backed search
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-09 16:16'
labels:
  - skills
  - webui
  - ux
  - power-user
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first power-user Skills discovery slice: make /skills search use a server-backed list query so large libraries can find matches outside the current page without client-only filtering.

Scope:
- Add a focused task plan for this reviewable slice.
- Add backend tests proving the skills list query filters before pagination and reports filtered totals.
- Add API/client/frontend tests for query-string search wiring.
- Extend the Skills list API/service/DB contract with a safe q search parameter.
- Update the Skills manager UI to request searched results from the server and reset pagination when search changes.

Out of scope:
- Bulk actions, dense mode, tag/type chips, shortcuts, import/export, and permissions changes.
- Visual redesign beyond the search/pagination state needed for this slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 GET /api/v1/skills accepts optional q and applies it before limit/offset pagination.
- [x] #2 Filtered list responses return only matching skills and total reflects the filtered result count.
- [x] #3 The frontend API client serializes trimmed q with limit/offset and omits blank q values.
- [x] #4 The Skills manager uses server-backed search results instead of filtering only the current page.
- [x] #5 Changing the search query resets pagination to the first page and shows the filtered total/empty state.
- [x] #6 Focused backend and frontend tests cover the search/pagination contract.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/Plans/IMPLEMENTATION_PLAN_skills_power_user_server_search_TASK_2342.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented server-backed Skills search for the first power-user discovery slice. The Skills list API now accepts optional q, the service/database apply the search to name/description before pagination with parameterized LIKE patterns, and filtered totals drive pagination metadata. The frontend client serializes trimmed q values and omits blank searches. The Skills manager now keys queries by search text, requests server-filtered results, resets to page 1 on search changes, and renders backend-returned rows instead of filtering only the current page.

Verification:
- Red checks confirmed missing behavior before implementation: service rejected q, API ignored q and returned the first page, client sent blank q, and manager did not request q.
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Skills/unit/test_skills_service.py tldw_Server_API/tests/Skills/integration/test_skills_api.py -q -> 92 passed, 6 warnings.
- bunx vitest run src/services/__tests__/tldw-api-client.boundary-slices.test.ts src/components/Option/Skills/__tests__/Manager.test.tsx -> 18 passed.
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit tldw_Server_API/app/api/v1/endpoints/skills.py tldw_Server_API/app/core/Skills/skills_service.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py -f json -o /tmp/bandit_task2342.json -> 0 findings, 0 errors.
- git diff --check -> passed.

Known skips/blockers: none.

PR: https://github.com/rmusser01/tldw_server/pull/2330
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
