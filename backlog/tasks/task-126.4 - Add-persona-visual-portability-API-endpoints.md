---
id: TASK-126.4
title: Add persona visual portability API endpoints
status: Done
assignee: []
created_date: '2026-05-09 02:41'
updated_date: '2026-05-09 02:50'
labels:
  - persona
  - visual-packs
  - portability
  - api
dependencies:
  - TASK-126.2
  - TASK-126.3
  - TASK-132
references:
  - >-
    Docs/superpowers/plans/2026-05-08-persona-visual-packs-implementation-plan.md
  - 'https://github.com/rmusser01/tldw_server/pull/1135'
parent_task_id: TASK-126
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose API routes for PR1135-aligned persona visual pack export and import-preview jobs using the new portability repository, job helpers, and worker contract. Keep frontend UX and import commit out of this slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 API schemas define portability job, export request/response, import preview request/response, and archive download/status models.
- [x] #2 Export endpoint creates a portability row, enqueues a Jobs payload with idempotency metadata, and returns the tracked job record.
- [x] #3 Export status endpoint returns the tracked portability row scoped to the user/persona/pack.
- [x] #4 Export download endpoint serves a completed archive only when the tracked row is completed and scoped to the caller.
- [x] #5 Import-preview endpoint creates a preview row and Jobs payload without mutating persona visual packs.
- [x] #6 Import-preview status endpoint returns the tracked preview row scoped to the caller.
- [x] #7 Focused API tests cover export start/status/download, import-preview start/status, and cross-user denial.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added persona visual portability API schemas for export requests/responses, export status, import-preview start, and import-preview status.

Added scoped export start/status/download endpoints under persona visual pack routes using PersonaVisualPortabilityRepository and Jobs enqueue helpers.

Added import-preview upload/start and scoped status endpoints that stage .tldw-persona-vpack archives and do not mutate existing visual packs.

Verification passed: pytest tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q --tb=short => 14 passed, 5 warnings.

Regression sweep passed: pytest test_persona_visuals_api.py test_persona_visual_service.py test_persona_visual_portability.py test_persona_visual_jobs.py test_persona_visual_portability_worker.py -q --tb=short => 33 passed, 5 warnings.

Bandit passed with no findings for touched API/schema/repository/worker/test scope using -s B101 for test assertions: /tmp/bandit_persona_visual_portability_api.json.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added PR1135-style persona visual portability API endpoints for export start/status/download and import-preview upload/status. The API creates scoped portability rows, enqueues Jobs payloads through the persona visual job manager, validates archive type and download location, exposes review metadata from preview rows, and preserves the import-preview no-mutation contract. Focused and regression tests pass, and Bandit reported no findings on the touched scope.
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
