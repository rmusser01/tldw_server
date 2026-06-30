---
id: TASK-126.3
title: Add persona visual portability persistence and worker handlers
status: Done
assignee: []
created_date: '2026-05-09 02:31'
updated_date: '2026-05-09 02:39'
labels:
  - persona
  - visual-packs
  - portability
  - jobs
dependencies: []
references:
  - >-
    Docs/superpowers/plans/2026-05-08-persona-visual-packs-implementation-plan.md
  - 'https://github.com/rmusser01/tldw_server/pull/1135'
parent_task_id: TASK-126
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add persistent persona visual export/import-preview bookkeeping and worker handlers that execute the PR1135-aligned visual pack export and import preview flows from Jobs payloads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Persona visual portability jobs can be persisted, fetched, and updated with operation, status, progress, archive, warning, and error metadata.
- [x] #2 Persona visual import previews can be persisted, fetched, and updated with target persona, archive, validation summary, proposed plan, warnings, and error metadata.
- [x] #3 A worker handler can run a visual pack export job, write the archive, and update the portability job record with exported status and checksums.
- [x] #4 A worker handler can run a visual pack import-preview job, validate the archive without mutating persona assets, and update both preview and job records.
- [x] #5 Focused tests cover persistence and worker export/import-preview behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented PersonaVisualPortabilityRepository with SQLite tables for persona visual portability jobs and import-preview records.

Implemented PersonaVisualPortabilityWorker to handle persona visual pack export and import-preview Jobs payloads while keeping generation worker concerns separate.

Verification passed: pytest tldw_Server_API/tests/Persona/test_persona_visual_portability_worker.py -q => 3 passed, 5 warnings.

Regression sweep passed: pytest test_persona_visual_service.py test_persona_visual_portability.py test_persona_visual_jobs.py test_persona_visual_portability_worker.py -q => 19 passed, 5 warnings.

Bandit passed with no findings for touched implementation/test scope using -s B101 for test assertions: /tmp/bandit_persona_visual_portability_worker.json.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added PR1135-style persona visual portability persistence and worker handlers. The slice introduces SQLite-backed portability job/import-preview records plus a dedicated worker that runs pack export and import-preview validation, updates status/checksum/fingerprint/progress metadata, and preserves preview no-mutation semantics. Focused and regression tests pass, and Bandit reported no findings on the touched scope.
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
