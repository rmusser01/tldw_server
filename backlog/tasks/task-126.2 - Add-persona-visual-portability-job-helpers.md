---
id: TASK-126.2
title: Add persona visual portability job helpers
status: Done
assignee: []
created_date: '2026-05-09 02:19'
updated_date: '2026-05-09 02:20'
labels:
  - persona
  - jobs
  - portability
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1135'
  - 'https://github.com/rmusser01/tldw_server/issues/1388'
  - 'https://github.com/rmusser01/tldw_server/issues/1389'
documentation:
  - >-
    Docs/superpowers/plans/2026-05-08-persona-visual-packs-implementation-plan.md
parent_task_id: TASK-126
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the Jobs contract helpers for persona visual pack portability so later API endpoints and workers can enqueue export and import-preview work using the same background-job/review-step model introduced for VN asset packs in PR #1135. Scope is payload builders, job type constants, idempotency keys, grouping, queue selection, and focused tests; API endpoints, workers, and frontend UI are out of scope for this slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Persona visual pack export and import-preview job type constants and payload builders are added
- [x] #2 Create-job helpers enqueue export and import-preview jobs with stable domain, queue, owner, group, idempotency, and retry values
- [x] #3 Idempotency keys include relevant request and option/archive inputs to avoid accidental collisions
- [x] #4 Focused tests cover payload shape, queue fallback, idempotency behavior, and create-job arguments
- [x] #5 Task notes identify the follow-up worker/API slice
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added persona visual portability Jobs contract helpers in tldw_Server_API/app/core/Persona/visual_jobs.py: export/import-preview job type constants, payload builders, portability queue fallback, batch-group helpers, idempotency keys, and create-job helpers. The helpers keep the existing persona_visuals domain and mirror the PR #1135 VN pack Jobs model for export/import-preview enqueue behavior.

Extended tldw_Server_API/tests/Persona/test_persona_visual_jobs.py with focused coverage for export and import-preview payload shape, default queue, batch group strings, owner attribution, options/archive digests in idempotency keys, and create-job arguments. Existing generation job tests still pass.

Verification passed: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_jobs.py -q (5 passed). Bandit passed with B101 skipped for pytest assertions: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Persona/visual_jobs.py tldw_Server_API/tests/Persona/test_persona_visual_jobs.py -s B101 -f json -o /tmp/bandit_persona_visual_portability_jobs.json (0 findings). git diff --check passed.

Follow-up worker/API slice: add persistent portability job/preview records, worker handlers using PersonaVisualPackExporter and PersonaVisualPackImportPreviewer, persona API endpoints for export job start/status/download/cancel and import preview upload/status/cancel/delete, then frontend review UX.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added persona visual portability Jobs helpers for export and import-preview enqueue flows, plus focused tests. This establishes the background job contract needed for the next worker/API/frontend review-flow slice while leaving endpoints and workers out of scope for this commit.
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
