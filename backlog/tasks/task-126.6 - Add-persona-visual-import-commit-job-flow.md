---
id: TASK-126.6
title: Add persona visual import commit job flow
status: Done
assignee: []
created_date: '2026-05-09 03:02'
updated_date: '2026-05-09 03:15'
labels:
  - persona
  - visual-packs
  - portability
  - api
  - jobs
dependencies:
  - TASK-126.3
  - TASK-126.4
  - TASK-126.5
references:
  - >-
    Docs/superpowers/plans/2026-05-08-persona-visual-packs-implementation-plan.md
  - 'https://github.com/rmusser01/tldw_server/pull/1135'
parent_task_id: TASK-126
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the PR1135-style commit step after a completed persona visual import preview. This slice should let a reviewed preview be accepted into a new draft visual pack through a Jobs-backed import_commit operation, remap archived source asset IDs to newly stored persona visual asset IDs, keep the target persona scoped to the route/user, and expose status for the commit job. Limit the first commit implementation to create_new target mode; update_existing/shared-library merge semantics can remain future work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Persona visual portability job helpers define import_commit job type, payload, group, idempotency key, and creation helper.
- [x] #2 Import commit worker validates the completed preview and archive checksum before mutating packs.
- [x] #3 Import commit creates a new draft persona visual pack for the target persona and imports present archive asset bytes through persona visual storage validation.
- [x] #4 Imported manifest references are remapped from archive source asset IDs to newly created asset IDs before persistence.
- [x] #5 API endpoints can start an import commit from a completed preview and return commit status scoped to the user/persona.
- [x] #6 Tests cover successful commit start/status, worker import mutation, source-to-new asset remapping, and rejection of incomplete previews.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add import_commit job helpers mirroring existing export/import-preview helper shape: type constant, payload builder, batch group, idempotency key, and create helper.
2. Add an import commit executor that loads a completed preview, validates archive readiness/checksum/fingerprint, creates a new draft target pack, imports present asset bytes through PersonaVisualService, remaps archived source asset ids to new asset ids, and persists the remapped manifest.
3. Route import_commit Jobs through PersonaVisualPortabilityWorker with scoped preview/job validation and progress/failure updates.
4. Expose API routes to start a commit from a completed preview and poll user/persona-scoped commit status.
5. Cover the behavior with focused job helper, worker mutation/remap, API start/status, and incomplete-preview rejection tests; run Bandit and diff checks before commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the import_commit flow for persona visual packs: new job helpers, importer, worker routing, API start/status routes, and tests for successful commit, scoped status, asset-id remapping, and incomplete-preview rejection.

Verification: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_jobs.py tldw_Server_API/tests/Persona/test_persona_visual_portability_worker.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q --tb=short` -> 28 passed, 5 warnings.

Security/quality checks: Bandit touched scope with `-s B101` wrote `/tmp/bandit_persona_visual_import_commit.json` and exited 0; `git diff --check` exited 0.

Broader regression: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_service.py tldw_Server_API/tests/Persona/test_persona_visual_portability.py tldw_Server_API/tests/Persona/test_persona_visual_jobs.py tldw_Server_API/tests/Persona/test_persona_visual_portability_worker.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q --tb=short` -> 39 passed, 5 warnings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a Jobs-backed persona visual import_commit flow that accepts a completed preview into a new draft visual pack. The worker revalidates the archive checksum/fingerprint before mutation, imports present asset bytes through existing visual storage validation, remaps source asset IDs in the manifest, and records progress/status through portability rows. The API now starts import commits and returns scoped status for the target persona/user. Focused pytest coverage passes (28 tests), broader persona visual backend regression passes (39 tests), Bandit on touched scope passes with B101 skipped for test assertions, and git diff whitespace checks pass.
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
