---
id: TASK-331
title: Harden Persona Visual import-commit eligibility revalidation
status: In Progress
assignee:
  - codex
created_date: '2026-05-14 03:19'
updated_date: '2026-05-14 03:49'
labels:
  - persona
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1510'
  - 'https://github.com/rmusser01/tldw_server/issues/1657'
  - 'https://github.com/rmusser01/tldw_server/pull/1678'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a narrow Persona/Buddy visual-pack hardening slice for GitHub #1657. Current origin/dev rejects blocked Manifest V2 renderer import previews at the API because their preview status is blocked, but the import-commit worker revalidates the archive and does not fail closed when the revalidated result is blocked or non-committable. Harden the server-side commit path so stale completed previews or capability-state changes cannot create partial draft packs/assets for unsupported renderer imports. Keep this scoped to Persona Visual Pack import-commit safety; do not add Live2D runtime activation, new renderer implementations, Persona Garden UI changes unless required by API contract, MCP provider expansion, VN/CYOA behavior, or live response mutation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Import-commit worker/importer fails closed when revalidated preview status is not completed or proposed_plan.commit_eligible is false.
- [x] #2 Stored preview metadata that already indicates commit ineligible is rejected before queuing a commit job when available.
- [x] #3 Blocked or non-committable renderer previews do not create draft visual packs or imported assets during failed commit attempts.
- [x] #4 Focused backend regression tests cover the stale completed/non-committable revalidation path and the API prequeue guard.
- [x] #5 Relevant Persona visual-pack docs or task notes are updated only if behavior/contract wording changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add red regression coverage in Persona visual import-commit tests for a stale completed preview whose archive revalidates as blocked/non-committable, asserting no draft pack/assets are created.
2. Add API regression coverage for a stored completed preview whose proposed_plan marks commit_eligible=false, asserting the commit request is rejected before queuing a job.
3. Add a small shared eligibility guard for preview metadata and apply it in the API prequeue path plus importer revalidation path.
4. Keep behavior limited to Persona Visual Pack import-commit safety; do not change renderer capabilities, activation, Persona Garden UI, MCP provider behavior, VN/CYOA, or live responses.
5. Run focused Persona visual portability/API tests, git diff checks, and Bandit on touched backend code before PR.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
Implemented shared import preview commit eligibility guards for API enqueue and worker revalidation paths.

Verification: red tests first confirmed stale blocked revalidation still created packs and commit-ineligible completed preview still queued a job before the fix.

Verification after fix: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_portability_worker.py -q --tb=short` passed 11 tests.

Verification after fix: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q --tb=short` passed 40 tests.

Verification after fix: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_portability.py tldw_Server_API/tests/Persona/test_persona_visual_import_preview_validators.py -q --tb=short` passed 21 tests.

Verification after fix: `git diff --check` passed.

Verification after fix: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit tldw_Server_API/app/core/Persona/visual_portability/commit_eligibility.py tldw_Server_API/app/core/Persona/visual_portability/importer.py tldw_Server_API/app/api/v1/endpoints/persona.py` reported no issues.

Draft PR opened: https://github.com/rmusser01/tldw_server/pull/1678. GitHub checks were pending at creation time; human-authored Change summary remains required before merge per repo policy.
<!-- SECTION:NOTES:END -->
