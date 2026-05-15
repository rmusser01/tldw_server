---
id: TASK-332
title: Harden Persona Visual import-commit eligibility revalidation
status: Done
assignee: []
created_date: '2026-05-14 03:21'
updated_date: '2026-05-14 05:56'
labels:
  - persona
  - visual-packs
  - backend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1657'
  - 'https://github.com/rmusser01/tldw_server/issues/1510'
  - 'https://github.com/rmusser01/tldw_server/pull/1684'
documentation:
  - Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md
  - Docs/Code_Documentation/Persona_Visual_Packs.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1657 as a focused Persona/Buddy visual-pack hardening slice. Import-commit must fail closed when stored or revalidated Persona Visual import previews are blocked or not commit-eligible, so stale completed previews or capability-state changes cannot create draft packs/assets before manifest validation fails. Scope stays backend/server-side only unless an existing API contract requires a minimal response adjustment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Import-commit rejects stored preview metadata that is not commit-eligible before starting draft pack or asset creation.
- [x] #2 Import-commit revalidates archive previews and rejects any non-completed or non-commit-eligible revalidation result before draft pack or asset creation.
- [x] #3 Blocked renderer preview commit attempts leave no partial draft packs/assets behind.
- [x] #4 Existing API-level completed-preview behavior remains intact for eligible previews.
- [x] #5 Focused backend regression tests cover stale stored ineligible preview metadata and revalidation-to-blocked behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add server-side API queueing guard for completed import previews whose stored proposed plan explicitly marks commit_eligible false.
2. Add importer-side guard before any pack or asset creation so stored previews and revalidated archive previews must be completed and commit-eligible.
3. Add focused regression coverage for stored ineligible preview metadata and revalidation-to-blocked behavior.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a stored preview commit-eligibility helper in the Persona API endpoint before import-commit jobs are created.
- Added shared importer validation that accepts legacy previews with no commit_eligible flag but rejects blocked previews or explicit commit_eligible false results.
- Added regression coverage proving blocked revalidation fails the job with no additional draft pack created.
- No docs changes were needed because the existing docs already describe blocked previews as commit-ineligible review results.
- Review follow-up: aligned API `blocked` preview reporting with worker `import_preview_not_commit_eligible` handling.
- Review follow-up: malformed or non-object stored `proposed_plan_json` now fails closed before job enqueueing or worker revalidation, while valid object metadata without `commit_eligible` remains eligible.
- Review follow-up: verification commands in this task use repo-relative invocations instead of machine-local absolute paths.
- PR #1684 review follow-up: added `replace_import_preview_proposed_plan_json()` so intentionally corrupted stored-plan fixture writes stay centralized in the DB repository instead of raw SQL in tests.
- PR #1684 review follow-up: normalized the API completion-status recheck with `.strip()`, made stored-plan JSON parsing bytes-safe, and restored importer top-level blank-line spacing.

## Verification

- RED: `python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_api.py::test_start_import_commit_rejects_ineligible_completed_preview tldw_Server_API/tests/Persona/test_persona_visual_portability_worker.py::test_persona_visual_import_commit_worker_rejects_revalidated_blocked_preview -q` failed with the API returning 202 and the worker not raising the new guard error.
- GREEN: same focused two-test command passed with 2 tests.
- Review RED: `python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_api.py::test_start_import_commit_reports_blocked_preview_as_not_commit_eligible tldw_Server_API/tests/Persona/test_persona_visuals_api.py::test_start_import_commit_rejects_invalid_stored_preview_plan tldw_Server_API/tests/Persona/test_persona_visual_portability_worker.py::test_persona_visual_import_commit_worker_rejects_invalid_stored_plan_before_revalidation -q` failed with 5 failing review-regression cases.
- Review GREEN: same review-regression command passed with 5 tests.
- Focused regression: `python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_api.py tldw_Server_API/tests/Persona/test_persona_visual_portability_worker.py -q` passed with 56 tests.
- Syntax: `python -m py_compile tldw_Server_API/app/api/v1/endpoints/persona.py tldw_Server_API/app/core/Persona/visual_portability/importer.py` passed.
- Whitespace: `git diff --check` passed.
- Bandit: `python -m bandit -r tldw_Server_API/app/api/v1/endpoints/persona.py tldw_Server_API/app/core/Persona/visual_portability/importer.py -f json -o /tmp/bandit_persona_visual_commit_guards_review.json` reported zero findings.
- PR #1684 review regression: `python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_import_commit_eligibility.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py tldw_Server_API/tests/Persona/test_persona_visual_portability_worker.py -q` passed with 66 tests.
- PR #1684 review syntax: `python -m py_compile tldw_Server_API/app/api/v1/endpoints/persona.py tldw_Server_API/app/core/Persona/visual_portability/importer.py tldw_Server_API/app/core/Persona/visual_portability/commit_eligibility.py tldw_Server_API/app/core/DB_Management/PersonaVisualPortability_DB.py` passed.
- PR #1684 review style: `python -m ruff check tldw_Server_API/app/core/Persona/visual_portability/importer.py --select E305 --preview` passed.
- PR #1684 review whitespace: `git diff --check` passed.
- PR #1684 review Bandit: `python -m bandit -r tldw_Server_API/app/api/v1/endpoints/persona.py tldw_Server_API/app/core/Persona/visual_portability/importer.py tldw_Server_API/app/core/Persona/visual_portability/commit_eligibility.py tldw_Server_API/app/core/DB_Management/PersonaVisualPortability_DB.py -f json -o /tmp/bandit_persona_visual_commit_guards_review2.json` reported zero findings.

## Known Skips

- Optional Ruff full-file check on the touched files still reports existing I001 import-order findings in legacy modules; this narrow backend guard slice did not auto-sort large unrelated import blocks.

PR #1684 review closeout refresh: rebased codex/persona-visual-commit-guards onto latest origin/dev so the PR diff is scoped back to Persona Visual import-commit guard files plus TASK-332.

Added docstrings to the newly introduced regression tests and local test doubles to address the remaining CodeRabbit docstring coverage warning while leaving existing unrelated tests unchanged.

Post-rebase verification: focused Persona pytest suite passed 66 tests; py_compile passed for changed API/core/tests; git diff --check passed; Ruff E305 passed for importer spacing; Bandit on touched production Python paths reported no findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened Persona Visual import-commit so stored previews with explicit `commit_eligible: false`, blocked status, malformed stored plan JSON, or non-object stored plan JSON are rejected before job queueing or worker revalidation. Worker-side revalidation also rejects blocked or non-commit-eligible previews before draft pack or asset creation. Added focused API and worker regressions while preserving existing eligible preview behavior for valid plan objects without `commit_eligible`. PR #1684 review follow-up also centralizes corrupted stored-plan fixture writes in the DB repository, normalizes API status checks consistently, and supports bytes-backed stored JSON parsing.

Post-rebase review closeout added docstrings for new regression tests, verified the PR diff is scoped to Persona Visual import-commit guard files, and reran focused pytest, py_compile, git diff --check, Ruff E305, and Bandit with no failures/findings.
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
