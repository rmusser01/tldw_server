---
id: TASK-2401
title: Implement Writing Playground manuscript annotations
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-06-23 15:03'
labels:
  - implementation
  - webui
  - extension
  - writing-playground
  - manuscripts
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-06-23-writing-playground-manuscript-annotations-implementation-plan.md
  - >-
    Docs/superpowers/specs/2026-05-24-writing-playground-manuscript-annotations-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-06-23-writing-playground-manuscript-annotations-implementation-plan.md task-by-task using subagent-driven development and TDD. Start with Task 1 pure annotation anchor helpers, then run spec and code-quality reviews before advancing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 1 complete: added pure manuscript annotation anchor constants/helpers and tests.

TDD evidence:
- Initial red run failed as expected before implementation.
- Review-fix red run failed as expected for absent selected-text context recovery and malformed scene_version handling.
- Green run: `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/Writing/test_manuscript_annotations_anchor.py -q` -> 16 passed, 6 warnings.

Review evidence:
- Spec review approved after re-checking the keyword-only API contract in the plan.
- Code-quality review found two anchor hardening issues; both were fixed.
- Code-quality re-review approved.

Security/static checks:
- Bandit on `tldw_Server_API/app/core/Writing/manuscript_annotations.py` wrote `/tmp/bandit_manuscript_annotations_task1_verify_after_fix.json` with no findings.
- `git diff --check HEAD~2..HEAD` passed.

Task 2 complete: added manuscript annotation persistence schema, migrations, DB helper methods, and regression tests.

TDD evidence:
- Initial DB red run failed as expected before the schema/helper implementation.
- Review-fix red run failed as expected for duplicate suppression, anchor_status constraints/filter validation, and structured tags/metadata validation.
- Green run: `../../.venv/bin/python -m pytest tldw_Server_API/tests/Writing/test_manuscript_annotations_db.py tldw_Server_API/tests/Writing/test_manuscript_annotations_anchor.py -q` -> 31 passed, 6 warnings.

Review evidence:
- Spec review initially found anchor_status filtering after SQL LIMIT could return incomplete results.
- Fix commit `68f2c9f412` filters derived anchor status before pagination and enforces the unbounded candidate cap.
- Spec re-review approved.
- Code-quality review found duplicate suppression ignored anchor identity and schema lacked anchor_status CHECK constraints.
- Fix commit `6766200de7` added anchor identity duplicate keys, SQLite/PostgreSQL anchor_status CHECK constraints, anchor_status filter validation, and structured tags/metadata validation.
- Code-quality re-review approved with no remaining Critical or Important issues.

Security/static checks:
- Bandit on `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` and `tldw_Server_API/app/core/DB_Management/ManuscriptDB.py` wrote `/tmp/bandit_task_2402.json` with no findings.
- `git diff --check origin/dev..HEAD` passed before the Task 2 review-fix commit and will be rerun before the next task starts.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
