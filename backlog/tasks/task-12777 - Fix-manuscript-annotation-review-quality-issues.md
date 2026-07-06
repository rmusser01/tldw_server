---
id: TASK-12777
title: Fix manuscript annotation review quality issues
status: Done
priority: medium
modified_files:
- tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
- tldw_Server_API/app/core/DB_Management/ManuscriptDB.py
- tldw_Server_API/tests/Writing/test_manuscript_annotations_db.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task 2 review fixes: include anchor identity in duplicate suppression, add anchor_status schema constraints, validate anchor_status filter values, and reject non-list tags/non-dict metadata in annotation create/update paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed Task 2 review issues by constraining stored manuscript annotation anchor_status values in SQLite/PostgreSQL v50->v51 DDL, validating anchor_status filters, rejecting non-list tags and non-dict metadata in annotation create/update paths, and including scene_version/anchor_start/anchor_end in duplicate candidate identity so repeated text at different offsets is retained while exact duplicate anchors are suppressed.

Verification:
- Red: `../../.venv/bin/python -m pytest tldw_Server_API/tests/Writing/test_manuscript_annotations_db.py -q` failed with 7 expected failures before implementation.
- Green: `../../.venv/bin/python -m pytest tldw_Server_API/tests/Writing/test_manuscript_annotations_db.py tldw_Server_API/tests/Writing/test_manuscript_annotations_anchor.py -q` passed 31 tests.
- Bandit: `../../.venv/bin/python -m bandit -r tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/ManuscriptDB.py -f json -o /tmp/bandit_task_2402.json` exited 0 with no findings.
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
