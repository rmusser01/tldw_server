---
id: TASK-13921
title: Workflow notes steps on SQLite use a nested ChaChaNotes.db, not the user's
  notes database
status: To Do
labels:
- workflows
- notes
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
run_notes_adapter (Workflows/adapters/knowledge/crud.py:177-181) passes DatabasePaths.get_user_base_directory(user), which is <base>/<id>, as NotesInteropService's base directory. NotesInteropService._get_db (Notes/Notes_Library.py) appends <id>/ChaChaNotes.db again, so on SQLite a workflow notes step reads and writes <base>/<id>/<id>/ChaChaNotes.db, while the Notes API uses DatabasePaths.get_chacha_db_path(id), <base>/<id>/ChaChaNotes.db. Seen in TASK-13897's PostgreSQL test log: 'Creating or loading DB for user_id 1 at path: .../user_dbs/1/1/ChaChaNotes.db'. On PostgreSQL the path does not choose the data, so only SQLite deployments are affected.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 On SQLite, a note that a workflow notes step creates is visible to the Notes API for the same user, with a test
- [ ] #2 Notes already written to the nested file are migrated, or the decision not to is recorded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

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
