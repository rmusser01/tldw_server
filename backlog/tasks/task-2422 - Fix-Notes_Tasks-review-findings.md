---
id: TASK-2422
title: Fix Notes_Tasks review findings
status: Done
assignee: []
created_date: '2026-06-23 14:44'
updated_date: '2026-06-23 15:05'
labels:
  - notes
  - tasks
  - review
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address current-code review findings in the Notes_Tasks module: prevent task text from being silently reinterpreted as metadata, stop current warning-state notes from being reprocessed as stale forever, and align stale checklist discovery with the parser-supported unordered checkbox bullets.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Literal task text containing allowlisted metadata token syntax is rejected or escaped so reconciliation cannot silently remove it.
- [x] #2 Current warning-state notes are not treated as stale candidates until their note version changes, while warning state remains visible to callers.
- [x] #3 Stale checklist discovery recognizes every unordered checkbox bullet syntax accepted by the parser.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Use TDD. Add focused regression coverage before production changes, then verify with Notes_Tasks unit tests, Notes task API integration tests, task-store tests covering stale discovery, and Bandit on touched backend scope.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the Notes_Tasks review findings by rejecting parseable metadata-token syntax in literal task text, returning current warning reconciliation state without reprocessing, and aligning stale checklist discovery with the parser-supported `-`, `*`, and `+` checkbox bullets. Added regression coverage in the Notes task service tests and ChaCha task-store tests.

Verification:
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Notes_Tasks/unit -q` -> 55 passed.
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chacha_task_store.py -q` -> 60 passed.
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Notes_NEW/integration/test_notes_tasks_api.py tldw_Server_API/tests/Notes_NEW/integration/test_notes_tasks_reconciliation_api.py -q` -> 37 passed.
- `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Notes_Tasks tldw_Server_API/app/core/DB_Management/chacha/task_store.py -f json -o /tmp/bandit_notes_tasks_review_fixes.json` -> 0 findings.

Known skips/blockers: none.
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
