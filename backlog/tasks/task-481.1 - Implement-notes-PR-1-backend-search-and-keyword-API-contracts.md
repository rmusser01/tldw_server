---
id: TASK-481.1
title: Implement notes PR 1 backend search and keyword API contracts
status: Done
labels:
- notes
- ux
- backend
- api
- planning
parent_task_id: TASK-481
modified_files:
- tldw_Server_API/app/api/v1/endpoints/notes.py
- tldw_Server_API/app/api/v1/schemas/notes_schemas.py
- tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
- tldw_Server_API/app/core/DB_Management/chacha/note_store.py
- apps/packages/ui/src/components/Notes/hooks/useNotesListManagement.tsx
- apps/packages/ui/src/services/note-keywords.ts
- tldw_Server_API/tests/Notes_NEW/integration/test_notes_api.py
- tldw_Server_API/tests/ChaChaNotesDB/test_chachanotes_db.py
- apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage11.search-filtering.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement PR 1 from the notes UX remediation plan: make /api/v1/notes/search return the standard paginated notes envelope, fix keyword search route ordering and aliases, align frontend search parsing, and add focused backend/frontend regression coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-27-notes-ux-remediation.md#pr-1-backend-search-and-keyword-api-contracts
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR 1 backend/frontend contract slice implemented in worktree /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/notes-ux-pr1 on branch codex/notes-ux-pr1.

Changed files:
- apps/packages/ui/src/components/Notes/hooks/useNotesListManagement.tsx
- apps/packages/ui/src/components/Notes/hooks/__tests__/useNotesListManagement.search-response.test.tsx
- tldw_Server_API/app/api/v1/endpoints/notes.py
- tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
- tldw_Server_API/app/core/DB_Management/chacha/note_store.py
- tldw_Server_API/tests/ChaChaNotesDB/test_chachanotes_db.py
- tldw_Server_API/tests/Notes_NEW/integration/test_notes_api.py

Verification:
- RED backend tests failed as expected for /keywords/search no-slash 422 and bare-array /notes/search response.
- RED frontend hook test failed as expected with total 1 instead of canonical 42.
- PASS: source .venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chachanotes_db.py::TestNotes::test_search_notes tldw_Server_API/tests/ChaChaNotesDB/test_chachanotes_db.py::TestNotes::test_count_notes_matching_keywords tldw_Server_API/tests/Notes_NEW/integration/test_notes_api.py::test_keywords_crud_and_linking tldw_Server_API/tests/Notes_NEW/integration/test_notes_api.py::test_search_notes_with_keyword_tokens_returns_pagination_total tldw_Server_API/tests/Notes_NEW/integration/test_notes_api.py::test_list_and_search_pagination_and_404s -v
- PASS: ./node_modules/.bin/vitest run src/components/Notes/hooks/__tests__/useNotesListManagement.search-response.test.tsx --maxWorkers=1 --no-file-parallelism
- PASS: source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/notes.py tldw_Server_API/app/core/DB_Management/chacha/note_store.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py -f json -o /tmp/bandit_notes_pr1.json (0 results, 0 errors)

Notes:
- Frontend Vitest in this worktree required workspace node_modules repair because package symlinks expected apps/node_modules. An interrupted bun install left ignored apps/node_modules artifacts only; tracked node_modules symlink was restored and is not in git status.
- No browser check was run because this PR changes API contracts and parser normalization rather than rendered UX controls.
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
