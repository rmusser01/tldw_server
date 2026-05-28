---
id: TASK-528
title: Address PR 2086 notes integration review feedback
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-28 03:12'
labels:
  - notes
  - ux
  - pr-review
  - integration
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase the consolidated /notes remediation branch on latest dev and address actionable PR #2086 review comments, CI failures, and integration issues. Keep scope limited to the /notes remediation branch and directly required test/backlog updates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Branch is rebased on latest origin/dev or equivalent latest-dev integration is documented.
- [x] #2 All actionable PR #2086 review comments/issues are either fixed or documented with technical rationale.
- [x] #3 Relevant focused tests and checks are run and results recorded.
- [x] #4 Backlog task records review items, changes, verification, and unresolved baseline limitations.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Review feedback addressed after rebasing PR #2086 branch on origin/dev 731c365b5:
- Gemini: moved recent-note persistence out of React state updater callbacks by introducing a recentNotesRef-backed updateRecentNotes helper in useNotesEditorState.
- Gemini: create_note_folder now returns the existing folder directly with HTTP 200 and no redundant create_note_folder_path write.
- Qodo: note-folder endpoint rate-limit check failures now log a warning before retaining the existing fail-open behavior.
- Qodo: list_note_folders/create_note_folder now have return type hints and docstrings.
- Qodo: new note-folder tests now have fixture/test return and parameter type hints.
- Qodo: PostgreSQL note_folders schema now has a unique LOWER(path) index and the duplicate _ensure_note_folder_schema_postgres definition was removed.
- Qodo: WebClipper folder/workspace picker load guards reset on cancellation/failure so users can retry after transient failures.

Verification:
- Red checks before fix: WebClipper retry tests failed; PostgreSQL lower-path schema test failed.
- Green targeted: WebClipper retry tests passed: 2 tests; note folder DB tests passed: 2 tests.
- Affected frontend: WebClipperPanel.save-flow + NotesManagerPage.stage47 passed: 2 files, 35 tests.
- Backend notes: test_note_folders.py + test_notes_api.py passed: 24 tests.
- Broad notes UI matrix passed: 74 files, 262 tests.
- apps/extension: bun run compile passed.
- Bandit touched Python notes scope passed with 0 findings and 0 errors: /tmp/bandit_notes_pr2086_review.json.
- git diff --check and git diff --check origin/dev..HEAD passed.
- apps/packages/ui typecheck still exits 2 only on unchanged baseline src/components/Option/Characters/__tests__/CharacterListContent.design-system.test.tsx:35, GalleryCardDensity rejects "comfortable".

Unstaged/untracked note:
- Backend tests generated tldw_Server_API/Config_Files/templates/watchlists/cti_osint_report_markdown.md and news_briefing_markdown.md; these remain untracked and are not part of this PR update.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased codex/notes-ux-remediation-integrated onto latest origin/dev and addressed all actionable PR #2086 review threads from Gemini and Qodo. The fixes cover React recent-note persistence purity, idempotent folder-create behavior, rate-limiter warning logs, endpoint/test type hints and docs, PostgreSQL case-insensitive folder path uniqueness, and retryable clipper destination picker loads. Focused and broad notes verification passed; the remaining typecheck failure is the unrelated pre-existing CharacterListContent density baseline.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Review comments/check failures resolved or explicitly documented
- [x] #3 Verification recorded
- [x] #4 Final summary added
<!-- DOD:END -->
