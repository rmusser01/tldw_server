---
id: TASK-481.19
title: Rebase notes PR 2318 on dev and address review comments
status: In Progress
labels:
- notes
- review
- webui
- backend
parent_task_id: TASK-481
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase PR #2318 (`codex/notes-ux-pr1`) onto the latest origin/dev and address all substantive PR review comments. Verify the rebased branch and update the PR base/head.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Branch is rebased onto latest origin/dev and force-pushed safely.
- [ ] #2 PR #2318 base is updated to dev if needed.
- [ ] #3 All substantive inline review comments are addressed or documented with technical rationale.
- [ ] #4 Focused frontend/backend tests and Bandit for touched backend scope are run or skips documented.
- [ ] #5 Backlog and PR status are updated with final verification.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Rebased `codex/notes-ux-pr1` from the stacked skills branch onto `origin/dev` (`785245fc4f`) with conflict resolution across Notes list reliability, save recovery, first-time create, tag terminology, capture provenance, Web Clipper destinations, connection labels, import workflow, duplicate shortcut, and AI-title test commits.
- Addressed PR review comments:
  - Replaced direct open-note title interpolation with a replacer function so `$&`/`$1` title text remains literal.
  - Preserved integer pagination totals when notes search count helpers fail by falling back to `offset + len(notes_data)`.
  - Kept the title input width override in Tailwind classes instead of viewport-dependent inline `minWidth`.
  - Excluded missing/deleted note relations from manual-link target options.
  - Reconciled hidden stale `workspaceId` state when workspace picker options load.
  - Strengthened offline queued-save status assertions by keeping a stable hidden idle save-status node.
  - Filled commented Backlog task AC blocks and reconciled `Done` task DoD checklists.
- Removed duplicate JSX props introduced by conflict resolution in `NotesManagerPage` and `NotesSidebar`.
- Verification:
  - `bunx vitest run src/components/Notes/__tests__/NotesListPanel.stage18.accessibility-selected-state.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage6.manual-links.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage41.offline-drafting-sync.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage10.ai-title.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage36.import-workflow.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage23.responsive-layout.test.tsx src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx` passed: 7 files, 58 tests.
  - `../../.venv/bin/python -m pytest tldw_Server_API/tests/Notes_NEW/integration/test_notes_api.py::test_search_notes_with_keyword_tokens_returns_pagination_total tldw_Server_API/tests/Notes_NEW/integration/test_notes_api.py::test_search_notes_falls_back_to_integer_total_when_count_fails tldw_Server_API/tests/Notes_NEW/integration/test_notes_api.py::test_list_and_search_pagination_and_404s` passed: 3 tests.
  - `../../.venv/bin/python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chachanotes_db.py -k test_count_notes_matching_keywords` passed: 1 selected test.
  - `../../.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/endpoints/notes.py -f json -o /tmp/bandit_notes_pr2318.json` passed with zero findings.
  - `git diff --check` passed.
- Local note: backend test startup created untracked watchlist template files under `tldw_Server_API/Config_Files/templates/watchlists/`; they are outside this PR scope and are not staged.

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
