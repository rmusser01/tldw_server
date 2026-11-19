---
id: TASK-15553
title: Page notes by the API contract in QuickNotesSection and NotesDockPanel
status: Done
labels:
- bug
- frontend
- notes
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Dev-side bug found during W4 convergence (#3149 follow-up): QuickNotesSection.tsx:261 and NotesDockPanel.tsx:456 call GET /api/v1/notes/ with page/results_per_page query params that the notes API ignores — it pages by limit/offset only, so these reads silently pin to page-1 semantics and any pagination UI built on them shows wrong totals/next pages. Fix both call sites to use the existing #3149 helper (buildNotesPageRequestPath in apps/packages/ui/src/services/notes-page-request.ts) or equivalent limit/offset params, keeping include_keywords. Update any tests pinning the old query strings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both call sites send limit/offset (no page/results_per_page) on /api/v1/notes/
- [x] #2 Existing pagination behavior of the components is preserved or corrected where it was silently page-1
- [x] #3 Tests updated and passing; typecheck green
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Merged via merge queue: PR #3486 -> dev (4ecfbe0816, 2026-11-17). Both call sites (QuickNotesSection.buildListPath, NotesDockPanel.fetchNotesList list branch) now send limit/offset — the notes API's only paging contract (#3149) — instead of the ignored page/results_per_page params. Queue eviction en route surfaced two suites the first verification missed (task-backed-todos, stage4.cache-sync mocks matched ?page=1); updated to ?limit= matching. Final verification: full NotesDock directory + QuickNotesSection suites 14 files 264/264.
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
