---
id: TASK-15553
title: Page notes by the API contract in QuickNotesSection and NotesDockPanel
status: To Do
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
- [ ] #1 Both call sites send limit/offset (no page/results_per_page) on /api/v1/notes/
- [ ] #2 Existing pagination behavior of the components is preserved or corrected where it was silently page-1
- [ ] #3 Tests updated and passing; typecheck green
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
