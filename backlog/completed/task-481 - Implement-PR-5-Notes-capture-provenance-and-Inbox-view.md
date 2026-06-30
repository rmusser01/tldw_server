---
id: TASK-481
title: Implement PR 5 Notes capture provenance and Inbox view
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-27 19:11'
labels:
  - notes
  - ux
  - webui
  - extension
  - pr5
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the PR 5 /notes UX remediation slice from the approved notes plan: ensure captured extension notes land in All Notes and a durable captured/Inbox view without relying on ignored arbitrary note metadata. Scope is limited to directly connected capture handoffs: sidepanel quick-save, Web Clipper save-to-notes, durable provenance/marker behavior, and captured-note filtering in /notes if supported by existing contracts. Product decision for this slice: captured notes should appear in All Notes and Inbox; implement Inbox/captured as a reserved durable tag/keyword marker plus existing Web Clipper provenance where available, without migrating existing notes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Web Clipper saves continue to persist source_url, capture_metadata, title/comment, and tags.
- [x] #2 Sidepanel quick-save no longer relies on arbitrary ignored note metadata for source URL/origin.
- [x] #3 Captured notes appear in All Notes.
- [x] #4 Inbox/captured view is backed by durable clipper provenance and/or a reserved capture tag.
- [x] #5 The reserved capture marker, if used, is documented in code comments/tests and does not silently rewrite existing notes.
- [x] #6 Capture failure clearly tells the user whether note creation, provenance storage, or destination placement failed.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved plan reference in source checkout: Docs/superpowers/plans/2026-05-27-notes-ux-remediation.md, PR 5. Reserved marker decision for this implementation: use a durable keyword/tag marker for captured notes so generic note saves are discoverable through existing Notes filtering; preserve Web Clipper source_url/capture_metadata provenance for richer clipper saves.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Implemented durable capture/InBox behavior:
- Added reserved `captured` tag marker for extension-created notes in `apps/packages/ui/src/services/note-capture.ts`; no migration or rewrite of existing notes.
- Web Clipper payloads now preserve user-entered tags and add `captured`, while keeping existing `source_url` and `capture_metadata` provenance.
- Sidepanel quick-save now persists source URL through note content and sends durable top-level note fields (`title`, `keywords`) rather than ignored `metadata.source_url` / `metadata.origin`.
- `/notes` has an Inbox view mode backed by `tokens=captured`; clearing filters returns to List.

Verification recorded:
- RED first: focused tests failed for missing captured tag, sidepanel metadata reliance, and missing Inbox view.
- PASS: `bunx vitest run src/components/Notes/__tests__/NotesManagerPage.stage13.navigation-filter-summary.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage39.organization-model.test.tsx src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx src/routes/__tests__/sidepanel-chat.note-quick-save-lazy-mount.guard.test.ts src/routes/__tests__/sidepanel-chat.note-capture-payload.guard.test.ts src/routes/__tests__/sidepanel-note-capture.test.ts` (35 tests).
- PASS: `git diff --check`.
- TypeScript: `bunx -p typescript@5.9.3 tsc --noEmit --pretty false` has unrelated repo-wide baseline errors; filtered touched-path check returned no Notes/Clipper/sidepanel/note-capture errors.
- Browser smoke: local Next dev server rendered `/notes` after setup skip and exposed one Inbox button.
- Bandit skipped because this is a frontend-only TypeScript/React slice.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR 5 slice completed: captured extension notes now get a durable `captured` tag, sidepanel quick-save no longer depends on ignored arbitrary note metadata for provenance, and `/notes` exposes an Inbox view backed by the captured tag. Focused UI/route tests and browser smoke were recorded; no Python backend touched.
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
