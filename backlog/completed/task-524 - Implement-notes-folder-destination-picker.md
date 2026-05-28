---
id: TASK-524
title: Implement notes folder destination picker
status: Done
labels:
- notes
- extension
- ux-remediation
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR 6B from the notes UX remediation plan: add or expose a public notes-folder list/create contract and wire the browser-extension capture destination UI to use a folder picker instead of relying on raw Folder ID entry. Preserve fallback behavior for unavailable folder data and avoid broad /notes redesign scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Public notes-folder list endpoint returns folders for the authenticated user with stable id/name/path fields.
- [x] #2 Public notes-folder create endpoint creates a user-scoped folder and handles duplicate or invalid names predictably.
- [x] #3 Capture destination UI loads note folders only when note destinations are relevant and exposes a picker when folder data is available.
- [x] #4 Capture save requests send the selected folder_id and preserve advanced/raw fallback when folder loading fails or no folder options exist.
- [x] #5 Focused backend and frontend regression tests cover the new folder API and picker behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-27-notes-ux-remediation.md#pr-6-destination-pickers-for-capture
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added public `GET/POST /api/v1/notes/folders/` routes backed by `CharactersRAGDB` note-folder helpers.
- Added frontend note-folder client methods and a browser-extension capture folder picker with advanced raw-ID fallback.
- Browser sidepanel live render was not re-run because the extension sidepanel requires extension runtime context; verification used focused component tests and extension compile instead.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the PR 6B notes-folder destination slice. Folder rows can now be listed/created through the notes API, duplicate paths are idempotent, and the Web Clipper capture form uses a folder picker when available while preserving manual ID fallback for unavailable or empty folder data.
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
