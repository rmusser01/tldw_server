---
id: TASK-481.6
title: Implement notes PR 6 capture provenance and inbox view
status: Done
labels:
- notes
- ux
- extension
- webui
- backend
parent_task_id: TASK-481
modified_files:
- Docs/superpowers/plans/2026-05-27-notes-ux-remediation.md
- apps/packages/ui/src/services/notes-capture.ts
- apps/packages/ui/src/services/__tests__/notes-capture.test.ts
- apps/packages/ui/src/components/Sidepanel/Clipper/WebClipperPanel.tsx
- apps/packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx
- apps/packages/ui/src/routes/sidepanel-chat.tsx
- apps/packages/ui/src/routes/__tests__/sidepanel-chat.note-quick-save-lazy-mount.guard.test.ts
- apps/packages/ui/src/components/Notes/NotesSidebar.tsx
- apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage39.organization-model.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement PR 6 from the notes UX remediation plan: choose and record the Inbox backing model, make captured notes discoverable through durable provenance or reserved tags, harden sidepanel quick-save source metadata, and verify clipper/quick-save behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Web Clipper and sidepanel quick-save notes use shared capture helpers that preserve source metadata.
- [x] #2 Captured notes receive a durable reserved captured-note keyword so they can be rediscovered from /notes.
- [x] #3 /notes exposes a Captured quick filter that finds captured notes without requiring manual tag entry.
- [x] #4 Focused tests verify capture helper output, Web Clipper save flow, quick-save guard behavior, and Captured filter rendering.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-27-notes-ux-remediation.md#pr-6-capture-provenance-and-inbox-view
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented PR6 frontend slice. Added shared capture helpers, reserved captured-tag insertion for Web Clipper and sidepanel quick-save, a Captured quick filter in /notes, and focused regression coverage. Verification: focused Vitest suite passed for notes-capture, WebClipperPanel save flow, sidepanel quick-save guard, and NotesManagerPage organization model (33 tests). Browser smoke reached /notes on the local Next dev server and confirmed the Captured filter renders; pointer activation was blocked by a broader existing layout issue where the notes list is shifted partly under the app/sidebar region. Package-wide `bunx tsc --noEmit -p tsconfig.json` remains blocked by unrelated baseline TypeScript errors across existing files. Bandit not run because no Python files were touched.
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
