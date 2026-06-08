---
id: TASK-481.7
title: Implement notes PR 7 destination pickers for capture
status: Done
labels:
- notes
- ux
- extension
- webui
parent_task_id: TASK-481
modified_files:
- Docs/superpowers/plans/2026-05-27-notes-ux-remediation.md
- apps/packages/ui/src/components/Sidepanel/Clipper/ClipDestinationFields.tsx
- apps/packages/ui/src/components/Sidepanel/Clipper/WebClipperPanel.tsx
- apps/packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx
- apps/packages/ui/src/routes/__tests__/sidepanel-clipper.test.tsx
- apps/packages/ui/src/services/__tests__/tldw-api-client.workspace-api.test.ts
- apps/packages/ui/src/services/tldw/domains/workspace-api.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement PR 7 from the notes UX remediation plan: verify destination APIs, replace raw destination IDs in capture flows where list data exists, preserve a clear fallback if needed, and test workspace/folder destination behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-27-notes-ux-remediation.md#pr-7-destination-pickers-for-capture
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed PR7A destination picker slice. Added frontend workspace list support through `workspaceApiMethods.listWorkspaces()`, loaded active workspaces in the Web Clipper panel, rendered a workspace select when list data is available, and preserved the raw workspace ID fallback when the picker cannot be populated. Folder picker work is explicitly deferred to PR7B because the Web Clipper folder destination uses `note_folders` internally but no public notes-folder list/create endpoint was found. Verification: `bunx vitest run src/services/__tests__/tldw-api-client.workspace-api.test.ts src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx src/routes/__tests__/sidepanel-clipper.test.tsx` passed with 31 tests; `git diff --check` passed. Browser sidepanel verification remains needs-verification because this surface is an extension sidepanel route rather than the routable `/notes` WebUI page. Bandit was not applicable because no Python files changed.
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
