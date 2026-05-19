---
id: TASK-412
title: Fix Workspace Playground Add Sources media library listing
status: Done
labels:
- webui
- ux
- workspace-playground
priority: medium
modified_files:
- apps/packages/ui/src/components/Option/WorkspacePlayground/SourcesPane/index.tsx
- apps/packages/ui/src/components/Option/WorkspacePlayground/ChatPane/index.tsx
- apps/packages/ui/src/components/Option/WorkspacePlayground/SourcesPane/media-library-normalization.ts
- apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/SourcesPane.stage2.test.tsx
- apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/AddSourceModal.stage2.intake.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure the Workspace Playground Add Sources > My Media tab exposes the user's existing media library correctly, including backend pagination totals and load-more behavior, so users with large libraries do not see an empty or truncated picker.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add Sources entry points expose the My Media library first.
- [x] #2 My Media handles backend pagination.total_items for large media libraries.
- [x] #3 Focused source/add-source tests cover the media-library entry and pagination behavior.
- [x] #4 Browser verification confirms the Workspace Playground modal opens on My Media with the local library visible.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a failing regression for the backend media-list response shape with pagination.total_items so My Media shows the correct total and load-more affordance.
2. Update Workspace media-library normalization to read the backend's canonical pagination total fields.
3. Run the AddSourceModal-focused Vitest suite and browser-check /workspace-playground Add Sources if local server state allows.
4. Record verification and any environment limitations in the Backlog task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Changed Workspace Playground source entry points so the left-panel Add Sources button, empty-source CTA, and chat empty-state Add Sources CTA open the Add Sources modal directly on the My Media tab. Added regression coverage proving the primary Add Sources action requests the existing-media tab. Hardened media-library response normalization to read pagination.total_items so large backend media libraries preserve the correct total and load-more behavior even when only the backend total_items field is present. Verification: SourcesPane.stage2 passed (19 tests); AddSourceModal.stage2 plus SourcesPane.stage2 passed (32 tests); the broader focused AddSourceModal/SourcesPane/media-library-normalization suite passed (10 files, 66 tests); browser smoke at http://localhost:3000/workspace-playground confirmed Add Sources opens with My Media active, search visible, and the local media library showing 50 of 918. git diff --check passed. Full TypeScript remains blocked by pre-existing repo-wide baseline errors, but filtered compiler output for the touched Workspace source/add-source files returned no matching errors. Bandit not applicable because touched code is frontend TypeScript/TSX only.
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
