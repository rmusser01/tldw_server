---
id: TASK-15571
title: Give Research Workspace a compact notebook layout
status: Done
labels:
- webui
- ux
- research-workspace
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved notebook layout in the existing shared owner workspace so research sources, conversations and saved outputs have priority on desktop, tablet and phone.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Three panes fit available desktop space and reachable tabs cover constrained widths with accessible pane handoffs.
- [x] #2 Compact header, source organization and contextual chat controls reclaim reading space while preserving existing actions and recovery.
- [x] #3 Studio leads with saved outputs, exposes generation on demand and avoids fixed empty sections.
- [x] #4 Verify owner workspace behavior, shared-client contracts and desktop/tablet/phone rendering.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR check: no new ADR. Docs/ADR/007-research-workspace-canonical-first-slice-shell.md governs the existing shared shell. No API or data ownership change. Plan stages: shell/navigation, sources/chat/studio, cross-client verification and review.
Approved mockup reviewed 2026-11-17. Implementation in codex/research-notebook-layout. Verify existing linked WebUI/extension contract. Stages and objectives in IMPLEMENTATION_PLAN_research_notebook_layout.md during development.
Implemented the shell and pane changes. Browser checks cover source intake, keyboard RAG, tablet citation preview with draft preservation, Quick Notes keyboard focus, Review/Download, generation cancellation/reload Retry, desktop restore rails and Chrome extension parity. Independent review found first-visit Studio discussion, cross-workspace draft isolation, and constrained-width tour targets; fixing and verifying all three before delivery.
Independent review fixes verified. Chat receives Studio discussions on first entry, drafts are scoped to the active chat, and every workspace tour pane has a visible tab fallback. The shared workspace and tutorial suite passed 1,530 tests across 83 files. Chromium passed all 18 WebUI browser workflows together, including desktop/tablet/phone navigation and light/dark composer readability. The production Chrome extension build and its shared parity workflow passed. Frontend TypeScript, scoped ESLint, and git diff --check passed. Bandit ran on the touched scope, with zero Python lines and zero findings. Live provider and real-backend generation were not exercised. User-facing guidance and tour targets were updated in the UI.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the approved compact notebook in the shared owner route. Source intake and selection, grounded and general chat, citations, model recovery, all Studio generators and saved-output actions, Quick Notes, search, workspace management, and keyboard navigation remain accessible. The layout uses actual workspace width, retains mounted pane state, and supports tablet side-by-side comparison. Source filtering and citation focus preserve user context. No backend, storage schema, or dependency changes.
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
