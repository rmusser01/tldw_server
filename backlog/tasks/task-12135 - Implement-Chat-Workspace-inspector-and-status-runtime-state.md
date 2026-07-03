---
id: TASK-12135
title: Implement Chat Workspace inspector and status runtime state
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-13 20:16'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/2033'
  - 'https://github.com/rmusser01/tldw_server/issues/1239'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #2033 for /chat-workspace: replace placeholder inspector/status rail behavior with accurate runtime and workspace state, degraded/offline recovery copy, hydration/send-disabled safety, and focused component/browser coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-03-chat-workspace-status-rails-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PR #2600 review remediation after rebase on latest origin/dev: added explicit hasModelSelected runtime state, made WorkspaceStatusStrip and InspectorRail require workspaceReady, verified selectedModelLabel is passed into InspectorRail, and strengthened stop-generation smoke coverage so delayed stream output does not land after abort. Verification: ChatWorkspace Vitest folder passed 8 files/74 tests; chat-workspace live-backend Playwright smoke passed 4/4; Stage 5 Chat Workspace release gate passed 1/1; git diff --check passed; UI package tsc --noEmit still fails on unrelated baseline outside Chat Workspace, and captured-log grep found no ChatWorkspace or rail errors; Bandit not applicable because no Python source changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented #2033 status/inspector runtime slice and PR #2600 review remediation: added sendError and explicit hasModelSelected runtime state, lifted failed-send state from WorkspaceChatPanel, passed workspace readiness and runtime setup state into WorkspaceStatusStrip and InspectorRail, required workspaceReady at rail boundaries, removed inactive approval/task-progress placeholder panels, added recovery copy for server unavailable/workspace hydration/send failure/missing model/no persona, and extended the live-backend smoke with visible streaming/failure rail assertions plus no-delayed-output stop-generation coverage. Verification: ChatWorkspace Vitest folder passed 8 files/74 tests; live-backend Playwright smoke passed 4/4; Stage 5 Chat Workspace release gate passed 1/1; git diff --check passed. UI package tsc --noEmit still fails on unrelated baseline errors outside touched Chat Workspace files; captured-log grep found no ChatWorkspace/InspectorRail/WorkspaceStatusStrip/WorkspaceChatPanel/ChatWorkspacePage errors. Bandit not applicable because no Python source changed.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
