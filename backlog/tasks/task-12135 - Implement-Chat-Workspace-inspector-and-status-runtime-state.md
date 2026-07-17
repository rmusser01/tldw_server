---
id: TASK-12135
title: Implement Chat Workspace inspector and status runtime state
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-14 01:58'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/2033'
  - 'https://github.com/rmusser01/tldw_server/issues/1239'
documentation:
  - >-
    Docs/superpowers/specs/2026-07-13-chat-workspace-hydration-offline-follow-up-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #2033 for /chat-workspace: replace placeholder inspector/status rail behavior with accurate runtime and workspace state, degraded/offline recovery copy, hydration/send-disabled safety, and focused component/browser coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Chat Workspace remains non-ready and chat sends stay disabled while the workspace store is not hydrated, even when a workspace ID is present.
- [x] #2 Live-backend browser coverage verifies connected-to-offline rail transitions and suppresses stale streaming state.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-03-chat-workspace-status-rails-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PR #2600 review remediation after rebase on latest origin/dev: added explicit hasModelSelected runtime state, made WorkspaceStatusStrip and InspectorRail require workspaceReady, verified selectedModelLabel is passed into InspectorRail, and strengthened stop-generation smoke coverage so delayed stream output does not land after abort. Verification: ChatWorkspace Vitest folder passed 8 files/74 tests; chat-workspace live-backend Playwright smoke passed 4/4; Stage 5 Chat Workspace release gate passed 1/1; git diff --check passed; UI package tsc --noEmit still fails on unrelated baseline outside Chat Workspace, and captured-log grep found no ChatWorkspace or rail errors; Bandit not applicable because no Python source changed.

2026-07-13 PR #2600 follow-up: addressing review findings for real storeHydrated readiness and the missing offline browser transition. Work will use TDD on the existing PR branch.

2026-07-13 PR #2600 follow-up complete. TDD evidence: the new non-empty-ID/storeHydrated=false page test failed before the readiness fix, then passed after ChatWorkspacePage derived one readiness boolean from hydration plus normalized identity. Added live browser coverage that starts streaming, transitions the real connection store to unreachable, and verifies both rails show server recovery while suppressing stale streaming state. Fresh verification: Chat Workspace Vitest 8 files/75 tests passed; frontend tsc --noEmit passed; live-backend Playwright 5/5 passed; focused ESLint exited 0 with only the pre-existing PERSONA_ID warning; git diff --check passed. Bandit is not applicable because the touched implementation and tests are TypeScript/TSX only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed issue #2033 and both final PR #2600 review findings. Chat Workspace now lifts send/runtime state into accurate status and inspector rails, avoids placeholder approval/task UI, gates chat and rail readiness on both workspace-store hydration and a normalized workspace ID, and provides degraded/offline/send-failure recovery states. Unit coverage proves a persisted ID cannot enable sends or ready rails before hydration; live-backend browser coverage proves active streaming rails transition to server-unavailable state without stale streaming labels. Verification: 75/75 Chat Workspace unit tests, TypeScript, 5/5 live-backend browser tests, focused ESLint (no errors; one pre-existing warning), and git diff --check all passed. No Python changed, so Bandit was not applicable.
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
