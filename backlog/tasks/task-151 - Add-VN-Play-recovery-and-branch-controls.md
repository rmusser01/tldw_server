---
id: TASK-151
title: Add VN Play recovery and branch controls
status: Done
assignee: []
created_date: '2026-05-09 04:39'
updated_date: '2026-05-09 05:01'
labels:
  - vn-play
  - frontend
  - recovery
  - webui
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1401'
  - 'https://github.com/rmusser01/tldw_server/issues/1391'
  - 'https://github.com/rmusser01/tldw_server/pull/1404'
documentation:
  - >-
    Docs/superpowers/plans/2026-05-09-vn-play-recovery-controls-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement issue #1401: make VN Play session recovery and branching controls first-class in the WebUI. The backend already exposes checkpoint, restore, retry-last-turn, events, session, and branch endpoints; this task should wire them into the existing VN Play workspace with explicit recovery states and focused frontend coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Selected sessions load and display checkpoint and branch metadata in the runtime inspector
- [x] #2 Users can create a checkpoint from the current scene and see the refreshed checkpoint list
- [x] #3 Users can restore a checkpoint and the selected session, scene state, events, checkpoints, and branches refresh coherently
- [x] #4 Users can retry the last recoverable failed/interrupted turn with a fresh idempotency key without duplicating the original action
- [x] #5 stale_scene_version and turn_in_progress conflicts show explicit recovery UI with reload/poll actions instead of generic errors
- [x] #6 Focused frontend tests cover checkpoint create, checkpoint restore, retry-last-turn, branch/checkpoint loading, and conflict-state recovery behavior
- [x] #7 Smoke/E2E coverage is updated where feasible, or exact environment blockers are documented
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan saved at Docs/superpowers/plans/2026-05-09-vn-play-recovery-controls-implementation-plan.md. Scope is frontend-first: load branch/checkpoint metadata, add checkpoint create/restore UX, add retry/stale/in-progress recovery controls, and verify focused VN Play tests plus smoke coverage where feasible.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented branch/checkpoint metadata loading, checkpoint create/restore controls, retry-last-turn recovery, stale_scene_version reload recovery, and turn_in_progress polling recovery in the VN Play WebUI.
Verification: bunx vitest run __tests__/vn-play/VNPlayWorkspace.test.tsx __tests__/vn-play/vnPlayApi.test.ts __tests__/vn-play/SceneStage.test.tsx -> 3 files / 15 tests passed.
Verification: TLDW_WEB_URL=http://localhost:18081 TLDW_WEB_CMD="bun run dev -- -p 18081" bunx playwright test e2e/smoke/vn-play.spec.ts --reporter=line -> 1 passed.
Verification: bunx eslint components/vn-play/VNPlayWorkspace.tsx components/vn-play/SceneInspector.tsx __tests__/vn-play/VNPlayWorkspace.test.tsx e2e/smoke/vn-play.spec.ts -> 0 errors/warnings.
Verification: git diff --check -> passed.
Bandit: skipped because this task touched only frontend TypeScript/TSX and docs/task metadata, no Python paths.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added VN Play WebUI recovery controls and branch/checkpoint inspection. The workspace now refreshes session, events, branches, and checkpoints together; the runtime inspector can create and restore checkpoints; recoverable turn failures expose retry/reload/poll actions with fresh idempotency keys. Focused Vitest, ESLint, git diff --check, and an isolated-port Playwright smoke test all pass.
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
