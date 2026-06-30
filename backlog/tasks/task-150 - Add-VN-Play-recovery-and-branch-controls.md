---
id: TASK-150
title: Add VN Play recovery and branch controls
status: Done
assignee: []
created_date: '2026-05-09 04:29'
updated_date: '2026-05-09 04:37'
labels:
  - vn-play
  - webui
  - implementation
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1401'
  - 'https://github.com/rmusser01/tldw_server/issues/1391'
documentation:
  - Docs/superpowers/specs/2026-05-01-vn-play-runtime-design.md
  - Docs/superpowers/plans/2026-05-01-vn-play-runtime-implementation.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next VN Play WebUI usability slice from issue #1401: expose checkpoint, restore, branch metadata, retry-last-turn, and conflict recovery controls in the existing VN Play workspace. Keep this focused on the frontend/runtime UX over the already-merged backend endpoints; avoid realtime image generation or new backend orchestration unless a small API/test fix is required.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Loads checkpoints and branches for the selected VN Play session and passes them into the runtime inspector
- [x] #2 Allows creating a checkpoint from the current scene and refreshes session metadata, events, checkpoints, and branches
- [x] #3 Allows restoring a checkpoint with explicit user intent and refreshes selected session state coherently
- [x] #4 Provides retry-last-turn behavior for recoverable failed turns using a fresh idempotency key
- [x] #5 Stale scene and turn-in-progress conflicts show explicit recovery UI instead of generic errors
- [x] #6 Focused VN Play frontend tests cover checkpoint create, restore, retry, and conflict-state behavior
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect existing VN Play API helpers, types, workspace, inspector, dialogue, choice, and tests.
2. Add failing tests for checkpoint/branch loading, checkpoint creation, restore refresh, retry-last-turn, and conflict state UI.
3. Implement minimal VNPlayWorkspace/SceneInspector changes using existing API helpers and UI primitives.
4. Re-run focused Vitest tests, run touched TypeScript checks where feasible, run git diff --check, and update the Backlog task.
5. Commit, push, and open a PR against dev linked to #1401.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red verification: bunx vitest run __tests__/vn-play/VNPlayWorkspace.test.tsx from apps/tldw-frontend failed with the intended missing behavior: checkpoint/branch metadata was not loaded, checkpoint create/restore controls were absent, retry-last-turn was absent, and stale_scene_version rendered as raw error text.

Implementation: wired existing VN Play checkpoint/branch/retry API helpers into VNPlayWorkspace, added recovery controls to SceneInspector, and suppressed raw recoverable conflict text in DialoguePanel so workspace-level recovery copy owns stale/in-progress states.

Focused verification: bunx vitest run __tests__/vn-play/VNPlayWorkspace.test.tsx __tests__/vn-play/SceneStage.test.tsx __tests__/vn-play/vnPlayApi.test.ts passed with 14 tests. git diff --check passed. Full frontend TypeScript check currently fails on pre-existing unrelated apps/packages/ui/src/services/persona-visuals.ts BlobPart typing; no touched VN Play diagnostics appeared in that run.

Post-rebase verification: branch rebased cleanly onto origin/dev after 8f6f94a0b. Focused command bunx vitest run __tests__/vn-play/VNPlayWorkspace.test.tsx __tests__/vn-play/SceneStage.test.tsx __tests__/vn-play/vnPlayApi.test.ts passed with 14 tests. git diff --check origin/dev..HEAD passed. Full tldw-frontend TypeScript check still fails only on the pre-existing ../packages/ui/src/services/persona-visuals.ts BlobPart typing issue; this slice did not touch that file. Bandit skipped because this task only changes frontend TypeScript/React and Backlog metadata.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added VN Play recovery controls for the existing runtime surface. VNPlayWorkspace now loads checkpoints and branch metadata for the selected session, refreshes those collections alongside events, creates checkpoints at the current scene version, restores checkpoints with restore-scoped idempotency keys, and retries the latest failed turn with retry-scoped idempotency keys. SceneInspector now exposes checkpoint create/restore controls and branch/checkpoint listings, while DialoguePanel suppresses raw stale/in-progress conflict strings so the workspace can show explicit recovery guidance. Focused VN Play Vitest coverage passes; package-wide TypeScript remains blocked by an unrelated persona-visuals BlobPart baseline error.
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
