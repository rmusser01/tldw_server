---
id: TASK-12131
title: Add Chat Workspace live-backend browser smoke coverage
status: Done
assignee: []
created_date: ''
updated_date: 2026-07-03 19:52
labels:
- WebUI
- Front-End
- ChatWorkspace
- E2E
dependencies: []
references:
- https://github.com/rmusser01/tldw_server/issues/2035
- https://github.com/rmusser01/tldw_server/issues/1239
- https://github.com/rmusser01/tldw_server/pull/2595
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #2035 for the Chat Workspace epic #1239: add focused Playwright coverage that opens /chat-workspace in an authenticated shell, exercises workspace-scoped send behavior against a live or realistic backend, verifies streaming/stop, error recovery, staged context, and links the release gate to this focused proof.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 WebUI Playwright coverage opens /chat-workspace in an authenticated shell against a live or realistic test backend.
- [x] #2 Coverage selects or seeds a usable model and persona/assistant state where applicable.
- [x] #3 Coverage sends a workspace-scoped message and verifies the request carries workspace scope.
- [x] #4 Coverage observes streaming or a deterministic mocked streaming state.
- [x] #5 Coverage verifies stop generation when streaming is active.
- [x] #6 Coverage exercises send failure/error recovery without clearing draft or staged context.
- [x] #7 Coverage stages at least one workspace source and verifies structured media ids or fallback summary behavior.
- [x] #8 The release gate references this focused proof rather than relying only on backend-unavailable route smoke.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-30-chat-workspace-live-backend-smoke-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented #2035 focused Chat Workspace browser smoke coverage.

Touched files:
- apps/tldw-frontend/e2e/smoke/chat-workspace-live-backend.spec.ts
- apps/tldw-frontend/e2e/smoke/stage5-release-gate.spec.ts
- apps/tldw-frontend/e2e/smoke/smoke.setup.ts
- Docs/superpowers/plans/2026-06-30-chat-workspace-live-backend-smoke-plan.md

Implementation notes:
- Added deterministic authenticated /chat-workspace Playwright coverage with seeded workspace, selected model, overlay persona, and staged sources.
- Backend fixture covers health/config/model/chat/RAG endpoints, captures request bodies, returns selected-source generated RAG answer, delays streaming so Stop generation is visible, and returns deterministic streaming failure for recovery assertions.
- Assertions verify workspace scope in chat creation, structured include_media_ids for ready staged media, fallback context injection for staged source without valid media id, stop button during active streaming, and draft/staged context preservation after failure.
- Stage 5 release gate now requires /chat-workspace to reference the focused proof spec and narrowly allowlists no-backend startup probes for notifications/persona on route-only smoke.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed focused #2035 Chat Workspace live-backend browser smoke coverage and release-gate reference.

Post-review update: made the smoke fixture deterministic by replacing `Date.now()` with the fixed seeded timestamp, seeded the current split workspace persistence keys for reliable workspace hydration, and made the streaming assertion wait for the captured chat-create request before checking scope. Stage 6 chat route smoke now stubs model/provider metadata so it tests the theme-toggle interaction without unrelated offline backend overlays.

Verification: npx playwright test e2e/smoke/chat-workspace-live-backend.spec.ts --project=chromium passed 4 tests; npx playwright test e2e/smoke/stage5-release-gate.spec.ts --project=chromium --grep "Chat Workspace" passed 1 test; npx playwright test e2e/smoke/stage6-interaction-stage1.spec.ts --project=chromium passed 2 tests; git diff --check passed. Bandit is not applicable because no Python files were touched.
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
