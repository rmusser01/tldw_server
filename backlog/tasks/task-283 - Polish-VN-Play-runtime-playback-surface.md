---
id: TASK-283
title: Polish VN Play runtime playback surface
status: Done
assignee: []
created_date: '2026-05-12 02:17'
labels:
  - vn-play
  - webui
  - runtime-playback
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1587'
  - 'https://github.com/rmusser01/tldw_server/issues/1391'
documentation:
  - Docs/API/VN.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1587: improve the main VN Play WebUI runtime playback surface so it consumes backend-provided scene visuals, generated choices, warnings, fallback states, and recovery copy without duplicating backend VN generation, asset-resolution, branch, or moderation rules.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Main VN Play scene surface renders backend-resolved background depth and sprite payloads with user-safe metadata and fallbacks.
- [x] #2 Generated visible choices are presented as first-class play actions and submit through existing backend turn APIs with idempotency and scene-version handling.
- [x] #3 Backend warning and recovery states such as missing visual assets no visible choices stale scene and turn in progress are visible as user-safe messages.
- [x] #4 Scripted generation inspector remains linked for audit/debug workflows but is not required for normal play.
- [x] #5 Focused frontend tests cover scene visual rendering generated choice flow fallback/warning states inspector separation and recoverable error copy.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Extend the focused scene component tests first for backend scene metadata, missing visual fallback copy, generated-choice labels, and inspector separation.
2. Improve `SceneStage` to render backend-owned visual metadata and warning/fallback copy from scene payloads without introducing frontend asset lookup rules.
3. Improve `ChoicePanel` to distinguish generated choices from authored choices using existing choice metadata only, preserve backend turn submission, and improve no-choice/in-progress copy.
4. Adjust `VNPlayWorkspace` layout/copy so normal play remains primary and the generation inspector remains an audit/debug link, not a required play step.
5. Run focused VN tests, targeted frontend lint, `git diff --check`, and document TypeScript/Bandit status.
<!-- SECTION:PLAN:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
- Created from GitHub issue #1587 after PR #1584 merged and #1578 closed.
- Implemented a focused WebUI runtime playback polish slice: `SceneStage` now renders backend-provided scene metadata, depth layers, sprite labels, visual fallback copy, and clearer warning details without asset lookup logic. `ChoicePanel` now labels generated choices from backend metadata while preserving existing `submitVNPlayTurn` behavior. `VNPlayWorkspace` tests lock the boundary that normal playback works while the generation inspector remains a separate audit/debug link.
- Verification: `bun run test:run __tests__/vn-play/SceneStage.test.tsx __tests__/vn-play/VNPlayWorkspace.test.tsx __tests__/vn-play/vnPlayRuntime.test.ts` passed with 40 tests; targeted frontend lint exited 0 with existing repo-wide warnings only; `git diff --check` passed. `bunx tsc --noEmit --pretty false` still fails only on existing `packages/ui` baseline errors in `EmbeddingsModelSelectionConfig.tsx` and `persona-visuals.ts`. Bandit skipped because only TypeScript/React frontend files and documentation/task metadata changed.
- PR #1590 review fixes: filtered invalid sprite URLs before rendering so sprite-only scenes keep consecutive alt labels and do not show the no-visuals fallback; generated-choice badges now also honor top-level `choice.source === "generated"`. Regression verification: `bun run test:run __tests__/vn-play/SceneStage.test.tsx -t 'does not show the no-visuals fallback|labels generated choices from the top-level source field'` passed; full focused VN test command passed with 42 tests.
<!-- SECTION:NOTES:END -->

## Final Summary
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Polished the VN Play runtime playback surface by rendering backend-provided visual metadata/fallbacks in `SceneStage`, labeling generated choices from backend metadata in `ChoicePanel`, and adding workspace coverage to keep normal play usable while the generation inspector remains a separate audit/debug route. The implementation keeps prompt generation, asset resolution, branch semantics, moderation, and debug state owned by the backend.
<!-- SECTION:FINAL_SUMMARY:END -->
