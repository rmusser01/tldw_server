---
id: TASK-157
title: Make VN Play session setup usable
status: In Progress
assignee: []
created_date: '2026-05-09 05:15'
labels:
  - vn-play
  - frontend
  - webui
  - setup
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1407'
  - 'https://github.com/rmusser01/tldw_server/issues/1391'
documentation:
  - Docs/API-related/VN_PLAY_API.md
  - Docs/superpowers/plans/2026-05-09-vn-play-session-setup-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement issue #1407: replace raw VN Play session setup IDs with data-driven character and VN asset-pack selection in the WebUI. Users should be able to start Freeform or Story/CYOA sessions by choosing named records, seeing compatibility/readiness/trust/content-rating warnings before submit, and falling back to manual ID entry only when selector data cannot load.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 New VN Play session dialog loads and displays selectable characters with identifying metadata
- [x] #2 Dialog loads VN asset packs and annotates or filters them by selected character compatibility
- [x] #3 Ready/approved compatible packs can be selected and create the existing VNPlaySessionCreate payload without raw ID typing
- [x] #4 Unready, draft, missing-byte, incompatible, trust-level, and content-rating conditions show explicit guidance before session creation
- [x] #5 When character or asset-pack selectors fail to load, manual ID entry remains available as a secondary fallback
- [x] #6 Empty states guide users toward character creation/import or VN asset pack preparation/review
- [x] #7 Focused frontend tests cover selector loading, compatibility behavior, warning states, fallback/manual entry, and create-session payloads
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implement in four stages: add minimal character API/types; replace raw IDs with selector-driven session creation; add readiness/compatibility/content-rating/empty-state/manual-fallback behavior; update smoke coverage and closeout verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created implementation plan Docs/superpowers/plans/2026-05-09-vn-play-session-setup-implementation-plan.md for issue #1407. The Backlog ID was moved to TASK-157 to avoid colliding with an existing TASK-155 in the main checkout.

Implemented a selector-driven VN Play session setup dialog, a minimal WebUI characters API wrapper/types, readiness and compatibility warning states, manual ID fallback on selector load failure, empty-state guidance, and selector-aware VN Play smoke mocks.

Verification recorded:
- RED: `bunx vitest run __tests__/vn-play/VNPlayWorkspace.test.tsx` failed 5 selector/warning/fallback tests before implementation.
- GREEN: `bunx vitest run __tests__/vn-play/VNPlayWorkspace.test.tsx __tests__/vn-play/vnPlayApi.test.ts __tests__/vn-play/SceneStage.test.tsx __tests__/vn-play/vnPlayRuntime.test.ts` passed 4 files / 21 tests.
- Lint: `bunx eslint components/vn-play/NewSessionDialog.tsx __tests__/vn-play/VNPlayWorkspace.test.tsx e2e/smoke/vn-play.spec.ts lib/api/characters.ts types/characters.ts` exited 0.
- Smoke: `TLDW_WEB_URL=http://localhost:18081 TLDW_WEB_CMD='bun run dev -- -p 18081' bunx playwright test e2e/smoke/vn-play.spec.ts --reporter=line` passed 1 test after rerunning outside the sandbox for port binding.
- Whitespace: `git diff --check` exited 0.
- Bandit: not applicable; touched production code is frontend TypeScript/React only.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Made VN Play session setup selectable from named character and VN asset-pack records while preserving the existing create-session payload. The dialog now loads characters, packs, and pack readiness; prefers compatible ready packs; disables incompatible/unready/draft selections; shows readiness, trust, and content-rating guidance; falls back to manual numeric IDs when selector loading fails; and updates the smoke test to cover the selector flow.
<!-- SECTION:FINAL_SUMMARY:END -->
