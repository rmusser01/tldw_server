---
id: TASK-291
title: Implement main /chat cockpit first-slice direct controls
status: Done
assignee: []
created_date: '2026-05-12 04:44'
updated_date: '2026-05-12 05:04'
labels:
  - webui
  - chat
  - frontend
  - cockpit
dependencies:
  - TASK-290
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1582'
documentation:
  - >-
    Docs/superpowers/plans/2026-05-12-main-chat-cockpit-first-slice-implementation-plan.md
  - Docs/superpowers/specs/2026-05-12-main-chat-cockpit-controls-gap-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first slice defined by the main /chat cockpit controls plan. The work is limited to the main WebUI /chat page and must preserve existing composer workflows while making the cockpit rails/status strip operate on the same chat state. Do not touch the browser-extension sidepanel/sidebar or broaden into the later cockpit maturity backlog.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Main /chat cockpit rails expose first-slice controls for web search, Search & Context entry, temporary/saved session state, model/provider summary, model settings, character/persona entry, and runtime/status state without duplicating chat state.
- [x] #2 Focus mode keeps the existing chat/composer workflow available and does not render cockpit rails.
- [x] #3 Component and integration tests prove cockpit controls use the same state paths as existing composer/dialog controls.
- [x] #4 Real-server browser smoke coverage verifies at least one state-changing cockpit action on /chat without mocked server data, page.route, synthetic API payloads, sidepanel/sidebar routes, or Computer Use.
- [x] #5 Verification results and any known skips are recorded before completion.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the first slice of main WebUI /chat cockpit direct controls in .worktrees/chat-degraded-health. Added a shared cockpit action bridge, wired context rail controls to existing PlaygroundForm handlers, added direct web-search and temporary/saved session rail controls, separated provider/model runtime display, added model and character settings callbacks, and expanded the status strip with provider/model/context/persistence/degraded/error state. Scope stayed on main /chat only; no browser-extension sidepanel/sidebar code was touched.

Verification:
- Focused Vitest passed: bunx vitest run src/components/Option/Playground/__tests__/playground-cockpit-actions.test.ts src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx src/components/Option/Playground/__tests__/PlaygroundStatusStrip.first-slice.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx. Result: 6 files / 23 tests passed. Existing stderr remains mocked tldw server not configured noise in Playground.cockpit-controls.
- Real-server Playwright passed against http://127.0.0.1:8000 using the live backend and no mocked API data: TLDW_E2E_SERVER_URL=http://127.0.0.1:8000 TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY TLDW_SERVER_URL=http://127.0.0.1:8000 NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 TLDW_WEB_URL=http://localhost:18014 TLDW_WEB_CMD='bun run dev -- -H 127.0.0.1 -p 18014' bunx playwright test e2e/workflows/chat-cockpit.real-server.spec.ts --project=chromium --reporter=line. Result: 3 passed. Backend health was degraded only for chacha_notes and /chat admitted with warnings.
- git diff --check passed.
- rg confirmed no page.route( usage in chat-cockpit.real-server.spec.ts.
- TypeScript check attempted: bunx tsc --noEmit --pretty false --project tsconfig.json in apps/packages/ui. It failed on existing repo-wide baseline errors across audio/composer/common/flashcards/onboarding/workspace/services/etc. No reported errors were in the new first-slice cockpit files.

Bandit skipped because this slice touched frontend TypeScript/TSX/Playwright and Markdown task files only; no Python code was touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the first slice of main /chat cockpit direct controls. The cockpit rails now use shared action wiring for web search, Search & Context, temporary/saved session intent, model settings, and character settings; the runtime rail separates provider/model display and supports explicit runtime status details; the status strip now reflects provider/model, context, persistence, degraded, and error state. Added focused component/integration tests plus real-server Playwright coverage proving a state-changing cockpit action and real chat attempt on /chat without mocked server data.
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
