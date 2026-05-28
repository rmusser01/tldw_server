---
id: TASK-517
title: Add chat cockpit rail regression guards
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-27 04:05
labels:
- chat
- frontend
- test
dependencies: []
documentation:
- Docs/superpowers/plans/2026-05-27-chat-rails-ux-rebaseline-implementation-plan.md
priority: high
modified_files:
- apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-regression.guard.test.ts
- apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx
- apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts
- Docs/Reviews/CHAT_RAILS_UX_REBASELINE_2026_05_27.md
- Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/evidence.json
- backlog/tasks/task-517 - Add-chat-cockpit-rail-regression-guards.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add focused regression coverage that proves the main /chat cockpit shell, context rail, runtime inspector, mobile rail panels, focus mode, and character rail remain wired on the origin/dev chat baseline before user-facing UX fixes proceed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A source-level regression guard proves the /chat page still imports and renders the cockpit shell, context rail, runtime inspector, mobile rail surface, focus mode affordance, and character rail.
- [x] #2 Existing cockpit component and real-server e2e coverage is run or updated to verify rail visibility on desktop and mobile.
- [x] #3 Screenshot artifacts for rail-enabled /chat are copied into the review asset directory for the refreshed UX audit.
- [x] #4 Verification results, skips, and any baseline failures are recorded in this task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 2 implementation notes - 2026-05-27:
- Added `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-regression.guard.test.ts` as a source-level guard using `import.meta.url`, `fileURLToPath`, and `path.resolve` to read `../Playground.tsx` and `../PlaygroundCockpitShell.tsx`.
- Guard asserts `Playground.tsx` still imports/uses and renders `PlaygroundCockpitShell`, `PlaygroundContextRail`, `PlaygroundRuntimeInspector`, and `CharacterControlRail`.
- Guard asserts `PlaygroundCockpitShell.tsx` still exposes `playground-cockpit-shell`, left/right/mobile rail test ids, `Enter focus chat`, and `Show cockpit panels`.
- Guard asserts `Playground.tsx` keeps `playgroundChatLayoutMode`, `"cockpit"`, and `"focus"`.
- Fixed the Task 2 existing rail-suite blocker in `Playground.cockpit-controls.test.tsx` by adding focused harness mocks for `@/hooks/useServerChatHistory`, `@/services/tldw-server`, and `@/services/chat-settings`; the controls tests now cover representative tracked and plain server chat history without requiring React Query integration.
- The cockpit server-readiness tests now clear the default selected character/assistant before rendering so they assert the cockpit health strip instead of character-chat readiness behavior.

Verification:
- `cd apps/packages/ui && bun install` completed earlier to restore missing local Vitest dependencies after the first guard run failed to resolve `vitest/config`.
- `cd apps/packages/ui && bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-regression.guard.test.ts`: PASS, 1 file, 3 tests.
- Initial existing rail-suite run failed because `Playground.cockpit-controls.test.tsx` rendered `Playground`, which now reaches `CharacterControlRail` / character sessions and `useServerChatHistory`; that test file had no `@/hooks/useServerChatHistory` mock and no `QueryClientProvider`, so React Query threw `No QueryClient set` in 11 tests.
- `cd apps/packages/ui && bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx`: PASS, 1 file, 16 tests after the harness fix.
- `cd apps/packages/ui && bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx src/components/Option/Playground/__tests__/CharacterControlRail.test.tsx`: PASS, 6 files, 99 tests.

DoD notes:
- AC #1 is complete for this Task 2 slice.
- AC #2 is partially complete: the existing cockpit component rail set now passes; real-server/screenshot evidence remains for Task 3.
- AC #3 remains open for Task 3 screenshot/evidence work.
- AC #4 is complete for this Task 2 slice.
- Bandit is not applicable for this TypeScript test and Backlog-only slice; no Python code was touched.
- The prior existing rail-suite blocker is resolved; no new production-code changes were made.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Task 3 implementation notes - 2026-05-27:
- Added `assertNoHorizontalOverflow(page)` to `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`; it evaluates `window.innerWidth`, `document.documentElement.scrollWidth`, and `document.body.scrollWidth`, then expects document/body scroll width to stay within `innerWidth + 1`.
- Wired the overflow assertion into stable desktop states after initial `/chat` cockpit render, desktop focus mode, and return to cockpit; wired it into mobile states after `/chat` load, cockpit context panel, runtime panel, return to focus, and return to cockpit.
- Coordinator-confirmed backend health succeeded with approved localhost access: `curl -sf http://127.0.0.1:8000/api/v1/health` returned status `ok`, auth mode `single_user`, and healthy database/metrics/ChaChaNotes checks.
- Agent-side sandboxed curl to the same health URL still failed with exit code 7 and no response body; recorded as sandbox-localhost access failure, not backend downtime.
- Planned live real-server Playwright command was first attempted inside the sandbox with `TLDW_WEB_CMD='bun run dev -- -H 127.0.0.1 -p 18014'`; it failed before tests because the sandbox blocked Next from binding `127.0.0.1:18014` (`listen EPERM`).
- Coordinator reran the planned live real-server Playwright command with approved localhost access. The run reached the backend and WebUI and produced 11 passing tests, with 4 existing/non-rail failures: stale Web search status-strip expectation, character clear state, disposable plain chat creation returning 422, and missing model-list scope toggle.
- Coordinator then started the WebUI with approved localhost access and manually captured the required rail-enabled screenshots from `/chat`: `desktop-cockpit.png`, `desktop-focus.png`, `mobile-focus.png`, and `mobile-cockpit.png`.
- Manual browser overflow metrics were clean in all four audit states: desktop cockpit, desktop focus, mobile focus, and mobile cockpit all had document/body scroll width equal to viewport width.
- The exact required focused static/unit Playwright command also failed before tests because Playwright config autostarted Next on `0.0.0.0:8080` and hit `listen EPERM`. Re-running the same grep with `TLDW_WEB_AUTOSTART=false` and the fake e2e API key passed: 2 tests, 595 ms.
- `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/evidence.json` now records coordinator-confirmed backend health, sandbox curl failure, escalated real-server Playwright results, focused static verification, manual browser capture, and captured viewport/screenshot entries.

DoD notes for Task 3:
- AC #2 is complete for this task scope: existing component coverage was completed in Task 2, and the real-server e2e coverage is now updated with desktop/mobile overflow assertions. Escalated live execution reached the real WebUI/backend and exposed four non-rail baseline failures.
- AC #3 is complete: evidence JSON exists and the required `/chat` screenshot artifacts are present in the review asset directory.
- Documentation was updated in the audit and evidence JSON.
- Bandit remains not applicable for this TypeScript/evidence-only slice; no Python code was touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 2 added the source-level cockpit rail wiring guard and fixed the cockpit-controls harness blocker exposed by the restored rails. The new guard passes, the isolated controls file passes, and the existing rail component set now passes. Screenshot/evidence criteria remain open for Task 3.

Task 3 hardened the real-server cockpit spec with horizontal overflow assertions and updated the audit/evidence scaffolding. Coordinator-confirmed backend health was OK. The full escalated real-server Playwright run reached the backend/WebUI and produced 11 passing tests with 4 existing/non-rail failures. Manual browser verification captured the four required `/chat` screenshots and measured no horizontal overflow in desktop cockpit, desktop focus, mobile focus, or mobile cockpit. Focused static/source Playwright verification passed with WebUI autostart disabled and the fake e2e API key.
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
