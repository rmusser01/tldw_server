---
id: TASK-418.13
title: Implement WebUI responsive shell and landmarks
status: Done
labels:
- ux
- webui
- extension
- implementation
- responsive
- accessibility
priority: high
parent_task_id: TASK-418
documentation:
- Docs/superpowers/plans/2026-05-17-webui-responsive-landmarks-implementation-plan.md
- Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md
- Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
modified_files:
- apps/tldw-frontend/e2e/smoke/stage4-responsive-landmarks.spec.ts
- apps/tldw-frontend/e2e/smoke/smoke.setup.ts
- apps/tldw-frontend/e2e/smoke/composer-mobile-viewport.spec.ts
- apps/tldw-frontend/package.json
- apps/tldw-frontend/components/layout/WebLayout.tsx
- apps/packages/ui/src/components/Common/PageShell.tsx
- apps/packages/ui/src/components/Layouts/SettingsOptionLayout.tsx
- apps/packages/ui/src/components/Layouts/__tests__/settings-layout-focus-order.test.tsx
- apps/packages/ui/src/routes/option-chat.tsx
- apps/packages/ui/src/components/Option/Prompt/index.tsx
- apps/packages/ui/src/components/Option/Sources/SourcesWorkspacePage.tsx
- apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx
- apps/packages/ui/src/components/Option/STT/SttPlaygroundPage.tsx
- apps/packages/ui/src/components/Option/STT/__tests__/SttPlaygroundPage.test.tsx
- apps/packages/ui/src/components/Option/TTS/TtsPlaygroundPage.tsx
- apps/packages/ui/src/components/Option/TTS/__tests__/TtsPlaygroundPage.defaults.test.tsx
- apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx
- apps/packages/ui/src/components/Option/Speech/__tests__/SpeechPlaygroundPage.render.test.tsx
- apps/packages/ui/src/components/Option/ChatWorkspace/ChatWorkspacePage.tsx
references:
- https://github.com/rmusser01/tldw_server/pull/1839
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the WP4 responsive shell and page landmarks remediation slice for WebUI/extension. Scope: add a route-level responsive landmark smoke gate, fix shared shell constraints first, then address only route-local overflow/heading failures for /chat, /media, /settings, /settings/model, /prompts, /workspace-playground, /setup, /sources, /mcp-hub, /stt, /tts, and /chat-workspace. Preserve existing product intent and avoid broad visual redesign or backend API changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Responsive landmark smoke gate covers the WP4 route matrix at 390px and asserts one semantic h1 per settled route.
- [x] #2 Shared WebUI shell constraints prevent page-level horizontal overflow without hiding reachable controls.
- [x] #3 Settings navigation is mobile-safe while preserving taxonomy, focus order, active route state, and filtering.
- [x] #4 Chat route exposes one semantic h1 and preserves sticky composer behavior at narrow width.
- [x] #5 Media route is structurally mobile-safe without breaking desktop split behavior.
- [x] #6 Route-local surfaces in scope expose semantic headings and constrain overflow locally where needed.
- [x] #7 Focused Vitest and Playwright checks pass or environment-specific skips are documented with evidence.
- [x] #8 Browser QA observations are recorded for changed routes.
- [x] #9 No backend API changes, broad visual redesign, or unrelated cleanup are included.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Baseline before WP4 product-code edits: created clean worktree `codex/webui-responsive-landmarks` from `origin/dev` at merge commit `65430b962`. Main checkout remains dirty and unrelated. `bun install` at repo root found no root package installs; `bun install` in `apps/tldw-frontend` installed local frontend dependencies without tracked lockfile changes. Baseline command `TLDW_WEB_CMD='bun run dev -- -H 127.0.0.1 -p 8080' bunx playwright test e2e/smoke/route-contract-stage2.spec.ts --reporter=line` was run after the sandboxed default bind failed with `listen EPERM 0.0.0.0:8080`. Static metadata checks passed, but the browser route-contract loop failed pre-existing local runtime readiness on `/connectors`: the page stayed on `Retrying server readiness` / disabled `Waiting` instead of mounting `route-placeholder-panel`. Evidence: `apps/tldw-frontend/test-results/smoke-route-contract-stage-33157-surface-and-do-not-misroute-chromium/error-context.md`.

TDD red gate captured for `apps/tldw-frontend/e2e/smoke/stage4-responsive-landmarks.spec.ts` with command `TLDW_WEB_CMD='bun run dev -- -H 127.0.0.1 -p 8080' bunx playwright test e2e/smoke/stage4-responsive-landmarks.spec.ts --reporter=line`. Result: 9 failed, 3 passed. Missing `h1` failures: `/chat`, `/settings`, `/settings/model`, `/sources`, `/mcp-hub`, `/stt`, `/tts`, `/chat-workspace`. `/prompts` exposed the existing `Prompts` h1 but failed page-level overflow with `rootScrollWidth` 497 at 390px. Passing routes at red baseline: `/media`, `/workspace-playground`, `/setup` or redirected setup state. Evidence files are under `apps/tldw-frontend/test-results/smoke-stage4-responsive-la-*`.

Implemented route-level h1 coverage, shared min-width containment, mobile-safe settings nav grouping, prompt mobile segmented overflow containment, sticky-chat viewport constraint, and smoke setup bypasses for non-under-test modals/tours. Browser diagnostic confirmed `/chat` now uses `data-chat-scroll-owner="transcript"` and keeps document height at the viewport while the transcript scrolls internally.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented WP4 responsive shell and landmarks. Added a 12-route Playwright gate for one h1 plus no page-level horizontal overflow at 390px, included it in the Stage 4 smoke script, fixed /chat sticky composer viewport containment, added route headings for settings/model/sources/MCP/STT/TTS/chat workspace, removed settings mobile nav forced width, constrained prompt mobile tabs, and stabilized smoke setup against backend/tour overlays. Verification: focused Vitest 9 files/55 tests passed; final affected Vitest 4 files/22 tests passed; responsive landmark Playwright 12/12 passed; updated Stage 4 smoke 28 passed/1 skipped; adjacent chat/sidebar/accessibility smoke 12/12 passed; design-system state verifier passed; git diff --check passed. PR: https://github.com/rmusser01/tldw_server/pull/1839. Bandit skipped because this slice only touches frontend TypeScript/TSX, Playwright tests, package script, and Backlog task metadata.
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
