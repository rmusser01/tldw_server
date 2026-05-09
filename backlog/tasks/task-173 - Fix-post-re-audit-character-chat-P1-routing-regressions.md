---
id: TASK-173
title: Fix post-re-audit character-chat P1 routing regressions
status: Done
assignee: []
created_date: '2026-05-09 17:51'
updated_date: '2026-05-09 18:20'
labels: []
dependencies: []
documentation:
  - Docs/Reviews/CHARACTER_CHAT_WEBUI_UX_REAUDIT_2026_05_09.md
  - >-
    Docs/superpowers/plans/2026-05-09-character-chat-route-aware-onboarding-plan.md
  - Docs/superpowers/plans/2026-05-09-character-chat-intent-preservation-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the live-browser regressions found in the 2026-05-09 character-chat UX re-audit: first-run character intent is preempted by the outer WebUI splash, and row-level Chat as can still leave character context when the selected model is stale or model availability has not resolved. Keep the fix scoped to preserving character-chat task context for first-time and returning users.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Direct first-time entry to /characters or /?intent=character-chat&returnTo=/characters bypasses the generic Build Your Assistant splash and reaches the character-chat onboarding lane or preserved setup route.
- [x] #2 The WebUI first-run Get Started path preserves character-chat intent instead of routing character-chat users to /persona.
- [x] #3 Row-level Chat as preserves the selected character and shows the in-context character-chat setup blocker when no usable chat model list is available or a stale selected model is not available.
- [x] #4 Existing non-character first-run setup behavior remains unchanged.
- [x] #5 Regression tests cover the character-intent first-run gate and stale-selected-model/no-available-model row chat path.
- [x] #6 Focused frontend tests and UI typecheck pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the two P1 re-audit regressions with focused tests.
2. Patch the outer WebUI first-run gate so character-chat intent bypasses the generic splash and keeps a canonical setup route.
3. Patch Characters row chat readiness so unresolved model catalog state cannot treat stale selected model state as ready.
4. Run focused Vitest checks, pinned UI typecheck, diff hygiene, and record any baseline-only blockers.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: direct `/characters` and explicit `/?intent=character-chat&returnTo=...` routes were intercepted by the outer Next `_app` `FirstRunGate` before the package-level route-aware onboarding code could render. The row-level Chat as path already blocked null selected model state, but it treated an unresolved model catalog as acceptable when a stale selected model value existed, allowing navigation to generic home before readiness was confirmed.

RED verification: `bunx vitest run __tests__/app/app-layout.test.tsx --testTimeout=30000` failed the two new character-chat app gate tests because `data-bypass` remained false. RED verification: `bunx vitest run src/components/PersonaGarden/__tests__/FirstRunGate.test.tsx src/components/Option/Characters/__tests__/Manager.first-use.test.tsx --testTimeout=30000` failed the new FirstRunGate bypass test and stale-selected-model row chat test.

GREEN verification: focused FirstRunGate, app layout, and Characters stale-model/row-chat Vitest checks passed. `../../tldw-frontend/node_modules/.bin/tsc --noEmit -p tsconfig.json --pretty false` passed from `apps/packages/ui`. Puppeteer/Chrome smoke `node /private/tmp/character-p1-smoke.mjs` passed and saved evidence under `Docs/Reviews/assets/2026-05-09-character-chat-p1-smoke`. `git diff --check` passed. Direct `apps/tldw-frontend` typecheck still fails on pre-existing app-wide baseline errors outside this patch; recorded as a known skip/blocker, not a regression from TASK-173. Bandit skipped because touched runtime code is TypeScript/React and no Python files changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the post-re-audit character-chat P1 routing regressions by allowing character-chat routes to bypass the outer generic first-run splash, preserving canonical character-chat setup routing from the WebUI gate, and treating unresolved model catalog state as unavailable for row-level Chat as handoff. Added regression coverage for the first-run gate and stale-selected-model row-chat blocker. Verified focused Vitest tests, pinned UI typecheck, diff hygiene, and a Puppeteer/Chrome smoke using an isolated backend profile. Direct apps/tldw-frontend typecheck still has unrelated baseline errors outside this patch.
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
