---
id: TASK-566
title: Address chat UX PR review comments after rebase
status: Done
labels:
- chat
- extension
- ux
- review-fix
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase PR #2160 onto latest dev and address review comments about sidepanel locale copy overriding the route-only handoff wording and E2E extension staging hardcoding the default locale.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR #2160 is rebased onto latest origin/dev.
- [x] #2 English sidepanel locale strings match the route-only /chat handoff contract so loaded i18n values do not override updated defaults.
- [x] #3 Extension launch staging reads manifest.default_locale and falls back safely to en when unavailable or invalid.
- [x] #4 Focused tests cover locale copy and staged default-locale behavior.
- [x] #5 Verification, Bandit applicability, and PR review-thread resolution are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Rebased PR #2160 onto `origin/dev` after fetching the latest branch. Rebase completed cleanly.

Addressed review comments:
- Qodo/Gemini default-locale staging: `prepareExtensionLaunchPath` now reads `manifest.json` and stages `_locales/<manifest.default_locale>/messages.json` when the manifest has a valid non-empty locale, falling back to `en` when the manifest is missing, invalid, or has an invalid value. Added RED/GREEN coverage for a non-`en` default locale.
- Qodo sidepanel i18n copy: English `sidepanel.json` now defines `header.openFullChatWebuiDescription` and updates the legacy route-only alias plus `controlRow.openFullAppDescription` / `openRolePlayFullAppDescription` to the route-only `/chat` wording. Added RED/GREEN locale-copy coverage so loaded i18n values cannot silently override the component defaults with stale copy.

Verification recorded:
- RED extension review test: `cd apps/extension && bunx vitest run tests/e2e/utils/extension-paths.test.ts` failed because `_locales/ja/messages.json` was not staged.
- RED sidepanel locale test: `cd apps/packages/ui && bun run test src/components/Sidepanel/Chat/__tests__/sidepanel-handoff-locale-copy.test.ts` failed because `header.openFullChatWebuiDescription` was missing from the English locale.
- GREEN extension target: `cd apps/extension && bunx vitest run tests/e2e/utils/extension-paths.test.ts` -> `1 passed`, `4 passed`.
- GREEN sidepanel locale target: `cd apps/packages/ui && bun run test src/components/Sidepanel/Chat/__tests__/sidepanel-handoff-locale-copy.test.ts` -> `1 passed`, `1 passed`.
- Focused extension harness suite: `cd apps/extension && bunx vitest run tests/e2e/utils/extension-paths.test.ts tests/e2e/utils/extension.launch.test.ts tests/e2e/utils/extension-build.test.ts` -> `3 passed`, `11 passed`.
- Focused sidepanel suite: `cd apps/packages/ui && bun run test src/components/Sidepanel/Chat/__tests__/sidepanel-handoff-locale-copy.test.ts src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.fullscreen-route.test.tsx src/components/Sidepanel/Chat/__tests__/ControlRow.role-play-handoff.test.tsx` -> `3 passed`, `7 passed`.
- Packaged extension smoke: `cd apps/extension && TLDW_E2E_EXTENSION_MINIMAL_LOCALES=1 TLDW_E2E_EXTENSION_LAUNCH_TIMEOUT_MS=90000 TLDW_E2E_EXTENSION_TARGET_WAIT_MS=90000 npx playwright test tests/e2e/sidepanel-chat-smoke.spec.ts --project=chromium-extension --reporter=line --workers=1 --grep 'keeps packaged /chat handoffs route-only and rail-safe'` -> `1 passed`.
- Real-server `/chat` proof against FastAPI `127.0.0.1:18023`, Next `localhost:18024`, and mock OpenAI `127.0.0.1:18088`: `cd apps/tldw-frontend && bun run e2e:chat-cockpit:real:focused` -> `5 passed`; no-skips assertion reported `executed=5 expected=5 skipped=0 unexpected=0 flaky=0`.
- Temporary FastAPI and mock OpenAI proof services were stopped, and ports `18023` and `18088` were confirmed no longer listening.
- Bandit was not run because this review fix changed TypeScript/JSON/test/Backlog files only; no Python source changed.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR #2160 was rebased onto latest `dev` and the review comments were addressed. English sidepanel locale copy now matches the route-only `/chat` handoff contract instead of overriding component defaults with stale state-preservation text. The E2E extension staging helper now follows the packaged extension manifest default locale rather than hardcoding `en`, with fallback behavior preserved. Focused unit tests, packaged sidepanel smoke, and the real-server `/chat` proof passed after the rebase.

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
