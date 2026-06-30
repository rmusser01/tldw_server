---
id: TASK-564
title: Refresh packaged extension and real-server chat proof after model recovery
  handoff
status: Done
labels:
- chat
- extension
- ux
- e2e
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve or route around the packaged extension sidepanel smoke launch environment for the current chat handoff branch, then refresh the real-server /chat green-path proof after the model recovery selector and packaged /chat carrier changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current packaged extension sidepanel smoke either passes in a known-good browser environment or records a root-caused environment skip with an actionable command.
- [x] #2 Current real-server /chat green path is verified against a configured backend/mock provider, including first send and visible response/loading/recovery behavior.
- [x] #3 Focused sidepanel route/model recovery regressions still pass after any changes.
- [x] #4 Verification commands, skips, blockers, and Bandit applicability are recorded before final handoff.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Rebased onto latest `origin/dev` before the final proof pass. The original local task ID collided with a newly merged `TASK-561`, so this handoff task was renumbered to `TASK-564`.

Root cause: Chrome-for-Testing could launch a minimal MV3 extension and the tldw extension without generated locales, but timed out when loading the full generated `_locales/*/messages.json` tree. System Chrome was not a reliable fallback because it ignored the unpacked extension in this run. The temporary `ignoreDefaultArgs: ['--disable-component-extensions-with-background-pages']` experiment exposed Chrome component-extension false positives and was not kept.

Implemented an E2E-only launch staging path for the packaged extension. When `TLDW_E2E_EXTENSION_MINIMAL_LOCALES=1` or equivalent locale mode is set, the harness copies the packaged extension into the Playwright profile, preserves the production build files, omits the generated locale tree, and writes a minimal `_locales/en/messages.json` for the browser launch. Production extension output is untouched.

Verification recorded:
- Extension launch utility tests: `cd apps/extension && bunx vitest run tests/e2e/utils/extension-paths.test.ts tests/e2e/utils/extension.launch.test.ts tests/e2e/utils/extension-build.test.ts` -> `3 passed`, `10 passed`.
- Packaged sidepanel smoke in known-good Chrome-for-Testing environment: `TLDW_E2E_EXTENSION_MINIMAL_LOCALES=1 TLDW_E2E_EXTENSION_LAUNCH_TIMEOUT_MS=90000 TLDW_E2E_EXTENSION_TARGET_WAIT_MS=90000 npx playwright test tests/e2e/sidepanel-chat-smoke.spec.ts --project=chromium-extension --reporter=line --workers=1 --grep 'keeps packaged /chat handoffs route-only and rail-safe'` -> `1 passed`.
- Real-server /chat proof against FastAPI on `127.0.0.1:18023`, Next on `localhost:18024`, and mock OpenAI on `127.0.0.1:18088`: `bun run e2e:chat-cockpit:real:focused` -> `5 passed`; no-skips assertion reported `executed=5 expected=5 skipped=0 unexpected=0 flaky=0`.
- Temporary FastAPI and mock OpenAI proof services were stopped, and ports `18023` and `18088` were confirmed no longer listening.
- Bandit was not run because this slice touched TypeScript E2E harness/test files and this Backlog task only; no Python source changed.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved the packaged extension smoke launch by routing the E2E harness through a staged Chrome-for-Testing copy with minimal locales, keeping the production packaged extension build unchanged. Refreshed the sidepanel /chat handoff smoke and the real-server /chat cockpit green path after rebasing onto latest `origin/dev`. No known skips remain for this proof slice; the only retained caveat is that the minimal-locale staging is a test-environment workaround for the current CFT launch behavior, not a product build change.

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
