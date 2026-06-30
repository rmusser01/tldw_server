---
id: TASK-498
title: Fix OpenUI extension WXT build hang regression
status: Done
references:
- TASK-306
- TASK-495
- apps/extension/scripts/build-with-profile.mjs
- apps/extension/wxt.config.ts
- apps/packages/ui/src/components/Common/DynamicUI/renderers/OpenUIRenderer.tsx
modified_files:
- apps/extension/entrypoints/copilot-popup.content.tsx
- apps/extension/tests/unit/copilot-entrypoint-lazy-import.test.ts
- backlog/tasks/task-498 - Fix-OpenUI-extension-WXT-build-hang-regression.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and fix the extension WXT Chrome build hang regression observed after the OpenUI Dynamic UI implementation. The build currently reaches WXT/Vite startup and then stops producing output, blocking extension build verification while extension OpenUI rendering remains fallback-only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Narrowest reproducible WXT hang command is identified with logs/process evidence.
- [x] #2 Root cause is documented and tied to a specific build/import/config path, or the task records an environment-only blocker with evidence.
- [x] #3 If repository code changes are needed, a failing regression test or static guard exists before production edits.
- [x] #4 `bun run compile` from `apps/extension` passes.
- [x] #5 The extension build command that previously hung either completes or has a documented non-code blocker and safe fallback decision.
- [x] #6 Bandit decision is recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the extension WXT Chrome dev build hang from the clean OpenUI worktree and capture the narrowest command that hangs.
2. Inspect process state and build logs to identify whether WXT is blocked by a static entry import, dependency resolution, CSS/runtime import side effect, or environment/build-runner issue.
3. Compare with the previous TASK-306 fix pattern and current OpenUI lazy adapter imports before forming a fix hypothesis.
4. Add a failing regression test or static guard for the identified import/build contract before changing production code.
5. Implement the smallest fix, then verify with focused tests, extension compile, and the extension build command that previously hung.
6. Record Bandit decision and close the Backlog task with evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Reproduced the narrow hang with `WXT_PREPARE_DEBUG_HANDLES=verbose node scripts/wxt-prepare.mjs` from `apps/extension`; the command reached WXT type generation, emitted duplicate-import/localStorage warnings, and stayed inside `prepare()` without reaching the active-handle dump.
- Process sampling showed an idle Node process with esbuild/FSEvents handles and loaded native `canvas` libraries, which indicated WXT prepare had imported the heavy shared UI graph rather than remaining on a lightweight extension entrypoint path.
- Compared against the previous TASK-306 pattern in commit `607f9a23a1`; the current branch had regressed `apps/extension/entrypoints/copilot-popup.content.tsx` back to a static re-export of `@tldw/ui/entries/copilot-popup.content`.
- Added `tests/unit/copilot-entrypoint-lazy-import.test.ts` first and verified it failed on the static re-export before changing production code.
- Restored the extension copilot content entrypoint as a WXT `defineContentScript` wrapper that dynamically imports the shared popup implementation from `main()`.
- Bandit was not run because this task touched only TypeScript/Backlog files and no Python code.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the extension WXT build hang by preventing the copilot content script from statically importing the shared popup implementation during WXT prepare/build. The extension entrypoint now stays lightweight for WXT and loads the shared popup implementation at runtime, with a regression test guarding the lazy-import contract.
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
