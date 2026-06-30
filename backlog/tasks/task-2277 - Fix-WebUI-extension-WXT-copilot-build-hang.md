---
id: TASK-2277
title: Fix WebUI extension WXT copilot build hang
status: Done
labels:
- extension
- webui
- build
- wxt
priority: high
references:
- apps/extension/entrypoints/copilot-popup.content.tsx
- apps/extension/tests/unit/copilot-entrypoint-lazy-import.test.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Port the current WebUI/browser extension build fix onto a clean branch from dev. Keep scope limited to the copilot content-script wrapper, a focused regression guard, and verification that extension production builds no longer hang.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the dev-branch copilot wrapper state and whether a code change is still needed.
2. Preserve the WXT metadata wrapper that dynamically imports the shared copilot content script in main().
3. Add a focused source guard to prevent static re-export regressions.
4. Verify with the focused test, compile, and browser production builds needed for the PR.
5. Record verification and Bandit applicability in the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Strengthened the existing regression guard for the WebUI extension copilot content-script wrapper by requiring the WXT defineContentScript import explicitly. The current origin/dev wrapper already uses defineContentScript with a runtime dynamic import, so this PR preserves that fix and prevents regression to the static re-export that caused WXT metadata discovery/build hangs. Verification: bun test tests/unit/copilot-entrypoint-lazy-import.test.ts passed; bun run compile passed; bun run build:chrome:prod passed; bun run build:firefox:prod passed; bun run build:edge:prod passed. Bandit was invoked on the touched TS test path via the project venv and produced zero findings; it reported the TypeScript file as non-Python syntax, as expected.
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
