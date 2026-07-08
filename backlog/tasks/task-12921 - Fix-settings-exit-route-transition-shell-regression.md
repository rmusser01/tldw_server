---
id: TASK-12921
title: Fix settings exit route transition shell regression
status: Done
labels:
- frontend
- bug
- settings
- navigation
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and fix the settings page Exit flow where navigating back to the previous page can leave the settings content visible without the WebUI header/sidebar during the route transition.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root cause: settings pages intentionally render as shell-less routes. On Exit, Next can update the browser URL back to the previous page before the new page component/hydration has replaced the old settings component, leaving the stale settings layout visible under the prior URL without the WebUI shell. Fix: settings Exit now uses document navigation in the Next web app so stale shell-less settings content is unloaded instead of kept through the SPA transition. Non-Next/shared UI contexts still use SPA navigate with React Router's flushSync option. The Next react-router shim now honors flushSync for callers that need immediate navigation semantics. Settings exit targets are normalized to app-relative paths before navigation so a corrupted sessionStorage return target cannot send document navigation off-origin. Verification: focused settings exit tests pass; existing settings layout tests pass; react-router shim transition tests pass; live Playwright repro against localhost no longer shows the stale settings page under /chat. Typecheck note: UI package tsc requires increased heap and still fails on unrelated existing baseline errors in tests/background code; no new typecheck evidence points at the touched settings files.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed settings Exit route-transition regression by using document navigation for settings Exit under the Next web app, keeping flushSync SPA navigation for non-Next contexts, and normalizing exit targets. Added focused regression coverage and verified with the live localhost repro.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
