---
id: TASK-590
title: Stabilize extension research workspace parity PR gate
status: In Progress
labels:
- ci
- e2e
- extension
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The PR rerun reproduced Extension Research Workspace Parity as a 60s Playwright test timeout after the extension build completed. Investigate and adjust the spec timeout budget so the shared parity contract can complete or fail at a concrete assertion instead of the global test timeout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Extension research workspace parity spec has an explicit timeout budget appropriate for the shared parity contract.
- [x] #2 The targeted extension parity E2E command is run locally or the inability to run it is documented.
- [ ] #3 PR status is rechecked after pushing the stabilization change.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The PR rerun reproduced the Extension Research Workspace Parity failure as a Playwright global 60s test timeout after the extension build completed. The failure did not include a parity assertion failure.

Added a per-test `test.setTimeout(120_000)` budget to `apps/extension/tests/e2e/research-workspace.parity.spec.ts`, matching the pattern used by other heavier extension E2E specs that launch the built extension and run multi-step workflows.

Verification before push:
- `git diff --check` -> clean
- `env CI=1 SKIP_WXT_PREPARE=1 TLDW_E2E_EXTENSION_LAUNCH_TIMEOUT_MS=90000 TLDW_E2E_EXTENSION_TARGET_WAIT_MS=90000 ../node_modules/.bin/playwright test tests/e2e/research-workspace.parity.spec.ts --reporter=line` from `apps/extension` -> exited 0 with 1 skipped. The command discovered the test and exercised the local Playwright entrypoint, but the browser-backed extension contract was skipped by the local launch helper in this environment.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

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
