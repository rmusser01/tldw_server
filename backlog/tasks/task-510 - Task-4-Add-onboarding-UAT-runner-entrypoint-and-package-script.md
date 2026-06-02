---
id: TASK-510
title: 'Task 4: Add onboarding UAT runner entrypoint and package script'
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 04:41'
labels:
  - onboarding-uat
  - runner
  - test
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the manual/dev onboarding UAT runner CLI entrypoint, command assembly helpers, package script, and focused tests without starting real services in unit tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 run.mjs exports command builders for mock server, backend, WebUI, and Playwright commands
- [x] #2 Runner CLI supports help and documented flags without starting services
- [x] #3 Package script e2e:onboarding:uat invokes the runner
- [x] #4 Focused tests cover command assembly and help-safe behavior
- [x] #5 Verification recorded for Vitest, help command, diff check, and Bandit skip if no Python touched
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the onboarding UAT runner entrypoint and package script. The runner now assembles deterministic commands for the repo mock OpenAI server, AuthNZ initialization, backend, WebUI, and Playwright; resolves Python from PYTHON/PYTHON3 or the project/main-worktree venv; starts the repo mock server via `python -m mock_openai.server` with a repo-local PYTHONPATH; supports help/scenario/viewport/mock-config/preserve/reviewed-evidence flags; writes redacted summary/log artifacts; scans artifacts for secret leaks; and can copy reviewed evidence into `Docs/Product/WebUI/evidence/onboarding_uat/<run-id>`. Verification: focused red/green command-assembly tests failed before implementation and passed after; `bunx vitest run scripts/__tests__/onboarding-uat-runner.test.ts` passed with 23 tests; `bun run e2e:onboarding:uat -- --help` exited 0 and printed usage without starting services; `env PYTHONPATH=mock_openai_server /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m mock_openai.server --help` exited 0; backend/AuthNZ/uvicorn import probe exited 0; `git diff --check` passed. Bandit skipped because Task 4 touched JS/TS/package/task files only; no Python code changed.
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
