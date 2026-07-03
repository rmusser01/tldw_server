---
id: TASK-12117
title: Fix PR 2571 release CI failures
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-03 13:44'
labels:
  - ci
  - release
  - pr-2571
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2571'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
<!-- SECTION:DESCRIPTION:BEGIN -->
<!-- SECTION:DESCRIPTION:END -->

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Second PR #2571 CI pass: current logs show docs MCP policy/config failures, Guardian notify timestamp mutation failure, UI playground a11y mock drift, and home-route smoke seed drift. Sandbox macOS/Python 3.12 cap failures are not reproduced locally on macOS/Python 3.11; collecting more evidence before changing that path.

User requested all current CodeQL issues be addressed. Expanding this task from test-failure fixes into CodeQL remediation/baseline cleanup for PR #2571.

Addressed the current 96 CodeQL annotations from check-run 84944194830. Existing justified suppressions were converted to LGTM suppression comments because the current gate did not honor the prior syntax. The valid TLS finding in the MCP docs validated transport was fixed by enforcing TLS 1.2 minimum on HTTPS sockets, with a regression test. Touched-scope Bandit now reports zero findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed current PR #2571 CI failures by allowing release README version updates to use the current combined beyond/post-release status line and by updating the Playground responsive parity guard to the current mobile/focus-mode condition. Validation: docs suite 117 passed; playground device-matrix 15 passed; Bandit on Helper_Scripts/release.py reported 0 results; git diff --check clean.

Follow-up validation for the CodeQL pass: focused MCP docs/Guardian pytest selection 6 passed; `bun run test:playground:a11y` in `apps/packages/ui` passed 10 files / 27 tests after repairing the local ignored Bun symlink for test execution; `git diff --check` clean; Bandit on touched Python files reported zero findings.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
