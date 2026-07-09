---
id: TASK-12929
title: Fix actionlint install failure in CI
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-09 03:09'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Current-head PR #2692 actionlint failed because the install step used process substitution for the download script; a raw.githubusercontent.com 429 left no ./actionlint binary while the install step still succeeded. Make the install fail-fast and less prone to unauthenticated raw GitHub rate limiting.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Actionlint install does not use process substitution that hides curl failures.
- [x] #2 Actionlint binary is installed from a pinned authenticated release download.
- [x] #3 Targeted workflow set passes local actionlint verification.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: raw.githubusercontent.com returned 429 in the process substitution used by bash <(curl ...). Bash then read an empty process-substitution file and exited successfully, so the next step failed because ./actionlint was never created. Replaced the install step with gh release download using github.token, pinned v1.7.12, then tar/install/version check.

Bandit not applicable because this task only changes a GitHub Actions workflow and Backlog metadata.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the actionlint CI install path so download failures are fail-fast and less prone to unauthenticated raw GitHub rate limits. Verified the targeted workflow set with a locally downloaded actionlint v1.7.12 binary.
<!-- SECTION:FINAL_SUMMARY:END -->

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
