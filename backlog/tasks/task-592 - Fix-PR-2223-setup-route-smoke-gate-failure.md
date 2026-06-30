---
id: TASK-592
title: Fix PR 2223 setup route smoke gate failure
status: Done
labels:
- ci
- frontend
- pr-review
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the PR #2223 UX Smoke Gate failure where /setup renders zero h1 elements in the completed-setup recovery state. Keep the change focused on the setup route landmark contract and verify with targeted frontend tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Completed `/setup` recovery state exposes exactly one semantic `h1`.
- [x] Wizard setup state keeps its existing single wizard `h1`.
- [x] Targeted unit and Playwright smoke verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added a route-level, screen-reader-only `h1` in `OptionSetup` when the setup wizard is not the rendered primary content. This covers completed setup recovery and loading states without creating a second `h1` when `UnifiedSetupWizard` owns the page heading.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the /setup completed-setup recovery landmark contract by rendering a route-level h1 when the setup wizard is not the primary content. Verification: focused Vitest regression failed before the production change and passed after it; targeted Playwright /setup responsive landmark case passed locally; frontend package lint exited 0 with existing warnings, though the direct UI-package file lint path is not available in this worktree's partial Bun install.
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
