---
id: TASK-12123
title: Design WebUI setup choice for API server setup handoff
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-03 22:54'
labels:
  - webui
  - setup
  - onboarding
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and plan a WebUI /setup pre-wizard choice screen that explains WebUI setup versus API server setup for first-run users who skipped directly from API install to WebUI access. The work should preserve backend setup security boundaries, link out to API server /setup separately, and avoid confusing nontechnical users.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design spec documents when and how the WebUI setup choice appears.
- [x] #2 Design accounts for browser-openable API setup URL resolution and local-only setup warnings.
- [x] #3 Design preserves existing WebUI first-run wizard behavior after choosing WebUI setup.
- [x] #4 Implementation plan covers focused frontend tests and onboarding UAT coverage.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-03-webui-setup-choice-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Design spec committed at Docs/superpowers/specs/2026-07-03-webui-setup-choice-design.md. Implementation plan written at Docs/superpowers/plans/2026-07-03-webui-setup-choice-implementation-plan.md and approved by plan review. Bandit skipped for the planning step because only Markdown/task files changed; implementation will need normal verification.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the WebUI /setup setup-choice design and implementation plan. The spec and plan define the pre-wizard choice, API server setup link/fallback behavior, blocked-state recovery handling, route integration, and focused frontend/Playwright verification. Implementation remains pending and should use a separate execution task before code edits begin.
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
