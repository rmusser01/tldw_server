---
id: TASK-517
title: Add chat cockpit rail regression guards
status: To Do
labels:
- chat
- frontend
- test
priority: high
documentation:
- Docs/superpowers/plans/2026-05-27-chat-rails-ux-rebaseline-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add focused regression coverage that proves the main /chat cockpit shell, context rail, runtime inspector, mobile rail panels, focus mode, and character rail remain wired on the origin/dev chat baseline before user-facing UX fixes proceed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A source-level regression guard proves the /chat page still imports and renders the cockpit shell, context rail, runtime inspector, mobile rail surface, focus mode affordance, and character rail.
- [ ] #2 Existing cockpit component and real-server e2e coverage is run or updated to verify rail visibility on desktop and mobile.
- [ ] #3 Screenshot artifacts for rail-enabled /chat are copied into the review asset directory for the refreshed UX audit.
- [ ] #4 Verification results, skips, and any baseline failures are recorded in this task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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
