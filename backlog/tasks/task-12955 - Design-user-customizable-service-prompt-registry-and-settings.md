---
id: TASK-12955
title: Design user-customizable service prompt registry and settings
status: Done
assignee: []
created_date: '2026-07-14 00:12'
updated_date: '2026-07-14 00:21'
labels:
  - prompts
  - design
  - backend
  - webui
  - browser-extension
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-07-12-user-customizable-service-prompts-design.md
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A reviewed design defines eligibility, ownership, precedence, multipart assembly, versioning, security, API, backup, WebUI, and browser-extension behavior.
- [x] #2 The requester explicitly approves the design before implementation planning begins.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Use the approved Service Prompt brainstorming and design workflow documented in Docs/superpowers/specs/2026-07-12-user-customizable-service-prompts-design.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Recreated on the current dev base because the original planning branch task ID collided with a task allocated on dev. Historical review decisions remain captured in the linked design specification.

Final verification on the current dev base: approved design artifact present; downstream planning and inventory artifacts use collision-free IDs; no Python changed, so Bandit is not applicable; full CI shards were intentionally skipped for planning-only work at requester direction.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed and requester-approved the governed design for user-customizable Service Prompts across backend, WebUI, and browser extension.
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
