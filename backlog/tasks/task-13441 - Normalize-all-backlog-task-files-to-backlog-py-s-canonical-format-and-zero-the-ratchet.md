---
id: TASK-13441
title: Normalize all backlog task files to backlog-py's canonical format and zero
  the ratchet
status: To Do
dependencies:
- TASK-13440
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up to TASK-13440: run the backlog-py normalize command over backlog/tasks (and archive if applicable) in a backlog-only PR, verify every file round-trips through backlog-py unchanged afterwards, and set the Node-style-section ratchet baseline to 0. Merge in a quiet window because it touches hundreds of task files other open PRs may also edit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every task file is in backlog-py canonical form and a second normalize run changes nothing
- [ ] #2 The Node-style-section ratchet baseline is 0 and CI enforces it
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
