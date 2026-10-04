---
id: TASK-13441
title: Normalize all backlog task files to backlog-py's canonical format
status: To Do
dependencies:
- TASK-13440
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up to TASK-13440: run 'backlog-py task normalize' over backlog/tasks in a backlog-only PR, then verify that a second 'normalize --check' run reports nothing. The ratchet is per-PR and has no baseline (ADR-059), so nothing else needs changing. Coordinate with peer sessions first and merge in a quiet window, because it touches about 2,300 task files that open PRs may also edit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every task file is in backlog-py canonical form and a second normalize run changes nothing
- [ ] #2 Peer sessions were pinged before the PR opened, and no open PR's task-file edits conflict at merge
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-10-03: reworded after the #3142 review replaced the global ratchet baseline with a per-PR diff check; there is no baseline left to zero.
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
