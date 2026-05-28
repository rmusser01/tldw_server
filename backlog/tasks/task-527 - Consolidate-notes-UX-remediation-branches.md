---
id: TASK-527
title: Consolidate notes UX remediation branches
status: In Progress
labels:
- notes
- ux
- integration
- webui
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a single integration branch for the approved /notes UX remediation slices, replaying the existing PR branches in plan order on current origin/dev, resolving conflicts narrowly, and recording focused verification. Scope is branch/history/file integration only; avoid new /notes feature work except conflict resolution required to make the existing slices coexist.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A clean integration branch contains the completed /notes remediation slice commits in approved PR order or equivalent resolved content.
- [ ] #2 Conflicts are resolved narrowly without reverting unrelated slice behavior.
- [ ] #3 Backlog task records the source branches/commits, conflicts, verification commands, and known baseline failures.
- [ ] #4 Focused /notes UI tests run on the integrated branch, or any failures are documented with evidence.
- [ ] #5 Frontend-only/Python verification requirements are documented, including Bandit skip or run as applicable.
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
- [ ] #2 Verification recorded
- [ ] #3 Known blockers/skips documented
- [ ] #4 Final summary added
<!-- DOD:END -->
