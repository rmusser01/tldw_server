---
id: TASK-12161
title: Sync accidental main release merge back into dev
status: Done
labels:
- release
- branch-sync
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
User accidentally merged PR #2596 into main after the release branch had been rebased from dev. Corrective goal: keep main as the release branch and dev as the forward working branch by merging origin/main back into current origin/dev, preserving dev-only PR #2653 and main release commits without rewriting either remote branch.
Merged origin/main into the dev sync branch with no content conflicts, preserving dev-only PR #2653 and absorbing the main release merge from PR #2596 plus main-only #2624 node_modules cleanup. Validation: git diff --cached --check passed; release docs and PyPI workflow contract tests passed 17/17.
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
