---
id: TASK-12148
title: 'Rebase PR #2633 and address review feedback'
status: In Progress
priority: High
references:
- https://github.com/rmusser01/tldw_server/pull/2633
- https://github.com/rmusser01/tldw_server/issues/2605
documentation:
- Docs/superpowers/specs/2026-07-12-pr-2633-review-rebase-design.md
- Docs/superpowers/plans/2026-07-12-pr-2633-review-rebase-implementation-plan.md
modified_files:
- Docs/superpowers/specs/2026-07-12-pr-2633-review-rebase-design.md
- Docs/superpowers/plans/2026-07-12-pr-2633-review-rebase-implementation-plan.md
- backlog/tasks/task-12148 - Rebase-PR-2633-and-address-review-feedback.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase the Research Workspace UAT and artifact verification PR onto current origin/dev, resolve conflicts without regressing current dev behavior, address every still-valid review comment, document rejected or stale feedback, verify touched frontend/backend scope, and update the existing PR branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 PR branch is rebased onto the latest origin/dev with conflicts resolved.
- [ ] #2 Every unresolved inline and outside-diff review finding is verified and either fixed or answered with technical rationale.
- [ ] #3 Valid stored ACP configuration takes precedence; runtime single-user API key is only a fallback for missing, blank, or placeholder stored keys, and multi-user auth remains isolated.
- [ ] #4 Focused frontend and backend regression tests pass.
- [ ] #5 Bandit reports no new findings in touched Python code and git diff --check passes.
- [ ] #6 Updated branch is force-pushed with lease and PR review/check status is re-inspected.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-12-pr-2633-review-rebase-implementation-plan.md
<!-- SECTION:PLAN:END -->

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
