---
id: TASK-12986
title: Split trusted license status into PR-bypassable rulesets
status: In Progress
assignee: []
created_date: '2026-07-24 04:08'
updated_date: '2026-07-24 04:45'
labels:
  - ci
  - github-actions
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-07-23-pr-only-license-status-bypass-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move the branch-qualified trusted frontend-license required statuses into dedicated rulesets with a pull-request-only repository-admin bypass. Keep the existing main and dev structural pull-request, deletion, and non-fast-forward protections active and non-bypassable. Do not change CI workflows or the trusted status publisher.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Main and dev structural protections remain active with no bypass actors.
- [ ] #2 Each branch-qualified trusted license status is enforced by a dedicated status-only ruleset.
- [ ] #3 Only repository administrators can bypass a status-only ruleset, and only while merging a pull request.
- [ ] #4 The rollout has no interval in which either required status or structural protection is absent.
- [ ] #5 Before-state, after-state, effective rules, and rollback evidence are recorded without secrets.
- [ ] #6 No GitHub Actions workflow or trusted publisher changes are made.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The original 25-workflow polling design was rejected after review: hosted-runner waiters could starve the trusted publisher, exceed the GitHub Free concurrency pool, consume excessive status API budget, and accept stale same-SHA statuses. The requester approved replacing it with status-only rulesets carrying a PR-only admin bypass. Live rulesets 5653432 and 19362594 currently combine structural and status rules and have no bypass actors.

Replacement design written after live revalidation. Rollout creates disabled status-only rulesets first, activates them before removing duplicated status rules from existing rulesets, and retains exact before-state for fail-closed rollback.

Independent spec review iteration 1 required three safety fixes: automatic restoration after a self-caused structural read-back mismatch, an exclusive ruleset-maintenance window with explicit non-transactional API semantics and final updated_at/payload comparisons, and a default_branch==main invariant. Iteration 2 approved the revised specification with no implementation-planning blockers.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
