---
id: TASK-12119
title: Plan conservative frontend licensing cutoff
status: In Progress
labels:
- licensing
- frontend
- planning
priority: high
documentation:
- Docs/superpowers/specs/2026-07-19-frontend-source-available-licensing-design.md
- Docs/superpowers/plans/2026-07-20-conservative-frontend-licensing-cutoff-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-07-20-conservative-frontend-licensing-cutoff-implementation-plan.md
- backlog/tasks/task-12120 - Implement-conservative-frontend-licensing-cutoff.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create an execution-ready implementation plan for the urgent, prospective pre-counsel licensing cutoff approved in TASK-12118. Scope the plan to standard PolyForm Perimeter and release-specific Countdown terms, authoritative path and historical records, metadata corrections, protected-artifact publication freeze, OpenAPI contract licensing, and fail-closed verification. Defer custom community/customer grants, CLA, trademark policy, and commercial agreements to counsel-reviewed follow-on work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The plan maps every immediate-cutoff requirement to exact repository files and executable verification commands.
- [ ] #2 Tasks are reviewer-sized, ordered, test-driven where behavior is introduced, and contain no unspecified implementation placeholders.
- [ ] #3 The plan prevents protected artifact publication and third-party protected/API-contract contributions until later legal and artifact gates exist.
- [ ] #4 The plan explicitly preserves public history, third-party terms, GPL-3.0-only backend implementation, and Apache-2.0 canonical OpenAPI contract material.
- [ ] #5 Post-counsel custom grants and full artifact release hardening are clearly deferred to separate follow-on plans.
- [ ] #6 The plan is self-reviewed against the approved spec, committed with this task record, and ready for an execution-mode choice.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Used the approved TASK-12118 design to produce an execution-ready six-task plan for the urgent pre-counsel cutoff.
- Split protected artifact release/Countdown completion and counsel-reviewed custom grants into two explicit follow-on plans because neither can safely ship in the source-only cutoff.
- Re-verified public GitHub refs on 2026-07-20: main 7a23be3202e360f2d8e7cfe208e13ba406cf0507, dev 29acaca8c781213e27b12066372df13855e2e7a6, and draft PR #2727 head 60ce244fb6a65a79489b3f77299340afa501be24.
- Fetched the five official legal texts read-only and recorded exact SHA-256 checksums for the implementation tests.
- Self-review covered spec requirements, placeholders, file paths, function names, workflow bootstrap behavior, Docker isolation, and deferred release mechanics. The placeholder scan and git diff whitespace check are clean.
- Pre-created TASK-12120 as the exact execution task so the plan does not contain an unknown Backlog path.
- Bandit is not applicable to this planning task because it changes Markdown task and plan records only.
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
