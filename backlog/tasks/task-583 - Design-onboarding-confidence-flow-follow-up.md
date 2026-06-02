---
id: TASK-583
title: Design onboarding confidence flow follow-up
status: In Progress
labels:
- onboarding
- webui
- setup
priority: High
documentation:
- Docs/superpowers/specs/2026-06-01-onboarding-confidence-flow-design.md
- Docs/superpowers/plans/2026-06-01-onboarding-confidence-flow-implementation-plan.md
modified_files:
- Docs/superpowers/specs/2026-06-01-onboarding-confidence-flow-design.md
- Docs/superpowers/plans/2026-06-01-onboarding-confidence-flow-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design the next unified solo onboarding follow-up PR: manual provider validation, setup readiness panel, inline first-chat recovery, and first-source guided milestone. Start from latest dev and do not rebase the stale unified-solo-onboarding worktree wholesale.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Design spec captures the approved four-stage order: provider validation, readiness panel, first-chat recovery, first-source milestone.
- [ ] #2 Spec explicitly preserves current dev readiness architecture and avoids resurrecting stale branch deletions.
- [ ] #3 Spec defines staged commits/tasks and test expectations for the eventual one-PR implementation.
- [ ] #4 Backlog task references the written spec and records review/verification status.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
['Design spec written and marked ready for user review.', 'Local review completed against approved sequence: provider validation, readiness panel, first-chat recovery, first-source milestone.', 'Subagent spec review not run because available subagent tooling is restricted to explicit user-requested delegation in this session.', 'Verification: git diff --cached --check passed for staged documentation/task changes.', 'Bandit: skipped because this commit changes only documentation and Backlog task metadata.', 'Follow-up review found and addressed validation-gate deadlock for syntax-only hosted providers by distinguishing ready vs accepted validation results.', 'Follow-up review found and addressed first-source readiness risk by requiring queryable/source readiness before offering a grounded question.', 'Follow-up review narrowed provider save semantics so persistence status is not overloaded with validation status.', 'Implementation plan written with four staged commits, exact touched files, TDD steps, focused verification commands, UAT checklist, and cleanup requirements.', 'Plan review subagent not run because available subagent tooling requires explicit user-requested delegation in this session.', 'Local plan review adjusted execution details: validate-first happy path preserves validation across safe key clearing, first-source ask action must not overclaim readiness, and exact verification commands now keep venv/workdir requirements explicit.']
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
