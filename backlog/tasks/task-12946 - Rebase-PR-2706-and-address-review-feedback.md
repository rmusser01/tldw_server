---
id: TASK-12946
title: Rebase PR 2706 and address review feedback
status: In Progress
labels:
- security
- codeql
- frontend
- code-review
priority: High
references:
- https://github.com/rmusser01/tldw_server/pull/2706
documentation:
- Docs/superpowers/plans/IMPLEMENTATION_PLAN_2026-07-10_pr2706_review_followup.md
modified_files:
- Docs/superpowers/plans/IMPLEMENTATION_PLAN_2026-07-10_pr2706_review_followup.md
- apps/packages/ui/src/components/Option/Watchlists/ItemsTab/items-utils.ts
- apps/packages/ui/src/components/Option/Watchlists/ItemsTab/__tests__/items-utils.test.ts
- apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourcesTab.tsx
- apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/SourcesTab.advanced-details.test.tsx
- backlog/tasks/task-12946 - Rebase-PR-2706-and-address-review-feedback.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase the CodeQL remediation PR #2706 onto the latest origin/dev, evaluate and address every actionable PR review comment, verify the focused frontend behavior and checks, reply in the original review threads, and update the PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 PR #2706 is rebased onto the latest origin/dev without dropping remediation changes.
- [ ] #2 All technically valid inline and summary review findings are addressed with focused regression coverage.
- [ ] #3 Focused tests, frontend typecheck, diff checks, and applicable security validation pass.
- [ ] #4 Review threads receive precise replies and the updated branch is pushed.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Review inventory on 2026-07-10 found two inline threads and one summary-only issue: accept optional whitespace around HTML src assignment; continue scanning after safeImageUrl rejects an earlier HTML/Markdown candidate; and avoid a truthiness check when deriving groups from number|null state. Root-cause verification confirmed the first two are user-visible preview correctness bugs. Backend group endpoints require IDs >=1, so group ID 0 is not currently server-valid, but an explicit null check is the minimal type-correct defensive change. `git fetch origin dev` followed by `git rebase origin/dev` reported the branch already up to date; no history rewrite occurred.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests and verification recorded
- [ ] #3 Bandit run for touched Python code or explicit not-applicable note
- [ ] #4 Final summary and PR link recorded
- [ ] #5 Known skips or blockers documented
<!-- DOD:END -->
