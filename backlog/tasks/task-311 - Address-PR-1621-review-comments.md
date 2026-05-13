---
id: TASK-311
title: Address PR 1621 review comments
status: In Progress
assignee: []
created_date: '2026-05-13 02:10'
updated_date: '2026-05-13 03:28'
labels:
  - moderation
  - pr-review
  - webui
  - backend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1621'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve still-actionable review threads on PR #1621 for moderation review/rules remediation. Verify each external suggestion against the codebase before changing it. Scope includes async capture behavior, decision-history responses, review-store query/schema details, RBAC seeding, frontend accessibility/error handling, test robustness, endpoint docstrings, and technical rationale for any intentionally unresolved item.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All still-valid unresolved review threads on PR #1621 are either fixed in code or explicitly answered with technical rationale.
- [x] #2 Focused backend and frontend tests covering touched moderation review/rules behavior pass locally.
- [x] #3 Security and hygiene checks for touched backend files pass or have documented baseline rationale.
- [ ] #4 Review fixes are committed and pushed to the PR branch.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Local verification completed before commit:
- Backend focused pytest: 57 passed.
- UI focused Vitest: 25 files / 241 tests passed.
- Moderation Playwright tier-5 subset: 10 passed.
- verify:design-system-state passed with baseline exceptions only.
- verify:openapi passed with 10 reviewed baseline exceptions.
- py_compile and git diff --check passed.
- Bandit wrote /tmp/bandit_pr1621_review_fixes.json; only finding is existing migrations.py B608 at line 616 outside the PR review-fix hunk.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [ ] #7 PR review-thread status refreshed after push
- [ ] #8 Remaining GitHub CI state reported separately from local verification
<!-- DOD:END -->
