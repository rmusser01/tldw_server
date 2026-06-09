---
id: TASK-530
title: Address PR 2313 media query mock review feedback
status: Done
labels:
- webui
- extension
- review
references:
- TASK-529
- https://github.com/rmusser01/tldw_server/pull/2313
modified_files:
- apps/tldw-frontend/__tests__/components/layout/WebLayout.chat-scroll-contract.test.tsx
- backlog/tasks/task-530 - Address-PR-2313-media-query-mock-review-feedback.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address PR #2313 review feedback by making the WebLayout media-query test mock resilient to additional exports, then rebase the PR branch onto latest dev and complete merge readiness checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 WebLayout test uses a partial useMediaQuery module mock that preserves non-overridden exports.
- [x] #2 Regression guard verifies useTablet and useMediaQuery remain available from the mocked module.
- [x] #3 PR branch is rebased onto latest origin/dev and reduced to the relevant review-remediation diff.
- [x] #4 Focused WebLayout and shared ChatSidebar/Layout Vitest suites pass after rebase.
- [x] #5 Bandit frontend-only/Markdown-only note and final summary are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Qodo review item verified as technically sound: the prior full module mock reproduced the missing-export failure once a guard checked non-overridden useMediaQuery exports. Replaced it with a partial mock using vi.importActual and overriding only useDesktop/useMobile. Added a regression guard that useTablet and useMediaQuery remain available from the mocked module. Next steps: rebase PR branch onto latest origin/dev, update PR base to dev, rerun focused suites, resolve review threads/comment, then merge if merge gates allow.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed Qodo PR #2313 feedback by replacing the full useMediaQuery module mock with a partial vi.importActual mock that preserves non-overridden exports while forcing useDesktop/useMobile false for this WebLayout test. Added a regression guard that useTablet and useMediaQuery remain available from the mocked module. Rebasing onto latest origin/dev reduced the PR to the single review-remediation commit because latest dev already contains the prior sidebar implementation and TASK-401/TASK-404 closeout via TASK-567/PR #2168. Verification passed after rebase: WebLayout chat scroll contract, 6 tests; shared ChatSidebar/Layout focused suite, 15 tests. Bandit not run because touched code is frontend TypeScript/test-only plus Backlog Markdown.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
