---
id: TASK-439
title: Address PR 1866 Character Chat review comments
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-19 19:14'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address live PR #1866 review comments for Character Chat Phase 2 readiness, including i18next interpolation usage and redundant PromptSelect loading status markup, with focused frontend regression tests and verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Verify live PR #1866 review comments before patching.
- [x] #2 PromptSelect loading trigger avoids redundant nested status announcements while keeping visible loading copy.
- [x] #3 Character Chat missing restored character copy uses i18next interpolation options.
- [x] #4 Character readiness copy delegates interpolation to the translation layer without manual placeholder replacement.
- [x] #5 Focused frontend tests pass and verification results are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified live PR #1866 review threads via GitHub GraphQL. Addressed Gemini comments by removing the nested prompt-loading role=status span, switching missing restored character copy to i18next interpolation options, and removing manual placeholder interpolation from the readiness copy helper. Updated local test translation mocks to support i18next-style options.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed all currently actionable PR #1866 review comments. Verification: red tests failed before the fix for PromptSelect redundant status markup and missing-character interpolation; focused Vitest suite passed with 6 files and 88 tests; git diff --check passed. TypeScript still fails on existing repo-wide baseline debt, and captured output in /tmp/tldw_pr1866_review_tsc.txt has no errors matching the files touched in this review pass.
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
