---
id: TASK-591
title: Rebase PR 2220 on dev and address review feedback
status: Done
labels:
- api-boundary
- pr-feedback
- rebase
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase PR #2220 onto latest origin/dev and address all review comments, PR comments, and reported issues before updating the pull request.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR #2220 branch rebased onto latest origin/dev.
- [x] #2 Valid review comments/issues addressed or documented as obsolete after base correction.
- [x] #3 CodeQL PR-check findings addressed or documented.
- [x] #4 Focused verification and Bandit results recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Rebased the API-boundary remediation series onto origin/dev and kept only the reviewable API-boundary commits on top of dev. Resolved rebase conflicts in router group metadata, Media DB schema bootstrap, and prototype promotion review routing.

Confirmed the Gemini Dockerfile inline comment is obsolete after rebasing/switching the PR base to dev because Dockerfiles/entrypoints/tldw-app-first-run.sh is no longer in the PR diff. Updated focused tests for dev-side audio preset bootstrap ordering and structured prototype error envelopes.

Resolved obsolete outdated review threads after the base correction left zero active unresolved review threads. Addressed the remaining CodeQL PR check failure by parsing URL hostnames before classifying Research Workspace source URLs as YouTube/Vimeo video sources, preventing spoofed URLs from matching by substring.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR #2220 has been rebased onto origin/dev locally. The remaining rebase follow-up updates test expectations for dev audio preset bootstrap and structured prototype promotion review errors, and fixes the CodeQL URL-substring sanitization finding in Research Workspace source type detection. Verification: the four previously failing tests passed; the focused API-boundary suite passed with 392 tests; AddSourceModal Stage 2 Vitest passed with 19 tests; git diff --check passed; Bandit over touched production paths reported 0 results and 0 errors.
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
