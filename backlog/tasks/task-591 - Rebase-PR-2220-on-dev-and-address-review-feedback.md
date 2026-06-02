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

Reviewed and addressed the active Cubic follow-up comments: completed/anchored the affected Backlog task records, derived the minimal-test research workspace router from canonical metadata, rejected null-only media item updates, allowed document annotation notes to be explicitly cleared, returned batch annotation sync results through one bulk read, and moved promotion review pending-state enforcement into the service with endpoint 409 conflict mapping.

Addressed follow-up frontend PR gate failures after the review fixes landed: ensured the `/setup` route always exposes exactly one semantic `h1` across completed, loading, and wizard states; refreshed the stale setup-route test around the current setup onboarding hook; and made the extension research workspace parity workflow build the Chrome extension before launching it under a headful Xvfb session. While touching that workflow, pinned the Node and Bun setup actions to immutable SHAs.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR #2220 has been rebased onto origin/dev locally. The rebase follow-up updates test expectations for dev audio preset bootstrap and structured prototype promotion review errors, fixes the CodeQL URL-substring sanitization finding in Research Workspace source type detection, and addresses the active Cubic review comments on minimal router coverage, media null updates, document annotation sync/update behavior, promotion request state handling, and Backlog task metadata. A second frontend follow-up fixes the `/setup` landmark contract and the extension parity workflow skip root cause. Verification: the Cubic-focused targeted tests passed with 5 tests; the focused API-boundary backend suite passed with 395 tests; AddSourceModal Stage 2 plus OptionSetup route Vitest passed with 22 tests; git diff --check passed; Bandit over touched production paths reported 0 results and 0 errors. Local execution of the new extension build step was blocked by this worktree's incomplete frontend dependency install (`cross-env` missing); CI runs `bun install --frozen-lockfile` before the added build step.
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
