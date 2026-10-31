---
id: TASK-13957
title: Keep Retry available after stopping whitespace-only chat output
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #3368 review found that whitespace-only output is classified as stopped before a reply but saved as generated_unsaved. Normalize no-output recovery and extend the real Playground regression for fresh and saved server chats. PR: https://github.com/rmusser01/tldw_server/pull/3368
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Stopping after whitespace retains one question and a usable saved-question Retry action.
- [x] #2 Retry sends the original question once and produces an answer.
- [x] #3 Uncertain server outcomes remain in guarded recovery after Stop.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR #3368 review: the stopped-before-reply predicate accepted whitespace yet recover saved it as generated_unsaved, hiding saved-question Retry. Independent review also found that applying this marker to server-owned or unadmitted turns hid uncertain recovery records. The two-line production fix requires a client-managed admitted question and passes empty content only for that no-visible-reply stop.

Red: the whitespace Playground case failed because the recovery card was absent; all four native Stop-before/after-admission cases failed because uncertain records were marked stopped_before_reply. Green: 464 tests passed across Playground interrupted replies, normalChatMode history selection, abort lifecycle, HistorySelectionReview, and draft restoration; 44 Retry service tests passed separately (508 total). All four CM-02 cases and all 26 native settlement cases also passed focused runs. Frontend bun run typecheck passed. Scoped ESLint passed with zero errors and seven warnings on unchanged lines. git diff --check passed. Independent final review found no remaining material issue.

Bandit is not applicable to these TypeScript/test and tracking-Markdown changes; no Python changed. Expected fake-server 404 and simulated stream-error messages occurred in integration tests; no failing or skipped tests in the final six-file suite. Repository-wide tests were not run. Hosted required checks remain separate and must pass before merge. Requester explicitly waived the human-written Change summary requirement only for #3368 and #3296 on 2026-10-30.

Evidence: temporary tldw-pr-landing-20261030 directory, pr3368-*.log and pr3368-eslint.json. Reviewed against c7ae162c548a2e97976aaf8c6212ec0fd779a086. PR: https://github.com/rmusser01/tldw_server/pull/3368
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Early Stop keeps Retry available after whitespace-only output while preserving guarded recovery for unconfirmed server writes. Added six regressions, verified 508 affected tests, typecheck and lint, and completed independent review. Ready for PR #3368 required CI and merge.
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
