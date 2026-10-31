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
2026-10-31 CI follow-up: investigating pre-commit whitespace failure in this task and the real-cockpit selected-history review failure on head 25dbb2f3e7. Auto-merge disarmed while preparing a reviewed update. ADR check: no new ADR required; preserve existing recovery/history ownership under Docs/ADR/049-chat-history-selection-ownership.md and normalize task records under Docs/ADR/059-backlog-py-task-editor-cutover.md. Plan: diagnose CI, integrate current dev, apply only scoped fixes, run affected tests/typecheck/lint and independent review before push.
2026-10-31 CI follow-up complete: integrated dev 1fe1348a3e cleanly, retained remote handoff-test fix 25dbb2f3e7, normalized this task with pre-commit, and moved the real-cockpit test follow-up exchange outside the unsaved-Stop branch. All three CI attempts showed a complete single exchange with Review hidden, as required by CC-04; both Stop outcomes now create the existing second question before selection/regeneration assertions. No production behavior changed. Independent review found no code issue.
Fresh validation: seven affected files passed 295 tests, including all 30 interrupted-reply cases and three history-controls cases; normalChatMode passed all 298 cases when rerun alone (593 tests verified). The initial concurrent eight-file run had one 5000ms tab-case timeout and 91 cascading state failures; its first two tab cases and then its complete 298-case file passed isolated without changing timeouts. Frontend typecheck passed. Changed E2E ESLint: zero errors, 53 existing warnings. All scoped pre-commit hooks passed using temporary Windows env/python3 command-name adapters. git diff --check origin/dev passed. Bandit remains inapplicable: changes beyond the clean dev merge are TypeScript test setup and tracking Markdown only.
Local real-browser run could not start: shared venv FastAPI lacked iter_route_contexts; existing temporary compatible overlay then hit missing cssselect, and Next dev exceeded its 120-second startup window. No shared dependencies changed and no browser test pass is claimed. Hosted UX and required checks must confirm the final pushed head before merge. Evidence: temporary pr3368-tests.log, pr3368-normal-recheck.log, pr3368-tab-diagnosis.log, pr3368-typecheck.log, pr3368-eslint.log, pr3368-precommit-local2.log and pr3368-ux-artifacts. Auto-merge remains disarmed for serial reviewed merge.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Early Stop preserves saved-question Retry for whitespace-only output and guarded recovery for uncertain server writes. CI follow-up normalizes the tracking task and gives both real-cockpit Stop outcomes a second exchange before history review. Verified 593 affected tests across successful runs, typecheck, lint and pre-commit; independent review complete. Hosted final-head CI remains required before merge.
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
