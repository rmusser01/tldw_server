---
id: TASK-15410
title: Establish two settled turns before chat cockpit history-review smoke
status: In Progress
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the baseline-equivalent UX Smoke Gate failure observed on PR3319 run41328474660 and independent baseline-equivalent run41327131002. The Stop path may retain an assistant answer, so its recovery-only follow-up is skipped and the test requests history review with only one exchange. Production intentionally hides review for one exchange. Make the existing follow-up unconditional before selected-history review; do not change production UI, relax assertions, add retries/timeouts, or duplicate older unrelated TASK15101 restoration work. Parent PR3319 integration TASK15400; explicit requester asks all issues fixed before auto-merge. ADR assessment: no new ADR, test precondition correction only, architecture unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focused smoke establishes a second settled turn before selected-history review whether Stop retained an answer or not
- [x] #2 Production visibility rules and test assertions remain unchanged
- [ ] #3 Focused local validation or exact local blocker recorded, final CI UX gate verified before auto-merge
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Baseline independently verified by read-only agent01a19816: allthreeattempts missed Review conversation history with exactlyoneuser/assistantpair. Independent31805c63 run has identical frontend/backend/mock/workflowtrees to base54b321dc and samefailure; no PR-introduced UI regression. Change only the relevant E2E test setup, not production.
Implemented in d9465bba5a52f942b1dd8065de527ba22907c57d: moved the existing eight-statement settled follow-up after the recovery conditional, no production UI or assertion/timeout/retry/skip change. TypeScript5.9.3 parse/transpile0; AST comparison preserves1221assertion/helper-call ASTs; focused Playwright discovery1Chromiumtest passes using existing shared dependencies via NODE_PATH, configuredhooks0, whitespace0. Initial discoverywxt resolution fixed by existing dependency path only, no installs. Real-server local E2E not run because this worktree has no standalone frontend bundle; report /tmp/embeddings-pr3319-ui-fix-report.md. CI UX gate must pass before final AC3/Done and parent auto-merge. Bandit not applicable to this TypeScript-only test fix. No servers/PG resources touched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
