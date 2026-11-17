---
id: TASK-13628
title: Align chat shell tests with current runtime contracts
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Quick prompt root tests explicitly require WebUI aliases but are in the shared native suite. Two shell source checks assume double-quoted imports and an older catalog guard. Preserve insertion/retirement assertions, qualify the current Scene-only no-catalog guard and accept equivalent import quoting.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All23 quick prompt cases remain unchanged and pass after moving to frontend ownership
- [x] #2 Both shell guard suites pass while checking the stronger Scene-only catalog condition and exact shared composer import
- [x] #3 Relevant setup regressions, normal client types, bounded review and hooks pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no; current runtime/test contracts only. RED shared UI3failedfiles: alias resolution plus2sourcechecks,7casespass. Original quick prompt23/23 underWebUI. Plan Docs/Plans/IMPLEMENTATION_PLAN_chat_shell_contracts_13628.md. PR https://github.com/rmusser01/tldw_server/pull/3229.
Final WebUI5files71passed, native2guards9passed, original quick23casesbodyunchanged. Both normal client source checks exit0; bounded review clear. Review Docs/Reviews/CHAT_SHELL_CONTRACTS_13628_2026_10_13.md. Bandit N/A TS test-only. Normal hooks next. Fresh fetch remote dev79aa6f00a4509919ce78c11cab9b3be40025866a; PR mergeable CONFLICTING/DIRTY, explains absent pull_request CI after4abf. Four license/bot successes are not required-gate qualification. Latest-dev integration follows this unit.
Final normal hooks pass. Quick23assertions preserved; WebUI71tests, native9guards, normal client source types and bounded review all pass. Bandit N/A test-only. Latest-dev conflict integration is necessary to restore required CI; final whole-PR review and merge still pending.
Completed owned plan retired to /private/tmp/IMPLEMENTATION_PLAN_chat_shell_contracts_13628_completed.md; durable evidence remains in the review.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Moved the WebUI quick prompt root fixture without changing its23cases. Source guards now accept equivalent import quote style and require current Scene-only model-catalog gating. Seventy-one WebUI tests, nine native guard tests, normal source types, review and hooks pass; no production change.
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
