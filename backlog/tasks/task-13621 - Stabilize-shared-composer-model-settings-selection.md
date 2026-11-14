---
id: TASK-13621
title: Stabilize shared composer model settings selection
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Actual model-store mounting reproduces runaway renders and heap exhaustion; restoring shallow equality for the shared composite selector preserves settings values without false snapshot changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Actual store one-time prompt and OpenUI/source tests complete without diagnostic adapters or heap increases
- [x] #2 Both client types, relevant regressions and independent review pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Found during TASK-13620. Bounded trace excludes setup and model writes; shallow equality-only diagnostic changes firstcase from OOM to250ms pass. ADR assessment: no new durable rule; use existing zustand/shallow convention as current settings panel does.
Sharedproductiondelta usesexisting shallow forcompositeprimitiveprojection; actualvalues/directsubscriptions/revisionguardsunchanged. Independentboundedsourceclear. Actualstore source53/cockpit31/shell63 andmodelregressions30qualified. Review Docs/Reviews/CHAT_COMPOSER_SELECTOR_EQUALITY_13621_2026_10_12.md. ADRassessment nonewADR; Bandit N/A TS/Markdown; finaltypes/hooks/build/wholePRgatesremain.
Final verification: the actual model store completes the source/OpenUI suite, all relevant model persistence/recovery/synchronization controls pass, and both normal client type configurations pass. Independent bounded source review is clear. No heap/timeout increase or ownership guard relaxation was introduced. Commit hooks are next; whole-PR build and merge acceptance remain separate.
Normal exact-file pre-commit hooks passed. Completed unique plan retired to /private/tmp/IMPLEMENTATION_PLAN_chat_composer_selector_equality_13621_completed.md. PR: https://github.com/rmusser01/tldw_server/pull/3229.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added existing shallow comparison to the shared composer's primitive model-settings projection, resolving actual-store render churn while preserving updates and ownership/revision checks. Actual-store suites and model recovery regressions pass; both client types, bounded review and normal hooks pass. Final whole-PR build and merge gates remain open.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
