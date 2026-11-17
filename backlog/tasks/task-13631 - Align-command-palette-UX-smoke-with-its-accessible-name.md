---
id: TASK-13631
title: Align command palette UX smoke with its accessible name
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX smoke tries to find a command palette dialog, but current canonical English dialog/input name is Search commands. Existing owned browser proves first Ctrl+K opens and focuses it; preserve keyboard focus and actual settings-command execution while correcting the locator.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Smoke locator uses exact current accessible dialog name and retains focus/arrow/Enter/navigation assertions
- [x] #2 Owned single-page keyboard flow opens, focuses, executes settings and preserves original9conversations28rows
- [x] #3 Focused palette regressions, independent bounded review and normal hooks pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no; test-only current-copy contract. Original CI38353280083 fails locator /command palette/i. Owned19769home Control+K actualdialogSearchcommands,inputFocusedtrue; no browserwindow/tabopened. Original9/28deepEqual before navigation. Plan Docs/Plans/IMPLEMENTATION_PLAN_chat_palette_smoke_name_13631.md. PR https://github.com/rmusser01/tldw_server/pull/3229.
Owned keyboard flow firstControlK opensSearchcommands/inputfocused, filteractualGoSettings selected, Enter/settings andpaletteclosed. Original9/28deepEqual after; no baseline rewrite/newbrowserwindow/tab/context. Focused4palette suites63passed; bounded reviewclear. ReviewDocs/Reviews/CHAT_PALETTE_SMOKE_NAME_13631_2026_10_13.md. BanditN/A TS test-only; entireCI smoke notclaimed. Normalhooksnext.
Normal exact-file hooks pass. One-line exactaccessible-name repair, ownedkeyboard execution/preservation,63regressions andboundedreviewcomplete. No production/type/buildchange; allotherPRissues andfinalreview/rebase/merge remainopen.
Completedownedplanretiredto/private/tmp/IMPLEMENTATION_PLAN_chat_palette_smoke_name_13631_completed.md; durableevidenceinreview.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Corrected the UX smoke dialog locator to exactSearchcommands while retainingeverykeyboard/focus/selection/navigation assertion. Existingownedpage opens/focusesfirstCtrlK andexecutesGoSettings byEnter; original9conversations28rows unchanged. Focused63tests, independentreview andnormalhooks pass; wholeCI smokepending.
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
