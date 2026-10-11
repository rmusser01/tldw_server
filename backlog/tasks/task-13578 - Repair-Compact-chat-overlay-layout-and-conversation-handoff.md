---
id: TASK-13578
title: Repair Compact chat overlay layout and conversation handoff
status: Done
labels:
- chat
- ux
- root-cause
- compact
- accessibility
dependencies:
- TASK-13577
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Current native390px Compact overlay computes position relative because Sidebar combines relative and fixed; it occupies288px of flex layout and squeezes chat to70px. New and selected history leave the overlay open. Repair the shared overlay geometry and explicit handoff using existing focus/modal patterns while preserving docked Pro and original load/account fences. Generated metadata Copy/provider pairing/full D7 matrix remain separate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Overlay preserves the chat width and blocks background interaction with usable keyboard dismissal/focus.
- [x] #2 Explicit New, tab and Recent local/server choices hand off to chat without a late old action closing a reopened overlay; docked Pro and blocked/retired loads remain safe.
- [x] #3 Behavior controls, types/static/security/hooks/current native geometry and task/evidence records qualify the bounded repair.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new durable architecture rule; existing ADR049 governs original load/account ownership and ADR059 tracking. Native current overlay width288/main70 at390, position relative, no dialog semantics/background inertness. Reuse existing installed UI focus/modal support; minimal root layout repair and user-triggered handoff, no new controller/store/dependency. Master13458/all ten remain open. Scope and red controls precede source edits.
Stage1 native390 RED saved /private/tmp/chat-compact-overlay-13578-native-red.json: relative sidebar288 leaves main70, no modal/inert. Mounted original causal RED12/14 plus capture-phase palette loader RED1. Two authorized independent source reviewers found early composer focus before draftReady, owned Drawer default-focus mismatch and an undismissible null Suspense fallback; repair now uses a thin route Drawer outside lazy Sidebar content, existing ChatErrorBoundary scoped fallback, actual ready callback and guarded presented Find. Existing H1/account/draft admission remains unchanged; local UI opening revision grants no transport authority. Added actual Sidebar+Drawer/Host controls and declared held/rejected content, newer-focus/close-reopen/account ABA. Verification/native closure pending. ADR assessment unchanged049/059; TS-only scope Bandit N/A.
Stage2 complete: original opening/account now gates retained Drawer Close and presented true/false before any ref writes. Focused real Drawer/Sidebar/nested Rename and pending palette retirement controls5 passed. Rename initial failure was test-mode duplicate title IDs, not actual row bubbling: no ContextMenu source change. Connected tests/types/static/current build and native remain pending.
Final source: thin existing Drawer outside lazy Sidebar, original opening/account guards all close/presentation/focus writes, passive production-ready callback after inert removal, one-use presented Find, accepted boolean New/tab/local/server handoff, background shortcut/palette retirement, local Tab boundary on existing workspace with exact DOM ancestor and nested portal exclusion. Final13 connected suites553/553; both client types exit0;9TS baseline42/current42/zero introduced lint; guarded AST-equivalent formatting; final webpack/tokens593.2KB shared600/795.4KB chat900 pass. Both authorized source reviewers bounded CLEAR. Current one-owned-tab native proof: held actual chunk dismisses with opener focus and cannot resurrect; full-width/inert/modal; initial ShiftTab and80 native forward/reverse Tabs stay inside; nested Rename Escape retains Chats; already-active tab waits real ready focus; New blank/zero rows; actual Recent H1 source exact IDs; docked928px chat with288px sidebar, collapse/reopen and mobile return. Network collector includes completions and all chat mutations: zero unintended writes; H1 selection POST is a read capture. All9 canonical responses/28 rows deep-equal. Evidence /private/tmp/chat-compact-overlay-13578-final-{cold,modal,handoff,geometry,preservation}.json and mobile/desktop PNG. Exclude observed label whitespace, inert-background role locator and duplicate docked Collapse diagnostics; outer debug main1280 is not workspace main928. Real local Recent/held retirement is mounted IO qualification, not native local import; packaged extension/AT/studies unmeasured. TS-only Bandit N/A. Latest dev3b68f2dbc2daa1b94099a8a3d56c809fd2c808db included; hooks/commit remain. Master/all ten remain open.
Normal exact-owned13-file pre-commit hooks pass without bypass. All four bounded stages complete; own plan archived /private/tmp/IMPLEMENTATION_PLAN_chat_compact_overlay_handoff_13578_completed.md and removed. Ledger records native scope and excluded diagnostics. No PR/push/merge or all-ten closure; incremental commit follows.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Compact Sidebar uses the installed modal Drawer without shrinking mobile chat. Exact original opening/account guards dismissal and focus; accepted New/tab/Recent handoff waits actual composer readiness. Local Tab boundary covers outer autofocus/first/last while respecting nested portals; pending content remains dismissible.553 connected/both types/zero introduced static/two source CLEAR/current build/budgets/normal hooks pass. One native headless tab qualifies held chunk/full width/80 Tabs/nested Escape/ready focus/New/tab/Recent/docked controls and zero unintended writes. Nine canonical conversations28 rows preserved. TS-only Bandit N/A; latest dev included. Native local import, packaged extension/AT/studies and wider parent queue remain open.
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
