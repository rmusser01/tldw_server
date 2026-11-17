---
id: TASK-13541
title: Protect Temporary tab closure and truthful compact bulk outcomes
status: Done
labels:
- chat-ux
- extension
- data-loss
dependencies:
- TASK-13540
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Compact New, switches and history opening retain Temporary snapshots. Close removes the target snapshot, including an inactive or last tab, and bulk closure currently does not await acceptance or count actual removals. Require an explicit keep-or-discard choice at target deletion and preserve cancellation and truthful aggregate outcomes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Cancelled close keeps exact active, inactive and last Temporary tab content; acknowledged close removes only the intended owned target.
- [x] #2 Compact New and tab/history switching keep retained Temporary content reachable without a discard prompt.
- [x] #3 Single and bulk close callers await accepted outcomes and do not report declined Temporary tabs as closed; existing relevant confirmation is not duplicated.
- [x] #4 Causal checks and applicable client/static gates plus independent reviews are recorded; all-ten program remains In Progress.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Read-only root trace during TASK-13540 identified sidepanel-chat.tsx handleCloseTab and compact Sidebar.tsx single/bulk close continuations as a distinct reviewable deletion unit. Highest task IDs just verified local13540/dev13530/openPR13526. Governing accepted Temporary lifetime and user-control design; no new architecture decision, ADR-049/050 and ADR-059 apply.
Plan: Docs/Design/IMPLEMENTATION_PLAN_chat_temporary_tab_close_13541.md. Full caller trace: route handleCloseTab removes active/inactive/last snapshots without Temporary admission; compact Sidebar single deletion ignores the async close outcome and bulk forEach does not await or count actual closure. Bulk server deletion also omits the current owner requestScope. Reuse TASK-13540 native Temporary-content helper at target removal, return boolean acceptance, await sequential bulk closures and count only actual accepted local closes. Existing bulk confirmation must explicitly describe Temporary discard and pass exact current-call consent to avoid a second question. Preserve all retained New/switch/history controls. ADR assessment: existing ADR-049/050 and ADR-059 govern, no new durable rule or dependency.
Failure-first evidence: causal route/Sidebar run has23 failures and4 retained controls passing across both actual entrypoints; scoped Undo and same-ID/Cancel/unmount retirement controls fail4/4 separately. A wrong-cwd auxiliary write made no changes and its obsolete run is excluded; corrected retirement log overwrites it. Reuse the existing route/H1 helper and existing bulk helper current predicate. A component-local action revision qualifies explicit bulk choice and selection/Cancel/unmount changes without treating owned successful sequential closures as retirement. Scope the sole Undo helper caller using the existing restoreChat requestScope option.
Reviewer regressions reproduced: seven clean failures cover actual same-ID snapshot replacement during active/last own transition, Cancel-to-new bulk attempt blocked by retired busy state, and stale single Trash errors; two real-route caller retirement cases fail by returning true after an explicitly retired admitted choice. Fixture-only query/read errors were corrected and excluded from the overwritten causal logs. Repair final target conversation/address ownership, revision-owned loading cleanup, current error feedback, and pass the existing bulk current predicate through the existing tab preparation guard.
Final affected matrix:380 passing checks in12 suites with no skips; after removing one redundant unmount counter update, all23 actual Sidebar cases pass again. Both final client types pass. Effective unchanged-rule lint covers all6 authored TS/TSX:10baseline/10current/zeroadded. Four owned formatting regions, exact three English-only locale mirrors and git diff check pass. Two independently authorized reviewers clear final source/test ownership, scoped Trash/Undo, awaited outcomes and cancellation recovery; their review is source-only. Bandit N/A: no Python repository code is authored. Fresh origin/dev2c5f19d0328360d45bce99b8bae75ce252cc312f remains included. Evidence: /private/tmp/chat-temporary-tab-close-13541-final-tests.log, -sidebar-final.log, -web-types.log, -extension-types.log and -eslint.json. Initial fixture setup/query failures are excluded from the overwritten causal logs.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Protected the real compact tab-removal root with exact Temporary discard admission and intended target identity/address checks. Retained New/switch/history controls remain direct. Single and sequential bulk callers await accepted closure; declined or failed targets stay selected, local counts reflect actual closure and current account scope reaches Trash/Undo. Existing bulk consent explicitly describes Temporary discard once. Caller retirement reaches held history preparation; Cancel releases old busy state without clearing newer work, and stale single failure feedback is suppressed. Final affected matrix:380 passing checks in12 suites with no skips; after removing one redundant unmount counter update, all23 actual Sidebar cases pass again. Both final client types pass. Effective unchanged-rule lint covers all6 authored TS/TSX:10baseline/10current/zeroadded. Four owned formatting regions, exact three English-only locale mirrors and git diff check pass. Two independently authorized reviewers clear final source/test ownership, scoped Trash/Undo, awaited outcomes and cancellation recovery; their review is source-only. Bandit N/A: no Python repository code is authored. Fresh origin/dev2c5f19d0328360d45bce99b8bae75ce252cc312f remains included. Evidence: /private/tmp/chat-temporary-tab-close-13541-final-tests.log, -sidebar-final.log, -web-types.log, -extension-types.log and -eslint.json. Initial fixture setup/query failures are excluded from the overwritten causal logs. Bounded source/test unit complete. No positive native dialog, packaged extension/docking, physical/zoom/VoiceOver or participant claim; TASK-13540 and the all-ten program remain In Progress. Four owned plan stages completed and only the TASK-13541 temporary plan is retired; accepted design and evidence retained in the master document.
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
