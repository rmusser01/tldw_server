---
id: TASK-13581
title: Clarify the scope and unit of chat finder badges
status: Done
labels:
- chat
- ux
- clarity
dependencies:
- TASK-13580
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Accepted chat UX follow-up: a public Full finder search acknowledges one result while the Server badge displays10 global chats without naming that scope. The existing Folders badge counts unique chats, not folders. Clarify these existing labels in shared Full WebUI/extension without changing queries, persistence or account guards. Compact has no corresponding badge.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Server badge names its global total; Folders badge names its chat unit; zero/unverified/stale handling and source selection remain intact.
- [x] #2 Real localized Sidebar behavior, affected owner/finder controls and full-client types/static checks qualify the minimal shared change.
- [x] #3 Current public Full and narrow finder geometry and exact canonical saved rows are verified in the single owned browser, with normal hooks and recorded limits.
- [x] #4 A recoverable paged refresh preserves its last acknowledged total alongside cached rows instead of relabeling loaded row count as the global total.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new durable rule; ADR049 source-owner and ADR059 canonical tracking govern existing behavior. Design basis is accepted all-ten chat remediation and closure ledger follow-up; public read-only RED /private/tmp/chat-finder-count-13580-red.json has global Server10/search total1. Existing Folders fixture/source proves one folder containing two unique chats produces Folders2. Reuse existing labels, real i18next, segmented wrapping and overview query; no new state/query/count helper. Plan Docs/Design/IMPLEMENTATION_PLAN_chat_finder_badges_13581.md. Primary checkout/unowned IAB untouched.
Independent read-only review found the deeper shared count root: deriveServerChatHistoryViewState overwrites acknowledged paged total with previousData.length on recoverable error (25 cached of60 becomes25). Reuse existing resolvedTotal through this existing helper (one production caller/two direct test callers), preserving owner/query/stale-state semantics. Real hook25-of60→429 RED/green control required before code. Existing app ICU adapter is used for folder singular/plural and missing-key fallback; base source labels remain separate. No new query/count state/helper/dependency.
First new RED exposed6 intended count failures plus3 invalid visibility assertions on Antd hidden radio inputs and4 inherited obsolete Shortcuts queries. Exact HEAD original Sidebar fixture independently reproduces4 failures/24passes: actual current Navigation label replaced old Shortcuts. Correct only those fixture queries while preserving expansion/storage assertions; new radio assertions use accessible-role presence, actual native geometry remains required. Real hook causal assertion receives25 instead of60. No production change yet. Existing English public locale has6 unrelated regenerateMessage drift keys; generate only new count keys using existing flatten helper, preserve that baseline drift.
Causal RED after excluding/fixing fixture diagnostics:6 intended failures/55passes across61 checks. Real paged hook returns25 for retained60 after429; real ICU Sidebar lacks total scope, singular/plural chat unit and fallback. No production source changed before this RED. Shared helper callers confirmed:one production/two direct tests. Preserve actual cached rows and stale state, use existing resolvedTotal.
Initial connected verification:15 suites290/290, both full types exit0,4TS differential lint baseline2/current2/zero added; production build/tokens/budgets593.2KB shared600/795.5KB chat900 pass. Actual desktop native label RED shows Server (10 total) scrollWidth120/clientWidth58 and computed whiteSpace nowrap despite existing whitespace-normal class. Ant Design style authority clips labels; repair this existing shared wrapping rule before completion. Public search acknowledges total1/global10 and source remains selected. Observer-runtime ReferenceError is an excluded harness diagnostic, no product console error.
Final first wrapping build passes290 connected checks/both types/lint0 added/build budgets. Actual desktop geometry has no overflow but visual review rejects four58px columns breaking Server/Folders/device across lines. This is not qualified usability. Reuse the installed Segmented vertical orientation at default size with left-aligned wrapping labels; preserve same radio values, count guards and selection. Native arrow navigation and desktop/390/320 readability required before closure; no new component/query/stylesheet.
Final source qualifies15 connected suites290/290, both full types exit0,4-file lint baseline2/current2/zero introduced and emitted build/tokens/budgets593.2KB shared600/795.5KB chat900. Final vertical installed Segmented passes actual1280/390/320 readable single-line28px geometry and desktop native Down/Up. Public search acknowledges1/global10; original two IDs/blank composer preserved. Final observer GET-only; nine original canonical responses28 rows deepEqual with no baseline rewrite. Earlier hidden-radio, obsolete Shortcuts, observer harness and fragmented wrapping diagnostics excluded. Two independent bounded source reviews CLEAR. Bandit N/A TS-only. Two generated locale keys exact; six unrelated baseline drifts retained. Evidence /private/tmp/chat-finder-badges-13581-*; normal exact-owned hooks next. Packaged/physical AT/studies and broader all-ten not credited.
Normal exact-owned repository hooks pass without bypass. Final canonical nine responses28 rows remain deepEqual; native1280/390/320 visual checks and actual keyboard passed. All bounded AC/DOD complete. Only own three-stage plan retires to /private/tmp/IMPLEMENTATION_PLAN_chat_finder_badges_13581_completed.md; incremental commit follows. Master/all-ten remain open.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Clarified global Server totals and unique folder chat counts, kept acknowledged paged totals through recoverable refreshes, and made the existing source selector readable and keyboard-operable at desktop/mobile widths. Six causal failures→61 focused and290 connected passes; both client types, zero-added static checks, current build/tokens/budgets, canonical9/28 preservation and normal owned hooks pass. Two independent bounded reviews clear. TS-only Bandit N/A; package/physical AT/studies and broad all-ten work remain open.
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
