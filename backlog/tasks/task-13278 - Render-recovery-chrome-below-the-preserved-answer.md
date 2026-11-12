---
id: TASK-13278
title: Render recovery chrome below the preserved answer
status: Done
assignee: []
created_date: 2026-09-19 14:47
updated_date: 2026-09-19 14:47
labels:
- chat
- webui
- error-handling
- ux-audit
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After stopping a stream the partial answer survives intact, which is a genuine strength. But the interruption notice and four recovery buttons render above it: measured at character index 32 for the notice and 345 for the prose. The user content the feature exists to preserve is pushed below the apology. The notice also offers a provider fallback action when no provider failed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Preserved assistant content renders above the recovery chrome.
- [x] #2 The recovery action set matches the cause; a user-initiated stop does not offer provider fallback.
- [x] #3 The recovery actions are listed once, not repeated as prose above the buttons.
- [x] #4 Live completion feedback does not contradict a stopped, interrupted, or saved partial reply; successful completion remains announced.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Resuming existing accepted all-ten NN/G task on f46fb2c641, with fresh dev97ea9cd5fa included. Shared Message interruption/saved-partial/transport notices render before content; interruption action array unconditionally includes provider fallback even for an intentional stop, and sr-only Recommended actions repeats button labels. Reach actual full PlaygroundChat and compact SidePanelBody plus Message integration/stop/pipeline/model/status neighbors. Preserve exact reply/usage/reasoning/images/provenance and existing action guards; move status/recovery after preserved content, offer cause-appropriate actions and remove duplicated interruption action prose. ADR049/050 apply to unchanged ownership, no new durable architecture/protocol; ADR059 tracking. Own staged plan Docs/Plans/IMPLEMENTATION_PLAN_chat_stopped_recovery_13278.md. Two previously authorized reviewers remain source-only independent reviewers. No repository Python changes intended; Bandit N/A.
Twelve actual renderer/full/compact controls fail causally before repair: five received prose/reasoning/image notice orders, both real parent orders, deliberate-stop fallback, three duplicated accessible action sets and reasoning-only false Continue/Retry prose. Source reviewers independently confirm one shared production root and compact lacks onContinue. The same duplicate sr-only action list exists in the actual shared ErrorBubble/empty-response notices, so remove all three actual lists under AC3; keep real buttons/live roles and provider-error/empty/non-stopped fallback positives. Unused MessageContent has no production caller and is not introduced as a new renderer. Pipeline diagnostic MISSING_FINAL_ANSWER_MESSAGE remains unchanged; display a concise localized reason only for the qualified missing-final view.
All12 causal regressions now green, total85/85 shared integration/source guard. Existing three partial/transport/interruption blocks move unchanged after response reasoning/Markdown and images, before research/general toolbar. Only deliberate stop omits provider fallback; Retry/Switch and supported Continue retain existing activity guards, other failures keep provider fallback. All three actual shared recovery notices no longer repeat native action labels in sr-only prose. Neutral interruption and qualified concise missing-final copy avoids unavailable compact Continue promises; exact pipeline diagnostic contract unchanged. English source/public entries verified with official flatten, entire public en playground has0 malformed message-shape entries after the prior13567 correction. Two owned TS formatted with runtime-equivalence assertion. Connected pipeline/full/compact/model/metadata matrix, both compilers/static and current native build/Stop proof pending.
Both independent bounded final source reviews CLEAR. Connected shared pipeline/full/compact/model/metadata/stop matrix2107/2107 in52 suites; both clients compile. Final shared85/85 after two test-only lint repairs; touched3 TypeScript have0 introduced findings against HEAD. Owned formatting runtime-equivalent. No authored repository Python, Bandit N/A. Fresh fetched dev97ea9cd5fa remains included. Stage2 complete, current-build actual native Stop/order/cold/network qualification now in progress; native/physical/AT/packaged and all-ten boundaries unchanged.
Current native build1T8PKGIllJ9LhvdKx1mSe tokens/assets799/budgets595.2/791.4 pass. One short120ms fixture timing control completed before Stop could click; no failed-production claim or replay. Extending only private response text creates16.44s deterministic stream, actual keyboard Stop11881 retains exact prefix The library door opens. I am Mira, your guid above notice, three actions/no fallback; cold2 rows/exact text/actions survive, saved25 canonical unchanged. Original private reply restored. Native DOM exposed adjacent false sr-only Response complete while Stopped is shown. Extend this bounded shared presentation task/AC4 before edits to suppress success announcement for actual interrupted and transport-partial notices, preserving true-success and saved-status precedence. Stage2 reopens for causal controls and review; no authority/transport/state changes.
Final96 shared checks and52 connected suites2118/2118 pass; both client compilers, two final independent source reviews and static3 files0 introduced pass. Two owned formatting regions runtime-equivalent; English source/flat Chrome mirrors correct. No authored repository Python/Bandit N/A. BuildtXY7wZDN2v6IXwEXBFM3i tokens/assets799/budgets595.2/791.4 pass. Native12441 false/noCID keyboardStop exactpartial before one truthful notice/three actions/no fallback/no false success; cold2 rows/exacttext/draft/actions/build preserved. Saved25 full response deep-equal/no auto completion/settings write. Initial11793 timing control and11881 pre-supplement Stop are excluded from final-source claim; private fixture restored. /private/tmp/chat-stopped-recovery-13278-native-proof.json and final-stopped/final-cold screenshots/logs qualify native full desktop headless automation plus declared source compact controls, not physical/AT/packaged/ordinary settlement. Stop BODY focus/Dismiss focus/Copy acknowledgement remain separate action roots. All stages complete; only own completed plan retired and preserved privately. ADR049/050 unchanged ownership/status;059 tracking. Fresh dev97ea9cd5fa included on2026-10-09. This legacy task has no DOD checklist, so the initial check-dod attempt was rejected atomically and is corrected via supported task edit, with quality evidence in notes. Parent13458/all ten remain In Progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Received reasoning/prose/images precede shared recovery notices; deliberate Stop offers cause-appropriate actions; repeated action prose and misleading Continue copy are removed; false success feedback is suppressed for partial outcomes. Existing saved-status precedence, passive history and guarded actions remain. 2118 connected checks, both clients, two reviewers and current native Stop/cold/network/canonical proof pass within declared bounds. Parent all-ten acceptance remains open.
<!-- SECTION:FINAL_SUMMARY:END -->
