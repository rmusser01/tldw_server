---
id: TASK-15203
title: Require explicit truthful recovery before repeating saved roleplay completion
status: Done
labels:
- chat
- recovery
- review
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two fresh independent reviewers confirm that ordinary Send silently reuses an unfinished same-input journal after completion dispatch uncertainty. The server receipt replay can generate again while the earlier provider is still running. Require existing explicit reviewed Retry, preserve durable unknown dispatch state and disclose that retry may generate another reply without adding another user input; keep owner fences and prior text.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Ordinary Send cannot silently adopt any unfinished original roleplay operation, including same exact input.
- [x] #2 Completion dispatch uncertainty survives cold journal reads; proven nondispatch remains distinguishable and original owner/input/result are preserved.
- [x] #3 Explicit original Retry keeps frozen IDs/receipts with truthful possible generation copy; held-provider and existing recovery controls pass both profiles.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: existing ADR-049 and original protected Resume contract. This bounded client/journal/copy fix changes no server schema or API and does not introduce a generation reservation framework. Existing explicit original-request replay remains supported; automatic adoption is the verified UI defect. Final review must assess actual semantics.
Independent follow-up confirms ordinary replay fix but identifies prior unsaved-text plus explicit retry lost-response branch: old result_text currently wins over new dispatchunknown. Add cold journal regression and preserve latest attempt uncertainty until a new validated completion receipt, retaining earlier text/receipt. Clarify fixture boundary: SDK onDispatch + transport failure uses controller stub/pure resolver and in-memory storage; unused providerWork promise is not a held real provider proof and will be removed. Actual H1 keyboard controls are separate.
Ordinary Send refuses all unfinished roleplay operations; only explicit original Retry may generate again. Durable unknown precedes transport and latest uncertainty outranks older retained text without discarding it. Actual held completion promise, journal and precise refusal cause qualify42 cases in both profiles, included in final190/profile; no physical provider concurrency claim. PR #3229; final bounded source review by two independent reviewers. TypeScript/JSON/Markdown-only unit: Bandit N/A locally; fresh full PR67-production-Python scan has0 introduced findings and0 scan errors vs exact merge-base. Installed extension/physical AT/touch/native-toolbar zoom/participants remain unrun. Overall merge and current Scene stability remain separate gates.
Renumbered from TASK-13676 to TASK-15203 on 2026-10-31 (TASK-13960): TASK-13676 now names only "Add capability-aware spelling hints to transcription" (backlog/tasks/task-13676 - Add-capability-aware-spelling-hints-to-transcription.md). Before 2026-10-31 both tasks used TASK-13676; Docs/Development/Backlog_Task_Id_Renumbering.md lists every renumbered id.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Ordinary Send refuses all unfinished roleplay operations; only explicit original Retry may generate again. Durable unknown precedes transport and latest uncertainty outranks older retained text without discarding it. Actual held completion promise, journal and precise refusal cause qualify42 cases in both profiles, included in final190/profile; no physical provider concurrency claim.
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
