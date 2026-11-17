---
id: TASK-13531
title: Reveal accepted quick prompts for owned variable editing
status: Done
labels:
- chat
- ux
- prompts
- accessibility
dependencies:
- TASK-13457
- TASK-13529
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Finish the source-confirmed long-template limit in the approved all-ten chat UX program. WebUI accepted quick prompts always collapse large text into a label; the shared owned raw-DOM equality fence then discards focus/variable selection. Reuse existing expanded restore behavior for accepted prompts, preserve ordinary paste/other producer policy and shared ownership/selection guards, and verify long Overwrite plus Append onto a collapsed draft through actual collapse/adapter/review and deployed binding.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Failure-first actual collapse/adapter/shared review plus deployed WebUI binding reproduces hidden variables/focus for long Overwrite and short variable Append onto a collapsed long source; short and unaffected raw controls remain valid.
- [x] #2 Accepted quick prompts use the existing expanded restore path so canonical text is editable and the existing owned selection can select its variable at correct append offset; normal paste/other publishers retain their collapse policy.
- [x] #3 Shared source/account/H1/input/command and raw-DOM equality fences remain intact; stale choices or moved focus cannot steal selection, and accepted selection survives relevant queued caret/measurement work.
- [x] #4 Relevant regressions, both client types, no-added lint and authored formatting, canonical task/diff checks and two independent reviews pass; Bandit scope assessment is recorded.
- [x] #5 Master/task records include ADR assessment, exact proof limits and remaining backend/native/assistive-technology program gates; parent remains In Progress.
- [x] #6 The bounded root repair and required task/design records are locally committed after verification.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Latest dev5ec8c7939f is integrated83969a296b; incoming Admin readiness code is independent. Existing private chat-copy task13527 is canonical13530 after verified backlog-py recreation, preserving original metadata/content/checks and avoiding incoming Admin13527. Highest remote13527/openPR13526/local13530 makes13531 fresh; duplicate search found no equivalent. Approved master: Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md. ADR assessment: no new durable rule; ADR-049/050 source/lifetime and ADR-059 tracking govern. Temporary IMPLEMENTATION_PLAN_chat_prompt_visibility_13531.md stages completed and retired after evidence was preserved here and in the master.
Root: actual PlaygroundForm QuickPromptInsertionReview setter collapseLarge:true hides variables at>1500; owned raw-DOM equality correctly drops selection. Existing restoreMessageValue(value,{wasExpanded:true}) reveals accepted prompts, retains last-variable semantics and Append offsets, and clears obsolete collapse refs. Normal paste remains collapsed. Three real queued caret callers (collapsed sync, short edit, pointer expansion) route through one private adapter scheduler; the helper keeps synchronous collapse state only. Explicit writes and unmount cancel the frame, including identical canonical text. All callers traced.
The final scheduler reuses captureChatDraftSource for LOCAL editing, with existing core draft/input/session revisions, focused original node and exact raw/display checks. Stale rendered source/session/collapse is rejected BEFORE replacing a current frame; postwrite revisions remain valid. Account events retire session restoreRevision through the existing watcher. New highlighted ranges cancel older work. Strict captureCompletionGuard remains unchanged for qualified review/dispatch/storage/import/send; owner verification is not required to edit local text. No new registry/protocol/dependency.
Failure-first: red.log6fail3controls, binding-only.log3fail6controls, owned-frame-red.log4fail6controls; current short-edit, stale raw entry, unchanged-text collapsed entry, newer selection and account-error local editing each have causal red/green evidence. Final lifetime-final.log28/28. Final affected matrix620/620 in30suites; WebUI typecheck and extension compile pass. Five authored TS/TSX files have zero ADDED lint findings against83969a296b; four owned formatting checks pass. Both independent reviewers clear final source/evidence. Logs: /private/tmp/chat-prompt-visibility-13531-{final-matrix,web-types,extension-types,lint,lifetime-final}.log; readonly format-check.cjs. Bandit N/A authored TS/TSX/docs; no Python source authored.
Actual worktree Next /chat native partial1800character paste/replacement yields Before x After caret8 before trailing6characters under account verification error. Screenshot /private/tmp/chat-prompt-visibility-13531-native-edit.jpg. Controlled input cleared and prior empty browser clipboard restored; no Chrome window/new browser tab opened. Native proof is local frontend editing only. Harness account/RAF I/O is declared and actual deployed JSX executes with qualified composer guard substituted at H1; full Form fixture retains surrounding mocks. Existing backend8000 is older. Full latest-dev backend generation/save/load/roleplay, successful qualified quick-prompt browser journey, packaged extension/docking, physical touch, font-only/200%zoom/themes and VoiceOver remain parent gates. TASK-13457/all-ten program remains In Progress.
All source/test gates were run against frozen final source before this local checkpoint; canonical tracking/diff checks precede commit. No push, PR or merge into dev is authorized by this unit.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Accepted long quick prompts now reuse expanded restore so variables are editable and selected at canonical offsets. One source/input/session/focus-owned adapter caret frame replaces three unowned timers, preserving newer highlighted ranges and local editing before verification without weakening storage/import/send/review authority. Failure-first cases and28focused/620affected checks pass, both client types/no-added lint/owned format/two reviews clear. Native partial paste replacement confirms caret8 before trailing text; input/clipboard restored. ADR and exact fixture/native/backend limits are recorded in the master; completed temporary plan retired. Bounded repair and canonical tracking are committed locally; parent remains In Progress pending latest-dev backend, packaged extension and accessibility acceptance.
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
