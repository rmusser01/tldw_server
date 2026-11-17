---
id: TASK-13559
title: Regenerate saved replies through both native chat interfaces without duplicating
  input
status: Done
labels:
- frontend
- chat
- ux
- bug
dependencies:
- TASK-13558
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete CM-01 native Regenerate client composition for full and compact ordinary/character chats using the existing server-owned reply sender and TASK-13558 admission contract. Preserve original rows and drafts, capture canonical assistant/direct-user source, validate original owner-issued admission separately, and retain explicit recovery when a saved alternate reply cannot be followed. Qualify exact ordered images/image-only input, loss/retirement/busy boundaries and actual fake-server contract; keep whole-program native/extension/AT acceptance open.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Full and compact ordinary and character native Regenerate send one canonical saved user with branch true and the existing operation marker, preserve all originals and append/follow only an acknowledged assistant sibling.
- [x] #2 Rendered H1 and draft/source guards reject stale or busy entry, validate current assistant/direct user and exact raw attachments before dispatch, and keep newer drafts and offscreen outcomes recoverable without automatic replay or keep.
- [x] #3 Strict original-admission and settlement parsing, explicit regenerate recovery/Open behavior, meaningful causal integration fixtures, both client type/static gates and two independent reviews qualify the unit while remaining program gates stay tracked.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new ADR. ADR049, ADR050 and accepted D7 section6.2 govern existing owner selection and explicit reply alternatives; ADR059 governs canonical tracking. Two independent state/transport design reviews recommend reusing native-history-character-send server receipt/recovery/follow tail for both ordinary and character Regenerate. Never relax normal client-managed pipeline into appending another user. Ordinary admission parser remains strict; reuse parser preserves original admission digest/messages/revision. Operation recovery keeps its new selection digest and labels the private regenerate action. No create/greeting/promotion or new endpoint/authority. Highest13558 checked across origin/dev/open PRs/both local backlogs; latestdev97ea9cd5fa included.
Native Regenerate entry/wire/recovery implemented across full/compact normal/character callers. Connected lifecycle423/423 and genuinely admitted actual Playground6/6 pass; both independent state/transport source reviews CLEAR. Repaired whitespace serialization, compact activity, shared provider ownership, Stop cleanup and source retirement during synchronous marker/busy publication. Broad31-suite run exposes29 failures/2errors in three old image/persona/overlay store doubles; isolated exact HEAD load reproduces the same failures, now repairing their missing real store state/setState boundary without weakening assertions. Fresh origin/dev97ea9cd5fa included.32 authored TS lint files have0 introduced findings; both client types pass. No authored Python/Bandit N/A. Plan Stages1–3 Complete; Stage4 qualification In Progress. Whole-program/native/packaged/AT/physical/large-history/participant gates stay open.
Qualification expansion: the repaired actual overlay store exposed a real legacy scratch-settings lease that stayed true after New. Add a causal current/retired source assertion and bind the existing settings lease to the rendered mounted draft revision/address while preserving first-ID history binding. This composes existing TASK13549 ownership rules and adds no protocol/registry/dependency. Final independent reviews and connected gates will include this small root correction.
Final qualified source and fixture checkpoint: 1,179/1,179 checks in33 suites;64/64 formatter-touched neighbors; both client types;37 authored TS files/zero introduced lint findings; shared owned formatting and diff pass. Both independent state/transport reviewers CLEAR. Causal held-save tests repair full/compact source retirement and exact optional preserveServerChatId forwarding, preserving first-ID binding and metadata. Inherited doubles and unchanged3-arg helper assertion repaired without dropping behavioral assertions. ADR049/050/D7/059 govern; no new ADR or dependency. No Python/Bandit N/A. Latestdev97ea9cd5fa included. Completed plan retained at/private/tmp/chat-native-regenerate-client-13559-completed-plan.md; only owned plan retired. Master design contains detailed evidence/limits. Whole-program/native/packaged/physical/AT/large-history/participant gates remain open.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Native Regenerate now works through both full/compact ordinary/character server-owned reply paths without duplicating input or modifying originals. Exact saved text/images, protected admission, assistant-only settlement and explicit recovery are qualified. Shared controller/Stop/publication and legacy late-save settings roots are corrected. 1,179 connected checks and both client types pass, zero introduced lint findings, two independent reviewers clear. Native browser/extension and remaining all-ten program work are still tracked separately.
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
