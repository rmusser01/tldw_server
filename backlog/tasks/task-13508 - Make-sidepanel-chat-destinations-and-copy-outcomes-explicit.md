---
id: TASK-13508
title: Make sidepanel chat destinations and copy outcomes explicit
status: Done
labels:
- chat
- ux
- extension
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Bounded Stage4 UI correction: accurately distinguish scoped selected-history expansion, existing route-only full extension opening, same-origin bounded draft/page-context copying, and configured WebUI callback handoff. Show persistent Opened/Partial copy feedback and source retention without claiming destination import. Preserve existing transfer formats, owner qualification and routing behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Native extension actions identify the actual full-page destination; configured WebUI callback and scoped history expansion remain distinct and retain their routes.
- [x] #2 The existing bounded copy action describes copied draft/visible page context and history/settings limitations; opened, shortened, missing-context and blocked-open results stay accessible after More closes without claiming Imported.
- [x] #3 The source draft is retained and sensitive draft/context content remains out of the bounded handoff URL; existing routing, duplicate-click and failure regressions pass.
- [x] #4 Source and generated English locale copy agree; relevant checks, shared types, extension compile, no-new lint and both independent reviews are recorded with explicit parent limitations.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved master Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md; parent TASK-13457 remains In Progress. Start13a7cfd9a7 on remote dev94854ca integrated292ea526d1. Official dedupe search found previous Done tasks12606/12610/12613, not this correction. Fresh67-PR path inventory has Header overlap with PR3152; inspected its header patch: editable full-title loading/rename baseline, independent of destination copy. PR3152 is reference only, not attached or integrated. No ControlRow/service overlap. ADR required:no for labels/status within existing payloads. Plan Docs/Plans/IMPLEMENTATION_PLAN_TASK_13508_chat_handoff_copy.md. Bandit N/A TS/JSON/docs. Remaining source account/navigation retirement, receiver owner/provenance/conflict/ACK and crossorigin local-ID/qualified server-reference issues remain parent continuity work; this bounded UI task does not close them.
Second independent reviews found copy overstatement (titles/URLs only) and active-page read failures masked by surviving document references. Corrected reference wording and explicit page-content exclusion. Moved the existing Form producer into existing sidepanel-chat-handoff-context.ts for transport-boundary testing; requested unavailable references preserve the existing truncated flag through the shared storage builder, without adding package fields or widening transfer. Context-only failures offer Retry and create/open no empty copy. Extraction baseline1pass; new producer/storage/UI red10fail26pass, including three intended missing-flag branches and misleading wording, plus a later blocked-open check affected by an unconsumed mocked call from the failed popup test. Final producer green36pass3suites, installed AntD Popover closes with result outside popup. Form overlaps inspected PR3147 persistence and PR2738 quick-ingest refactoring are independent references; neither touches this producer, and no PR integrated. Remote dev still94854ca. Native geometry, VoiceOver, receiver ACK/ownership/navigation remain parent gates.
Final13-suite matrix141passed, after supplying the actual full-Form readiness initializer and ref dependency to the stale source-extraction harness; no assertion removed or product dispatch change. Final shared types0, extension entrypoint/config compile0, no-new touched lint0, diff0. Official locale sync narrowed13English generated keys; unrelated mirrors restored byte-for-byte. Both independent reviewers clear source and added harness repair; source/callback/transport doubles remain declared. Bandit N/A TS/TSX/JSON/docs. Master checkpoint updated; own plan complete and removed before commit. Parent continuity/native gates stay open.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Explicit extension/WebUI destinations and Copy draft & page references clarify titles/URLs, excluded page content/history/settings, and source retention. Opened/Partial outcomes remain outside More. Existing producer/storage partial flag survives active-page read failures; context-only failure offers Retry without empty copying. Verified141checks13suites, shared types, extension compile, no-new lint, official locale agreement and diff checks; both independent reviews clear. Receiver ownership/provenance/ACK/navigation, full-title and native/package journeys remain parent gates.
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
