---
id: TASK-13457
title: 'Chat UX stage 4: accessible responsive controls and extension continuity'
status: In Progress
labels:
- chat
- ux
- webui
- extension
- nng
dependencies:
- TASK-13454
- TASK-13455
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve findings 5 and 9 across shared UI and both extension surfaces: viewport-safe model selection, meaningful accessible names and landmarks, contrast/focus, visible conversation title, and truthful handoff destinations/state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Model menu stays operable within 320px and 390px viewports and at 200 percent zoom; touch, keyboard, focus return, overflow, and desktop placement are verified using the existing popup component.
- [ ] #2 Greeting switches and history search have accessible names; textarea uses valid semantics, each document has a coherent heading/landmark structure, and keyboard plus screen-reader operation is checked.
- [ ] #3 Shared light/dark tokens meet WCAG AA normal text contrast and applicable focus/target requirements; reported target dimensions are assessed with spacing exceptions rather than asserted as failures without evidence.
- [ ] #4 Compact sidepanel always exposes the active conversation title, mode, and save status with a clear path to full details.
- [ ] #5 Open in full extension and Continue in WebUI accurately describe carried history, draft, model, and page context; server references are owner-validated, origin-local IDs are not treated as portable history, unavailable items remain recoverable at the source.
- [ ] #6 All three existing transfer mechanisms and actual native Chrome sidepanel docking/reopen are verified against their actual destination/field contracts, including drafts, truncation, expiry, changed account/server/workspace, local-only history, missing context and failed loading.
- [ ] #7 Source feedback distinguishes opened/unconfirmed from destination imported/partial; draft conflicts offer insert/replace/cancel before payload consumption, foreign local IDs do not block transferable drafts, and unproven server references remain unconfirmed without a new transfer/acknowledgement protocol.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Program: TASK-13453. Design: Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md. Base: origin/dev 75ab224081bf140ef52017c1a9b0a04f6878d488, verified 2026-10-04. User approved equal audience priority and all ten findings plus enhancements in five stages. Written-design approval and the requested critical review are recorded on 2026-10-04; application implementation has not begun. ADR required: no for the client remediation; ADR-049 and ADR-050 govern owner-bound history and native fork boundaries, ADR-059 governs canonical task editing. Reassess if implementation changes public APIs or persistence authority. Reuse existing shared UI, loaders, presets, and test harnesses; preserve temporary-chat privacy and account/workspace fences. Findings 5, 9. Reuse AntD popup, shared semantic color tokens, SidepanelHeaderSimple/ControlRow, and the existing sidepanel-chat-webui-handoff service. Reverify #3152 after it reaches dev. No CSP weakening, new handoff service, or credentials/transcripts in transfer URLs.
Scheduling clarification: Stage 4 follows the reliable state/navigation work in Stages 1 and 2 and may proceed while the conditional native roleplay server dependency in Stage 3 remains open. Stage 5 still requires all stage outcomes.
Pre-implementation review gate: document three existing mechanisms separately: selected-history expansion, same-origin one-shot draft/page-context/route-intent package, and WebUI URL-fragment package. The current bounded-package ControlRow action opens extension options; its label must name that destination. The package has no model/history/settings payload; verify any independent tab-state restoration before promising it. Source reports Opened/unconfirmed; destination reports Imported/Partially imported. No new acknowledgement protocol. Foreign local IDs cannot prevent draft import; portable references require provenance and the destination owner-bound loader, never direct server-ID authority. Reuse insert/replace/cancel conflicts before consuming payloads; verify truncation, unsupported/missing context, different origins/accounts/workspaces and actual destination URLs.
Stage 2 organization review identified adjacent baseline unqualified Option Sidebar and compact Sidebar/FolderPickerModal/legacy bulk consumers. Stage 2 .6 qualifies actual ServerChatList bulk pickers; .7 covers shared ChatSidebar Folders discovery. Stage 4 must capture existing owner/view operations through compact/Option folder consumers, qualify visible labels/feature availability, stop stale single/bulk callbacks and guard read-only/Temporary operations. Reuse the existing store/operation/transport; do not invent another folder cache authority.
2026-10-06: Started bounded TASK-13507 shared model picker/composer accessibility repair. Active-character paged history already implemented in TASK-13455.3; not repeated. Native responsive viewport/VoiceOver and extension/handoff gates remain open. Baseline94854ca3db6ec3eaa0c27b4e9537bdd47d222611 current; detailed plan Docs/Plans/IMPLEMENTATION_PLAN_TASK_13507_chat_accessible_model_picker.md.
TASK-13507 bounded picker/composer accessibility checkpoint complete: final9suites182passed, both independent reviews clear, shared types/extension entrypoint compile/no-new lint/locale/diff pass. Source/adapter AX proof recorded in master; native320/390px/zoom/themes/touch/VoiceOver/package and handoff continuity remain open. Stage4 stays In Progress.
TASK-13508 bounded destination/copy UI complete: truthful extension/configured WebUI names, titles/URLs-only copy with exclusions, persistent Opened/Partial results and unavailable-page Retry.141pass13suites, shared types/extension/no-new lint/official locale/diff pass, both independent source reviews clear. Master checkpoint records actual extracted producer, installed Popover closure and IO/source-harness limits. Source account/navigation retirement, receiver provenance/conflict/freshness/ACK, crossorigin local-ID and qualified server-reference work, compact title, native geometry/VoiceOver/package and complete journeys remain open. Stage4 remains In Progress.
TASK13509 bounded receiver delivery complete: exact draft whitespace, fresh explicit conflict choices, actual form revision edit/Undo retirement, current route/hash/unmount lifetime and real-reader Retry. Final158 related checks, both client gates, no-new lint and both independent source reviews clear. Master checkpoint records evidence and parent boundaries. Stage4 remains In Progress: cold readiness/initial auto-import, owner/provenance/source retirement, cross-origin local IDs/qualified references, receipt outcomes, compact full-title and native/browser/packaged/assistive checks remain open.
TASK-13512 destination ownership repair complete: actual cold composer restoration/account watcher/H1 guard, stale initial/explicit/Cancel retirement, guarded consume removal, page-context retirement and review recovery across readiness/route/fence changes. Latest dev7ba48 integrated at2f7fbb8653; final567+66checks/bothclientgates/no-addedlint/locale/diffpass; independent reviews clear. Master checkpoint records actual source/harness limits. Source action retirement/provenance/unknown owner review, cross-origin references, receipts and native/packaged/assistive gates remain open.
TASK-13519 source action root repair complete: actual restored/rendered composer owner, draft/input/H1 fence; stale start/read/write/open retirement and attempt/receipt identity; unique unpublished-copy cleanup. Final20suites600pass, bothclientgates/noaddedlint/diffpass, both independent reviews clear. Latest dev7ba48 remains integrated. Master checkpoint records exact source/harness/native ceilings. Source retirement unit closes; payload provenance, foreign local IDs/unqualified server references, destination receipts, native/packaged/geometry/VoiceOver and D7 prerequisites remain open. Stage4 remains In Progress.
TASK13520 bounded WebUI fragment reference repair complete: no origin-local lookup or direct payload server-ID selection; visible history-not-transferred notice names actual Chats sidebar. Final116checks/3suites, fresh bothclientgates/noaddedlint/locale/task/diffpass, two independent reviews clear. Real root/coordinator tests use Form/IO doubles; full fragment owner/readiness/conflict/settings/document/receipt and native/D7 gates remain open. Master checkpoint updated; Stage4 remains In Progress.
TASK13521 exact WebUI draft root repair complete: raw strings including empty/whitespace; actual hydrated owned Insert/Replace/Cancel for fragment and encoded query; fresh choice/owner/input gates; no fragment local IO; draft review independent of settings/document waits; matching parent cleanup and explicit revealed-query review. Final serial7suites252pass; both clients/no-addedlint7TSfiles/task/diff checks pass; two reviews clear. Master checkpoint records actual composer registry/account watcher and declared Form/native limits. Automatic settings/document/context qualification/receipt, source WebUI opener input lifetime, unknown-owner same-origin review, compact title, native/packaged/VoiceOver and D7 remain open. Stage4 remains In Progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
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
