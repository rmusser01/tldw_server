---
id: TASK-13457
title: 'Chat UX stage 4: accessible responsive controls and extension continuity'
status: To Do
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
