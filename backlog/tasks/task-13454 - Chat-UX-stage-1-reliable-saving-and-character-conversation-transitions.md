---
id: TASK-13454
title: 'Chat UX stage 1: reliable saving and character conversation transitions'
status: In Progress
labels:
- chat
- ux
- webui
- extension
- nng
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve findings 1, 2, and 8 through truthful persistence and safe new-chat/character transitions. Reuse or integrate existing fixes after they reach dev; preserve prior chats, drafts, selected ancestry, and ownership checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Empty, temporary, local-acknowledged, server-acknowledged, pending, and failed states describe actual writes; a connected server or bare conversation ID never proves the latest turn was saved.
- [ ] #2 A standard local chat can be left, found, and reopened with its correct messages and configuration through the main conversation UI.
- [ ] #3 Standard local chat to explicit new character conversation to first turn succeeds with resolved provider and a new owner-bound native session; stale callbacks cannot bind the wrong view or account.
- [ ] #4 Unusable model/provider, unsupported context, invalid ownership, and server failures show a reason plus a valid recovery action before or after Send as appropriate, retaining the draft.
- [ ] #5 All new-chat entry points identify Saved, Temporary, or Character behavior consistently; Use no prior messages has an accurate effect and does not promote local history or disable owner guards.
- [ ] #6 An unconfigured first visit names connection/provider/model prerequisites and links to the existing setup UI, preserves the draft during configuration, and reaches a usable first send without introducing a new setup wizard.
- [ ] #7 Unknown first-create and known-ID append outcomes are distinguished across lost responses and reload; retry does not duplicate creation/provider dispatch or claim saving without an applicable acknowledgement/replay contract.
- [ ] #8 Greeting-only startup uses supported creation options and captured provider/model/sampling, verifies owner-bound stored greeting rows, dispatches no provider call, and cannot duplicate or publish stale creation on repeated activation or scope change.
- [ ] #9 Full shared preflight runs before new-character setup changes the active conversation; successful, failed and cancelled transitions preserve the source draft/owner, with an explicit choice before discarding temporary content.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Program: TASK-13453. Design: Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md. Base: origin/dev 75ab224081bf140ef52017c1a9b0a04f6878d488, verified 2026-10-04. User approved equal audience priority and all ten findings plus enhancements in five stages. Written-design approval and the requested critical review are recorded on 2026-10-04; application implementation has not begun. ADR required: no for the client remediation; ADR-049 and ADR-050 govern owner-bound history and native fork boundaries, ADR-059 governs canonical task editing. Reassess if implementation changes public APIs or persistence authority. Reuse existing shared UI, loaders, presets, and test harnesses; preserve temporary-chat privacy and account/workspace fences. Findings 1, 2, 8. Integrate/reverify #3147 and assess #3165/#3174/#3175 only when their contracts are accepted and merged. Do not implement a second save-status store or implicit local-to-server conversion.
Pre-implementation review gate: distinguish unknown first-create outcomes from known-ID prefix reconciliation. Preserve dispatched/unconfirmed state across reload through existing saved-mode recovery records; no replay without an applicable tested contract. Capture supported model/provider/sampling at native creation. Greeting-only startup uses existing create query options and owner-bound loading; actual stored rows, not a seed-status header, prove the greeting. Separate preview insertion from commitment. Full preflight runs before source detachment as well as dispatch. Keep successful-transition source drafts, scope reset to the active compact tab, and explicitly handle temporary content. Lost create/append response, rapid repeat, stale scope and zero-provider-call greeting checks are required.
Preflight clarification: validate the proposed fresh target without interpreting the source local owner as native ownership, then retain dispatch-time active-owner checks. Seed the displayed default/exact alternate index rather than choosing random again. Card changes between preview/create show the actual accepted opening and review action; they never silently reseed/create another session.
Execution started after user continue on 2026-10-04. Fresh origin/dev remains 75ab224081bf140ef52017c1a9b0a04f6878d488; PRs #3147, #3152, #3174 and #3178 remain open and are not imported. First independently reviewable delivery: shared clear/new-chat actions release the active history controller and fence late callbacks, including actions outside the mounted provider; preserve compact logical-view isolation and cancelled navigation. Plan: Docs/Plans/IMPLEMENTATION_PLAN_chat_ux_safe_clear_2026_10_04.md. ADR required: no; this restores ADR-049 ownership lifecycle through existing reset/event contracts without new APIs or storage authority. Other Stage 1 acceptance criteria remain open. Reuse installed dependencies via ignored worktree symlinks; no package installation.
Safe-clear delivery verified 2026-10-04: optional ancestor reset after accepted navigation plus mounted owner-qualified clear-event reset. Initial real-controller inside/outside regressions failed on retained owner; outside delayed capture failed before the mounted fix. Final complete coordinator/navigation/controller suites: 132 passed; active-only reset preserves another logical tab. WebUI typecheck exit 0 using installed dependencies (shared test-helper Playwright types linked locally). Touched ESLint: zero errors and no new warnings against HEAD; existing Playground 18/controller-test 37 warnings remain. Prettier flags the same four files on both HEAD and delivery; existing style retained without unrelated formatting. git diff --check clean. Both authorized independent reviewers found no Critical/Important issues. Bandit not applicable to TypeScript/docs-only scope. Local dependency symlinks are excluded from staging. Application implementation has now begun; this supersedes the preparation checkpoint above. Stage 1 remains In Progress; no complete acceptance criterion is claimed by this foundation fix.
Character creation setup delivery: create uses the already resolved model/provider and only explicit supported temperature/top_p, preserving zero and backend character defaults when omitted. No penalty remapping or new API/storage store. Plan: Docs/Plans/IMPLEMENTATION_PLAN_chat_character_creation_setup_2026_10_04.md. Completed safe-clear plan archived in commit 7496dc9a6c and removed per policy. Baseline actions suite 61 passed; three create-input regressions failed on omitted fields; final complete suite including qualified-provider precedence, explicit zero, absent sampling and stale ACK: 62 passed. WebUI typecheck exit 0. Touched lint zero errors/no new warnings (action-test baseline 103 warnings). Both files have baseline Prettier failures retained without unrelated formatting. git diff --check clean. Both independent reviewers verified schema/endpoint/factory compatibility with no Critical/Important issues. Bandit not applicable to TS/docs-only scope. This verifies client inputs with external server doubles; actual lifecycle/reopen stays in Stage 5. All wider Stage 1 acceptance criteria remain open; continue shared preflight, drafts, recovery, greeting startup and explicit modes.
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
