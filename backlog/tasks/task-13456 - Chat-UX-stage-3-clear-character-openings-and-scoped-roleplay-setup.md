---
id: TASK-13456
title: 'Chat UX stage 3: clear character openings and scoped roleplay setup'
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
Resolve findings 4, 6, and 7 and approved roleplay enhancements: prominent greeting, familiar identity/context terminology, predictable editing scope, reusable setups, and truthful next-turn context preview. Keep unsupported native context extensions tracked against the existing server contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A selected card replaces generic onboarding with its character introduction and visible greeting; Start with this greeting persists at most once through the valid owner-bound path, and users can instead type the first turn.
- [ ] #2 AI character, Your role, Scene, Inherited lore, and Additional pinned context are distinguished with concise contextual help; zero pinned sources never means no inherited character context.
- [ ] #3 Every editor identifies This conversation, Character library, or Provider-model default; staged editors support Apply/Cancel and dirty state, immediate controls state their timing, and library/default writes require their explicit existing action.
- [ ] #4 Existing saved roleplay setups can be previewed, applied, modified, and reused without changing library cards or global model defaults implicitly.
- [ ] #5 Next-turn summary distinguishes selected, supported/planned, server-acknowledged, excluded, and unknown context; it does not claim lore injection or custom prompt/scene support from UI selection or route presence.
- [ ] #6 Imported PNG/JSON card, inherited worldbook/dictionary, custom scene/prompt/context, cancel/retry, and reopened-session restoration checks pass for supported contracts; unmet native resume support remains open under TASK-13160 and its consuming enhancement cannot be marked complete.
- [ ] #7 Greeting preview, alternate selection and reroll do not alter transcript or request history; own-message and greeting startup verify their actual stored rows and context separately.
- [ ] #8 Apply reports local and server acknowledgements or failed/unknown outcomes under captured owner/view/account/workspace scope; partial retries skip acknowledged operations and stale completions cannot publish into another conversation.
- [ ] #9 A field-to-scope/support map governs editor actions; supported conversation edits and Cancel leave provider-model defaults and another conversation using the same model unchanged on switch/reopen.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Program: TASK-13453. Design: Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md. Base: origin/dev 75ab224081bf140ef52017c1a9b0a04f6878d488, verified 2026-10-04. User approved equal audience priority and all ten findings plus enhancements in five stages. Written-design approval and the requested critical review are recorded on 2026-10-04; application implementation has not begun. ADR required: no for the client remediation; ADR-049 and ADR-050 govern owner-bound history and native fork boundaries, ADR-059 governs canonical task editing. Reassess if implementation changes public APIs or persistence authority. Reuse existing shared UI, loaders, presets, and test harnesses; preserve temporary-chat privacy and account/workspace fences. Findings 4, 6, 7. Reuse RolePlaySetupDrawer/Preview, ChatGreetingPicker, useChatSettingsRecord, saved-role-play-setups and compatibility helpers. TASK-13160 is the existing conditional prerequisite for snapshot-based native resume; this task consumes it instead of inventing a competing prompt protocol. Existing native character path rejects client scene/actor/custom context; expose that honestly until authoritative support is delivered.
ADR detail: backlog/decisions/002-character-conversation-behavior-snapshot-and-fenced-completion.md governs snapshot-authoritative native resume and supported versioned settings mutations. TASK-13160 is a prerequisite, not a promise that arbitrary client scene/prompt overlays become supported. Never inject client overrides or fall back to less guarded completion; assess any missing durable API before extension and retain its consuming enhancement as open.
Pre-implementation review gate: greeting preview/reroll changes neither transcript nor request history; own-message startup inserts no assistant greeting. Reuse Stage 1 greeting persistence. Label new-chat setup Use setup for new chat and existing supported edits Apply to this conversation. Preserve explicit live sections rather than removing every autosave; Cancel controls unapplied fields only. Map fields to actual storage/completion support before changing labels. Conversation generation overrides currently support temperature/top_p/repetition_penalty/stop; model-default fields do not become conversation fields by renaming. Extend ordinary settings helpers with captured request scope and explicit local/server failed/unknown outcomes; retry only unresolved operations, and reject stale Apply publishing. Two-chat model isolation, held PUT across scope changes, partial retry call counts and nested-focus checks are required. TASK-13160 still gates exact advertised native behavior mutation, not arbitrary client overlays.
Starting the approved explicit character-opening unit under TASK-13454.11, linked to the existing Stage1 native greeting prerequisites. Independent source reviews precede implementation. Scoped roleplay setup and full Stage3 acceptance remain open.
TASK13456.1 closes shared greeting placeholder display/alternate label root on latest dev e3c345b: explicit names or blank User fallback, unknown characters retained, raw seeds/IDs/index/history/editor/wire unchanged.85/85 checks in7suites, both client types, no-added lint/formatter, two source reviews pass. Evidence is declared UI/H1 fixtures, not native reload or full backend/editor. Master updated; local commit includes task. Stage3 remains In Progress: scoped edits/acknowledgements/lore/native support and live journeys still required.
TASK13532 starts the actual scene reader failure root after de2ba9bb92: failed chat/profile reads become defaults, bypass drawer load-error gating, and can be written by RolePlaySetupDrawer, ActorPopout and CurrentChatModelSettings. Native/normal preflight readers also assume actor disabled on failure. Plan records strict shared reads, three sibling editor gates, current source/session lifetime controls and exact remaining backend/native/AT limits. No new roleplay capability or durable architecture rule; parent remains In Progress.
TASK13532 scene absence/error root verified: strict actual persistent chat/fallback reads, three editor Save/Apply/Retry lifetimes and native controls, true write ACK, stable scene result codes and human notices. Final701 passed/one existing owned-native opt-in skip30suites; both client types,16files no-added lint,17owned format and two independent source reviews pass. Native current-worktree /chat finite unavailable-account scene error/guidance/disabled Apply/Cancel proof saved at /private/tmp/chat-scene-read-13532-native-unavailable.png; drawer/menu closed and existing tab retained. Master preserves red-green evidence and exact native/backend/storage/cache-double limits; owned plan retired. ADR049/050/059, Bandit N/A TS/TSX/docs. Current latest dev5ec8c7939f remains included. Parent remains In Progress: scoped model/conversation ACKs, lore/provenance/native capabilities and full latest-backend/extension/AT journeys remain open.
TASK13548 completes truthful selected/planned projections at the shared roleplay/template/cockpit/rail roots, additional versus unknown inherited lore, scene drafts and absent-preview footprint.464/464 checks in25 suites, both client types,28-file no-added lint/23 formatted modules, two independent clear source reviews and retained-IAB native final labels pass; all5 saved transcripts unchanged. Master design records artifacts and limits, owned plan retired. Parent remains In Progress: scoped settings ACK/cache/retry/editor commitment, actual inherited snapshot/capability preview and native/Resume/D7 prerequisites remain required.
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
