---
id: TASK-13532
title: Preserve unread conversation scenes across editor and send failures
status: Done
labels:
- chat
- ux
- roleplay
- recovery
dependencies:
- TASK-13456
- TASK-13531
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the source-confirmed absence/error conflation in shared scene reads. Failed storage or profile reads must never certify an empty scene, let an editor overwrite it, or let native/normal dispatch silently omit it. Reuse existing storage certification, scene service, three writable editors, current preflight/error gates and native retries. Known absence still uses existing character-profile/neutral fallback; no new roleplay server capability or scene protocol.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Actual shared storage/service failure-first tests distinguish successful absence, chat/profile read rejection and nonpersistent fallback; known valid chat/profile precedence and migration controls remain valid.
- [x] #2 Shared chat/fallback readers reject unavailable scene data; all native/normal preparation consumers retain errors before provider dispatch or identity changes without assuming actor disabled.
- [x] #3 All three writable consumers (RolePlaySetupDrawer, ActorPopout, CurrentChatModelSettings) expose load failure, protect unread scenes from Save/Apply, and allow scoped retry; current loaded edits and known-absence fallback remain usable.
- [x] #4 Source, character, close/reopen and account/session retirement prevent held hydration or save callbacks from publishing or saving another view; reuse existing source/lifetime primitives.
- [x] #5 Relevant regression checks, both client types, no-added lint/owned format, canonical tracking/diff and two independent source reviews pass; Bandit scope assessment and exact native/backend/fixture proof limits are recorded.
- [x] #6 The approved master and parent remain current/In Progress, the bounded completed plan is retired after its evidence is preserved, and the verified repair is committed locally.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Duplicate scene-settings search found no equivalent. Latest remote dev5ec8c7939f is included83969a296b; previous repair de2ba9bb92. Highest remote13527/openPR13526/local13531 makes13532 fresh. ADR assessment: no new durable storage/API rule; ADR-049/050 source/lifetime and ADR-059 tracking govern. Plan: Docs/Design/IMPLEMENTATION_PLAN_chat_scene_read_recovery_13532.md. Shared actor-settings catches rejected reads as null and fallback returns defaults; existing drawer throw-only fixture bypasses actual failure. Traced three writable editors and four normal/native/selection read paths plus explicit profile application. Retain unsupported TASK13160/D7 capabilities and complete latest-dev/native/accessibility journeys as parent gates.
Failure-first shared read4/4 controls; sibling/drawer lifecycle initial8UI red5/3; actual H1 red4/36; retained Retry and writer ACK red4 plus false Saved UI case; stable caller-code/native display red5 plus one raw-notice contract expectation. Later initial authority/native ActorEditor controls red4/1 positive. All three editors now gate native controls and stale rendered Save/Apply/Retry; unavailable initial authority settles with recovery guidance rather than a spinner/no-op Retry. Three old drawer expectations on exact previous source de2 reproduced3 failures/21 controls and now retain the deployed apply context. Old model-prompt fixture writer now returns async true to match the real boolean ACK API. Stopped repeated unchanged two-modal timeout checks, inspected three neighboring native/Modal/cache implementations and measured6.3s isolated for its5s cap; retain all assertions and give only that real two-modal test10s. No global timeout increase or disabled test. ADR-049/050/059 still govern; no new durable scene/model namespace.
Final independent review caught a warm model-config cache dependency: fresh scenes still reread model-default storage. Added a real cached-model/query plus actual scene-reader failure-first control (1 failed/36 filtered), then reused captured existing model data on a cache hit; strict scene reads remain fresh. Unqualified-owner Retry is hidden with existing account/selection recovery guidance in all three editors. English copy is synchronized through the existing converter; no cache namespace, dependencies or model behavior added.
Second final review found retired-view feedback persisting into an unavailable new account. Failure-first controls reproduced all three editors (3 failed/65 filtered), then moved local completion error resets to existing view retirement before guard refusal. Fresh expanded checks: shared scene/editor 76 passed, model prompt cases 34 passed plus a real two-modal 5s CSS timeout, native/normal neighbors 586 passed plus an obsolete Llama source guard. The exact preceding HEAD source reproduces the source-guard failure unchanged; maintained it against deployed shared catalog/model resolver and current scope wiring, retaining Llama form/Save assertions. Reassessed remount test cost: exercise both cached numeric/remount cases through the existing actual Drawer/Form, keeping their assertions and reverting the temporary 10s cap; other real Modal cases remain. Cached model rejection proof declares getAllModelSettings I/O; actual model service catches storage failures to an empty object, so it is not actual storage rejection propagation.
The Drawer experiment did not remove the measured two-render generation fixture cost (one timeout while all five selected feedback/cache/static controls passed), so reverted that unneeded fixture parameter/prop. Kept original actual Modal/Form controls and every functional assertion. Both known 5.2–6.3s cache/remount cases now alone use local 10s caps; global and other test timeouts unchanged. Final Stage2 source is complete; Stage3 fresh gates remain in progress. No passing claim based on a filtered/skipped run.
Final fresh full matrix /private/tmp/chat-scene-read-13532-final-matrix.log:701 passed,1 pre-existing opt-in owned Quick Chat backend fixture skipped across30suites (no owned19765API fixture running). Scene service/editors79 passing. WebUI and extension types exit0,16 authored TS/TSX files zero added lint findings,17 owned formatting checks; two final reviewers clear storage/request and view/feedback boundaries. Native IAB /chat finite unavailable-owner error/guidance/disabled Apply and Cancel screenshot /private/tmp/chat-scene-read-13532-native-unavailable.png; drawer/menu closed, empty composer retained, no new tab/Chrome. Fresh remote5ec8c7939f included; master and parent updated, completed owned temporary plan retired. Bandit N/A TS/TSX/docs, no Python repository source. Existing backend is older; no full qualified/backend/packaged-extension/AT claim. No push, PR or merge.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Preserved unread chat scenes at the shared storage service and all writable/preparation callers, with strict persistent read/write ACKs, scoped load/save/retry, native busy controls and finite unavailable-owner guidance. Retained original known-absence/profile/model-cache behavior and stable failure codes; truthful notifications and retired error cleanup preserve current edits. Failure-first controls plus fresh701passing/one existing opt-in skip30suites, both client types,16no-added-lint files,17format checks, two reviews and bounded native unavailable-source screenshot validate the local repair. Master retains exact proof limits and remaining full-program/backend/extension/accessibility gates; parent stays In Progress. Verified changes are checkpointed locally with this canonical record; no push/PR/dev merge.
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
