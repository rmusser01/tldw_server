---
id: TASK-13535
title: Preserve unapplied character launch intents during startup hydration
status: Done
labels:
- chat
- ux
- webui
- extension
- character
dependencies:
- TASK-13458
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Native latest-dev Characters Chat-as launched a fresh card command while the preceding saved ordinary conversation remained selected. Connection/catalog or prior-chat hydration invalidates the owned preparation guard, which correctly returns retired without applying. The coordinator wrongly treats that still-current no-op as acceptance, consumes characterId/mode and silently leaves the old standard chat. Preserve unapplied current intents with the existing recovery/Retry; consume only an acknowledged application. Keep genuine source/account/input/H1/Clear supersession guards. Restore the actual Form test double subscription contract rather than disable its failing cases.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Causal regressions preserve a current character route and source when passive hydration retires preparation; Retry applies only with current inputs.
- [x] #2 Only accepted application consumes the route; genuine Clear, route/source/account and newer input supersession remain fenced.
- [x] #3 Actual Form regression tests exercise held card reads and changing readiness; existing test doubles provide the real store subscription interface.
- [x] #4 Native latest-dev Characters Chat-as either accepts the fresh card or visibly retains recovery without silently opening the old standard chat; prior saved chat remains reachable.
- [x] #5 Relevant regressions, both client types, formatting/lint/security assessment, two existing independent reviews and evidence are recorded; broad parent stays open.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Parent native finding and evidence /private/tmp/chat-native-card-launch-old-chat-20261007.png. Source independently traced by existing root_state reviewer at PlaygroundForm applyCharacterRoute and Playground route coordinator. The real prior-chat loader also hydrates any retained serverChatId before new card setup is accepted; preserve that acknowledged owner and draft. No new controller/protocol/dependency. ADR required: no; ADR-049/050 govern accepted history source and client lifetime, existing setup guards are preserved. TASK-13534 cookie repair is separately committed344db3ea8f; latest dev2c5f19d032 integrated68c828b295. Before this task source edits, all previous gates must finish. The broadened actual Form suite currently has80 setup failures because its local message-store double lacks subscribe; no difference in those files between pre/post dev merge. Fix the double contract and obtain meaningful red-first route/hydration failures. Highest local/dev/open-PR task ID13534 verified before reserving13535. Existing TASK13530 has a new upstream collision with an unrelated Knowledge task; do not edit either by numeric ID.
Current native production delta now keeps the unapplied launch at ?mode=character&characterId=4; explicit Retry accepts Mira opening and consumes only characterId. Evidence /private/tmp/chat-native-character-launch-recovery-13535-20261007.png and /private/tmp/chat-native-character-launch-retry-13535-20261007.png; the original ordinary four rows and saved Mira five rows remain in the disposable server. Remaining startup root: useMessageOption always enables the real server loader for the prior retained ID, competing with the fresh card command despite initializePlayground already excluding prior automatic bookmarks. Next bounded TDD step: suspend that existing loader via its enabled option only while a current fresh card command is pending; qualified saved targets, literal copies, retired/Clear intent and normal entry retain loader behavior. No source, input, H1 or account guard relaxation. Add actual-loader held-card regression and repeat native first-click launch; reviews must cover this further delta.
Final shared root consumes only applied character setup. A current retired attempt preserves the URL, original source and existing recovery/Retry; settled model fetch errors keep their finite recovery. Existing filtered route derivation now precedes useMessageOption, whose existing loader is disabled only for a current fresh characterId without chatId. Normal/mode-only, qualified saved targets, literal-copy masking and retired/Clear routes retain loading. No Form/loader production guard, protocol, dependency or controller added; prior H1 reads already started before entry may still settle only under existing fences.
Causal root red: /private/tmp/chat-character-launch-13535-red.log. Actual Form subscription mock repaired (production unchanged); three held-read connection/catalog/source controls retire without writes and Retry succeeds while eight prior supersession negatives remain. Actual coordinator/loader/H1/store/selected-assistant/WebUI storage held-card causal red captures the old source twice: /private/tmp/chat-character-launch-13535-loader-causal-red.log. Test acceptance boundary/server I/O are declared doubles. Initial controlled-storage fixture did not settle and exhausted its worker; bounded profiler exposed that fixture path, then existing actual WebUI shim controls were reused. Capture counters are reset at arrange; final selection is asserted through actual owned storage, not a controlled-only field. No test or guard disabled.
Final verification: 407 passed / one explicit environment-gated native Quick fixture skipped in ten suites, /private/tmp/chat-character-launch-13535-verified-matrix.log. Both final client types exit0: /private/tmp/chat-character-launch-13535-verified-web-typecheck.log and /private/tmp/chat-character-launch-13535-verified-extension-typecheck.log. Existing ESLint rules actually cover all four authored TSX files using apps as base: 47 baseline /47 current, zero added after normalizing only embedded Hook line-location shifts, /private/tmp/chat-character-launch-13535-static/eslint-covered-final.json. Earlier frontend-base comparison skipped shared files and is superseded, not counted as lint evidence. All eight owned formatting checks and git diff --check pass. Bandit N/A for this TSX/docs/tracking scope; no Python repository code changed. Two existing authorized independent reviewers clear the final source and evidence limits.
Native exact current worktree over integrated dev2c5f19d032 (68c828b295) with disposable local cookie/data and deterministic local provider: ordinary saved source -> Characters -> Chat as Mira accepts greeting preview on first click, consumes characterId only, no retry or unintended greeting send. /private/tmp/chat-native-character-launch-first-click-13535-20261007.png. Existing retired-recovery /Retry proofs /private/tmp/chat-native-character-launch-recovery-13535-20261007.png and /private/tmp/chat-native-character-launch-retry-13535-20261007.png remain. Prior ordinary four server rows and saved Mira five rows retained, /private/tmp/chat-native-character-launch-source-retained-13535-20261007.json. No browser interception/storage seeding, extra tab or Chrome window. This proves setup/retention, not an additional generated-turn or roleplay-quality/participant/packaged-extension/AT claim. Fresh fetch confirms dev2c5f19d032 included. ADR-049/050/059 apply; no new ADR.
The owned temporary three-stage implementation plan is complete and retired after preserving its decisions/evidence here and in the parent design. Stage5 and the all-ten parent remain In Progress; D7/scoped editor/preview provenance, packaged extension/docking, physical touch/zoom/themes, VoiceOver and participant gates remain open.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fresh card startup no longer competes with the preceding conversation loader. Only accepted setup consumes its command; current interrupted setup remains recoverable. Native first-click Mira preview succeeds and prior saved messages remain intact. 407 affected tests pass, one opt-in native fixture remains skipped, both client types pass, effective lint adds no findings, formatting passes and both independent reviews clear. Broader all-ten UX validation stays open.
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
