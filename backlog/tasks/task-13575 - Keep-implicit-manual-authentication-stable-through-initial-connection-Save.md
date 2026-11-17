---
id: TASK-13575
title: Keep implicit manual authentication stable through initial connection Save
status: Done
labels:
- chat
- bug
- ux
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Native TASK13573 review reproduced retained first-Send text with no chat IO after public connection Test/Save. Draft owner normalizes omitted authSource to manual; shared request target comparison does not, so its unchanged original scope is falsely revoked. Align only that existing default in the shared comparator; preserve cookie-session, server, auth mode, organization, API-key and principal gates across consumers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implicit manual and explicit manual target scopes compare equally in both directions; cookie-session and other target changes remain rejected.
- [x] #2 Original first-Send snapshot survives public initial connection Save without a reload and retains original-account checks.
- [x] #3 Connected client regression, both types, authored static checks and normal commit hooks pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new ADR. This is a consistency repair to existing manual authSource defaults in chatDraftOwnerKey, serverChatMirrorOwnerKey and connectionAuthoritiesMatch; it changes no durable protocol or ownership rule. Investigation found all shared comparator callers in request-core, service-prompts, background proxy/extension, credential locks, loader authority, connection authority and onboarding. Two independent reviewers remain authorized; no new browser is needed.
Causal RED4: captured original Service Prompt snapshot and omitted/null/empty manual target cases fail with the native request_config_scope_changed error. Minimal shared comparator repair defaults authSource using the same existing manual rule as owner/connection identities, leaving other target and separate principal/credential gates intact. Connected comparator/snapshot/background407 and credential/loader/persistence/surface82 pass. Both authorized independent reviewers bounded source CLEAR. Native RED preserved as /private/tmp/chat-manual-auth-13575-native-red.json and .png: retained public draft,0 chat requests; original scope authSource omitted and current saved config manual. Native GREEN requires a fresh owned profile with public Test/Save then Send and no intervening reload; pending rebuild. Bandit N/A: only TS tests and shared TS comparator touched; no Python repository source.
Final original-account GREEN: fresh one-headless-profile public Open settings/Test/Save/Return/model pick/first Send succeeds on the same page without reload. Canonical2 rows, clear composer, save_to_db true, no browser message POST; actual second Send has exact original H1 admission/final IDs and canonical4 with complete metadata. Evidence /private/tmp/chat-native-server-settlement-13573-native-two-proof.json and canonical-two.json. First stream CDP body cannot be read and is explicitly excluded; second stream uses passive response.clone only, leaving original IO/consumer unchanged. Exact-message notification wait was corrected after actual success text included You can continue. Final18 connected1032 plus auth407/82 pass, both types exit0,22 authored TS scopes0 introduced findings, final webpack/token/budget593.1/795.4 pass. Normal exact4-file hooks and diff check pass. Both bounded source reviewers CLEAR; no new durable rule or Python scope.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Repaired the shared request-target comparison so the existing omitted-manual authSource default agrees with draft/connection ownership. Initial connection Save no longer falsely retires a first Send; cookie-session and other target/credential boundaries remain strict. Causal4 RED,489 connected account-scope GREEN and1032 chat regressions, both types/static/build/hooks, public first-time no-reload Send and second-turn canonical receipt qualify the bounded repair. Native request/body recording limitations are explicit; packaged extension, physical AT and participant studies remain separate.
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
