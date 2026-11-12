---
id: TASK-13538
title: Fence model catalogue config reads before request reuse
status: Done
labels:
- chat
- webui
- extension
- bug
dependencies:
- TASK-13458
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A model request started before a setup/account invalidation can finish its held getConfig after the shared in-flight pointer has been cleared, install a retired promise, and recursively await itself. Recheck the existing catalogue generation after config resolution before reading cached models, joining current discovery or starting discovery. Keep options, auth boundaries and current request reuse intact.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Held config invalidation regressions expose and repair the self-await for strict and advisory reads.
- [x] #2 Retired config results cannot use stale cached or cache-only models and request options are preserved.
- [x] #3 Same-generation concurrent and post-update successor requests retain correct sharing and failure handling.
- [x] #4 Affected regressions, both client type checks, covered static checks and independent review pass.
- [x] #5 Native current-build model readiness and disposable Temporary completed-send/reload/Clear are verified or their concrete limits are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Read-only duplicate searches found no held-config catalogue task. Highest local/dev/open-PR IDs13537/13530/13526, so13538 is unique. Fresh origin/dev2c5f19d032 remains included. Source root independently confirmed in fetchChatModels340-443: invalidation during awaited config allows the old caller to install a retired promise; recursive dedupe can then await itself. Existing test invalidates after lower discovery and starts a successor, missing this seam. TASK-13537 currently has444passing checks, both types, zero added effective lint and two clear source reviews; native completed-send verification is waiting on catalogue readiness. ADR assessment: no new decision; retain existing catalogue generation/cache/auth contracts and ADR-049/050 ownership/lifetime, ADR-059 tracking. Plan: Docs/Design/IMPLEMENTATION_PLAN_chat_model_config_fence_13538.md. Entire all-ten program remains In Progress.
Causal red19checks: four real failures /15passing controls in /private/tmp/chat-model-config-fence-13538-red.log. Strict/advisory held-config invalidation leaves the promise unsettled; forced refresh redundantly discovers twice; current successor joins without rereading retired config. Shared root now checks existing generation immediately after config await and retries unchanged options before any cache/join/new discovery. No lower transport, account guard, cache lifetime or dependency changed.
Final closeout: all four owned plan stages are Complete; only this completed plan is retired. The shared root adds a generation check immediately after awaited config and before cache access, in-flight reuse or discovery, retrying all five unchanged options. It neither moves generation capture after the await nor changes cache/auth/error policy. Same-generation and held current-successor controls preserve one legitimate discovery; the successor stays held until the old caller rereads config, avoiding the earlier fixture-timing failure. No new timer, request registry, helper, dependency or protocol.
Final post-format combined matrix:543passed in17suites, no skips (/private/tmp/chat-temporary-and-model-13537-13538-final-matrix.log). Relevant service/catalogue/selector/API normalization cases are included. Both client types pass. Actual unchanged-rule lint covers19authored TS/TSX files:34baseline/34current/zeroadded;14Temporary owned formatting regions and one catalogue test region pass, as does diff checking. Both existing independent reviewers clear the final source and causal tests. Bandit N/A: TS/TSX/docs only, no Python repository file changes.
Current native dropdown reaches Healthy and the real disposable local provider completes a Temporary turn without manual cache refresh. Actual reload retains the exact two message rows and next unsent draft including two trailing spaces; New saved chat uses the existing Clear boundary and subsequent reload has zero articles and empty composer. Native receipts/photos are recorded by Done TASK-13537. Authenticated actual server content search has zero Temporary-question matches and ordinary4/Mira5 original saved rows retain exact SHA256 values (/private/tmp/chat-native-temporary-server-history-13537-20261007.json). Current origin/dev2c5f19d032 remains included. One existing IAB tab only; no browser interception/storage seeding or new Chrome window.
Limits: causal retirement/config/discovery controls declare lower I/O doubles; the native current-build success establishes readiness and tested Temporary flow, not instrumentation proving the startup race occurred at the exact held-config seam. It does not establish packaged worker, browser crash/duplicate-tab, assistive technology, generated roleplay quality or full D7/editor ACK/lore/provenance. TASK-13458 and the all-ten program remain In Progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the shared model-catalogue root fix: a config read retired during invalidation retries before entering cached or in-flight discovery, preserving all caller options and current-request sharing. Four causal failures are repaired; final543checks, both types, zero added covered lint and two independent reviews pass. Native readiness and Temporary send/reload/New succeed; saved ordinary/Mira fixtures remain exact. Parent all-ten program stays In Progress.
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
