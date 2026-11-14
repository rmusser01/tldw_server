---
id: TASK-13652
title: Qualify startup model Retry faults at native device boundary
status: Done
labels:
- chat
- ci
- tests
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair two reproduced startup Retry assertions whose DOM storage fault misses native sync storage. Preserve exact failed operation, identity ACK and frozen library behavior. PR#3229.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Actual native or WebUI sync write failure is observed without replaying identity acknowledgement
- [x] #2 Exact operation and frozen proposal assertions retained; physical model value verified
- [x] #3 Both configurations and independent review pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
RED:2startup device cases fail in current native suite. Old localStorage.setItem fault doesnot intercept capturednativeSyncAPI. ADR assessment:no newADR,test boundary only. Plan:Docs/Plans/IMPLEMENTATION_PLAN_chat_startup_device_fixture_13652.md
Finalnative/WebUIeach19tests1suite0skipprocess0;realdefaultStorage readback andderivedfault area. ScopedESLint0errors/warnings;whitespace0;BanditN/ATypeScript. Initial nonexistentconnectedsuite pathnotcredited;stalecommentremoved. Review:Docs/Reviews/CHAT_STARTUP_DEVICE_FIXTURE_13652_2026_10_13.md. No productionchanges.
Independentboundedreview correcteddefaultStorage/faultareamismatch andrecheckedclear. Nativeexactcommand frontendcwd vitest run --config vitest.extension.config.ts UItest;WebUI --config vitest.config.ts UItest. IdenticalRUNcwdnotaliasproof;configsexplicitinorchestratedcommands. Noothermodelsuite credited.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
TwooriginalnativefailuresqualifiedthroughactualdefaultSDKstorage;19startup/modelcases passineach explicitclientconfig,identity/frozenretry assertions retained,review/lint/hooks clear. PR#3229.
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
