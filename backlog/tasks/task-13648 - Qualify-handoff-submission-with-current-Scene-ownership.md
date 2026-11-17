---
id: TASK-13648
title: Qualify handoff submission with current Scene ownership
status: Done
labels:
- chat
- ci
- tests
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair six reproduced handoff submit fixture failures using real Scene source capture with scoped IO only. PR#3229.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Ordinary and queued positive submissions retain exact imported context assertions
- [x] #2 Failed Scene reads and retired accounts preserve composer and context without dispatch
- [x] #3 Original package failure and storage serialization controls retained
- [x] #4 Both client suites and independent bounded review pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
RED:6failed71passed in native suite. Readonly reviewer confirmed missing controller/account/actual fresh draft. ADR assessment:no newADR; current source ownership policy preserved. Plan:Docs/Plans/IMPLEMENTATION_PLAN_chat_handoff_fixture_13648.md
Final native/WebUI each102tests2suites0skipprocess0 inclactualScene-target controls. Independent boundedreview clear. ESLint0errors0new6unchangedwarnings;diffcheck0;BanditN/A TypeScript. Review:Docs/Reviews/CHAT_HANDOFF_SCENE_FIXTURE_13648_2026_10_13.md. No source enforcement weakened.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Six original handoff failures repaired through qualified fixture ownership;102tests pass in each client configuration;negative preparation/retirement controls and bounded review clear. PR#3229.
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
