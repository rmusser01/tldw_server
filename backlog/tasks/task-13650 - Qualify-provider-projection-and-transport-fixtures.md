---
id: TASK-13650
title: Qualify provider projection and transport fixtures
status: Done
labels:
- chat
- ci
- tests
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair five reproduced frontend contract failures without changing production projection or transport admission. PR#3229.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Actual projection declarations and dependency helpers preserve native and local history behavior
- [x] #2 Pre-aborted requests reject with zero transport IO; dispatched cancellation retained
- [x] #3 Model refresh uses declared asynchronous runtime IO and preserves failure/success assertions
- [x] #4 Both client configurations and independent review pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
RED:5failed26passed across3nativefixtures. ADR assessment:no newADR;test-onlyqualifications preservingcurrentinference/cancellation/discoverycontracts. Plan:Docs/Plans/IMPLEMENTATION_PLAN_chat_provider_fixtures_13650.md
Finalnative/WebUIeach35pass3suites0skipprocess0. Independentboundedreviewclear. ExistingconfigESLint0findings/errors;whitespace0;BanditN/A TypeScript. Review:Docs/Reviews/CHAT_PROVIDER_FIXTURES_13650_2026_10_13.md. No production changes.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fivefixture failures qualified to currentproductioncontracts;35tests pass ineachclient;actualprojection,zeroIOpreabort and runtimefallback coverage retained;review/static/hooks clear. PR#3229.
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
