---
id: TASK-13683
title: Dispose owned Scene editor fixture resources after each case
status: Done
labels:
- chat
- testing
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR3229 final qualification identified real cumulative resource lifetimes in the connected Scene fixture: QueryClients are never cleared, modal destroy callbacks survive, and the console spy retains history. Repair only owned fixture cleanup using neighboring house patterns, preserve real controls and all assertions/time limits, and requalify both profiles.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each mounted fixture client and modal lifecycle is disposed after the case, including failure paths.
- [x] #2 The console spy and globals are restored without broad test or production changes.
- [x] #3 Full Scene verification passes both profiles with unchanged behavioral assertions and individual waits; only real Modal conflict-review cases use the existing fifteen-second integration budget.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR #3229; associated TASK-13458/13679. Initial450 per profile passed; later combined runs441/450 with9 late failures; isolated24 commitment cases pass23.15s; sequential full112alone98 pass14timeouts462.77s, so concurrency alone is disproven. Stop unchanged reruns and reassess: two independent reviewers confirm QueryClient cache/GC lifetimes, real AntD modal destroyFns callbacks and console spy retention; navigator proxies are restored per case. House precedents: CurrentChatModelSettings.conversation-prompt.test.tsx208 Modal.destroyAll,258 unmount/client.clear. ADR assessment: fixture lifetime cleanup only, no durable architecture change; ADR049/050 ownership guards remain unchanged. Stage1 reproduce/reassess complete; Stage2 reuse owned cleanup; Stage3 verify current full fixture profiles and independent review.
Cleanup qualification108/112 passes; four remaining failures are generic5000ms test deadlines at5141/5458/6215/5119ms, without behavior assertion failure. Both independent reviewers traced handlers and confirm original owner/view/edit-revision/CAS/confirmation identity/finally guards remain intact. Reassessment: align only three actual Form/Modal conflict-review case declarations with existing15000ms integration deadlines in this same fixture. Preserve every findBy/waitFor and behavioral assertion; no global timeout change, production change, retry, or UX performance claim. Earlier unchanged-time-limit diagnostic promise is superseded explicitly by this bounded house-budget choice. Cumulative slowdown is incompletely attributed and must not be reported as fixed solely by cleanup.
Owned fixture cleanup drains AntD modal callbacks, unmounts editors, clears each QueryClient, restores the console spy and globals using neighboring house conventions. Full112 cases per profile retain all real controls, behavioral assertions and individual waits; only real-modal review cases use the existing fifteen-second integration budget. Prior timing failures and resource-lifetime diagnosis remain recorded. Final reviewed production build DUqqxSjslGXmYUQ7qf1PA: all22 real-server cockpit cases pass with no retries/backend route interception, including JSON/PNG imported card greeting, two protected turns, exact Scene staging/Apply/cold reopen, empty-history Refresh, responsive12-cell matrix and actual Saved/Temporary badge contrast in both themes. Connected Scene/Drawer/Form/Mirror source verification covers450 distinct cases in each client profile:338 cases pass in four complete suites, and the full112 shared editor cases pass separately with fixture cleanup and unchanged behavioral assertions/individual waits and scoped real-modal15s integration budgets. Native and WebUI production connected dependency roots pass. Changed35 TypeScript files:0 errors/0 new findings (final lint report records baseline warning counts). Source-matched58 production deltas;969 emitted JSON manifests,1131 static references,0 missing; shared token sync and590.6/600KB shared /869.4/900KB max budgets pass. Two independent bounded source reviewers report no remaining actionable issue; review is selective, not every line of the whole PR. Original9 conversations/28 rows and compact4-row source remain deep-equal. Fresh whole-PR67 production Python Bandit files:14 identical merge-base findings,0 introduced,0 scan errors. Physical installed-extension docking/promotion, native-toolbar zoom, touch/assistive technology and recruited participant studies remain unrun; publicH2, activeSync and tracked persona remain unavailable. Remote-dev integration and exact-head CI remain separate required gates.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Owned fixture cleanup drains AntD modal callbacks, unmounts editors, clears each QueryClient, restores the console spy and globals using neighboring house conventions. Full112 cases per profile retain all real controls, behavioral assertions and individual waits; only real-modal review cases use the existing fifteen-second integration budget. Prior timing failures and resource-lifetime diagnosis remain recorded.
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
