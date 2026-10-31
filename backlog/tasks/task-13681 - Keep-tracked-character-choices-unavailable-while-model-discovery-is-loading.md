---
id: TASK-13681
title: Keep tracked character choices unavailable while model discovery is loading
status: Done
labels:
- chat
- ux
- accessibility
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Real PR #3229 runtime-rail selection reaches the actual tracked Form callback with all source/account/model revision guards current, but fresh model usability reports loading with no catalog and refuses native_history_model_unavailable. The character list falsely looks selectable before this prerequisite settles. Reuse parent catalog state in the shared picker, visibly explain waiting and disable only tracked selection during discovery; preserve overlays, editor staging and ownership guards.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Tracked character choices show an accessible waiting status and remain disabled until the required catalog settles, then become selectable without changing the requested mode.
- [x] #2 Full and compact native callers pass existing discovery state; ordinary overlays and staged editor identities remain usable.
- [x] #3 Causal regressions, both profiles/types/source review and the uninstrumented full real-server cockpit pass; temporary diagnostics are removed.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root proof: /private/tmp/chat-pr3229-runtime-picker-13679-model-qualified-probe-results.json reports pending=false, tracked callback=true, current lifetime=true, every entry guard=true, model status=loading, modelsLoading=true, catalog size=0, guarded model-unavailable refusal. Earlier guard-retirement attribution is withdrawn. Prior 20/21 current cockpit result is not a success claim. ADR assessment: existing shared readiness/ADR-049 ownership; no new API, await/replay or state framework. Plan: regression, shared availability/caller propagation, both-profile/current-build/browser/review qualification.
Tracked character choices wait for current model discovery through shared AssistantSelect in full and compact callers. Character editor and ordinary overlays remain available; unsupported tracked persona offers one clear unavailable explanation and explicit ordinary-chat overlay guidance. Final reviewed production build DUqqxSjslGXmYUQ7qf1PA: all22 real-server cockpit cases pass with no retries/backend route interception, including JSON/PNG imported card greeting, two protected turns, exact Scene staging/Apply/cold reopen, empty-history Refresh, responsive12-cell matrix and actual Saved/Temporary badge contrast in both themes. Connected Scene/Drawer/Form/Mirror source verification covers450 distinct cases in each client profile:338 cases pass in four complete suites, and the full112 shared editor cases pass separately with fixture cleanup and unchanged behavioral assertions/individual waits and scoped real-modal15s integration budgets. Native and WebUI production connected dependency roots pass. Changed35 TypeScript files:0 errors/0 new findings (final lint report records baseline warning counts). Source-matched58 production deltas;969 emitted JSON manifests,1131 static references,0 missing; shared token sync and590.6/600KB shared /869.4/900KB max budgets pass. Two independent bounded source reviewers report no remaining actionable issue; review is selective, not every line of the whole PR. Original9 conversations/28 rows and compact4-row source remain deep-equal. Fresh whole-PR67 production Python Bandit files:14 identical merge-base findings,0 introduced,0 scan errors. Physical installed-extension docking/promotion, native-toolbar zoom, touch/assistive technology and recruited participant studies remain unrun; publicH2, activeSync and tracked persona remain unavailable. Remote-dev integration and exact-head CI remain separate required gates.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Tracked character choices wait for current model discovery through shared AssistantSelect in full and compact callers. Character editor and ordinary overlays remain available; unsupported tracked persona offers one clear unavailable explanation and explicit ordinary-chat overlay guidance.
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
