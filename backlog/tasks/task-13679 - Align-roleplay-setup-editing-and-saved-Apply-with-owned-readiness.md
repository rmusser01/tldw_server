---
id: TASK-13679
title: Align roleplay setup editing and saved Apply with owned readiness
status: Done
labels:
- chat
- ux
- review
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Final PR #3229 browser qualification exposed a source-confirmed gap: Scene fields appear editable without a current original-read acknowledgement, so guarded staging can silently discard input. Also qualify saved, non-identity Scene/settings Apply against the original protected conversation instead of fresh-chat model readiness. Preserve all ownership, account, revision, and publication fences.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Scene controls remain disabled until the original owned read is acknowledged and become disabled after retirement.
- [x] #2 Saved non-identity Scene/settings Apply preserves the protected owner and does not require an unrelated fresh model selection; fresh or changed identity still requires readiness.
- [x] #3 Both client profiles pass causal regressions and real-server browser qualification, with independent review and no original data mutation.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR #3229; source inspection: RolePlaySetupDrawer stage/isDraftCurrent versus fieldset readiness, PlaygroundForm applyAssistantSelection nativeCharacter preflight. ADR assessment: reuse existing actor-scene ownership and protected history settings contracts (ADR-049/050); no new durable architecture rule. Plan: 1 causal regressions; 2 shared readiness fixes; 3 current-build browser and review qualification. Runtime failure attribution remains bounded until causal evidence confirms it.
Final uninstrumented 21-case production run on OE782 passed20, failed PNG Scene staging: field editable but actual fill reset to blank before Apply. Investigate callback original-read ownership and event-value handling; do not weaken fences or relax the input oracle. Existing fieldset fix alone is insufficient for this observed case.
Reassessment after three bounded diagnostics: original uninstrumented PNG fill/reopen failure remains unresolved; first boolean probe observed actual onChange original guard true but notes missing on reopen, second same-asset trace passed yet cached marker coverage was insufficient, third routed trace did not reach drawer and is excluded. No speculative production Scene fix or relaxed input oracle added. Alternatives reviewed: parent Apply/close ordering (parent ends isCurrent, does not close), actor-settings original record read/CAS (saved record precedence retained), and passive read-effect resets. Next use read-only connected source review and actual DOM lifecycle receipt instead of mutable static chunk instrumentation.
Root now attributed by actual source-built diagnostic: valid notes input/stage→historyId-only mirror hydration→default reread→blank Apply/save. Native Scene key remains original account/scope/CID. Implement Scene-specific canonical cache alias handling with exact owner/CID/scope mirror receipt after successful existing transaction; shared history comparisons and sender contracts remain unchanged. Cover staged/held read/save, failed/aborted transaction and foreign ownership/scope/alias controls. Both independent reviewers validated this architecture boundary; no new durable ADR needed.
2026-10-14 root-cause extension: both ActorPopout and CurrentChatModelSettings reproduced staged Scene loss when an acknowledged native mirror alias hydrates (2 causal failures before shared fix). Reused useActorSceneHistoryId in the existing actor-scene-source module for all three editors, preserving exact account/CID/workspace alias acknowledgement plus original target, source, H1, lease, address/revision and CAS guards. Two independent reviewers found no additional ownership/security widening; requested removal of one misplaced comparator property and explicit callback dependencies. Complete affected suites in both profiles are running; current production/browser qualification remains pending.
Final source regression accounting: initial complete5suites450 cases per profile passed. Translator callback dependency churn in a follow-up run was corrected with latest-copy ref and those terminated runs are excluded. A later combined build/browser/two-profile run passed441 cases but timed out9 shared settings conflict cases (one late close after neighboring timeouts). Unchanged isolated conflict/default commitment block24 cases passed in23.15s. Full112 shared settings cases are now run separately per profile with original time limits and assertions; no timeout increase, test suppression or disabled assertion. Browser21 existing journeys including both Scene imports pass; new badge qualification separately identified header background precedence13682.
Fixed native Scene loss at the shared root: acknowledged local mirror aliases no longer replace the same qualified native Scene identity. RolePlaySetupDrawer, ActorPopout and CurrentChatModelSettings share the existing actor-scene-source helper; exact account/CID/workspace provenance is published only after successful mirror transaction and all original address/revision/lease/H1/owner/view/CAS guards remain. Saved non-identity Apply retains its native owner; read-only retired controls and fresh model/catalog readiness remain guarded. Locale copy changes do not rehydrate staged Scene edits. Final reviewed production build DUqqxSjslGXmYUQ7qf1PA: all22 real-server cockpit cases pass with no retries/backend route interception, including JSON/PNG imported card greeting, two protected turns, exact Scene staging/Apply/cold reopen, empty-history Refresh, responsive12-cell matrix and actual Saved/Temporary badge contrast in both themes. Connected Scene/Drawer/Form/Mirror source verification covers450 distinct cases in each client profile:338 cases pass in four complete suites, and the full112 shared editor cases pass separately with fixture cleanup and unchanged behavioral assertions/individual waits and scoped real-modal15s integration budgets. Native and WebUI production connected dependency roots pass. Changed35 TypeScript files:0 errors/0 new findings (final lint report records baseline warning counts). Source-matched58 production deltas;969 emitted JSON manifests,1131 static references,0 missing; shared token sync and590.6/600KB shared /869.4/900KB max budgets pass. Two independent bounded source reviewers report no remaining actionable issue; review is selective, not every line of the whole PR. Original9 conversations/28 rows and compact4-row source remain deep-equal. Fresh whole-PR67 production Python Bandit files:14 identical merge-base findings,0 introduced,0 scan errors. Physical installed-extension docking/promotion, native-toolbar zoom, touch/assistive technology and recruited participant studies remain unrun; publicH2, activeSync and tracked persona remain unavailable. Remote-dev integration and exact-head CI remain separate required gates.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed native Scene loss at the shared root: acknowledged local mirror aliases no longer replace the same qualified native Scene identity. RolePlaySetupDrawer, ActorPopout and CurrentChatModelSettings share the existing actor-scene-source helper; exact account/CID/workspace provenance is published only after successful mirror transaction and all original address/revision/lease/H1/owner/view/CAS guards remain. Saved non-identity Apply retains its native owner; read-only retired controls and fresh model/catalog readiness remain guarded. Locale copy changes do not rehydrate staged Scene edits.
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
