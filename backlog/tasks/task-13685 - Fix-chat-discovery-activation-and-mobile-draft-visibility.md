---
id: TASK-13685
title: Fix chat discovery activation and mobile draft visibility
status: Done
labels:
- chat
- ux
- performance
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Final PR3229 CI exposes duplicate current-generation model discovery requests and runtime rail overlap with the mobile composer on the fuller server configuration. Diagnose and fix shared roots without weakening ownership readiness, layout acceptance or request budgets.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current-generation fresh and ordinary model discovery share eligible network work without stale readiness or cross-account reuse
- [x] #2 Long mobile runtime content and constrained composer layout do not overlap and remain keyboard reachable
- [x] #3 Unit and browser acceptance retains current request budget and no-overlap assertions
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: existing ADR049/050 account/settings ownership; no new durable rule expected. CI RED: UX request budget28 vs25±1 with duplicated metadata/providers; mobile390x844 runtime panelBottom414.1875 vs composerTop378.515625 on all3attempts. Stage1 trace dependencies and reproduce; stage2 test shared discovery invariants and mobile geometry; stage3 minimal root fixes; stage4 independent review and current-source acceptance.
Measured baseline: native110 discovery checks pass after raw-promise sharing, but real request capture shows sequential metadata/providers lanes after completion. Added RED recent-live-read test:1failed1passed; authoritative reuse must require actual in-process successful current-generation provenance, nonempty result, existing30-second freshness interval, noncookie scope; hydrated cache, empty results, account invalidation, forced refresh and strict errors stay excluded. Mobile diagnosis ruled out hidden-main scrolling: all hidden ancestor scrollTop0; actual composer scrollporttop591.5height252.5scrollTop221content733 leaves textarea top414.5 clipped177px. Casual mobile cockpit duplicates full toolbar, while pro already uses compact toolbar. Added explicit draft-within-scrollport browser assertion; short600px case alone passed and is not credited as causal RED.
Independent review rejected completed-live reuse because explicit Refresh and post-activation reads must remain fresh. Removed that entire candidate and its tests; its RED/build logs are diagnostic only. Smaller root: page catalog waits for exact isConnected+CONNECTED as Form, and automatic character-route intent remains pending until that gate and catalog settle. New actual page test RED1failure proves discovery before connection. Mobile correction preserves Customize/model/roleplay/Buddy controls: qualified rail focus scrolls draft without changing rail focus. Reviewer found helper-only scheduling gap; existing focusin listener now admits that same qualified rail focus. Browser case waits earlier timers out and asserts draft within its own scrollport plus preserved Runtime-button focus. Native connected-page/Form/catalog qualification is running.
Final reviewed production scope is three existing files: exact connected page/catalog+route gate; current raw network discovery coalescing with strict per-caller errors; existing mobile focus listener/scroll helper for rail focus. Native467 tests across9 connected page/Form/catalog suites pass, plus earlier654 fixture tests across11 suites. Both independent source reviewers report no remaining actionable issue; runtime/client-profile and exact-head remote checks remain pending. Native Form214 suite took186seconds; no deadline or assertion changes.
Final production build XhDB6DEu1RRotqawBxBrD passes source hashes/tokens/budgets; all23 real-server cockpit cases pass without retries/backend interception, including explicit clipped-draft regression and preserved Runtime focus after earlier timers retire. Final shared17 TypeScript files:0errors181baseline warnings0new/0ignored. Production request measurement improves24→22 requests and removes both metadata/providers duplicates (one each). Whole ratchet fails only because its development/StrictMode baseline includes additional Persona/Buddy duplicate requests absent in production; baseline is unchanged. Qualify the actual development environment before treating the whole ratchet as passing.
Actual development/StrictMode request ratchet passes unchanged:1real case26.6seconds,26requests within25±1, exactly1metadata+1providers request and no new duplicate entries. Production22requests has lower known StrictMode duplicate counts, so its whole-ratchet diagnostic was not credited as passing. First development startup failed temporary directory naming validation; its skipped pre-readiness run is not credited. Correct owned .next-live-tier-* development preview qualifies the actual metric.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed shared raw discovery coalescing and premature page discovery/character intent by the exact confirmed connection gate. Qualified mobile rail focus scrolls the clipped draft without stealing focus or hiding customization controls. Native467connected cases plus654fixtures pass; WebUI/extension1121cases each pass. Both connected type roots pass. Production build XhDB6DEu1RRotqawBxBrD has65 verified source deltas,969 JSON manifests,1130 physical static references and0missing; tokens and590.2/600KB shared /868.9/900KB max budgets pass. All23real-server cockpit cases pass; calibrated unchanged development request ratchet passes26requests with one metadata/providers read each. Lint17sharedTSfiles0errors181unchanged warnings0new/ignored. Two independent source reviews cleared findings. Original9conversations28rows and compact4rows remain deep-equal. Completed-cache reuse was rejected and removed; failed diagnostics were not credited. Parent TASK13458 tracks final integration and merge; physical extension/AT/touch/native zoom and participant testing remain unrun.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
