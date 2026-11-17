---
id: TASK-13571
title: Load shared splash content and extension visuals only on request
status: Done
labels:
- chat
- ux
- frontend
- performance
dependencies:
- TASK-13458
- TASK-13562
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The reviewed chat build exceeds the unchanged shared-shell budget at602.4KB. Splash hook imports cards/messages eagerly, and common UI settings pulls the full card definitions just to get default names. Extension AppShell still imports the visual barrel eagerly. Split the existing lightweight preference names from optional card content and load content/visuals only on an explicit splash request, preserving preferences, forced events, duration, dismissal and app usability during held or rejected loading.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Hidden or disabled splash does not load optional card/message/visual modules; default enabled names remain identical to canonical cards.
- [x] #2 Explicit and forced requests retain existing settings and dismissal; dismissed, superseded, disabled or unmounted pending requests cannot display late.
- [x] #3 Held or failed optional loading preserves usable app children and records a diagnostic; both WebUI and extension paths use the existing loader conventions.
- [x] #4 Causal and existing splash tests, both client types, static checks, two independent reviews and current build/token/unchanged budget gates pass, with runtime/package limits recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Allocated13571 after search: existing13562 Done covers WebUI visual loading only, extension/shared data remain explicit follow-ups. Highest local13570; fresh dev and open-PR paths max13534. ADR assessment:no new ADR; existing optional UI loading patterns and ADR059 govern, no authority/storage/protocol/dependency decision. Plan Docs/Design/IMPLEMENTATION_PLAN_chat_shared_splash_loading_13571.md. Initial current build failure /private/tmp/chat-roleplay-resume-13160-reviewed-budget.json:616898 bytes vs614400 shared; /chat824007 vs921600. No limits increased.
Final actual splash consumer gate10 suites50/50 passes: real hook8, held module loading/semantic retirement10, extension shell5, actual lazy extension held/rejected visuals2, canonical names3/fidelity4, actual overlay timer7, auth3/2 and WebUI Next provider6. Causal initial hidden-loading2 failures and real fade-lifetime2 failures precede fixes. Later same-name/same-duration replacement passes. Evidence /private/tmp/chat-shared-splash-13571-{red.log,fade-red.log,lifecycle-green.log,final-consumers.log}. Both independent source reviewers bounded CLEAR after canonical names, preference ABA/equivalent hydration and original fade ownership repairs. Ten scoped files formatted with runtime-AST equivalence at /private/tmp/chat-shared-splash-13571-format.json.

Initial broad regex included three nested credit names; exact fidelity test exposed it, then extraction of only TS AST top-level canonical card.name properties yields78 names in original order. Initial extension rejection assertion expected the direct failure but Vitest wraps it; actual diagnostic cause is asserted and children remain usable. The initially named nonexistent splash-settings suite was not selected, so only ten actual suites50 are credited. Initial WebUI compiler caught Node ambient timer overload; browser fade ref now uses native numeric timer ID. These diagnostics are explicitly not final green. Final types/static/current emitted budget/hooks/native/packaged pending. Bandit N/A:no authored Python in13571. No current browser/participant/all-ten result inferred.
2026-10-10 final TASK13571 qualification: all10 actual consumer suites50/50 pass; both final WebUI/extension compilers exit0;61 authored TS scopes introduce0 lint findings; executable-AST-equivalent owned formatting and explicit pre-commit hooks pass. Both approved independent reviewers bounded CLEAR. Fresh emitted build aQCHfC0j_EKsNHcGpJ7WU passes tokens/all814 referenced JS/CSS assets and unchanged600/900KB budgets: shared610241bytes(595.9KB), /chat817350bytes(798.2KB). Canonical78 preference names exactly match cards. Card/message marker payloads are absent from shared startup JS and emitted as async chunks. Current existing headless tab shows0 optional content loads before the actual public forced-login event, then both content chunks load; actual Splash screen displays/dismisses and restored7-row transcript is unchanged. Evidence /private/tmp/chat-shared-splash-13571-{final-consumers.log,final-web-types.log,final-extension-types.log,final-static.jsonl,format.json,build.log,tokens.log,budget.json,assets.json,optional-assets.json,native-final.log}; applicable hooks /private/tmp/chat-roleplay-splash-final-hooks.log. No budget increase, dependency, auth/storage/protocol change or new ADR. ADR049/050/059 remain governing. Bandit N/A:no authored Python in this bounded unit.
Exclude initial regex nested-name diagnostic, wrapped rejection assertion, Node timer typing, nonexistent suite, preview launcher cwd/JSON banner/runtime-auth/TIME_WAIT diagnostics, collapsed-sidebar finder wait and premature saved-banner assertion. Corrected public runtime-config is available and actual saved setup/seventh article explicitly awaited before native qualification; no auth policy was weakened. Original full/compact draft buckets remain stored. Legacy auth namespace continuity is not qualified by a different preview runtime authority. Packaged extension, physical AT and participant evidence remain open; real extension hook/shell controls are automated, not installed-package proof. All4 bounded stages complete; only this owned plan retires to /private/tmp/IMPLEMENTATION_PLAN_chat_shared_splash_loading_13571_completed.md. Parent/all-ten remains In Progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Deferred shared splash cards/messages and extension visuals with original-request retirement and owned fade timers. Exact preferences and app usability retained;50 connected checks, both types/static/reviews/hooks and fresh unchanged-budget/native on-request loading pass. Packaged/physical/participant limits remain in parent13458.
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
