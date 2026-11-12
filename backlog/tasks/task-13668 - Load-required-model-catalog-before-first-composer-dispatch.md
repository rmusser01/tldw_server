---
id: TASK-13668
title: Load required model catalog before first composer dispatch
status: Done
labels:
- chat
- ux
- model-readiness
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A cold full composer with a restored model gates its own admission catalog observer on optional picker visibility/engagement. Parent Playground.tsx has an unconditional observer, so there is not a whole-page absence of producers. Add explicit connected activation for the Form, distinguish active loading before ordinary/queued preparation, preserve source/model/owner fences and qualify the actual connection transition/fresh result with picker unopened.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Connected full composer fetches its required model catalog with picker closed and never engaged; disconnected composer does not fetch
- [x] #2 Ordinary submission and queued validation distinguish active catalog loading from an unavailable model without dispatch or source writes; supported ready and frozen roleplay paths remain available
- [x] #3 Native/WebUI regressions, source checks and actual cold mobile/streaming cockpit workflows qualify the root fix with canonical conversation preservation
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root trace:FullForm896 optional model-catalog selector -> shouldEnableOptionalResource uses visible&&engaged; FullForm1439 disables required query until picker interaction. Compare compact form connected-query and transcript default-enabled query; an empty full transcript has no query producer. Existing source guard requires a catalog and correctly refuses, but missing producer is the bug. TASK13656 metadata-only fixes/pure controls pass; strict character opening/reply/cold qualification now passes. No protocol/model admission bypass. ADR assessment:no new durable architecture rule; restore existing required model/admission dependency, ADR049 ownership preserved. PR https://github.com/rmusser01/tldw_server/pull/3229.
Independent source review clear but corrected causal wording: Playground.tsx427 already has unconditional shared catalog observer. Form RED proves only its enabled option; disconnected control is local, not whole-app network absence. Added actual QueryClient shared-parent early error -> connected admission retry -> held fresh result control. Do not claim observed browser error sequence without evidence. Native/WebUI7 connected each304/0 skips pass before added observer control; browser/build pending. TASK13656 tools text locator was ambiguous; now exact visible popup Clear conversation action plus Escape hide/opener focus, with no inferred Button click-handler bug.
Native/WebUI connected7 suites each304 pass/0 skips; added actual QueryClient observer suite5 pass each (overlaps4 prior controls) proves early shared failure -> connection-enabled admission retry -> withheld held fresh result -> verified ready. Final2-root production dependency type profiles and changed cockpit/imported helpers type profile exit0. Actual root lint5 TS files:0 errors/104 unchanged HEAD warnings/0 ignored/0 new. Browser spec now captures real metadata result with selected provider/model and asserts model picker unopened before mobile/streaming dispatch. Specific historical browser error sequence remains unclaimed. Build/new-dist gates/native browser/preservation pending.
New source-matched buildW3UtHkn0Q870UEMAie6sw passes154 pages,42 delta hashes/emitted969 JSON1130 refs0missing/new-dist tokens/budgets590.3/869.0. Actual remaining7 run still4 pass/3 fail: strict character plus3 catalog controls pass, tools exact popup absent, mobile/stream ordinary model unavailable despite real browser200 metadata result. Stop repeating prior attempts; connected observer fix is not claimed as causal closure. Reassess with bounded actual cached model data/visible picker selection diagnostics, preserving all original oracles and avoiding config/transcript/API-key dumps. OpenAI endpoint host verified127.0.0.1:19768 localmock.
Controlled real QueryClient early-empty-success activation RED exposed[] instead of withholding; shared hook now verifies false->true activation without changing30s stale policy or refetching incidental initially enabled mounts. Stable refetch dependency and query-owned error handling explicitfalse; generation/effect/activation identity reject old settlement. Final8 hook controls pass incl failure-retained-data and old-account completion. Lint6 TS0errors/104 unchangedwarnings/0ignored/0new. Temporary safe diagnostic retired privately: no cache beforefirstSend, actual picker publishes219 configuredmodels includinggpt, observed GET metadata/providers200; no API-key/config/transcript dumps. This is causal evidence plus controlled mechanism, not yet browser closure. Broader profiles and new build follow.
Final connected qualification: native and WebUI each 7 suites / 311 tests passed, zero skipped. Three production roots type-check in both surfaces; actual root lint six TS files zero errors / 104 unchanged warnings / zero new findings or ignored files. Public QueryClient controls prove initial-enabled fresh-empty verification, waiting a prior same-generation in-flight read before a fresh activation read, and zero extra IO for an initial warm nonempty catalog. Correct earlier root-trace note: Playground.tsx already has an unconditional producer; empty transcript is not evidence of a producer-free application. Browser closure remains pending the latest v2 production build and strict first-send workflows.
Final bounded independent source review: no actionable correctness finding; both prior activation findings covered by red/green real QueryClient regressions, old settlements fenced, warm nonempty mounts preserve no-extra-IO behavior. Production v2 build HgLBTVIPscnaI6-4Qt2EC passed 154-page build, actual-dist token sync and bundle limits (shared 590.3KB/600KB, heaviest 869.0KB/900KB), 43 source deltas with matching hashes, 969 emitted JSON manifests / 1129 physical JS-CSS references / zero missing. Original nine conversations/28 rows and existing synthetic four-row fixture remain deeply equal before browser checks. Seven real-server cases currently running on v2; no browser closure claimed yet.
V2 actual browser remains 4 passed/3 failed (Tools, mobile premature Send, stream premature Send); no native first-send closure. Reassessed after repeated failures: filtered real QueryClient diagnostic showed pending/fetching then success/idle with70 models and no activation hold; clean installed AntD Tools opens. Root UI problem: ordinary Send appeared enabled while catalog admission still correctly rejected loading. Added actual Form red/green tests for disabled loading and ready settlement, with positive image-command and saved-native-character exceptions; seven connected suites now314 tests passed each native/WebUI, zero skipped, both three-root types0 and lint0new. Existing disabled-reason and loading-label UI reused; no timers or weakening of admission. Exact prior Tools picker sequence is investigated separately.
Final combined pointer/catalog verification:14 suites405 each native/WebUI,4root types both0, lint13TS0new, six actual browser cases pass including unopened mobile firstSend and strict greeting/reply/cold reopen. Model admission and pointer roots are verified; AC3/full streaming workflow remains open under13656 after stop-before-admission and cold-empty-reopen evidence. Preserve source guards and do not assume optimistic input acceptance. Bounded review Docs/Reviews/CHAT_COMPOSER_REQUIRED_CATALOG_POINTER_13668_13669_2026_10_14.md; original9/28+4rowfixture unchanged; no fullPR approval.
Normal22-file hooks passed. Source-only bounded commit planned with TASK13669 and all connected callers;13656 cockpit E2E file/task remain separate while early-stop workflow assertion is unresolved. AC3 intentionally open. BanditN/A TS/Markdown-only; wholePR backend validation remains required.
Full17 cockpit allpass39.2s,0skips/retries/flakes on source-matchedgMU build, including unopened-picker coldmobileSend and actual streamingStop/reload/explicitH1boundary/nextSend withoriginaljournal deepEqual and raw0automaticcompletion/admissionattempts. This completes AC3 boundedcatalogqualification. Required model observer/loading/activation controls remain14suites405each nativeWebUI/source4roots/types/lint qualified in6aa69d48ee. Normal final hooks/taskclosure follow. Newlyfound bound-private live-global readiness is separate; no wholePR/merge claim.
Normal15-file finalhooks passed; own completed required-catalog plan retired. All3AC/6DOD complete. Distinct bound-popup liveglobal readiness is new task; completed catalog unit not reopened.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Required connected composer catalog and explicit verification/loading control complete. Warm verified mounts avoid extra IO; earlyempty/priorinflight/account retirements covered. Both runtime14suites405 pass, source types/lint and currentnative mobile/streaming/full17cockpit qualify. Original9/28+4fixture preserved. Source implementation6aa69d48ee; final taskclosure accompanies qualifiedcockpit recovery. BanditN/A TS/Markdown scope; wholePR gates remain separate.
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
