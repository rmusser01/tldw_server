---
id: TASK-13690
title: Apply shared web timeout policy to the actual request URL
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Integrating current dev PR3238 exposed that ordinary api.get/post pass URL separately while createAbortSignal reads config.url, so shared path floors see an empty path. Pass the actual request path at the one shared call site and verify real fetch AbortSignal behavior with fake timers, preserving explicit overrides, generic30s defaults, cancellation and RetryAfter seconds.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Behavioral timer regressions reproduce missing floors on incoming dev
- [x] #2 All web request callers use the actual path for shared timeout policy without new transport abstraction
- [x] #3 Existing API behavior and connected source/type/build verification pass; final independent reviews and exact-head CI precede merge
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new ADR; existing shared transport StageA plan in incoming PR3238 governs intended parity and frozen web error semantics; ADR069 remains Proposed contract-package boundary, not promoted. Minimal fix is data flow into existing deriveRequestTimeout. Related parent TASK13458 and final CI plan13685. Reviewer A confirmed ordinary request config lacks URL; actual request URL is already resolved before creating signal.
Plan: Docs/Design/IMPLEMENTATION_PLAN_web_timeout_path_13690.md. Merge-preserving rebase complete on6cbf1e943: normal flattening caused historical chat conflicts because the branch contains30 integration merges; correct --rebase-merges preserves them. Whole-tree comparison vs93994cf shows only four incoming transport/plan/task paths changed, all previous chat/backend/fixture source unchanged. Incoming TASK13526 lacks leading YAML delimiter; restore known canonical backup metadata and use backlog-py to preserve all new upstream notes and ADR069 Proposed correction. Behavioral6-case RED now exercises actualGET/POST paths, metadata query/absoluteURL, generic30s and explicit5s override.
Behavioral RED verified on actual incoming6cb code:4 floor cases fail because signal already aborted,28 pass includinggeneric30s/positiveoverride/caller cancellation. Minimal3-line repair adds resolved requestUrl to private signal helper and its solecaller; resolved URL includes /api/v1 even when callers use /llm or /slides. Do not pass rawcallerpath because shared path predicates require actual APIprefix.
Minimal resolvedURL fix GREEN: actualwebfetch34cases including6 deadline+2RetryAfter formats, total7API/path/refresh timeout suites81pass. No frontendrequestwrapper or auth/error changes. Complete3configurationScene112 qualification unaffected byrebase; originalchat/backend/fixtures exactbyte-identicalto939 aftermerge-preserving rebase. Finalproductionbuild,types,readonlydelta reviews running. TS-onlyproductionchange: Bandit prior67productionfiles0new and fixture1line0new remain current; noPythonchange fromlatestdevtransport.
Final exactworking-source production build passes:97productionhashes match source+owned artifact,969emittedJSONmanifests/1134physicalstaticreferences,0missing. Build l86jhxirQAbtXh5lmtcBx. Tokens590.2/600KB and heaviest868.9/900KB. WebUI types pass; two changed TS files lint0errors0warnings0ignored0new. Both readonlyreviewers confirm minimalURLfinding addressed/no actionable delta issue. Guarded preview19769 nowPID41946/newbuild; all23realcockpit acceptance cases executing zero retries. Nativeproduction/backend and allprevious fixtures unchanged bywhole-tree rebase comparison.
All23finalproductioncockpit cases pass with0skips/flaky/errors/retries, includingactualv2 card import/greeting/two turns/Scene/reopen andmobilefocus/userdata recovery. Original9conversations28rows deepEqualtrue;4originalcompactrows unchanged;97productionhashes still match. Five-file normal hooks pass. Final exactcommitted-head independentreviews/GitHubstatuses/merge remain Stage3.
2026-10-14 continuation: remote dev advanced to4ffb0f01fd via TASK13526 StageB1 request-plan/credential extraction. Before edits, preserve6aec fixture-reviewed candidate, integrate four incoming paths with merge-preserving rebase, preserve resolved-URL web abort fix and canonical task metadata, then qualify connected request tests/types/new production build and renew both independent reviews. Published4b CI also reports completed db-privileges failure; inspect root cause before final publication.
Latest-dev merge-preserving integration exact0173d7d3 changes only shared request-core, incoming plan fixture and canonical TASK13526 versus6aec; existing shell-independent server-capability guard already covers upstream correction. IndependentA cleared integrated cancellation/ownership paths. Web315/native266/extension266 cases PASS. Production build FxEtgrYFhJR0EH0eSabUF passes969JSON1134static refs0missing/98source hashes, but integration lint identifies incoming B1 unused isAbsolute destructure and new any plan predicate type: remove unused binding, use unknown at plan boundary, then refresh source build/proof before final qualification. Existing six any warnings remain baseline; no credential/cancellation semantics change.
Final production source qualified: normal WebUI typecheck exit0; build LwZwWIX9oU5QB_ybRXenN,98matching sourcehashes,969JSON1134static references0missing; budgets590.2/600KB and868.9/900KB. Final23realcockpit cases PASS0skip/flaky/error/retry and original9conversations28rows deepEqual immutablebaseline. Final core lint0errors6baseline anywarnings0new. Both independent source reviews CLEAR; final published exact-head CI/merge remains pending.
Final integrated cfd8a74 production source qualified: all8direct request suites81pass afterlintcleanup; WebUI315/native266/extension266 connected before type-only cleanup; configured types/build98hashes and23real-server cockpit cases pass. Both independent exact-head source reviewsCLEAR; all7required GitHub checksPASS. New TASK13697 fixture/gate-only batch retains all98production hashes and protected9conversations28rows deepEqual. Its final published-head CI and broader workflow completion remain required before authorized merge.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Pass the actual resolved request URL into shared timeout policy, retaining configured overrides, cancellation and RetryAfter semantics; preserve current dev request-plan integration. Connected runtime/type/build/cockpit verification and independent review pass; all7required cfd checks pass. Final fixture-only head and broader workflows remain landing gates.
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
