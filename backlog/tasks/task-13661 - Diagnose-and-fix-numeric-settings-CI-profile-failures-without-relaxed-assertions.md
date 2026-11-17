---
id: TASK-13661
title: Diagnose and fix numeric settings CI profile failures without relaxed assertions
status: Done
labels:
- chat
- testing
- ci
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Actual UI package CI setup reproduces two numeric settings timeouts while frontend native setup passes. Isolate the causal environment/test-lifetime boundary before editing production or fixtures; preserve actual AntD behavior, staged/recovered0.45 and existing deadlines.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Exact package CI profile numeric cases pass with original expectations and deadlines.
- [x] #2 Causal diagnostic identifies and repairs the underlying seam without replacing actual numeric controls or disabling coverage.
- [x] #3 Connected settings tests, static/security gates where applicable and independent review pass; source/runner limits recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR3229. RED actual package vitest.config, maxWorkers1, global15000 unchanged and six-edit local10000 unchanged:14passed2timeouts,103.592s. Previously frontend native16pass does not qualify package CI setup. Dependencies/physical module paths identical across both packages. Setup getComputedStyle/matchMedia and restoreMocks differ but none causally proved yet. ADR required:no for diagnostics; preserve current ownership/settings architecture. Use bounded profiling/A-B with no assertion/time-budget changes.
Exact profile dependency versions and physical paths match frontend:React18.3.1,AntD6.2.1,Vitest4.0.18,jsdom28.1.0. Frontend setup imports shared UI setup and preserves real React Query when provider exists, so getComputedStyle/matchMedia and Query mocks are not yet proved causes. Package config.restoreMocks differs; React plugin/aliases also differ. Private profiling retains base setup and measures style cost without product/test assertions edits. Initial private config cwd-resolution failure occurred before collection, corrected with absolute source path and excluded. Plan Docs/Plans/IMPLEMENTATION_PLAN_chat_numeric_ci_profile_13661.md.
Private style profiler (syntax/cwd startup failures corrected and excluded) measures6-edit518stylecalls8.015s/8.456s and lostACK527calls15.375s/16.019s. Motion:false A/B retainedsamecalls and firsttimedout10.843s; wrapperfullyreverted. QueryA/B reducesnumeric repeatedrole lookup to actual linked labels, retainingexplicitroles:157/359calls;paired2pass(14filterednotcredited). Addexplicitvisibility beforeediting/afterreopen/afterRecovery; nowrunall16 with exactpackageconfig, deadlines/0.45/writecountsunchanged. No production or globalsetup change. Full-profile gatepending; independent review requested.
Full exact UIpackage CI profile passes16/16,0skips,86.681s tests90.05s total, preservinglocal10s/global15s and allvalues/writecounts. Query optimization retains explicit spinbutton roles/visibility and actual labels/controlDOM; dedicated numeric a11y/value/bounds/native-disabled cases retainoriginalrolequeries. Independent frontend source review clear. This resolves reproduced fixture query bottleneck, not a claimed production setting overwrite. Normal frontend native/WebUI16 and final static gates pending.
Reassessment after motion/query/visibility experiments: exactpackage16passes, but normalfrontendnative5s budgetfails2othercases(alreadyAppliedtrue5.326s and selector naming5.929s), with six-edit9.903s near10. Stop acceptance here; no failuredisabled/timeoutraised. Full scope remains open. Inspect2–3 existing accessibility/fixture patterns and reduce redundant query work preserving actual role, accessible name, visibility, scopedstorage and transport checks. Coreproductionoverwrite still notproved; fixture-only candidate notcommitted.
Reassessment candidate passed all16 defaultfrontendnative cases with unchanged5s/default and six-edit10s. Added naming-case exact accessible-name/role/visible assertions on labelled actual controls; native committed recovery caches originalreview removal and nowalsoasserts exact currentReview textabsence per reviewer minor; actualApply cache mustremainInDocument before disabled. Original numericrole/bounds cases preserved. Final exactCI+defaultnative+WebUIall16qualification underway on final source. No production/globalsetup/motion/CSS mock or timeoutchange.
Final source passesall16/0skips in actualpackageCI46.511s, defaultnative45.714s anddefaultWebUI63.011s, preservingCI15s/default5s/local6-edit10s. Originalvalues/persistence/writecounts/mocks/storage/controlsemantics unchanged. Roles/exactaccessibleNames/visibilityexplicit; actualApplystillInDocument; actualReviewdetachplusnoCurrentReviewlabelnegative. Finalindependentfrontendboundedreviewclear, Minorresolved.1TSscope0introducedlint/0errors/1baselinewarning. Test-only TS:sourcecompilers excludeit;Vitesttranspile/executionqualified;BanditN/A. Review Docs/Reviews/CHAT_NUMERIC_CI_PROFILE_13661_2026_10_13.md. No productionoverwritefix claimed; broaderwholePR/CI/integrationremaining.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed measured numeric fixture query cost without weakening actual controls, accessibility, persistence or time budgets. Full16cases pass exact packageCI and both default frontend runtimes; independent bounded review and scoped static gate pass. No production/global setup change.
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
