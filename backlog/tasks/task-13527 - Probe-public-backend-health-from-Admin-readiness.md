---
id: TASK-13527
title: Probe public backend health from Admin readiness
status: In Progress
labels:
- bug
- admin-ui
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Admin unauthenticated readiness incorrectly probes protected versioned /api/v1/health, so multi-user deployments return401 and report503 despite healthy backend. Use the configured backend origin with public /health while preserving auth on versioned health, request-host isolation, timeout and failure semantics. Plan: reproduce with existing ready-route tests; make the one-site URL correction; run readiness/config regression tests and Admin typecheck; independent review and upstream PR to current dev. ADR required: no; existing public control-plane health contract is restored, not changed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Readiness probes public /health at the trusted configured backend origin, never request-supplied hostname or protected versioned health.
- [x] #2 Existing timeout,503/error/no-store semantics remain and regression tests plus typecheck pass.
- [ ] #3 Independent review complete and source fix submitted against current dev with private operational evidence excluded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Two readiness regressions failed against the original route, then14 readiness/config/liveness cases passed after one-site URL correction. Admin tsc --noEmit --incremental false passed; targetedESLint --max-warnings0passed. FullAdminVitest executed and NOTclaimedgreen: changed117failed/701passed/818total; unchanged origin/dev export with same lockeddependencies has identical117failedassertionnames and zero newlyfailednames (698passed/815total; totaldifferencesrequireclassification). Machine-readable per-test names: /private/tmp/task13527-admin-baseline-results-20261007.json and /private/tmp/task13527-admin-changed-results-20261007.json. Existing frontend-required implements baseline ratchet; requiredCI stillmustpass. GlobalNode24 emits DEP0205 Vitestmodule.register deprecation; no app warnings suppressed. Independent source review found noP1/P2. NoPython sourcechanged; Banditnotapplicable toTS-onlyscope. ADRrequired:no. Prior humanChangeSummarywaiver remains explicit.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
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
