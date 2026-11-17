---
id: TASK-13615
title: Remove inert duplicate notification bridge from the app shell
status: Done
labels:
- frontend
- performance
- ci
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The authenticated WebLayout owns NotificationLifecycleProvider and its working toast bridge. AppProviders also imports/mounts an inert bridge outside that context, making public/shared shell bundles pay for notification lifecycle code. Remove that duplicate and its obsolete AppProviders-only flag, retain the real layout owner, and qualify actual notification behavior plus the unchanged production Turbopack budget.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Only the real provider-owned notification toast bridge remains; no public shell eager import.
- [x] #2 All obsolete AppProviders flag callers and fixtures are removed without weakening authentication or notification-owner tests.
- [x] #3 Actual toast lifecycle/ownership and app-provider/layout tests pass.
- [x] #4 Exact locked Bun1.3.2 production Turbopack, token sync and unchanged600/900KB limits pass with normal client types and bounded review.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required:no; remove inert duplicate using existing React context/provider ownership, no durable architecture change. Source evidence WebLayout NotificationRuntimeOwner owns real provider+bridge at904-907; AppProviders duplicate outsidecontext at100. Fresh exactDocker-version locked Turbopack HEADeb8d1ec197 compiles/tokenspass butshared600.5KB exceeds600by0.5; diagnosticcopy attempts failed fromescaped/incomplete dependency graph and were reassessed to housefrozeninstall. Highest local/origin/openPR taskID13614 checked; latestorigin/dev remains2a8ffa78a8. User authorized all remaining PR3229 root fixes/review/rebase/merge.
Plan Docs/Plans/IMPLEMENTATION_PLAN_shell_notification_owner_13615.md. Design: remove only inert duplicate notification bridge from AppProviders and its private obsolete flag/callers. Existing WebLayout NotificationRuntimeOwner retains provider, bridge, auth/demo gating and stream authority. ToastProvider staysglobal so provider-owned bridge stillhasitsancestor. No dynamicimport budget workaround, no payloadthreshold change, no durabledecision change. Scope reaches AppProviders,_app,app-layout fixture and AppProviders import/component tests, plus actualnotification toast/lifecycle regression checks.
Final actualnotification/auth/splash/dependency suites132passed5files; both normal clienttypespassed. FreshBun1.3.2 frozenlockedNext16.1.4 productionTurbopack build/token/budgetpassed:shared589.7KB versus600.5baseline (capunchanged600), heaviest868.4KB(cap900). Boundedreviewclear. Review Docs/Reviews/CHAT_SHELL_NOTIFICATION_OWNER_13615_2026_10_12.md; own completedplanretired to/private/tmp. BanditN/A TS/Markdown-only. Docker/remoteCI/packagedextension/physicalAT/wholePR/H2/finalreview/rebase/merge remainopen.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Delete inert out-of-context AppProviders notification bridge and obsolete private flag/callers. Preserve the working layout-owned provider/bridge and auth/demo gates.132tests/types/exactlockedproductionbuild/tokens/budgets passed; sharedshell shrinks10.8KBgzip to589.7KB without raisinglimits.
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
