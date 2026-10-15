---
id: TASK-13684
title: Complete chat fixture contracts exposed by final PR CI
status: Done
labels:
- chat
- testing
- ci
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair incomplete shared-hook fixture exports and diagnose the remaining Scene/reset test failures on the rebased PR3229 head, preserving production ownership guards and behavioral assertions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Affected fixture modules retain actual shared hook and query client contracts
- [x] #2 Scene conflict and reset-refusal regressions are diagnosed and pass without relaxed assertions or raised deadlines
- [x] #3 Affected native UI and client profiles pass and independent reviewers inspect the final delta
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new durable architecture decision; existing ADR049/050 ownership and ADR059 task editing govern this test fixture repair. CI RED: head716e8cd shards3/4/6 show missing useQueryClient / useActorSceneHistoryId exports, one Scene read-count failure, and one reset-refusal timeout. Plan: inventory caller mocks; reproduce under native UI CI profile; minimal fixture repair; verify affected suites and independent review; publish and rerun required gates.
Final actual provider-failure journey initially received200because the owned mock was still in local-success mode; this is fixture configuration, not a relaxed401 assertion. Native654affected fixture cases pass. Both WebUI/extension profiles are running all20 affected/connected suites (1121cases perprofile expected). Protected provider-failure qualification will use the existing CI mock configuration and retain401/receipt/canonical-row/explicitRetry checks.
Currenta34frontendshard2job116127515077: four focus-visibility callback harness tests fail ReferenceError isMobileViewport. Production callbacks are qualified by23realbrowsercases but the older AST harness omitted newly referenced mobile viewport/cockpit flags. Add declaredscope and real extracted-callback positive/negative railfocus cases; preserve existing4behaviorassertions and no productionchange.
Current a34 CI shard7 reproduced locally using native packages/ui Vitest: 5 files, 36 failures, 428 passed. Complete three partial React Query mocks with real empty QueryClient, preserve catalog result branches; correct only two dispatched provider recovery expectations to unknown; qualify forced splash after actual settings hydration. No production authorization or request retirement bypass.
Shard8 read-only reviewer diagnosis: three cockpit-controls expectations remain Checking model because their ready/degraded catalog fixtures never declare transport connection. Qualify existing cockpit-control harness as CONNECTED without changing production readiness gate or degraded/provider assertions. Native shard7 now all464 pass.
Shard5 CI diagnosis: two more partial Query mocks missing useQueryClient; SceneSettingsRecovery actual mirror needs hoisted transactional memory Dexie in jsdom; old compact settings mount contract must preserve delayed afterClose lifetime; compact AST harness must declare new persona-disabled variable for ordinary picker. Actual CI RED logs /private/tmp/chat-pr3229-a34-shard5-independent.log. Keep real mirror/editor/CAS and source-extracted inline JSX. Prior10file532 each WebUI/extension GREEN; 464+65native GREEN.
Native remaining4files full195/199: four pre-existing Scene defaults cases hit15-second local deadlines under concurrent heavy suites/type checking. All8 exact held/retained/mirror boundaries pass isolated with unchanged15-second limits, including both real mirror cases. Sequential complete qualification remains required. Optional package-wide native tsc finds legacy fixture/dependency diagnostics; all60 diagnostics in changed UI files are on unchanged HEAD expressions. Established native connected production tsc and WebUI tsc pass; do not call full native test typing clean.
Native complete Scene112 now passes164.44s sequentially; no raised deadlines or relaxed assertions. Earlier concurrent native195/199 plus this full112 gives all199 affected outcomes. Complete WebUI/extension Scene profiles now executing sequentially; other3 suites had all87 passing in each. Local physical compact-shim preference suite6passes in WebUI config; extension config intentionally collects only shared UI and excludes frontend shim tests (not a pass claim). All13 TS files lint0errors,167unchanged warnings,0ignored,0new. Normal18file hooks pass.
Exact9f CI shard3 failed one interrupted-reply integration boundary: request arrival was treated as admission receipt, then Stop sometimes occurred before the protected admission was consumed. Native isolated full16 passes locally; investigating receipt timing independently before changing test. Preserve unknown-outcome production handling, original loading/no-send/draft assertions, and synchronize the actual retained admission for the admitted-turn scenario. Failure log /private/tmp/chat-pr3229-final-shard3-failure.log. ADR unchanged049/050/059; existing final CI plan Stage4 remains In Progress.
Receipt-timing hypothesis rejected: isolated zero-token scenarios with0/1000ms admission-delay both pass. IndependentA identified actual jsdom realm leak: simulatePageReload left old provider/controller/30s watchdog alive across singleton-store resets; CI old watchdog fires during next zero-token Stop case. New regression RED confirms reload leaves old controller un-aborted. Repair only shared fixture lifetime: unmount then abort without user-Stop marking, wait real controller release and HistoryTurnSettlement before store reuse/remount; drain mounts afterEach. No production edit or weaker loading/no-send oracle.
Reload-drain regression GREEN; complete9harness-affected suites77pass native,77WebUI,77extension sequentially; latest WebUI tsc passes. Two TS files lint0errors0warnings0ignored0new. Both independent reviewers clear exact working fixture lifetime delta. Current final production97hashes unchanged, so earlier final build23real cockpit and security evidence retained. Exact deterministic native CI shard3 context running before commit;6of7published9f required statuses pass, frontend needs this repair. Touched TypeScript fixtures only: Bandit not applicable to this delta; previously verified67production Python files unchanged.
Final deterministic CI native shard3 context:51files987tests PASS82.42s, including the original loading/no-send case and new reload-drain regression;0failures, unchanged15-second limits. Complete9affected suites77pass each native/WebUI/extension. Both final readonly reviewers clear source supplement; final WebUI types, TS lint0findings,4file normal hooks and97unchanged production hashes qualify. Source ready for normal commit; final new-head7statuses and authorized merge remain parent13458/13690 landing gates.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed remaining a34 CI fixture contracts with real partial Query exports, exact declared connection/mobile scope, only dispatched unknown expectations, real splash hydration, transactional memory-Dexie and actual compact JSX/dialog lifetime. Native464+65+focus8 pass; ten affected suites532 each WebUI/extension. Remaining message/comparison/shell87 pass each; full Scene112 passes sequentially native164s/WebUI174s/extension154s with unchanged deadlines, superseding local resource-pressure timeout diagnostics. Eight exact boundaries also pass isolated. Compact shim6 passes owningWebUI; extension profile excludes it.13 TS files lint0errors,167baseline warnings,0new; normal hooks pass. Native connected production and WebUI types pass; optional full native fixture typing is not clean (60 changed-file diagnostics on unchanged expressions). Both readonly reviewers cleared current fixture deltas. Parent13458 and13690 track latestdev integration, exacthead independent review,7statuses and authorized merge.
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
