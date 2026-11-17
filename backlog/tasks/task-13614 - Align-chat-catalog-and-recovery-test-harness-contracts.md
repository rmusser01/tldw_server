---
id: TASK-13614
title: Align chat catalog and recovery test harness contracts
status: Done
labels:
- chat
- tests
- ci
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair PR3229 frontend harnesses that still describe obsolete model query keys or incomplete query results, omit required QueryClient ownership or chat-settings exports, and assert a superseded mutation callback spelling. Preserve readiness and recovery negative controls; do not weaken production code or CI gates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Character first-use fixtures supply current verified catalog contract and retain unavailable/loading/error controls.
- [x] #2 Research-context and Notes harnesses retain real required providers/module exports while controlling only intended IO.
- [x] #3 Composer recovery assertions verify current acknowledged callback behavior.
- [x] #4 Affected suites and normal type configurations pass with bounded independent review.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required:no; fixture ownership/contract repair under existing model catalog and acknowledged recovery decisions. User authorized all remaining PR3229 fixes and eventual review/rebase/merge; whole-PR acceptance remains open. Highest local/origin/openPR task ID13613 checked before creation.
Plan Docs/Plans/IMPLEMENTATION_PLAN_chat_frontend_harness_contracts_13614.md. Current catalog query key is playground:chatModels plus generation, with verified=isSuccess&&!isFetching. Character fixtures still use getModelsForFieldGeneration and omit isSuccess. Research integration mounts Playground without QueryClientProvider. Notes replaces whole chat-settings module and omits getChatSettingsStorageKey. Composer signal guard expects superseded one-argument onError callback. Production validation/recovery will remain unchanged.
First green pass161passed/3failed. Reassessed popup renderer boundary after catalog+controller fixes did not restore sending: real useDraftPersistence uses durable local registry, unavailable in native shared runner without an extension backend; creation recovery ledger has same boundary. Compared existing useCharacterQuickChat.ownership renderer adapters (scene/retention/ledger), opt-in native.integration real-owned-server path, and Manager existing mocked native send. Use bounded renderer IO adapters with explicit retention-failure negative control; preserve separate real durability/admission suites. Notes remaining assertion contradicted H1 staged publication: identity/messages committed before navigation while serverChatMetaLoaded intentionally false pending metadata admission.
Final six-suite run193passed/1unrelated prompt-editor60s timeout; both normal clienttypespassed. Actualpopup ownershipnegative suite included. Isolating timeout without increasingthreshold before deciding whether furtherfixture repair is required. Focused popup/private-retention/Notes controls4passed129excluded; bounded source reviewer clear, wholePR gates remainopen.
The isolated prompt-preset timeout control passed unchanged (1passed100excluded). Complete Manager101-case rerun now runsalone withoneworker; other five suites from finalrun passed, includingpopupownership and research/Notes/catalog/recovery guards. Both normaltypespassedagain afterTask13615 sourcechange. Boundedreviews sourceclear; fullcharacter-rerun pending. Task13615 committed86dae8125e resolveslockedTurbopack budget to589.7KB with132actualshell/notificationchecks.
Complete Manager rerun passed101/101 withoneworker; five other complete affected suites passed93cases in finalrun. All194casesqualified acrossfinalcompletesuite runs. Isolated prompt-editor timeout control also passedunchanged; no raisedthreshold. Bothnormaltypespassedafterfinalfixtures/source. Boundedreviewclear. Review Docs/Reviews/CHAT_FRONTEND_HARNESS_CONTRACTS_13614_2026_10_12.md. Owncompletedplanretired/private/tmp. BanditN/A tests/Markdown-only. RendererIOadaptersdo notclaimdurablestorage/installedextension; registryunavailablediagnosticwarningsnotdurableacceptance. WholePR/CI/H2/finalreview/rebase/merge remainopen.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Align character catalog result/key fixtures, research query ownership, Notes real exports/staged metadata and acknowledged-focus guard. Declare popup renderer I/O consistently and verify retention refusal preserves input without dispatch/navigation. All194affectedcasesqualified acrosscompletesuite runs, bothnormaltypespassed, boundedreviewclear; no productionguard weakened.
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
