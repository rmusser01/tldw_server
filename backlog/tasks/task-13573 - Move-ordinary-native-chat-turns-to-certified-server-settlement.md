---
id: TASK-13573
title: Move ordinary native chat turns to certified server settlement
status: Done
labels:
- chat
- ux
- root-cause
- d7
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute accepted D7 P2/P5 for ordinary saved native Full/WebUI/extension and Compact chat. Current shared pipeline opts out of server persistence, admits input separately and writes the assistant from the browser. Reuse existing native selection/admission/settlement and metadata rendering; retain original source/account/Stop and primary-operation capabilities, uncertain receipts and enrolled Sync boundaries. This is the next finite original-scope root after TASK13572.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Qualified ordinary native Full and Compact dispatch one server-settled completion under the original CID and selected context without duplicate browser input or assistant writes.
- [x] #2 Original admission and final receipts are strictly consumed; missing, conflicting, foreign or retired acknowledgements preserve uncertainty and cannot adopt latest, tip, text matches or arbitrary assistant children.
- [x] #3 Stop, transport loss and length-limited replies preserve qualified partials and generation metadata under original ownership without claiming an unacknowledged save.
- [x] #4 Local, Temporary, SDK, Compare and enrolled Sync callers retain their supported semantics; no server-native mutation is routed around Sync ownership.
- [x] #5 Existing generation-status/model metadata UI consumes canonical data; legacy states and recovery remain usable.
- [x] #6 Connected native/shared, backend contract when touched, both types, static/security/hooks, current native and latest-dev verification are recorded with packaging/AT/participant limits.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Master TASK13458 and stage1 TASK13454 remain In Progress. Accepted design Docs/Design/2026-10-04-d7-conversation-model.md P2/P5; read-only offline Q1 means no offline-send outbox. Native TASK13572 proof has one browser user append, save_to_db false completion, one browser assistant append; canonical metadata is empty. No original settlement receipt can be inferred from messages GET, whose tldw_history_admission_v1 is null. Two authorized independent source reviewers trace actual shared and uncertainty/Sync boundaries. ADR check: no new ADR for consuming the existing owner-bound completion contract under ADR049 and accepted D7. Reassess and record a new ADR before adding public capability/receipt endpoints, persisted protocol or run ownership; ADR050 is specifically native-fork storage, not blanket turn authority. Plan Docs/Design/IMPLEMENTATION_PLAN_chat_native_server_settlement_13573.md.
Stage1 causal mounted RED qualified: original Full and Compact normal native consumers both dispatch save_to_db false at unchanged held-Bound positives; /private/tmp/chat-native-server-settlement-13573-dispatch-red.log and patch archive. Test file restored exact committed SHA272661626e38d0cf7bf0da3322bb415e1d944af74c62036b5616a40192685423. Split finite prerequisite TASK13574: opt-in per original owner capture completion capability, preserving older strict clients and enrolled Sync. ADR required yes for this public capability, proposed Docs/ADR/068-native-completion-capture-capability.md; core P2 consumption remains existing ADR049. Provider-error frame can precede saved stream_end, and ordinary interrupted Retry cannot replay consumed selection. Those receipt/recovery constraints remain parent Stage2/3.
TASK13574 capability implementation is qualified and awaiting normal final hooks/commit: opt-in scoped capture, legacy JSON/older omission, exact context and actual Sync scoped helper; no dispatch changes. 638 connected clients, authenticated125PASS5skips, final service-backed Sync35PASS, both types and0 introduced lint/security, two source reviews CLEAR. Next shared seams: explicit current-input provenance; completion primary gate rather than MessageCreate; original raw receipt observer; save-helper/recovery server discriminator; error before late partial receipt and monotonic saved evidence; retain current explicit Retry until its consumed-selection protocol is supported. No guessed generic GET/text/tip/ancestry ACKs or after-dispatch browser fallback.
TASK13574 Done with all4AC/6DOD and17-file normal hooks passed. Original-owner capability prerequisite is qualified and staged for normal commit; shared ordinary dispatch still opts out save_to_db and remains the next root. Current plan stays active; no package/native settlement/physical AT/participant acceptance inferred.
Stage2 begins from committed capability prerequisite fa85bea287. Scope reaches explicit Normal current-input provenance; shared prepare/dispatch and original raw admission/final receipts; model/SDK original ChatScope; save-helper, keep hook and Review discriminator; connected actual Full/Compact and transport/recovery tests. Two independent source reviewers identify error-before-partial-ACK ordering and accepted-operation lifetime after display New. Unsupported Actor transcript transformations and already-admitted Retry retain the established pre-dispatch lane until qualified. No automatic replay or generic receipt lookup.
Current connected first-send causal RED is 2/2 actual Full/Compact save_to_db false cases. After implementation the first positives pass. Expanded receipt controls include duplicate admission/final, admission-only primary frames, missing/foreign receipts, provider-error then partial ACK, and known save followed by drop. Three ordinary memory tests causally fail at existing Roleplay-only admission (qualified RED); expanded negative fixture initially queried getCurrent().recoveries instead of the hook property and is excluded as a fixture diagnostic. Independent source review found a preflight validation insertion and compiler caught its temporal-dead-zone plus nonexported import; corrected before further qualification. No complete/native claim yet.
Broader 18-suite run did not pass (847 pass/180 fail). Migrated incomplete Actor test mock to preserve actual exports; ordinary in-memory recovery fixture now starts with the actual durable dispatch intent and asserts dismissal; moved new extension scope cases after existing health-cache scenarios. These are fixture fixes, with no production cache reset or test bypass. Both type checks were not reached in that failed set-e run.
The fixture recheck passed DB91 and actual transport17; remaining action failures isolated to an incomplete Zustand subscribe export and an old client-managed Retry fixture incorrectly labeled persistence:server with a canonical assistant ID. Preserved actual store subscribe and migrated only that fixture to client persistence and a device-only reply ID. Real canonical server replies continue to use native Regenerate; no production gate was weakened.
Independent reviewer B found two additional root storage failures. Causal focused RED: Full/Compact acknowledged saved pair became failed when completed journal and dismissal writes hit quota; ordinary window recovery retained every streaming prefix. Narrow fix treats only acknowledged server-result dismissal as housekeeping (proof remains), and restricts distinct retained text accumulation to its original Roleplay contract. Focused RED3 precedes implementation; action regression328 now passes with honest store frame identity.
Stages1/2 complete; Stage3 final checks in progress. Connected18 suites1030 and both client types pass; proper shared-source lint274 baseline/273 current with0 additions. Both independent reviewers bounded source CLEAR after causal quota RED3/green3. Added ordinary-only Open gate RED2 and fix, retaining existing Edit/Regenerate completion requirements. Ignored ESLint paths, wrong-cwd no-test run and discarded unsafe formatting candidates are excluded diagnostics. Prior private runtime/evidence files are missing; new owned isolated API/provider ready, pending current build/headless native proof.
Final current qualification: connected18 suites1032/1032; both current WebUI/extension compilers pass; authored static0 additions; two authorized independent source reviewers bounded CLEAR. Emitted build tokens and original600/900KB budgets pass593.1/795.4KB. One headless public Full Test/Save/model/first Send without reload after TASK13575 creates canonical2; second exact raw ACK creates4; actual Stop creates6 with interrupted metadata and honest unknown-final-ACK Review/Copy, then Find/reopen/cold6. Compact debug route390x844 original raw ACK two-turn canonical4, close/Recent/reopen/cold4, empty draft and only GETs. No browser message POSTs; exact Full6/Compact4 canonical data preserved. Private proof prefix /private/tmp/chat-native-server-settlement-13573-. First Full CDP body excluded; passive later clones qualify. Bandit N/A: TS-only authored scope. Latest dev3b68f2dbc2daa1b94099a8a3d56c809fd2c808db included; ADR049/proposed067 existing. Completed own plan archived privately and retired. Final hooks pending. Linked server mirror Local badge/overview refresh is a separate investigated root; original receipt/run reattachment, import/view-only/Sync, package/physical AT/participants and all-ten remain open.
All exact owned applicable normal repository hooks pass /private/tmp/chat-native-server-settlement-13573-hooks.log; final authored19TS static274 baseline/273 current/0 additions. Six bounded AC and six DOD verified. No authored Python: Bandit N/A. No PR/push/merge; parent/master/all ten remain In Progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Ordinary qualified native Full/Compact now server-settle exactly one original turn with strict admission/final receipts, explicit current-input provenance and canonical metadata. Missing or foreign ACK remains honest unknown; no browser duplicate write or automatic replay. Supported alternate lanes preserved. 1032 connected checks, both compilers,0 new static findings, two bounded source reviews, current build/token/budgets and actual Full2/4/Stop6 plus Compact4/reopen/cold exact canonical proofs pass; normal owned hooks pass. TS-only Bandit N/A; latest dev included. Completed own plan retired privately. Original receipt/run reattachment, mirror/refresh and remaining parent surface/package/physical AT/participant gates stay open.
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
