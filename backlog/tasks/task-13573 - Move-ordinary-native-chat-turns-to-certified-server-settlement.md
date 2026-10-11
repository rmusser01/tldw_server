---
id: TASK-13573
title: Move ordinary native chat turns to certified server settlement
status: In Progress
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
- [ ] #1 Qualified ordinary native Full and Compact dispatch one server-settled completion under the original CID and selected context without duplicate browser input or assistant writes.
- [ ] #2 Original admission and final receipts are strictly consumed; missing, conflicting, foreign or retired acknowledgements preserve uncertainty and cannot adopt latest, tip, text matches or arbitrary assistant children.
- [ ] #3 Stop, transport loss and length-limited replies preserve qualified partials and generation metadata under original ownership without claiming an unacknowledged save.
- [ ] #4 Local, Temporary, SDK, Compare and enrolled Sync callers retain their supported semantics; no server-native mutation is routed around Sync ownership.
- [ ] #5 Existing generation-status/model metadata UI consumes canonical data; legacy states and recovery remain usable.
- [ ] #6 Connected native/shared, backend contract when touched, both types, static/security/hooks, current native and latest-dev verification are recorded with packaging/AT/participant limits.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Master TASK13458 and stage1 TASK13454 remain In Progress. Accepted design Docs/Design/2026-10-04-d7-conversation-model.md P2/P5; read-only offline Q1 means no offline-send outbox. Native TASK13572 proof has one browser user append, save_to_db false completion, one browser assistant append; canonical metadata is empty. No original settlement receipt can be inferred from messages GET, whose tldw_history_admission_v1 is null. Two authorized independent source reviewers trace actual shared and uncertainty/Sync boundaries. ADR check: no new ADR for consuming the existing owner-bound completion contract under ADR049 and accepted D7. Reassess and record a new ADR before adding public capability/receipt endpoints, persisted protocol or run ownership; ADR050 is specifically native-fork storage, not blanket turn authority. Plan Docs/Design/IMPLEMENTATION_PLAN_chat_native_server_settlement_13573.md.
Stage1 causal mounted RED qualified: original Full and Compact normal native consumers both dispatch save_to_db false at unchanged held-Bound positives; /private/tmp/chat-native-server-settlement-13573-dispatch-red.log and patch archive. Test file restored exact committed SHA272661626e38d0cf7bf0da3322bb415e1d944af74c62036b5616a40192685423. Split finite prerequisite TASK13574: opt-in per original owner capture completion capability, preserving older strict clients and enrolled Sync. ADR required yes for this public capability, proposed Docs/ADR/067-native-completion-capture-capability.md; core P2 consumption remains existing ADR049. Provider-error frame can precede saved stream_end, and ordinary interrupted Retry cannot replay consumed selection. Those receipt/recovery constraints remain parent Stage2/3.
TASK13574 capability implementation is qualified and awaiting normal final hooks/commit: opt-in scoped capture, legacy JSON/older omission, exact context and actual Sync scoped helper; no dispatch changes. 638 connected clients, authenticated125PASS5skips, final service-backed Sync35PASS, both types and0 introduced lint/security, two source reviews CLEAR. Next shared seams: explicit current-input provenance; completion primary gate rather than MessageCreate; original raw receipt observer; save-helper/recovery server discriminator; error before late partial receipt and monotonic saved evidence; retain current explicit Retry until its consumed-selection protocol is supported. No guessed generic GET/text/tip/ancestry ACKs or after-dispatch browser fallback.
TASK13574 Done with all4AC/6DOD and17-file normal hooks passed. Original-owner capability prerequisite is qualified and staged for normal commit; shared ordinary dispatch still opts out save_to_db and remains the next root. Current plan stays active; no package/native settlement/physical AT/participant acceptance inferred.
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
