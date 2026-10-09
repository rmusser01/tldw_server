---
id: TASK-13567
title: Keep retained chat recoveries compact and distinguish no-admission refusals
status: Done
labels:
- chat-ux
- webui
- extension
- recovery
dependencies:
- TASK-13458
- TASK-13563
- TASK-13565
- TASK-13566
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Actual current native recovery shows old retained operation cards crowding out the conversation, and a definitive no-admission409 being marked as unacknowledged save failure by the generic write tracker. Correct shared full/compact presentation and exact scoped refusal status without deleting/deduplicating records, hiding current choices, replaying writes or treating unknown assistant outcomes as saved.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current history recovery choices remain prominent; retained records have accurate counted disclosure with every exact payload and existing owned per-record action accessible, including same-owner refresh and new-owner reset.
- [x] #2 Only exact qualified strict-false native no-admission rejection preserves the previous save outcome; malformed/foreign/ambiguous/conflicting/transport/assistant outcomes remain failed and errors pass through unchanged.
- [x] #3 Causal component/domain/tracker controls, connected neighbors, both clients, introduced static/security checks, two independent source reviews and current native clutter/badge/no-write evidence pass; broader all-ten limits remain explicit.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Accepted all-ten progressive disclosure/visibility/recovery scope; actual buildD7F825-h96xHLo8ujYwqc screenshot /private/tmp/chat-native-recovery-focus-13566-branch.png shows two complete old records plus a third before any messages. Existing relatedTASK13278 is specifically stopped-answer recovery ordering/provider-fallback and remains separate; TASK13457.3 protects unknown assistant ACK and must not be weakened. Fresh dev97ea9cd5fa included. ADR049/050 and acceptedD7 private presentation/acknowledged persistence rules govern; no new durable architecture or protocol, ADR059 tracking. Two authorized reviewers traced shared Review and actual addChatMessage/tracker roots, source-only.
Causal evidence: original density/status regressions red; sparse ID vector red then Array.from validation green; both reviewers independently found A expanded -> B no cards -> A disclosure reopening, owner-red.log reproduces it. Existing owner retirement effect now clears disclosure, native Toggle is guarded by original mounted/draft/H1 fence; same-owner expansion preserved. Final presentation41/41 and prior connected250/250. Distinct identical-input operations remain individually retained, chronological actions and explicit saved/uncertain-result access qualified. Existing nested Inspect disclosure intentionally remains. Two bounded final source reviews CLEAR. Seven owned TS formatted with runtime-equivalence assertion; all eight touched TS 0 introduced static findings. Both client compilers/expanded matrix/current emitted build/native UI checks remain in progress. No authored repository Python; Bandit N/A. Evidence /private/tmp/chat-recovery-presentation-13567-.
Final44-suite matrix1986/1986, both client compilers pass;8 touchedTS0 introduced static findings,7 owned formatting runtime-equivalent, English mirrors agree, two final source reviews CLEAR. No authored repository Python, Bandit N/A. BuildQXKfdWQJDouv3-Aum_OEk tokens/all799 assets and budgets595.1/791.3 pass. Actual same native full desktop saved continuation five-card region752px ->88px; keyboard Enter reaches every retained Copy/Dismiss and exact input inspection. Actual scoped strict-false POST10625 returns exact409, keeps prior Saved on server badge/exact draft and current choices ahead of five old cards. Explicit Latest then cold reload keep six collapsed recoveries/22 selected rows/exact draft; Latest returns composer focus. Full canonical25-row responses deep-equal,0 generation and no input admitted. Temporary finder preview intentionally read-only; explicit New saved then Find is the continuation precondition. Both compilers/shared controlled compact behavior qualify source reach, not packaged/mobile/physical/AT/clipboard-success/all ten. Fresh dev97ea9cd5fa included. Master updated; own completed plan preserved at /private/tmp/chat-recovery-presentation-13567-completed-plan.md. Evidence /private/tmp/chat-recovery-presentation-13567-native-proof.json and associated logs/screenshots. Dismiss-last focus/TASK13278/backend settlement/Edit/settings/lore/import/Resume/Sync/extension splash/full acceptance remain separate.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Compacted older retained outcomes without discarding any operation, prioritized current recovery choices and preserved live prior status only for exact original scoped no-admission refusals. Causal edge-case,1986 connected checks, both clients, introduced static/equivalent formatting/source reviews and current native canonical/no-generation proof pass. All broader acceptance limits remain explicit.
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
