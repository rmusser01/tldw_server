---
id: TASK-13550
title: Offer history choices only when conversation history exists
status: Done
labels:
- chat
- ux
- webui
- extension
- nng
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair identified NN/G CC-04 / #3106 in the shared WebUI and extension history review. A fresh first send creates a qualified ready owner with an empty captured source; readiness alone currently exposes Use no prior messages even though there is nothing to omit. Gate history-path actions by actual source availability while retaining loading, error, recovery, fork and expand controls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A fresh full Playground first send with a held provider response exposes neither history-path control; the existing known-defect wrapper becomes a normal passing regression only after causal failure is verified.
- [x] #2 A qualified nonempty source still exposes Review and Use no prior messages, and choosing empty preserves omitted alternatives.
- [x] #3 Empty source loading, stale/error, unsupported, retained recovery and expansion remain visible where applicable.
- [x] #4 Both client types, appropriate real-source tests, authored lint, canonical task format and two independent reviews pass; bounded browser evidence and unrun gates are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Parent TASK13458 / accepted all-ten design Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md. User requested all root causes and enhancements. Read-only investigation confirms normalChatMode installs a valid empty captured native/local source on first Send; HistorySelectionReview gates Review by nodes but empty action only by ready. Correct the shared action availability root, preserve H1 semantics and other state controls, and avoid changing source creation or completion contracts. Sequence: reproduce actual known-defect regression and component boundary controls; minimal shared condition; affected tests/static/native and independent review; working incremental commit. ADR assessment:no new durable decision; ADR049 selection ownership is preserved and ADR059 governs task editing. Task id13550 exceeds latest fetched origin/dev, all53 open PRs and local13549.
Initial red run:2 direct shared availability assertions fail causally,19 controls pass; full Playground fails before provider dispatch and is excluded from causal evidence. Traced fixture mismatch: current client sends its durable UUID in POST /chats and requires the matching receipt, but fake-tldw-server POST ignores body.id and returns server-chat-N. Correct only the declared HTTP fixture to echo/persist the supplied caller ID as the deployed endpoint does, then repeat the red before touching production. No provider/server capability or ownership guard will be weakened.
Valid red after the declared UUID fixture correction: full Playground reaches1 real completion request with matching created/persisted caller UUID, then fails on the visible Use no prior messages button during the first held turn. The2 direct empty-source availability assertions fail for the same button;19 controls pass. Fix only the shared ready/nonempty action predicate. Loading/errors/recovery/fork/expansion and native/local owner creation remain intact.
Expanded11-suite gate passes10 suites/149 checks but exposes2 held-owner session-fixture failures: outcome remains pending, not an incorrect owner publication. Their resolveServicePromptScope double creates a fresh unresolved promise on every read and retains only the last resolver, while real composition has multiple consumers/rechecks of one account verification. Correct the two declared fixtures to share the same pending verification promise across reads, preserving rejection/restoration assertions and all production owner fences. This gate is not recorded as passing.
Corrected held-owner fixture gate43/43 passes; original account refusal, automatic-allocation retention, source retirement and restored H1 receipt assertions are retained. No production session or principal behavior changed. Both client type gates pass. Native initial first-send AX has Queue request/Stop Streaming and an empty selection region with no history-path buttons; real backend7e50a83e-a370-4ea0-84ae-67ea8750cee9 stores exactly user+assistant, verified in /private/tmp/chat-fresh-history-13550-native.json. Screenshot /private/tmp/chat-fresh-history-13550.jpg records the settled chat, not the in-flight response. Mock deterministic provider proves lifecycle, not roleplay quality. Both original source reviewers clear the bounded availability fix; fixture tail supplement is pending.
Final11-suite151/151 runner checks, comprising148 positive checks and3 retained expected-failure counterexamples for unrelated model-startup and saved-message Regenerate/Edit-and-Send, not acceptance for those known defects. All8 affected HTTP-harness consumers execute; original UUID and held-scope mismatch failures excluded from causal evidence. Both final client types0;5 effective TS lint files0 authored diagnostics,24 inherited; authored formatting and canonical diff/task checks pass. Both independent reviewers clear the source and fixture tail. Master design updated with exact current frontier. TS-only production/test delta: Bandit N/A, no Python changes. Native first-send and2 actual stored messages bounded to mock-provider lifecycle; packaged/physical/AT/participants and broader all-ten work remain open. Only this availability task completes.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
History-path actions now appear only when captured history exists. Fresh first Send no longer offers Use no prior messages for an empty source; nonempty review/empty choice, recovery, errors, loading, fork and compact expansion remain available. Full Playground regression is causally verified and converted to a normal test. Shared HTTP/held-principal fixtures match current declared contracts.151 runner checks, both types, authored lint, two reviews and bounded native2-row first-send evidence recorded;148 positive checks and3 remaining known-defect counterexamples distinguished. Master design updated; broader remediation continues.
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
