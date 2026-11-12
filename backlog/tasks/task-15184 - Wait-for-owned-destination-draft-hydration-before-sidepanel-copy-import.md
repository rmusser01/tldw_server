---
id: TASK-15184
title: Wait for owned destination draft hydration before sidepanel copy import
status: Done
labels:
- chat
- ux
- extension
- bug
dependencies:
- TASK-13457
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue approved Stage4 root repair from TASK13509. The initial one-shot receiver can prefill/consume before destination draft hydration and retains imported page context across owner/conversation changes. Reuse actual composer draft readiness and existing H1/account/action guards to retire initial reads and explicit choices; no transfer-format or authority changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Initial reads/apply/consume wait for actual owned destination draft readiness; restored drafts receive existing conflict choices.
- [x] #2 Held initial and explicit reads cannot apply or consume after destination owner/conversation retirement.
- [x] #3 Imported page context clears on owner/conversation change without erasing unrelated destination text.
- [x] #4 Actual hook/Form-boundary failure-first tests, client gates and independent review pass; parent provenance/native gates stay explicit.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Latest dev1fc353 integrated atffecdcaf78; catalog root repair839d7a04ee complete. Parent Stage4 remains In Progress. Focus this bounded unit on readiness and destination ownership; initial source provenance/unknown owner review, cross-origin server references and native docking remain separate required work.
Root investigation reproduced cold destination hydration, held initial and choice retirement, imported-context carryover, callback/StrictMode cancellation latch, stale Cancel, readiness recovery, and consume-service post-await retirement. Failure-first checks recorded in /private/tmp/chat-handoff-destination-red.log, chat-handoff-lifetime-red.log and chat-handoff-consume-red.log. Receiver reuses actual composer owner/readiness and H1/action fence; shared consume now rechecks captured owner at its final remove boundary. Review UI shows copied plaintext before explicit choice. No package-format/source-provenance authority added.
Final review reproduced same-URL route cleanup stranding and unrelated file-retrieval/assistant preparation invalidating displayed choices. Retirement is consolidated across navigation/readiness. The import fence is narrowed to real owner, mounted H1 selected history, IDs and draft address/revision; H1 controller changes recapture discovery. No unrelated assistant/context revision is treated as destination ownership. Added failing regression evidence in chat-handoff-navigation-red.log and chat-handoff-fence-red.log.
Both authorized independent reviewers clear the final bounded source; neither ran the final matrix. Final checks and latest-dev integration remain the completion gate. Incoming dev advanced to1047ce10ecc191f78b0bee7e1b8ad19780c68015 (includes PR3170 shared domain/session authority fixes); commit this verified unit incrementally, merge current dev, then run affected plus incoming authority regressions before marking Done. Bandit is not applicable to TS/TSX/JSON/docs; no Python implementation changed.
Final latest-dev checks:18suites567passed (/private/tmp/chat-handoff-latest-dev-final-regressions.log), catalog66passed separately (/private/tmp/chat-handoff-latest-dev-models.log), WebUI types and extension compile exit0, no-added touched lint9files, official English agreement/unrelated locale byte checks and diff checks pass. Both independent source reviewers clear final bounded fix; merged authority integration also reviewed clear. Live remote dev7ba48f251ec47a1e0bb680f49f9b7d86ec2b988d includes checked1047ce10 and is integrated at2f7fbb8653. Root implementationac925515a3. Master design updated; own completed temporary plan retired. Parent13457 remains In Progress with source/provenance/qualified references/native/receipt gates. No Python implementation touched; Bandit N/A.
Renumbered from TASK-13512 to TASK-15184 on 2026-10-31 (TASK-13960): TASK-13512 now names only "Add performance baseline harness and query-counting test utilities" (backlog/tasks/task-13512 - Add-performance-baseline-harness-and-query-counting-test-utilities.md). Before 2026-10-31 both tasks used TASK-13512; Docs/Development/Backlog_Task_Id_Renumbering.md lists every renumbered id.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reused actual composer restoration, account owner, selected-history fence and draft address/revision for copy discovery/choices/consumption. Fixed canceled discovery and conflict recovery across readiness, navigation and retired held choices; previewed exact copied plaintext. Final latest-dev567+66 checks and both client gates pass. Source provenance/full native/packaged composition remain explicit parent work.
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
