---
id: TASK-13503
title: Keep compact character selection reversible until accepted
status: In Progress
labels:
- chat
- ux
- extension
- character
dependencies:
- TASK-13454.14
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Within the approved all-ten chat UX program, fix the compact character controls entry point: opening/cancelling a picker must preserve the source overlay, assistant, chat and typed draft; choosing must use the existing guarded picker callback and preserve the source draft before a navigation-approved destination transition. Prevent delayed selection work from publishing after modal close, source/owner changes or navigation cancellation. Reuse existing native preflight, account/selection fences and draft retention rather than adding a parallel conversation framework. Library and Quick Chat entry points remain separate bounded work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Opening and cancelling either tracked picker leave source assistant, overlay, chat identity, draft and resume marker untouched.
- [x] #2 The real picker delegates tracked selection; cancelled navigation and failed preparation preserve the source, while accepted changes retain its exact draft and bind only the current destination.
- [x] #3 Held preparation/identity writes are fenced across picker close, account changes and New/Open; supported native configuration is checked before releasing the source.
- [x] #4 Relevant red/green component and surface regressions, both client types, no-new touched lint and two independent reviews are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved design: Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md. Governing ADR-049/050 owner and persistence decisions remain unchanged; ADR-059 task editor applies. No new ADR: this restores the existing account, H1 and private-draft contracts at another entry point. Baseline latest dev 587cd8e9fe integrated in f460db867c; prior model checkpoint 61da94ba19. Plan will be Docs/Plans/IMPLEMENTATION_PLAN_compact_character_selection_2026_10_05.md. Allocated TASK-13503 after checking dev, local task paths and all open PR file paths (highest13502); no duplicate character entry task found.
Review found live tool choice omitted, synchronous image/text ABA unguarded, void identity ACK on supersession, later Sheet intents not retiring preparation, and failed verification stuck loading. Added eight failing regressions (81 initial passes) then repaired these using useSimpleForm.getRevision, existing selected-assistant operation revision/serialized write ACK, keyed picker intents, and existing useChatDraftOwner retry. Shared full native character acceptance now also honors declined ACK; no new owner authority or clearChat flags. Focused first verification: 152 tests in five suites pass. Callback adapter is source-boundary evidence, not packaged extension/focus/hydration proof.
Additional independent reviews reproduced last-accepted rollback being overwritten by optimistic hook render and then by backend watch-before-ACK. Ref-only update still failed; direct owned Storage removed optimism but watch-before-ACK remained. Reassessed the existing acknowledged composer/store patterns and serialized selection writer. Correction must insulate the existing accepted baseline during the pending owner write queue, including installed Plasmo and WebUI watch events. Clear overlay held patch is now fenced by Sheet intent/unmount and live source; source-change test uses immutable replacement to mirror real Zustand. Remote dev advanced to f225562fc26182c7931b4048f689d8622da4c8e5; fetched. Upstream changes concern bounded memory/PDF diff behavior; integrate after the clean local checkpoint and repeat relevant gates.
Reproduced and repaired two further queue edges with actual Storage shim and both installed hooks: newer queued write rejection after an unaccepted predecessor, and same-owner logout/login ABA (warm/cold reopen, with/without a newer accepted choice). Restoration is private-key compensation only while that owned key has no newer writer; it cannot establish a history/server owner. Failed restoration rejects explicitly, preserves the accepted RAM baseline for retry, and does not claim cold restoration. Negative test confirms cold snapshot can remain unaccepted on IO failure; successful retry then cold reopen proves recovery. Six focused suites: 179 passes before localized recovery-error assertion; both actual storage hooks still use the test Storage shim, not native physical tabs.
Final pre-integration verification: 457 tests in 16 relevant suites pass (/private/tmp/chat-compact-checkpoint-regressions.log), WebUI typecheck and extension compile pass, no new touched ESLint diagnostics against HEAD, official English locale sync and git diff --check pass. Both independent reviewers clear current source with no remaining substantiated Important/Critical finding. Bandit N/A: touched repository source is TS/TSX/JSON/docs only; no Python code changed. Code checkpoint will precede integration of newly fetched f225562fc26; Stage 3 remains In Progress until post-merge gates. Packaged extension/native physical tabs/focus/actual whole-form hydration and real-provider generation remain parent-program gates. Failed preference rollback proves accepted RAM preservation and explicit failure/retry, not durable restoration while IO fails.
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
