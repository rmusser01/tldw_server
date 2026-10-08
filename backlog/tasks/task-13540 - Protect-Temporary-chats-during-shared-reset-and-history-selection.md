---
id: TASK-13540
title: Protect Temporary chats during shared reset and history selection
status: In Progress
labels:
- chat-ux
- webui
- extension
- data-loss
dependencies:
- TASK-13454
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Shared New/Clear and full-page history loading silently replace a completed Temporary conversation. Compact New and tab switches retain snapshots, but closing a Temporary tab deletes its snapshot. Apply the accepted keep-or-discard policy at actual interactive loss boundaries, preserve synchronous cancellation and repair caller continuations without adding persistence or an asynchronous transition framework.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Retained compact New/tab switches and empty or ordinary saved chat transitions do not gain redundant Temporary prompts; existing confirmed actions receive at most one relevant discard question.
- [x] #2 Caller continuations stop when reset or selection is declined; source/account/navigation fences and native character preflight remain intact.
- [x] #3 Failure-first behavior checks, appropriate neighbor suites, both client types, scoped static checks and two independent authorized reviews pass.
- [ ] #4 Native Temporary Cancel/Discard outcomes are recorded against latest dev, with server fixtures and broader program limits explicit.
- [x] #5 New/Clear and saved or on-device history selection offer an explicit keep-or-discard choice before losing Temporary content; cancellation preserves the owned view and draft.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Duplicate search found no Temporary discard task. Highest local/dev/open-PR task IDs checked before creation. Accepted master Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md Starting/changing requires a retain/copy-or-discard choice; Docs/Design/2026-10-04-d7-conversation-model.md requires Temporary content to remain on device until explicitly closed/cleared. ADR assessment: no new durable decision; implement existing UI policy under ADR-049/050 ownership/lifetime and ADR-059 tracking. Root investigation distinguishes retained compact switches from deletion, and keeps fresh native character Temporary preflight as a no-loss refusal.
Scope split after full caller trace: compact closure is a separate deletion-and-bulk-count unit tracked by TASK-13541. This unit preserves compact snapshot-retaining New/switch controls, fixes shared reset/select boundaries and their cancellation continuations, and keeps explicit System Reset admission ahead of DB/storage deletion.
Plan: Docs/Design/IMPLEMENTATION_PLAN_chat_temporary_discard_13540.md. Selected synchronous native confirmation to preserve existing direct event/keyboard/mutation callback contracts. One reusable predicate/choice at actual discard boundaries; no UI confirmation in generic store allocation or automatic restoration. Cancel keeps current conversation; copy/export capability is not assumed. Compact target deletion remains TASK-13541.
Valid pre-fix evidence: shared reset/saved selection8causal failures8controls; on-device selection2failures18controls; Header earlyprivacy/typed-reset admission3failures7controls; actual Server/Folders sidebar continuation2failures26controls. Fixture/setup failures excluded and corrected to model immutable Zustand snapshots. First green46checks. Independent review found a distinct Notes history-only/greeting-only source cancellation bypass and held-confirmation fences to repair before final gates; compact closure remains TASK13541.
Review repairs: shared captureChatDraftSource now includes session restore/source-selection revisions; H1 fences cover shared clear/selection, character selectors, Notes and both held delete-all actions; Form clear reuses its existing input/connection preparation guard. Notes obtains Settings and Temporary admission before async identity/session writes, honors identity ACK and shared-selector false, and requests normal mode inside admitted selection before qualified intent capture. Ordinary Temporary read-only history clicks remain unchanged. Legacy sidebar deletion checks live IDs and local selection checks Settings admission. First repaired matrix: 126 passing checks. Causal follow-ups: corrected actual saved Character fixture proves one stale accepted load intent; both delete-all callbacks prove two stale H1 failures with two owned controls. Final broader gates pending.
Native IAB: declined New kept exact two-message Temporary transcript and draft including two trailing spaces, also after reload. Native JS confirm acceptance was not exposed by documented getJsDialog; later saved-selection click timed out. No extra Chrome windows created. Positive acceptance/packaged browser claim remains open. Read-only backend receipt confirms 4-row ordinary and 5-row character fixture hashes unchanged; Temporary content absent from server history. Evidence: /private/tmp/chat-native-temporary-declined-reload-13540-20261007.{png,json}; /private/tmp/chat-native-temporary-server-history-13540-20261007.json.
Final regression evidence: broad34-suite matrix956passed/5fixture-errors; repaired neighbor fixture removes obsolete store/simple-form doubles and reuses actual primitives; final targeted8checks pass (five Research + three selector). Unique declared matrix961behavior checks pass across those completed runs, with no skipped tests in the broad or final focused matrices. Final WebUI and extension types pass. All30authored TS/TSX effective lint187baseline/185current/zeroadded;3owned format checks and3English confirmation mirrors pass, no unrelated locale bytes. Both independent source reviews clear after accepted-mode and H1 repairs. Fresh origin/dev2c5f19d0328360d45bce99b8bae75ce252cc312f included. Bandit N/A, no Python repository code authored. Native positive acceptance/packaged-browser gate remains explicitly open; status and owned plan remain In Progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implementation checkpoint: shared Temporary keep-or-discard admission and caller refusal propagation are implemented and reviewed, including held-confirmation session/H1/input ownership, linked Notes admission/mode ordering, and live legacy sidebar cleanup. Regression matrix and both client types are verified; native declined New retains exact transcript/draft across reload and server fixtures remain unchanged. Native positive confirmation acceptance is still unverified in IAB, so this task is not closed. Compact closure/bulk outcomes are TASK-13541.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
