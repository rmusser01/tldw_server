# Initial server setup draft continuity

TASK-13454.10. Design Docs/Design/2026-10-05-initial-server-setup-draft-continuity.md.

## Stage 1: Keep Chat mounted through setup open and dismissal
**Goal**: Reuse the existing settings leaf in a chat dialog for readiness/composer actions.
**Success Criteria**: Existing composer instance/text/image survive open/Close; dirty dismissal cancels or explicitly discards settings edits; focus returns and diagnostics does not unmount chat.
**Tests**: Actual application event/wiring and dialog regressions; live absent-backend WebUI open/Close, dirty Cancel, focus/keyboard.
**Status**: Complete

## Stage 2: Qualify deliberate first configuration
**Goal**: Preserve only never-owned fresh input through this exact initial setup commit and normal owner verification.
**Success Criteria**: Existing account/Core clearing remains effective; no unowned storage writes; failed/stale/ABA/logout/New/newer-input cases cannot resurrect a checkpoint.
**Tests**: Real shared composer and configuration receipt cases; storage/owner privacy regressions; actual first configuration with an isolated test backend.
**Status**: Complete

## Stage 3: Integrate compact setup and verify journeys
**Goal**: Review the actual compact owner-loader/lifetime and both surfaces' setup behavior.
**Success Criteria**: Genuine product first-time and existing-account paths are distinguished; no shared-hook-only claim closes native/package behavior.
**Tests**: Actual compact route, WebUI/extension types, relevant regressions, touched lint and both independent source reviews. Bandit N/A if no Python production scope.
**Status**: Complete

## Stage 4: Record bounded completion
**Goal**: Backlog evidence and an incremental local commit.
**Success Criteria**: Only verified acceptance closes; parent all-ten program remains open for remaining native greeting/roleplay/responsive/Sync/backend usability work.
**Tests**: Staging/diff verification; omit generated runtime/dependency symlinks; no push/PR/remote merge.
**Status**: In Progress

## Stage 1 implementation checkpoint (2026-10-05)

The Chat-local dialog reuses the existing settings leaf and keeps the editor mounted. Readiness and composer settings requests use the existing cancelable navigation seam; diagnostics opens separately. Native Modal focus return has a Chat main-region fallback. Unsaved settings use the existing app confirmation API with explicit Keep editing/Discard controls, including confirmation teardown. Pending Save disables normal dismissal and preserves its source/credential/account guards before persistence begins. The manual guard travels through the shared validation and credential-lock seam; the lock recheck uses the existing read-only effective-config resolver rather than recursive initialization. No first-owner input exception, H1 binding, anonymous persistent storage or checkpoint has been added.

Eight exact affected suites: 159 passing (/private/tmp/chat-setup-stage1-final3-green.log); WebUI typecheck, extension entrypoint compile, no added touched lint. Two independent read-only source reviewers cleared the bounded stage after a newly introduced credential-lock recursion was reproduced red and fixed. The first foreign-account fixture lacked the watched event; the corrected StorageEvent ABA case is green and the earlier failure is not counted as qualified red. Other red checks cover actual delayed-validation writes, diagnostics route departure, absent focus fallback and lock settlement.

Live actual WebUI /chat on port19764 with backend19765 absent: exact two-line input and source editor remain after setup open/Return; focus returns to Server settings. The composer Open settings action opens the same dialog. Screenshot /private/tmp/chat-ux-server-setup-dialog.jpg. Live dirty Cancel reached a native prompt that the browser controls could not dismiss; the implementation now uses in-app confirmation, verified with the actual AntD dialogs. Its live follow-up remains pending. First-Save continuity, strict initial absence, storage mutation atomicity and the actual compact pre-owner route remain open. Bandit N/A: no Python production code authored in this stage.

Fresh fetch advanced origin/dev from27ce976387 to9a8bdba10d (CI/merge-queue changes only). Integrate this head in the isolated remediation branch immediately after the incremental Stage1 commit; no PR/push/remote merge requested.

Stage1 commit41dcd854d0 and latest dev integration6c36816276 are now local. origin/dev9a8bdba10d is an ancestor of HEAD. Stage2 investigation begins from this integrated head; the live dirty-confirm follow-up remains pending.

## Verified Stage 2–3 checkpoint (2026-10-05)

The deliberate first Save uses a RAM-only checkpoint of exact fresh text/image metadata. Strict raw configuration absence is checked before initial configuration work and inside the existing credential lock; durable readback, source lifetime, account revision and normal owner verification must all qualify before input is restored. Web Locks are required for this narrow exception. Ordinary settings Save still works without them, with an explicit not-retained warning. Existing ownership, account clearing and stale continuation guards remain active. URL-only hints no longer manufacture an auth configuration before Save. No anonymous persistent draft, synthetic H1 binding, provider request or queue is introduced.

Compact cold connection failure now uses the existing recovery callout with announced Retry/checking and the real runtime-specific full settings destination. Retry keeps the action surface mounted. Positive authority loss revokes private live state through the existing account boundary; transient failure preserves a verified owner, and durable saved data is not erased. Real WebUI shim/runtime and native extension ID cases are covered separately.

Actual WebUI dirty Keep editing/Discard preserves the exact two-line draft and returns focus to Server settings (/private/tmp/chat-ux-dirty-dismissal-retained.jpg). The first session-only Save against isolated latest-dev backend5775d3fbbe retains `FIRST SAVE REVIEW — KEEP THIS IDEA\nSECOND LINE WITH EXACT SPACING  `, including two trailing spaces, through Save, Return to chat and Retry (/private/tmp/chat-ux-first-save-session-retained.jpg). An initial post-Save reachability failure resolved through actual Retry; profile/health/storage returned200. The limited-router backend has no LLM provider, no Send was performed, and this is not full generation or packaged extension proof.

Final verification:21 affected suites610 tests and3 compact suites182 tests, all792 passing. Logs /private/tmp/chat-setup-stage2-final-suites.log and /private/tmp/chat-setup-compact-final-green.log. WebUI types and extension entrypoint compile pass; touched lint has no added findings; git diff --check is clean. Both authorized independent read-only reviewers found no remaining Critical/Important issues in the bounded source unit. Bandit N/A: authored scope is TypeScript and documentation. Native packaged lifetime, mobile/zoom, screen-reader behavior, real provider generation, Sync and later roleplay acceptance remain open in the parent program.

Existing fixture failures were checked against an immutable git-archive HEAD baseline, without overwriting working files. Coordinator fixtures now use coherent initial stores and actual accepted H1 load receipts; queue expectations preserve full payload while asserting the intentional blocked restore transition and timestamps; Flashcards fixtures provide typed owner/runtime/Core state. Original privacy, popup, transfer and parent/child address assertions remain.

Fresh origin/dev5775d3fbbe includes quota/backend work and nullable API types. The verified source unit will be committed first, then that dev head integrated at the clean checkpoint and overlapping types rechecked. Stage4 remains open until the local commit and tracker completion are recorded.
