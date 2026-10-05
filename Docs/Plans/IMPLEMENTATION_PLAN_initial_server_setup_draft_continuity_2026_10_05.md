# Initial server setup draft continuity

TASK-13454.10. Design Docs/Design/2026-10-05-initial-server-setup-draft-continuity.md.

## Stage 1: Keep Chat mounted through setup open and dismissal
**Goal**: Reuse the existing settings leaf in a chat dialog for readiness/composer actions.
**Success Criteria**: Existing composer instance/text/image survive open/Close; dirty dismissal cancels or explicitly discards settings edits; focus returns and diagnostics does not unmount chat.
**Tests**: Actual application event/wiring and dialog regressions; live absent-backend WebUI open/Close, dirty Cancel, focus/keyboard.
**Status**: In Progress

## Stage 2: Qualify deliberate first configuration
**Goal**: Preserve only never-owned fresh input through this exact initial setup commit and normal owner verification.
**Success Criteria**: Existing account/Core clearing remains effective; no unowned storage writes; failed/stale/ABA/logout/New/newer-input cases cannot resurrect a checkpoint.
**Tests**: Real shared composer and configuration receipt cases; storage/owner privacy regressions; actual first configuration with an isolated test backend.
**Status**: Not Started

## Stage 3: Integrate compact setup and verify journeys
**Goal**: Review the actual compact owner-loader/lifetime and both surfaces' setup behavior.
**Success Criteria**: Genuine product first-time and existing-account paths are distinguished; no shared-hook-only claim closes native/package behavior.
**Tests**: Actual compact route, WebUI/extension types, relevant regressions, touched lint and both independent source reviews. Bandit N/A if no Python production scope.
**Status**: Not Started

## Stage 4: Record bounded completion
**Goal**: Backlog evidence and an incremental local commit.
**Success Criteria**: Only verified acceptance closes; parent all-ten program remains open for remaining native greeting/roleplay/responsive/Sync/backend usability work.
**Tests**: Staging/diff verification; omit generated runtime/dependency symlinks; no push/PR/remote merge.
**Status**: Not Started

## Stage 1 implementation checkpoint (2026-10-05)

The Chat-local dialog reuses the existing settings leaf and keeps the editor mounted. Readiness and composer settings requests use the existing cancelable navigation seam; diagnostics opens separately. Native Modal focus return has a Chat main-region fallback. Unsaved settings use the existing app confirmation API with explicit Keep editing/Discard controls, including confirmation teardown. Pending Save disables normal dismissal and preserves its source/credential/account guards before persistence begins. The manual guard travels through the shared validation and credential-lock seam; the lock recheck uses the existing read-only effective-config resolver rather than recursive initialization. No first-owner input exception, H1 binding, anonymous persistent storage or checkpoint has been added.

Eight exact affected suites: 159 passing (/private/tmp/chat-setup-stage1-final3-green.log); WebUI typecheck, extension entrypoint compile, no added touched lint. Two independent read-only source reviewers cleared the bounded stage after a newly introduced credential-lock recursion was reproduced red and fixed. The first foreign-account fixture lacked the watched event; the corrected StorageEvent ABA case is green and the earlier failure is not counted as qualified red. Other red checks cover actual delayed-validation writes, diagnostics route departure, absent focus fallback and lock settlement.

Live actual WebUI /chat on port19764 with backend19765 absent: exact two-line input and source editor remain after setup open/Return; focus returns to Server settings. The composer Open settings action opens the same dialog. Screenshot /private/tmp/chat-ux-server-setup-dialog.jpg. Live dirty Cancel reached a native prompt that the browser controls could not dismiss; the implementation now uses in-app confirmation, verified with the actual AntD dialogs. Its live follow-up remains pending. First-Save continuity, strict initial absence, storage mutation atomicity and the actual compact pre-owner route remain open. Bandit N/A: no Python production code authored in this stage.

Fresh fetch advanced origin/dev from27ce976387 to9a8bdba10d (CI/merge-queue changes only). Integrate this head in the isolated remediation branch immediately after the incremental Stage1 commit; no PR/push/remote merge requested.
