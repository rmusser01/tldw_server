# Acknowledged recovery and owned focus — TASK-13568

The accepted all-ten design and source-confirmed native Stop/tour focus failures govern this work. TASK13271 is qualified; overall TASK13458 remains open. Copy must report only an acknowledged exact clipboard write. Dismiss must confirm the actual journal transaction, retain text on failure and respect the initiating source/account/view across every await. A failed cleanup cannot authorize Retry dispatch. Explicit keyboard Stop and tutorial exits must return focus only when the initiating action still owns it.

ADR assessment: the ADR index was searched. No new durable architecture rule is required: internal committed receipts and transient UI focus implement ADR049/050 ownership and explicit operation lifecycle; ADR059 governs tracking. DOM elements/callbacks never enter persisted tutorial progress or conversation metadata. No dependency or general focus framework.

## Source and change reach

Acknowledgement: HistorySelectionReview → useHistorySelection.dismissRecovery → db/dexie/history-selection.dismissHistoryTurnRecovery. Add an optional existing HistoryOperationOptions lease and boolean committed receipt; check it around profile/read/write awaits and throw inside the transaction after retirement so writes roll back. The controller composes caller guard, captured epoch, owner/view/profile and native/local lease, catches errors and publishes feedback only while current. Existing proven completion callers in normalChatMode, native-history-character-send, history-turn-keep and accepted open-result cleanup remain optional/offscreen-compatible. Explicit open-result cleanup must use its accepted installed epoch rather than its now-retired original one. useChatActions Retry must consume the receipt before dispatch; useKeepInterruptedHistoryTurns benefits from the controller handling failures. Update affected positive receipt mocks and exact argument assertions without disabling substantive controls.

Copy/Dismiss UI: await exact result_text or input_text, use one current attempt identity, show acknowledged success or actionable failure, cancel stale feedback on source/view retirement. Capture exact focused Dismiss initiator; only an acknowledged removal can return focus after commit. Preserve focus moved away, another editor/dialog, source/H1 changes and stale completions; retain all recovery records/text on failed mutation.

Explicit Stop: keep useChatActions/useMessage raw stopStreamingRequest and queued/unmount/tab cancellation unchanged. Wrap only UI props in Playground, its Form, compact Form and Sidepanel Body, covering SendControl, StatusStrip, RuntimeInspector, Message/Compare controls, legacy compact buttons and all three composer variants. Capture a current source/H1/mount/input lease and exact focused initiating control before abort; return through the existing composer registry after removal and idle commit only while focus remains owned. Textarea keyboard Stop already owns its input and requires no focus transfer.

Tour/other returns: one transient attempt in the existing tutorial runtime carries a launcher target and source/H1/route predicate captured before Help/Quick launchers close. Preserve the original visible Help opener and suppress its competing cleanup during handoff. TutorialRunner consumes it only for current explicit Close/Skip/Finish; missing-target exhaustion, replacement/restart, external end, navigation/reset/unmount cancel. Real Joyride controls own the tour focus scope; any newer user focus/pointer/editor/dialog cancels return. Apply the same current-source/user-focus principles to the palette's remaining BODY/hidden-opener lifetime seam without reopening its completed destination-preservation fix.

## Stage 1: Transaction and controller acknowledgement
**Goal**: Dismiss returns a committed receipt under the captured owner and cannot authorize a failed Retry.
**Success Criteria**: Current removal succeeds once; missing/refused/failed/retired operations retain records or roll back, never dispatch another request and never publish to a replacement view.
**Tests**: Current/missing journal; lease retirement before/after profile, read and write; rollback after held write; wrong profile; same-ID epoch reset; stale callback before rerender; controller error retention and Retry cleanup refusal. Existing completed cleanup controls remain.
**Status**: Complete

## Stage 2: Exact recovery feedback and UI-owned focus
**Goal**: Await Copy/Dismiss and restore only still-owned removed-control/tour focus.
**Success Criteria**: Exact bytes survive rejected clipboard/storage; only acknowledged current attempts report success; Stop internals never move focus; newer focus/source/attempts cancel stale return; Close/Skip/Finish retain completion semantics.
**Tests**: Held/rejected clipboard, late old versus new attempt, raw whitespace/reasoning bytes; Dismiss ACK/failure/removed-control and focus ABA; full/compact real UI Stop/all variants/internal negative; real Help/Quick→Joyride actions, same-ID restart, missing-target/unmount and non-persisted handoff; palette source/hidden/ABA controls.
**Status**: In Progress

## Stage 3: Integrated and native qualification
**Goal**: Both clients and actual owned headless full/compact surfaces qualify the completed behavior.
**Success Criteria**: Meaningful connected tests, compilers/static/security applicability and independent source reviews pass; emitted assets/tokens/budgets remain; exact drafts/canonical response are preserved outside deliberate disposable test actions.
**Tests**: Real IndexedDB or explicitly declared transaction-double boundary; actual Copy/Dismiss/keyboard Stop/tour exit and focus; source/account retirement; full/compact drafts, no replay/unrelated writes. Physical AT, packaged extension and participant measurements remain explicit overall limits.
**Status**: Not Started

## Stage 4: Record and commit
**Goal**: Reviewable incremental commits, current Backlog and master/closure evidence.
**Success Criteria**: No broad acceptance closed from a sub-step; task criteria reflect actual qualification; only owned files staged; own plan retired only after all stages complete.
**Tests**: Canonical Backlog format, diff/self-review and commit hooks; Bandit for authored Python if any, otherwise record TypeScript applicability.
**Status**: Not Started


2026-10-09 incremental acknowledgement evidence: 84 established connected suites / 2,513 tests pass (`/private/tmp/chat-recovery-ack-13568-matrix-accepted.log`); final dependency change also passes all59 Review controls (`/private/tmp/chat-recovery-ack-13568-review-qualified.log`). Both client compilers pass. Eight TypeScript scopes introduce zero lint findings (`/private/tmp/chat-recovery-ack-13568-static-qualified.jsonl`); owned formatting was checked for runtime equivalence. Both independent reviewers source-clear the acknowledgement portion after fixing Copy admission before account-lease-retired clipboard access and keeping the existing abort listener through transaction acknowledgement. DB rollback and pending-commit tests are explicitly transaction doubles, not native IndexedDB qualification. Bandit does not apply to the authored TypeScript scope; no authored repository Python or dependency change. Copy/Dismiss UI ownership is unit-qualified; explicit Stop, tour/palette lifetimes and native full/compact ACK qualification remain open in Stages2–3. Overall TASK13458 remains In Progress; physical AT/package and participant measurements are not claimed.
