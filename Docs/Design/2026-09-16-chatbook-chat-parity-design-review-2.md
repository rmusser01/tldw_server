# Chatbook chat parity: second design review

Date: 2026-09-16. Tracking: TASK-13261. Scope: the revised [audit/design](2026-09-16-chatbook-chat-parity-call-path-audit.md) and [inventory](2026-09-16-chatbook-console-parity-matrix.md), after [review one](2026-09-16-chatbook-chat-parity-design-review.md).

The [closure addendum](2026-09-16-chatbook-chat-parity-review-closure.md) records subsequent iterative review and current decisions. These findings describe this historical pass; deferred choices are superseded by the addendum.

**Five additional findings; the design documents are corrected.** This pass challenged the earlier corrections and their delivery dependencies. It does not count the unchanged R1–R9 issues again or claim the underlying application defects are fixed. Implementation remains pending the applicable sub-project specification and approval.

Source pins remain Chatbook `24094f23d59c7a9d3cfac964c19fd263bc0393b2` and server `59049e094e0845a4611ea725ae19b7c1754ea709`. Three independent reviews covered fork policy, consent lifecycle and run/context lifetimes; the primary reviewer verified findings, ran focused probes and checked the delivery gates.

## Additional findings

| ID | Priority | Issue | Correction |
|---|---|---|---|
| S1 | P1 | Late sync completion can undo disable/detach. | Version consent changes, fence pending work, and preserve receipts without restoring enrollment. |
| S2 | P2 | Revised fork rules retain request state the source explicitly resets. | Clear summaries/compaction memory and pinned/one-shot prefill; preserve declarative policy. |
| S3 | P2 | Queue success does not imply its next prompt still has the reviewed context. | Pause on semantic context change and require revision-checked adoption. |
| S4 | P2 | A delayed voice save can clear a newer turn in the same chat. | Immutable turn/attempt identity and independent settlement. |
| S5 | P2 | H1's recovery gate depends on later work, while specification gates are too broad. | Move each runtime proof to its implementing delivery and scope prerequisites to that delivery. |

### S1. Make disable/detach a durable consent transition

The revised E4 and privacy/ownership contract define opt-in enrollment but did not define its revocation while work is pending. The source service reads profile mode/device/dataset, awaits transport, and later persists those captured fields on success or error. Its repository update is unconditional. A concurrent disable or detach can therefore be overwritten by the old completion. [Profile capture](https://github.com/rmusser01/tldw_chatbook/blob/24094f23d59c7a9d3cfac964c19fd263bc0393b2/tldw_chatbook/Sync_Interop/local_first_sync_service.py#L105), [batch loop](https://github.com/rmusser01/tldw_chatbook/blob/24094f23d59c7a9d3cfac964c19fd263bc0393b2/tldw_chatbook/Sync_Interop/local_first_sync_service.py#L292), [late write](https://github.com/rmusser01/tldw_chatbook/blob/24094f23d59c7a9d3cfac964c19fd263bc0393b2/tldw_chatbook/Sync_Interop/local_first_sync_service.py#L926), [repository update](https://github.com/rmusser01/tldw_chatbook/blob/24094f23d59c7a9d3cfac964c19fd263bc0393b2/tldw_chatbook/Sync_Interop/sync_state_repository.py#L1997).

A controlled service probe held pull transport open, changed the real temporary profile, and released the old cycle. Both cases reproduced the defect:

| User transition while transport waits | State immediately after transition | State after the old cycle completes |
|---|---|---|
| Disable | `local_only`; original dataset/device | `local_first`; original dataset/device restored |
| Detach | `local_only`; dataset/device cleared | `local_first`; original dataset/device restored |

This used the actual service and repository with synthetic identities and fake transport. It is not an end-user UI or live-server test. Pending envelopes also remain selected by profile/dataset/domain, so re-enrollment needs an explicit backlog rule. [Pending selection](https://github.com/rmusser01/tldw_chatbook/blob/24094f23d59c7a9d3cfac964c19fd263bc0393b2/tldw_chatbook/Sync_Interop/sync_state_repository.py#L1211).

**Design correction:** distinguish pause, disable/detach and remote deletion. Pause retains enrollment/backlog and stops new dispatch; already accepted work may settle. Disable/detach increments a durable consent revision and prevents further batches/local apply under the old revision. Retain unsent work under its original binding without releasing it automatically; re-enrollment reviews that backlog. Late responses can record original-operation receipts for reconciliation, but cannot rewrite consent or retarget work. Already accepted remote effects cannot be promised undone.

Reuse the existing browser server/account request lease; this is an additional durable consent guard, not a replacement routing layer. [Existing lease](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/services/service-prompts.ts#L604). Add held-transport fixtures for success/error completion, restart, and explicit re-enrollment. Corrected audit E4, ownership/H4 gates and matrix F04/F05.

### S2. Reset derived request state in a fork

The first review changed the fork settings rule to drop or recompute summaries only when their range falls outside the selected path. That permits an in-path summary to enter the child, and the broad declarative-settings wording also leaves pinned prefill eligible. The source does neither: it explicitly clears pinned prefill and initializes the child summary to `(None, None)`. Its test sets a summary ending at the selected boundary and requires it absent from the child. [Settings projection](https://github.com/rmusser01/tldw_chatbook/blob/24094f23d59c7a9d3cfac964c19fd263bc0393b2/tldw_chatbook/Chat/console_chat_store.py#L6892), [summary reset](https://github.com/rmusser01/tldw_chatbook/blob/24094f23d59c7a9d3cfac964c19fd263bc0393b2/tldw_chatbook/Chat/console_chat_store.py#L8347), [exact exclusion fixture](https://github.com/rmusser01/tldw_chatbook/blob/24094f23d59c7a9d3cfac964c19fd263bc0393b2/Tests/Chat/test_console_chat_fork.py#L2275).

**Impact:** the child's next request can contain a prefilled answer or compressed context the source fork would not use. The same source fixture also allows excluded draft/prefill/summary changes without invalidating the fork snapshot, so a generic settings-version equality check must not silently turn them into copied-state changes.

**Design correction:** clear one-shot/pinned prefill and existing summary/compaction state regardless of its range. Preserve their policy controls and accepted character behavior; a later explicit child compaction is a separate operation. Compare the actual allowlisted copy projection while retaining required source conversation/leaf/authority guards. A coarse revision change can trigger revalidation rather than automatically treating an excluded field as changed content.

The source exclusion test and three ownership cases passed in this review. Add a destination fixture that reopens the child and inspects its next request. Settled stopped/failed text remains eligible with its honest status; this review did not find a new rejection of partial terminal state. Corrected the fork projection, settings/fence rows, H2/H3 gates and matrix B02.

### S3. Pause queued work when its effective context changes

The first correction preserved full-page failure handling and added Stop/pause, but still left success sufficient to drain the queue. Chatbook additionally checks a context epoch and requires explicit, revision-checked adoption of changed context. Its epoch intentionally excludes ordinary linear appends, streaming, terminal status and persistence bookkeeping. [Context check](https://github.com/rmusser01/tldw_chatbook/blob/24094f23d59c7a9d3cfac964c19fd263bc0393b2/tldw_chatbook/Chat/console_prompt_queue_coordinator.py#L692), [adoption](https://github.com/rmusser01/tldw_chatbook/blob/24094f23d59c7a9d3cfac964c19fd263bc0393b2/tldw_chatbook/Chat/console_prompt_queue_coordinator.py#L1008), [epoch semantics](https://github.com/rmusser01/tldw_chatbook/blob/24094f23d59c7a9d3cfac964c19fd263bc0393b2/tldw_chatbook/Chat/console_chat_store.py#L21454).

The browser queue snapshots model/prompt/tool settings but has no equivalent context token; full-page dispatch combines those settings with current context. An intervening branch change, historical edit, rewind or compaction can change what a waiting prompt means. [Queue snapshot](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/utils/chat-request-queue.ts#L5), [current-context dispatch](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundQueueManagement.ts#L435).

**Design correction:** bind the queue's baseline to the owning conversation's effective context. Add a `context_changed` pause and explicit adoption of the reviewed current context, fenced by both queue revision and context token. A second change during review invalidates resume. Normal linear responses must continue draining without requiring repeated review. This token is distinct from `history_version` and the queued model/settings snapshot.

This finding is statically verified; no new queue runtime result is claimed. Required fixture: change the selected context while an item waits, observe no next send after success, explicitly adopt the context, and verify only the intended queued request runs. Corrected audit E3 and matrix C10.

### S4. Give each accepted spoken turn independent settlement

The revised E8 covers chat/context/device ownership, but the same chat can contain overlapping turn acceptance and persistence. The shared helper keeps one current-turn ref; finalization captures it, awaits a save, then clears it unconditionally. Both forms fire finalization without awaiting it. If B begins while A's save waits, A's completion clears B, and B's later deltas/finalization are ignored. [Turn replacement](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/hooks/useVoiceChatMessages.tsx#L65), [save and clear](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/hooks/useVoiceChatMessages.tsx#L110), [full-page callback](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx#L984), [sidepanel callback](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/components/Sidepanel/Chat/form.tsx#L739).

A controlled hook probe reproduced the exact order: begin A, hold A's save, begin B and append partial text, finish A's save, then deliver B's remaining text/final. Only A was saved; B remained at its partial placeholder. This isolates settlement behavior using the real hook and the repository's existing test mocks; it does not qualify microphones or transport.

**Design correction:** assign immutable accepted-turn/attempt identities and settle each independently. A completion may clear only its matching turn. Define how interruption settles partial text/status before a later transcript supersedes it; the current interrupted event handles playback/state only. [Interruption event](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/hooks/useVoiceChatStream.tsx#L454). Add delayed-save, duplicate-final, interrupted-partial and cross-view fixtures. Corrected E8 and matrix M08/M09.

### S5. Put each delivery gate where its implementation exists

The revised H1 gate required lost-response recovery, but the child transaction and durable receipt are implemented in later native-server/browser deliveries. H1 can define the response contract and prevent automatic fallback; it cannot establish complete committed-result recovery without taking on that later implementation. The final specification paragraph also required local-model/WebUI and sync decisions too broadly, despite describing queue, ACP and native fork work as independent.

This is a dependency defect in the revised plan, not another count of the previously recorded missing idempotency implementation.

**Design correction:** H1 establishes selected-path/safe-copy behavior and blocks owner-changing fallback on unknown outcomes. H2 implements and tests the native receipt and lost-response resolution; H3 implements and tests local publication/cache recovery. H4 owns the sync protocol and consent lifecycle proof. Apply only each delivery's relevant specification gates. No model call is needed to verify a fork mutation; independent generation and cold WebUI availability remain mandatory F02 work, rather than prerequisites for unrelated mutations.

A partial H2/H3 release still does not complete all B01–B03 modes or F02. Corrected the H1–H4 table, specification gate paragraph, and inventory decomposition.

## Verification and limits

| Work performed this pass | Result | Interpretation |
|---|---|---|
| Existing source fork exclusion and ownership tests | 4 passed | Source resets excluded state and preserves independent ownership in these fixtures. |
| Held-transport sync consent reproducer | 2 cases reproduced | Both disable and detach were overwritten by stale completion. Passing probe assertions describe the defect, not the desired behavior. |
| Delayed-save voice hook reproducer | 1 case reproduced | A's save cleared B; B's finalization was lost. Passing probe assertions are not a fix. |
| Queue context and delivery dependencies | Static review | Acceptance tests are specified; no new queue runtime qualification is claimed. |

The initial standalone consent probe was refused by the repository's recovery admission (`recovery_scope_uncertain`). It was rerun under the existing pytest harness without weakening that guard. Existing dependency/runtime warnings remained visible. Earlier mixed application-suite results were not rerun or reclassified.

Production source remained pinned and unmodified. Two temporary diagnostic test files were added only inside the task's extracted snapshots. Repository changes are design documents and TASK-13261 tracking; Bandit is not applicable to that documentation scope. No live model, microphone, external host or full two-surface journey was exercised.

Original diagnostic commands used the activated environments and pinned snapshot/profile-core `PYTHONPATH` from the audit. The temporary probe files listed below no longer exist at the subsequent H1 handoff; their described fixtures must be reconstructed before rerunning the probe commands:

```text
Chatbook:
python -m pytest -q --tb=short --show-capture=no \
  Tests/Chat/test_console_chat_fork.py::test_unrelated_excluded_state_does_not_stale_or_enter_the_snapshot \
  Tests/Chat/test_console_chat_fork.py::test_stage_and_register_fork_preserves_source_and_allocates_fresh_ownership
python -m pytest -q -s --tb=short Tests/Sync_Interop/test_review2_consent_probe.py

Shared UI:
node node_modules/vitest/vitest.mjs run \
  src/hooks/__tests__/useVoiceChatMessages.review2-probe.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

The consent probe uses the pinned test suite's `_repo_with_profile`, `FakeLocalFirstServer` and `RecordingLocalStore`; an event holds `pull_v2_envelopes` while the real repository switches to `local_only`, optionally clearing dataset/device. Releasing the event reproduces `local_first/dataset-1/device-1` restoration. The voice probe reuses the existing hook test's store/ID/save mocks and delays only the first save. These are diagnostic fixtures, not proposed application tests or implementation.

Original temporary evidence paths (no longer present at the H1 specification handoff):

- Source fork log: `/private/tmp/tldw-chat-parity-audit.Tq0UXo/logs/design-review2-fork-policy.log`.
- Consent probe: `/private/tmp/tldw-chat-parity-audit.Tq0UXo/chatbook/Tests/Sync_Interop/test_review2_consent_probe.py`; result: `/private/tmp/tldw-chat-parity-audit.Tq0UXo/logs/design-review2-sync-consent.log`.
- Voice probe: `/private/tmp/tldw-chat-parity-audit.Tq0UXo/server/apps/packages/ui/src/hooks/__tests__/useVoiceChatMessages.review2-probe.test.tsx`; result: `/private/tmp/tldw-chat-parity-audit.Tq0UXo/logs/design-review2-voice-overlap.log`.

Document validation passed: 244 pinned targets across 166 source files, 12 local link targets, all 90 unique ordered inventory IDs, E1–E12 and S1–S5 sections, reference labels, table structure and whitespace. Inventory counts remain 70 Partial, nine Missing in the inspected path and 11 Unverified; zero Equivalent.

The review deliberately preserves the agreed component boundaries: both full browser clients, independent local operation with optional sync, and `tldw-agent` solely for server-directed external OS execution. No additional verified E5/E6/E7 findings were found in this pass. The applicable sub-project specifications remain to complete before implementation.
