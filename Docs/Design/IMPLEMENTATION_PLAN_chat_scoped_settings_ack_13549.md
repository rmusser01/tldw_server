# Scoped chat settings acknowledgements — TASK-13549

Approved scope: master chat UX remediation Stage 3; the user requested all issues/enhancements and root-cause repairs. Latest origin/dev2c5f19d0328360d45bce99b8bae75ce252cc312f is included in5191a7566e. No provider/default/library changes are inferred from a conversation edit.

ADR assessment: derived qualified settings keys and leases enforce ADR049; ADR059 governs tracking. Private exact-intent/device reconciliation records follow the approved D7 settings design and existing local-bucket CAS. They grant no server authority, native behavior capability, cross-client operation contract, credential persistence or background replay. ADR050 provides a no-blind-replay persistence precedent only; it does not establish ordinary-settings capability. A new authoritative durable protocol would require a separate ADR. Bare-ID caches never certify an authenticated native destination.

## Stage 1: Shared settings failure and acknowledgement roots
**Goal**: Stop GET failures and rejected/malformed PUT receipts from becoming apparent server success or overwriting server state.
**Success Criteria**: A failed GET causes no PUT/cache/source cleanup. Only matching conversation/settings responses certify server acknowledgement. Initialization/storage failure is surfaced; canonical returned settings are preserved.
**Tests**: Actual shared service causal failures, valid200-empty and matching-ACK controls, scratch/source preservation, storage failure and cancelled/stale boundaries.
**Status**: Complete

## Stage 2: Captured ownership and durable device receipt
**Goal**: Qualify ordinary saved/draft settings to their actual owner/address/workspace and carry existing scoped credentials, cancellation and current H1/view fences through every caller.
**Success Criteria**: Account/workspace/chat collisions and held H1 ABA cannot mutate another owner or clean up a newer pending edit. Exact patches and server/device outcomes remain distinct; unknown write outcomes reconcile by GET before Retry. Intentional detach captures its source lease before publication.
**Tests**: Existing bucket/owner/request transport controls, held GET/PUT retirement, device failure after server ACK, newer edits versus old receipts and only unresolved patch Retry.
**Status**: In Progress

## Stage 3: Editor commitment and supported field scopes
**Goal**: Make conversation Apply/Cancel/dirty/retry reflect supported per-conversation fields and preserve independent model defaults.
**Success Criteria**: Two chats using one model remain independent, resetting/changing model cannot overwrite newer edits, closing staged forms has a dirty admission, live/default-only controls describe timing/scope. Settings/scene partial Retry skips acknowledged steps, including roleplay drawer. Direct dictionary assignment follows captured transport/ACK rules.
**Tests**: Actual CurrentChatModelSettings, ConversationTab, RolePlaySetupDrawer and useChatSettingsRecord/loader/link/dictionary callers; settings+scene partial failure and reopened native saved chat.
**Status**: In Progress

## Stage 4: Verify, review and record
**Goal**: Complete all scoped settings criteria with truthful evidence and incremental working commits.
**Success Criteria**: Both client types, affected causal/integration/static/security gates, two independent source reviews and bounded native evidence pass. Task/master current; broader native/D7/Resume/packaged/physical/AT/participant gates remain explicit.
**Tests**: Appropriate affected suites and native same destination/account/workspace controls, authored lint/format/diff, canonical tasks and commit hooks; Bandit for any Python scope.
**Status**: In Progress

Checkpoint (2026-10-08): shared failure/ACK and exact-intent repair is implemented; qualified settings cache and private receipts reuse existing local durable buckets. The platform lock serializes read/compose/PUT; differing unresolved edits remain staged rather than replace pending intent. Intentional patches omit client top-level timestamps, and canonical ACK must actually contain the edited fields. Legacy canonical schema remains readable; only backend-defined attachment deletions treat known absence as a null clear. Local/fork storage guard and complete local memory-map replacement remain intact. Render records carry their requested storage key, and old callbacks check rendered IDs/address/revision before capturing live state. Shared loader/first-bind/detach callers carry original scope/signals/leases. Remaining Stage2 integration/static/native checks and all Stage3 editors/dictionary/metadata/sampling behavior are still open. No completion or native capability claim is made at this checkpoint.


Incremental editor checkpoint (2026-10-08): 27 affected suites/615 checks pass. Actual Current46/scene48/drawer45 and shared model11/field-scope26 are included. Final ownership-hook dependency repair preserves the source tuple/H1 fence; actual Playground research controls24 pass, including a red-first synchronous New/held restore continuation. The one history-control expected-failure counterexample is retained as unfinished work, never positive acceptance. WebUI and extension type gates pass; authored TS lint has no added findings across40 files; Python reader40 checks and the unchanged-scope Bandit baseline are recorded in TASK13549. The task and Stages2/3 remain In Progress.

Next supported slice: replace ConversationTab change/blur saves with explicit Apply/Cancel, report dirty/pending to the parent and preserve edited fields through hydration. Retain an exact component-local proposal and acknowledged step flags for partial Retry, using shared settings reconciliation; add no queue/store. Use captured ordinary target for metadata and retain fork/pending restrictions. Enforce the deployed expected_version with one PUT and strict raw matching ID/version/edited-field ACK; failed discovery or a conflict must not trigger an unversioned or silently rebased write. Remove the sibling feedback fallback that defeats this rule. Topic Clear sends null; persona memory mode remains unavailable because the deployed update schema does not accept it. Model system prompt remains explicitly a model default. Saved previews are labelled as applied state. Native sampling/int-participant and dictionary flows continue afterward with unchanged capability refusals.

ADR assessment for the next slice: no new ADR. ADR049 and the accepted behavior-snapshot/versioned-settings decision already require captured ownership and versioned updates. This corrects the client use of that deployed CAS boundary, adds no server authority or native capability, and does not change the storage protocol. Any new public operation/behavior/import contract still requires its own assessment.

Final checkpoint addition: source review found same-chat Attach/Remove and held follow-up races beyond New. Two valid red controls now pass after the existing attachment commands advance one local intent revision; all26 research integration controls pass. This is a bounded presentation/continuation fence, not a new persisted protocol or native verification. The expected-failure empty-history control remains open.

Metadata checkpoint (2026-10-08): deployed versioned updates now make one PUT, preserve conflicts, and require raw matching ID/advanced actual version/edited-field ACK. Captured advisory ratings do not replay, publish after source retirement, or downgrade live versions; the dead metadata writer is removed. Four affected suites98 checks and scoped consumer controls pass after correcting the declared discovery/ACK fixture. Both client type gates and authored lint pass; two independent source reviews identify no remaining introduced blocker. This is an incremental transport/caller checkpoint, not ConversationTab/native/all-ten completion. The explicit editor, native sampling/participants and dictionary slices remain In Progress.

Supported-control truth checkpoint: persona mode is read-only under the deployed update schema; the prompt explicitly identifies its model-default Save scope, and attachment removal identifies filename/immediate timing. Four red controls became ten green, plus46 existing Current editor controls; both types and authored lint pass. Native acceptance is still open: the existing preview bundle displays old wording and must be refreshed/verified before new UI claims. Explicit conversation commitment remains the next slice.
