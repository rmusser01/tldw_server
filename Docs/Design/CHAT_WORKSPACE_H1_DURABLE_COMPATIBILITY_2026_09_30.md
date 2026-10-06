# Chat Workspace H1 / Durable-Turn Compatibility

Tracking: TASK-13398.12. Date: 2026-09-30.

**Status: Concrete bounded V1 approved for implementation on 2026-09-30.**
The human approved the reviewed 20-source, 1,000-scalar excerpt and 64-KiB result
limits, explicit Retry semantics and context exclusions. Implementation and
native acceptance remain separate gates; this approval does not authorize
unguarded provider mounting, broader native forks, commits or publication.
Stage 1 qualified checkpoint storage remains preserved.

## Boundary And Baseline

This document is the only sidecar write. Investigation was limited to the
existing H1 admission/settlement, durable user identity, recovery records, RAG
composition, and their workspace handoff boundaries. No application changes,
runtime/browser checks, task-record writes, or Git operations were performed.

Requester-supplied baseline: worktree
`/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y`,
latest dev HEAD `2256bc82afa154891c635df3ef955ed7a6bc61b3`, with the large dirty
candidate preserved. HEAD was not independently checked using Git. Evidence
below describes the files read in this candidate; line numbers are not immutable
commit citations and must be rechecked before implementation.

Inputs:

- [Approved checkpoint specification](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/Docs/Design/CHAT_WORKSPACE_OWNER_CHECKPOINTS_2026_09_29.md:40): owner-validated restoration, no cached-row authority, selected ancestry and native-fork behavior preserved.
- [Checkpoint implementation plan](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/IMPLEMENTATION_PLAN_chat_workspace_owner_checkpoints_2026_09_30.md:19): Stage 2/3 integration is blocked on this compatibility decision, not on Stage 1 storage.
- [Read-only feasibility review](/private/tmp/chat-workspace-h1-rag-feasibility-review-20260930.md): mounting a provider does not deliver H1-compatible grounded or durable sends.

## Verified Current Contracts

| Boundary | Current behavior and evidence |
| --- | --- |
| Durable request | `TLDWTurnSpec` contains only `user_message_id: UUID` and forbids extras. Durable requests require an existing conversation, explicit persistence, one text user and optional request-local systems; top-level H1, legacy Retry/regeneration and continuation are excluded. [Schema](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py:771), [validation](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py:1266). |
| Durable row ledger | `insert_or_validate_user_turn` rechecks owner/workspace/assistant context, inserts or verifies the UUID/text, and uses `message_insertion_order` to return history through that anchor. It rejects a later user turn. It does not resolve a selected H1 branch. [Store](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/core/DB_Management/chacha/message_store.py:1258). |
| Retry guarantee | Durable identity deduplicates the user, not inference or assistant outcomes. The HTTP regression sends the same UUID twice and expects one user plus two assistants, each parented to that UUID. [Existing test](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/tests/Chat_NEW/integration/test_durable_user_turn_api.py:103). This is test-source evidence, not a test run in this sidecar. |
| H1 native completion | The endpoint authenticates the transport/account owner, verifies scope, rejects unsupported sync/skills, and prepares current inputs. The service validates selected content and appends a server-ID input chain in one transaction. The plural append rejects a consumed selection. [Endpoint](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/api/v1/endpoints/chat.py:4555), [service](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/core/Chat/chat_service.py:4142), [plural append](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/core/DB_Management/chacha/message_store.py:784). |
| H1 reusable single input | `append_selected_history_input` already accepts a caller ID. Matching ID, exact input intent and exact selection replay the stored admission after accepted-parent validation. A new input derives its parent from the validated retained path, including null for empty/before-first. [Single append](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/core/DB_Management/chacha/message_store.py:700). |
| Selected provider history | H1 reads canonical selected content, not visible rows or insertion-order history, and binds the assistant parent to the admitted input. [Projection](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/core/Chat/chat_service.py:4651). |
| Settlement | `_validate_history_parent` checks protected admission, live input revision/state, input chain and scope, without checking another view's current branch. `settle_history_admission` atomically writes result metadata and provenance; same result ID/intent replays, changed or deleted results conflict. [Parent fence](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/core/DB_Management/chacha/message_store.py:880), [settlement](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/core/DB_Management/chacha/message_store.py:928). |
| Server acknowledgements | Native H1 wraps the completion saver with accepted settlement and returns its admission in SSE/JSON. Saved canonical IDs are emitted by the completion service. [Saver](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/api/v1/endpoints/chat.py:4613), [SSE](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/api/v1/endpoints/chat.py:5706), [result IDs](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/core/Chat/chat_service.py:7493). |
| Browser recovery ledger | Existing H1 `pending_turns` and outcome tombstones are owner/profile-qualified, credential-free observations, not server write authority. The writer protects immutable identities and does not replace stronger receipts with weaker observations. [Types](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/db/dexie/types.ts:266), [writer/read/dismiss](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/db/dexie/history-selection.ts:918). There is no provider-attempt claim in these inspected paths. |
| Workspace transport | Workspace allocates the durable UUID before scope resolution and retains it in its failed-request snapshot. `ChatTldw` strips prior client history for durable requests. [Submit](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/components/Option/ChatWorkspace/WorkspaceChatPanel.tsx:494), [Retry](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/components/Option/ChatWorkspace/WorkspaceChatPanel.tsx:596), [transport](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/models/ChatTldw.ts:468). |
| Intentional exclusions | Selected H1 rejects RAG, legacy regenerate and tracked persona. Normal mode and the shared pipeline independently reject durable/H1 combinations. Native client-managed persistence also rejects source/result metadata in preparation, serialization and saving. [Actions](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/hooks/chat/useChatActions.ts:3433), [normal](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/hooks/chat-modes/normalChatMode.ts:817), [pipeline](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts:583), [serializer](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/services/chat-history-selection.ts:462), [save](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/hooks/chat-helper/index.ts:584). |

"Durable ledger reuse" here means the existing message UUID/insertion ledger,
protected H1 admission JSON, and scoped browser recovery ledger. It does not
mean an already delivered exactly-once provider-operation service.

## Ownership Choice

**Recommend an explicit selected-history mode inside the existing durable turn
envelope, with server-owned admission and settlement.** Reuse H1's single-input
append with the durable UUID, H1 selected-content projection and accepted-result
settlement. Do not call both admission implementations. No new queue, database,
operation service, provider-attempt registry, or workspace persistence subsystem.

Alternatives considered:

1. Client-managed H1 append -> stateless inference -> H1 settle can preserve the
   UUID as the input ID and is a supported architectural pattern. However,
   adopting it for Workspace replaces its server-owned completion boundary,
   requires native citation serialization and recovery changes, and introduces
   an accepted-but-unsent interval. Keep it an alternative, not a silent fallback.
2. Combining current top-level fields or removing guards retains two incompatible
   admission owners, and durable insertion-order history still ignores the H1
   cursor. Reject this approach.
3. A new exactly-once turn subsystem exceeds current Retry guarantees and lacks
   evidence of necessity for this checkpoint. Reconsider only if exactly-once
   inference, cross-tab execution claims or durable per-attempt receipts become
   separately approved requirements.

## Proposed Request And Result

The reviewed **Concrete V1 Wire Candidate** below supersedes these illustrative
shapes for the proposed v1: it requires an explicit result projection, including
an empty source list for plain sends, and a fresh request-context digest for an
accepted-reference inference. Neither section describes a delivered API or
grants implementation approval.

The following **new fields are illustrative protocol names requiring approval**,
not fields supported by the current server. Extend strict `TLDWTurnSpec` with an
optional `history_v1` tagged union and optional `result_v1`. Retain the existing
top-level `tldw_turn + tldw_history_selection_v1` prohibition.

```typescript
type SelectedDurableHistoryV1 =
  | { version: 1; kind: "selection"; selection: HistorySelectionV1 }
  | { version: 1; kind: "admission"; admission: HistoryAdmissionReferenceV1 }

type SelectedDurableTurn = {
  user_message_id: string // Existing UUID, unchanged throughout logical Retry.
  history_v1: SelectedDurableHistoryV1
  result_v1?: { version: 1; sources: NormalizedMessageSource[] }
}
```

`NormalizedMessageSource` is a contract placeholder for the existing RAG
presentation shape, not a type delivered by TASK22 or a request to introduce a
second source model. TASK22 only restricts link eligibility in MessageSource.
The current RAG adapter's source shape is `name`, `type`, `mode: "rag"`, `url`,
`pageContent`, `metadata`; it builds the source list from retrieved documents.
[Current source projection](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/hooks/chat-modes/ragMode.ts:226),
[retrieval](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/hooks/chat-modes/ragMode.ts:487).

Wire example, with the complete H1 value abbreviated only for readability:

```json
{
  "conversation_id": "owned-workspace-conversation",
  "save_to_db": true,
  "messages": [
    {"role": "system", "content": "Frozen request-local retrieval instructions"},
    {"role": "user", "content": "Original question"}
  ],
  "tldw_turn": {
    "user_message_id": "f61d4e19-3600-4715-96fd-1f7b45cf8d77",
    "history_v1": {"version": 1, "kind": "selection", "selection": "<complete HistorySelectionV1>"},
    "result_v1": {"version": 1, "sources": []}
  }
}
```

Rules:

- Existing authenticated workspace scope stays in the existing scope channel;
  a selection or source ID does not authorize a workspace/account/target.
- Exactly one original text user, optional request-local systems, explicit saved
  conversation; no client historical prefix, tools, continuation or legacy
  regenerate/failed-Retry flags. `result_v1` requires the selected durable mode.
- First send uses `kind: "selection"`, `purpose: "send"`, matching owner and
  conversation. Retry of an accepted input uses `kind: "admission"`, with
  `input_message_id == user_message_id`; it is not legacy regeneration.
- Retrieval and source normalization finish before admission. The composer freezes
  explicit provider/model/settings, selected manifest and request-local evidence.
  RAG's rendered question/context must not replace the persisted original user.
- Reuse `prepareHistoryContext`/`finalizeHistorySelection` for the final request
  projection. Exclude only the enclosing `history_v1` provenance to avoid a
  digest cycle; include the durable UUID, explicit inference parameters,
  instructions/current user and normalized `result_v1`. No late prompt rewrite
  or retrieval refresh under that digest. [Existing finalization](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/services/chat-history-selection.ts:338).
- Specify one shared digest projection before implementation: explicit versus
  default/omitted values and transport `stream` handling must agree across
  frontend and backend. Do not invent a second hashing algorithm. The server
  freezes its resolved selected content and supported saved context before
  dispatch; a client digest is never itself admission authority.
- Strip all nested history/result controls from provider parameters and reject
  attempts to replace these controls through generic extension bodies. Only
  request-local evidence instructions and supported inference values reach the
  provider; result metadata remains inert server persistence input.

Reuse current result fields: `tldw_history_admission_v1`,
`tldw_user_message_id`, `tldw_conversation_id`, and post-save `tldw_message_id`.
For the new mode, user receipt and H1 admission must identify the same UUID.
Propose an additional `tldw_history_result_v1` containing version, canonical
result ID, accepted input ID, attempt request-context digest and normalized
sources. Emit it only after answer + sources + provenance commit. Validate the
same binding for SSE and JSON, independent of legacy metadata display flags.
Provider-generated identity/receipt fields cannot count as server receipts.

## Admission And Settlement Sequence

| Phase | Owner and proposed behavior |
| --- | --- |
| Prepare | Browser captures current owner/workspace/reference/generation and the existing H1 view lease. Capture is read-only. Retrieve selected ready media under the pinned scope; compose/normalize evidence and verify supported context. A retrieval failure/no-evidence diagnostic admits nothing and dispatches no answer completion. |
| Admit new input | Server reuses current authenticated native H1 scope/sync/skills/context gates. In one DB transaction lock/recheck the owned conversation and workspace, validate the H1 selection and freeze selected content/context. Call `append_selected_history_input` with `id = tldw_turn.user_message_id`, original user text and server-derived role metadata. Never also call plural H1 append or `insert_or_validate_user_turn`. Retain the message insertion ledger generated by ordinary message insertion. |
| Admit replay | Same UUID + exact input/selection returns its protected H1 admission; different user, owner, conversation, parent or selection conflicts without writes. If replay is followed by new inference, re-read the retained message IDs/revisions under the owner transaction; a replay receipt alone does not prove that an old source is still eligible for new generation. |
| Accepted Retry | Server validates the supplied reference against protected input authority, original text, live UUID/revision/state and unchanged scope. Derive the prior path from the stored admitted selection, re-read its exact members/revisions, and resolve supported context for this attempt. Do not append a user, select the latest leaf, or reuse insertion-order history. Changed retained content blocks new inference explicitly. Explicit model Retry may change request-local model/provider/settings, never the input identity or admitted ancestry. |
| Dispatch | Server composes from the frozen validated selected rows plus original user and current request-local systems, reusing the H1 projection. Do not fall through the legacy durable branch or run both history paths. No database lock survives inference. Return admission/user receipts before consuming provider output when streaming. |
| Settle | Server alone generates a result ID and calls `settle_history_admission` against the accepted reference. Merge only validated citation presentation data into existing message metadata, then store answer, sources and protected result provenance in the same transaction. No browser generic user/result append or later best-effort metadata patch. |
| Display/checkpoint | Browser follows a verified result only while the captured owner/workspace/reference, H1 `view_session_id`/`selection_revision` and view generation remain current. Fresh-document inspection cannot recreate an expired follow lease. A different mounted view cannot redirect the server operation. Mark recovery completed only after a validated settlement receipt; checkpoint writeback uses its own outgoing captured qualification. |

The Retry context read is a bounded DB-abstraction extension using existing
protected authority and snapshot readers, not a new history resolver. The
current native completion endpoint cannot obtain this behavior merely by
relaxing schema validation: its current service uses the plural append, while
the legacy durable branch uses insertion-order history.

Settlement deliberately does not recheck the *current selected cursor* or reject
because another branch was appended. An edited/deleted accepted input, changed
input metadata, lost owner or moved scope rejects settlement. Old source edits
after admission do not redirect an already generated answer; a subsequent new
inference must validate its retained source separately. Preserve the existing
message-before-conversation edit lock order.

## Citations Are Result Data, Not Authority

Use the existing `message_metadata.extra` storage and H1 metadata revision
binding, not a citation database or unprotected side write. Proposed allowlisted
key: `history_result_v1`, carrying normalized source presentation and the
attempt request digest. The store's protected `history_admission_json` remains
server-only; the public key cannot manufacture a receipt or permission.
[Metadata revision capture](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/core/DB_Management/chacha/message_store.py:209).

Preserve source order, citation markers and the corresponding excerpt/locator
through settlement, selected-history read and fresh-document restoration.
Never store raw retrieval options, credentials, arbitrary execution metadata or
source-owned asset capabilities. URLs are inert display data: do not fetch them
during settlement, and retain existing safe-link rendering/access checks.
Source IDs must not grant media access outside the authenticated scope.

The current selected-history formatter only reconstructs presentation from
`extra_metadata.local_history`; storing the new key alone will not render
citations. A future parent integration must map the allowlisted result key back
to its canonical MessageSource representation, without spreading arbitrary
metadata into authority. [Formatter](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/db/dexie/helpers.ts:367).

The compatibility wire allowlist and count/text/JSON limits still require a
concrete bounded specification and approval using the existing RAG presentation
shape. TASK22 rendering acceptance is not H1/durable protocol acceptance.
Validate the complete result projection before new
admission, and fail explicitly if required citation state exceeds the approved
limits; do not silently truncate it or fall back to uncited ordinary chat.
Raw RAG metadata is presently open-ended and is not an acceptable wire schema.

Adding citation persistence does not expand native `plain_v1` fork eligibility.
Its current copier rejects required metadata; preserve that gate for cited
histories until rich independent projection is approved. A source URL/media ID
is not child ownership. [Current fork restrictions](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/services/chat-history-selection.ts:656).

## Retry, Unknown Outcomes And Stop

Keep the existing durable UUID from initial submission through every explicit
Retry. Do not substitute a new UUID on timeout, mounted-provider change, model
change or checkpoint restore. Different explicit inference attempts may produce
different canonical result IDs, as current durable behavior already permits.
Replay of the *same persisted result ID and intent* stays idempotent; it is not
a promise of exactly-once inference.

Each explicit new inference attempt gets a fresh existing browser `operation_id`
and finalized request digest, while retaining the logical user UUID/admission.
Earlier unknown observations remain intact. Complete/dismiss only that operation;
never dismiss unresolved siblings because another Retry completed. This preserves
the existing recovery writer's immutable digest and single observed result ID,
without introducing a server attempt registry.

After reload, exact-payload Retry is disabled when the detached request-local
evidence payload is unavailable: a digest and source list cannot reconstruct it.
An explicit re-preparation is a new observed attempt/digest under the same user
UUID and admitted ancestry. Reauthorize/retrieve required media; unavailable
evidence fails explicitly. Do not persist credentials or add an evidence cache
to make exact replay appear available.

Reuse H1's scoped recovery ledger. For this server-owned mode, extend its narrow
whitelist with the known durable input UUID before admission, the complete
credential-free finalized selection needed to identify an uncertain send,
and normalized result sources/digest when observed. Current `persistence:
"server"` records cannot carry an input ID without admission; introduce an
explicit logical-user-ID field rather than weakening that receipt invariant.
Never persist request-scope config, raw headers, provider keys or write leases.
Checkpoint records remain references/drafts, not retry queues or accepted history.

| Observation | Required behavior |
| --- | --- |
| Stop/scope change during capture, capability discovery, prompt preparation or retrieval | Recheck generation/leases after every await; no admission/completion dispatch. Remove only local pending preparation/stubs; preserve the unsent draft. |
| Admission may have committed, receipt lost | Keep UUID/selection as `unknown`; no automatic resend or latest-history fallback. Reauthorize the same owner/workspace and inspect the user plus protected admission. |
| Admission known, no saved result | Keep `accepted_unsent` or `generated_unsaved` according to observation. Absence of a result is not proof that inference has stopped. Explicit Retry uses the accepted-reference mode and same UUID; user must understand it may repeat inference. |
| Answer committed, final receipt lost | Read the owned user and canonical settled children/citation metadata. Follow a uniquely verified matching result; if concurrent attempts make it ambiguous, expose scoped inspection, not a latest-row guess. |
| Scope/account/workspace A/B/A or New Chat | Abort consumption and invalidate mounted generation. Do not write the old result/draft into the new checkpoint. Preserve original operation observations under their original owner; do not expose or adopt them for another account. |
| Stop after dispatch | Abort browser consumption through existing signals; this is not a rollback receipt and cannot revoke a remote commit. Do not create a synthetic assistant as proof of save or automatically restart. Verified remote persistence stays attached only to the original conversation. |

For unknown admission inspection, the existing scoped message GET/list paths
are sufficient route surfaces, but **do not currently return verified H1
authority on ordinary reads**. Propose an opt-in verified admission/settlement
projection on those reads, produced by the DB abstraction after live owner,
input-state and scope checks. A result receipt additionally requires the live
assistant's protected settlement binding, expected version, non-deleted state,
intent and `result_state_digest`, equivalent to settlement replay validation.
A parent link/public metadata alone is not proof: edited, forged or ambiguous
children remain inspection-only and cannot mark recovery complete.
Do not expose raw protected JSON or derive an
admission from public metadata. Missing/deleted/unreadable records remain
unresolved or rejected; do not recreate them. [Current scoped GET](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/api/v1/endpoints/character_messages.py:1003).

Recovery does not claim a per-attempt execution ledger. A duplicate completion
request can still infer twice; a later user input or moved display must not
silently reparent its result. If stronger duplicate-result/attempt correlation
is required, this proposed minimum is insufficient and needs a separate
approved extension with concurrency and crash-recovery proof.

## Rollout, Defaults And ADRs

- No fields: existing durable API/UUID semantics and unrelated surfaces unchanged.
  Existing top-level native H1 behavior, including consumed-selection rejection,
  remains unchanged. Legacy clients do not enter the new branch.
- Require both a successful owner-bound H1 capture and authoritative capability
  discovery for the nested selected-turn request, citation result and verified
  recovery-read contracts. Existing presence detection of `tldw_turn` alone is
  insufficient. Reuse pinned, cancellable OpenAPI discovery, not a new probe
  subsystem. [Current detection](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/services/tldw/server-capabilities.ts:538),
  [pinned lookup](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/services/tldw/server-capabilities.ts:893).
- Mixed/older serving nodes or malformed capability/receipt responses fail closed.
  No retry without the nested fields, no dropping the UUID, no switching to the
  current latest branch. Roll out coherent serving-node support before exposing
  Workspace sending through a restored H1 selection.
- No native DB schema migration is proposed: reuse messages, insertion order,
  metadata and protected H1 authority. Existing rows are not backfilled into
  admissions. A legacy durable UUID lacking H1 authority remains on its legacy
  path; it cannot acquire a selected-parent receipt by adoption.
- Extend browser recovery records additively; old records remain inspectable,
  missing selection/UUID details never authorize replay. No Stage 1 record,
  split-key bounds, legacy-rejection or IndexedDB behavior changes here.
- [ADR-008](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/Docs/ADR/008-workspace-split-key-persistence-and-indexeddb-offload.md:10)
  continues to govern the existing split store. [ADR-049](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/Docs/ADR/049-chat-history-selection-ownership.md:10)
  governs verified selected ancestry and owner authority. [ADR-050](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/Docs/ADR/050-native-chat-fork-storage-lifecycle.md:10)
governs independent fork receipts and destination closure; turn admission
  cannot bypass those fences or mint fork authority. No accepted ADR is edited.

The `plain_v1` native-fork gate continues to reject any required metadata, not
only citations. An uncited send is not automatically a plain-fork candidate.

Mounting the provider and lifting route-specific restrictions sequence only
after the new backend/transport/result/recovery contracts are approved and
verified together. Keep tracked persona, unsupported saved context, tools,
uploads, dynamic UI and richer fork operations capability-gated. A plain
supported send and a citation-bearing selected-media send are separate tests.

## Focused Future Verification

These are proposed acceptance cases, **not executed tests**. Extend existing
suites rather than create a new harness:

| Scope | Required failing-first cases and success criteria |
| --- | --- |
| Schema/API compatibility | In durable schema/API suites, old bodies retain behavior and the old top-level combination remains rejected. Nested selector/admission are mutually exclusive, versions/roles/UUIDs/scopes are strict, result-only and forged receipts fail before mutations. Unsupported serving nodes trigger no fallback. |
| Real DB admission | Extend `test_history_selection_transactions.py`: selected non-latest sibling -> exact retained provider IDs/order and correct new user parent; before-first -> null. Same UUID/input/selection replay -> one user/admission; changed ID intent/parent/text or foreign scope -> no writes. Exercise independent SQLite connections and existing PostgreSQL fixtures/lock order. |
| Retry context | Accepted-reference Retry keeps UUID and original selected path even after unrelated sibling appends; explicit provider/model switch keeps ancestry. Changed/deleted retained member, changed input metadata or moved workspace rejects new generation. Do not apply legacy "later user" insertion-order semantics to an H1 sibling. |
| Result and citations | Stream/nonstream settlement stores answer + ordered sources + digest atomically. Fault in metadata/provenance write -> no result/ack. Same result ID/intent replays; changed citations/parent or public forged authority conflicts. Owner response fields cannot come from provider payload. Native plain-fork gate stays closed for required cited metadata. |
| Browser boundaries | Shared action/pipeline/service and Workspace panel suites: plain saved selected send and selected-ready-media RAG both use one server-owned path. No generic second user/result write. No evidence/retrieval error -> local diagnostic only, zero admission/answer dispatch. Retry is accepted-input Retry, not generic regenerate. |
| Recovery/fences | Lose admission or final ack; reload, inspect and explicitly Retry under original scope/UUID. Exercise Stop at each await, account/target/workspace A/B/A, selection during retrieval, New Chat, changed model and stale outgoing checkpoint effects. No automatic send; no cross-view display mutation or draft loss. |
| Native acceptance | Parent-run real Chrome with real API/DB/local model/embeddings: restored earlier branch + ready media, exact request ancestry, persisted parent/citations, plain saved send, unknown outcome, explicit Retry and Stop/account/workspace fences. Mocks do not satisfy this gate. |

Existing tests to retain include the admission/settlement replay and drift tests
at [DB transactions](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/tests/DB_Management/test_history_selection_transactions.py:478)
and [metadata drift](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/tests/DB_Management/test_history_selection_transactions.py:666),
plus [client admission boundary regressions](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/hooks/chat-modes/__tests__/chatModePipeline.history-admission.test.ts:140).
Keep their current exclusion assertions until a separately approved selected
durable branch has its own coverage; do not erase evidence of unsupported paths.
Future touched Python requires project-venv tests/Bandit and parent review before
completion. This document-only sidecar provides no runtime certification.

## Decisions Still Requiring Human Approval

1. Approve the server-owned nested durable/H1 mode and accepted-reference Retry,
   or deliberately choose the separate client-managed alternative. Neither is
   authorized by approval of this document-writing checkpoint.
2. Approve the concrete candidate's citation wire allowlist/limits, canonical
   request-digest projection and capability marker/verified recovery-read shape
   before source changes. Prove backend byte parity before advertising support;
   preserve required citation state or explicitly reject it.
3. Confirm the existing guarantee of one logical user with potentially multiple
   explicit/duplicate inference results. If a response must be attributable to
   one durable attempt after any crash/concurrency race, require a stronger
   attempt-receipt design before claiming that property.
4. Confirm the supported workspace assistant/context matrix. Current tracked
   persona and other saved-context gates are real additional blockers; this
   design does not certify them or permit a default-assistant substitution.
5. Approve coordinated backend/transport/H1 handoff work and subsequent parent
   native acceptance. Stage 1 and the unrelated MessageSource implementation
   remain independently owned and unchanged by this draft.

Self-review: one admission/settlement owner; stable durable UUID retained;
server-selected ancestry, inert atomic citations, unknown-outcome honesty and
Stop/owner/workspace/account fences preserved. No new subsystem or permission
to remove guards is implied. The named remaining decisions are approval gates,
not implementation placeholders to fill without review.

Independent design review identified three blocking ambiguities, addressed above:
live result-state proof on recovery, distinct browser Retry observations, and
fresh-document evidence replay policy. Its optional view-lease/plain-fork
precision is included. The remaining protocol-shape gates are still unresolved;
no application implementation or native checkpoint acceptance follows from this
document review.

## Concrete V1 Wire Contract (Approved)

Parent continuation on verified dev f3f1b4fdbe3fe461b371ece30887c5fff8476d9d.
This contract defines the previously illustrative protocol questions and was
approved by the human for implementation. Implementation/parity/native proof
remain required before claiming a supported capability. The independent
review's body-versus-query-scope P2 was verified against transport/endpoint code,
corrected to strictly body-only hashing with separate mandatory pinned leases
and server owner/scope checks, and re-reviewed without a new contradiction.
Source references describe the preserved current candidate, not executed tests.

### 1. Envelope And Ownership

Keep `tldw_turn.user_message_id` as the durable UUID, explicit `save_to_db:true`, owned conversation and exactly one original text user plus request-local systems [S1].
Propose the existing draft's strict `history_v1` union: `{version:1,kind:"selection",selection:HistorySelectionV1}` or `{version:1,kind:"admission",admission:HistoryAdmissionReferenceV1,request_context_digest:Hex64}`.
The admission variant's digest is **new** attempt context, not a replacement for its original `selection_digest`; the existing admission reference alone cannot carry this need.
`Hex64` is exactly 64 lowercase hexadecimal characters. Reuse existing strict H1 DTOs unchanged; `purpose:"send"`, matching owner/conversation and admission input ID equal to durable UUID are mandatory [S2].
`result_v1`, only in this nested mode, is exactly `{version:1,sources:SourceV1[]}`. Require it, with `sources:[]` for plain sends; no result-only mode.
Keep top-level H1/durable, continuation and legacy failed-Retry/regeneration exclusions. Reject tools/functions, client historical prefixes and generic-body overrides of protocol controls before any write.
Preferred proposal: one server single-input H1 append using the durable UUID, selected-content projection, then server settlement [S3]; never call both admission paths or the plural append.
No new queue, database or provider registry; no exactly-once inference or server per-attempt ledger. One logical user may have multiple explicit/duplicate result IDs.
Keep server-owned `tldw_history_admission_v1`, `tldw_user_message_id`, `tldw_conversation_id` and post-save `tldw_message_id`; their input identity must agree with the durable UUID, never provider-supplied identity.

### 2. Strict Source Projection And Bounds

`SourceV1` is a narrowed wire projection of **existing `RagSourceEntry`**, not a replacement presentation system [S4]. All six top-level keys are required:
`{name:string,type:string,mode:"rag",url:string,pageContent:string,metadata:SourceMetadataV1}`.
`SourceMetadataV1` permits only optional `source`, `title`, `chunk_id`, `retrieval_strategy`, `source_type`, `selection_reason`, `score`, `page`, `loc`, `media_id`, `author`, `chunk_index`, `total_chunks`, `start_char`, `end_char`, `chunk_start`, `chunk_end`.
`loc`, when present, is exactly `{lines:{from:SafeInt,to:SafeInt}}`; no extra keys at any depth. `SafeInt` is an integer in `0..9007199254740991`, not bool; require `from<=to`.
`page`, `chunk_index` and the four character locators are `SafeInt`; `total_chunks` is a `SafeInt` of at least 1. When both chunk index and total are supplied require `chunk_index<total_chunks`. Each character pair (`start_char`/`end_char`, `chunk_start`/`chunk_end`) must be absent or complete and ordered start<=end. Preserve both forms exactly, without aliasing or requiring equality between different ranges. All other metadata members except `score`/`loc` are strings. `score` is a finite JSON number, not bool; no invented 0..1 restriction because the renderer also presents other finite scores [S5].
The eight additional metadata fields are the bounded extension approved on 2026-10-01 after native retrieval exposed the strict projection mismatch. `media_id` is an inert locator, not access authority. Capability matching requires all 17 exact metadata properties and the `media_id_utf8_bytes:512`, `paired_character_ranges:true`, and `chunk_index_within_total:true` bounds; old nine-property servers fail closed. No version-based fallback or relaxed unknown-field handling.
No nullable optional members: absent means omitted. `name`, `type`, `pageContent` must be nonblank; `url` may be empty; supplied optional strings must be nonblank.

| Field / collection | Proposed hard bound | Evidence / distinction |
| --- | --- | --- |
| `sources` | 0..20; grounded selected-media send requires 1..20 | Reuse adjacent citation count precedent [S6], not RAG `top_k`'s 100-result ceiling [S7]. |
| `pageContent` | <=1,000 Unicode scalar values and <=4,000 UTF-8 bytes each; <=16,000 scalars total | Adjacent quote/excerpt policy [S6]; **not** an existing RAG limit. |
| `name`, metadata `source`, `title`, `selection_reason`, `author` | <=1,000 UTF-8 bytes each | Adjacent source-title policy [S6]; reason/author limit reuse. |
| metadata `chunk_id`, `media_id` | <=512 UTF-8 bytes | Adjacent source-ID policy [S6]; inert locator, no media-access capability. |
| `type`, metadata `retrieval_strategy`, `source_type` | <=128 UTF-8 bytes each | Proposed compact-label limit, not a delivered source constraint. |
| `url` | <=2,048 UTF-8 bytes | Proposed URL-display limit; no claim an existing URL validator enforces it. |
| Complete `result_v1` | <=65,536 UTF-8 bytes of `canonicalHistoryJson` | Reuse adjacent 64-KiB budget [S6], but its ASCII JSON serializer is **not** the same byte measure. |

Validate scalar-value strings (reject lone surrogates), actual UTF-8 byte counts, shape and complete budget before admission; no truncation, source dropping or uncited fallback.
Normalize raw retrieval metadata *before* strict wire validation. Preserve existing top-level name/type/URL/excerpt; resolve displayed aliases using MessageSource's precedence: score/relevance/rerank_score/bm25_norm -> `score`; chunk_id/chunkId -> `chunk_id`; retrieval_strategy/reranking_strategy/search_mode -> `retrieval_strategy`; source_type/type -> `source_type`; selection_reason/rationale/reason/why_selected -> `selection_reason` [S5].
Only display-irrelevant raw keys may be omitted. A required locator, attribution or evidence value outside this projection is a rejection gate, not permission to silently erase it; credentials, raw options, execution data, headers and asset capabilities are never eligible wire data.
URLs remain exact inert presentation strings, including non-HTTP provenance labels. No server fetch; existing safe HTTP navigation rules still apply [S5].
Keep source order and citation-marker/excerpt/locator association through atomic settlement and restoration. **Concrete mapping gate:** `formatDocs` deduplicates by excerpt and numbers `<doc id='0'>...`, while `ragMode` maps all documents to sources [S8]. For this candidate reject duplicate excerpts before admission; use the same ordered validated list for evidence IDs and sources. Any alternate/academic marker scheme needs an explicit faithful adapter, not assumed support.
`RagContextDocument`/`RagContext` are not safe exact substitutes: both allow extras, arbitrary metadata and lack these bounds [S9]. Longer required excerpts fail this proposed v1; approval of a larger budget must precede support.
Restoration needs an explicit allowlisted `history_result_v1` -> MessageSource adapter; current `formatSelectedHistory` reconstructs presentation from `local_history`, so storing the new key alone does not deliver citations [S18].

### 3. Digest Projection And Canonicalization

Reuse `prepareHistoryContext`/`finalizeHistorySelection`, `canonicalHistoryJson` and `historyDigest`; retain the H1 nine-member selection tuple unchanged [S10].
Define request projection `P` as the **final dispatched JSON body**, deleting only `tldw_turn.history_v1` (including its embedded digest); include `user_message_id`, `result_v1`, conversation ID, original user, frozen retrieval systems, explicit provider/model and every body-transmitted inference value. `P` is strictly body-only: transport target, account/auth headers and scope query parameters are excluded, not synthesized as body fields. Existing pinned target/account/scope leases and server owner/scope checks remain separate mandatory fences. A composite body-plus-scope digest would require a separately approved projection, not an implicit implementation choice.
Require explicit `stream:true|false` before preparation and **include it** in `P`; stream/nonstream are different exact payloads. No post-finalization stream rewrite (current transport performs one [S11]).
Missing, explicit null and explicit default are distinct; do not hash `model_dump` with server-added defaults or omit present nulls. Optional undefined members are removed before wire construction, matching the existing canonicalizer. Freeze/detach `P` once; attach only the finalized excluded H1 envelope, with no change to prepared evidence/inference values. Server must compare the selection/admission attempt digest to `historyDigest(P)` before admission/dispatch.
Current canonicalization sorts object keys, preserves array order and strings, rejects opaque/nonfinite values, and hashes compact ECMAScript JSON as UTF-8 SHA-256 to unprefixed lowercase hex [S10]. No Unicode normalization, trimming or URL normalization during hashing.
**Backend gate:** there is no inspected native request verifier for this projection. `_history_digest` is storage provenance (`sha256:` prefix, Python JSON number/key semantics), not a drop-in browser request digest [S3]. A backend equivalent must prove byte parity for decimals/exponents, negative zero, astral strings, numeric object keys and omission/null/default cases before capability advertisement; do not invent a second hash or claim `json.dumps(sort_keys=True)` suffices.
The server separately validates/freezes selected rows and supported saved context; a client request digest never supplies ownership, ancestry or input authority.
Each new inference gets a fresh browser `operation_id` and newly finalized digest. Identical exact bodies may hash identically; operation IDs, not a salted hash or server attempt claim, distinguish observations. Preserve all earlier unknown siblings [S12].
Reload without detached frozen evidence disables exact-payload Retry. Explicit reprepare reauthorizes/retrieves evidence and makes a new observed attempt with the same logical UUID and admitted ancestry; never reconstruct evidence from a digest/source list.

### 4. Versioned OpenAPI Capability

Propose this exact operation extension on `POST /api/v1/chat/completions`:
```json
{
  "x-tldw-selected-durable-turn": {
    "version": 1, "history": "h1_single_input_v1", "result": "rag_source_v1",
    "request_digest": "history_context_wire_v1", "recovery_read": "protected_live_v1",
    "inference_guarantee": "multiple_results_possible"
  }
}
```
Each scoped GET/list operation below must additionally advertise `"x-tldw-history-recovery-read":{"version":1,"projection":"protected_live_v1"}`.
All marker keys/values and referenced strict request/result/recovery schemas must match, including source limits and the opt-in query parameter. Mere `tldw_turn` presence is insufficient [S13].
Use existing pinned `/openapi.json` discovery with abort/generation checks, no cached/fallback-spec authority or new probe subsystem. Missing/malformed/unknown versions or incoherent serving nodes fail closed; never strip fields, lose UUID or switch ancestry.
Advertise only after coherent backend/transport/atomic-result/verified-read support and its own verification. The marker is a deployment promise, not authority derived from a provider response or a dynamic claim that every assistant context is supported.

### 5. Exact Verified Recovery-Read Proposal

Add opt-in `include_history_recovery_v1=true` to existing scoped `GET /api/v1/messages/{message_id}` and `GET /api/v1/chats/{chat_id}/messages`; the router is mounted directly at `/api/v1` [S14].
Return normal MessageResponse plus optional `tldw_history_recovery_v1` below. GET uses exact stored text; list requires `format_for_completions=false`, `include_character_context=false`, `include_deleted=false`, `render_placeholders=false`. Reject incompatible opt-in flags; keep current pagination bounds/fields. No raw protected JSON and no dependency on `include_metadata`.
```typescript
type ScopeV1 = {scope_type:"workspace";workspace_id:string}
             | {scope_type:"global";workspace_id:null}
type ResultV1 = {
  version:1; result_message_id:UUID; result_message_revision:"1";
  admission:HistoryAdmissionReferenceV1; request_context_digest:Hex64;
  sources:SourceV1[];
}
type RecoveryReadV1 =
  | {version:1;status:"input_verified";scope:ScopeV1;admission:HistoryAdmissionV1}
  | {version:1;status:"result_verified";scope:ScopeV1;result:ResultV1}
  | {version:1;status:"unverified";code:"no_protected_binding"|"live_state_mismatch"|"unsupported_projection"}
```
These are new strict `HistoryWireModel` projections, not current MessageResponse behavior; ordinary GET/list conversion does not install verified admissions [S14]. ResultV1 also specifies the proposed SSE/JSON `tldw_history_result_v1` receipt, emitted only after answer+sources+protected provenance commit.
`UUID` uses the existing durable UUID scalar validation; result IDs are server-generated. Legacy durable UUIDs without protected H1 authority return `no_protected_binding`, never backfill/adoption.
Inside one owner-validated read transaction, recheck current account/target namespace, owned conversation/workspace and requested scope. For an input, validate strict protected version-1 admission/selection, role user, matching UUID/reference, live revision/nondeleted state, exact original text, input-state digest/input-chain and stored scope using `_validate_history_parent` semantics [S3].
For a result, first validate that live input; then require assistant role, same conversation/parent, protected version-1 `settled:true`, exact settlement reference, live row version 1/nondeleted, and `_history_message_state == result_state_digest` [S3]. Public metadata/parent alone never establishes this proof.
**Intent-verification contract proposal:** constrain the text-only saver and read verifier to the same `_history_intent_digest` dictionary: `{id,sender:"assistant",content,images:[],tool_calls:null,extra_metadata:{sender_role:"assistant",history_result_v1:{version:1,request_context_digest,sources}},parent_message_id:inputUUID}`. Exclude DB-injected client/conversation/default fields. Reconstruct from exact stored text and allowlisted metadata, reject extra intent-affecting fields, and compare the protected `intent_digest`. No new protected attempt ledger [S3,S15].
This read-only verifier and normalized saver are **not delivered helpers**. Existing differently shaped/legacy results cannot be adopted by guessing their intent; they return `unverified`. Malformed authority/metadata must not degrade into a receipt. Inaccessible/missing/deleted records retain current scoped 404 behavior; no recreation or absence-as-inference-stop claim.
The browser additionally matches its captured scope/owner, UUID, original selection digest, request digest and any already observed result ID. Missing fields/version mismatch fail closed. More than one verified matching result is inspection-only, not latest-row selection.
Paginated list observations cannot prove unique results or inference completion under concurrent writes. Without a known canonical result ID or a separately approved coherent enumeration contract, ambiguous unknown outcomes stay unresolved; no exactly-once/per-attempt attribution claim.
Recovery completion/dismissal applies only to the selected browser operation. Fresh-document inspection grants no expired view follow lease, and never completes unknown siblings or overwrites another checkpoint [S12].
Extend only the existing browser recovery whitelist: immutable `logical_user_message_id:UUID`, credential-free `finalized_selection:HistorySelectionV1`, optional validated `observed_result:ResultV1`. Keep server `input_id` unavailable without admission; these extra observations are not receipts or write authority. Do not persist detached evidence systems, credentials, request config or follow leases.

### 6. Assistant / Context Matrix

"Eligible" means proposed nested mode only after its gates; it does **not** mean currently integrated or native-accepted. Both first admission and each new accepted-reference inference must revalidate live retained revisions and supported owner context [S16].
| Assistant / effective context | Candidate disposition |
| --- | --- |
| Neutral owned conversation: no saved assistant/character identity, missing behavior snapshot | Eligible for existing neutral default prompt path only; no substitution for an unavailable bound assistant. Require explicit resolved provider/model and verified skill absence. |
| One saved character, valid behavior snapshot/materialized base binding, matching identity, single turn-taking | Eligible only for exact `project_history_context` projection: saved prompt/preset and bound sampling; no live card/profile/preset lookup or silent fallback. |
| Tracked **or inherited** persona, any memory mode | Reject: projector rejects persona; inherited workspace default does not waive the gate. |
| Missing/invalid/stale character snapshot, identity/participant mismatch, multi-character/group context | Reject, preserve selected assistant/draft; no default-character replacement. |
| Active overlay, world books/lore, exemplars, memory, greeting, author note, prompt context, pins, auto-summary, unknown saved effects | Reject unless current projector proves inactive/absent; bookkeeping names alone are not proof. |
| Skills visible/eligible/unresolved/recovery state; sync-owned messages | Reject via current H1 native gates; no sync adapter or registry introduced. |
| Raw-passthrough prompt template; frozen request-local text systems | Eligible; named/custom late-rewriting template rejected. |
| Ready selected-media RAG with validated evidence/SourceV1 and faithful marker mapping | Eligible only with this result/digest/recovery capability; retrieval failure/no evidence/overflow admits nothing, no ordinary-chat fallback. |
| Plain text/no sources | Eligible with explicit empty result projection; same H1 authority/recovery requirements. |
| New uploads/images/tools, dynamic UI, continuation, legacy regenerate, comparison/richer fork | Reject in this bounded v1; existing unrelated native H1 features are unchanged. |
| Native `plain_v1` fork of any required metadata | Reject, **not only citations**. Even `sender_role` metadata can close plain copying; uncited is not synonymous with plain-fork eligibility [S15,S17]. |

### 7. Remaining Decisions And Review Gates

1. Human selects/approves server-owned nested admission/accepted-reference Retry (still preferred proposal), the added Retry digest member and multiple-result guarantee; no implementation authorization follows from writing this candidate.
2. Approve the restrictive source bounds/alias-loss policy and marker rejection rules; decide whether real required excerpts/locators demand a larger or richer separately reviewed projection.
3. Approve exact wire/default/stream policy and a backend canonicalization parity solution; approve normalized text-result intent plus read-only verifier and whether unresolved paginated ambiguity is acceptable. Stronger attribution/enumeration needs a separate explicit contract.
4. Confirm matrix exclusions, especially inherited persona, saved effects and required-metadata forks. Parent may incorporate this reviewed proposal into the design now; application integration requires these decisions and independent implementation review.
Future checks must cover schema rejection before writes, digest golden vectors, duplicate-excerpt mapping, atomic fault rollback, edited/deleted/forged result proof, multiple outcomes, missing reload evidence and owner/view/scope/Stop fences; parent still owes real checkpoint/native UAT. None ran here.

### Source Index

- S1: [durable envelope](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py:771), [exclusions](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py:1266).
- S2: [strict H1 DTOs](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/api/v1/schemas/history_selection_schemas.py:21), [browser admission reference](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/types/history-selection.ts:145).
- S3: [storage hash](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/core/DB_Management/chacha/message_store.py:50), [single append](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/core/DB_Management/chacha/message_store.py:700), [live input/settlement fences](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/core/DB_Management/chacha/message_store.py:880).
- S4: [RagSourceEntry](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/hooks/chat-modes/ragMode.ts:226), [source composition](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/hooks/chat-modes/ragMode.ts:503).
- S5: [MessageSource presentation/links/alias precedence](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/components/Common/Playground/MessageSource.tsx:7).
- S6: [adjacent citation limits](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/core/DB_Management/chacha/shared_workspace_chat_store.py:40), [strict citation validation](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/core/DB_Management/chacha/shared_workspace_chat_store.py:1162).
- S7: [RAG top_k bounds](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py:212).
- S8: [deduplicating evidence formatter](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/utils/format-docs.ts:7).
- S9: [open RAG context schemas](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py:1309).
- S10: [canonicalHistoryJson/historyDigest](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/db/dexie/history-selection.ts:46), [prepare/finalize](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/services/chat-history-selection.ts:338), [selection tuple](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/utils/history-selection.ts:135).
- S11: [transport stream rewrite](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/services/tldw/domains/chat-rag.ts:435), [durable original-user projection](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/models/ChatTldw.ts:468).
- S12: [browser recovery DTO](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/db/dexie/types.ts:266), [immutable recovery observations/dismissal](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/db/dexie/history-selection.ts:918).
- S13: [presence-only capability](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/services/tldw/server-capabilities.ts:538), [pinned discovery](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/services/tldw/server-capabilities.ts:893).
- S14: [GET](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/api/v1/endpoints/character_messages.py:1003), [list](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/api/v1/endpoints/character_messages.py:609), [response DTO](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py:491), [mount prefix](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/api/v1/router_groups/content.py:504).
- S15: [native intent preparation](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/core/Chat/persistence_service.py:165).
- S16: [supported saved-context projector](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/core/Chat/history_context.py:53), [native sync/skill fences](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/api/v1/endpoints/chat.py:4558), [selected provider content](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/tldw_Server_API/app/core/Chat/chat_service.py:4651).
- S17: [plain native-fork metadata rejection](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/services/chat-history-selection.ts:656).
- S18: [selected-history display projection](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-workspace-a11y/apps/packages/ui/src/db/dexie/helpers.ts:368).

### Latest-Dev Raw Retrieval Compatibility (2026-10-01)

The raw RAG adapter accepts bounded `source_id` (512 UTF-8 bytes),
`evidence_origin` (128 bytes), `section_path` (1000 bytes), and up to 20
`ancestry_titles` of 1000 bytes each. These are backend retrieval bookkeeping,
not fields consumed by the Chat Workspace/Research chat citation renderer or
its source navigation. They are validated and omitted from the existing
17-property SourceV1 metadata projection, just like highlighting and snippets.
KnowledgeQA's separate trust-state pipeline is unchanged. Source excerpts,
media IDs, attribution and approved locators remain exact; malformed values,
unknown metadata and credential fields still reject before admission. This
does not add authority, change wire version, or relax protected recovery.
