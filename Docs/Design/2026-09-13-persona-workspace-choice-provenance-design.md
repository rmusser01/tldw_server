# Workspace Persona Choices And Startup Provenance

Status: Proposed Stage 2 contract, awaiting user approval. No runtime changes are authorized by this document alone.

Tracking: [#2950](https://github.com/rmusser01/tldw_server/issues/2950), TASK-13245, design TASK-13245.1. Depends on Stage 1 [#2957](https://github.com/rmusser01/tldw_server/pull/2957) and the [parity plan](../superpowers/plans/2026-09-13-persona-workspace-parity-implementation-plan.md).

## Scope And Corrected Baseline

Make Workspace opt-out durable and record how a Workspace chat's initial assistant was chosen. Preserve existing callers; add an explicit versioned, retry-safe selection protocol for adopting clients. No Buddy, animation, UI, design-system, tool-profile activation, provisioning, or Workspace Sync rollout.

Inspected server branch: Stage 1 commit `e713fc448a9996fae5dfec7f15fceb1cd3c10087`; server dev rechecked at `beac8e9449b0e2fa90bdab89cf8cbf7905d2b915`. Relevant startup/schema/DB/Sync files are unchanged between their dev baselines. Chatbook dev: `392ce191fd28953550f85154ea1f8e4eda4ab7f3`. Read Chatbook committed files, not local working changes.

The earlier assessment missed startup behavior already introduced by server commit `77e2f3765b22ce3166187b564a7bbcb88ba2880b`. These are corrections, not newly implemented capabilities:

| Existing behavior | Source evidence |
| --- | --- |
| New Workspace chat with omitted assistant fields already inherits its saved Persona default. | `tldw_Server_API/app/core/Workspaces/assistant_defaults.py`: `resolve_new_conversation_assistant`; `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`: `create_chat_session` calls it. |
| Explicit null in `assistant_kind`, `assistant_id`, or `character_id` suppresses inheritance; global and parent/fork requests do not inherit. | Same resolver uses `model_fields_set`; `tldw_Server_API/tests/Workspaces/test_workspace_assistant_creation.py` verifies resolver and HTTP creation. |
| Workspace chat create does not use Sync v2. Its current `Idempotency-Key` does not provide deduplication outside the sync branch. | `_active_chat_sync_service` returns None for Workspace scope; `create_chat_session` otherwise generates a fresh UUID. |
| Current create responses/list/tree projections do not preserve selection provenance. | `_conversation_list_item_fields`; `chat_session_schemas.py`; `chat_conversation_schemas.py`; DB `chacha/conversation_store.py`. |
| Workspace operation receipts expire, permit key reuse, and are committed separately from conversation creation. | `ChaChaNotes_DB.py`: `create_workspace_operation`, `_delete_expired_workspace_operation_idempotency`. |
| Chatbook persists explicit-None separately, but its startup fallback/prompt behavior is not identical to the server. | Chatbook `Workspaces/registry_service.py`: set/clear; `Chat/console_assistant_defaults.py`: `resolve_new_console_assistant`. |

Baseline execution: `test_workspace_assistant_creation.py` **9 passed, 4 warnings**. Omission of assistant metadata in a Research RAG create payload is therefore not proof that the server stores no Persona. End-to-end RAG prompting still needs Stage 5 tests; no UI/runtime parity claim follows from this correction.

## Approaches

1. **Recommended: preserve legacy semantics and add a strict protocol.** Existing omission/null behavior stays; new clients can explicitly request `inherit` or `none` with replay guarantees. This adds a compatibility branch but avoids changing already-tested clients.
2. Replace implicit inheritance with opt-in only. Simpler eventual API, but breaks existing Workspace callers and contradicts current tests; requires a separately approved breaking transition.
3. Store only the opt-out bit and infer provenance from identity/default equality. Smallest storage change, but loses historical intent and cannot prevent retry rebinding; does not meet Stage 2.

## Durable Workspace Choice

Add `assistant_defaults_explicit_none: bool` to Workspace storage and the settings-authorized read model. It is not a directly writable request field. Existing PATCH commands determine it in the same version-checked write as defaults:

| Operation/state | Stored defaults | Explicit None |
| --- | --- | --- |
| Newly created Workspace, no selection | SQL NULL | false |
| Existing default omitted from PATCH | unchanged | unchanged |
| PATCH `assistant_defaults: null` | SQL NULL | true |
| PATCH valid Persona default | reference object | false |
| Legacy SQL NULL during migration | SQL NULL | true |
| Legacy non-null default during migration | unchanged, including corruption | false |

Legacy NULL means unknown intent, not proof that a person clicked Clear. Treating it as opted out is a conservative backfill safeguard; do not display it as a verified historical action. Clearing an already-empty Workspace still persists true and increments the version. Unrelated PUT/PATCH/rename/archive/restore operations preserve the bit. No public reset-to-unset command in this slice; a later reviewed provisioning command can replace the selection explicitly.

Never interpret malformed storage as consent. Corrupt non-null defaults remain `invalid_default`; a non-null default plus true is inconsistent and also unavailable, not a valid configured default. Repair requires an explicit owner write. Fresh and upgrade paths must match on SQLite/PostgreSQL and use the next registered migration at implementation time, not alter v49 or bypass catalog guards.

Clone/import paths must explicitly preserve a valid same-owner choice or conservatively mark the destination opted out if they cannot represent it. They must not create an apparently never-configured destination from a source that was deliberately cleared. This does not introduce cross-user Persona reference portability.

## Startup Selection Protocol

Existing `POST /api/v1/chats/` remains the endpoint. Add optional `workspace_assistant_selection: "inherit" | "none"` and `workspace_assistant_default_version: positive integer` to `ChatSessionCreate`.

```json
{
  "scope_type": "workspace",
  "workspace_id": "workspace-id",
  "workspace_assistant_selection": "inherit",
  "workspace_assistant_default_version": 7,
  "title": "Literature review"
}
```

New-protocol calls require a non-empty `Idempotency-Key` header, 1-128 ASCII characters matching `[A-Za-z0-9][A-Za-z0-9._:-]*`. Require Workspace scope, no parent/fork, and no supplied `assistant_kind`, `assistant_id`, `character_id`, or `persona_memory_mode`, including explicit null. Only `inherit` accepts/requires the version; it refers to the existing whole-Workspace version, not a new defaults-only counter. Unrelated Workspace changes can therefore cause a safe, refreshable conflict.

Reject ambiguous combinations with 422 before any write. The new protocol does not accept Character-only options: participants, prompt preset, per-Character memory, non-default provider/model/sampling overrides, or greeting seeding. Existing explicit Character/Persona requests keep their current route and behavior. No silent loss of requested options.

| Request | Resolution |
| --- | --- |
| No new selector, explicit tracked assistant | Existing explicit assistant wins. |
| No new selector, explicit null identity field | Existing explicit None wins. |
| No new selector, new Workspace chat and all identity fields omitted | Existing implicit inheritance remains, hardened to use the same effective resolver. No mandatory key/version is retrofitted. |
| No new selector, global or parent request | Existing global/fork semantics remain. |
| `inherit` with matching Workspace version and available Persona | Persist its reference and saved memory mode. |
| `inherit` with matching version and truly unset/explicitly cleared default | Persist an untracked assistant, source `system_fallback`; do not provision anything. |
| `inherit` with unavailable/invalid default | 409 `workspace_assistant_unavailable`, with a bounded degraded reason and no hidden identity. |
| `inherit` with disabled Persona support and a configured default | 503 `persona_feature_disabled`; explicit None remains possible. |
| `inherit` with stale version | 409 `workspace_assistant_version_conflict`, before insertion. |
| `none` | Persist an untracked assistant, source `explicit_none`; do not consult Persona availability. |

Missing/inaccessible Workspace remains 404. Archived Workspaces must not accept new strict-protocol chats (409); this does not change legacy archive behavior in this slice. Transient DB/quota failures remain mapped service errors, never a successful fallback. Legacy implicit inheritance also fails closed on corrupt/disabled configured defaults after sharing the Stage 1 resolver; legacy omission never starts inheriting in global/fork scope.

Explicit `inherit` is the caller's selection intent, not authority to write memory. Only a currently valid saved read-write default is inherited under the existing save confirmation gate; callers cannot override its mode in this protocol. Persona policy and per-turn memory admission remain in force.

## Provenance

Expose a read-only `assistant_startup` object on chat detail/list and conversation list/tree metadata. Store its bounded canonical JSON alongside conversation identity, with no Persona content or duplicate assistant id/memory mode:

```json
{
  "assistant_startup": {
    "schema_version": 1,
    "source": "workspace_default",
    "workspace_id": "workspace-id",
    "workspace_version": 7
  }
}
```

Source values: `workspace_default`, `explicit`, `explicit_none`, `system_fallback`, `fork`, `unknown`. `workspace_default` requires the originating Workspace id/version; `system_fallback` carries them only when the Workspace default was actually examined. Other sources have null Workspace id/version. Use a 1 KiB JSON size cap, fixed keys, and existing Workspace identifier validation. NULL/legacy provenance projects to schema_version 1/source unknown/null references. Never guess old origin by matching today's default or a stored assistant id.

Every new Workspace create path, including legacy omission/null/explicit calls, writes honest provenance atomically with identity. Parent-based create is `fork` and retains existing validated `parent_conversation_id`/`forked_from_message_id`; it does not copy parent provenance or newly implement parent assistant inheritance. Cross-Character forks remain supported by existing behavior. Global creation is unchanged and projects unknown unless a later contract records provenance there.

Provenance describes creation, not a permission grant or live default pointer. Default edits and metadata-only conversation edits do not change it. Existing identity-changing paths must invalidate it to unknown in the same write unless a trusted domain operation supplies a new defined origin; otherwise the origin could misleadingly describe a different assistant. Workspace moves preserve the stored origin. When a reader cannot access the originating Workspace, project the entire public object to schema_version 1/source unknown/null references without changing storage. This keeps the public `workspace_default` invariant valid and avoids exposing inaccessible references. Restore preserves the stored record; fork/import never copy it as server-verified default resolution. Shared-recipient adoption remains excluded.

Reject caller-supplied `assistant_startup` in chat create/update rather than trusting it or using it for resolution. The DB serializer accepts only internally constructed, validated provenance and never serializes request bodies, prompts, names, or policy snapshots. This is distinct from roleplay behavior snapshots, which are not modified here.

## Atomic Creation And Retry

Reuse the existing resolver module; do not introduce a parallel startup resolver. Separate effective-state calculation from HTTP projection so Workspace reads and startup use the same validation/feature/privacy rules. Selection for a first accepted creation must read Workspace version/default and visible active Persona under the transaction used for insertion. Merely reading the Workspace twice outside that transaction is not a fence.

For strict-protocol calls, add a small DB-owned `workspace_chat_startup_receipts` table: authenticated owner id, key digest, request fingerprint, accepted binding digest, originating Workspace id, created conversation reference (nullable on hard delete), creation timestamp, and nullable invalidated timestamp. Unique `(owner_id, key_digest)` within this operation-specific table; changing Workspace with the same key is a conflict, not a second create. The immutable binding digest covers the normalized stored assistant kind/id/character_id/memory mode, startup provenance, and conversation scope, not display metadata. Store neither raw key/request nor a response or Persona snapshot. Do not reuse `workspace_operations`' expiring, separately committed asynchronous lifecycle.

1. Authenticate, validate syntax/scope, and apply request rate limiting. Look up an accepted receipt before resolving today's defaults or enforcing new-chat count quota. Receipt access still requires current owner/Workspace/conversation access.
2. Fingerprint the validated original request before server-derived assistant/title fields are added. Include selector, Workspace/version, caller metadata, and effective query options; preserve omission-versus-null distinctions when they affect behavior. Exclude generated ids/timestamps and display names. Same key/different fingerprint returns 409 `idempotency_key_conflict` without revealing the earlier request.
3. An accepted matching receipt returns the same conversation with its original binding/provenance only while that binding is unchanged. It may expose current caller-edited title/metadata; it never regenerates them or re-resolves today's default. Return 200 with `Idempotency-Replayed: true`; first acceptance is 201. Revalidate current access and, in a consistent read, the receipt's invalidated timestamp and accepted binding digest against the conversation. A changed binding returns 409 `workspace_chat_startup_changed`, never restores the old assistant or creates another chat. Quota exhaustion after first acceptance does not block replay. Current Persona revocation/disable must not cause a new conversation or rebinding: reject with the mapped unavailable error if replay cannot safely expose/use the original identity.
4. Without a receipt, take a DB write transaction, recheck receipt, acquire Workspace/selected-Persona row locks, and recheck the receipt after every blocking acquisition before evaluating current defaults/version. A matching committed receipt wins over a now-stale default. Then create conversation plus provenance and receipt in that transaction. SQLite uses its existing write-transaction serialization; PostgreSQL needs row locks and uniqueness handling. A receipt uniqueness conflict rolls back the attempted insert and rereads the winning receipt in a fresh transaction through the same replay/conflict checks. Use the existing `add_conversation(..., conn=...)` rather than a second INSERT implementation. Keep SQL in DB_Management. No LLM/network call inside the transaction.
5. Concurrent identical keys produce one conversation. A crash before commit rolls back both records; a crash after commit is recovered by receipt. A clear/default edit that commits before selection yields a version conflict; after creation it cannot rewrite that chat's identity. Quota rejection commits neither conversation nor receipt.
6. Soft-deleted or hard-deleted accepted conversations return 410 `workspace_chat_deleted`, never recreate. Preserve an owner-scoped hashed receipt tombstone when the conversation is hard-deleted. No TTL/key recycling in this slice; owner account/database deletion removes receipts. This makes retention explicit and avoids a delayed retry silently selecting a new Persona.

All DB identity/memory-mode changes and scope transfers must irreversibly invalidate associated receipts in the same transaction, including sync materialization. Only actual changes to the normalized binding do so; metadata-only/no-op updates do not. Returning the assistant to its old value does not reactivate the key. Scope transfers preserve historical provenance but invalidate a receipt for creation in the old scope. Deletion takes precedence over changed-binding errors; restore may recover an unchanged soft-deleted conversation, but never clears an identity/scope invalidation. The binding digest is a defensive check for mutation paths, not a substitute for auditing and testing those paths. Read replay state consistently with concurrent mutation so no response combines a validated old digest with a newly edited identity.

Legacy calls without the new selector retain their existing retry behavior; do not advertise the new guarantee for them. The strict path excludes forks/Character factories and seeding to keep its transaction free of unrelated side effects. Extending replay guarantees to those flows is a separate reviewable decision, not an accidental expansion of this stage.

## Sync, Prompting, And Lifecycle Boundaries

- Workspace startup stays outside Sync v2. Do not turn on `_active_chat_sync_service` for Workspace scope or claim cross-app provenance sync. `assistant_startup` is server-owned and is not a new Sync v2 wire field in this slice.
- Existing sync materialization may update conversations through `upsert_conversation_from_sync`; metadata-only updates must preserve local provenance, identity changes must invalidate it to unknown, and remote payloads must not forge it. A new remotely materialized row starts unknown. Tests cover this even though Workspace create itself bypasses sync.
- Existing JSON/chatbook/export/import serializers must use explicit allowed fields: private receipts never export; provenance must not appear as trusted server authority on import. Preserve existing local metadata without guessing a Workspace/Persona mapping. Any transport unable to represent provenance reports/uses unknown, not a fabricated default source. Cross-server verified provenance portability is deferred.
- At send/preview/complete boundaries, a persisted Persona that is inactive/deleted/disabled cannot silently become a plain assistant. Audit both ordinary chat service and Persona-backed session completion helpers. This guard is needed independently of how the assistant was selected.
- Do not change prompt composition to imitate Chatbook. Chatbook custom prompts can bypass default inheritance; the server's chat-create request has no equivalent custom system-prompt field, and explicit request prompts are applied later in its own pipeline. Preserve tested server Persona boundaries/exemplars/memory precedence. Stage 2 does not certify Research Workspace RAG generation.

## Delivery Boundaries

After approval, turn these into focused implementation plans and independently tested PRs; do not implement the entire stage as a single unreviewed schema-and-runtime change.

| Slice | Deliverable | Principal files | Verification |
| --- | --- | --- | --- |
| 2A | Durable Workspace explicit None and migration invariants | `app/core/DB_Management/ChaChaNotes_DB.py`, `app/api/v1/schemas/workspace_schemas.py`, `app/api/v1/endpoints/workspaces.py`; existing defaults DB/API tests | Fresh/upgrade SQLite/PostgreSQL; legacy null; clear/set/omit; version conflict; corruption; clone/restore behavior. |
| 2B | Shared resolver and locally persisted creation provenance | Existing `app/core/Workspaces/assistant_defaults.py`, conversation store, create schemas/response builders; chat/default creation tests | Omitted/null/explicit/fork/global matrix, active/disabled/corrupt reads, old rows unknown, all response projections, metadata-only versus identity updates. |
| 2C | Strict versioned selection and transactional receipts | Existing create endpoint/resolver, DB-owned startup receipt helper, registered migration; new `tests/Workspaces/test_workspace_assistant_startup.py` | Exact replay/request conflict, binding mutation conflict, current auth, quota, delete, competing defaults/clear, concurrent keys, crash before/after commit; no duplicated side effects. |
| 2D | Lifecycle, sync-boundary, and send-time validation closure | Existing sync materializer/conversation store, ordinary and session chat Persona resolution, affected export/import serializers | Preserve/invalidate provenance correctly; no forged Sync/import provenance; inactive Persona cannot generate; prompt/memory regressions; existing global/Character flows unchanged. |

Paths in this table are relative to `tldw_Server_API/`. Confirm the current registered migration version and exact transport consumers before each implementation PR. 2B must inventory all callers of `add_conversation`, `update_conversation`, `upsert_conversation_from_sync`, list/tree projections, and scope-transfer paths; a shared row change cannot be judged from the create endpoint alone. Stage 2 completes only after 2D; 2A alone does not establish startup provenance/retry safety.

## Acceptance Matrix

- New Workspace unset, legacy null protected, clear persists through restart, rebind resets protection, unrelated mutation preserves it, corrupt records never become auto-provision candidates.
- Legacy omitted identity still inherits; each explicit-null identity field still opts out; explicit Persona/Character wins; parent/global requests bypass inheritance.
- Strict inherit with current version chooses read-only or previously confirmed read-write exactly; stale version conflicts; explicit none remains possible while Persona support is disabled.
- Original accepted identity/provenance survives default edits and retries while the conversation binding is unchanged; changed requests conflict. Assistant/memory-mode edits or scope moves invalidate replay without undoing the edit, even after a change back. Concurrent mutation cannot leak a mismatched replay response.
- Two processes using the same key produce one row. On PostgreSQL, a duplicate waiting for the Workspace lock replays a committed receipt even if a queued default edit advanced the version. Deleted targets never recreate; wrong owner/scope cannot replay them.
- Every owned response/list/tree/resume surface projects the same bounded provenance; inaccessible originating Workspaces produce unknown/null references while stored origin survives. Test the public schema after move/access loss as well as ordinary accessible `workspace_default`. No names/prompts/raw keys in provenance or logs.
- Fork uses validated lineage without inheriting today's defaults. Metadata updates/restore preserve provenance; identity mutation resets it; remote imports cannot forge authority.
- Current Persona revocation is checked before generation. Existing global sync and Character behavior snapshots remain unchanged. No claim that Workspace sync or RAG adoption was implemented.

## Review Gate

Approve or amend the recommended compatibility approach, the strict-protocol-only replay boundary, and local-only provenance before implementation. The implementation stage remains open; this design is not evidence of runtime completion. Every code slice still requires TDD, backend-specific migration/concurrency tests where applicable, Bandit, and the human-written PR Change summary gate.

## Validation Record

Independent source-backed design review found three contract issues: replay against mutable identity, a PostgreSQL receipt recheck race, and redaction that violated the public provenance shape. The proposal now defines irreversible receipt invalidation plus binding verification, post-lock receipt checks, and unknown/null public projection for inaccessible origins. Focused re-review found no remaining material issues at contract level; complete mutation-path and backend concurrency verification remains implementation work.

This document changes no runtime or test code. Existing startup tests passed **9 tests, 4 warnings**. The ordinary prompt/memory baseline (`test_persona_prompt_assembly.py` plus `integration/test_persona_backed_chat_conversations.py`) produced **13 passes, 9 failures, 6 warnings**. All nine failures return HTTP 503 `missing_provider_credentials` before the mocked provider call. The Persona integration fixture patches `chat.API_KEYS`, but credential resolution now obtains a frozen snapshot through `provider_credential_runtime.load_server_config_snapshot`.

A diagnostic-only rerun of those same two suites with an in-memory autouse fixture supplying the existing dummy OpenAI credential at the snapshot loader passed **22 tests, 6 warnings**. The provider transport remained mocked; no real provider call or credential was used. This isolates the fixture mismatch, not a production fix or a green unmodified suite. Working examples already patch this loader in `tests/Chat/integration/test_chat_endpoint_simplified.py`. Under TASK-13245, update the Persona integration fixture at that boundary and rerun the ordinary suites before accepting prompt/memory regression coverage for implementation.

Live PostgreSQL, Chatbook runtime tests, browser UAT, and the full repository suite were not run. Bandit is not applicable to this docs/Backlog-only change; it remains required for the later code slices.
