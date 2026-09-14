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

**Deployment prerequisite:** no mixed-version writers against a migrated database. Drain and stop old API processes, background workers, and direct DB writers before migration; migrate affected per-user databases and restart only compatible binaries before reopening writes. Existing cached DB handles only probe `SELECT 1`, so initialization-time schema checks do not fence an already-running older writer. Such a writer can save `(Persona, true)` or clear to `(NULL, false)` without maintaining the new bit. Use an offline maintenance upgrade for this slice, not an unverified rolling upgrade. Rehearse the old cached-handle hazard and the drain/migrate/restart procedure on both backends; if writer quiescence cannot be guaranteed, block rollout. Rollback must not reopen migrated data with old writers or discard the new choice/provenance/receipt data.

Clone/import paths must explicitly preserve a valid same-owner choice or conservatively mark the destination opted out if they cannot represent it. They must not create an apparently never-configured destination from a source that was deliberately cleared. This does not introduce cross-user Persona reference portability.

## Startup Selection Protocol

Keep `POST /api/v1/chats/` as the legacy endpoint. Put strict selection on a distinct `POST /api/v1/chats/workspace-startup` route, using a dedicated `WorkspaceChatStartupCreate` model with `extra="forbid"`, required `workspace_assistant_selection: "inherit" | "none"`, and conditional `workspace_assistant_default_version: StrictInt > 0`. Reuse common metadata validators and the existing resolver/DB creation operations, not the full Character creation schema or a second implementation of creation.

This route boundary is necessary: existing `ChatSessionCreate` silently ignores unknown fields, so a strict `none` payload sent to an older `/chats/` implementation would lose the selector and inherit a Persona. An older server has no strict route and must return 404/405 without creating a chat. Clients must never downgrade a failed strict request to legacy creation, including after timeouts. New servers reject misplaced strict fields on the legacy route with targeted validation; do not globally forbid every legacy extra field as an incidental breaking change. The strict route must be registered/tested against dynamic chat-id routes and remain unavailable until its lifecycle prerequisites below are complete.

```json
{
  "scope_type": "workspace",
  "workspace_id": "workspace-id",
  "workspace_assistant_selection": "inherit",
  "workspace_assistant_default_version": 7,
  "title": "Literature review"
}
```

Strict-route calls require a non-empty `Idempotency-Key` header, 1-128 ASCII characters matching `[A-Za-z0-9][A-Za-z0-9._:-]*`. Require Workspace scope, no parent/fork, and no supplied `assistant_kind`, `assistant_id`, `character_id`, or `persona_memory_mode`, including explicit null. Only `inherit` accepts/requires the version; it refers to the existing whole-Workspace version, not a new defaults-only counter. Unrelated Workspace changes can therefore cause a safe, refreshable conflict. Boolean/string versions are invalid rather than coerced.

Reject ambiguous combinations with 422 before any write. The dedicated model accepts only selection/scope plus existing `title`, `state`, `topic_label`, `cluster_id`, `source`, and `external_ref` metadata. Character-only fields, including default-valued provider/model/sampling fields, are not accepted. Greeting query parameters and unknown query options are rejected, not silently ignored. Existing explicit Character/Persona requests keep their current route and behavior. No silent loss of requested options.

| Request | Resolution |
| --- | --- |
| Legacy route, explicit tracked assistant | Existing explicit assistant wins. |
| Legacy route, explicit null identity field | Existing explicit None wins. |
| Legacy route, new Workspace chat and all identity fields omitted | Existing implicit inheritance remains, hardened to use the same effective resolver. No mandatory key/version is retrofitted. |
| Legacy route, global or parent request | Existing global/fork semantics remain. |
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

Bound permanent storage separately from live-chat quota: propose a finite `WORKSPACE_CHAT_STARTUP_RECEIPT_LIMIT_PER_USER` (default 10,000; positive integer) counting all accepted receipts, including invalidated/deleted tombstones across Workspaces. At capacity, reject only previously unseen keys with 409 `workspace_chat_receipt_capacity_exceeded`; exact replay and ordinary chat access remain available. No automatic eviction or key reuse. Owners can request an operator-configured higher budget; deleting chats does not recover this lifetime retry budget. Enforce the check and insertion under the same owner-scoped transaction serialization, not just a Workspace lock, so concurrent requests in different Workspaces cannot overshoot it. This is a count over the receipt table, not a second quota ledger. Use existing DB advisory-lock patterns on PostgreSQL and write serialization on SQLite. Acceptance tests use a small configured cap and explicitly exercise create/delete loops and concurrency.

1. Authenticate, validate syntax/scope, and apply request rate limiting. Look up an accepted receipt before resolving today's defaults or enforcing new-chat count quota. Receipt access still requires current owner/Workspace/conversation access.
2. Fingerprint the validated original request before server-derived assistant/title fields are added. Include selector, Workspace/version, caller metadata, and effective query options; preserve omission-versus-null distinctions when they affect behavior. Exclude generated ids/timestamps and display names. Same key/different fingerprint returns 409 `idempotency_key_conflict` without revealing the earlier request.
3. An accepted matching receipt returns the same conversation with its original binding/provenance only while that binding is unchanged. It may expose current caller-edited title/metadata; it never regenerates them or re-resolves today's default. Return 200 with `Idempotency-Replayed: true`; first acceptance is 201. Revalidate current access and, in a consistent read, the receipt's invalidated timestamp and accepted binding digest against the conversation. A changed binding returns 409 `workspace_chat_startup_changed`, never restores the old assistant or creates another chat. Quota exhaustion after first acceptance does not block replay. Current Persona revocation/disable must not cause a new conversation or rebinding: reject with the mapped unavailable error if replay cannot safely expose/use the original identity.
4. Without a receipt, take a DB write transaction, acquire owner-scoped receipt-admission serialization before Workspace/selected-Persona row locks, and recheck the receipt after every blocking acquisition before evaluating current defaults/version or capacity. A matching committed receipt wins over a now-stale default or exhausted capacity. Then create conversation plus provenance and receipt in that transaction. SQLite uses its existing write-transaction serialization; PostgreSQL needs advisory/row locks and uniqueness handling. A receipt uniqueness conflict rolls back the attempted insert and rereads the winning receipt in a fresh transaction through the same replay/conflict checks. Use the existing `add_conversation(..., conn=...)` rather than a second INSERT implementation. Keep SQL in DB_Management. No LLM/network call inside the transaction.
5. Concurrent identical keys produce one conversation. A crash before commit rolls back both records; a crash after commit is recovered by receipt. A clear/default edit that commits before selection yields a version conflict; after creation it cannot rewrite that chat's identity. Quota rejection commits neither conversation nor receipt.
6. Soft-deleted or hard-deleted accepted conversations return 410 `workspace_chat_deleted`, never recreate. Preserve an owner-scoped hashed receipt tombstone when the conversation is hard-deleted. No TTL/key recycling in this slice; owner account/database deletion removes receipts. This makes retention explicit and avoids a delayed retry silently selecting a new Persona.

All DB identity/memory-mode changes and scope transfers must irreversibly invalidate associated receipts in the same transaction, including sync materialization. Only actual changes to the normalized binding do so; metadata-only/no-op updates do not. Returning the assistant to its old value does not reactivate the key. Scope transfers preserve historical provenance but invalidate a receipt for creation in the old scope. Deletion takes precedence over changed-binding errors; restore may recover an unchanged soft-deleted conversation, but never clears an identity/scope invalidation. The binding digest is a defensive check for mutation paths, not a substitute for auditing and testing those paths. Read replay state consistently with concurrent mutation so no response combines a validated old digest with a newly edited identity.

Legacy-route calls retain their existing retry behavior; do not advertise the new guarantee for them. The strict path excludes forks/Character factories and seeding to keep its transaction free of unrelated side effects. Extending replay guarantees to those flows is a separate reviewable decision, not an accidental expansion of this stage.

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
| 2A | Durable Workspace explicit None and migration invariants | `app/core/DB_Management/ChaChaNotes_DB.py`, `app/api/v1/schemas/workspace_schemas.py`, `app/api/v1/endpoints/workspaces.py`; existing defaults DB/API tests | Fresh/upgrade SQLite/PostgreSQL; legacy null; clear/set/omit; version conflict; corruption; clone/restore; maintenance upgrade and old-writer exclusion. |
| 2B | Shared resolver, local provenance, and its lifecycle protections together | Existing `app/core/Workspaces/assistant_defaults.py`, conversation store, sync materializer, affected export/import serializers, create schemas/response builders; chat/default creation tests | Omitted/null/explicit/fork/global matrix; active/disabled/corrupt reads; old rows unknown; all projections; sync/direct metadata preservation and identity invalidation; import authority. |
| 2C | Dedicated strict-startup route, transactional receipts, and their lifecycle protections together | Existing chat router/resolver, dedicated request schema, DB-owned startup receipt helper and all binding/scope/delete writers, registered migration, ordinary/session Persona admission; new `tests/Workspaces/test_workspace_assistant_startup.py` | Old-server fail-closed/no downgrade; replay/conflict/capacity; current auth; quota; delete/scope/identity invalidation; competing defaults; concurrent keys/capacity; crash recovery; inactive Persona cannot generate. |
| 2D | Cross-surface regression and operational verification closure | Existing sync/transport tests, ordinary and session chat tests, migration/deployment runbook | Verify safeguards already delivered in 2A-2C; complete prompt/memory and lifecycle matrix; existing global/Character flows unchanged; no first activation of required safety mechanisms here. |

Paths in this table are relative to `tldw_Server_API/`. Confirm the current registered migration version and exact transport consumers before each implementation PR. 2B must inventory all callers of `add_conversation`, `update_conversation`, `upsert_conversation_from_sync`, list/tree projections, and scope-transfer paths; a shared row change cannot be judged from the create endpoint alone. No 2B provenance writes/projection may activate before sync/direct mutation and import protections pass; no 2C strict route may activate before receipt invalidation/deletion/capacity and send-time admission pass. If a slice is split into smaller PRs, keep its behavior inactive until all prerequisites land. Stage 2 completes only after 2D; 2A alone does not establish startup provenance/retry safety.

## Acceptance Matrix

- New Workspace unset, legacy null protected, clear persists through restart, rebind resets protection, unrelated mutation preserves it, corrupt records never become auto-provision candidates.
- Legacy omitted identity still inherits; each explicit-null identity field still opts out; explicit Persona/Character wins; parent/global requests bypass inheritance.
- Old servers reject the distinct strict route without writes; clients never fall back to legacy creation after rejection or an ambiguous timeout. Strict request fields on the new server's legacy route are rejected. Strict unknown fields/query options and coerced version types are rejected.
- Strict inherit with current version chooses read-only or previously confirmed read-write exactly; stale version conflicts; explicit none remains possible while Persona support is disabled.
- Original accepted identity/provenance survives default edits and retries while the conversation binding is unchanged; changed requests conflict. Assistant/memory-mode edits or scope moves invalidate replay without undoing the edit, even after a change back. Concurrent mutation cannot leak a mismatched replay response.
- Two processes using the same key produce one row. On PostgreSQL, a duplicate waiting for owner-scoped admission replays a committed receipt despite subsequent default/version changes or exhausted capacity. Keep the general receipt recheck after blocking acquisitions. Deleted targets never recreate; wrong owner/scope cannot replay them.
- Receipt capacity includes tombstones and spans Workspaces; concurrent fresh keys cannot exceed it, while accepted replay still works at capacity. Draining old writers is required before each incompatible storage rollout; required lifecycle safeguards activate with their dependent features, not afterward.
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

### Requester-Requested Follow-Up Review

Reviewed the published design at `5cdeed0f34175cc25fb4a645ca4acaddf44f3e1c` against the same backend implementation. Four additional design issues were verified and amended before runtime work:

| Priority | Finding and evidence | Design correction |
| --- | --- | --- |
| P1 | `ChatSessionCreate` ignores extra fields; `resolve_new_conversation_assistant` then inherits when identity is omitted. A schema/resolver probe with a stub DB reproduced `workspace_assistant_selection: none` becoming a Persona. | Distinct strict route and extra-forbid model; no downgrade to legacy creation. |
| P1 | `update_workspace` only writes `assistant_defaults_json`; cached dependencies in `ChaCha_Notes_DB_Deps.py` check liveness, not current schema compatibility. An already-open old writer can violate the migrated choice pair. | Drained offline maintenance upgrade, compatible writers only, explicit rollback limits. |
| P2 | `upsert_conversation_from_sync` overwrites identity/scope independently of `update_conversation`; delaying its protections until 2D would leave newly recorded provenance stale. | Deliver lifecycle/import safeguards with 2B/2C; 2D is verification, not their first activation. |
| P2 | `count_conversations_for_user` excludes deleted chats by default; it does not bound permanent receipts across create/delete cycles. | Owner-scoped receipt budget including all tombstones, atomically admitted across Workspaces; accepted replay bypasses capacity rejection. |

Fresh follow-up validation: the read-only schema/resolver probe passed its assertions with no DB/provider calls; three JSON examples and five relative document links validated; `git diff --check` clean. No runtime, test, or configuration files changed. The ordinary-suite failures recorded above remain an implementation prerequisite, not reclassified as passing by this review.

Independent focused re-review found no remaining material contract issues. Its minor acceptance-matrix correction now reflects owner-scoped admission rather than waiting first at a Workspace lock. Migration quiescence, route dispatch, concurrent capacity admission, and complete mutation coverage still require implementation verification.
