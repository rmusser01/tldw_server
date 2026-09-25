# H2 Native Fork Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create an independently usable native child with a single atomic commit and durable same-operation recovery in both full-page clients.

**Architecture:** Keep ownership, operation receipts, selected graph, accepted settings/snapshot and attachment claims in ChaCha. Prepare immutable bytes through existing ingestion/storage machinery before a short owner transaction; compose future character requests from accepted frozen data. Use the shared client controller and retain the original owner after uncertain outcomes.

**Tech Stack:** Python/FastAPI/Pydantic, repository SQLite/PostgreSQL abstractions, existing verified local blob store/upload sink, shared React/TypeScript/Dexie, Vitest and Playwright. No new runtime dependency.

**Spec:** [H2 native fork design](Docs/Design/2026-09-18-chatbook-h2-native-fork-design.md). Read the [source audit](Docs/Reviews/CHATBOOK_H2_NATIVE_FORK_SOURCE_AUDIT_2026_09_18.md) and its asset/character/retention evidence before implementation.

## Global Constraints

- Full parity belongs in the WebUI and extension full-page UI; the compact sidepanel expands inside the extension.
- H2 covers native ownership. H3, H4, F02 and unsupported persona/multi-participant/comparison behavior remain separately named gates. No tldw-agent integration belongs here.
- Same accepted operation key never becomes a fresh create, including after source/child deletion or receipt compaction. No owner-changing fallback after dispatch.
- Stage external bytes before the native transaction. No network, filesystem transfer, provider, ingestion or live behavior lookup inside the commit transaction.
- Use existing advertised destination ingestion/storage limits; strict asset fidelity is default. Explicit degraded review changes the digest and blocks required-asset sends until resolved.
- Canonical settings and the accepted child snapshot remain authoritative after cold reopen and edits; no live-card creation, greeting seeding or current preset/worldbook lookup.
- The original H1 integrated baseline was `c13bad37fefd85ad15e3d0e9145db308a2a4723e` on server dev `91e8bbf84c25d3afbba2bb53ed06280d44c35307`; the current draft stack is rebased on server dev `91c32e3126baad20aba2b74399927bd63004a5f7` through H1 `a7a89a80db`. The H2 source comparison was originally pinned to Chatbook `4d3e2d380e6ebb7a8e465d2d90a7564a82e7f6ef`; refresh it before later client implementation. H1's ChaCha schema is SQLite v69 / PostgreSQL v73; the first Task 1.2 storage increment installs v70/v74 after rechecking server dev. H1 repairs old H1-v68 databases by catalog evidence. The [increment review](Docs/Reviews/CHATBOOK_H2_NATIVE_STORE_INCREMENT_REVIEW_2026_09_23.md) names the remaining unqualified Task 1.2 interfaces.
- Preserve current-dev strict ordered image reads and detail, saved-turn retry, and PostgreSQL operation-scoped RLS. H2 native multi-image cold reopen needs exact representation-aware admission; the general loader refusal remains until that path is qualified. Agent-created routed chats are separate TASK-13261.6 work, not ordinary H2 native fork.
- Work only in this isolated worktree on `codex/chatbook-h2-native-fork`, stacked on H1 PR2968. Do not modify main/UAT, its processes/data/ports or the untracked `.next-live-tier-h1-production/` evidence directory.
- TASK-13261.2 tracks specification only. Before implementation edits, search/create linked Backlog tasks for the reviewable stages through official MCP/CLI, then record results/commits there. Read full applicable frontend/extension AGENTS.md before those edits.
- Each task follows red → green → focused regression → independent review → commit. No `--no-verify`. Run Bandit on touched Python with the project venv; do not expand unrelated testing after adequate proof.

## Interface map

New files are proposals, not existing implementations. Existing source references use the H1 baseline. Python prefixes below are `tldw_Server_API/app/`; UI prefixes are `apps/packages/ui/src/`.

| File | Responsibility and exported interface |
|---|---|
| Create `api/v1/schemas/native_fork_schemas.py` | Strict wire types from spec §3: `NativeScopeV1`, `NativeAssetManifestEntryV1`, `NativeForkCaptureRequestV1`, `NativeForkCaptureV1`, `NativeForkRequestV1`, `NativeForkResultV1`, `NativeForkResolveRequestV1`, `NativeAssetRetentionRequestV1`, `NativeAssetRetentionResultV1`, `NativeAssetContextUpdateV1`, capabilities and binding descriptor. |
| Create `core/Chat/native_fork_projection.py` | `native_fork_request_digest(request) -> str`; `project_native_fork_context(state, selected_rows, asset_manifest) -> ProjectedNativeForkContext`; `capture_native_fork(db, owner, capture_request) -> NativeForkCaptureV1`. Fork-purpose coherent projection only; H1 send validation unchanged. |
| Create `core/DB_Management/chacha/native_fork_store.py` | `NativeForkStore.reserve_operation`, `claim_attempt`, `resolve_operation`, `commit_fork`, `mark_child_gone`; all accept the same authenticated owner and caller connection where atomic composition is needed. All SQL stays here or existing DB stores. |
| Create `core/DB_Management/chacha/native_chat_asset_store.py` | Candidate/claim/ref CAS, quota-intent keys and exact owner/revision reads; transactional retention and release, cleanup eligibility. Quota authority is the existing service, not this table. No byte I/O. |
| Create `core/Chat/native_chat_assets.py` | `NativeChatAssetService.prepare_retention`, `prepare_fork_assets`, `reconcile_candidates`, `read_owned_asset`; reuses upload sink/validator and native-root byte adapter. |
| Create `core/Chat/frozen_history_behavior.py` | `build_frozen_history_behavior(context) -> FrozenHistoryBehaviorPlan`; `compose_frozen_history_behavior(plan, selected_messages, current_inputs) -> FrozenHistoryPromptResult`; pure supported effect validation/composition. |
| Create `core/Chat/native_fork.py` | `NativeForkService.capture`, `execute`, `resolve`, `retain_assets`, `resolve_retention`; orchestrates storage/preparation with injected owner and existing DB/asset dependencies. No second HTTP copier. |
| Create `api/v1/endpoints/native_forks.py` | Thin authenticated/scoped/rate-limited routes from spec §3; register under chat prefix without growing the existing huge chat.py module. |
| Create UI `types/native-fork.ts`, `services/native-fork.ts` | Wire types, tuple digest and typed client protocol/recovery; exact endpoint wrappers remain in existing chat-rag/TldwApiClient modules. |
| Modify UI existing history/operation/controller/loader files | Route capable native owners to atomic service, preserve legacy pending records, keep owner leases, render native binding/assets and recovery. |

Definitions shared by tasks (implement under the named schema/projection/behavior modules, not duplicated per task):

```python
# Immutable service records; endpoint dependencies construct Owner from auth,
# never by trusting the request's owner_key/client_id.
@dataclass(frozen=True)
class AuthorizedNativeOwner:
    client_id: str
    owner_key: str
    scope: NativeScopeV1

@dataclass(frozen=True)
class ProjectedNativeForkContext:
    settings_json: str
    snapshot_schema: str | None
    snapshot_json: str | None
    snapshot_digest: str | None
    binding: NativeForkBindingTemplateV1 | None
    retained_context_digest: str
    required_effects: tuple[str, ...]

@dataclass(frozen=True)
class NativeAttempt:
    operation_kind: str
    operation_id: str
    generation: int
    request_digest: str

@dataclass(frozen=True)
class FrozenHistoryBehaviorPlan:
    child_id: str
    settings_revision: int
    history_revision: int  # captured after current input append in admission
    identity_json: str
    behavior_json: str  # validated supported frozen effects and their policies
    parameters_json: str

@dataclass(frozen=True)
class NativeChildSummaryUpdate:
    child_id: str
    expected_settings_revision: int
    expected_history_revision: int
    summary_json: str

@dataclass(frozen=True)
class FrozenHistoryPromptResult:
    messages: tuple[Mapping[str, Any], ...]  # recursively frozen via H1 helper
    parameters_json: str
    summary_update: NativeChildSummaryUpdate | None
```

`NativeForkBindingTemplateV1` is the pure projector's validated mode/accepted primary participant/display-name/snapshot digest template; it has no child ID because allocation has not happened. Task3.1 binds it to the newly allocated child ID and produces read-only `NativeAssistantBindingV1`. `state` and `selected_rows` in the pure projector are the coherent detached rows from existing snapshot/resume reads; their adapter must distinguish absent from unreadable/corrupt settings. The generated wire types carry no arbitrary Python object. `NativeForkResultV1` is a discriminated union: committed (`child_id`, immutable `message_map`, projection), pending, not_recorded, rejected, expired, gone; terminal noncommitted variants carry a bounded code, no candidate child masquerading as committed. Reuse the existing H1 `state` discriminator and committed-result names exactly. HTTP authentication/authorization failures remain ordinary denied responses. Retention results carry a typed immutable asset map instead of a child map; `NativeOperationResultV1` is the union of fork and retention results, tagged by operation kind.

## Stage 1: Native operation and projection contracts

**Goal:** A testable storage contract with immutable request identity, deletion-independent receipts and coherent fork-purpose projection.
**Success Criteria:** H2-A1 and storage portion of H2-A2/A3/A8 pass on both database engines; no public route is enabled yet.
**Tests:** canonical fixture/property tests, schema migration/RLS, same-key race/replay, deletion/expiry, and retained-vs-excluded source changes.
**Status:** In Progress

### Task 1.1: Strict wire contract and retained projection

**Files:** Create schema/projection files in the interface map; create `tldw_Server_API/tests/Chat/unit/test_native_fork_projection.py`, `tldw_Server_API/tests/fixtures/native_fork_v1.json`; modify `core/DB_Management/chacha/message_store.py` only for a fork-purpose coherent read adapter. Existing graph resolvers remain in `core/Chat/history_selection.py`.

**Interfaces:** Produces the wire types, `AuthorizedNativeOwner`, `ProjectedNativeForkContext`, canonical tuple function and capture/projector; consumes H1 snapshot/graph resolver and canonical accepted behavior builders.

- [x] Write fixtures for Unicode, empty cursor, explicit legacy projection, changed title/asset/fidelity, active rows, unknown carrier and no-op/raw settings formatting. Add property checks for deterministic digest and changed semantic tuple members. The retained projection excludes cached summary in public and materialized controls while retaining declarative policy.

```python
def test_summary_content_changes_do_not_change_fork_projection(accepted_state):
    before = project_native_fork_context(accepted_state, (), ())
    changed = accepted_state.with_cached_summary("new cached text")
    after = project_native_fork_context(changed, (), ())
    assert after.retained_context_digest == before.retained_context_digest

def test_retained_worldbook_content_changes_fork_projection(accepted_state):
    before = project_native_fork_context(accepted_state, (), ())
    changed = accepted_state.with_accepted_worldbook_text("different lore")
    after = project_native_fork_context(changed, (), ())
    assert after.retained_context_digest != before.retained_context_digest
```

`accepted_state` is a new test-only immutable builder in `tests/Chat/unit/test_native_fork_projection.py`, seeded with the existing canonical snapshot/materialized-envelope builders. Its two shown methods return validated new detached states without touching live card storage. Include a corrupt required envelope and a neutral absent snapshot as separate cases.

- [x] Run `python -m pytest -o addopts='' tldw_Server_API/tests/Chat/unit/test_native_fork_projection.py -q` after activating the project venv. Observe behavioral/import failure before implementation.
- [x] Implement the fixed tuple from spec §3 and strict wire validators; project settings, replay metadata, image/generated-event classification and references by allowlist. Re-resolve the full selected graph through H1; require settled retained nodes, preserve supported stopped/failed text, and reject incomplete tool replay groups. Use canonical accepted snapshot/envelope builders when exclusions change nested bytes.

```python
def native_fork_request_digest(request: NativeForkRequestV1) -> str:
    encoded = json.dumps(native_fork_request_tuple(request), ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
```

`native_fork_request_tuple` is defined in this task with the exact ordered members in spec §3; it does not hash an arbitrary request dictionary. Capture and commit must use this same projector; do not rename H1's raw storage digest into a fork digest.

- [x] Re-run focused tests and existing `test_history_selection.py`/`test_history_context.py`; review exclusions against D2. Commit the tested contract and fixture as `feat(chat): define native fork retained projection`.

### Task 1.2: Migration, receipt state and deletion

**Refreshed baseline brief:** [2026-09-23 Task 1.2 brief](Docs/Design/2026-09-23-chatbook-h2-task-1-2-brief.md).

**Current checkpoint:** Operation, candidate, deletion/restore, Sync/identity and workspace admission guards are incremental and reviewed; the [workspace admission review](Docs/Reviews/CHATBOOK_H2_WORKSPACE_ADMISSION_INCREMENT_2026_09_23.md) records the admission boundary. The [candidate reclamation review](Docs/Reviews/CHATBOOK_H2_CANDIDATE_RECLAMATION_INCREMENT_2026_09_23.md) adds database-only cleanup fencing. Final child commit, physical byte/claim/quota lifecycle and remaining admin paths are still open.

**Files:** Create native_fork_store and asset_store from interface map; modify `core/DB_Management/ChaChaNotes_DB.py`, `core/DB_Management/backends/pg_rls_policies.py`, `core/DB_Management/chacha/conversation_store.py`. Create `tests/DB_Management/test_native_fork_migration.py`, `test_native_fork_transactions.py`, `test_native_fork_tenancy.py` under `tldw_Server_API/`. Preserve existing concurrency assertions while adapting the deletion barriers in `tests/ChaChaNotesDB/test_workspace_source_saved_views_db.py` and `test_workspace_source_saved_views_postgres.py` to the new outermost-only staged deletion contract.

**Interfaces:** Produces `NativeAttempt` and owner store methods:

```text
reserve_operation(owner, kind, operation_id, request_digest, canonical_request, *, conn)
claim_attempt(owner, kind, operation_id, request_digest, *, now, conn) -> NativeAttempt | NativeForkResultV1
resolve_operation(owner, kind, operation_id, request_digest, *, conn) -> NativeOperationResultV1
mark_child_gone(owner, child_id, *, conn) -> None
```

All methods validate stored owner/scope/digest; application SQL is confined to DB_Management. Store methods do not call network/byte APIs. Create candidate/claim/ref and quota-intent schema now because its atomic FK/ownership lifecycle belongs to this migration; byte preparation remains Task 2.1 and aggregate accounting Task 2.2.

- [ ] Use the real SQLite/PostgreSQL `history_db` fixture pattern in `test_history_selection_transactions.py:19`, backed by repository `pg_database_config`, as the basis of a new `native_fork_db` fixture. Do not depend on the H1 test's default character card for neutral fixtures. Write upgrade/reopen/repeated-init tests and transaction fixtures for committed map, missing source, child soft/hard/bulk delete, compaction and wrong account. Assert receipt remains after source/child cascades; actual app-role RLS cannot enumerate another user's operation.

```python
def test_deleted_child_operation_never_becomes_fresh(native_fork_db, committed_fork):
    db, owner = native_fork_db
    db.hard_delete_conversation(committed_fork.child_id)
    with db.transaction() as conn:
        result = db.native_forks.resolve_operation(
            owner, "native_fork_v1", committed_fork.operation_id,
            committed_fork.request_digest, conn=conn)
    assert result.state == "gone"
```

`committed_fork` is a test fixture using the store to insert an owned minimal ready child and receipt in one transaction; it exposes the three immutable fields in this test. `hard_delete_conversation` is the existing store entry point at `conversation_store.py:1389`.

- [ ] Run the three new storage suites red. Add SQLite migration v70 after integrated H1 v69 and PostgreSQL migration v74 after integrated H1 v73 (recheck current remote versions before editing), tables with direct-owner RLS and no cascading receipt FK, protected conversation binding/projection columns, and index child creation-operation mapping for deletion. Candidate states are reserved/prepared/adopted/reclaiming/discarded; claims are live/released. Store durable quota intent keys at transitions; external accounting is outside this DB transaction. Cover clean installs, current-dev lineage, previously initialized H1-v68 lineage and repeated initialization before writing native rows.
- [ ] Implement locked reservation/attempt generation and committed-first resolution. Creation/delete lock operations before conversations; bulk sort keys/IDs; old message rows remain unlocked. Deletion paths, including bulk/export-import replacement/admin APIs and public restore, retain permanent gone/expired keys. Soft delete keeps claims/bytes charged and restorable; hard purge releases them. Test cleanup between soft delete and restore for exact bytes, and hard-purge cleanup separately. Conservatively reconcile a missing child on lookup. Do not prune accepted keys.
- [ ] Implement W1 from the [workspace lifecycle review](Docs/Reviews/CHATBOOK_H2_NATIVE_WORKSPACE_LIFECYCLE_REVIEW_2026_09_18.md): migrate protected `workspaces.native_chat_admission_closed` (default false), and provide the owner-bound workspace admission guard for final native writes. Native order is operation → workspace → conversation → assets. Begin-close locks only workspace, validates ordinary owner/current expected version for soft deletion, closes admission without bumping public version, and commits before enumeration. Preserve hard purge's versionless signature and absent-row idempotence. Both staged deletion methods reject an enclosing transaction before any mutation, without force commits or a second connection.
- [ ] Keep closure after partial/cascade/CAS failure and permit retry against the current public workspace version with fresh enumeration. Final workspace transition checks for residual protected rows (no live protected row for soft delete; none at all for hard purge), aborting/releasing before any per-operation repair. Closed scope denies new native admission/context/restore; destructive cleanup stays allowed. Do not mark a still-existing child's receipt gone merely because scope is closed. Protect native scope/identity/deleted state and closure from generic import/Sync/upsert bypass; no workspace reopening API is added.
- [ ] Add real SQLite/PostgreSQL deterministic barriers for admission-before-close, close-before-prepared-admission, partial failure/current-version retry, stale/wrong-owner close, residual protected rows, same-key resolve during closure and after actual deletion, and nested transaction rejection preserving unrelated caller work. Move the two existing saved-view delete-first barriers inside the short final workspace transition after its deleted update and before commit; retain their waiting, `source_view_not_found` and empty-metadata assertions. Mutation-first barriers remain. Task2.3/4.1 add actual retention/context and HTTP/fork integration over this store protocol; no filesystem/quota work enters these locks.
- [ ] Run migration/RLS/transaction suites green, plus H1 transaction/migration tests. Check failed update rollback on both engines and deterministic concurrent same-key reservation. Run touched Python format/lint/Bandit, independent review and commit `feat(chat): persist native fork operations and tombstones`.

## Stage 2: Native attachment retention and lifetime

**Goal:** Typed native attachments have exact bytes, independent ownership and safe preparation/reclamation without cross-store adoption authority.
**Success Criteria:** H2-A4/A6 storage proofs pass, including explicit retention; no Media ID or upload draft can masquerade as a native claim.
**Tests:** byte/hash/MIME/limit validation, original-vs-extracted text, selected revision, retention replay, source/child deletion, quota concurrency and GC/adoption races.
**Status:** Not Started

### Task 2.1: Native byte adapter, candidates and cleanup

**Files:** Create `core/Chat/native_chat_assets.py`; extend asset_store; create `tests/Chat/unit/test_native_chat_assets.py` and `tests/DB_Management/test_native_chat_asset_transactions.py`. Reuse `core/Sync/v2/blob_store.py`, `core/Infrastructure/distributed_lock.py` and `core/Ingestion_Media_Processing/input_sourcing.py`. Extend the secure blob-store lock helper into a reusable native guard only with tests; do not rely on its existing GC-only lock to fence publishers.

**Interfaces:** `prepare_fork_assets(owner, attempt, manifest) -> tuple[PreparedNativeAsset, ...]`; `reconcile_candidates(owner, operation_id) -> CandidateReconciliation`; `PreparedNativeAsset` contains candidate UUID, attempt generation, source native ID/revision, namespace/hash/size/MIME/representation. `CandidateReconciliation` contains counts of retained/reclaiming/removed records and retryable errors, never byte contents or source paths.

- [ ] Write tests using a real temporary byte root and repository DB fixture: reserve-before-write crash, publish-before-prepared crash, hash/size/MIME mismatch, symlink/path swap, abandoned generation, and deterministic GC/adoption barrier. Pause before chunk publish and final rename, fence generation, start cleanup, resume writer: after cleanup no final bytes/chunks can appear. Reuse byte-store security fixtures; add native ownership assertions instead of rerunning unchanged Sync service tests as native proof. Quota concurrency is Task 2.2.

```python
def test_reclaiming_candidate_cannot_be_adopted(native_asset_case):
    candidate = native_asset_case.prepare_exact_image()
    native_asset_case.mark_reclaiming(candidate)
    with pytest.raises(NativeAssetConflict, match="candidate_not_adoptable"):
        native_asset_case.adopt(candidate)
```

Define `native_asset_case` in the new transaction suite as a test adapter over real native stores/blob root; its methods perform actual reserve/prepare/CAS/adoption, not canned outcomes. `NativeAssetConflict` is the typed native asset error in `native_chat_assets.py`, mapped to a bounded 409 code.

- [ ] Run both suites red, then implement reserve → guarded verified chunk write/commit → prepared CAS using independent native namespace. All physical writers take stable `candidate_io_guard(candidate_id)` outside the deletable directories and recheck durable generation/state before writing. Adoption locks candidates and installs live claims in caller transaction. Cleanup commits non-adoptable state first, then waits on the same guard outside its DB transaction; no DB lock is held while waiting. Keep uncertain outcomes; failed unlink retries from durable state. No cross-conversation physical deduplication.

```python
def reclaim_candidate(store, adapter, owner, candidate_id):
    candidate = store.claim_reclamation(owner, candidate_id)
    if candidate is None:
        return False
    with adapter.candidate_io_guard(candidate_id):
        adapter.blobs.delete_namespace_blob(
            storage_key=candidate.storage_key,
            storage_namespace_id=candidate.namespace,
            payload_hash=candidate.content_hash,
            expected_size=candidate.size_bytes)
        adapter.discard_upload_verified(candidate.upload_id)
        store.finish_reclamation(owner, candidate_id, candidate.generation)
    return True
```

Implement `claim_reclamation`/`finish_reclamation` in asset_store. `candidate_io_guard` is the new native adapter wrapper over secure stable lock-file/platform primitives. `discard_upload_verified` calls existing `discard_upload(upload_id)` then verifies the exact upload directory is absent: that existing helper suppresses removal errors, so a silent return alone cannot release quota or mark cleanup finished. If either removal/verification fails, leave reclaiming state durable and surface a retryable error. Adopted assets use the same release/reclamation machinery after hard purge/final claim release; soft deletion does not release them.

- [ ] Run green plus selected existing multi-image, original-retirement and byte-store cases if their shared helper changed. Review GC/quota races and commit `feat(chat): own native attachment bytes and cleanup`.

### Task 2.2: Existing quota authority and idempotent native charges

**Files:** Modify `services/storage_quota_service.py` (under `tldw_Server_API/app/`), existing AuthNZ migration registration in `core/AuthNZ/migrations.py`/`pg_migrations_extra.py`, and `core/AuthNZ/repos/storage_quotas_repo.py` for the service adapter. Create SQL-owning `core/DB_Management/storage_quota_reservation_store.py` and `tldw_Server_API/tests/Storage/test_native_chat_quota_reservations.py`; extend `tests/Storage/test_storage_quota_service.py` and native asset crash tests.

**Interfaces:** `reserve_native_bytes(owner, candidate_key, bytes_count) -> NativeQuotaReservation`, `resolve_native_reservation(owner, candidate_key)`, `release_native_bytes(owner, candidate_key)` on existing service. `NativeQuotaReservation` contains immutable authenticated user/applicable team/org, exact candidate key/byte count and active/released state. Reservation is idempotent; mismatched bytes/scope conflicts. No new native allowance is configured.

- [ ] Test two native reservations racing for the same remaining budget; only one fits. Prime the legacy quota cache, reserve native bytes, and assert existing user/combined checks see the native charge. Test reservation response loss/replay, unlink then release response loss/replay, adopted and soft-deleted charge persistence. A cache-prime → reserve → publish → recalculate → reserve/release sequence must produce identical total usage before/after recalculation for user and applicable team/org pools, with no double charge. Use existing AuthNZ SQLite/PostgreSQL fixtures. Include a held pre-admitted legacy upload race and report the inherited check-then-write semantics honestly; H2 does not promise a new global hard cap.

```python
async def test_quota_reservation_replay_does_not_double_charge(quota_case):
    await quota_case.reserve("candidate-a", 8 * 1024 * 1024)
    await quota_case.reserve("candidate-a", 8 * 1024 * 1024)
    assert await quota_case.native_reserved_bytes() == 8 * 1024 * 1024
```

`quota_case` wraps the production quota service and real AuthNZ fixture, with a 10 MiB remaining budget; no fake counters. It exposes reserve and authoritative native ledger query for assertions.

- [ ] Run red; implement operation/candidate-keyed ledger in the existing quota authority. Native reserve locks applicable quota owner rows in stable user/team/org order, reads uncached existing usage plus active native charges and inserts the unique allocation in that transaction. Keep stored/cached legacy component L native-exclusive; all admission/public usage summaries compute L+fresh N once from the active native ledger. `calculate_user_storage` excludes the managed native root, stores only L and returns aggregate L+N; it never stores the aggregate into L. Native bytes never also call `update_usage`. Apply the split to shared pools and existing public usage serializers so consumers do not show raw L as total. Keep embedded DB image/other legacy accounting semantics unchanged and explicit.
- [ ] Persist ChaCha quota intent before external reservation. Reconcile identical key before writing if response is unknown. Native commit retains charge; guarded physical cleanup produces durable release intent, and service release is idempotent. A crash cannot grant a second reservation or release live bytes. No AuthNZ/network call in the chat commit. Scope-derived team/org limits and identity are checked by existing services; request fields cannot choose a cheaper scope.
- [ ] Run service/native asset crash suites green, review the cross-store accounting boundary and commit `feat(storage): account for native chat asset reservations`.

### Task 2.3: Explicit representation-aware retention and read

**Files:** Extend native_chat_assets/store/schemas; create route/service retention handlers under native_forks/native_fork from interface map; create `tests/Chat_NEW/integration/test_native_asset_retention_api.py`. Client workflow is Task 4.2. Existing upload seams: `api/v1/endpoints/media/process_documents.py:176`, `core/Ingestion_Media_Processing/input_sourcing.py:358`, `document_upload_preflight.py`.

**Interfaces:** `prepare_retention(owner, attempt, request: NativeAssetRetentionRequestV1, validated_uploads) -> tuple[PreparedNativeAsset, ...]`; `retain_assets(owner, request, validated_uploads) -> NativeAssetRetentionResultV1`; `read_owned_asset(owner, conversation_id, asset_id, revision)` returns authorized descriptor/byte stream. Retention request binds source native attachment fence and selected revision plus per-file actual byte hash/size/MIME/representation; it cannot contain executable Media/FileArtifact IDs.

Also produces `update_native_asset_context(owner, conversation_id, request: NativeAssetContextUpdateV1) -> NativeAssetContextV1`, `read_native_asset_context(owner, conversation_id) -> NativeAssetContextV1` and the exact scoped PATCH/GET collection routes from spec §3. `NativeAssetContextV1` is the canonical ordered reference manifest plus `asset_manifest_revision`. The strict update carries expected manifest revision, exact target reference ID, expected reference/asset revisions and action; replacement/restoration additionally requires a live claim/revision in the same conversation. All mutations lock conversation before references and return revised canonical context. Removing one of two refs to the same claim changes only the chosen context ref; bytes and the other ref remain.

For workspace scope, use Task1.2's open-admission guard before the conversation lock: retention locks operation → workspace → conversation → claims, and context updates lock workspace → conversation → references. A prepared upload losing to closure cannot commit new claims; its same-key cleanup remains safe. Test actual retention/context calls held across begin-close and preserve committed live charges until actual purge.

- [ ] Write actual HTTP multipart cases for binary original, extracted text with truthful MIME, chosen generated revision and missing preview. Use ordinary upload fixtures and mocked external download only at the user-side boundary. Hold upload while chat/owner/selected revision changes: no stale attachment. Drop a success response and resolve same retention operation; no second claim.

```python
def test_extracted_text_cannot_claim_pdf_original(retention_api):
    response = retention_api.upload(
        representation="document_original_v1", mime="application/pdf",
        filename="source.pdf", content=b"extracted words only")
    assert response.status_code == 422
```

`retention_api` is a new fixture following `history_api`'s authenticated client/dependency override pattern and exposes `upload` as a multipart wrapper over the production retention endpoint. It does not fake the native store or validator.

- [ ] Run red; implement the thin destination adapter after existing validated upload. Stage bytes first, then lock retention operation/source/claims, CAS native attachment fence, atomically publish native manifest and committed asset map. Download remains owner/revision bound; set safe MIME/content-disposition and reuse path-containment machinery. Reject active Sync owners and unsupported representations before publication.
- [ ] Implement the context PATCH: remove-from-context preserves display/bytes and disables only required request references; replace uses already retained verified bytes; restore requires known original hash. Advance manifest/context fences and compare expected revisions. No browser cancellation/source-job calls. Test stale concurrent patches, cross-owner/other-conversation references, wrong restore hash, lost response then canonical reload, and removal unblocking actual admission. No automatic replay with a newly refreshed revision.
- [ ] Add tests: downloaded bytes explicitly uploaded after external source deletion remain a valid new native copy; external IDs never resolve as claim IDs; old pre-promotion fork digest rejects; original missing yields truthful text-only/degraded outcome. Read/source-delete/child-delete in both orders verify independent exact bytes. Re-run relevant preflight tests and commit `feat(chat): retain reviewed attachments under native ownership`.

## Stage 3: Frozen character identity and future context

**Goal:** A ready copied character can reopen/edit/send from independent accepted behavior after the live card and definitions change or disappear.
**Success Criteria:** H2-A5 and character part of A6 pass with live readers poisoned, not just equal stored snapshots.
**Tests:** null card FK, cold API identity, single-character rich provider payload, summary reset/policy, neutral chats, protected update/import/legacy route guards.
**Status:** Not Started

### Task 3.1: Protected snapshot binding and canonical readers

**Files:** Modify `core/DB_Management/chacha/conversation_store.py`, `conversation_resume_store.py`, `api/v1/schemas/chat_session_schemas.py`, `chat_conversation_schemas.py`, `api/v1/endpoints/character_chat_sessions.py`, `core/Character_Chat/chat_settings_validation.py`; create `tests/Character_Chat_NEW/integration/test_native_fork_character_binding.py`.

**Interfaces:** `add_snapshot_bound_conversation(owner, child_id, projected_context, *, conn) -> str`, and `update_snapshot_bound_settings(owner, child_id, expected_settings_revision, update, *, conn)` are protected store/service entry points. Produces read-only `NativeAssistantBindingV1` and required projection descriptor. Public numeric-card creation remains unchanged.

- [ ] Write a child-bound fixture using accepted snapshot/envelope builders and real DB; hard-delete the actual source card, then list/detail/get settings/edit sampling/reset child. Assert child survives and no live card/preset materializer is called. Public identity spoofing and legacy sync/import replacement fail. Neutral ordinary fixture remains neutral in a workspace with default persona/card.
- [ ] Run red; use a dedicated internal identity normalization path, null card FK and child-local snapshot ID. Add read-only descriptor to serializers and fixed list predicates; live-card-specific lists do not absorb snapshot children. Copy accepted projection in caller transaction and protect binding/readiness/internal settings on generic update paths.
- [ ] Carry `assistant_binding_mode` in the coherent fork-purpose resume read once this migration exists. Test a second fork from a snapshot-bound H2 child: its assistant ID is child-local, while the accepted participant's original card ID is inert provenance. Reject a legacy row that merely imitates the `snapshot:` ID without protected mode.

```python
def test_hard_deleting_source_card_preserves_child(snapshot_child_case):
    snapshot_child_case.hard_delete_source_card()
    detail = snapshot_child_case.get_child_detail()
    assert (detail["character_id"], detail["assistant_binding"]["mode"]) == (
        None, "snapshot_v1")
```

`snapshot_child_case` is defined in the new integration suite using real card CRUD and production detail/settings endpoints; after insertion it replaces live behavior readers with failing sentinels. Include authorized/unauthorized workspace cases.

- [ ] Run green and affected existing character snapshot API cases. Review every normalizer/caller enumerated in the character audit; commit `feat(chat): bind forked characters to accepted snapshots`.

### Task 3.2: Frozen rich composition and every-entry admission

**Files:** Create `core/Chat/frozen_history_behavior.py`; modify `core/Chat/history_context.py`, `chat_service.py`, relevant admission in `api/v1/endpoints/chat.py`/`character_chat_sessions.py`/`character_messages.py`; extract pure reusable portions from `core/Character_Chat/world_book_manager.py`, `world_book_prompt_context.py`, `modules/persona_exemplar_selector.py`. Create `tests/Chat/unit/test_frozen_history_behavior.py`, `tests/Chat_NEW/integration/test_native_fork_context_admission.py`.

**Interfaces:** Consumes projected accepted context + exact admitted selected rows. Produces immutable `FrozenHistoryBehaviorPlan` and `FrozenHistoryPromptResult` from the interface map. Optional summary updates bind child ID plus admitted settings and post-input history revisions; compare both under the child lock through existing CAS semantics at `character_chat_sessions.py:4209`. A stale update loses rather than replacing newer child history/settings.

- [ ] Add golden provider-message tests for accepted preset section order, worldbook match/recursion/budget, memory/exemplar policy, author-note placement, pins, sampling and child-only summary policy. Poison all live card/preset/book/memory readers. Unsupported persona/multi-participant/overlay fails **before current input append**. Test generic completion, character completion and browser-managed admission, including a bypass attempt without projection header.

```python
def test_frozen_book_survives_live_edit(frozen_plan, selected_messages):
    result = compose_frozen_history_behavior(frozen_plan, selected_messages, ())
    rendered = "\n".join(row["content"] for row in result.messages)
    assert "accepted lore" in rendered and "new live lore" not in rendered
```

The two fixtures use the new pure accepted builder and actual selected-history rows. Add integration assertions against the intercepted provider payload, not only this string unit test.

- [ ] Run red; validate/freeze supported effects inside H1 owner admission before input append, then capture post-append history revision in that same transaction. Compose pure frozen inputs afterwards. Extract existing matching/packing algorithms rather than instantiate fake DB services. Reuse `_resolve_auto_summary_config`, deterministic summary construction and child-only fenced update semantics from character sessions; drop source summaries everywhere during fork, while allowing new child summary policy. Test held composition followed by actual message edit/delete/append with unchanged settings: the old summary cannot persist because history revision changed. Never seed greeting for empty fork or first continuation.
- [ ] Guard protected typed owners on direct/legacy entry points; a missing supported header returns 409 before lossy read/mutation/dispatch. Required missing asset markers block every dependent request before persistence; native claims feed the existing image/document/provider assembly. No foreign refine/file IDs or approvals remain executable. Run unit/integration + H1 selected context/continuation/system-message tests; independent review and commit `feat(chat): compose copied character context from frozen inputs`.

### Task 3.3: Native reference-image provider adapter

**Files:** Modify `api/v1/endpoints/files.py`, `api/v1/schemas/file_artifacts_schemas.py`, `core/File_Artifacts/file_artifacts_service.py`, `core/File_Artifacts/adapters/image_adapter.py`, `core/Image_Generation/reference_images.py` and `request_validation.py`; create `tldw_Server_API/tests/Chat_NEW/integration/test_native_chat_reference_image.py`. Client fields are Task4.2.

**Interfaces:** Strict `NativeChatImageReferenceV1` has kind `native_chat_asset_v1`, conversation ID, asset ID, exact revision and scope. The image payload carries `reference_image` mutually exclusive with legacy `reference_file_id`. `resolve_native_chat_reference(owner, reference) -> ResolvedReferenceImage` consumes the native asset service and verifies current exact owned claim/bytes; it never consults MediaFiles. Inject the resolver via FileArtifactsService/image request context derived from authenticated request, not a payload user ID. Normalization/export preserve and dispatch the discriminator.

- [ ] Test actual `/api/v1/files/create` synchronous inline image request with the injected real native store and a capturing image backend. Delete original MediaFiles first; assert `ImageGenRequest.reference_image` contains exact retained bytes. Reject both reference forms together, missing/unavailable/wrong-owner/wrong-scope/revision, unsupported model/backend, absent projection header and native reference on async export/re-export before new work is created.
- [ ] Implement the typed resolver after ordinary owner/scope admission, reusing existing MIME/content/size checks and detached `ResolvedReferenceImage`. Extend `_ImageAdapterRequestContext`/service setup at both normalize and export sites. Do not silently discard the new field in `normalize`, and export must test either discriminator rather than only `reference_file_id`. Keep legacy managed-reference path behavior unchanged. No provider work occurs in fork or a native DB transaction.
- [ ] Run new integration and affected image-adapter/reference tests; inspect captured provider payload and review capability gates. Commit `feat(images): resolve retained native chat references`.

## Stage 4: Atomic orchestration and both client consumers

**Goal:** End-to-end capture → reviewed dispatch → atomic commit/recovery → cold usable child, with honest old-server compatibility.
**Success Criteria:** H2-A1–A7 API/client gates pass; held responses cannot cause duplicate child or wrong-owner UI effects.
**Tests:** API fault injection/races and mounted shared controllers, actual scoped proxy transport, both shell browser acceptance in Stage 5.
**Status:** Not Started

### Task 4.1: Native coordinator and authenticated routes

**Files:** Implement native_fork service/route from interface map; register using `router.include_router(native_forks.router)` beside existing includes in `api/v1/endpoints/chat.py:413`. Extend native_fork_store `commit_fork`; create `tests/Chat_NEW/integration/test_native_fork_api.py`; extend transaction tests.

**Interfaces:** `execute(owner, request) -> NativeForkResultV1`, `resolve(owner, operation_id, request_digest) -> NativeForkResultV1`, `capture(owner, request) -> NativeForkCaptureV1`; `commit_fork(owner, attempt, request, prepared_assets, *, conn) -> NativeForkResultV1`. Consumes Stages 1–3; emits ready state only with receipt commit.

- [ ] Write one test per fault boundary: graph row, selected message/image, settings, snapshot, binding, asset ref, map/receipt and ready commit. Use actual caller transaction rollback and failpoints only at the intended write boundary. Two concurrent same-key requests yield one exact map/child; same key changed title/fidelity conflicts. A 20,001-row source selection is complete, not page-limited.

```python
def test_lost_response_then_source_delete_replays_same_child(native_fork_api):
    request = native_fork_api.reviewed_request()
    committed = native_fork_api.post(request).json()
    native_fork_api.delete_source()
    replay = native_fork_api.post(request).json()
    assert replay["message_map"] == committed["message_map"]
    assert replay["child_id"] == committed["child_id"]
```

Define `native_fork_api` like H1 `history_api`, with real authenticated routes and native DB. Its request builder calls capture and canonical digest. The separate lost-response test drops the HTTP response *after* production commit, then resolves; this basic replay test alone is not transport-loss proof.

- [ ] Run red; implement spec §5 exactly. Receipt lookup precedes source checks; stage external native bytes before caller transaction. Reproject under source lock and accept irrelevant excluded-only changes, reject retained changes. Insert fresh graph/provenance and independent projected settings/snapshot/assets, map and ready receipt on that connection. Never use normal character creation or old createChat/addChatMessage methods.
- [ ] Apply W1 at final fork admission with operation → workspace → conversation ordering; early capture/preparation checks are insufficient. Qualify both admission/closure winners with actual child/receipt creation, no FK scope detachment, retryable incomplete workspace deletion, and source-independent receipt resolution after workspace removal. A committed live child during closure returns lifecycle denial without leaking its map or burning the receipt. Only actual missing/deleted child yields gone. Map lifecycle failures truthfully in HTTP; no new-key/owner fallback.
- [ ] Add owner/scope/active-Sync denial, deletion/restore/expiry, permission-loss and actual metadata/settings writer races. Validate OpenAPI and exact error variants (201 new, 200 replay, 202 pending, 409 conflict/upgrade, terminal gone/expired outcomes, ordinary denied HTTP response). Expose capabilities only for qualified representation/effect versions. Run green + affected H1 APIs/transactions/Bandit; review and commit `feat(chat): commit native forks with durable recovery`.

### Task 4.2: Shared client protocol, retention, cold reopen and recovery

**Files:** Create UI native-fork types/service and `services/__tests__/native-fork.test.ts`; modify `db/dexie/types.ts`, `fork-operations.ts` and its tests; `services/chat-history-selection.ts`, `services/tldw/TldwApiClient.ts`, `services/tldw/domains/chat-rag.ts`, `services/tldw/service-prompt-scope-error.ts`; `hooks/handlers/messageHandlers.ts`, `hooks/chat/useHistorySelection.ts`, `useServerChatLoader.ts`, `effective-assistant-state.ts`, `native-history-character-send.ts`; `components/Common/Playground/HistorySelectionReview.tsx`; representation-aware retention in `services/chat-document-processing.ts` and native branch of `hooks/chat/useFileUpload.ts`. Carry typed native reference through `utils/image-generation-chat.ts` (`ImageGenerationRequestSnapshot`), `hooks/chat-modes/normalChatMode.ts`, both image client serializers (`services/tldw/TldwApiClient.ts:7504`, `services/tldw/domains/media.ts:1524`) and image generation/refinement requests in `components/Option/Playground/hooks/usePlaygroundImageGen.ts`. Modify actual shared transport handlers `entries/background.ts`, `services/background-proxy.ts`, `services/tldw/request-core.ts`, plus canonical locales and generated output through existing sync script. The extension entrypoint is only a re-export; test WebUI through its actual shim resolution without inventing a second transport implementation.

**Interfaces:** `captureAtomicNativeFork(owner, view, fidelity)`, `commitAtomicNativeFork(owner, frozenRequest)`, `resolveAtomicNativeFork(owner, operation)` and `retainNativeAttachments(owner, retentionRequest, files)` in new shared service. All use frozen `NativeHistoryOwnerV1` and explicit scope. `ForkOperation.protocol` distinguishes native_atomic_v1/legacy_native_v1/current local protocols, and its request type is a tagged union using the matching canonical digest validator; the existing legacy `historyDigest` validator cannot validate an atomic tuple request. Read-only `native_bundle` and optional character binding descriptors are carried by `ServerChatSummary`; neutral plain children also load canonical settings without a local receipt.

- [ ] Write shared digest fixtures against the Python JSON corpus; mounted tests cover no dispatch before pending write, held response after account/scope switch, same-key pending/resolve, not_recorded exact resubmit, committed result-cache/open failure, gone/expiry and forbidden legacy migration. Add proxy tests for exact allowed routes/header and encoded slash/trailing-path negatives. Type extracted text distinctly from original binary before building multipart.

```typescript
it("does not reinterpret a legacy unknown operation as atomic", async () => {
  const operation = makeLegacyUnknownOperation()
  await expect(resolveAtomicNativeFork(owner, operation))
    .rejects.toThrow("fork_protocol_mismatch")
  expect(transport.post).not.toHaveBeenCalled()
})
```

Define `makeLegacyUnknownOperation` in the new test using the actual existing ForkOperation shape with omitted protocol; use current scoped-transport test spies for `transport`, and actual immutable owner fixture. Add a real Dexie browser proof in Stage 5; this unit test does not prove persistent transaction ordering.

- [ ] Run affected Vitest suites red. Implement explicit capability negotiation, immutable pending request persistence, same-ID resolution/resume and owner lease checks at every await. No fallback on mutation 404 or timeout. Keep server commit feedback even if local result storage/open fails. Render explicit retention review, freeze byte/revision inputs, upload through bgUpload and recapture only after native retention receipt. Unknown/missing bytes get honest strict/degraded choices.
- [ ] Carry protected server descriptor through list/detail/loader and use accepted display/settings instead of live card. Install `native_bundle`/required projection for every H2 child, including a neutral empty child, plus sources that acquired typed retention. Add cold-browser ordinary and character cases with no local receipt and poisoned ambient settings. Send projection header on exact protected-owner paths. Wire native claims to image/document/refine context adapters; old IDs are inert provenance. Missing required asset blocks send; restore/replace/remove updates canonical revision. Source/current view/account mismatch cannot adopt the child or attachment.
- [ ] Add `updateNativeAssetContext` shared method using exact scoped PATCH, expected revisions and canonical result. Native remove/replace/restore never calls existing source draft/job cancellation. Lost result reloads original-owner context; no unconditional rebase/retry. Propagate `NativeChatImageReferenceV1` through request snapshot, both image serializers and reference-dependent generation/refinement controls to Task3.3; strip executable foreign reference IDs. The separate text-only “Refine with LLM” prompt helper does not read image bytes and does not prove reference-image admission. Test mounted reference-dependent generation after deleting original source and a changed owner while native reference bytes are held.
- [ ] Extend mounted loader/editor/review/controller tests plus true extension/WebUI proxy tests; inspect both consumer dependency closures. Sync locales via existing script; use semantic labels/focus/keyboard/error state patterns. Commit reviewed working consumer changes `feat(ui): recover native forks and retained attachments`.

## Stage 5: Qualification, review closure and delivery evidence

**Goal:** Demonstrate each required gate at a known final commit and preserve the exact limits of those results.
**Success Criteria:** All in-scope H2-A1–A8 pass; all identified P1/P2 and source-audit A/C/R obligations resolved; broader-parity gates remain explicit. Required matrix skips cannot count as pass.
**Tests:** production-backed WebUI/extension flows, SQLite/PostgreSQL, relevant H1 regression, affected builds/types/lint/security and documented rollback compatibility.
**Status:** Not Started

### Task 5.1: Both shells and native-owner failure matrix

**Files:** Create `apps/tldw-frontend/e2e/workflows/chat-native-fork.spec.ts` and `apps/extension/tests/e2e/chat-native-fork.spec.ts`; reuse current H1 `chat-history-selection.spec.ts` harness, preserving actual route/controller tests. Add only task-owned test fixtures/config needed for isolated runtime. Create final `Docs/Reviews/CHATBOOK_H2_NATIVE_FORK_VERIFICATION_2026_09_18.md` when actual evidence exists.

- [ ] Run a fresh isolated test API/WebUI/extension profile and repository PG fixture, recording owned ports/process IDs and exact commands. Discover free ports; do not reuse or terminate main/UAT or assume previous H1 test infrastructure still belongs to this run. Use `TLDW_TEST_POSTGRES_REQUIRED=1` and fixture-provisioned Postgres. Activate `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate` before Python. Follow existing H1 harness launch/teardown patterns, replacing only owned run identifiers.
- [ ] Implement and run these distinct browser cases in each full-page shell: ordinary before/after/empty selected copy; exact multi-image/document original+text/generated revision; explicit retention and pre-promotion digest rejection; character source-card deletion then cold reopen/edit/next send; lost response/reload recovery; deleted child then retry; account/workspace switch during held operation; strict/degraded missing asset; postcommit cache/open failure. Compact extension review expansion remains in-extension and preserves operation. Disable Sync for native proof, then separately enable and verify native new-operation gate. Record exact provider payload through isolated provider test server; do not contact a real paid model.
- [ ] For a browser failure use deterministic repro, repair under the relevant task, re-run affected matrix and fresh review. Keep no retries/skips hidden in headline totals. Real backend tests own rollback/lock proof; mocked browser network alone cannot establish it.

### Task 5.2: Cross-gate checks, final review and tracking

- [ ] Run affected native suites once at final relevant head, including new projection/store/asset/binding/context/API suites and existing H1 `test_history_selection_transactions.py`, `test_history_selection_migration.py`, `test_history_selection_api.py`, `test_history_context.py`, multi-image and continuation/system-message cases. Capture command/env/exit/pass/skip details. If a required PG fixture is unavailable, keep its acceptance gate open; do not roll an ad hoc alternative database.
- [ ] Run affected shared Vitest tests from `apps/packages/ui`, full actual WebUI `bun run typecheck`, extension `bun run compile`, corresponding production builds plus token/bundle budgets and touched lint. Compare diagnostics by identity to a fresh baseline at the implementation base, not merely the count; H1's 90 existing WebUI diagnostics and two traced-copy warnings are historical context, not a waiver. Never describe focused aliases as full consumer qualification.
- [ ] Run Bandit and compile checks on exact touched Python paths using project venv; inspect findings before completion. Verify generated API/locale changes, scoped proxy/header/CORS tests and unsupported-client guards. Do not include `.next-live-tier-h1-production/` or secrets/profiles/test DBs in staging.
- [ ] Fresh reviewer reads final change and evidence against every H2-A gate. Fix concrete P1/P2, requalify changed scope and repeat until none; retain full reports and disposition. A receipt/plain-image subset must stay a milestone if rich A4/A5 is incomplete. Update parent parity ledger without marking full B01–B03 Equivalent.
- [ ] Archive this completed plan only after all stages pass, repair its links, and finalize linked Backlog tasks with tests/limitations/commit/PR links. Commit scoped evidence normally. Keep H1 PR separate; a later H2 PR explains stacked base and human-written Change summary merge gate. No merge or deployment is implied by this plan.

## Coverage and planned commands

| Acceptance | Implementing tasks |
|---|---|
| A1 canonical namespace/protocol | 1.1, 1.2, 4.1, 4.2 |
| A2 atomicity/selection/concurrency | 1.2, 2.1, 4.1, 5.1 |
| A3 replay/deletion/expiry/access | 1.2, 4.1, 4.2, 5.1 |
| A4 retained assets/explicit promotion/lifetime | 2.1, 2.2, 2.3, 3.3, 4.2, 5.1 |
| A5 accepted character reopen/edit/send/reset | 1.1, 3.1, 3.2, 4.2, 5.1 |
| A6 unavailable assets/authority exclusions | 1.1, 2.3, 3.2, 3.3, 4.1, 4.2 |
| A7 consumers/recovery/legacy | 4.1, 4.2, 5.1 |
| A8 Sync/RLS/migration/security/static checks | 1.2, 2.2, 2.3, 3.2, 4.1, 4.2, 5.2 |

Example focused commands after the named files exist (all from this worktree, with fixture-owned isolated env configured):

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -o addopts='' tldw_Server_API/tests/Chat/unit/test_native_fork_projection.py -q
python -m pytest -o addopts='' tldw_Server_API/tests/DB_Management/test_native_fork_transactions.py -v -rs
bun run --cwd apps/packages/ui test src/services/__tests__/native-fork.test.ts
bun run --cwd apps/tldw-frontend e2e:pw e2e/workflows/chat-native-fork.spec.ts --workers=1 --retries=0
bun run --cwd apps/extension test:e2e tests/e2e/chat-native-fork.spec.ts --workers=1 --retries=0
```

These are planned commands, not results. Documentation verification for TASK-13261.2 is recorded separately from implementation qualification.
