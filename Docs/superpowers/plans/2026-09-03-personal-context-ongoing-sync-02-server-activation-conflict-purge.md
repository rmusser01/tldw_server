# Personal Context Ongoing Sync: Server Activation, Conflict, Purge, and Rollout Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Safely activate linked profiles, resolve conflicts through canonical authority, fence restrictive cleanup and global deletion, then truthfully enable server ongoing-sync version 1.

**Architecture:** Activation, conflict resolution, cleanup, and purge are replayable journals spanning `Personalization.db` and Sync V2 rather than cross-database transactions. Every state transition is bound to immutable IDs and receipts; Personalization remains canonical, while Sync stores delivery, cursor, protected candidate, and device acknowledgment state. Version 1 remains unavailable until every required component passes readiness.

**Tech Stack:** Python 3.11, Pydantic v2, SQLite Personalization, SQLite/PostgreSQL Sync V2, FastAPI, pytest, MkDocs.

**Spec:** `Docs/superpowers/specs/2026-09-02-personal-context-ongoing-sync-design.md`

## Global Constraints

- ADR required: no new ADR. Use `backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md`.
- Execute after TASK-13159 through TASK-13161.
- Preserve one home authority, stable canonical object IDs, authoritative server manifest sequencing, and ingress-only client envelopes.
- Treat activation, conflicts, cleanup, and purge as journals with exact replay receipts; never claim cross-database atomicity.
- Encrypt all baselines, candidates, resolution bodies, and cleanup-sensitive data at rest.
- Keep `ongoing_sync_version = 0` until TASK-13165 passes its readiness gate.
- Use the existing batched conflict endpoint; do not add a second resolution API.

---

### Task 1: TASK-13162 — Activate links with continuity fencing

**Files:**

- Create: `tldw_Server_API/app/core/Personalization/personal_context_activation.py`
- Modify: `tldw_Server_API/app/core/DB_Management/Personalization_DB.py`
- Modify: `tldw_Server_API/app/core/Personalization/personal_context_publication.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/store.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/sync.py`
- Test: `tldw_Server_API/tests/Personalization/test_personal_context_activation.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_activation.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_transport.py`

**Interfaces:**

- Consumes: TASK-13159 `PersonalContextExchangeProof` and TASK-13161 `PersonalContextRelay`.
- Produces: `PersonalContextActivationService.prepare()`, `.install()`, `.acknowledge()`, and `.validate_exchange()`.
- Produces API behavior: ongoing activation preparation through the versioned `/api/v1/sync/personal-context/bootstrap` fields and acknowledgment through `POST /api/v1/sync/personal-context/activation/acknowledge`.

- [ ] **Step 1: Add failing whole-batch and restart tests**

```python
def test_prepare_watermark_never_splits_publication_batch(activation, publications) -> None:
    publications.append_test_batch(size=3)
    prepared = activation.prepare(**activation_args())
    batch = publications.batch(prepared.publication_watermark)
    assert prepared.publication_watermark == batch.profile_publication_sequence
    assert batch.batch_size == 3


def test_compaction_retains_activation_covered_terminal_proof(activation, publications) -> None:
    prepared = activation.prepare(**activation_args())
    receipt = activation.install(prepared.activation_id)
    publications.compact_covered_bodies(prepared.profile_id)
    assert publications.covered_through(prepared.profile_id) == prepared.publication_watermark
    assert activation.install(prepared.activation_id) == receipt
```

- [ ] **Step 2: Run activation tests and confirm RED**

Run: `.venv/bin/python -m pytest tldw_Server_API/tests/Personalization/test_personal_context_activation.py tldw_Server_API/tests/Sync/test_sync_v2_personal_context_activation.py -q`

Expected: activation service and journal tables are absent.

- [ ] **Step 3: Add Personalization and Sync activation journals**

In Personalization, store an encrypted exact-head baseline plus clear bounded activation ID, digest, generation, whole-batch watermark, state, Sync receipt ID, and timestamps. In Sync, store deterministic baseline envelope IDs, installation receipt, per-device acknowledgment, and expiry. Use one row per activation and one row per device acknowledgment.

```python
class ActivationState(StrEnum):
    PREPARED = "prepared"
    INSTALLED = "installed"
    ACTIVE_FOR_DEVICE = "active_for_device"


@dataclass(frozen=True, slots=True)
class PreparedActivation:
    activation_id: str
    profile_id: str
    baseline_digest: str
    purge_generation: int
    publication_watermark: int
    state: ActivationState


@dataclass(frozen=True, slots=True)
class SyncActivationInstallReceipt:
    receipt_id: str
    activation_id: str
    baseline_digest: str
    last_server_cursor: int


@dataclass(frozen=True, slots=True)
class ActivationInstallationReceipt:
    activation_id: str
    baseline_digest: str
    purge_generation: int
    publication_watermark: int
    sync_receipt_id: str
    home_server_cursor: int
```

For Sync tables, implement equivalent SQLite and PostgreSQL DDL plus repository tests. The watermark must equal the sequence of a fully committed batch or zero when no batch exists.

- [ ] **Step 4: Prepare and install the deterministic baseline**

```python
def prepare(self, *, user_id: str, profile_id: str) -> PreparedActivation:
    with self.publications.profile_lease(profile_id):
        snapshot = self.profile_service(user_id).sync_snapshot()
        watermark = self.publications.latest_complete_batch_boundary(profile_id)
        return self.store.create_or_get_prepared(snapshot, watermark)


def install(self, *, user_id: str, activation_id: str) -> ActivationInstallationReceipt:
    prepared = self.store.require_prepared(activation_id)
    sync_receipt = self.sync_installer.install_exact_baseline(prepared)
    self.sync_installer.verify_receipt(sync_receipt, prepared.baseline_digest)
    return self.store.cover_and_install(prepared, sync_receipt)
```

`cover_and_install()` runs under the same profile lease and one immediate Personalization transaction. It CASes the prepared activation state, marks every incomplete batch through the exact watermark `covered_by_activation`, binds activation ID/digest/receipt, advances `activation_covered_through_sequence`, and only then permits encrypted source-body compaction.

- [ ] **Step 5: Issue and enforce continuity proof**

Generate the activation epoch and continuity token with `secrets.token_urlsafe(32)`, store them durably with profile and purge generation, and return them only after installation. Add:

```python
def validate_exchange(
    self,
    *,
    user_id: str,
    dataset_id: str,
    device_id: str,
    proof: PersonalContextExchangeProof,
) -> PersonalContextExchangeProof:
    current = self.store.current_exchange_proof(user_id, dataset_id, device_id)
    if not hmac.compare_digest(proof.activation_epoch, current.activation_epoch):
        raise PersonalContextActivationRequired("personal_context_activation_required")
    if not hmac.compare_digest(proof.continuity_token, current.continuity_token):
        raise PersonalContextActivationRequired("personal_context_activation_required")
    return current
```

Invoke before any version-1 push mutation, pull delivery, conflict list, or conflict resolution. Echo the exact current proof in every successful response. If publication journaling cannot commit, either reject canonical mutation or advance epoch and invalidate continuity in the same canonical transaction.

- [ ] **Step 6: Add per-device acknowledgment and downgrade behavior**

The acknowledgment endpoint accepts activation ID and baseline digest, writes Sync device acknowledgment idempotently, then advances the Personalization server journal only after verifying the exact receipt. A capability downgrade never deletes activation, publication, or device state. Restoration with a changed/unverifiable pair returns `personal_context_activation_required`.

```python
def acknowledge_for_device(
    self,
    request: SyncPersonalContextActivationAcknowledgeRequest,
) -> PersonalContextActivationReceipt:
    sync_receipt = self.store.acknowledge_personal_context_activation(
        dataset_id=request.dataset_id,
        device_id=request.device_id,
        activation_id=request.activation_id,
        baseline_digest=request.baseline_digest,
    )
    self.activations.verify_and_mark_active(
        activation_id=request.activation_id,
        baseline_digest=request.baseline_digest,
        sync_receipt_id=sync_receipt.receipt_id,
    )
    return self.activations.receipt_for_device(request.device_id)
```

- [ ] **Step 7: Run activation, continuity, and backend contract tests**

```bash
.venv/bin/python -m pytest \
  tldw_Server_API/tests/Personalization/test_personal_context_activation.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_activation.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_transport.py \
  tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py -q
.venv/bin/ruff check \
  tldw_Server_API/app/core/Personalization/personal_context_activation.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/api/v1/endpoints/sync.py \
  tldw_Server_API/tests/Personalization/test_personal_context_activation.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_activation.py
git diff --check
```

Expected: every interruption resumes the same activation bytes; racing server writes land in baseline or post-watermark stream; capability remains version 0.

- [ ] **Step 8: Commit TASK-13162**

```bash
git add tldw_Server_API/app/core/Personalization/personal_context_activation.py \
  tldw_Server_API/app/core/DB_Management/Personalization_DB.py \
  tldw_Server_API/app/core/Personalization/personal_context_publication.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/api/v1/endpoints/sync.py \
  tldw_Server_API/tests/Personalization/test_personal_context_activation.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_activation.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_transport.py \
  backlog/tasks/task-13162\ -\ Activate-existing-Personal-Context-links-with-continuity-fencing.md
git commit -m "feat(sync): activate Personal Context continuity"
```

### Task 2: TASK-13163 — Resolve conflicts through the batched API

**Files:**

- Create: `tldw_Server_API/app/core/Sync/v2/personal_context_conflicts.py`
- Modify: `tldw_Server_API/app/core/Personalization/personal_context_service.py`
- Modify: `tldw_Server_API/app/core/Personalization/personal_context_publication.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/models.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/store.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/sync.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_conflicts.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_service.py`
- Test: `tldw_Server_API/tests/Personalization/test_personal_context_service.py`

**Interfaces:**

- Consumes: TASK-13162 exchange validation and TASK-13161 internal authority insertion.
- Produces: `PersonalContextConflictService.ensure_authority_candidate()` and `.resolve_batch_item()`.
- Extends: existing `POST /api/v1/sync/conflicts/resolve`; no new conflict route.

- [ ] **Step 1: Add failing candidate and replay tests**

```python
def test_push_conflict_returns_protected_current_authority_candidate(sync_service) -> None:
    result = sync_service.push(**stale_personal_context_push())
    conflict = result.conflicts[0]
    assert conflict.remote_envelope_id
    assert conflict.remote_envelope.authority.role == "home_authority"
    assert sync_service.store.get_envelope(conflict.remote_envelope_id).retention_pinned


def test_resolution_replay_does_not_duplicate_manifest_or_publication(conflicts) -> None:
    first = conflicts.resolve_batch_item(**overwrite_resolution())
    replay = conflicts.resolve_batch_item(**overwrite_resolution())
    assert replay == first
    assert conflicts.publication_count(first.idempotency_key) == 1
```

- [ ] **Step 2: Run conflict tests and confirm RED**

Run: `.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_personal_context_conflicts.py -q`

Expected: push conflicts do not carry a protected authority envelope and expected IDs are not enforced.

- [ ] **Step 3: Create or reuse the deterministic authority candidate before reporting conflict**

```python
def ensure_authority_candidate(
    self,
    *,
    user_id: str,
    dataset_id: str,
    conflict: SyncConflict,
) -> SyncEnvelope:
    canonical = self.profile_service(user_id).canonical_head(conflict.domain, conflict.entity_id)
    envelope_id = stable_conflict_authority_envelope_id(conflict.conflict_id, canonical.version_id)
    envelope = self.authority.insert_or_get(
        dataset_id=dataset_id,
        envelope_id=envelope_id,
        canonical=canonical,
        retention_pin=f"conflict:{conflict.conflict_id}",
    )
    self.store.attach_remote_envelope(conflict.conflict_id, envelope.envelope_id)
    return envelope
```

Return a terminal push conflict only after the envelope and conflict reference are durable. Replay returns the same ID and canonical bytes even when the canonical version predates the requesting cursor.

- [ ] **Step 4: Enforce expected candidates and narrow freezes**

Add `expected_local_envelope_id`, `expected_remote_envelope_id`, and `idempotency_key` to each Personal Context resolution. Reject stale IDs before claiming the conflict. Record ordinary freeze by object ID; for semantic-key collisions record both object IDs plus a hash-bound `(scope_id, kind, semantic_key)` slot inside encrypted conflict state, exposing only opaque identifiers in generic metadata.

```python
@dataclass(frozen=True, slots=True)
class PersonalContextResolutionCommand:
    user_id: str
    dataset_id: str
    conflict_id: str
    action: Literal["skip", "overwrite", "duplicate_rename"]
    expected_local_envelope_id: str
    expected_remote_envelope_id: str
    idempotency_key: str
    resolution_envelope: SyncV2Envelope | None = None


def require_expected_candidates(self, request: PersonalContextResolutionCommand) -> SyncConflict:
    conflict = self.store.get_conflict(request.conflict_id, dataset_id=request.dataset_id)
    if (
        conflict.local_envelope_id != request.expected_local_envelope_id
        or conflict.remote_envelope_id != request.expected_remote_envelope_id
    ):
        raise SyncConflictError("personal_context_stale_conflict_review")
    return conflict
```

- [ ] **Step 5: Route mutating actions through Personalization authority**

```python
@dataclass(frozen=True, slots=True)
class ResolutionReceipt:
    conflict_id: str
    action: Literal["skip", "overwrite", "duplicate_rename"]
    status: Literal["resolved"]
    authority_publication_batch_id: str | None


def resolve_batch_item(self, request: PersonalContextResolutionCommand) -> ResolutionReceipt:
    conflict = self.require_expected_candidates(request)
    if request.action == "skip":
        return self.resolve_skip_in_sync(conflict, request.idempotency_key)
    canonical = self.profile_service(request.user_id).apply_conflict_resolution(
        conflict_token=conflict.conflict_id,
        expected_local_envelope_id=request.expected_local_envelope_id,
        expected_remote_envelope_id=request.expected_remote_envelope_id,
        action=request.action,
        resolution=request.resolution_envelope,
        idempotency_key=request.idempotency_key,
    )
    return self.finalize_from_personalization_receipt(conflict, canonical)
```

`overwrite` handles keep-local and reviewed merge payloads. `duplicate_rename` requires a new object ID and noncolliding semantic key. The canonical transaction writes the result, manifest, publication batch, conflict receipt, and expected candidates before Sync resolves the generic conflict. `skip` changes no canonical object.

- [ ] **Step 6: Prove stale review, collision scope, retention, and partial batch behavior**

Run:

```bash
.venv/bin/python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_conflicts.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_retention.py \
  tldw_Server_API/tests/Personalization/test_personal_context_service.py \
  tldw_Server_API/tests/Personalization/test_personal_context_plaintext_canary.py -q
.venv/bin/ruff check \
  tldw_Server_API/app/core/Sync/v2/personal_context_conflicts.py \
  tldw_Server_API/app/core/Personalization/personal_context_service.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_conflicts.py
git diff --check
```

Expected: unrelated objects continue, candidates remain pinned, stale choices mutate nothing, and each mutating replay creates one canonical publication.

- [ ] **Step 7: Commit TASK-13163**

```bash
git add tldw_Server_API/app/core/Sync/v2/personal_context_conflicts.py \
  tldw_Server_API/app/core/Personalization/personal_context_service.py \
  tldw_Server_API/app/core/Personalization/personal_context_publication.py \
  tldw_Server_API/app/core/Sync/v2/models.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/api/v1/endpoints/sync.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_conflicts.py \
  backlog/tasks/task-13163\ -\ Resolve-Personal-Context-conflicts-through-the-batched-Sync-API.md
git commit -m "feat(sync): resolve Personal Context conflicts"
```

### Task 3: TASK-13164 — Fence restrictive cleanup and global purge

**Files:**

- Create: `tldw_Server_API/app/core/Personalization/personal_context_cleanup.py`
- Modify: `tldw_Server_API/app/core/DB_Management/Personalization_DB.py`
- Modify: `tldw_Server_API/app/core/DB_Management/Personal_Context_Repository.py`
- Modify: `tldw_Server_API/app/core/Personalization/personal_context_service.py`
- Modify: `tldw_Server_API/app/core/LLM_Calls/context_builders/personal_context.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/personal_context_relay.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/store.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/sync.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/personal_context.py`
- Test: `tldw_Server_API/tests/Personalization/test_personal_context_cleanup.py`
- Test: `tldw_Server_API/tests/Personalization/test_personal_context_plaintext_canary.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_purge.py`

**Interfaces:**

- Produces: `PersonalContextCleanupJournal.require()`, `.run_due()`, and `.acknowledge()`.
- Produces: `PersonalContextService.request_global_purge()` returning a durable purge receipt, the TASK-13159 signed Sync purge endpoint, and the existing direct server purge path routed through the same generation fence.
- Consumes: TASK-13161 profile relay lease and TASK-13162 continuity/generation state.

- [ ] **Step 1: Add failing restrictive-read and purge-interleaving tests**

```python
def test_user_only_head_is_excluded_before_cleanup_ack(context_builder, service) -> None:
    record = service.create_record(agent_visible_record())
    context_builder.snapshot()
    service.update_record(record.record_id, controls=user_only_controls())
    assert record.record_id not in context_builder.snapshot().record_ids
    assert service.cleanup_status().pending == 1


@pytest.mark.parametrize("failure_point", PURGE_FAILURE_POINTS)
def test_old_generation_cannot_publish_after_barrier(purge_harness, failure_point) -> None:
    purge_harness.fail_at(failure_point)
    purge_harness.run_and_restart()
    assert purge_harness.authority_generations_after_barrier() == []
    assert purge_harness.old_ingress_status() == "stale_generation"
```

- [ ] **Step 2: Run cleanup and purge tests and confirm RED**

Run: `.venv/bin/python -m pytest tldw_Server_API/tests/Personalization/test_personal_context_cleanup.py tldw_Server_API/tests/Sync/test_sync_v2_personal_context_purge.py -q`

Expected: version-bound cleanup journal and globally ordered purge barrier are absent.

- [ ] **Step 3: Journal restrictive cleanup in the canonical transaction**

```python
def require(
    self,
    connection: sqlite3.Connection,
    *,
    profile_id: str,
    object_id: str,
    version_id: str,
    purge_generation: int,
    cleanup_kinds: frozenset[str],
) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO personal_context_cleanup_requirements "
        "(profile_id, object_id, version_id, purge_generation, cleanup_kinds, state, created_at) "
        "VALUES (?, ?, ?, ?, ?, 'pending', ?)",
        (profile_id, object_id, version_id, purge_generation, encode_kinds(cleanup_kinds), now_text()),
    )
```

Insert this before exposing the restrictive head. Context/search/summary/tool builders must re-read the current head and reject broader cached/indexed state immediately. Cleanup deletes or rewrites derived artifacts, invalidates in-memory snapshots, and acknowledges only the exact version/generation.

- [ ] **Step 4: Implement the signed durable purge request and canonical transaction**

```python
def request_global_purge_from_sync(
    self,
    *,
    request: SyncPersonalContextPurgeRequest,
) -> PersonalContextPurgeReceipt:
    signed_bytes = canonical_json_bytes(
        {
            "dataset_id": request.dataset_id,
            "device_id": request.device_id,
            "request_id": request.request_id,
            "expected_purge_generation": request.expected_purge_generation,
            "idempotency_key": request.idempotency_key,
        }
    )
    self.verify_device_purge_signature(request.device_id, signed_bytes, request.signature)
    with self.publications.profile_lease(self.profile_id):
        return self.repository.purge_and_publish_barrier(
            request_id=request.request_id,
            idempotency_key=request.idempotency_key,
            expected_generation=request.expected_purge_generation,
        )
```

The Sync purge request deliberately does not require a current activation epoch/token: global deletion must remain recoverable after a capability or continuity failure. Authentication, registered-device key verification, the signed dataset/device/request/generation/idempotency tuple, and current generation check still fail closed. The direct authenticated `/api/v1/personal-context/purge` path calls the same leased repository transaction with a server-generated request/idempotency pair after its existing typed confirmation; it does not emulate a device signature. One immediate Personalization transaction advances generation, destroys canonical and derived-readable content, terminalizes every older unstaged source batch, invalidates continuity, and writes one deterministic new-generation purge-barrier batch. An exact request replay returns the stored receipt; a changed payload under the same ID fails closed.

- [ ] **Step 5: Fence ingress, relay, Sync history, and profile recreation**

Recheck generation before ingress canonical commit and before each authority staging operation under the shared lease. Mark older uncommitted ingress `stale_generation`. Stage the barrier only after permissible older authority rows precede it and every remaining old source batch is terminal. When the barrier is acknowledged, crypto-shred readable Personal Context Sync payload history and retain only profile/dataset/device IDs, generation, timestamps, and acknowledgment state.

```python
def require_current_generation(self, *, profile_id: str, envelope_generation: int) -> None:
    current = self.publications.current_purge_generation(profile_id)
    if envelope_generation != current:
        raise StalePersonalContextGeneration(
            "personal_context_ingress_stale_generation"
        )


def finalize_acknowledged_purge(self, receipt: PurgeBarrierReceipt) -> None:
    with self.store.transaction() as connection:
        self.store.require_barrier_acknowledged(connection, receipt)
        self.store.crypto_shred_personal_context_payloads_before_generation(
            connection,
            profile_id=receipt.profile_id,
            generation=receipt.purge_generation,
        )
        self.store.record_content_free_purge_ack(connection, receipt)
```

- [ ] **Step 6: Run cleanup, purge, context, retention, and canary tests**

```bash
.venv/bin/python -m pytest \
  tldw_Server_API/tests/Personalization/test_personal_context_cleanup.py \
  tldw_Server_API/tests/Personalization/test_personal_context_plaintext_canary.py \
  tldw_Server_API/tests/Personalization/test_personal_context_service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_purge.py \
  tldw_Server_API/tests/Sync/test_sync_v2_retention.py -q
.venv/bin/ruff check \
  tldw_Server_API/app/core/Personalization/personal_context_cleanup.py \
  tldw_Server_API/app/core/Personalization/personal_context_service.py \
  tldw_Server_API/app/core/LLM_Calls/context_builders/personal_context.py \
  tldw_Server_API/app/core/Sync/v2/personal_context_relay.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_purge.py
git diff --check
```

Expected: restrictive content is inaccessible before cleanup acknowledgment; all purge interleavings converge; no old-generation row stages after the barrier; readable Sync history is absent after shredding.

- [ ] **Step 7: Commit TASK-13164**

```bash
git add tldw_Server_API/app/core/Personalization/personal_context_cleanup.py \
  tldw_Server_API/app/core/DB_Management/Personalization_DB.py \
  tldw_Server_API/app/core/DB_Management/Personal_Context_Repository.py \
  tldw_Server_API/app/core/Personalization/personal_context_service.py \
  tldw_Server_API/app/core/LLM_Calls/context_builders/personal_context.py \
  tldw_Server_API/app/core/Sync/v2/personal_context_relay.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/api/v1/endpoints/sync.py \
  tldw_Server_API/app/api/v1/endpoints/personal_context.py \
  tldw_Server_API/tests/Personalization/test_personal_context_cleanup.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_purge.py \
  backlog/tasks/task-13164\ -\ Fence-Personal-Context-privacy-cleanup-and-global-purge.md
git commit -m "feat(personal-context): fence cleanup and purge"
```

### Task 4: TASK-13165 — Enable and document server version 1

**Files:**

- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/factory.py`
- Modify: `Docs/User_Guides/Server/Personal_Context_Profile.md`
- Modify: `Docs/Code_Documentation/Personal_Context_Developer_Guide.md`
- Modify: `Docs/API-related/Personal_Context_API.md`
- Regenerate: `Docs/Published/User_Guides/Server/Personal_Context_Profile.md`
- Regenerate: `Docs/Published/Code_Documentation/Personal_Context_Developer_Guide.md`
- Regenerate: `Docs/Published/API-related/Personal_Context_API.md`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_readiness.py`
- Test: `tldw_Server_API/tests/Personalization/integration/test_personal_context_composed_app.py`

**Interfaces:**

- Consumes: completed TASK-13159 through TASK-13164.
- Produces: truthful `PersonalContextSyncCapabilities.ongoing_sync_version == 1` only under complete readiness.
- Produces: current server operator, API, developer, and future-client documentation.

- [ ] **Step 1: Add a failing readiness matrix**

```python
@pytest.mark.parametrize(
    "missing_component, blocker",
    [
        ("publication", "personal_context_publication_unavailable"),
        ("relay", "personal_context_relay_unavailable"),
        ("activation", "personal_context_activation_unavailable"),
        ("conflicts", "personal_context_conflicts_unavailable"),
        ("cleanup", "personal_context_cleanup_unavailable"),
        ("purge", "personal_context_purge_unavailable"),
    ],
)
def test_ongoing_v1_requires_every_component(missing_component, blocker, readiness) -> None:
    capabilities = readiness.with_component_disabled(missing_component).capabilities()
    assert capabilities.ongoing_sync_version == 0
    assert blocker in capabilities.ongoing_sync_blockers
```

- [ ] **Step 2: Run readiness tests and confirm RED**

Run: `.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_personal_context_readiness.py -q`

Expected: version 1 is never advertised and component-specific blockers are absent.

- [ ] **Step 3: Compute readiness from operational components**

```python
def ongoing_sync_readiness(self, *, user_id: str) -> tuple[int, tuple[str, ...]]:
    blockers = tuple(
        code
        for code, ready in self._personal_context_component_checks(user_id)
        if not ready
    )
    return (1, ()) if not blockers else (0, blockers[:8])
```

Checks must prove schema, profile key custody, publication journal, reserved authority identity, relay, activation store, continuity proof, batched conflict extension, cleanup journal, and purge fence. Configuration booleans alone are insufficient. Existing first-link fields and unrelated domain capability output stay compatible.

- [ ] **Step 4: Update canonical and generated documentation**

Document event-triggered rather than real-time delivery, activation, continuity failures, pull-time relay budgets, conflict action mapping, privacy cleanup state, device removal, global purge, recovery limits, and the exact schema/manifest vendoring procedure for future clients. Regenerate only with:

Run: `./Helper_Scripts/refresh_docs_published.sh`

- [ ] **Step 5: Run the server certification set**

```bash
.venv/bin/python -m pytest \
  tldw_Server_API/tests/Personalization/test_personal_context_contract.py \
  tldw_Server_API/tests/Personalization/test_personal_context_repository.py \
  tldw_Server_API/tests/Personalization/test_personal_context_service.py \
  tldw_Server_API/tests/Personalization/test_personal_context_plaintext_canary.py \
  tldw_Server_API/tests/Personalization/integration/test_personal_context_composed_app.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_adapter.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_materializer.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_transport.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_ongoing_contract.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_relay.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_activation.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_conflicts.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_purge.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_readiness.py -q
.venv/bin/ruff check tldw_Server_API/app/core/Personalization tldw_Server_API/app/core/Sync/v2 tldw_Server_API/app/api/v1/schemas/sync_v2_models.py
.venv/bin/python -m bandit -q -lll -r tldw_Server_API/app/core/Personalization tldw_Server_API/app/core/Sync/v2
./Helper_Scripts/check_top_guides_docs_path_hygiene.py
git diff --check
```

Expected: targeted server certification passes; regenerated docs are stable on a second run; blockers accurately force version 0; complete readiness reports version 1.

- [ ] **Step 6: Commit TASK-13165**

```bash
git add tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/factory.py \
  Docs/User_Guides/Server/Personal_Context_Profile.md \
  Docs/Code_Documentation/Personal_Context_Developer_Guide.md \
  Docs/API-related/Personal_Context_API.md \
  Docs/Published/User_Guides/Server/Personal_Context_Profile.md \
  Docs/Published/Code_Documentation/Personal_Context_Developer_Guide.md \
  Docs/Published/API-related/Personal_Context_API.md \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_readiness.py \
  tldw_Server_API/tests/Personalization/integration/test_personal_context_composed_app.py \
  backlog/tasks/task-13165\ -\ Enable-and-document-server-Personal-Context-ongoing-sync-v1.md
git commit -m "feat(sync): enable Personal Context ongoing v1"
```

## Plan self-review

- Spec coverage: whole-batch activation, activation-covered terminal proof, continuity, downgrade, candidate delivery, authoritative batch resolution, narrow freezes, cleanup acknowledgments, generation fencing, purge barrier, crypto-shredding, readiness, and server documentation are assigned.
- Placeholder scan: no deferred implementation markers remain.
- Type consistency: every task consumes the contract, journal, relay, and exchange proof produced by the preceding server plan; version 1 is enabled only after both conflict and purge branches complete.
