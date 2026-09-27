# Personal Context Relay Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the reopened TASK-13161 through seven separately reviewable remediation gates without activating ongoing Personal Context synchronization.

**Architecture:** Preserve `Personalization.db` as the only canonical profile authority and Sync V2 as encrypted transport. First make canonical receipts and authority envelopes prove complete immutable identity, then make the cross-database relay recoverable at every crash boundary. Bind cryptographic cleanup to a durable authorization minted only by the authenticated direct full-profile purge, consolidate pull recovery under one exact budget, narrow activation checks to the selected Personal Context operation, and finish with production-factory certification.

**Tech Stack:** Python 3.11, Pydantic v2, SQLite, FastAPI/TestClient, Sync V2, AES-GCM, HMAC-SHA256, pytest, Ruff, Bandit.

**Spec:** `Docs/superpowers/specs/2026-09-02-personal-context-ongoing-sync-design.md`

**Governing ADR:** `backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md`

## Why TASK-13161 is split

The fifth review of TASK-13161 rejected completion for independent defects in identity binding, legacy receipt validation, crash recovery, failure classification, pull budgeting, conflict gating, production coverage, and stale-generation retention. Continuing to patch all of them inside one task made review state and completion claims unreliable. TASK-13161 remains In Progress; TASK-13166 through TASK-13172 are the remediation gates, and TASK-13172 is the only task allowed to close it.

## Global constraints

- ADR required: no new ADR. ADR-002 and the approved ongoing-sync specification already decide authority, cross-database journals, encryption, purge fencing, and activation. If implementation reveals a genuinely different custody or deletion model, stop and amend the ADR before code proceeds.
- Keep `ongoing_sync_version = 0`. Do not implement or activate TASK-13162 through TASK-13165 in this plan.
- `Personalization.db` remains canonical. Sync V2 stores encrypted ingress, hidden staging, authority delivery, cursors, and content-free receipts only.
- The user authorized irreversible cryptographic shredding only as execution of an authenticated, explicitly confirmed direct **Delete Entire Profile** operation. Ordinary pull, relay, compaction, listing, and mutation cannot mint that authority.
- A restart may resume a durable purge-cleanup intent only when that intent was atomically minted by the explicitly authorized direct purge. Recovery must not infer authorization from a generation mismatch alone.
- Authenticated pull may return only encrypted, applied home-authority Personal Context envelopes for the same linked profile and generation. Client-ingress, hidden pending authority, plaintext bodies, wrapped keys, and content-derived diagnostics never egress.
- Poison means authenticated canonical-source corruption. Adapter conflicts, current-head contention, lease loss, database failures, deadlines, and process interruption are retryable.
- The exact recovery ceiling is 100 total inspected source-plus-Sync rows and 100 milliseconds per pull. Selection and decryption consume the same budget as relay and raw scanning.
- Use the shared environment from this worktree as `../../.venv/bin/python`. Run targeted tests only; do not run the full repository suite unless the user explicitly requests it.
- At the start of each task: move its Backlog item to In Progress and add that task's implementation plan. At completion: check every acceptance criterion, add concise implementation notes, record exact verification, run self-review, and only then mark it Done.
- Commit after each task. Do not push, open a PR, rebase, or merge as part of this plan.

---

### Task 1: TASK-13166 — Bind Personal Context authority confirmation identity

**Files:**

- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/Sync_DB.py`
- Modify: `tldw_Server_API/app/core/Personalization/personal_context_publication.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/server_origin.py` only if the insert seam must accept an explicit expected identity
- Create: `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_authority_identity.py`
- Update: `backlog/tasks/task-13166 - Bind-Personal-Context-authority-confirmation-identity.md`

**Interfaces:**

- Add `SyncV2Store.get_personal_context_ingress_receipt(server_cursor: int) -> Mapping[str, Any] | None` and the matching `SyncDatabase` query.
- Add `PersonalContextPublicationRelayStore.canonical_ingress_receipt_for_source(row)` so the relay can compare Sync's ingress receipt with the canonical Personalization receipt that originated the publication batch.
- Add one private normalized authority attestation helper in `service.py`; keep it local to authority staging rather than creating a general Sync abstraction.
- Change `_is_exact_ingress_confirmation()` to consume the source publication row, canonical receipt, and stored Sync ingress receipt, not merely the current head and decrypted bytes.
- Do not weaken `_evaluate_envelope()` or normal current-head CAS behavior.

- [ ] **Step 1: Put TASK-13166 In Progress and record its plan**

```bash
backlog task edit 13166 -s "In Progress" -a @codex
backlog task edit 13166 --plan "1. Add RED identity and tamper tests. 2. Persist and read the exact ingress receipt. 3. Compare the complete immutable authority fingerprint and canonical receipt binding. 4. Run targeted security and regression checks. 5. Self-review and close the task. ADR required: no new ADR; ADR-002 governs."
```

- [ ] **Step 2: Add RED tests for complete replay identity**

Create a real two-database fixture that commits a canonical ingress receipt, stages its authority replay, and can mutate one persisted Sync field at a time. Parameterize at least:

```python
@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("base_server_cursor", 991),
        ("base_object_revision", 77),
        ("base_object_hash", "sha256:" + "0" * 64),
        ("object_revision", 88),
        ("stable_key", "tampered-stable-key"),
        ("client_sequence", 991),
        ("client_timestamp", "2099-01-01T00:00:00Z"),
        ("mutation_group_id", "tampered-group"),
    ],
)
def test_existing_authority_row_rejects_any_immutable_drift(
    authority_harness, field: str, changed: object
) -> None:
    receipt = authority_harness.commit_ingress()
    cursor = authority_harness.stage_without_source_ack(receipt.publication_batch_id)
    authority_harness.tamper_sync_envelope(cursor, field, changed)

    with pytest.raises(SyncStoreError, match="authority receipt is invalid"):
        authority_harness.retry_stage(receipt.publication_batch_id)

    assert authority_harness.source_row_state() == "pending"
```

Add parallel cases for dependency IDs, profile ID, generation, encryption policy/key version/wrapped-DEK metadata, routing metadata, authority batch facts, originating device, client envelope ID, canonical digest, resulting object/version, manifest revision/version, source sequence, and batch ordinal/size. Assert no source acknowledgement and no poison for every tampered Sync row.

- [ ] **Step 3: Run the focused test and confirm RED**

```bash
../../.venv/bin/python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_authority_identity.py -q
```

Expected: at least the currently omitted lineage and receipt-binding cases fail.

- [ ] **Step 4: Authenticate one complete normalized immutable fingerprint**

At first insert, protect the payload, construct the final authority metadata, then HMAC the complete immutable persisted envelope with the canonical profile integrity key. Store the tag in routing metadata as `authority_envelope_tag`. On deterministic reuse, verify the tag over the persisted row before accepting any of its lineage as fact:

```python
def _authority_attested_fields(
    envelope: SyncEnvelope | SyncEnvelopeCreate,
    authority: PersonalContextAuthorityMetadata,
) -> tuple[object, ...]:
    return (
        envelope.dataset_id,
        envelope.client_envelope_id,
        envelope.device_id,
        envelope.domain,
        envelope.operation,
        envelope.object_id,
        envelope.base_server_cursor,
        envelope.base_object_revision,
        envelope.base_object_hash,
        envelope.object_revision,
        envelope.stable_key,
        envelope.parent_id,
        envelope.base_version,
        envelope.entity_version,
        envelope.schema_version,
        envelope.adapter_version,
        envelope.client_sequence,
        envelope.client_timestamp,
        tuple(envelope.dependencies),
        envelope.mutation_group_id,
        envelope.deleted,
        envelope.payload_hash,
        envelope.payload_size_bytes,
        _normalized_encryption_metadata(envelope.encryption_metadata),
        _routing_metadata_without_authority_tag(envelope.routing_metadata),
        authority.role,
        authority.publication_batch_id,
        authority.profile_publication_sequence,
        authority.batch_ordinal,
        authority.batch_size,
    )
```

Use the real model field names discovered during implementation; the tuple above is the required semantic set, not permission to add duplicate fields. The normalized encryption metadata includes policy, algorithm, and key version. Random nonce, ciphertext, and wrapped-DEK bytes are authenticated by successful envelope decryption and are not regenerated for equality on retry. Include the canonical payload digest in the tag input. This attestation closes the existing AAD gap for lineage fields while preserving randomized AES-GCM encryption. A retry verifies the persisted tag and source binding; it does not compare a newly randomized ciphertext package with the original.

- [ ] **Step 5: Bind ingress confirmation to its canonical receipt**

Read `sync_personal_context_ingress_receipts` by the current ingress cursor. Separately read the canonical `CanonicalApplyReceipt` from Personalization by the source publication batch/sequence/result identity, and require exact equality among both receipts, the ingress head, and the source row:

```python
def _is_exact_ingress_confirmation(
    self,
    *,
    dataset: SyncDataset,
    current_head: SyncEnvelope | None,
    envelope: SyncEnvelopeCreate,
    canonical: bytes,
    source_row: PublicationSourceRow,
    canonical_receipt: CanonicalApplyReceipt | None,
    sync_ingress_receipt: Mapping[str, Any] | None,
) -> bool:
    if current_head is None or canonical_receipt is None or sync_ingress_receipt is None:
        return False
    return _receipt_binds_ingress_and_source(
        current_head=current_head,
        authority=envelope,
        source_row=source_row,
        canonical_receipt=canonical_receipt,
        sync_receipt=sync_ingress_receipt,
        canonical=canonical,
    )
```

The binding must prove dataset, originating device and client envelope, ingress cursor/status/role, canonical digest, profile/generation, resulting object/internal version, wire version, manifest revision/version, publication batch/sequence, and exact authority payload. A pending ingress may confirm only when the same authenticated canonical receipt exists; an applied ingress must match that same stored receipt.

- [ ] **Step 6: Run regression and security checks**

```bash
../../.venv/bin/python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_authority_identity.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_relay.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_materializer.py -q
../../.venv/bin/python -m ruff check \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/app/core/Personalization/personal_context_publication.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_authority_identity.py
../../.venv/bin/python -m bandit -q \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/app/core/Personalization/personal_context_publication.py
git diff --check
```

- [ ] **Step 7: Complete task hygiene, self-review, and commit**

Review the diff specifically for omitted mutable metadata and content-bearing diagnostics. Update TASK-13166 ACs, verification, notes, status, then commit:

```bash
git add tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/app/core/Personalization/personal_context_publication.py \
  tldw_Server_API/app/core/Sync/v2/server_origin.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_authority_identity.py \
  backlog/tasks/task-13166\ -\ Bind-Personal-Context-authority-confirmation-identity.md
git commit -m "fix(sync): bind Personal Context authority identity"
```

---

### Task 2: TASK-13167 — Harden legacy Personal Context receipt backfill

**Files:**

- Modify: `tldw_Server_API/app/core/Personalization/personal_context_publication.py`
- Modify: `tldw_Server_API/app/core/DB_Management/Personal_Context_Repository.py` only if a bounded current-manifest query is not already exposed
- Create: `tldw_Server_API/tests/Personalization/test_personal_context_receipt_backfill.py`
- Update: `backlog/tasks/task-13167 - Harden-legacy-Personal-Context-receipt-backfill.md`

**Interfaces:**

- Keep `PersonalContextPublicationJournal.read_ingress_receipt(connection, identity)` as the public seam.
- Extract a private `_validate_legacy_receipt_source()` helper returning the exact wire version to CAS, or raising the existing content-free replay error.
- Backfill only `wire_entity_version = ''`; no other legacy column is repaired opportunistically.

- [ ] **Step 1: Put TASK-13167 In Progress and record its plan**

```bash
backlog task edit 13167 -s "In Progress" -a @codex
backlog task edit 13167 --plan "1. Add RED legacy mismatch tests. 2. Validate the complete stored receipt against decrypted source and current manifest. 3. Backfill the empty wire identity with one checked CAS. 4. Run targeted security and regression checks. 5. Self-review and close the task. ADR required: no new ADR; ADR-002 governs."
```

- [ ] **Step 2: Add RED legacy validation matrix**

Build a valid old receipt with empty `wire_entity_version`, then mutate one fact per test. Cover receipt ID, profile/generation, batch sequence and size, result object/version, manifest revision/version, source role/domain/operation, manifest sibling, current manifest, digest, ciphertext, and encryption key.

```python
def test_legacy_backfill_rejects_stale_manifest_without_mutating_receipt(harness) -> None:
    identity = harness.insert_valid_legacy_receipt()
    harness.tamper_historical_manifest_revision_without_changing_legacy_receipt()

    with pytest.raises(ValueError, match="identity reused"):
        harness.journal.read_ingress_receipt(harness.connection, identity)

    assert harness.stored_wire_entity_version(identity) == ""
```

- [ ] **Step 3: Run the focused test and confirm RED**

```bash
../../.venv/bin/python -m pytest \
  tldw_Server_API/tests/Personalization/test_personal_context_receipt_backfill.py -q
```

- [ ] **Step 4: Validate all legacy facts before mutation**

Within the caller's existing transaction:

1. Select the exact receipt identity.
2. Select the exact batch and result source row plus its manifest sibling.
3. Authenticate and decrypt both rows.
4. Parse their canonical domains and bodies.
5. Compare source role/operation/object/version, batch generation/sequence/size, digest, result IDs, and the authenticated historical manifest row's revision/version/lineage. If that historical version is still the current manifest head, require exact current-head agreement; a later valid manifest head does not invalidate an older idempotent receipt.
6. Only then CAS the empty wire version.

```python
updated = connection.execute(
    """
    UPDATE personal_context_ingress_receipts
       SET wire_entity_version = ?
     WHERE dataset_id = ? AND device_id = ? AND client_envelope_id = ?
       AND wire_entity_version = ''
    """,
    (identity.wire_entity_version, identity.dataset_id,
     identity.device_id, identity.client_envelope_id),
)
if updated.rowcount != 1:
    raise ValueError("ingress identity reused with a different payload")
```

Never catch validation failure and continue. Never return a receipt synthesized from the request when persisted or decrypted facts disagree.

- [ ] **Step 5: Run regression and security checks**

```bash
../../.venv/bin/python -m pytest \
  tldw_Server_API/tests/Personalization/test_personal_context_receipt_backfill.py \
  tldw_Server_API/tests/Personalization/test_personal_context_publication.py -q
../../.venv/bin/python -m ruff check \
  tldw_Server_API/app/core/Personalization/personal_context_publication.py \
  tldw_Server_API/tests/Personalization/test_personal_context_receipt_backfill.py
../../.venv/bin/python -m bandit -q \
  tldw_Server_API/app/core/Personalization/personal_context_publication.py
git diff --check
```

- [ ] **Step 6: Complete task hygiene, self-review, and commit**

```bash
git add tldw_Server_API/app/core/Personalization/personal_context_publication.py \
  tldw_Server_API/app/core/DB_Management/Personal_Context_Repository.py \
  tldw_Server_API/tests/Personalization/test_personal_context_receipt_backfill.py \
  backlog/tasks/task-13167\ -\ Harden-legacy-Personal-Context-receipt-backfill.md
git commit -m "fix(personal-context): harden legacy receipt backfill"
```

---

### Task 3: TASK-13168 — Make Personal Context relay staging crash-safe

**Files:**

- Modify: `tldw_Server_API/app/core/Personalization/personal_context_publication.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/personal_context_relay.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/Sync_DB.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/server_origin.py`
- Create: `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_relay_recovery.py`
- Update: `backlog/tasks/task-13168 - Make-Personal-Context-relay-staging-crash-safe.md`

**Interfaces:**

- Preserve deterministic `PublicationSourceRow.deterministic_envelope_id` as the cross-database recovery key.
- Add a bounded source query for unfinished staged identities, including terminalized/shredded metadata needed to compensate an orphan.
- Make `record_staged_row()`, `acknowledge_row()`, `complete_if_acknowledged()`, `mark_attention()`, lease renew/release, and authority finalization enforce owner/status/batch/generation CAS with checked row counts.
- Split `PersonalContextAuthoritySourceError` from retryable `SyncStoreError`/adapter/head failures; do not add multiple retry exception hierarchies.

- [ ] **Step 1: Put TASK-13168 In Progress and record its plan**

```bash
backlog task edit 13168 -s "In Progress" -a @codex
backlog task edit 13168 --plan "1. Add RED crash, CAS, and failure-classification tests. 2. Recover deterministic hidden staging from source identity. 3. Fence all relay transitions and compensate lost claims. 4. Keep only authenticated source corruption poisonable. 5. Run targeted security and concurrency checks. 6. Self-review and close the task. ADR required: no new ADR; ADR-002 governs."
```

- [ ] **Step 2: Add deterministic crash-boundary tests**

Inject a failure after each durable action: Sync insert, source `record_staged_row`, Sync finalization, source acknowledgement, and batch completion. Restart with new service/store instances over the same two database files.

```python
def test_restart_after_sync_insert_before_source_cursor_recovers_exactly_once(two_db_harness) -> None:
    two_db_harness.fail_once("after_authority_insert")
    first = two_db_harness.relay()
    assert first.continuation == "personal_context_relay_pending"
    assert two_db_harness.visible_authority_count() == 0

    recovered = two_db_harness.restart().relay()

    assert recovered.continuation == "complete"
    assert two_db_harness.authority_count_for_source() == 1
    assert two_db_harness.visible_authority_count() == 1
    assert two_db_harness.source_row_state() == "acknowledged"
```

Add two-relay-instance lease races, lease expiry during slow stage, source purge after orphan insert, current-head contention, adapter rejection, database exception, and authenticated corrupt source. Capture logs and scan Personalization DB, Sync DB, and WAL files for unique plaintext canaries.

- [ ] **Step 3: Run the focused test and confirm RED**

```bash
../../.venv/bin/python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_relay_recovery.py -q
```

- [ ] **Step 4: Recover hidden staging by deterministic source identity**

Before inserting, look up the exact deterministic envelope. After insert, return a structured stage receipt whose identity can be checked on retry:

```python
@dataclass(frozen=True, slots=True)
class AuthorityStageReceipt:
    server_cursor: int
    deterministic_envelope_id: str
    publication_batch_id: str
    profile_publication_sequence: int
    batch_ordinal: int
    purge_generation: int
```

If source state still owns the row, bind the recovered cursor through `record_staged_row()`. If the source has been terminalized by the authorized purge fence, call the exact pending-authority compensation seam and repair the current head. Never expose a pending authority row merely to make recovery easier.

- [ ] **Step 5: Fence every state transition**

Each mutator must include the immutable profile/batch/sequence/ordinal/generation plus the expected prior state and live owner token where ownership applies. A zero or multi-row update is a retryable race, not success. Authority finalization must require `apply_status='pending'`, the exact stage identity, and an acknowledged source receipt; idempotent replay may accept the already-applied exact row.

```python
def finalize_personal_context_authority(
    self,
    row: PublicationSourceRow,
    receipt: AuthorityStageReceipt,
    dataset_id: str,
    user_id: str,
) -> None:
    """CAS one exact hidden authority row from pending to applied."""
```

Keep compensation exact and idempotent. It may remove only the matching hidden pending row and may repair only a current head that still points to that row.

- [ ] **Step 6: Correct poison classification**

`stage_personal_context_authority()` raises `PersonalContextAuthoritySourceError` only after source authentication/decryption/integrity or canonical-shape validation proves corruption. `_evaluate_envelope()` conflicts, stale heads, database errors, lease races, and injected failures remain pending. Add a captured-log assertion proving no warning/error labels a retryable head conflict as poisoned.

- [ ] **Step 7: Run regression, concurrency, and security checks**

```bash
../../.venv/bin/python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_relay_recovery.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_authority_identity.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_relay.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_materializer.py -q
../../.venv/bin/python -m ruff check \
  tldw_Server_API/app/core/Personalization/personal_context_publication.py \
  tldw_Server_API/app/core/Sync/v2/personal_context_relay.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_relay_recovery.py
../../.venv/bin/python -m bandit -q \
  tldw_Server_API/app/core/Personalization/personal_context_publication.py \
  tldw_Server_API/app/core/Sync/v2/personal_context_relay.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/DB_Management/Sync_DB.py
git diff --check
```

- [ ] **Step 8: Complete task hygiene, self-review, and commit**

```bash
git add tldw_Server_API/app/core/Personalization/personal_context_publication.py \
  tldw_Server_API/app/core/Sync/v2/personal_context_relay.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/app/core/Sync/v2/server_origin.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_relay_recovery.py \
  backlog/tasks/task-13168\ -\ Make-Personal-Context-relay-staging-crash-safe.md
git commit -m "fix(sync): recover Personal Context relay staging"
```

---

### Task 4: TASK-13169 — Purge stale Personal Context Sync ciphertext safely

**Files:**

- Modify: `tldw_Server_API/app/core/DB_Management/Personalization_DB.py`
- Modify: `tldw_Server_API/app/core/DB_Management/Personal_Context_Repository.py`
- Modify: `tldw_Server_API/app/core/Personalization/personal_context_service.py`
- Modify: `tldw_Server_API/app/api/v1/API_Deps/personal_context_deps.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/Sync_DB.py`
- Test: `tldw_Server_API/tests/Personalization/test_personal_context_publication.py`
- Create: `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_purge_retention.py`
- Update: `backlog/tasks/task-13169 - Purge-stale-Personal-Context-Sync-ciphertext-safely.md`

**Interfaces:**

- Add a content-free `personal_context_purge_cleanup_intents` journal in Personalization storage. It is inserted only when `journal_destruction_authorization is _DIRECT_CONFIRMED_FULL_PROFILE_PURGE`.
- Add `PersonalContextRepository.claim_direct_purge_cleanup()` and `complete_direct_purge_cleanup()` with idempotent owner fencing.
- Add `SyncV2Service.shred_authorized_personal_context_history(intent)` backed by one exact `SyncDatabase` transaction.
- Add an after-commit purge-cleanup callback separate from the ordinary relay callback. Do not let pull or relay mint or execute an unjournaled deletion.

- [ ] **Step 1: Put TASK-13169 In Progress and record its plan**

```bash
backlog task edit 13169 -s "In Progress" -a @codex
backlog task edit 13169 --plan "1. Add RED direct-purge authorization and ciphertext-retention tests. 2. Journal cleanup authority only in the confirmed direct purge transaction. 3. Shred old-generation Sync payload/key material idempotently and repair barriers. 4. Keep pull and relay non-destructive. 5. Inventory application-owned backup boundaries and verify canaries. 6. Self-review and close the task. ADR required: no new ADR; ADR-002 and the approved spec govern."
```

- [ ] **Step 2: Add RED authorization and retention tests**

Cover direct confirmed purge, wrong confirmation, client-originated purge barrier application, pull, relay, compaction, retry after cleanup failure, other profiles/datasets, current generation, pending/orphan authority, ingress receipt, conflicts, WAL, and key rotation.

```python
def test_only_confirmed_direct_purge_mints_sync_cleanup_authority(purge_harness) -> None:
    purge_harness.apply_remote_purge_barrier()
    assert purge_harness.cleanup_intents() == []

    purge_harness.direct_purge(confirmation="DELETE EVERYWHERE")

    intent = purge_harness.only_cleanup_intent()
    assert intent.origin == "direct_confirmed_full_profile_purge"
    assert intent.purge_generation == 1
```

Keep one unique canary in each old-generation authority, ingress, conflict candidate, source row, and wrapped-DEK package. After cleanup, the active service and retained active-store/WAL artifacts must not decrypt or expose it.

- [ ] **Step 3: Run the focused test and confirm RED**

```bash
../../.venv/bin/python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_purge_retention.py -q
```

- [ ] **Step 4: Mint a durable cleanup intent only in direct purge**

The canonical purge transaction must atomically advance generation, shred canonical/source bodies as already authorized, publish the barrier, and insert a content-free cleanup intent. A remote Sync purge application does not receive the private capability and therefore cannot insert this intent.

```python
@dataclass(frozen=True, slots=True)
class DirectPurgeCleanupIntent:
    intent_id: str
    profile_id: str
    old_generation_through: int
    purge_generation: int
    state: Literal["pending", "claimed", "complete"]
    owner_token: str | None
```

The intent contains no profile body, semantic identity, or key bytes. The direct endpoint's service callback may execute it after commit; retrying the same explicit purge resumes the same intent. A dedicated startup recovery path may resume previously minted intents, but generic pull/relay entry points may not call it.

- [ ] **Step 5: Shred old Sync payload and key material transactionally**

Within a dataset/profile/generation-scoped Sync transaction:

- overwrite or remove old-generation encrypted payload packages, wrapped DEKs, nonces, protected authority/conflict candidates, and any stored recovery key record that can decrypt those generations;
- retain only content-free IDs, generation fences, timestamps, statuses, and acknowledgements needed for idempotency;
- remove/repair stale current-head pointers and pending barriers without changing unrelated domain heads;
- preserve the new purge barrier and current generation;
- check every affected-row expectation and make retries converge.

```python
def shred_personal_context_profile_history(
    self,
    *,
    dataset_id: str,
    profile_id: str,
    old_generation_through: int,
    purge_generation: int,
) -> PersonalContextHistoryShredReceipt:
    """Irreversibly remove readable old-generation Sync material."""
```

Before implementation, inventory application-owned backups and migration snapshots. The specification permits encrypted external/operator backups but does not permit a false claim that an old wrapped key is unrecoverable if an application-owned backup still contains both it and usable custody. Delete or invalidate application-owned copies inside the documented purge boundary; if that requires a new custody model, stop for an ADR amendment.

- [ ] **Step 6: Keep ordinary recovery non-destructive**

`scan_personal_context_authority()` must skip stale generations and continue past their hidden rows without deleting them or moving a current-generation cursor incorrectly. Relay may compensate one exact orphan pending row under TASK-13168's recovery contract, but broad history shredding requires the durable direct-purge intent.

- [ ] **Step 7: Run regression and security checks**

```bash
../../.venv/bin/python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_purge_retention.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_relay_recovery.py \
  tldw_Server_API/tests/Personalization/test_personal_context_publication.py \
  tldw_Server_API/tests/Personalization/test_personal_context_service.py -q
../../.venv/bin/python -m ruff check \
  tldw_Server_API/app/core/DB_Management/Personalization_DB.py \
  tldw_Server_API/app/core/DB_Management/Personal_Context_Repository.py \
  tldw_Server_API/app/core/Personalization/personal_context_service.py \
  tldw_Server_API/app/api/v1/API_Deps/personal_context_deps.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_purge_retention.py
../../.venv/bin/python -m bandit -q \
  tldw_Server_API/app/core/DB_Management/Personalization_DB.py \
  tldw_Server_API/app/core/DB_Management/Personal_Context_Repository.py \
  tldw_Server_API/app/core/Personalization/personal_context_service.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/DB_Management/Sync_DB.py
git diff --check
```

- [ ] **Step 8: Complete task hygiene, self-review, and commit**

Self-review the authorization provenance, backup-boundary statement, and every destructive SQL predicate. Update the task and commit only after proving other datasets/profiles remain byte-for-byte unchanged.

```bash
git add tldw_Server_API/app/core/DB_Management/Personalization_DB.py \
  tldw_Server_API/app/core/DB_Management/Personal_Context_Repository.py \
  tldw_Server_API/app/core/Personalization/personal_context_service.py \
  tldw_Server_API/app/api/v1/API_Deps/personal_context_deps.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/tests/Personalization/test_personal_context_publication.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_purge_retention.py \
  backlog/tasks/task-13169\ -\ Purge-stale-Personal-Context-Sync-ciphertext-safely.md
git commit -m "fix(sync): shred purged Personal Context history"
```

---

### Task 5: TASK-13170 — Unify bounded Personal Context relay recovery

**Files:**

- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/store.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/personal_context_relay.py`
- Modify: `tldw_Server_API/app/core/Personalization/personal_context_publication.py`
- Create: `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_recovery_budget.py`
- Update: `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_transport.py`
- Update: `backlog/tasks/task-13170 - Unify-bounded-Personal-Context-relay-recovery.md`

**Interfaces:**

- Replace the passive frozen `_PersonalContextRecovery` values with one mutable budget object used by source lookup, source decryption, relay, raw scan, and page lookahead.
- Pass an absolute `deadline_ns` and budget object into `earliest_nonterminal_batch()`, `PersonalContextRelay.relay_profile()`, and `scan_personal_context_authority()`.
- Keep one `_coordinate_personal_context_recovery()` call for legacy, signed, mixed, and subset pulls.

- [ ] **Step 1: Put TASK-13170 In Progress and record its plan**

```bash
backlog task edit 13170 -s "In Progress" -a @codex
backlog task edit 13170 --plan "1. Add RED exact-budget and watermark tests. 2. Introduce one shared row/deadline budget. 3. Route all Personal Context pull modes through it. 4. Correct page-plus-one and mixed-stream cursor handling. 5. Run targeted regression and security checks. 6. Self-review and close the task. ADR required: no new ADR; ADR-002 governs."
```

- [ ] **Step 2: Add exact 100/101 and deadline tests**

Test source-only, raw-only, and combined partitions such as 40+60, 99+1, and 100+1. Assert the 100th inspected row returns a valid continuation and no source method receives zero. Inject a monotonic clock that expires during source selection and decryption.

```python
@pytest.mark.parametrize(("source_rows", "raw_rows"), [(100, 0), (0, 100), (40, 60)])
def test_exact_hundred_rows_is_a_valid_bounded_attempt(
    recovery_harness, source_rows: int, raw_rows: int
) -> None:
    recovery_harness.seed(source_rows=source_rows, raw_rows=raw_rows)
    result = recovery_harness.pull()
    assert recovery_harness.total_inspected == 100
    assert result.personal_context_relay.state in {
        "complete", "personal_context_relay_pending"
    }
    assert 0 not in recovery_harness.source_limits
```

- [ ] **Step 3: Run the focused test and confirm RED**

```bash
../../.venv/bin/python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_recovery_budget.py -q
```

- [ ] **Step 4: Implement one consumable budget**

```python
@dataclass(slots=True)
class _PersonalContextRecoveryBudget:
    remaining_rows: int
    deadline_ns: int
    clock_ns: Callable[[], int] = monotonic_ns

    def can_inspect(self) -> bool:
        return self.remaining_rows > 0 and self.clock_ns() < self.deadline_ns

    def consume(self) -> None:
        if not self.can_inspect():
            raise PersonalContextRecoveryBudgetExhausted
        self.remaining_rows -= 1
```

Consume before decrypting each selected source row and before classifying each raw Sync row. Never derive a second 100-row allowance. When `remaining_rows == 0`, return the bounded continuation without invoking another source query.

- [ ] **Step 5: Correct lookahead and safe watermarks**

Fetch page-plus-one only inside the remaining budget. Advance a stream's raw watermark past safe hidden ingress, but never past a pending authority barrier, unresolved conflict barrier, expired deadline boundary, or uninspected row. Preserve unrelated Notes delivery and the requested domain subset. Restore encrypted authority only after it has passed role, status, profile, generation, key, and activation checks.

- [ ] **Step 6: Run the transport matrix and security checks**

```bash
../../.venv/bin/python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_recovery_budget.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_transport.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_relay_recovery.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py -q
../../.venv/bin/python -m ruff check \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/Sync/v2/personal_context_relay.py \
  tldw_Server_API/app/core/Personalization/personal_context_publication.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_recovery_budget.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_transport.py
../../.venv/bin/python -m bandit -q \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/Sync/v2/personal_context_relay.py
git diff --check
```

- [ ] **Step 7: Complete task hygiene, self-review, and commit**

```bash
git add tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/Sync/v2/personal_context_relay.py \
  tldw_Server_API/app/core/Personalization/personal_context_publication.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_recovery_budget.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_transport.py \
  backlog/tasks/task-13170\ -\ Unify-bounded-Personal-Context-relay-recovery.md
git commit -m "fix(sync): bound Personal Context pull recovery"
```

---

### Task 6: TASK-13171 — Enforce active Personal Context exchange without breaking Sync V2

**Files:**

- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/Sync_DB.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/sync.py`
- Create: `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_exchange_gate.py`
- Update: `tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py`
- Update: `backlog/tasks/task-13171 - Enforce-active-Personal-Context-exchange-without-breaking-Sync-V2.md`

**Interfaces:**

- Keep `require_active_exchange()` as the one exact proof validator.
- Extend conflict listing with optional backward-compatible `domain` and `device_id` query parameters; thread the domain filter through service/store/database queries.
- Replace dataset-wide conflict-list gating with a selected-domain/selected-page gate that always receives the real requesting device ID for Personal Context.
- Preserve the request and response models created by TASK-13159; only the optional GET query parameters are added.

- [ ] **Step 1: Put TASK-13171 In Progress and record its plan**

```bash
backlog task edit 13171 -s "In Progress" -a @codex
backlog task edit 13171 --plan "1. Add RED TestClient proof and mixed-conflict tests. 2. Centralize selected-operation Personal Context detection. 3. Require exact proof plus completed device link only for selected Personal Context work. 4. Preserve version-zero and unrelated Sync behavior. 5. Run targeted endpoint and security checks. 6. Self-review and close the task. ADR required: no new ADR; ADR-002 governs."
```

- [ ] **Step 2: Add production TestClient gate matrix**

Cover push, pull, conflict list, and conflict resolve with exact proof, missing proof, stale epoch, stale token, wrong device, incomplete link, tampered stored proof, and version zero. Build a mixed dataset containing unresolved Notes and Personal Context conflicts.

```python
def test_mixed_dataset_lists_notes_conflict_without_personal_context_proof(client, mixed_dataset) -> None:
    response = client.get(
        "/api/v1/sync/conflicts",
        params={"dataset_id": mixed_dataset.id, "domain": "notes"},
        headers=mixed_dataset.auth_headers,
    )
    assert response.status_code == 200
    assert [item["domain"] for item in response.json()["conflicts"]] == ["notes"]
```

Add the converse: listing or resolving a selected Personal Context conflict without exact proof and completed link returns `personal_context_activation_required` before cursor advancement or mutation.

- [ ] **Step 3: Run the focused test and confirm RED**

```bash
../../.venv/bin/python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_exchange_gate.py -q
```

- [ ] **Step 4: Gate the selected operation, not dataset membership**

For push and pull, derive whether the requested envelopes/streams include Personal Context. For conflict listing, fetch the bounded filtered page first, then require `device_id` and proof only if that selected page contains Personal Context; a `domain=notes` page from a mixed dataset remains legacy-compatible. For resolution, load the requested conflict IDs and gate only if any selected valid conflict is Personal Context.

```python
def verified_exchange_for_selected_conflicts(
    self,
    *,
    user_id: str,
    dataset_id: str,
    device_id: str,
    conflict_ids: Sequence[str],
    exchange: object | None,
) -> PersonalContextExchangeProof | None:
    conflicts = self._selected_dataset_conflicts(dataset_id, conflict_ids)
    if not any(item.domain in PERSONAL_CONTEXT_SYNC_DOMAINS for item in conflicts):
        return None
    return self.verified_active_exchange(
        user_id=user_id,
        dataset_id=dataset_id,
        device_id=device_id,
        exchange=exchange,
    )
```

Use `hmac.compare_digest()` for epoch/token and echo only the verified persisted proof. Never echo unverified request fields. Version-zero Personal Context work returns `personal_context_activation_required` before state change; legacy first-link and unrelated domains retain existing behavior.

- [ ] **Step 5: Run endpoint and security checks**

```bash
../../.venv/bin/python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_exchange_gate.py \
  tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_transport.py -q
../../.venv/bin/python -m ruff check \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/app/api/v1/endpoints/sync.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_exchange_gate.py \
  tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py
../../.venv/bin/python -m bandit -q \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/app/api/v1/endpoints/sync.py
git diff --check
```

- [ ] **Step 6: Complete task hygiene, self-review, and commit**

```bash
git add tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/app/api/v1/endpoints/sync.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_exchange_gate.py \
  tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py \
  backlog/tasks/task-13171\ -\ Enforce-active-Personal-Context-exchange-without-breaking-Sync-V2.md
git commit -m "fix(sync): scope Personal Context exchange gates"
```

---

### Task 7: TASK-13172 — Certify Personal Context relay and close TASK-13161

**Files:**

- Modify only if certification exposes a defect: production files owned by TASK-13166 through TASK-13171
- Create: `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_certification.py`
- Update: `tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py`
- Update: `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_transport.py`
- Update: `.superpowers/sdd/2026-09-03-personal-context-ongoing-sync-01-server-contract-publication/task-3-report.md`
- Update: `.superpowers/sdd/2026-09-03-personal-context-ongoing-sync-01-server-contract-publication/progress.md`
- Update when an incident yields reusable evidence: `backlog/docs/lessons-testing-evidence.md` or `backlog/docs/lessons-live-verification.md`
- Update: `backlog/tasks/task-13172 - Certify-Personal-Context-relay-and-close-TASK-13161.md`
- Update last, only after acceptance: `backlog/tasks/task-13161 - Relay-ordered-Personal-Context-authority-publications-through-Sync-V2.md`

**Interfaces:**

- Add no new production API for certification.
- Construct the application through the production Sync/Personal Context dependency and factory paths.
- Treat any production defect as a failing test plus the smallest fix in the task that owns the violated invariant; rerun that task's complete gate before certification continues.

- [ ] **Step 1: Put TASK-13172 In Progress and record its plan**

```bash
backlog task edit 13172 -s "In Progress" -a @codex
backlog task edit 13172 --plan "1. Add a production-factory RED certification flow. 2. Prove after-commit failure, restart debt, exact-once pull, and single-dataset invariants. 3. Run the complete targeted remediation matrix and static/security checks. 4. Request independent specification and code-quality reviews. 5. Correct any finding in its owning task and repeat review. 6. Close TASK-13161 only after all evidence is accepted. ADR required: no new ADR; ADR-002 governs."
```

- [ ] **Step 2: Add the production-factory certification flow**

Use real temporary Personalization and Sync database files, production `personal_context_service_for_user()`, `sync_v2_service_for_user()`, FastAPI dependency wiring, and `TestClient`. Prove:

1. direct canonical create/update commits a publication batch;
2. client ingress stays hidden while canonical application produces a receipt;
3. after-commit relay failure does not turn the accepted canonical commit into an HTTP failure;
4. new process/service instances recover durable debt before a later push or pull;
5. exactly one applied authority envelope is pulled and decrypts to the canonical contract object;
6. repeating pull advances safely and returns no duplicate;
7. no ingress/pending/plaintext state egresses;
8. one user/profile cannot be enrolled into multiple authoritative Personal Context datasets.

```python
def test_production_factory_recovers_after_commit_relay_failure_exactly_once(
    production_personal_context_app,
) -> None:
    app = production_personal_context_app.fail_next_relay_after_commit()
    accepted = app.client.post("/api/v1/personal-context/records", json=app.record_request())
    assert accepted.status_code == 200

    restarted = app.restart()
    pulled = restarted.pull_personal_context()

    assert [item["authority"]["role"] for item in pulled["envelopes"]] == ["home_authority"]
    assert restarted.count_authority_for_canonical_version() == 1
    assert restarted.pull_personal_context()["envelopes"] == []
```

- [ ] **Step 3: Run the focused certification test and confirm RED or GREEN**

```bash
../../.venv/bin/python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_certification.py -q
```

If RED, add the failure to the owning task's AC/notes before changing production code. Reopen that task if it was marked Done, implement the smallest TDD fix, and rerun its full gate.

- [ ] **Step 4: Run the complete targeted remediation matrix**

```bash
../../.venv/bin/python -m pytest \
  tldw_Server_API/tests/Personalization/test_personal_context_publication.py \
  tldw_Server_API/tests/Personalization/test_personal_context_receipt_backfill.py \
  tldw_Server_API/tests/Personalization/test_personal_context_service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_authority_identity.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_relay.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_relay_recovery.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_materializer.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_purge_retention.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_recovery_budget.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_exchange_gate.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_transport.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_certification.py \
  tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py -q
../../.venv/bin/python -m ruff check \
  tldw_Server_API/app/core/Personalization/personal_context_publication.py \
  tldw_Server_API/app/core/Personalization/personal_context_service.py \
  tldw_Server_API/app/core/DB_Management/Personal_Context_Repository.py \
  tldw_Server_API/app/core/Sync/v2/personal_context_relay.py \
  tldw_Server_API/app/core/Sync/v2/server_origin.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/app/api/v1/endpoints/sync.py \
  tldw_Server_API/app/api/v1/API_Deps/personal_context_deps.py \
  tldw_Server_API/tests/Personalization/test_personal_context_receipt_backfill.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_authority_identity.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_relay_recovery.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_purge_retention.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_recovery_budget.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_exchange_gate.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_certification.py
../../.venv/bin/python -m bandit -q \
  tldw_Server_API/app/core/Personalization/personal_context_publication.py \
  tldw_Server_API/app/core/Personalization/personal_context_service.py \
  tldw_Server_API/app/core/DB_Management/Personal_Context_Repository.py \
  tldw_Server_API/app/core/Sync/v2/personal_context_relay.py \
  tldw_Server_API/app/core/Sync/v2/server_origin.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/app/api/v1/endpoints/sync.py
git diff --check
```

Also scan the isolated DBs, WALs, logs, diagnostic artifacts, migration snapshots, and application-owned backup fixtures for each unique canary. Record the exact artifact paths and whether the canary is absent, encrypted, or explicitly content-free.

- [ ] **Step 5: Enforce the single authoritative dataset invariant**

Test both bootstrap/configuration and runtime resolution. Repeating enrollment into the same dataset is idempotent; attempting a second active Personal Context dataset for the same profile is rejected before key wrapping, mutation, or cursor creation. Dataset lookup must fail closed if legacy corruption already violates the invariant.

- [ ] **Step 6: Request independent reviews**

Use `superpowers:requesting-code-review` for two separate reviews:

- specification review: map every TASK-13161 and TASK-13166–TASK-13172 AC to code and executable evidence;
- code-quality/security review: inspect CAS ownership, failure classification, cryptographic deletion authority, budget accounting, egress filtering, and production factory wiring.

Do not mark a finding resolved from explanation alone. Add a failing regression test or concrete code/document evidence, fix in the owning task, rerun that gate and the certification matrix, then request review again.

- [ ] **Step 7: Close task records honestly**

After both reviews accept:

- check every TASK-13172 AC and DoD item;
- update the SDD report and progress ledger with exact commits, commands, counts, and review outcomes;
- add a lesson only if the remediation produced an incident-backed reusable rule;
- re-read TASK-13161 and check each original AC/DoD item against current evidence;
- add concise implementation notes to TASK-13161 and mark it Done only if no requirement remains.

- [ ] **Step 8: Commit certification and closure**

```bash
git add tldw_Server_API/tests/Sync/test_sync_v2_personal_context_certification.py \
  tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_transport.py \
  .superpowers/sdd/2026-09-03-personal-context-ongoing-sync-01-server-contract-publication/task-3-report.md \
  .superpowers/sdd/2026-09-03-personal-context-ongoing-sync-01-server-contract-publication/progress.md \
  backlog/tasks/task-13172\ -\ Certify-Personal-Context-relay-and-close-TASK-13161.md \
  backlog/tasks/task-13161\ -\ Relay-ordered-Personal-Context-authority-publications-through-Sync-V2.md
git commit -m "test(sync): certify Personal Context authority relay"
```

## Plan self-review

- **Finding coverage:** Complete authority identity and ingress receipt binding are isolated in TASK-13166; legacy receipt backfill in TASK-13167; crash recovery, CAS fencing, and poison classification in TASK-13168; direct-purge-only retention in TASK-13169; exact shared budgets and watermarks in TASK-13170; selected-operation proof gating in TASK-13171; production-factory, single-dataset, and combined evidence in TASK-13172.
- **Dependency safety:** TASK-13166 and TASK-13167 share only the receipt contract and can be reasoned about independently. TASK-13168 consumes both. Retention follows crash-safe staging; bounded recovery follows retention; endpoint gates follow recovery; certification consumes all six.
- **Authority safety:** No task activates ongoing sync. No task permits client home-authority submission, client-ingress egress, plaintext egress, or deletion authorization inferred from pull state.
- **Deletion realism:** The plan distinguishes active/application-owned artifacts from operator/external backups and explicitly stops for ADR work if existing custody makes the promised cryptographic deletion impossible.
- **Budget exactness:** Source selection, decryption, relay, raw scan, and lookahead all consume one 100-row/100-millisecond budget; zero-limit source calls are forbidden.
- **Review honesty:** TASK-13161 stays open until independent specification and quality reviews accept the integrated result. A reopened owning task is required for any certification defect.
- **Scope:** No Chatbook code, activation task, conflict-resolution feature, rollout flag, PR, rebase, push, or merge is included.

## Execution handoff

Two supported execution modes:

1. **Subagent-driven development (recommended):** execute one Backlog task at a time with a fresh implementation worker and independent specification/quality reviews at each gate.
2. **Inline execution:** execute this plan serially in the current task, stopping after each Backlog task for review.

Do not begin either mode until the user chooses it.
