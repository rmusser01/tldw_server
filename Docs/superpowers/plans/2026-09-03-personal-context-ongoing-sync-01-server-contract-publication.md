# Personal Context Ongoing Sync: Server Contract and Publication Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the server-owned versioned wire contract, encrypted canonical publication journal, and ordered Sync V2 relay for ongoing Personal Context synchronization.

**Architecture:** `Personalization.db` remains canonical and writes an encrypted publication batch in the same transaction as every eligible mutation. A separate relay claims those batches in global profile order and writes deterministic already-applied home-authority envelopes into Sync V2; client envelopes remain ingress-only until a canonical replay receipt exists. Contract models live on the server and produce a checked-in schema plus manifest that clients vendor exactly.

**Tech Stack:** Python 3.11, Pydantic v2, SQLite, FastAPI, Sync V2, AES-GCM envelope storage, pytest.

**Spec:** `Docs/superpowers/specs/2026-09-02-personal-context-ongoing-sync-design.md`

> **Remediation status (2026-09-04):** TASK-13161 was reopened after its fifth review found unresolved identity, legacy receipt, crash recovery, failure-classification, recovery-budget, conflict-gating, production-certification, and stale-ciphertext retention defects. Do not treat Task 3 below as complete. Execute `Docs/superpowers/plans/2026-09-04-personal-context-relay-remediation.md` (TASK-13166 through TASK-13172) before TASK-13161 can close.

## Global Constraints

- ADR required: no new ADR. Use `backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md`.
- Keep `Personalization.db` authoritative; Sync V2 is transport and conflict state only.
- Advertise `ongoing_sync_version = 0` throughout TASK-13159 through TASK-13161.
- Never store profile bodies, labels, semantic keys, proposals, or conflict candidates in plaintext persistence, logs, metrics, or error text.
- Never permit a registered client to submit or register the reserved home-authority identity.
- Preserve all existing Sync domains and first-link behavior.
- Use targeted tests for touched Personal Context and Sync paths; do not run the full repository suite unless the user requests it.

---

### Task 1: TASK-13159 — Version the ongoing-sync wire contract

**Files:**

- Create: `tldw_Server_API/app/core/Sync/v2/personal_context_ongoing_contract.py`
- Create: `tldw_Server_API/app/core/Sync/v2/contracts/personal-context-ongoing-v1.schema.json`
- Create: `tldw_Server_API/app/core/Sync/v2/contracts/personal-context-ongoing-v1.manifest.json`
- Create: `Helper_Scripts/generate_personal_context_ongoing_contract.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sync_v2_models.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/sync.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_ongoing_contract.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_models.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py`

**Interfaces:**

- Produces: `PersonalContextExchangeProof`, `PersonalContextAuthorityMetadata`, `PersonalContextRelayContinuation`, `PersonalContextActivationReceipt`, `PersonalContextCleanupAck`, and `PersonalContextPurgeReceipt`.
- Produces: optional `personal_context_exchange` on push, pull, conflict-list, and conflict-resolution request/response boundaries plus the complete version-1 fields on bootstrap, activation acknowledgment, conflict, and purge models.
- Consumes: existing `SyncV2Envelope`, `SyncPushRequest`, `SyncPullResponse`, and `PersonalContextSyncCapabilitiesResponse` models.

- [ ] **Step 1: Add failing strict-contract tests**

```python
def test_exchange_proof_requires_exact_version_epoch_and_token() -> None:
    proof = PersonalContextExchangeProof.model_validate(
        {
            "ongoing_sync_version": 1,
            "activation_epoch": "epoch_0123456789abcdef",
            "continuity_token": "continuity_0123456789abcdef",
        }
    )
    assert proof.ongoing_sync_version == 1


def test_client_envelope_cannot_claim_home_authority() -> None:
    with pytest.raises(ValueError, match="home authority"):
        validate_client_personal_context_metadata(
            PersonalContextAuthorityMetadata(
                role="home_authority",
                publication_batch_id="batch_0123456789abcdef",
                profile_publication_sequence=1,
                batch_ordinal=0,
                batch_size=2,
            )
        )


def test_pull_relay_continuation_distinguishes_pending_from_poisoned() -> None:
    continuation = PersonalContextRelayContinuation.model_validate(
        {
            "state": "relay_poisoned",
            "scan_watermark": "cursor_0123456789abcdef",
        }
    )
    assert continuation.state == "relay_poisoned"
```

- [ ] **Step 2: Run the contract test and confirm RED**

Run: `.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_personal_context_ongoing_contract.py -q`

Expected: collection fails because `personal_context_ongoing_contract` does not exist.

- [ ] **Step 3: Add the bounded Pydantic contract models**

```python
_OPAQUE_TOKEN = r"[A-Za-z0-9._~-]{16,256}"


class PersonalContextExchangeProof(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    ongoing_sync_version: Literal[1]
    activation_epoch: StrictStr = Field(pattern=_OPAQUE_TOKEN)
    continuity_token: StrictStr = Field(pattern=_OPAQUE_TOKEN)


class PersonalContextAuthorityMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    role: Literal["client_ingress", "home_authority"]
    publication_batch_id: StrictStr | None = Field(None, min_length=16, max_length=128)
    profile_publication_sequence: StrictInt | None = Field(None, ge=1)
    batch_ordinal: StrictInt | None = Field(None, ge=0)
    batch_size: StrictInt | None = Field(None, ge=1, le=100)


class PersonalContextRelayContinuation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    state: Literal["complete", "personal_context_relay_pending", "relay_poisoned"]
    scan_watermark: StrictStr | None = Field(None, min_length=16, max_length=512)


class PersonalContextActivationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    activation_id: StrictStr = Field(min_length=16, max_length=128)
    baseline_digest: StrictStr = Field(pattern=r"[0-9a-f]{64}")
    purge_generation: StrictInt = Field(ge=0)
    publication_watermark: StrictInt = Field(ge=0)
    home_server_cursor: StrictInt = Field(ge=0)
    home_manifest_revision: StrictInt = Field(ge=0)
    home_manifest_version_id: StrictStr = Field(min_length=16, max_length=128)
    state: Literal["prepared", "installed", "acknowledged", "active"]


class PersonalContextCleanupAck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    object_id: StrictStr = Field(min_length=16, max_length=128)
    version_id: StrictStr = Field(min_length=16, max_length=128)
    purge_generation: StrictInt = Field(ge=0)
    server_cleanup_complete: StrictBool


class PersonalContextPurgeReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    request_id: StrictStr = Field(min_length=16, max_length=128)
    profile_id: StrictStr = Field(min_length=16, max_length=128)
    purge_generation: StrictInt = Field(ge=1)
    barrier_envelope_id: StrictStr = Field(min_length=16, max_length=128)
    state: Literal["accepted", "barrier_pending", "acknowledged"]
```

Add an after-validator requiring all four publication fields for `home_authority` and forbidding them for `client_ingress`. Limit blockers to eight safe reason codes, conflict pages to 20, and every opaque identifier to 128 characters.

- [ ] **Step 4: Extend the existing API models without changing unrelated domains**

```python
class SyncPushRequest(BaseModel):
    dataset_id: str
    device_id: str = Field(..., min_length=1)
    personal_context_exchange: PersonalContextExchangeProof | None = None
    # retain every existing field unchanged


class SyncPushResponse(BaseModel):
    dataset_id: str
    accepted: list[SyncPushAcceptedEnvelope] = Field(default_factory=list)
    rejected: list[SyncPushRejectedEnvelope] = Field(default_factory=list)
    conflicts: list[SyncPushConflictEnvelope] = Field(default_factory=list)
    next_cursor: str | None = None
    personal_context_exchange: PersonalContextExchangeProof | None = None


class SyncPullResponse(BaseModel):
    # retain every existing field unchanged
    personal_context_relay: PersonalContextRelayContinuation | None = None
    personal_context_exchange: PersonalContextExchangeProof | None = None


class PersonalContextSyncCapabilitiesResponse(BaseModel):
    available: bool = False
    blockers: list[str] = Field(default_factory=list, max_length=8)
    ongoing_sync_version: Literal[0, 1] = 0
    ongoing_sync_blockers: list[str] = Field(default_factory=list, max_length=8)
    activation_epoch: str | None = Field(None, min_length=16, max_length=256)
    continuity_token: str | None = Field(None, min_length=16, max_length=256)

    @model_validator(mode="after")
    def validate_ongoing_state(self) -> "PersonalContextSyncCapabilitiesResponse":
        if (self.activation_epoch is None) != (self.continuity_token is None):
            raise ValueError("activation epoch and continuity token must appear together")
        if self.ongoing_sync_version == 1 and (
            self.ongoing_sync_blockers
            or self.activation_epoch is None
            or self.continuity_token is None
        ):
            raise ValueError("ongoing sync version 1 requires an unblocked continuity pair")
        return self


class SyncPersonalContextActivationAcknowledgeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    dataset_id: StrictStr = Field(min_length=1, max_length=128)
    device_id: StrictStr = Field(min_length=1, max_length=128)
    activation_id: StrictStr = Field(min_length=16, max_length=128)
    baseline_digest: StrictStr = Field(pattern=r"[0-9a-f]{64}")
    local_receipt_id: StrictStr = Field(min_length=16, max_length=128)
    personal_context_exchange: PersonalContextExchangeProof


class SyncPersonalContextActivationAcknowledgeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    receipt: PersonalContextActivationReceipt
    personal_context_exchange: PersonalContextExchangeProof


class SyncPersonalContextPurgeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    dataset_id: StrictStr = Field(min_length=1, max_length=128)
    device_id: StrictStr = Field(min_length=1, max_length=128)
    request_id: StrictStr = Field(min_length=16, max_length=128)
    expected_purge_generation: StrictInt = Field(ge=0)
    idempotency_key: StrictStr = Field(min_length=16, max_length=128)
    signature: StrictStr = Field(min_length=32, max_length=512)


class SyncPersonalContextPurgeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    receipt: PersonalContextPurgeReceipt
```

Use the same optional proof type in the pull response and batched conflict response. Return `PersonalContextRelayContinuation` on Personal Context pull so `complete`, retryable `personal_context_relay_pending`, and actionable `relay_poisoned` remain distinguishable. Extend `SyncPushConflictEnvelope` with the protected authority candidate and expected local/remote IDs. Extend the existing `SyncConflictResolution` with those expected IDs and an idempotency key; retain only the existing `skip`, `overwrite`, and `duplicate_rename` actions. Extend `SyncPersonalContextBootstrapRequest`/`SyncPersonalContextBootstrapResponse` for ongoing activation and add `SyncPersonalContextActivationAcknowledgeRequest`/`SyncPersonalContextActivationAcknowledgeResponse`; do not change the legacy 204 response of `/sync/personal-context/complete`. Add `SyncPersonalContextPurgeRequest`/`SyncPersonalContextPurgeResponse` under Sync V2 for signed device-originated global purge; the existing direct Personal Context purge endpoint keeps its current public shape. These are versioned shapes only: later tasks implement their state transitions while the advertised version remains 0.

Add these fields to the existing models without renaming their current fields:

```python
class SyncPersonalContextBootstrapRequest(BaseModel):
    ongoing_sync_version: Literal[1] | None = None


class SyncPersonalContextBootstrapResponse(BaseModel):
    activation: PersonalContextActivationReceipt | None = None
    personal_context_exchange: PersonalContextExchangeProof | None = None


class SyncPushConflictEnvelope(BaseModel):
    expected_local_envelope_id: str | None = None
    expected_remote_envelope_id: str | None = None
    authority_candidate: SyncV2EnvelopeResponse | None = None


class SyncConflictResolution(BaseModel):
    expected_local_envelope_id: str | None = None
    expected_remote_envelope_id: str | None = None
    idempotency_key: str | None = None


class SyncConflictResolveRequest(BaseModel):
    personal_context_exchange: PersonalContextExchangeProof | None = None


class SyncConflictResolveResponse(BaseModel):
    personal_context_exchange: PersonalContextExchangeProof | None = None
```

The enclosing request validator requires all expected IDs and the idempotency key when `personal_context_exchange` is present, and forbids Personal Context-only fields for non-Personal-Context conflicts.

Register the two new versioned POST paths immediately, but fail closed until their later state owners report readiness:

```python
@router.post("/personal-context/activation/acknowledge")
def acknowledge_personal_context_activation(
    request: SyncPersonalContextActivationAcknowledgeRequest,
    service: SyncV2Service = Depends(get_sync_v2_service),
) -> SyncPersonalContextActivationAcknowledgeResponse:
    raise HTTPException(
        status_code=409,
        detail={"code": "personal_context_ongoing_sync_unavailable"},
    )


@router.post("/personal-context/purge")
def purge_personal_context_everywhere(
    request: SyncPersonalContextPurgeRequest,
    service: SyncV2Service = Depends(get_sync_v2_service),
) -> SyncPersonalContextPurgeResponse:
    raise HTTPException(
        status_code=409,
        detail={"code": "personal_context_ongoing_sync_unavailable"},
    )
```

For GET pull and conflict-list requests, parse `personal_context_activation_epoch` and `personal_context_continuity_token` together and reject a half-present pair before calling the service. The checked-in contract endpoint map is fixed to:

```python
PERSONAL_CONTEXT_ONGOING_ENDPOINTS = {
    "capabilities": ("GET", "/api/v1/sync/capabilities"),
    "activation_prepare": ("POST", "/api/v1/sync/personal-context/bootstrap"),
    "activation_acknowledge": ("POST", "/api/v1/sync/personal-context/activation/acknowledge"),
    "push": ("POST", "/api/v1/sync/push"),
    "pull": ("GET", "/api/v1/sync/pull"),
    "conflict_list": ("GET", "/api/v1/sync/conflicts"),
    "conflict_resolve": ("POST", "/api/v1/sync/conflicts/resolve"),
    "purge": ("POST", "/api/v1/sync/personal-context/purge"),
}
```

- [ ] **Step 5: Export a reproducible schema and provenance manifest**

```python
def export_personal_context_ongoing_contract() -> dict[str, object]:
    model_classes = (
        PersonalContextSyncCapabilitiesResponse,
        SyncPersonalContextBootstrapRequest,
        SyncPersonalContextBootstrapResponse,
        SyncPersonalContextActivationAcknowledgeRequest,
        SyncPersonalContextActivationAcknowledgeResponse,
        SyncPushRequest,
        SyncPushResponse,
        SyncPullResponse,
        SyncConflictResolveRequest,
        SyncConflictResolveResponse,
        SyncPersonalContextPurgeRequest,
        SyncPersonalContextPurgeResponse,
    )
    _, schema = models_json_schema(
        [(model, "validation") for model in model_classes],
        title="tldw Personal Context ongoing sync v1",
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:tldw:personal-context:ongoing-sync:v1",
        "x-tldw-contract-version": 1,
        "x-tldw-endpoints": PERSONAL_CONTEXT_ONGOING_ENDPOINTS,
        **schema,
    }
```

The generator accepts `--source-commit`, writes canonical sorted JSON with a trailing newline, computes SHA-256 over the schema bytes, and builds the manifest with the supplied commit and computed digest:

```python
manifest = {
    "contract": "personal-context-ongoing-v1",
    "schema_version": 1,
    "server_source_commit": source_commit,
    "sha256": f"sha256:{hashlib.sha256(schema_bytes).hexdigest()}",
}
```

Artifact tests use `0000000000000000000000000000000000000000` as their fixed source commit. The generator also accepts `--output-dir`, so tests write to `tmp_path` without changing checked-in artifacts.

- [ ] **Step 6: Prove model, artifact, and endpoint behavior**

Run:

```bash
.venv/bin/python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_ongoing_contract.py \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py -q
.venv/bin/ruff check \
  tldw_Server_API/app/core/Sync/v2/personal_context_ongoing_contract.py \
  tldw_Server_API/app/api/v1/schemas/sync_v2_models.py \
  tldw_Server_API/app/api/v1/endpoints/sync.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_ongoing_contract.py
git diff --check
```

Expected: all targeted model and generator tests pass from temporary output; capability output still reports version 0.

- [ ] **Step 7: Commit the contract source used by the provenance manifest**

```bash
git add \
  Helper_Scripts/generate_personal_context_ongoing_contract.py \
  tldw_Server_API/app/core/Sync/v2/personal_context_ongoing_contract.py \
  tldw_Server_API/app/api/v1/schemas/sync_v2_models.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/api/v1/endpoints/sync.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_ongoing_contract.py \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py
git commit -m "feat(sync): define Personal Context ongoing contract"
```

- [ ] **Step 8: Generate and verify the checked-in artifact from that exact source commit**

```bash
.venv/bin/python Helper_Scripts/generate_personal_context_ongoing_contract.py \
  --source-commit "$(git rev-parse HEAD)"
.venv/bin/python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_ongoing_contract.py \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py -q
git diff --check
```

Expected: the manifest source commit resolves to the immediately preceding source commit, its SHA-256 matches the exact schema bytes, a second generation produces no diff, and capability output still reports version 0.

- [ ] **Step 9: Commit the published artifact and close TASK-13159**

```bash
git add \
  tldw_Server_API/app/core/Sync/v2/contracts/personal-context-ongoing-v1.schema.json \
  tldw_Server_API/app/core/Sync/v2/contracts/personal-context-ongoing-v1.manifest.json \
  backlog/tasks/task-13159\ -\ Version-Personal-Context-ongoing-sync-wire-contract.md
git commit -m "build(sync): publish Personal Context ongoing contract v1"
```

### Task 2: TASK-13160 — Journal canonical publications atomically

**Files:**

- Create: `tldw_Server_API/app/core/Personalization/personal_context_publication.py`
- Modify: `tldw_Server_API/app/core/DB_Management/Personalization_DB.py`
- Modify: `tldw_Server_API/app/core/DB_Management/Personal_Context_Repository.py`
- Modify: `tldw_Server_API/app/core/Personalization/personal_context_service.py`
- Test: `tldw_Server_API/tests/Personalization/test_personal_context_publication.py`
- Test: `tldw_Server_API/tests/Personalization/test_personal_context_repository.py`
- Test: `tldw_Server_API/tests/Personalization/test_personal_context_service.py`
- Test: `tldw_Server_API/tests/Personalization/test_personal_context_plaintext_canary.py`

**Interfaces:**

- Consumes: canonical `ProfileManifest`, `ProfileScope`, `ProfileRecord`, and `ProfileProposal` bytes from Shared Profile Core.
- Produces: `PublicationObject`, `PublicationBatchReceipt`, `IngressIdentity`, `CanonicalApplyReceipt`, and `PersonalContextPublicationJournal.append_batch()`.
- Produces: `PersonalContextService.apply_sync_ingress()` for the Sync materializer used in TASK-13161.

- [ ] **Step 1: Add failing schema and atomicity tests**

```python
def test_record_mutation_commits_manifest_and_publication_batch_atomically(service, db) -> None:
    record = service.create_record(active_record())
    rows = db.connection_for_test().execute(
        "SELECT role, batch_ordinal, batch_size FROM personal_context_publication_rows ORDER BY batch_ordinal"
    ).fetchall()
    assert [row["role"] for row in rows] == ["semantic", "manifest"]
    assert {(row["batch_ordinal"], row["batch_size"]) for row in rows} == {(0, 2), (1, 2)}


def test_ingress_replay_returns_original_result_without_second_manifest_advance(service) -> None:
    first = service.apply_sync_ingress(**ingress_kwargs("client-envelope-1"))
    replay = service.apply_sync_ingress(**ingress_kwargs("client-envelope-1"))
    assert replay == first
    assert service.get_manifest().revision == first.manifest_revision
```

- [ ] **Step 2: Run the publication tests and confirm RED**

Run: `.venv/bin/python -m pytest tldw_Server_API/tests/Personalization/test_personal_context_publication.py -q`

Expected: the publication tables and service method are absent.

- [ ] **Step 3: Add idempotent Personalization schema objects**

Add these tables in `PersonalizationDB._ensure_schema()` with foreign keys and uniqueness constraints:

```sql
CREATE TABLE IF NOT EXISTS personal_context_publication_profiles (
    profile_id TEXT PRIMARY KEY,
    next_sequence INTEGER NOT NULL CHECK (next_sequence >= 1),
    activation_covered_through_sequence INTEGER NOT NULL DEFAULT 0,
    purge_generation INTEGER NOT NULL CHECK (purge_generation >= 0),
    activation_epoch TEXT,
    continuity_token TEXT,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS personal_context_publication_batches (
    profile_id TEXT NOT NULL,
    profile_publication_sequence INTEGER NOT NULL,
    publication_batch_id TEXT NOT NULL,
    purge_generation INTEGER NOT NULL,
    batch_size INTEGER NOT NULL CHECK (batch_size >= 1),
    status TEXT NOT NULL CHECK (status IN ('pending','relaying','complete','covered_by_activation','purge_terminal')),
    activation_id TEXT,
    baseline_digest TEXT,
    sync_receipt_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (profile_id, profile_publication_sequence),
    UNIQUE (profile_id, publication_batch_id)
);
CREATE TABLE IF NOT EXISTS personal_context_publication_rows (
    profile_id TEXT NOT NULL,
    profile_publication_sequence INTEGER NOT NULL,
    publication_batch_id TEXT NOT NULL,
    batch_ordinal INTEGER NOT NULL CHECK (batch_ordinal >= 0),
    batch_size INTEGER NOT NULL CHECK (batch_size >= 1),
    purge_generation INTEGER NOT NULL CHECK (purge_generation >= 0),
    role TEXT NOT NULL CHECK (role IN ('semantic','manifest','purge_barrier')),
    opaque_object_id TEXT NOT NULL,
    opaque_version_id TEXT NOT NULL,
    operation TEXT NOT NULL CHECK (operation IN ('upsert','tombstone')),
    algorithm TEXT NOT NULL,
    key_version INTEGER NOT NULL,
    nonce BLOB NOT NULL,
    wrapped_dek BLOB NOT NULL,
    wrapped_dek_nonce BLOB NOT NULL,
    ciphertext BLOB NOT NULL,
    integrity_tag TEXT NOT NULL,
    payload_size_bytes INTEGER NOT NULL CHECK (payload_size_bytes >= 0),
    deterministic_envelope_id TEXT NOT NULL,
    sync_server_cursor INTEGER,
    row_state TEXT NOT NULL CHECK (row_state IN ('pending','staged','acknowledged','shredded')),
    PRIMARY KEY (profile_id, profile_publication_sequence, batch_ordinal),
    UNIQUE (profile_id, deterministic_envelope_id),
    FOREIGN KEY (profile_id, profile_publication_sequence)
        REFERENCES personal_context_publication_batches(profile_id, profile_publication_sequence)
);
CREATE TABLE IF NOT EXISTS personal_context_ingress_receipts (
    dataset_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    client_envelope_id TEXT NOT NULL,
    canonical_payload_digest TEXT NOT NULL,
    purge_generation INTEGER NOT NULL CHECK (purge_generation >= 0),
    resulting_object_id TEXT NOT NULL,
    resulting_version_id TEXT NOT NULL,
    resulting_manifest_revision INTEGER NOT NULL CHECK (resulting_manifest_revision >= 0),
    resulting_manifest_version_id TEXT NOT NULL,
    publication_batch_id TEXT NOT NULL,
    profile_publication_sequence INTEGER NOT NULL CHECK (profile_publication_sequence >= 1),
    receipt_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (dataset_id, device_id, client_envelope_id),
    UNIQUE (receipt_id)
);
```

Add indexes for `(profile_id, row_state, profile_publication_sequence, batch_ordinal)` and `(profile_id, status, profile_publication_sequence)`. The receipt stores only digest, opaque resulting IDs/revisions, generation, and batch identity in clear.

- [ ] **Step 4: Implement the transaction-scoped journal**

```python
@dataclass(frozen=True, slots=True)
class PublicationObject:
    domain: str
    object_id: str
    version_id: str
    operation: Literal["upsert", "tombstone"]
    role: Literal["semantic", "manifest", "purge_barrier"]
    canonical: bytes


class PersonalContextPublicationJournal:
    def append_batch(
        self,
        connection: sqlite3.Connection,
        *,
        profile_id: str,
        purge_generation: int,
        objects: Sequence[PublicationObject],
        ingress: IngressIdentity | None = None,
    ) -> PublicationBatchReceipt:
        sequence = self._claim_next_sequence(connection, profile_id)
        batch_id = self._stable_batch_id(profile_id, sequence, purge_generation)
        self._insert_encrypted_rows(connection, batch_id, objects)
        return self._record_receipt(connection, batch_id, sequence, ingress)
```

Use the repository's existing profile encryption key and AEAD helper. A publication row's AAD includes profile ID, batch ID, sequence, ordinal, role, and purge generation. No readable domain label, semantic key, profile label, or payload enters logs.

- [ ] **Step 5: Integrate every canonical mutation inside its existing transaction**

For record/scope/proposal mutations that advance the manifest, pass the semantic object followed by the exact new manifest to `append_batch()`. For canonical manifest-only changes, publish only the manifest. For purge, defer the barrier body to TASK-13164 but reserve the role. Keep runtime policy out because it is peer-local.

Add the ingress-only service seam:

```python
def apply_sync_ingress(
    self,
    *,
    identity: IngressIdentity,
    domain: str,
    value: ProfileManifest | ProfileScope | ProfileRecord | ProfileProposal | Mapping[str, Any],
    base_object_hash: str | None,
) -> CanonicalApplyReceipt:
    return self._repository.apply_ingress_and_publish(
        identity=identity,
        domain=domain,
        value=value,
        base_object_hash=base_object_hash,
    )
```

The repository first checks the ingress key and payload digest. An exact replay returns its stored receipt; a reused ID with different bytes raises a bounded idempotency conflict before canonical mutation.

- [ ] **Step 6: Prove rollback, concurrency, encryption, and pre-activation compaction**

Run:

```bash
.venv/bin/python -m pytest \
  tldw_Server_API/tests/Personalization/test_personal_context_publication.py \
  tldw_Server_API/tests/Personalization/test_personal_context_repository.py \
  tldw_Server_API/tests/Personalization/test_personal_context_service.py \
  tldw_Server_API/tests/Personalization/test_personal_context_plaintext_canary.py -q
.venv/bin/ruff check \
  tldw_Server_API/app/core/Personalization/personal_context_publication.py \
  tldw_Server_API/app/core/DB_Management/Personalization_DB.py \
  tldw_Server_API/app/core/DB_Management/Personal_Context_Repository.py \
  tldw_Server_API/app/core/Personalization/personal_context_service.py \
  tldw_Server_API/tests/Personalization/test_personal_context_publication.py
git diff --check
```

Expected: rollback leaves neither canonical nor publication changes; simultaneous mutations allocate distinct contiguous sequences; plaintext canaries appear only inside authenticated ciphertext; capability remains version 0.

- [ ] **Step 7: Commit TASK-13160**

```bash
git add \
  tldw_Server_API/app/core/DB_Management/Personalization_DB.py \
  tldw_Server_API/app/core/DB_Management/Personal_Context_Repository.py \
  tldw_Server_API/app/core/Personalization/personal_context_publication.py \
  tldw_Server_API/app/core/Personalization/personal_context_service.py \
  tldw_Server_API/tests/Personalization/test_personal_context_publication.py \
  tldw_Server_API/tests/Personalization/test_personal_context_repository.py \
  tldw_Server_API/tests/Personalization/test_personal_context_service.py \
  tldw_Server_API/tests/Personalization/test_personal_context_plaintext_canary.py \
  backlog/tasks/task-13160\ -\ Journal-canonical-Personal-Context-publications-atomically.md
git commit -m "feat(personal-context): journal authority publications"
```

### Task 3: TASK-13161 — Relay authority publications through Sync V2

**Files:**

- Create: `tldw_Server_API/app/core/Sync/v2/personal_context_relay.py`
- Modify: `tldw_Server_API/app/core/Personalization/personal_context_publication.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/materializers/personal_context.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/server_origin.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/store.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/models.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/factory.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_relay.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_materializer.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_transport.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_service.py`

**Interfaces:**

- Consumes: TASK-13160 publication batches and `PersonalContextService.apply_sync_ingress()`.
- Produces: `PersonalContextRelay.relay_profile()` and `PersonalContextRelayResult`.
- Produces: permanent `client_ingress` versus `home_authority` envelope classification used only by Personal Context pull.

- [ ] **Step 1: Add failing interruption and visibility tests**

```python
def test_relay_never_stages_manifest_before_semantic_siblings(relay, stores) -> None:
    stores.publication.fail_after_acknowledged_ordinal = 0
    with pytest.raises(InjectedFailure):
        relay.relay_profile(**relay_args())
    assert [row.role for row in stores.sync.authority_rows()] == ["semantic"]
    stores.publication.fail_after_acknowledged_ordinal = None
    relay.relay_profile(**relay_args())
    assert [row.role for row in stores.sync.authority_rows()] == ["semantic", "manifest"]


def test_pull_hides_client_ingress_even_after_canonical_acceptance(sync_service) -> None:
    ingress = sync_service.push(**personal_context_push()).accepted[0]
    page = sync_service.pull(**other_device_pull())
    assert ingress.client_envelope_id not in {item.client_envelope_id for item in page.envelopes}
    assert {item.authority.role for item in page.envelopes} == {"home_authority"}
```

- [ ] **Step 2: Run focused relay tests and confirm RED**

Run: `.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_personal_context_relay.py -q`

Expected: relay module and role-aware storage are absent.

- [ ] **Step 3: Implement one leased globally ordered relay**

```python
@dataclass(frozen=True, slots=True)
class PersonalContextRelayResult:
    staged_rows: int
    source_exhausted: bool
    visible_lookahead: bool
    continuation: Literal["complete", "personal_context_relay_pending", "relay_poisoned"]


class PersonalContextRelay:
    def relay_profile(
        self,
        *,
        user_id: str,
        profile_id: str,
        dataset_id: str,
        after_server_cursor: int | None,
        row_budget: int = 100,
        wall_time_ms: int = 100,
    ) -> PersonalContextRelayResult:
        with self.publications.profile_lease(profile_id):
            return self._relay_earliest_batches(
                user_id=user_id,
                profile_id=profile_id,
                dataset_id=dataset_id,
                after_server_cursor=after_server_cursor,
                row_budget=row_budget,
                deadline_ns=self.clock_ns() + wall_time_ms * 1_000_000,
            )
```

Claim only the earliest batch whose state is not `complete`, `covered_by_activation`, or `purge_terminal`. Walk ordinals in order. Refuse the manifest until every semantic row has a verified deterministic Sync envelope receipt. A malformed row marks profile Attention and blocks later batches.

- [ ] **Step 4: Materialize ingress through the canonical replay receipt boundary**

Update `PersonalContextMaterializer.apply()` to call `apply_sync_ingress()` with dataset, real device, client envelope ID, canonical digest, and purge generation. Only after the returned receipt matches the stored envelope does Sync mark ingress `applied`. Re-entry after a crash performs the equality replay and does not create another canonical version or publication batch.

```python
def apply(self, envelope: SyncEnvelope, *, user_id: str) -> CanonicalApplyReceipt:
    canonical = self.decrypt_and_validate(envelope)
    receipt = self.profile_service(user_id).apply_sync_ingress(
        identity=IngressIdentity(
            dataset_id=envelope.dataset_id,
            device_id=envelope.device_id,
            client_envelope_id=envelope.client_envelope_id,
            canonical_digest=sha256(canonical).hexdigest(),
            purge_generation=envelope.purge_generation,
        ),
        domain=envelope.domain,
        value=self.shared_core.parse(envelope.domain, canonical),
        base_object_hash=envelope.base_object_hash,
    )
    self.store.mark_personal_context_ingress_applied(
        server_cursor=envelope.server_cursor,
        expected_client_envelope_id=envelope.client_envelope_id,
        canonical_receipt_id=receipt.receipt_id,
    )
    return receipt
```

- [ ] **Step 5: Insert authority egress through an internal-only server-origin seam**

```python
def insert_personal_context_authority(
    self,
    *,
    envelope: SyncEnvelopeCreate,
    authority: PersonalContextAuthorityMetadata,
) -> SyncEnvelope:
    if authority.role != "home_authority":
        raise SyncStoreError("Personal Context authority role is required")
    stored = self.store.insert_envelope(
        replace(envelope, device_id=SERVER_ORIGIN_DEVICE_ID, status="accepted")
    )
    self.store.mark_envelope_apply_status(stored.server_cursor, apply_status="applied")
    return self.store.get_envelope_by_server_cursor(stored.server_cursor)
```

Keep this function outside public push. Public push rejects the reserved ID and any client-supplied `home_authority` role before insertion.

- [ ] **Step 6: Make Personal Context pull recover and filter without skipping**

Before a Personal Context pull page is assembled, invoke `relay_profile()` until a post-filter lookahead exists, the source is exhausted, or the 100-row/100-millisecond budget expires. Maintain the raw scan watermark in the signed pull token and return only `applied` home-authority rows. Do not promote client ingress after materialization; the canonical publication is a distinct later envelope.

```python
def pull_personal_context_page(
    self,
    *,
    user_id: str,
    profile_id: str,
    request: SyncPullRequest,
) -> SyncPullResponse:
    relay = self.personal_context_relay.relay_profile(
        user_id=user_id,
        profile_id=profile_id,
        dataset_id=request.dataset_id,
        after_server_cursor=request.scan_watermark,
        row_budget=100,
        wall_time_ms=100,
    )
    page = self.store.scan_personal_context_authority(
        dataset_id=request.dataset_id,
        after_server_cursor=request.scan_watermark,
        limit=request.limit,
        require_role="home_authority",
        require_apply_status="applied",
    )
    return SyncPullResponse(
        envelopes=page.visible_envelopes,
        next_cursor=self.tokens.sign_scan_watermark(page.raw_scan_watermark),
        personal_context_relay=PersonalContextRelayContinuation(
            state=relay.continuation,
            scan_watermark=self.tokens.sign_scan_watermark(page.raw_scan_watermark),
        ),
        personal_context_exchange=self.require_exchange_proof(request),
    )
```

- [ ] **Step 7: Run relay, materializer, cursor, and reserved-identity tests**

Run:

```bash
.venv/bin/python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_relay.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_materializer.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_transport.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_server_trusted_encryption.py -q
.venv/bin/ruff check \
  tldw_Server_API/app/core/Sync/v2/personal_context_relay.py \
  tldw_Server_API/app/core/Sync/v2/materializers/personal_context.py \
  tldw_Server_API/app/core/Sync/v2/server_origin.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_relay.py
git diff --check
```

Expected: interleaved after-commit and pull relay preserves profile sequence; every crash resumes deterministic IDs; ingress never appears in another device's pull; capability remains version 0.

- [ ] **Step 8: Commit TASK-13161**

```bash
git add \
  tldw_Server_API/app/core/Personalization/personal_context_publication.py \
  tldw_Server_API/app/core/Sync/v2/personal_context_relay.py \
  tldw_Server_API/app/core/Sync/v2/materializers/personal_context.py \
  tldw_Server_API/app/core/Sync/v2/server_origin.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/Sync/v2/models.py \
  tldw_Server_API/app/core/Sync/v2/factory.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_relay.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_materializer.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_transport.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  backlog/tasks/task-13161\ -\ Relay-ordered-Personal-Context-authority-publications-through-Sync-V2.md
git commit -m "feat(sync): relay Personal Context authority batches"
```

## Plan self-review

- Spec coverage: contract ownership, encrypted source journal, atomic canonical publication, ingress replay, reserved authority identity, global batch order, manifest gating, pull-time recovery budget, role-aware egress, and split scan/application facts are assigned.
- Placeholder scan: no deferred implementation markers remain.
- Type consistency: TASK-13160 produces the journal and ingress receipt consumed by TASK-13161; TASK-13159 produces the authority and continuation models used by both.
