# Chatbook Sync Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the unified Sync v2 engine described by the Chatbook Sync Engine PRD, with `tldw_chatbook` as the first local-first client while preserving local-only and server-front-end modes.

**Architecture:** Add a generic server-side Sync v2 substrate under the existing `/api/v1/sync` route family, backed by device, dataset, envelope, cursor, conflict, and key-record storage. Domain adapters bridge Sync v2 envelopes to existing tldw server domain stores, while Chatbook adds a local sync profile, outbox/inbox, encryption hooks, and restore flow around its existing `Sync_Interop` scaffolding. Existing media-only sync is migrated into a compatibility domain rather than kept as a competing protocol.

**Tech Stack:** FastAPI, Pydantic, SQLite/PostgreSQL-compatible DB abstractions, pytest, Loguru, existing AuthNZ dependencies, existing Media DB and ChaChaNotes DB helpers, tldw_chatbook Python/Textual services, client-side encryption primitives, and existing Chatbook sync tests.

---

## Scope

Implement the approved product/design direction from:

- `Docs/superpowers/specs/2026-05-10-chatbook-sync-engine-prd-design.md`
- Backlog task `TASK-208`

This plan covers the end-to-end Sync v2 feature family, but it should not be
implemented as one giant pull request. Before implementation, split this into
reviewable Backlog tasks and branches:

1. Server protocol and storage substrate.
2. Server media compatibility and domain-adapter contract.
3. Server V1 domain adapters for notes, chat, workspaces/source refs, and source cache.
4. Chatbook client protocol substrate.
5. Chatbook local-first encryption and recovery-bundle integration.
6. Restore flow and conflict review.
7. End-to-end integration, migration docs, and hardening.

The first shippable server slice should expose capabilities, device
registration, dataset enrollment, push, pull, and restore manifest endpoints
with one working compatibility domain. The first shippable Chatbook slice should
negotiate capabilities, register a device, enroll a personal dataset, persist
cursors, and run a no-content dry-run sync without altering local-only behavior.

## Non-Scope For Initial Implementation

- CRDT text editing.
- Full real-time collaboration.
- Large media binary replication.
- Embedding/vector replication.
- Server-side decryption for local-first private datasets.
- Automatic recovery when all client keys and recovery bundles are lost.
- Broad UI redesign in Chatbook beyond sync mode/status/conflict surfaces needed
  to complete the feature.

## Current File Map

### Server Files

API and schemas:

- Modify `tldw_Server_API/app/api/v1/endpoints/sync.py`
  - Keep `/send` and `/get` as legacy wrappers or explicitly gated
    compatibility endpoints until the migration decision is made.
  - Add Sync v2 endpoints under the same router:
    `/capabilities`, `/devices/register`, `/datasets/enroll`,
    `/restore-manifest`, `/push`, `/pull`, `/conflicts`,
    `/conflicts/{id}/resolve`, `/attachments`, and
    `/keys/recovery-bundle`.
- Keep or deprecate `tldw_Server_API/app/api/v1/schemas/sync_server_models.py`
  for legacy media sync models.
- Create `tldw_Server_API/app/api/v1/schemas/sync_v2_models.py`
  - Pydantic request/response models for Sync v2.
  - Domain, operation, dataset scope, encryption policy, and conflict status
    literals.
  - Examples that do not expose plaintext private content.
- Verify router mount in
  `tldw_Server_API/app/api/v1/router_groups/core.py`.

Core substrate:

- Create `tldw_Server_API/app/core/Sync/v2/__init__.py`
- Create `tldw_Server_API/app/core/Sync/v2/models.py`
  - Internal dataclasses or typed structures for devices, datasets, envelopes,
    cursors, conflicts, key records, and adapter outcomes.
- Create `tldw_Server_API/app/core/Sync/v2/store.py`
  - DB operations for Sync v2 tables.
  - Transaction boundaries and idempotent envelope insert helpers.
- Create `tldw_Server_API/app/core/Sync/v2/service.py`
  - Business logic for capability negotiation, device registration, dataset
    enrollment, push, pull, conflict listing/resolution, restore manifests, and
    key-record persistence.
- Create `tldw_Server_API/app/core/Sync/v2/adapters.py`
  - Adapter protocol/base class and registry.
  - Result types for accepted, rejected, conflict, and deferred envelopes.
- Create `tldw_Server_API/app/core/Sync/v2/security.py`
  - Server-side validation that private payload fields stay encrypted/opaque.
  - Log-safe redaction helpers for envelope payloads and key records.
- Create `tldw_Server_API/app/core/Sync/v2/errors.py`
  - Sync-specific exceptions mapped by the API layer.

Storage and migrations:

- Add migrations/bootstrap logic where this repo currently manages per-user DB
  schemas. Candidate locations to verify during implementation:
  - `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
  - `tldw_Server_API/app/core/DB_Management/media_db/`
  - existing per-user DB bootstrap helpers under `tldw_Server_API/app/core/DB_Management/`
- If Sync v2 storage is kept separate from ChaChaNotes/Media DB, create a new
  focused DB helper under `tldw_Server_API/app/core/DB_Management/Sync_DB.py`
  and tests under `tldw_Server_API/tests/Sync/`.

Domain adapters:

- Create `tldw_Server_API/app/core/Sync/v2/domain_adapters/`
- Create `tldw_Server_API/app/core/Sync/v2/domain_adapters/media.py`
  - Compatibility bridge for existing media sync semantics.
- Create `tldw_Server_API/app/core/Sync/v2/domain_adapters/notes.py`
  - Adapter for notes and note metadata in ChaChaNotes.
- Create `tldw_Server_API/app/core/Sync/v2/domain_adapters/chat.py`
  - Adapter for conversations and append-only messages.
- Create `tldw_Server_API/app/core/Sync/v2/domain_adapters/workspaces.py`
  - Adapter for workspace records and source references.
- Create `tldw_Server_API/app/core/Sync/v2/domain_adapters/source_cache.py`
  - Adapter for extracted text/transcript/summary cache metadata and small
    attachment references.

Server tests:

- Create `tldw_Server_API/tests/Sync/test_sync_v2_models.py`
- Create `tldw_Server_API/tests/Sync/test_sync_v2_store.py`
- Create `tldw_Server_API/tests/Sync/test_sync_v2_service.py`
- Create `tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py`
- Create `tldw_Server_API/tests/Sync/test_sync_v2_security.py`
- Create `tldw_Server_API/tests/Sync/test_sync_v2_media_compat.py`
- Create `tldw_Server_API/tests/Sync/test_sync_v2_domain_adapters.py`
- Extend `tldw_Server_API/tests/Sync/test_sync_error_mapping.py` if endpoint
  error behavior changes.

Documentation:

- Update `tldw_Server_API/app/core/Sync/README.md`.
- Update `Docs/Design/Sync-Engine.md` from placeholder links into a short index
  pointing to the PRD, implementation plan, protocol docs, and migration notes.
- Add API/protocol docs under `Docs/API/` once endpoint shapes stabilize.

### Chatbook Files

The Chatbook repo is a sibling checkout:
`/Users/macbook-dev/Documents/GitHub/tldw_chatbook`.

Client API and schemas:

- Modify `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/tldw_api/sync_schemas.py`
  - Add Sync v2 client models aligned with server `sync_v2_models.py`.
  - Preserve legacy media sync models until compatibility is removed.
- Modify `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/tldw_api/client.py`
  - Add methods for Sync v2 endpoints.
  - Keep existing `send_sync_changes` and `get_sync_changes` until migration
    is complete.

Sync state and services:

- Modify `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/Sync_Interop/sync_state.py`
  - Add sync profile mode, dataset state, device state, per-domain cursors,
    envelope metadata, key-record references, and conflict summary structures.
- Modify `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/Sync_Interop/sync_state_repository.py`
  - Persist Sync v2 profile/device/dataset/cursor/outbox/inbox/conflict state.
  - Add migrations from current dry-run state if needed.
- Modify `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/Sync_Interop/server_sync_service.py`
  - Replace media-only send/get orchestration with Sync v2 capability/register/
    enroll/push/pull flows.
- Modify `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/Sync_Interop/sync_scope_service.py`
  - Route local-only, local-first sync, and server-front-end modes explicitly.
- Modify `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/Sync_Interop/sync_readiness.py`
  - Report readiness by profile mode, auth state, key state, server
    capabilities, conflicts, and pending outbox.

New Chatbook modules:

- Create `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/Sync_Interop/envelope_builder.py`
  - Converts local domain changes into Sync v2 envelopes.
- Create `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/Sync_Interop/envelope_applier.py`
  - Applies pulled envelopes to local stores with adapter-specific merge rules.
- Create `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/Sync_Interop/crypto.py`
  - Client-side content encryption/decryption and recovery-bundle wrapping.
- Create `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/Sync_Interop/domain_adapters/`
  - Local adapters for notes, chat messages, workspaces/source refs, source
    cache, and media compatibility.
- Create `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/Sync_Interop/restore_service.py`
  - Restore manifest fetch, selection, pull, decrypt, and apply flow.

Chatbook tests:

- Modify or create tests under:
  - `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Sync_Interop/test_sync_state.py`
  - `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Sync_Interop/test_sync_state_repository.py`
  - `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Sync_Interop/test_server_sync_service.py`
  - `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Sync_Interop/test_sync_scope_service.py`
  - `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Sync_Interop/test_sync_readiness.py`
  - `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/tldw_api/test_sync_client.py`
- Create:
  - `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Sync_Interop/test_envelope_builder.py`
  - `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Sync_Interop/test_envelope_applier.py`
  - `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Sync_Interop/test_sync_crypto.py`
  - `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Sync_Interop/test_restore_service.py`

## Implementation Tasks

### Task 1: Define Sync v2 Protocol Schemas

**Files:**

- Create: `tldw_Server_API/app/api/v1/schemas/sync_v2_models.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_models.py`
- Reference: `tldw_Server_API/app/api/v1/schemas/sync_server_models.py`

- [ ] **Step 1: Write failing schema tests**

Create tests for:

```python
def test_sync_envelope_rejects_plaintext_private_payload():
    payload = {
        "client_envelope_id": "env-1",
        "dataset_id": "dataset-1",
        "domain": "notes",
        "entity_id": "note-1",
        "operation": "upsert",
        "adapter_version": 1,
        "routing_metadata": {"entity_kind": "note"},
        "payload_ciphertext": None,
        "payload_clear": {"body": "known plaintext"},
        "payload_hash": "sha256:test",
    }

    with pytest.raises(ValidationError):
        SyncV2Envelope.model_validate(payload)
```

```python
def test_push_response_reports_per_envelope_outcomes():
    response = SyncPushResponse.model_validate(
        {
            "dataset_id": "dataset-1",
            "accepted": [{"client_envelope_id": "env-1", "server_sequence": 1}],
            "rejected": [],
            "conflicts": [],
            "next_cursor": "1",
        }
    )

    assert response.accepted[0].server_sequence == 1
```

- [ ] **Step 2: Run schema tests and verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_models.py -v
```

Expected: FAIL because `sync_v2_models.py` does not exist.

- [ ] **Step 3: Implement minimal Pydantic models**

Add models for:

- `SyncCapabilitiesResponse`
- `SyncDeviceRegisterRequest`
- `SyncDeviceRegisterResponse`
- `SyncDatasetEnrollRequest`
- `SyncDatasetEnrollResponse`
- `SyncRestoreManifestResponse`
- `SyncV2Envelope`
- `SyncPushRequest`
- `SyncPushResponse`
- `SyncPullResponse`
- `SyncAttachmentUploadRequest`
- `SyncAttachmentUploadResponse`
- `SyncConflictRecord`
- `SyncConflictResolveRequest`
- `SyncKeyRecoveryBundleRequest`

`SyncV2Envelope` must include `adapter_version` as a required integer so the
service layer can reject unsupported adapter versions without guessing from
domain defaults.

Use string literal unions for V1 domains:

```python
SyncDomain = Literal["notes", "chat", "workspaces", "source_cache", "media"]
SyncOperation = Literal["upsert", "delete", "link", "unlink", "resolve_conflict"]
DatasetScopeType = Literal["personal", "workspace"]
EncryptionPolicy = Literal["client_private_v1", "server_trusted", "shared_workspace_v1"]
```

- [ ] **Step 4: Pass schema tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_models.py -v
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py
git commit -m "feat(sync): define sync v2 protocol schemas"
```

### Task 2: Add Sync v2 Store And Migration Path

**Files:**

- Create: `tldw_Server_API/app/core/Sync/v2/models.py`
- Create: `tldw_Server_API/app/core/Sync/v2/store.py`
- Create: `tldw_Server_API/app/core/Sync/v2/errors.py`
- Create or modify: `tldw_Server_API/app/core/DB_Management/Sync_DB.py`
- Modify as needed: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_store.py`

- [ ] **Step 1: Decide storage location before coding**

Pick one storage strategy for the first implementation PR:

1. Per-user Sync DB next to existing per-user DBs.
2. Sync tables inside ChaChaNotes DB.
3. Central Sync DB keyed by `user_id`.

Default recommendation: per-user Sync DB if restore queries and backup behavior
stay manageable; central Sync DB only if multi-device restore inventory across
many user DBs becomes too slow.

Record the decision in the PR description and update this plan if it changes
file ownership materially.

- [ ] **Step 2: Write failing store tests**

Cover:

- device upsert idempotency
- dataset enrollment idempotency
- envelope insert idempotency by `(dataset_id, client_envelope_id)`
- pull after cursor returns deterministic server-sequence order
- conflict insert/list/resolve lifecycle
- key record stores wrapped blobs but never plaintext keys

Example:

```python
def test_insert_envelope_is_idempotent(sync_store):
    envelope = make_envelope(client_envelope_id="env-1", domain="notes")

    first = sync_store.insert_envelope(envelope)
    second = sync_store.insert_envelope(envelope)

    assert second.server_sequence == first.server_sequence
    assert sync_store.list_envelopes_after(envelope.dataset_id, 0) == [first]
```

- [ ] **Step 3: Run store tests and verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_store.py -v
```

Expected: FAIL because store code and/or DB bootstrap does not exist.

- [ ] **Step 4: Implement tables and store helpers**

Implement logical tables from the PRD:

- `sync_devices`
- `sync_datasets`
- `sync_domain_state`
- `sync_envelopes`
- `sync_device_cursors`
- `sync_conflicts`
- `sync_key_records`

Keep SQL parameterized. Use existing transaction helpers. Do not log payload
ciphertext or wrapped key blobs.

- [ ] **Step 5: Pass focused store tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_store.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Sync/v2 \
  tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/tests/Sync/test_sync_v2_store.py
git commit -m "feat(sync): add sync v2 storage substrate"
```

Adjust staged paths to match the chosen storage strategy.

### Task 3: Add Sync v2 Service Layer

**Files:**

- Create: `tldw_Server_API/app/core/Sync/v2/adapters.py`
- Create: `tldw_Server_API/app/core/Sync/v2/service.py`
- Create: `tldw_Server_API/app/core/Sync/v2/security.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/store.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_service.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_security.py`

- [ ] **Step 1: Write failing service tests**

Cover:

- capabilities returns protocol version, domains, limits, encryption policies
- device registration creates and refreshes the same device
- dataset enrollment creates personal dataset by default
- minimal adapter registry accepts known domains and rejects unknown domains
- push rejects envelopes for datasets the user cannot access
- push returns per-envelope accepted/rejected/conflict outcomes
- push rejects unsupported adapter versions with a per-envelope rejection
- pull uses stable server cursor
- pull honors dataset/domain filters, same-device echo policy, page size, and
  `has_more`/next-cursor paging
- restore manifest is metadata-only for private encrypted datasets
- restore manifest includes device last-seen metadata, unresolved conflict
  counts, attachment availability/size classes, and encryption/key recovery
  status

Security example:

```python
def test_private_restore_manifest_has_no_plaintext_labels(sync_service):
    manifest = sync_service.restore_manifest(user_id="user-1")

    assert "known private note" not in repr(manifest)
    assert manifest.datasets[0].encryption_policy == "client_private_v1"
```

- [ ] **Step 2: Run service tests and verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_security.py \
  -v
```

Expected: FAIL because service/security code does not exist.

- [ ] **Step 3: Implement minimal service behavior**

Implement `SyncV2Service` with dependencies injected:

- store
- authenticated user context
- minimal adapter registry from `adapters.py`
- clock function for deterministic tests
- settings/capabilities object

The first version of `adapters.py` belongs in this task so adapter-version
validation is not duplicated later. Implement only the protocol, registry, base
result types, and test doubles needed by the service. Concrete media/notes/chat
adapters come in later tasks.

Protocol invariants implemented here:

- `push` is idempotent by `(dataset_id, client_envelope_id)`.
- `push` validates dataset access and adapter version before accepting an
  envelope.
- `push` returns accepted, rejected, and conflict outcomes without failing the
  whole batch for one bad envelope unless the request itself is malformed.
- `pull` filters by dataset and domain.
- `pull` exposes deterministic ordering by server sequence.
- `pull` supports same-device echo exclusion or marking based on request
  options.
- `pull` returns `next_cursor` plus `has_more` for paged results.
- restore manifests never include private plaintext before client-side unlock.

Keep payload logging behind redaction helpers:

```python
def redact_envelope_for_log(envelope: SyncV2Envelope) -> dict[str, object]:
    data = envelope.model_dump(exclude={"payload_ciphertext"})
    data["payload_ciphertext"] = "<redacted>" if envelope.payload_ciphertext else None
    return data
```

- [ ] **Step 4: Pass focused service/security tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_security.py \
  -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Sync/v2/adapters.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/security.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_security.py
git commit -m "feat(sync): add sync v2 service layer"
```

### Task 4: Wire Sync v2 API Endpoints

**Files:**

- Modify: `tldw_Server_API/app/api/v1/endpoints/sync.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/core.py` only if needed
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py`
- Modify: `tldw_Server_API/tests/Sync/test_sync_error_mapping.py`

- [ ] **Step 1: Write failing endpoint tests**

Use FastAPI dependency overrides or direct endpoint calls consistent with
existing sync tests.

Cover:

- `GET /api/v1/sync/capabilities`
- `POST /api/v1/sync/devices/register`
- `POST /api/v1/sync/datasets/enroll`
- `GET /api/v1/sync/restore-manifest`
- `POST /api/v1/sync/push`
- `GET /api/v1/sync/pull`
- `GET /api/v1/sync/conflicts`
- `POST /api/v1/sync/conflicts/{id}/resolve`
- `POST /api/v1/sync/attachments`
- `POST /api/v1/sync/keys/recovery-bundle`

Protocol invariants to cover directly in endpoint tests:

- `POST /push` is idempotent by `client_envelope_id`.
- `POST /push` rejects unsupported adapter versions per envelope.
- `GET /restore-manifest` honors dataset filters.
- `GET /restore-manifest` honors domain filters.
- `GET /pull` honors dataset filters.
- `GET /pull` honors domain filters.
- `GET /pull` supports same-device echo exclusion or explicit echo marking.
- `GET /pull` returns `has_more` and `next_cursor` when a page is truncated.
- `POST /attachments` either accepts small encrypted attachment chunks when the
  server capability is enabled or returns the documented gated response when
  attachment upload is disabled.
- endpoint logs and errors do not expose encrypted payloads, wrapped keys, or
  known plaintext.

Example:

```python
@pytest.mark.asyncio
async def test_push_returns_per_envelope_outcomes(app_client, auth_headers):
    response = await app_client.post(
        "/api/v1/sync/push",
        headers=auth_headers,
        json={"dataset_id": "dataset-1", "device_id": "device-1", "envelopes": []},
    )

    assert response.status_code == 200
    assert set(response.json()) >= {"accepted", "rejected", "conflicts", "next_cursor"}
```

- [ ] **Step 2: Run endpoint tests and verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py -v
```

Expected: FAIL because endpoints are missing.

- [ ] **Step 3: Add endpoints as thin service wrappers**

Keep endpoint functions small:

- auth and DB dependencies
- request parsing through Pydantic
- service method call
- exception-to-HTTP mapping

Do not duplicate merge/conflict logic in the endpoint file.

For `/attachments`, implement one of two explicit policies in this task:

1. Functional V1 small encrypted attachment upload with size/type/capability
   checks.
2. Capability-gated endpoint that returns a documented 409 or 501-style
   response until the attachment storage tranche lands.

Do not leave the route silently missing. The response shape must be tested so
clients can feature-detect attachment behavior.

- [ ] **Step 4: Preserve legacy endpoint behavior deliberately**

Choose and test one policy:

1. `/send` and `/get` remain unchanged as legacy compatibility endpoints.
2. `/send` and `/get` become wrappers that translate to Sync v2 media domain.
3. `/send` and `/get` return a documented deprecation error when Sync v2 is enabled.

Default recommendation: keep them unchanged in the first Sync v2 PR, then add a
compatibility wrapper in the media compatibility task.

- [ ] **Step 5: Pass endpoint tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py \
  tldw_Server_API/tests/Sync/test_sync_error_mapping.py \
  -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/sync.py \
  tldw_Server_API/app/api/v1/router_groups/core.py \
  tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py \
  tldw_Server_API/tests/Sync/test_sync_error_mapping.py
git commit -m "feat(sync): expose sync v2 api endpoints"
```

### Task 5: Add Media Compatibility Domain

**Files:**

- Create: `tldw_Server_API/app/core/Sync/v2/domain_adapters/__init__.py`
- Create: `tldw_Server_API/app/core/Sync/v2/domain_adapters/media.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/adapters.py`
- Modify: `tldw_Server_API/app/core/Sync/sync_contract.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_media_compat.py`
- Test: `tldw_Server_API/tests/MediaDB2/test_sync_server.py`
- Test: `tldw_Server_API/tests/MediaDB2/test_sync_client.py`

- [ ] **Step 1: Write failing adapter tests**

Cover:

- media adapter registers through the existing minimal adapter registry
- media adapter accepts legacy `Media`, `Keywords`, and `MediaKeywords`
- link/unlink semantics preserve current media sync behavior
- legacy payloads translate to Sync v2 envelopes without plaintext leakage
- old media sync tests still pass or are intentionally updated

- [ ] **Step 2: Run adapter/media tests and verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_media_compat.py \
  tldw_Server_API/tests/MediaDB2/test_sync_server.py \
  tldw_Server_API/tests/MediaDB2/test_sync_client.py \
  -v
```

Expected: FAIL until the concrete media adapter exists.

- [ ] **Step 3: Implement media adapter**

Use the protocol created in Task 3. If Task 3 did not define these methods yet,
extend it with the smallest interface equivalent to:

```python
class SyncDomainAdapter(Protocol):
    domain: str
    adapter_version: int

    def validate_envelope(self, envelope: SyncV2Envelope) -> None: ...
    def apply_envelope(self, envelope: SyncV2Envelope, context: SyncApplyContext) -> SyncAdapterResult: ...
    def build_restore_summary(self, dataset_id: str, user_id: str) -> SyncDomainRestoreSummary: ...
```

Keep current media operations parameterized and transactional.

- [ ] **Step 4: Pass adapter/media tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_media_compat.py \
  tldw_Server_API/tests/MediaDB2/test_sync_server.py \
  tldw_Server_API/tests/MediaDB2/test_sync_client.py \
  -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Sync/v2/adapters.py \
  tldw_Server_API/app/core/Sync/v2/domain_adapters \
  tldw_Server_API/app/core/Sync/sync_contract.py \
  tldw_Server_API/tests/Sync/test_sync_v2_media_compat.py \
  tldw_Server_API/tests/MediaDB2/test_sync_server.py \
  tldw_Server_API/tests/MediaDB2/test_sync_client.py
git commit -m "feat(sync): add media compatibility adapter"
```

### Task 6: Add Server V1 Domain Adapters

**Files:**

- Create: `tldw_Server_API/app/core/Sync/v2/domain_adapters/notes.py`
- Create: `tldw_Server_API/app/core/Sync/v2/domain_adapters/chat.py`
- Create: `tldw_Server_API/app/core/Sync/v2/domain_adapters/workspaces.py`
- Create: `tldw_Server_API/app/core/Sync/v2/domain_adapters/source_cache.py`
- Modify as needed: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify as needed: existing workspace/source DB helpers under
  `tldw_Server_API/app/core/DB_Management/`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_domain_adapters.py`
- Reuse fixtures from `tldw_Server_API/tests/ChaChaNotesDB/`

- [x] **Step 1: Write failing domain adapter tests**

Cover the PRD merge rules:

- notes: tag/status safe merge, title/body conflicts recorded
- chat: append-only messages merge by stable message ID/time
- chat: same message ID with different hashes creates conflict
- workspace source refs: stable source ID membership merges
- source cache: source ID plus content hash can coexist
- delete-vs-update creates manual conflict

Example:

```python
def test_note_body_concurrent_update_creates_conflict(notes_adapter, sync_context):
    base = make_note_envelope(entity_id="note-1", body_hash="base")
    local = make_note_envelope(entity_id="note-1", base_version=base.entity_version, body_hash="a")
    remote = make_note_envelope(entity_id="note-1", base_version=base.entity_version, body_hash="b")

    notes_adapter.apply_envelope(local, sync_context)
    result = notes_adapter.apply_envelope(remote, sync_context)

    assert result.status == "conflict"
    assert result.conflict_type == "encrypted_content_edit"
```

- [x] **Step 2: Run adapter tests and verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_domain_adapters.py -v
```

Expected: FAIL until adapters exist.

- [x] **Step 3: Implement adapters incrementally**

Implementation order:

1. notes
2. chat
3. workspaces/source refs
4. source cache

Do not introduce broad DB refactors. Add narrow DB helper methods only where
existing APIs cannot express the required sync operation.

- [x] **Step 4: Pass focused adapter tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_domain_adapters.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_note_store.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_message_store.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_workspace_sub_resources_db.py \
  -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Sync/v2/domain_adapters \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/tests/Sync/test_sync_v2_domain_adapters.py \
  tldw_Server_API/tests/ChaChaNotesDB
git commit -m "feat(sync): add sync v2 domain adapters"
```

### Task 7: Add Chatbook Sync v2 Client Models And API Methods

**Files:**

- Modify: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/tldw_api/sync_schemas.py`
- Modify: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/tldw_api/client.py`
- Test: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/tldw_api/test_sync_client.py`

- [ ] **Step 1: Write failing Chatbook API client tests**

Cover:

- capabilities request path
- device registration request/response
- dataset enrollment
- push and pull request serialization
- conflict resolve request
- encrypted attachment upload request or gated-response handling
- recovery-bundle request
- legacy send/get methods still exist during migration

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook
python -m pytest Tests/tldw_api/test_sync_client.py -v
```

Expected: FAIL because Sync v2 methods are missing.

- [ ] **Step 3: Implement client schemas and methods**

Keep schemas aligned with server `sync_v2_models.py`. If duplication becomes
painful, defer shared package extraction until after the first working
implementation; do not block V1 on cross-repo packaging.

- [ ] **Step 4: Pass client tests**

Run:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook
python -m pytest Tests/tldw_api/test_sync_client.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit in Chatbook repo**

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook
git add tldw_chatbook/tldw_api/sync_schemas.py \
  tldw_chatbook/tldw_api/client.py \
  Tests/tldw_api/test_sync_client.py
git commit -m "feat(sync): add sync v2 api client"
```

### Task 8: Add Chatbook Sync State, Profiles, And Orchestration

**Files:**

- Modify: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/Sync_Interop/sync_state.py`
- Modify: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/Sync_Interop/sync_state_repository.py`
- Modify: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/Sync_Interop/server_sync_service.py`
- Modify: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/Sync_Interop/sync_scope_service.py`
- Modify: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/Sync_Interop/sync_readiness.py`
- Tests:
  - `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Sync_Interop/test_sync_state.py`
  - `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Sync_Interop/test_sync_state_repository.py`
  - `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Sync_Interop/test_server_sync_service.py`
  - `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Sync_Interop/test_sync_scope_service.py`
  - `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Sync_Interop/test_sync_readiness.py`

- [ ] **Step 1: Write failing profile/state tests**

Cover:

- default profile mode is `local_only`
- `server_frontend` mode never creates local outbox records
- `local_first_sync` requires auth, device, dataset, and key readiness
- cursors persist per dataset/domain
- duplicate client envelope IDs are not regenerated on retry

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook
python -m pytest Tests/Sync_Interop -v
```

Expected: FAIL until new state fields and orchestration exist.

- [ ] **Step 3: Implement state/profile support**

Add explicit profile modes:

```python
SyncProfileMode = Literal["local_only", "local_first_sync", "server_frontend"]
```

Persist:

- device registration state
- dataset enrollment state
- per-domain pull cursors
- local outbox entries
- unresolved conflict summaries
- key/recovery readiness

- [ ] **Step 4: Implement server sync orchestration**

`server_sync_service.py` should orchestrate:

1. capabilities
2. device register
3. dataset enroll
4. push pending outbox
5. pull after cursor
6. apply or record conflicts

Keep mode routing in `sync_scope_service.py` explicit and testable.

- [ ] **Step 5: Pass Sync_Interop tests**

Run:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook
python -m pytest Tests/Sync_Interop -v
```

Expected: PASS.

- [ ] **Step 6: Commit in Chatbook repo**

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook
git add tldw_chatbook/Sync_Interop \
  Tests/Sync_Interop
git commit -m "feat(sync): add sync v2 profile orchestration"
```

### Task 9: Add Chatbook Encryption And Recovery Bundle

**Files:**

- Create: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/Sync_Interop/crypto.py`
- Modify: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/Sync_Interop/server_sync_service.py`
- Modify: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/Sync_Interop/sync_state.py`
- Test: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Sync_Interop/test_sync_crypto.py`
- Test: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Sync_Interop/test_server_sync_service.py`

- [ ] **Step 1: Choose encryption primitive and dependency policy**

Prefer an existing dependency if Chatbook already ships one suitable for
authenticated encryption. If adding a new dependency, document why and update
packaging/test requirements in the same PR.

Minimum requirement: authenticated encryption with random nonce per payload and
versioned envelope metadata.

- [ ] **Step 2: Write failing crypto tests**

Cover:

- encrypt/decrypt round-trip
- two encryptions of same plaintext produce different ciphertext
- wrong key fails
- recovery bundle stores wrapped material, not plaintext dataset key
- known plaintext never appears in serialized envelope

Example:

```python
def test_encrypt_private_payload_does_not_leak_plaintext():
    encrypted = encrypt_sync_payload({"body": "known private text"}, key=dataset_key)

    serialized = encrypted.model_dump_json()
    assert "known private text" not in serialized
    assert decrypt_sync_payload(encrypted, key=dataset_key)["body"] == "known private text"
```

- [ ] **Step 3: Run crypto tests and verify failure**

Run:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook
python -m pytest Tests/Sync_Interop/test_sync_crypto.py -v
```

Expected: FAIL until crypto module exists.

- [ ] **Step 4: Implement crypto and recovery bundle hooks**

Add:

- dataset key generation
- payload encryption/decryption
- recovery bundle wrap/unwrap primitives
- sync state references for key readiness
- server API call for `/api/v1/sync/keys/recovery-bundle`

Do not log plaintext, dataset keys, or recovery material.

- [ ] **Step 5: Pass crypto and service tests**

Run:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook
python -m pytest \
  Tests/Sync_Interop/test_sync_crypto.py \
  Tests/Sync_Interop/test_server_sync_service.py \
  -v
```

Expected: PASS.

- [ ] **Step 6: Commit in Chatbook repo**

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook
git add tldw_chatbook/Sync_Interop/crypto.py \
  tldw_chatbook/Sync_Interop/server_sync_service.py \
  tldw_chatbook/Sync_Interop/sync_state.py \
  Tests/Sync_Interop/test_sync_crypto.py \
  Tests/Sync_Interop/test_server_sync_service.py
git commit -m "feat(sync): encrypt local-first sync payloads"
```

### Task 10: Add Chatbook Envelope Builders, Appliers, And Local Domain Adapters

**Files:**

- Create: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/Sync_Interop/envelope_builder.py`
- Create: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/Sync_Interop/envelope_applier.py`
- Create: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/Sync_Interop/domain_adapters/__init__.py`
- Create: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/Sync_Interop/domain_adapters/notes.py`
- Create: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/Sync_Interop/domain_adapters/chat.py`
- Create: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/Sync_Interop/domain_adapters/workspaces.py`
- Create: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/Sync_Interop/domain_adapters/source_cache.py`
- Create: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/Sync_Interop/domain_adapters/media.py`
- Tests:
  - `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Sync_Interop/test_envelope_builder.py`
  - `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Sync_Interop/test_envelope_applier.py`

- [ ] **Step 1: Write failing builder/applier tests**

Cover:

- note body goes into encrypted payload
- chat messages append by stable ID
- workspace source ref add/remove maps to link/unlink style envelopes
- source cache uses source ID plus content hash
- local conflict is recorded instead of overwriting encrypted content edits

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook
python -m pytest \
  Tests/Sync_Interop/test_envelope_builder.py \
  Tests/Sync_Interop/test_envelope_applier.py \
  -v
```

Expected: FAIL until modules exist.

- [ ] **Step 3: Implement builders and appliers by domain**

Implementation order:

1. note metadata and encrypted content
2. chat threads/messages
3. workspace source references
4. source cache
5. media compatibility

Keep adapters small and unit-testable. Do not bury merge policy inside UI code.

- [ ] **Step 4: Pass builder/applier tests**

Run:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook
python -m pytest \
  Tests/Sync_Interop/test_envelope_builder.py \
  Tests/Sync_Interop/test_envelope_applier.py \
  Tests/Sync_Interop/test_sync_scope_service.py \
  -v
```

Expected: PASS.

- [ ] **Step 5: Commit in Chatbook repo**

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook
git add tldw_chatbook/Sync_Interop/envelope_builder.py \
  tldw_chatbook/Sync_Interop/envelope_applier.py \
  tldw_chatbook/Sync_Interop/domain_adapters \
  Tests/Sync_Interop/test_envelope_builder.py \
  Tests/Sync_Interop/test_envelope_applier.py \
  Tests/Sync_Interop/test_sync_scope_service.py
git commit -m "feat(sync): add chatbook sync envelopes and adapters"
```

### Task 11: Add Restore Manifest And Conflict Review Flow

**Files:**

Server:

- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/sync.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_service.py`

Chatbook:

- Create: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/Sync_Interop/restore_service.py`
- Modify: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/Sync_Interop/server_sync_service.py`
- Modify: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/tldw_chatbook/Sync_Interop/sync_readiness.py`
- Test: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Sync_Interop/test_restore_service.py`

- [ ] **Step 1: Write failing restore/conflict tests**

Server coverage:

- restore manifest lists datasets/domains/counts without plaintext
- restore manifest supports dataset filters for selective restore preview
- restore manifest supports domain filters for selective restore preview
- restore manifest includes registered devices and last-seen timestamps
- restore manifest includes unresolved conflict counts per dataset/domain
- restore manifest includes attachment availability and size-class summaries
- restore manifest includes encryption policy and key/recovery-bundle readiness
  status
- conflicts list unresolved records
- conflict resolve creates a resolution envelope

Chatbook coverage:

- restore service fetches manifest
- restore service can distinguish locked encrypted datasets from restorable
  datasets with available local key or recovery bundle
- restore service exposes conflict counts and attachment availability in
  metadata without requiring plaintext decrypt
- user selection becomes filtered pull requests
- encrypted content is decrypted before local apply
- conflicts remain visible until resolved

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py \
  -v
```

Run:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook
python -m pytest Tests/Sync_Interop/test_restore_service.py -v
```

Expected: FAIL until restore/conflict implementation exists.

- [ ] **Step 3: Implement restore and conflict flows**

Server:

- build metadata-only restore manifest
- apply dataset/domain filters before calculating restore summaries
- include dataset scope, domains, approximate counts, byte estimates, and last
  update time
- include registered device names/IDs and last-seen timestamps
- include unresolved conflict counts by dataset/domain
- include attachment availability and small/large size-class summaries
- include encryption policy, key-record presence, and recovery-bundle readiness
- list conflict summaries
- accept conflict resolution envelope

Chatbook:

- fetch manifest
- surface locked/unlocked/recovery-available dataset state from manifest
- surface conflict counts and attachment availability before restore
- present/select dataset/domain IDs through service API
- pull selected domains
- decrypt and apply
- record conflicts locally

UI wiring can be a separate follow-up if services expose testable state first.

- [ ] **Step 4: Pass restore/conflict tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py \
  -v
```

Run:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook
python -m pytest \
  Tests/Sync_Interop/test_restore_service.py \
  Tests/Sync_Interop/test_sync_readiness.py \
  -v
```

Expected: PASS.

- [ ] **Step 5: Commit in each repo**

Server:

```bash
git add tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/api/v1/endpoints/sync.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py
git commit -m "feat(sync): add restore and conflict endpoints"
```

Chatbook:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook
git add tldw_chatbook/Sync_Interop/restore_service.py \
  tldw_chatbook/Sync_Interop/server_sync_service.py \
  tldw_chatbook/Sync_Interop/sync_readiness.py \
  Tests/Sync_Interop/test_restore_service.py \
  Tests/Sync_Interop/test_sync_readiness.py
git commit -m "feat(sync): add restore flow"
```

### Task 12: Add End-To-End Sync Restore Coverage And Documentation

**Files:**

- Create: `tldw_Server_API/tests/e2e/test_chatbook_sync_v2_restore.py`
- Update: `tldw_Server_API/app/core/Sync/README.md`
- Update: `Docs/Design/Sync-Engine.md`
- Create or update: `Docs/API/sync-v2.md`
- Update Chatbook docs as needed under
  `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Docs/`

- [ ] **Step 1: Write failing end-to-end test**

Scenario:

1. Register device A.
2. Enroll personal dataset.
3. Push encrypted note, chat message, workspace source ref, and source cache
   envelope.
4. Register device B.
5. Fetch restore manifest.
6. Pull selected domains.
7. Verify no plaintext private content appears in server-visible manifest/log
   representations.
8. Verify duplicate push is idempotent.

- [ ] **Step 2: Run e2e test and verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/e2e/test_chatbook_sync_v2_restore.py -v
```

Expected: FAIL until the server/client pieces are wired.

- [ ] **Step 3: Implement missing glue uncovered by e2e test**

Only fix integration gaps. Do not rewrite already-tested unit behavior in this
task.

- [ ] **Step 4: Update docs**

Document:

- endpoint overview
- protocol invariants
- encryption expectations
- legacy `/send` and `/get` policy
- restore flow
- conflict policy
- operational limits and known non-scope

- [ ] **Step 5: Run verification suite**

Server:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Sync \
  tldw_Server_API/tests/e2e/test_chatbook_sync_v2_restore.py \
  -v
python -m bandit -r tldw_Server_API/app/core/Sync tldw_Server_API/app/api/v1/endpoints/sync.py -f json -o /tmp/bandit_sync_v2.json
```

Chatbook:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook
python -m pytest \
  Tests/Sync_Interop \
  Tests/tldw_api/test_sync_client.py \
  -v
```

Expected: PASS, with any Bandit findings triaged and fixed for touched code.

- [ ] **Step 6: Commit final hardening/docs**

```bash
git add tldw_Server_API/tests/e2e/test_chatbook_sync_v2_restore.py \
  tldw_Server_API/app/core/Sync/README.md \
  Docs/Design/Sync-Engine.md \
  Docs/API/sync-v2.md
git commit -m "docs(sync): document sync v2 restore workflow"
```

Commit Chatbook doc/test updates separately in the Chatbook repo.

## Cross-Cutting Requirements

### Backlog And Branching

- Create or reuse a Backlog task before each implementation PR.
- Prefer one branch per work package.
- Do not mix server and Chatbook changes in one repo commit.
- If a feature requires cross-repo coordination, reference both commit SHAs in
  the final task notes.

### Compatibility

- Preserve current `/api/v1/sync/send` and `/api/v1/sync/get` behavior until a
  compatibility wrapper or deprecation is explicitly implemented and tested.
- Do not remove legacy Chatbook media sync methods until the new media
  compatibility path passes tests.

### Security

- Treat note titles, note bodies, message content, extracted source text,
  transcripts, summaries, and small attachment contents as private encrypted
  payload for local-first personal datasets.
- Do not log payload ciphertext, wrapped key blobs, plaintext, dataset keys, or
  recovery secrets.
- Keep restore manifests metadata-only for encrypted personal datasets.
- Run Bandit on touched server code before each server implementation PR.

### Testing

- Use TDD for each task.
- Keep domain adapter tests focused on merge/conflict behavior.
- Add endpoint tests for auth, authorization, idempotency, and error mapping.
- Add at least one end-to-end restore test before calling the feature complete.

### Observability And Operations

- Add structured Loguru logs with payload redaction.
- Track counts for accepted, rejected, and conflicted envelopes.
- Expose enough conflict and cursor state for admin diagnostics without leaking
  private content.
- Add quotas and attachment-size limits before enabling large user-facing
  payloads.

## Completion Criteria

Sync v2 is complete when:

- Chatbook can remain local-only with no sync side effects.
- Chatbook can register as a device, enroll a personal dataset, push encrypted
  envelopes, pull remote envelopes, and persist cursors.
- A second Chatbook device can view a restore manifest and selectively restore
  V1 domains.
- Server-front-end mode is selectable and does not create local sync state.
- Existing media sync behavior is either preserved or intentionally migrated
  with compatibility tests.
- Private content is absent from server-visible clear payload fields, restore
  manifests, and logs.
- Risky concurrent edits produce conflict records instead of silent overwrites.
- Server `tldw_Server_API/tests/Sync` and the new restore e2e test pass.
- Chatbook `Tests/Sync_Interop` and `Tests/tldw_api/test_sync_client.py` pass.
- Bandit has been run against touched server code and new findings are fixed or
  explicitly triaged.
