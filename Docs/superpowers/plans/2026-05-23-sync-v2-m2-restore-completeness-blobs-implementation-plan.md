# Sync v2 M2 Restore Completeness And Blobs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Sync v2 M2 blob transfer and restore completeness so a Chatbook user can safely restore selected personal Notes/Chat data and referenced blobs on a new or existing device.

**Architecture:** Extend the existing Sync v2 M1 service, store, schemas, and `/api/v1/sync` endpoints in place. Keep the append-only envelope log and materialized server state as-is, add blob metadata/session/chunk storage, and keep blob bytes under the per-user encrypted storage scope through a small local blob-store adapter.

**Tech Stack:** FastAPI, Pydantic, SQLite/Postgres-compatible Sync DB migrations, local filesystem blob storage, pytest, Bandit.

---

## Stage 1: M2 Protocol Models And Capabilities

**Goal:** Define the public M2 request/response contract before storage or endpoint behavior.

**Success Criteria:** Model tests validate upload sessions, chunk manifests, quota/status fields, restore completeness fields, and capability advertisement.

**Tests:** `tldw_Server_API/tests/Sync/test_sync_v2_models.py`, `tldw_Server_API/tests/Sync/test_sync_v2_service.py`

**Status:** Not Started

**Files:**

- Modify: `tldw_Server_API/app/api/v1/schemas/sync_v2_models.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/models.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_models.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_service.py`

- [ ] **Step 1: Write failing schema tests**

Add tests for:

- `SyncBlobUploadCreateRequest`
- `SyncBlobUploadSessionResponse`
- `SyncBlobChunkUploadResponse`
- `SyncBlobUploadCompleteResponse`
- `SyncBlobDownloadManifestResponse`
- `SyncRestoreCompletenessResponse` or equivalent restore preview fields
- capabilities `blob_transfer` and quota details

- [ ] **Step 2: Run schema tests and verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_models.py -v
```

Expected: FAIL because M2 models do not exist.

- [ ] **Step 3: Add Pydantic and core dataclass models**

Implement the minimal public fields from `Docs/Design/Sync_V2_M2_Restore_Completeness_and_Blobs.md`:

- upload ID/session status
- chunk index, offset, size, and `sha256:<hex>` hash
- full payload hash
- quota limits and usage
- blob availability/status
- restore profile status and per-domain/per-blob detail

Keep M2 default encryption policy `server_trusted_v1`; do not require
`client_private_v1` for blob transfer.

- [ ] **Step 4: Extend capabilities**

Update `SyncV2Settings` and `SyncV2Capabilities` so M2 can advertise:

- `resumable_upload`
- `resumable_download`
- `chunk_checksums`
- `full_checksum`
- `max_blob_bytes`
- `max_chunk_bytes`
- `max_active_uploads`
- user/dataset quota values

Keep the runtime default disabled until later stages pass.

- [ ] **Step 5: Run targeted tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py::test_capabilities_include_sync_v2_m1_contract \
  -v
```

Expected: PASS after model and capability updates.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/sync_v2_models.py \
  tldw_Server_API/app/core/Sync/v2/models.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py
git commit -m "feat(sync): define sync v2 m2 blob protocol"
```

## Stage 2: Blob Store, DB Schema, And Quotas

**Goal:** Add durable metadata, upload-session state, chunk tracking, and local blob storage without exposing new endpoints yet.

**Success Criteria:** Store tests prove migrations, idempotent upload sessions, chunk writes, quota reserve/release, checksum verification, dedupe, and safe path handling.

**Tests:** `tldw_Server_API/tests/Sync/test_sync_v2_store.py`, new `tldw_Server_API/tests/Sync/test_sync_v2_blob_store.py`

**Status:** Not Started

**Files:**

- Modify: `tldw_Server_API/app/core/DB_Management/Sync_DB.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/store.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Create: `tldw_Server_API/app/core/Sync/v2/blob_store.py`
- Create: `tldw_Server_API/tests/Sync/test_sync_v2_blob_store.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_store.py`

- [ ] **Step 1: Write failing store/blob tests**

Cover:

- `sync_blob_objects`, `sync_blob_upload_sessions`, and `sync_blob_chunks` exist after initialization.
- upload session create is idempotent by `(dataset_id, device_id, idempotency_key)`.
- quota reservation is recorded on session create and released on cancel/expiry.
- chunk writes reject wrong offset, size, hash, or duplicate mismatched content.
- completion verifies all chunks and full hash before committing.
- duplicate full payload hash dedupes to one committed blob object.
- filesystem paths stay under the configured per-user `sync_blobs` root.

- [ ] **Step 2: Run store tests and verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_blob_store.py \
  -v
```

Expected: FAIL because schema and blob store do not exist.

- [ ] **Step 3: Add migrations and store methods**

Add DB tables for blob objects, upload sessions, and chunks to both SQLite and
Postgres schema strings. Add store methods for:

- create/get/cancel upload session
- record chunk
- list missing chunks
- complete upload metadata
- get blob by attachment ID or payload hash
- summarize quota usage

- [ ] **Step 4: Add local blob-store adapter**

Implement a small adapter responsible for:

- resolving a per-user blob root
- writing chunk temp files
- reading chunk bytes
- assembling verified blob files atomically
- deleting abandoned temp files
- rejecting unsafe paths

Do not add cloud object-store behavior in M2.

- [ ] **Step 5: Run targeted store/blob tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_blob_store.py \
  -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/blob_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_blob_store.py
git commit -m "feat(sync): add sync v2 blob storage ledger"
```

## Stage 3: Resumable Upload API

**Goal:** Expose upload session, chunk upload, status, complete, cancel, and small-blob wrapper APIs.

**Success Criteria:** API tests verify auth isolation, validation, quota errors, idempotency, resume status, checksum failures, and successful completion.

**Tests:** `tldw_Server_API/tests/Sync/test_sync_v2_api.py`

**Status:** Not Started

**Files:**

- Modify: `tldw_Server_API/app/api/v1/endpoints/sync.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_api.py`

- [ ] **Step 1: Write failing upload API tests**

Cover:

- capabilities keep upload disabled until configured.
- create upload rejects inaccessible datasets and unenrolled domains.
- create upload returns the existing session for the same idempotency key.
- chunk upload rejects bad chunk hash, wrong index, or excessive size.
- status returns uploaded and missing chunk indexes.
- complete rejects incomplete sessions and full-hash mismatch.
- complete returns committed blob metadata and availability `available`.
- cancel releases reserved quota.
- existing `POST /api/v1/sync/attachments` routes through the same small-blob commit path.

- [ ] **Step 2: Run upload API tests and verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_api.py -k "blob or attachment or capabilities" -v
```

Expected: FAIL because M2 endpoints are not implemented.

- [ ] **Step 3: Implement service upload operations**

Add service methods that orchestrate dataset authorization, quota checks,
store updates, blob-store writes, completion verification, and safe errors.

- [ ] **Step 4: Add FastAPI routes**

Add routes under `/api/v1/sync/attachments/uploads` and update
`POST /api/v1/sync/attachments` to use the same service path for small blobs.

- [ ] **Step 5: Run targeted API tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_api.py -k "blob or attachment or capabilities" -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/sync.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_api.py
git commit -m "feat(sync): expose resumable blob uploads"
```

## Stage 4: Download, Restore Completeness, And Selective Restore

**Goal:** Make restore preview/manifest aware of server-held blobs and expose resumable download manifests.

**Success Criteria:** Restore tests distinguish metadata-only, blocked-by-conflicts, blob-incomplete, content-complete, and verified-complete states.

**Tests:** `tldw_Server_API/tests/Sync/test_sync_v2_service.py`, `tldw_Server_API/tests/e2e/test_chatbook_sync_v2_restore.py`

**Status:** Not Started

**Files:**

- Modify: `tldw_Server_API/app/core/Sync/v2/restore.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/sync.py`
- Modify: `tldw_Server_API/tests/Sync/test_sync_v2_service.py`
- Modify: `tldw_Server_API/tests/e2e/test_chatbook_sync_v2_restore.py`

- [ ] **Step 1: Write failing restore/download tests**

Cover:

- download manifest returns chunk map, full hash, size, and availability.
- download chunk returns bytes only for the authenticated owner.
- restore preview uses server blob state over client-authored availability.
- metadata-only restore remains possible only when the client explicitly requests it.
- new-device restore with all blobs available reports `content_complete`.
- existing-profile restore with conflicting Note/conversation metadata reports `blocked_by_conflicts`.
- client-supplied verified blob inventory can move status to `verified_complete`.

- [ ] **Step 2: Run restore/download tests and verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/e2e/test_chatbook_sync_v2_restore.py \
  -k "restore or blob or attachment" \
  -v
```

Expected: FAIL because download and completeness logic is not implemented.

- [ ] **Step 3: Implement download service and routes**

Add download manifest and chunk/whole-blob routes with owner checks, chunk
hashes, and safe HTTP errors. Range support can be added if the adapter can
serve ranges without bypassing authorization.

- [ ] **Step 4: Implement restore completeness**

Extend restore preview/manifest output with profile-level status and per-domain
details. Preserve M1 object conflict semantics and append-only message merge.

- [ ] **Step 5: Run targeted restore/download tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/e2e/test_chatbook_sync_v2_restore.py \
  -k "restore or blob or attachment" \
  -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Sync/v2/restore.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/api/v1/endpoints/sync.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/e2e/test_chatbook_sync_v2_restore.py
git commit -m "feat(sync): report restore completeness with blobs"
```

## Stage 5: Key Recovery Hardening, Docs, And Final Verification

**Goal:** Finish M2 readiness by hardening key recovery status, documenting the contract, and running relevant verification.

**Success Criteria:** Restore manifest/preview report recovery readiness safely, docs describe M2 behavior, and targeted tests plus Bandit pass.

**Tests:** `tldw_Server_API/tests/Sync/test_sync_v2_security.py`, targeted Sync suite, Bandit touched production scope.

**Status:** Not Started

**Files:**

- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Modify: `tldw_Server_API/app/core/DB_Management/Sync_DB.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sync_v2_models.py`
- Modify: `Docs/API/Sync_V2_M1.md` or create `Docs/API/Sync_V2_M2.md`
- Modify: `Docs/Design/Sync_V2_M2_Restore_Completeness_and_Blobs.md`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_security.py`

- [ ] **Step 1: Write failing key recovery hardening tests**

Cover:

- invalid key purpose or wrapping metadata is rejected.
- recovery records for other users/datasets are inaccessible.
- revoked records are excluded from readiness.
- restore preview warns when no active recovery bundle exists.
- safe error/log paths never include wrapped key material.

- [ ] **Step 2: Run security tests and verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_security.py -v
```

Expected: FAIL until hardening is implemented.

- [ ] **Step 3: Implement hardening and docs**

Tighten validation and restore readiness, then document:

- M2 capabilities
- upload/download flows
- quota fields
- restore completeness statuses
- M2 non-goals and M3 deferred encryption modes

- [ ] **Step 4: Run targeted Sync tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_security.py \
  tldw_Server_API/tests/Sync/test_sync_v2_api.py \
  tldw_Server_API/tests/e2e/test_chatbook_sync_v2_restore.py \
  -v
```

Expected: PASS or document unrelated pre-existing failures with exact tests.

- [ ] **Step 5: Run Bandit on touched production scope**

Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/api/v1/endpoints/sync.py \
  tldw_Server_API/app/api/v1/schemas/sync_v2_models.py \
  tldw_Server_API/app/core/Sync/v2 \
  tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  -f json -o /tmp/bandit_sync_v2_m2.json
```

Expected: PASS with no new findings in touched code.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/app/api/v1/schemas/sync_v2_models.py \
  Docs/API/Sync_V2_M2.md \
  Docs/Design/Sync_V2_M2_Restore_Completeness_and_Blobs.md \
  tldw_Server_API/tests/Sync/test_sync_v2_security.py
git commit -m "docs(sync): finalize sync v2 m2 blob restore contract"
```

## Final Verification For M2

Before opening the M2 implementation PR:

- [ ] Run targeted Sync v2 tests listed above.
- [ ] Run the e2e restore test.
- [ ] Run Bandit on touched production paths.
- [ ] Run `git diff --check`.
- [ ] Update `TASK-490.12` and all child Backlog tasks with modified files,
  verification output, known skips, and final summary.
- [ ] Ensure the PR includes a human-written Change summary explaining what
  changed and why these implementation choices were made.
