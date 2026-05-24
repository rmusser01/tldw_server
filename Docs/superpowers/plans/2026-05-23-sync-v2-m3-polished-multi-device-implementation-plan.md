# Sync v2 M3 Polished Multi-Device Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move Sync v2 from reliable manual personal sync into capability-gated polished multi-device sync.

**Architecture:** Add M3 features incrementally behind explicit capabilities while preserving the envelope log, materializer, restore, and blob contracts established by M1 and M2. Start with device lifecycle and acknowledgments before background sync, workspace datasets, broader domains, stricter encryption, retention/GC, and diagnostics.

**Tech Stack:** FastAPI, Pydantic, SQLite-backed Sync v2 store, existing Sync v2 service/factory/materializer boundaries, pytest, Ruff, Bandit, Backlog.md.

---

## File Map

- `Docs/Design/Sync_V2_M3_Polished_Multi_Device.md`: design gate for M3 scope and sequencing.
- `Docs/API/Sync_V2_M3.md`: API contract draft for capability-gated M3 additions.
- `tldw_Server_API/app/core/Sync/v2/models.py`: new M3 dataclasses for device lifecycle, acknowledgments, background policy, workspace datasets, retention, and diagnostics.
- `tldw_Server_API/app/api/v1/schemas/sync_v2_models.py`: Pydantic request/response models for M3 endpoints.
- `tldw_Server_API/app/core/Sync/v2/store.py`: schema migrations and repository methods for M3 storage.
- `tldw_Server_API/app/core/Sync/v2/service.py`: capability gating and orchestration.
- `tldw_Server_API/app/core/Sync/v2/profile.py`: profile/background/device status aggregation.
- `tldw_Server_API/app/api/v1/endpoints/sync.py`: API wiring under `/api/v1/sync`.
- `tldw_Server_API/app/core/Sync/v2/domain_adapters/*.py`: domain admission and conflict behavior as domains graduate.
- `tldw_Server_API/app/core/Sync/v2/materializers/*.py`: projection behavior for newly enabled domains.
- `tldw_Server_API/tests/Sync/`: focused unit/integration tests for each M3 slice.
- `tldw_Server_API/tests/e2e/test_chatbook_sync_v2_restore.py`: end-to-end smoke coverage after device/background/workspace flows are usable.

## Stage 1: M3 Contract And Backlog Gate

**Goal:** Refine M3 requirements after M1/M2 and split implementation into reviewable Backlog tasks.

**Success Criteria:** Design/API docs exist, parent and child Backlog tasks identify M3 slices, and no production code is changed.

**Tests:** `git diff --check`; stale-placeholder scan over new docs; Bandit skipped as docs-only.

**Status:** Complete

- [x] **Step 1: Create TASK-490.13.1**

  Track this planning gate as a child of `TASK-490.13`.

- [x] **Step 2: Write M3 design and API docs**

  Create:

  - `Docs/Design/Sync_V2_M3_Polished_Multi_Device.md`
  - `Docs/API/Sync_V2_M3.md`

- [x] **Step 3: Create implementation child tasks**

  Add Backlog children for device lifecycle, background sync policy/status,
  workspace dataset foundation, broader domain staging, stricter encryption/key
  rotation, retention/GC, observability, and final verification.

- [x] **Step 4: Verify docs**

  Run:

  ```bash
  git diff --check
  rg -n "T[B]D|T[O]DO|FIX[M]E|\\bM[2]\\b.*M[3]|client-only.*server-front-end.*m[u]st" Docs/Design/Sync_V2_M3_Polished_Multi_Device.md Docs/API/Sync_V2_M3.md Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
  ```

  Expected: `git diff --check` exits 0. `rg` should return no stale placeholders or contradictions unless the match is intentionally reviewed and documented.

- [ ] **Step 5: Commit planning gate**

  ```bash
  git add Docs/Design/Sync_V2_M3_Polished_Multi_Device.md Docs/API/Sync_V2_M3.md Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md backlog/tasks
  git commit -m "docs(sync): plan sync v2 m3 multi-device roadmap"
  ```

## Stage 2: Device Lifecycle And Acknowledgments

**Goal:** Make registered devices manageable and add acknowledgment primitives required for later background sync and retention.

**Success Criteria:** Users can list/update/pause/authorize/revoke devices; revoked devices fail closed across existing Sync v2 APIs; per-device domain/blob acknowledgments are persisted and idempotent.

**Tests:** Store tests, service tests, endpoint tests, cross-user/revoked-device regression tests.

**Status:** Complete

- [x] **Step 1: Write failing store tests**

  Cover device status fields, pending authorization, revocation, idempotent
  acknowledgments, and cursor lag queries in
  `tldw_Server_API/tests/Sync/test_sync_v2_store.py`.

- [x] **Step 2: Implement store schema and methods**

  Extend `sync_devices`, add acknowledgment tables, and expose repository
  helpers through `SyncV2Store`.

- [x] **Step 3: Write failing service/endpoint tests**

  Cover `GET /sync/devices`, `PATCH /sync/devices/{device_id}`,
  `POST /sync/device-authorizations`,
  `POST /sync/device-authorizations/{authorization_id}/approve`,
  `POST /sync/devices/{device_id}/revoke`, and
  `POST /sync/device-acknowledgments`.

- [x] **Step 4: Implement service and endpoint behavior**

  Add Pydantic models, service methods, revoked-device checks, and safe error
  mapping.

- [x] **Step 5: Verify and commit**

  Run targeted Sync tests, Ruff on touched files, Bandit on touched production
  scope, and `git diff --check`.

## Stage 3: Background Sync Policy And Status

**Goal:** Add server-side policy, status, and lease primitives for client-run background sync.

**Success Criteria:** A device can fetch policy hints, store pause/resume intent, create advisory leases, and retrieve background sync status without replacing push/pull/blob idempotency.

**Tests:** Store/service/endpoint tests for policy, leases, status aggregation, lease expiry, idempotency, and cross-user isolation.

**Status:** Complete

- [x] **Step 1: Write failing model and schema tests**

  Validate background policy, lease, and status request/response shapes.

- [x] **Step 2: Implement storage**

  Add policy and lease persistence keyed by dataset/device.

- [x] **Step 3: Implement service aggregation**

  Combine cursor lag, last push/pull, conflict count, replayable failures,
  quota pressure, and blob completeness.

- [x] **Step 4: Wire endpoints**

  Add `GET/PATCH /sync/background-policy`,
  `POST /sync/background-leases`, and `GET /sync/background-status`.

- [x] **Step 5: Verify and commit**

  Run targeted tests, relevant Sync suite, Ruff, Bandit, and `git diff --check`.

## Stage 4: Workspace Dataset Foundation

**Goal:** Introduce workspace-scoped datasets with explicit permission and key policy boundaries before enabling broad collaborative content sync.

**Success Criteria:** Dataset scope supports `personal` and `workspace`; workspace enrollment and all dataset-scoped operations fail closed when membership is missing; initial workspace domains are limited to workspace metadata/source references.

**Tests:** Store/service/endpoint tests for workspace enrollment, permission changes, cross-user isolation, and domain admission.

**Status:** Complete

- [x] **Step 1: Identify existing workspace auth helpers**

  Read existing workspace/RBAC code and document the helper boundary in the
  task notes before production edits.

- [x] **Step 2: Write failing dataset scope tests**

  Cover workspace dataset creation, membership checks, and personal/workspace
  object isolation.

- [x] **Step 3: Implement dataset scope and permission checks**

  Extend dataset records and centralize permission enforcement for push, pull,
  restore, blobs, keys, conflicts, and repair.

- [x] **Step 4: Enable first workspace domains**

  Register only `workspaces.workspace` and `workspaces.source_ref` where
  materialization ownership is clear.

- [x] **Step 5: Verify and commit**

  Run targeted tests, relevant Sync tests, Ruff, Bandit, and `git diff --check`.

## Stage 5: Broader Domain Expansion

**Goal:** Expand sync domain coverage without mixing source-of-truth ownership.

**Success Criteria:** Source cache and media metadata domains have stable object identity, conflict rules, tombstones, restore inventory, and projection behavior before derived content domains are considered.

**Tests:** Domain adapter tests, materializer tests, restore-preview tests, replay/repair tests, and cross-user/workspace isolation tests.

**Status:** Complete

- [x] **Step 1: Promote source cache domain**

  Write failing adapter/materializer/restore tests, then implement the smallest
  projection path.

- [x] **Step 2: Promote media metadata domains**

  Add `media.item`, `media.keyword`, and `media.keyword_link` with metadata-only
  semantics and blob references through M2 attachment/blob paths.

- [x] **Step 3: Reassess derived content**

  Document whether transcripts, summaries, embeddings, and evaluation artifacts
  are source-of-truth sync domains or rebuildable cache.

- [x] **Step 4: Verify and commit each domain group**

  Keep commits per domain family.

## Stage 6: Stricter Encryption And Key Rotation

**Goal:** Add passphrase/device-wrapped policies and key rotation while keeping server-trusted mode working.

**Success Criteria:** Dataset policies advertise honest capabilities; key epochs and rewrap status are tracked; revoked/superseded keys cannot be used for new envelopes; client-only limitations are explicit.

**Tests:** Model/store/service/endpoint tests for policy validation, rotation preview/commit, revoked key rejection, safe redaction, and server-front-end limitations.

**Status:** Complete

- [x] **Step 1: Write failing policy model tests**

  Cover `server_trusted_v1`, `passphrase_wrapped_v1`, `device_wrapped_v1`, and
  `client_private_v1` validation.

- [x] **Step 2: Implement key epoch storage**

  Extend key record metadata and add rotation state.

- [x] **Step 3: Add rotation preview and commit APIs**

  Implement idempotent rotation flows with safe error mapping and redaction.

- [x] **Step 4: Gate server-front-end limitations**

  Prevent server-side mutation of opaque client-private fields.

- [x] **Step 5: Verify and commit**

  Run targeted tests, security/redaction tests, Ruff, Bandit, and `git diff --check`.

## Stage 7: Retention, Compaction, GC, And Observability

**Goal:** Add safe retention/GC and diagnostics after device acknowledgments exist.

**Success Criteria:** Dry-run identifies candidates without mutation; compaction is policy-gated; blob GC requires no active refs and no unacknowledged restore windows; diagnostics are useful and redacted.

**Tests:** Retention dry-run tests, compaction safety tests, blob GC tests, diagnostics redaction tests, and audit-event tests.

**Status:** Not Started

- [ ] **Step 1: Write failing retention dry-run tests**

  Cover unacknowledged devices, tombstone windows, audit mode, and blob refs.

- [ ] **Step 2: Implement dry-run only**

  Add candidate calculation without deletion.

- [ ] **Step 3: Add diagnostics endpoint**

  Report health, counts, lag, quota pressure, key blockers, and retention
  candidates with redaction.

- [ ] **Step 4: Implement guarded compaction/GC**

  Only after dry-run tests prove safety and defaults remain conservative.

- [ ] **Step 5: Verify and commit**

  Run targeted tests, broader Sync suite, Ruff, Bandit, and `git diff --check`.

## Stage 8: End-To-End M3 Verification

**Goal:** Prove the polished multi-device path works across realistic scenarios.

**Success Criteria:** E2E coverage includes two devices with background status, revoked-device denial, workspace dataset access changes, conflict preview/resolution, stricter key policy behavior, retention dry-run, and diagnostics redaction.

**Tests:** E2E restore/sync tests plus full relevant Sync suite.

**Status:** Not Started

- [ ] **Step 1: Extend e2e restore/sync tests**

  Add M3 multi-device scenarios without requiring a real Chatbook client process.

- [ ] **Step 2: Update API docs**

  Reflect implemented subset and explicit deferrals.

- [ ] **Step 3: Run final verification**

  Run Sync tests, e2e tests, Ruff, Bandit on touched production scope, and
  `git diff --check`.

- [ ] **Step 4: Close Backlog tasks**

  Update child tasks and parent `TASK-490.13` with final summaries and known
  deferrals.
