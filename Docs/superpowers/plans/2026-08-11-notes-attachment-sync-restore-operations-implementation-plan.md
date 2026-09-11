# Notes Attachment Restore And Operations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete version-aware restore, guarded binding release and physical blob GC, operational diagnostics, public documentation, and end-to-end rollout verification for canonical Notes attachments.

**Architecture:** Restore orders live v2 refs after their note providers and derives completeness through revision binding→blob ID. Retention first releases eligible historical bindings, then atomically fences a dataset-namespaced blob `available→deleting`; every binding/upload writer shares that authority. Storage unlink is idempotent and finalizes `deleted`, while legacy global keys can never be dataset-locally unlinked.

**Tech Stack:** Sync v2 restore/repair/retention services, filesystem blob store, FastAPI diagnostics, SQLite/PostgreSQL contracts, pytest, Ruff, Bandit.

**Backlog task:** `TASK-13005.4`
**Depends on:** `TASK-13005.1`, `TASK-13005.2`, `TASK-13005.3`
**ADR required:** no
**ADR path:** `Docs/ADR/038-canonical-notes-attachment-registry-and-blob-lifecycle.md`
**Reason:** This is direct implementation of the approved restore/retention authority.

---

### Task 1: Make restore adapter-version and binding aware

**Files:**
- Modify: `tldw_Server_API/app/core/Sync/v2/restore.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/replay.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sync_v2_models.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py`

- [ ] Write RED tests for v2 live ref→note dependency ordering, tombstone with no live
  dependency, registered-device version filtering, device-less v1 compatibility,
  adapter version in ordered actions, v1/v2 collision, local inventory, exact group
  metadata, current binding resolution, and missing/quarantined/deleted completeness.
- [ ] Run RED:

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py \
  tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py \
  -k 'attachment_ref_v2 or adapter_version or attachment_binding'
```

- [ ] Add `attachment.ref` v2 to restore allowlists without exposing it to v1-only
  devices. Load current blob by revision binding/blob ID, never provenance attachment
  ID. Keep metadata replay possible when bytes are unavailable.
- [ ] Run GREEN and commit:

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py \
  tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py \
  -k 'attachment_ref_v2 or adapter_version or attachment_binding'
git add tldw_Server_API/app/core/Sync/v2/restore.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/replay.py \
  tldw_Server_API/app/api/v1/schemas/sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py \
  tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py
git commit -m "feat(sync): restore attachment ref v2 history"
```

### Task 2: Release historical bindings monotonically

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/Sync_DB.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/store.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_service.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_store.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_notes_postgres_contract.py`

- [ ] Write RED tests for hidden live blockers, tombstone/audit/restore windows,
  adapter-version domain acks, blob-ID acks, quarantine/repair holds, current vs
  historical revision, 1,000-row cursor pages, monotonic release, replacement
  preserving old evidence, and later restore creating a new protected binding.
- [ ] Implement dry-run candidates and CAS `retention_released_at` without erasing
  digest/size/cursor/blob ID. Never infer eligibility from mutable ref counts.
- [ ] Add bounded query-count assertions and PostgreSQL index-plan contracts for
  release-candidate and outstanding-binding scans.
- [ ] Run RED, then GREEN and commit:

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_postgres_contract.py \
  -k 'binding_release or retention_candidate or attachment_query_plan'
git add tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_postgres_contract.py
git commit -m "feat(sync): release historical attachment bindings"
```

### Task 3: Add the physical-deletion fence and idempotent cleanup seam

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/Sync_DB.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/store.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/blob_store.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_blob_store.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_service.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_workspace_blobs.py`

- [ ] Write barrier RED tests proving a concurrent binding/create/upload cannot target
  a blob after the fence and bytes cannot disappear before the fence.
- [ ] Cover `available→deleting→deleted`, retry after crash between unlink/finalize,
  transient unlink failure, already-absent file, quarantine, same-digest repair only
  after deleted, namespace validation, cross-dataset same digest, and global-key
  nonretryable blocker.
- [ ] Implement one guarded plan/fence transaction. Unlink only after durable
  `deleting`; all binding/resolution/upload-completion paths reject/retry deleting or
  deleted. Never auto-clear deleting to available.
- [ ] Run this command once RED before implementation and again GREEN after
  implementation, then commit:

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Sync/test_sync_v2_blob_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_workspace_blobs.py \
  -k 'retention or physical_gc or deleting or storage_namespace'
git add tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/Sync/v2/blob_store.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_blob_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_workspace_blobs.py
git commit -m "feat(sync): guard attachment blob collection"
```

### Task 4: Add bounded diagnostics and recovery actions

**Files:**
- Modify: `tldw_Server_API/app/core/Sync/v2/profile.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/sync.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sync_v2_models.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py`

- [ ] Write RED tests for owner/dataset-scoped registry/live/hidden/tombstone,
  binding, missing/verify-failed/quarantined/deleting/deleted, orphan, bootstrap,
  cleanup-candidate, retention-blocker, and failed/pending projection counts.
- [ ] Default samples to zero; enforce 100/category and 500/response with 413 rather
  than clamp. Samples contain only authorized IDs and stable codes.
- [ ] Add idempotent retry upload, repair projection, resolve conflict, restore,
  quarantine release, bootstrap resume, and GC retry actions. Destructive action
  remains the existing confirmation-gated retention workflow. Diagnostic responses
  expose only machine-readable recovery-action descriptors/hints; diagnostics never
  invoke those actions or mutate state.
- [ ] Assert before/after product and Sync snapshots are identical for every
  diagnostic request, including when action descriptors are returned.
- [ ] Run RED, then GREEN and commit:

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py \
  -k 'attachment_diagnostic or recovery_action or read_only'
git add tldw_Server_API/app/core/Sync/v2/profile.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/api/v1/endpoints/sync.py \
  tldw_Server_API/app/api/v1/schemas/sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py
git commit -m "feat(sync): diagnose attachment lifecycle state"
```

### Task 5: Public docs and end-to-end verification

**Files:**
- Modify: `Docs/API/Sync_V2_M1.md`
- Modify: `Docs/API/Sync_V2_M2.md`
- Modify: `Docs/API/Sync_V2_M3.md`
- Modify: `Docs/ADR/038-canonical-notes-attachment-registry-and-blob-lifecycle.md`
- Create: `tldw_Server_API/tests/e2e/test_notes_attachment_sync_v2.py`
- Test: `tldw_Server_API/tests/Notes/test_notes_attachment_sync_api.py`
- Test: `tldw_Server_API/tests/ChaChaNotesDB/test_note_attachment_postgres_tenancy.py`

- [ ] Document adapter-version negotiation/cursors/acks, exact payload and API
  schemas, gates/readiness, range behavior, restore completeness, bootstrap limits,
  retention fence, namespace migration, safe errors, and rollback.
- [ ] Add SQLite full lifecycle e2e: one-shot create→rename→replace→trash note→restore
  note→delete attachment→restore→new device pull/restore→retention dry run.
- [ ] Add required live PostgreSQL two-owner identical filename/digest/RLS lifecycle
  and single/suffix/open-ended byte-range reads with exact 206/416 headers; skip only
  without DSN and record that limitation.
- [ ] Run full affected gates:

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_attachment_migration_v59.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_attachment_store.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_attachment_postgres_tenancy.py \
  tldw_Server_API/tests/Notes/test_notes_attachment_sync_api.py \
  tldw_Server_API/tests/Sync/test_sync_v2_attachment_refs.py \
  tldw_Server_API/tests/Sync/test_sync_v2_attachment_materializer.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_attachment_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py \
  tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py \
  tldw_Server_API/tests/Sync/test_sync_v2_blob_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/e2e/test_notes_attachment_sync_v2.py
```

- [ ] Run exact touched Ruff/formatter, Bandit, `py_compile`, `git diff --check`, and
  existing Notes attachment/shared blob/restore/retention regression gates.
- [ ] Commit docs/e2e:

```bash
git add Docs/API/Sync_V2_M1.md Docs/API/Sync_V2_M2.md Docs/API/Sync_V2_M3.md \
  Docs/ADR/038-canonical-notes-attachment-registry-and-blob-lifecycle.md \
  tldw_Server_API/tests/e2e/test_notes_attachment_sync_v2.py \
  tldw_Server_API/tests/Notes/test_notes_attachment_sync_api.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_attachment_postgres_tenancy.py
git commit -m "docs(sync): document Notes attachment lifecycle"
```

- [ ] Update `TASK-13005.4` notes/AC/DoD, and mark it Done.
- [ ] Only after all four children are Done, update parent TASK-13005 evidence and
  check its AC/DoD; do not mark the parent Done earlier.
