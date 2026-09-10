# Notes Attachment Legacy Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Source-verify and resumably import legacy Notes attachment files into canonical v2 registry/blob state without deleting or trusting legacy paths as authority.

**Architecture:** A dedicated bootstrapper pages authoritative owned note IDs, enumerates each confined legacy directory with hard caps, records an immutable source map, uploads bytes through the existing verified blob path, and appends trusted bootstrap envelopes. Readiness is independent from schema, organization, and notes.link state; cleanup candidates remain non-authoritative evidence only.

**Tech Stack:** Python pathlib/stat/hashlib, Sync v2 trusted bootstrap, SQLite/PostgreSQL Sync store, ChaChaNotes registry, pytest, Ruff, Bandit.

**Backlog task:** `TASK-13005.3`
**Depends on:** `TASK-13005.1`, `TASK-13005.2`
**ADR required:** no
**ADR path:** `Docs/ADR/038-canonical-notes-attachment-registry-and-blob-lifecycle.md`
**Reason:** The approved ADR already defines source verification, non-destructive import, readiness, and cleanup authority.

---

### Task 1: Extract a read-only legacy source seam

**Files:**
- Create: `tldw_Server_API/app/core/Notes/legacy_attachment_source.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/notes.py`
- Create: `tldw_Server_API/tests/Notes/test_legacy_attachment_source.py`

- [ ] Write RED tests for authoritative note-ID directory derivation, root confinement,
  symlink/path escape, sorted immutable source keys, 200-note pages, 1,000 candidates
  per note/run, 64 KiB sidecar cap, 4,096-byte source cursor, and soft-deleted notes.
- [ ] Implement read-only enumeration/stat/hash; do not move, rename, or delete files.
- [ ] Reuse the seam from inactive routes without changing response behavior.
- [ ] Run RED, then GREEN and commit:

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Notes/test_legacy_attachment_source.py
git add tldw_Server_API/app/core/Notes/legacy_attachment_source.py \
  tldw_Server_API/app/api/v1/endpoints/notes.py \
  tldw_Server_API/tests/Notes/test_legacy_attachment_source.py
git commit -m "refactor(notes): isolate legacy attachment source"
```

### Task 2: Persist readiness, source map, and cleanup candidates

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/Sync_DB.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/models.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/store.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_store.py`
- Create: `tldw_Server_API/tests/Sync/test_sync_v2_notes_attachment_bootstrap.py`

- [ ] Write RED tests for idempotent `notes_attachment_v2` begin/transition CAS,
  stable bootstrap ID, adapter target 2, source counts/hash/cursor, failed safe code,
  one real UUIDv4 allocated once per source key, owner-root-relative internal path,
  public-safe path hash, and no routing/log path leakage.
- [ ] Add bounded Sync tables for source mappings and cleanup candidates. Identity is
  `(dataset_id, bootstrap_id, source_key_hash)`; source paths never enter envelopes.
- [ ] Existing v1 heads remain pullable; v2 writes remain closed until verified ready.
- [ ] Run RED, then GREEN and commit:

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Sync/test_sync_v2_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_attachment_bootstrap.py \
  -k 'attachment_bootstrap or source_map or cleanup_candidate'
git add tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/app/core/Sync/v2/models.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_attachment_bootstrap.py
git commit -m "feat(sync): persist attachment bootstrap state"
```

### Task 3: Implement source-verified bootstrap orchestration

**Files:**
- Create: `tldw_Server_API/app/core/Sync/v2/notes_attachment_bootstrap.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/profile.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/factory.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/server_origin_batch.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_notes_attachment_bootstrap.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py`

- [ ] Write RED tests for empty source, multi-page resume, interrupted upload/append/
  projection, source change before/after hashing, same-name suffixes, malformed or
  oversized sidecars/files, ambiguous v1 ID collision, soft-deleted parent visibility,
  exact rerun, and count/fingerprint verification.
- [ ] Implement a bootstrapper patterned after `notes_link_bootstrap.py`, but keep
  streaming outside the dataset guard. Re-stat/re-hash before accepting each source.
- [ ] Import through namespaced verified blob completion, then use trusted
  server-origin v2 capture with a bootstrap-step verifier.
- [ ] Mark ready only after source count and canonical fingerprint match; any source
  blocker leaves files untouched and state failed with a safe code.
- [ ] Run this command once RED before implementation and again GREEN after
  implementation, then commit:

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Notes/test_legacy_attachment_source.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_attachment_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py
git add tldw_Server_API/app/core/Sync/v2/notes_attachment_bootstrap.py \
  tldw_Server_API/app/core/Sync/v2/profile.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/factory.py \
  tldw_Server_API/app/core/Sync/v2/server_origin_batch.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_attachment_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py
git commit -m "feat(sync): bootstrap legacy Notes attachments"
```

### Task 4: Add bounded bootstrap API and diagnostics

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/sync.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sync_v2_models.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/profile.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py`

- [ ] Write RED endpoint tests for owner/dataset authority, dry-run counts, start/
  resume/status, hard caps, safe errors, initializing/failed/ready transitions, and
  no local path/name leakage.
- [ ] Reuse profile bootstrap/resume endpoints where possible; do not add a second
  background worker or automatic startup migration.
- [ ] Return safe counts/cursors only. Cleanup-candidate public samples contain hash,
  stable attachment ID, state, and blocker code—not source path.
- [ ] Run RED, then GREEN and commit:

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py \
  tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py \
  -k 'attachment_bootstrap or cleanup_candidate'
git add tldw_Server_API/app/api/v1/endpoints/sync.py \
  tldw_Server_API/app/api/v1/schemas/sync_v2_models.py \
  tldw_Server_API/app/core/Sync/v2/profile.py \
  tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py \
  tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py
git commit -m "feat(sync): expose attachment bootstrap status"
```

### Task 5: Prove non-destructive rollout and rollback

- [ ] Verify schema upgrade alone creates no registry/blob/source rows.
- [ ] Verify gate-off preserves legacy files/routes before initialization.
- [ ] Verify partial bootstrap never falls back to a second mutable authority.
- [ ] Verify disabling the gate after ready leaves canonical metadata read-only and
  every original legacy file untouched.
- [ ] Run relevant Notes/Sync/blob suites, touched Ruff/formatter, Bandit, compile, and
  `git diff --check`; record live-PG skip truthfully.
- [ ] Update only `TASK-13005.3` notes/AC/DoD and mark it Done.
