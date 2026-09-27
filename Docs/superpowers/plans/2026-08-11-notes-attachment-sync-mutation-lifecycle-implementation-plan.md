# Notes Attachment Mutation Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route active Notes attachment create, resumable attach/replace, rename, delete, restore, list/detail, and download through one fail-closed Sync v2 coordinator while preserving inactive filename behavior.

**Architecture:** A focused Notes attachment coordinator owns dataset/readiness/idempotency preflight and calls the existing server-origin append/materialization seam. The v2 materializer projects into the v59 registry; bytes remain in the shared blob service. Existing routes become compatibility aliases, while additive stable-ID routes expose strict ETag and pagination contracts.

**Tech Stack:** FastAPI, Pydantic, Sync v2 server-origin batches, ChaChaNotes attachment store, resumable blob service, pytest, Ruff, Bandit.

**Backlog task:** `TASK-13005.2`
**Depends on:** `TASK-13005.1`
**ADR required:** no
**ADR path:** `Docs/ADR/038-canonical-notes-attachment-registry-and-blob-lifecycle.md`
**Reason:** This is direct implementation of the approved coordinator/API boundary.

---

### Task 1: Add strict Notes attachment API schemas

**Files:**
- Create: `tldw_Server_API/app/api/v1/schemas/notes_attachments.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sync_v2_models.py`
- Create: `tldw_Server_API/tests/Notes/test_notes_attachment_sync_api.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_models.py`

- [ ] Write failing tests for one-shot responses, canonical items/pages, rename-only
  PATCH, strict delete/restore reason, `{upload_id}` from-upload, attachment intent,
  ETag grammar, 128-byte idempotency key, 512-byte keyset cursor, and extra fields.
- [ ] Run RED:

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Notes/test_notes_attachment_sync_api.py \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py \
  -k 'schema or etag or attachment_intent'
```

- [ ] Implement only the models/validators; keep routes disabled.
- [ ] Run GREEN and commit:

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Notes/test_notes_attachment_sync_api.py \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py \
  -k 'schema or etag or attachment_intent'
git add tldw_Server_API/app/api/v1/schemas/notes_attachments.py \
  tldw_Server_API/app/api/v1/schemas/sync_v2_models.py \
  tldw_Server_API/tests/Notes/test_notes_attachment_sync_api.py \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py
git commit -m "feat(notes): define attachment lifecycle api"
```

### Task 2: Implement the v2 domain adapter and materializer

**Files:**
- Create: `tldw_Server_API/app/core/Sync/v2/domain_adapters/attachment_refs.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/materializers/attachment_refs.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/factory.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_attachment_refs.py`
- Create: `tldw_Server_API/tests/Sync/test_sync_v2_attachment_materializer.py`

- [ ] Write RED tests for create, pending binding, resolved binding, rename, replace,
  tombstone, routing-only restore, exact replay, stale base, name collision, hidden
  parent behavior, postcondition replay, and v1/v2 identity collision. Pair present-
  and missing-byte acceptance cases to prove immutable `availability_at_acceptance`,
  exact digest/size resolution, and no payload/object-hash/idempotency rehash or
  post-submit enrichment when a pending binding later resolves.
- [ ] Use real SQLite Notes + Sync stores; recording-only materializers are not enough.
- [ ] Run RED, implement minimal adapter/materializer, then GREEN:

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Sync/test_sync_v2_attachment_refs.py \
  tldw_Server_API/tests/Sync/test_sync_v2_attachment_materializer.py
```

- [ ] Ensure deterministic validation/caps happen before product writes and exact
  postcondition replay does not advance revision/timestamps twice.
- [ ] Commit:

```bash
git add tldw_Server_API/app/core/Sync/v2/domain_adapters/attachment_refs.py \
  tldw_Server_API/app/core/Sync/v2/materializers/attachment_refs.py \
  tldw_Server_API/app/core/Sync/v2/factory.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_attachment_refs.py \
  tldw_Server_API/tests/Sync/test_sync_v2_attachment_materializer.py
git commit -m "feat(sync): materialize Notes attachment refs"
```

### Task 3: Add the owner-bound attachment coordinator

**Files:**
- Create: `tldw_Server_API/app/core/Sync/v2/notes_attachment_coordinator.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/server_origin.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/server_origin_batch.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Create: `tldw_Server_API/tests/Sync/test_sync_v2_notes_attachment_coordinator.py`

- [ ] Write RED tests for optional dataset resolution, exact default-personal
  authority, inactive `None`, sticky canonical authority, gate/readiness failures,
  note read-set races, upload-intent drift, idempotency replay, and zero product writes
  before deterministic rejection. Add failure injection after blob commit, mutation
  manifest persistence, envelope append, registry projection commit, apply-status
  update, and response-manifest persistence. Exact retry must reuse the verified blob
  and must not advance attachment revision, stable timestamps, quota accounting, or
  response data twice.
- [ ] Implement one coordinator with explicit owner; never infer owner from device ID
  or silently fall back after canonical initialization.

```python
class NotesAttachmentCoordinator:
    def require_mutation_ready(self, *, owner_id: str, dataset_id: str | None) -> ReadyAttachmentDataset: ...
    def capture(self, plan: NotesAttachmentMutationPlan) -> NotesAttachmentMutationResult: ...
```

- [ ] Hold the dataset projection guard only for final note/registry/blob recheck,
  append, and projection—not while streaming request bytes.
- [ ] Run RED, then GREEN and commit:

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_attachment_coordinator.py \
  -k 'dataset or readiness or crash_window or exact_retry'
git add tldw_Server_API/app/core/Sync/v2/notes_attachment_coordinator.py \
  tldw_Server_API/app/core/Sync/v2/server_origin.py \
  tldw_Server_API/app/core/Sync/v2/server_origin_batch.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_attachment_coordinator.py
git commit -m "feat(sync): coordinate Notes attachment mutations"
```

### Task 4: Bind resumable upload intent and namespaced blob completion

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/sync.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sync_v2_models.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/store.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/blob_store.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_workspace_blobs.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_blob_store.py`
- Test: `tldw_Server_API/tests/Notes/test_notes_attachment_sync_api.py`

- [ ] Write RED tests for strict create/replace intent, requested-name binding,
  positive size/lowercase digest, owner/dataset/note/attachment immutability, opaque
  upload ID, long-upload same-name race, namespaced storage, completion retry, and
  from-upload cross-note/cross-attachment denial.
- [ ] Extend the existing `/api/v1/sync/blob-uploads` request only for
  `domain="attachment.ref"`; keep other domain behavior unchanged.
- [ ] Allocate the final create suffix under the commit guard and persist it in the
  response manifest. Replacement must match the session base and `If-Match`.
- [ ] Run RED, then GREEN and commit:

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Sync/test_sync_v2_workspace_blobs.py \
  tldw_Server_API/tests/Sync/test_sync_v2_blob_store.py \
  tldw_Server_API/tests/Notes/test_notes_attachment_sync_api.py \
  -k 'attachment_intent or namespace or completion or from_upload'
git add tldw_Server_API/app/api/v1/endpoints/sync.py \
  tldw_Server_API/app/api/v1/schemas/sync_v2_models.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/Sync/v2/blob_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_workspace_blobs.py \
  tldw_Server_API/tests/Sync/test_sync_v2_blob_store.py \
  tldw_Server_API/tests/Notes/test_notes_attachment_sync_api.py
git commit -m "feat(sync): bind Notes attachment upload intent"
```

### Task 5: Add canonical APIs and route active compatibility writes

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/notes.py`
- Modify: `tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/notes_attachment_coordinator.py`
- Test: `tldw_Server_API/tests/Notes/test_notes_attachment_sync_api.py`
- Test: `tldw_Server_API/tests/Notes/test_notes_api_integration.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_notes_postgres_contract.py`

- [ ] Write endpoint RED tests for canonical list/detail/content/from-upload/PATCH/
  DELETE/restore, static-before-filename route order, keyset pagination, ETag 428/409,
  exact replay, strict errors, range conditionals, hidden notes, tombstone filter, and
  active/inactive/partial/failed gate matrices.
- [ ] Assert canonical list/detail use bounded query counts and PostgreSQL contract
  tests prove index-backed owner/dataset/keyset lookups rather than wall-clock timing.
- [ ] Refactor legacy filesystem helpers without changing inactive semantics. Under
  active Sync, route POST/list/download/delete through the coordinator/registry.
- [ ] Implement only single `bytes=` ranges and exact 200/206/304/400/416 behavior.
- [ ] Preserve optional Idempotency-Key on compatibility routes; do not claim exact
  replay when it is omitted.
- [ ] Run this command once RED before implementation and again GREEN after
  implementation, then commit:

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Notes/test_notes_attachment_sync_api.py \
  tldw_Server_API/tests/Notes/test_notes_api_integration.py \
  tldw_Server_API/tests/Sync/test_sync_v2_attachment_materializer.py \
  tldw_Server_API/tests/Sync/test_sync_v2_workspace_blobs.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_postgres_contract.py
git add tldw_Server_API/app/api/v1/endpoints/notes.py \
  tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py \
  tldw_Server_API/app/api/v1/schemas/notes_attachments.py \
  tldw_Server_API/app/core/Sync/v2/notes_attachment_coordinator.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/tests/Notes/test_notes_attachment_sync_api.py \
  tldw_Server_API/tests/Notes/test_notes_api_integration.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_postgres_contract.py
git commit -m "feat(notes): synchronize attachment mutations"
```

### Task 6: Slice verification

- [ ] Run the existing attachment/blob/endpoint/service/materializer suites plus the
  new tests; rerun only sandbox-denied SQLite cases with authorized worktree access.
- [ ] Run touched Ruff and formatter checks, Bandit on production files, `py_compile`,
  and `git diff --check`.
- [ ] Prove rollout-off leaves inactive legacy behavior unchanged and canonical
  mutation unavailable.
- [ ] Update only `TASK-13005.2` AC/DoD/notes and mark it Done.
