# Media Authorization And Tenant Storage Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close `AUDIT-2026-06-27-MEDIA-001`, `AUDIT-2026-06-27-MEDIA-002`, and `AUDIT-2026-06-27-MEDIA-003` on current `origin/dev`.

**Architecture:** Reuse the existing `MEDIA_CREATE` and `media.create` authorization contract for no-DB processing routes rather than adding a new RBAC permission in this slice. Thread request-scoped Media DB and user identity into MediaWiki ingest so DB writes, vector writes, and checkpoints are per user. Add compensating cleanup when original-file storage succeeds but `MediaFiles` registration fails.

**Tech Stack:** FastAPI dependencies, AuthNZ `RequirePermission`, media DB request sessions, MediaWiki ingestion helpers, pytest.

---

## Source Context

- Backlog task: `TASK-12091`
- Baseline: `origin/dev` at `30495536d3`
- Branch: `codex/audit-media-storage-2026-07-02`
- Audit IDs: `AUDIT-2026-06-27-MEDIA-001`, `AUDIT-2026-06-27-MEDIA-002`, `AUDIT-2026-06-27-MEDIA-003`
- Permission decision: use existing `MEDIA_CREATE` and `rbac_rate_limit("media.create")`.
- Checkpoint decision: scope MediaWiki checkpoints by user for persisted imports; keep ephemeral process-only checkpoints unused.

## File Map

- Modify: `tldw_Server_API/app/api/v1/endpoints/media/process_audios.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/process_documents.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/process_pdfs.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/process_ebooks.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/process_code.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/process_emails.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/process_mediawiki.py`
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/MediaWiki/Media_Wiki.py`
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py`
- Create: `tldw_Server_API/tests/AuthNZ_Unit/test_media_processing_permissions_claims.py`
- Modify: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_mediawiki_db_persistence.py`
- Modify: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_mediawiki_vector_storage.py`
- Modify: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_persistence_original_storage.py`

### Task 1: Add Permission Gates To No-DB Media Processing Routes

**Files:**
- Modify: media processing endpoint files listed in File Map
- Create: `tldw_Server_API/tests/AuthNZ_Unit/test_media_processing_permissions_claims.py`

- [ ] **Step 1: Write the failing route dependency test**

Create a parameterized test that imports each processing router module and asserts the route dependencies include both `RequirePermission(MEDIA_CREATE)` and `rbac_rate_limit("media.create")`. Include these paths:

```python
ROUTE_CASES = [
    ("process_audios", "/process-audios"),
    ("process_documents", "/process-documents"),
    ("process_pdfs", "/process-pdfs"),
    ("process_ebooks", "/process-ebooks"),
    ("process_code", "/process-code"),
    ("process_emails", "/process-emails"),
    ("process_mediawiki", "/mediawiki/ingest-dump"),
    ("process_mediawiki", "/mediawiki/process-dump"),
]
```

Assert against the real `APIRoute.dependant.dependencies` so the test fails before implementation on the ungated routes.

- [ ] **Step 2: Verify the test fails**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tldw_Server_API/tests/AuthNZ_Unit/test_media_processing_permissions_claims.py -q
```

Expected before implementation: the new test fails for the listed ungated routes.

- [ ] **Step 3: Add dependencies and imports**

For each endpoint module, import:

```python
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import RequirePermission, rbac_rate_limit
from tldw_Server_API.app.core.AuthNZ.permissions import MEDIA_CREATE
```

Add these dependencies before quota/billing dependencies:

```python
Depends(RequirePermission(MEDIA_CREATE)),
Depends(rbac_rate_limit("media.create")),
```

For `process_emails.py`, also add `require_within_limit(LimitCategory.API_CALLS_DAY, 1)` to match the other processing-only routes unless an existing focused test shows email deliberately lacks API-call billing.

- [ ] **Step 4: Verify the route dependency test passes**

Run the same pytest command from Step 2. Expected: pass.

### Task 2: Thread Request-Scoped MediaWiki DB And Vector User

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/process_mediawiki.py`
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/MediaWiki/Media_Wiki.py`
- Modify: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_mediawiki_db_persistence.py`
- Modify: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_mediawiki_vector_storage.py`

- [ ] **Step 1: Write failing MediaWiki DB injection tests**

Update `test_import_mediawiki_dump_reuses_single_managed_media_database` into two tests:

```python
def test_import_mediawiki_dump_uses_injected_media_writer_without_managed_database(...):
    ...
    results = list(Media_Wiki.import_mediawiki_dump(..., media_writer=fake_repo, vector_user_id=42))
    assert managed_calls == []
    assert len(fake_repo.calls) == 2
```

Keep a separate legacy fallback test proving `managed_media_database(client_id="mediawiki_import")` is still used only when `store_to_db=True` and no writer is injected.

- [ ] **Step 2: Write failing vector user test**

Extend `test_process_single_item_stores_vectors` so `process_single_item(..., vector_user_id=42)` constructs `ChromaDBManager(user_id="42", ...)` instead of the configured single-user fixed ID.

- [ ] **Step 3: Write failing checkpoint scope test**

Add a unit test that calls `import_mediawiki_dump(..., media_writer=fake_repo, checkpoint_scope="user_42")` and monkeypatches `get_safe_checkpoint_path` to capture the name. Expected checkpoint key should include both the safe wiki name and `user_42`.

- [ ] **Step 4: Verify the new MediaWiki tests fail**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_mediawiki_db_persistence.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_mediawiki_vector_storage.py -q
```

Expected before implementation: tests fail because the importer always opens a managed DB and vector storage uses `SINGLE_USER_FIXED_ID`.

- [ ] **Step 5: Implement MediaWiki parameters**

Add optional parameters:

```python
vector_user_id: str | int | None = None
media_writer: Any | None = None
checkpoint_scope: str | None = None
```

Thread `vector_user_id` from `import_mediawiki_dump` to `process_single_item` to `_store_mediawiki_chunks_in_vector_db`. Use `settings.get("SINGLE_USER_FIXED_ID", 1)` only when `vector_user_id is None`.

In `import_mediawiki_dump`, use an injected `media_writer` when provided; open `managed_media_database` only for the fallback path. Build checkpoint names with a sanitized combined key when `checkpoint_scope` is provided.

- [ ] **Step 6: Wire the endpoint**

In `process_mediawiki.py`, add request-scoped dependencies to `_process_mediawiki_dump` and the two public endpoints:

```python
db: Any = Depends(get_media_db_for_user)
current_user: User = Depends(get_request_user)
```

For persisted ingest, pass `media_writer=get_media_repository(db)`, `vector_user_id=current_user.id_int or current_user.id`, and a sanitized checkpoint scope derived from the same user ID. For process-only endpoint, keep `store_to_db=False` and `store_to_vector_db=False`.

- [ ] **Step 7: Verify MediaWiki tests pass**

Run the command from Step 4. Expected: pass.

### Task 3: Delete Stored Originals When MediaFiles Registration Fails

**Files:**
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py`
- Modify: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_persistence_original_storage.py`

- [ ] **Step 1: Write failing cleanup test**

Add a test near `test_original_storage_uses_processing_source` using a fake storage backend that records `store()` and `delete()` calls. Configure `fake_db.insert_media_file` to raise after storage succeeds.

Expected assertions:

```python
assert storage.stored_paths == ["..."]
assert storage.deleted_paths == storage.stored_paths
assert response.status_code == 200
assert result["original_file_stored"] is False
```

- [ ] **Step 2: Verify cleanup test fails**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tldw_Server_API/tests/MediaIngestion_NEW/unit/test_persistence_original_storage.py -q
```

Expected before implementation: stored path is not deleted.

- [ ] **Step 3: Implement compensating cleanup**

In `persistence.py`, initialize `storage_path: str | None = None` before `storage.store(...)`. In the exception handler for original-file persistence, if `storage_path` is set, call `await storage.delete(user_id=user_id_str, storage_path=storage_path)` or the storage backend's actual delete signature. Log cleanup failure at warning/debug level without failing the ingestion.

- [ ] **Step 4: Verify cleanup test passes**

Run the command from Step 2. Expected: pass.

### Task 4: Final Focused Verification

**Files:**
- All files above
- Backlog task `TASK-12091`

- [ ] **Step 1: Run focused tests**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tldw_Server_API/tests/AuthNZ_Unit/test_media_processing_permissions_claims.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_mediawiki_db_persistence.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_mediawiki_vector_storage.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_persistence_original_storage.py -q
```

- [ ] **Step 2: Run Bandit on touched production paths**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m bandit \
  tldw_Server_API/app/api/v1/endpoints/media/process_audios.py \
  tldw_Server_API/app/api/v1/endpoints/media/process_documents.py \
  tldw_Server_API/app/api/v1/endpoints/media/process_pdfs.py \
  tldw_Server_API/app/api/v1/endpoints/media/process_ebooks.py \
  tldw_Server_API/app/api/v1/endpoints/media/process_code.py \
  tldw_Server_API/app/api/v1/endpoints/media/process_emails.py \
  tldw_Server_API/app/api/v1/endpoints/media/process_mediawiki.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/MediaWiki/Media_Wiki.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py \
  -f json -o /tmp/bandit_media_storage_12091.json
```

- [ ] **Step 3: Run whitespace check**

```bash
git diff --check
```

- [ ] **Step 4: Update `TASK-12091`**

Record verification results, closed findings, residual risk, and touched files in the Backlog task.
