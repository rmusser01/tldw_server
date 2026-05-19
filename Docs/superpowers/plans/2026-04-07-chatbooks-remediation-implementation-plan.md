# Chatbooks Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the confirmed Chatbooks runtime, contract, documentation, and test-confidence findings on current `dev`, and add regression coverage that prevents those behaviors from drifting again.

**Architecture:** This plan keeps the remediation narrow and sequential. It fixes runtime correctness first, then tightens caller-visible API behavior, then aligns static contracts/docs to the live implementation, and only then performs the bounded test-structure cleanup required by the review. Each task ends with a focused verification slice and a dedicated commit so the branch can be reviewed or bisected safely.

**Tech Stack:** Python 3, FastAPI, Pydantic, pytest, Bandit, git, rg, JSON Schema, Markdown

---

## Scope Lock

Keep these decisions fixed while implementing:

- work in `/Users/appledev/Documents/GitHub/tldw_server/.worktrees/chatbooks-remediation`, not the dirty main checkout
- target the current `dev` branch behavior, not the older review snapshot
- preserve broad remove-job semantics:
  - export jobs removable in `completed`, `cancelled`, `failed`, `expired`
  - import jobs removable in `completed`, `cancelled`, `failed`
- treat preview failures as strict HTTP errors:
  - `400` for caller-caused archive and manifest problems
  - `500` only for unexpected server faults
- do not preserve `file_path` in the live sync export response contract
- do not preserve public async continuation export support
- populate sync import `imported_items` from the import operation result itself
- do bounded test cleanup only; do not refactor the Chatbooks runtime modules

## File Map

**Primary runtime files:**

- `tldw_Server_API/app/core/Chatbooks/jobs_adapter.py`
- `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
- `tldw_Server_API/app/core/Chatbooks/chatbook_validators.py`
- `tldw_Server_API/app/core/Chatbooks/chatbook_models.py`
- `tldw_Server_API/app/core/Chatbooks/services/jobs_worker.py`

**Primary API and schema files:**

- `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`
- `tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py`

**Static contract and documentation files:**

- `Docs/API-related/chatbook_openapi.yaml`
- `Docs/Schemas/chatbooks_manifest_v1.json`
- `tldw_Server_API/app/core/Chatbooks/README.md`
- `Docs/Code_Documentation/Guides/Chatbooks_Code_Guide.md`
- `Docs/Product/Chatbooks_PRD.md`

**Existing tests to tighten or mine for fixtures:**

- `tldw_Server_API/tests/Chatbooks/test_chatbooks_cancellation.py`
- `tldw_Server_API/tests/Chatbooks/test_chatbooks_api_preview.py`
- `tldw_Server_API/tests/Chatbooks/test_chatbooks_api_error_and_preview_mapping.py`
- `tldw_Server_API/tests/Chatbooks/test_chatbooks_export_sync.py`
- `tldw_Server_API/tests/Chatbooks/test_chatbooks_signed_urls.py`
- `tldw_Server_API/tests/Chatbooks/test_chatbook_integration.py`
- `tldw_Server_API/tests/Chatbooks/test_chatbook_service.py`
- `tldw_Server_API/tests/Chatbooks/test_chatbook_security.py`
- `tldw_Server_API/tests/Chatbooks/test_chatbooks_api_path_guard.py`

**New tests expected from this plan:**

- `tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_adapter.py`
- `tldw_Server_API/tests/Chatbooks/test_chatbooks_import_cleanup.py`
- `tldw_Server_API/tests/Chatbooks/test_chatbooks_import_validation.py`
- `tldw_Server_API/tests/Chatbooks/test_chatbooks_sync_contracts.py`
- `tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_contract.py`
- `tldw_Server_API/tests/Chatbooks/test_chatbook_service_job_lifecycle.py`
- `tldw_Server_API/tests/Chatbooks/test_chatbook_service_preview_import_safety.py`
- `tldw_Server_API/tests/Chatbooks/test_chatbook_service_continuation.py`
- `tldw_Server_API/tests/Chatbooks/test_chatbook_service_cleanup.py`

## Stage Overview

## Stage 1: Runtime Lifecycle and Cleanup Fixes
**Goal:** Fix cancellation readback, tokenized import cleanup, and helper validation drift.
**Success Criteria:** Cancelled jobs stay cancelled on get/list, tokenized staged imports are deletable, and helper validation rejects unsafe archives the same way the main validator does.
**Tests:** Focused unit and service tests only.
**Status:** Not Started

## Stage 2: API Contract Tightening
**Goal:** Enforce strict preview, continuation, sync export, sync import, and remove-job behavior at the route/schema layer.
**Success Criteria:** Unsupported continuation returns `400`, preview stops returning `200 + error`, sync export contracts match across routes, sync import returns real `imported_items`, and remove-route messaging matches actual service semantics.
**Tests:** Focused endpoint tests only.
**Status:** Not Started

## Stage 3: Static Contract and Docs Alignment
**Goal:** Regenerate or replace stale manifest/OpenAPI/docs artifacts so they describe the live Chatbooks behavior.
**Success Criteria:** Real exports validate against the canonical manifest schema and docs no longer advertise stale request/response shapes.
**Tests:** Contract and manifest validation tests.
**Status:** Not Started

## Stage 4: Test Confidence Cleanup and Full Verification
**Goal:** Split the biggest Chatbooks test hotspot, tighten weak assertions, and run touched verification plus Bandit.
**Success Criteria:** The touched Chatbooks tests fail when risky branches are skipped, the hotspot file is decomposed by responsibility, and the touched scope passes pytest and Bandit.
**Tests:** Touched Chatbooks slices, integration spot checks, Bandit.
**Status:** Not Started

### Task 1: Lock the Worktree Baseline and Verify the Starting Point

**Files:**
- Create: none
- Modify: none
- Inspect: `Docs/superpowers/specs/2026-04-07-chatbooks-remediation-design.md`
- Inspect: `Docs/superpowers/plans/2026-04-07-chatbooks-remediation-implementation-plan.md`
- Inspect: `tldw_Server_API/app/core/Chatbooks`
- Inspect: `tldw_Server_API/tests/Chatbooks`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_export_sync.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_api_preview.py`

- [ ] **Step 1: Confirm the isolated worktree is clean before implementation**

Run:
```bash
git status --short
```

Expected: no modified Chatbooks files before new edits begin.

- [ ] **Step 2: Record the implementation baseline hash**

Run:
```bash
git rev-parse --short HEAD
```

Expected: one short commit hash to cite in later review and commit messages.

- [ ] **Step 3: Re-read the approved design before touching code**

Run:
```bash
sed -n '1,220p' Docs/superpowers/specs/2026-04-07-chatbooks-remediation-design.md
```

Expected: the runtime, contract, and documentation requirements are locked before test writing starts.

- [ ] **Step 4: Run the narrow baseline smoke slice from the worktree root**

Run:
```bash
source ../../.venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_export_sync.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_api_preview.py -q
```

Expected: a fast baseline showing current behavior before remediation commits start.

### Task 2: Fix Cancelled-Job Reconciliation and Tokenized Import Cleanup

**Files:**
- Create: `tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_adapter.py`
- Create: `tldw_Server_API/tests/Chatbooks/test_chatbooks_import_cleanup.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/jobs_adapter.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
- Modify: `tldw_Server_API/tests/Chatbooks/test_chatbooks_cancellation.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_adapter.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_import_cleanup.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_cancellation.py`

- [ ] **Step 1: Write the failing adapter tests for cancelled terminal-state protection**

Add tests shaped like:
```python
def test_apply_export_status_keeps_cancelled_when_jobs_row_lags():
    adapter = ChatbooksJobsAdapter()
    job = SimpleNamespace(status=ExportStatus.CANCELLED)
    adapter.apply_export_status(job, {"status": "running"})
    assert job.status is ExportStatus.CANCELLED


def test_apply_import_status_keeps_cancelled_when_jobs_row_lags():
    adapter = ChatbooksJobsAdapter()
    job = SimpleNamespace(status=ImportStatus.CANCELLED)
    adapter.apply_import_status(job, {"status": "queued"})
    assert job.status is ImportStatus.CANCELLED
```

Run:
```bash
source ../../.venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_adapter.py -q
```

Expected: failure showing the adapter still remaps cancelled jobs.

- [ ] **Step 2: Write the failing cleanup tests for tokenized import paths**

Add tests shaped like:
```python
def test_try_delete_import_file_resolves_temp_token(service, tmp_path):
    staged = Path(service.temp_dir) / "imports" / "sample.chatbook"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(b"zip")

    token = service._build_import_file_token(staged)
    assert service._try_delete_import_file(token) == 1
    assert not staged.exists()
```

Also cover an `import/...` token and a path outside the allowed bases.

Run:
```bash
source ../../.venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_import_cleanup.py -q
```

Expected: failure showing `_try_delete_import_file(...)` does not resolve tokens yet.

- [ ] **Step 3: Tighten the existing cancellation API assertions before the implementation**

Replace permissive assertions like:
```python
assert status in ("cancelled", "completed", "failed", "in_progress", "pending")
```

with a deterministic shape:
```python
assert status == "cancelled"
```

and stub the backing service/job state so the test no longer depends on race timing.

Run:
```bash
source ../../.venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_cancellation.py -q
```

Expected: red test if readback still drifts away from `cancelled`.

- [ ] **Step 4: Implement the adapter and cleanup fixes**

Update the runtime roughly like:
```python
def apply_export_status(self, job, job_row: dict[str, Any] | None = None) -> None:
    if getattr(job, "status", None) == ExportStatus.CANCELLED:
        return
    mapped = _map_export_status((job_row or self._get_job(job.job_id, "chatbook_export") or {}).get("status"))
    if mapped is not None:
        job.status = mapped
```

and:
```python
def _try_delete_import_file(self, file_path_str: str) -> int:
    try:
        file_path = self._resolve_import_archive_path(file_path_str)
    except ValidationError:
        logger.warning("Refusing to delete unresolved import file token")
        return 0
    ...
```

Keep the existing containment checks after resolution.

- [ ] **Step 5: Re-run the lifecycle and cleanup slice**

Run:
```bash
source ../../.venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_adapter.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_import_cleanup.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_cancellation.py -q
```

Expected: green tests proving cancelled remains terminal and tokenized cleanup works.

- [ ] **Step 6: Commit the runtime lifecycle fix**

Run:
```bash
git add \
  tldw_Server_API/app/core/Chatbooks/jobs_adapter.py \
  tldw_Server_API/app/core/Chatbooks/chatbook_service.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_adapter.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_import_cleanup.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_cancellation.py
git commit -m "fix: harden chatbooks job reconciliation and cleanup"
```

### Task 3: Harden Helper Validation and Preview Failure Semantics

**Files:**
- Create: `tldw_Server_API/tests/Chatbooks/test_chatbooks_import_validation.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_validators.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`
- Modify: `tldw_Server_API/tests/Chatbooks/test_chatbook_security.py`
- Modify: `tldw_Server_API/tests/Chatbooks/test_chatbooks_api_preview.py`
- Modify: `tldw_Server_API/tests/Chatbooks/test_chatbooks_api_error_and_preview_mapping.py`
- Modify: `tldw_Server_API/tests/Chatbooks/test_chatbooks_api_path_guard.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_import_validation.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_api_preview.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_api_error_and_preview_mapping.py`

- [ ] **Step 1: Write the failing helper-validation tests**

Add tests shaped like:
```python
def test_validate_chatbook_file_rejects_symlink_archive(service, tmp_path):
    chatbook_path = build_symlink_zip(tmp_path)
    result = service.validate_chatbook_file(str(chatbook_path))
    assert result["valid"] is False
    assert "symlink" in result["error"].lower()
```

Add one traversal member case and one dangerous-file-type case.

Run:
```bash
source ../../.venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_import_validation.py -q
```

Expected: failure showing helper validation still bypasses the hardened ZIP validator.

- [ ] **Step 2: Write the failing preview contract tests**

Add endpoint tests shaped like:
```python
def test_preview_invalid_manifest_returns_400(client):
    files = {"file": ("broken.chatbook", build_invalid_manifest_zip(), "application/zip")}
    response = client.post("/api/v1/chatbooks/preview", files=files)
    assert response.status_code == 400
    assert "detail" in response.json()


def test_preview_unexpected_service_error_returns_500(client, monkeypatch):
    monkeypatch.setattr(ChatbookService, "preview_chatbook", lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))
    response = client.post("/api/v1/chatbooks/preview", files={"file": ("ok.chatbook", build_valid_zip(), "application/zip")})
    assert response.status_code == 500
```

Run:
```bash
source ../../.venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_api_preview.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_api_error_and_preview_mapping.py -q
```

Expected: failure because caller-caused preview errors still return `200`.

- [ ] **Step 3: Implement the hardened helper and preview mapping changes**

Route helper validation through the ZIP validator first:
```python
def validate_chatbook(self, file_path: str) -> bool:
    resolved_path = self._resolve_import_archive_path(file_path)
    ok, err = ChatbookValidator.validate_zip_file(str(resolved_path))
    if not ok:
        raise ValidationError(err or "Invalid chatbook archive", field="file_path")
    ...
```

Map preview failures explicitly:
```python
manifest, error = service.preview_chatbook(str(temp_file))
if manifest is None:
    raise HTTPException(status_code=400, detail=error or "Invalid chatbook")
```

Keep the broad `except _CHATBOOKS_NONCRITICAL_EXCEPTIONS` block for true server faults only.

- [ ] **Step 4: Add live `/import` and `/preview` malicious-archive coverage**

Add API tests shaped like:
```python
def test_import_rejects_symlink_archive(client):
    response = client.post("/api/v1/chatbooks/import", files=build_upload(build_symlink_zip_bytes()))
    assert response.status_code == 400
```

Reuse shared fixture helpers where possible instead of duplicating ZIP builders.

- [ ] **Step 5: Re-run the preview and validation slice**

Run:
```bash
source ../../.venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_import_validation.py \
  tldw_Server_API/tests/Chatbooks/test_chatbook_security.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_api_preview.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_api_error_and_preview_mapping.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_api_path_guard.py -q
```

Expected: green tests proving helper validation and preview HTTP semantics are strict.

- [ ] **Step 6: Commit the validation and preview hardening**

Run:
```bash
git add \
  tldw_Server_API/app/core/Chatbooks/chatbook_service.py \
  tldw_Server_API/app/core/Chatbooks/chatbook_validators.py \
  tldw_Server_API/app/api/v1/endpoints/chatbooks.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_import_validation.py \
  tldw_Server_API/tests/Chatbooks/test_chatbook_security.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_api_preview.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_api_error_and_preview_mapping.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_api_path_guard.py
git commit -m "fix: harden chatbooks archive validation and preview errors"
```

### Task 4: Tighten Continuation, Sync Export, Sync Import, and Remove-Job Contracts

**Files:**
- Create: `tldw_Server_API/tests/Chatbooks/test_chatbooks_sync_contracts.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
- Modify: `tldw_Server_API/tests/Chatbooks/test_chatbooks_export_sync.py`
- Modify: `tldw_Server_API/tests/Chatbooks/test_chatbooks_signed_urls.py`
- Modify: `tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_worker_import_defaults.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_sync_contracts.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_export_sync.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_signed_urls.py`

- [ ] **Step 1: Write the failing continuation and sync-response contract tests**

Add tests shaped like:
```python
def test_continue_export_rejects_async_mode_true(client):
    response = client.post(
        "/api/v1/chatbooks/export/continue",
        json={"export_id": "exp-1", "continuations": [{"type": "evaluation"}], "async_mode": True},
    )
    assert response.status_code == 400


def test_continue_export_sync_matches_export_response_shape(client, monkeypatch):
    monkeypatch.setattr(ChatbookService, "continue_chatbook_export", AsyncMock(return_value=(True, "ok", "/tmp/fake.chatbook")))
    response = client.post("/api/v1/chatbooks/export/continue", json={"export_id": "exp-1", "continuations": [{"type": "evaluation"}]})
    body = response.json()
    assert body["success"] is True
    assert "job_id" in body
    assert "download_url" in body
    assert "file_path" not in body
```

Run:
```bash
source ../../.venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_sync_contracts.py -q
```

Expected: failure because continuation still accepts `async_mode` and returns `file_path`.

- [ ] **Step 2: Write the failing sync import and remove-route tests**

Add tests shaped like:
```python
def test_import_sync_returns_imported_items(client):
    response = client.post("/api/v1/chatbooks/import", files=build_sync_import_upload())
    body = response.json()
    assert body["imported_items"] == {"conversation": 1, "note": 1}
    assert body["warnings"] == []


def test_import_sync_omits_skipped_and_unrequested_types(client):
    response = client.post("/api/v1/chatbooks/import", files=build_partial_sync_import_upload())
    body = response.json()
    assert body["imported_items"]["conversation"] == 1
    assert "media" not in body["imported_items"]
    assert "embedding" not in body["imported_items"]


def test_remove_export_job_allows_failed_and_expired(client, monkeypatch):
    monkeypatch.setattr(ChatbookService, "delete_export_job", lambda *_args, **_kwargs: True)
    response = client.delete("/api/v1/chatbooks/export/jobs/job-1/remove")
    assert response.status_code == 200
```

Also add:
- a rename-path case where renamed imports still increment `imported_items`
- a skip-path case where skipped conflicts do not increment `imported_items`
- a negative remove-route test that still rejects non-terminal states with the broadened message

Run:
```bash
source ../../.venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_sync_contracts.py -q
```

Expected: failure because `imported_items` is empty and remove-route messaging is stale.

- [ ] **Step 3: Implement the sync and route contract changes**

Make the continuation endpoint reject async mode up front:
```python
if request_data.async_mode:
    raise HTTPException(status_code=400, detail="Async continuation export is not supported")
```

Return the persisted-job sync shape from both sync export routes:
```python
return CreateChatbookResponse(
    success=True,
    message=message,
    job_id=job_id,
    download_url=download_url,
)
```

Remove `file_path` from `CreateChatbookResponse`, and make sync import populate:
```python
return ImportChatbookResponse(
    success=True,
    message=message,
    imported_items=result["imported_items"],
    warnings=result["warnings"],
)
```

If the service currently returns only warnings for sync import, refactor it to return a deterministic dict such as:
```python
{
    "imported_items": {"conversation": 1, "note": 1},
    "warnings": [],
}
```

- [ ] **Step 4: Tighten the signed-download and sync export tests so they cannot early-return**

Replace weak patterns like:
```python
if not download_url:
    return
```

with hard assertions:
```python
assert download_url, "sync export should always emit a download URL"
```

and verify the continuation sync route produces the same `job_id` plus job-backed download semantics as the primary sync export route.

- [ ] **Step 5: Re-run the sync contract slice**

Run:
```bash
source ../../.venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_sync_contracts.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_export_sync.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_signed_urls.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_worker_import_defaults.py -q
```

Expected: green tests proving the stricter public contract.

- [ ] **Step 6: Commit the contract-tightening changes**

Run:
```bash
git add \
  tldw_Server_API/app/api/v1/endpoints/chatbooks.py \
  tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py \
  tldw_Server_API/app/core/Chatbooks/chatbook_service.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_sync_contracts.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_export_sync.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_signed_urls.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_worker_import_defaults.py
git commit -m "fix: align chatbooks sync contracts with live behavior"
```

### Task 5: Align the Manifest Schema, OpenAPI, and Chatbooks Docs

**Files:**
- Create: `tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_contract.py`
- Modify: `Docs/Schemas/chatbooks_manifest_v1.json`
- Modify: `Docs/API-related/chatbook_openapi.yaml`
- Modify: `tldw_Server_API/app/core/Chatbooks/README.md`
- Modify: `Docs/Code_Documentation/Guides/Chatbooks_Code_Guide.md`
- Modify: `Docs/Product/Chatbooks_PRD.md`
- Modify: `tldw_Server_API/tests/Chatbooks/test_chatbook_integration.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_contract.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbook_integration.py`

- [x] **Step 1: Write the failing manifest schema contract test**

Add a test shaped like:
```python
def test_real_export_manifest_matches_canonical_schema(service, tmp_path):
    success, _, archive_path = asyncio.run(
        service.export_chatbook(name="Schema Contract", description="contract", content_types=["conversations"], async_mode=False)
    )
    assert success is True

    manifest = read_manifest_from_zip(archive_path)
    schema = json.loads(Path("Docs/Schemas/chatbooks_manifest_v1.json").read_text())
    jsonschema.validate(manifest, schema)
```

Run:
```bash
source ../../.venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_contract.py -q
```

Expected: failure while the stale schema still expects the wrong root shape or version.

- [x] **Step 2: Replace the stale manifest schema and OpenAPI shapes**

Update the manifest schema to match the real `ChatbookManifest` payload, including canonical `1.0.0`.

Update the OpenAPI file so it matches:
```yaml
CreateChatbookResponse:
  type: object
  required: [success, message]
  properties:
    success: {type: boolean}
    message: {type: string}
    job_id: {type: string, nullable: true}
    download_url: {type: string, nullable: true}
```

and ensure `/preview`, `/import`, and `/export/continue` document the strict status codes and request shapes actually implemented.

Add or update a route-level contract assertion proving `/import` still uses multipart form fields directly rather than the older `options` JSON wrapper shape.

- [x] **Step 3: Align README, code guide, and PRD statements**

Update the prose so it no longer claims:
```md
- preview returns `200` with an `error` body for invalid chatbooks
- sync export returns a local `file_path`
- continuation export supports async mode
- import uses a JSON `options` wrapper
```

Leave future or planned gaps labeled as planned rather than as present-tense behavior.

- [x] **Step 4: Re-run the manifest and contract slice**

Run:
```bash
source ../../.venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_contract.py \
  tldw_Server_API/tests/Chatbooks/test_chatbook_integration.py -q
```

Expected: real exports validate against the new canonical manifest schema and touched integration assertions remain green.

- [x] **Step 5: Commit the static contract and documentation alignment**

Run:
```bash
git add \
  Docs/Schemas/chatbooks_manifest_v1.json \
  Docs/API-related/chatbook_openapi.yaml \
  tldw_Server_API/app/core/Chatbooks/README.md \
  Docs/Code_Documentation/Guides/Chatbooks_Code_Guide.md \
  Docs/Product/Chatbooks_PRD.md \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_contract.py \
  tldw_Server_API/tests/Chatbooks/test_chatbook_integration.py
git commit -m "docs: align chatbooks contracts and manifest schema"
```

### Task 6: Split the Chatbook Service Test Hotspot and Tighten Weak Assertions

**Files:**
- Create: `tldw_Server_API/tests/Chatbooks/test_chatbook_service_job_lifecycle.py`
- Create: `tldw_Server_API/tests/Chatbooks/test_chatbook_service_preview_import_safety.py`
- Create: `tldw_Server_API/tests/Chatbooks/test_chatbook_service_continuation.py`
- Create: `tldw_Server_API/tests/Chatbooks/test_chatbook_service_cleanup.py`
- Modify: `tldw_Server_API/tests/Chatbooks/test_chatbook_service.py`
- Modify: `tldw_Server_API/tests/Chatbooks/test_chatbooks_signed_urls.py`
- Modify: `tldw_Server_API/tests/Chatbooks/test_chatbook_integration.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbook_service.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbook_service_job_lifecycle.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbook_service_preview_import_safety.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbook_service_continuation.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbook_service_cleanup.py`

- [x] **Step 1: Move lifecycle, preview/import-safety, continuation, and cleanup tests into dedicated files**

Use this split as the minimum target:
```python
# test_chatbook_service_job_lifecycle.py
def test_get_export_job_parses_varied_timestamps(...): ...
def test_cancel_export_job(...): ...

# test_chatbook_service_preview_import_safety.py
def test_validate_chatbook_file(...): ...
def test_preview_chatbook_cleans_temp_dir_on_failure(...): ...

# test_chatbook_service_continuation.py
async def test_continue_export_produces_linked_chatbook(...): ...

# test_chatbook_service_cleanup.py
def test_clean_old_exports(...): ...
```

Keep shared helpers in `test_chatbook_service.py` only if they are still reused.

- [x] **Step 2: Tighten the weak integration assertions that masked reviewed defects**

Replace patterns like:
```python
assert response.status_code in (200, 400)
assert body.get("error") is None or "detail" in body
```

with branch-specific checks that assert the intended side effects, for example:
```python
assert response.status_code == 200
assert body["download_url"].endswith(body["job_id"])
assert body["success"] is True
```

Remove any swallowed setup failures that allow the test to pass without exercising the risky path.

- [x] **Step 3: Run the service-structure and touched integration slice**

Run:
```bash
source ../../.venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Chatbooks/test_chatbook_service.py \
  tldw_Server_API/tests/Chatbooks/test_chatbook_service_job_lifecycle.py \
  tldw_Server_API/tests/Chatbooks/test_chatbook_service_preview_import_safety.py \
  tldw_Server_API/tests/Chatbooks/test_chatbook_service_continuation.py \
  tldw_Server_API/tests/Chatbooks/test_chatbook_service_cleanup.py \
  tldw_Server_API/tests/Chatbooks/test_chatbook_integration.py -q
```

Expected: the split suite stays green and the tightened tests would fail if the risky branch were skipped.

- [x] **Step 4: Run the touched Chatbooks verification bundle**

Run:
```bash
source ../../.venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_adapter.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_import_cleanup.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_import_validation.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_sync_contracts.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_contract.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_api_preview.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_api_error_and_preview_mapping.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_export_sync.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_signed_urls.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_cancellation.py \
  tldw_Server_API/tests/Chatbooks/test_chatbook_service.py \
  tldw_Server_API/tests/Chatbooks/test_chatbook_service_job_lifecycle.py \
  tldw_Server_API/tests/Chatbooks/test_chatbook_service_preview_import_safety.py \
  tldw_Server_API/tests/Chatbooks/test_chatbook_service_continuation.py \
  tldw_Server_API/tests/Chatbooks/test_chatbook_service_cleanup.py -q
```

Expected: the touched Chatbooks regression suite passes before final completion.

- [x] **Step 5: Run Bandit on the touched Chatbooks scope**

Run:
```bash
source ../../.venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/Chatbooks \
  tldw_Server_API/app/api/v1/endpoints/chatbooks.py \
  tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py \
  tldw_Server_API/tests/Chatbooks \
  -f json -o /tmp/bandit_chatbooks_remediation.json
```

Expected: a Bandit JSON report for the touched scope with any new findings addressed before final handoff.

- [x] **Step 6: Commit the test cleanup and verification pass**

Run:
```bash
git add \
  tldw_Server_API/tests/Chatbooks/test_chatbook_service.py \
  tldw_Server_API/tests/Chatbooks/test_chatbook_service_job_lifecycle.py \
  tldw_Server_API/tests/Chatbooks/test_chatbook_service_preview_import_safety.py \
  tldw_Server_API/tests/Chatbooks/test_chatbook_service_continuation.py \
  tldw_Server_API/tests/Chatbooks/test_chatbook_service_cleanup.py \
  tldw_Server_API/tests/Chatbooks/test_chatbook_integration.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_signed_urls.py
git commit -m "test: harden chatbooks regression coverage"
```

## Final Verification Checklist

- [ ] `git status --short` shows only intentional Chatbooks remediation files before the final handoff
- [ ] every finding from `Docs/superpowers/specs/2026-04-07-chatbooks-remediation-design.md` is mapped to a completed task above
- [ ] preview caller faults return `400`, not `200 + error`
- [ ] cancelled jobs stay cancelled on readback
- [ ] sync export and sync continuation both return `job_id` and `download_url`
- [ ] sync import returns deterministic `imported_items` plus `warnings`
- [ ] remove-job docs and error text match the broader allowed terminal states
- [ ] real export manifests validate against `Docs/Schemas/chatbooks_manifest_v1.json`
- [ ] touched Chatbooks pytest slices and Bandit have been run and recorded
