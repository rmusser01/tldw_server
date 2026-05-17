# llama.cpp Model Acquisition And Import Workflows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe llama.cpp acquisition workflow that improves local import/register UX first, then adds cancellable Jobs-backed remote downloads that produce normal `LlamaCppAsset` inventory entries.

**Architecture:** Keep `llamacpp_inventory_service` as the local asset source of truth and keep managed profiles/runtime supervision separate from acquisition. Local folder import remains synchronous and admin-only; remote downloads become Jobs-domain work with explicit status/cancel endpoints, bounded progress, partial-file cleanup, checksum support, and atomic registration only after validation. The WebUI exposes acquisition state inside the existing Admin assets panel without creating or starting profiles automatically.

**Tech Stack:** FastAPI, Pydantic v2, existing llama.cpp Admin schemas/endpoints, `llamacpp_inventory_service`, core Jobs `JobManager`/`WorkerSDK`, pytest/TestClient, React/Ant Design shared UI, Vitest/testing-library, Bandit.

**Command note:** Run Python verification commands after activating the project virtual environment from the repository root, for example `source .venv/bin/activate`.

---

## References

- Roadmap spec: `Docs/superpowers/specs/2026-05-16-llamacpp-managed-runtime-roadmap-design.md`
- Current asset endpoints: `tldw_Server_API/app/api/v1/endpoints/llamacpp.py`
- Current Admin schemas: `tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py`
- Current inventory service: `tldw_Server_API/app/core/Local_LLM/llamacpp_inventory_service.py`
- Current managed profile models: `tldw_Server_API/app/core/Local_LLM/llamacpp_runtime_models.py`
- Jobs dependency helper: `tldw_Server_API/app/api/v1/API_Deps/jobs_deps.py`
- WorkerSDK pattern: `tldw_Server_API/app/core/File_Artifacts/jobs_worker.py`
- Worker startup registration: `tldw_Server_API/app/services/startup_content_jobs_pollers.py`
- Existing Admin assets panel: `apps/packages/ui/src/components/Option/Admin/LlamacppAssetsPanel.tsx`
- Existing Admin page: `apps/packages/ui/src/components/Option/Admin/LlamacppAdminPage.tsx`
- Tracking task: `TASK-416`

## Scope Guardrails

- Do not create, start, stop, or wire managed profiles from acquisition actions.
- Do not treat a downloaded file as trusted executable configuration.
- Do not download into arbitrary destinations; destination roots must be under configured llama.cpp model/import allowlists.
- Do not expose raw secrets from URLs, headers, or errors in logs, job results, or UI messages.
- Do not add a curated model marketplace in this slice.
- Do not make Hugging Face account/token integration required for direct URL downloads.
- Do not weaken existing `registered_model_paths` and `imported_asset_folders` validation.
- Do not hard-block advisory disk/model metadata warnings when the actual path/download request is safe.

## Current Baseline

Already available on `origin/dev`:

- `GET /api/v1/llamacpp/assets`
- `POST /api/v1/llamacpp/assets/register-path`
- `POST /api/v1/llamacpp/assets/import-folder`
- Asset inventory covers `gguf`, `mmproj`, `folder`, and `unknown`.
- Imported local folders are persisted in `[LlamaCpp].imported_asset_folders`.
- `LlamacppAssetsPanel` has basic "Register asset" and "Import folder" inputs.
- Managed profiles support `model_id`, `model_path`, `mmproj_model_id`, modes, metadata, and saved profile editing.

Missing:

- no non-mutating local import/register preview;
- no richer import result that reports discovered files, pair candidates, or stale warnings;
- no acquisition job API/status model;
- no download worker, destination policy, partial-file cleanup, or checksum contract;
- no WebUI status/cancel surface for downloads;
- no docs explaining local import versus remote download semantics.

## File Structure

### Backend

- Modify: `tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py`
  - Add preview, acquisition request, acquisition job response, and download source/destination models.
- Modify: `tldw_Server_API/app/core/Local_LLM/llamacpp_inventory_service.py`
  - Add local import/register preview helpers.
  - Add summary helpers when tests need shared count/warning/result shaping for import/register/download completion.
  - Keep actual path persistence inside existing config write lock.
- Create: `tldw_Server_API/app/core/Local_LLM/llamacpp_acquisition_service.py`
  - Validate download URLs and destinations.
  - Build redacted source metadata.
  - Resolve temporary and final destination paths.
  - Validate checksum and final asset registration.
- Create: `tldw_Server_API/app/core/Local_LLM/llamacpp_acquisition_jobs.py`
  - Thin job creation/status/cancel/result helpers around `JobManager`.
  - Domain: `llamacpp`.
  - Queue: `acquisition`.
  - Job type: `llamacpp_asset_download`.
- Create: `tldw_Server_API/app/services/llamacpp_acquisition_jobs_worker.py`
  - WorkerSDK loop for download jobs.
  - Progress reporting through `JobManager.update_job_progress()` or WorkerSDK's `progress_cb` renewal hook.
  - Stream download to a `.partial` file, update progress, support cancellation, atomically rename, register final asset, clean partial files.
- Modify: `tldw_Server_API/app/services/startup_content_jobs_pollers.py`
  - Start the worker behind `LLAMACPP_ACQUISITION_JOBS_WORKER_ENABLED`.
- Modify: `tldw_Server_API/app/api/v1/endpoints/llamacpp.py`
  - Add preview/status/cancel/download endpoints.
  - Keep all endpoints admin-only.
- Modify: `tldw_Server_API/app/api/v1/API_Deps/jobs_deps.py`
  - Reuse `get_job_manager`; no llama.cpp-specific manager singleton unless tests prove it is needed.
- Modify: `Docs/API-related/llamacpp_integration_modes.md`
- Modify: `Docs/Published/API-related/llamacpp_integration_modes.md`

### Backend Tests

- Modify: `tldw_Server_API/tests/LLM_Local/test_llamacpp_asset_inventory_service.py`
- Modify: `tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py`
- Create: `tldw_Server_API/tests/LLM_Local/test_llamacpp_acquisition_service.py`
- Create: `tldw_Server_API/tests/LLM_Local/test_llamacpp_acquisition_api.py`
- Create: `tldw_Server_API/tests/LLM_Local/test_llamacpp_acquisition_jobs_worker.py`
- Modify: `tldw_Server_API/tests/AuthNZ_Unit/test_llamacpp_permissions_claims.py`

### Frontend

- Modify: `apps/packages/ui/src/types/llamacpp-admin.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/models-audio.ts`
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Modify: `apps/packages/ui/src/components/Option/Admin/LlamacppAssetsPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/Admin/LlamacppAdminPage.tsx`

### Frontend Tests

- Modify: `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAssetsPanel.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx`
- Modify: `apps/packages/ui/src/services/__tests__/tldw-api-client.models-normalization.test.ts`

---

## Task 1: Local Import Preview And Result Contract

**Files:**

- Modify: `tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py`
- Modify: `tldw_Server_API/app/core/Local_LLM/llamacpp_inventory_service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/llamacpp.py`
- Test: `tldw_Server_API/tests/LLM_Local/test_llamacpp_asset_inventory_service.py`
- Test: `tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py`

- [ ] **Step 1: Write failing service tests for folder import preview**

Add tests for:

```python
def test_preview_import_asset_folder_summarizes_assets_without_persisting(monkeypatch, tmp_path):
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / "chat.Q4_K_M.gguf").write_text("model")
    (folder / "mmproj-chat.gguf").write_text("projector")
    configure_llamacpp_paths(monkeypatch, models_dir=folder, allowed_paths=[folder])

    preview = llamacpp_inventory_service.preview_import_asset_folder(folder)

    assert preview.folder.display_name == "models"
    assert preview.asset_counts["gguf"] == 1
    assert preview.asset_counts["mmproj"] == 1
    assert preview.scan_limited is False
    assert "imported_asset_folders" not in read_persisted_llamacpp_config()
```

Also cover:

- non-existent folder returns `ServerError`;
- file path returns `ServerError`;
- folder outside allowed paths fails closed;
- scan-limited previews set `scan_limited=true`;
- preview emits warnings for unreadable or unknown files without failing the whole folder.

- [ ] **Step 2: Run preview service tests to verify failure**

Run:

```bash
python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_asset_inventory_service.py \
  -k "import_asset_folder_preview" -v
```

Expected: FAIL because preview helpers do not exist.

- [ ] **Step 3: Add preview schemas**

In `llamacpp_admin_schemas.py`, add:

```python
class LlamaCppAssetImportPreviewResponse(BaseModel):
    folder: LlamaCppAsset
    assets: list[LlamaCppAsset] = Field(default_factory=list)
    asset_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    scan_limited: bool = False
    will_persist: bool = False
```

Keep `assets` bounded by the same inventory limit so the preview is safe to render.

- [ ] **Step 4: Implement preview helpers**

In `llamacpp_inventory_service.py`, add:

```python
def preview_import_asset_folder(path: Path, *, limit: int = 500) -> LlamaCppAssetImportPreviewResponse:
    canonical = _canonical_path(path, "Imported asset folder")
    _validate_path_for_config(canonical, "imported_asset_folders", "Imported asset folder")
    if not canonical.exists():
        raise ServerError("Imported asset folder does not exist.")
    if not canonical.is_dir():
        raise ServerError("Imported asset path is not a folder.")
    saved_config = _read_saved_config()
    allowed_bases = _allowed_bases_for_config(saved_config)
    if not allowed_bases or not handler_utils.is_path_allowed(canonical, allowed_bases):
        raise ServerError("Imported asset folder is outside allowed llama.cpp paths.")
    warnings: list[str] = []
    folder_asset = _folder_asset_for_path(canonical, allowed_bases=allowed_bases)
    assets = [
        asset
        for asset in (
            _asset_for_path(item, source="imported_folder", allowed_bases=allowed_bases, warnings=warnings)
            for item in _iter_asset_files(canonical, warnings, limit)
        )
        if asset is not None
    ]
    _attach_mmproj_candidates(assets)
    return LlamaCppAssetImportPreviewResponse(
        folder=folder_asset,
        assets=assets,
        asset_counts=_asset_counts(assets),
        warnings=warnings,
        scan_limited=len(assets) >= limit,
        will_persist=False,
    )
```

Use existing private helpers rather than duplicating allowlist or asset-kind logic.

- [ ] **Step 5: Write failing API tests for preview endpoint**

Add coverage for:

- `POST /api/v1/llamacpp/assets/import-folder/preview` returns preview response;
- preview does not mutate config;
- endpoint is admin-only in permission claims test;
- errors map to `400` with existing `ServerError` detail.

- [ ] **Step 6: Add preview endpoint**

In `llamacpp.py`, add:

```python
@router.post(
    "/llamacpp/assets/import-folder/preview",
    summary="Preview llama.cpp Asset Folder Import",
    response_model=LlamaCppAssetImportPreviewResponse,
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def preview_llamacpp_asset_folder_endpoint(
    payload: LlamaCppImportAssetFolderRequest,
    llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager),
) -> LlamaCppAssetImportPreviewResponse:
    _ = llm_manager
    try:
        return await run_in_threadpool(llamacpp_inventory_service.preview_import_asset_folder, Path(payload.path))
    except ServerError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
```

- [ ] **Step 7: Run Task 1 tests**

Run:

```bash
python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_asset_inventory_service.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_llamacpp_permissions_claims.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit Task 1**

```bash
git add \
  tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py \
  tldw_Server_API/app/core/Local_LLM/llamacpp_inventory_service.py \
  tldw_Server_API/app/api/v1/endpoints/llamacpp.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_asset_inventory_service.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_llamacpp_permissions_claims.py
git commit -m "feat: preview llama.cpp asset folder imports"
```

## Task 2: Acquisition Job Contract And API

**Files:**

- Modify: `tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py`
- Create: `tldw_Server_API/app/core/Local_LLM/llamacpp_acquisition_service.py`
- Create: `tldw_Server_API/app/core/Local_LLM/llamacpp_acquisition_jobs.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/llamacpp.py`
- Test: `tldw_Server_API/tests/LLM_Local/test_llamacpp_acquisition_service.py`
- Test: `tldw_Server_API/tests/LLM_Local/test_llamacpp_acquisition_api.py`

- [ ] **Step 1: Write failing acquisition service tests**

Cover:

- rejects empty URLs;
- rejects `file://`, `ftp://`, and unsupported schemes;
- rejects localhost/private/link-local hosts by default;
- allows private-network URLs only when explicit config says so;
- redacts userinfo and query secrets from source display/log metadata;
- resolves destination under configured `models_dir` or `allowed_paths`;
- rejects destination traversal and delimiter characters;
- reserves `.partial` paths under the final directory.

- [ ] **Step 2: Run service tests to verify failure**

Run:

```bash
python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_acquisition_service.py -v
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Add acquisition schemas**

Add request/response models:

```python
class LlamaCppAssetDownloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(..., min_length=1)
    destination_dir: str | None = None
    filename: str | None = None
    expected_sha256: str | None = None
    expected_size_bytes: int | None = Field(default=None, ge=1)
    source_label: str | None = None
    overwrite: bool = False
    register_asset: bool = True


class LlamaCppAcquisitionJobResponse(BaseModel):
    job_id: str
    status: str
    operation: Literal["download"]
    queue: str
    source_label: str | None = None
    destination_path: str | None = None
    asset_id: str | None = None
    progress: dict[str, object] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error_message: str | None = None
```

Keep response fields path-aware because endpoints are admin-only, but redact source URLs and never echo credentials.

- [ ] **Step 4: Implement acquisition service validation**

In `llamacpp_acquisition_service.py`, implement pure helpers:

- `validate_download_request(payload, saved_config) -> LlamaCppValidatedDownload`
- `redacted_source_label(url: str) -> str`
- `resolve_download_destination(payload, saved_config) -> Path`
- `partial_download_path(final_path: Path, job_id: str) -> Path`
- `validate_completed_download(path, expected_sha256, expected_size_bytes) -> list[str]`
- `register_completed_download(path) -> LlamaCppAsset`

Use stdlib `urllib.parse`, `ipaddress`, and `socket` for URL host checks. Treat DNS lookup failures as warnings only when the worker can still attempt the request, but block literal local/private IPs unless private downloads are explicitly enabled.

- [ ] **Step 5: Write failing API tests**

Cover:

- `POST /api/v1/llamacpp/assets/downloads` creates a Jobs row with domain `llamacpp`, queue `acquisition`, job type `llamacpp_asset_download`;
- request payload is stored without raw credentials;
- `GET /api/v1/llamacpp/assets/downloads/{job_id}` returns normalized job state;
- `GET /api/v1/llamacpp/assets/downloads` lists recent llama.cpp acquisition jobs;
- `DELETE /api/v1/llamacpp/assets/downloads/{job_id}` requests cancellation;
- admin permission claims include all new endpoints.

- [ ] **Step 6: Add job helper module and endpoints**

In `llamacpp_acquisition_jobs.py`, wrap `JobManager` creation/status mapping so endpoints remain thin:

```python
LLAMACPP_ACQUISITION_DOMAIN = "llamacpp"
LLAMACPP_ACQUISITION_QUEUE = "acquisition"
LLAMACPP_DOWNLOAD_JOB_TYPE = "llamacpp_asset_download"
```

Add endpoint routes:

```text
POST   /api/v1/llamacpp/assets/downloads
GET    /api/v1/llamacpp/assets/downloads
GET    /api/v1/llamacpp/assets/downloads/{job_id}
DELETE /api/v1/llamacpp/assets/downloads/{job_id}
```

Use `get_job_manager` from `API_Deps.jobs_deps`.

- [ ] **Step 7: Run Task 2 tests**

Run:

```bash
python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_acquisition_service.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_acquisition_api.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_llamacpp_permissions_claims.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit Task 2**

```bash
git add \
  tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py \
  tldw_Server_API/app/core/Local_LLM/llamacpp_acquisition_service.py \
  tldw_Server_API/app/core/Local_LLM/llamacpp_acquisition_jobs.py \
  tldw_Server_API/app/api/v1/endpoints/llamacpp.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_acquisition_service.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_acquisition_api.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_llamacpp_permissions_claims.py
git commit -m "feat: add llama.cpp asset acquisition job API"
```

## Task 3: Download Worker And Startup Wiring

**Files:**

- Create: `tldw_Server_API/app/services/llamacpp_acquisition_jobs_worker.py`
- Modify: `tldw_Server_API/app/services/startup_content_jobs_pollers.py`
- Modify: `tldw_Server_API/app/services/shutdown_primary_late_stop_workers.py` only if the startup worker group requires an explicit shutdown helper.
- Modify: `tldw_Server_API/app/core/Local_LLM/llamacpp_acquisition_service.py`
- Test: `tldw_Server_API/tests/LLM_Local/test_llamacpp_acquisition_jobs_worker.py`

- [ ] **Step 1: Write failing worker tests**

Use a local aiohttp/httpx test server or a monkeypatched stream adapter. Cover:

- successful download writes to `.partial`, validates size/checksum, atomically renames, registers asset, and completes job with `asset_id`;
- checksum mismatch deletes the partial/final file and fails terminally;
- cancellation deletes partial file and leaves no registered asset;
- existing destination with `overwrite=false` fails terminally;
- worker progress records bytes downloaded and total bytes when known;
- source URL credentials are never stored in job result/errors.

- [ ] **Step 2: Run worker tests to verify failure**

Run:

```bash
python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_acquisition_jobs_worker.py -v
```

Expected: FAIL because the worker does not exist.

- [ ] **Step 3: Implement WorkerSDK handler**

Follow `File_Artifacts/jobs_worker.py` structure:

```python
async def _handle_llamacpp_download_job(job: dict[str, Any]) -> dict[str, Any]:
    payload = job.get("payload") or {}
    validated = acquisition_service.validate_download_payload(payload)
    partial_path = acquisition_service.partial_download_path(validated.destination_path, str(job["id"]))
    try:
        bytes_written = await acquisition_service.download_to_partial(
            validated,
            partial_path,
            progress_callback=lambda progress: jobs_manager.update_job_progress(
                int(job["id"]),
                progress_percent=progress.get("progress_percent"),
                progress_message=progress.get("progress_message"),
            ),
        )
        warnings = acquisition_service.validate_completed_download(...)
        final_path = acquisition_service.promote_partial_download(partial_path, validated.destination_path)
        asset = acquisition_service.register_completed_download(final_path)
        return {"status": "ready", "asset_id": asset.asset_id, "bytes": bytes_written, "warnings": warnings}
    finally:
        acquisition_service.cleanup_partial_if_needed(partial_path)
```

Keep the network download implementation bounded:

- explicit timeout;
- max bytes if configured or `expected_size_bytes` present;
- chunked writes;
- no shell commands;
- no automatic profile creation.

- [ ] **Step 4: Wire worker startup**

In `startup_content_jobs_pollers.py`, add a worker gate controlled by an env flag:

- env: `LLAMACPP_ACQUISITION_JOBS_WORKER_ENABLED`;
- label: `llamacpp-acquisition`;
- coroutine factory imports `run_llamacpp_acquisition_jobs_worker`.

Use the existing worker inventory registration helper so shutdown diagnostics see it.

- [ ] **Step 5: Run worker/startup tests**

Run:

```bash
python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_acquisition_jobs_worker.py \
  tldw_Server_API/tests/Services/test_startup_content_jobs_pollers.py -v
```

If the startup test file does not exist, add focused coverage in the nearest existing startup worker test file rather than creating a broad app-lifespan test.

- [ ] **Step 6: Commit Task 3**

```bash
git add \
  tldw_Server_API/app/services/llamacpp_acquisition_jobs_worker.py \
  tldw_Server_API/app/services/startup_content_jobs_pollers.py \
  tldw_Server_API/app/core/Local_LLM/llamacpp_acquisition_service.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_acquisition_jobs_worker.py
git commit -m "feat: add llama.cpp acquisition download worker"
```

## Task 4: Admin WebUI Acquisition Surface

**Files:**

- Modify: `apps/packages/ui/src/types/llamacpp-admin.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/models-audio.ts`
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Modify: `apps/packages/ui/src/components/Option/Admin/LlamacppAssetsPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/Admin/LlamacppAdminPage.tsx`
- Test: `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAssetsPanel.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx`
- Test: `apps/packages/ui/src/services/__tests__/tldw-api-client.models-normalization.test.ts`

- [ ] **Step 1: Write failing frontend tests**

Cover:

- local import preview renders counts/warnings and requires an explicit import click;
- download form validates URL/destination client-side enough to prevent empty submissions;
- queued download appears with status/progress and a cancel action;
- completed download triggers an asset rescan;
- download completion does not create/start/wire a profile.

- [ ] **Step 2: Run frontend tests to verify failure**

Run from `apps/packages/ui`:

```bash
./node_modules/.bin/vitest run \
  src/components/Option/Admin/__tests__/LlamacppAssetsPanel.test.tsx \
  src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx \
  src/services/__tests__/tldw-api-client.models-normalization.test.ts
```

Expected: FAIL because acquisition methods and UI state do not exist.

- [ ] **Step 3: Add client types and methods**

Extend `llamacpp-admin.ts` with:

- `LlamacppAssetImportPreviewResponse`;
- `LlamacppAssetDownloadRequest`;
- `LlamacppAcquisitionJobResponse`;
- `LlamacppAcquisitionJobListResponse`.

Add `tldwClient` methods:

- `previewLlamacppAssetFolder(path)`;
- `startLlamacppAssetDownload(payload)`;
- `listLlamacppAssetDownloads()`;
- `getLlamacppAssetDownload(jobId)`;
- `cancelLlamacppAssetDownload(jobId)`.

- [ ] **Step 4: Update assets panel workflow**

Keep the panel compact:

- "Register asset path" remains synchronous.
- "Import folder" becomes preview then confirm.
- Add "Download asset" as a collapsed or secondary section, not a dominant hero.
- Render queued/running/completed/failed downloads as a small status list.
- Use warnings for disk/source/checksum issues where backend returns warnings.
- Do not add profile creation controls here unless a later Admin Console V2 task explicitly asks for it.

- [ ] **Step 5: Run frontend tests**

Run:

```bash
./node_modules/.bin/vitest run \
  src/components/Option/Admin/__tests__/LlamacppAssetsPanel.test.tsx \
  src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx \
  src/services/__tests__/tldw-api-client.models-normalization.test.ts
```

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

```bash
git add \
  apps/packages/ui/src/types/llamacpp-admin.ts \
  apps/packages/ui/src/services/tldw/domains/models-audio.ts \
  apps/packages/ui/src/services/tldw/TldwApiClient.ts \
  apps/packages/ui/src/components/Option/Admin/LlamacppAssetsPanel.tsx \
  apps/packages/ui/src/components/Option/Admin/LlamacppAdminPage.tsx \
  apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAssetsPanel.test.tsx \
  apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx \
  apps/packages/ui/src/services/__tests__/tldw-api-client.models-normalization.test.ts
git commit -m "feat: add llama.cpp acquisition workflow UI"
```

## Task 5: Docs, E2E Smoke, And Closeout

**Files:**

- Modify: `Docs/API-related/llamacpp_integration_modes.md`
- Modify: `Docs/Published/API-related/llamacpp_integration_modes.md`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-4-admin/admin-llamacpp.spec.ts`
- Modify: `backlog/tasks/task-416 - Plan-llama.cpp-model-acquisition-and-import-workflows.md`

- [ ] **Step 1: Update docs**

Document:

- local register path versus import folder;
- import preview does not mutate config;
- remote download is a Jobs-backed acquisition action;
- downloaded assets appear in normal inventory only after validation completes;
- acquisition never creates/starts/wires profiles automatically;
- private network URL policy and destination allowlist policy.

- [ ] **Step 2: Add or extend admin E2E smoke**

Mock backend responses for:

- import preview;
- confirmed folder import;
- queued download;
- completed download plus asset refresh.

Do not require real remote downloads in E2E.

- [ ] **Step 3: Run focused backend tests**

Run:

```bash
python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_asset_inventory_service.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_acquisition_service.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_acquisition_api.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_acquisition_jobs_worker.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_llamacpp_permissions_claims.py -v
```

Expected: PASS.

- [ ] **Step 4: Run focused frontend tests**

Run from `apps/packages/ui`:

```bash
./node_modules/.bin/vitest run \
  src/components/Option/Admin/__tests__/LlamacppAssetsPanel.test.tsx \
  src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx \
  src/services/__tests__/tldw-api-client.models-normalization.test.ts
```

Expected: PASS.

- [ ] **Step 5: Run Playwright smoke if dev-server prerequisites are available**

Run:

```bash
bunx playwright test apps/tldw-frontend/e2e/workflows/tier-4-admin/admin-llamacpp.spec.ts --reporter=line
```

If the suite requires a running backend/frontend server, start the established dev server workflow or record the blocker clearly.

- [ ] **Step 6: Run Bandit on touched Python paths**

Run:

```bash
python -m bandit \
  -r tldw_Server_API/app/core/Local_LLM/llamacpp_inventory_service.py \
     tldw_Server_API/app/core/Local_LLM/llamacpp_acquisition_service.py \
     tldw_Server_API/app/core/Local_LLM/llamacpp_acquisition_jobs.py \
     tldw_Server_API/app/api/v1/endpoints/llamacpp.py \
     tldw_Server_API/app/services/llamacpp_acquisition_jobs_worker.py \
  -f json -o /tmp/bandit_llamacpp_acquisition.json
```

Expected: no high/medium findings in touched code. If Bandit flags network/download behavior, fix the policy or document a narrow justified suppression only after reviewing the finding.

- [ ] **Step 7: Run diff checks**

Run:

```bash
git diff --check
git status --short --branch
```

Expected: no whitespace errors and only intentional files.

- [ ] **Step 8: Update Backlog and commit docs closeout**

Update `TASK-416` with exact verification results, known skips, and final summary.

```bash
git add \
  Docs/API-related/llamacpp_integration_modes.md \
  Docs/Published/API-related/llamacpp_integration_modes.md \
  apps/tldw-frontend/e2e/workflows/tier-4-admin/admin-llamacpp.spec.ts \
  "backlog/tasks/task-416 - Plan-llama.cpp-model-acquisition-and-import-workflows.md"
git commit -m "docs: document llama.cpp acquisition workflows"
```

Do not create an empty commit.

## Follow-Up Boundaries

After this acquisition slice lands, keep later work split as:

1. `llamacpp-admin-console-v2`
   - create/duplicate profile from selected asset;
   - searchable llama-server option browser;
   - readiness/assets/profiles/runtime layout cleanup.
2. `llamacpp-huggingface-catalog`
   - authenticated Hugging Face metadata/catalog browsing;
   - license/trust display;
   - source-specific resume/checksum support.
3. `llamacpp-advanced-routing`
   - Chat/Knowledge profile selection beyond the single global llama.cpp provider endpoint;
   - multi-user provider alias semantics;
   - profile-aware request routing.
