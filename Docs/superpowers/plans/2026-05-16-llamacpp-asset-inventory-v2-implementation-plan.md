# llama.cpp Asset Inventory V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand llama.cpp inventory from a GGUF-only model list into a local asset inventory that supports registered files, imported folders, stale-path warnings, and explicit mmproj candidate pairing while preserving the Stage 1 managed runtime contracts.

**Architecture:** Reuse the existing `llamacpp_inventory_service` as the inventory boundary and add asset-shaped schemas around it instead of creating a parallel catalog. Keep path registration in `[LlamaCpp]` config under the existing config lock, adding only an `imported_asset_folders` list for folder registrations; remote downloads and long-running acquisition jobs stay out of this slice. Legacy `/inventory` and `/models/register-path` endpoints become compatibility adapters over the new asset service so existing WebUI and API clients keep working.

**Tech Stack:** FastAPI, Pydantic v2, existing llama.cpp config lock and path allowlist helpers, pytest/TestClient, React/Ant Design shared WebUI, Vitest, Bandit.

---

## Scope Check

This plan implements Stage 3 from `Docs/superpowers/specs/2026-05-16-llamacpp-managed-runtime-roadmap-design.md`.

In scope:

- Asset response model for `gguf`, `mmproj`, `folder`, and `unknown` assets.
- Safe local file registration through a new asset endpoint.
- Safe local folder registration through a new "import folder" endpoint.
- Discovery of GGUF base models and mmproj projector files under configured roots and imported folders.
- Stale/missing/non-readable/outside-allowlist warnings attached to assets instead of whole-scan failure.
- Candidate mmproj/base model pairing metadata with warnings because pairing is inferred, not proven.
- Legacy inventory compatibility.
- Minimal WebUI assets panel that lists assets and exposes register/import actions.

Out of scope:

- Remote model downloads, URL ingestion, catalogs, checksum validation, disk quota jobs, or partial-file cleanup.
- Model-family routing through `/api/v1/llm/models/metadata`.
- Full Admin Console V2 redesign.
- Automatic profile mutation or automatic mmproj pairing.
- Chat image attachment routing.
- Copying, moving, uploading, or deleting model files.

## Design Decisions

- "Import folder" means "register an existing local folder and scan it". It must never copy/upload/move data in this slice.
- Asset IDs remain path-derived and stable for canonical resolved paths: `gguf:<hash>`, `mmproj:<hash>`, `folder:<hash>`, and `unknown:<hash>`.
- Symlinks are resolved before allowlist checks and before ID generation.
- Filename-based capability inference must be conservative. Use `unknown` warnings instead of claiming exact model capabilities from filenames alone.
- mmproj pairing is candidate metadata only. Profiles select a base model and projector explicitly in a later profile-editing flow.
- Keep `registered_model_paths` as the explicit file registration store and add `imported_asset_folders` as a sibling global config list. The name is legacy, but the new asset endpoint may store GGUF, mmproj, unknown, and stale file paths there for backward compatibility; legacy inventory must continue to filter that list down to GGUF models only. Do not introduce a second asset store in this slice.
- Existing `model_id` and `mmproj_model_id` fields remain in profile APIs for Stage 1 compatibility. Asset V2 may expose `asset_id`; profile field renames to `model_asset_id`/`mmproj_asset_id` are deferred to a migration plan.

## File Structure

Create:

- `tldw_Server_API/tests/LLM_Local/test_llamacpp_asset_inventory_service.py`
  - Unit coverage for asset discovery, stale warnings, imported folder handling, asset ID stability, and candidate pairing.
- `apps/packages/ui/src/components/Option/Admin/LlamacppAssetsPanel.tsx`
  - Minimal asset inventory panel for the existing Admin page.
- `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAssetsPanel.test.tsx`
  - Component tests for asset groups, warnings, register/import controls, and candidate pairing labels.

Modify:

- `tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py`
  - Add asset schemas and request/response types.
- `tldw_Server_API/app/core/Local_LLM/llamacpp_config_service.py`
  - Include `imported_asset_folders` in saved config parsing and config response.
- `tldw_Server_API/app/core/Local_LLM/llamacpp_inventory_service.py`
  - Add asset scanning, registration, folder import, pairing helpers, and legacy inventory adapter.
- `tldw_Server_API/app/api/v1/endpoints/llamacpp.py`
  - Add asset endpoints and keep legacy routes.
- `tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py`
  - Extend API coverage for asset endpoints and legacy compatibility.
- `apps/packages/ui/src/types/llamacpp-admin.ts`
  - Add asset types and asset request/response types.
- `apps/packages/ui/src/services/tldw/domains/models-audio.ts`
  - Add asset API methods.
- `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
  - Mirror/re-export asset API methods where this client still mirrors the domain client.
- `apps/packages/ui/src/components/Option/Admin/LlamacppAdminPage.tsx`
  - Load assets and render the minimal asset panel next to the existing inventory/launch flow.
- `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx`
  - Cover asset load/render and register/import reload behavior.

Do not modify:

- Download/acquisition job systems.
- `/api/v1/llm/models/metadata`.
- Knowledge or Chat provider selection flows.
- `llamacpp_profile_store.py` except for consuming existing IDs if a tiny compatibility helper is unavoidable.

## Shared Implementation Rules

- Use admin-only dependencies for every new endpoint:
  `dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))]`.
- Preserve warnings rather than hard blocking for stale, missing, non-GGUF, unknown capability, and inferred pairing states.
- Hard fail path traversal, unresolvable paths, non-allowlisted paths, and unsupported config delimiter characters.
- Reuse `handler_utils.build_allowed_paths` and `handler_utils.is_path_allowed`.
- Reuse `llamacpp_config_write_lock` for mutations to `[LlamaCpp]` config.
- Do not expose raw OS errors that include sensitive parent paths. Return asset-level warnings with the basename or the configured/imported root when possible.
- Keep legacy `LlamaCppInventoryResponse.models` filtered to base GGUF models only. mmproj assets must not appear in legacy model lists.
- Use `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python` in worktrees when `.venv` is absent.

## Data Contract

Add these schema shapes in `llamacpp_admin_schemas.py`:

```python
class LlamaCppAssetMetadata(BaseModel):
    quantization: str | None = None
    parameter_hint: str | None = None
    context_hint: int | None = None
    family_hint: str | None = None


class LlamaCppAsset(BaseModel):
    asset_id: str
    kind: Literal["gguf", "mmproj", "folder", "unknown"]
    identity_basis: Literal["resolved_path", "manual"]
    path: str
    resolved_path: str | None = None
    display_name: str
    source: Literal["models_dir", "registered_path", "imported_folder"]
    size_bytes: int | None = None
    modified_at: str | None = None
    metadata: LlamaCppAssetMetadata = Field(default_factory=LlamaCppAssetMetadata)
    capabilities: list[str] = Field(default_factory=list)
    mmproj_asset_ids: list[str] = Field(default_factory=list)
    base_model_asset_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LlamaCppAssetsResponse(BaseModel):
    assets: list[LlamaCppAsset]
    warnings: list[str] = Field(default_factory=list)
    scan_limited: bool = False


class LlamaCppRegisterAssetPathRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(..., min_length=1)


class LlamaCppImportAssetFolderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(..., min_length=1)
```

If Python 3.10 compatibility rejects dynamic `Literal` usage during tests, replace the `Literal[...]` fields with `str` plus a Pydantic `field_validator` using local string sets.

## Task 1: Asset Schema And Config Contract

**Files:**

- Modify: `tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py`
- Modify: `tldw_Server_API/app/core/Local_LLM/llamacpp_config_service.py`
- Test: `tldw_Server_API/tests/LLM_Local/test_llamacpp_asset_inventory_service.py`

- [x] **Step 1: Write failing schema/config tests**

Create `tldw_Server_API/tests/LLM_Local/test_llamacpp_asset_inventory_service.py` with:

```python
from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path

import pytest


def _llamacpp_parser(models_dir: Path, **overrides: str) -> ConfigParser:
    parser = ConfigParser()
    parser.add_section("LlamaCpp")
    values = {
        "enabled": "true",
        "models_dir": str(models_dir),
        "allowed_paths": "",
        "registered_model_paths": "",
        "imported_asset_folders": "",
    }
    values.update(overrides)
    parser["LlamaCpp"] = values
    return parser


@pytest.mark.unit
def test_config_state_reads_imported_asset_folders(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_config_service

    models_dir = tmp_path / "models"
    imported = tmp_path / "external"
    models_dir.mkdir()
    imported.mkdir()
    monkeypatch.setattr(
        llamacpp_config_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir, imported_asset_folders=str(imported)),
    )

    state = llamacpp_config_service.get_config_state(llm_manager=object())

    assert state["saved_config"]["imported_asset_folders"] == [str(imported)]
```

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_asset_inventory_service.py::test_config_state_reads_imported_asset_folders -v
```

Expected: FAIL because `imported_asset_folders` is not parsed yet.

- [x] **Step 3: Add schemas and saved config field**

In `llamacpp_admin_schemas.py`:

- add `Literal` to imports if compatible, otherwise use validated strings;
- add `imported_asset_folders: list[str] = Field(default_factory=list)` to `LlamaCppSavedConfig`;
- add the asset schema classes from the Data Contract section.

In `llamacpp_config_service.py`:

- add `"imported_asset_folders"` to `_LIST_FIELDS`;
- add `"imported_asset_folders"` to `_SAVED_FIELDS`;
- do not add it to `RESTART_FIELDS`; registering a folder changes inventory, not an active process;
- do not add an environment override in this slice unless existing config docs already define one.

- [x] **Step 4: Run test to verify it passes**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_asset_inventory_service.py::test_config_state_reads_imported_asset_folders -v
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py \
  tldw_Server_API/app/core/Local_LLM/llamacpp_config_service.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_asset_inventory_service.py
git commit -m "Add llama.cpp asset inventory schema contract"
```

## Task 2: Asset Discovery Service

**Files:**

- Modify: `tldw_Server_API/app/core/Local_LLM/llamacpp_inventory_service.py`
- Test: `tldw_Server_API/tests/LLM_Local/test_llamacpp_asset_inventory_service.py`

- [x] **Step 1: Write failing asset discovery tests**

Append:

```python
@pytest.mark.unit
def test_scan_assets_discovers_gguf_mmproj_and_folder(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service

    models_dir = tmp_path / "models"
    imported = tmp_path / "imported"
    models_dir.mkdir()
    imported.mkdir()
    (models_dir / "Llama-3-8B-Q4_K_M.gguf").write_text("base")
    (models_dir / "mmproj-Llama-3-vision-f16.gguf").write_text("projector")
    (imported / "notes.txt").write_text("not a model")

    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir, allowed_paths=str(imported), imported_asset_folders=str(imported)),
    )

    result = llamacpp_inventory_service.scan_assets(limit=500)

    by_kind = {asset.kind: asset for asset in result.assets}
    assert by_kind["gguf"].asset_id.startswith("gguf:")
    assert by_kind["mmproj"].asset_id.startswith("mmproj:")
    assert by_kind["folder"].asset_id.startswith("folder:")
    assert "vision_projector" in by_kind["mmproj"].capabilities
    assert by_kind["folder"].source == "imported_folder"


@pytest.mark.unit
def test_scan_assets_reports_stale_imported_folder_without_failing(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service

    models_dir = tmp_path / "models"
    missing = tmp_path / "missing"
    models_dir.mkdir()
    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir, imported_asset_folders=str(missing)),
    )

    result = llamacpp_inventory_service.scan_assets(limit=500)

    folder = next(asset for asset in result.assets if asset.kind == "folder")
    assert folder.resolved_path == str(missing)
    assert any("missing" in warning.lower() for warning in folder.warnings)
```

- [x] **Step 2: Run tests to verify failure**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_asset_inventory_service.py::test_scan_assets_discovers_gguf_mmproj_and_folder \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_asset_inventory_service.py::test_scan_assets_reports_stale_imported_folder_without_failing -v
```

Expected: FAIL because `scan_assets` does not exist.

- [x] **Step 3: Implement asset scanning helpers**

In `llamacpp_inventory_service.py`:

- add `asset_id_for_path(path: Path, kind: str) -> str`;
- add `scan_assets(config_state: dict[str, Any] | None = None, limit: int = 500) -> LlamaCppAssetsResponse`;
- add `_asset_for_path(...) -> LlamaCppAsset | None`;
- add `_folder_asset_for_path(...) -> LlamaCppAsset`;
- add `_iter_asset_files(root: Path, warnings: list[str], limit: int)`;
- update `_read_saved_config()` to include `imported_asset_folders`;
- update `_saved_config_from_state()` path to understand `imported_asset_folders`.

Implementation notes:

- scan registered model paths first, then configured `models_dir`, then imported folders;
- include a `folder` asset for every imported folder path even if it is missing;
- include files ending in `.gguf`;
- classify filename prefixes containing `mmproj` or `projector` as `mmproj`;
- classify other `.gguf` files as `gguf`;
- classify non-GGUF registered paths as `unknown` with a warning, but do not recursively include arbitrary non-GGUF files from scanned folders;
- use current traversal limits and warning style from `_iter_gguf_models`;
- keep de-duplication by `asset_id`;
- return assets sorted by source, kind, display name, and path for stable UI/tests.

- [x] **Step 4: Run tests to verify pass**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_asset_inventory_service.py -v
```

Expected: PASS.

- [x] **Step 5: Run legacy inventory tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py::test_inventory_recursively_scans_gguf_and_skips_mmproj \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py::test_inventory_model_ids_are_stable_for_canonical_path -v
```

Expected: PASS. If these fail because legacy inventory changed shape, fix the adapter instead of changing the tests.

- [x] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Local_LLM/llamacpp_inventory_service.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_asset_inventory_service.py
git commit -m "Add llama.cpp local asset discovery"
```

## Task 3: mmproj Candidate Pairing

**Files:**

- Modify: `tldw_Server_API/app/core/Local_LLM/llamacpp_inventory_service.py`
- Test: `tldw_Server_API/tests/LLM_Local/test_llamacpp_asset_inventory_service.py`

- [x] **Step 1: Write failing pairing tests**

Append:

```python
@pytest.mark.unit
def test_scan_assets_adds_inferred_mmproj_candidates_with_warnings(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    base = models_dir / "llava-7b-Q4_K_M.gguf"
    projector = models_dir / "mmproj-llava-7b-f16.gguf"
    base.write_text("base")
    projector.write_text("projector")
    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir),
    )

    result = llamacpp_inventory_service.scan_assets(limit=500)

    base_asset = next(asset for asset in result.assets if asset.kind == "gguf")
    projector_asset = next(asset for asset in result.assets if asset.kind == "mmproj")
    assert projector_asset.asset_id in base_asset.mmproj_asset_ids
    assert base_asset.asset_id in projector_asset.base_model_asset_ids
    assert any("inferred" in warning.lower() for warning in base_asset.warnings)
```

- [x] **Step 2: Run test to verify failure**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_asset_inventory_service.py::test_scan_assets_adds_inferred_mmproj_candidates_with_warnings -v
```

Expected: FAIL because candidate pairing is not populated.

- [x] **Step 3: Implement conservative pairing**

Add a helper such as `_attach_mmproj_candidates(assets: list[LlamaCppAsset]) -> None`.

Rules:

- Pair assets in the same directory when normalized token overlap is strong enough.
- Normalize names by lowercasing, stripping `.gguf`, removing `mmproj`, `projector`, quantization tokens, and common separators.
- If there is exactly one mmproj in the same directory as one or more base GGUF assets, treat it as a candidate for each base asset in that directory.
- If several candidates match, include all plausible IDs but add a warning that the pairing is inferred.
- Never remove or hide assets because pairing is ambiguous.
- Never auto-populate profile `mmproj_model_id` from candidate metadata.

- [x] **Step 4: Run pairing and service tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_asset_inventory_service.py -v
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Local_LLM/llamacpp_inventory_service.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_asset_inventory_service.py
git commit -m "Infer llama.cpp mmproj asset candidates"
```

## Task 4: Asset Registration And Folder Import APIs

**Files:**

- Modify: `tldw_Server_API/app/core/Local_LLM/llamacpp_inventory_service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/llamacpp.py`
- Modify: `tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py`

- [x] **Step 1: Write failing API tests**

Append tests to `test_llamacpp_inventory_api.py`:

```python
@pytest.mark.unit
def test_assets_endpoint_lists_gguf_and_mmproj(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "chat.Q4_K_M.gguf").write_text("base")
    (models_dir / "mmproj-chat-f16.gguf").write_text("projector")
    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir),
    )
    monkeypatch.setattr(lp.llamacpp_config_service, "get_config_state", lambda llm_manager: _config_state(models_dir))
    app = _make_app_with_manager(_Manager())

    with TestClient(app) as client:
        response = client.get("/api/v1/llamacpp/assets")

    assert response.status_code == 200, response.text
    kinds = {asset["kind"] for asset in response.json()["assets"]}
    assert {"gguf", "mmproj"} <= kinds


@pytest.mark.unit
def test_import_folder_persists_allowlisted_folder_and_returns_folder_asset(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service

    models_dir = tmp_path / "models"
    imported = tmp_path / "imported"
    models_dir.mkdir()
    imported.mkdir()
    updates: list[dict[str, dict[str, str]]] = []
    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir, allowed_paths=str(imported), imported_asset_folders=""),
    )
    monkeypatch.setattr(llamacpp_inventory_service.setup_manager, "update_config", updates.append)
    monkeypatch.setattr(llamacpp_inventory_service, "refresh_config_cache", lambda: None)
    app = _make_app_with_manager(_Manager())

    with TestClient(app) as client:
        response = client.post("/api/v1/llamacpp/assets/import-folder", json={"path": str(imported)})

    assert response.status_code == 200, response.text
    assert response.json()["kind"] == "folder"
    assert updates[-1]["LlamaCpp"]["imported_asset_folders"] == str(imported)
```

- [x] **Step 2: Run tests to verify failure**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py::test_assets_endpoint_lists_gguf_and_mmproj \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py::test_import_folder_persists_allowlisted_folder_and_returns_folder_asset -v
```

Expected: FAIL because routes/service methods do not exist.

- [x] **Step 3: Add service mutation methods**

In `llamacpp_inventory_service.py`:

- add `register_asset_path(path: Path) -> LlamaCppAsset`;
- add `import_asset_folder(path: Path) -> LlamaCppAsset`;
- keep `register_model_path(path: Path) -> LlamaCppInventoryItem` as a wrapper that calls the new registration path and adapts the result;
- validate paths with `_canonical_path`;
- reject non-allowlisted canonical paths with `ServerError`;
- reject delimiter characters using the existing config validation pattern;
- persist explicit file asset paths to `registered_model_paths` even when they are mmproj, unknown, or stale; the legacy inventory adapter is responsible for filtering non-GGUF entries;
- persist folders to `imported_asset_folders`;
- preserve existing registered paths/folders by stable ID and unresolved-path fallback.

- [x] **Step 4: Add endpoints**

In `llamacpp.py`, add routes near the legacy inventory routes:

```python
@router.get(
    "/llamacpp/assets",
    summary="List llama.cpp Local Assets",
    response_model=LlamaCppAssetsResponse,
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def get_llamacpp_assets_endpoint(
    llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager),
) -> LlamaCppAssetsResponse:
    config_state = llamacpp_config_service.get_config_state(llm_manager)
    return llamacpp_inventory_service.scan_assets(config_state)
```

Add `POST /llamacpp/assets/register-path` and `POST /llamacpp/assets/import-folder` with `ServerError` mapped to HTTP 400 and `from e` exception chaining.

- [x] **Step 5: Run API tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py -v
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Local_LLM/llamacpp_inventory_service.py \
  tldw_Server_API/app/api/v1/endpoints/llamacpp.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py
git commit -m "Add llama.cpp asset inventory APIs"
```

## Task 5: Legacy Inventory Compatibility Adapter

**Files:**

- Modify: `tldw_Server_API/app/core/Local_LLM/llamacpp_inventory_service.py`
- Modify: `tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py`

- [x] **Step 1: Write failing regression tests**

Append:

```python
@pytest.mark.unit
def test_legacy_inventory_excludes_mmproj_assets_after_asset_v2(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "chat.gguf").write_text("base")
    (models_dir / "mmproj-chat.gguf").write_text("projector")
    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir),
    )

    inventory = llamacpp_inventory_service.scan_inventory(limit=500)

    assert [item.basename for item in inventory.models] == ["chat.gguf"]


@pytest.mark.unit
def test_resolve_model_id_rejects_mmproj_asset_id(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service
    from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ModelNotFoundError

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    projector = models_dir / "mmproj-chat.gguf"
    projector.write_text("projector")
    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir),
    )

    with pytest.raises(ModelNotFoundError):
        llamacpp_inventory_service.resolve_model_id(llamacpp_inventory_service.asset_id_for_path(projector, "mmproj"))
```

- [x] **Step 2: Run tests to verify current behavior**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py::test_legacy_inventory_excludes_mmproj_assets_after_asset_v2 \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py::test_resolve_model_id_rejects_mmproj_asset_id -v
```

Expected: FAIL only if the adapter has not been corrected yet. If they already pass, keep the tests and continue.

- [ ] **Step 3: Adapt legacy scan over assets**

Update `scan_inventory()` so it can call `scan_assets()` internally, filter `kind == "gguf"`, and convert to `LlamaCppInventoryItem`.

Implementation note: no production adapter change was made in this step because the regression tests passed with the current legacy scanner, and adapting it directly through `scan_assets()` would risk dropping existing non-GGUF registered-path warning entries that legacy tests still protect.

Keep:

- `model_id_for_path(path)` returning `gguf:<hash>`;
- existing warning strings where tests depend on them;
- `registered_path` priority sorting;
- `scan_limited` behavior;
- `resolve_model_id()` resolving only available GGUF files.

- [x] **Step 4: Run focused backend tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_asset_inventory_service.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_runtime_api.py -v
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Local_LLM/llamacpp_inventory_service.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py
git commit -m "Preserve llama.cpp legacy inventory compatibility"
```

## Task 6: Frontend Types And API Client

**Files:**

- Modify: `apps/packages/ui/src/types/llamacpp-admin.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/models-audio.ts`
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`

- [x] **Step 1: Write failing type/client tests if a client test harness exists**

Search first:

```bash
rg -n "getLlamacppInventory|TldwApiClient|models-audio" apps/packages/ui/src --glob '*test*'
```

If an existing service test harness covers `models-audio`, add tests for:

- `getLlamacppAssets()` calls `/api/v1/llamacpp/assets`;
- `registerLlamacppAssetPath(path)` posts `{ path }`;
- `importLlamacppAssetFolder(path)` posts `{ path }`.

If no matching harness exists, document the skip in the commit body and rely on component/Admin page tests in Tasks 7 and 8.

Implementation note: no dedicated `models-audio` service test harness exists; coverage for the new methods is deferred to the component/Admin tests in Tasks 7 and 8.

- [x] **Step 2: Add TypeScript asset types**

Add:

```ts
export type LlamacppAssetKind = "gguf" | "mmproj" | "folder" | "unknown"
export type LlamacppAssetSource = "models_dir" | "registered_path" | "imported_folder"

export interface LlamacppAssetMetadata {
  quantization?: string | null
  parameter_hint?: string | null
  context_hint?: number | null
  family_hint?: string | null
}

export interface LlamacppAsset {
  asset_id: string
  kind: LlamacppAssetKind
  identity_basis: "resolved_path" | "manual"
  path: string
  resolved_path?: string | null
  display_name: string
  source: LlamacppAssetSource
  size_bytes?: number | null
  modified_at?: string | null
  metadata: LlamacppAssetMetadata
  capabilities: string[]
  mmproj_asset_ids: string[]
  base_model_asset_ids: string[]
  warnings: string[]
}

export interface LlamacppAssetsResponse {
  assets: LlamacppAsset[]
  warnings: string[]
  scan_limited: boolean
}
```

- [x] **Step 3: Add API methods**

Add methods to the domain client and mirrored client:

- `getLlamacppAssets(): Promise<LlamacppAssetsResponse>`
- `registerLlamacppAssetPath(path: string): Promise<LlamacppAsset>`
- `importLlamacppAssetFolder(path: string): Promise<LlamacppAsset>`

- [ ] **Step 4: Run type checks for touched paths**

Run:

```bash
bunx tsc --noEmit --pretty false
```

Expected: PASS or known repo-wide baseline failures unrelated to touched files. If repo-wide failures occur, run the narrow touched-path TypeScript check used in this repo's design-system slices and document the baseline separately.

Implementation note: `bunx tsc --noEmit --pretty false` was attempted, but Bun could not write to its temp directory inside the sandbox. The required escalated rerun was rejected by the approval reviewer, and retrying with `TMPDIR=/private/tmp` failed with the same tempdir access error. Frontend behavioral validation continues through Vitest in Tasks 7 and 8.

- [x] **Step 5: Commit**

```bash
git add apps/packages/ui/src/types/llamacpp-admin.ts \
  apps/packages/ui/src/services/tldw/domains/models-audio.ts \
  apps/packages/ui/src/services/tldw/TldwApiClient.ts
git commit -m "Add llama.cpp asset API client types"
```

## Task 7: Minimal Assets Panel

**Files:**

- Create: `apps/packages/ui/src/components/Option/Admin/LlamacppAssetsPanel.tsx`
- Create: `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAssetsPanel.test.tsx`

- [x] **Step 1: Write failing component tests**

Create tests that render:

- separate visible groups for GGUF models, mmproj projectors, and imported folders;
- global warnings and per-asset warnings;
- register file path control;
- import folder path control;
- candidate projector labels on GGUF assets.

Example:

```tsx
it("renders asset groups warnings and inferred projector candidates", async () => {
  render(
    <LlamacppAssetsPanel
      assets={{
        assets: [
          {
            asset_id: "gguf:base",
            kind: "gguf",
            identity_basis: "resolved_path",
            path: "/models/base.gguf",
            resolved_path: "/models/base.gguf",
            display_name: "base",
            source: "models_dir",
            size_bytes: 100,
            modified_at: null,
            metadata: {},
            capabilities: ["unknown"],
            mmproj_asset_ids: ["mmproj:vision"],
            base_model_asset_ids: [],
            warnings: ["Projector pairing is inferred."]
          },
          {
            asset_id: "mmproj:vision",
            kind: "mmproj",
            identity_basis: "resolved_path",
            path: "/models/mmproj-base.gguf",
            resolved_path: "/models/mmproj-base.gguf",
            display_name: "mmproj-base",
            source: "models_dir",
            size_bytes: 50,
            modified_at: null,
            metadata: {},
            capabilities: ["vision_projector"],
            mmproj_asset_ids: [],
            base_model_asset_ids: ["gguf:base"],
            warnings: []
          }
        ],
        warnings: [],
        scan_limited: false
      }}
      loading={false}
      registeringPath={false}
      importingFolder={false}
      error={null}
      onRegisterPath={vi.fn()}
      onImportFolder={vi.fn()}
      onReload={vi.fn()}
    />
  )

  expect(screen.getByText("GGUF models")).toBeTruthy()
  expect(screen.getByText("mmproj projectors")).toBeTruthy()
  expect(screen.getByText("Projector pairing is inferred.")).toBeTruthy()
})
```

- [x] **Step 2: Run test to verify failure**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAssetsPanel.test.tsx
```

Expected: FAIL because component does not exist.

Verification note: `bunx vitest run packages/ui/src/components/Option/Admin/__tests__/LlamacppAssetsPanel.test.tsx` from `apps/` failed red because `../LlamacppAssetsPanel` did not exist. Earlier root/package attempts exposed missing worktree dependency links; `bun install` in `apps/` repaired local test dependencies, and the unrelated tracked dependency-link pruning was restored before continuing.

- [x] **Step 3: Implement minimal panel**

Use existing local conventions from `LlamacppInventoryPanel.tsx`:

- Ant `Card`, `Button`, `Input`, `List`, `Space`, `Tag`, `Typography`;
- `DesignSystemAlert` for warnings/errors;
- `RefreshCw` icon for reload;
- two separate path inputs:
  - "Register local asset path"
  - "Import local asset folder"
- group assets by `kind`;
- show `source`, size, conservative capability tags, basename/path, and warnings;
- show candidate IDs as labels, not as automatic "paired" state;
- keep text compact and avoid redesigning the whole Admin page.

- [x] **Step 4: Run panel tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAssetsPanel.test.tsx
```

Expected: PASS.

Verification note: `./node_modules/.bin/vitest run src/components/Option/Admin/__tests__/LlamacppAssetsPanel.test.tsx` passed from `apps/packages/ui`.

- [x] **Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Option/Admin/LlamacppAssetsPanel.tsx \
  apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAssetsPanel.test.tsx
git commit -m "Add llama.cpp assets panel"
```

## Task 8: Admin Page Integration

**Files:**

- Modify: `apps/packages/ui/src/components/Option/Admin/LlamacppAdminPage.tsx`
- Modify: `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx`

- [x] **Step 1: Write failing Admin page tests**

Add tests for:

- page loads assets once during strict-mode mount together with existing status/config/inventory/runtime requests;
- register asset path calls the new asset endpoint, reloads assets, and keeps the existing legacy inventory selection working;
- import folder calls the folder endpoint and reloads assets;
- asset endpoint failure is shown without hiding the legacy inventory panel.

- [x] **Step 2: Run selected Admin page tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx
```

Expected: FAIL because Admin page does not yet load/render assets.

Verification note: `./node_modules/.bin/vitest run src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx` failed with 5 expected asset-integration failures before implementation.

- [x] **Step 3: Integrate asset state**

In `LlamacppAdminPage.tsx`:

- add state for `assets`, `loadingAssets`, `assetError`, `registeringAssetPath`, and `importingAssetFolder`;
- add `loadAssets()`;
- call `loadAssets()` in the same initial load effect as config/status/inventory/runtime, preserving the strict-mode single-load guard;
- add handlers:
  - `handleRegisterAssetPath(path): Promise<boolean>`
  - `handleImportAssetFolder(path): Promise<boolean>`
- reload assets after successful asset actions;
- reload legacy inventory after `registerAssetPath` only if the registered file is a GGUF asset, or simply reload both inventory and assets if the service does not return enough information yet;
- render `<LlamacppAssetsPanel />` near the existing inventory panel;
- keep the existing `LlamacppInventoryPanel` and launch flow intact.

- [x] **Step 4: Run Admin tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAssetsPanel.test.tsx \
  apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx
```

Expected: PASS.

Verification note: `./node_modules/.bin/vitest run src/components/Option/Admin/__tests__/LlamacppAssetsPanel.test.tsx src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx` passed from `apps/packages/ui`.

- [x] **Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Option/Admin/LlamacppAdminPage.tsx \
  apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx
git commit -m "Wire llama.cpp assets into admin page"
```

## Task 9: Final Verification And Security Sweep

**Files:**

- All touched files.
- Backlog task for the implementation slice.

- [ ] **Step 1: Run backend tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_asset_inventory_service.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_runtime_api.py -v
```

Expected: PASS.

- [ ] **Step 2: Run frontend tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAssetsPanel.test.tsx \
  apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppInventoryPanel.test.tsx \
  apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx
```

Expected: PASS.

- [ ] **Step 3: Run Bandit on touched Python scope**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit \
  -r tldw_Server_API/app/core/Local_LLM/llamacpp_inventory_service.py \
     tldw_Server_API/app/core/Local_LLM/llamacpp_config_service.py \
     tldw_Server_API/app/api/v1/endpoints/llamacpp.py \
  -f json -o /tmp/bandit_llamacpp_asset_inventory_v2.json
```

Expected: PASS with no new findings in touched code. If Bandit reports existing baseline issues outside the changed lines, document them separately and do not hide new issues.

- [ ] **Step 4: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 5: Update Backlog task**

Update the implementation task with:

- final summary;
- verification commands and outcomes;
- Bandit JSON path and result;
- known skips or blockers;
- PR link if one is opened.

- [ ] **Step 6: Final commit**

If any verification/backlog edits remain:

```bash
git add backlog/tasks/<implementation-task>.md
git commit -m "Document llama.cpp asset inventory verification"
```

## Review Checklist For Implementers

- [ ] New endpoints are admin-only and rate-limited.
- [ ] Registered files and imported folders must be under configured `models_dir` or `allowed_paths` after symlink resolution.
- [ ] Legacy `/api/v1/llamacpp/inventory` response shape is unchanged.
- [ ] Legacy `start-by-model` can still resolve GGUF `model_id` values.
- [ ] mmproj assets are visible in `/assets` but excluded from legacy `/inventory`.
- [ ] Folder import is registration only; no file copy/upload/move/download occurs.
- [ ] Candidate mmproj pairing is shown as inferred and never auto-applied to a profile.
- [ ] Remote download work remains deferred.
- [ ] No secrets or raw command-line args are introduced into asset warnings.

## Expected Follow-Up Work

- Model-family mode metadata and provider routing.
- Explicit profile editor controls for selecting base GGUF plus mmproj assets.
- Full Admin Console V2 layout across readiness/assets/profiles/runtime/advanced args.
- Remote download/acquisition jobs feeding the same `LlamaCppAsset` contract.
