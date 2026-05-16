# llama.cpp Model-Family And mmproj Profile Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the next llama.cpp managed runtime slice: explicit model-family modes, safe GGUF/mmproj profile wiring, managed-profile capability metadata, and minimal Admin UI/client visibility.

**Architecture:** Keep Asset Inventory V2 as the local asset source of truth and keep the Stage 1 supervisor/profile store as the lifecycle source of truth. Add a small backend resolver that validates a profile's base model asset, optional mmproj asset, mode, and launch args before start, then reuse that same resolved capability contract for `/api/v1/llm/models/metadata` and the Admin UI. Remote downloads, catalogs, and the full profile editor remain out of scope.

**Tech Stack:** FastAPI, Pydantic v2, existing llama.cpp profile store/supervisor/process runner, existing `/api/v1/llm/models/metadata`, pytest/TestClient, React/Ant Design shared UI, Vitest, Bandit.

---

## Scope Check

The approved roadmap still has several independent follow-ups after Stage 1 and Asset Inventory V2:

- model-family modes and mmproj/profile launch wiring;
- full Admin Console V2 with a profile editor;
- remote download/acquisition jobs and catalogs;
- advanced routing from Chat/Knowledge into multiple managed profiles.

This plan covers only the first item, with the smallest WebUI/client changes needed to expose the new fields and prevent users from losing visibility into mode/mmproj state. It does not implement remote downloads, Hugging Face catalog flows, automatic profile mutation from the asset panel, or a full profile editor.

## Current Baseline

Already landed:

- `LlamaCppProfile.mode` includes `chat`, `vision`, `embedding`, `rerank`, and `server_generic`.
- `LlamaCppProfile.model_id`, `model_path`, and `mmproj_model_id` exist.
- `LlamaCppAsset.kind` includes `gguf`, `mmproj`, `folder`, and `unknown`.
- `scan_assets()` attaches candidate `mmproj_asset_ids` and `base_model_asset_ids`.
- `LlamaCppProcessRunner` already knows how to format `server_args["mmproj"]` into `--mmproj`.
- The Admin runtime panel displays profile mode, but does not surface mmproj/profile capability state.

Missing:

- no single resolver validates profile mode against selected GGUF/mmproj assets;
- `mmproj_model_id` is not resolved or injected into runner args;
- vision profiles can start without a projector and projector paths can conflict with manual `server_args["mmproj"]`;
- managed profiles are not represented as capability-aware local model entries in `/api/v1/llm/models/metadata`;
- frontend types/panels do not show profile capability warnings or mmproj associations.

## File Structure

Create:

- `tldw_Server_API/app/core/Local_LLM/llamacpp_profile_capabilities.py`
  - Profile launch resolution and capability metadata helpers.
  - Resolve `profile.model_id`/`profile.model_path` to a base GGUF path.
  - Resolve `profile.mmproj_model_id` to an mmproj path when present.
  - Validate mode/assets/server args before launch.
  - Build bounded capability metadata for model catalog entries.
- `tldw_Server_API/tests/LLM_Local/test_llamacpp_profile_capabilities.py`
  - Unit tests for profile validation and metadata shaping.

Modify:

- `tldw_Server_API/app/core/Local_LLM/llamacpp_inventory_service.py`
  - Add a public asset resolver such as `resolve_asset_id(asset_id: str, expected_kind: str | None = None, assets: list[LlamaCppAsset] | None = None) -> LlamaCppAsset`.
  - Let callers pass a pre-scanned asset list so profile capability resolution and metadata generation do not repeat full model-directory scans.
  - Keep `resolve_model_id()` as the legacy GGUF-only helper.
- `tldw_Server_API/app/core/Local_LLM/llamacpp_supervisor_service.py`
  - Use the new resolver before `runner.start()`.
  - Pass normalized server args with resolved `mmproj` path when needed.
  - Reuse validation on profile create/update where possible without doing filesystem-heavy work on every list.
- `tldw_Server_API/app/core/Local_LLM/llamacpp_process_runner.py`
  - Add regression-only changes if needed after the supervisor starts passing normalized args.
  - Do not add a second mmproj resolution path here.
- `tldw_Server_API/app/core/Local_LLM/llamacpp_runtime_models.py`
  - Add optional runtime/profile capability fields only if needed by API responses.
  - Prefer keeping current profile field names in this slice: `model_id` for base GGUF inventory ID and `mmproj_model_id` for the mmproj asset ID.
- `tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py`
  - Add `capabilities`, `modalities`, `capability_warnings`, and `mmproj_path`/`mmproj_display_name` response fields only if the UI/API needs them.
  - Keep request compatibility with existing `mmproj_model_id`.
- `tldw_Server_API/app/api/v1/endpoints/llm_providers.py`
  - Append managed llama.cpp profile metadata entries in `/api/v1/llm/models/metadata`.
  - Keep existing static provider metadata behavior for `llama.cpp`.
- `tldw_Server_API/tests/LLM_Local/test_llamacpp_runtime_api.py`
  - Regression coverage for vision profile start and invalid mmproj cases.
- `tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py`
  - Resolver/API coverage for asset IDs and kind filtering.
- `tldw_Server_API/tests/LLM_Local/test_llm_models_metadata_llamacpp_profiles.py`
  - Endpoint coverage for managed profile metadata.
- `apps/packages/ui/src/types/llamacpp-admin.ts`
  - Type the new profile/runtime capability fields.
- `apps/packages/ui/src/components/Option/Admin/LlamacppRuntimePanel.tsx`
  - Display profile mode, mmproj/capability warnings, and vision capability status without adding a full editor.
- `apps/packages/ui/src/components/Option/Admin/LlamacppAssetsPanel.tsx`
  - Only add visible candidate labels if needed by tests; do not create profiles from assets in this slice.
- `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppRuntimePanel.test.tsx`
  - Verify capability/mmproj display.
- `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx`
  - Verify updated runtime payloads do not break page loading.

Do not modify:

- remote download/catalog code;
- Chat message image attachment behavior;
- Knowledge retrieval routing;
- provider secret storage;
- full Admin profile editor flows;
- global config update behavior except through existing `use-in-chat`.

## Shared Implementation Rules

- Preserve V1 compatibility: `/api/v1/llamacpp/start-by-model` keeps targeting the default profile and should not require mmproj fields.
- Keep profile start explicit. Do not auto-wire a profile into Chat just because it starts.
- Fail closed for unsafe local paths, unknown asset IDs, wrong asset kinds, and conflicting manual `server_args["mmproj"]`.
- Prefer warnings over hard blocks for uncertain metadata, but hard-block impossible launches such as `vision` without a valid mmproj.
- Do not re-scan huge folders in `/api/v1/llm/models/metadata`; use profile store data and resolve only the selected assets needed for managed profile metadata.
- Keep metadata bounded: no raw local paths in public model metadata unless the existing Admin-only endpoint already exposes them.
- Keep remote downloads deferred.

## Task 1: Asset Resolver And Profile Capability Contract

**Files:**

- Create: `tldw_Server_API/app/core/Local_LLM/llamacpp_profile_capabilities.py`
- Modify: `tldw_Server_API/app/core/Local_LLM/llamacpp_inventory_service.py`
- Test: `tldw_Server_API/tests/LLM_Local/test_llamacpp_profile_capabilities.py`
- Test: `tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py`

- [ ] **Step 1: Write failing asset resolver tests**

Add tests for:

```python
def test_resolve_asset_id_rejects_wrong_kind(monkeypatch, tmp_path):
    base = tmp_path / "models" / "chat.gguf"
    mmproj = tmp_path / "models" / "mmproj-chat.gguf"
    base.parent.mkdir()
    base.write_text("base")
    mmproj.write_text("projector")
    configure_assets(monkeypatch, models_dir=base.parent)

    base_id = llamacpp_inventory_service.asset_id_for_path(base, "gguf")

    with pytest.raises(ModelNotFoundError):
        llamacpp_inventory_service.resolve_asset_id(base_id, expected_kind="mmproj")
```

Also cover a missing asset ID and a stale registered asset path that yields a clear `ModelNotFoundError`.

- [ ] **Step 2: Run asset resolver tests to verify failure**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py::test_resolve_asset_id_rejects_wrong_kind -v
```

Expected: FAIL because `resolve_asset_id()` does not exist.

- [ ] **Step 3: Add public asset resolver**

Implement in `llamacpp_inventory_service.py`:

```python
def resolve_asset_id(
    asset_id: str,
    expected_kind: str | None = None,
    assets: list[LlamaCppAsset] | None = None,
) -> LlamaCppAsset:
    wanted = str(asset_id or "").strip()
    if not wanted:
        raise ModelNotFoundError("Asset ID is required.")
    search_pool = assets if assets is not None else scan_assets().assets
    for asset in search_pool:
        if asset.asset_id != wanted:
            continue
        if expected_kind and asset.kind != expected_kind:
            raise ModelNotFoundError(f"Asset ID {wanted} does not reference a {expected_kind} asset.")
        if asset.resolved_path and Path(asset.resolved_path).exists():
            return asset
        raise ModelNotFoundError(f"Asset ID {wanted} does not reference an available local asset.")
    raise ModelNotFoundError(f"Asset ID {wanted} was not found.")
```

Keep exceptions aligned with `resolve_model_id()` so API mapping remains familiar.

- [ ] **Step 4: Write failing profile capability tests**

Create `test_llamacpp_profile_capabilities.py` covering:

```python
def test_chat_profile_resolves_base_model_without_mmproj(monkeypatch, tmp_path):
    profile = LlamaCppProfile(
        profile_id="chat",
        name="Chat",
        mode=LlamaCppProfileMode.CHAT,
        model_id=asset_id_for_path(base, "gguf"),
    )
    resolved = resolve_profile_launch(profile)
    assert resolved.model_path == base.resolve()
    assert "mmproj" not in resolved.server_args
    assert resolved.capabilities["vision"] is False


def test_vision_profile_requires_mmproj_asset(monkeypatch, tmp_path):
    profile = LlamaCppProfile(
        profile_id="vision",
        name="Vision",
        mode=LlamaCppProfileMode.VISION,
        model_id=asset_id_for_path(base, "gguf"),
    )
    with pytest.raises(ServerError, match="mmproj"):
        resolve_profile_launch(profile)
```

Add coverage for:

- valid vision profile injects `server_args["mmproj"]` with the resolved projector path;
- manual `server_args["mmproj"]` plus `mmproj_model_id` conflict is rejected unless both resolve to the same path;
- embedding mode derives embeddings capability and text-only modalities;
- rerank mode derives rerank capability and text-only modalities;
- server_generic mode does not claim Chat/Knowledge capabilities.

- [ ] **Step 5: Run profile capability tests to verify failure**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_profile_capabilities.py -v
```

Expected: FAIL because the module does not exist.

- [ ] **Step 6: Add profile capability module**

Implement:

```python
class LlamaCppResolvedProfileLaunch(BaseModel):
    profile: LlamaCppProfile
    model_path: Path
    mmproj_path: Path | None = None
    server_args: dict[str, object]
    capabilities: dict[str, bool]
    modalities: dict[str, list[str]]
    warnings: list[str] = Field(default_factory=list)
```

Add:

```python
def resolve_profile_launch(
    profile: LlamaCppProfile,
    assets: list[LlamaCppAsset] | None = None,
) -> LlamaCppResolvedProfileLaunch: ...


def profile_capability_metadata(
    profile: LlamaCppProfile,
    assets: list[LlamaCppAsset] | None = None,
) -> dict[str, object]: ...
```

When resolving multiple profiles, scan once and pass the same asset list into each helper call.

Mode mapping:

- `chat`: text input/output, chat capability true, vision false.
- `vision`: text+image input, text output, chat and vision true, mmproj required.
- `embedding`: text input, embedding/vector output, embeddings true, chat false.
- `rerank`: text input, score output, rerank true, chat false.
- `server_generic`: text input/output only, no specialized claim unless explicit safe metadata is added later.

Use current field names:

- `profile.model_id`: preferred base GGUF asset/model ID.
- `profile.model_path`: legacy/manual path fallback.
- `profile.mmproj_model_id`: mmproj asset ID in this slice.

- [ ] **Step 7: Run Task 1 tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_profile_capabilities.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add \
  tldw_Server_API/app/core/Local_LLM/llamacpp_profile_capabilities.py \
  tldw_Server_API/app/core/Local_LLM/llamacpp_inventory_service.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_profile_capabilities.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py
git commit -m "feat: add llama.cpp profile capability resolver"
```

## Task 2: Supervisor Launch Validation And mmproj Injection

**Files:**

- Modify: `tldw_Server_API/app/core/Local_LLM/llamacpp_supervisor_service.py`
- Modify: `tldw_Server_API/app/core/Local_LLM/llamacpp_runtime_models.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py`
- Test: `tldw_Server_API/tests/LLM_Local/test_llamacpp_supervisor_service.py`
- Test: `tldw_Server_API/tests/LLM_Local/test_llamacpp_runtime_api.py`

- [ ] **Step 1: Write failing supervisor tests**

Add tests that patch asset resolution and runner factory:

```python
async def test_supervisor_starts_vision_profile_with_resolved_mmproj(monkeypatch, tmp_path):
    _base_path, mmproj_path = configure_supervisor_assets(
        monkeypatch,
        tmp_path,
        base_asset_id="gguf:base",
        mmproj_asset_id="mmproj:projector",
    )
    profile = LlamaCppProfile(
        profile_id="vision",
        name="Vision",
        mode=LlamaCppProfileMode.VISION,
        model_id="gguf:base",
        mmproj_model_id="mmproj:projector",
        server_args={"ctx_size": 4096},
    )
    store.upsert(profile)

    runtime = await supervisor.start_profile("vision")

    assert runtime.profile_id == "vision"
    assert runner.starts[0].server_args["mmproj"] == str(mmproj_path)
```

Also cover:

- vision profile without mmproj returns/raises a `ServerError`;
- wrong-kind mmproj asset is rejected before runner spawn;
- manual `server_args["mmproj"]` path outside allowlist still fails;
- `resolved_args`/runtime response includes the injected `--mmproj` path redacted through the existing command redaction path.

- [ ] **Step 2: Run supervisor tests to verify failure**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_supervisor_service.py::test_supervisor_starts_vision_profile_with_resolved_mmproj -v
```

Expected: FAIL because supervisor still sends raw profile args without profile capability resolution.

- [ ] **Step 3: Wire resolver into supervisor start**

In `_start_profile_unlocked()`:

```python
resolved = resolve_profile_launch(profile)
runtime = await runner.start(
    resolved.model_path,
    profile.model_copy(update={"server_args": resolved.server_args}),
)
```

Keep all existing per-profile locking and port checks.

- [ ] **Step 4: Preserve profile create/update behavior**

On profile create/update:

- keep cheap Pydantic/profile field validation in create/update;
- do not force expensive asset scanning during every update unless `model_id`, `model_path`, `mmproj_model_id`, `mode`, or `server_args` changed;
- when validating changed asset-related fields, map `ModelNotFoundError` to `400` from the endpoint layer and `LlamaCppProfileConflictError` to `409`.

Do not silently mutate `server_args` in persisted profiles just because a valid `mmproj_model_id` exists. Inject normalized launch args at start time.

- [ ] **Step 5: Extend runtime/profile response fields only if needed**

If frontend tests need it, add optional response-only fields:

```python
capabilities: dict[str, bool] = Field(default_factory=dict)
modalities: dict[str, list[str]] = Field(default_factory=dict)
capability_warnings: list[str] = Field(default_factory=list)
mmproj_path: str | None = None
```

Prefer response-only fields over changing request field names. Keep `mmproj_model_id` in request/response for compatibility.

- [ ] **Step 6: Run runtime API tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_supervisor_service.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_runtime_api.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add \
  tldw_Server_API/app/core/Local_LLM/llamacpp_supervisor_service.py \
  tldw_Server_API/app/core/Local_LLM/llamacpp_runtime_models.py \
  tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_supervisor_service.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_runtime_api.py
git commit -m "feat: wire llama.cpp mmproj profiles into launch"
```

## Task 3: Managed Profile Metadata In `/api/v1/llm/models/metadata`

**Files:**

- Create or modify: `tldw_Server_API/app/core/Local_LLM/llamacpp_profile_capabilities.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/llm_providers.py`
- Modify if needed: `tldw_Server_API/app/core/Local_LLM/llamacpp_supervisor_service.py`
- Test: `tldw_Server_API/tests/LLM_Local/test_llm_models_metadata_llamacpp_profiles.py`

- [ ] **Step 1: Write failing metadata endpoint tests**

Create tests for:

```python
def test_models_metadata_includes_managed_llamacpp_profiles(client, monkeypatch):
    profile = LlamaCppProfile(
        profile_id="vision",
        name="Vision profile",
        mode=LlamaCppProfileMode.VISION,
        model_id="gguf:base",
        mmproj_model_id="mmproj:projector",
        provider_alias="llamacpp-vision",
    )
    monkeypatch_profile_store([profile])

    response = client.get("/api/v1/llm/models/metadata")
    models = response.json()["models"]

    entry = next(item for item in models if item["llamacpp_profile_id"] == "vision")
    assert entry["provider"] == "llama.cpp"
    assert entry["model"] == "llamacpp-vision"
    assert entry["capabilities"]["vision"] is True
    assert entry["modalities"]["input"] == ["text", "image"]
```

Also cover:

- disabled profiles are included as `is_configured=false` or skipped consistently; choose one behavior and document it in the test;
- `type=chat` and `input_modality=image` filters include/exclude managed entries correctly;
- invalid/stale profile assets produce a bounded warning entry instead of failing the entire metadata endpoint.

- [ ] **Step 2: Run metadata tests to verify failure**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llm_models_metadata_llamacpp_profiles.py -v
```

Expected: FAIL because managed profile metadata is not appended.

- [ ] **Step 3: Add profile metadata builder**

In `llamacpp_profile_capabilities.py`, add:

```python
def managed_profile_model_metadata(
    profile: LlamaCppProfile,
    assets: list[LlamaCppAsset] | None = None,
) -> dict[str, object]:
    alias = profile.provider_alias or f"llamacpp:{profile.profile_id}"
    capabilities = profile_capability_metadata(profile, assets=assets)
    return {
        "provider": "llama.cpp",
        "model": alias,
        "name": profile.name,
        "type": _type_for_mode(profile.mode),
        "llamacpp_profile_id": profile.profile_id,
        "source": "managed_llamacpp_profile",
        "provider_is_configured": profile.enabled,
        "is_configured": profile.enabled,
        "catalog_only": False,
        **capabilities,
    }
```

Do not expose resolved local paths in this public metadata payload. If a profile has stale assets, include `warnings` and conservative capabilities.

- [ ] **Step 4: Append entries in `llm_providers.py`**

Reuse the existing manager/supervisor path rather than constructing a separate `JsonLlamaCppProfileStore` in the endpoint. If the endpoint needs access to managed profiles, add a `Request` parameter to `get_models_metadata()` and read `request.app.state.llm_manager.llamacpp_supervisor`, using the same fallback pattern as `llamacpp.py` tests if needed.

Add a small helper near `get_models_metadata()`:

```python
def _managed_llamacpp_profile_metadata_entries(supervisor: LlamaCppSupervisor | None) -> list[dict[str, Any]]:
    if supervisor is None:
        return []
    try:
        profiles = supervisor.list_profiles()
        assets = llamacpp_inventory_service.scan_assets().assets if profiles else []
        return [managed_profile_model_metadata(profile, assets=assets) for profile in profiles]
    except Exception:
        logger.debug("Failed to load managed llama.cpp profiles for model metadata", exc_info=True)
        return []
```

Then append before image models and run the existing `_model_matches_filters()` check on each entry.

- [ ] **Step 5: Run metadata endpoint tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llm_models_metadata_llamacpp_profiles.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_runtime_api.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add \
  tldw_Server_API/app/core/Local_LLM/llamacpp_profile_capabilities.py \
  tldw_Server_API/app/api/v1/endpoints/llm_providers.py \
  tldw_Server_API/tests/LLM_Local/test_llm_models_metadata_llamacpp_profiles.py
git commit -m "feat: expose managed llama.cpp profile metadata"
```

## Task 4: Minimal WebUI Capability Visibility

**Files:**

- Modify: `apps/packages/ui/src/types/llamacpp-admin.ts`
- Modify: `apps/packages/ui/src/components/Option/Admin/LlamacppRuntimePanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/Admin/LlamacppAssetsPanel.tsx`
- Test: `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppRuntimePanel.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAssetsPanel.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx`

- [ ] **Step 1: Write failing frontend tests**

Add runtime-panel coverage:

```tsx
it("shows vision and mmproj state for a managed runtime profile", () => {
  render(
    <LlamacppRuntimePanel
      profiles={[visionProfile]}
      runtimes={[visionRuntime]}
      onRefresh={vi.fn()}
      onStart={vi.fn()}
      onStop={vi.fn()}
      onPause={vi.fn()}
      onResume={vi.fn()}
      onUseInChat={vi.fn()}
    />
  )
  expect(screen.getByText("vision")).toBeInTheDocument()
  expect(screen.getByText("Vision input")).toBeInTheDocument()
  expect(screen.getByText("mmproj-chat.gguf")).toBeInTheDocument()
})
```

Add assets-panel coverage only if display labels change:

- base GGUF shows projector candidate IDs/names;
- mmproj asset shows candidate base model IDs/names.

- [ ] **Step 2: Run frontend tests to verify failure**

Run from `apps/packages/ui`:

```bash
./node_modules/.bin/vitest run \
  src/components/Option/Admin/__tests__/LlamacppRuntimePanel.test.tsx \
  src/components/Option/Admin/__tests__/LlamacppAssetsPanel.test.tsx \
  src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx
```

Expected: FAIL because capability/mmproj fields are not rendered or typed.

- [ ] **Step 3: Extend TypeScript types**

Add optional fields matching backend responses:

```ts
capabilities?: Record<string, boolean>
modalities?: Record<string, string[]>
capability_warnings?: string[]
mmproj_path?: string | null
mmproj_display_name?: string | null
```

Keep all fields optional so older servers still render.

- [ ] **Step 4: Render capability state**

In `LlamacppRuntimePanel`:

- keep the compact list layout;
- add `Vision input` tag for profile/runtime capability `vision`;
- add `Embeddings` and `Rerank` tags for non-chat modes;
- show a small `mmproj` tag/secondary text when a profile/runtime has a projector;
- show capability warnings as orange tags or the existing warning surface;
- preserve current actions and fallback behavior.

Do not add profile editing controls in this task.

- [ ] **Step 5: Run frontend tests**

Run:

```bash
./node_modules/.bin/vitest run \
  src/components/Option/Admin/__tests__/LlamacppRuntimePanel.test.tsx \
  src/components/Option/Admin/__tests__/LlamacppAssetsPanel.test.tsx \
  src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add \
  apps/packages/ui/src/types/llamacpp-admin.ts \
  apps/packages/ui/src/components/Option/Admin/LlamacppRuntimePanel.tsx \
  apps/packages/ui/src/components/Option/Admin/LlamacppAssetsPanel.tsx \
  apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppRuntimePanel.test.tsx \
  apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAssetsPanel.test.tsx \
  apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx
git commit -m "feat: show llama.cpp profile capabilities"
```

## Task 5: Focused Verification And Task Closeout

**Files:**

- Modify: `backlog/tasks/task-407 - Plan-llama.cpp-model-family-and-mmproj-profile-wiring.md` or the implementation task record created for code execution.
- Modify: `Docs/superpowers/plans/2026-05-16-llamacpp-model-family-mmproj-profile-wiring-plan.md` only for verification notes if needed.

- [ ] **Step 1: Run focused backend tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_profile_capabilities.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_asset_inventory_service.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_supervisor_service.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_runtime_api.py \
  tldw_Server_API/tests/LLM_Local/test_llm_models_metadata_llamacpp_profiles.py -v
```

Expected: PASS.

- [ ] **Step 2: Run focused frontend tests**

Run from `apps/packages/ui`:

```bash
./node_modules/.bin/vitest run \
  src/components/Option/Admin/__tests__/LlamacppRuntimePanel.test.tsx \
  src/components/Option/Admin/__tests__/LlamacppAssetsPanel.test.tsx \
  src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx \
  src/services/__tests__/model-settings.llamacpp-controls.test.ts \
  src/utils/__tests__/build-llamacpp-server-args.test.ts
```

Expected: PASS.

- [ ] **Step 3: Run Bandit on touched Python paths**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit \
  -r tldw_Server_API/app/core/Local_LLM/llamacpp_profile_capabilities.py \
     tldw_Server_API/app/core/Local_LLM/llamacpp_inventory_service.py \
     tldw_Server_API/app/core/Local_LLM/llamacpp_supervisor_service.py \
     tldw_Server_API/app/api/v1/endpoints/llm_providers.py \
     tldw_Server_API/app/api/v1/endpoints/llamacpp.py \
  -f json -o /tmp/bandit_llamacpp_model_family_mmproj.json
```

Expected: no high/medium findings in touched code. If Bandit reports path or subprocess findings, verify there is no shell invocation and add a narrow `# nosec` only when justified.

- [ ] **Step 4: Run diff checks**

Run:

```bash
git diff --check
git status --short --branch
```

Expected: no whitespace errors and only intentional files.

- [ ] **Step 5: Update Backlog task**

Update the implementation task with:

- final summary;
- exact verification commands and results;
- any known skips, especially repo-wide TypeScript debt if package-level `tsc` is attempted and fails outside touched files;
- PR URL after creation.

- [ ] **Step 6: Commit verification notes if needed**

```bash
git add <task-or-plan-files>
git commit -m "docs: record llama.cpp model-family verification"
```

Do not create an empty commit.

## Follow-Up Boundaries

After this plan lands and the implementation PR merges, the remaining roadmap work should be split as:

1. `llamacpp-admin-console-v2`
   - full profile editor;
   - create/duplicate profile from selected asset;
   - searchable llama-server option browser;
   - readiness/assets/profiles/runtime layout.
2. `llamacpp-download-acquisition-jobs`
   - remote source downloads;
   - cancellation/retry/checksum/disk warnings;
   - atomic asset registration only after complete validation.
3. `llamacpp-advanced-routing`
   - Chat/Knowledge profile selection beyond the single global llama.cpp provider endpoint;
   - multi-user provider alias semantics;
   - profile-aware request routing.
