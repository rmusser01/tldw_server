# VN Platform API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the backend-owned `/api/v1/vn/vn-*` VN platform API contract for capabilities, assets, scripts, play/runtime, policy, and VN audio.

**Architecture:** Reuse the existing VN asset and VN play modules instead of rebuilding them. Add a small shared VN platform layer for stable errors, idempotency, capabilities, route registration, and profile snapshot rules; add focused modules for scripts, policy/generation profiles, and VN TTS metadata. Keep implementation API-first and backend-owned so custom frontends can use the server without WebUI-only coupling.

**Tech Stack:** FastAPI, Pydantic v2, SQLite-backed per-user `ChaChaNotes.db`, existing AuthNZ/generated-file storage, core Jobs, existing VN asset/play services, existing TTS service stack, pytest, Bandit, OpenAPI contract tests.

---

## Source Documents

- Reviewed API spec: `Docs/superpowers/specs/2026-05-10-vn-platform-api-design.md`
- Existing VN asset docs: `Docs/API-related/VN_ASSET_PACKS_API.md`
- Existing VN play docs: `Docs/API-related/VN_PLAY_API.md`
- Existing VN asset spec/plan:
  - `Docs/superpowers/specs/2026-04-24-vn-asset-packs-design.md`
  - `Docs/superpowers/plans/2026-04-24-vn-asset-packs-implementation-plan.md`
- Existing VN play spec/plan:
  - `Docs/superpowers/specs/2026-05-01-vn-play-runtime-design.md`
  - `Docs/superpowers/plans/2026-05-01-vn-play-runtime-implementation.md`
- Existing portability spec/plan:
  - `Docs/superpowers/specs/2026-04-25-vn-pack-portability-design.md`
  - `Docs/superpowers/plans/2026-04-25-vn-pack-portability-implementation.md`

## Scope Boundary

Implement this as backend/API work. Do not build new WebUI surfaces in this plan.

Allowed WebUI/client touchpoints:

- update typed API clients or smoke fixtures only where route migration breaks existing tests;
- update docs links from old `/api/v1/vn-assets` and `/api/v1/vn-play` paths to canonical `/api/v1/vn/vn-assets` and `/api/v1/vn/vn-play`.

Out of scope:

- realtime image generation during play;
- session/script export/import;
- marketplace or cross-user sharing;
- multiplayer/co-op;
- SSE/WebSocket/webhooks;
- rich media timeline/lip sync;
- full built-in inventory/stat systems;
- collaborative script editing;
- frontend authoring workbench.

## Delivery Shape

This is intentionally split into independent PR-sized slices. Each task below should leave the repo in a working state and should be committed separately.

Recommended execution order:

1. Platform shell and canonical route namespace.
2. Asset API migration and safety hardening.
3. Policy and generation profiles.
4. Script authoring and publish API.
5. Scripted runtime and shared runtime hardening.
6. VN audio TTS API.
7. Docs, OpenAPI, migration, and verification closeout.

If the branch grows too large, stop after any task and open a PR for that slice.

## Cross-Cutting Test Requirements

Every resource task must include focused tests for:

- authentication and owner scoping;
- content/preview endpoints validating owner, generated-file source feature,
  media type, generated-file metadata, and policy before serving bytes;
- stable VN error `detail` object shape for new endpoints;
- idempotency replay and conflict behavior for every mutating endpoint that
  creates work, publishes, uploads, advances state, restores state, creates save
  slots, or creates TTS jobs;
- offset pagination on list endpoints;
- route registration under `/api/v1/vn` and OpenAPI path coverage.

Do not defer these to the final docs task unless the resource has already added
the behavior and only needs cross-module OpenAPI aggregation.

## Required Idempotency Matrix

Implement and test idempotency for:

- VN assets: generation, export, import preview upload, import commit, cleanup
  execution, item upload, item regenerate, slot retry;
- VN scripts: publish;
- VN play: Story start, Freeform/Story turns, scripted advance, scripted choice,
  scripted regenerate, checkpoint restore, branch restore, save-slot creation,
  save-slot restore;
- VN audio: TTS job creation.

Each task that owns one of these commands must add both same-payload replay tests
and same-key/different-payload conflict tests.

## File Map

### Existing Files To Reuse

- `tldw_Server_API/app/api/v1/router_groups/content.py`
  - Register canonical VN routers under `/api/v1/vn`.
  - Remove old root-level VN route registration in the target API slice.
- `tldw_Server_API/app/api/v1/endpoints/vn_assets.py`
  - Keep existing asset router, but include it under canonical `/api/v1/vn`.
  - Add multipart idempotency and cleanup blocker behavior.
- `tldw_Server_API/app/api/v1/endpoints/vn_play.py`
  - Keep existing play router, but include it under canonical `/api/v1/vn`.
  - Add story start, scripted story endpoints, save slots, and action request hardening.
- `tldw_Server_API/app/api/v1/schemas/vn_asset_schemas.py`
  - Add canonical response/error fields only where needed.
- `tldw_Server_API/app/api/v1/schemas/vn_play_schemas.py`
  - Extend modes from `freeform|story` to include `scripted_story`.
  - Add script runtime request/response schemas.
- `tldw_Server_API/app/core/DB_Management/VNAssetPacks_DB.py`
  - Add cleanup reference checks and upload idempotency metadata if not better isolated elsewhere.
- `tldw_Server_API/app/core/DB_Management/VNPlay_DB.py`
  - Extend sessions/action requests/save slots/scripted state.
- `tldw_Server_API/app/core/VN_Assets/service.py`
  - Add cleanup blockers and upload idempotency behavior.
- `tldw_Server_API/app/core/VN_Play/service.py`
  - Add action request recovery, story start, scripted story orchestration, save slots.
- `tldw_Server_API/app/core/VN_Play/setup_options.py`
  - Include published scripts and canonical setup warnings.
- `tldw_Server_API/app/core/VN_Play/gates.py`
  - Extend safety metadata and policy gate behavior.
- `tldw_Server_API/app/core/VN_Play/state.py`
  - Add spoiler-safe public script state support.
- `tldw_Server_API/app/core/VN_Play/branch_navigation.py`
  - Keep branch navigation as derived read model.
- `tldw_Server_API/app/core/VN_Assets/portability/*`
  - Keep import/export public flow; only update paths and idempotency rules.
- `tldw_Server_API/app/services/vn_asset_jobs_worker.py`
  - Keep generation worker behavior.
- `tldw_Server_API/app/services/startup_content_jobs_pollers.py`
  - Add VN audio worker only if Task 6 introduces a separate Jobs worker.

### New Backend Files

- `tldw_Server_API/app/core/VN_Platform/__init__.py`
  - Shared VN platform exports.
- `tldw_Server_API/app/core/VN_Platform/errors.py`
  - Stable VN error codes and helper to build FastAPI-compatible `detail` objects.
- `tldw_Server_API/app/core/VN_Platform/idempotency.py`
  - Canonical JSON/form payload hashing, multipart file hash helpers, idempotency conflict utilities.
- `tldw_Server_API/app/core/VN_Platform/capabilities.py`
  - Build `GET /vn-capabilities` response from route/config/profile state.
- `tldw_Server_API/app/api/v1/schemas/vn_common_schemas.py`
  - Shared error/warning/pagination/job response models.
- `tldw_Server_API/app/api/v1/schemas/vn_capabilities_schemas.py`
  - Capabilities response schema.
- `tldw_Server_API/app/api/v1/endpoints/vn_capabilities.py`
  - `GET /api/v1/vn/vn-capabilities`.
- `tldw_Server_API/app/core/DB_Management/VNPolicy_DB.py`
  - Admin/global policy and generation profile definitions plus immutable snapshot helpers.
- `tldw_Server_API/app/core/VN_Policy/__init__.py`
- `tldw_Server_API/app/core/VN_Policy/service.py`
  - Policy evaluation and generation profile resolution.
- `tldw_Server_API/app/api/v1/schemas/vn_policy_schemas.py`
- `tldw_Server_API/app/api/v1/endpoints/vn_policy.py`
  - `/api/v1/vn/vn-policy` endpoints.
- `tldw_Server_API/app/core/DB_Management/VNScripts_DB.py`
  - Per-user script metadata, drafts, versions, manifest snapshots, profile snapshots.
- `tldw_Server_API/app/core/VN_Scripts/__init__.py`
- `tldw_Server_API/app/core/VN_Scripts/models.py`
  - Internal typed script models where Pydantic API schemas are not enough.
- `tldw_Server_API/app/core/VN_Scripts/validator.py`
  - Canonical JSON opcode validator.
- `tldw_Server_API/app/core/VN_Scripts/interpreter.py`
  - Script cursor execution until stop point.
- `tldw_Server_API/app/core/VN_Scripts/service.py`
  - Script CRUD, draft save, validation, publish, snapshot orchestration.
- `tldw_Server_API/app/api/v1/schemas/vn_script_schemas.py`
- `tldw_Server_API/app/api/v1/endpoints/vn_scripts.py`
  - `/api/v1/vn/vn-scripts` endpoints.
- `tldw_Server_API/app/core/DB_Management/VNAudio_DB.py`
  - Per-user VN TTS job/output metadata.
- `tldw_Server_API/app/core/VN_Audio/__init__.py`
- `tldw_Server_API/app/core/VN_Audio/service.py`
  - TTS job creation/status/output metadata and generated-file registration.
- `tldw_Server_API/app/core/VN_Audio/jobs.py`
  - VN TTS Jobs payload helpers if TTS generation is asynchronous.
- `tldw_Server_API/app/services/vn_audio_jobs_worker.py`
  - Optional worker entrypoint if Task 6 uses a distinct Jobs domain.
- `tldw_Server_API/app/api/v1/schemas/vn_audio_schemas.py`
- `tldw_Server_API/app/api/v1/endpoints/vn_audio.py`
  - `/api/v1/vn/vn-audio` endpoints.

### Tests

- Create `tldw_Server_API/tests/VN_Platform/`
  - `test_vn_platform_errors.py`
  - `test_vn_platform_idempotency.py`
  - `test_vn_capabilities_api.py`
  - `test_vn_route_namespace.py`
- Extend `tldw_Server_API/tests/VN_Assets/`
  - `test_vn_assets_api.py`
  - `test_storage_cleanup.py`
  - `test_portability_api.py`
- Create `tldw_Server_API/tests/VN_Policy/`
  - `test_vn_policy_db.py`
  - `test_vn_policy_service.py`
  - `test_vn_policy_api.py`
- Create `tldw_Server_API/tests/VN_Scripts/`
  - `test_vn_scripts_db.py`
  - `test_vn_script_validator.py`
  - `test_vn_scripts_api.py`
  - `test_vn_script_publish_snapshots.py`
  - `test_vn_script_interpreter.py`
- Extend `tldw_Server_API/tests/VN_Play/`
  - `test_vn_play_db.py`
  - `test_vn_play_turns.py`
  - `test_vn_play_api.py`
  - add `test_vn_play_scripted_story.py`
  - add `test_vn_play_save_slots.py`
  - add `test_vn_play_action_requests.py`
- Create `tldw_Server_API/tests/VN_Audio/`
  - `test_vn_audio_db.py`
  - `test_vn_audio_service.py`
  - `test_vn_audio_api.py`
- Extend cross-module tests:
  - `tldw_Server_API/tests/Services/test_router_groups_contract.py`
  - `tldw_Server_API/tests/Services/test_openapi_contracts.py`
  - `tldw_Server_API/tests/Services/test_startup_content_jobs_pollers.py` if a worker is added.

### Docs

- Modify `Docs/API-related/VN_ASSET_PACKS_API.md`
- Modify `Docs/API-related/VN_PLAY_API.md`
- Create `Docs/API-related/VN_PLATFORM_API.md`
- Keep `Docs/superpowers/specs/2026-05-10-vn-platform-api-design.md` as source spec; update only for design corrections.

## Preflight

- [ ] **Step 1: Confirm branch and clean state**

Run:

```bash
git status --short --branch
test -f Docs/superpowers/specs/2026-05-10-vn-platform-api-design.md
```

Expected: implementation branch is based on current `dev`; worktree has no unrelated changes.

- [ ] **Step 2: Create implementation Backlog task**

Use Backlog.md before code edits. Suggested title:

```text
Implement VN platform API slice N
```

Include references:

```text
Docs/superpowers/specs/2026-05-10-vn-platform-api-design.md
Docs/superpowers/plans/2026-05-10-vn-platform-api-implementation-plan.md
https://github.com/rmusser01/tldw_server/issues/1391
https://github.com/rmusser01/tldw_server/issues/1486
```

Expected: task is `In Progress` before file edits.

- [ ] **Step 3: Run focused baseline**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/VN_Assets \
  tldw_Server_API/tests/VN_Play \
  tldw_Server_API/tests/Services/test_router_groups_contract.py \
  tldw_Server_API/tests/Services/test_openapi_contracts.py -q
```

Expected: current VN and route/OpenAPI baseline is understood before changes. If failures exist on `dev`, record them in the task notes before editing.

## Task 1: Platform Shell And Canonical Namespace

**Files:**
- Create: `tldw_Server_API/app/core/VN_Platform/errors.py`
- Create: `tldw_Server_API/app/core/VN_Platform/idempotency.py`
- Create: `tldw_Server_API/app/core/VN_Platform/capabilities.py`
- Create: `tldw_Server_API/app/api/v1/schemas/vn_common_schemas.py`
- Create: `tldw_Server_API/app/api/v1/schemas/vn_capabilities_schemas.py`
- Create: `tldw_Server_API/app/api/v1/endpoints/vn_capabilities.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/content.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/vn_assets.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/vn_play.py`
- Modify: `apps/tldw-frontend/lib/api/vnAssets.ts`
- Modify: `apps/tldw-frontend/lib/api/vnPlay.ts`
- Test: `apps/tldw-frontend/__tests__/vn-assets/vnAssetsApi.test.ts`
- Test: `apps/tldw-frontend/__tests__/vn-play/vnPlayApi.test.ts`
- Test: `tldw_Server_API/tests/VN_Platform/test_vn_platform_errors.py`
- Test: `tldw_Server_API/tests/VN_Platform/test_vn_platform_idempotency.py`
- Test: `tldw_Server_API/tests/VN_Platform/test_vn_capabilities_api.py`
- Test: `tldw_Server_API/tests/VN_Platform/test_vn_route_namespace.py`
- Test: `tldw_Server_API/tests/Services/test_router_groups_contract.py`
- Test: `tldw_Server_API/tests/Services/test_openapi_contracts.py`

- [ ] **Step 1: Write failing namespace and capability tests**

Add tests that include the canonical routers through `content.iter_content_router_specs()` and assert:

```python
assert "/api/v1/vn/vn-capabilities" in paths
assert "/api/v1/vn/vn-assets/packs" in paths
assert "/api/v1/vn/vn-play/sessions" in paths
assert "/api/v1/vn-assets/packs" not in paths
assert "/api/v1/vn-play/sessions" not in paths
```

Add an API test:

```python
def test_vn_capabilities_returns_canonical_paths(client, api_headers):
    response = client.get("/api/v1/vn/vn-capabilities", headers=api_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "vn_capabilities.v1"
    assert body["base_path"] == "/api/v1/vn"
    assert body["resources"]["assets"] == "/api/v1/vn/vn-assets"
    assert body["resources"]["scripts"] == "/api/v1/vn/vn-scripts"
    assert body["resources"]["play"] == "/api/v1/vn/vn-play"
    assert body["resources"]["policy"] == "/api/v1/vn/vn-policy"
    assert body["resources"]["audio"] == "/api/v1/vn/vn-audio"
    assert body["features"]["realtime_image_generation"] is False
    assert "visible_policy_profiles" in body
    assert "visible_generation_profiles" in body
    assert "supported_media_types" in body
    assert body["route_migration"]["canonical"] == "/api/v1/vn/vn-*"
```

- [ ] **Step 2: Run failing shell tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/VN_Platform/test_vn_capabilities_api.py \
  tldw_Server_API/tests/VN_Platform/test_vn_route_namespace.py -q
```

Expected: tests fail because canonical routers/capabilities do not exist yet.

- [ ] **Step 3: Add stable VN error and idempotency helpers**

In `errors.py`, define stable code constants and a helper:

```python
def vn_error_detail(
    code: str,
    message: str,
    *,
    details: dict[str, object] | None = None,
    retryable: bool = False,
) -> dict[str, object]:
    return {
        "code": code,
        "message": message,
        "details": details or {},
        "retryable": retryable,
    }
```

In `idempotency.py`, implement:

```python
def canonical_payload_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
```

Add multipart hash helper that accepts canonical form fields plus a streaming file digest.

- [ ] **Step 4: Add capabilities service and endpoint**

Build `VNCapabilitiesResponse` from static V1 feature flags plus configured
limits, visible policy profiles, visible generation profiles, supported image/TTS
media types, route migration metadata, and docs/OpenAPI links. Enabled resources
must be derived from registered routes/configured modules so a partial slice does
not advertise unavailable policy/script/audio APIs as enabled. Keep the endpoint
read-only and side-effect-free.

- [ ] **Step 5: Move VN route registration under `/api/v1/vn`**

In `content.py`, include the existing `vn_assets` and `vn_play` routers with `prefix=f"{API_V1_PREFIX}/vn"` while their router prefixes remain `/vn-assets` and `/vn-play`.

Do not register old `/api/v1/vn-assets` and `/api/v1/vn-play` aliases in the target API.

- [ ] **Step 6: Update bundled API path constants**

Update only the existing bundled client path constants and API-client tests so
the WebUI calls the same backend-owned canonical contract as custom frontends:

- `apps/tldw-frontend/lib/api/vnAssets.ts`
- `apps/tldw-frontend/lib/api/vnPlay.ts`
- `apps/tldw-frontend/__tests__/vn-assets/vnAssetsApi.test.ts`
- `apps/tldw-frontend/__tests__/vn-play/vnPlayApi.test.ts`

Do not move setup eligibility, readiness, or policy logic client-side.

- [ ] **Step 7: Run shell tests and route/OpenAPI contracts**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/VN_Platform/test_vn_platform_errors.py \
  tldw_Server_API/tests/VN_Platform/test_vn_platform_idempotency.py \
  tldw_Server_API/tests/VN_Platform/test_vn_capabilities_api.py \
  tldw_Server_API/tests/VN_Platform/test_vn_route_namespace.py \
  tldw_Server_API/tests/Services/test_router_groups_contract.py \
  tldw_Server_API/tests/Services/test_openapi_contracts.py -q
bunx vitest run \
  apps/tldw-frontend/__tests__/vn-assets/vnAssetsApi.test.ts \
  apps/tldw-frontend/__tests__/vn-play/vnPlayApi.test.ts
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit**

```bash
git add \
  tldw_Server_API/app/core/VN_Platform \
  tldw_Server_API/app/api/v1/schemas/vn_common_schemas.py \
  tldw_Server_API/app/api/v1/schemas/vn_capabilities_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/vn_capabilities.py \
  tldw_Server_API/app/api/v1/router_groups/content.py \
  tldw_Server_API/tests/VN_Platform \
  tldw_Server_API/tests/Services/test_router_groups_contract.py \
  tldw_Server_API/tests/Services/test_openapi_contracts.py \
  apps/tldw-frontend/lib/api/vnAssets.ts \
  apps/tldw-frontend/lib/api/vnPlay.ts \
  apps/tldw-frontend/__tests__/vn-assets/vnAssetsApi.test.ts \
  apps/tldw-frontend/__tests__/vn-play/vnPlayApi.test.ts
git commit -m "Add VN platform API shell"
```

## Task 2: Canonical VN Assets API Migration And Safety Hardening

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/vn_assets.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/vn_asset_schemas.py`
- Modify: `tldw_Server_API/app/core/DB_Management/VNAssetPacks_DB.py`
- Modify: `tldw_Server_API/app/core/VN_Assets/service.py`
- Modify: `tldw_Server_API/app/core/VN_Assets/storage.py`
- Modify: `tldw_Server_API/app/core/VN_Assets/portability/preview.py`
- Modify: `Docs/API-related/VN_ASSET_PACKS_API.md`
- Test: `tldw_Server_API/tests/VN_Assets/test_vn_assets_api.py`
- Test: `tldw_Server_API/tests/VN_Assets/test_storage_cleanup.py`
- Test: `tldw_Server_API/tests/VN_Assets/test_portability_api.py`

- [ ] **Step 1: Write failing tests for canonical asset paths**

Update API tests to call `/api/v1/vn/vn-assets/...` only. Add a negative route test asserting old `/api/v1/vn-assets/packs` returns 404 when route flags use the target API.

- [ ] **Step 2: Write failing idempotency tests for all asset mutators**

Cover same-payload replay and same-key/different-payload conflict for:

- `POST /packs/{pack_id}/generate`
- `POST /packs/{pack_id}/cleanup` when execution is requested
- `POST /packs/{pack_id}/export`
- `POST /import/previews`
- `POST /import/commit`
- `POST /packs/{pack_id}/slots/{slot_id}/retry`
- `POST /packs/{pack_id}/items/{item_id}/regenerate`
- `POST /packs/{pack_id}/items/upload`

- [ ] **Step 3: Write failing content/preview validation tests**

Cover `GET /items/{item_id}/preview` and `GET /items/{item_id}/content` denying:

- wrong owner or cross-user pack/item access;
- generated-file records whose `source_feature` is not `vn_assets`;
- generated-file records whose `source_ref` does not match the item;
- non-image or disallowed media types;
- policy-blocked item access.

- [ ] **Step 4: Write failing multipart idempotency tests**

Add tests for `POST /api/v1/vn/vn-assets/packs/{pack_id}/items/upload`:

```python
def test_item_upload_replays_same_idempotency_key(client, api_headers, pack_and_slot, png_bytes):
    first = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack_and_slot.pack_id}/items/upload",
        headers=api_headers,
        data={"slot_id": str(pack_and_slot.slot_id), "idempotency_key": "upload-1"},
        files={"file": ("sprite.png", png_bytes, "image/png")},
    )
    second = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack_and_slot.pack_id}/items/upload",
        headers=api_headers,
        data={"slot_id": str(pack_and_slot.slot_id), "idempotency_key": "upload-1"},
        files={"file": ("sprite.png", png_bytes, "image/png")},
    )
    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]
```

Add conflict coverage with changed bytes under the same key expecting `409 idempotency_key_conflict`.

- [ ] **Step 5: Write failing cleanup blocker tests**

In `test_storage_cleanup.py`, create item records referenced by:

- a manifest snapshot row from `VNScripts_DB.py` when available;
- a `vn_play_sessions` row;
- a checkpoint/save slot row after Task 5;
- a branch restore target or branch event lineage row after Task 5;
- a persisted VN audio output row after Task 6.

For early Task 2 implementation, make the cleanup service accept a pluggable blocker provider so tests can verify blocker behavior before script/audio tables exist.

Expected dry-run result includes `blocked_count` and stable `cleanup_blocked` details. Confirmed cleanup skips blocked files.

- [ ] **Step 6: Add asset idempotency persistence**

Prefer a small VN idempotency table in `VNAssetPacks_DB.py` or reuse an existing operation table if it already captures operation, key, payload hash, and response ID.

Minimum fields:

- `owner_user_id`
- `scope`
- `resource_id`
- `idempotency_key`
- `payload_hash`
- `result_type`
- `result_id`
- `response_json`
- timestamps

- [ ] **Step 7: Add cleanup blocker provider**

Implement blocker checks in `VN_Assets/service.py` with an interface that can query scripts/play/audio modules when present. Keep Task 2 safe before later modules land by supporting no-op providers.

- [ ] **Step 8: Run focused asset tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/VN_Assets/test_vn_assets_api.py \
  tldw_Server_API/tests/VN_Assets/test_storage_cleanup.py \
  tldw_Server_API/tests/VN_Assets/test_portability_api.py -q
```

Expected: selected tests pass.

- [ ] **Step 9: Commit**

```bash
git add \
  tldw_Server_API/app/api/v1/endpoints/vn_assets.py \
  tldw_Server_API/app/api/v1/schemas/vn_asset_schemas.py \
  tldw_Server_API/app/core/DB_Management/VNAssetPacks_DB.py \
  tldw_Server_API/app/core/VN_Assets \
  tldw_Server_API/tests/VN_Assets \
  Docs/API-related/VN_ASSET_PACKS_API.md
git commit -m "Migrate VN assets to platform API"
```

## Task 3: VN Policy And Generation Profiles

**Files:**
- Create: `tldw_Server_API/app/core/DB_Management/VNPolicy_DB.py`
- Create: `tldw_Server_API/app/core/VN_Policy/__init__.py`
- Create: `tldw_Server_API/app/core/VN_Policy/service.py`
- Create: `tldw_Server_API/app/api/v1/schemas/vn_policy_schemas.py`
- Create: `tldw_Server_API/app/api/v1/endpoints/vn_policy.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/content.py`
- Modify: `tldw_Server_API/app/core/VN_Play/gates.py`
- Test: `tldw_Server_API/tests/VN_Policy/test_vn_policy_db.py`
- Test: `tldw_Server_API/tests/VN_Policy/test_vn_policy_service.py`
- Test: `tldw_Server_API/tests/VN_Policy/test_vn_policy_api.py`

- [ ] **Step 1: Write failing profile repository tests**

Cover:

- built-in local default profile exists after initialization;
- admin profile CRUD stores versioned definitions;
- snapshot creation copies effective settings and does not change after profile update.

Add generation-profile schema/service tests for required fields:

- provider/model routing;
- structured-output capability;
- temperature and token defaults/bounds;
- allowed content ratings;
- max choices and branch depth;
- max model expansion scope;
- TTS permission;
- output persistence bounds;
- audit/logging mode.

Assert invalid bounds and unsupported combinations fail validation before a
profile can be used by scripts, sessions, or VN audio.

- [ ] **Step 2: Write failing safety metadata tests**

Use service-level tests for:

```python
("general", "missing", "local_default") -> decision warn, requires_acknowledgement True
("general", "unknown_or_ambiguous", "local_default") -> decision warn, requires_acknowledgement True
("mature", "missing", "local_default") -> decision block
("mature", "unknown_or_ambiguous", "local_default") -> decision block
("general", "conflicting", "local_default") -> decision block
("general", "imported_untrusted", "local_default") -> decision warn, requires_acknowledgement True
("general", "missing", "strict_hosted") -> decision block
("general", "unknown_or_ambiguous", "strict_hosted") -> decision block
("general", "conflicting", "strict_hosted") -> decision block
("general", "imported_untrusted", "strict_hosted") -> decision block
```

- [ ] **Step 3: Write failing admin/RBAC and route tests**

Cover:

- normal users can list/read usable profiles but cannot create, patch, or delete
  policy/generation profiles;
- admin users can create, patch, and disable profiles;
- cross-user evaluation does not expose another user's script, pack, session, or
  audio metadata;
- `/api/v1/vn/vn-policy` paths appear in OpenAPI after router registration;
- list endpoints expose offset pagination fields.

- [ ] **Step 4: Implement policy DB and service**

Use admin/global storage consistent with existing AuthNZ/admin infrastructure. If a global config DB abstraction is not available, keep the first implementation config-backed plus snapshot rows in per-user `ChaChaNotes.db` for user-owned scripts/sessions. Do not store user-owned script/session snapshots only in the central AuthNZ DB.

- [ ] **Step 5: Add API schemas and endpoints**

Implement:

- `POST /api/v1/vn/vn-policy/evaluate`
- `GET /profiles`
- `GET /profiles/{profile_id}`
- admin `POST/PATCH/DELETE /profiles`
- `GET /generation-profiles`
- `GET /generation-profiles/{profile_id}`
- admin `POST/PATCH/DELETE /generation-profiles`

Use stable VN error detail objects.

- [ ] **Step 6: Register policy router and assert OpenAPI paths**

Register `vn_policy.router` under `prefix=f"{API_V1_PREFIX}/vn"` in
`content.py`. Add route/OpenAPI assertions for `/api/v1/vn/vn-policy/evaluate`,
`/profiles`, and `/generation-profiles`.

- [ ] **Step 7: Run policy tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/VN_Policy -q
```

Expected: all VN policy tests pass.

- [ ] **Step 8: Commit**

```bash
git add \
  tldw_Server_API/app/core/DB_Management/VNPolicy_DB.py \
  tldw_Server_API/app/core/VN_Policy \
  tldw_Server_API/app/api/v1/schemas/vn_policy_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/vn_policy.py \
  tldw_Server_API/app/api/v1/router_groups/content.py \
  tldw_Server_API/tests/VN_Policy
git commit -m "Add VN policy profiles API"
```

## Task 4: VN Scripts Draft, Validation, And Publish API

**Files:**
- Create: `tldw_Server_API/app/core/DB_Management/VNScripts_DB.py`
- Create: `tldw_Server_API/app/core/VN_Scripts/__init__.py`
- Create: `tldw_Server_API/app/core/VN_Scripts/models.py`
- Create: `tldw_Server_API/app/core/VN_Scripts/validator.py`
- Create: `tldw_Server_API/app/core/VN_Scripts/service.py`
- Create: `tldw_Server_API/app/api/v1/schemas/vn_script_schemas.py`
- Create: `tldw_Server_API/app/api/v1/endpoints/vn_scripts.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/content.py`
- Modify: `tldw_Server_API/app/core/VN_Assets/manifest.py` only if manifest snapshot construction needs a pure helper.
- Test: `tldw_Server_API/tests/VN_Scripts/test_vn_scripts_db.py`
- Test: `tldw_Server_API/tests/VN_Scripts/test_vn_script_validator.py`
- Test: `tldw_Server_API/tests/VN_Scripts/test_vn_scripts_api.py`
- Test: `tldw_Server_API/tests/VN_Scripts/test_vn_script_publish_snapshots.py`

- [ ] **Step 1: Write failing DB tests**

Assert scripts tables are created in per-user `ChaChaNotes.db`:

- `vn_scripts`
- `vn_script_drafts`
- `vn_script_versions`
- `vn_script_manifest_snapshots`
- `vn_profile_snapshots`

Test draft optimistic revision conflict with `if_revision`.

- [ ] **Step 2: Write failing validator tests**

Cover:

- entry label exists;
- missing jump/choice target errors;
- typed variable assignment errors;
- structured condition operand errors;
- visual slot key missing from approved manifest;
- BGM/SFX/voice media references are inaccessible or wrong media type;
- model generation settings not allowed by selected generation profile;
- warning for unreachable labels.

Add publish idempotency tests for same-key replay and same-key/different-payload
conflict.

Add publish policy tests proving `POST /publish` repeats authoritative policy
evaluation and blocks/warns independently of prior advisory `/vn-policy/evaluate`
responses.

- [ ] **Step 3: Implement repository and validator**

Keep validation pure where possible. The service should load manifest/profile context and pass it into the pure validator.

Do not accept text DSL as source truth in V1.

- [ ] **Step 4: Implement script API**

Implement:

- `POST /api/v1/vn/vn-scripts/scripts`
- `GET /scripts`
- `GET/PATCH/DELETE /scripts/{script_id}`
- `GET /scripts/{script_id}/draft`
- `PUT /scripts/{script_id}/draft`
- `POST /scripts/{script_id}/draft/validate`
- `GET /scripts/{script_id}/draft/diagnostics`
- `POST /scripts/{script_id}/publish`
- version read/list/manifest-snapshot endpoints
- `POST /scripts/{script_id}/versions/{version_id}/policy/evaluate`.

Publish must snapshot:

- approved asset manifest;
- effective policy profile;
- effective generation profile;
- script defaults.

- [ ] **Step 5: Register scripts router and assert OpenAPI paths**

Register `vn_scripts.router` under `prefix=f"{API_V1_PREFIX}/vn"` in
`content.py`. Add route/OpenAPI assertions for draft, validation, publish,
version, manifest snapshot, and version policy-evaluate endpoints.

- [ ] **Step 6: Run script tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/VN_Scripts -q
```

Expected: all VN scripts tests pass.

- [ ] **Step 7: Commit**

```bash
git add \
  tldw_Server_API/app/core/DB_Management/VNScripts_DB.py \
  tldw_Server_API/app/core/VN_Scripts \
  tldw_Server_API/app/api/v1/schemas/vn_script_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/vn_scripts.py \
  tldw_Server_API/app/api/v1/router_groups/content.py \
  tldw_Server_API/tests/VN_Scripts
git commit -m "Add VN scripts authoring API"
```

## Task 5: Scripted Story Runtime And Shared Runtime Hardening

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/VNPlay_DB.py`
- Modify: `tldw_Server_API/app/core/VN_Play/constants.py`
- Modify: `tldw_Server_API/app/core/VN_Play/models.py`
- Modify: `tldw_Server_API/app/core/VN_Play/service.py`
- Modify: `tldw_Server_API/app/core/VN_Play/state.py`
- Modify: `tldw_Server_API/app/core/VN_Play/gates.py`
- Modify: `tldw_Server_API/app/core/VN_Play/setup_options.py`
- Create or extend: `tldw_Server_API/app/core/VN_Scripts/interpreter.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/vn_play_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/vn_play.py`
- Test: `tldw_Server_API/tests/VN_Play/test_vn_play_action_requests.py`
- Test: `tldw_Server_API/tests/VN_Play/test_vn_play_save_slots.py`
- Test: `tldw_Server_API/tests/VN_Play/test_vn_play_scripted_story.py`
- Test: `tldw_Server_API/tests/VN_Play/test_vn_play_api.py`
- Test: `tldw_Server_API/tests/VN_Play/test_vn_play_turns.py`

- [ ] **Step 1: Write failing action request recovery tests**

Cover:

- stale `client_scene_version` rejects before execution;
- duplicate in-flight key replays active status or returns `turn_in_progress`;
- abandoned lease becomes `action_request_abandoned`;
- completed key replays stored response;
- failed key replays stable failure payload.

- [ ] **Step 2: Write failing save-slot tests**

Add `vn_play_save_slots` table and endpoint tests for create/list/read/patch/delete/restore. Save slots are per-session and reference checkpoint semantics. Cover idempotency replay/conflict for save-slot creation and restore.

- [ ] **Step 3: Write failing setup-options tests**

Extend `GET /api/v1/vn/vn-play/setup-options` tests to include:

- published script versions;
- script/version readiness;
- policy warnings and acknowledgement requirements;
- default policy/generation profiles;
- empty states for no ready scripts, no ready packs, and filtered results.

The backend owns these setup rules; do not move eligibility or policy logic into
frontend clients.

- [ ] **Step 4: Write failing scripted story tests**

Create a minimal published script version and assert:

- `POST /sessions` accepts `mode="scripted_story"` with `script_version_id`;
- session pins script version and manifest/profile snapshots;
- session creation repeats authoritative policy evaluation and blocks/warns even
  if a previous advisory `/vn-policy/evaluate` response was stale;
- `POST /script/advance` runs until visible choice;
- `POST /script/choices/{choice_id}` advances target branch;
- `GET /script/state` is spoiler-safe;
- `GET /script/debug-state` exposes raw label/cursor only to owner/admin.

Add replay tests proving persisted model-generated narration/dialogue/choices are
returned on replay, and seeded random opcode results are deterministic and stored
in runtime events.

- [ ] **Step 5: Extend VN Play schema and repository**

Add:

- `script_version_id`
- `script_position_json`
- `policy_snapshot_id`
- `generation_profile_snapshot_id`
- generalized `vn_play_action_requests` fields if existing `vn_play_session_actions` is insufficient;
- `vn_play_save_slots`.

Preserve existing `freeform` and `story` behavior.

- [ ] **Step 6: Implement story start endpoint**

Add `POST /api/v1/vn/vn-play/sessions/{session_id}/story/start` as a stable backend command instead of private `custom_action` startup.

- [ ] **Step 7: Implement setup-options script support**

Extend `setup_options.py` so custom frontends can discover characters, ready
asset packs, published script versions, policy warnings, defaults, and empty
states from one backend-owned contract.

- [ ] **Step 8: Implement script runtime endpoints**

Add:

- `POST /sessions/{session_id}/script/advance`
- `POST /sessions/{session_id}/script/choices/{choice_id}`
- `POST /sessions/{session_id}/script/regenerate`
- `GET /sessions/{session_id}/script/state`
- `GET /sessions/{session_id}/script/debug-state`

Keep model calls synchronous with persisted failure state. Regeneration must fork
or create a new lineage and must not rewrite historical events. Persist model
expansions and seeded random results as events so replay is deterministic.

- [ ] **Step 9: Run VN Play tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/VN_Play -q
```

Expected: all VN Play tests pass.

- [ ] **Step 10: Commit**

```bash
git add \
  tldw_Server_API/app/core/DB_Management/VNPlay_DB.py \
  tldw_Server_API/app/core/VN_Play \
  tldw_Server_API/app/core/VN_Scripts/interpreter.py \
  tldw_Server_API/app/api/v1/schemas/vn_play_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/vn_play.py \
  tldw_Server_API/tests/VN_Play
git commit -m "Add scripted VN play runtime"
```

## Task 6: VN Audio TTS API

**Files:**
- Create: `tldw_Server_API/app/core/DB_Management/VNAudio_DB.py`
- Create: `tldw_Server_API/app/core/VN_Audio/__init__.py`
- Create: `tldw_Server_API/app/core/VN_Audio/service.py`
- Create: `tldw_Server_API/app/core/VN_Audio/jobs.py`
- Create: `tldw_Server_API/app/api/v1/schemas/vn_audio_schemas.py`
- Create: `tldw_Server_API/app/api/v1/endpoints/vn_audio.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/content.py`
- Modify: `tldw_Server_API/app/services/startup_content_jobs_pollers.py`
- Create: `tldw_Server_API/app/services/vn_audio_jobs_worker.py`
- Test: `tldw_Server_API/tests/VN_Audio/test_vn_audio_db.py`
- Test: `tldw_Server_API/tests/VN_Audio/test_vn_audio_service.py`
- Test: `tldw_Server_API/tests/VN_Audio/test_vn_audio_api.py`

- [ ] **Step 1: Write failing VN audio DB/API tests**

Cover:

- create TTS job with `scope="script_pregen"`;
- idempotency replay and same-key/different-payload conflict;
- `job_id` is returned and Generic Jobs owns queued/processing/completed/failed
  lifecycle state;
- status read/list;
- cancel queued job;
- persisted output metadata creates generated-file record with VN metadata;
- transient output has expiry and authenticated content serving rules.

Add preview/content tests denying:

- wrong owner or cross-user output access;
- generated-file records whose source feature is not VN audio;
- generated-file records whose source ref does not match the output row;
- non-audio or disallowed media types;
- policy-blocked access after output creation.

- [ ] **Step 2: Implement VN audio metadata repository**

Store job/output metadata in per-user `ChaChaNotes.db`.

Use generated-file storage for persisted audio bytes. Do not store audio bytes in `ChaChaNotes.db`.

- [ ] **Step 3: Implement TTS job service**

Prefer wrapping existing TTS service/provider stack. Keep provider/model access constrained by generation profile policy.

VN TTS generation is always Jobs-backed in V1. Create a `vn_audio` Jobs domain,
return the Generic Jobs `job_id`, and keep VN audio rows for VN-specific stage,
scope, output, and line-reference metadata. Reads reconcile VN metadata with the
Jobs lifecycle state.

- [ ] **Step 4: Implement VN audio endpoints**

Add:

- `POST /api/v1/vn/vn-audio/tts/jobs`
- `GET /tts/jobs`
- `GET /tts/jobs/{job_id}`
- `POST /tts/jobs/{job_id}/cancel`
- `GET /tts/outputs/{output_id}`
- `GET /tts/outputs/{output_id}/preview`
- `GET /tts/outputs/{output_id}/content`
- `DELETE /tts/outputs/{output_id}`

- [ ] **Step 5: Register audio router and assert OpenAPI paths**

Register `vn_audio.router` under `prefix=f"{API_V1_PREFIX}/vn"` in `content.py`.
Add route/OpenAPI assertions for `/api/v1/vn/vn-audio/tts/jobs` and output
preview/content endpoints.

- [ ] **Step 6: Run VN audio tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/VN_Audio -q
```

Expected: all VN audio tests pass.

- [ ] **Step 7: Commit**

```bash
git add \
  tldw_Server_API/app/core/DB_Management/VNAudio_DB.py \
  tldw_Server_API/app/core/VN_Audio \
  tldw_Server_API/app/api/v1/schemas/vn_audio_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/vn_audio.py \
  tldw_Server_API/app/api/v1/router_groups/content.py \
  tldw_Server_API/tests/VN_Audio
git commit -m "Add VN audio TTS API"
```

## Task 7: Docs, OpenAPI, Migration, And Final Verification

**Files:**
- Create: `Docs/API-related/VN_PLATFORM_API.md`
- Modify: `Docs/API-related/VN_ASSET_PACKS_API.md`
- Modify: `Docs/API-related/VN_PLAY_API.md`
- Modify: `tldw_Server_API/tests/Services/test_openapi_contracts.py`
- Modify: `tldw_Server_API/tests/Services/test_router_groups_contract.py`
- Modify frontend API path constants only if existing tests require route migration:
  - `apps/tldw-frontend/lib/api/vnAssets.ts`
  - `apps/tldw-frontend/lib/api/vnPlay.ts`
  - `apps/tldw-frontend/types/vn-assets.ts`
  - `apps/tldw-frontend/types/vn-play.ts`

- [ ] **Step 1: Write final OpenAPI assertions**

Assert OpenAPI includes:

- `/api/v1/vn/vn-capabilities`
- `/api/v1/vn/vn-assets/...`
- `/api/v1/vn/vn-scripts/...`
- `/api/v1/vn/vn-play/...`
- `/api/v1/vn/vn-policy/...`
- `/api/v1/vn/vn-audio/...`

Assert old root-level VN paths are absent unless a separate migration exception is explicitly approved.

- [ ] **Step 2: Update API docs**

Create `VN_PLATFORM_API.md` as the overview and cross-link existing docs. Keep old route migration explicit and avoid promising deprecated aliases.

- [ ] **Step 3: Run full focused backend verification**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/VN_Platform \
  tldw_Server_API/tests/VN_Assets \
  tldw_Server_API/tests/VN_Policy \
  tldw_Server_API/tests/VN_Scripts \
  tldw_Server_API/tests/VN_Play \
  tldw_Server_API/tests/VN_Audio \
  tldw_Server_API/tests/Services/test_router_groups_contract.py \
  tldw_Server_API/tests/Services/test_openapi_contracts.py -q
```

Expected: selected backend suite passes.

- [ ] **Step 4: Run Bandit on touched backend scope**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r \
  tldw_Server_API/app/api/v1/endpoints/vn_assets.py \
  tldw_Server_API/app/api/v1/endpoints/vn_play.py \
  tldw_Server_API/app/api/v1/endpoints/vn_capabilities.py \
  tldw_Server_API/app/api/v1/endpoints/vn_policy.py \
  tldw_Server_API/app/api/v1/endpoints/vn_scripts.py \
  tldw_Server_API/app/api/v1/endpoints/vn_audio.py \
  tldw_Server_API/app/core/VN_Platform \
  tldw_Server_API/app/core/VN_Policy \
  tldw_Server_API/app/core/VN_Scripts \
  tldw_Server_API/app/core/VN_Audio \
  tldw_Server_API/app/core/VN_Assets \
  tldw_Server_API/app/core/VN_Play \
  -f json -o /tmp/bandit_vn_platform_api.json
```

Expected: no new findings in touched production code. If tests are scanned separately, use `-s B101` only for test files.

- [ ] **Step 5: Run final diff checks**

Run:

```bash
git diff --check
git status --short --branch
```

Expected: no whitespace errors; branch only contains intended changes.

- [ ] **Step 6: Commit docs and verification closeout**

```bash
git add \
  Docs/API-related/VN_PLATFORM_API.md \
  Docs/API-related/VN_ASSET_PACKS_API.md \
  Docs/API-related/VN_PLAY_API.md \
  tldw_Server_API/tests/Services/test_openapi_contracts.py \
  tldw_Server_API/tests/Services/test_router_groups_contract.py
git commit -m "Document VN platform API"
```

## Cross-Cutting Implementation Notes

- Keep per-user VN metadata in `ChaChaNotes.db`.
- Keep generated image/audio bytes in AuthNZ generated-file storage.
- Keep old `/api/v1/vn-assets` and `/api/v1/vn-play` aliases out of the target API unless explicitly approved.
- Use stable VN `detail` objects for new endpoints; avoid mixing string-only details into new VN routes.
- Use idempotency for every mutating endpoint that creates work, publishes, uploads, or advances/restores runtime state.
- Do not call LLM/TTS providers before idempotency, policy, ownership, and scene-version checks pass.
- Do not physically delete generated files referenced by published manifests, sessions, checkpoints, save slots, branch targets, or persisted audio output metadata.
- Do not expose raw script label/cursor/interpreter internals in public runtime state.
- Do not create new Jobs for interactive runtime model calls; use durable action requests instead.

## Final Review Checklist

- [ ] Canonical namespace is `/api/v1/vn/vn-*`.
- [ ] Capabilities endpoint reflects the actually enabled VN modules.
- [ ] Route/OpenAPI tests prove old paths are not target API aliases.
- [ ] Asset generation, portability, cleanup, and VN audio use Jobs where expected.
- [ ] Runtime model calls are synchronous and recoverable through action-request records.
- [ ] Script versions and sessions snapshot effective profile config.
- [ ] Character safety metadata behavior is deterministic per policy profile.
- [ ] Public script state is spoiler-safe.
- [ ] Bandit and focused pytest verification are recorded in Backlog task notes before PR.
