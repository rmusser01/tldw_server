# llama.cpp Managed Runtime Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first backend-owned llama.cpp managed runtime slice: durable instance profiles, multi-instance supervisor state, admin APIs, V1 default-profile compatibility, and minimal WebUI/client boundaries.

**Approved Spec:** `Docs/superpowers/specs/2026-05-16-llamacpp-managed-runtime-roadmap-design.md`

**Architecture:** Keep `[LlamaCpp]` config as global bootstrap state and add a dedicated profile store plus supervisor service under `tldw_Server_API/app/core/Local_LLM/`. The supervisor owns per-profile process runners and runtime state, while existing V1 endpoints become wrappers around a reserved default profile. This plan deliberately defers remote downloads, full asset inventory v2, mmproj pairing UI, and model-family routing to follow-up plans.

**Tech Stack:** FastAPI, Pydantic v2, asyncio subprocess management, existing llama.cpp handler utilities, pytest/TestClient, Ant Design/React shared WebUI, Vitest, Bandit.

---

## Scope Check

The roadmap spec spans several independent subsystems: runtime supervision,
asset inventory v2, mmproj/model-family metadata, full Admin console UX, and
future download/catalog jobs. This plan covers Stage 1 only, plus the smallest
Stage 2 hook needed to make supervisor lifecycle explicit. Follow-up plans
should cover:

- Asset Inventory V2: assets, folder import, mmproj pairing, stale-path state.
- Model-Family Modes: chat/vision/embedding/rerank metadata and routing.
- Admin Console UX: full readiness/assets/profiles/runtime redesign.
- Downloads/Catalogs: remote acquisition jobs and trust/disk policy.

Stage 1 success is backend-first: multiple durable profiles can be created,
started on distinct ports, inspected, stopped independently, and accessed
through explicit admin APIs. The old single-server endpoints still work through
the default profile.

## File Structure

Create:

- `tldw_Server_API/app/core/Local_LLM/llamacpp_runtime_models.py`
  - Internal enums and dataclasses/Pydantic helpers for profile state, runtime
    state, port policy, restart policy, and health status.
- `tldw_Server_API/app/core/Local_LLM/llamacpp_profile_store.py`
  - Backend-owned profile persistence. Use
    `setup_manager.get_config_file_path().with_name("llamacpp_profiles.json")`
    as the default JSON path for this slice, with repository-shaped methods so
    it can move to SQLite later.
- `tldw_Server_API/app/core/Local_LLM/llamacpp_process_runner.py`
  - One-process runner extracted from the current `LlamaCppHandler` launch,
    stop, log, and status behavior. It must not register atexit handlers per
    instance.
- `tldw_Server_API/app/core/Local_LLM/llamacpp_supervisor_service.py`
  - Profile/running-process coordinator with per-profile locks, explicit port
    validation, manual start/stop/pause/resume, and shutdown cleanup.
- `tldw_Server_API/tests/LLM_Local/test_llamacpp_profile_store.py`
- `tldw_Server_API/tests/LLM_Local/test_llamacpp_process_runner.py`
- `tldw_Server_API/tests/LLM_Local/test_llamacpp_supervisor_service.py`
- `tldw_Server_API/tests/LLM_Local/test_llamacpp_runtime_api.py`
- `apps/packages/ui/src/components/Option/Admin/LlamacppRuntimePanel.tsx`
- `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppRuntimePanel.test.tsx`

Modify:

- `tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py`
  - Add API schemas for profile CRUD, runtime status, lifecycle actions, and
    profile-scoped log tails.
- `tldw_Server_API/app/api/v1/endpoints/llamacpp.py`
  - Add new profile/runtime endpoints and route V1 wrappers through the
    supervisor default profile.
- `tldw_Server_API/app/core/Local_LLM/LlamaCpp_Handler.py`
  - Preserve public behavior; delegate shared process command construction and
    cleanup to `llamacpp_process_runner` only if that extraction can be done
    without widening the patch. Otherwise leave this file mostly unchanged and
    duplicate only minimal runner code in the new runner with a follow-up TODO
    linked to TASK-397.
- `tldw_Server_API/app/core/Local_LLM/LLM_Inference_Manager.py`
  - Attach `llamacpp_supervisor` when llama.cpp config is enabled and call
    supervisor shutdown cleanup from `cleanup_on_exit`.
- `tldw_Server_API/app/services/lifespan_startup_sequence.py` or the existing
  service that constructs `LLMInferenceManager`
  - Only modify if the manager construction path needs app-state publication
    for the supervisor.
- `apps/packages/ui/src/types/llamacpp-admin.ts`
  - Add profile/runtime API types.
- `apps/packages/ui/src/services/tldw/domains/models-audio.ts`
  - Add client methods for profile/runtime endpoints.
- `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
  - Add or re-export matching client methods if this file still mirrors the
    domain client.
- `apps/packages/ui/src/components/Option/Admin/LlamacppAdminPage.tsx`
  - Render a minimal runtime panel behind the existing setup/inventory/launch
    flow. Do not replace the full page in this slice.

Do not modify:

- Remote download flows.
- `/api/v1/llm/models/metadata` routing.
- Knowledge settings.
- Chat image attachment behavior.
- Full mmproj pairing UI.

## Shared Implementation Rules

- Use admin-only dependencies for every new endpoint:
  `dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))]`.
- Preserve warnings-first resource behavior.
- Hard-fail duplicate enabled explicit host/port combinations.
- Resolve symlinks before allowlist checks.
- Do not expose arbitrary log paths.
- Do not silently wire Chat on start.
- Keep V1 response shapes compatible where existing tests assert them.
- Use `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python`
  in worktrees when `.venv` is absent.

## Task 1: Runtime Schemas And Profile Store

**Files:**

- Create: `tldw_Server_API/app/core/Local_LLM/llamacpp_runtime_models.py`
- Create: `tldw_Server_API/app/core/Local_LLM/llamacpp_profile_store.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py`
- Test: `tldw_Server_API/tests/LLM_Local/test_llamacpp_profile_store.py`

- [ ] **Step 1: Write failing profile-store tests**

Add tests for:

```python
def test_profile_store_bootstraps_default_profile_from_config(tmp_path):
    store = JsonLlamaCppProfileStore(tmp_path / "profiles.json")
    profile = store.ensure_default_profile(
        model_id="gguf:abc",
        server_args={"port": 8080},
    )
    assert profile.profile_id == "default"
    assert profile.name == "Default llama.cpp server"
    assert profile.port_policy == "explicit"


def test_profile_store_rejects_duplicate_enabled_explicit_ports(tmp_path):
    store = JsonLlamaCppProfileStore(tmp_path / "profiles.json")
    store.upsert(profile("one", host="127.0.0.1", port=8181, enabled=True))
    with pytest.raises(LlamaCppProfileConflictError, match="host/port"):
        store.upsert(profile("two", host="127.0.0.1", port=8181, enabled=True))
```

Define a local `profile()` test helper that returns a valid
`LlamaCppProfile`. Avoid relying on implementation defaults for fields that are
central to a test assertion.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_profile_store.py -v
```

Expected: FAIL because store/models do not exist.

- [ ] **Step 3: Add runtime models**

Implement minimal models:

```python
class LlamaCppProfileMode(str, Enum):
    CHAT = "chat"
    VISION = "vision"
    EMBEDDING = "embedding"
    RERANK = "rerank"
    SERVER_GENERIC = "server_generic"

class LlamaCppPortPolicy(str, Enum):
    EXPLICIT = "explicit"
    AUTOSELECT = "autoselect"

class LlamaCppProfile(BaseModel):
    profile_id: str
    name: str
    enabled: bool = True
    mode: LlamaCppProfileMode = LlamaCppProfileMode.CHAT
    model_id: str | None = None
    model_path: str | None = None
    mmproj_model_id: str | None = None
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    port_policy: LlamaCppPortPolicy = LlamaCppPortPolicy.EXPLICIT
    server_args: dict[str, object] = Field(default_factory=dict)
    autostart: bool = False
    restart_policy: dict[str, object] = Field(default_factory=dict)
    provider_alias: str | None = None
    tags: list[str] = Field(default_factory=list)
```

Also add `LlamaCppRuntimeState` with states `defined`, `starting`, `running`,
`stopped`, `failed`, and `paused`.

Add narrow exception classes in the same module:

```python
class LlamaCppProfileStoreError(RuntimeError): ...
class LlamaCppProfileNotFoundError(LlamaCppProfileStoreError): ...
class LlamaCppProfileConflictError(LlamaCppProfileStoreError): ...
```

- [ ] **Step 4: Add JSON profile store**

Implement:

```python
class JsonLlamaCppProfileStore:
    def __init__(self, path: Path): ...
    def list_profiles(self) -> list[LlamaCppProfile]: ...
    def get(self, profile_id: str) -> LlamaCppProfile | None: ...
    def upsert(self, profile: LlamaCppProfile) -> LlamaCppProfile: ...
    def delete(self, profile_id: str) -> bool: ...
    def ensure_default_profile(...) -> LlamaCppProfile: ...
```

Use atomic write through a temporary sibling file and `Path.replace()`. Validate
duplicate enabled explicit host/port combinations before writing.

Expose:

```python
def default_profile_store_path() -> Path:
    return setup_manager.get_config_file_path().expanduser().resolve().with_name("llamacpp_profiles.json")
```

- [ ] **Step 5: Add API schemas**

In `llamacpp_admin_schemas.py`, add request/response schemas that mirror the
internal model without exposing store internals:

- `LlamaCppProfileCreateRequest`
- `LlamaCppProfileUpdateRequest`
- `LlamaCppProfileResponse`
- `LlamaCppProfileListResponse`
- `LlamaCppRuntimeResponse`
- `LlamaCppRuntimeListResponse`
- `LlamaCppLifecycleActionResponse`

- [ ] **Step 6: Run profile-store tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_profile_store.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add \
  tldw_Server_API/app/core/Local_LLM/llamacpp_runtime_models.py \
  tldw_Server_API/app/core/Local_LLM/llamacpp_profile_store.py \
  tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_profile_store.py
git commit -m "feat: add llama.cpp profile store"
```

## Task 2: Process Runner For One Managed Instance

**Files:**

- Create: `tldw_Server_API/app/core/Local_LLM/llamacpp_process_runner.py`
- Modify: `tldw_Server_API/app/core/Local_LLM/LlamaCpp_Handler.py`
- Test: `tldw_Server_API/tests/LLM_Local/test_llamacpp_process_runner.py`
- Existing reference: `tldw_Server_API/app/core/Local_LLM/handler_utils.py`

- [ ] **Step 1: Write failing runner tests**

Test command construction, path checks, port policy, and independent stop:

```python
async def test_runner_starts_without_stopping_other_runner(tmp_path, monkeypatch):
    first = LlamaCppProcessRunner(config, profile_id="one")
    second = LlamaCppProcessRunner(config, profile_id="two")
    await first.start(model_path, profile=profile("one", port=8181))
    await second.start(model_path, profile=profile("two", port=8182))
    assert first.runtime.port == 8181
    assert second.runtime.port == 8182
```

Patch `asyncio.create_subprocess_exec` and `wait_for_http_ready` so no real
llama-server binary is required.

- [ ] **Step 2: Run runner tests to verify failure**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_process_runner.py -v
```

Expected: FAIL because the runner module does not exist.

- [ ] **Step 3: Implement runner**

Create `LlamaCppProcessRunner` with:

- `start(model_path: Path, profile: LlamaCppProfile) -> LlamaCppRuntime`
- `stop() -> LlamaCppRuntime`
- `status() -> LlamaCppRuntime`
- `tail_logs(lines: int) -> dict[str, object]`
- `cleanup_sync() -> None`

Start by moving only reusable, low-risk logic from `LlamaCppHandler`:

- path allowlist checks
- denylist check
- host/client-host normalization
- explicit/autoselect port resolution
- command redaction
- subprocess group creation
- bounded shutdown
- log handle ownership

Do not register signal handlers or atexit hooks in the runner.

- [ ] **Step 4: Keep existing handler compatible**

Either:

- make `LlamaCppHandler` delegate its current single-server methods to one
  default runner, or
- leave `LlamaCppHandler` untouched and keep the runner code separate.

Prefer delegation only if tests remain focused and the patch stays readable.
Preserve all existing `LlamaCppHandler` public methods.

- [ ] **Step 5: Run old and new runner tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_process_runner.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_management_api.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add \
  tldw_Server_API/app/core/Local_LLM/llamacpp_process_runner.py \
  tldw_Server_API/app/core/Local_LLM/LlamaCpp_Handler.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_process_runner.py
git commit -m "feat: add llama.cpp process runner"
```

## Task 3: Supervisor Service

**Files:**

- Create: `tldw_Server_API/app/core/Local_LLM/llamacpp_supervisor_service.py`
- Modify: `tldw_Server_API/app/core/Local_LLM/LLM_Inference_Manager.py`
- Test: `tldw_Server_API/tests/LLM_Local/test_llamacpp_supervisor_service.py`

- [x] **Step 1: Write failing supervisor tests**

Cover:

```python
async def test_supervisor_starts_two_profiles_on_distinct_ports(...):
    await supervisor.start_profile("one")
    await supervisor.start_profile("two")
    states = supervisor.list_runtimes()
    assert {state.profile_id for state in states} == {"one", "two"}

async def test_supervisor_rejects_duplicate_enabled_explicit_port(...):
    with pytest.raises(LlamaCppProfileConflictError):
        store.upsert(profile("two", port=8181, enabled=True))

async def test_supervisor_serializes_same_profile_start(...):
    await asyncio.gather(supervisor.start_profile("one"), supervisor.start_profile("one"))
    assert runner_factory.calls["one"] == 1
```

- [x] **Step 2: Run tests to verify failure**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_supervisor_service.py -v
```

Expected: FAIL because supervisor does not exist.

- [x] **Step 3: Implement supervisor**

Implement:

```python
class LlamaCppSupervisor:
    def list_profiles(self) -> list[LlamaCppProfile]: ...
    async def create_profile(self, request: LlamaCppProfileCreateRequest) -> LlamaCppProfile: ...
    async def update_profile(self, profile_id: str, request: LlamaCppProfileUpdateRequest) -> LlamaCppProfile: ...
    async def delete_profile(self, profile_id: str) -> bool: ...
    async def start_profile(self, profile_id: str) -> LlamaCppRuntime: ...
    async def stop_profile(self, profile_id: str, disable: bool = False) -> LlamaCppRuntime: ...
    async def pause_profile(self, profile_id: str) -> LlamaCppRuntime: ...
    async def resume_profile(self, profile_id: str) -> LlamaCppRuntime: ...
    def list_runtimes(self) -> list[LlamaCppRuntime]: ...
    async def shutdown(self) -> None: ...
```

Use `dict[str, asyncio.Lock]` for per-profile locks. Validate explicit
host/port conflicts before start and before saving enabled profiles.

- [x] **Step 4: Add default profile bridge**

Add methods:

```python
async def ensure_default_profile_from_model(model_id: str, server_args: dict[str, object]) -> LlamaCppProfile: ...
async def start_default_by_model(model_id: str, server_args: dict[str, object]) -> LlamaCppRuntime: ...
async def stop_default() -> LlamaCppRuntime: ...
def default_status_compat() -> dict[str, object]: ...
```

Use `llamacpp_inventory_service.resolve_model_id()` for model path resolution.

- [x] **Step 5: Attach to manager**

In `LLM_Inference_Manager.__init__`, after `self.llamacpp` is created, attach:

```python
self.llamacpp_supervisor = LlamaCppSupervisor.from_manager(self)
```

If llama.cpp is disabled, set `self.llamacpp_supervisor = None`.

In `cleanup_on_exit`, call supervisor cleanup before or instead of direct
handler cleanup when supervisor exists. Because `cleanup_on_exit` is
synchronous, the supervisor needs a `cleanup_sync()` method for this path even
if it also has async shutdown for future lifespan integration.

- [x] **Step 6: Run supervisor tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_supervisor_service.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add \
  tldw_Server_API/app/core/Local_LLM/llamacpp_supervisor_service.py \
  tldw_Server_API/app/core/Local_LLM/LLM_Inference_Manager.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_supervisor_service.py
git commit -m "feat: add llama.cpp supervisor"
```

## Task 4: Admin Runtime APIs And V1 Wrappers

**Files:**

- Modify: `tldw_Server_API/app/api/v1/endpoints/llamacpp.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py`
- Test: `tldw_Server_API/tests/LLM_Local/test_llamacpp_runtime_api.py`
- Test: `tldw_Server_API/tests/LLM_Local/test_llamacpp_management_api.py`
- Test: `tldw_Server_API/tests/LLM_Local/test_llamacpp_provider_and_logs_api.py`

- [x] **Step 1: Write failing API tests**

Cover:

```python
def test_profiles_crud_requires_admin_and_returns_profile():
    response = client.post("/api/v1/llamacpp/profiles", json={...})
    assert response.status_code == 200
    assert response.json()["profile"]["name"] == "Qwen fixed port"

def test_instances_lists_two_runtimes():
    response = client.get("/api/v1/llamacpp/instances")
    assert response.status_code == 200
    assert len(response.json()["instances"]) == 2

def test_v1_start_by_model_targets_default_profile_only():
    response = client.post("/api/v1/llamacpp/start-by-model", json={...})
    assert response.json()["model_id"] == "gguf:abc"
    assert supervisor.started_profile_ids == ["default"]
```

- [x] **Step 2: Run API tests to verify failure**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_runtime_api.py -v
```

Expected: FAIL because endpoints do not exist.

- [x] **Step 3: Add supervisor resolver**

In `llamacpp.py`, add:

```python
def _resolve_llamacpp_supervisor(llm_manager: LLMInferenceManager) -> LlamaCppSupervisor:
    supervisor = getattr(llm_manager, "llamacpp_supervisor", None)
    if supervisor is None:
        raise _llamacpp_unavailable()
    return supervisor
```

- [x] **Step 4: Add profile/runtime endpoints**

Add endpoints from the spec:

- `GET /api/v1/llamacpp/profiles`
- `POST /api/v1/llamacpp/profiles`
- `GET /api/v1/llamacpp/profiles/{profile_id}`
- `PUT /api/v1/llamacpp/profiles/{profile_id}`
- `DELETE /api/v1/llamacpp/profiles/{profile_id}`
- `POST /api/v1/llamacpp/profiles/{profile_id}/start`
- `POST /api/v1/llamacpp/profiles/{profile_id}/stop`
- `POST /api/v1/llamacpp/profiles/{profile_id}/pause`
- `POST /api/v1/llamacpp/profiles/{profile_id}/resume`
- `POST /api/v1/llamacpp/profiles/{profile_id}/use-in-chat`
- `GET /api/v1/llamacpp/instances`
- `GET /api/v1/llamacpp/instances/{profile_id}`
- `GET /api/v1/llamacpp/instances/{profile_id}/logs/tail`

Map invalid user input to `400`, missing profile to `404`, resource conflicts
to `409`, unavailable llama.cpp manager to `503`, and unexpected failures to
sanitized `500`.

- [x] **Step 5: Route V1 wrappers through default profile**

Change these to prefer supervisor when present:

- `start_llamacpp_by_model_endpoint`
- `start_llamacpp_server_endpoint`
- `stop_llamacpp_server_endpoint`
- `get_llamacpp_status_endpoint`
- `tail_llamacpp_logs_endpoint`
- `use_llamacpp_in_chat_endpoint`
- `run_llamacpp_inference_endpoint`

Keep manager/handler fallback for tests and compatibility.

- [x] **Step 6: Run API and compatibility tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_runtime_api.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_management_api.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_provider_and_logs_api.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_lifecycle_api_contract.py -v
```

Expected: PASS.

Task 4 review fix coverage:

- profile-scoped `use-in-chat`
- V1 split-brain protection when supervisor and legacy handler both exist
- `start-by-model` followed by V1 inference on the supervisor default profile
- fresh `stop_server` idempotency when no default profile exists

- [ ] **Step 7: Commit**

```bash
git add \
  tldw_Server_API/app/api/v1/endpoints/llamacpp.py \
  tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_runtime_api.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_management_api.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_provider_and_logs_api.py
git commit -m "feat: expose llama.cpp runtime APIs"
```

## Task 5: Minimal WebUI Client And Runtime Panel

**Files:**

- Modify: `apps/packages/ui/src/types/llamacpp-admin.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/models-audio.ts`
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Create: `apps/packages/ui/src/components/Option/Admin/LlamacppRuntimePanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/Admin/LlamacppAdminPage.tsx`
- Test: `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppRuntimePanel.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx`

- [ ] **Step 1: Write failing frontend tests**

Test that the runtime panel shows multiple instances and actions:

```tsx
it("renders multiple llama.cpp runtimes with independent actions", async () => {
  render(<LlamacppRuntimePanel runtimes={[runningOne, stoppedTwo]} onStop={fn} onStart={fn} />)
  expect(screen.getByText("Qwen fixed port")).toBeInTheDocument()
  expect(screen.getByText("8181")).toBeInTheDocument()
  expect(screen.getByText("Stopped")).toBeInTheDocument()
})
```

- [ ] **Step 2: Run frontend tests to verify failure**

```bash
bunx vitest run \
  apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppRuntimePanel.test.tsx \
  apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx
```

Expected: FAIL because panel/types do not exist.

- [ ] **Step 3: Add TypeScript types and client methods**

Add:

- `LlamacppProfile`
- `LlamacppRuntime`
- `LlamacppProfileListResponse`
- `LlamacppRuntimeListResponse`
- `LlamacppLifecycleActionResponse`

Client methods:

- `listLlamacppProfiles`
- `createLlamacppProfile`
- `updateLlamacppProfile`
- `deleteLlamacppProfile`
- `startLlamacppProfile`
- `stopLlamacppProfile`
- `pauseLlamacppProfile`
- `resumeLlamacppProfile`
- `listLlamacppInstances`
- `tailLlamacppInstanceLogs`

- [ ] **Step 4: Add minimal runtime panel**

Keep it operational and dense:

- runtime table/list
- profile name
- state
- endpoint
- pid
- restart count
- warnings
- start/stop/pause/resume buttons
- no remote download controls
- no full profile editor yet

- [ ] **Step 5: Wire panel into Admin page**

Load `listLlamacppInstances()` alongside status/config/inventory. Render the
panel under the existing readiness/status band. If the new endpoint fails with
404/503 on older servers, degrade to the existing single-server view.

- [ ] **Step 6: Run frontend tests**

```bash
bunx vitest run \
  apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppRuntimePanel.test.tsx \
  apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx \
  apps/packages/ui/src/services/__tests__/tldw-api-client.models-normalization.test.ts
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add \
  apps/packages/ui/src/types/llamacpp-admin.ts \
  apps/packages/ui/src/services/tldw/domains/models-audio.ts \
  apps/packages/ui/src/services/tldw/TldwApiClient.ts \
  apps/packages/ui/src/components/Option/Admin/LlamacppRuntimePanel.tsx \
  apps/packages/ui/src/components/Option/Admin/LlamacppAdminPage.tsx \
  apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppRuntimePanel.test.tsx \
  apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx
git commit -m "feat: add llama.cpp runtime panel"
```

## Task 6: Focused Verification And Security Pass

**Files:**

- Modify only files touched by earlier tasks if verification requires fixes.
- Update `backlog/tasks/task-397.1 - Plan-llama.cpp-managed-runtime-implementation.md` or follow-up implementation task records with final verification notes.

- [ ] **Step 1: Run focused backend tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_profile_store.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_process_runner.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_supervisor_service.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_runtime_api.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_admin_config_api.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_management_api.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_provider_and_logs_api.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_lifecycle_api_contract.py -v
```

Expected: PASS.

- [ ] **Step 2: Run focused frontend tests**

```bash
bunx vitest run \
  apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppRuntimePanel.test.tsx \
  apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx \
  apps/packages/ui/src/services/__tests__/model-settings.llamacpp-controls.test.ts \
  apps/packages/ui/src/utils/__tests__/build-llamacpp-server-args.test.ts
```

Expected: PASS.

- [ ] **Step 3: Run Bandit on touched backend paths**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit \
  -r tldw_Server_API/app/core/Local_LLM tldw_Server_API/app/api/v1/endpoints/llamacpp.py \
  -f json -o /tmp/bandit_llamacpp_runtime_stage1.json
```

Expected: no new high/medium findings in touched code. If Bandit reports
subprocess usage, verify commands use argument lists and no shell; add
`# nosec` only with a narrow explanation when the finding is expected.

- [ ] **Step 4: Run diff checks**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors. Status should show only intentional files.

- [ ] **Step 5: Commit verification notes**

If verification fixes changed files, commit them:

```bash
git add <fixed-files>
git commit -m "test: verify llama.cpp runtime stage 1"
```

Otherwise do not create an empty commit.

## Follow-Up Plan Boundaries

Create separate plans after Stage 1 lands:

1. `llamacpp-asset-inventory-v2`
   - `LlamaCppAsset`
   - folder import/register
   - mmproj discovery and pairing
   - stale path warnings
2. `llamacpp-model-family-metadata`
   - profile modes in `/api/v1/llm/models/metadata`
   - chat/vision/embedding/rerank capability routing
3. `llamacpp-admin-console-v2`
   - full readiness/assets/profiles/runtime layout
   - profile editor
   - option browser from `llama-server --help`
4. `llamacpp-download-acquisition-jobs`
   - direct URL/source downloads
   - cancellation/retry/checksum/disk warnings
   - atomic asset registration

Do not start these follow-ups inside the Stage 1 implementation branch unless
the user explicitly expands scope.
