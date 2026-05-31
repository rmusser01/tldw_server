# llama.cpp Server Management WebUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the approved single-server llama.cpp WebUI management flow: guided config, validation, inventory, start-by-model, warnings-first hardware guidance, explicit chat provider wiring, and safe log access.

**Architecture:** Keep the existing `LlamaCppHandler` as the process authority and add a narrow admin facade for config, inventory, validation, provider wiring, and logs. The backend owns config/path/argument safety; the frontend renders a guided console against typed responses and preserves the existing advanced launch controls. Model inventory introduces stable `model_id` values so the UI never passes arbitrary absolute paths to the start endpoint.

**Tech Stack:** FastAPI, Pydantic, existing setup/config helpers, existing `LlamaCppHandler`, pytest/httpx TestClient, Next.js/React, Ant Design, shared `tldwClient`, Vitest/testing-library, Playwright smoke where practical.

---

## References

- Spec: `Docs/superpowers/specs/2026-05-15-llamacpp-server-management-webui-design.md`
- Backlog: `TASK-365`
- Existing backend endpoints: `tldw_Server_API/app/api/v1/endpoints/llamacpp.py`
- Existing handler: `tldw_Server_API/app/core/Local_LLM/LlamaCpp_Handler.py`
- Existing config loader: `tldw_Server_API/app/core/config.py`
- Existing comment-preserving config writer: `tldw_Server_API/app/core/Setup/setup_manager.py`
- Existing WebUI page: `apps/packages/ui/src/components/Option/Admin/LlamacppAdminPage.tsx`
- Existing frontend API client: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Existing domain client mirror: `apps/packages/ui/src/services/tldw/domains/models-audio.ts`

## Scope Guardrails

- Keep V1 to one managed llama.cpp server process.
- Do not add downloads, uploads, registry browsing, or model import from remote URLs.
- Do not silently change chat behavior during `start`.
- Do not add server-side launch profile storage in required V1.
- Do not let parsed `llama-server --help` output bypass backend allowlists.
- Do not serve arbitrary files through logs or path registration.
- Warnings should guide users; hardware estimates must not hard-block launch.

## File Map

### Backend

- Create: `tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py`
  - Pydantic contracts for config, validation, inventory, hardware, start-by-model, provider wiring, and log tail.
- Create: `tldw_Server_API/app/core/Local_LLM/llamacpp_config_service.py`
  - Saved/active config facade, env override detection, restart-required calculation, typed config updates, binary validation.
- Create: `tldw_Server_API/app/core/Local_LLM/llamacpp_inventory_service.py`
  - Bounded recursive GGUF scan, registered path parsing, stable model IDs, model ID resolution.
- Create: `tldw_Server_API/app/core/Local_LLM/llamacpp_hardware_service.py`
  - Best-effort RAM/CPU/GPU snapshot and warning generation.
- Create: `tldw_Server_API/app/core/Local_LLM/llamacpp_provider_service.py`
  - Explicit `use-in-chat` provider wiring for `Local-API.llama_api_IP`.
- Modify: `tldw_Server_API/app/core/Local_LLM/LlamaCpp_Handler.py`
  - Add safe start-by-path helper while preserving existing `start_server(model_filename, ...)`.
  - Return current active state fields needed by the admin facade.
- Modify: `tldw_Server_API/app/api/v1/endpoints/llamacpp.py`
  - Add new facade endpoints and keep existing endpoints compatible.
- Modify: `tldw_Server_API/app/core/config.py`
  - Teach config parsing about optional `registered_model_paths` if that key is introduced.
- Modify: `tldw_Server_API/Config_Files/config.txt`
  - Add `registered_model_paths =` under `[LlamaCpp]` so `setup_manager.update_config()` can safely persist path registration.

### Backend Tests

- Create: `tldw_Server_API/tests/LLM_Local/test_llamacpp_admin_config_api.py`
- Create: `tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py`
- Create: `tldw_Server_API/tests/LLM_Local/test_llamacpp_provider_and_logs_api.py`
- Modify: `tldw_Server_API/tests/Local_LLM/test_llamacpp_handler.py`
- Modify: `tldw_Server_API/tests/Local_LLM/test_llamacpp_hardening.py`
- Modify: `tldw_Server_API/tests/AuthNZ_Unit/test_llamacpp_permissions_claims.py`

### Frontend

- Create: `apps/packages/ui/src/types/llamacpp-admin.ts`
  - Shared TypeScript response/request types for the admin console.
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
  - Add facade client methods.
- Modify: `apps/packages/ui/src/services/tldw/domains/models-audio.ts`
  - Mirror facade client methods if this domain file remains the split client source.
- Modify: `apps/packages/ui/src/services/tldw/client-ownership.ts`
  - Add ownership entries for new client methods.
- Modify: `apps/packages/ui/src/components/Option/Admin/LlamacppAdminPage.tsx`
  - Reshape into readiness, inventory, and launch areas.
- Create: `apps/packages/ui/src/components/Option/Admin/LlamacppReadinessPanel.tsx`
- Create: `apps/packages/ui/src/components/Option/Admin/LlamacppInventoryPanel.tsx`
- Create: `apps/packages/ui/src/components/Option/Admin/LlamacppLaunchPanel.tsx`
  - Keep `LlamacppAdminPage.tsx` as orchestration and state loading; keep panels focused and testable.

### Frontend Tests

- Modify: `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx`
- Create: `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppReadinessPanel.test.tsx`
- Create: `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppInventoryPanel.test.tsx`
- Create: `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppLaunchPanel.test.tsx`
- Modify: `apps/packages/ui/src/utils/__tests__/build-llamacpp-server-args.test.ts` only if launch settings mapping changes.
- Modify: `apps/tldw-frontend/e2e/workflows/tier-4-admin/admin-llamacpp.spec.ts`

### Docs

- Modify: `Docs/API-related/llamacpp_integration_modes.md`
- Modify: `Docs/Published/API-related/llamacpp_integration_modes.md`

---

## Task 1: Backend Config Facade and Validation

**Files:**
- Create: `tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py`
- Create: `tldw_Server_API/app/core/Local_LLM/llamacpp_config_service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/llamacpp.py`
- Modify: `tldw_Server_API/Config_Files/config.txt`
- Test: `tldw_Server_API/tests/LLM_Local/test_llamacpp_admin_config_api.py`

- [ ] **Step 1: Write failing tests for saved vs active config**

Add tests that build a FastAPI app around `llamacpp.router`, monkeypatch config-service helpers, and assert the endpoint shape.

```python
@pytest.mark.unit
def test_llamacpp_config_reports_saved_active_and_restart_required(monkeypatch):
    app = _make_app_with_manager(_ManagerWithoutHandler())
    monkeypatch.setattr(
        lp.llamacpp_config_service,
        "get_config_state",
        lambda llm_manager: {
            "saved_config": {"enabled": True, "models_dir": "models/gguf_models"},
            "active_config": {"handler_configured": False},
            "restart_required": True,
            "restart_reasons": ["handler_not_configured"],
            "env_overrides": {"models_dir": False},
        },
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/llamacpp/config")

    assert response.status_code == 200
    body = response.json()
    assert body["restart_required"] is True
    assert body["active_config"]["handler_configured"] is False
```

- [ ] **Step 2: Run the failing config test**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/LLM_Local/test_llamacpp_admin_config_api.py -q
```

Expected: FAIL because `/api/v1/llamacpp/config` does not exist.

- [ ] **Step 3: Add Pydantic schemas**

Create `llamacpp_admin_schemas.py` with typed request/response models. Keep the fields permissive enough for current config while avoiding `dict[str, Any]` at the endpoint boundary except for warning metadata.

Key model skeleton:

```python
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LlamaCppSavedConfig(BaseModel):
    enabled: bool = False
    executable_path: str | None = None
    models_dir: str | None = None
    default_host: str | None = None
    default_port: int | None = None
    default_threads: int | None = None
    default_n_gpu_layers: int | None = None
    default_ctx_size: int | None = None
    allow_unvalidated_args: bool | None = None
    allow_cli_secrets: bool | None = None
    port_autoselect: bool | None = None
    port_probe_max: int | None = None
    allowed_paths: list[str] = Field(default_factory=list)
    registered_model_paths: list[str] = Field(default_factory=list)
    log_output_file: str | None = None


class LlamaCppActiveConfig(BaseModel):
    handler_configured: bool
    enabled: bool | None = None
    executable_path: str | None = None
    models_dir: str | None = None
    default_host: str | None = None
    default_port: int | None = None
    active_model: str | None = None
    active_host: str | None = None
    active_port: int | None = None
    active_pid: int | None = None


class LlamaCppConfigResponse(BaseModel):
    saved_config: LlamaCppSavedConfig
    active_config: LlamaCppActiveConfig
    restart_required: bool
    restart_reasons: list[str] = Field(default_factory=list)
    env_overrides: dict[str, bool] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class LlamaCppConfigUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    executable_path: str | None = None
    models_dir: str | None = None
    default_host: str | None = None
    default_port: int | None = Field(default=None, ge=1, le=65535)
    default_threads: int | None = Field(default=None, ge=1)
    default_n_gpu_layers: int | None = None
    default_ctx_size: int | None = Field(default=None, ge=1)
    port_autoselect: bool | None = None
    port_probe_max: int | None = Field(default=None, ge=0)
    allowed_paths: list[str] | None = None
    log_output_file: str | None = None
```

- [ ] **Step 4: Implement config service read helpers**

In `llamacpp_config_service.py`, read saved config from `load_comprehensive_config()` and active config from `llm_manager.llamacpp.config` if present.

Implement small pure helpers first:

```python
LLAMACPP_ENV_OVERRIDES = {
    "enabled": "LLAMACPP_ENABLED",
    "executable_path": "LLAMACPP_EXECUTABLE_PATH",
    "models_dir": "LLAMACPP_MODELS_DIR",
    "default_host": "LLAMACPP_HOST",
    "default_port": "LLAMACPP_PORT",
    "default_threads": "LLAMACPP_THREADS",
    "default_n_gpu_layers": "LLAMACPP_N_GPU_LAYERS",
    "default_ctx_size": "LLAMACPP_CTX_SIZE",
    "port_autoselect": "LLAMACPP_PORT_AUTOSELECT",
    "port_probe_max": "LLAMACPP_PORT_PROBE_MAX",
    "allowed_paths": "LLAMACPP_ALLOWED_PATHS",
    "log_output_file": "LLAMACPP_LOG_OUTPUT_FILE",
}

RESTART_FIELDS = {
    "enabled",
    "executable_path",
    "models_dir",
    "allowed_paths",
    "log_output_file",
}
```

`get_config_state(llm_manager)` should return:

- saved config parsed from `[LlamaCpp]`;
- active config from handler config and active process state;
- `restart_required=True` if saved handler fields differ from active handler fields or saved `enabled=true` but no handler exists;
- env override booleans for each known field.

- [ ] **Step 5: Implement typed config update**

Add `update_config_state(payload, llm_manager)` that:

- drops `None` fields;
- rejects writes for env-overridden fields with `409 Conflict` and a response body that identifies the locked fields;
- calls `setup_manager.update_config({"LlamaCpp": updates})`;
- calls `refresh_config_cache()`;
- returns `get_config_state(llm_manager)`.

Do not use `ConfigParser.write()`.

- [ ] **Step 6: Add config endpoints**

In `llamacpp.py`, import the schema module and config service. Add:

```python
@router.get(
    "/llamacpp/config",
    response_model=LlamaCppConfigResponse,
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def get_llamacpp_config_endpoint(
    llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager),
):
    return llamacpp_config_service.get_config_state(llm_manager)
```

Add `PUT /llamacpp/config` with `LlamaCppConfigUpdateRequest`.

- [ ] **Step 7: Add binary validation tests**

Test `POST /api/v1/llamacpp/validate` for:

- missing binary path;
- existing executable that returns help/version text;
- invalid executable path does not leak host paths in client-facing errors.

- [ ] **Step 8: Implement validation endpoint**

Add `LlamaCppValidationRequest` and `LlamaCppValidationResponse`. The validation service should:

- resolve the binary path;
- check file exists and is executable where the platform supports that check;
- run a bounded `--help` or `--version` subprocess with timeout;
- return command support as warnings, not a process-start attempt.

Use `asyncio.create_subprocess_exec` or `subprocess.run` through `asyncio.to_thread`; keep timeout low.

- [ ] **Step 9: Run focused backend tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_admin_config_api.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_management_api.py \
  -q
```

Expected: PASS.

- [ ] **Step 10: Commit Task 1**

```bash
git add \
  tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py \
  tldw_Server_API/app/core/Local_LLM/llamacpp_config_service.py \
  tldw_Server_API/app/api/v1/endpoints/llamacpp.py \
  tldw_Server_API/Config_Files/config.txt \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_admin_config_api.py
git commit -m "feat: add llama.cpp admin config facade"
```

---

## Task 2: Inventory Resolver and Start-by-Model

**Files:**
- Create: `tldw_Server_API/app/core/Local_LLM/llamacpp_inventory_service.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py`
- Modify: `tldw_Server_API/app/core/Local_LLM/LlamaCpp_Handler.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/llamacpp.py`
- Modify: `tldw_Server_API/app/core/config.py`
- Test: `tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py`
- Test: `tldw_Server_API/tests/Local_LLM/test_llamacpp_handler.py`
- Test: `tldw_Server_API/tests/Local_LLM/test_llamacpp_hardening.py`

- [ ] **Step 1: Write failing inventory tests**

Create tests for:

- recursive scan returns nested `*.gguf`;
- `mmproj*.gguf` files are skipped;
- unreadable/outside paths return item warnings rather than failing whole inventory;
- model IDs are stable for the same canonical path.

Example assertion:

```python
assert [item["basename"] for item in body["models"]] == ["nested-model.gguf"]
assert body["models"][0]["model_id"].startswith("gguf:")
assert body["models"][0]["source"] == "models_dir"
```

- [ ] **Step 2: Run inventory tests to verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py -q
```

Expected: FAIL because inventory endpoint/service does not exist.

- [ ] **Step 3: Implement inventory schemas**

Add these models to `llamacpp_admin_schemas.py`:

```python
class LlamaCppModelMetadata(BaseModel):
    quantization: str | None = None
    parameter_hint: str | None = None
    context_hint: int | None = None


class LlamaCppInventoryItem(BaseModel):
    model_id: str
    display_name: str
    basename: str
    source: str
    path: str
    size_bytes: int | None = None
    modified_at: str | None = None
    metadata: LlamaCppModelMetadata = Field(default_factory=LlamaCppModelMetadata)
    warnings: list[str] = Field(default_factory=list)


class LlamaCppInventoryResponse(BaseModel):
    models: list[LlamaCppInventoryItem]
    warnings: list[str] = Field(default_factory=list)
    scan_limited: bool = False
```

- [ ] **Step 4: Implement inventory service**

Implement:

- `scan_inventory(config_state, limit=500) -> LlamaCppInventoryResponse`;
- `register_model_path(path: Path) -> LlamaCppInventoryItem`;
- `resolve_model_id(model_id: str) -> Path`.

Use a deterministic ID based on canonical resolved path:

```python
def model_id_for_path(path: Path) -> str:
    canonical = str(path.expanduser().resolve())
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
    return f"gguf:{digest}"
```

For V1 persistence, store registered local paths in `[LlamaCpp] registered_model_paths` as a comma-separated value. Add the key to `config.txt` before writing through `setup_manager.update_config()`.

- [ ] **Step 5: Refactor handler start logic safely**

In `LlamaCpp_Handler.py`, preserve current `start_server(model_filename, ...)`, but move the shared launch body behind a helper that accepts a validated model path and display label.

Shape:

```python
async def start_server_by_path(
    self,
    model_path: Path,
    *,
    model_label: str | None = None,
    server_args: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    model_path = model_path.expanduser().resolve()
    if not self._is_path_allowed(model_path):
        raise ServerError("Model path must be under allowed directories.")
    if model_path.suffix.lower() != ".gguf":
        raise ServerError("Model path must reference a GGUF file.")
    if not model_path.is_file():
        raise ModelNotFoundError(f"Model file {model_path.name} was not found.")
    return await self._start_server_for_model_path(
        model_path,
        model_label=model_label or model_path.name,
        server_args=server_args,
    )
```

Then make existing `start_server()` call `start_server_by_path(self.models_dir / model_filename, model_label=model_filename, ...)`.

Do not accept absolute paths in `start_server(model_filename, ...)`.

- [ ] **Step 6: Add hardening tests for start-by-path**

Add tests that:

- `start_server_by_path` accepts allowed registered path;
- rejects non-GGUF path;
- rejects path outside allowlist;
- existing traversal rejection for `start_server("../../x.gguf")` still passes.

- [ ] **Step 7: Add inventory endpoints**

Add:

```text
GET  /api/v1/llamacpp/inventory
POST /api/v1/llamacpp/models/register-path
POST /api/v1/llamacpp/start-by-model
```

`start-by-model` body:

```python
class LlamaCppStartByModelRequest(BaseModel):
    model_id: str
    server_args: dict[str, Any] = Field(default_factory=dict)
```

Endpoint logic:

1. resolve handler with `_resolve_llamacpp_target(llm_manager, ("start_server_by_path",))`;
2. resolve `model_id` to path through inventory service;
3. call handler `start_server_by_path`;
4. return the existing start result plus `backend="llamacpp"` and `model_id`.

- [ ] **Step 8: Run focused backend tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py \
  tldw_Server_API/tests/Local_LLM/test_llamacpp_handler.py \
  tldw_Server_API/tests/Local_LLM/test_llamacpp_hardening.py \
  -q
```

Expected: PASS.

- [ ] **Step 9: Commit Task 2**

```bash
git add \
  tldw_Server_API/app/core/Local_LLM/llamacpp_inventory_service.py \
  tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py \
  tldw_Server_API/app/core/Local_LLM/LlamaCpp_Handler.py \
  tldw_Server_API/app/api/v1/endpoints/llamacpp.py \
  tldw_Server_API/app/core/config.py \
  tldw_Server_API/Config_Files/config.txt \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py \
  tldw_Server_API/tests/Local_LLM/test_llamacpp_handler.py \
  tldw_Server_API/tests/Local_LLM/test_llamacpp_hardening.py
git commit -m "feat: add llama.cpp model inventory resolver"
```

---

## Task 3: Provider Wiring, Hardware Snapshot, and Log Tail

**Files:**
- Create: `tldw_Server_API/app/core/Local_LLM/llamacpp_hardware_service.py`
- Create: `tldw_Server_API/app/core/Local_LLM/llamacpp_provider_service.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/llamacpp.py`
- Test: `tldw_Server_API/tests/LLM_Local/test_llamacpp_provider_and_logs_api.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_llamacpp_permissions_claims.py`

- [ ] **Step 1: Write failing provider wiring tests**

Tests should verify:

- `POST /api/v1/llamacpp/use-in-chat` requires a running managed process;
- endpoint writes only `Local-API.llama_api_IP`;
- endpoint calls config cache refresh;
- a future env override for the provider endpoint returns `effective=false` with a warning;
- no model field is written.

Use monkeypatches for `setup_manager.update_config` and `refresh_config_cache`.

- [ ] **Step 2: Write failing log safety tests**

Tests should verify:

- log tail reads only configured `log_output_file`;
- request cannot pass arbitrary path;
- response is bounded by requested line/byte limit;
- missing log returns empty lines with warning, not 500.

- [ ] **Step 3: Write failing hardware tests**

Tests should verify the service returns structured `unavailable` warnings when optional GPU probes are missing.

Do not require NVIDIA hardware in tests.

- [ ] **Step 4: Run failing tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/LLM_Local/test_llamacpp_provider_and_logs_api.py -q
```

Expected: FAIL because endpoints/services do not exist.

- [ ] **Step 5: Implement provider wiring service**

In `llamacpp_provider_service.py`, implement:

```python
def normalize_llamacpp_base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def use_managed_server_in_chat(llm_manager: Any) -> dict[str, Any]:
    handler = getattr(llm_manager, "llamacpp", None)
    status = await_or_sync(handler.get_server_status())
    if status.get("status") != "running":
        raise ValueError("Managed llama.cpp server is not running.")
    endpoint = normalize_llamacpp_base_url(status["host"], int(status["port"]))
    setup_manager.update_config({"Local-API": {"llama_api_IP": endpoint}})
    refresh_config_cache()
    return {
        "provider": "llama",
        "endpoint": endpoint,
        "updated": True,
        "effective": True,
        "warnings": [],
    }
```

Keep the actual implementation async if it awaits handler status.
If a future environment override makes the saved provider endpoint ineffective,
return `effective=false` with a warning instead of pretending the chat provider
was changed.

- [ ] **Step 6: Implement safe log tail**

Add schema:

```python
class LlamaCppLogTailResponse(BaseModel):
    lines: list[str]
    truncated: bool = False
    warnings: list[str] = Field(default_factory=list)
```

Endpoint:

```text
GET /api/v1/llamacpp/logs/tail?lines=200
```

Implementation:

- gets `handler.config.log_output_file`;
- resolves and checks it is the configured path;
- never accepts a path query parameter;
- reads from the end with a byte cap;
- redacts obvious `api_key=...`, `token=...`, `hf_token=...`.

- [ ] **Step 7: Implement hardware snapshot**

Add:

```text
GET /api/v1/llamacpp/hardware
```

Return shape:

```python
class LlamaCppHardwareSnapshot(BaseModel):
    ram_total_bytes: int | None = None
    ram_available_bytes: int | None = None
    cpu_count: int | None = None
    gpus: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
```

Use `psutil` if available. Use optional NVML only behind import guards. Do not shell out to privileged tools in V1 unless there is already an accepted project helper for that pattern.

- [ ] **Step 8: Add permission claim coverage**

Update `test_llamacpp_permissions_claims.py` to include the new admin endpoints.

- [ ] **Step 9: Run focused backend tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_provider_and_logs_api.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_llamacpp_permissions_claims.py \
  -q
```

Expected: PASS.

- [ ] **Step 10: Commit Task 3**

```bash
git add \
  tldw_Server_API/app/core/Local_LLM/llamacpp_hardware_service.py \
  tldw_Server_API/app/core/Local_LLM/llamacpp_provider_service.py \
  tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/llamacpp.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_provider_and_logs_api.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_llamacpp_permissions_claims.py
git commit -m "feat: wire managed llama.cpp server to chat explicitly"
```

---

## Task 4: Frontend Client Types and API Methods

**Files:**
- Create: `apps/packages/ui/src/types/llamacpp-admin.ts`
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/models-audio.ts`
- Modify: `apps/packages/ui/src/services/tldw/client-ownership.ts`
- Test: existing type/build checks and `LlamacppAdminPage` tests once page uses methods.

- [ ] **Step 1: Add TypeScript contracts**

Create `llamacpp-admin.ts` matching backend response names:

```ts
export interface LlamacppConfigResponse {
  saved_config: LlamacppSavedConfig
  active_config: LlamacppActiveConfig
  restart_required: boolean
  restart_reasons: string[]
  env_overrides: Record<string, boolean>
  warnings: string[]
}

export interface LlamacppInventoryItem {
  model_id: string
  display_name: string
  basename: string
  source: string
  path: string
  size_bytes?: number | null
  modified_at?: string | null
  metadata: {
    quantization?: string | null
    parameter_hint?: string | null
    context_hint?: number | null
  }
  warnings: string[]
}
```

- [ ] **Step 2: Add client methods**

Add methods to both client surfaces:

```ts
async getLlamacppConfig(): Promise<LlamacppConfigResponse>
async updateLlamacppConfig(payload: Partial<LlamacppSavedConfig>): Promise<LlamacppConfigResponse>
async validateLlamacpp(payload: LlamacppValidationRequest): Promise<LlamacppValidationResponse>
async getLlamacppInventory(): Promise<LlamacppInventoryResponse>
async registerLlamacppModelPath(path: string): Promise<LlamacppInventoryItem>
async startLlamacppModel(modelId: string, serverArgs?: Record<string, any>): Promise<any>
async useLlamacppInChat(): Promise<LlamacppUseInChatResponse>
async tailLlamacppLogs(lines?: number): Promise<LlamacppLogTailResponse>
async getLlamacppHardware(): Promise<LlamacppHardwareSnapshot>
```

- [ ] **Step 3: Update client ownership**

Add new method names to `apps/packages/ui/src/services/tldw/client-ownership.ts` so ownership checks do not flag the facade methods.

- [ ] **Step 4: Run focused frontend tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx
```

Expected: Existing tests should still pass before page reshape.

- [ ] **Step 5: Commit Task 4**

```bash
git add \
  apps/packages/ui/src/types/llamacpp-admin.ts \
  apps/packages/ui/src/services/tldw/TldwApiClient.ts \
  apps/packages/ui/src/services/tldw/domains/models-audio.ts \
  apps/packages/ui/src/services/tldw/client-ownership.ts
git commit -m "feat: add llama.cpp admin client methods"
```

---

## Task 5: WebUI Guided Console

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Admin/LlamacppAdminPage.tsx`
- Create: `apps/packages/ui/src/components/Option/Admin/LlamacppReadinessPanel.tsx`
- Create: `apps/packages/ui/src/components/Option/Admin/LlamacppInventoryPanel.tsx`
- Create: `apps/packages/ui/src/components/Option/Admin/LlamacppLaunchPanel.tsx`
- Test: `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppReadinessPanel.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppInventoryPanel.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppLaunchPanel.test.tsx`

- [ ] **Step 1: Write failing UI tests for readiness**

Mock:

```ts
apiMock.getLlamacppConfig.mockResolvedValue({
  saved_config: { enabled: true, models_dir: "models/gguf_models" },
  active_config: { handler_configured: false },
  restart_required: true,
  restart_reasons: ["handler_not_configured"],
  env_overrides: {},
  warnings: []
})
```

Assert the page renders:

- "Restart required";
- saved models directory;
- no claim that the handler is active.

- [ ] **Step 2: Write failing UI tests for inventory and start-by-model**

Mock inventory with one model item and assert:

- model display name appears;
- warning tags appear;
- clicking start calls `startLlamacppModel(model_id, serverArgs)`, not `startLlamacppServer(filename, ...)`.

- [ ] **Step 3: Write failing UI tests for explicit chat wiring**

After mocked start success, assert:

- "Use this in Chat" appears;
- clicking it calls `useLlamacppInChat`;
- provider wiring success is displayed.

- [ ] **Step 4: Run failing UI tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx
```

Expected: FAIL because the page does not call the new facade.

- [ ] **Step 5: Reshape page state loading**

Replace the initial `Promise.all([loadStatus(), loadModels()])` with:

```ts
await Promise.all([
  loadConfig(),
  loadStatus(),
  loadInventory(),
  loadHardware()
])
```

Keep strict-mode duplicate-load protection.

- [ ] **Step 6: Add readiness section**

Render:

- enabled/disabled;
- binary path validity if validation has run;
- saved vs active config;
- restart-required alert;
- env override tags;
- status banner.

Use warnings and actions instead of long prose. Keep the existing admin guard behavior.

- [ ] **Step 7: Add inventory section**

Render inventory items as a table/list with:

- display name;
- basename/path source;
- size;
- quantization/parameter hints;
- warning tags;
- active model marker;
- select/start action.

Add a local path registration input. It should call `registerLlamacppModelPath(path)` and then reload inventory. Do not add upload controls.

- [ ] **Step 8: Preserve launch controls**

Move existing common controls into the launch section. Keep current advanced collapsibles for:

- Other Options;
- Multimodal;
- Speculative decoding;
- Network & Runtime;
- Raw argument overrides.

The default start path should call `startLlamacppModel(selectedModelId, buildLlamacppServerArgs(settings))`.

- [ ] **Step 9: Add hardware warnings**

Display hardware snapshot warnings near launch controls. If the backend returns unknown hardware, show a neutral warning state and still enable start.

- [ ] **Step 10: Add explicit provider wiring**

After start success or when status indicates a running managed server, show `Use this in Chat`. Do not call it automatically.

- [ ] **Step 11: Run focused UI tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx
```

Expected: PASS.

- [ ] **Step 12: Commit Task 5**

```bash
git add \
  apps/packages/ui/src/components/Option/Admin/LlamacppAdminPage.tsx \
  apps/packages/ui/src/components/Option/Admin/LlamacppReadinessPanel.tsx \
  apps/packages/ui/src/components/Option/Admin/LlamacppInventoryPanel.tsx \
  apps/packages/ui/src/components/Option/Admin/LlamacppLaunchPanel.tsx \
  apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx \
  apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppReadinessPanel.test.tsx \
  apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppInventoryPanel.test.tsx \
  apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppLaunchPanel.test.tsx
git commit -m "feat: guide llama.cpp server management in admin UI"
```

---

## Task 6: Docs, E2E Smoke, and Final Verification

**Files:**
- Modify: `Docs/API-related/llamacpp_integration_modes.md`
- Modify: `Docs/Published/API-related/llamacpp_integration_modes.md`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-4-admin/admin-llamacpp.spec.ts`
- Modify: `backlog/tasks/<implementation-task>.md` through Backlog MCP during execution.

- [x] **Step 1: Update llama.cpp integration docs**

Document:

- managed plane vs provider plane remains separate;
- new config/inventory/provider endpoints;
- restart-required semantics;
- warnings-first hardware guidance;
- explicit `Use this in Chat`.

- [x] **Step 2: Update E2E smoke test**

Mock backend responses for:

- config;
- inventory;
- hardware;
- start-by-model;
- use-in-chat.

Assert the normal flow:

1. page loads readiness;
2. model inventory appears;
3. user starts selected model;
4. provider wiring prompt appears;
5. user confirms `Use this in Chat`.

- [x] **Step 3: Run backend focused suite**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_admin_config_api.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_provider_and_logs_api.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_management_api.py \
  tldw_Server_API/tests/Local_LLM/test_llamacpp_handler.py \
  tldw_Server_API/tests/Local_LLM/test_llamacpp_hardening.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_llamacpp_permissions_claims.py \
  -q
```

Expected: PASS.

- [x] **Step 4: Run frontend focused suite**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx \
  apps/packages/ui/src/utils/__tests__/build-llamacpp-server-args.test.ts
```

Expected: PASS.

- [x] **Step 5: Run E2E smoke if server harness is available**

Run the existing project command for the tier-4 admin llama.cpp spec. If the repo requires a running dev server, start it with the established `bun run dev` workflow and record the URL/port used.

Run:

```bash
bunx playwright test apps/tldw-frontend/e2e/workflows/tier-4-admin/admin-llamacpp.spec.ts --reporter=line
```

Expected: PASS, or document the environment blocker.

- [x] **Step 6: Run Bandit on touched backend scope**

Run:

```bash
source .venv/bin/activate
python -m bandit \
  -r tldw_Server_API/app/api/v1/endpoints/llamacpp.py \
     tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py \
     tldw_Server_API/app/core/Local_LLM/LlamaCpp_Handler.py \
     tldw_Server_API/app/core/Local_LLM/llamacpp_config_service.py \
     tldw_Server_API/app/core/Local_LLM/llamacpp_inventory_service.py \
     tldw_Server_API/app/core/Local_LLM/llamacpp_hardware_service.py \
     tldw_Server_API/app/core/Local_LLM/llamacpp_provider_service.py \
  -f json \
  -o /tmp/bandit_llamacpp_admin.json
```

Expected: no new high/medium findings in touched code. Fix new findings before finalizing.

- [x] **Step 7: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no whitespace errors.

- [x] **Step 8: Commit Task 6**

```bash
git add \
  Docs/API-related/llamacpp_integration_modes.md \
  Docs/Published/API-related/llamacpp_integration_modes.md \
  apps/tldw-frontend/e2e/workflows/tier-4-admin/admin-llamacpp.spec.ts
git commit -m "docs: document llama.cpp admin management flow"
```

---

## Final Acceptance Checklist

- [x] `/admin/llamacpp` renders a guided readiness, inventory, and launch workflow.
- [x] Config facade reports saved config, active config, env overrides, and restart-required reasons.
- [x] Config updates use `setup_manager.update_config()` and refresh config caches.
- [x] Inventory supports bounded recursive GGUF scan and registered local paths.
- [x] `model_id` start path works without passing arbitrary absolute paths from the UI.
- [x] Existing `/llamacpp/start_server` filename flow still works.
- [x] Hardware snapshot warnings never hard-block launch.
- [x] `Use this in Chat` updates only `Local-API.llama_api_IP` after explicit confirmation.
- [x] Log tail endpoint is bounded and cannot read arbitrary paths.
- [x] New endpoints are admin-only and rate-limited consistently with existing lifecycle endpoints.
- [x] Focused backend tests pass.
- [x] Focused frontend tests pass.
- [x] E2E smoke either passes or has a documented environment blocker.
- [x] Bandit has no new actionable findings in touched backend code.
- [x] `git diff --check` passes.

## Implementation Notes

- Start with backend contracts. The frontend should not guess restart semantics or model path safety.
- Keep response field names stable and snake_case to match existing API style; map them in TypeScript types rather than reshaping ad hoc in components.
- Avoid adding a new profile persistence system during this implementation. The `model_id` contract is enough future-proofing for profiles.
- If `setup_manager.update_config()` rejects a new config key, add the key to `Config_Files/config.txt` before attempting to persist it.
- If both `TldwApiClient.ts` and `domains/models-audio.ts` need methods, update both in the same commit to avoid client drift.
- Do not resolve unrelated dirty files in the main worktree. Stage only files touched for this feature.
- Final closeout verification on the post-merge `origin/dev` baseline passed: backend focused llama.cpp suite `180 passed`; package-local frontend llama.cpp/admin suite `58 passed`; tier-4 Playwright admin llama.cpp smoke `6 passed`; Bandit on touched backend scope reported zero findings; `git diff --check` passed.
