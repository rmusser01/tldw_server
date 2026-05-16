# Managed vLLM Instances Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build durable multi-instance managed `vLLM` lifecycle support with local and SSH execution, request-scoped routing for chat/embeddings/multimodal calls, and an admin UI for instance operations.

**Architecture:** Keep inference on the existing OpenAI-compatible `vllm` adapter, but add a new `VLLM_Management` control plane for persistent instance storage, executor-backed lifecycle management, capability tracking, and request-time instance resolution. Use Jobs for long-running lifecycle work and keep all routing request-scoped so concurrent requests can target different `vLLM` instances safely.

**Tech Stack:** FastAPI, Pydantic v2, Loguru, SQLite-first repository abstraction for v1, existing Jobs `JobManager` and `WorkerSDK`, existing chat and embeddings adapters, React, Ant Design, Vitest, Playwright smoke inventory wiring.

**Implementation scope note:** v1 should ship with a SQLite-backed persistent repository behind an interface designed for a later Postgres implementation. Do not claim multi-process/shared-state Postgres support unless a real Postgres-backed repository is added during execution.

---

### Task 1: Create the Persistent vLLM Instance Repository
**Status:** Complete

**Files:**
- Create: `tldw_Server_API/app/core/VLLM_Management/__init__.py`
- Create: `tldw_Server_API/app/core/VLLM_Management/models.py`
- Create: `tldw_Server_API/app/core/VLLM_Management/repository.py`
- Create: `tldw_Server_API/app/core/VLLM_Management/sqlite_repo.py`
- Create: `tldw_Server_API/tests/VLLM_Management/test_repository.py`

**Step 1: Write the failing test**

```python
def test_repository_round_trips_instance_and_default_route(tmp_path):
    repo = SqliteVLLMInstanceRepository(db_path=tmp_path / "vllm_instances.db")
    created = repo.create_instance(
        VLLMInstanceCreate(
            name="vision-a100",
            execution_mode="ssh",
            transport_config={
                "host": "gpu-a100-01.internal",
                "port": 22,
                "username": "ubuntu",
                "launcher_path": "/usr/local/bin/tldw-vllm-launcher",
                "auth": {"secret_ref": "ssh-key-vllm-prod"},
            },
            launch_spec={"model": "Qwen/Qwen2.5-VL-7B-Instruct", "port": 8001},
            routing_policy={"is_default": False},
            declared_capabilities={"chat": True, "embeddings": False, "vision": True},
        )
    )

    repo.set_default_instance(created.instance_id)
    fetched = repo.get_instance(created.instance_id)

    assert fetched is not None
    assert fetched.name == "vision-a100"
    assert repo.get_default_instance_id() == created.instance_id
```

**Step 2: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/VLLM_Management/test_repository.py::test_repository_round_trips_instance_and_default_route -v
```

Expected: FAIL with import errors because the repository and models do not exist yet.

**Step 3: Write minimal implementation**

```python
@dataclass
class VLLMInstanceRecord:
    instance_id: str
    name: str
    execution_mode: str
    transport_config: dict[str, Any]
    launch_spec: dict[str, Any]
    routing_policy: dict[str, Any]
    declared_capabilities: dict[str, Any]
    desired_state: str
    observed_state: str


class SqliteVLLMInstanceRepository:
    def create_instance(self, payload: VLLMInstanceCreate) -> VLLMInstanceRecord: ...
    def get_instance(self, instance_id: str) -> VLLMInstanceRecord | None: ...
    def list_instances(self) -> list[VLLMInstanceRecord]: ...
    def set_default_instance(self, instance_id: str | None) -> None: ...
    def get_default_instance_id(self) -> str | None: ...
```

Add the SQLite schema and CRUD helpers inside the new repository module. Keep raw SQL isolated to this abstraction.

Repository shape requirement:

- ship a SQLite-backed implementation in v1
- keep the repository interface storage-agnostic enough for a future Postgres-backed implementation
- persist `transport_config` and secret references without expanding secrets into logs or API echoes

**Step 4: Run test to verify it passes**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/VLLM_Management/test_repository.py -v
```

Expected: PASS for repository CRUD and default-route tests.

**Step 5: Commit**

```bash
git add tldw_Server_API/app/core/VLLM_Management/__init__.py \
        tldw_Server_API/app/core/VLLM_Management/models.py \
        tldw_Server_API/app/core/VLLM_Management/repository.py \
        tldw_Server_API/app/core/VLLM_Management/sqlite_repo.py \
        tldw_Server_API/tests/VLLM_Management/test_repository.py
git commit -m "feat: add persistent vllm instance repository"
```

### Task 2: Add the Request-Scoped Instance Resolver
**Status:** Complete

**Files:**
- Create: `tldw_Server_API/app/core/VLLM_Management/resolver.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/embeddings_models.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py`
- Modify: `tldw_Server_API/app/core/LLM_Calls/providers/local_adapters.py`
- Modify: `tldw_Server_API/app/core/Chat/chat_service.py`
- Create: `tldw_Server_API/tests/VLLM_Management/test_request_resolver.py`
- Create: `tldw_Server_API/tests/LLM_Calls/test_vllm_instance_routing.py`

**Step 1: Write the failing test**

```python
def test_resolver_prefers_request_instance_id_over_default():
    repo = InMemoryVLLMInstanceRepository(
        instances={
            "default-id": fake_instance("default-id", "http://127.0.0.1:8000/v1"),
            "vision-id": fake_instance("vision-id", "http://10.0.0.9:8000/v1"),
        },
        default_instance_id="default-id",
    )

    resolved = resolve_vllm_instance_for_request(
        provider="vllm",
        provider_instance_id="vision-id",
        required_capability="chat",
        repository=repo,
    )

    assert resolved.base_url == "http://10.0.0.9:8000/v1"
    assert resolved.instance_id == "vision-id"
```

**Step 2: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/VLLM_Management/test_request_resolver.py \
  tldw_Server_API/tests/LLM_Calls/test_vllm_instance_routing.py -v
```

Expected: FAIL because `provider_instance_id` and resolver wiring do not exist yet.

**Step 3: Write minimal implementation**

```python
class ResolvedVLLMRoute(BaseModel):
    instance_id: str
    base_url: str
    model: str | None
    effective_capabilities: dict[str, bool]


def resolve_vllm_instance_for_request(...):
    if provider != "vllm":
        return None
    # request instance id first, then default route
```

Add `provider_instance_id` to both chat and embeddings request schemas. Add an explicit `provider` field to embeddings request schemas so the endpoint can keep using provider-aware resolution rather than model-only heuristics. Then inject the resolved `base_url` and model into the `VLLMAdapter` call path via request payload overrides rather than global config mutation.

**Step 4: Run test to verify it passes**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/VLLM_Management/test_request_resolver.py \
  tldw_Server_API/tests/LLM_Calls/test_vllm_instance_routing.py -v
```

Expected: PASS, with per-request `vllm_api_url` overrides resolved from managed instance records.

**Step 5: Commit**

```bash
git add tldw_Server_API/app/core/VLLM_Management/resolver.py \
        tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py \
        tldw_Server_API/app/api/v1/schemas/embeddings_models.py \
        tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py \
        tldw_Server_API/app/core/LLM_Calls/providers/local_adapters.py \
        tldw_Server_API/app/core/Chat/chat_service.py \
        tldw_Server_API/tests/VLLM_Management/test_request_resolver.py \
        tldw_Server_API/tests/LLM_Calls/test_vllm_instance_routing.py
git commit -m "feat: add request scoped vllm instance routing"
```

### Task 3: Build the Structured Command Builder and Capability Model
**Status:** Complete

**Files:**
- Create: `tldw_Server_API/app/core/VLLM_Management/command_builder.py`
- Create: `tldw_Server_API/app/core/VLLM_Management/capabilities.py`
- Create: `tldw_Server_API/tests/VLLM_Management/test_command_builder.py`
- Create: `tldw_Server_API/tests/VLLM_Management/test_capabilities.py`

**Step 1: Write the failing test**

```python
def test_command_builder_prefers_structured_fields_over_extra_args():
    argv = build_vllm_serve_argv(
        launch_spec={
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "port": 8002,
            "tensor_parallel_size": 2,
            "extra_args": ["--port", "9999", "--dtype", "float16"],
        }
    )

    assert argv[:2] == ["vllm", "serve"]
    assert "--port" in argv
    assert "8002" in argv
    assert "9999" not in argv
```

**Step 2: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/VLLM_Management/test_command_builder.py \
  tldw_Server_API/tests/VLLM_Management/test_capabilities.py -v
```

Expected: FAIL because the command builder and capability helpers do not exist.

**Step 3: Write minimal implementation**

```python
STRUCTURED_FLAG_MAP = {
    "port": "--port",
    "served_model_name": "--served-model-name",
    "tensor_parallel_size": "--tensor-parallel-size",
}


def build_vllm_serve_argv(launch_spec: dict[str, Any]) -> list[str]:
    argv = ["vllm", "serve", launch_spec["model"]]
    # add structured flags first
    # add validated extra_args only when non-conflicting
    return argv
```

Add `declared_capabilities`, `probed_capabilities`, and `effective_capabilities` helpers so route resolution can reject unsupported embeddings or multimodal requests clearly.

**Step 4: Run test to verify it passes**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/VLLM_Management/test_command_builder.py \
  tldw_Server_API/tests/VLLM_Management/test_capabilities.py -v
```

Expected: PASS with argv-safe command construction and capability derivation.

**Step 5: Commit**

```bash
git add tldw_Server_API/app/core/VLLM_Management/command_builder.py \
        tldw_Server_API/app/core/VLLM_Management/capabilities.py \
        tldw_Server_API/tests/VLLM_Management/test_command_builder.py \
        tldw_Server_API/tests/VLLM_Management/test_capabilities.py
git commit -m "feat: add vllm command builder and capability model"
```

### Task 4: Implement the Local and SSH Executors
**Status:** Complete

**Files:**
- Create: `tldw_Server_API/app/core/VLLM_Management/executors/__init__.py`
- Create: `tldw_Server_API/app/core/VLLM_Management/executors/base.py`
- Create: `tldw_Server_API/app/core/VLLM_Management/executors/local.py`
- Create: `tldw_Server_API/app/core/VLLM_Management/executors/ssh.py`
- Create: `tldw_Server_API/app/core/VLLM_Management/executors/agent.py`
- Create: `tldw_Server_API/app/core/VLLM_Management/ssh_launcher.py`
- Create: `tldw_Server_API/tests/VLLM_Management/test_local_executor.py`
- Create: `tldw_Server_API/tests/VLLM_Management/test_ssh_executor.py`

**Step 1: Write the failing test**

```python
def test_ssh_executor_uses_launcher_contract_not_shell_backgrounding():
    runner = RecordingSSHRunner()
    executor = SSHVLLMExecutor(ssh_runner=runner)

    executor.start(fake_ssh_instance())

    assert "nohup" not in runner.last_command
    assert "&" not in runner.last_command
    assert runner.last_command[:2] == ["/usr/local/bin/tldw-vllm-launcher", "start"]
```

**Step 2: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/VLLM_Management/test_local_executor.py \
  tldw_Server_API/tests/VLLM_Management/test_ssh_executor.py -v
```

Expected: FAIL because executor modules and launcher contract do not exist yet.

**Step 3: Write minimal implementation**

```python
class VLLMExecutor(Protocol):
    def start(self, instance: VLLMInstanceRecord) -> LifecycleResult: ...
    def stop(self, instance: VLLMInstanceRecord, handle: dict[str, Any]) -> StopResult: ...
    def probe(self, instance: VLLMInstanceRecord) -> ProbeResult: ...


class SSHVLLMExecutor:
    def start(self, instance: VLLMInstanceRecord) -> LifecycleResult:
        return self._runner.run([
            instance.transport_config["launcher_path"],
            "start",
            "--host",
            instance.transport_config["host"],
            "--json-spec",
            json.dumps(instance.launch_spec),
        ])
```

For the local executor, use `subprocess.Popen`-style argv execution with log capture and pid metadata. For the SSH executor, pull host/user/auth/launcher details from `transport_config` rather than `launch_spec`. For the agent executor, add a placeholder class that raises a clear “not implemented” error until phase 2.

**Step 4: Run test to verify it passes**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/VLLM_Management/test_local_executor.py \
  tldw_Server_API/tests/VLLM_Management/test_ssh_executor.py -v
```

Expected: PASS for local lifecycle behavior and SSH launcher contract enforcement.

**Step 5: Commit**

```bash
git add tldw_Server_API/app/core/VLLM_Management/executors/__init__.py \
        tldw_Server_API/app/core/VLLM_Management/executors/base.py \
        tldw_Server_API/app/core/VLLM_Management/executors/local.py \
        tldw_Server_API/app/core/VLLM_Management/executors/ssh.py \
        tldw_Server_API/app/core/VLLM_Management/executors/agent.py \
        tldw_Server_API/app/core/VLLM_Management/ssh_launcher.py \
        tldw_Server_API/tests/VLLM_Management/test_local_executor.py \
        tldw_Server_API/tests/VLLM_Management/test_ssh_executor.py
git commit -m "feat: add vllm local and ssh executors"
```

### Task 5: Add the Admin API for Instance CRUD and Default Routing
**Status:** Complete

**Files:**
- Create: `tldw_Server_API/app/api/v1/schemas/vllm_management.py`
- Create: `tldw_Server_API/app/api/v1/endpoints/vllm_management.py`
- Modify: `tldw_Server_API/app/core/VLLM_Management/models.py`
- Modify: `tldw_Server_API/app/core/VLLM_Management/repository.py`
- Modify: `tldw_Server_API/app/core/VLLM_Management/sqlite_repo.py`
- Modify: `tldw_Server_API/app/main.py`
- Create: `tldw_Server_API/tests/LLM_Local/test_vllm_management_api.py`
- Create: `tldw_Server_API/tests/AuthNZ_Unit/test_vllm_permissions_claims.py`

**Step 1: Write the failing test**

```python
def test_create_instance_returns_backend_metadata_and_persisted_record(client):
    response = client.post(
        "/api/v1/llm/providers/vllm/instances",
        json={
            "name": "embed-box",
            "execution_mode": "local",
            "launch_spec": {"model": "BAAI/bge-m3", "port": 8010},
            "declared_capabilities": {"embeddings": True},
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["backend"] == "vllm"
    assert body["instance"]["name"] == "embed-box"
```

**Step 2: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_vllm_management_api.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_vllm_permissions_claims.py -v
```

Expected: FAIL because the schema, endpoint module, and router registration do not exist.

**Step 3: Write minimal implementation**

```python
@router.post("/llm/providers/vllm/instances", status_code=201)
async def create_vllm_instance(payload: VLLMInstanceCreateRequest, ...):
    record = repo.create_instance(payload.to_domain())
    return {"backend": "vllm", "instance": record}
```

Add CRUD endpoints, list/detail responses, and default-route mutation. Expand the repository contract so the admin API can support:

- instance spec updates (`update_instance`)
- instance deletion with safety checks (`delete_instance`)
- persisted runtime metadata reads needed by list/detail responses

The repo/model layer should now carry the mutable fields the lifecycle task will need next, including `probed_capabilities`, `effective_capabilities`, `last_known_base_url`, `last_error`, and executor handle metadata. Reuse the `check_rate_limit` and `require_roles("admin")` dependency pattern used by `mlx` and `llama.cpp`.

**Step 4: Run test to verify it passes**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_vllm_management_api.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_vllm_permissions_claims.py -v
```

Expected: PASS for admin-only CRUD/default-route behavior.

**Step 5: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/vllm_management.py \
        tldw_Server_API/app/api/v1/endpoints/vllm_management.py \
        tldw_Server_API/app/core/VLLM_Management/models.py \
        tldw_Server_API/app/core/VLLM_Management/repository.py \
        tldw_Server_API/app/core/VLLM_Management/sqlite_repo.py \
        tldw_Server_API/app/main.py \
        tldw_Server_API/tests/LLM_Local/test_vllm_management_api.py \
        tldw_Server_API/tests/AuthNZ_Unit/test_vllm_permissions_claims.py
git commit -m "feat: add vllm instance management api"
```

### Task 6: Add Jobs-Backed Start, Stop, Restart, Probe, and Reconciliation
**Status:** Complete

**Files:**
- Modify: `tldw_Server_API/app/core/VLLM_Management/models.py`
- Modify: `tldw_Server_API/app/core/VLLM_Management/repository.py`
- Modify: `tldw_Server_API/app/core/VLLM_Management/sqlite_repo.py`
- Create: `tldw_Server_API/app/core/VLLM_Management/service.py`
- Create: `tldw_Server_API/app/core/VLLM_Management/job_handlers.py`
- Create: `tldw_Server_API/app/core/VLLM_Management/reconciler.py`
- Create: `tldw_Server_API/app/services/vllm_management_worker.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/vllm_management.py`
- Modify: `tldw_Server_API/app/main.py`
- Create: `tldw_Server_API/tests/VLLM_Management/test_jobs_service.py`
- Create: `tldw_Server_API/tests/VLLM_Management/test_reconciler.py`
- Create: `tldw_Server_API/tests/VLLM_Management/test_worker.py`

**Step 1: Write the failing test**

```python
def test_start_endpoint_returns_job_metadata_instead_of_blocking(client, seeded_instance):
    response = client.post(
        f"/api/v1/llm/providers/vllm/instances/{seeded_instance.instance_id}/start",
        json={},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["job_id"] is not None
    assert body["requested_action"] == "start"
```

**Step 2: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/VLLM_Management/test_jobs_service.py \
  tldw_Server_API/tests/VLLM_Management/test_reconciler.py \
  tldw_Server_API/tests/VLLM_Management/test_worker.py -v
```

Expected: FAIL because lifecycle work is not yet routed through `JobManager`, the dedicated worker entrypoint does not exist, or reconciler startup hooks are missing.

**Step 3: Write minimal implementation**

```python
class VLLMManagementService:
    def enqueue_start(self, instance_id: str, owner_user_id: str | None) -> dict[str, Any]:
        return self.job_manager.create_job(
            domain="vllm_management",
            job_type="vllm_instance_start",
            payload={"instance_id": instance_id},
        )
```

Wire the job handlers to call the repository, resolver, command builder, and executor stack. This task must extend repository persistence so workers can atomically record lifecycle state transitions and runtime metadata, for example:

- desired and observed state changes
- probe results and effective capabilities
- last known base URL
- last error
- executor handle / pid / remote pid metadata

Use the existing shared `get_job_manager` dependency from `tldw_Server_API/app/api/v1/API_Deps/jobs_deps.py` rather than importing it from a media endpoint module. Add a dedicated `WorkerSDK`-based `vllm_management_worker.py` entrypoint for the `vllm_management` domain and register reconciler startup in `app/main.py` so persisted records are probed on boot.

**Step 4: Run test to verify it passes**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/VLLM_Management/test_jobs_service.py \
  tldw_Server_API/tests/VLLM_Management/test_reconciler.py \
  tldw_Server_API/tests/VLLM_Management/test_worker.py -v
```

Expected: PASS, with lifecycle endpoints returning Jobs metadata and reconciler updating observed state.

**Step 5: Commit**

```bash
git add tldw_Server_API/app/core/VLLM_Management/service.py \
        tldw_Server_API/app/core/VLLM_Management/models.py \
        tldw_Server_API/app/core/VLLM_Management/repository.py \
        tldw_Server_API/app/core/VLLM_Management/sqlite_repo.py \
        tldw_Server_API/app/core/VLLM_Management/job_handlers.py \
        tldw_Server_API/app/core/VLLM_Management/reconciler.py \
        tldw_Server_API/app/services/vllm_management_worker.py \
        tldw_Server_API/app/api/v1/endpoints/vllm_management.py \
        tldw_Server_API/app/main.py \
        tldw_Server_API/tests/VLLM_Management/test_jobs_service.py \
        tldw_Server_API/tests/VLLM_Management/test_reconciler.py \
        tldw_Server_API/tests/VLLM_Management/test_worker.py
git commit -m "feat: add job backed vllm lifecycle orchestration"
```

### Task 7: Expose Managed vLLM Metadata in Provider Listings and Embeddings
**Status:** Complete

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/llm_providers.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py`
- Modify: `tldw_Server_API/app/core/LLM_Calls/provider_metadata.py`
- Modify: `tldw_Server_API/app/core/LLM_Calls/tokenizer_resolver.py`
- Create: `tldw_Server_API/tests/LLM_Calls/test_vllm_provider_listing.py`
- Create: `tldw_Server_API/tests/VLLM_Management/test_embeddings_instance_resolution.py`

**Step 1: Write the failing test**

```python
def test_provider_listing_includes_managed_vllm_default_and_capabilities(client, seeded_repo):
    response = client.get("/api/v1/llm/providers")
    body = response.json()
    vllm = next(item for item in body["providers"] if item["name"] == "vllm")

    assert vllm["managed_instances"]["default_instance_id"] == "vision-id"
    assert vllm["managed_instances"]["count"] == 2
```

**Step 2: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/LLM_Calls/test_vllm_provider_listing.py \
  tldw_Server_API/tests/VLLM_Management/test_embeddings_instance_resolution.py -v
```

Expected: FAIL because provider listings and embeddings requests do not yet understand managed instance state.

**Step 3: Write minimal implementation**

```python
provider_entry["managed_instances"] = {
    "count": len(records),
    "default_instance_id": repo.get_default_instance_id(),
}
```

Also thread `provider_instance_id` through the embeddings endpoint so embeddings requests can resolve the correct managed `vLLM` target and enforce capability checks.

**Step 4: Run test to verify it passes**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/LLM_Calls/test_vllm_provider_listing.py \
  tldw_Server_API/tests/VLLM_Management/test_embeddings_instance_resolution.py -v
```

Expected: PASS with provider metadata enriched and embeddings routing honoring managed instances.

**Step 5: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/llm_providers.py \
        tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py \
        tldw_Server_API/app/core/LLM_Calls/provider_metadata.py \
        tldw_Server_API/app/core/LLM_Calls/tokenizer_resolver.py \
        tldw_Server_API/tests/LLM_Calls/test_vllm_provider_listing.py \
        tldw_Server_API/tests/VLLM_Management/test_embeddings_instance_resolution.py
git commit -m "feat: expose managed vllm metadata and embeddings routing"
```

### Task 8: Build the Admin UI and Client Wiring
**Status:** Complete

**Files:**
- Create: `apps/packages/ui/src/components/Option/Admin/VllmAdminPage.tsx`
- Create: `apps/packages/ui/src/components/Option/Admin/__tests__/VllmAdminPage.test.tsx`
- Create: `apps/packages/ui/src/routes/option-admin-vllm.tsx`
- Modify: `apps/packages/ui/src/routes/route-registry.tsx`
- Modify: `apps/packages/ui/src/components/Layouts/header-shortcut-items.ts`
- Modify: `apps/packages/ui/src/services/settings/ui-settings.ts`
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Create: `apps/tldw-frontend/extension/routes/option-admin-vllm.tsx`
- Modify: `apps/tldw-frontend/extension/routes/route-registry.tsx`
- Create: `apps/tldw-frontend/pages/admin/vllm.tsx`
- Modify: `apps/tldw-frontend/e2e/page-mapping.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/page-inventory.ts`

**Step 1: Write the failing test**

```tsx
it("loads managed instances and renders start controls", async () => {
  apiMock.listVllmInstances.mockResolvedValue({
    backend: "vllm",
    instances: [{ instance_id: "vision-id", name: "vision-a100", observed_state: "stopped" }]
  })

  render(<VllmAdminPage />)

  expect(await screen.findByText("vision-a100")).toBeInTheDocument()
  expect(screen.getByRole("button", { name: /start/i })).toBeInTheDocument()
})
```

**Step 2: Run test to verify it fails**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Admin/__tests__/VllmAdminPage.test.tsx
```

Expected: FAIL because the page, client methods, and routes do not exist yet.

**Step 3: Write minimal implementation**

```tsx
export const VllmAdminPage: React.FC = () => {
  const [instances, setInstances] = React.useState<VllmInstanceSummary[]>([])
  React.useEffect(() => { void tldwClient.listVllmInstances().then((data) => setInstances(data.instances)) }, [])
  return <List dataSource={instances} renderItem={(item) => <List.Item>{item.name}</List.Item>} />
}
```

Add TldwApiClient methods for instance list/create/update/delete/start/stop/restart/probe/default-route operations. Keep the first UI cut functional and admin-oriented before polishing advanced forms.

**Step 4: Run test to verify it passes**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Admin/__tests__/VllmAdminPage.test.tsx
```

Expected: PASS for the page’s basic lifecycle controls and error rendering.

**Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Option/Admin/VllmAdminPage.tsx \
        apps/packages/ui/src/components/Option/Admin/__tests__/VllmAdminPage.test.tsx \
        apps/packages/ui/src/routes/option-admin-vllm.tsx \
        apps/packages/ui/src/routes/route-registry.tsx \
        apps/packages/ui/src/components/Layouts/header-shortcut-items.ts \
        apps/packages/ui/src/services/settings/ui-settings.ts \
        apps/packages/ui/src/services/tldw/TldwApiClient.ts \
        apps/tldw-frontend/extension/routes/option-admin-vllm.tsx \
        apps/tldw-frontend/extension/routes/route-registry.tsx \
        apps/tldw-frontend/pages/admin/vllm.tsx \
        apps/tldw-frontend/e2e/page-mapping.ts \
        apps/tldw-frontend/e2e/smoke/page-inventory.ts
git commit -m "feat: add managed vllm admin ui"
```

### Task 9: Verify, Secure, and Document the Final Slice
**Status:** Complete

**Files:**
- Modify: `README.md`
- Modify: `Docs/Code_Documentation/Local_LLM.md`
- Modify: `Docs/Published/Overview/Feature_Status.md`
- Optionally create: `Docs/User_Guides/Integrations_Experiments/Managed_vLLM.md`

**Step 1: Write the failing verification checklist**

```markdown
- chat request with `provider="vllm"` and `provider_instance_id` routes to managed instance
- embeddings request with `provider_instance_id` rejects when capability is absent
- multimodal request rejects unhealthy or non-vision instances
- lifecycle endpoints create Jobs instead of blocking
- SSH executor never uses raw shell backgrounding
```

**Step 2: Run verification commands**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/VLLM_Management \
  tldw_Server_API/tests/LLM_Local/test_vllm_management_api.py \
  tldw_Server_API/tests/LLM_Calls/test_vllm_instance_routing.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_vllm_permissions_claims.py -v
python -m pytest --cov=tldw_Server_API --cov-report=term-missing \
  tldw_Server_API/tests/VLLM_Management \
  tldw_Server_API/tests/LLM_Local/test_vllm_management_api.py
python -m bandit -r tldw_Server_API/app/core/VLLM_Management \
  tldw_Server_API/app/api/v1/endpoints/vllm_management.py \
  tldw_Server_API/app/api/v1/schemas/vllm_management.py \
  -f json -o /tmp/bandit_vllm_managed_instances.json
```

Expected: PASS for tests; Bandit finds no new high-signal issues in the touched scope.

**Step 3: Write minimal documentation updates**

```markdown
## Managed vLLM

- Create one or more structured managed instances
- Start locally or over SSH
- Route chat or embeddings with `provider_instance_id`
- Use the admin page to inspect health, default route, and lifecycle jobs
```

Document:

- direct-reachability requirement
- SSH launcher contract
- `provider_instance_id`
- capability declaration vs probe behavior
- Jobs-backed lifecycle semantics

**Step 4: Re-run the highest-signal checks**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/VLLM_Management -v
cd apps/packages/ui
bunx vitest run src/components/Option/Admin/__tests__/VllmAdminPage.test.tsx
```

Expected: PASS for the final backend slice and admin UI smoke coverage.

**Step 5: Commit**

```bash
git add README.md \
        Docs/Code_Documentation/Local_LLM.md \
        Docs/Published/Overview/Feature_Status.md \
        Docs/User_Guides/Integrations_Experiments/Managed_vLLM.md
git commit -m "docs: document managed vllm instances"
```
