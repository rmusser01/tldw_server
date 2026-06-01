# MCP Unified Stage 4N External Lifecycle Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add package-owned external MCP server lifecycle runtime controls, credential-secret handling contracts, and safe install/update foundations for the standalone gateway.

**Architecture:** Create a package-local `GatewayExternalRuntimeManager` that is separate from registry CRUD. The manager reads `ExternalServerDefinition` rows from `ExternalRegistryStore`, manages active injected transport instances, emits audit events, resolves brokered credentials only at execution time, and exposes disabled-by-default install/update adapter hooks. FastAPI can mount runtime routes for the in-process gateway; CLI durable lifecycle control is intentionally deferred until a daemon-control client exists.

**Tech Stack:** Python 3.11, Pydantic v2, FastAPI, package-local MCP Unified federation/storage contracts, pytest, Bandit.

---

## Scope And Constraints

Spec: `Docs/superpowers/specs/2026-05-31-mcp-unified-stage4n-external-lifecycle-runtime-design.md`

Backlog: `TASK-581`

Do not move host stdio/WebSocket adapters into `mcp_unified` in this slice. The stdio adapter still depends on `tldw_Server_API.app.core.Agent_Client_Protocol.stdio_client`; importing that from package code would violate the package boundary.

Do not add package-manager execution, network download, shell execution, or marketplace behavior. Install/update methods return deterministic not-configured or unsupported responses unless a test-injected installer adapter is supplied.

Do not add durable lifecycle CLI commands that pretend to control a separate running gateway process. FastAPI/in-process manager routes are the lifecycle control surface for this slice.

Keep all `mcp_unified` files free of `tldw_Server_API` imports.

## File Structure

- Create: `mcp_unified/gateway/external_runtime.py`
  - `GatewayExternalRuntimeError`
  - `GatewayExternalRuntimeManager`
  - lifecycle/status/credential summary helpers
- Create: `mcp_unified/federation/installers.py`
  - installer protocol and disabled-by-default implementation
- Modify: `mcp_unified/federation/transports.py`
  - allow runtime-auth-aware transport protocol and fake transport behavior
- Modify: `mcp_unified/federation/__init__.py`
  - export installer contracts if needed
- Modify: `mcp_unified/gateway/bootstrap.py`
  - carry optional `external_runtime_manager`
- Modify: `mcp_unified/gateway/fastapi.py`
  - mount runtime lifecycle routes when configured
- Modify: `mcp_unified/gateway/__init__.py`
  - lazily export external runtime manager/error if useful
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py`
  - manager lifecycle, credential, install/update contract coverage
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`
  - route mounting/error mapping coverage
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`
  - export/import-boundary coverage
- Modify: `backlog/tasks/task-581 - Implement-MCP-external-server-lifecycle-runtime-integration.md`
  - task notes, verification, final summary

## Task 1: Runtime Test Harness And Lifecycle Red Tests

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py`

- [x] **Step 1: Write test doubles**

Create copy-isolated test doubles:

```python
class InMemoryExternalRegistryStore:
    def __init__(self, servers=None) -> None:
        self.servers = {server.id: server.model_copy(deep=True) for server in servers or ()}

    async def get_server(self, server_id: str):
        server = self.servers.get(server_id)
        return None if server is None else server.model_copy(deep=True)

    async def list_servers(self):
        return await self.list_server_definitions()

    async def list_server_definitions(self, *, enabled=None):
        rows = [
            server for server in self.servers.values()
            if enabled is None or server.enabled is enabled
        ]
        return [server.model_copy(deep=True) for server in sorted(rows, key=lambda item: item.id)]
```

Add `RecordingAuditStore` and fake transport classes that record `connect_count`, `close_count`, `calls`, `runtime_auth`, and configurable discovery/call failures.

- [x] **Step 2: Write red tests for start/status/stop**

Tests:

```python
async def test_external_runtime_start_discovers_tools_and_reports_healthy_status():
    ...
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda server: transport,
        audit_store=audit,
    )

    payload = await manager.start_server("research")
    rows = await manager.list_runtime_servers()
    tools = await manager.list_virtual_tools()

    assert payload["ok"] is True
    assert payload["reason_code"] == "external_server_started"
    assert rows["servers"][0]["status"] == "healthy"
    assert [tool.virtual_name for tool in tools] == ["ext.research.search"]
    assert transport.connect_count == 1
```

```python
async def test_external_runtime_stop_is_idempotent_and_clears_tools():
    ...
    await manager.start_server("research")
    first = await manager.stop_server("research")
    second = await manager.stop_server("research")

    assert first["reason_code"] == "external_server_stopped"
    assert second["reason_code"] == "external_server_already_stopped"
    assert await manager.list_virtual_tools() == []
```

- [x] **Step 3: Verify red**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py -q
```

Expected: import failure for missing `mcp_unified.gateway.external_runtime`.

## Task 2: Package Runtime Manager Lifecycle

**Files:**
- Create: `mcp_unified/gateway/external_runtime.py`
- Modify: `mcp_unified/federation/transports.py`
- Modify: `mcp_unified/gateway/__init__.py`

- [x] **Step 1: Add runtime manager skeleton**

Implement:

```python
class GatewayExternalRuntimeError(RuntimeError):
    def __init__(self, message: str, *, reason_code: str, server_id: str | None = None) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.server_id = server_id

    def to_payload(self) -> dict[str, Any]:
        payload = {"ok": False, "error": str(self), "reason_code": self.reason_code}
        if self.server_id is not None:
            payload["server_id"] = self.server_id
        return payload
```

```python
class GatewayExternalRuntimeManager:
    def __init__(
        self,
        *,
        external_registry_store: ExternalRegistryStore,
        transport_factory: Callable[[ExternalServerDefinition], ExternalFederationTransport],
        audit_store: AuditStore | None = None,
        credential_broker: ExternalCredentialBroker | None = None,
        installer: ExternalServerInstaller | None = None,
    ) -> None: ...
```

- [x] **Step 2: Implement start/status/stop**

Use one `asyncio.Lock`. Store active transports in `self._transports`, loaded definitions in `self._servers`, virtual tools in `self._virtual_tools`, and error strings in `self._last_errors`.

`start_server()` should:

- load server with `get_server`
- reject unknown with `external_server_not_found`
- reject disabled with `external_server_disabled`
- stop an already active runtime before replacing it
- build/connect/discover with a temporary transport first
- commit active state only after successful discovery
- close the temporary transport on failure

`stop_server()` should:

- load the server if needed to distinguish unknown from known stopped
- close active transport if present
- clear tools and state
- return `external_server_already_stopped` for inactive known servers

- [x] **Step 3: Implement virtual tool discovery**

Reuse the `ext.<server_id>.<tool>` naming pattern and write classification from `ExternalFederationManager` if possible. Return caller-owned `VirtualExternalTool.copy()` values.

- [x] **Step 4: Run lifecycle tests**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py -q
```

Expected: new start/stop tests pass; future red tests may still fail as they are added.

- [x] **Step 5: Commit lifecycle core**

```bash
git add mcp_unified/gateway/external_runtime.py mcp_unified/federation/transports.py mcp_unified/gateway/__init__.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py
git commit -m "feat: add gateway external runtime lifecycle"
```

## Task 3: Refresh, Restart, And Reconcile

**Files:**
- Modify: `mcp_unified/gateway/external_runtime.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py`

- [x] **Step 1: Add red tests for refresh failure isolation**

Test that discovery failure on one server clears only that server's tools, marks
its status degraded/unhealthy, and preserves other active servers.

- [x] **Step 2: Add red tests for restart reload**

Update the in-memory store between start and restart. Assert the old transport
closed, a new transport connected, and discovered tool metadata reflects the
updated server definition.

- [x] **Step 3: Add red tests for reconcile**

Tests:

- newly enabled `auto_start=True` server starts
- disabled active server stops
- deleted active server stops
- changed active definition is replaced
- unchanged active server is refreshed

- [x] **Step 4: Implement `refresh_server`, `restart_server`, and `reconcile`**

Keep partial failures per-server. Response envelopes should include:

```python
{
    "ok": True,
    "reason_code": "external_server_reconciled",
    "server_id": server_id,
    "started_servers": int,
    "stopped_servers": int,
    "refreshed_servers": int,
    "total_servers": int,
    "errors": {...},
}
```

- [x] **Step 5: Run focused runtime tests**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py -q
```

Expected: all runtime manager lifecycle/reconcile tests pass.

- [x] **Step 6: Commit refresh/reconcile**

```bash
git add mcp_unified/gateway/external_runtime.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py
git commit -m "feat: reconcile external runtime lifecycle"
```

## Task 4: Credential Broker Secret Handling

**Files:**
- Modify: `mcp_unified/gateway/external_runtime.py`
- Modify: `mcp_unified/federation/transports.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py`

- [x] **Step 1: Add red credential-injection test**

Use sentinel values:

```python
SECRET_HEADER = "Bearer do-not-leak-header"
SECRET_ENV = "do-not-leak-env"
```

Execute a virtual tool through a broker that returns:

```python
BrokeredExternalCredential(
    headers={"Authorization": SECRET_HEADER},
    env={"TOKEN": SECRET_ENV},
    metadata={
        "credential_mode": "brokered_ephemeral",
        "credential_source": "test",
        "unsafe_note": SECRET_ENV,
    },
)
```

Assert:

- fake transport receives the credential for the call
- result metadata includes `credential_mode`, `credential_source`, and injected key names
- result metadata and audit payload do not contain sentinel secret values
- active server definitions are unchanged

- [x] **Step 2: Add red missing credential tests**

For a server with `credential_slots=["api_key"]`:

- no broker and no effective grant denies with `credential_broker_unavailable`
- broker returns `None` denies with `required_credential_grant_missing`

- [x] **Step 3: Implement credential broker protocol and summary helpers**

Define a local protocol in `external_runtime.py` unless it is clearer to add
`mcp_unified/interfaces/credentials.py`:

```python
class ExternalCredentialBroker(Protocol):
    async def resolve_external_credential(...) -> BrokeredExternalCredential | None: ...
```

Implement `_public_runtime_auth_metadata()` and `_summarize_runtime_auth()` so
only key names are exposed.

- [x] **Step 4: Pass runtime_auth through fake transport**

Update `ExternalFederationTransport` and `FakeExternalTransport.call_tool()` to
accept `runtime_auth: BrokeredExternalCredential | None = None`. Store a copied
credential in the fake for assertions; never include secret values in default
metadata.

- [x] **Step 5: Run credential tests**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py -q
```

Expected: all runtime and credential tests pass.

- [x] **Step 6: Commit credential handling**

```bash
git add mcp_unified/gateway/external_runtime.py mcp_unified/federation/transports.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py
git commit -m "feat: add external runtime credential brokering"
```

## Task 5: Install And Update Contracts

**Files:**
- Create: `mcp_unified/federation/installers.py`
- Modify: `mcp_unified/federation/__init__.py`
- Modify: `mcp_unified/gateway/external_runtime.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py`

- [x] **Step 1: Add red tests for default install/update responses**

Assert:

```python
install = await manager.install_server("research")
update = await manager.update_server("research")

assert install["reason_code"] == "external_server_install_not_configured"
assert update["reason_code"] == "external_server_update_not_configured"
```

- [x] **Step 2: Add red tests for unsupported injected installer**

Create a test installer that returns unsupported for the server and assert the
manager returns `external_server_install_unsupported` and
`external_server_update_unsupported` without mutating registry or active
transport state.

- [x] **Step 3: Implement installer protocol**

Create:

```python
class ExternalServerInstaller(Protocol):
    async def install_server(self, server: ExternalServerDefinition, *, context: Any = None) -> dict[str, Any]: ...
    async def update_server(self, server: ExternalServerDefinition, *, context: Any = None) -> dict[str, Any]: ...
    async def get_status(self, server: ExternalServerDefinition) -> dict[str, Any]: ...
```

Create `NullExternalServerInstaller` that returns not-configured responses and
`available: False` status.

- [x] **Step 4: Wire manager methods**

`install_server()` and `update_server()` should load the registry definition and
delegate to the configured installer. Unknown/disabled semantics should match
lifecycle methods unless tests prove update should work for disabled drafts.

- [x] **Step 5: Run installer tests**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py -q
```

Expected: all runtime, credential, and install/update tests pass.

- [x] **Step 6: Commit install/update contracts**

```bash
git add mcp_unified/federation/installers.py mcp_unified/federation/__init__.py mcp_unified/gateway/external_runtime.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py
git commit -m "feat: add external runtime install contracts"
```

## Task 6: FastAPI Runtime Routes

**Files:**
- Modify: `mcp_unified/gateway/bootstrap.py`
- Modify: `mcp_unified/gateway/fastapi.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`

- [x] **Step 1: Add red FastAPI tests**

In `test_gateway_fastapi_package.py`, add tests that create a fake runtime
manager and mount:

```python
app = create_gateway_app(
    runtime,
    prefix="/mcp",
    external_runtime_manager=manager,
)
```

Assert:

- `GET /mcp/external-servers/runtime` returns status rows
- `POST /mcp/external-servers/research/start` calls manager and returns payload
- `POST /mcp/external-servers/research/stop` calls manager
- `POST /mcp/external-servers/refresh` calls manager with `None`
- `POST /mcp/external-servers/research/refresh` calls manager with server id
- expected manager errors map to HTTP status codes

- [x] **Step 2: Extend bootstrap dataclass**

Add:

```python
external_runtime_manager: GatewayExternalRuntimeManager | None = None
```

Keep existing bootstrap call sites compatible by defaulting to `None`.

- [x] **Step 3: Add FastAPI resolver and route mount**

Add `external_runtime_manager` and `enable_external_runtime_management` params
to `create_gateway_router()` and `create_gateway_app()`.

Fail fast when explicit enable is true without a manager.

- [x] **Step 4: Add route handlers**

Implement the routes from the spec except durable CLI control. Use manager
methods directly and map `GatewayExternalRuntimeError.to_payload()` to status
codes:

- not found: 404
- disabled/conflict: 409
- transport/credential unavailable: 503
- invalid request: 422
- otherwise: 500

- [x] **Step 5: Run FastAPI tests**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py -q
```

Expected: runtime manager and FastAPI route tests pass.

- [x] **Step 6: Commit FastAPI routes**

```bash
git add mcp_unified/gateway/bootstrap.py mcp_unified/gateway/fastapi.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py
git commit -m "feat: expose external runtime gateway routes"
```

## Task 7: Boundary, Exports, And Compatibility Verification

**Files:**
- Modify: `mcp_unified/gateway/__init__.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`
- Modify: `backlog/tasks/task-581 - Implement-MCP-external-server-lifecycle-runtime-integration.md`
- Modify: this plan file

- [x] **Step 1: Add boundary/export tests**

Assert:

- `mcp_unified.gateway.GatewayExternalRuntimeManager` resolves lazily or imports
  without FastAPI-only side effects
- `mcp_unified.federation.NullExternalServerInstaller` and
  `ExternalServerInstaller` export if they are public
- package import-boundary test still finds no `tldw_Server_API` imports

- [x] **Step 2: Run focused compatibility suite**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_federation_shell_contracts.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_external_server_manager.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_external_credential_broker_runtime.py \
  -q
```

Expected: all focused tests pass.

- [x] **Step 3: Run lint/security/whitespace checks**

Run:

```bash
source ../../.venv/bin/activate
python -m ruff check mcp_unified/gateway mcp_unified/federation tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py
python -m bandit -r mcp_unified/gateway mcp_unified/federation -f json -o /tmp/bandit_mcp_stage4n_external_runtime.json
git diff --check
```

Expected: Ruff passes, Bandit reports no new findings in touched package code,
and diff check passes.

- [x] **Step 4: Update Backlog and plan status**

Record:

- tests run and results
- Bandit output path
- known skips, especially no durable CLI lifecycle commands by design
- final summary

- [x] **Step 5: Commit final validation**

```bash
git add mcp_unified/gateway/__init__.py tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py Docs/superpowers/plans/2026-05-31-mcp-unified-stage4n-external-lifecycle-runtime-implementation-plan.md backlog/tasks/task-581\ -\ Implement-MCP-external-server-lifecycle-runtime-integration.md
git commit -m "chore: validate external runtime integration"
```

## Final Verification Before PR

Run:

```bash
source ../../.venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_federation_shell_contracts.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_external_server_manager.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_external_credential_broker_runtime.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_registry_management.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py \
  -q
python -m ruff check mcp_unified/gateway mcp_unified/federation tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py
python -m bandit -r mcp_unified/gateway mcp_unified/federation -f json -o /tmp/bandit_mcp_stage4n_external_runtime.json
git diff --check
git status --short --branch
```

Expected:

- focused pytest suite passes
- Ruff passes
- Bandit JSON reports no new findings for touched package code
- `git diff --check` passes
- branch is clean after final commit

## Deliberate Deferrals

- Durable lifecycle CLI commands are deferred until there is a daemon-control
  client. A short-lived CLI process cannot truthfully start/stop transports in a
  separate running gateway.
- Real package-owned stdio process spawning remains deferred until executable
  allowlists, cwd validation, minimal environment construction, resource limits,
  and process audit policy are implemented in package code.
- Third-party server install/update execution is deferred; this slice only adds
  adapter contracts and deterministic disabled/unsupported responses.
