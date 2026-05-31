# MCP Unified Stage 4M Gateway External Registry Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement manager-owned standalone gateway external MCP server registry management over package CLI and FastAPI surfaces.

**Architecture:** Add a package-owned `GatewayExternalRegistryManager` as the only mutation boundary for external server definitions. Reuse `ExternalRegistryStore`, `CredentialGrantStore`, `AuditStore`, and `SQLiteMCPStore`, then keep config/bootstrap, FastAPI, and CLI thin over the same manager/storage bundle. Registry mutations persist definitions only; they do not start, stop, refresh, or hot-reload external federation runtime lifecycle.

**Tech Stack:** Python 3.11, Pydantic v2, FastAPI, SQLAlchemy Core-backed SQLite store, pytest, Bandit, package-local MCP Unified contracts.

---

## Scope And Constraints

Spec: `Docs/superpowers/specs/2026-05-31-mcp-unified-stage4m-gateway-external-registry-management-design.md`

Backlog: `TASK-579`

Do not implement real upstream stdio process spawning, WebSocket lifecycle, MCP server install/update flows, credential secret handling, credential grant CRUD, profile grant editing, approval policy editing, path-scope editing, or UI changes.

Keep `mcp_unified` package files free of `tldw_Server_API` imports. The host tests under `tldw_Server_API/app/core/MCP_unified/tests/` may import package code.

Use SQLAlchemy Core for SQLite changes. Do not add raw `sqlite3` usage to package code.

Run commands from this dedicated worktree. The shared virtual environment lives at `../../.venv` from this worktree path, so use `source ../../.venv/bin/activate` unless you intentionally create a local worktree venv.

## File Structure

- Modify: `mcp_unified/interfaces/storage.py`
  - Add external registry atomic create protocol and store-unavailable/duplicate exceptions.
- Modify: `mcp_unified/storage/sqlite.py`
  - Implement `create_server` atomically with SQLAlchemy Core and async offload.
- Create: `mcp_unified/gateway/external_registry.py`
  - Define `GatewayStoreMetadata` or reuse/generalize existing metadata.
  - Define `GatewayExternalRegistryManagementError`.
  - Define `GatewayExternalRegistryManager`.
- Modify: `mcp_unified/gateway/config.py`
  - Add `GatewayExternalRegistryStorageBundle`.
  - Add external registry storage/manager builder helpers that reuse the same SQLite store for registry, credential grants, and audit.
- Modify: `mcp_unified/gateway/bootstrap.py`
  - Only if the implementation chooses to expose `external_registry_manager` from a bootstrap object. Keep profile behavior compatible.
- Modify: `mcp_unified/gateway/fastapi.py`
  - Add external registry request/response models, route mounting, and error mapping.
- Modify: `mcp_unified/gateway/cli.py`
  - Add external registry commands using the shared config/storage builder.
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_registry_management.py`
  - New manager-focused tests.
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py`
  - Atomic create/server storage contract tests.
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`
  - Route mounting, success, and error mapping tests.
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py`
  - CLI command tests.
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py` only if new public exports need boundary coverage.
- Modify: `backlog/tasks/task-579 - Plan-MCP-Unified-Stage-4M-gateway-external-registry-management-implementation.md`
  - Keep task notes, verification, and final summary current.

## Task 1: Storage Contracts And Atomic SQLite Create

**Files:**
- Modify: `mcp_unified/interfaces/storage.py`
- Modify: `mcp_unified/storage/sqlite.py`
- Modify: `mcp_unified/storage/__init__.py` if exception exports are needed
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py`

- [ ] **Step 1: Add failing tests for atomic external server create**

Append tests near `test_sqlite_store_lists_external_server_definitions`:

```python
@pytest.mark.asyncio
async def test_sqlite_store_create_server_rejects_duplicate_id(tmp_path: Path) -> None:
    from mcp_unified.interfaces.storage import ExternalServerAlreadyExistsError
    from mcp_unified.storage import ExternalServerDefinition, SQLiteMCPStore

    store = SQLiteMCPStore(tmp_path / "mcp.sqlite")
    server = ExternalServerDefinition(
        id="search",
        name="Search",
        transport="websocket",
        url="wss://example.test/mcp",
    )

    created = await store.create_server(server)
    with pytest.raises(ExternalServerAlreadyExistsError) as exc_info:
        await store.create_server(server.model_copy(update={"name": "Other"}))

    assert created.id == "search"
    assert exc_info.value.server_id == "search"
    assert (await store.get_server("search")).name == "Search"
    await store.aclose()
```

- [ ] **Step 2: Run the storage test and verify it fails**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py::test_sqlite_store_create_server_rejects_duplicate_id -q
```

Expected: fail because `ExternalServerAlreadyExistsError` and/or `create_server` does not exist.

- [ ] **Step 3: Add external registry exceptions and protocol method**

In `mcp_unified/interfaces/storage.py`, add:

```python
class ExternalRegistryStoreUnavailableError(RuntimeError):
    """Raised when an external registry store cannot serve requests."""


class ExternalServerAlreadyExistsError(RuntimeError):
    """Raised when an atomic external server create conflicts with an existing id."""

    def __init__(self, server_id: str) -> None:
        super().__init__(f"External server already exists: {server_id}")
        self.server_id = server_id
```

Add to `ExternalRegistryStore`:

```python
async def create_server(
    self,
    server: ExternalServerDefinition,
) -> ExternalServerDefinition:
    """Create an external server definition and reject existing ids."""
    ...
```

- [ ] **Step 4: Implement `SQLiteMCPStore.create_server`**

In `mcp_unified/storage/sqlite.py`, add async and sync methods using the existing `_run_db`, `_dump_model`, `_load_model`, and `sqlite_insert` patterns:

```python
async def create_server(
    self,
    server: ExternalServerDefinition,
) -> ExternalServerDefinition:
    """Create an external server definition only when its id is absent."""
    return await self._run_db(self._create_server_sync, server)

def _create_server_sync(
    self,
    server: ExternalServerDefinition,
) -> ExternalServerDefinition:
    payload = self._dump_model(server)
    table = self._table("mcp_external_servers")
    statement = sqlite_insert(table).values(
        id=server.id,
        enabled=int(server.enabled),
        transport=server.transport,
        updated_at=server.updated_at.isoformat(),
        payload=payload,
    )
    with self._engine.begin() as connection:
        result = connection.execute(
            statement.on_conflict_do_nothing(index_elements=[table.c.id])
        )
    if not result.rowcount:
        raise ExternalServerAlreadyExistsError(server.id)
    return self._load_model(payload, ExternalServerDefinition)
```

- [ ] **Step 5: Export exceptions if tests/imports require it**

If tests import from `mcp_unified.storage`, update `mcp_unified/storage/__init__.py`. Prefer importing exceptions from `mcp_unified.interfaces.storage` in tests to avoid broad storage-module churn.

- [ ] **Step 6: Run focused storage tests**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py::test_sqlite_store_lists_external_server_definitions tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py::test_sqlite_store_create_server_rejects_duplicate_id -q
```

Expected: pass.

- [ ] **Step 7: Commit storage contract slice**

```bash
git add mcp_unified/interfaces/storage.py mcp_unified/storage/sqlite.py mcp_unified/storage/__init__.py tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py
git commit -m "feat: add external registry atomic create"
```

## Task 2: Gateway External Registry Manager

**Files:**
- Create: `mcp_unified/gateway/external_registry.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_registry_management.py`

- [ ] **Step 1: Write manager test doubles and first red tests**

Create `test_gateway_external_registry_management.py` with:

```python
from __future__ import annotations

from datetime import timezone

import pytest
from mcp_unified.storage.models import AuditEvent, CredentialGrant, ExternalServerDefinition

UTC = timezone.utc


class InMemoryExternalRegistryStore:
    def __init__(self, servers=None) -> None:
        self.servers = {}
        for server in servers or ():
            self.servers[server.id] = server.model_copy(deep=True)

    async def get_server(self, server_id: str):
        server = self.servers.get(server_id)
        return None if server is None else server.model_copy(deep=True)

    async def list_server_definitions(self, *, enabled=None):
        servers = [
            server for server in self.servers.values()
            if enabled is None or server.enabled is enabled
        ]
        return [server.model_copy(deep=True) for server in sorted(servers, key=lambda item: item.id)]

    async def create_server(self, server: ExternalServerDefinition):
        from mcp_unified.interfaces.storage import ExternalServerAlreadyExistsError
        if server.id in self.servers:
            raise ExternalServerAlreadyExistsError(server.id)
        self.servers[server.id] = server.model_copy(deep=True)
        return self.servers[server.id].model_copy(deep=True)

    async def upsert_server(self, server: ExternalServerDefinition):
        self.servers[server.id] = server.model_copy(deep=True)
        return self.servers[server.id].model_copy(deep=True)

    async def delete_server(self, server_id: str) -> bool:
        return self.servers.pop(server_id, None) is not None


class InMemoryCredentialGrantStore:
    def __init__(self, grants=None) -> None:
        self.grants = list(grants or [])

    async def list_grants(self, *, profile_id=None, external_server_id=None):
        return [
            grant.model_copy(deep=True)
            for grant in self.grants
            if (profile_id is None or grant.profile_id == profile_id)
            and (external_server_id is None or grant.external_server_id == external_server_id)
        ]


class InMemoryAuditStore:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def append_event(self, event: AuditEvent) -> AuditEvent:
        self.events.append(event.model_copy(deep=True))
        return event.model_copy(deep=True)


def _grant(
    *,
    grant_id: str = "grant-1",
    server_id: str = "github",
    slot: str = "token",
    enabled: bool = True,
) -> CredentialGrant:
    return CredentialGrant(
        id=grant_id,
        profile_id="profile-1",
        broker_id="broker-1",
        credential_slot=slot,
        external_server_id=server_id,
        enabled=enabled,
    )
```

Add red tests:

```python
@pytest.mark.asyncio
async def test_external_registry_manager_lists_servers_with_metadata() -> None:
    from mcp_unified.gateway.external_registry import (
        GatewayExternalRegistryManager,
        GatewayStoreMetadata,
    )

    manager = GatewayExternalRegistryManager(
        external_registry_store=InMemoryExternalRegistryStore([
            ExternalServerDefinition(id="draft", name="Draft", transport="stdio", enabled=False),
            ExternalServerDefinition(id="search", name="Search", transport="websocket", url="wss://example.test/mcp"),
        ]),
        store_metadata=GatewayStoreMetadata(kind="memory", persistent=False),
    )

    payload = await manager.list_servers()

    assert payload["ok"] is True
    assert [server["id"] for server in payload["servers"]] == ["draft", "search"]
    assert payload["store"] == {"kind": "memory", "persistent": False}
```

- [ ] **Step 2: Run the new manager test and verify it fails**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_registry_management.py::test_external_registry_manager_lists_servers_with_metadata -q
```

Expected: fail because `mcp_unified.gateway.external_registry` is missing.

- [ ] **Step 3: Implement manager skeleton and metadata**

Create `mcp_unified/gateway/external_registry.py` with:

```python
@dataclass(frozen=True, slots=True)
class GatewayStoreMetadata:
    """User-facing metadata describing a gateway management store."""

    kind: Literal["memory", "sqlite"]
    persistent: bool

    def to_payload(self) -> dict[str, Any]:
        return {"kind": self.kind, "persistent": self.persistent}


class GatewayExternalRegistryManagementError(RuntimeError):
    """Domain error for expected gateway external-registry failures."""

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

Then add a `GatewayExternalRegistryManager` with `list_servers` and `_dump_server`.

- [ ] **Step 4: Run manager list test**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_registry_management.py::test_external_registry_manager_lists_servers_with_metadata -q
```

Expected: pass.

- [ ] **Step 5: Add red tests for show/create/duplicate/audit**

Add tests for:

- `show_server` returns copy-isolated JSON-safe payload.
- `show_server` missing raises `external_server_not_found` and audits `external_server.show_failed`.
- `create_server` normalizes id/name, validates slug ids, and audits `external_server.created`.
- duplicate create raises `external_server_already_exists`.
- enabled websocket create rejects non-`ws://`/`wss://` URL with `invalid_external_server_request`.

Use expected payload fragments:

```python
assert exc_info.value.reason_code == "external_server_already_exists"
assert audit_store.events[-1].event_type == "external_server.created"
assert payload["server"]["id"] == "search"
assert payload["store"] == {"kind": "memory", "persistent": False}
```

- [ ] **Step 6: Implement create/show validation**

Implement:

- `_require_text(value, field=...)`
- `_optional_text(value, field=...)`
- `_validate_server_id(value)` with `[a-z0-9_-]+`
- `_validate_public_server_contract(server)` for enabled websocket URL scheme
- `_append_audit_event(...)`
- `_audit_expected_failure(...)`
- `show_server(...)`
- `create_server(...)`

Catch only expected validation errors:

```python
except (TypeError, ValueError) as exc:
    raise self._error("Invalid external server request", reason_code="invalid_external_server_request") from exc
```

Catch `ExternalServerAlreadyExistsError` and `ExternalRegistryStoreUnavailableError` explicitly.

- [ ] **Step 7: Add red tests for patch semantics and credential-slot guards**

Add tests for:

- patch replaces allowed scalar/list fields and audits changed fields.
- empty patch and unsupported fields raise `invalid_external_server_patch`.
- enabled-server credential slot addition is allowed.
- enabled-server credential slot removal raises `credential_slot_change_requires_disabled_server`.
- same patch that disables an enabled server may remove slots when no grants exist.
- disabled server slot removal with enabled grants raises `external_server_has_credential_grants`.
- slot removal without grant store raises `credential_grant_store_unavailable`.

Example assertion:

```python
with pytest.raises(GatewayExternalRegistryManagementError) as exc_info:
    await manager.patch_server("github", {"credential_slots": []})
assert exc_info.value.reason_code == "credential_slot_change_requires_disabled_server"
```

- [ ] **Step 8: Implement patch and grant guards**

Supported patch fields:

```python
_SERVER_PATCH_FIELDS = frozenset({
    "name", "transport", "command", "url", "cwd", "env_allowlist",
    "credential_slots", "metadata", "provenance", "enabled", "auto_start",
})
```

Guard credential slot relaxation before persisting:

```python
old_slots = set(_normalized_strings(existing.credential_slots))
new_slots = set(_normalized_strings(updated.credential_slots))
removed = old_slots - new_slots
if removed:
    if existing.enabled and updated.enabled:
        raise self._error(
            "Credential slot removal requires disabling the external server",
            reason_code="credential_slot_change_requires_disabled_server",
            server_id=server_id,
        )
    await self._ensure_no_enabled_grants(server_id)
```

Treat `CredentialGrant.enabled is False` as not blocking. If no grant store is configured and a slot relaxation needs grant knowledge, fail with `credential_grant_store_unavailable`.

- [ ] **Step 9: Add red tests for delete guards**

Add tests for:

- deleting missing server raises `external_server_not_found`.
- deleting server with enabled credential grants raises `external_server_has_credential_grants`.
- deleting without grant store raises `credential_grant_store_unavailable`.
- deleting ungranted server succeeds and audits `external_server.deleted`.

- [ ] **Step 10: Implement delete guard and success path**

Delete flow:

1. Normalize server id.
2. Load server; if missing, audit and raise not found.
3. Require an explicitly configured credential grant store for deletion. The config builder may pass the same SQLite object as both registry and grant store, but the manager should not implicitly discover grant methods on the registry store.
4. Query grants for `external_server_id`.
5. If any enabled grants exist, raise `external_server_has_credential_grants`.
6. Delete via `external_registry_store.delete_server`.
7. Translate false result to not found.
8. Audit success.

- [ ] **Step 11: Run manager test file**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_registry_management.py -q
```

Expected: all tests in the new manager file pass.

- [ ] **Step 12: Commit manager slice**

```bash
git add mcp_unified/gateway/external_registry.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_registry_management.py
git commit -m "feat: add gateway external registry manager"
```

## Task 3: Config And Bootstrap Wiring

**Files:**
- Modify: `mcp_unified/gateway/config.py`
- Modify: `mcp_unified/gateway/bootstrap.py` if needed
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`

- [ ] **Step 1: Add failing config tests**

Add tests near existing config bootstrap tests:

```python
def test_gateway_config_builds_sqlite_external_registry_storage(tmp_path: Path) -> None:
    from mcp_unified.gateway.config import (
        GatewayProfileStoreConfig,
        build_gateway_external_registry_storage,
    )
    from mcp_unified.storage.sqlite import SQLiteMCPStore

    bundle = build_gateway_external_registry_storage(
        GatewayProfileStoreConfig(kind="sqlite", sqlite_path=tmp_path / "gateway.db")
    )

    assert isinstance(bundle.external_registry_store, SQLiteMCPStore)
    assert bundle.credential_grant_store is bundle.external_registry_store
    assert bundle.audit_store is bundle.external_registry_store
    assert bundle.metadata.to_payload() == {"kind": "sqlite", "persistent": True}
```

Add a test that injected registry store without credential grant store still builds, but manager operations requiring grants fail at manager level.

- [ ] **Step 2: Run config tests and verify they fail**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py::test_gateway_config_builds_sqlite_external_registry_storage -q
```

Expected: fail because `build_gateway_external_registry_storage` is missing.

- [ ] **Step 3: Implement external registry storage bundle**

In `mcp_unified/gateway/config.py`, add:

```python
@dataclass(frozen=True, slots=True)
class GatewayExternalRegistryStorageBundle:
    """Resolved external registry, credential grant, audit stores and metadata."""

    external_registry_store: ExternalRegistryStore
    credential_grant_store: CredentialGrantStore | None
    audit_store: AuditStore | None
    metadata: GatewayStoreMetadata
```

Add `build_gateway_external_registry_storage(...)`:

- Reuse injected `external_registry_store` when provided.
- For `memory`, return an in-memory registry only if an in-memory external store is implemented. Otherwise raise `ValueError("external registry management requires sqlite store")` for CLI/manager factory use.
- For `sqlite`, create one `SQLiteMCPStore` and cast it to external registry, credential grant, and audit stores.

If a memory registry test double is not needed in production config, keep memory unsupported in this builder and use direct manager construction in unit tests.

- [ ] **Step 4: Add manager factory helper**

Add:

```python
def external_registry_manager_from_storage(
    bundle: GatewayExternalRegistryStorageBundle,
) -> GatewayExternalRegistryManager:
    return GatewayExternalRegistryManager(
        external_registry_store=bundle.external_registry_store,
        credential_grant_store=bundle.credential_grant_store,
        audit_store=bundle.audit_store,
        store_metadata=bundle.metadata,
    )
```

Use a clearer name if local style prefers `_manager_from_bundle`.

- [ ] **Step 5: Run config/bootstrap tests**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -k "external_registry_storage or config_bootstrap_uses_sqlite_profile_store" -q
```

Expected: pass.

- [ ] **Step 6: Commit config slice**

```bash
git add mcp_unified/gateway/config.py mcp_unified/gateway/bootstrap.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py
git commit -m "feat: wire gateway external registry storage"
```

## Task 4: FastAPI External Registry Routes

**Files:**
- Modify: `mcp_unified/gateway/fastapi.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`

- [ ] **Step 1: Add FastAPI route mounting tests**

Add manager double methods similar to `_ProfileManagementManagerDouble`:

```python
class _ExternalRegistryManagerDouble:
    def __init__(self) -> None:
        self.calls = []

    async def list_servers(self, *, enabled=None):
        self.calls.append(("list_servers", (), {"enabled": enabled}))
        return {"ok": True, "servers": [{"id": "search", "name": "Search"}], "store": {"kind": "memory", "persistent": False}}
```

Tests:

- routes are not mounted by default.
- routes mount with explicit `external_registry_manager`.
- `enable_external_registry_management=True` without manager raises `ValueError`.
- explicit manager wins over bootstrap manager if both are present.

- [ ] **Step 2: Run route mounting tests and verify they fail**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -k "external_registry_management_routes" -q
```

Expected: fail because route parameters and mounting are missing.

- [ ] **Step 3: Add FastAPI models and route resolver**

In `mcp_unified/gateway/fastapi.py`:

- Import `GatewayExternalRegistryManager` and `GatewayExternalRegistryManagementError`.
- Add `_EXTERNAL_REGISTRY_STATUS_CODES`.
- Add request/response models:
  - `CreateExternalServerRequest`
  - `PatchExternalServerRequest`
  - `ExternalServerListResponse`
  - `ExternalServerResponse`
  - `DeleteExternalServerResponse`
- Add `_external_registry_error_response`.
- Add `_resolve_external_registry_manager(...)`.
- Add `_mount_external_registry_routes(...)`.

- [ ] **Step 4: Extend router/app factory signatures**

Add optional parameters:

```python
external_registry_manager: GatewayExternalRegistryManager | None = None
enable_external_registry_management: bool = False
```

If the implementation extends a bootstrap object, read `external_registry_manager` with `getattr(profile_bootstrap, "external_registry_manager", None)` instead of changing existing tests that use doubles.

- [ ] **Step 5: Add success and error mapping tests**

Cover:

- `GET /mcp/external-servers`
- `GET /mcp/external-servers?enabled=true`
- `GET /mcp/external-servers/{id}`
- `POST /mcp/external-servers`
- `PATCH /mcp/external-servers/{id}`
- `DELETE /mcp/external-servers/{id}`
- reason-code mappings: `external_server_not_found` -> 404, `external_server_already_exists` -> 409, `credential_grant_store_unavailable` -> 503, invalid request/patch -> 422.

- [ ] **Step 6: Implement routes**

Route bodies should delegate directly to the manager:

```python
@router.get("/external-servers", response_model=ExternalServerListResponse)
async def list_external_servers(enabled: bool | None = None):
    try:
        return await manager.list_servers(enabled=enabled)
    except GatewayExternalRegistryManagementError as exc:
        return _external_registry_error_response(exc)
```

Use `request.model_dump(mode="json", exclude_unset=True)` for PATCH.

- [ ] **Step 7: Run FastAPI package tests**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -k "external_registry or profile_management_routes_are_not_mounted_by_default" -q
```

Expected: pass.

- [ ] **Step 8: Commit FastAPI slice**

```bash
git add mcp_unified/gateway/fastapi.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py
git commit -m "feat: expose gateway external registry routes"
```

## Task 5: CLI External Registry Commands

**Files:**
- Modify: `mcp_unified/gateway/cli.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py`

- [ ] **Step 1: Add CLI red tests for list/show/create**

Add tests near profile CRUD CLI tests:

```python
def _server_payload(server_id: str, name: str) -> dict[str, object]:
    return {
        "id": server_id,
        "name": name,
        "transport": "websocket",
        "url": "wss://example.test/mcp",
    }
```

Tests:

- `create-external-server --server-file <json> --config <sqlite config>` persists.
- `show-external-server <id>` returns stored server.
- `list-external-servers --enabled true` filters enabled servers.
- `create-external-server --server-file -` accepts stdin JSON.

- [ ] **Step 2: Run CLI tests and verify they fail**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py -k "external_server" -q
```

Expected: fail because commands are missing.

- [ ] **Step 3: Add CLI parser commands**

In `_build_parser()` add:

- `list-external-servers`
- `show-external-server`
- `create-external-server`
- `patch-external-server`
- `delete-external-server`

Use `--server-file` for create and `--patch-file` for patch. Reuse `_load_json_argument_file`.

- [ ] **Step 4: Add external registry CLI command runner**

Add a runner parallel to `_handle_profile_management_command`:

```python
def _handle_external_registry_command(
    args: argparse.Namespace,
    operation: _ExternalRegistryOperation,
    *,
    require_persistent: bool = True,
) -> int:
    config_path = _config_path_from_args(args)
    bundle = None
    try:
        config = load_gateway_profile_bootstrap_config(config_path)
        bundle = build_gateway_external_registry_storage(config.store)
        if require_persistent and not bundle.metadata.persistent:
            raise GatewayExternalRegistryManagementError(
                "External registry management requires a persistent gateway store",
                reason_code="external_registry_store_unavailable",
            )
        manager = external_registry_manager_from_storage(bundle)
        payload = _run_async(operation(manager))
    except GatewayExternalRegistryManagementError as exc:
        _emit_json(exc.to_payload(), sys.stderr)
        return 1
    except Exception as exc:
        _emit_json({"error": str(exc), "ok": False, "path": str(config_path)}, sys.stderr)
        return 1
    finally:
        if bundle is not None:
            _run_async(_close_external_registry_bundle(bundle))
    _emit_json(_cli_payload(payload), sys.stdout)
    return 0
```

Close unique stores from registry, grant, and audit fields.

- [ ] **Step 5: Add CLI error tests**

Cover:

- malformed `--server-file` JSON returns exit 2 and no traceback.
- non-object JSON returns exit 2.
- duplicate create returns reason `external_server_already_exists`.
- patch with unsupported field returns `invalid_external_server_patch`.
- delete with credential grant returns `external_server_has_credential_grants`.
- memory config returns `external_registry_store_unavailable`.

- [ ] **Step 6: Implement handler methods**

Add:

- `_handle_list_external_servers`
- `_handle_show_external_server`
- `_handle_create_external_server`
- `_handle_patch_external_server`
- `_handle_delete_external_server`

Normalize `--enabled` with choices `("true", "false")` and convert to bool.

- [ ] **Step 7: Run CLI package tests**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py -q
```

Expected: pass.

- [ ] **Step 8: Commit CLI slice**

```bash
git add mcp_unified/gateway/cli.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py
git commit -m "feat: add gateway external registry CLI"
```

## Task 6: Integration Sweep And Boundary Validation

**Files:**
- Modify: any touched files from earlier tasks if fixes are needed
- Modify: `backlog/tasks/task-579 - Plan-MCP-Unified-Stage-4M-gateway-external-registry-management-implementation.md`
- Possibly modify: `Docs/superpowers/plans/2026-05-31-mcp-unified-stage4m-gateway-external-registry-management-implementation-plan.md` if implementation discoveries require plan corrections

- [ ] **Step 1: Run focused Stage 4M suite**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_registry_management.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run package import boundary test alone**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -q
```

Expected: pass and no `mcp_unified` package file imports `tldw_Server_API`.

- [ ] **Step 3: Run Bandit on touched package scope**

Run:

```bash
source ../../.venv/bin/activate
python -m bandit -r mcp_unified/gateway mcp_unified/storage mcp_unified/interfaces -f json -o /tmp/bandit_mcp_stage4m.json
```

Expected: 0 new findings in touched code. If Bandit reports findings, inspect `/tmp/bandit_mcp_stage4m.json` and fix new issues before continuing.

- [ ] **Step 4: Run whitespace validation**

Run:

```bash
git diff --check
```

Expected: no output, exit 0.

- [ ] **Step 5: Update Backlog task**

Use Backlog MCP or CLI to set implementation task status and final summary once implementation exists. Include:

- files changed
- focused test counts
- Bandit output summary
- known skips or blockers

- [ ] **Step 6: Commit validation/closeout changes**

```bash
git add backlog/tasks/task-579\ -\ Plan-MCP-Unified-Stage-4M-gateway-external-registry-management-implementation.md
git commit -m "chore: close Stage 4M external registry task"
```

Only make this commit if the task file changes during implementation closeout.

## Final Implementation Checklist

- [ ] External registry atomic create exists and duplicate conflicts are deterministic.
- [ ] Gateway external registry manager owns validation, audit, and guards.
- [ ] Credential-slot relaxation cannot silently remove credential requirements from an enabled server.
- [ ] Delete fails closed when grant state is unavailable or grants reference the server.
- [ ] CLI and FastAPI use the same storage bundle semantics.
- [ ] Registry mutations do not start, stop, refresh, or hot-reload external federation runtime.
- [ ] `mcp_unified` package import boundary remains clean.
- [ ] Focused pytest suite passes.
- [ ] Bandit touched-scope scan is clean or documented with no new findings.
- [ ] `git diff --check` passes.
