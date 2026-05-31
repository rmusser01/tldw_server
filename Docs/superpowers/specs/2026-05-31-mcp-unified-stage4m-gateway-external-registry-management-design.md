# MCP Unified Stage 4M Gateway External Registry Management Design

Date: 2026-05-31
Status: Draft for review
Backlog: TASK-578

## Summary

Stage 4M adds the next standalone gateway management surface after editable
profile CRUD: manager-owned external MCP server registry management for the
package gateway.

The slice should let standalone gateway operators list, inspect, create, patch,
enable, disable, and delete stored external server definitions through CLI and
FastAPI routes. It should reuse the existing package-owned
`ExternalServerDefinition`, `ExternalRegistryStore`, `AuditStore`, and
`SQLiteMCPStore` primitives. It must not start real upstream processes or turn
external tools executable just because a registry row exists.

## Goals

- Add a package-owned `GatewayExternalRegistryManager` or equivalent manager as
  the single mutation boundary for external server definitions.
- Expose deterministic CLI and FastAPI management operations for external
  registry rows.
- Preserve package isolation: no `mcp_unified` import may depend on
  `tldw_Server_API`.
- Persist definitions through the existing SQLite store and keep memory stores
  read-only for mutating CLI workflows unless a later design explicitly changes
  that posture.
- Emit compact audit events for successful and failed registry mutations when an
  audit store is configured.
- Keep registry management separate from external federation runtime lifecycle,
  discovery refresh, credential brokering, and policy execution.

## Non-Goals

- No real upstream stdio process spawning.
- No WebSocket upstream connection lifecycle beyond stored definition metadata.
- No install, update, package-manager, marketplace, or dependency-resolution
  behavior for third-party MCP servers.
- No secret storage or credential value handling.
- No credential grant CRUD, profile grant editing, approval policy editing, path
  scope editing, or UI changes.
- No automatic hot-reload notification to a running gateway process after CLI
  mutations.
- No `tldw_server` MCP Hub management replacement. In `tldw_server`, MCP Hub
  remains authoritative for managed external servers.

## Current Foundation

The package already has the primitives needed for a narrow management slice:

- `mcp_unified.storage.models.ExternalServerDefinition`
- `mcp_unified.interfaces.storage.ExternalRegistryStore`
- `mcp_unified.storage.sqlite.SQLiteMCPStore`
- `mcp_unified.federation.ExternalFederationManager`
- gateway profile-management patterns in `mcp_unified.gateway.profiles`,
  `mcp_unified.gateway.fastapi`, and `mcp_unified.gateway.cli`

Stage 2F intentionally created a non-spawning federation shell. Stage 4M should
not expand that into process lifecycle. It should only make registry definitions
manageable through the standalone gateway product surface.

## Approach Options

### Option A: Extend `ExternalFederationManager`

This would put CRUD methods directly on the lifecycle manager.

Tradeoff: it is convenient for future refresh workflows, but it mixes stored
configuration mutation with runtime transport state. That makes CLI usage
awkward because CLI commands should not need to construct transport factories or
start lifecycle state just to edit the registry.

### Option B: Add A Separate Registry Manager

Create a `GatewayExternalRegistryManager` that depends on a typed
`ExternalRegistryStore`, an optional `CredentialGrantStore`, an optional
`AuditStore`, and store metadata. Runtime lifecycle remains owned by
`ExternalFederationManager`.

Tradeoff: this adds one small manager, but it keeps responsibilities clear and
matches Stage 4K/4L.

### Option C: Expose Store Methods Directly In CLI/FastAPI

FastAPI and CLI handlers could call `ExternalRegistryStore` methods directly.

Tradeoff: this is shortest initially, but duplicates validation, error mapping,
audit behavior, and deletion guards across transports.

## Recommended Approach

Use Option B.

Add a package-owned gateway registry manager and keep CLI/FastAPI thin over it.
This mirrors the existing profile-management architecture and gives later slices
a stable place to add guarded delete, grant-aware checks, import/export, and
runtime refresh coordination without spreading policy across transports.

## Data Model

Stage 4M should use the existing `ExternalServerDefinition` model without a
schema migration:

```text
ExternalServerDefinition
  id: str
  name: str
  transport: "stdio" | "websocket"
  command: list[str]
  url: str | None
  cwd: str | None
  env_allowlist: list[str]
  credential_slots: list[str]
  metadata: dict
  provenance: dict
  enabled: bool
  auto_start: bool
  created_at: aware datetime
  updated_at: aware datetime
```

The spec intentionally treats `command` as configuration data only. Runtime
spawn validation belongs to the future lifecycle slice that introduces direct
exec allowlists, cwd canonicalization, env allowlists, resource limits, and
process audit events.

Server ids become part of virtual tool names such as `ext.<server_id>.<tool>`.
The manager should therefore restrict ids to a conservative lowercase slug
format, matching the existing external config schema: `[a-z0-9_-]+`. Names are
display strings and may be less constrained, but they must still be non-blank.

## Manager Contract

The manager should expose async methods:

```python
list_servers(enabled: bool | None = None) -> dict
show_server(server_id: str) -> dict
create_server(server_document: ExternalServerDefinition | Mapping[str, Any]) -> dict
patch_server(server_id: str, patch_document: Mapping[str, Any]) -> dict
delete_server(server_id: str) -> dict
```

If the implementation wants explicit enable/disable helpers, they should be
thin wrappers around `patch_server(server_id, {"enabled": bool})` rather than
separate persistence paths.

The manager owns:

- id/name normalization and required text checks
- server id slug validation for stable virtual-tool names
- Pydantic validation error translation
- allowed patch-field validation
- transport-specific semantic validation
- duplicate detection
- guarded delete behavior
- audit event emission
- deterministic response envelopes

The manager must return caller-owned payloads, not live mutable store objects.

## Store Capabilities

Existing `ExternalRegistryStore` supports `get_server`, `list_servers`,
`list_server_definitions`, `upsert_server`, and `delete_server`.

Stage 4M should add a store-level `create_server` capability for persistent
stores so duplicate create is atomic, matching the Stage 4L profile create
hardening pattern. If the SQLite implementation is touched, use SQLAlchemy Core
and async offload through the existing `_run_db` helper.

The manager should require typed registry-store behavior for management. It
should prefer `list_server_definitions` for list operations and should not build
editable management responses from runtime health/status rows returned by a
legacy `list_servers` implementation.

Delete needs one guard: if credential grants reference the server, deletion must
fail with `external_server_has_credential_grants` when a `CredentialGrantStore`
is available. If no grant store is configured, deletion should fail closed for
persistent stores unless the implementation can prove no grants exist in the
same store. Disabling remains allowed because it does not orphan grants.

## Patch Semantics

Supported top-level patch fields:

- `name`
- `transport`
- `command`
- `url`
- `cwd`
- `env_allowlist`
- `credential_slots`
- `metadata`
- `provenance`
- `enabled`
- `auto_start`

Unsupported fields should fail with `invalid_external_server_patch`. Empty patch
documents should also fail with `invalid_external_server_patch`.

Patch is replace-style for each supported field, not merge-patch. Nested
metadata/provenance merge behavior can be added later if there is a concrete UI
need.

Changing `transport` should revalidate the whole resulting document. For enabled
servers:

- `stdio` requires a non-empty `command`
- `websocket` requires a non-empty `ws://` or `wss://` URL

The manager should normalize list fields by dropping blank items and should not
scrub unused transport fields automatically. Operators can clear no-longer-used
fields by patching them to `[]` or `null` where the model supports it.
`auto_start` remains a future lifecycle hint in Stage 4M and must not start a
transport.

Disabled draft servers may omit transport-specific runtime details, matching the
current model behavior.

## FastAPI Surface

Mount routes only when an external registry manager is explicitly provided or
enabled through bootstrap wiring.

Proposed routes under the existing gateway router prefix:

```text
GET    /external-servers
POST   /external-servers
GET    /external-servers/{server_id}
PATCH  /external-servers/{server_id}
DELETE /external-servers/{server_id}
```

Optional query:

```text
GET /external-servers?enabled=true|false
```

Responses should follow the Stage 4K/4L management style:

```json
{
  "ok": true,
  "server": {},
  "store": {"kind": "sqlite", "persistent": true}
}
```

List response:

```json
{
  "ok": true,
  "servers": [],
  "store": {"kind": "sqlite", "persistent": true}
}
```

Delete response:

```json
{
  "ok": true,
  "server_id": "research",
  "store": {"kind": "sqlite", "persistent": true}
}
```

The FastAPI layer should define Pydantic request/response models and delegate all
domain behavior to the manager.

## CLI Surface

Add commands to `mcp-unified-gateway`:

```text
list-external-servers --config <path> [--enabled true|false]
show-external-server <server_id> --config <path>
create-external-server --server-file <path|-> --config <path>
patch-external-server <server_id> --patch-file <path|-> --config <path>
delete-external-server <server_id> --config <path>
```

Until gateway config grows an explicit external-registry seed/import shape, all
Stage 4M CLI commands should require a persistent configured store. This avoids
inventing transient memory semantics that differ from the writable SQLite
registry. CLI output should be deterministic JSON. Domain errors should be
emitted on stderr with `ok: false`, `error`, and `reason_code`.

## Error Mapping

Expected reason codes:

```text
external_registry_store_unavailable -> HTTP 503
external_server_not_found -> HTTP 404
external_server_already_exists -> HTTP 409
external_server_has_credential_grants -> HTTP 409
invalid_external_server_request -> HTTP 422
invalid_external_server_patch -> HTTP 422
unexpected_external_server_delete_result -> HTTP 500
```

Transport-level JSON parsing and Pydantic request body errors should keep the
existing FastAPI/CLI behavior for malformed payloads. Manager-level validation
should produce the domain reason codes above.

## Audit Posture

When an `AuditStore` is present, successful operations should append:

- `external_server.created`
- `external_server.patched`
- `external_server.deleted`

Expected failures should append:

- `external_server.create_failed`
- `external_server.patch_failed`
- `external_server.delete_failed`
- `external_server.show_failed`

Payloads should include stable reason codes, server id, changed fields for patch
success, and no secrets. Audit failures remain best-effort for registry
management, matching the current profile-management posture. Future execution
and credential-bearing slices can impose stricter audit requirements.

## Concurrency And Runtime State

Registry mutations update persistence only. They do not mutate an already
started `ExternalFederationManager` in memory.

If a running gateway wants new definitions to take effect, a later lifecycle
slice should add explicit refresh/reload behavior. Stage 4M should document this
clearly in CLI and API behavior so callers do not assume that `create` or
`patch` immediately changes active virtual tools.

## Security Notes

- Creating an enabled stdio definition is not equivalent to process execution.
- Registry visibility is not execution authority.
- External tools remain non-executable until profile policy, credential grants,
  external server grants, RBAC ceilings, and lifecycle health allow them.
- `credential_slots` are names only. Secret material must not appear in server
  documents, audit payloads, CLI output, or logs.
- `command`, `cwd`, and env names are stored for future process-policy
  validation; Stage 4M should not claim they are safe to execute.

## Validation Strategy

Focused tests should cover:

- manager list/show/create/patch/delete success paths
- duplicate create conflict
- malformed create and patch payloads
- server id slug validation and name normalization
- empty patch and unsupported patch fields
- transport revalidation when patching enabled servers, including websocket URL
  scheme checks
- `auto_start` persistence without lifecycle side effects
- delete not found
- delete blocked by credential grants when grant information is available
- SQLite persistence round trips and caller-owned returned payloads
- FastAPI success/error status mappings
- CLI success/error JSON and persistent-store requirement
- package import boundary: `mcp_unified` still has no `tldw_Server_API` imports

Validation commands for implementation planning should include:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_registry_management.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -q
python -m bandit -r mcp_unified/gateway mcp_unified/storage mcp_unified/interfaces -f json -o /tmp/bandit_mcp_stage4m.json
git diff --check
```

## Implementation Boundaries

Expected touched areas for the implementation plan:

- `mcp_unified/gateway/external_registry.py` or equivalent new manager module
- `mcp_unified/gateway/fastapi.py`
- `mcp_unified/gateway/cli.py`
- `mcp_unified/gateway/config.py` if storage bundle wiring needs a named
  external-registry manager factory
- `mcp_unified/interfaces/storage.py` for the persistent-store atomic
  `create_server` capability and any guarded-delete helper selected during
  planning
- `mcp_unified/storage/sqlite.py` for atomic create and any guarded-delete helper
- focused tests under `tldw_Server_API/app/core/MCP_unified/tests/`

The implementation should not touch host MCP Hub services unless a later
host-adapter slice is explicitly planned.

## Acceptance Criteria

- The design is tracked by Backlog task `TASK-578`.
- The spec defines a manager-first Stage 4M scope for external server registry
  management over CLI and FastAPI.
- The spec reuses existing package storage primitives and preserves the
  `mcp_unified` import boundary.
- The spec defines deterministic error mappings, audit behavior, concurrency
  expectations, and focused verification.
- The spec explicitly defers real external process lifecycle, credential secret
  handling, install/update flows, profile grant editing, and UI work.
