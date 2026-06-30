# MCP Unified Stage 4N External Lifecycle Runtime Design

Date: 2026-05-31
Status: Approved for spec review
Backlog: TASK-581

## Summary

Stage 4N adds the standalone gateway runtime layer for managed external MCP
servers after Stage 4M made registry rows editable. The slice should give the
package real lifecycle semantics for configured external servers: start, stop,
restart, refresh discovery, reconcile against registry changes, and report
health/state. It should also add the credential-secret and install/update
contracts needed for safe operation without introducing a package-manager or
marketplace implementation yet.

The core design is a package-owned runtime manager that operates over
`ExternalRegistryStore`, `AuditStore`, optional credential broker interfaces,
and injected transport/installer adapters. The package remains free of
`tldw_Server_API` imports. Host applications may inject real stdio/WebSocket
transport adapters, while package tests use deterministic fake adapters.

## Goals

- Add a package-local external server runtime manager for start, stop, restart,
  refresh, reconcile, list status, and virtual-tool execution.
- Preserve the existing `mcp_unified` import boundary: no package code may
  import `tldw_Server_API`.
- Reuse `ExternalServerDefinition`, `ExternalRegistryStore`, `CredentialGrant`,
  `AuditEvent`, and existing federation model contracts where possible.
- Keep registry management and runtime lifecycle separate: registry CRUD remains
  owned by `GatewayExternalRegistryManager`; lifecycle owns active transport
  state.
- Resolve credential material only at execution time through an explicit
  broker interface and expose only safe key-name summaries in public metadata.
- Add install/update flow contracts and gateway status responses that are safe
  by default when no installer adapter is configured.
- Keep real process/network side effects behind injected adapters with explicit
  lifecycle methods and tests.

## Non-Goals

- No third-party MCP package-manager, marketplace, dependency resolver, or
  automatic install/update execution in this slice.
- No client-facing stdio MCP gateway entrypoint.
- No secret values stored in `ExternalServerDefinition`, profile documents,
  audit events, logs, or long-lived transport state.
- No replacement for `tldw_server` MCP Hub external-server authority.
- No WebUI changes.
- No broad refactor of the existing host external federation manager beyond
  optional compatibility adapters needed to preserve current behavior.

## Current Foundation

Stage 4M added gateway registry management over the package store:

- `mcp_unified.gateway.external_registry.GatewayExternalRegistryManager`
- `mcp_unified.interfaces.storage.ExternalRegistryStore`
- `mcp_unified.storage.models.ExternalServerDefinition`
- `mcp_unified.storage.sqlite.SQLiteMCPStore`

The package also has a non-spawning federation shell:

- `mcp_unified.federation.manager.ExternalFederationManager`
- `mcp_unified.federation.transports.ExternalFederationTransport`
- `mcp_unified.federation.transports.FakeExternalTransport`

The real upstream transport implementation currently lives in the host tree:

- `tldw_Server_API.app.core.MCP_unified.external_servers.manager.ExternalServerManager`
- host stdio and WebSocket transport adapters
- host credential broker and MCP Hub external registry services

That code proves the needed behavior, but the stdio adapter depends on
`tldw_Server_API.app.core.Agent_Client_Protocol.stdio_client`. Stage 4N should
not move that dependency into `mcp_unified`. Instead, the package defines the
neutral lifecycle contract and host adapters can bridge to it.

## Approach Options

### Option A: Move Host Real Adapters Into `mcp_unified`

This would immediately expose the existing stdio/WebSocket implementations from
the package.

Tradeoff: it moves too much host coupling into the package because stdio still
depends on `ACPStdioClient` under `tldw_Server_API`. It risks breaking the import
boundary that earlier stages established.

### Option B: Add Package Runtime Manager With Injected Adapters

Create a package-owned lifecycle manager that uses injected transport and
installer factories. The manager owns state transitions, locking, audit, health,
reconcile, credential-broker calls, and virtual-tool execution. Real transports
can be supplied by hosts or later package extras.

Tradeoff: this creates a small abstraction layer before moving real adapters,
but it keeps the package safe and testable.

### Option C: Build Install/Update Flows First

Add install/update commands and endpoints before lifecycle and credential
contracts are complete.

Tradeoff: this is the wrong dependency order. Install/update actions are
side-effectful and need lifecycle, process policy, audit, and secret boundaries
before they can be safe.

## Recommended Approach

Use Option B.

Stage 4N should add lifecycle and credential contracts first, then expose
install/update as disabled-by-default adapter hooks. This gives gateway users and
future UI work a stable runtime status/control surface without pretending that
arbitrary MCP server installation is safe.

## Runtime Manager Contract

Add a package-owned runtime manager, for example:

```python
class GatewayExternalRuntimeManager:
    async def start_server(server_id: str) -> dict: ...
    async def stop_server(server_id: str) -> dict: ...
    async def restart_server(server_id: str) -> dict: ...
    async def refresh_server(server_id: str | None = None) -> dict: ...
    async def reconcile(server_id: str | None = None) -> dict: ...
    async def list_runtime_servers() -> dict: ...
    async def list_virtual_tools() -> list[VirtualExternalTool]: ...
    async def execute_virtual_tool(...) -> FederatedToolResult: ...
```

The manager should own:

- per-server runtime state
- active transport instances
- discovered virtual-tool cache
- lifecycle locks
- graceful stop behavior that continues across individual close failures
- reconciliation against current enabled registry definitions
- lifecycle/discovery/execution audit events
- safe public status payloads
- credential broker calls for execution

The manager should not own:

- registry CRUD
- credential grant CRUD
- profile CRUD
- installer execution implementation
- host-specific MCP Hub precedence rules

## Transport Adapter Contract

Use or extend the package transport protocol so every real or fake adapter can
provide:

```python
server_id: str
transport_name: str
async connect() -> None
async close() -> None
async health_check() -> dict[str, bool]
async list_tools() -> list[ExternalToolDefinition]
async call_tool(tool_name, arguments, *, context=None, runtime_auth=None) -> ExternalToolCallResult
```

Actual stdio process spawning and WebSocket connections are allowed only through
adapters supplied to the runtime manager. The default package factory should be
safe: fake or unsupported until a caller opts into real transport extras or host
adapters.

## Lifecycle Semantics

`start_server(server_id)`:

- loads the current server definition from `ExternalRegistryStore`
- rejects unknown, disabled, or invalid server definitions with reason codes
- builds a transport through the injected factory
- connects the transport
- discovers tools
- stores the transport and virtual tools only after successful connect and
  discovery
- closes partially connected transports on failure
- emits lifecycle and discovery audit events

`stop_server(server_id)`:

- closes the active transport if present
- clears virtual tools for that server
- leaves the registry definition untouched
- returns idempotent `already_stopped` for inactive known servers
- captures close/audit failures in the response without leaking secrets

`restart_server(server_id)`:

- stops the active transport
- reloads the registry definition
- starts it again
- returns combined stop/start status and reason codes

`refresh_server(server_id | None)`:

- re-runs `list_tools` for active transports only
- clears tools for servers whose discovery fails
- keeps other servers active
- returns per-server error maps

`reconcile(server_id | None)`:

- reloads enabled registry definitions
- starts newly enabled `auto_start` servers
- stops disabled/deleted active servers
- replaces active transports when material transport fields change
- refreshes unchanged active servers
- isolates partial failures to affected servers

## State And Reason Codes

Runtime status rows should include:

```text
id
name
transport
configured
active
status: stopped | starting | healthy | degraded | unhealthy | stopping
tool_count
last_error
checks
lifecycle
install
update
```

Initial reason codes should include:

- `external_server_started`
- `external_server_stopped`
- `external_server_already_stopped`
- `external_server_restarted`
- `external_server_refreshed`
- `external_server_reconciled`
- `external_server_not_found`
- `external_server_disabled`
- `external_server_start_failed`
- `external_server_stop_failed`
- `external_server_discovery_failed`
- `external_server_transport_unavailable`
- `external_server_install_not_configured`
- `external_server_update_not_configured`
- `external_server_install_unsupported`
- `external_server_update_unsupported`
- `required_credential_grant_missing`
- `credential_broker_unavailable`

Execution should fail closed when required policy, credential, or transport
state is unavailable.

## Credential-Secret Handling

Add a package-local credential broker protocol for execution-time material:

```python
class ExternalCredentialBroker(Protocol):
    async def resolve_external_credential(
        self,
        *,
        server: ExternalServerDefinition,
        tool_name: str,
        arguments: dict[str, Any],
        effective_policy: Any,
        context: Any = None,
    ) -> BrokeredExternalCredential | None: ...
```

Brokered credentials are ephemeral and may contain headers/env values for one
transport call. Public metadata may include only:

- credential mode/source strings
- injected header names
- injected environment variable names
- grant ids or slot names only when they are not secret values

The runtime manager must not:

- persist credential values
- include credential values in audit events
- include credential values in log messages
- mutate long-lived server definitions or active transport config with
  credential values

If a server requires credential slots and no broker/grant can satisfy them,
execution must deny with `required_credential_grant_missing` or
`credential_broker_unavailable`.

## Install And Update Contracts

Stage 4N should define safe install/update contracts without executing package
manager commands by default.

Recommended protocol:

```python
class ExternalServerInstaller(Protocol):
    async def install_server(server: ExternalServerDefinition, *, context=None) -> dict: ...
    async def update_server(server: ExternalServerDefinition, *, context=None) -> dict: ...
    async def get_status(server: ExternalServerDefinition) -> dict: ...
```

When no installer is configured:

- install requests return `external_server_install_not_configured`
- update requests return `external_server_update_not_configured`
- status reports `available: false`

When an installer is configured but cannot support the server:

- install returns `external_server_install_unsupported`
- update returns `external_server_update_unsupported`

The runtime manager may expose these as methods and FastAPI/CLI surfaces, but no
default adapter may run shell commands, package managers, or network downloads.

Future installer adapters must require explicit command allowlists, no shell,
bounded cwd, audit, and operator-visible status before execution.

## Gateway Surface

Add lifecycle routes under the existing package gateway when an external runtime
manager is configured:

```text
GET  /external-servers/runtime
POST /external-servers/{server_id}/start
POST /external-servers/{server_id}/stop
POST /external-servers/{server_id}/restart
POST /external-servers/refresh
POST /external-servers/{server_id}/refresh
POST /external-servers/reconcile
POST /external-servers/{server_id}/reconcile
POST /external-servers/{server_id}/install
POST /external-servers/{server_id}/update
```

CLI support should not imply durable control over a separate running gateway
process. A one-shot CLI process can safely expose registry-backed runtime
inspection, install/update not-configured responses, and foreground smoke
operations only when an injected runtime factory is explicitly available. Durable
start/stop/restart of a running gateway belongs to the FastAPI/in-process manager
surface until a future daemon-control client exists.

If CLI commands are added in this slice, they must use the same config/bootstrap
helper and make process-lifetime semantics explicit:

```text
external-server-runtime-list
external-server-start
external-server-stop
external-server-restart
external-server-refresh
external-server-reconcile
external-server-install
external-server-update
```

Default CLI lifecycle commands should return deterministic
`external_runtime_not_configured` or install/update not-configured results rather
than pretending to control another process.

If the runtime manager is not configured, route mounting should be opt-in and
fail fast when explicitly enabled without dependencies.

## Integration With Existing Host Runtime

The existing `tldw_server` external manager should keep current behavior. Stage
4N may add compatibility adapters that let host code supply its real transport
factory to the package manager later, but it should not make MCP Hub lose
authority over managed external servers.

Host compatibility requirements:

- `tldw_Server_API.app.core.MCP_unified.external_servers` imports keep working.
- Existing host external federation tests continue passing.
- MCP Hub external registry and credential broker services remain authoritative
  in `tldw_server`.
- Package code remains import-clean.

## Testing Strategy

Package-focused tests should cover:

- start connects/discovers tools and records status
- start failure closes partial transport and leaves no active tools
- stop is idempotent and clears virtual tools
- restart reloads changed registry data
- refresh isolates discovery failure to one server
- reconcile starts enabled auto-start servers and stops disabled/deleted active
  servers
- execution injects brokered credentials only for the call and redacts values
  from metadata/audit
- missing broker or grant denies required credential slots
- install/update return deterministic not-configured results by default
- configured installer unsupported results are surfaced without side effects
- FastAPI and CLI route/command mappings
- package import-boundary remains clean

Host compatibility tests should cover:

- existing host external manager behavior still passes
- existing host credential broker runtime tests still pass
- no new package import depends on `tldw_Server_API`

Verification should include focused pytest, `git diff --check`, and Bandit on
touched package Python files.

## Risks And Mitigations

- Hidden host imports in package lifecycle code: keep real transports injectable
  and extend import-boundary tests.
- Lifecycle manager becomes a second registry manager: keep CRUD out of the
  runtime manager.
- Secrets leak through status/audit/logging: centralize brokered credential
  public-summary logic and test with sentinel secret values.
- Install/update scope expands: only define adapter hooks and disabled default
  responses in this slice.
- Process-spawning safety gaps: no default package transport should spawn
  unless a later adapter implements executable allowlists, cwd validation,
  minimal env, resource limits, and audit.

## Acceptance Criteria

- A package-owned external runtime manager supports start, stop, restart,
  refresh, reconcile, status, and virtual-tool execution over injected
  transports.
- The standalone gateway can expose lifecycle controls only when configured with
  the runtime manager.
- Credential values are never persisted or returned; execution uses brokered
  ephemeral credentials when granted.
- Install/update flows are represented with deterministic disabled/unsupported
  outcomes and no default side effects.
- Existing host behavior and package import-boundary tests remain compatible.
- Focused tests, Bandit on touched Python package scope, and diff whitespace
  checks pass before PR.
