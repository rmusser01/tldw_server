# MCP Unified Upstream Stdio Transport Design

Date: 2026-06-01
Status: Approved for implementation
Backlog: TASK-582

## Summary

Add a package-owned upstream stdio transport for configured external MCP
servers. This fills the gap left after Stage 4N: the standalone gateway runtime
can already manage lifecycle state through injected `ExternalFederationTransport`
instances, but `mcp_unified` still lacks a real transport that launches a stdio
MCP server process without importing host `tldw_Server_API` code.

The transport should live inside `mcp_unified.federation`, implement the existing
`ExternalFederationTransport` protocol, use `asyncio.create_subprocess_exec`
with newline-delimited JSON-RPC, and keep all secrets out of logs, persisted
state, audit payloads, and exception messages.

## Goals

- Launch configured `transport="stdio"` external MCP server commands from the
  standalone package.
- Implement connect, discovery, tool calls, health checks, timeout handling,
  and close cleanup through the existing transport protocol.
- Keep subprocess execution shell-free and environment handling explicit.
- Support brokered runtime credentials only through per-call JSON-RPC metadata;
  do not mutate a long-lived process environment after start.
- Preserve the package boundary: no imports from `tldw_Server_API`.
- Add focused tests for validation, subprocess round trips, failures, cleanup,
  timeout behavior, environment allowlisting, and credential redaction.

## Non-Goals

- No client-facing stdio gateway changes. `mcp_unified.gateway.stdio` already
  handles stdin/stdout for front-end clients.
- No WebSocket upstream transport in this slice.
- No package-manager install/update implementation.
- No shell command parsing or expansion.
- No durable daemon/process supervisor. The transport owns one subprocess for
  the in-process runtime manager.
- No per-call process environment mutation. A subprocess environment is
  process-scoped, so per-call env credentials would require a separate
  short-lived adapter model and are intentionally deferred.

## Existing Contracts

The transport plugs into:

- `mcp_unified.storage.models.ExternalServerDefinition`
- `mcp_unified.federation.transports.ExternalFederationTransport`
- `mcp_unified.federation.models.ExternalToolDefinition`
- `mcp_unified.federation.models.ExternalToolCallResult`
- `mcp_unified.federation.models.BrokeredExternalCredential`
- `mcp_unified.gateway.external_runtime.GatewayExternalRuntimeManager`

`ExternalServerDefinition.command` already stores an argv list. The first item
is the executable and the remaining items are arguments. `cwd` is optional.
`env_allowlist` stores names of current-process environment variables that may
be inherited by the child process.

## Transport Design

Create `mcp_unified/federation/stdio_transport.py` with:

- `StdioExternalTransport`
- `StdioExternalTransportError`
- `create_external_transport`

`StdioExternalTransport(server, request_timeout_s=30.0, connect_timeout_s=30.0)`
validates:

- `server.transport == "stdio"`
- `server.command` is non-empty
- all command items are non-empty strings
- `server.cwd`, when provided, resolves to an existing directory

Process launch uses:

```python
asyncio.create_subprocess_exec(
    *server.command,
    stdin=asyncio.subprocess.PIPE,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
    cwd=resolved_cwd,
    env=allowed_env,
)
```

There is no shell. The child environment is built from only allowlisted names in
`os.environ`. Static secret values are not read from `ExternalServerDefinition`.

## JSON-RPC Behavior

Requests and responses use JSON-RPC 2.0 with one compact JSON object per line.
The transport sends:

- `initialize` during `connect`
- `tools/list` during discovery
- `tools/call` for execution
- `ping` for health when initialized and the process is still running

Response normalization follows the host adapter behavior:

- `tools/list` accepts `{"tools": [...]}` and filters invalid tool rows.
- invalid or missing `inputSchema` falls back to `{"type": "object"}`.
- invalid metadata falls back to `{}`.
- `tools/call` maps MCP `content` and `isError` into `ExternalToolCallResult`.
- JSON-RPC error responses become `ExternalToolCallResult(is_error=True)` for
  tool calls and structured transport errors for lifecycle/discovery methods.

The first implementation serializes requests with an async lock. That keeps the
client small and deterministic while still fitting the runtime manager's awaited
transport interface. Concurrent pipelining can be added later without changing
the public protocol.

## Runtime Credentials

`BrokeredExternalCredential` can contain `headers`, `env`, and public metadata.
For stdio, headers and env cannot be injected as HTTP headers or process env per
call. The transport therefore supports request metadata only:

```json
{
  "name": "tool",
  "arguments": {},
  "_meta": {
    "mcp_unified_runtime_auth": {
      "headers": {"Authorization": "..."},
      "env": {"TOKEN": "..."},
      "metadata": {"credential_source": "..."}
    }
  }
}
```

This intentionally sends credentials only to the upstream server for the one
tool call. The transport must not log, persist, include in health data, or echo
secret values in exception messages. Runtime manager public summaries continue
to expose only credential key names.

## Error Handling And Cleanup

`StdioExternalTransportError` includes a safe `reason_code`, `server_id`, and
optional safe metadata. Error messages include method names and reason codes but
not request bodies, environment values, stderr contents, or credential values.

`close()` is idempotent:

- close stdin when possible
- terminate the process
- wait up to a short shutdown timeout
- kill on timeout
- cancel reader/stderr tasks
- resolve pending requests with safe transport errors

`connect()` must close partial transports if initialize fails. `health_check()`
must not throw for ordinary process exits; it returns configured/connected/
initialized booleans and marks exited processes disconnected.

## Tests

Add `tldw_Server_API/app/core/MCP_unified/tests/test_stdio_external_transport.py`.
The tests use small temporary Python stdio MCP server scripts so no network or
third-party MCP server is required.

Coverage:

- validation rejects non-stdio definitions, empty commands, and bad cwd values
- environment inheritance includes only allowlisted variables
- connect initializes and reports healthy process state
- list_tools normalizes tool definitions
- call_tool round-trips arguments and maps success/error results
- runtime_auth appears in the outbound JSON-RPC `_meta` for the call but secret
  values do not appear in transport exceptions
- request timeout raises a safe structured error
- process exit updates health and pending calls fail safely
- close is idempotent and cleans up the subprocess
- package boundary tests confirm `mcp_unified.federation.stdio_transport` does
  not import `tldw_Server_API`

## Rollout

This slice only adds the transport and a package factory helper. Gateway users
can opt in by passing `create_external_transport` as the runtime manager's
`transport_factory`. A later slice can wire this factory into CLI/config
bootstrap defaults after policy and install/update UX are ready.
