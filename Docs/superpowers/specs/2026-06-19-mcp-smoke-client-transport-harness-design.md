# MCP Smoke Client Transport Harness Design

## Context

`TASK-2281` was opened to design and implement LSP-backed MCP code intelligence
tools. Before adding more MCP tool surface area, the server needs a small client
that can run the MCP gateway through realistic protocol flows and make regressions
easy to diagnose across transports.

The repository already has strong unit and integration coverage for protocol
handlers, gateway runtimes, FastAPI routes, WebSocket paths, stdio helpers,
profiles, resources, prompts, external federation, and tool reporting. The
missing piece is an operator/developer smoke client that drives a server the way
a real MCP client does, across deterministic in-process tests and live transports.

## Goal

Add a reusable MCP smoke client harness that can validate a standalone or
tldw-hosted MCP server through standard JSON-RPC scenarios. The same scenario
runner should work against:

- an in-process runtime or test app for deterministic CI;
- a live HTTP endpoint;
- a live WebSocket endpoint;
- a stdio subprocess that speaks newline-delimited JSON-RPC.

The harness should produce a concise terminal summary and an optional structured
JSON report that is safe to attach to PRs, CI artifacts, or Backlog task notes.

## Non-Goals

- Do not build a complete MCP SDK.
- Do not replace existing pytest suites.
- Do not create a graphical UI.
- Do not execute arbitrary tools from user-provided scenario files without an
  explicit allowlist.
- Do not require a live server for default CI.
- Do not implement LSP tools in this task. The LSP design should consume this
  harness after the baseline exists.

## Design Principles

1. **One scenario model, multiple transports.** Protocol expectations should be
   written once and executed through transport adapters.
2. **Deterministic by default.** The in-process mode is the default CI path.
   Live transports are opt-in smoke modes.
3. **Safe output.** Reports include method names, request ids, statuses, elapsed
   time, reason codes, and bounded error details. They do not include secrets,
   raw environment variables, bearer tokens, absolute local paths, or full file
   contents.
4. **Protocol-first.** The harness should validate MCP JSON-RPC behavior before
   asserting product-specific conveniences.
5. **Profile-aware.** Scenarios should be able to run under a named profile and
   verify profile-filtered tool visibility.
6. **Extensible by feature.** LSP, WebFetch, Git, CodeGraph, or filesystem
   scenarios should be additive modules on top of the baseline client.
7. **Capability-adaptive.** Optional resources, prompts, and fixture tools should
   be skipped with explicit reason codes unless strict mode requests them.

## Architecture

### Components

`McpSmokeClient`
: Small async client facade with `request()`, `notify()`, `initialize()`,
  `list_tools()`, `call_tool()`, `list_resources()`, `read_resource()`,
  `list_prompts()`, and `get_prompt()` helpers.

`McpSmokeTransport`
: Protocol implemented by transport adapters. It should expose `start()`,
  `request(payload)`, `notify(payload)`, and `close()` methods.

`ScenarioRunner`
: Executes ordered scenario steps, captures request ids, validates responses,
  records timings, and returns a `SmokeReport`.

`SmokeReport`
: JSON-serializable report with run metadata, transport metadata, per-step
  outcome, elapsed time, result summary, error reason code, and sanitized detail.

`ScenarioCatalog`
: Built-in scenarios for baseline MCP behavior. Later feature-specific scenario
  files can register additional scenario groups.

### Proposed File Layout

- `mcp_unified/smoke/__init__.py`
- `mcp_unified/smoke/client.py`
- `mcp_unified/smoke/transports.py`
- `mcp_unified/smoke/scenarios.py`
- `mcp_unified/smoke/reporting.py`
- `mcp_unified/smoke/cli.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py`
- `tldw_Server_API/app/core/MCP_unified/tests/fixtures/smoke_stdio_server.py`
- `Docs/MCP/Unified/Smoke_Client.md`

The code lives in `mcp_unified` because the standalone gateway package should be
able to use it without importing tldw Server internals. tldw-specific tests may
build an in-process test app/runtime using existing support helpers.

## Transport Modes

### In-Process

The default mode should run without network sockets or subprocesses. Two adapter
forms are useful:

- `InProcessGatewayTransport` for a `GatewayRuntime` using
  `mcp_unified.gateway.jsonrpc.handle_jsonrpc()`.
- `InProcessFastApiTransport` for a FastAPI `TestClient`, using
  `/api/v1/mcp/request` or the standalone gateway route mounted in a test app.

This mode is the CI baseline because it is deterministic, fast, and does not
need port allocation.

### Live HTTP

The HTTP adapter targets an already-running MCP endpoint, for example:

```text
http://127.0.0.1:8000/api/v1/mcp/request
```

It should support:

- bearer token or API key headers loaded from explicit CLI args or environment;
- profile headers such as `x-mcp-profile`;
- timeout and retry settings;
- session id capture from initialize responses when the server returns one.

The adapter must never print credentials in reports or logs.

HTTP notification behavior should accept HTTP 204 or a JSON-RPC no-response
equivalent, depending on the endpoint implementation. The adapter should treat
both as a passed notification when no JSON-RPC response object is expected.

### Live WebSocket

The WebSocket adapter targets an already-running MCP WebSocket endpoint, for
example:

```text
ws://127.0.0.1:8000/api/v1/mcp/ws?client_id=smoke
```

It should support:

- optional auth headers where the client library supports them;
- profile selection through query params or headers where supported;
- request/response correlation by JSON-RPC id;
- notification-only calls that expect no response;
- bounded receive timeouts with useful diagnostics.

The adapter should keep one WebSocket connection open for the scenario so it can
exercise initialization, notifications, request correlation, and connection
cleanup as a client would.

### Stdio Subprocess

The stdio adapter starts a configured command and exchanges one JSON-RPC object
per line over stdin/stdout. It should support:

- command plus argument array, not a shell string;
- explicit `cwd`;
- explicit allowlisted environment variables;
- startup timeout and per-request timeout;
- stderr capture with bounded redaction for diagnostics;
- process cleanup on success, failure, timeout, or cancellation.

This adapter is for validating packaged standalone gateway entrypoints and
external MCP servers. It must not become a general shell execution path.

Notification steps over stdio should assert that no response line is produced
within a short bounded timeout, then continue using the same subprocess.

## Baseline Scenario Set

The initial built-in scenario group should include:

1. `initialize`
   - request server capabilities;
   - assert `serverInfo.name` and capability shape are present;
   - capture session metadata when available.

2. `notifications/initialized`
   - send as a notification;
   - assert no response is returned.

3. `tools/list`
   - assert response contains a tool list;
   - assert every tool has `name`, `description`, and `inputSchema`;
   - optionally assert expected tools are present.

4. `ping`
   - send a request-form ping;
   - assert a successful response or accepted empty result according to the
     server contract.

5. `tools/call` read-only happy path
   - call a configured harmless tool such as `profile.tools.list`,
     `tool_search`, `tool_describe`, or a fixture tool;
   - assert JSON-RPC success and structured content/result.

6. `tools/call` unknown tool
   - call a definitely missing tool;
   - assert JSON-RPC error with a stable error code and sanitized details.

7. Profile-filtered visibility
   - run `tools/list` under a profile when configured;
   - assert denied tools are absent and discovery tools remain available when
     policy allows them.

8. Resources
   - call `resources/list`;
   - if resources exist, call `resources/read` for one safe resource;
   - if none exist, assert the empty response is well formed.

9. Prompts
   - call `prompts/list`;
   - if prompts exist, call `prompts/get` with safe arguments;
   - if none exist, assert the empty response is well formed.

10. JSON-RPC batch
    - send a small batch containing `ping` and `tools/list`;
    - assert response correlation and per-item status.

11. Malformed request
    - send an invalid JSON-RPC request shape;
    - assert an error response without crashing the connection.

12. Policy denial
    - when a fixture/runtime is available, attempt one intentionally denied
      tool call or write path;
    - assert denial uses the expected structured error contract.

The scenario runner should allow `--strict` and `--best-effort` modes. Strict
mode fails when optional resource/prompt/tool expectations are unavailable.
Best-effort mode records skipped optional steps with reason codes.

## CLI Shape

The CLI should be small and JSON-friendly:

```bash
python -m mcp_unified.smoke.cli inprocess --scenario baseline --json-report /tmp/mcp-smoke.json
python -m mcp_unified.smoke.cli http --url http://127.0.0.1:8000/api/v1/mcp/request --api-key-env SINGLE_USER_API_KEY
python -m mcp_unified.smoke.cli websocket --url ws://127.0.0.1:8000/api/v1/mcp/ws?client_id=smoke
python -m mcp_unified.smoke.cli stdio --command python --arg tldw_Server_API/app/core/MCP_unified/tests/fixtures/smoke_stdio_server.py
```

Default test and CI invocations should use `inprocess`. Live HTTP, WebSocket,
and stdio subprocess commands are explicit operator smoke checks and should be
marked or documented as environment-dependent.

The stdio example should initially target a test fixture subprocess. If the
standalone gateway later exposes a packaged stdio entrypoint, the docs can add a
second example for that executable without changing the transport contract.

Exit codes:

- `0`: all required steps passed;
- `1`: required scenario step failed;
- `2`: invalid client configuration or CLI arguments;
- `3`: transport startup/connect failure;
- `4`: scenario skipped because required capability was unavailable in strict
  mode.

## Report Contract

Each report should include:

- `ok`;
- `transport`;
- `server_info`;
- `started_at`;
- `duration_ms`;
- `scenario`;
- `profile_id`;
- `steps`;
- `summary`.

Each step should include:

- `name`;
- `method`;
- `request_id`;
- `status`: `passed`, `failed`, or `skipped`;
- `elapsed_ms`;
- `error_code`;
- `reason_code`;
- `message`;
- `result_summary`.

`result_summary` must be bounded. For tool calls, summarize content item counts,
tool names, resource counts, prompt counts, or error reason codes rather than
dumping full payloads.

## Security And Safety

- Auth material may be read only from explicit CLI flags or named environment
  variables.
- Reports must redact auth headers, environment values, and subprocess command
  environment.
- Stdio command execution must use argv arrays and `asyncio.create_subprocess_exec`
  or equivalent, never `shell=True`.
- Live transport modes must require explicit URLs.
- In-process mode must not open network sockets.
- The client should cap response size and fail with a structured
  `response_too_large` diagnostic.
- The client should cap stderr capture for stdio subprocess mode.
- Scenario files, if added later, must be declarative and must not allow
  arbitrary Python imports, shell fragments, or unreviewed tool calls by default.

## Relationship To LSP Tools

The LSP tool work should depend on this harness. Once the baseline client is in
place, `TASK-2281` can add LSP-specific scenarios:

- `lsp.status` returns unavailable when no provider is configured;
- `lsp.diagnostics` returns bounded diagnostics for a fixture file when a fake
  provider is injected;
- `lsp.definition`, `lsp.references`, and `lsp.hover` obey path grants and
  return bounded workspace-relative locations;
- denied paths fail closed with structured policy errors.

This keeps LSP implementation honest without requiring a real language server in
the first LSP PR.

## Implementation Slices

1. **Core client and report model**
   - JSON-RPC id generation;
   - request/notification helpers;
   - report types and sanitizer.

2. **In-process transport and baseline scenario**
   - deterministic CI path;
   - fixture runtime or FastAPI test app;
   - focused pytest coverage.

3. **Live HTTP transport**
   - URL, headers, auth env support, session capture, timeout behavior.

4. **Live WebSocket transport**
   - request correlation, notification behavior, timeout diagnostics.

5. **Stdio subprocess transport**
   - argv-based process launch, bounded stderr, cleanup, startup failure tests.

6. **CLI and docs**
   - command parser;
   - exit codes;
   - `Docs/MCP/Unified/Smoke_Client.md`;
   - Backlog and validation closeout.

## Testing Strategy

- Unit tests for report sanitization and JSON-RPC helpers.
- In-process scenario tests with a fake `GatewayRuntime`.
- HTTP tests using FastAPI `TestClient` or an ASGI test transport.
- WebSocket tests against the existing in-process WebSocket test app.
- Stdio tests against a tiny fixture subprocess that implements initialize,
  tools/list, tools/call, resources/list, and prompts/list.
- Failure tests for malformed JSON-RPC, unknown method/tool, timeout, oversized
  response, subprocess startup failure, and redaction.
- Live transport commands should have documented manual smoke examples but should
  not block default CI unless a job explicitly provisions the target server.

Verification for the implementation PR should include:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py -q
python -m ruff check mcp_unified/smoke tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py
python -m py_compile mcp_unified/smoke/*.py
python -m bandit -r mcp_unified/smoke -f json -o /tmp/bandit_mcp_smoke_client.json
git diff --check
```

## Acceptance Criteria

- A spec and implementation plan exist before MCP LSP implementation resumes.
- The client design supports in-process, live HTTP, live WebSocket, and stdio
  subprocess transports.
- The baseline scenario covers initialization, tools, resources, prompts,
  ping, unknown tool/error behavior, JSON-RPC batch, optional profile filtering,
  and optional policy denial.
- Report output is bounded and redacted.
- Stdio subprocess execution is argv-based and never shell-based.
- LSP work has a clear follow-up hook for adding LSP-specific smoke scenarios.
