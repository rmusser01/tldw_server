# MCP Smoke Client Transport Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable MCP smoke client that exercises baseline JSON-RPC flows across deterministic in-process execution plus live HTTP, live WebSocket, and stdio subprocess transports.

**Architecture:** Add a small `mcp_unified.smoke` package with an async client facade, transport adapters, scenario runner, and report sanitizer. Keep tldw-specific fixtures and tests under `tldw_Server_API/app/core/MCP_unified/tests/` so the standalone package code does not import tldw internals.

**Tech Stack:** Python async/await, `httpx.AsyncClient` with ASGI transport, `websockets`, FastAPI gateway helpers, pytest, pytest-asyncio, Bandit.

---

## File Structure

- Create `mcp_unified/smoke/__init__.py`
  - Public exports for client, transport protocols, scenario runner, and report types.
- Create `mcp_unified/smoke/client.py`
  - JSON-RPC id generation, `McpSmokeClient`, request/notify helpers, initialize/call/list wrappers.
- Create `mcp_unified/smoke/transports.py`
  - `McpSmokeTransport` protocol, `InProcessGatewayTransport`, `InProcessFastApiTransport`, `LiveHttpTransport`, `LiveWebSocketTransport`, `StdioSubprocessTransport`.
- Create `mcp_unified/smoke/scenarios.py`
  - Built-in baseline scenario steps, strict/best-effort gating, capability-aware resource/prompt steps.
- Create `mcp_unified/smoke/reporting.py`
  - `SmokeReport`, `SmokeStepReport`, redaction, bounded result summaries, optional debug trace summaries.
- Create `mcp_unified/smoke/fixtures.py`
  - Minimal standalone fake runtime helpers that do not import tldw server internals.
- Create `mcp_unified/smoke/cli.py`
  - Argument parser, transport construction, exit codes, report output.
- Create `tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py`
  - Unit and integration coverage for client, reports, scenarios, in-process, HTTP, WebSocket, and stdio behavior.
- Create `tldw_Server_API/app/core/MCP_unified/tests/fixtures/smoke_stdio_server.py`
  - Tiny newline-delimited JSON-RPC subprocess fixture for stdio smoke tests.
- Modify `pyproject.toml`
  - Add `mcp-unified-smoke = "mcp_unified.smoke.cli:main"` after the module shape works.
- Create `Docs/MCP/Unified/Smoke_Client.md`
  - Operator docs for in-process, live HTTP, live WebSocket, stdio subprocess, auth/profile flags, reports, and exit codes.
- Modify `backlog/tasks/task-2387 - Design-MCP-smoke-client-transport-harness.md`
  - Link this plan, record validation, and close the design task when complete.
- Modify `backlog/tasks/task-2281 - Add-LSP-backed-code-intelligence-MCP-tools.md`
  - Keep the dependency on TASK-2387; do not implement LSP in this task.

## Task 1: Report Model And JSON-RPC Client Core

**Files:**
- Create: `mcp_unified/smoke/__init__.py`
- Create: `mcp_unified/smoke/client.py`
- Create: `mcp_unified/smoke/reporting.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py`

- [ ] **Step 1: Write failing tests for report sanitization**

Add tests that assert secrets, bearer tokens, environment values, absolute paths, and oversized payloads are redacted or summarized.

```python
def test_smoke_report_redacts_sensitive_details() -> None:
    from mcp_unified.smoke.reporting import summarize_result

    summary = summarize_result(
        {
            "headers": {"authorization": "Bearer secret-token"},
            "path": "/Users/example/private/file.txt",
            "content": [{"type": "text", "text": "x" * 5000}],
        }
    )

    rendered = repr(summary)
    assert "secret-token" not in rendered
    assert "/Users/example" not in rendered
    assert "content_count" in summary
```

- [ ] **Step 2: Run the focused failing tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py::test_smoke_report_redacts_sensitive_details -q
```

Expected: fail because `mcp_unified.smoke.reporting` does not exist.

- [ ] **Step 3: Implement `reporting.py`**

Add dataclasses or Pydantic-free dataclasses for:

- `SmokeStepReport`
- `SmokeReport`
- `SmokeTraceSummary`

Add helpers:

- `summarize_result(result: object, max_text_chars: int = 240) -> dict[str, object]`
- `redact_detail(value: object) -> object`
- `report_to_json(report: SmokeReport) -> dict[str, object]`

Keep summaries bounded and never include full tool arguments or full file contents.

- [ ] **Step 4: Write failing tests for `McpSmokeClient` request/notification behavior**

Use a fake transport that records payloads and returns canned JSON-RPC responses.

```python
async def test_smoke_client_request_assigns_id_and_returns_result() -> None:
    from mcp_unified.smoke.client import McpSmokeClient

    transport = _RecordingTransport(
        [{"jsonrpc": "2.0", "id": "smoke-1", "result": {"pong": True}}]
    )
    client = McpSmokeClient(transport)

    result = await client.request("ping")

    assert result == {"pong": True}
    assert transport.payloads[0]["method"] == "ping"
    assert transport.payloads[0]["id"] == "smoke-1"
```

- [ ] **Step 5: Implement `client.py`**

Implement:

- sequential stable ids such as `smoke-1`, `smoke-2`;
- `request(method, params=None)`;
- `notify(method, params=None)`;
- `initialize(client_name="mcp-smoke-client")`;
- `ping()`;
- `list_tools()`;
- `call_tool(name, arguments)`;
- `list_resources()`;
- `read_resource(uri)`;
- `list_prompts()`;
- `get_prompt(name, arguments=None)`.

Raise a small `McpSmokeClientError` for malformed JSON-RPC responses.

- [ ] **Step 6: Run Task 1 tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py -q -k "smoke_report or smoke_client"
```

Expected: pass.

- [ ] **Step 7: Commit Task 1**

```bash
git add mcp_unified/smoke tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py
git commit -m "feat: add mcp smoke client core"
```

## Task 2: In-Process Transport And Baseline Scenario Runner

**Files:**
- Modify: `mcp_unified/smoke/transports.py`
- Modify: `mcp_unified/smoke/scenarios.py`
- Create: `mcp_unified/smoke/fixtures.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py`

- [ ] **Step 1: Write failing tests for `InProcessGatewayTransport`**

Use a fake `GatewayRuntime` that advertises `echo.search`, one resource, and one prompt.

```python
async def test_inprocess_gateway_transport_runs_ping() -> None:
    from mcp_unified.smoke.client import McpSmokeClient
    from mcp_unified.smoke.fixtures import SmokeFixtureGatewayRuntime
    from mcp_unified.smoke.transports import InProcessGatewayTransport

    client = McpSmokeClient(InProcessGatewayTransport(SmokeFixtureGatewayRuntime()))

    assert await client.ping() == {"pong": True}
```

- [ ] **Step 2: Implement `McpSmokeTransport` and `InProcessGatewayTransport`**

In `transports.py`, define an async protocol with:

- `async start() -> None`
- `async request(payload: dict[str, object] | list[object]) -> object | None`
- `async notify(payload: dict[str, object]) -> None`
- `async close() -> None`

Implement gateway transport by calling `mcp_unified.gateway.jsonrpc.handle_jsonrpc()` directly.

- [ ] **Step 3: Add fixture runtime**

In `fixtures.py`, implement `SmokeFixtureGatewayRuntime` with:

- `name = "smoke-fixture-gateway"`
- `version = "0.0-test"`
- one read-only `echo.search` tool;
- one safe resource `resource://smoke/doc`;
- one safe prompt `smoke.review`;
- an optional denied tool path for policy tests.

- [ ] **Step 4: Write failing tests for baseline scenario**

```python
async def test_baseline_scenario_passes_in_best_effort_mode() -> None:
    from mcp_unified.smoke.scenarios import run_baseline_scenario
    from mcp_unified.smoke.fixtures import SmokeFixtureGatewayRuntime
    from mcp_unified.smoke.transports import InProcessGatewayTransport

    report = await run_baseline_scenario(
        InProcessGatewayTransport(SmokeFixtureGatewayRuntime()),
        mode="best_effort",
    )

    assert report.ok is True
    assert {step.name for step in report.steps} >= {"initialize", "tools/list", "ping"}
```

- [ ] **Step 5: Implement `scenarios.py`**

Implement ordered scenario steps:

1. `initialize`
2. `notifications/initialized`
3. follow-up `ping`
4. `tools/list`
5. safe read-only `tools/call`
6. unknown tool
7. optional profile-filtered visibility
8. capability-gated resources
9. capability-gated prompts
10. JSON-RPC batch
11. malformed request
12. optional policy denial

Ensure `notifications/initialized` does not pass on no-response alone; it must be followed by a successful request on the same transport.

- [ ] **Step 6: Add `InProcessFastApiTransport`**

Use `httpx.AsyncClient` with `httpx.ASGITransport(app=app)` and a configurable request path. Do not use sync `TestClient` inside the async client flow.

- [ ] **Step 7: Run Task 2 tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py -q -k "inprocess or baseline"
```

Expected: pass.

- [ ] **Step 8: Commit Task 2**

```bash
git add mcp_unified/smoke tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py
git commit -m "feat: add mcp smoke baseline scenario"
```

## Task 3: Live HTTP Transport And Retry Policy

**Files:**
- Modify: `mcp_unified/smoke/transports.py`
- Modify: `mcp_unified/smoke/cli.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py`

- [ ] **Step 1: Write failing tests for HTTP transport success**

Test against an ASGI app using `create_gateway_app()` and `InProcessFastApiTransport` first, then test `LiveHttpTransport` with `httpx.MockTransport`.

```python
async def test_live_http_transport_sends_profile_header() -> None:
    seen_headers: list[str | None] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(request.headers.get("x-mcp-profile"))
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": "smoke-1", "result": {"pong": True}})

    transport = LiveHttpTransport(
        "http://mcp.test/request",
        profile_id="reviewer",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    assert await transport.request({"jsonrpc": "2.0", "id": "smoke-1", "method": "ping"}) == {
        "jsonrpc": "2.0",
        "id": "smoke-1",
        "result": {"pong": True},
    }
    assert seen_headers == ["reviewer"]
```

- [ ] **Step 2: Implement `LiveHttpTransport`**

Support:

- explicit URL;
- bearer token and API key values passed by caller;
- `--api-key-env` and `--bearer-token-env` resolution in CLI later;
- `x-mcp-profile` header;
- timeout;
- HTTP 204 as notification success;
- JSON-RPC response body parsing.

- [ ] **Step 3: Write failing retry-policy tests**

Add a test proving transmitted `tools/call` is not replayed by default when the server disconnects or returns a retryable 5xx after receiving the body.

- [ ] **Step 4: Implement conservative retry policy**

Default behavior:

- no automatic retry for `tools/call`, `resources/read`, or `prompts/get`;
- optional retry for connection setup or explicitly idempotent methods only;
- report `transport_retry_skipped_non_idempotent` when retry is suppressed.

- [ ] **Step 5: Run Task 3 tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py -q -k "http or retry"
```

Expected: pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add mcp_unified/smoke tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py
git commit -m "feat: add mcp smoke http transport"
```

## Task 4: Live WebSocket Transport

**Files:**
- Modify: `mcp_unified/smoke/transports.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py`

- [x] **Step 1: Write failing tests for WebSocket request correlation**

Use `websockets.serve()` on `127.0.0.1` with an ephemeral port. The fixture server should return responses out of order for two requests so the transport must correlate by id.

- [x] **Step 2: Implement `LiveWebSocketTransport`**

Use the `websockets` package already present in project dependencies. Support:

- URL;
- optional auth headers where supported;
- profile query parameter fallback;
- one connection per scenario;
- request/response id correlation;
- notification sends with no response expectation;
- receive timeout diagnostics.

- [x] **Step 3: Write failing tests for notification suppression**

Send a notification and then a `ping` over the same connection. Assert only the `ping` response is returned.

- [x] **Step 4: Implement notification-safe receive loop**

Keep a pending response map. Ignore unrelated notifications from the server only if they are valid server-side notifications; surface malformed frames as transport errors.

- [x] **Step 5: Run Task 4 tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py -q -k "websocket"
```

Expected: pass.

- [x] **Step 6: Commit Task 4**

```bash
git add mcp_unified/smoke tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py
git commit -m "feat: add mcp smoke websocket transport"
```

## Task 5: Stdio Subprocess Transport

**Files:**
- Modify: `mcp_unified/smoke/transports.py`
- Create: `tldw_Server_API/app/core/MCP_unified/tests/fixtures/smoke_stdio_server.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py`

- [ ] **Step 1: Write failing tests for stdio subprocess object payloads**

Start the fixture subprocess via argv:

```python
transport = StdioSubprocessTransport(
    command=sys.executable,
    args=[str(FIXTURE_PATH)],
    cwd=str(REPO_ROOT),
    env_allowlist=["PYTHONPATH"],
)
```

Assert `initialize`, `ping`, and `tools/list` work.

- [ ] **Step 2: Create `smoke_stdio_server.py` fixture**

The fixture should:

- read one newline-delimited JSON-RPC payload at a time;
- accept object or batch-array payloads;
- write JSON-RPC responses to stdout only;
- write diagnostics to stderr only;
- suppress notification responses;
- implement `initialize`, `ping`, `tools/list`, `tools/call`, `resources/list`, `prompts/list`.

- [ ] **Step 3: Implement `StdioSubprocessTransport`**

Use `asyncio.create_subprocess_exec`, never `shell=True`.

Support:

- command + args;
- explicit cwd;
- allowlisted env inheritance;
- startup timeout;
- per-request timeout;
- bounded stderr capture;
- cleanup on success, failure, timeout, and cancellation.

- [ ] **Step 4: Write failing tests for batch payloads and cleanup**

Assert a batch containing a notification and `ping` returns only the `ping` response. Assert the subprocess exits or is terminated after `close()`.

- [ ] **Step 5: Implement batch support and cleanup**

Accept list payloads in `request()`. Serialize compact JSON plus newline. Parse exactly one stdout line as the response unless the request is notification-only.

- [ ] **Step 6: Write failing tests for secret redaction in stderr**

Have the fixture write a secret-looking value to stderr on a controlled request. Assert the transport error/report does not expose it.

- [ ] **Step 7: Implement stderr redaction and bounds**

Cap stderr bytes and run captured text through the same redaction helpers used by `SmokeReport`.

- [ ] **Step 8: Run Task 5 tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py -q -k "stdio"
```

Expected: pass.

- [ ] **Step 9: Commit Task 5**

```bash
git add mcp_unified/smoke tldw_Server_API/app/core/MCP_unified/tests
git commit -m "feat: add mcp smoke stdio transport"
```

## Task 6: CLI, Docs, Packaging, And Validation

**Files:**
- Modify: `mcp_unified/smoke/cli.py`
- Modify: `pyproject.toml`
- Create: `Docs/MCP/Unified/Smoke_Client.md`
- Modify: `backlog/tasks/task-2387 - Design-MCP-smoke-client-transport-harness.md`
- Modify: `backlog/tasks/task-2281 - Add-LSP-backed-code-intelligence-MCP-tools.md`

- [ ] **Step 1: Write failing tests for CLI exit codes**

Use `main([...])` directly with in-process mode.

Assert:

- `0` for passed baseline;
- `1` for required step failure;
- `2` for invalid CLI arguments;
- `3` for transport startup/connect failure;
- `4` for strict-mode capability skip.

- [ ] **Step 2: Implement CLI parser**

Support:

- `inprocess`;
- `http --url`;
- `websocket --url`;
- `stdio --command --arg ... --cwd ...`;
- `--scenario baseline`;
- `--mode best-effort|strict`;
- `--profile-id`;
- `--api-key-env`;
- `--bearer-token-env`;
- `--json-report`;
- `--debug-trace`;
- `--timeout`.

- [ ] **Step 3: Add console script**

Modify `pyproject.toml`:

```toml
[project.scripts]
mcp-unified-smoke = "mcp_unified.smoke.cli:main"
```

Keep the existing `mcp-unified-gateway` entrypoint unchanged.

- [ ] **Step 4: Write operator docs**

Create `Docs/MCP/Unified/Smoke_Client.md` with:

- purpose and scope;
- in-process command;
- live HTTP command;
- live WebSocket command;
- stdio subprocess command;
- auth/profile flag behavior;
- retry behavior and non-idempotent call warning;
- report schema summary;
- exit codes;
- CI recommendations.

- [ ] **Step 5: Update Backlog tasks**

Update `TASK-2387` with final summary, checked acceptance criteria, and validation results. Keep `TASK-2281` dependent on `TASK-2387` and mention LSP scenarios should be added after the baseline harness lands.

- [ ] **Step 6: Run full smoke-client verification**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py -q
python -m py_compile mcp_unified/smoke/*.py
python -m ruff check mcp_unified/smoke tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py
python -m bandit -r mcp_unified/smoke -f json -o /tmp/bandit_mcp_smoke_client.json
git diff --check
```

Expected:

- smoke-client tests pass;
- py_compile passes;
- ruff has no new findings in touched scope;
- Bandit reports no new actionable findings in `mcp_unified/smoke`;
- diff check passes.

- [ ] **Step 7: Commit Task 6**

```bash
git add mcp_unified/smoke pyproject.toml Docs/MCP/Unified/Smoke_Client.md backlog/tasks \
  tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py \
  tldw_Server_API/app/core/MCP_unified/tests/fixtures/smoke_stdio_server.py
git commit -m "feat: add mcp smoke client cli"
```

## Final Verification Before PR

- [ ] Run the smoke-client focused suite:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py -q
```

- [ ] Run nearby gateway regression tests:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_jsonrpc_batch.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_http_batch.py \
  -q
```

- [ ] Run static/security checks:

```bash
source .venv/bin/activate
python -m py_compile mcp_unified/smoke/*.py
python -m ruff check mcp_unified/smoke tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py
python -m bandit -r mcp_unified/smoke -f json -o /tmp/bandit_mcp_smoke_client.json
git diff --check
```

- [ ] Manually try documented commands when a local server is available:

```bash
source .venv/bin/activate
mcp-unified-smoke http --url http://127.0.0.1:8000/api/v1/mcp/request --api-key-env SINGLE_USER_API_KEY
mcp-unified-smoke websocket --url ws://127.0.0.1:8000/api/v1/mcp/ws?client_id=smoke
mcp-unified-smoke stdio --command python --arg tldw_Server_API/app/core/MCP_unified/tests/fixtures/smoke_stdio_server.py
```

Record unavailable live-server checks as manual skips with reason.
