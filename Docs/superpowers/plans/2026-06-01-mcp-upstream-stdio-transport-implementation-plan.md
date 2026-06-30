# MCP Upstream Stdio Transport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a package-owned upstream stdio subprocess transport for external MCP servers managed by the standalone gateway runtime.

**Architecture:** Implement a focused `mcp_unified.federation.stdio_transport` module that launches one subprocess with shell-free argv execution, speaks newline-delimited JSON-RPC over stdio, normalizes MCP tool discovery/call responses, and satisfies the existing `ExternalFederationTransport` protocol. Tests drive behavior with temporary Python MCP stdio stub servers.

**Tech Stack:** Python asyncio subprocesses, JSON-RPC 2.0 line framing, Pydantic storage models, pytest, Bandit.

---

## Scope And Constraints

Spec: `Docs/superpowers/specs/2026-06-01-mcp-upstream-stdio-transport-design.md`

Backlog: `TASK-582`

Keep package code free of `tldw_Server_API` imports. Do not reuse the host
`ACPStdioClient`; this transport must be package-owned.

Do not add shell execution, install/update behavior, WebSocket upstream support,
or default gateway bootstrap wiring in this slice.

Use TDD: write failing tests first, verify red, then implement the minimal code.

## File Structure

- Create: `mcp_unified/federation/stdio_transport.py`
  - process launch, JSON-RPC request/response handling, response normalization,
    credential metadata injection, cleanup, and safe transport errors
- Modify: `mcp_unified/federation/__init__.py`
  - export `StdioExternalTransport`, `StdioExternalTransportError`, and
    `create_external_transport`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_stdio_external_transport.py`
  - subprocess-backed behavior tests and package-boundary checks
- Modify: `backlog/tasks/task-582 - Implement-MCP-upstream-stdio-external-server-transport.md`
  - progress notes and verification summary

## Task 1: Validation And Package Boundary Red Tests

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_stdio_external_transport.py`

- [x] **Step 1: Write failing tests**

Tests:

```python
def test_stdio_transport_import_does_not_import_host_package():
    code = (
        "import sys; import mcp_unified.federation.stdio_transport; "
        "print('tldw_Server_API' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"
```

```python
def test_stdio_transport_rejects_non_stdio_definition():
    server = ExternalServerDefinition(
        id="ws",
        name="WebSocket",
        transport="websocket",
        url="ws://example.invalid",
    )
    with pytest.raises(StdioExternalTransportError, match="unsupported_transport"):
        StdioExternalTransport(server)
```

```python
def test_stdio_transport_rejects_missing_cwd(tmp_path):
    server = _server(command=[sys.executable, "-c", "pass"], cwd=str(tmp_path / "missing"))
    with pytest.raises(StdioExternalTransportError, match="invalid_cwd"):
        StdioExternalTransport(server)
```

- [x] **Step 2: Verify red**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_stdio_external_transport.py -q
```

Expected: import failure for missing `mcp_unified.federation.stdio_transport`.

## Task 2: Subprocess Round Trip

**Files:**
- Create: `mcp_unified/federation/stdio_transport.py`
- Modify: `mcp_unified/federation/__init__.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_stdio_external_transport.py`

- [x] **Step 1: Add failing subprocess tests**

Create a temporary Python script in the test that reads JSON lines from stdin
and responds to `initialize`, `tools/list`, `tools/call`, and `ping`.

Tests:

```python
async def test_stdio_transport_connect_list_call_and_close(tmp_path):
    transport = StdioExternalTransport(_server(command=[sys.executable, "-u", script]))
    await transport.connect()
    assert (await transport.health_check())["initialized"] is True
    tools = await transport.list_tools()
    assert [tool.name for tool in tools] == ["docs.search", "docs.defaulted"]
    result = await transport.call_tool("docs.search", {"q": "hello"})
    assert result.content == [{"type": "text", "text": "search:hello"}]
    await transport.close()
    await transport.close()
```

```python
async def test_stdio_transport_uses_only_allowlisted_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_ALLOWED", "yes")
    monkeypatch.setenv("MCP_BLOCKED", "no")
    result = await transport.call_tool("docs.env", {})
    assert result.content == {"allowed": "yes", "blocked": None}
```

- [x] **Step 2: Implement minimal transport**

Implement:

- constructor validation
- `connect()`
- `_request()`
- `list_tools()`
- `call_tool()`
- `health_check()`
- `close()`
- `create_external_transport()`

Use `asyncio.create_subprocess_exec(*server.command, ...)` and a compact JSON
line protocol. Serialize requests through an async lock.

- [x] **Step 3: Verify green**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_stdio_external_transport.py -q
```

Expected: validation and round-trip tests pass.

## Task 3: Failure, Timeout, Credential Redaction

**Files:**
- Modify: `mcp_unified/federation/stdio_transport.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_stdio_external_transport.py`

- [x] **Step 1: Add failing tests**

Tests:

```python
async def test_stdio_transport_timeout_error_is_safe(tmp_path):
    transport = StdioExternalTransport(_server(command=[sys.executable, "-u", script]), request_timeout_s=0.05)
    with pytest.raises(StdioExternalTransportError) as exc_info:
        await transport.call_tool("docs.slow", {})
    assert exc_info.value.reason_code == "request_timeout"
    assert "docs.slow" not in str(exc_info.value)
```

```python
async def test_stdio_transport_sends_runtime_auth_in_meta_without_leaking_secret(tmp_path):
    secret = "super-secret-token"
    result = await transport.call_tool(
        "docs.auth",
        {},
        runtime_auth=BrokeredExternalCredential(env={"DOCS_TOKEN": secret}),
    )
    assert result.content["has_secret"] is True
    with pytest.raises(StdioExternalTransportError) as exc_info:
        await transport.call_tool(
            "docs.slow",
            {},
            runtime_auth=BrokeredExternalCredential(env={"DOCS_TOKEN": secret}),
        )
    assert secret not in str(exc_info.value)
```

```python
async def test_stdio_transport_health_marks_exited_process_disconnected(tmp_path):
    await transport.connect()
    await transport.call_tool("docs.exit", {})
    health = await transport.health_check()
    assert health["connected"] is False
```

- [x] **Step 2: Harden implementation**

Add safe structured error handling:

- `StdioExternalTransportError(reason_code=...)`
- request timeout wrapping
- JSON-RPC error mapping
- process exit detection
- pending request cleanup
- stderr drain without including stderr text in public errors
- runtime-auth `_meta` injection with no logging

- [x] **Step 3: Verify green**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_stdio_external_transport.py -q
```

Expected: all stdio transport tests pass.

## Task 4: Integration Verification And Cleanup

**Files:**
- Modify: `backlog/tasks/task-582 - Implement-MCP-upstream-stdio-external-server-transport.md`

- [x] **Step 1: Run focused existing package runtime tests**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_stdio_external_transport.py \
  -q
```

- [x] **Step 2: Run Ruff on touched package/tests**

Run:

```bash
source ../../.venv/bin/activate
python -m ruff check \
  mcp_unified/federation/stdio_transport.py \
  mcp_unified/federation/__init__.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_stdio_external_transport.py
```

- [x] **Step 3: Run Bandit on touched Python source**

Run:

```bash
source ../../.venv/bin/activate
python -m bandit -r \
  mcp_unified/federation/stdio_transport.py \
  -f json -o /tmp/bandit_mcp_stdio_transport.json
```

- [x] **Step 4: Run whitespace check**

Run:

```bash
git diff --check
```

- [x] **Step 5: Update Backlog task**

Record files changed, verification commands, and final notes in TASK-582.

- [x] **Step 6: Commit**

```bash
git add \
  Docs/superpowers/specs/2026-06-01-mcp-upstream-stdio-transport-design.md \
  Docs/superpowers/plans/2026-06-01-mcp-upstream-stdio-transport-implementation-plan.md \
  mcp_unified/federation/stdio_transport.py \
  mcp_unified/federation/__init__.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_stdio_external_transport.py \
  "backlog/tasks/task-582 - Implement-MCP-upstream-stdio-external-server-transport.md"
git commit -m "feat: add package upstream stdio transport"
```
