# MCP Gateway External Runtime Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose active standalone external MCP runtime virtual tools through the gateway `GatewayRuntime` protocol so HTTP, WebSocket, and stdio transports can list and call them.

**Architecture:** Add a small package-owned adapter that composes an optional base `GatewayRuntime` with `GatewayExternalRuntimeManager`. Keep transport code unchanged: JSON-RPC still delegates to `GatewayRuntime`, while `ProfileAwareGatewayRuntime` enriches delegated call contexts with the already-derived effective policy so the external manager can enforce external-server grants, credential grants, and audit behavior.

**Tech Stack:** Python 3.11, asyncio, package-local MCP Unified gateway/federation contracts, pytest, Ruff, Bandit.

---

## Scope And Constraints

Backlog: `TASK-589`

In scope:

- Convert active `VirtualExternalTool` rows into normal gateway tool descriptors.
- Dispatch `tools/call` for external virtual tool names to `GatewayExternalRuntimeManager.execute_virtual_tool()`.
- Preserve profile policy enforcement by having `ProfileAwareGatewayRuntime` pass the resolved `EffectivePolicy` into downstream context metadata.
- Preserve base runtime behavior when an injected base runtime handles local tools/resources/prompts/modules.

Out of scope:

- Durable daemon-control CLI commands.
- A client-facing stdio serve loop.
- Real package-manager install/update execution.
- New external process policy rules beyond existing process-policy enforcement.
- Broad packaging/extras metadata changes.

## File Structure

- Create: `mcp_unified/gateway/external_runtime_adapter.py`
  - `ExternalRuntimeGatewayRuntime`
  - virtual-tool descriptor conversion
  - context policy extraction
  - federated result conversion to MCP tool-call JSON
- Modify: `mcp_unified/gateway/profile_runtime.py`
  - add a package-private metadata key for effective policy
  - enrich delegated call contexts after profile policy is resolved
- Modify: `mcp_unified/gateway/__init__.py`
  - lazy/public export for the adapter and policy metadata key if useful
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime_adapter.py`
  - focused adapter and profile-context integration tests
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`
  - package export/import-boundary coverage
- Modify: `Docs/superpowers/plans/2026-06-02-mcp-gateway-external-runtime-adapter-plan.md`
  - execution evidence and status updates
- Modify: `backlog/tasks/task-589 - Expose-MCP-external-runtime-tools-through-gateway-runtime.md`
  - acceptance criteria, verification, final summary

Baseline evidence before this plan: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py -q` passed with `63 passed, 4 warnings`.

## Task 1: Adapter Red Tests

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime_adapter.py`

- [x] **Step 1: Write test doubles**

Create an in-memory external registry store, a recording transport, and a small base gateway runtime. Reuse `ExternalServerDefinition`, `ExternalToolDefinition`, and `ExternalToolCallResult` from package models.

- [x] **Step 2: Add a listing test**

Assert that an adapter with a base runtime and a started external runtime returns both local and external tools:

```python
async def test_external_runtime_gateway_lists_base_and_external_tools():
    store = InMemoryExternalRegistryStore([_server("research")])
    transport = RecordingTransport(
        tools=[
            ExternalToolDefinition(
                name="search",
                description="Search papers",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
                metadata={"capability": "research.search", "annotations": {"readOnlyHint": True}},
            )
        ]
    )
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda _server: transport,
    )
    await manager.start_server("research")

    runtime = ExternalRuntimeGatewayRuntime(
        base_runtime=BaseGatewayRuntime(),
        external_runtime_manager=manager,
    )
    tools = await runtime.list_tools(GatewayRequestContext(request_id="list"))

    assert [tool["name"] for tool in tools] == ["local.echo", "ext.research.search"]
    external = tools[1]
    assert external["inputSchema"]["properties"]["query"]["type"] == "string"
    assert external["metadata"]["external_server_id"] == "research"
    assert external["metadata"]["upstream_tool_name"] == "search"
    assert external["metadata"]["source"] == "external_runtime"
```

- [x] **Step 3: Add an execution dispatch test**

Assert local names delegate to the base runtime and external names call `execute_virtual_tool()` with a copy of arguments and the context actor id:

```python
result = await runtime.call_tool(
    "ext.research.search",
    {"query": "mcp"},
    GatewayRequestContext(
        request_id="call",
        user_id="user-1",
        metadata={EFFECTIVE_POLICY_METADATA_KEY: {"external_server_grants": [{"server_id": "research"}]}},
    ),
)

assert result == {
    "content": {"matches": ["paper-1"]},
    "isError": False,
    "metadata": {
        "server_id": "research",
        "upstream_tool_name": "search",
        "virtual_tool_name": "ext.research.search",
    },
}
assert transport.calls[0][0] == "search"
```

- [x] **Step 4: Add a missing external tool fallback test**

Assert `ExternalRuntimeGatewayRuntime` delegates unknown/non-external names to the base runtime when one is configured, and raises `ValueError` for unknown names when no base runtime is available.

- [x] **Step 5: Verify red**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime_adapter.py -q
```

Expected: fail because `mcp_unified.gateway.external_runtime_adapter` does not exist.

Evidence: RED failed during collection with `ModuleNotFoundError: No module named 'mcp_unified.gateway.external_runtime_adapter'`.

## Task 2: Adapter Implementation

**Files:**
- Create: `mcp_unified/gateway/external_runtime_adapter.py`
- Modify: `mcp_unified/gateway/__init__.py`

- [x] **Step 1: Add the adapter skeleton**

Implement:

```python
class ExternalRuntimeGatewayRuntime:
    name = "mcp-unified-gateway"
    version = "0.1.0"

    def __init__(
        self,
        *,
        external_runtime_manager: GatewayExternalRuntimeManager,
        base_runtime: GatewayRuntime | None = None,
        name: str | None = None,
        version: str | None = None,
    ) -> None: ...
```

Resolve `name` and `version` from explicit values, then `base_runtime`, then stable defaults.

- [x] **Step 2: Convert virtual tools to gateway descriptors**

Map `VirtualExternalTool` to:

```python
{
    "name": virtual_tool.virtual_name,
    "description": virtual_tool.description,
    "inputSchema": copy.deepcopy(virtual_tool.input_schema),
    "metadata": {
        **copy.deepcopy(virtual_tool.metadata),
        "source": "external_runtime",
        "external_server_id": virtual_tool.server_id,
        "upstream_tool_name": virtual_tool.upstream_tool_name,
        "is_write": virtual_tool.is_write,
    },
}
```

Do not mutate the `VirtualExternalTool` returned by the manager.

- [x] **Step 3: Implement local/external tool dispatch**

`call_tool()` should:

- route names currently returned by `list_virtual_tools()` to `execute_virtual_tool()`
- pass `effective_policy` from `context.metadata[EFFECTIVE_POLICY_METADATA_KEY]`
- pass `actor_id=context.user_id`
- pass the original context to the external manager
- delegate all other names to `base_runtime.call_tool()` when configured
- raise `ValueError("Unknown gateway tool: <name>")` when no route exists

- [x] **Step 4: Convert federated results**

Convert `FederatedToolResult` into MCP tool-call JSON:

```python
{
    "content": copy.deepcopy(result.content),
    "isError": result.is_error,
    "metadata": {
        **copy.deepcopy(result.metadata),
        "server_id": result.server_id,
        "upstream_tool_name": result.upstream_tool_name,
        "virtual_tool_name": result.virtual_tool_name,
    },
}
```

- [x] **Step 5: Delegate non-tool methods**

For resources, prompts, modules, and module health, delegate to `base_runtime` when available. Return empty lists or empty health details when no base runtime exists.

- [x] **Step 6: Export lazily**

Update `mcp_unified/gateway/__init__.py` so `ExternalRuntimeGatewayRuntime` can be imported without eager FastAPI imports.

- [x] **Step 7: Run green adapter tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime_adapter.py -q
```

Expected: adapter tests pass.

Evidence: adapter tests passed with `5 passed, 4 warnings`.

## Task 3: Profile Policy Context Integration

**Files:**
- Modify: `mcp_unified/gateway/profile_runtime.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime_adapter.py`

- [x] **Step 1: Add a red profile integration test**

Build a profile with:

```python
MCPProfile(
    id="researcher",
    name="Researcher",
    policy_document=ProfilePolicy(allowed_tools=["ext.research.search"]),
    external_server_grants=[{"server_id": "research"}],
    credential_grants=[{"server_id": "research", "credential_slots": ["api_key"]}],
)
```

Use `ProfileAwareGatewayRuntime(ExternalRuntimeGatewayRuntime(...), profile_store=store, default_profile_id="researcher")` and assert:

- `tools/list` includes `ext.research.search`
- `tools/call` succeeds for the external tool
- the external manager receives effective policy grants
- a required credential slot is satisfied through the profile credential grant

- [x] **Step 2: Add a red profile denial test**

Use the same started external server but a profile without `external_server_grants`. Assert the profile gate may list the tool if allowed by tool name, but `tools/call` fails with `FederationPolicyDenied`/JSON-RPC policy denial reason `external_server_not_granted`.

- [x] **Step 3: Add context enrichment helper**

In `profile_runtime.py`, add:

```python
EFFECTIVE_POLICY_METADATA_KEY = "_gateway_effective_policy"
```

After `call_tool()` obtains a resolved `EffectivePolicyResult`, create a new `GatewayRequestContext` with copied metadata plus the JSON-safe effective policy data. Do not mutate the incoming context.

- [x] **Step 4: Run profile integration tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime_adapter.py -q
```

Expected: all adapter and profile-context tests pass.

Evidence: covered by the adapter test file pass with `5 passed, 4 warnings`.

## Task 4: JSON-RPC And Boundary Coverage

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime_adapter.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`

- [x] **Step 1: Add a JSON-RPC smoke test**

Use `handle_jsonrpc()` or `create_gateway_app()` with `ProfileAwareGatewayRuntime(ExternalRuntimeGatewayRuntime(...))` and assert `tools/list` and `tools/call` return the external tool through the existing JSON-RPC path.

- [x] **Step 2: Add export/import-boundary tests**

Assert:

- importing `mcp_unified.gateway.ExternalRuntimeGatewayRuntime` works
- importing `mcp_unified.gateway.external_runtime_adapter` does not import `tldw_Server_API`
- importing the adapter does not require FastAPI

- [x] **Step 3: Run focused package tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime_adapter.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  -q
```

Expected: focused package tests pass.

Evidence: adapter plus runtime package-boundary tests passed with `18 passed, 4 warnings`.

## Task 5: Verification And Handoff

**Files:**
- Modify: `Docs/superpowers/plans/2026-06-02-mcp-gateway-external-runtime-adapter-plan.md`
- Modify: `backlog/tasks/task-589 - Expose-MCP-external-runtime-tools-through-gateway-runtime.md`

- [x] **Step 1: Run compatibility tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime_adapter.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_federation_shell_contracts.py \
  -q
```

Expected: focused adapter, manager, gateway, boundary, and federation tests pass.

Evidence: focused MCP compatibility suite passed with `201 passed, 5 warnings`.

- [x] **Step 2: Run lint/security/whitespace checks**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check \
  mcp_unified/gateway \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime_adapter.py
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r \
  mcp_unified/gateway/external_runtime_adapter.py \
  mcp_unified/gateway/profile_runtime.py \
  -f json -o /tmp/bandit_mcp_gateway_external_runtime_adapter.json
git diff --check
```

Expected: Ruff passes, Bandit reports no findings for touched package files, and diff check passes.

Evidence: Ruff reported `All checks passed!`; Bandit JSON at `/tmp/bandit_mcp_gateway_external_runtime_adapter.json` had `0` results; `git diff --check` exited cleanly.

- [x] **Step 3: Update plan and Backlog task**

Record test evidence, Bandit output path, known deferrals, and final summary in this plan and `TASK-589`.

- [ ] **Step 4: Commit and open PR**

Stage only this worktree's task, plan, source, and test changes. Commit with:

```bash
git commit -m "feat: expose external runtime tools through gateway"
```

Push `codex/mcp-gateway-external-runtime-adapter` and open a PR against `dev`.

## Final Verification Before PR

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime_adapter.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_federation_shell_contracts.py \
  -q
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check \
  mcp_unified/gateway \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime_adapter.py
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r \
  mcp_unified/gateway/external_runtime_adapter.py \
  mcp_unified/gateway/profile_runtime.py \
  -f json -o /tmp/bandit_mcp_gateway_external_runtime_adapter.json
git diff --check
git status --short --branch
```

Expected:

- focused pytest suite passes
- Ruff passes
- Bandit JSON reports no findings for touched package code
- `git diff --check` passes
- branch is clean after commit

## Deliberate Deferrals

- The client-facing stdio serve loop remains deferred until there is a concrete gateway runtime factory to launch from configuration.
- Durable lifecycle CLI control remains deferred until there is a daemon-control client.
- Package installer execution remains disabled by default and out of this adapter scope.
