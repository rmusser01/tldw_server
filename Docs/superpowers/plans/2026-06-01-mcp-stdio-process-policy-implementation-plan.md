# MCP Unified Stdio Process Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add runtime-configurable process policy enforcement before package-owned upstream stdio MCP transports spawn external processes.

**Architecture:** Create a focused `mcp_unified.federation.process_policy` module for policy coercion and validation, then keep `stdio_transport.py` responsible for subprocess JSON-RPC behavior while delegating executable/cwd/env decisions to the policy helper. Gateway bootstrap config will parse the policy and wrap the default stdio transport factory only when a config policy is supplied, preserving current default factory identity otherwise.

**Tech Stack:** Python 3.10+, dataclasses, pathlib, asyncio subprocesses, Pydantic model inputs, pytest/pytest-asyncio, existing `mcp_unified` gateway/federation contracts.

---

## File Map

- Create `mcp_unified/federation/process_policy.py`
  - Own `StdioProcessPolicy`, `coerce_stdio_process_policy()`, and validation helpers for executable, cwd, env, PATH lookup, and shell wrapper denial.
- Modify `mcp_unified/federation/stdio_transport.py`
  - Accept optional `process_policy`, call policy validation during construction, use the effective cwd and env-name filtering returned by helpers, and pass policy through `create_external_transport()`.
- Modify `mcp_unified/federation/__init__.py`
  - Re-export `StdioProcessPolicy` and `coerce_stdio_process_policy`.
- Modify `mcp_unified/federation/transports.py`
  - Update stale docstrings that still claim the package contract is non-spawning.
- Modify `mcp_unified/gateway/config.py`
  - Add `process_policy` to `GatewayExternalRuntimeBootstrapConfig`, validate mappings, and wrap the default factory only when policy is configured.
- Modify `mcp_unified/gateway/cli.py`
  - Add compact policy summary to `validate-config`.
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_stdio_external_transport.py`
  - Add transport-level policy tests.
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`
  - Add bootstrap and runtime-manager redaction/status tests.
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py`
  - Update validate-config expectations and add configured-policy summary coverage.
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`
  - Add public export coverage for the new policy contract.
- Update Backlog task `TASK-588` with touched files and verification results.

## Stage 1: Policy Helper Module

**Goal:** Add policy data model and pure validation helpers without touching subprocess launch behavior.

**Success Criteria:** Helper tests fail before implementation, then pass with no subprocesses required.

**Tests:** Focused unit tests in `test_stdio_external_transport.py` or a new nearby test section covering coercion, shell basename detection, PATH rules, cwd roots, env-name policy, and safe error details.

**Status:** Not Started

### Task 1: Write failing helper tests

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_stdio_external_transport.py`
- Create later: `mcp_unified/federation/process_policy.py`

- [ ] **Step 1: Add failing tests for policy coercion and denials**

Add tests equivalent to:

```python
from mcp_unified.federation.process_policy import (
    StdioProcessPolicy,
    coerce_stdio_process_policy,
)


def test_stdio_process_policy_rejects_invalid_mapping_values() -> None:
    with pytest.raises(ValueError, match="allowed_executables"):
        coerce_stdio_process_policy({"allowed_executables": ["python", ""]})
    with pytest.raises(ValueError, match="allow_path_lookup"):
        coerce_stdio_process_policy({"allow_path_lookup": "false"})


def test_stdio_process_policy_defaults_block_shell_wrappers() -> None:
    server = _server(command=["/bin/bash", "-lc", "echo no"])
    with pytest.raises(StdioExternalTransportError) as exc_info:
        StdioExternalTransport(server)
    assert exc_info.value.reason_code == "process_policy_shell_denied"
    assert "echo no" not in str(exc_info.value)


def test_stdio_process_policy_allows_explicit_shell_executable() -> None:
    policy = StdioProcessPolicy(allowed_executables=("/bin/bash",))
    transport = StdioExternalTransport(
        _server(command=["/bin/bash", "--version"]),
        process_policy=policy,
    )
    assert transport.server_id == "docs"
```

- [ ] **Step 2: Run helper tests and confirm they fail**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_stdio_external_transport.py \
  -k "process_policy" -q
```

Expected: FAIL because `mcp_unified.federation.process_policy` and `process_policy` constructor support do not exist.

### Task 2: Implement policy model and pure helpers

**Files:**
- Create: `mcp_unified/federation/process_policy.py`

- [ ] **Step 1: Add `StdioProcessPolicy` and coercion helper**

Implement:

```python
@dataclass(frozen=True, slots=True)
class StdioProcessPolicy:
    allowed_executables: tuple[str, ...] = ()
    allowed_cwd_roots: tuple[str | Path, ...] = ()
    allowed_env_names: tuple[str, ...] | None = None
    allow_path_lookup: bool = True
    reject_shell_executables: bool = True
    default_cwd: str | Path | None = None
```

Add `coerce_stdio_process_policy(value)` that:

- returns default policy for `None`
- returns the same immutable model for `StdioProcessPolicy`
- accepts mappings from JSON/TOML config
- rejects blank strings, non-string list entries, and non-bool booleans
- preserves `allowed_env_names=None` distinct from `allowed_env_names=()`

- [ ] **Step 2: Add validation helpers**

Add helpers that can be called from `stdio_transport.py`:

```python
def validate_stdio_process_policy(
    *,
    server_id: str,
    command: tuple[str, ...],
    cwd: str | None,
    env_allowlist: list[str],
    policy: StdioProcessPolicy,
    error_factory: Callable[..., Exception],
) -> StdioProcessDecision:
    ...
```

The returned decision should contain:

- `command`
- `cwd`
- `allowed_env_names`

Use safe reason codes from the spec.

- [ ] **Step 3: Run helper tests**

Run the same focused pytest command. Expected: failures should now be only in transport integration, if any.

## Stage 2: Transport Enforcement

**Goal:** Enforce process policy before spawning and keep stdio JSON-RPC behavior unchanged for allowed commands.

**Success Criteria:** Existing stdio transport tests keep passing, new policy denial tests pass, and secret/command material is not leaked in error messages.

**Tests:** `test_stdio_external_transport.py`.

**Status:** Not Started

### Task 3: Wire policy into `StdioExternalTransport`

**Files:**
- Modify: `mcp_unified/federation/stdio_transport.py`
- Modify: `mcp_unified/federation/__init__.py`
- Modify: `mcp_unified/federation/transports.py`

- [ ] **Step 1: Update constructor and factory signatures**

Add:

```python
process_policy: StdioProcessPolicy | Mapping[str, Any] | None = None
```

to `StdioExternalTransport.__init__()` and `create_external_transport()`.

- [ ] **Step 2: Apply policy during construction**

After `_validate_command()`, call `coerce_stdio_process_policy()` and the validation helper. Store:

- `self._process_policy`
- `self._cwd`
- `self._allowed_env_names`

Use `StdioExternalTransportError` with policy reason codes.

- [ ] **Step 3: Apply env-name intersection in `_build_child_env()`**

When `allowed_env_names` is not `None`, inherit only names present in both:

- `server.env_allowlist`
- `policy.allowed_env_names`
- `os.environ`

Do not include values in exceptions or logs.

- [ ] **Step 4: Export the policy and fix stale docstrings**

Update `mcp_unified/federation/__init__.py` public exports and revise `mcp_unified/federation/transports.py` docstrings from “non-spawning” to neutral language.

- [ ] **Step 5: Run stdio transport tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_stdio_external_transport.py -q
```

Expected: PASS.

## Stage 3: Gateway Config And CLI Wiring

**Goal:** Let gateway configs supply process policy while preserving existing default behavior and injected factory semantics.

**Success Criteria:** Config validates policy mappings; default manager factory identity is unchanged when policy is omitted; configured policy wraps the default stdio factory; custom factories remain untouched.

**Tests:** `test_gateway_fastapi_package.py`, `test_gateway_cli_package.py`, `test_runtime_package_boundary.py`.

**Status:** Not Started

### Task 4: Write failing gateway config and CLI tests

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`

- [ ] **Step 1: Add bootstrap tests**

Add tests that assert:

- `GatewayExternalRuntimeBootstrapConfig(process_policy={"allow_path_lookup": False})` is accepted
- invalid process policy mapping raises `ValueError`
- manager default factory identity remains `create_external_transport` when policy is omitted
- manager default factory is wrapped when policy is configured
- custom `external_transport_factory` is preserved even when config has policy

- [ ] **Step 2: Add CLI summary test**

Update existing validate-config success expectation to include:

```json
"process_policy": {
  "configured": false,
  "allowed_executables": 0,
  "allowed_cwd_roots": 0,
  "allowed_env_names": null,
  "allow_path_lookup": true,
  "reject_shell_executables": true,
  "default_cwd": false
}
```

Add a configured-policy test where counts are non-zero and paths are not echoed.

- [ ] **Step 3: Add public export test**

Assert `mcp_unified.federation.StdioProcessPolicy is mcp_unified.federation.process_policy.StdioProcessPolicy`.

- [ ] **Step 4: Run focused tests and confirm failures**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  -k "process_policy or validate_config_reports_success_json or builds_stdio_external_runtime_manager or external_runtime_uses_injected_transport_factory or public_exports" -q
```

Expected: FAIL until config, CLI, and exports are implemented.

### Task 5: Implement gateway config and CLI support

**Files:**
- Modify: `mcp_unified/gateway/config.py`
- Modify: `mcp_unified/gateway/cli.py`
- Modify: `mcp_unified/federation/__init__.py`

- [ ] **Step 1: Extend `GatewayExternalRuntimeBootstrapConfig`**

Add `process_policy` field and normalize it in `__post_init__()`.

Preserve whether the caller explicitly supplied policy, for example with an internal boolean or `None` field, so factory wrapping happens only when configured.

- [ ] **Step 2: Thread policy through bootstrap**

Add `process_policy` parameter to `external_runtime_manager_from_storage()`.

When `transport_factory is None` and policy was configured, build a wrapper closure:

```python
def _policy_transport_factory(server: ExternalServerDefinition) -> ExternalFederationTransport:
    return create_external_transport(server, process_policy=process_policy)
```

When no policy was configured, pass `create_external_transport` directly.

- [ ] **Step 3: Add CLI summary**

Update `_validated_config_payload()` with a compact `process_policy` summary. Do not expose raw paths or executable values.

- [ ] **Step 4: Run gateway tests**

Run the focused pytest command from Task 4. Expected: PASS.

## Stage 4: Runtime Manager Redaction Coverage

**Goal:** Verify process-policy denial through the runtime manager produces safe status and error metadata.

**Success Criteria:** A policy-denied start fails with existing public reason code and status redaction does not include command args or env values.

**Tests:** `test_gateway_fastapi_package.py` or `test_gateway_external_runtime.py`, whichever has the simplest existing fixtures.

**Status:** Not Started

### Task 6: Add and pass runtime-manager redaction test

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`
- Modify only if needed: `mcp_unified/gateway/external_runtime.py`

- [ ] **Step 1: Write failing redaction test**

Create a sqlite bootstrap with `external_runtime.enabled=True` and a restrictive process policy. Store an enabled stdio server with a denied command containing a unique argument marker and env allowlist containing a unique env name.

Assert:

- `manager.start_server("research")` raises `GatewayExternalRuntimeError` with reason `external_server_start_failed`
- `manager.list_runtime_servers()` returns status containing a safe `last_error`
- denied command argument marker is absent from the exception string and status JSON
- env value marker is absent from the exception string and status JSON

- [ ] **Step 2: Run test and confirm failure if runtime redaction needs adjustment**

Run the focused gateway pytest command. Expected: FAIL only if current runtime stores too much exception text.

- [ ] **Step 3: Adjust runtime error summarization only if needed**

If the test fails due leakage, change only the start-failure path to store safe policy reason details instead of raw exception text. Keep existing payload shapes.

- [ ] **Step 4: Run focused gateway tests**

Expected: PASS.

## Stage 5: Verification And Completion

**Goal:** Validate touched behavior and update tracking records.

**Success Criteria:** Targeted tests pass, Bandit runs on touched code, Backlog task records results, and commits are clean.

**Tests:** Targeted pytest plus Bandit on touched code.

**Status:** Not Started

### Task 7: Run final verification

**Files:**
- No code changes expected unless verification exposes failures.

- [ ] **Step 1: Run targeted pytest suite**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_stdio_external_transport.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run Bandit on touched package code**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit \
  -r mcp_unified/federation mcp_unified/gateway \
  -f json -o /tmp/bandit_mcp_stdio_process_policy.json
```

Expected: completes with no new findings in touched code. If Bandit is not installed, record that explicitly and run any available security checks for touched files.

- [ ] **Step 3: Update Backlog task**

Record:

- touched files
- test commands and results
- Bandit result or skip reason
- final summary

- [ ] **Step 4: Commit implementation**

Commit after tests pass:

```bash
git add mcp_unified/federation/process_policy.py \
  mcp_unified/federation/stdio_transport.py \
  mcp_unified/federation/__init__.py \
  mcp_unified/federation/transports.py \
  mcp_unified/gateway/config.py \
  mcp_unified/gateway/cli.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_stdio_external_transport.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  "backlog/tasks/task-588 - Harden-MCP-external-stdio-process-policy.md"
git commit -m "feat(mcp): enforce stdio process policy"
```

Expected: commit succeeds with a clean worktree afterward.
