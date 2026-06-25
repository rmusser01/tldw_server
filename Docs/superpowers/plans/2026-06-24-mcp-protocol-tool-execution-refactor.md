# MCP Protocol Tool Execution Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the security-sensitive `tools/call` path from `tldw_Server_API/app/core/MCP_unified/protocol.py` into focused tool-execution modules while preserving public behavior.

**Architecture:** `MCPProtocol` remains the public JSON-RPC facade. Tool execution moves behind a `ToolExecutionCoordinator` that depends on explicit services, a reporter facade, and focused security/hooks/runtime helpers. Early stages use a compatibility callback ledger so temporary protocol-bound behavior is visible and removed deliberately.

**Tech Stack:** Python, pytest, Pydantic, Loguru, existing MCP Unified module registry/RBAC/rate limiter/telemetry/tool-use reporting interfaces.

---

## Source Spec

- `Docs/superpowers/specs/2026-06-23-mcp-protocol-tool-execution-refactor-design.md`
- Backlog task: `TASK-2424`

## File Structure

- Create `tldw_Server_API/app/core/MCP_unified/protocol_types.py`
  - Shared protocol/tool-execution types and compatibility-auth helpers:
    `RequestContext`, `PreparedToolCall`, `InvalidParamsException`,
    `GovernanceDeniedError`, `ApprovalRequiredError`,
    `_trusted_compat_claims_metadata`, and `_has_trusted_compat_claims`.
- Create `tldw_Server_API/app/core/MCP_unified/tool_execution/__init__.py`
  - Package marker with explicit exports for coordinator, dependencies, reporter, security, hooks, and runtime classes.
- Create `tldw_Server_API/app/core/MCP_unified/tool_execution/models.py`
  - Internal stage result types used only by the extracted package.
- Create `tldw_Server_API/app/core/MCP_unified/tool_execution/dependencies.py`
  - `ToolExecutionDependencies` dataclass and `CompatibilityCallbackLedgerEntry`.
- Create `tldw_Server_API/app/core/MCP_unified/tool_execution/reporting.py`
  - `ToolExecutionReporter` facade first, then reporting internals in Stage 6.
- Create `tldw_Server_API/app/core/MCP_unified/tool_execution/coordinator.py`
  - `ToolExecutionCoordinator` that owns `tools/call` stage ordering and public nested prepare/execute flows.
- Create `tldw_Server_API/app/core/MCP_unified/tool_execution/security.py`
  - `ToolExecutionSecurity` for validation, resolution, hardening, write classification, RBAC/API scopes, effective policy, external access, path scope, approval, governance preflight, prepared-call integrity, and input schema validation.
- Create `tldw_Server_API/app/core/MCP_unified/tool_execution/hooks.py`
  - `ToolExecutionHooks` for pre/post hook context construction, hook decision coercion, hook payload shaping, and hook execution.
- Create `tldw_Server_API/app/core/MCP_unified/tool_execution/runtime.py`
  - `ToolExecutionRuntime` for rate limiting, idempotency, circuit-breaker execution, result formatting, execution eval metadata, audit/metrics calls through the reporter, and post-hooks.
- Modify `tldw_Server_API/app/core/MCP_unified/protocol.py`
  - Re-export compatibility symbols, construct dependencies/coordinator/reporter, delegate `tools/call`, keep JSON-RPC parsing, coarse authorization, non-tool handlers, and response models.
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py`
  - Add compatibility import and import-boundary tests.
- Modify or add focused tests under `tldw_Server_API/app/core/MCP_unified/tests/`
  - Characterization, coordinator, security, hooks, runtime, reporting, idempotency, and boundary tests.

## Temporary Compatibility Callback Ledger

Final active ledger state: empty. The table below is retained as resolved history for auditability; each behavior now has an extracted owner, with protocol wrappers kept only for compatibility and test monkeypatch seams where current callers still exist.

| Callback | Original owner | Final owner | Resolved in | Parity test |
| --- | --- | --- | --- | --- |
| `prepare_tool_call_impl` | `MCPProtocol._prepare_tool_call_inline` | `ToolExecutionSecurity.prepare_tool_call` | Stage 4c | `test_tool_execution_coordinator_delegates_prepare_then_execute` |
| `execute_prepared_tool_call_impl` | `MCPProtocol._execute_prepared_tool_call_inline` | `ToolExecutionRuntime.execute_prepared_tool_call` | Stage 5 | `test_tool_execution_coordinator_delegates_prepare_then_execute` |
| `validate_input_schema` | `MCPProtocol._validate_input_schema` | `ToolExecutionSecurity.validate_input_schema` | Stage 4a | `test_protocol_invalid_tool_arguments_remain_invalid_params` |
| `generic_exception_like` | `MCPProtocol._generic_exception_like` | `ToolExecutionRuntime._generic_exception_like` | Stage 5 | `test_runtime_sanitizes_tool_execution_errors` |
| `make_idempotency_cache_key` | `MCPProtocol._make_idempotency_cache_key` | `ToolExecutionRuntime.make_idempotency_cache_key` | Stage 5 | `test_idempotency_key_isolated_across_users` |
| `extract_eval_profile_id` | `MCPProtocol._extract_eval_profile_id` | `ToolExecutionRuntime.extract_eval_profile_id` | Stage 5 | `test_protocol_tool_use_reporting_includes_execution_eval_metadata` |
| `tool_use_event_builder` | `MCPProtocol._build_tool_use_event` | `ToolExecutionReporter.build_tool_use_event` | Stage 6 | `test_protocol_records_successful_tool_use_event` |
| `record_process_request_failure` | `MCPProtocol._record_process_request_tool_use_failure` | `ToolExecutionReporter.record_process_request_failure` | Stage 6 | `test_tools_call_coarse_authorization_denial_records_denied_tool_use` |

## Stage 1: Characterization And Boundary Tests

**Goal:** Lock down current behavior before moving code.

**Success Criteria:** New tests fail only where the future extraction package does not exist yet; existing behavior tests continue to pass.

**Tests:** `test_extraction_contracts.py`, `test_tool_execution_coordinator.py`, `test_tool_use_reporting_protocol.py`, `test_idempotency_and_category.py`.

### Task 1: Add Compatibility Import And Import-Boundary Tests

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py`

- [x] **Step 1: Add the failing tests**

Append these tests after `test_protocol_instances_do_not_share_prepared_call_secrets`:

```python
def test_protocol_reexports_tool_execution_shared_symbols() -> None:
    from tldw_Server_API.app.core.MCP_unified import protocol

    expected = {
        "RequestContext",
        "PreparedToolCall",
        "InvalidParamsException",
        "GovernanceDeniedError",
        "ApprovalRequiredError",
        "IdempotencyManager",
        "_trusted_compat_claims_metadata",
    }

    missing = sorted(name for name in expected if not hasattr(protocol, name))
    assert missing == []


def test_tool_execution_package_does_not_import_protocol_facade() -> None:
    package_dir = MCP_ROOT / "tool_execution"
    assert package_dir.is_dir()
    forbidden = {
        f"{MCP_PACKAGE}.protocol",
        "tldw_Server_API.app.core.MCP_unified.protocol",
    }
    offenders: dict[str, list[str]] = {}

    def package_for(path: Path) -> str:
        relative_parent = path.relative_to(MCP_ROOT).parent
        return ".".join((MCP_PACKAGE, *relative_parent.parts))

    for path in package_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        current_package = package_for(path)
        imports = _resolved_import_sources_for(path, current_package)
        bad_imports = [
            source
            for source in imports
            if source in forbidden
            or source.endswith(".MCP_unified.protocol")
        ]
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                source = _resolve_import_from_source(current_package, node.module, node.level)
                if source == MCP_PACKAGE and any(alias.name == "protocol" for alias in node.names):
                    bad_imports.append(f"{source}.protocol")
        if bad_imports:
            offenders[str(path.relative_to(MCP_ROOT))] = bad_imports

        uses_facade = [
            "MCPProtocol"
            for node in ast.walk(tree)
            if (
                isinstance(node, ast.Name)
                and node.id == "MCPProtocol"
            )
            or (
                isinstance(node, ast.Attribute)
                and node.attr == "MCPProtocol"
            )
        ]
        if uses_facade:
            offenders.setdefault(str(path.relative_to(MCP_ROOT)), []).append("MCPProtocol")

    assert offenders == {}
```

- [x] **Step 2: Run the tests to verify they fail for missing package**

Run:

```bash
source .venv/bin/activate
TEST_MODE=true ENABLE_TRACING=false OTEL_METRICS_EXPORTER=none python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py::test_protocol_reexports_tool_execution_shared_symbols tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py::test_tool_execution_package_does_not_import_protocol_facade -q
```

Expected: the re-export test passes and the package-boundary test fails because `tool_execution/` does not exist yet.

- [x] **Step 3: Leave the failing test in place for Stage 2**

Do not skip or xfail this test. Stage 2 creates the package and makes it pass.

### Task 2: Add Coordinator Characterization Tests

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_tool_execution_coordinator.py`

- [x] **Step 1: Write the failing coordinator tests**

Create the file with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext


@dataclass
class _Prepared:
    name: str
    context: RequestContext


class _Reporter:
    def __init__(self) -> None:
        self.prepare_failures: list[dict[str, Any]] = []

    def should_record(self, context: RequestContext) -> bool:
        return context.metadata.get("mcp_tool_use_observed") is not True

    async def record_prepare_failure(
        self,
        *,
        context: RequestContext,
        params: dict[str, Any],
        exc: Exception,
        start_ts: float,
    ) -> None:
        del start_ts
        self.prepare_failures.append(
            {
                "request_id": context.request_id,
                "name": params.get("name"),
                "error_type": exc.__class__.__name__,
            }
        )


@pytest.mark.asyncio
async def test_tool_execution_coordinator_delegates_prepare_then_execute() -> None:
    from tldw_Server_API.app.core.MCP_unified.tool_execution.coordinator import (
        ToolExecutionCoordinator,
    )

    calls: list[str] = []
    context = RequestContext(request_id="coord-ok", user_id="u1", client_id="c1")
    prepared = _Prepared(name="demo.echo", context=context)

    async def prepare_impl(
        *,
        params: dict[str, Any],
        context: RequestContext,
        idempotency_key: str | None = None,
    ) -> _Prepared:
        assert params == {"name": "demo.echo", "arguments": {"value": "x"}}
        assert idempotency_key is None
        calls.append("prepare")
        return prepared

    async def execute_impl(prepared_call: _Prepared) -> dict[str, Any]:
        assert prepared_call is prepared
        calls.append("execute")
        return {"content": [{"type": "text", "text": "ok"}], "tool": prepared_call.name}

    coordinator = ToolExecutionCoordinator(
        prepare_tool_call_impl=prepare_impl,
        execute_prepared_tool_call_impl=execute_impl,
        reporter=_Reporter(),
    )

    result = await coordinator.handle_tools_call(
        {"name": "demo.echo", "arguments": {"value": "x"}},
        context,
    )

    assert calls == ["prepare", "execute"]
    assert result["tool"] == "demo.echo"


@pytest.mark.asyncio
async def test_tool_execution_coordinator_reports_prepare_failure_before_reraising() -> None:
    from tldw_Server_API.app.core.MCP_unified.protocol import InvalidParamsException
    from tldw_Server_API.app.core.MCP_unified.tool_execution.coordinator import (
        ToolExecutionCoordinator,
    )

    reporter = _Reporter()
    context = RequestContext(request_id="coord-fail", user_id="u1", client_id="c1")

    async def prepare_impl(
        *,
        params: dict[str, Any],
        context: RequestContext,
        idempotency_key: str | None = None,
    ) -> _Prepared:
        del params, context, idempotency_key
        raise InvalidParamsException("bad arguments")

    async def execute_impl(prepared_call: _Prepared) -> dict[str, Any]:
        del prepared_call
        raise AssertionError("execute should not run")

    coordinator = ToolExecutionCoordinator(
        prepare_tool_call_impl=prepare_impl,
        execute_prepared_tool_call_impl=execute_impl,
        reporter=reporter,
    )

    with pytest.raises(InvalidParamsException):
        await coordinator.handle_tools_call({"name": "demo.echo"}, context)

    assert reporter.prepare_failures == [
        {
            "request_id": "coord-fail",
            "name": "demo.echo",
            "error_type": "InvalidParamsException",
        }
    ]
```

- [x] **Step 2: Run the tests to verify they fail for missing coordinator**

Run:

```bash
source .venv/bin/activate
TEST_MODE=true ENABLE_TRACING=false OTEL_METRICS_EXPORTER=none python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_tool_execution_coordinator.py -q
```

Expected: import failure for `tool_execution.coordinator`.

### Task 3: Add Authorization Boundary Characterization Tests

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py`

- [x] **Step 1: Add coarse and deep authorization tests**

Append:

```python
@pytest.mark.asyncio
async def test_tools_call_coarse_authorization_denial_records_denied_tool_use() -> None:
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPRequest

    recorder = _RecordingToolUseRecorder()
    protocol = MCPProtocol()
    protocol._tool_use_recorder = recorder

    async def deny_request(_request: MCPRequest, _context: RequestContext) -> bool:
        return False

    protocol._check_authorization = deny_request  # type: ignore[method-assign]
    context = RequestContext(request_id="coarse-deny", user_id="u1", client_id="c1")
    request = MCPRequest(
        method="tools/call",
        params={"name": "test.read", "arguments": {"value": "x"}},
        id="coarse-deny",
    )

    response = await protocol.process_request(request, context)

    assert response.error is not None
    assert response.error.code == ErrorCode.AUTHORIZATION_ERROR
    assert recorder.events
    event = recorder.events[-1]
    assert event.status == "denied"
    assert event.reason_code == "permission_denied"
    assert event.execution_origin == "failed_before_execution"


@pytest.mark.asyncio
async def test_tools_call_deep_authorization_denial_records_prepare_failure() -> None:
    recorder = _RecordingToolUseRecorder()
    protocol = MCPProtocol()
    protocol._tool_use_recorder = recorder

    async def allow_request(_request: Any, _context: RequestContext) -> bool:
        return True

    async def deny_prepare(
        *,
        params: dict[str, Any],
        context: RequestContext,
        idempotency_key: str | None = None,
    ) -> Any:
        del params, context, idempotency_key
        raise PermissionError("Permission denied for tool: test.read")

    protocol._check_authorization = allow_request  # type: ignore[method-assign]
    protocol.prepare_tool_call = deny_prepare  # type: ignore[method-assign]
    context = RequestContext(request_id="deep-deny", user_id="u1", client_id="c1")

    response = await protocol.process_request(
        {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": "test.read", "arguments": {"value": "x"}},
            "id": "deep-deny",
        },
        context,
    )

    assert response.error is not None
    assert response.error.code == ErrorCode.AUTHORIZATION_ERROR
    assert recorder.events
    event = recorder.events[-1]
    assert event.status == "denied"
    assert event.reason_code == "permission_denied"
    assert event.execution_origin == "failed_before_execution"
```

- [x] **Step 2: Run the authorization tests**

Run:

```bash
source .venv/bin/activate
TEST_MODE=true ENABLE_TRACING=false OTEL_METRICS_EXPORTER=none python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py::test_tools_call_coarse_authorization_denial_records_denied_tool_use tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py::test_tools_call_deep_authorization_denial_records_prepare_failure -q
```

Expected: both tests pass before extraction and continue passing after each stage.

## Stage 2: Shared Types And Package Skeleton

**Goal:** Create import-safe shared types and an empty extraction package.

**Success Criteria:** Compatibility imports from `protocol.py` still work; the package-boundary test passes.

**Tests:** Stage 1 tests plus focused py_compile.

### Task 4: Move Shared Types Into `protocol_types.py`

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/protocol_types.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/protocol.py`

- [x] **Step 1: Create `protocol_types.py`**

Move these exact existing definitions from `protocol.py` into the new file:

```python
class InvalidParamsException(Exception):
    """Raised when tool parameters fail validation or validators are missing for write tools."""
    pass


class GovernanceDeniedError(PermissionError):
    """Permission error carrying structured governance decision details."""

    def __init__(self, message: str, governance: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.governance = governance or {}


class ApprovalRequiredError(PermissionError):
    """Permission error carrying structured MCP Hub approval request details."""

    def __init__(self, message: str, approval: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.approval = approval or {}
```

Also move the existing `RequestContext`, `_TrustedCompatClaimsSentinel`, `_trusted_compat_claims_metadata`, `_has_trusted_compat_claims`, and `PreparedToolCall` definitions unchanged. Include these imports at the top:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger

from .modules.base import BaseModule
```

- [x] **Step 2: Re-export from `protocol.py`**

Replace the moved definitions in `protocol.py` with:

```python
from .protocol_types import (
    ApprovalRequiredError,
    GovernanceDeniedError,
    InvalidParamsException,
    PreparedToolCall,
    RequestContext,
    _has_trusted_compat_claims,
    _trusted_compat_claims_metadata,
)
```

- [x] **Step 3: Run compatibility tests**

Run:

```bash
source .venv/bin/activate
TEST_MODE=true ENABLE_TRACING=false OTEL_METRICS_EXPORTER=none python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py::test_protocol_reexports_tool_execution_shared_symbols tldw_Server_API/app/core/MCP_unified/tests/test_protocol_scope_enforcement.py -q
```

Expected: all selected tests pass.

### Task 5: Create Tool-Execution Package Skeleton

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/tool_execution/__init__.py`
- Create: `tldw_Server_API/app/core/MCP_unified/tool_execution/models.py`
- Create: `tldw_Server_API/app/core/MCP_unified/tool_execution/reporting.py`
- Create: `tldw_Server_API/app/core/MCP_unified/tool_execution/dependencies.py`
- Create: `tldw_Server_API/app/core/MCP_unified/tool_execution/coordinator.py`
- Create: `tldw_Server_API/app/core/MCP_unified/tool_execution/security.py`
- Create: `tldw_Server_API/app/core/MCP_unified/tool_execution/hooks.py`
- Create: `tldw_Server_API/app/core/MCP_unified/tool_execution/runtime.py`

- [x] **Step 1: Add package marker**

Use:

```python
"""Tool execution pipeline for the MCP protocol facade."""

from .coordinator import ToolExecutionCoordinator
from .dependencies import CompatibilityCallbackLedgerEntry, ToolExecutionDependencies
from .reporting import ToolExecutionReporter

__all__ = [
    "CompatibilityCallbackLedgerEntry",
    "ToolExecutionCoordinator",
    "ToolExecutionDependencies",
    "ToolExecutionReporter",
]
```

- [x] **Step 2: Add internal models**

Use:

```python
"""Internal models for staged MCP tool execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolResolution:
    tool_name: str
    tool_args: Any
    module: Any
    module_id: str | None
    tool_def: dict[str, Any] | None
    is_write: bool | None


@dataclass(slots=True)
class PolicyEvaluation:
    effective_policy: dict[str, Any] | None
    external_access_result: dict[str, Any] = field(default_factory=dict)
    path_scope_result: dict[str, Any] = field(default_factory=dict)
    scope_payload: dict[str, Any] | None = None
    within_effective_policy: bool = True
    within_resolved_scope: bool = True
```

- [x] **Step 3: Add reporter facade**

Use:

```python
"""Tool-use reporting facade for MCP tool execution."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from mcp_unified.tool_use_reporting.models import ToolUseEvent

from ..protocol_types import RequestContext


@dataclass(slots=True)
class ToolExecutionReporter:
    should_record: Callable[[RequestContext], bool]
    record_event: Callable[[ToolUseEvent], Awaitable[None]]
    build_event: Callable[..., ToolUseEvent]
    duration_ms: Callable[[float], float]
    execution_origin_for_failure: Callable[[str], str]
    record_process_request_failure_impl: Callable[..., Awaitable[None]]

    async def record_process_request_failure(self, **kwargs: Any) -> None:
        await self.record_process_request_failure_impl(**kwargs)

    async def record_prepare_failure(
        self,
        *,
        context: RequestContext,
        params: dict[str, Any],
        exc: Exception,
        start_ts: float,
    ) -> None:
        from mcp_unified.tool_use_reporting.builders import classify_tool_use_exception

        from ..protocol_types import GovernanceDeniedError

        if not self.should_record(context):
            return
        status, reason_code = classify_tool_use_exception(exc)
        scope_payload = None
        if isinstance(exc, GovernanceDeniedError) and isinstance(exc.governance, dict):
            path_scope = exc.governance.get("path_scope")
            if isinstance(path_scope, dict):
                scope_payload = path_scope
        event = self.build_event(
            context=context,
            requested_tool_name=params.get("name") if isinstance(params, dict) else None,
            status=status,
            execution_origin="failed_before_execution",
            duration_ms=self.duration_ms(start_ts),
            reason_code=reason_code,
            tool_args=params.get("arguments") if isinstance(params, dict) else None,
            scope_payload=scope_payload,
        )
        await self.record_event(event)
```

- [x] **Step 4: Add dependencies dataclasses**

Use:

```python
"""Explicit dependencies for the MCP tool execution pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .reporting import ToolExecutionReporter


@dataclass(frozen=True, slots=True)
class CompatibilityCallbackLedgerEntry:
    callback: str
    current_owner: str
    target_owner: str
    removal_stage: str
    parity_test: str


@dataclass(slots=True)
class ToolExecutionDependencies:
    module_registry: Any
    rbac_policy: Any
    rate_limiter: Any
    metrics: Any
    telemetry: Any
    hook_manager: Any
    tool_use_recorder: Any
    idempotency: Any
    config_provider: Callable[[], Any]
    effective_policy_resolver: Any
    path_scope_enforcer: Any
    approval_evaluator: Any
    external_access_evaluator: Any
    reporter: ToolExecutionReporter
```

- [x] **Step 5: Add coordinator**

Use:

```python
"""Coordinator for the MCP tools/call execution path."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger

from ..protocol_types import RequestContext
from .reporting import ToolExecutionReporter


@dataclass(slots=True)
class ToolExecutionCoordinator:
    prepare_tool_call_impl: Callable[..., Awaitable[Any]]
    execute_prepared_tool_call_impl: Callable[[Any], Awaitable[dict[str, Any]]]
    reporter: ToolExecutionReporter

    async def handle_tools_call(
        self,
        params: dict[str, Any],
        context: RequestContext,
    ) -> dict[str, Any]:
        start_ts = time.time()
        try:
            prepared = await self.prepare_tool_call(params=params, context=context)
        except Exception as exc:
            try:
                await self.reporter.record_prepare_failure(
                    context=context,
                    params=params,
                    exc=exc,
                    start_ts=start_ts,
                )
            except Exception as record_exc:
                logger.warning(
                    "Failed to build or record prepare-failure tool-use event: {}",
                    record_exc.__class__.__name__,
                )
            raise
        return await self.execute_prepared_tool_call(prepared)

    async def prepare_tool_call(
        self,
        params: dict[str, Any],
        context: RequestContext,
        idempotency_key: str | None = None,
    ) -> Any:
        return await self.prepare_tool_call_impl(
            params=params,
            context=context,
            idempotency_key=idempotency_key,
        )

    async def execute_prepared_tool_call(self, prepared: Any) -> dict[str, Any]:
        return await self.execute_prepared_tool_call_impl(prepared)
```

- [x] **Step 6: Add empty implementation modules**

Use this for `security.py`, `hooks.py`, and `runtime.py`:

```python
"""Focused helper module for MCP tool execution."""

from __future__ import annotations
```

- [x] **Step 7: Run skeleton tests**

Run:

```bash
source .venv/bin/activate
TEST_MODE=true ENABLE_TRACING=false OTEL_METRICS_EXPORTER=none python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py::test_tool_execution_package_does_not_import_protocol_facade tldw_Server_API/app/core/MCP_unified/tests/test_tool_execution_coordinator.py -q
```

Expected: all selected tests pass.

## Stage 3: Coordinator Delegation In `MCPProtocol`

**Goal:** Route `tools/call`, `prepare_tool_call()`, and `execute_prepared_tool_call()` through the coordinator while the original logic remains in protocol-private compatibility methods.

**Success Criteria:** Public methods still exist and behavior is unchanged; the compatibility ledger has exactly two high-value behavior callbacks at the end of this stage.

**Tests:** Coordinator tests, authorization boundary tests, idempotency tests, tool-use reporting tests.

### Task 6: Wire Coordinator Into `MCPProtocol`

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/protocol.py`

- [x] **Step 1: Import tool-execution classes**

Add:

```python
from .tool_execution import ToolExecutionCoordinator, ToolExecutionDependencies, ToolExecutionReporter
```

- [x] **Step 2: Build reporter and dependency objects in `__init__`**

After `_governance_lock` is initialized, add:

```python
        self._tool_execution_reporter = ToolExecutionReporter(
            should_record=self._should_record_tool_use,
            record_event=self._record_tool_use_event,
            build_event=self._build_tool_use_event,
            duration_ms=self._tool_use_duration_ms,
            execution_origin_for_failure=self._tool_use_execution_origin_for_failure,
            record_process_request_failure_impl=self._record_process_request_tool_use_failure,
        )
        self._tool_execution_dependencies = ToolExecutionDependencies(
            module_registry=self.module_registry,
            rbac_policy=self.rbac_policy,
            rate_limiter=self.rate_limiter,
            metrics=self.metrics,
            telemetry=self.telemetry,
            hook_manager=self._tool_call_hook_manager,
            tool_use_recorder=self._tool_use_recorder,
            idempotency=self._idempotency,
            config_provider=get_config,
            effective_policy_resolver=self.dependencies.effective_policy_resolver,
            path_scope_enforcer=self.dependencies.path_scope_enforcer,
            approval_evaluator=self.dependencies.approval_evaluator,
            external_access_evaluator=self.dependencies.external_access_evaluator,
            reporter=self._tool_execution_reporter,
        )
        self._tool_execution = ToolExecutionCoordinator(
            prepare_tool_call_impl=self._prepare_tool_call_inline,
            execute_prepared_tool_call_impl=self._execute_prepared_tool_call_inline,
            reporter=self._tool_execution_reporter,
        )
```

- [x] **Step 3: Rename original implementation methods**

Rename:

```python
async def prepare_tool_call(
```

to:

```python
async def _prepare_tool_call_inline(
```

Rename:

```python
async def execute_prepared_tool_call(self, prepared: PreparedToolCall) -> dict[str, Any]:
```

to:

```python
async def _execute_prepared_tool_call_inline(self, prepared: PreparedToolCall) -> dict[str, Any]:
```

- [x] **Step 4: Replace public methods with delegators**

Add public methods where the renamed methods used to be:

```python
    async def prepare_tool_call(
        self,
        params: dict[str, Any],
        context: RequestContext,
        idempotency_key: str | None = None,
    ) -> PreparedToolCall:
        """Prepare a tool invocation through protocol policy, validation, and governance checks."""
        return await self._tool_execution.prepare_tool_call(
            params=params,
            context=context,
            idempotency_key=idempotency_key,
        )

    async def execute_prepared_tool_call(self, prepared: PreparedToolCall) -> dict[str, Any]:
        """Execute a previously prepared tool invocation."""
        return await self._tool_execution.execute_prepared_tool_call(prepared)
```

- [x] **Step 5: Replace `_handle_tools_call` body with coordinator delegation**

Use:

```python
    async def _handle_tools_call(
        self,
        params: dict[str, Any],
        context: RequestContext
    ) -> dict[str, Any]:
        """Execute a tool."""
        return await self._tool_execution.handle_tools_call(params, context)
```

- [x] **Step 6: Run focused tests**

Run:

```bash
source .venv/bin/activate
TEST_MODE=true ENABLE_TRACING=false OTEL_METRICS_EXPORTER=none python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_tool_execution_coordinator.py tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py::test_tools_call_coarse_authorization_denial_records_denied_tool_use tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py::test_tools_call_deep_authorization_denial_records_prepare_failure tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_and_category.py::test_idempotency_dedupes_write_calls -q
```

Expected: all selected tests pass.

## Stage 4a: Extract Validation, Resolution, Hardening, And Integrity

**Goal:** Move pure and near-pure tool preparation helpers into `ToolExecutionSecurity`.

**Success Criteria:** `protocol.py` wrappers delegate to `ToolExecutionSecurity`; no extracted module imports `protocol.py`.

**Tests:** validation/sanitization, prepared integrity, import-boundary tests.

### Task 7: Implement `ToolExecutionSecurity` Core Helpers

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tool_execution/security.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/protocol.py`

- [x] **Step 1: Add `ToolExecutionSecurity` constructor**

Use:

```python
"""Security stages for MCP tool execution."""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from ..modules.base import BaseModule
from ..protocol_types import InvalidParamsException, PreparedToolCall, RequestContext
from .dependencies import ToolExecutionDependencies


class ToolExecutionSecurity:
    def __init__(
        self,
        *,
        dependencies: ToolExecutionDependencies,
        tool_name_re: re.Pattern[str],
        prepared_call_secret: bytes,
        noncritical_exceptions: tuple[type[BaseException], ...],
    ) -> None:
        self.dependencies = dependencies
        self.module_registry = dependencies.module_registry
        self.rbac_policy = dependencies.rbac_policy
        self.metrics = dependencies.metrics
        self._tool_name_re = tool_name_re
        self._prepared_call_secret = prepared_call_secret
        self._noncritical_exceptions = noncritical_exceptions
```

- [x] **Step 2: Move helper methods unchanged into the class**

Move these existing methods from `MCPProtocol` into `ToolExecutionSecurity` and update `self.metrics`, `self.module_registry`, and `self.dependencies` references to the class attributes above:

```text
_hash_arguments
_resolve_tool_definition
_classify_write_tool_call
_resolve_write_classification
_strip_forbidden_tool_argument_overrides
_harden_and_sanitize_tool_arguments
_prepared_tool_call_payload
_build_prepared_tool_call_integrity_tag
_verify_prepared_tool_call_integrity
_fingerprint_request_context
_context_json_safe
_normalize_idempotency_key
_validate_input_schema
```

- [x] **Step 3: Construct security helper in protocol `__init__`**

After `_tool_execution_dependencies`, add:

```python
        from .tool_execution.security import ToolExecutionSecurity

        self._tool_execution_security = ToolExecutionSecurity(
            dependencies=self._tool_execution_dependencies,
            tool_name_re=self._tool_name_re,
            prepared_call_secret=self._prepared_call_secret,
            noncritical_exceptions=_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS,
        )
```

- [x] **Step 4: Keep compatibility wrappers in `protocol.py`**

Replace each moved method body with a direct wrapper:

```python
    @staticmethod
    def _hash_arguments(arguments: dict[str, Any]) -> Optional[str]:
        return ToolExecutionSecurity.hash_arguments(arguments)
```

For instance methods, use:

```python
    async def _resolve_tool_definition(
        self,
        module: BaseModule,
        tool_name: str,
    ) -> Optional[dict[str, Any]]:
        return await self._tool_execution_security.resolve_tool_definition(module, tool_name)
```

Apply this same wrapper pattern for all moved helpers. Preserve public compatibility for existing tests that access private protocol helpers.

- [x] **Step 5: Update inline prepare/execute calls to use security helper directly**

Inside `_prepare_tool_call_inline` and `_execute_prepared_tool_call_inline`, replace `self._resolve_tool_definition`, `self._harden_and_sanitize_tool_arguments`, `self._resolve_write_classification`, `self._validate_input_schema`, `self._normalize_idempotency_key`, `self._hash_arguments`, and integrity calls with `self._tool_execution_security.<method_name>`.

- [x] **Step 6: Run tests**

Run:

```bash
source .venv/bin/activate
TEST_MODE=true ENABLE_TRACING=false OTEL_METRICS_EXPORTER=none python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_validation_and_sanitization.py tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py::test_tool_execution_package_does_not_import_protocol_facade tldw_Server_API/tests/MCP_unified/test_phase3_3_small_core_sanitizers.py -q
```

Expected: all selected tests pass.

## Stage 4b: Extract Tool Authorization And Scope Checks

**Goal:** Move tool-specific RBAC, MCP scope, API-key scope, and alias authorization helpers into `ToolExecutionSecurity`.

**Success Criteria:** Tool preparation uses security helper methods for module/tool permission gates; protocol resource/prompt handlers keep behavior through wrappers.

**Tests:** allowed-tools, scope enforcement, sandbox auth binding, HTTP auth paths.

### Task 8: Move Tool Authorization Helpers

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tool_execution/security.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/protocol.py`

- [x] **Step 1: Move helper methods into `ToolExecutionSecurity`**

Move these existing methods and keep behavior unchanged:

```text
_rbac_check
_scoped_permissions
_mcp_scopes
_api_key_scopes
_api_key_scope_level
_api_key_allows
_scope_matches
_scope_allows
_has_module_permission
_has_tool_permission
_extract_allowed_tools
_extract_tool_command
_matches_allowed_tool_pattern
_tool_authorization_names
_matches_tool_authorization_pattern
_scope_allows_tool_name
_is_tool_allowed_by_context
```

Import these in `security.py`:

```python
from ..auth.rbac import Action, Resource
from ..protocol_types import _has_trusted_compat_claims
```

- [x] **Step 2: Keep protocol compatibility wrappers**

For each moved method, leave a `protocol.py` wrapper. Example:

```python
    async def _has_tool_permission(
        self,
        context: RequestContext,
        tool_name: str,
        *,
        is_write: Optional[bool] = None,
    ) -> bool:
        return await self._tool_execution_security.has_tool_permission(
            context,
            tool_name,
            is_write=is_write,
        )
```

- [x] **Step 3: Update `_check_authorization` and `_handle_tools_list`**

Replace direct protocol calls for tool and scope logic with wrapper calls that delegate to security. Keep `_has_resource_permission`, `_has_prompt_permission`, and `_has_namespaced_prompt_permission` in `protocol.py` because non-tool handlers still own resource and prompt behavior.

- [x] **Step 4: Run tests**

Run:

```bash
source .venv/bin/activate
TEST_MODE=true ENABLE_TRACING=false OTEL_METRICS_EXPORTER=none python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_scope_enforcement.py tldw_Server_API/tests/MCP_unified/test_sandbox_module_auth_binding.py tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py::test_tool_execution_package_does_not_import_protocol_facade -q
```

Expected: all selected tests pass.

## Stage 4c: Extract Effective Policy, External Access, Path Scope, Approval, Governance, And Pre-Hooks

**Goal:** Finish prepare-phase extraction so `ToolExecutionSecurity.prepare_tool_call()` owns the full security gate order.

**Success Criteria:** `prepare_tool_call_impl` ledger entry is removed; coordinator calls `ToolExecutionSecurity.prepare_tool_call` directly.

**Tests:** governance preflight, path scope, external federation, tool hooks, stage-order tests.

### Task 9: Extract Hook Helper

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tool_execution/hooks.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/protocol.py`

- [x] **Step 1: Implement `ToolExecutionHooks`**

Move these existing methods from `MCPProtocol` into `ToolExecutionHooks`:

```text
_hook_safe_copy
_hook_safe_metadata
_hook_safe_tool_args
_redact_hook_visible_tool_args
_hook_safe_scope_payload
_build_tool_hook_context
_coerce_tool_hook_action
_coerce_tool_hook_decision
_tool_hook_payload
_run_pre_tool_hooks
_run_post_tool_hooks
```

Use this constructor:

```python
class ToolExecutionHooks:
    def __init__(
        self,
        *,
        hook_manager: Any,
        reporter: Any,
        noncritical_exceptions: tuple[type[BaseException], ...],
    ) -> None:
        self._tool_call_hook_manager = hook_manager
        self._reporter = reporter
        self._noncritical_exceptions = noncritical_exceptions
```

- [x] **Step 2: Construct hooks helper in protocol `__init__`**

Add:

```python
        from .tool_execution.hooks import ToolExecutionHooks

        self._tool_execution_hooks = ToolExecutionHooks(
            hook_manager=self._tool_call_hook_manager,
            reporter=self._tool_execution_reporter,
            noncritical_exceptions=_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS,
        )
```

- [x] **Step 3: Keep protocol wrappers**

Replace moved hook methods in `protocol.py` with wrappers that call `self._tool_execution_hooks`.

- [x] **Step 4: Run hook tests**

Run:

```bash
source .venv/bin/activate
TEST_MODE=true ENABLE_TRACING=false OTEL_METRICS_EXPORTER=none python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_protocol_tool_hooks.py tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py::test_protocol_records_pre_hook_denial_metadata_without_raw_payload -q
```

Expected: all selected tests pass.

### Task 10: Move Policy And Governance Helpers

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tool_execution/security.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/protocol.py`

- [x] **Step 1: Move policy/governance methods into `ToolExecutionSecurity`**

Move:

```text
_resolve_effective_tool_policy
_is_tool_allowed_by_effective_policy
_evaluate_runtime_approval
_evaluate_path_scope
_extract_path_scope_candidates
_evaluate_external_access
_governance_preflight_bypassed
_governance_summary
_resolve_governance_category
_resolve_governance_rollout_mode
_record_governance_check
_serialize_governance_decision
_ensure_governance_service
_run_governance_preflight
```

Add these instance fields to `ToolExecutionSecurity.__init__`:

```python
        self._governance_service: Any | None = None
        self._governance_store: Any | None = None
        self._governance_lock = asyncio.Lock()
```

- [x] **Step 2: Move prepare implementation into `ToolExecutionSecurity.prepare_tool_call`**

Move the complete body of `_prepare_tool_call_inline` into a new method with this exact signature:

```python
    async def prepare_tool_call(
        self,
        params: dict[str, Any],
        context: RequestContext,
        idempotency_key: str | None = None,
        hooks: Any | None = None,
    ) -> PreparedToolCall:
        """Prepare a tool invocation through protocol policy, validation, and governance checks."""
```

Inside the moved method, replace the old pre-hook call:

```python
await self._run_pre_tool_hooks(
    tool_name=tool_name,
    tool_args=tool_args,
    module_id=module_id,
    tool_def=tool_def if isinstance(tool_def, dict) else None,
    is_write=is_write,
    arguments_hash=args_hash,
    context=context,
    scope_payload=scope_payload,
)
```

with:

```python
if hooks is None:
    raise RuntimeError("ToolExecutionHooks dependency is required")
await hooks.run_pre_tool_hooks(
    tool_name=tool_name,
    tool_args=tool_args,
    module_id=module_id,
    tool_def=tool_def if isinstance(tool_def, dict) else None,
    is_write=is_write,
    arguments_hash=args_hash,
    context=context,
    scope_payload=scope_payload,
)
```

- [x] **Step 3: Remove `prepare_tool_call_impl` callback from coordinator construction**

Create a wrapper in `MCPProtocol.__init__` before coordinator construction so the extracted prepare method always receives the hook dependency:

```python
        async def _prepare_with_hooks(
            *,
            params: dict[str, Any],
            context: RequestContext,
            idempotency_key: str | None = None,
        ) -> PreparedToolCall:
            return await self._tool_execution_security.prepare_tool_call(
                params=params,
                context=context,
                idempotency_key=idempotency_key,
                hooks=self._tool_execution_hooks,
            )
```

Then pass `prepare_tool_call_impl=_prepare_with_hooks`.

- [x] **Step 4: Leave `_prepare_tool_call_inline` as a compatibility delegator**

Use:

```python
    async def _prepare_tool_call_inline(
        self,
        params: dict[str, Any],
        context: RequestContext,
        idempotency_key: str | None = None,
    ) -> PreparedToolCall:
        return await self._tool_execution_security.prepare_tool_call(
            params=params,
            context=context,
            idempotency_key=idempotency_key,
            hooks=self._tool_execution_hooks,
        )
```

- [x] **Step 5: Run policy and hook tests**

Run:

```bash
source .venv/bin/activate
TEST_MODE=true ENABLE_TRACING=false OTEL_METRICS_EXPORTER=none python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_protocol_governance_preflight.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_path_scope_candidates.py tldw_Server_API/tests/MCP_unified/test_mcp_protocol_path_scope.py tldw_Server_API/tests/MCP_unified/test_mcp_protocol_external_federation.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_tool_hooks.py tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py::test_tool_execution_package_does_not_import_protocol_facade -q
```

Expected: all selected tests pass.

## Stage 5: Extract Runtime And Idempotency

**Goal:** Move execution-phase behavior into `ToolExecutionRuntime`.

**Success Criteria:** `execute_prepared_tool_call_impl` ledger entry is removed; `IdempotencyManager` stays import-compatible from `protocol.py` and is injected into runtime through `ToolExecutionDependencies`; nested `run` behavior still works.

**Tests:** idempotency/category, run command module, reporting, execution errors.

### Task 11: Move Runtime Execution

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tool_execution/runtime.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/protocol.py`

- [x] **Step 1: Implement `ToolExecutionRuntime` constructor**

Use:

```python
"""Runtime execution phase for prepared MCP tool calls."""

from __future__ import annotations

import contextlib
import time
from typing import Any

from loguru import logger

from mcp_unified.tool_use_reporting.builders import classify_tool_use_exception

from ..auth.rate_limiter import RateLimitExceeded
from ..execution_eval import (
    attach_execution_eval_metadata,
    execution_eval_metadata_from_tool_definition,
    sanitize_eval_profile_id,
)
from ..protocol_types import InvalidParamsException, PreparedToolCall, RequestContext
from .dependencies import ToolExecutionDependencies


class ToolExecutionRuntime:
    def __init__(
        self,
        *,
        dependencies: ToolExecutionDependencies,
        security: Any,
        hooks: Any,
        noncritical_exceptions: tuple[type[BaseException], ...],
        tool_execution_error: str,
        generic_exception_like: Any,
        make_idempotency_cache_key: Any,
    ) -> None:
        self.dependencies = dependencies
        self.security = security
        self.hooks = hooks
        self.rate_limiter = dependencies.rate_limiter
        self.metrics = dependencies.metrics
        self.telemetry = dependencies.telemetry
        self.idempotency = dependencies.idempotency
        self.reporter = dependencies.reporter
        self.config_provider = dependencies.config_provider
        self._noncritical_exceptions = noncritical_exceptions
        self._tool_execution_error = tool_execution_error
        self._generic_exception_like = generic_exception_like
        self._make_idempotency_cache_key = make_idempotency_cache_key
```

- [x] **Step 2: Move runtime helpers**

Move from `MCPProtocol` into `ToolExecutionRuntime`:

```text
_extract_eval_profile_id
```

Move the complete body of `_execute_prepared_tool_call_inline` into a new method with this exact signature:

```python
    async def execute_prepared_tool_call(self, prepared: PreparedToolCall) -> dict[str, Any]:
        """Execute a previously prepared tool invocation."""
```

Replace:

```python
self._verify_prepared_tool_call_integrity(prepared)
```

with:

```python
self.security.verify_prepared_tool_call_integrity(prepared)
```

Replace `get_config()` with `self.config_provider()`.

- [x] **Step 3: Construct runtime helper in protocol `__init__`**

Add:

```python
        from .tool_execution.runtime import ToolExecutionRuntime

        self._tool_execution_runtime = ToolExecutionRuntime(
            dependencies=self._tool_execution_dependencies,
            security=self._tool_execution_security,
            hooks=self._tool_execution_hooks,
            noncritical_exceptions=_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS,
            tool_execution_error=_MCP_TOOL_EXECUTION_ERROR,
            generic_exception_like=self._generic_exception_like,
            make_idempotency_cache_key=self._make_idempotency_cache_key,
        )
```

- [x] **Step 4: Point coordinator at runtime**

Change coordinator construction to:

```python
            execute_prepared_tool_call_impl=self._tool_execution_runtime.execute_prepared_tool_call,
```

- [x] **Step 5: Keep public compatibility wrappers**

Use:

```python
    async def _execute_prepared_tool_call_inline(self, prepared: PreparedToolCall) -> dict[str, Any]:
        return await self._tool_execution_runtime.execute_prepared_tool_call(prepared)
```

Keep `IdempotencyManager` defined or re-exported from `protocol.py`. Move `_make_idempotency_cache_key` behavior into `ToolExecutionRuntime.make_idempotency_cache_key`, and leave `MCPProtocol._make_idempotency_cache_key` as a compatibility wrapper that calls runtime.

- [x] **Step 6: Run runtime tests**

Run:

```bash
source .venv/bin/activate
TEST_MODE=true ENABLE_TRACING=false OTEL_METRICS_EXPORTER=none python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_and_category.py tldw_Server_API/app/core/MCP_unified/tests/test_run_command_module.py tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py::test_tool_execution_package_does_not_import_protocol_facade -q
```

Expected: all selected tests pass.

## Stage 6: Extract Reporting Internals

**Goal:** Move reporting/audit helper internals behind `ToolExecutionReporter` while preserving all event fields and failure swallowing.

**Success Criteria:** Protocol owns only reporter construction and process-level error mapping; event construction lives in `reporting.py`.

**Tests:** full tool-use reporting matrix, logging redaction, process-request failure tests.

### Task 12: Move Tool-Use Reporting Helpers

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tool_execution/reporting.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/protocol.py`

- [x] **Step 1: Convert `ToolExecutionReporter` from callback facade to implementation class**

Move these existing methods from `MCPProtocol` into `ToolExecutionReporter`:

```text
_should_record_tool_use
_record_tool_use_event
_safe_tool_use_name
_tool_use_duration_ms
_tool_use_execution_origin_for_failure
_tool_use_eval_metadata
_tool_use_file_policy_decisions
_tool_use_hook_results
_tool_hook_summary_items
_append_tool_hook_summary
_tool_use_decision_grant_outcome
_tool_use_value_present
_tool_use_contains_key
_tool_use_category
_build_tool_use_event
_record_process_request_tool_use_failure
_audit_tool_event
```

Use this constructor:

```python
class ToolExecutionReporter:
    def __init__(
        self,
        *,
        recorder: Any,
        metrics: Any,
        tool_name_re: Any,
        noncritical_exceptions: tuple[type[BaseException], ...],
    ) -> None:
        self._tool_use_recorder = recorder
        self.metrics = metrics
        self._tool_name_re = tool_name_re
        self._noncritical_exceptions = noncritical_exceptions
```

- [x] **Step 2: Update reporter construction**

Replace the callback-style reporter construction with:

```python
        self._tool_execution_reporter = ToolExecutionReporter(
            recorder=self._tool_use_recorder,
            metrics=self.metrics,
            tool_name_re=self._tool_name_re,
            noncritical_exceptions=_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS,
        )
```

- [x] **Step 3: Keep protocol wrappers**

For each moved method, leave a wrapper in `protocol.py`. Example:

```python
    def _should_record_tool_use(self, context: RequestContext) -> bool:
        return self._tool_execution_reporter.should_record_tool_use(context)
```

Use wrapper names that preserve current protocol-private call sites until Stage 7 removes dead wrappers.

- [x] **Step 4: Run reporting tests**

Run:

```bash
source .venv/bin/activate
TEST_MODE=true ENABLE_TRACING=false OTEL_METRICS_EXPORTER=none python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py tldw_Server_API/app/core/MCP_unified/tests/test_log_redaction.py tldw_Server_API/app/core/MCP_unified/tests/test_jsonrpc_notifications.py -q
```

Expected: all selected tests pass.

## Stage 7: Clean Protocol Facade

**Goal:** Remove dead tool-path helpers from `protocol.py` while keeping compatibility exports and non-tool handlers intact.

**Success Criteria:** `protocol.py` owns JSON-RPC facade responsibilities only; no ledger entries remain.

**Tests:** focused MCP protocol/tool slice, py_compile, Bandit.

### Task 13: Remove Dead Tool-Path Helpers

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/protocol.py`
- Modify: `Docs/superpowers/plans/2026-06-24-mcp-protocol-tool-execution-refactor.md`
- Modify: `backlog/tasks/task-2424 - Verify-and-remediate-MCP-Unified-review-findings.md`

- [x] **Step 1: Remove wrappers with no in-repo callers**

Use `rg` to confirm each wrapper has no caller outside tests or compatibility imports before removing it:

```bash
rg -n "_prepare_tool_call_inline|_execute_prepared_tool_call_inline|_build_tool_use_event|_run_pre_tool_hooks|_run_post_tool_hooks|_resolve_effective_tool_policy|_evaluate_path_scope|_evaluate_external_access|_evaluate_runtime_approval" tldw_Server_API/app/core/MCP_unified tldw_Server_API/tests/MCP_unified
```

Remove only wrappers that are not required by existing tests or external compatibility.

Final check: no wrappers were removed because each searched wrapper is still used by an in-repo test, compatibility monkeypatch seam, or run-command integration.

- [x] **Step 2: Keep compatibility exports**

Ensure `protocol.py` still imports and exposes:

```python
ApprovalRequiredError
GovernanceDeniedError
IdempotencyManager
InvalidParamsException
MCPError
MCPProtocol
MCPRequest
MCPResponse
PreparedToolCall
RequestContext
_trusted_compat_claims_metadata
```

- [x] **Step 3: Update the ledger**

Mark every compatibility callback as removed or resolved in the task notes. The final ledger state must be empty.

- [x] **Step 4: Run final verification**

Run:

```bash
source .venv/bin/activate
python -m py_compile tldw_Server_API/app/core/MCP_unified/protocol.py tldw_Server_API/app/core/MCP_unified/protocol_types.py tldw_Server_API/app/core/MCP_unified/tool_execution/*.py
```

Run:

```bash
source .venv/bin/activate
TEST_MODE=true ENABLE_TRACING=false OTEL_METRICS_EXPORTER=none python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_tool_execution_coordinator.py tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_and_category.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_scope_enforcement.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_governance_preflight.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_tool_hooks.py tldw_Server_API/app/core/MCP_unified/tests/test_run_command_module.py -q
```

Run:

```bash
source .venv/bin/activate
python -m bandit -r tldw_Server_API/app/core/MCP_unified/protocol.py tldw_Server_API/app/core/MCP_unified/protocol_types.py tldw_Server_API/app/core/MCP_unified/tool_execution -f json -o /tmp/bandit_mcp_tool_execution_refactor.json
```

Expected:

```text
py_compile exits 0
focused pytest slice passes
Bandit reports 0 findings for the touched scope
```

## Final Self-Review Checklist

- [x] Every spec goal maps to a stage in this plan.
- [x] The plan does not change JSON-RPC response shapes or HTTP error mappings.
- [x] `MCPProtocol` remains the JSON-RPC facade and owns non-tool handlers.
- [x] Extracted modules do not import `MCPProtocol` or `MCP_unified.protocol`.
- [x] Coarse authorization and deep tool authorization are both tested.
- [x] `ToolExecutionReporter` is present before coordinator extraction changes reporting call sites.
- [x] `IdempotencyManager` is injected into runtime and remains import-compatible from `protocol.py` before Stage 5 completes.
- [x] The callback ledger is empty by the end of Stage 7.
- [x] Focused tests and Bandit are recorded in `TASK-2424`.
