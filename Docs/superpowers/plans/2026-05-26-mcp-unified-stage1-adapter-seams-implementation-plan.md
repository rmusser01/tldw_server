# MCP Unified Stage 1 Adapter Seams Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare MCP Unified for standalone extraction by adding explicit adapter seams, boundary tests, and a module ownership inventory while preserving all current `tldw_server` MCP behavior.

**Architecture:** This first slice keeps all runtime code inside `tldw_Server_API/app/core/MCP_unified` and introduces interfaces plus default `tldw_server` adapters in place. Existing import paths, route behavior, singleton shims, MCP Hub resolution, AuthNZ/RBAC, path scopes, approvals, and credential behavior must remain compatible. No new standalone `mcp_unified` package or gateway entrypoint is created in this slice.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic, pytest, Loguru, existing MCP Unified runtime and tests.

---

## Source Spec

- Design spec: `Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md`
- Backlog task: `TASK-480`

## Scope Boundary

In scope:

- Adapter protocol files under `tldw_Server_API/app/core/MCP_unified/interfaces/`
- Default host adapter files under `tldw_Server_API/app/core/MCP_unified/adapters/`
- Narrow constructor injection seams for `MCPProtocol`, `MCPServer`, and `BaseModule`
- Boundary and compatibility tests
- Module ownership inventory document

Out of scope:

- Creating a top-level standalone `mcp_unified` package
- Moving modules out of `tldw_Server_API`
- Building the standalone gateway
- Changing `/api/v1/mcp/*` route contracts
- Changing MCP Hub persistence or AuthNZ schemas

## File Structure

Create:

- `tldw_Server_API/app/core/MCP_unified/interfaces/__init__.py`
- `tldw_Server_API/app/core/MCP_unified/interfaces/runtime.py`
- `tldw_Server_API/app/core/MCP_unified/interfaces/policy.py`
- `tldw_Server_API/app/core/MCP_unified/interfaces/storage.py`
- `tldw_Server_API/app/core/MCP_unified/adapters/__init__.py`
- `tldw_Server_API/app/core/MCP_unified/adapters/tldw_runtime.py`
- `tldw_Server_API/app/core/MCP_unified/adapters/tldw_policy.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py`
- `Docs/MCP/mcp_unified_module_ownership_inventory.md`

Modify:

- `tldw_Server_API/app/core/MCP_unified/protocol.py`
- `tldw_Server_API/app/core/MCP_unified/server.py`
- `tldw_Server_API/app/core/MCP_unified/modules/base.py`
- `tldw_Server_API/app/core/MCP_unified/__init__.py`
- `backlog/tasks/task-480 - Design-MCP-Unified-standalone-library-and-gateway-extraction.md`

Do not modify:

- `tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py` unless a compatibility test proves the adapter seam cannot be installed without endpoint changes.
- Any domain module implementation beyond what is needed to inventory it.

## Task 1: Add Boundary Tests For The Future Extraction Contract

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py`

- [x] **Step 1: Write failing import-boundary tests**

Add tests that describe the intended Stage 1 contract before implementation.

```python
from __future__ import annotations

import ast
from pathlib import Path

import pytest

MCP_ROOT = Path("tldw_Server_API/app/core/MCP_unified")


def _imports_for(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def test_new_interface_modules_do_not_import_tldw_server_api() -> None:
    interface_dir = MCP_ROOT / "interfaces"
    assert interface_dir.exists()
    offenders: dict[str, list[str]] = {}
    for path in interface_dir.glob("*.py"):
        imports = [
            name
            for name in _imports_for(path)
            if name == "tldw_Server_API" or name.startswith("tldw_Server_API.")
        ]
        if imports:
            offenders[str(path)] = imports
    assert offenders == {}


def test_mcp_protocol_accepts_runtime_dependencies() -> None:
    from tldw_Server_API.app.core.MCP_unified.adapters.tldw_runtime import (
        build_default_runtime_dependencies,
    )
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol

    deps = build_default_runtime_dependencies()
    protocol = MCPProtocol(dependencies=deps)
    assert protocol.module_registry is deps.module_registry
    assert protocol.rbac_policy is deps.rbac_policy


def test_mcp_server_accepts_runtime_dependencies() -> None:
    from tldw_Server_API.app.core.MCP_unified.adapters.tldw_runtime import (
        build_default_runtime_dependencies,
    )
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    deps = build_default_runtime_dependencies()
    server = MCPServer(dependencies=deps)
    assert server.protocol.module_registry is deps.module_registry


def test_protocol_instances_do_not_share_prepared_call_secrets() -> None:
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol

    first = MCPProtocol()
    second = MCPProtocol()
    assert first is not second
    assert first._prepared_call_secret != second._prepared_call_secret  # noqa: SLF001
```

- [x] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py -v
```

Expected: failures for missing `interfaces`, missing `adapters.tldw_runtime`, and unsupported `dependencies=` constructor arguments.

- [x] **Step 3: Commit only the failing tests if using strict TDD checkpoints**

```bash
git add tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py
git commit -m "test: define mcp extraction boundary contracts"
```

If the team prefers one commit per complete task, leave this unstaged until Task 2 passes.

## Task 2: Add Interface Protocols And Default Runtime Dependency Bundle

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/interfaces/__init__.py`
- Create: `tldw_Server_API/app/core/MCP_unified/interfaces/runtime.py`
- Create: `tldw_Server_API/app/core/MCP_unified/interfaces/policy.py`
- Create: `tldw_Server_API/app/core/MCP_unified/interfaces/storage.py`
- Create: `tldw_Server_API/app/core/MCP_unified/adapters/__init__.py`
- Create: `tldw_Server_API/app/core/MCP_unified/adapters/tldw_runtime.py`
- Create: `tldw_Server_API/app/core/MCP_unified/adapters/tldw_policy.py`

- [x] **Step 1: Create runtime interfaces**

Create `interfaces/runtime.py` with small protocols and dataclasses. Keep this file free of `tldw_Server_API` imports. The contracts should describe only behavior used by MCP Unified in this slice; do not mirror the full host service APIs.

```python
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol


class TelemetryProvider(Protocol):
    def record_event(self, event_name: str, payload: dict[str, Any] | None = None) -> None: ...


class MetricsCollectorLike(Protocol):
    def record_request(self, method: str, success: bool, duration_ms: float) -> None: ...


class RateLimiterLike(Protocol):
    async def check_rate_limit(self, key: str, limit_type: str = "default") -> bool: ...


class ModuleRegistryLike(Protocol):
    async def find_module_for_tool(self, tool_name: str) -> Any | None: ...
    def get_module_id_for_tool(self, tool_name: str) -> str | None: ...


class CircuitBreakerFactory(Protocol):
    def __call__(self, *, name: str, config: Any) -> Any: ...


class DatabasePathResolver(Protocol):
    def resolve_user_db_paths(self, user_id: str | int | None) -> dict[str, str]: ...


class ApiKeyScopeNormalizer(Protocol):
    def normalize(self, raw_scopes: Any) -> set[str]: ...


@dataclass(slots=True)
class MCPRuntimeDependencies:
    module_registry: ModuleRegistryLike
    rbac_policy: Any
    rate_limiter: Any
    metrics_collector: Any
    telemetry_provider: Any
    database_path_resolver: DatabasePathResolver
    api_key_scope_normalizer: ApiKeyScopeNormalizer
    effective_policy_resolver: Any
    approval_evaluator: Any
    path_scope_enforcer: Any
    external_access_evaluator: Any
    redis_client_factory: Callable[..., Awaitable[Any]]
    circuit_breaker_factory: CircuitBreakerFactory
```

- [x] **Step 2: Create policy interfaces**

Create `interfaces/policy.py`.

```python
from __future__ import annotations

from typing import Any, Protocol


class EffectivePolicyResolver(Protocol):
    async def resolve_for_context(self, *, user_id: str | None, metadata: dict[str, Any]) -> dict[str, Any] | None: ...


class ApprovalEvaluator(Protocol):
    async def evaluate_tool_call(
        self,
        *,
        effective_policy: dict[str, Any],
        tool_name: str,
        tool_args: Any,
        context: Any,
        tool_def: dict[str, Any] | None,
        is_write: bool | None,
        within_effective_policy: bool,
        force_approval: bool = False,
        approval_reason: str | None = None,
        scope_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


class PathScopeEnforcer(Protocol):
    async def evaluate_tool_call(
        self,
        *,
        effective_policy: dict[str, Any] | None,
        context: Any,
        tool_name: str,
        tool_args: Any,
        tool_def: dict[str, Any] | None,
    ) -> dict[str, Any]: ...


class ExternalAccessEvaluator(Protocol):
    async def resolve_for_sources(
        self,
        *,
        sources: list[dict[str, Any]],
        effective_policy: dict[str, Any],
    ) -> dict[str, Any]: ...
```

- [x] **Step 3: Create storage interfaces**

Create `interfaces/storage.py`.

```python
from __future__ import annotations

from typing import Any, Protocol


class ProfileStore(Protocol):
    async def get_profile(self, profile_id: str) -> dict[str, Any] | None: ...


class ExternalRegistryStore(Protocol):
    async def list_servers(self) -> list[dict[str, Any]]: ...


class AuditStore(Protocol):
    async def append_event(self, event: dict[str, Any]) -> None: ...
```

- [x] **Step 4: Export interfaces**

Create `interfaces/__init__.py`.

```python
from .policy import ApprovalEvaluator, EffectivePolicyResolver, ExternalAccessEvaluator, PathScopeEnforcer
from .runtime import MCPRuntimeDependencies
from .storage import AuditStore, ExternalRegistryStore, ProfileStore

__all__ = [
    "ApprovalEvaluator",
    "AuditStore",
    "EffectivePolicyResolver",
    "ExternalAccessEvaluator",
    "ExternalRegistryStore",
    "MCPRuntimeDependencies",
    "PathScopeEnforcer",
    "ProfileStore",
]
```

- [x] **Step 5: Create default runtime adapters**

Create `adapters/tldw_runtime.py`. This file may import `tldw_Server_API`; it is the host adapter.

```python
from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.Infrastructure.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from tldw_Server_API.app.core.Infrastructure.redis_factory import create_async_redis_client
from tldw_Server_API.app.core.MCP_unified.auth.authnz_rbac import get_rbac_policy
from tldw_Server_API.app.core.MCP_unified.auth.rate_limiter import get_rate_limiter
from tldw_Server_API.app.core.MCP_unified.interfaces.runtime import MCPRuntimeDependencies
from tldw_Server_API.app.core.MCP_unified.modules.registry import get_module_registry
from tldw_Server_API.app.core.MCP_unified.monitoring.metrics import get_metrics_collector
from tldw_Server_API.app.core.Metrics.telemetry import get_telemetry_manager


class TldwDatabasePathResolver:
    def resolve_user_db_paths(self, user_id: str | int | None) -> dict[str, str]:
        if user_id is None:
            return {}
        try:
            uid_int = int(str(user_id))
        except (TypeError, ValueError):
            return {}
        from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

        paths = DatabasePaths.get_all_user_db_paths(uid_int)
        return {key: str(value) for key, value in paths.items()}


class TldwApiKeyScopeNormalizer:
    def normalize(self, raw_scopes: Any) -> set[str]:
        try:
            from tldw_Server_API.app.core.AuthNZ.api_key_manager import normalize_scope
        except Exception:
            return set()
        try:
            return set(normalize_scope(raw_scopes))
        except Exception:
            return set()


def create_tldw_circuit_breaker(*, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
    return CircuitBreaker(name=name, config=config)


def build_default_runtime_dependencies() -> MCPRuntimeDependencies:
    from tldw_Server_API.app.core.MCP_unified.adapters.tldw_policy import (
        TldwApprovalEvaluator,
        TldwEffectivePolicyResolver,
        TldwExternalAccessEvaluator,
        TldwPathScopeEnforcer,
    )

    return MCPRuntimeDependencies(
        module_registry=get_module_registry(),
        rbac_policy=get_rbac_policy(),
        rate_limiter=get_rate_limiter(),
        metrics_collector=get_metrics_collector(),
        telemetry_provider=get_telemetry_manager(),
        database_path_resolver=TldwDatabasePathResolver(),
        api_key_scope_normalizer=TldwApiKeyScopeNormalizer(),
        effective_policy_resolver=TldwEffectivePolicyResolver(),
        approval_evaluator=TldwApprovalEvaluator(),
        path_scope_enforcer=TldwPathScopeEnforcer(),
        external_access_evaluator=TldwExternalAccessEvaluator(),
        redis_client_factory=create_async_redis_client,
        circuit_breaker_factory=create_tldw_circuit_breaker,
    )
```

- [x] **Step 6: Create default policy adapters**

Create `adapters/tldw_policy.py` with thin wrappers around the existing MCP Hub services. Do not change behavior yet.

```python
from __future__ import annotations

from typing import Any


class TldwEffectivePolicyResolver:
    async def resolve_for_context(self, *, user_id: str | None, metadata: dict[str, Any]) -> dict[str, Any] | None:
        from tldw_Server_API.app.services.mcp_hub_policy_resolver import get_mcp_hub_policy_resolver

        resolver = await get_mcp_hub_policy_resolver()
        return await resolver.resolve_for_context(user_id=user_id, metadata=metadata)


class TldwApprovalEvaluator:
    async def evaluate_tool_call(self, **kwargs: Any) -> dict[str, Any]:
        from tldw_Server_API.app.services.mcp_hub_approval_service import get_mcp_hub_approval_service

        service = await get_mcp_hub_approval_service()
        return await service.evaluate_tool_call(**kwargs)


class TldwPathScopeEnforcer:
    async def evaluate_tool_call(self, **kwargs: Any) -> dict[str, Any]:
        from tldw_Server_API.app.services.mcp_hub_path_enforcement_service import get_mcp_hub_path_enforcement_service

        service = await get_mcp_hub_path_enforcement_service()
        return await service.evaluate_tool_call(**kwargs)


class TldwExternalAccessEvaluator:
    async def resolve_for_sources(
        self,
        *,
        sources: list[dict[str, Any]],
        effective_policy: dict[str, Any],
    ) -> dict[str, Any]:
        from tldw_Server_API.app.services.mcp_hub_external_access_resolver import (
            get_mcp_hub_external_access_resolver,
        )

        resolver = await get_mcp_hub_external_access_resolver()
        return await resolver.resolve_for_sources(
            sources=sources,
            effective_policy=dict(effective_policy or {}),
        )
```

These wrapper names intentionally match the current service APIs used by `protocol.py`: `McpHubPolicyResolver.resolve_for_context`, `McpHubApprovalService.evaluate_tool_call`, `McpHubPathEnforcementService.evaluate_tool_call`, and `McpHubExternalAccessResolver.resolve_for_sources`.

- [x] **Step 7: Export adapters**

Create `adapters/__init__.py`.

```python
from .tldw_runtime import build_default_runtime_dependencies

__all__ = ["build_default_runtime_dependencies"]
```

- [x] **Step 8: Run boundary test**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py::test_new_interface_modules_do_not_import_tldw_server_api -v
```

Expected: PASS.

## Task 3: Wire `MCPProtocol` To Runtime Dependencies

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/protocol.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py`

- [x] **Step 1: Update `MCPProtocol.__init__`**

Change constructor to accept optional dependencies while preserving current default behavior.

```python
from .adapters.tldw_runtime import build_default_runtime_dependencies
from .interfaces.runtime import MCPRuntimeDependencies


class MCPProtocol:
    def __init__(self, dependencies: MCPRuntimeDependencies | None = None):
        self.dependencies = dependencies or build_default_runtime_dependencies()
        self.module_registry = self.dependencies.module_registry
        self.rbac_policy = self.dependencies.rbac_policy
        self.rate_limiter = self.dependencies.rate_limiter
        self.metrics = self.dependencies.metrics_collector
        ...
```

- [x] **Step 2: Move DB path resolution through dependency**

In `RequestContext`, avoid direct `DatabasePaths` import by accepting optional `db_paths` or a resolver call from `MCPProtocol` context construction.

Minimal safe approach:

```python
class RequestContext:
    def __init__(..., db_paths: Optional[dict[str, str]] = None):
        ...
        self.db_paths = dict(db_paths or {})
```

Then update protocol/server call sites that create `RequestContext` to pass `dependencies.database_path_resolver.resolve_user_db_paths(user_id)` where available.

- [x] **Step 3: Move API key scope normalization through dependency**

Replace direct imports of `normalize_scope` in `protocol.py` with `self.dependencies.api_key_scope_normalizer.normalize(raw)`.

- [x] **Step 4: Move MCP Hub policy service lookups through dependency**

Replace direct imports in:

- `_resolve_effective_tool_policy`
- `_evaluate_runtime_approval`
- `_evaluate_path_scope`
- `_evaluate_external_access`

Use:

```python
policy = await self.dependencies.effective_policy_resolver.resolve_for_context(
    user_id=context.user_id,
    metadata=metadata,
)
approval = await self.dependencies.approval_evaluator.evaluate_tool_call(...)
path_scope = await self.dependencies.path_scope_enforcer.evaluate_tool_call(...)
external_access = await self.dependencies.external_access_evaluator.resolve_for_sources(
    sources=[dict(item) for item in sources if isinstance(item, dict)],
    effective_policy=policy,
)
```

Keep the existing fail-closed fallback behavior in each method. If an adapter raises where the old direct service lookup raised, the returned reasons must remain `policy_resolution_failed`, `approval_unavailable`, `path_scope_unavailable`, or `external_access_unavailable` as today.

- [x] **Step 5: Move Redis idempotency factory behind dependency or constructor**

If changing `IdempotencyManager` is low risk, allow it to accept `redis_client_factory`. If that is too broad for this slice, leave direct Redis wiring in place and record it in the module ownership inventory as remaining extraction debt.

- [x] **Step 6: Run extraction contract tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py -v
```

Expected: PASS.

- [x] **Step 7: Run policy compatibility tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add \
  tldw_Server_API/app/core/MCP_unified/interfaces \
  tldw_Server_API/app/core/MCP_unified/adapters \
  tldw_Server_API/app/core/MCP_unified/protocol.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py
git commit -m "feat: add mcp runtime dependency seam"
```

## Task 4: Wire `BaseModule` Circuit Breaker Creation Through An Adapter

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/base.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/interfaces/runtime.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/adapters/tldw_runtime.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py`

- [ ] **Step 1: Add optional circuit breaker factory to `ModuleConfig`**

Add a field that defaults to `None` so current callers are unchanged.

```python
from typing import Callable

@dataclass
class ModuleConfig:
    ...
    circuit_breaker_factory: Any | None = None
```

- [ ] **Step 2: Create a default local fallback factory**

In `modules/base.py`, move the current direct import into a small function.

```python
def _default_circuit_breaker_factory(name: str, config: Any) -> Any:
    from tldw_Server_API.app.core.Infrastructure.circuit_breaker import CircuitBreaker
    return CircuitBreaker(name=name, config=config)
```

This still imports `tldw_Server_API`, but isolates the remaining dependency to one hook for the later package move.

- [ ] **Step 3: Use the configured factory in `BaseModule.__init__`**

```python
factory = config.circuit_breaker_factory or _default_circuit_breaker_factory
self._circuit_breaker = factory(
    name=f"mcp_{config.name}",
    config=_CBCfg(...),
)
```

If `CircuitBreakerConfig` is still imported directly, either wrap it in the factory input or record it as extraction debt. Prefer keeping behavior unchanged in Stage 1.

- [ ] **Step 4: Add a test with a fake factory**

Add to `test_basic_functionality.py`:

```python
def test_module_config_accepts_circuit_breaker_factory():
    created = {}

    class FakeBreaker:
        pass

    def factory(*, name, config):
        created["name"] = name
        created["config"] = config
        return FakeBreaker()

    module = TestModule(ModuleConfig(name="factory_test", circuit_breaker_factory=factory))
    assert created["name"] == "mcp_factory_test"
    assert module._circuit_breaker.__class__ is FakeBreaker  # noqa: SLF001
```

- [ ] **Step 5: Run module tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py::test_module_config_accepts_circuit_breaker_factory -v
```

Expected: PASS.

- [ ] **Step 6: Run focused registry/module tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_registry_iteration_race.py \
  -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add \
  tldw_Server_API/app/core/MCP_unified/modules/base.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py
git commit -m "feat: add mcp module runtime factory seam"
```

## Task 5: Wire `MCPServer` To Runtime Dependencies Without Route Changes

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/server.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/__init__.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_server_batch_and_formatting.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_websocket_smoke.py`

- [ ] **Step 1: Add optional dependencies to `MCPServer.__init__`**

```python
from .adapters.tldw_runtime import build_default_runtime_dependencies
from .interfaces.runtime import MCPRuntimeDependencies


class MCPServer:
    def __init__(self, dependencies: MCPRuntimeDependencies | None = None):
        self.dependencies = dependencies or build_default_runtime_dependencies()
        self.config = get_config()
        self.protocol = MCPProtocol(dependencies=self.dependencies)
        self.module_registry = self.dependencies.module_registry
        self.rbac_policy = self.dependencies.rbac_policy
        self.rate_limiter = self.dependencies.rate_limiter
        ...
```

- [ ] **Step 2: Keep singleton API unchanged**

Ensure `get_mcp_server()` and `reset_mcp_server()` signatures remain unchanged. Add an internal helper only if tests need it.

- [ ] **Step 3: Ensure `MCPServer` creates `RequestContext` with resolved db paths**

When `server.py` constructs `RequestContext`, pass `db_paths=self.dependencies.database_path_resolver.resolve_user_db_paths(user_id)`.

- [ ] **Step 4: Run server seam test**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py::test_mcp_server_accepts_runtime_dependencies -v
```

Expected: PASS.

- [ ] **Step 5: Run focused HTTP/WS tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_server_batch_and_formatting.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_websocket_smoke.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_ws_parse_error_jsonrpc.py \
  -v
```

Expected: PASS or document environment-limited websocket failures separately if local socket binding is blocked.

- [ ] **Step 6: Commit**

```bash
git add \
  tldw_Server_API/app/core/MCP_unified/server.py \
  tldw_Server_API/app/core/MCP_unified/__init__.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py
git commit -m "feat: add mcp server dependency seam"
```

## Task 6: Add Module Ownership Inventory

**Files:**
- Create: `Docs/MCP/mcp_unified_module_ownership_inventory.md`
- Modify: `backlog/tasks/task-480 - Design-MCP-Unified-standalone-library-and-gateway-extraction.md`

- [ ] **Step 1: Create inventory document header**

```markdown
# MCP Unified Module Ownership Inventory

Date: 2026-05-26
Backlog: TASK-480

This inventory classifies current MCP Unified modules before standalone package extraction.

Classification:

- `runtime-neutral`: safe to move into a standalone package with no `tldw_Server_API` dependency.
- `adapter-backed`: reusable only after explicit host adapters replace current `tldw_Server_API` dependencies.
- `tldw-owned`: remains in `tldw_server` and is exposed as a host-provided module.

| Module file | Tool families | Classification | Data stores/dependencies | Capability/risk notes | Migration recommendation | Protecting tests |
| --- | --- | --- | --- | --- | --- | --- |
```

- [ ] **Step 2: Inventory each module implementation**

Add one table row for each file under `tldw_Server_API/app/core/MCP_unified/modules/implementations/`.

Initial recommended classification:

- `external_federation_module.py`: adapter-backed
- `filesystem_module.py`: adapter-backed
- `mcp_discovery_module.py`: adapter-backed
- `run_command_module.py`: runtime-neutral or adapter-backed depending on current dependency scan
- `template_module.py`: likely runtime-neutral if dependency scan confirms no tldw DB usage
- Domain modules such as `media`, `knowledge`, `notes`, `chats`, `characters`, `prompts`, `flashcards`, `quizzes`, `slides`, `kanban`, `persona_visuals`, `codegraph`, `governance`, `sandbox`: likely `tldw-owned` or `adapter-backed`

Use `rg -n "from tldw_Server_API|import tldw_Server_API" <module-file>` to ground each classification.

- [ ] **Step 3: Add extraction debt section**

```markdown
## Extraction Debt

- `protocol.py`: remaining direct imports that require adapter seams.
- `server.py`: remaining direct imports that require adapter seams.
- `modules/base.py`: remaining circuit breaker config coupling if not fully removed in Task 4.
- module rows classified as `adapter-backed`: one future plan per module family before migration.
```

- [ ] **Step 4: Update Backlog task modified files**

Use MCP task edit to include the inventory file in `TASK-480` modified files and implementation notes.

- [ ] **Step 5: Commit**

```bash
git add \
  Docs/MCP/mcp_unified_module_ownership_inventory.md \
  "backlog/tasks/task-480 - Design-MCP-Unified-standalone-library-and-gateway-extraction.md"
git commit -m "docs: inventory mcp module extraction ownership"
```

## Task 7: Add Compatibility And Boundary Verification Pass

**Files:**
- Modify as needed only if tests reveal missing coverage:
  - `tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py`
  - `tldw_Server_API/app/core/MCP_unified/tests/test_protocol_catalog_filter.py`
  - `tldw_Server_API/app/core/MCP_unified/tests/test_scope_and_fallbacks.py`

- [ ] **Step 1: Run all new and directly touched tests**

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_server_batch_and_formatting.py \
  -v
```

Expected: PASS.

- [ ] **Step 2: Run host MCP compatibility tests**

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/MCP_unified/test_mcp_http_auth_paths.py \
  tldw_Server_API/tests/MCP_unified/test_mcp_protocol_path_scope.py \
  tldw_Server_API/tests/MCP_unified/test_mcp_protocol_external_federation.py \
  -v
```

Expected: PASS or document any pre-existing/environment-limited failures with exact failure text.

- [ ] **Step 3: Run Bandit on touched code**

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/MCP_unified/protocol.py \
  tldw_Server_API/app/core/MCP_unified/server.py \
  tldw_Server_API/app/core/MCP_unified/modules/base.py \
  tldw_Server_API/app/core/MCP_unified/interfaces \
  tldw_Server_API/app/core/MCP_unified/adapters \
  -f json -o /tmp/bandit_mcp_unified_stage1_adapter_seams.json
```

Expected: no new findings in touched code. If baseline findings appear, document them separately and fix any new issue introduced by this slice.

- [ ] **Step 4: Record verification in Backlog**

Update `TASK-480` with:

- test commands run
- pass/fail result
- Bandit output path
- any known skips or environment constraints

- [ ] **Step 5: Final commit**

```bash
git add \
  tldw_Server_API/app/core/MCP_unified \
  Docs/MCP/mcp_unified_module_ownership_inventory.md \
  "backlog/tasks/task-480 - Design-MCP-Unified-standalone-library-and-gateway-extraction.md"
git commit -m "test: verify mcp extraction adapter seams"
```

## Final Verification Before PR

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_server_batch_and_formatting.py \
  tldw_Server_API/tests/MCP_unified/test_mcp_http_auth_paths.py \
  tldw_Server_API/tests/MCP_unified/test_mcp_protocol_path_scope.py \
  tldw_Server_API/tests/MCP_unified/test_mcp_protocol_external_federation.py \
  -v
```

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/MCP_unified/protocol.py \
  tldw_Server_API/app/core/MCP_unified/server.py \
  tldw_Server_API/app/core/MCP_unified/modules/base.py \
  tldw_Server_API/app/core/MCP_unified/interfaces \
  tldw_Server_API/app/core/MCP_unified/adapters \
  -f json -o /tmp/bandit_mcp_unified_stage1_adapter_seams.json
```

Expected:

- targeted pytest commands pass
- no new Bandit findings in touched code
- `TASK-480` includes final verification notes
- current routes/imports remain compatible
