# MCP Unified Stage 2 Package Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Create the first runtime-neutral `mcp_unified` package boundary while preserving all existing `tldw_Server_API.app.core.MCP_unified` imports.

**Architecture:** This slice creates a top-level package containing host-neutral interface contracts and profile schema primitives. The existing in-repo MCP Unified interface modules become compatibility shims that re-export those package contracts; no protocol dispatch, server routes, domain modules, or gateway entrypoints move in this PR.

**Tech Stack:** Python 3.10+, Pydantic v2, pytest, existing Backlog.md task tracking, Bandit.

---

## Source Spec

- Spec: `Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md`
- Stage 1 inventory: `Docs/MCP/mcp_unified_module_ownership_inventory.md`
- Backlog task: `TASK-518`

## Scope Boundary

In scope:

- Create `mcp_unified/` as a top-level package included by setuptools discovery.
- Move/copy the neutral runtime, policy, and storage protocols into `mcp_unified.interfaces`.
- Add profile model and resolver/store primitives that encode the profile fields from the spec without enforcing policy yet.
- Convert `tldw_Server_API.app.core.MCP_unified.interfaces.*` into compatibility shims that re-export the new package contracts.
- Add boundary tests proving the new package imports without `tldw_Server_API` and existing host imports still resolve to the same contracts.

Out of scope:

- Moving `protocol.py`, `server.py`, `modules/base.py`, or any domain module into the package.
- Creating a standalone gateway, FastAPI router, CLI, stdio server, SQLite store, or external server manager in the new package.
- Changing `/api/v1/mcp/*` behavior or route contracts.
- Changing MCP Hub, AuthNZ, credential, approval, or path-scope behavior.

## File Structure

Create:

- `mcp_unified/__init__.py`
- `mcp_unified/interfaces/__init__.py`
- `mcp_unified/interfaces/policy.py`
- `mcp_unified/interfaces/runtime.py`
- `mcp_unified/interfaces/storage.py`
- `mcp_unified/profiles/__init__.py`
- `mcp_unified/profiles/models.py`
- `mcp_unified/profiles/resolver.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`

Modify:

- `pyproject.toml`
- `tldw_Server_API/app/core/MCP_unified/interfaces/__init__.py`
- `tldw_Server_API/app/core/MCP_unified/interfaces/policy.py`
- `tldw_Server_API/app/core/MCP_unified/interfaces/runtime.py`
- `tldw_Server_API/app/core/MCP_unified/interfaces/storage.py`
- `backlog/tasks/task-518 - Implement-MCP-Unified-Stage-2-runtime-neutral-package-boundary.md`

## Task 1: Add Failing Package Boundary Tests

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`

- [x] **Step 1: Write boundary tests**

Add tests that fail until the `mcp_unified` package and shims exist:

```python
from __future__ import annotations

import ast
import importlib
from pathlib import Path


PACKAGE_ROOT = Path("mcp_unified")


def _tldw_imports_for(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(
                alias.name
                for alias in node.names
                if alias.name == "tldw_Server_API"
                or alias.name.startswith("tldw_Server_API.")
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "tldw_Server_API" or node.module.startswith("tldw_Server_API."):
                imports.append(node.module)
    return imports


def test_runtime_package_boundary_has_no_tldw_server_imports() -> None:
    assert PACKAGE_ROOT.exists()
    offenders: dict[str, list[str]] = {}
    for path in PACKAGE_ROOT.rglob("*.py"):
        imports = _tldw_imports_for(path)
        if imports:
            offenders[str(path)] = imports
    assert offenders == {}


def test_host_interface_shims_reexport_package_contracts() -> None:
    package_runtime = importlib.import_module("mcp_unified.interfaces.runtime")
    host_runtime = importlib.import_module(
        "tldw_Server_API.app.core.MCP_unified.interfaces.runtime"
    )
    assert host_runtime.MCPRuntimeDependencies is package_runtime.MCPRuntimeDependencies
    assert host_runtime.ModuleRegistry is package_runtime.ModuleRegistry


def test_profile_defaults_are_safe_and_preserve_extension_metadata() -> None:
    from mcp_unified.profiles.models import MCPProfile

    profile = MCPProfile(
        id="architect",
        name="Architect",
        metadata={"agent_metadata": {"system_prompt": "review architecture"}},
    )

    assert profile.enabled is True
    assert profile.policy_document.allowed_tools == []
    assert profile.policy_document.capabilities == []
    assert profile.credential_grants == []
    assert profile.external_server_grants == []
    assert profile.metadata["agent_metadata"]["system_prompt"] == "review architecture"
```

- [x] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -v
```

Expected: FAIL because `mcp_unified` does not exist.

## Task 2: Create Runtime-Neutral Package Contracts

**Files:**
- Create: `mcp_unified/__init__.py`
- Create: `mcp_unified/interfaces/__init__.py`
- Create: `mcp_unified/interfaces/policy.py`
- Create: `mcp_unified/interfaces/runtime.py`
- Create: `mcp_unified/interfaces/storage.py`
- Create: `mcp_unified/profiles/__init__.py`
- Create: `mcp_unified/profiles/models.py`
- Create: `mcp_unified/profiles/resolver.py`

- [x] **Step 1: Create `mcp_unified/__init__.py`**

Expose only stable, neutral package metadata:

```python
"""Standalone MCP Unified runtime package boundary."""

__all__ = ["__version__"]
__version__ = "0.1.0"
```

- [x] **Step 2: Copy neutral interface contracts**

Create `mcp_unified/interfaces/policy.py`, `runtime.py`, and `storage.py` from the current Stage 1 interface contents. Keep these files free of `tldw_Server_API` imports.

- [x] **Step 3: Export interface contracts**

Create `mcp_unified/interfaces/__init__.py` that exports:

```python
from .policy import (
    ApprovalEvaluator,
    EffectivePolicyResolver,
    ExternalAccessEvaluator,
    PathScopeEnforcer,
)
from .runtime import (
    ApiKeyScopeNormalizer,
    CircuitBreakerFactory,
    DatabasePathResolver,
    MCPRuntimeDependencies,
    MetricsCollector,
    ModuleRegistry,
    RateLimiter,
    RbacPolicy,
    RedisClientFactory,
    TelemetryProvider,
)
from .storage import AuditStore, ExternalRegistryStore, ProfileStore
```

- [x] **Step 4: Add profile models**

Create `mcp_unified/profiles/models.py` with Pydantic models for the profile fields in the spec. Use safe defaults:

```python
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ProfilePolicy(BaseModel):
    allowed_tools: list[str] = Field(default_factory=list)
    denied_tools: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    denied_capabilities: list[str] = Field(default_factory=list)
    tool_patterns: list[str] = Field(default_factory=list)
    module_patterns: list[str] = Field(default_factory=list)
    risk_classes: list[str] = Field(default_factory=list)
    resource_constraints: dict[str, Any] = Field(default_factory=dict)


class MCPProfile(BaseModel):
    id: str
    name: str
    description: str = ""
    schema_version: int = 1
    preset_id: str | None = None
    preset_version: str | None = None
    enabled: bool = True
    policy_document: ProfilePolicy = Field(default_factory=ProfilePolicy)
    approval_policy: dict[str, Any] = Field(default_factory=dict)
    path_scopes: list[dict[str, Any]] = Field(default_factory=list)
    external_server_grants: list[dict[str, Any]] = Field(default_factory=list)
    credential_grants: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
```

- [x] **Step 5: Add resolver/store protocols**

Create `mcp_unified/profiles/resolver.py`:

```python
from __future__ import annotations

from typing import Protocol

from .models import MCPProfile


class ProfileResolver(Protocol):
    async def resolve_profile(self, profile_id: str | None, *, user_id: str | None = None) -> MCPProfile | None: ...
```

- [x] **Step 6: Export profile primitives**

Create `mcp_unified/profiles/__init__.py` that exports `MCPProfile`, `ProfilePolicy`, and `ProfileResolver`.

- [x] **Step 7: Run package boundary test**

Run:

```bash
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py::test_runtime_package_boundary_has_no_tldw_server_imports -v
```

Expected: PASS.

## Task 3: Convert Host Interfaces To Compatibility Shims

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/interfaces/__init__.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/interfaces/policy.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/interfaces/runtime.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/interfaces/storage.py`
- Modify: `pyproject.toml`

- [x] **Step 1: Replace host interface definitions with package re-exports**

Each host interface module should import and re-export the matching `mcp_unified.interfaces` contracts. Example for `runtime.py`:

```python
"""Compatibility re-exports for MCP Unified runtime interfaces."""

from mcp_unified.interfaces.runtime import (
    ApiKeyScopeNormalizer,
    CircuitBreakerFactory,
    DatabasePathResolver,
    MCPRuntimeDependencies,
    MetricsCollector,
    ModuleRegistry,
    RateLimiter,
    RbacPolicy,
    RedisClientFactory,
    TelemetryProvider,
)

__all__ = [...]
```

- [x] **Step 2: Update setuptools package discovery**

In `pyproject.toml`, update `[tool.setuptools.packages.find] include` to include the new package:

```toml
include = ["tldw_Server_API", "tldw_Server_API.*", "mcp_unified", "mcp_unified.*"]
```

- [x] **Step 3: Run shim identity test**

Run:

```bash
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py::test_host_interface_shims_reexport_package_contracts -v
```

Expected: PASS.

- [x] **Step 4: Run Stage 1 extraction contracts**

Run:

```bash
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py -v
```

Expected: PASS.

## Task 4: Verify, Record, And Commit The Stage 2 Slice

**Files:**
- Modify: `backlog/tasks/task-518 - Implement-MCP-Unified-Stage-2-runtime-neutral-package-boundary.md`

- [x] **Step 1: Run focused tests**

Run:

```bash
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py \
  -v
```

Expected: PASS.

- [x] **Step 2: Run Bandit on touched package and interface code**

Run:

```bash
python -m bandit -r \
  mcp_unified \
  tldw_Server_API/app/core/MCP_unified/interfaces \
  -f json -o /tmp/bandit_mcp_unified_stage2_package_boundary.json
```

Expected: 0 findings.

- [x] **Step 3: Update Backlog task**

Record:

- implementation plan path
- modified files
- test commands and results
- Bandit result
- any known skips or blockers

- [x] **Step 4: Commit**

Run:

```bash
git add \
  pyproject.toml \
  mcp_unified \
  tldw_Server_API/app/core/MCP_unified/interfaces \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  Docs/superpowers/plans/2026-05-27-mcp-unified-stage2-package-boundary-implementation-plan.md \
  "backlog/tasks/task-518 - Implement-MCP-Unified-Stage-2-runtime-neutral-package-boundary.md"
git commit -m "feat: add mcp unified package boundary"
```

## Final Verification Before PR

Run:

```bash
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py \
  -v
```

Run:

```bash
python -m bandit -r \
  mcp_unified \
  tldw_Server_API/app/core/MCP_unified/interfaces \
  -f json -o /tmp/bandit_mcp_unified_stage2_package_boundary.json
```

Expected:

- Package boundary tests pass.
- Stage 1 extraction contracts still pass.
- Basic MCP functionality still passes.
- Bandit reports 0 findings.
- `TASK-518` contains final verification notes.
