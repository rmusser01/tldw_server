# MCP Unified Stage 2C Structured Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add structured profile and effective-policy resolution primitives for MCP Unified without wiring profiles into runtime tool execution.

**Architecture:** Keep this slice inside the standalone `mcp_unified.profiles` package and package tests. `StoreBackedProfileResolver` gains a structured result path while preserving the existing `resolve_profile()` convenience behavior. A small effective-policy primitive records reason codes, provenance, deny-over-allow/default-deny behavior, and workspace-binding requirements for write-capable profiles, but does not modify FastAPI routes, `MCPProtocol`, `MCPServer`, SQLite persistence, external server lifecycle, or gateway entrypoints.

**Tech Stack:** Python 3.11, Pydantic v2 models, pytest, Ruff, Mypy, Bandit.

---

## Source Design

- Spec: `Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md`
- Prior Stage 2B plan: `Docs/superpowers/plans/2026-05-27-mcp-unified-profile-registry-resolver-implementation-plan.md`
- Backlog task for plan hardening: `TASK-522`

## Scope

In scope:
- Structured `ProfileResolutionResult` and `EffectivePolicyResult` package models.
- Machine-readable reason codes for resolved, required, missing, disabled, store-unavailable, denied, and workspace-scope-required outcomes.
- Store-backed resolver result method plus compatibility-preserving `resolve_profile()` behavior.
- Effective-policy primitive that enforces deny-over-allow/default-deny and workspace-binding requirements for write-capable profiles.
- Preset/resource-constraint tests that mark write-capable bundled presets as assignment-time workspace-bound templates.

Out of scope:
- FastAPI route changes.
- Runtime `MCPProtocol` or `MCPServer` execution enforcement.
- SQLite persistence or migrations.
- Profile assignment APIs.
- External MCP process spawning, stdio lifecycle, or gateway entrypoints.
- Host MCP Hub policy rewiring.

## Files

- Create: `mcp_unified/profiles/resolution.py`
  - Result models, reason-code literals, effective-policy helper, and write-capability workspace-binding checks.
- Modify: `mcp_unified/profiles/resolver.py`
  - Add `resolve_profile_result()` and keep `resolve_profile()` as a convenience wrapper.
- Modify: `mcp_unified/profiles/presets.py`
  - Mark write-capable bundled presets with `resource_constraints.requires_workspace_binding`.
- Modify: `mcp_unified/profiles/__init__.py`
  - Export structured result primitives.
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_profile_structured_resolution.py`
  - Package tests for result statuses, reason codes, provenance, default behavior, disabled/missing/store-unavailable states, and compatibility wrapper behavior.
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py`
  - Assert write-capable presets advertise workspace-binding requirements.
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`
  - Keep package-boundary and shim-export assertions current if exports change.
- Modify: `backlog/tasks/<new-task>.md`
  - Track the implementation task, verification, and final summary.

## Task 1: RED Tests For Structured Profile Resolution

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_profile_structured_resolution.py`

- [ ] **Step 1: Write failing profile-resolution tests**

Add tests covering:

```python
@pytest.mark.asyncio
async def test_profile_result_reports_required_when_no_explicit_or_default_profile() -> None:
    resolver = StoreBackedProfileResolver(InMemoryProfileStore())

    result = await resolver.resolve_profile_result(None)

    assert result.status == "profile_required"
    assert result.reason_code == "profile_required"
    assert result.profile is None


@pytest.mark.asyncio
async def test_profile_result_reports_disabled_profile_with_provenance() -> None:
    store = InMemoryProfileStore([
        MCPProfile(id="disabled", name="Disabled", enabled=False),
    ])
    resolver = StoreBackedProfileResolver(store)

    result = await resolver.resolve_profile_result("disabled")

    assert result.status == "profile_disabled"
    assert result.reason_code == "profile_disabled"
    assert result.provenance["profile_id"] == "disabled"


@pytest.mark.asyncio
async def test_resolve_profile_keeps_legacy_none_wrapper_behavior() -> None:
    resolver = StoreBackedProfileResolver(InMemoryProfileStore())

    assert await resolver.resolve_profile(None) is None
```

- [ ] **Step 2: Run RED test**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_profile_structured_resolution.py -v
```

Expected: FAIL because `resolve_profile_result` and result models do not exist.

## Task 2: Add Resolution Result Models

**Files:**
- Create: `mcp_unified/profiles/resolution.py`
- Modify: `mcp_unified/profiles/__init__.py`

- [ ] **Step 1: Implement result models**

Add Pydantic models:

```python
ProfileResolutionStatus = Literal[
    "resolved",
    "profile_required",
    "profile_not_found",
    "profile_disabled",
    "store_unavailable",
]

EffectivePolicyStatus = Literal[
    "resolved",
    "denied",
    "approval_required",
    "degraded",
]

class ProfileResolutionResult(BaseModel):
    status: ProfileResolutionStatus
    profile: MCPProfile | None = None
    reason_code: str
    provenance: dict[str, Any] = Field(default_factory=dict)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
```

Also add `EffectivePolicy` and `EffectivePolicyResult` with the same provenance/warning shape.

- [ ] **Step 2: Export result primitives**

Export from `mcp_unified.profiles`.

- [ ] **Step 3: Run focused import tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -v
```

Expected: PASS after exports are stable.

## Task 3: Add Store-Backed Structured Resolution

**Files:**
- Modify: `mcp_unified/profiles/resolver.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_profile_structured_resolution.py`

- [ ] **Step 1: Implement `resolve_profile_result()`**

Add structured behavior:
- no explicit/default id -> `profile_required`
- missing profile -> `profile_not_found`
- disabled profile -> `profile_disabled`
- store unavailable -> `store_unavailable`
- enabled profile -> `resolved`

Each result must include provenance with at least `requested_profile_id`, `resolved_profile_id`, `used_default_profile`, and `resolver`.

- [ ] **Step 2: Preserve wrapper behavior**

Keep `resolve_profile()` returning `MCPProfile | None` for existing primitive callers by delegating to `resolve_profile_result()` and returning the copied profile only when status is `resolved`.

- [ ] **Step 3: Run focused tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_profile_structured_resolution.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_registry_resolver.py -v
```

Expected: PASS.

## Task 4: Add Effective Policy Workspace-Binding Primitive

**Files:**
- Modify: `mcp_unified/profiles/resolution.py`
- Modify: `mcp_unified/profiles/presets.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_profile_structured_resolution.py`

- [ ] **Step 1: Write failing workspace-binding tests**

Add tests proving:
- write-capable profiles without `path_scopes`, host binding, or assignment binding return `workspace_scope_required`
- read-only profiles can resolve effective policy without workspace binding
- deny entries override allow entries
- no allowed tools/capabilities defaults to deny for execution

- [ ] **Step 2: Mark write-capable presets**

Set `policy_document.resource_constraints["requires_workspace_binding"] = True` for bundled presets with mutating/write-scoped capabilities.

- [ ] **Step 3: Implement effective-policy helper**

Add a package-local helper, for example:

```python
def build_effective_policy_result(
    profile: MCPProfile,
    *,
    host_caps: dict[str, Any] | None = None,
    assignment_binding: dict[str, Any] | None = None,
) -> EffectivePolicyResult:
    ...
```

Do not call this from runtime execution in this slice.

- [ ] **Step 4: Run focused tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_profile_structured_resolution.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py -v
```

Expected: PASS.

## Task 5: Regression And Quality Gates

**Files:**
- Modify: implementation Backlog task file

- [ ] **Step 1: Run focused regression tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_structured_resolution.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_registry_resolver.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  -v
```

Expected: PASS.

- [ ] **Step 2: Run static and security checks**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check mcp_unified tldw_Server_API/app/core/MCP_unified/tests/test_profile_structured_resolution.py
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m mypy mcp_unified/profiles mcp_unified/interfaces/storage.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_structured_resolution.py
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r mcp_unified/profiles mcp_unified/interfaces/storage.py -f json -o /tmp/bandit_mcp_unified_stage2c_resolution.json
jq '.metrics._totals, (.results | length)' /tmp/bandit_mcp_unified_stage2c_resolution.json
git diff --check
```

Expected: Ruff passes, Mypy passes, runtime Bandit reports 0 findings, and diff whitespace is clean.

- [ ] **Step 3: Update Backlog task and commit**

Record RED/GREEN evidence, verification, known skips, and final summary, then commit:

```bash
git add \
  mcp_unified/profiles/resolution.py \
  mcp_unified/profiles/resolver.py \
  mcp_unified/profiles/presets.py \
  mcp_unified/profiles/__init__.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_structured_resolution.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_registry_resolver.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  "backlog/tasks/<new-task>.md"
git commit -m "feat: add mcp profile structured resolution primitives"
```

Expected: commit succeeds with no runtime execution wiring.
