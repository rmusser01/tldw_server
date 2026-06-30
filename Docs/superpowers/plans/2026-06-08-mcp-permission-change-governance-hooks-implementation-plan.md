# MCP Permission Change Governance Hooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add a minimal governance seam for MCP profile permission mutations so profile and path-grant changes can be allowed, denied, or routed to future approval before persistence.

**Architecture:** Keep the executable profile store unchanged and add a small gateway-level governance contract used by `GatewayProfileManager` before permission-changing mutations. The default governor allows existing behavior; injected governors can return `deny` or `ask`, which blocks the mutation and emits redacted audit metadata.

**Tech Stack:** Python, Pydantic profile models, async gateway manager methods, pytest.

---

### Task 1: Governance Contract And Manager Injection

**Files:**
- Create: `mcp_unified/gateway/profile_governance.py`
- Modify: `mcp_unified/gateway/profiles.py`
- Modify: `mcp_unified/gateway/__init__.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py`

- [x] **Step 1: Write the failing contract/injection tests**

Add test doubles in `test_gateway_profile_management.py`:

```python
class RecordingPermissionGovernor:
    def __init__(self, outcome: str = "allow") -> None:
        self.outcome = outcome
        self.requests = []

    async def evaluate_permission_change(self, request):
        self.requests.append(request)
        return PermissionChangeDecision(outcome=self.outcome, reason_code=f"{self.outcome}_for_test")
```

Add tests that instantiate `_manager(..., permission_governor=governor)` and verify `create_profile()` calls the governor with action `profile.create`, profile id, changed fields, and redacted risk metadata.

- [x] **Step 2: Run the focused test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py::test_create_profile_calls_permission_governor_with_redacted_summary -q`

Expected: FAIL because `permission_governor` and governance models do not exist yet.

- [x] **Step 3: Add minimal governance models**

Create `profile_governance.py` with:

```python
PermissionChangeOutcome = Literal["deny", "ask", "allow"]

@dataclass(frozen=True, slots=True)
class PermissionChangeRequest:
    action: str
    profile_id: str | None
    target_type: str
    target_id: str
    changed_fields: tuple[str, ...] = ()
    policy_fields: tuple[str, ...] = ()
    risk_flags: tuple[str, ...] = ()

@dataclass(frozen=True, slots=True)
class PermissionChangeDecision:
    outcome: PermissionChangeOutcome
    reason_code: str = "allowed"
    message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

Add `AllowPermissionChangeGovernor` and `PermissionChangeGovernor` protocol with async `evaluate_permission_change()`.

- [x] **Step 4: Inject the default governor**

Add `permission_governor` to `GatewayProfileManager.__init__`, defaulting to `AllowPermissionChangeGovernor()`, and export the new types from `mcp_unified/gateway/__init__.py`.

- [x] **Step 5: Run the focused test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py::test_create_profile_calls_permission_governor_with_redacted_summary -q`

Expected: PASS.

### Task 2: Enforcement, Audit Metadata, And Path-Grant Patch Support

**Files:**
- Modify: `mcp_unified/gateway/profiles.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py`

- [x] **Step 1: Write failing enforcement tests**

Add tests for:
- `deny` decision blocks `patch_profile()` before persistence with reason `permission_change_denied`.
- `ask` decision blocks with reason `permission_change_requires_approval`.
- Denial audit payloads include action, profile id, changed fields, policy fields, risk flags, and decision reason, but not raw `policy_document`, path prefixes, or tool patterns.
- `policy_document.path_grants` is accepted by semantic patch validation and included in governance `policy_fields`.

- [x] **Step 2: Run the new focused tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py -k "permission_governor or path_grants" -q`

Expected: FAIL because enforcement and `path_grants` patch support are missing.

- [x] **Step 3: Add permission-change request summarization**

In `GatewayProfileManager`, add helpers that produce redacted request summaries:
- `profile.create`
- `profile.duplicate_from_preset`
- `profile.patch`
- `profile.delete`
- `profile.default_change`

Risk flags should be conservative and content-free, for example:
- `profile_enabled`
- `default_profile_change`
- `policy_allowed_tools_changed`
- `policy_tool_patterns_changed`
- `path_grants_changed`
- `path_grants_write_or_edit`
- `wildcard_tool_policy`

- [x] **Step 4: Enforce governance decisions before persistence**

Before mutating stores, call the governor. If the outcome is:
- `allow`: proceed and add redacted governance metadata to success audit payloads.
- `ask`: audit a blocked permission change and raise `GatewayProfileManagementError(reason_code="permission_change_requires_approval")`.
- `deny`: audit a blocked permission change and raise `GatewayProfileManagementError(reason_code="permission_change_denied")`.

- [x] **Step 5: Allow path-grant policy patch fields**

Extend `_POLICY_PATCH_FIELDS` to include `path_grants` plus authored path-grant keys already recognized by `mcp_unified.profiles.path_grants`.

- [x] **Step 6: Run focused tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py -k "permission_governor or path_grants" -q`

Expected: PASS.

### Task 3: HTTP Mapping, Backlog Notes, And Verification

**Files:**
- Modify: `mcp_unified/gateway/fastapi.py`
- Modify: `backlog/tasks/task-2304 - Add-MCP-permission-change-governance-hooks.md`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py` if status mapping coverage already exists nearby.

- [x] **Step 1: Add failing HTTP status expectations if feasible**

If existing FastAPI route tests have a compact profile-management error mapping fixture, add coverage that `permission_change_denied` maps to 403 and `permission_change_requires_approval` maps to 409.

- [x] **Step 2: Add status-code mapping**

In `mcp_unified/gateway/fastapi.py`, add:
- `permission_change_denied: 403`
- `permission_change_requires_approval: 409`

- [x] **Step 3: Update Backlog task**

Record acceptance criteria, plan path, touched files, and verification commands in `TASK-2304`.

- [x] **Step 4: Run focused verification**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py -q`

Expected: PASS.

- [x] **Step 5: Run security verification on touched Python scope**

Run: `source .venv/bin/activate && python -m bandit -r mcp_unified/gateway/profiles.py mcp_unified/gateway/profile_governance.py mcp_unified/gateway/fastapi.py -f json -o /tmp/bandit_mcp_permission_governance.json`

Expected: exit 0 or only pre-existing/non-touched findings documented.

- [x] **Step 6: Commit**

Run:

```bash
git add Docs/superpowers/plans/2026-06-08-mcp-permission-change-governance-hooks-implementation-plan.md \
  "backlog/tasks/task-2304 - Add-MCP-permission-change-governance-hooks.md" \
  mcp_unified/gateway/profile_governance.py \
  mcp_unified/gateway/profiles.py \
  mcp_unified/gateway/__init__.py \
  mcp_unified/gateway/fastapi.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py
git commit -m "feat: add MCP permission governance hooks"
```
