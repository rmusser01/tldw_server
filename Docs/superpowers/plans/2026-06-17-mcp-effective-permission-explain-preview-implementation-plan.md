# MCP Effective Permission Explain And Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only MCP gateway policy explanation and profile tool-preview surface for admin API and CLI users.

**Architecture:** Add one package-owned `policy_explain` service that assembles redacted response models, strict audit events, and degraded-state metadata while delegating decisions to existing policy primitives. Keep model-facing tool discovery filtered, but add a public admin catalog provider for unfiltered preview rows. Expose the service through optional standalone gateway admin routes, remote admin client methods, and CLI commands.

**Tech Stack:** Python 3.10+, Pydantic v2-compatible models, FastAPI optional gateway routes, argparse CLI, stdlib `urllib`, pytest.

---

## Reference Documents

- Spec: `Docs/superpowers/specs/2026-06-16-mcp-effective-permission-explain-preview-design.md`
- Backlog: `TASK-2369`

## File Map

Create:

- `mcp_unified/gateway/policy_explain.py`
  - Request/response models.
  - Public service class.
  - Strict audit append helper.
  - Redaction helpers.
  - Public `GatewayPolicyExplainError`.

- `tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_service.py`
  - Unit coverage for service outcomes, audit, redaction, static/runtime mode, and degraded states.

- `tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_api.py`
  - FastAPI route coverage for auth, permission, audit failure, validation, and success responses.

- `tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_cli.py`
  - CLI and remote admin client coverage.

Modify:

- `mcp_unified/gateway/tool_discovery.py`
  - Add a public unfiltered admin catalog provider.
  - Preserve current model-facing `list_profile_tools`, `search_profile_tools`, and direct-call behavior.

- `mcp_unified/gateway/admin_auth.py`
  - Add `GatewayAdminIdentity`.
  - Add `GatewayAdminPermissionChecker` protocol and `DefaultGatewayAdminPermissionChecker`.
  - Add an identity-producing dependency while preserving existing auth error responses.

- `mcp_unified/gateway/fastapi.py`
  - Add optional policy explain route mounting.
  - Add route request/response schemas or import them from `policy_explain.py`.
  - Use strict audit and stable error envelope.

- `mcp_unified/gateway/remote_admin.py`
  - Add `explain_policy` and `preview_profile_tools` client methods.
  - Allow POST request bodies in `_request_json`.

- `mcp_unified/gateway/cli.py`
  - Add `explain-policy` and `preview-profile-tools`.
  - Support local and remote modes.
  - Support `--args-json`, `--args-json-file`, and `--args-stdin`.
  - Leave `simulate-policy` unchanged.

- `mcp_unified/gateway/__init__.py`
  - Export the new service and admin auth types only if exports are useful to embedders.

- `mcp_unified/README.md`
  - Add a short admin policy explain section.

- `mcp_unified/USER_GUIDE.md`
  - Add user-facing examples for API and CLI usage.

## Implementation Notes

- Tests should live in `tldw_Server_API/tests/MCP_unified` because this checkout does not currently have a standalone `mcp_unified/tests` tree.
- Activate the repo virtual environment before running Python commands:

```bash
source .venv/bin/activate
```

- Prefer small commits after each task if executing inline. If using subagent-driven execution, review and commit each task result before dispatching the next task.
- Do not implement model-facing MCP permission explanation tools in this plan.
- Do not change the behavior of `simulate-policy`.
- Service dependency callables such as profile resolvers and catalog providers should accept either sync or async implementations. Normalize them inside `policy_explain.py` so CLI, tests, and FastAPI wiring do not need duplicate adapters.

---

### Task 1: Add Policy Explain Service Models, Redaction, And Strict Audit

**Files:**
- Create: `mcp_unified/gateway/policy_explain.py`
- Create: `tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_service.py`

- [ ] **Step 1: Write failing service tests**

Create `tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_service.py` with tests for:

```python
from __future__ import annotations

import pytest

from mcp_unified.gateway.policy_explain import (
    GatewayPolicyExplainError,
    GatewayPolicyExplainService,
    PolicyExplainRequest,
    ProfileToolPreviewRequest,
)
from mcp_unified.profiles import MCPProfile, ProfilePolicy
from mcp_unified.storage.models import AuditEvent


class _MemoryAuditStore:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.events: list[AuditEvent] = []

    async def append_event(self, event: AuditEvent) -> None:
        if self.fail:
            raise RuntimeError("audit backend failed")
        self.events.append(event)


def _profile() -> MCPProfile:
    return MCPProfile(
        id="backend-engineer",
        name="Backend Engineer",
        policy_document=ProfilePolicy(
            allowed_tools=["fs.patch"],
            denied_tools=["shell.exec"],
        ),
    )


@pytest.mark.asyncio
async def test_explain_tool_call_redacts_subjects_and_audits() -> None:
    audit = _MemoryAuditStore()
    service = GatewayPolicyExplainService(
        profile_resolver=lambda profile_id: _profile(),
        audit_store=audit,
        actor_id="operator-1",
    )

    response = await service.explain_tool_call(
        PolicyExplainRequest(
            profile_id="backend-engineer",
            tool_name="fs.patch",
            arguments={"path": "/Users/example/project/src/app.py"},
        )
    )

    assert response.ok is True
    assert response.final_outcome == "allow"
    assert response.subjects[0].redaction_state in {"sanitized", "redacted"}
    rendered = response.model_dump_json()
    assert "/Users/example" not in rendered
    assert audit.events[0].event_type == "policy.explain.requested"
    assert "/Users/example" not in str(audit.events[0].payload)


@pytest.mark.asyncio
async def test_explain_tool_call_fails_closed_when_audit_append_fails() -> None:
    service = GatewayPolicyExplainService(
        profile_resolver=lambda profile_id: _profile(),
        audit_store=_MemoryAuditStore(fail=True),
        actor_id="operator-1",
    )

    with pytest.raises(GatewayPolicyExplainError) as exc_info:
        await service.explain_tool_call(
            PolicyExplainRequest(profile_id="backend-engineer", tool_name="fs.patch")
        )

    assert exc_info.value.reason_code == "audit_store_unavailable"


@pytest.mark.asyncio
async def test_preview_requires_catalog_for_complete_denied_counts() -> None:
    audit = _MemoryAuditStore()
    service = GatewayPolicyExplainService(
        profile_resolver=lambda profile_id: _profile(),
        audit_store=audit,
        actor_id="operator-1",
    )

    response = await service.preview_profile_tools(
        ProfileToolPreviewRequest(profile_id="backend-engineer")
    )

    assert response.degraded is True
    assert "catalog_unavailable" in response.degraded_reasons
    assert response.redacted is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_service.py -v
```

Expected: FAIL because `mcp_unified.gateway.policy_explain` does not exist.

- [ ] **Step 3: Implement minimal service module**

Create `mcp_unified/gateway/policy_explain.py` with:

- Pydantic models:
  - `PolicyExplainMode`
  - `PolicyExplainRequest`
  - `ProfileToolPreviewRequest`
  - `PolicyExplainSubject`
  - `PolicyExplainResponse`
  - `ProfileToolPreviewEntry`
  - `ProfileToolPreviewSummary`
  - `ProfileToolPreviewResponse`
  - `PolicyExplainErrorResponse`
- `GatewayPolicyExplainError`.
- `GatewayPolicyExplainService`.
- `_append_audit_event_strict`.
- `_redact_subject_value`.

Implementation rules:

- `mode` defaults to `runtime_effective`.
- `arguments` defaults to `{}` and is never stored directly.
- Enforce a conservative serialized argument size cap, for example `64 * 1024` bytes.
- Call `simulate_tool_call_policy` for single-call explain.
- Call `explain_profile_tool_decision` to derive final tool-level outcome and visibility.
- Convert simulator subject values to `PolicyExplainSubject` with `redaction_state`.
- Build audit payload from redacted response data only.
- Raise `GatewayPolicyExplainError(reason_code="audit_store_unavailable")` if `audit_store` is missing or append fails.

- [ ] **Step 4: Run service tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_service.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcp_unified/gateway/policy_explain.py tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_service.py
git commit -m "feat: add MCP policy explain service"
```

---

### Task 2: Add Public Admin Tool Catalog Provider

**Files:**
- Modify: `mcp_unified/gateway/tool_discovery.py`
- Modify: `tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_service.py`

- [ ] **Step 1: Add failing catalog/preview tests**

Extend `test_standalone_policy_explain_service.py`:

```python
@pytest.mark.asyncio
async def test_preview_includes_denied_installed_tools_from_admin_catalog() -> None:
    audit = _MemoryAuditStore()
    service = GatewayPolicyExplainService(
        profile_resolver=lambda profile_id: _profile(),
        audit_store=audit,
        actor_id="operator-1",
        installed_tool_catalog=lambda: [
            {
                "name": "fs.patch",
                "description": "Patch file",
                "metadata": {"category": "filesystem"},
            },
            {
                "name": "shell.exec",
                "description": "Shell",
                "metadata": {"category": "process"},
            },
        ],
    )

    response = await service.preview_profile_tools(
        ProfileToolPreviewRequest(profile_id="backend-engineer")
    )

    by_name = {entry.tool_name: entry for entry in response.tools}
    assert by_name["fs.patch"].outcome == "allow"
    assert by_name["shell.exec"].outcome == "deny"
    assert by_name["shell.exec"].visibility == "hidden"
    assert response.degraded is False
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_service.py::test_preview_includes_denied_installed_tools_from_admin_catalog -v
```

Expected: FAIL because the service/catalog hook does not yet produce unfiltered rows.

- [ ] **Step 3: Add public catalog helpers**

Modify `mcp_unified/gateway/tool_discovery.py`:

- Add a public dataclass, for example `AdminToolCatalogEntry`.
- Add `list_admin_tool_catalog(profile, backend_tools, *, include_recommendations=True)`.
- Reuse existing normalization/sorting helpers where possible.
- Do not call `_visible_entries` for the admin catalog if doing so would hide denied tools.
- Keep existing exports and model-facing functions stable.

- [ ] **Step 4: Wire service preview to the catalog provider**

Modify `GatewayPolicyExplainService.preview_profile_tools`:

- Prefer an injected `admin_tool_catalog_provider`.
- Fall back to an injected `installed_tool_catalog` plus `list_admin_tool_catalog`.
- Mark preview degraded with `catalog_unavailable` when neither is available.
- Include denied installed tools by default.
- Apply `limit`, `include_denied`, `include_recommendations`, and `category`.

- [ ] **Step 5: Run focused tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_service.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mcp_unified/gateway/tool_discovery.py mcp_unified/gateway/policy_explain.py tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_service.py
git commit -m "feat: add admin tool preview catalog"
```

---

### Task 3: Add Admin Identity And Permission Seam

**Files:**
- Modify: `mcp_unified/gateway/admin_auth.py`
- Create: `tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_api.py`

- [ ] **Step 1: Write failing auth seam tests**

Create `test_standalone_policy_explain_api.py` with tests that directly exercise the auth helpers first:

```python
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mcp_unified.gateway.admin_auth import (
    DefaultGatewayAdminPermissionChecker,
    GatewayAdminAuthConfig,
    GatewayAdminIdentity,
    GatewayAdminPermissionError,
    GatewayAdminPermissionChecker,
)


def test_default_admin_identity_has_policy_explain_permission_when_auth_disabled() -> None:
    identity = GatewayAdminIdentity.local_admin()

    assert identity.actor_id == "local-admin"
    assert "mcp.policy.explain" in identity.permissions


@pytest.mark.asyncio
async def test_permission_checker_denies_missing_policy_explain_permission() -> None:
    checker = DefaultGatewayAdminPermissionChecker()
    identity = GatewayAdminIdentity(actor_id="viewer", permissions=frozenset())

    with pytest.raises(GatewayAdminPermissionError) as exc_info:
        await checker.require_permission(identity, "mcp.policy.explain")

    assert exc_info.value.reason_code == "admin_permission_denied"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_api.py -v
```

Expected: FAIL because the new identity and permission checker do not exist.

- [ ] **Step 3: Implement auth seam**

Modify `mcp_unified/gateway/admin_auth.py`:

- Add `GatewayAdminIdentity`.
- Add `GatewayAdminPermissionError`.
- Add `GatewayAdminPermissionChecker` as a protocol.
- Add `DefaultGatewayAdminPermissionChecker` as the concrete default implementation.
- Add a default local admin identity helper.
- Add an identity-producing dependency helper, for example `gateway_admin_identity_dependency(config)`.
- Preserve `GatewayAdminAuthError`, `gateway_admin_auth_dependencies`, and current auth error response payloads.
- Add a permission error response helper with the stable error envelope.

- [ ] **Step 4: Run auth tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_api.py -v
```

Expected: PASS for the auth seam tests.

- [ ] **Step 5: Commit**

```bash
git add mcp_unified/gateway/admin_auth.py tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_api.py
git commit -m "feat: add gateway admin permission seam"
```

---

### Task 4: Mount Policy Explain Admin API Routes

**Files:**
- Modify: `mcp_unified/gateway/fastapi.py`
- Modify: `mcp_unified/gateway/policy_explain.py`
- Modify: `tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_api.py`

- [ ] **Step 1: Add failing API route tests**

Extend `test_standalone_policy_explain_api.py` with:

```python
from mcp_unified.gateway.fastapi import create_gateway_app
from mcp_unified.gateway.runtime import GatewayRequestContext
from mcp_unified.profiles import MCPProfile, ProfilePolicy
from mcp_unified.storage.models import AuditEvent


class _Runtime:
    name = "test-runtime"
    version = "0.1"

    async def list_tools(self, context: GatewayRequestContext) -> list[dict]:
        return [
            {"name": "fs.patch", "description": "Patch file", "metadata": {"category": "filesystem"}},
            {"name": "shell.exec", "description": "Shell", "metadata": {"category": "process"}},
        ]

    async def call_tool(self, name: str, arguments: dict, context: GatewayRequestContext) -> dict:
        raise AssertionError("policy explain must not execute tools")


class _Audit:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.events: list[AuditEvent] = []

    async def append_event(self, event: AuditEvent) -> None:
        if self.fail:
            raise RuntimeError("audit failed")
        self.events.append(event)


def _profile() -> MCPProfile:
    return MCPProfile(
        id="backend-engineer",
        name="Backend Engineer",
        policy_document=ProfilePolicy(allowed_tools=["fs.patch"], denied_tools=["shell.exec"]),
    )


def test_policy_explain_route_requires_admin_key() -> None:
    app = create_gateway_app(
        _Runtime(),
        enable_policy_explain_management=True,
        policy_explain_profile_resolver=lambda profile_id: _profile(),
        policy_explain_audit_store=_Audit(),
        admin_auth=GatewayAdminAuthConfig(enabled=True, api_key="secret"),
    )
    client = TestClient(app)

    response = client.post("/mcp/policy/explain", json={"profile_id": "backend-engineer", "tool_name": "fs.patch"})

    assert response.status_code == 401
    assert response.json()["reason_code"] == "admin_auth_required"


def test_policy_explain_route_success_audits() -> None:
    audit = _Audit()
    app = create_gateway_app(
        _Runtime(),
        enable_policy_explain_management=True,
        policy_explain_profile_resolver=lambda profile_id: _profile(),
        policy_explain_audit_store=audit,
        admin_auth=GatewayAdminAuthConfig(enabled=True, api_key="secret"),
    )
    client = TestClient(app)

    response = client.post(
        "/mcp/policy/explain",
        headers={"X-MCP-Gateway-Admin-Key": "secret"},
        json={"profile_id": "backend-engineer", "tool_name": "fs.patch", "arguments": {"path": "src/app.py"}},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["final_outcome"] == "allow"
    assert audit.events[0].event_type == "policy.explain.requested"


def test_policy_preview_route_includes_denied_tool() -> None:
    audit = _Audit()
    app = create_gateway_app(
        _Runtime(),
        enable_policy_explain_management=True,
        policy_explain_profile_resolver=lambda profile_id: _profile(),
        policy_explain_audit_store=audit,
    )
    client = TestClient(app)

    response = client.post("/mcp/profiles/backend-engineer/tool-preview", json={})

    assert response.status_code == 200
    by_name = {tool["tool_name"]: tool for tool in response.json()["tools"]}
    assert by_name["shell.exec"]["outcome"] == "deny"
```

- [ ] **Step 2: Run route tests to verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_api.py -v
```

Expected: FAIL because `create_gateway_app` does not accept policy explain args and routes are not mounted.

- [ ] **Step 3: Add route mounting and service resolution**

Modify `mcp_unified/gateway/fastapi.py`:

- Add optional args to `create_gateway_router` and `create_gateway_app`:
  - `enable_policy_explain_management: bool = False`
  - `policy_explain_service: GatewayPolicyExplainService | None = None`
  - `policy_explain_profile_resolver: Callable[[str], MCPProfile | Awaitable[MCPProfile | None]] | None = None`
  - `policy_explain_audit_store: AuditStore | None = None`
  - `policy_explain_permission_checker: GatewayAdminPermissionChecker | None = None`
- Add `_mount_policy_explain_routes`.
- Use POST `/policy/explain`.
- Use POST `/profiles/{profile_id}/tool-preview`.
- Use `runtime.list_tools` with a neutral `GatewayRequestContext` as the installed tool catalog provider.
- Return the stable policy explain error envelope for `GatewayPolicyExplainError` and permission errors.
- Preserve existing `GatewayAdminAuthError` behavior.

- [ ] **Step 4: Run API tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_api.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcp_unified/gateway/fastapi.py mcp_unified/gateway/policy_explain.py tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_api.py
git commit -m "feat: expose MCP policy explain admin API"
```

---

### Task 5: Add Remote Admin Client And CLI Commands

**Files:**
- Modify: `mcp_unified/gateway/remote_admin.py`
- Modify: `mcp_unified/gateway/cli.py`
- Modify: `tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_cli.py`

- [ ] **Step 1: Write failing remote client and CLI tests**

Create `test_standalone_policy_explain_cli.py`:

```python
from __future__ import annotations

import io
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

from mcp_unified.gateway import cli as gateway_cli
from mcp_unified.gateway.remote_admin import RemoteGatewayAdminClient, RemoteGatewayAdminConfig


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_remote_admin_client_posts_policy_explain_body() -> None:
    seen: dict[str, Any] = {}

    def opener(request: Any, timeout: float) -> _Response:
        seen["url"] = request.full_url
        seen["method"] = request.get_method()
        seen["data"] = json.loads(request.data.decode("utf-8"))
        return _Response({"ok": True})

    client = RemoteGatewayAdminClient(
        RemoteGatewayAdminConfig(gateway_url="http://localhost/mcp", admin_key="secret"),
        opener=opener,
    )

    assert client.explain_policy({"profile_id": "backend-engineer", "tool_name": "fs.patch"}) == {"ok": True}
    assert seen["url"].endswith("/policy/explain")
    assert seen["method"] == "POST"
    assert seen["data"]["tool_name"] == "fs.patch"


def test_cli_args_json_file_avoids_command_line_json(tmp_path: Path) -> None:
    args_path = tmp_path / "args.json"
    args_path.write_text('{"path":"src/app.py"}', encoding="utf-8")

    parsed = gateway_cli._policy_explain_arguments_from_args(
        Namespace(args_json=None, args_json_file=str(args_path), args_stdin=False)
    )

    assert parsed == {"path": "src/app.py"}
```

- [ ] **Step 2: Run CLI tests to verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_cli.py -v
```

Expected: FAIL because remote methods and CLI parsing helpers do not exist.

- [ ] **Step 3: Extend remote admin client**

Modify `mcp_unified/gateway/remote_admin.py`:

- Change `_request_json` to accept optional `payload: Mapping[str, Any] | None = None`.
- Serialize provided payload for POST requests.
- Add:
  - `def explain_policy(self, payload: Mapping[str, Any]) -> dict[str, Any]`
  - `def preview_profile_tools(self, profile_id: str, payload: Mapping[str, Any]) -> dict[str, Any]`

- [ ] **Step 4: Add CLI subcommands**

Modify `mcp_unified/gateway/cli.py`:

- Register:
  - `explain-policy`
  - `preview-profile-tools`
- Add shared args:
  - `--profile`
  - `--gateway-url`
  - `--admin-key`
  - `--admin-header-name`
  - `--static-policy-only`
  - `--session-id`
- Add explain args:
  - `--tool`
  - `--args-json`
  - `--args-json-file`
  - `--args-stdin`
  - `--capability`
- Add preview args:
  - `--category`
  - `--include-recommendations`
  - `--exclude-recommendations`
  - `--include-denied`
  - `--exclude-denied`
  - `--limit`
- Local mode should build `GatewayPolicyExplainService` from the same config/storage helpers used by `simulate-policy`.
- Remote mode should call `RemoteGatewayAdminClient`.
- Emit JSON to stdout on success and stderr on errors.

- [ ] **Step 5: Run CLI tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_cli.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mcp_unified/gateway/remote_admin.py mcp_unified/gateway/cli.py tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_cli.py
git commit -m "feat: add MCP policy explain CLI"
```

---

### Task 6: Export Public Types And Update Package Docs

**Files:**
- Modify: `mcp_unified/gateway/__init__.py`
- Modify: `mcp_unified/README.md`
- Modify: `mcp_unified/USER_GUIDE.md`

- [ ] **Step 1: Add minimal export smoke test**

Add to `test_standalone_policy_explain_service.py`:

```python
def test_policy_explain_public_exports() -> None:
    from mcp_unified.gateway import GatewayPolicyExplainService, PolicyExplainRequest

    assert GatewayPolicyExplainService is not None
    assert PolicyExplainRequest is not None
```

- [ ] **Step 2: Run export test to verify failure if exports are missing**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_service.py::test_policy_explain_public_exports -v
```

Expected: FAIL if exports are not present.

- [ ] **Step 3: Add exports**

Modify `mcp_unified/gateway/__init__.py`:

- Export `GatewayPolicyExplainService`.
- Export `PolicyExplainRequest`.
- Export `ProfileToolPreviewRequest`.
- Export `GatewayPolicyExplainError`.
- Keep lazy import style if adding direct imports would create optional dependency problems.

- [ ] **Step 4: Update docs**

Modify `mcp_unified/README.md` and `mcp_unified/USER_GUIDE.md` with:

- What `explain-policy` does.
- What `preview-profile-tools` does.
- Local CLI example.
- Remote CLI example.
- Admin API examples.
- Security notes: audited, redacted, no raw arguments echoed, `--args-json-file` or `--args-stdin` preferred for sensitive arguments.

- [ ] **Step 5: Run export test**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_service.py::test_policy_explain_public_exports -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mcp_unified/gateway/__init__.py mcp_unified/README.md mcp_unified/USER_GUIDE.md tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_service.py
git commit -m "docs: document MCP policy explain surface"
```

---

### Task 7: Full Verification And Security Scan

**Files:**
- No new source files expected.
- May update tests or docs only if verification reveals issues.

- [ ] **Step 1: Run focused standalone policy explain tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_service.py \
  tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_api.py \
  tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_cli.py \
  -v
```

Expected: PASS.

- [ ] **Step 2: Run related gateway regression tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/MCP_unified/test_mcp_http_auth_paths.py \
  tldw_Server_API/tests/MCP_unified/test_mcp_config_sanitization.py \
  -v
```

Expected: PASS or only pre-existing unrelated failures documented with exact failure names.

- [ ] **Step 3: Run package import smoke**

Run:

```bash
source .venv/bin/activate
python - <<'PY'
from mcp_unified.gateway import GatewayPolicyExplainService, PolicyExplainRequest
print(GatewayPolicyExplainService.__name__, PolicyExplainRequest.__name__)
PY
```

Expected output:

```text
GatewayPolicyExplainService PolicyExplainRequest
```

- [ ] **Step 4: Run Bandit on touched package scope**

Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  mcp_unified/gateway/policy_explain.py \
  mcp_unified/gateway/tool_discovery.py \
  mcp_unified/gateway/admin_auth.py \
  mcp_unified/gateway/fastapi.py \
  mcp_unified/gateway/remote_admin.py \
  mcp_unified/gateway/cli.py \
  -f json -o /tmp/bandit_mcp_policy_explain.json
```

Expected: no new findings in touched code. If Bandit reports findings, fix changed-code findings before finishing.

- [ ] **Step 5: Run diff checks**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors. Status should show only intended changes before final commit or be clean after commits.

- [ ] **Step 6: Final commit if verification required fixes**

If verification caused edits:

```bash
git add mcp_unified/gateway tldw_Server_API/tests/MCP_unified mcp_unified/README.md mcp_unified/USER_GUIDE.md
git commit -m "test: verify MCP policy explain surface"
```

If no edits were needed, do not create an empty commit.

---

## Execution Handoff

Recommended execution mode: **Subagent-Driven**. The tasks have clear boundaries and can be reviewed one at a time:

1. Service and strict audit.
2. Admin catalog provider.
3. Auth permission seam.
4. FastAPI routes.
5. Remote client and CLI.
6. Docs and exports.
7. Verification.

Use inline execution only if the current session needs to keep all edits local.
