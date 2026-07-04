# MCP WebSocket Scoped JWT Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remediate AUDIT-2026-06-27-MCP-WS-001 by making ACP and sandbox bearer-token WebSocket handshakes enforce scoped AuthNZ JWT restrictions consistently with HTTP endpoints.

**Architecture:** Add one shared AuthNZ helper that projects a WebSocket handshake into the existing token-scope guard inputs: bearer credentials, logical endpoint id, synthetic method, path, quota action, and request state. ACP stream and ACP SSH use write-capable enforcement; sandbox run stream uses read enforcement for the current stream endpoint while preserving ownership and signed-url checks. Focused tests first prove scoped JWTs with insufficient claims are rejected before the endpoint accepts the socket.

**Tech Stack:** FastAPI and Starlette WebSocket handlers, existing AuthNZ `require_token_scope` logic, pytest, FastAPI `TestClient`, Bandit.

---

## Stage 1: Tracking And Plan Artifacts
**Goal:** Start TASK-12138 and commit non-production tracking artifacts before production edits.
**Success Criteria:** TASK-12138 is `In Progress`; this plan exists under `Docs/superpowers/plans/`; a first commit contains only backlog and plan changes.
**Tests:** `git show --stat --oneline HEAD` after commit shows only the task file and this plan file.
**Status:** Complete

**Files:**
- Modify: `backlog/tasks/task-12138 - Remediate-ACP-and-sandbox-WebSocket-scoped-JWT-audit-finding.md`
- Create: `Docs/superpowers/plans/2026-07-02-mcp-ws-scoped-jwt-remediation-plan.md`

- [ ] **Step 1: Update backlog task**

Use Backlog MCP `task_edit` to set `TASK-12138` to `In Progress` and append a note that the audit artifact was reviewed.

- [ ] **Step 2: Add this implementation plan**

Create this file with five stages covering tracking, red tests, implementation, verification, and task finalization.

- [ ] **Step 3: Commit tracking artifacts**

Run:

```bash
git add "backlog/tasks/task-12138 - Remediate-ACP-and-sandbox-WebSocket-scoped-JWT-audit-finding.md" Docs/superpowers/plans/2026-07-02-mcp-ws-scoped-jwt-remediation-plan.md
git commit -m "docs: plan mcp websocket scope remediation"
```

Expected: commit succeeds and no production Python files are staged.

## Stage 2: ACP Scoped-JWT Red Tests
**Goal:** Prove ACP stream and SSH WebSocket handshakes currently accept bearer JWTs that should be blocked by scoped claims.
**Success Criteria:** New focused tests fail before implementation because `_authenticate_ws` returns a user id instead of rejecting insufficient scoped JWTs.
**Tests:** Run the two new ACP tests and observe failure before production code changes.
**Status:** Complete

**Files:**
- Modify: `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py`

- [ ] **Step 1: Add scoped JWT test helpers**

Add helpers near the existing WebSocket auth fixtures:

```python
class _ScopedJWTManager:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def verify_token(self, token: str):
        self.calls.append(token)
        return types.SimpleNamespace(user_id=1)
```

The endpoint-level tests will monkeypatch the shared WebSocket guard to raise `HTTPException(403, "Forbidden: endpoint not permitted for token")` only after implementation wires it in.

- [ ] **Step 2: Add ACP stream rejection test**

Add:

```python
def test_websocket_stream_rejects_scoped_jwt_without_acp_endpoint_permission(
    self, client_user_only, mock_get_runner_client, monkeypatch
):
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints
    from fastapi import HTTPException

    guard_calls: list[dict[str, object]] = []

    async def _deny_scope(**kwargs):
        guard_calls.append(kwargs)
        raise HTTPException(status_code=403, detail="Forbidden: endpoint not permitted for token")

    monkeypatch.setattr(acp_endpoints, "get_jwt_manager", lambda: _ScopedJWTManager())
    monkeypatch.setattr(acp_endpoints, "enforce_websocket_token_scope", _deny_scope, raising=False)

    with pytest.raises(Exception):
        with client_user_only.websocket_connect(
            "/api/v1/acp/sessions/test-session/stream?token=scoped.jwt.token"
        ):
            pass

    assert guard_calls
    assert guard_calls[0]["required_scope"] == "write"
    assert guard_calls[0]["endpoint_id"] == "acp.sessions.stream"
```

- [ ] **Step 3: Add ACP SSH rejection test**

Add:

```python
def test_websocket_ssh_rejects_scoped_jwt_without_acp_endpoint_permission(
    self, client_user_only, monkeypatch
):
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints
    from fastapi import HTTPException

    guard_calls: list[dict[str, object]] = []

    async def _deny_scope(**kwargs):
        guard_calls.append(kwargs)
        raise HTTPException(status_code=403, detail="Forbidden: endpoint not permitted for token")

    monkeypatch.setattr(acp_endpoints, "get_jwt_manager", lambda: _ScopedJWTManager())
    monkeypatch.setattr(acp_endpoints, "enforce_websocket_token_scope", _deny_scope, raising=False)

    with pytest.raises(Exception):
        with client_user_only.websocket_connect(
            "/api/v1/acp/sessions/test-session/ssh?token=scoped.jwt.token"
        ):
            pass

    assert guard_calls
    assert guard_calls[0]["required_scope"] == "write"
    assert guard_calls[0]["endpoint_id"] == "acp.sessions.ssh"
```

- [ ] **Step 4: Verify ACP tests are red**

Run:

```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py::TestACPWebSocketConnection::test_websocket_stream_rejects_scoped_jwt_without_acp_endpoint_permission \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py::TestACPWebSocketConnection::test_websocket_ssh_rejects_scoped_jwt_without_acp_endpoint_permission
```

Expected before implementation: both tests fail because the guard is not called.

## Stage 3: Sandbox Scoped-JWT Red Test
**Goal:** Prove sandbox run stream WebSocket handshake currently skips scoped JWT restrictions.
**Success Criteria:** New sandbox test fails before implementation because `_resolve_sandbox_ws_user_id` returns a user id without calling the shared scoped-token guard.
**Tests:** Run the new sandbox test and observe failure before production code changes.
**Status:** Complete

**Files:**
- Modify: `tldw_Server_API/tests/sandbox/test_ws_stdin_caps.py`

- [ ] **Step 1: Add sandbox rejection test**

Add:

```python
def test_ws_stream_rejects_scoped_jwt_without_sandbox_endpoint_permission(monkeypatch) -> None:
    from fastapi import HTTPException
    from tldw_Server_API.app.api.v1.endpoints import sandbox as sb

    guard_calls: list[dict[str, object]] = []

    async def _deny_scope(**kwargs):
        guard_calls.append(kwargs)
        raise HTTPException(status_code=403, detail="Forbidden: endpoint not permitted for token")

    class _JWTService:
        async def verify_token_async(self, token: str, token_type: str = "access"):
            assert token == "scoped.jwt.token"
            assert token_type == "access"
            return {"user_id": 1, "scope": "read", "allowed_endpoints": ["chat.completions"]}

    class _SessionManager:
        async def is_token_blacklisted(self, token: str, jti=None) -> bool:
            return False

    async def _session_manager():
        return _SessionManager()

    monkeypatch.setattr(sb, "enforce_websocket_token_scope", _deny_scope, raising=False)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.jwt_service.get_jwt_service",
        lambda: _JWTService(),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.session_manager.get_session_manager",
        _session_manager,
    )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(sb._resolve_sandbox_ws_user_id(_FakeSandboxWebSocket(), token="scoped.jwt.token", api_key=None))

    assert exc.value.status_code == 403
    assert guard_calls
    assert guard_calls[0]["required_scope"] == "read"
    assert guard_calls[0]["endpoint_id"] == "sandbox.runs.stream"
```

Also add `_FakeSandboxWebSocket` with `headers`, `scope`, `state`, `query_params`, and `client` attributes if an equivalent stub is not already in this file.

- [ ] **Step 2: Verify sandbox test is red**

Run:

```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/sandbox/test_ws_stdin_caps.py::test_ws_stream_rejects_scoped_jwt_without_sandbox_endpoint_permission
```

Expected before implementation: the test fails because the guard is not called.

## Stage 4: Shared WebSocket Scope Enforcement
**Goal:** Reuse the existing AuthNZ scoped-token enforcement for bearer-token WebSocket handshakes.
**Success Criteria:** ACP and sandbox JWT handshakes call the shared helper before returning a user id; scoped claims enforce `scope`, `allowed_endpoints`, `allowed_methods`, `allowed_paths`, quotas, and schedule claims through existing guard code.
**Tests:** Red tests from Stages 2 and 3 pass; existing read-only API-key WebSocket tests still pass.
**Status:** Complete

**Files:**
- Modify: `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/sandbox.py`

- [x] **Step 1: Add shared helper in AuthNZ dependencies**

Add a small helper in `auth_deps.py` after `require_token_scope` or near the scoped-token section:

```python
async def enforce_websocket_token_scope(
    websocket: WebSocket,
    *,
    token: str,
    required_scope: str,
    endpoint_id: str,
    method: str = "GET",
    count_as: Optional[str] = None,
    require_schedule_match: bool = True,
) -> None:
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    scope = dict(getattr(websocket, "scope", {}) or {})
    scope.setdefault("path", getattr(getattr(websocket, "url", None), "path", "") or "")
    request_like = SimpleNamespace(
        app=getattr(websocket, "app", None),
        client=getattr(websocket, "client", None),
        headers=getattr(websocket, "headers", {}),
        method=str(method).upper(),
        path_params=scope.get("path_params", {}),
        query_params=getattr(websocket, "query_params", {}),
        scope=scope,
        state=getattr(websocket, "state", SimpleNamespace()),
        url=getattr(websocket, "url", None),
    )
    checker = require_token_scope(
        required_scope,
        require_if_present=True,
        require_schedule_match=require_schedule_match,
        allow_admin_bypass=False,
        endpoint_id=endpoint_id,
        count_as=count_as,
    )
    await checker(request_like, credentials=credentials, jwt_service=get_jwt_service(), db_pool=await get_db_pool())
```

Adjust imports for `SimpleNamespace`, `WebSocket`, and `HTTPAuthorizationCredentials` if not already present. Keep the helper narrow and avoid changing normal HTTP dependency behavior.

- [x] **Step 2: Wire ACP JWT branch**

Import the helper into `agent_client_protocol.py` and extend `_authenticate_ws` to accept `endpoint_id` and `count_as`. In the token branch, call:

```python
if endpoint_id:
    await enforce_websocket_token_scope(
        websocket=websocket,
        token=token,
        required_scope=required_scope,
        endpoint_id=endpoint_id,
        method="POST" if required_scope == "write" else "GET",
        count_as=count_as,
    )
```

Pass `endpoint_id="acp.sessions.stream", count_as="call"` from `acp_session_stream` and `endpoint_id="acp.sessions.ssh", count_as="call"` from `acp_session_ssh`.

- [x] **Step 3: Wire sandbox JWT branch**

Import the helper into `sandbox.py` and call it in `_resolve_sandbox_ws_user_id` after the token is decoded and blacklist-checked but before returning the subject:

```python
await enforce_websocket_token_scope(
    websocket=websocket,
    token=token,
    required_scope="read",
    endpoint_id="sandbox.runs.stream",
    method="GET",
    count_as="call",
)
```

Let `HTTPException` propagate so `stream_run_logs` closes the socket with the existing authentication-failure path.

- [x] **Step 4: Verify focused tests are green**

Run:

```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py::TestACPWebSocketConnection::test_websocket_stream_rejects_scoped_jwt_without_acp_endpoint_permission \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py::TestACPWebSocketConnection::test_websocket_ssh_rejects_scoped_jwt_without_acp_endpoint_permission \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py::TestACPWebSocketConnection::test_websocket_stream_rejects_read_only_api_key_in_multi_user_mode \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py::TestACPWebSocketConnection::test_websocket_ssh_rejects_read_only_api_key_in_multi_user_mode \
  tldw_Server_API/tests/sandbox/test_ws_stdin_caps.py::test_ws_stream_rejects_scoped_jwt_without_sandbox_endpoint_permission
```

Expected after implementation: selected tests pass.

## Stage 5: Verification, Backlog Finalization, And Commit
**Goal:** Verify the scoped remediation, record results in TASK-12138, and commit implementation changes without pushing.
**Success Criteria:** Focused tests pass; Bandit touched-scope scan has no new production high/medium findings; `git diff --check` passes; task contains touched files, verification, residual risks, and final summary; implementation commit exists.
**Tests:** Focused pytest, Bandit on touched production Python paths, and whitespace check.
**Status:** Complete

**Files:**
- Modify: `backlog/tasks/task-12138 - Remediate-ACP-and-sandbox-WebSocket-scoped-JWT-audit-finding.md`
- Verify: `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`
- Verify: `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py`
- Verify: `tldw_Server_API/app/api/v1/endpoints/sandbox.py`

- [x] **Step 1: Run focused pytest**

Run:

```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py::TestACPWebSocketConnection::test_websocket_stream_rejects_scoped_jwt_without_acp_endpoint_permission \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py::TestACPWebSocketConnection::test_websocket_ssh_rejects_scoped_jwt_without_acp_endpoint_permission \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py::TestACPWebSocketConnection::test_websocket_stream_rejects_read_only_api_key_in_multi_user_mode \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py::TestACPWebSocketConnection::test_websocket_ssh_rejects_read_only_api_key_in_multi_user_mode \
  tldw_Server_API/tests/sandbox/test_ws_stdin_caps.py::test_ws_stream_rejects_scoped_jwt_without_sandbox_endpoint_permission
```

- [x] **Step 2: Run Bandit on touched production files**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/api/v1/API_Deps/auth_deps.py \
  tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py \
  tldw_Server_API/app/api/v1/endpoints/sandbox.py \
  -f json -o /tmp/bandit_task_12138_mcp_ws_scope.json
```

- [x] **Step 3: Run whitespace check**

Run:

```bash
git diff --check
```

- [x] **Step 4: Update TASK-12138**

Use Backlog MCP `task_edit` to add touched files, verification results, residual risks, and final summary. Check acceptance criteria and definition-of-done items that are satisfied.

- [x] **Step 5: Commit implementation**

Run:

```bash
git add \
  "backlog/tasks/task-12138 - Remediate-ACP-and-sandbox-WebSocket-scoped-JWT-audit-finding.md" \
  tldw_Server_API/app/api/v1/API_Deps/auth_deps.py \
  tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py \
  tldw_Server_API/app/api/v1/endpoints/sandbox.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py \
  tldw_Server_API/tests/sandbox/test_ws_stdin_caps.py
git commit -m "fix: enforce scoped jwt claims on mcp websockets"
```

Expected: commit succeeds. Do not push.
