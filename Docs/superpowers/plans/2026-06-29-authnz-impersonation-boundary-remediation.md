# AuthNZ Impersonation Boundary Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the AuthNZ impersonation audit findings by issuing true short-lived impersonation tokens, preserving actor-plus-subject attribution in request context, writing durable issuance audit events, and removing backend-specific raw SQL from the impersonation endpoint.

**Architecture:** Keep the public endpoint small. Move impersonation lookup and issuance helpers behind existing AuthNZ repository and audit-service boundaries, extend the JWT service with an explicit expiry override, and extend `AuthPrincipal` with optional impersonation metadata so downstream audit hooks can distinguish actor and subject without changing normal user semantics.

**Tech Stack:** FastAPI, Pydantic, PyJWT, existing AuthNZ repositories, unified audit service, pytest, pytest-asyncio, Bandit.

---

## Current Task

Backlog task: `TASK-12073 - Plan and remediate AuthNZ impersonation audit findings`

Audit findings covered:

- `AUDIT-2026-06-27-AUTH-001`: response advertises a 15 minute TTL but token uses normal access-token lifetime.
- `AUDIT-2026-06-27-AUTH-002`: impersonation actor metadata is not preserved for durable audit attribution.
- `AUDIT-2026-06-27-AUTH-003`: endpoint uses raw `pool.acquire()` with SQLite placeholders on a PostgreSQL-capable path.

Wave 0 reconfirmation on current `origin/dev` found all three findings still open.

## File Structure

- Modify: `tldw_Server_API/app/core/AuthNZ/jwt_service.py`
  - Add an optional access-token expiry override and a named impersonation-token helper.
- Modify: `tldw_Server_API/app/core/AuthNZ/principal_model.py`
  - Add optional `impersonated_by_user_id` and `impersonation` fields to `AuthPrincipal`.
- Modify: `tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py`
  - Read impersonation claims from decoded JWTs and place them on `AuthPrincipal` and `request.state`.
- Modify: `tldw_Server_API/app/services/admin_audit_service.py`
  - Add a mandatory audit helper for impersonation token issuance using `flush(raise_on_failure=True)`.
- Modify: `tldw_Server_API/app/api/v1/endpoints/admin/admin_impersonation.py`
  - Replace raw SQL with `AuthnzUsersRepo` and `AuthnzRbacRepo`.
  - Issue impersonation tokens through the new helper.
  - Persist mandatory audit before returning success.
- Modify: `tldw_Server_API/tests/AuthNZ/test_admin_impersonation.py`
  - Cover TTL, repository lookups, mandatory audit failure, and endpoint behavior.
- Modify: `tldw_Server_API/tests/AuthNZ/unit/test_jwt_service.py`
  - Cover explicit access-token expiry override and impersonation token claims.
- Create: `tldw_Server_API/tests/AuthNZ/unit/test_impersonation_auth_context.py`
  - Cover decoded impersonation metadata propagation into request `AuthContext`.

## Design Notes

- The default `JWTService.create_access_token()` behavior must remain unchanged for normal logins and existing tests.
- The new expiry override must reject non-positive values with `ValueError` so callers cannot accidentally mint already-expired or nonsensical tokens.
- The endpoint must not use `pool.acquire()` or raw cursor calls. User lookup should use `AuthnzUsersRepo.get_user_by_id()`. Role lookup should use `AuthnzRbacRepo.get_user_roles()` and fall back to the legacy `role` column or `"user"`.
- Issuance audit must be mandatory. If durable audit cannot flush, the endpoint should return HTTP 503 and no token should be reported as successfully issued.
- Step-up reauthentication remains out of scope for this first code slice because the current endpoint schema has no request body for admin password or reauth token. Record this as residual risk unless the implementation naturally introduces an approved request body without breaking clients.

---

### Task 1: Add JWT Expiry Support And Impersonation Token Tests

**Files:**
- Modify: `tldw_Server_API/tests/AuthNZ/unit/test_jwt_service.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/jwt_service.py`

- [x] **Step 1: Add failing JWT service tests**

Append these tests to `tldw_Server_API/tests/AuthNZ/unit/test_jwt_service.py`:

```python
from datetime import timedelta


def test_create_access_token_accepts_expiry_override(jwt_service):
    token = jwt_service.create_access_token(
        user_id=42,
        username="target",
        role="user",
        expires_delta=timedelta(minutes=15),
    )

    payload = jwt.decode(
        token,
        jwt_service._decode_key,
        algorithms=[jwt_service.algorithm],
        options={"verify_aud": False},
    )

    assert payload["sub"] == "42"
    assert int(payload["exp"]) - int(payload["iat"]) == 15 * 60


def test_create_impersonation_access_token_marks_actor_and_short_ttl(jwt_service):
    token = jwt_service.create_impersonation_access_token(
        user_id=42,
        username="target",
        role="user",
        impersonated_by=7,
        expires_delta=timedelta(minutes=15),
    )

    payload = jwt.decode(
        token,
        jwt_service._decode_key,
        algorithms=[jwt_service.algorithm],
        options={"verify_aud": False},
    )

    assert payload["sub"] == "42"
    assert payload["impersonation"] is True
    assert payload["impersonated_by"] == 7
    assert int(payload["exp"]) - int(payload["iat"]) == 15 * 60


def test_create_access_token_rejects_non_positive_expiry_override(jwt_service):
    with pytest.raises(ValueError, match="expires_delta must be positive"):
        jwt_service.create_access_token(
            user_id=42,
            username="target",
            role="user",
            expires_delta=timedelta(seconds=0),
        )
```

- [x] **Step 2: Run the new JWT tests and verify they fail**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tldw_Server_API/tests/AuthNZ/unit/test_jwt_service.py::test_create_access_token_accepts_expiry_override \
  tldw_Server_API/tests/AuthNZ/unit/test_jwt_service.py::test_create_impersonation_access_token_marks_actor_and_short_ttl \
  tldw_Server_API/tests/AuthNZ/unit/test_jwt_service.py::test_create_access_token_rejects_non_positive_expiry_override \
  -q
```

Expected: fail because `expires_delta` and `create_impersonation_access_token` do not exist yet.

- [x] **Step 3: Implement expiry override and impersonation token helper**

In `tldw_Server_API/app/core/AuthNZ/jwt_service.py`, update `create_access_token`:

```python
    def create_access_token(
        self,
        user_id: int,
        username: str,
        role: str,
        additional_claims: Optional[dict[str, Any]] = None,
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        if expires_delta is not None:
            if expires_delta.total_seconds() <= 0:
                raise ValueError("expires_delta must be positive")
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=self.settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        issued_at = datetime.now(timezone.utc)
```

Then set `"iat": issued_at` in the payload. Add this helper below `create_access_token`:

```python
    def create_impersonation_access_token(
        self,
        *,
        user_id: int,
        username: str,
        role: str,
        impersonated_by: int,
        expires_delta: timedelta,
    ) -> str:
        return self.create_access_token(
            user_id=user_id,
            username=username,
            role=role,
            additional_claims={
                "impersonated_by": int(impersonated_by),
                "impersonation": True,
            },
            expires_delta=expires_delta,
        )
```

- [x] **Step 4: Run JWT tests and verify they pass**

Run the same command from Step 2.

Expected: 3 passed.

- [x] **Step 5: Commit JWT token changes**

Run:

```bash
git add tldw_Server_API/app/core/AuthNZ/jwt_service.py tldw_Server_API/tests/AuthNZ/unit/test_jwt_service.py
git commit -m "fix: support short-lived impersonation tokens"
```

Expected: commit succeeds.

---

### Task 2: Propagate Impersonation Metadata Into AuthContext

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/principal_model.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py`
- Create: `tldw_Server_API/tests/AuthNZ/unit/test_impersonation_auth_context.py`

- [x] **Step 1: Add failing AuthContext propagation test**

Create `tldw_Server_API/tests/AuthNZ/unit/test_impersonation_auth_context.py`:

```python
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import verify_jwt_and_fetch_user


class _Headers(dict):
    def get(self, key: str, default=None):
        return super().get(key, default)


def _request() -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(request_id="req-1"),
        client=SimpleNamespace(host="127.0.0.1"),
        headers=_Headers({"User-Agent": "pytest", "X-Request-ID": "req-1"}),
        scope={},
    )


@pytest.mark.asyncio
async def test_impersonation_claims_populate_auth_context(monkeypatch):
    request = _request()

    class JwtStub:
        def decode_access_token(self, _token: str):
            return {
                "sub": "42",
                "username": "target",
                "impersonation": True,
                "impersonated_by": 7,
            }

    class RepoStub:
        @classmethod
        async def from_pool(cls):
            return cls()

        async def get_user_by_id(self, user_id: int):
            assert user_id == 42
            return {
                "id": 42,
                "uuid": "target-uuid",
                "username": "target",
                "email": "target@example.com",
                "role": "user",
                "is_active": True,
                "is_verified": True,
                "is_superuser": False,
            }

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.User_DB_Handling.get_jwt_service",
        lambda: JwtStub(),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.User_DB_Handling.get_session_manager",
        AsyncMock(return_value=SimpleNamespace(is_token_blacklisted=AsyncMock(return_value=False))),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.User_DB_Handling.AuthnzUsersRepo",
        RepoStub,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.User_DB_Handling.list_memberships_for_user",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.User_DB_Handling.apply_scoped_permissions",
        AsyncMock(return_value=SimpleNamespace(permissions=[], active_org_id=None, active_team_id=None)),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.User_DB_Handling._enrich_user_with_rbac",
        lambda *_, **__: (["user"], [], False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.User_DB_Handling.set_scope",
        lambda **_: None,
    )

    user = await verify_jwt_and_fetch_user("token", request)

    assert user.id == 42
    assert request.state.impersonation is True
    assert request.state.impersonated_by_user_id == 7
    assert request.state.auth.principal.user_id == 42
    assert request.state.auth.principal.impersonation is True
    assert request.state.auth.principal.impersonated_by_user_id == 7
```

- [x] **Step 2: Run the AuthContext test and verify it fails**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tldw_Server_API/tests/AuthNZ/unit/test_impersonation_auth_context.py -q
```

Expected: fail because impersonation fields are absent from `AuthPrincipal` or request state.

- [x] **Step 3: Add impersonation fields to `AuthPrincipal`**

In `tldw_Server_API/app/core/AuthNZ/principal_model.py`, add these fields after `jti`:

```python
    impersonation: bool = Field(
        default=False,
        description="True when this principal was authenticated with an impersonation token.",
    )
    impersonated_by_user_id: int | None = Field(
        default=None,
        description="Admin actor user id when this is an impersonated request.",
    )
```

- [x] **Step 4: Populate request state and principal fields**

In `verify_jwt_and_fetch_user`, after token membership claim extraction, add:

```python
        token_impersonation = bool(payload.get("impersonation"))
        token_impersonated_by = payload.get("impersonated_by")
        impersonated_by_user_id: Optional[int] = None
        if token_impersonated_by is not None:
            try:
                impersonated_by_user_id = int(token_impersonated_by)
            except (TypeError, ValueError):
                logger.warning("Token impersonated_by claim is invalid")
                raise credentials_exception
```

When setting request state near the existing `request.state.user_id = user.id`, add:

```python
        request.state.impersonation = token_impersonation
        request.state.impersonated_by_user_id = impersonated_by_user_id
```

When constructing `AuthPrincipal`, pass:

```python
            impersonation=token_impersonation,
            impersonated_by_user_id=impersonated_by_user_id,
```

- [x] **Step 5: Run the AuthContext test and verify it passes**

Run the same command from Step 2.

Expected: 1 passed.

- [x] **Step 6: Commit AuthContext propagation**

Run:

```bash
git add tldw_Server_API/app/core/AuthNZ/principal_model.py tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py tldw_Server_API/tests/AuthNZ/unit/test_impersonation_auth_context.py
git commit -m "fix: propagate impersonation auth context"
```

Expected: commit succeeds.

---

### Task 3: Replace Raw Impersonation Lookups And Add Mandatory Issuance Audit

**Files:**
- Modify: `tldw_Server_API/app/services/admin_audit_service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/admin/admin_impersonation.py`
- Modify: `tldw_Server_API/tests/AuthNZ/test_admin_impersonation.py`

- [x] **Step 1: Add failing endpoint tests for repository lookup and short TTL**

In `tldw_Server_API/tests/AuthNZ/test_admin_impersonation.py`, replace the current `test_success` body with this repository-based test:

```python
    @pytest.mark.asyncio
    async def test_success_uses_repositories_short_ttl_and_mandatory_audit(self):
        principal = _admin_principal()

        class UsersRepoStub:
            @classmethod
            async def from_pool(cls):
                return cls()

            async def get_user_by_id(self, user_id: int):
                assert user_id == 42
                return {"id": 42, "username": "targetuser", "is_active": True, "role": "legacy"}

        class RbacRepoStub:
            def get_user_roles(self, user_id: int):
                assert user_id == 42
                return [{"name": "user"}]

        mock_jwt_svc = MagicMock()
        mock_jwt_svc.create_impersonation_access_token = MagicMock(return_value="mock.jwt.token")
        audit = AsyncMock()

        with (
            patch(
                "tldw_Server_API.app.api.v1.endpoints.admin.admin_impersonation.AuthnzUsersRepo",
                UsersRepoStub,
            ),
            patch(
                "tldw_Server_API.app.api.v1.endpoints.admin.admin_impersonation.AuthnzRbacRepo",
                return_value=RbacRepoStub(),
            ),
            patch(
                "tldw_Server_API.app.api.v1.endpoints.admin.admin_impersonation.get_jwt_service",
                return_value=mock_jwt_svc,
            ),
            patch(
                "tldw_Server_API.app.api.v1.endpoints.admin.admin_impersonation.emit_impersonation_issuance_audit_event",
                audit,
            ),
        ):
            result = await create_impersonation_token(42, principal)

        assert result.token == "mock.jwt.token"
        assert result.impersonated_user_id == 42
        assert result.impersonated_by == 1
        mock_jwt_svc.create_impersonation_access_token.assert_called_once()
        token_kwargs = mock_jwt_svc.create_impersonation_access_token.call_args.kwargs
        assert token_kwargs["user_id"] == 42
        assert token_kwargs["username"] == "targetuser"
        assert token_kwargs["role"] == "user"
        assert token_kwargs["impersonated_by"] == 1
        assert token_kwargs["expires_delta"].total_seconds() == 15 * 60
        audit.assert_awaited_once()
```

- [x] **Step 2: Add failing mandatory-audit failure test**

Append:

```python
    @pytest.mark.asyncio
    async def test_mandatory_audit_failure_returns_503(self):
        principal = _admin_principal()

        class UsersRepoStub:
            @classmethod
            async def from_pool(cls):
                return cls()

            async def get_user_by_id(self, user_id: int):
                return {"id": 42, "username": "targetuser", "is_active": True, "role": "user"}

        class RbacRepoStub:
            def get_user_roles(self, user_id: int):
                return []

        mock_jwt_svc = MagicMock()
        mock_jwt_svc.create_impersonation_access_token = MagicMock(return_value="mock.jwt.token")

        with (
            patch(
                "tldw_Server_API.app.api.v1.endpoints.admin.admin_impersonation.AuthnzUsersRepo",
                UsersRepoStub,
            ),
            patch(
                "tldw_Server_API.app.api.v1.endpoints.admin.admin_impersonation.AuthnzRbacRepo",
                return_value=RbacRepoStub(),
            ),
            patch(
                "tldw_Server_API.app.api.v1.endpoints.admin.admin_impersonation.get_jwt_service",
                return_value=mock_jwt_svc,
            ),
            patch(
                "tldw_Server_API.app.api.v1.endpoints.admin.admin_impersonation.emit_impersonation_issuance_audit_event",
                AsyncMock(side_effect=HTTPException(status_code=503, detail="Mandatory audit persistence unavailable")),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await create_impersonation_token(42, principal)

        assert exc_info.value.status_code == 503
        assert exc_info.value.detail == "Mandatory audit persistence unavailable"
```

- [x] **Step 3: Run endpoint tests and verify they fail**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tldw_Server_API/tests/AuthNZ/test_admin_impersonation.py -q
```

Expected: fail because endpoint still imports DB pool directly and has no mandatory audit helper.

- [x] **Step 4: Add mandatory impersonation audit helper**

In `tldw_Server_API/app/services/admin_audit_service.py`, import `MandatoryAuditWriteError` and add:

```python
async def emit_impersonation_issuance_audit_event(
    *,
    actor_id: int | None,
    target_user_id: int,
    expires_in_minutes: int,
) -> None:
    try:
        svc = await get_or_create_audit_service_for_user_id_optional(actor_id)
        ctx = AuditContext(
            user_id=str(actor_id) if actor_id is not None else None,
            endpoint="/api/v1/admin/impersonate/{user_id}/token",
            method="POST",
        )
        await svc.log_event(
            event_type=AuditEventType.AUTH_TOKEN_CREATED,
            category=AuditEventCategory.AUTHENTICATION,
            context=ctx,
            resource_type="user_impersonation",
            resource_id=str(target_user_id),
            action="admin.impersonation.token_issued",
            metadata={
                "actor_id": actor_id,
                "target_user_id": target_user_id,
                "expires_in_minutes": expires_in_minutes,
                "impersonation": True,
            },
        )
        await svc.flush(raise_on_failure=True)
    except MandatoryAuditWriteError:
        raise
    except Exception as exc:
        logger.warning("Mandatory impersonation audit emission failed: {}", exc)
        raise MandatoryAuditWriteError("impersonation issuance audit failed") from exc
```

Use `AuditEventType.AUTH_TOKEN_CREATED` with a specific `action` and `resource_type` so impersonation issuance is represented as token creation while remaining distinguishable in audit searches.

- [x] **Step 5: Refactor the impersonation endpoint**

In `tldw_Server_API/app/api/v1/endpoints/admin/admin_impersonation.py`:

- Add imports:

```python
from datetime import timedelta

from tldw_Server_API.app.core.AuthNZ.repos.rbac_repo import AuthnzRbacRepo
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
from tldw_Server_API.app.core.Audit.unified_audit_service import MandatoryAuditWriteError
from tldw_Server_API.app.services.admin_audit_service import emit_impersonation_issuance_audit_event
```

- Replace raw `get_db_pool()` lookup with:

```python
        repo = await AuthnzUsersRepo.from_pool()
        row = await repo.get_user_by_id(user_id)
```

- Normalize target fields from dicts:

```python
        target_user_id = int(row["id"])
        target_username = str(row["username"])
        target_is_active = bool(row.get("is_active", False))
```

- Replace role lookup with:

```python
        try:
            role_rows = AuthnzRbacRepo().get_user_roles(target_user_id)
        except Exception:
            logger.warning("Unable to load RBAC roles for impersonation target; falling back to user row role")
            role_rows = []
        target_role = str(role_rows[0].get("name")) if role_rows else str(row.get("role") or "user")
```

- Replace token creation with:

```python
        jwt_svc = get_jwt_service()
        token = jwt_svc.create_impersonation_access_token(
            user_id=target_user_id,
            username=target_username,
            role=target_role,
            impersonated_by=principal.user_id,
            expires_delta=timedelta(minutes=_IMPERSONATION_TTL_MINUTES),
        )
```

- Before returning, call:

```python
        try:
            await emit_impersonation_issuance_audit_event(
                actor_id=principal.user_id,
                target_user_id=target_user_id,
                expires_in_minutes=_IMPERSONATION_TTL_MINUTES,
            )
        except MandatoryAuditWriteError as exc:
            logger.error("Mandatory audit write failed while creating impersonation token")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Mandatory audit persistence unavailable",
            ) from exc
```

- [x] **Step 6: Run endpoint tests and verify they pass**

Run the command from Step 3.

Expected: all impersonation endpoint tests pass.

- [x] **Step 7: Commit endpoint and audit changes**

Run:

```bash
git add \
  tldw_Server_API/app/services/admin_audit_service.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_impersonation.py \
  tldw_Server_API/tests/AuthNZ/test_admin_impersonation.py
git commit -m "fix: harden admin impersonation issuance"
```

Expected: commit succeeds.

---

### Task 4: Verification, Bandit, And Backlog Closure Evidence

**Files:**
- Modify: `backlog/tasks/task-12073 - Plan-and-remediate-AuthNZ-impersonation-audit-findings.md`

- [x] **Step 1: Run focused AuthNZ tests**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tldw_Server_API/tests/AuthNZ/test_admin_impersonation.py \
  tldw_Server_API/tests/AuthNZ/unit/test_jwt_service.py \
  tldw_Server_API/tests/AuthNZ/unit/test_impersonation_auth_context.py \
  -q
```

Expected: all selected tests pass.

- [x] **Step 2: Run broader AuthNZ safety tests**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tldw_Server_API/tests/AuthNZ \
  tldw_Server_API/tests/AuthNZ_Unit \
  -q
```

Expected: pass, or record unrelated pre-existing failures with evidence.

- [x] **Step 3: Run Bandit on touched production paths**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m bandit \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_impersonation.py \
  tldw_Server_API/app/core/AuthNZ/jwt_service.py \
  tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py \
  tldw_Server_API/app/core/AuthNZ/principal_model.py \
  tldw_Server_API/app/services/admin_audit_service.py \
  -f json -o /tmp/bandit_authnz_impersonation.json
```

Expected: no new high or medium findings in touched paths. If Bandit reports existing unrelated findings, record them explicitly.

- [x] **Step 4: Run whitespace and status checks**

Run:

```bash
git diff --check
git status --short --branch
```

Expected: no whitespace errors; tracked changes limited to the AuthNZ remediation files and `TASK-12073`.

- [x] **Step 5: Update Backlog task evidence**

Use Backlog MCP to update `TASK-12073`:

- Check acceptance criteria 1-5.
- Check definition of done 1-6.
- Add implementation notes listing exact test commands and Bandit command results.
- Final summary must state whether `AUTH-001`, `AUTH-002`, and `AUTH-003` are closed or have residual risk.

- [x] **Step 6: Commit final task update**

Run:

```bash
git add "backlog/tasks/task-12073 - Plan-and-remediate-AuthNZ-impersonation-audit-findings.md"
git commit -m "docs: close authnz impersonation remediation task"
```

Expected: commit succeeds.

---

## Self-Review Checklist

- `AUTH-001` has a test that decodes the actual token and checks `exp - iat`.
- `AUTH-002` has a test that proves impersonation metadata reaches `request.state.auth.principal`.
- `AUTH-002` issuance has mandatory durable audit behavior and a 503 failure path.
- `AUTH-003` removes raw `pool.acquire()` SQL from the endpoint.
- Normal access token lifetime behavior remains unchanged.
- No new request-body contract is introduced unless the worker also updates endpoint tests and docs.
- Bandit is run over touched production paths before completion.

## Completion Evidence

- Focused AuthNZ remediation tests: 40 passed, 170 warnings.
- Broader AuthNZ/AuthNZ_Unit safety tests: 1306 passed, 175 skipped, 10846 warnings.
- Bandit on touched production paths: no high or medium findings; 13 low token-type literal false positives recorded in `TASK-12073`.
- Endpoint search confirmed no remaining `get_db_pool`, `pool.acquire`, raw user/user-role `SELECT`, or generic `create_access_token` usage in `admin_impersonation.py`.
- `TASK-12073` is marked Done with acceptance criteria and definition of done checked.
