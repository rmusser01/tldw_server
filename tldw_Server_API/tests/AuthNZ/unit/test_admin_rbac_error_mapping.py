from __future__ import annotations

from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.admin import admin_rbac
from tldw_Server_API.app.api.v1.schemas.admin_schemas import ToolPermissionGrantRequest
from tldw_Server_API.app.api.v1.schemas.admin_rbac_schemas import (
    OverrideEffect,
    PermissionCreateRequest,
    RoleCreateRequest,
    UserOverrideUpsertRequest,
)
from tldw_Server_API.app.core.AuthNZ import settings as auth_settings


class _ExplodingDB:
    def __init__(self, message: str) -> None:
        self.message = message

    async def execute(self, *_args, **_kwargs):
        raise RuntimeError(self.message)


async def _fake_is_pg() -> bool:
    return False


async def _allow_admin_scope(*_args, **_kwargs) -> None:
    return None


async def _assert_admin_rbac_log_sanitized(
    call: Callable[[], Awaitable[Any]],
    *,
    expected_log: str,
    raw_marker: str,
) -> None:
    messages: list[str] = []
    sink_id = admin_rbac.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        with pytest.raises(HTTPException) as excinfo:
            await call()
    finally:
        admin_rbac.logger.remove(sink_id)

    assert excinfo.value.status_code == 500
    joined = "\n".join(messages)
    assert expected_log in joined
    assert raw_marker not in joined
    assert "/private/" not in joined


def _configure_error_mapping_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(admin_rbac, "is_test_mode", lambda: True)
    monkeypatch.setattr(admin_rbac, "_get_is_postgres_backend_fn", lambda: _fake_is_pg)
    monkeypatch.setattr(admin_rbac, "_enforce_admin_user_scope", _allow_admin_scope)
    monkeypatch.setattr(
        auth_settings,
        "get_settings",
        lambda: SimpleNamespace(AUTH_MODE="multi_user", SINGLE_USER_FIXED_ID=1),
    )
    monkeypatch.setattr(auth_settings, "is_single_user_mode", lambda: False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("service_attr", "call_factory", "expected_log", "raw_marker"),
    [
        (
            "svc_list_roles",
            lambda: admin_rbac.list_roles(db=object()),
            "Failed to list roles",
            "roles endpoint failed",
        ),
        (
            "svc_create_role",
            lambda: admin_rbac.create_role(
                RoleCreateRequest(name="new-role", description="desc"),
                db=object(),
            ),
            "Failed to create role",
            "role create endpoint failed",
        ),
        (
            "svc_delete_role",
            lambda: admin_rbac.delete_role(role_id=42, db=object()),
            "Failed to delete role",
            "role delete endpoint failed",
        ),
        (
            "svc_list_role_permissions",
            lambda: admin_rbac.list_role_permissions(role_id=42, db=object()),
            "Failed to list role permissions",
            "role permissions endpoint failed",
        ),
        (
            "svc_list_tool_permissions",
            lambda: admin_rbac.list_tool_permissions(db=object()),
            "Failed to list tool permissions",
            "tool permissions endpoint failed",
        ),
        (
            "svc_delete_tool_permission",
            lambda: admin_rbac.delete_tool_permission(perm_name="tools.execute:test", db=object()),
            "Failed to delete tool permission",
            "tool permission delete endpoint failed",
        ),
        (
            "svc_grant_tool_perm",
            lambda: admin_rbac.grant_tool_permission_to_role(
                role_id=42,
                payload=ToolPermissionGrantRequest(tool_name="test"),
                db=object(),
            ),
            "Failed to grant tool permission",
            "tool permission grant endpoint failed",
        ),
        (
            "svc_revoke_tool_perm",
            lambda: admin_rbac.revoke_tool_permission_from_role(
                role_id=42,
                tool_name="test",
                db=object(),
            ),
            "Failed to revoke tool permission",
            "tool permission revoke endpoint failed",
        ),
    ],
)
async def test_role_tool_permission_endpoints_sanitize_backend_failure_logs(
    monkeypatch: pytest.MonkeyPatch,
    service_attr: str,
    call_factory: Callable[[], Awaitable[Any]],
    expected_log: str,
    raw_marker: str,
) -> None:
    async def _raise_backend_error(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(f"{raw_marker} at /private/rbac-endpoint.db")

    monkeypatch.setattr(admin_rbac, service_attr, _raise_backend_error)

    await _assert_admin_rbac_log_sanitized(
        call_factory,
        expected_log=expected_log,
        raw_marker=raw_marker,
    )


@pytest.mark.asyncio
async def test_create_permission_sanitizes_backend_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_error_mapping_env(monkeypatch)

    with pytest.raises(HTTPException) as excinfo:
        await admin_rbac.create_permission(
            PermissionCreateRequest(name="persona.write", description="write", category="persona"),
            db=_ExplodingDB("permission backend exploded"),
        )

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Failed to create permission"


@pytest.mark.asyncio
async def test_upsert_user_override_sanitizes_backend_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_error_mapping_env(monkeypatch)

    with pytest.raises(HTTPException) as excinfo:
        await admin_rbac.upsert_user_override(
            user_id=42,
            payload=UserOverrideUpsertRequest(permission_id=7, effect=OverrideEffect.allow),
            principal=object(),
            db=_ExplodingDB("override backend exploded"),
        )

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Failed to upsert user override"


@pytest.mark.asyncio
async def test_delete_user_override_sanitizes_backend_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_error_mapping_env(monkeypatch)

    with pytest.raises(HTTPException) as excinfo:
        await admin_rbac.delete_user_override(
            user_id=42,
            permission_id=7,
            principal=object(),
            db=_ExplodingDB("delete backend exploded"),
        )

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Failed to delete user override"
