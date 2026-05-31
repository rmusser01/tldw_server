from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import mcp_catalogs_manage
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.exceptions import ToolCatalogConflictError


class _LoggerStub:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.debugs: list[str] = []

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.errors.append(str(message))

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.debugs.append(str(message))


def _assert_sanitized_error_log(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.errors == [expected_message]
    rendered = " ".join(logger_stub.errors)
    assert "/private/" not in rendered
    assert "exploded" not in rendered


def _assert_sanitized_debug_log(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.debugs == [expected_message]
    rendered = " ".join(logger_stub.debugs)
    assert "/private/" not in rendered
    assert "exploded" not in rendered


def _principal(*, user_id: int | None, roles: list[str] | None = None, is_admin: bool = False) -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=user_id,
        roles=roles or [],
        permissions=[],
        is_admin=is_admin,
        org_ids=[],
        team_ids=[],
    )


@pytest.mark.asyncio
async def test_require_org_manager_allows_admin_role_claim_without_membership_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fail_if_called(*args, **kwargs):  # pragma: no cover - should not run
        raise AssertionError("admin principal should bypass org membership lookup")

    monkeypatch.setattr(mcp_catalogs_manage, "list_org_members", _fail_if_called)

    await mcp_catalogs_manage._require_org_manager(
        _principal(user_id=None, roles=["admin"], is_admin=False),
        org_id=12,
    )


@pytest.mark.asyncio
async def test_require_org_manager_rejects_boolean_admin_without_claims() -> None:
    with pytest.raises(HTTPException) as exc:
        await mcp_catalogs_manage._require_org_manager(
            _principal(user_id=None, roles=["user"], is_admin=True),
            org_id=12,
        )
    assert exc.value.status_code == 403
    assert exc.value.detail == "Org manager role required"


@pytest.mark.asyncio
async def test_require_org_manager_denies_missing_user_id_for_non_admin() -> None:
    with pytest.raises(HTTPException) as exc:
        await mcp_catalogs_manage._require_org_manager(
            _principal(user_id=None, roles=["user"], is_admin=False),
            org_id=7,
        )
    assert exc.value.status_code == 403
    assert exc.value.detail == "Org manager role required"


@pytest.mark.asyncio
async def test_require_org_manager_membership_backend_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()

    async def _raise_membership_error(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("org membership backend exploded at /private/mcp-catalogs.db")

    monkeypatch.setattr(mcp_catalogs_manage, "logger", logger_stub)
    monkeypatch.setattr(mcp_catalogs_manage, "list_org_members", _raise_membership_error)

    with pytest.raises(HTTPException) as exc:
        await mcp_catalogs_manage._require_org_manager(
            _principal(user_id=42, roles=["user"], is_admin=False),
            org_id=7,
        )

    assert exc.value.status_code == 403
    assert exc.value.detail == "Org manager role required"
    _assert_sanitized_debug_log(logger_stub, "Org manager check failed")


@pytest.mark.asyncio
async def test_require_team_manager_allows_lead_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_team_members(team_id: int):
        assert team_id == 88
        return [
            {"user_id": 5, "role": "member"},
            {"user_id": 42, "role": "lead"},
        ]

    monkeypatch.setattr(mcp_catalogs_manage, "list_team_members", _fake_team_members)

    await mcp_catalogs_manage._require_team_manager(
        _principal(user_id=42, roles=["user"], is_admin=False),
        team_id=88,
    )


@pytest.mark.asyncio
async def test_require_team_manager_denies_non_manager_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_team_members(team_id: int):
        assert team_id == 3
        return [{"user_id": 42, "role": "member"}]

    monkeypatch.setattr(mcp_catalogs_manage, "list_team_members", _fake_team_members)

    with pytest.raises(HTTPException) as exc:
        await mcp_catalogs_manage._require_team_manager(
            _principal(user_id=42, roles=["user"], is_admin=False),
            team_id=3,
        )
    assert exc.value.status_code == 403
    assert exc.value.detail == "Team manager role required"


@pytest.mark.asyncio
async def test_require_team_manager_membership_backend_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()

    async def _raise_membership_error(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("team membership backend exploded at /private/mcp-catalogs.db")

    monkeypatch.setattr(mcp_catalogs_manage, "logger", logger_stub)
    monkeypatch.setattr(mcp_catalogs_manage, "list_team_members", _raise_membership_error)

    with pytest.raises(HTTPException) as exc:
        await mcp_catalogs_manage._require_team_manager(
            _principal(user_id=42, roles=["user"], is_admin=False),
            team_id=3,
        )

    assert exc.value.status_code == 403
    assert exc.value.detail == "Team manager role required"
    _assert_sanitized_debug_log(logger_stub, "Team manager check failed")


@pytest.mark.asyncio
async def test_get_scoped_catalog_rejects_wrong_org_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_get_tool_catalog(_db, catalog_id: int):
        assert catalog_id == 5
        return {"id": 5, "org_id": 99, "team_id": None}

    monkeypatch.setattr(
        mcp_catalogs_manage.admin_tool_catalog_service,
        "get_tool_catalog",
        _fake_get_tool_catalog,
    )

    result = await mcp_catalogs_manage._get_scoped_catalog(
        db=object(),
        catalog_id=5,
        org_id=7,
    )
    assert result is None


@pytest.mark.asyncio
async def test_create_org_tool_catalog_maps_service_conflict_to_409(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _allow_manager(*_args, **_kwargs):
        return None

    async def _raise_conflict(*_args, **_kwargs):
        raise ToolCatalogConflictError("exists")

    monkeypatch.setattr(mcp_catalogs_manage, "_require_org_manager", _allow_manager)
    monkeypatch.setattr(
        mcp_catalogs_manage.admin_tool_catalog_service,
        "create_tool_catalog",
        _raise_conflict,
    )

    with pytest.raises(HTTPException) as exc:
        await mcp_catalogs_manage.create_org_tool_catalog(
            org_id=7,
            payload=mcp_catalogs_manage.ToolCatalogCreateRequest(name="dup-cat"),
            principal=_principal(user_id=42, roles=["lead"], is_admin=False),
            db=object(),
        )
    assert exc.value.status_code == 409
    assert exc.value.detail == "Catalog already exists"


@pytest.mark.asyncio
async def test_list_org_tool_catalogs_sanitizes_generic_error_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()

    async def _allow_manager(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def _raise_backend_error(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("org catalog backend exploded at /private/mcp-catalogs.db")

    monkeypatch.setattr(mcp_catalogs_manage, "logger", logger_stub)
    monkeypatch.setattr(mcp_catalogs_manage, "_require_org_manager", _allow_manager)
    monkeypatch.setattr(
        mcp_catalogs_manage.admin_tool_catalog_service,
        "list_tool_catalogs",
        _raise_backend_error,
    )

    with pytest.raises(HTTPException) as exc:
        await mcp_catalogs_manage.list_org_tool_catalogs(
            org_id=7,
            principal=_principal(user_id=42, roles=["lead"], is_admin=False),
            db=object(),
        )

    assert exc.value.status_code == 500
    assert exc.value.detail == "Failed to list org tool catalogs"
    _assert_sanitized_error_log(logger_stub, "Failed to list org tool catalogs")


@pytest.mark.asyncio
async def test_create_org_tool_catalog_sanitizes_generic_error_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()

    async def _allow_manager(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def _raise_backend_error(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("org catalog backend exploded at /private/mcp-catalogs.db")

    monkeypatch.setattr(mcp_catalogs_manage, "logger", logger_stub)
    monkeypatch.setattr(mcp_catalogs_manage, "_require_org_manager", _allow_manager)
    monkeypatch.setattr(
        mcp_catalogs_manage.admin_tool_catalog_service,
        "create_tool_catalog",
        _raise_backend_error,
    )

    with pytest.raises(HTTPException) as exc:
        await mcp_catalogs_manage.create_org_tool_catalog(
            org_id=7,
            payload=mcp_catalogs_manage.ToolCatalogCreateRequest(name="new-cat"),
            principal=_principal(user_id=42, roles=["lead"], is_admin=False),
            db=object(),
        )

    assert exc.value.status_code == 500
    assert exc.value.detail == "Failed to create tool catalog"
    _assert_sanitized_error_log(logger_stub, "Failed to create org tool catalog")


@pytest.mark.asyncio
async def test_list_team_tool_catalogs_sanitizes_generic_error_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()

    async def _allow_manager(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def _raise_backend_error(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("team catalog backend exploded at /private/mcp-catalogs.db")

    monkeypatch.setattr(mcp_catalogs_manage, "logger", logger_stub)
    monkeypatch.setattr(mcp_catalogs_manage, "_require_team_manager", _allow_manager)
    monkeypatch.setattr(
        mcp_catalogs_manage.admin_tool_catalog_service,
        "list_tool_catalogs",
        _raise_backend_error,
    )

    with pytest.raises(HTTPException) as exc:
        await mcp_catalogs_manage.list_team_tool_catalogs(
            team_id=11,
            principal=_principal(user_id=42, roles=["lead"], is_admin=False),
            db=object(),
        )

    assert exc.value.status_code == 500
    assert exc.value.detail == "Failed to list team tool catalogs"
    _assert_sanitized_error_log(logger_stub, "Failed to list team tool catalogs")


@pytest.mark.asyncio
async def test_create_team_tool_catalog_sanitizes_generic_error_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()

    async def _allow_manager(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def _raise_backend_error(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("team catalog backend exploded at /private/mcp-catalogs.db")

    monkeypatch.setattr(mcp_catalogs_manage, "logger", logger_stub)
    monkeypatch.setattr(mcp_catalogs_manage, "_require_team_manager", _allow_manager)
    monkeypatch.setattr(
        mcp_catalogs_manage.admin_tool_catalog_service,
        "create_tool_catalog",
        _raise_backend_error,
    )

    with pytest.raises(HTTPException) as exc:
        await mcp_catalogs_manage.create_team_tool_catalog(
            team_id=11,
            payload=mcp_catalogs_manage.ToolCatalogCreateRequest(name="new-cat"),
            principal=_principal(user_id=42, roles=["lead"], is_admin=False),
            db=object(),
        )

    assert exc.value.status_code == 500
    assert exc.value.detail == "Failed to create tool catalog"
    _assert_sanitized_error_log(logger_stub, "Failed to create team tool catalog")


@pytest.mark.asyncio
async def test_add_org_catalog_entry_sanitizes_generic_error_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()

    async def _allow_manager(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def _scoped_catalog(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"id": 5, "org_id": 7, "team_id": None}

    async def _raise_backend_error(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("org catalog entry backend exploded at /private/mcp-catalogs.db")

    monkeypatch.setattr(mcp_catalogs_manage, "logger", logger_stub)
    monkeypatch.setattr(mcp_catalogs_manage, "_require_org_manager", _allow_manager)
    monkeypatch.setattr(mcp_catalogs_manage, "_get_scoped_catalog", _scoped_catalog)
    monkeypatch.setattr(
        mcp_catalogs_manage.admin_tool_catalog_service,
        "add_tool_catalog_entry",
        _raise_backend_error,
    )

    with pytest.raises(HTTPException) as exc:
        await mcp_catalogs_manage.add_org_catalog_entry(
            org_id=7,
            catalog_id=5,
            payload=mcp_catalogs_manage.ToolCatalogEntryCreateRequest(tool_name="tool.echo"),
            principal=_principal(user_id=42, roles=["lead"], is_admin=False),
            db=object(),
        )

    assert exc.value.status_code == 500
    assert exc.value.detail == "Failed to add tool catalog entry"
    _assert_sanitized_error_log(logger_stub, "Failed to add org tool catalog entry")


@pytest.mark.asyncio
async def test_add_team_catalog_entry_sanitizes_generic_error_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()

    async def _allow_manager(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def _scoped_catalog(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"id": 5, "org_id": None, "team_id": 11}

    async def _raise_backend_error(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("team catalog entry backend exploded at /private/mcp-catalogs.db")

    monkeypatch.setattr(mcp_catalogs_manage, "logger", logger_stub)
    monkeypatch.setattr(mcp_catalogs_manage, "_require_team_manager", _allow_manager)
    monkeypatch.setattr(mcp_catalogs_manage, "_get_scoped_catalog", _scoped_catalog)
    monkeypatch.setattr(
        mcp_catalogs_manage.admin_tool_catalog_service,
        "add_tool_catalog_entry",
        _raise_backend_error,
    )

    with pytest.raises(HTTPException) as exc:
        await mcp_catalogs_manage.add_team_catalog_entry(
            team_id=11,
            catalog_id=5,
            payload=mcp_catalogs_manage.ToolCatalogEntryCreateRequest(tool_name="tool.echo"),
            principal=_principal(user_id=42, roles=["lead"], is_admin=False),
            db=object(),
        )

    assert exc.value.status_code == 500
    assert exc.value.detail == "Failed to add tool catalog entry"
    _assert_sanitized_error_log(logger_stub, "Failed to add team tool catalog entry")


@pytest.mark.asyncio
async def test_delete_org_tool_catalog_sanitizes_generic_error_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()

    async def _allow_manager(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def _scoped_catalog(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"id": 5, "org_id": 7, "team_id": None}

    async def _raise_backend_error(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("org catalog delete backend exploded at /private/mcp-catalogs.db")

    monkeypatch.setattr(mcp_catalogs_manage, "logger", logger_stub)
    monkeypatch.setattr(mcp_catalogs_manage, "_require_org_manager", _allow_manager)
    monkeypatch.setattr(mcp_catalogs_manage, "_get_scoped_catalog", _scoped_catalog)
    monkeypatch.setattr(
        mcp_catalogs_manage.admin_tool_catalog_service,
        "delete_tool_catalog",
        _raise_backend_error,
    )

    with pytest.raises(HTTPException) as exc:
        await mcp_catalogs_manage.delete_org_tool_catalog(
            org_id=7,
            catalog_id=5,
            principal=_principal(user_id=42, roles=["lead"], is_admin=False),
            db=object(),
        )

    assert exc.value.status_code == 500
    assert exc.value.detail == "Failed to delete tool catalog"
    _assert_sanitized_error_log(logger_stub, "Failed to delete org tool catalog")


@pytest.mark.asyncio
async def test_delete_org_catalog_entry_sanitizes_generic_error_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()

    async def _allow_manager(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def _scoped_catalog(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"id": 5, "org_id": 7, "team_id": None}

    async def _raise_backend_error(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("org catalog entry delete backend exploded at /private/mcp-catalogs.db")

    monkeypatch.setattr(mcp_catalogs_manage, "logger", logger_stub)
    monkeypatch.setattr(mcp_catalogs_manage, "_require_org_manager", _allow_manager)
    monkeypatch.setattr(mcp_catalogs_manage, "_get_scoped_catalog", _scoped_catalog)
    monkeypatch.setattr(
        mcp_catalogs_manage.admin_tool_catalog_service,
        "delete_tool_catalog_entry",
        _raise_backend_error,
    )

    with pytest.raises(HTTPException) as exc:
        await mcp_catalogs_manage.delete_org_catalog_entry(
            org_id=7,
            catalog_id=5,
            tool_name="tool.echo",
            principal=_principal(user_id=42, roles=["lead"], is_admin=False),
            db=object(),
        )

    assert exc.value.status_code == 500
    assert exc.value.detail == "Failed to delete tool catalog entry"
    _assert_sanitized_error_log(logger_stub, "Failed to delete org tool catalog entry")


@pytest.mark.asyncio
async def test_delete_team_tool_catalog_sanitizes_generic_error_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()

    async def _allow_manager(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def _scoped_catalog(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"id": 5, "org_id": None, "team_id": 11}

    async def _raise_backend_error(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("team catalog delete backend exploded at /private/mcp-catalogs.db")

    monkeypatch.setattr(mcp_catalogs_manage, "logger", logger_stub)
    monkeypatch.setattr(mcp_catalogs_manage, "_require_team_manager", _allow_manager)
    monkeypatch.setattr(mcp_catalogs_manage, "_get_scoped_catalog", _scoped_catalog)
    monkeypatch.setattr(
        mcp_catalogs_manage.admin_tool_catalog_service,
        "delete_tool_catalog",
        _raise_backend_error,
    )

    with pytest.raises(HTTPException) as exc:
        await mcp_catalogs_manage.delete_team_tool_catalog(
            team_id=11,
            catalog_id=5,
            principal=_principal(user_id=42, roles=["lead"], is_admin=False),
            db=object(),
        )

    assert exc.value.status_code == 500
    assert exc.value.detail == "Failed to delete tool catalog"
    _assert_sanitized_error_log(logger_stub, "Failed to delete team tool catalog")


@pytest.mark.asyncio
async def test_delete_team_catalog_entry_sanitizes_generic_error_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()

    async def _allow_manager(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def _scoped_catalog(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"id": 5, "org_id": None, "team_id": 11}

    async def _raise_backend_error(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("team catalog entry delete backend exploded at /private/mcp-catalogs.db")

    monkeypatch.setattr(mcp_catalogs_manage, "logger", logger_stub)
    monkeypatch.setattr(mcp_catalogs_manage, "_require_team_manager", _allow_manager)
    monkeypatch.setattr(mcp_catalogs_manage, "_get_scoped_catalog", _scoped_catalog)
    monkeypatch.setattr(
        mcp_catalogs_manage.admin_tool_catalog_service,
        "delete_tool_catalog_entry",
        _raise_backend_error,
    )

    with pytest.raises(HTTPException) as exc:
        await mcp_catalogs_manage.delete_team_catalog_entry(
            team_id=11,
            catalog_id=5,
            tool_name="tool.echo",
            principal=_principal(user_id=42, roles=["lead"], is_admin=False),
            db=object(),
        )

    assert exc.value.status_code == 500
    assert exc.value.detail == "Failed to delete tool catalog entry"
    _assert_sanitized_error_log(logger_stub, "Failed to delete team tool catalog entry")
