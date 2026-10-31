from __future__ import annotations

import base64
from pathlib import Path

import pytest

from tldw_Server_API.app.core.AuthNZ.repos.mcp_hub_repo import McpHubRepo
from tldw_Server_API.app.core.exceptions import BadRequestError


def _b64_key(byte_char: bytes) -> str:
    return base64.b64encode(byte_char * 32).decode("ascii")


async def _build_service(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services.mcp_hub_service import McpHubService

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))

    reset_settings()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    repo = McpHubRepo(pool)
    await repo.ensure_tables()
    return McpHubService(repo=repo), repo


@pytest.mark.asyncio
async def test_list_external_servers_reports_no_auth_template_status(tmp_path, monkeypatch, fresh_services) -> None:
    svc, _repo = await _build_service(tmp_path, monkeypatch)

    await svc.create_external_server(
        server_id="docs",
        name="Docs",
        transport="websocket",
        config={"websocket": {"url": "wss://docs.example/ws"}},
        owner_scope_type="global",
        owner_scope_id=None,
        enabled=True,
        actor_id=1,
    )

    rows = await svc.list_external_servers()
    docs = next(row for row in rows if row["id"] == "docs")

    assert docs["auth_template_present"] is False
    assert docs["auth_template_valid"] is False
    assert docs["auth_template_blocked_reason"] == "no_auth_template"


@pytest.mark.asyncio
async def test_list_external_servers_reports_missing_required_slot_secret_status(tmp_path, monkeypatch, fresh_services) -> None:
    svc, repo = await _build_service(tmp_path, monkeypatch)

    await svc.create_external_server(
        server_id="docs",
        name="Docs",
        transport="websocket",
        config={"websocket": {"url": "wss://docs.example/ws"}},
        owner_scope_type="global",
        owner_scope_id=None,
        enabled=True,
        actor_id=1,
    )
    await repo.create_external_server_credential_slot(
        server_id="docs",
        slot_name="token_readonly",
        display_name="Read-only token",
        secret_kind="bearer_token",
        privilege_class="read",
        is_required=True,
        actor_id=1,
    )
    await svc.update_external_server(
        "docs",
        config={
            "websocket": {"url": "wss://docs.example/ws"},
            "auth": {
                "mode": "template",
                "mappings": [
                    {
                        "slot_name": "token_readonly",
                        "target_type": "header",
                        "target_name": "Authorization",
                        "prefix": "Bearer ",
                        "suffix": "",
                        "required": True,
                    }
                ],
            },
        },
        actor_id=1,
    )

    rows = await svc.list_external_servers()
    docs = next(row for row in rows if row["id"] == "docs")

    assert docs["auth_template_present"] is True
    assert docs["auth_template_valid"] is False
    assert docs["auth_template_blocked_reason"] == "required_slot_secret_missing"


@pytest.mark.asyncio
async def test_update_external_server_rejects_transport_mismatched_template_mapping(tmp_path, monkeypatch, fresh_services) -> None:
    svc, repo = await _build_service(tmp_path, monkeypatch)

    await svc.create_external_server(
        server_id="docs",
        name="Docs",
        transport="stdio",
        config={"stdio": {"command": "npx", "args": ["-y", "@docs/server"]}},
        owner_scope_type="global",
        owner_scope_id=None,
        enabled=True,
        actor_id=1,
    )
    await repo.create_external_server_credential_slot(
        server_id="docs",
        slot_name="token_readonly",
        display_name="Read-only token",
        secret_kind="bearer_token",
        privilege_class="read",
        is_required=True,
        actor_id=1,
    )

    with pytest.raises(BadRequestError, match="transport"):
        await svc.update_external_server(
            "docs",
            config={
                "stdio": {"command": "npx", "args": ["-y", "@docs/server"]},
                "auth": {
                    "mode": "template",
                    "mappings": [
                        {
                            "slot_name": "token_readonly",
                            "target_type": "header",
                            "target_name": "Authorization",
                            "prefix": "Bearer ",
                            "suffix": "",
                            "required": True,
                        }
                    ],
                },
            },
            actor_id=1,
        )
