from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.api.v1.utils.chatops.telegram_support import (
    TelegramScope,
    _peek_pending_telegram_approval_for_tests,
    build_telegram_approval_callback_data,
)
from tldw_Server_API.app.services.mcp_hub_approval_service import _scope_key_for_tool_call


class _FakeTelegramApprovalsRepo:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, object]] = {}

    async def create_pending_approval(
        self,
        *,
        approval_token: str,
        scope_type: str,
        scope_id: int,
        approval_policy_id: int | None,
        context_key: str,
        conversation_id: str | None,
        tool_name: str,
        scope_key: str,
        initiating_auth_user_id: int,
        expires_at: datetime,
        now: datetime | None = None,
    ) -> dict[str, object]:
        row = {
            "approval_token": approval_token,
            "scope_type": scope_type,
            "scope_id": scope_id,
            "approval_policy_id": approval_policy_id,
            "context_key": context_key,
            "conversation_id": conversation_id,
            "tool_name": tool_name,
            "scope_key": scope_key,
            "initiating_auth_user_id": initiating_auth_user_id,
            "expires_at": expires_at,
            "created_at": now or datetime.now(timezone.utc),
            "consumed_at": None,
        }
        self.rows[approval_token] = row
        return dict(row)

    async def get_pending_approval_by_token(
        self,
        approval_token: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, object] | None:
        row = self.rows.get(approval_token)
        if row is None:
            return None
        current = now or datetime.now(timezone.utc)
        if row.get("consumed_at") is not None:
            return None
        expires_at = row.get("expires_at")
        if isinstance(expires_at, datetime) and expires_at <= current:
            return None
        return dict(row)

    async def consume_pending_approval(
        self,
        approval_token: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, object] | None:
        row = await self.get_pending_approval_by_token(approval_token, now=now)
        if row is None:
            return None
        stored = self.rows[approval_token]
        stored["consumed_at"] = now or datetime.now(timezone.utc)
        row["consumed_at"] = stored["consumed_at"]
        return row


@pytest.mark.asyncio
async def test_telegram_approval_callback_data_preserves_exact_scope_fingerprint() -> None:
    repo = _FakeTelegramApprovalsRepo()
    scope_payload = {
        "path_scope_mode": "cwd_descendants",
        "workspace_root": "/tmp/project",
        "scope_root": "/tmp/project/src",
        "normalized_paths": ["/tmp/project/README.md"],
        "reason": "path_outside_current_folder_scope",
    }
    scope_key = _scope_key_for_tool_call(
        "Bash",
        {"command": "git status", "args": ["--short"]},
        scope_payload=scope_payload,
    )

    callback_data = await build_telegram_approval_callback_data(
        approval_policy_id=17,
        context_key="user:202|group:88|persona:researcher",
        conversation_id="conv-1",
        tool_name="Bash",
        tool_args={"command": "git status", "args": ["--short"]},
        scope=TelegramScope(scope_type="group", scope_id=88),
        initiating_auth_user_id=202,
        scope_payload=scope_payload,
        repo=repo,
    )

    pending = await _peek_pending_telegram_approval_for_tests(callback_data, repo=repo)

    assert pending is not None
    assert pending["scope_key"] == scope_key
    assert pending["tool_name"] == "Bash"
    assert pending["context_key"] == "user:202|group:88|persona:researcher"
    assert len(callback_data) <= 64


@pytest.mark.asyncio
async def test_telegram_approval_callback_data_changes_when_scope_payload_changes() -> None:
    repo = _FakeTelegramApprovalsRepo()

    first = await build_telegram_approval_callback_data(
        approval_policy_id=17,
        context_key="user:202|group:88|persona:researcher",
        conversation_id="conv-1",
        tool_name="files.read",
        tool_args={"path": "../README.md"},
        scope=TelegramScope(scope_type="group", scope_id=88),
        initiating_auth_user_id=202,
        scope_payload={
            "path_scope_mode": "cwd_descendants",
            "workspace_root": "/tmp/project",
            "scope_root": "/tmp/project/src",
            "normalized_paths": ["/tmp/project/README.md"],
            "reason": "path_outside_current_folder_scope",
        },
        repo=repo,
    )
    second = await build_telegram_approval_callback_data(
        approval_policy_id=17,
        context_key="user:202|group:88|persona:researcher",
        conversation_id="conv-1",
        tool_name="files.read",
        tool_args={"path": "../README.md"},
        scope=TelegramScope(scope_type="group", scope_id=88),
        initiating_auth_user_id=202,
        scope_payload={
            "path_scope_mode": "cwd_descendants",
            "workspace_root": "/tmp/project",
            "scope_root": "/tmp/project/src",
            "normalized_paths": ["/tmp/project/docs/README.md"],
            "reason": "path_outside_current_folder_scope",
        },
        repo=repo,
    )

    first_pending = await _peek_pending_telegram_approval_for_tests(first, repo=repo)
    second_pending = await _peek_pending_telegram_approval_for_tests(second, repo=repo)

    assert first_pending is not None
    assert second_pending is not None
    assert first_pending["scope_key"] != second_pending["scope_key"]

