from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import Request

import tldw_Server_API.app.api.v1.endpoints.telegram_support as telegram_support
from tldw_Server_API.app.api.v1.endpoints.telegram_support import (
    TelegramScope,
    TelegramWebhookContext,
    build_telegram_approval_callback_data,
    telegram_webhook_impl,
)
from tldw_Server_API.app.services.mcp_hub_approval_service import _scope_key_for_tool_call


@dataclass
class _RecordedApprovalDecision:
    id: int
    approval_policy_id: int | None
    context_key: str
    conversation_id: str | None
    tool_name: str
    scope_key: str
    decision: str
    consume_on_match: bool
    expires_at: datetime | None
    actor_id: int | None


class _FakeApprovalService:
    def __init__(self) -> None:
        self.recorded: list[_RecordedApprovalDecision] = []

    async def record_decision(
        self,
        *,
        approval_policy_id: int | None,
        context_key: str,
        conversation_id: str | None,
        tool_name: str,
        scope_key: str,
        decision: str,
        consume_on_match: bool = False,
        expires_at: datetime | None = None,
        actor_id: int | None = None,
    ) -> dict[str, object]:
        row = _RecordedApprovalDecision(
            id=len(self.recorded) + 1,
            approval_policy_id=approval_policy_id,
            context_key=context_key,
            conversation_id=conversation_id,
            tool_name=tool_name,
            scope_key=scope_key,
            decision=decision,
            consume_on_match=consume_on_match,
            expires_at=expires_at,
            actor_id=actor_id,
        )
        self.recorded.append(row)
        return {
            "id": row.id,
            "approval_policy_id": row.approval_policy_id,
            "context_key": row.context_key,
            "conversation_id": row.conversation_id,
            "tool_name": row.tool_name,
            "scope_key": row.scope_key,
            "decision": row.decision,
            "consume_on_match": row.consume_on_match,
            "expires_at": row.expires_at,
            "created_by": row.actor_id,
        }


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


class _FakeTelegramRuntimeRepo:
    def __init__(self) -> None:
        self.links: dict[tuple[str, int, int], dict[str, object]] = {}
        self.receipts: set[str] = set()

    async def store_webhook_receipt(
        self,
        *,
        dedupe_key: str,
        scope_type: str,
        scope_id: int,
        update_id: int,
        expires_at: datetime,
        now: datetime | None = None,
    ) -> bool:
        _ = (scope_type, scope_id, update_id, expires_at, now)
        if dedupe_key in self.receipts:
            return False
        self.receipts.add(dedupe_key)
        return True

    async def upsert_actor_link(
        self,
        *,
        scope_type: str,
        scope_id: int,
        telegram_user_id: int,
        auth_user_id: int,
        telegram_username: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, object]:
        row = {
            "scope_type": scope_type,
            "scope_id": scope_id,
            "telegram_user_id": telegram_user_id,
            "auth_user_id": auth_user_id,
            "telegram_username": telegram_username,
            "updated_at": now or datetime.now(timezone.utc),
        }
        self.links[(scope_type, scope_id, telegram_user_id)] = row
        return dict(row)

    async def get_actor_link(
        self,
        *,
        scope_type: str,
        scope_id: int,
        telegram_user_id: int,
    ) -> dict[str, object] | None:
        row = self.links.get((scope_type, scope_id, telegram_user_id))
        return dict(row) if row is not None else None


class _LoggerStub:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def error(self, message: str, *args: object, **kwargs: object) -> None:
        self.errors.append(message.format(*args, **kwargs) if args or kwargs else message)


def _approval_callback_payload(callback_data: str, *, telegram_user_id: int = 202) -> dict[str, object]:
    return {
        "callback_query": {
            "from": {"id": telegram_user_id, "is_bot": False},
            "data": callback_data,
        }
    }


def _assert_sanitized_error_log(logger: _LoggerStub, expected: list[str]) -> None:
    assert logger.errors == expected
    rendered = "\n".join(logger.errors)
    assert "exploded" not in rendered
    assert "/private/" not in rendered


def _build_request(*, update_id: int, telegram_user_id: int, callback_data: str) -> Request:
    body = {
        "update_id": update_id,
        "callback_query": {
            "id": f"cb-{update_id}",
            "from": {"id": telegram_user_id, "is_bot": False, "username": f"user-{telegram_user_id}"},
            "message": {
                "message_id": 21,
                "chat": {"id": -1001, "type": "group"},
            },
            "data": callback_data,
        },
    }
    headers = [
        (b"content-type", b"application/json"),
        (b"x-telegram-bot-api-secret-token", b"secret-123"),
    ]
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/telegram/webhook",
        "query_string": b"",
        "headers": headers,
    }

    async def _receive() -> dict[str, object]:
        return {"type": "http.request", "body": json.dumps(body).encode("utf-8"), "more_body": False}

    return Request(scope, _receive)


@pytest.mark.asyncio
async def test_telegram_approval_repo_resolution_failure_log_is_sanitized(monkeypatch):
    logger = _LoggerStub()
    monkeypatch.setattr(telegram_support, "logger", logger)

    async def _raise_repo() -> object:
        raise RuntimeError("telegram approvals repo exploded at /private/telegram-approvals.db")

    async def _unused_service() -> object:
        raise AssertionError("approval service should not be resolved")

    response = await telegram_support._handle_telegram_approval_callback_query(
        scope=TelegramScope(scope_type="group", scope_id=88),
        payload=_approval_callback_payload("tgapp1.repo-token"),
        runtime_repo=_FakeTelegramRuntimeRepo(),
        get_telegram_approvals_repo=_raise_repo,
        get_approval_service=_unused_service,
    )

    assert response.status_code == 503
    assert json.loads(response.body)["error"] == "approval_service_unavailable"
    _assert_sanitized_error_log(logger, ["Failed to resolve Telegram approvals repo"])


@pytest.mark.asyncio
async def test_telegram_approval_service_resolution_failure_log_is_sanitized(monkeypatch):
    logger = _LoggerStub()
    monkeypatch.setattr(telegram_support, "logger", logger)
    repo = _FakeTelegramApprovalsRepo()
    runtime_repo = _FakeTelegramRuntimeRepo()
    await runtime_repo.upsert_actor_link(
        scope_type="group",
        scope_id=88,
        telegram_user_id=202,
        auth_user_id=202,
    )
    callback_data = await build_telegram_approval_callback_data(
        approval_policy_id=17,
        context_key="user:202|group:88|persona:researcher",
        conversation_id="conv-1",
        tool_name="Bash",
        tool_args={"command": "git status"},
        scope=TelegramScope(scope_type="group", scope_id=88),
        initiating_auth_user_id=202,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        repo=repo,
    )

    async def _get_repo() -> _FakeTelegramApprovalsRepo:
        return repo

    async def _raise_service() -> object:
        raise RuntimeError("telegram approval service exploded at /private/telegram-approvals.db")

    response = await telegram_support._handle_telegram_approval_callback_query(
        scope=TelegramScope(scope_type="group", scope_id=88),
        payload=_approval_callback_payload(callback_data),
        runtime_repo=runtime_repo,
        get_telegram_approvals_repo=_get_repo,
        get_approval_service=_raise_service,
    )

    assert response.status_code == 503
    assert json.loads(response.body)["error"] == "approval_service_unavailable"
    _assert_sanitized_error_log(logger, ["Failed to resolve Telegram approval service"])


@pytest.mark.asyncio
async def test_telegram_approval_decision_failure_log_is_sanitized(monkeypatch):
    class _FailingApprovalService:
        async def record_decision(self, **_kwargs: object) -> dict[str, object]:
            raise RuntimeError("telegram approval decision exploded at /private/telegram-approvals.db")

    logger = _LoggerStub()
    monkeypatch.setattr(telegram_support, "logger", logger)
    repo = _FakeTelegramApprovalsRepo()
    runtime_repo = _FakeTelegramRuntimeRepo()
    await runtime_repo.upsert_actor_link(
        scope_type="group",
        scope_id=88,
        telegram_user_id=202,
        auth_user_id=202,
    )
    callback_data = await build_telegram_approval_callback_data(
        approval_policy_id=17,
        context_key="user:202|group:88|persona:researcher",
        conversation_id="conv-1",
        tool_name="Bash",
        tool_args={"command": "git status"},
        scope=TelegramScope(scope_type="group", scope_id=88),
        initiating_auth_user_id=202,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        repo=repo,
    )

    async def _get_repo() -> _FakeTelegramApprovalsRepo:
        return repo

    async def _get_service() -> _FailingApprovalService:
        return _FailingApprovalService()

    response = await telegram_support._handle_telegram_approval_callback_query(
        scope=TelegramScope(scope_type="group", scope_id=88),
        payload=_approval_callback_payload(callback_data),
        runtime_repo=runtime_repo,
        get_telegram_approvals_repo=_get_repo,
        get_approval_service=_get_service,
    )

    assert response.status_code == 503
    assert json.loads(response.body)["error"] == "approval_service_unavailable"
    _assert_sanitized_error_log(logger, ["Failed to persist Telegram approval decision"])


@pytest.mark.asyncio
async def test_telegram_callback_rejects_approval_from_non_initiating_linked_user(monkeypatch):
    repo = _FakeTelegramApprovalsRepo()
    runtime_repo = _FakeTelegramRuntimeRepo()
    await runtime_repo.upsert_actor_link(
        scope_type="group",
        scope_id=88,
        telegram_user_id=303,
        auth_user_id=303,
    )

    async def _fake_get_org_secret_repo() -> object:
        return object()

    async def _fake_resolve_webhook_scope_from_secret(*, repo: object, webhook_secret: str):
        return TelegramWebhookContext(scope=TelegramScope(scope_type="group", scope_id=88), bot_username="example_bot")

    service = _FakeApprovalService()

    async def _fake_get_approval_service() -> _FakeApprovalService:
        return service

    async def _fake_get_telegram_approvals_repo() -> _FakeTelegramApprovalsRepo:
        return repo

    async def _fake_get_telegram_runtime_repo() -> _FakeTelegramRuntimeRepo:
        return runtime_repo

    monkeypatch.setattr(telegram_support, "_resolve_webhook_scope_from_secret", _fake_resolve_webhook_scope_from_secret)

    callback_data = await build_telegram_approval_callback_data(
        approval_policy_id=17,
        context_key="user:202|group:88|persona:researcher",
        conversation_id="conv-1",
        tool_name="Bash",
        tool_args={"command": "git status"},
        scope=TelegramScope(scope_type="group", scope_id=88),
        initiating_auth_user_id=202,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        repo=repo,
    )
    request = _build_request(update_id=9001, telegram_user_id=303, callback_data=callback_data)

    response = await telegram_webhook_impl(
        request=request,
        get_org_secret_repo=_fake_get_org_secret_repo,
        get_telegram_approvals_repo=_fake_get_telegram_approvals_repo,
        get_telegram_runtime_repo=_fake_get_telegram_runtime_repo,
        get_approval_service=_fake_get_approval_service,
    )

    assert response.status_code == 403
    assert json.loads(response.body)["error"] == "approval_not_authorized"
    assert service.recorded == []


@pytest.mark.asyncio
async def test_telegram_callback_approves_initiating_linked_user_and_is_single_use(monkeypatch):
    repo = _FakeTelegramApprovalsRepo()
    runtime_repo = _FakeTelegramRuntimeRepo()
    await runtime_repo.upsert_actor_link(
        scope_type="group",
        scope_id=88,
        telegram_user_id=202,
        auth_user_id=202,
    )

    async def _fake_get_org_secret_repo() -> object:
        return object()

    async def _fake_resolve_webhook_scope_from_secret(*, repo: object, webhook_secret: str):
        return TelegramWebhookContext(scope=TelegramScope(scope_type="group", scope_id=88), bot_username="example_bot")

    service = _FakeApprovalService()

    async def _fake_get_approval_service() -> _FakeApprovalService:
        return service

    async def _fake_get_telegram_approvals_repo() -> _FakeTelegramApprovalsRepo:
        return repo

    async def _fake_get_telegram_runtime_repo() -> _FakeTelegramRuntimeRepo:
        return runtime_repo

    monkeypatch.setattr(telegram_support, "_resolve_webhook_scope_from_secret", _fake_resolve_webhook_scope_from_secret)

    callback_data = await build_telegram_approval_callback_data(
        approval_policy_id=17,
        context_key="user:202|group:88|persona:researcher",
        conversation_id="conv-1",
        tool_name="Bash",
        tool_args={"command": "git status"},
        scope=TelegramScope(scope_type="group", scope_id=88),
        initiating_auth_user_id=202,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        repo=repo,
    )

    first = await telegram_webhook_impl(
        request=_build_request(update_id=9003, telegram_user_id=202, callback_data=callback_data),
        get_org_secret_repo=_fake_get_org_secret_repo,
        get_telegram_approvals_repo=_fake_get_telegram_approvals_repo,
        get_telegram_runtime_repo=_fake_get_telegram_runtime_repo,
        get_approval_service=_fake_get_approval_service,
    )
    second = await telegram_webhook_impl(
        request=_build_request(update_id=9004, telegram_user_id=202, callback_data=callback_data),
        get_org_secret_repo=_fake_get_org_secret_repo,
        get_telegram_approvals_repo=_fake_get_telegram_approvals_repo,
        get_telegram_runtime_repo=_fake_get_telegram_runtime_repo,
        get_approval_service=_fake_get_approval_service,
    )

    assert first.status_code == 200
    assert json.loads(first.body)["status"] == "approved"
    assert len(service.recorded) == 1
    assert service.recorded[0].decision == "approved"
    assert service.recorded[0].consume_on_match is True
    assert service.recorded[0].actor_id == 202
    assert service.recorded[0].expires_at is None
    assert service.recorded[0].scope_key == _scope_key_for_tool_call("Bash", {"command": "git status"})
    assert second.status_code == 409
    assert json.loads(second.body)["error"] == "approval_unavailable"


@pytest.mark.asyncio
async def test_telegram_callback_rejects_scope_mismatch_as_unavailable(monkeypatch):
    repo = _FakeTelegramApprovalsRepo()
    runtime_repo = _FakeTelegramRuntimeRepo()
    await runtime_repo.upsert_actor_link(
        scope_type="group",
        scope_id=88,
        telegram_user_id=202,
        auth_user_id=202,
    )

    async def _fake_get_org_secret_repo() -> object:
        return object()

    async def _fake_resolve_webhook_scope_from_secret(*, repo: object, webhook_secret: str):
        return TelegramWebhookContext(scope=TelegramScope(scope_type="group", scope_id=99), bot_username="example_bot")

    service = _FakeApprovalService()

    async def _fake_get_approval_service() -> _FakeApprovalService:
        return service

    async def _fake_get_telegram_approvals_repo() -> _FakeTelegramApprovalsRepo:
        return repo

    async def _fake_get_telegram_runtime_repo() -> _FakeTelegramRuntimeRepo:
        return runtime_repo

    monkeypatch.setattr(telegram_support, "_resolve_webhook_scope_from_secret", _fake_resolve_webhook_scope_from_secret)

    callback_data = await build_telegram_approval_callback_data(
        approval_policy_id=17,
        context_key="user:202|group:88|persona:researcher",
        conversation_id="conv-1",
        tool_name="Bash",
        tool_args={"command": "git status --porcelain"},
        scope=TelegramScope(scope_type="group", scope_id=88),
        initiating_auth_user_id=202,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        repo=repo,
    )
    request = _build_request(update_id=9002, telegram_user_id=202, callback_data=callback_data)

    response = await telegram_webhook_impl(
        request=request,
        get_org_secret_repo=_fake_get_org_secret_repo,
        get_telegram_approvals_repo=_fake_get_telegram_approvals_repo,
        get_telegram_runtime_repo=_fake_get_telegram_runtime_repo,
        get_approval_service=_fake_get_approval_service,
    )

    assert response.status_code == 409
    assert json.loads(response.body)["error"] == "approval_unavailable"
    assert service.recorded == []


@pytest.mark.asyncio
async def test_telegram_callback_rejects_expired_approval(monkeypatch):
    repo = _FakeTelegramApprovalsRepo()
    runtime_repo = _FakeTelegramRuntimeRepo()
    await runtime_repo.upsert_actor_link(
        scope_type="group",
        scope_id=88,
        telegram_user_id=202,
        auth_user_id=202,
    )

    async def _fake_get_org_secret_repo() -> object:
        return object()

    async def _fake_resolve_webhook_scope_from_secret(*, repo: object, webhook_secret: str):
        return TelegramWebhookContext(scope=TelegramScope(scope_type="group", scope_id=88), bot_username="example_bot")

    service = _FakeApprovalService()

    async def _fake_get_approval_service() -> _FakeApprovalService:
        return service

    async def _fake_get_telegram_approvals_repo() -> _FakeTelegramApprovalsRepo:
        return repo

    async def _fake_get_telegram_runtime_repo() -> _FakeTelegramRuntimeRepo:
        return runtime_repo

    monkeypatch.setattr(telegram_support, "_resolve_webhook_scope_from_secret", _fake_resolve_webhook_scope_from_secret)

    callback_data = await build_telegram_approval_callback_data(
        approval_policy_id=17,
        context_key="user:202|group:88|persona:researcher",
        conversation_id="conv-1",
        tool_name="Bash",
        tool_args={"command": "git status"},
        scope=TelegramScope(scope_type="group", scope_id=88),
        initiating_auth_user_id=202,
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        repo=repo,
    )

    response = await telegram_webhook_impl(
        request=_build_request(update_id=9005, telegram_user_id=202, callback_data=callback_data),
        get_org_secret_repo=_fake_get_org_secret_repo,
        get_telegram_approvals_repo=_fake_get_telegram_approvals_repo,
        get_telegram_runtime_repo=_fake_get_telegram_runtime_repo,
        get_approval_service=_fake_get_approval_service,
    )

    assert response.status_code == 409
    assert json.loads(response.body)["error"] == "approval_unavailable"
    assert service.recorded == []
