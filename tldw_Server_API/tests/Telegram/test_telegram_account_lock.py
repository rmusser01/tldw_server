"""A locked linked user cannot ask or approve through Telegram.

The existing Telegram fixtures have no AuthNZ database, so ``is_user_security_locked`` is replaced in
``telegram_support`` with a plain function that answers for the locked id only.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.api.v1.utils.chatops import telegram_support
from tldw_Server_API.app.api.v1.utils.chatops.telegram_support import (
    TelegramScope,
    TelegramWebhookContext,
    _register_telegram_actor_link_for_tests,
    build_telegram_approval_callback_data,
    telegram_webhook_impl,
)
from tldw_Server_API.tests.Telegram.test_telegram_approvals import (
    _build_request,
    _FakeApprovalService,
    _FakeTelegramApprovalsRepo,
    _FakeTelegramRuntimeRepo,
)
from tldw_Server_API.tests.Telegram.test_telegram_jobs_and_delivery import (  # noqa: F401  (fixtures)
    _post_ask,
    _seed_telegram_bot,
    client,
    job_manager_override,
    principal_override,
)

LOCKED_ID = 901


def _lock_only(monkeypatch, locked_id: int) -> None:
    async def _is_locked(user_id: int) -> bool:
        return user_id == locked_id

    monkeypatch.setattr(telegram_support, "is_user_security_locked", _is_locked)


def _link_and_seed(client, principal_override, *, auth_user_id: int) -> None:
    _seed_telegram_bot(
        client, principal_override, scope_type="team", scope_id=22, bot_token="123:abc", webhook_secret="secret-123"
    )
    _register_telegram_actor_link_for_tests(
        scope_type="team", scope_id=22, telegram_user_id=77, auth_user_id=auth_user_id
    )


def test_telegram_ask_from_a_locked_linked_user_queues_nothing(
    client, principal_override, job_manager_override, monkeypatch
) -> None:
    _lock_only(monkeypatch, LOCKED_ID)
    _link_and_seed(client, principal_override, auth_user_id=LOCKED_ID)

    response = _post_ask(client, update_id=13001, secret="secret-123", telegram_user_id=77)

    assert response.status_code == 403
    assert response.json() == {"ok": False, "error": "account_locked"}
    assert job_manager_override.list_jobs(owner_user_id=str(LOCKED_ID)) == []


def test_telegram_ask_from_an_unlocked_linked_user_still_queues(
    client, principal_override, job_manager_override, monkeypatch
) -> None:
    _lock_only(monkeypatch, LOCKED_ID)
    _link_and_seed(client, principal_override, auth_user_id=902)

    response = _post_ask(client, update_id=13002, secret="secret-123", telegram_user_id=77)

    assert response.json()["status"] == "queued"
    assert len(job_manager_override.list_jobs(owner_user_id="902")) == 1


async def _approve(monkeypatch, *, user_id: int):
    """Run an approval callback for ``user_id``; returns ``(response, decisions_recorded)``."""
    repo = _FakeTelegramApprovalsRepo()
    runtime_repo = _FakeTelegramRuntimeRepo()
    await runtime_repo.upsert_actor_link(scope_type="group", scope_id=88, telegram_user_id=202, auth_user_id=user_id)
    service = _FakeApprovalService()

    async def _org_secret_repo() -> object:
        return object()

    async def _scope(*, repo: object, webhook_secret: str):
        return TelegramWebhookContext(scope=TelegramScope(scope_type="group", scope_id=88), bot_username="example_bot")

    async def _service() -> _FakeApprovalService:
        return service

    async def _approvals() -> _FakeTelegramApprovalsRepo:
        return repo

    async def _runtime() -> _FakeTelegramRuntimeRepo:
        return runtime_repo

    monkeypatch.setattr(telegram_support, "_resolve_webhook_scope_from_secret", _scope)
    callback_data = await build_telegram_approval_callback_data(
        approval_policy_id=17,
        context_key=f"user:{user_id}|group:88|persona:researcher",
        conversation_id="conv-1",
        tool_name="Bash",
        tool_args={"command": "git status"},
        scope=TelegramScope(scope_type="group", scope_id=88),
        initiating_auth_user_id=user_id,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        repo=repo,
    )
    response = await telegram_webhook_impl(
        request=_build_request(update_id=9100 + user_id, telegram_user_id=202, callback_data=callback_data),
        get_org_secret_repo=_org_secret_repo,
        get_telegram_approvals_repo=_approvals,
        get_telegram_runtime_repo=_runtime,
        get_approval_service=_service,
    )
    return response, service.recorded


@pytest.mark.asyncio
async def test_telegram_approval_from_a_locked_linked_user_records_nothing(monkeypatch) -> None:
    _lock_only(monkeypatch, 202)

    response, recorded = await _approve(monkeypatch, user_id=202)

    assert response.status_code == 403
    assert json.loads(response.body) == {"ok": False, "error": "account_locked"}
    assert recorded == []


@pytest.mark.asyncio
async def test_telegram_approval_from_an_unlocked_linked_user_is_recorded(monkeypatch) -> None:
    _lock_only(monkeypatch, 202)

    response, recorded = await _approve(monkeypatch, user_id=203)

    assert json.loads(response.body)["status"] == "approved"
    assert [row.actor_id for row in recorded] == [203]
