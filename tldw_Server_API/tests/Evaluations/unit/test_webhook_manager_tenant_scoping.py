"""Regression (TASK-13296): webhook lookup must stay per-user and per-event under TEST_MODE.

``_get_webhooks`` relaxed its scoping whenever the ``TEST_MODE`` env var was set, which a
misconfigured deployment can carry. The relaxation must not change delivery targets.
"""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Evaluations.webhook_manager import WebhookEvent, WebhookManager

pytestmark = pytest.mark.unit


@pytest.fixture
async def manager(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")
    mgr = WebhookManager(db_path=str(tmp_path / "webhooks.db"))
    await mgr.register_webhook(
        "user-a", "https://a.example.com/hook", [WebhookEvent.EVALUATION_COMPLETED],
        secret="secret-a", skip_validation=True,
    )
    await mgr.register_webhook(
        "user-b", "https://b.example.com/hook", [WebhookEvent.EVALUATION_COMPLETED],
        secret="secret-b", skip_validation=True,
    )
    # Simulate a deployed server with TEST_MODE leaked into its env (no pytest runtime).
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    return mgr


@pytest.mark.asyncio
async def test_user_without_webhooks_gets_none_of_other_users(manager) -> None:
    rows = await manager._get_webhooks("user-c", WebhookEvent.EVALUATION_COMPLETED)
    assert rows == []


@pytest.mark.asyncio
async def test_user_only_gets_own_webhook(manager) -> None:
    rows = await manager._get_webhooks("user-a", WebhookEvent.EVALUATION_COMPLETED)
    assert [r["secret"] for r in rows] == ["secret-a"]


@pytest.mark.asyncio
async def test_unsubscribed_event_is_not_delivered_under_test_mode_env(manager) -> None:
    rows = await manager._get_webhooks("user-a", WebhookEvent.EVALUATION_FAILED)
    assert rows == []


@pytest.mark.asyncio
async def test_delivery_keeps_ssrf_check_under_test_mode_env(manager, monkeypatch) -> None:
    """TEST_MODE in a deployed env must not skip the DNS/SSRF delivery-target check."""
    from tldw_Server_API.app.core.Evaluations import webhook_manager as wm

    checked: list[str] = []

    async def reject(url: str):
        checked.append(url)
        raise ValueError("resolves to a private address")

    def must_not_deliver(*_args, **_kwargs):
        raise AssertionError("delivery attempted without SSRF validation")

    # pytest re-sets PYTEST_CURRENT_TEST for the call phase; clear it to look deployed.
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(wm.webhook_validator, "resolve_safe_delivery_target_async", reject)
    monkeypatch.setattr(manager, "_create_delivery_record", must_not_deliver)
    webhook = (await manager._get_webhooks("user-a", WebhookEvent.EVALUATION_COMPLETED))[0]
    payload = wm.WebhookPayload(event="evaluation.completed", evaluation_id="e1", timestamp="t", data={})

    await manager._deliver_webhook(webhook, payload)

    assert checked == ["https://a.example.com/hook"]
