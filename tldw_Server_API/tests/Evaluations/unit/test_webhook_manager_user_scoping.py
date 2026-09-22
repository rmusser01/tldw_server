"""Regression guard for TASK-13296.

`WebhookManager._get_webhooks` ended with a "final safety" fallback that, when the
per-user lookup returned nothing and TEST_MODE was set, selected **every** active
webhook registration with no `user_id` predicate — including each row's `secret` —
and returned them as if they belonged to the caller.

The gate was `core/testing.py:is_test_mode`, which reads an environment variable
rather than checking the pytest runtime. `TEST_MODE=1` present in a shared, staging
or misconfigured deployment therefore leaked other users' webhook URLs and signing
secrets, and delivered the caller's evaluation payloads to those endpoints.

The fallback was also redundant: the block immediately above it runs the same query
correctly scoped by `user_id`, so a user with no active webhooks legitimately has
none.
"""

from contextlib import contextmanager
from typing import Any

import pytest


from tldw_Server_API.app.core.Evaluations.webhook_manager import (
    WebhookEvent,
    WebhookManager,
)

# Suite marker: these are fast, isolated regression guards.
pytestmark = pytest.mark.unit

# Rows belonging to two different users. Only user-a may ever be returned to user-a.
_ROWS = [
    {
        "id": "wh-a",
        "user_id": "user-a",
        "url": "https://a.example/hook",
        "secret": "secret-a",
        "retry_count": 1,
        "timeout_seconds": 5,
        "active": 1,
        "events": '["evaluation.completed"]',
    },
    {
        "id": "wh-b",
        "user_id": "user-b",
        "url": "https://b.example/hook",
        "secret": "secret-b-MUST-NOT-LEAK",
        "retry_count": 1,
        "timeout_seconds": 5,
        "active": 1,
        "events": '["evaluation.completed"]',
    },
]


class _ScopedAdapter:
    """Adapter that honours a user_id predicate when the query carries one."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.queries: list[str] = []

    @contextmanager
    def transaction(self):
        yield self

    def fetch_all(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        self.queries.append(sql)
        if "user_id = ?" in sql:
            user_id = params[0]
            return [r for r in self.rows if r["user_id"] == user_id]
        # No user predicate: the database happily returns every active row.
        return list(self.rows)


def _manager(adapter: _ScopedAdapter) -> WebhookManager:
    mgr = WebhookManager.__new__(WebhookManager)
    mgr.db_adapter = adapter
    return mgr


@pytest.fixture(autouse=True)
def _force_test_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """is_test_mode() reads an env var -- this is the deployment hazard."""
    monkeypatch.setenv("TEST_MODE", "1")


async def test_user_never_receives_another_users_webhook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # user-c owns no webhooks at all, so every user-scoped lookup returns empty
    # and the unscoped fallback is what fires.
    adapter = _ScopedAdapter(_ROWS)
    mgr = _manager(adapter)
    monkeypatch.setattr(mgr, "_is_postgres_backend", lambda: False, raising=False)

    webhooks = await mgr._get_webhooks("user-c", WebhookEvent.EVALUATION_COMPLETED)

    returned_ids = {w["id"] for w in webhooks}
    assert returned_ids == set(), (
        f"user-c owns no webhooks but received {sorted(returned_ids)} -- "
        "an unscoped query leaked other users' registrations"
    )


async def test_other_users_secret_is_never_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _ScopedAdapter(_ROWS)
    mgr = _manager(adapter)
    monkeypatch.setattr(mgr, "_is_postgres_backend", lambda: False, raising=False)

    webhooks = await mgr._get_webhooks("user-c", WebhookEvent.EVALUATION_COMPLETED)

    secrets = {w["secret"] for w in webhooks}
    assert "secret-b-MUST-NOT-LEAK" not in secrets, (
        "another user's webhook signing secret was returned to this caller"
    )


async def test_every_lookup_is_scoped_by_user_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No query issued on this path may omit the user predicate."""
    adapter = _ScopedAdapter(_ROWS)
    mgr = _manager(adapter)
    monkeypatch.setattr(mgr, "_is_postgres_backend", lambda: False, raising=False)

    await mgr._get_webhooks("user-c", WebhookEvent.EVALUATION_COMPLETED)

    unscoped = [q for q in adapter.queries if "user_id = ?" not in q]
    assert not unscoped, (
        "a webhook lookup ran without a user_id predicate:\n"
        + "\n---\n".join(unscoped)
    )


async def test_owner_still_receives_their_own_webhook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: removing the fallback must not break legitimate delivery."""
    adapter = _ScopedAdapter(_ROWS)
    mgr = _manager(adapter)
    monkeypatch.setattr(mgr, "_is_postgres_backend", lambda: False, raising=False)

    webhooks = await mgr._get_webhooks("user-a", WebhookEvent.EVALUATION_COMPLETED)

    assert {w["id"] for w in webhooks} == {"wh-a"}
    assert webhooks[0]["secret"] == "secret-a"


@pytest.mark.parametrize("test_mode", ["1", None], ids=["test_mode_on", "test_mode_off"])
async def test_no_unscoped_query_in_either_mode(
    monkeypatch: pytest.MonkeyPatch,
    test_mode: str | None,
) -> None:
    """The unscoped fallback must stay unreachable in both modes.

    When TEST_MODE is set, an unconditional early return in the first branch means
    the later fallbacks are never reached. When it is unset, their own
    `_is_test_mode()` guard is false. So the unscoped SELECT is dead either way --
    which is why the leak is latent rather than live, and why deleting the block is
    safe. This test is the guard against it being reactivated by a future refactor
    of that early return.
    """
    if test_mode is None:
        monkeypatch.delenv("TEST_MODE", raising=False)
    else:
        monkeypatch.setenv("TEST_MODE", test_mode)

    adapter = _ScopedAdapter(_ROWS)
    mgr = _manager(adapter)
    monkeypatch.setattr(mgr, "_is_postgres_backend", lambda: False, raising=False)

    await mgr._get_webhooks("user-c", WebhookEvent.EVALUATION_COMPLETED)

    unscoped = [q for q in adapter.queries if "user_id = ?" not in q]
    assert not unscoped, (
        f"TEST_MODE={test_mode!r}: an unscoped webhook query ran:\n"
        + "\n---\n".join(unscoped)
    )
