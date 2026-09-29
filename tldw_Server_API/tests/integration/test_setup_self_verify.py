from datetime import datetime, timezone

import pytest

import tldw_Server_API.app.api.v1.endpoints.setup as setup_endpoint
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal

# Since 5f31630280 users writes go through the versioned profile gateway: it
# reads the profile-version candidates, runs the schema-qualified UPDATE and
# touches the anchor on the same connection. The fakes answer those reads.
_ANCHOR = datetime(2026, 2, 9, 11, 0, tzinfo=timezone.utc)


class _SQLiteCursor:
    rowcount = 1

    def __init__(self, user_id) -> None:
        self._user_id = user_id

    async def fetchall(self):
        return [("user", self._user_id, _ANCHOR)]


class _FakeSQLiteConn:
    def __init__(self) -> None:
        self.calls = []
        self.commits = 0

    async def execute(self, query, params):
        self.calls.append((getattr(query, "text", query), params))
        return _SQLiteCursor(params[-1])

    async def commit(self):
        self.commits += 1


class _FakeAsyncPGConn:
    def __init__(self) -> None:
        self.calls = []

    async def fetch(self, query, *params):
        return [{"source_tag": "user", "source_id": params[0], "candidate_value": _ANCHOR}]

    async def execute(self, query, *params):
        self.calls.append((getattr(query, "text", query), params))
        return "UPDATE 1"


def _verify_update(calls):
    matches = [call for call in calls if "SET is_verified" in call[0]]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.asyncio
async def test_setup_self_verify_updates_sqlite(monkeypatch):
    """SQLite-compatible execute receives normalized placeholders and commits."""
    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "get_status_snapshot",
        lambda: {"needs_setup": True},
    )

    fake_db = _FakeSQLiteConn()

    result = await setup_endpoint.setup_self_verify(
        principal=AuthPrincipal(kind="user", user_id=9, username="setup-user"),
        db=fake_db,
        _guard=None,
    )

    assert result["success"] is True
    query, params = _verify_update(fake_db.calls)
    assert query == "UPDATE main.users SET is_verified = ?, updated_at = ? WHERE id = ?"
    assert params[0] is True
    assert params[2] == 9
    assert fake_db.commits == 1


@pytest.mark.asyncio
async def test_setup_self_verify_updates_asyncpg(monkeypatch):
    """AsyncPG-style execute preserves `$n` placeholders."""
    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "get_status_snapshot",
        lambda: {"needs_setup": True},
    )

    fake_db = _FakeAsyncPGConn()

    result = await setup_endpoint.setup_self_verify(
        principal=AuthPrincipal(kind="user", user_id=3, username="setup-user"),
        db=fake_db,
        _guard=None,
    )

    assert result["success"] is True
    query, params = _verify_update(fake_db.calls)
    assert query == "UPDATE public.users SET is_verified = $1, updated_at = $2 WHERE id = $3"
    assert params[0] is True
    assert params[2] == 3
