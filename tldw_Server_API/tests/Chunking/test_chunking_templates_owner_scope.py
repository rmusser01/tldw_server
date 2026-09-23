"""Listing chunking templates must be scoped to the authenticated caller.

`user_id` arrives from the query string. It used to be passed straight through
to the repository, so `?user_id=<someone else>` returned their templates and
omitting it returned everyone's. The in-memory fallback store is a
process-global dict keyed by user id, so that second path held on SQLite too,
not only on a shared PostgreSQL content backend.
"""

from uuid import NAMESPACE_URL, uuid5

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import chunking_templates
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User


class _RecordingDB:
    """Captures the user_id the endpoint actually scopes the query with."""

    def __init__(self):
        self.seen_user_ids = []

    def list_chunking_templates(self, *args, **kwargs):
        self.seen_user_ids.append(kwargs.get("user_id"))
        return []


def _fallback_record(name: str, user_id: str) -> dict:
    """A record shaped the way the fallback store holds them."""
    return {
        "id": 1, "uuid": str(uuid5(NAMESPACE_URL, name)), "name": name, "description": None,
        "template_json": "{}", "is_builtin": False, "tags": [],
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "version": 1, "user_id": user_id,
    }


@pytest.fixture()
def caller():
    return User(id=1, username="owner", email=None, is_active=True)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_listing_scopes_to_the_caller_when_no_user_id_is_given(caller):
    db = _RecordingDB()

    await chunking_templates.list_templates(
        include_builtin=True, include_custom=True, tags=None,
        user_id=None, current_user=caller, db=db, response=None,
    )

    assert db.seen_user_ids == ["1"], "must scope to the caller, not to everyone"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_listing_another_users_templates_is_refused(caller):
    db = _RecordingDB()

    with pytest.raises(HTTPException) as exc_info:
        await chunking_templates.list_templates(
            include_builtin=True, include_custom=True, tags=None,
            user_id="2", current_user=caller, db=db, response=None,
        )

    assert exc_info.value.status_code == 403
    assert db.seen_user_ids == [], "the query must not run at all"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_caller_may_name_their_own_id(caller):
    db = _RecordingDB()

    await chunking_templates.list_templates(
        include_builtin=True, include_custom=True, tags=None,
        user_id="1", current_user=caller, db=db, response=None,
    )

    assert db.seen_user_ids == ["1"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fallback_store_does_not_aggregate_every_user(caller, monkeypatch):
    """The fallback path is a process-global dict, so it leaks on SQLite too."""
    monkeypatch.setattr(
        chunking_templates,
        "_FALLBACK_TEMPLATES",
        {
            "1": {"mine": _fallback_record("mine", "1")},
            "2": {"theirs": _fallback_record("theirs", "2")},
        },
        raising=True,
    )

    class _NoNativeSupportDB:
        pass

    result = await chunking_templates.list_templates(
        include_builtin=True, include_custom=True, tags=None,
        user_id=None, current_user=caller, db=_NoNativeSupportDB(), response=None,
    )

    names = {template.name for template in result.templates}
    assert names == {"mine"}, f"another user's templates leaked: {names}"
