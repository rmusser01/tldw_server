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

    def __init__(self) -> None:
        self.seen_user_ids: list[str | None] = []

    def list_chunking_templates(self, *args: object, **kwargs: object) -> list[dict]:
        """Record the scope the endpoint passed and return nothing."""
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
    """With no parameter the listing must be the caller's, not everyone's."""
    db = _RecordingDB()

    await chunking_templates.list_templates(
        include_builtin=True, include_custom=True, tags=None,
        user_id=None, current_user=caller, db=db, response=None,
    )

    assert db.seen_user_ids == ["1"], "must scope to the caller, not to everyone"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_listing_another_users_templates_is_refused(caller):
    """A query parameter naming someone else is refused, not silently ignored."""
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
    """Naming your own id stays valid; only a foreign id is refused."""
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


class _RoundTripDB:
    """Stores what create persists and filters listings the way the repo does."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def create_chunking_template(
        self, *, name: str, template_json: str, description: str | None,
        is_builtin: bool, tags: list[str] | None, user_id: str | None,
    ) -> dict:
        """Persist a row exactly as handed over, so the owner can be asserted."""
        row = _fallback_record(name, user_id)
        row.update({"template_json": template_json, "description": description,
                    "is_builtin": is_builtin, "tags": tags})
        self.rows.append(row)
        return row

    def get_chunking_template(self, *, name: str) -> dict:
        """Return the stored row by name."""
        return next(r for r in self.rows if r["name"] == name)

    def list_chunking_templates(
        self, *, include_builtin: bool, include_custom: bool,
        tags: list[str] | None, user_id: str | None, include_deleted: bool,
    ) -> list[dict]:
        """Filter the way the repository does: owner's rows plus built-ins."""
        return [
            r for r in self.rows
            if (user_id is None or r["user_id"] == user_id or r["is_builtin"])
        ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_created_template_is_listable_when_the_body_omits_user_id(caller):
    """A client that omits user_id must still see what it just created.

    Listing is scoped to the owner, so a create that persisted a null owner
    would store a row its own author could never list again.
    """
    db = _RoundTripDB()

    await chunking_templates.create_template(
        template_data=chunking_templates.ChunkingTemplateCreate(
            name="mine",
            template=chunking_templates.TemplateConfig(
                chunking={"method": "words", "config": {"max_size": 100}},
            ),
        ),
        current_user=caller, db=db, response=None,
    )

    assert db.rows[0]["user_id"] == "1", "create must stamp the caller as owner"

    listed = await chunking_templates.list_templates(
        include_builtin=True, include_custom=True, tags=None,
        user_id=None, current_user=caller, db=db, response=None,
    )
    assert {t.name for t in listed.templates} == {"mine"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_creating_a_template_owned_by_someone_else_is_refused(caller):
    """`user_id` in the body is request input, same as the query parameter."""
    db = _RoundTripDB()

    with pytest.raises(HTTPException) as exc_info:
        await chunking_templates.create_template(
            template_data=chunking_templates.ChunkingTemplateCreate(
                name="theirs",
                template=chunking_templates.TemplateConfig(
                    chunking={"method": "words", "config": {"max_size": 100}},
                ),
                user_id="2",
            ),
            current_user=caller, db=db, response=None,
        )

    assert exc_info.value.status_code == 403
    assert db.rows == []
