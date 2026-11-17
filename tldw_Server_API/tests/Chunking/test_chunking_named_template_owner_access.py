"""Named-template reads and mutations obey the authenticated owner in shared stores."""

import copy
import json
from contextlib import nullcontext
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import chunking_templates as endpoint
from tldw_Server_API.app.api.v1.schemas.chunking_templates_schemas import ApplyTemplateRequest, ChunkingTemplateUpdate
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.Chunking.templates import TemplateClassifier
from tldw_Server_API.app.core.DB_Management.media_db.errors import InputError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.media_db.runtime import chunk_template_ops
from tldw_Server_API.app.core.Ingestion_Media_Processing.chunking_options import apply_chunking_template_if_any

pytestmark = pytest.mark.integration
NAME = "s06-private-owner-template"
CONFIG = {
    "preprocessing": [],
    "chunking": {"method": "words", "config": {"max_size": 2, "overlap": 0}},
    "postprocessing": [],
}


@pytest.fixture(params=["sqlite", "postgresql", "fallback"])
def shared_template_store(request, tmp_path, monkeypatch):
    monkeypatch.setattr(endpoint, "_FALLBACK_TEMPLATES", {})
    if request.param == "fallback":
        monkeypatch.setenv("CHUNKING_TEMPLATES_FALLBACK_ENABLED", "1")
        record = {
            "id": 1,
            "uuid": str(uuid5(NAMESPACE_URL, NAME)),
            "name": NAME,
            "description": "Private owner template",
            "template_json": json.dumps(CONFIG),
            "is_builtin": False,
            "tags": ["private"],
            "user_id": "1",
            "version": 1,
            "created_at": "2026-11-02T00:00:00+00:00",
            "updated_at": "2026-11-02T00:00:00+00:00",
        }
        endpoint._FALLBACK_TEMPLATES["1"] = {NAME: record}
        yield SimpleNamespace(), lambda: copy.deepcopy(endpoint._FALLBACK_TEMPLATES.get("1", {}).get(NAME))
        return
    backend = request.getfixturevalue("pg_restricted_backend") if request.param == "postgresql" else None
    db = MediaDatabase(tmp_path / "shared-templates.db", client_id="template-fixture", backend=backend)
    try:
        with db.transaction():
            db.create_chunking_template(
                name=NAME,
                template_json=json.dumps(CONFIG),
                description="Private owner template",
                is_builtin=False,
                tags=["private"],
                user_id="1",
            )
        yield db, lambda: copy.deepcopy(db.get_chunking_template(name=NAME))
    finally:
        db.release_context_connection()
        db.close_connection()


@pytest.mark.parametrize("action", ["read", "update", "soft-delete", "hard-delete"])
async def test_foreign_named_template_is_unavailable_and_unchanged(shared_template_store, action):
    db, stored = shared_template_store
    bob = User(id=2, username="template-bob", email=None, is_active=True)
    boundary = db.transaction() if isinstance(db, MediaDatabase) else nullcontext()
    with boundary:
        before = stored()
        with pytest.raises(HTTPException) as refused:
            if action == "read":
                await endpoint.get_template(NAME, current_user=bob, db=db)
            elif action == "update":
                await endpoint.update_template(
                    NAME, ChunkingTemplateUpdate(description="Forbidden edit"), current_user=bob, db=db, response=None
                )
            else:
                await endpoint.delete_template(
                    NAME, hard_delete=action == "hard-delete", current_user=bob, db=db, response=None
                )
        assert refused.value.status_code == 404  # nosec B101 # Foreign and missing custom names share the refusal.
        assert stored() == before  # nosec B101 # A refused read or mutation preserves the complete private row.


@pytest.mark.parametrize("shared_template_store", ["sqlite", "postgresql"], indirect=True)
@pytest.mark.parametrize("identifier", ["name", "template_id", "uuid"])
def test_scoped_lookup_hides_foreign_rows_for_every_identifier(shared_template_store, identifier):
    db, stored = shared_template_store
    with db.transaction():
        row = stored()
        key = row["id"] if identifier == "template_id" else row[identifier]
        assert db.get_chunking_template(**{identifier: key}, owner_user_id="1") == row  # nosec B101 -- Own named/id/UUID reads remain intact.
        assert db.get_chunking_template(**{identifier: key}, owner_user_id="2") is None  # nosec B101 -- All identifiers enforce the owner.


@pytest.mark.parametrize("shared_template_store", ["sqlite", "postgresql"], indirect=True)
def test_public_builtins_and_trusted_reads_remain_but_ownerless_customs_are_hidden(shared_template_store):
    db, stored = shared_template_store
    with db.transaction():
        db.create_chunking_template(name="public-catalog", template_json=json.dumps(CONFIG), is_builtin=True)
        db.create_chunking_template(name="ownerless-custom", template_json=json.dumps(CONFIG))
        assert db.get_chunking_template(name="public-catalog", owner_user_id="2")["is_builtin"]  # nosec B101 -- Built-ins remain public.
        assert db.get_chunking_template(name="ownerless-custom", owner_user_id="1") is None  # nosec B101 -- Null is not a public custom owner.
        assert db.get_chunking_template(name=NAME) == stored()  # nosec B101 -- Trusted initialization keeps its existing unscoped API.
        with pytest.raises(InputError):
            db.get_chunking_template(name=NAME, owner_user_id=" ")
        with pytest.raises(InputError):
            db.update_chunking_template(name="public-catalog", description="Forbidden", owner_user_id="2")
        with pytest.raises(InputError):
            db.delete_chunking_template(name="public-catalog", owner_user_id="2")


@pytest.mark.parametrize("shared_template_store", ["sqlite", "postgresql"], indirect=True)
@pytest.mark.parametrize("action", ["update", "soft-delete", "hard-delete"])
def test_mutation_checks_owner_even_after_a_stale_authorized_lookup(shared_template_store, monkeypatch, action):
    db, stored = shared_template_store
    with db.transaction():
        before = stored()
        # Model an ownership change after the preliminary read without weakening the real SQL write.
        with monkeypatch.context() as lookup:
            lookup.setattr(chunk_template_ops, "get_chunking_template", lambda *a, **k: copy.deepcopy(before))
            if action == "update":
                changed = db.update_chunking_template(name=NAME, description="Forbidden", owner_user_id="2")
            else:
                changed = db.delete_chunking_template(name=NAME, hard_delete=action == "hard-delete", owner_user_id="2")
        assert changed is False  # nosec B101 -- The write must authorize independently of the prior read.
        assert stored() == before  # nosec B101 -- Stale lookup cannot change/delete the real private row.


async def test_owner_can_read_update_and_delete_own_template(shared_template_store):
    db, _ = shared_template_store
    owner = User(id=1, username="template-owner", email=None, is_active=True)
    boundary = db.transaction() if isinstance(db, MediaDatabase) else nullcontext()
    with boundary:
        assert (await endpoint.get_template(NAME, current_user=owner, db=db)).user_id == "1"  # nosec B101 -- Authorized public read remains usable.
        result = await endpoint.update_template(
            NAME, ChunkingTemplateUpdate(description="Owner edit"), current_user=owner, db=db, response=None
        )
        assert result.description == "Owner edit"  # nosec B101 -- Authorized write/readback stays scoped.
        await endpoint.delete_template(NAME, hard_delete=True, current_user=owner, db=db, response=None)
        with pytest.raises(HTTPException) as refused:
            await endpoint.get_template(NAME, current_user=owner, db=db)
        assert refused.value.status_code == 404  # nosec B101 -- Authorized deletion completes.


async def test_foreign_template_cannot_be_applied(shared_template_store):
    db, stored = shared_template_store
    bob = User(id=2, username="template-bob", email=None, is_active=True)
    boundary = db.transaction() if isinstance(db, MediaDatabase) else nullcontext()
    with boundary:
        before = stored()
        with pytest.raises(HTTPException) as refused:
            await endpoint.apply_template(
                ApplyTemplateRequest(template_name=NAME, text="One two three four."),
                include_metadata=False,
                current_user=bob,
                db=db,
                response=None,
            )
        assert refused.value.status_code == 404  # nosec B101 -- Applying a private foreign template matches an unknown name.
        assert stored() == before  # nosec B101 -- Application refusal cannot mutate the source definition.


@pytest.mark.parametrize("shared_template_store", ["sqlite", "postgresql"], indirect=True)
@pytest.mark.parametrize("selection", ["named", "automatic"])
def test_ingestion_selects_only_owned_or_public_templates(shared_template_store, selection):
    db, _ = shared_template_store
    config = copy.deepcopy(CONFIG)
    config["chunking"]["config"]["hierarchical_template"] = {"boundaries": []}
    config["classifier"] = {"media_types": ["document"]}
    form = SimpleNamespace(
        perform_chunking=True,
        media_type="document",
        chunk_method=None,
        chunking_template_name=NAME if selection == "named" else None,
        auto_apply_template=selection == "automatic",
        hierarchical_chunking=False,
    )
    with db.transaction():
        db.update_chunking_template(name=NAME, template_json=json.dumps(config))
        for owner, expected in (("1", True), ("2", False)):
            result = apply_chunking_template_if_any(
                form, db, {"method": "words"}, owner_user_id=owner, TemplateClassifier=TemplateClassifier
            )
            assert bool(result.get("hierarchical")) is expected  # nosec B101 -- Actual named and classifier paths obey the authenticated owner.


@pytest.mark.parametrize("shared_template_store", ["sqlite", "postgresql"], indirect=True)
async def test_match_never_ranks_a_foreign_template(shared_template_store):
    db, _ = shared_template_store
    config = copy.deepcopy(CONFIG)
    config["classifier"] = {"media_types": ["document"]}
    with db.transaction():
        db.update_chunking_template(name=NAME, template_json=json.dumps(config))
        owner = User(id=1, username="template-owner", email=None, is_active=True)
        bob = User(id=2, username="template-bob", email=None, is_active=True)
        own = await endpoint.match_templates(
            media_type="document", title=None, url=None, filename=None, current_user=owner, db=db
        )
        foreign = await endpoint.match_templates(
            media_type="document", title=None, url=None, filename=None, current_user=bob, db=db
        )
        assert NAME in {r["name"] for r in own["matches"]}  # nosec B101 -- Legitimate classifier discovery remains available.
        assert NAME not in {r["name"] for r in foreign["matches"]}  # nosec B101 -- Ranking must not disclose foreign names.


@pytest.mark.parametrize("operation", ["get", "list", "match", "learn"])
async def test_missing_authenticated_owner_is_rejected_before_database_access(operation):
    class NoAccess:
        def __getattr__(self, name):
            raise AssertionError("Database touched before owner validation")

    args = {"current_user": SimpleNamespace(id=None), "db": NoAccess()}
    with pytest.raises(HTTPException) as refused:
        if operation == "get":
            await endpoint.get_template(NAME, **args)
        elif operation == "list":
            await endpoint.list_templates(
                include_builtin=True, include_custom=True, tags=None, user_id=None, response=None, **args
            )
        elif operation == "match":
            await endpoint.match_templates(media_type=None, title=None, url=None, filename=None, **args)
        else:
            await endpoint.learn_template(endpoint.LearnTemplateRequest(name="no-owner", save=True), **args)
    assert refused.value.status_code == 401  # nosec B101 -- An absent authenticated owner cannot turn into an unscoped read.


@pytest.mark.parametrize("shared_template_store", ["sqlite", "postgresql"], indirect=True)
async def test_json_chunking_treats_foreign_template_like_an_unknown_name(shared_template_store):
    from starlette.requests import Request

    from tldw_Server_API.app.api.v1.endpoints import chunking
    from tldw_Server_API.app.api.v1.schemas.chunking_schema import ChunkingOptionsRequest, ChunkingTextRequest

    db, _ = shared_template_store
    bob = User(id=2, username="template-bob", email=None, is_active=True)
    results = []
    with db.transaction():
        for name in (NAME, "unknown-template"):
            result = await chunking.process_text_for_chunking_json(
                ChunkingTextRequest(
                    text_content="One two three four.",
                    file_name="example.txt",
                    options=ChunkingOptionsRequest(method="words", max_size=8, overlap=0, template_name=name),
                ),
                http_request=Request({"type": "http", "method": "POST", "path": "/chunk_text", "headers": []}),
                current_user=bob,
                media_db=db,
            )
            results.append([chunk.model_dump() for chunk in result.chunks])
    assert results[0] == results[1]  # nosec B101 -- Foreign config must not affect chunking or metadata.


def test_missing_ingestion_owner_does_not_read_templates():
    class NoAccess:
        def __getattr__(self, name):
            raise AssertionError("Template store accessed without an owner")

    form = SimpleNamespace(perform_chunking=True, chunking_template_name=NAME)
    options = {"method": "words", "max_size": 8}
    result = apply_chunking_template_if_any(form, NoAccess(), options, owner_user_id=None)
    assert result == options  # nosec B101 -- Missing identity preserves normal options without an unscoped lookup.


async def test_fallback_public_builtins_stay_readable_and_immutable(monkeypatch):
    record = {
        "id": 3,
        "uuid": str(uuid5(NAMESPACE_URL, "public-fallback")),
        "name": "public-fallback",
        "description": "Public built-in",
        "template_json": json.dumps(CONFIG),
        "is_builtin": True,
        "tags": [],
        "user_id": None,
        "version": 1,
        "created_at": "2026-11-02T00:00:00+00:00",
        "updated_at": "2026-11-02T00:00:00+00:00",
    }
    before = copy.deepcopy(record)
    ownerless = {**record, "name": "ownerless", "is_builtin": False}
    monkeypatch.setattr(
        endpoint, "_FALLBACK_TEMPLATES", {"catalog": {"public-fallback": record, "ownerless": ownerless}}
    )
    monkeypatch.setenv("CHUNKING_TEMPLATES_FALLBACK_ENABLED", "1")
    bob = User(id=2, username="template-bob", email=None, is_active=True)
    db = SimpleNamespace()
    assert (await endpoint.get_template("public-fallback", current_user=bob, db=db)).is_builtin  # nosec B101 -- Catalog rows are public across fallback buckets.
    listed = await endpoint.list_templates(
        include_builtin=True, include_custom=True, tags=None, user_id=None, current_user=bob, db=db, response=None
    )
    assert [r.name for r in listed.templates] == ["public-fallback"]  # nosec B101 -- An ownerless custom is not public.
    for operation in ("update", "delete"):
        with pytest.raises(HTTPException) as refused:
            if operation == "update":
                await endpoint.update_template(
                    "public-fallback",
                    ChunkingTemplateUpdate(description="Forbidden"),
                    current_user=bob,
                    db=db,
                    response=None,
                )
            else:
                await endpoint.delete_template(
                    "public-fallback", hard_delete=True, current_user=bob, db=db, response=None
                )
        assert refused.value.status_code == 400  # nosec B101 -- Public does not grant mutation permission.
    assert endpoint._FALLBACK_TEMPLATES["catalog"]["public-fallback"] == before  # nosec B101 -- The exact catalog definition survives refusals.
