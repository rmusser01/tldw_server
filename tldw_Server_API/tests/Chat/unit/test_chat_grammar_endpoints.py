from __future__ import annotations

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoints
from tldw_Server_API.app.api.v1.endpoints import chat_grammars as chat_grammar_endpoints
from tldw_Server_API.app.api.v1.schemas.chat_grammar_schemas import (
    ChatGrammarCreate,
    ChatGrammarUpdate,
)
from tldw_Server_API.app.core.Character_Chat.chat_grammar import ChatGrammarService
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)


@pytest.fixture()
def chacha_db(tmp_path):
    db_path = tmp_path / "ChaChaNotes.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="test-client")
    try:
        yield db
    finally:
        db.close_connection()


def test_chat_grammar_router_exposes_expected_paths():
    paths = {route.path for route in chat_grammar_endpoints.router.routes}
    assert "/grammars" in paths
    assert "/grammars/{grammar_id}" in paths


def test_chat_grammar_route_handlers_have_docstrings():
    handlers = (
        chat_grammar_endpoints.create_chat_grammar,
        chat_grammar_endpoints.list_chat_grammars,
        chat_grammar_endpoints.get_chat_grammar,
        chat_grammar_endpoints.update_chat_grammar,
        chat_grammar_endpoints.delete_chat_grammar,
    )

    for handler in handlers:
        assert handler.__doc__


@pytest.mark.asyncio
async def test_create_and_list_chat_grammars(chacha_db: CharactersRAGDB):
    created = await chat_endpoints.create_chat_grammar(
        ChatGrammarCreate(name="Root", description="desc", grammar_text='root ::= "ok"'),
        db=chacha_db,
    )
    assert created.name == "Root"

    listing = await chat_endpoints.list_chat_grammars(
        include_archived=False,
        limit=100,
        offset=0,
        db=chacha_db,
    )
    assert listing.total == 1
    assert listing.items[0].id == created.id


@pytest.mark.asyncio
async def test_list_chat_grammars_returns_canonical_pagination(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ChatGrammarService, "__init__", lambda self, db: None)
    monkeypatch.setattr(
        ChatGrammarService,
        "list_grammars",
        lambda self, include_archived=False, limit=100, offset=0: [
            {
                "id": "grammar-1",
                "name": "Root",
                "description": "desc",
                "grammar_text": 'root ::= "ok"',
                "validation_status": "valid",
                "validation_error": None,
                "last_validated_at": None,
                "is_archived": False,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "version": 1,
            }
        ],
    )
    monkeypatch.setattr(ChatGrammarService, "count_grammars", lambda self, include_archived=False: 1)

    listing = await chat_grammar_endpoints.list_chat_grammars(
        include_archived=False,
        limit=100,
        offset=0,
        db=object(),
    )

    assert listing.total == 1
    assert listing.items[0].id == "grammar-1"
    assert listing.pagination.total == 1
    assert listing.pagination.limit == 100
    assert listing.pagination.offset == 0
    assert listing.pagination.has_more is False
    assert listing.pagination.next_offset is None
    assert listing.has_more is False
    assert listing.next_offset is None


@pytest.mark.asyncio
async def test_get_archived_grammar_requires_include_archived(chacha_db: CharactersRAGDB):
    service = ChatGrammarService(chacha_db)
    grammar_id = service.create_grammar(
        name="Archived Grammar",
        description="desc",
        grammar_text='root ::= "archived"',
    )
    service.archive_grammar(grammar_id)

    with pytest.raises(HTTPException) as excinfo:
        await chat_endpoints.get_chat_grammar(grammar_id, include_archived=False, db=chacha_db)
    assert excinfo.value.status_code == 404

    archived = await chat_endpoints.get_chat_grammar(
        grammar_id,
        include_archived=True,
        db=chacha_db,
    )
    assert archived.id == grammar_id
    assert archived.is_archived is True


@pytest.mark.asyncio
async def test_update_chat_grammar_returns_refreshed_record(chacha_db: CharactersRAGDB):
    created = await chat_endpoints.create_chat_grammar(
        ChatGrammarCreate(
            name="Versioned Grammar",
            description="v1",
            grammar_text='root ::= "v1"',
        ),
        db=chacha_db,
    )

    updated = await chat_endpoints.update_chat_grammar(
        created.id,
        ChatGrammarUpdate(description="v2", grammar_text='root ::= "v2"'),
        db=chacha_db,
    )

    assert updated.description == "v2"
    assert updated.grammar_text == 'root ::= "v2"'
    assert updated.version == created.version + 1


@pytest.mark.asyncio
async def test_delete_chat_grammar_hides_record_from_default_reads(chacha_db: CharactersRAGDB):
    created = await chat_endpoints.create_chat_grammar(
        ChatGrammarCreate(
            name="Delete Grammar",
            description="desc",
            grammar_text='root ::= "delete"',
        ),
        db=chacha_db,
    )

    response = await chat_endpoints.delete_chat_grammar(created.id, db=chacha_db)
    assert response.status_code == 204

    with pytest.raises(HTTPException) as excinfo:
        await chat_endpoints.get_chat_grammar(created.id, include_archived=False, db=chacha_db)
    assert excinfo.value.status_code == 404


@pytest.mark.asyncio
async def test_create_chat_grammar_maps_database_error_to_contextual_500(
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    def _raise_db_error(self, *, name: str, description: str | None = None, grammar_text: str) -> str:
        _ = (self, name, description, grammar_text)
        raise CharactersRAGDBError("sqlite backend unavailable")

    monkeypatch.setattr(ChatGrammarService, "create_grammar", _raise_db_error)

    with pytest.raises(HTTPException) as excinfo:
        await chat_endpoints.create_chat_grammar(
            ChatGrammarCreate(name="Broken", description="desc", grammar_text='root ::= "broken"'),
            db=chacha_db,
        )

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Failed to create chat grammar"


@pytest.mark.asyncio
async def test_list_chat_grammars_maps_database_error_to_contextual_500(
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    def _raise_db_error(
        self,
        *,
        include_archived: bool = False,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        _ = (self, include_archived, include_deleted, limit, offset)
        raise CharactersRAGDBError("sqlite backend unavailable")

    monkeypatch.setattr(ChatGrammarService, "list_grammars", _raise_db_error)

    with pytest.raises(HTTPException) as excinfo:
        await chat_endpoints.list_chat_grammars(
            include_archived=False,
            limit=100,
            offset=0,
            db=chacha_db,
        )

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Failed to list chat grammars"


@pytest.mark.asyncio
async def test_get_chat_grammar_maps_database_error_to_contextual_500(
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    def _raise_db_error(self, grammar_id: str, include_archived: bool = False):
        _ = (self, grammar_id, include_archived)
        raise CharactersRAGDBError("sqlite backend unavailable")

    monkeypatch.setattr(ChatGrammarService, "get_grammar", _raise_db_error)

    with pytest.raises(HTTPException) as excinfo:
        await chat_endpoints.get_chat_grammar(
            "grammar-1",
            include_archived=False,
            db=chacha_db,
        )

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Failed to get chat grammar"


@pytest.mark.asyncio
async def test_update_chat_grammar_maps_database_error_to_contextual_500(
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        ChatGrammarService,
        "get_grammar",
        lambda self, grammar_id, include_archived=False: {
            "id": grammar_id,
            "name": "Grammar",
            "description": "desc",
            "grammar_text": 'root ::= "v1"',
            "is_archived": False,
            "version": 1,
        },
    )

    def _raise_db_error(
        self,
        grammar_id: str,
        updates: dict,
        *,
        expected_version: int | None = None,
    ) -> dict:
        _ = (self, grammar_id, updates, expected_version)
        raise CharactersRAGDBError("sqlite backend unavailable")

    monkeypatch.setattr(ChatGrammarService, "update_grammar", _raise_db_error)

    with pytest.raises(HTTPException) as excinfo:
        await chat_endpoints.update_chat_grammar(
            "grammar-1",
            ChatGrammarUpdate(description="v2"),
            db=chacha_db,
        )

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Failed to update chat grammar"


@pytest.mark.asyncio
async def test_delete_chat_grammar_maps_input_error_to_404(
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    def _raise_input(self, grammar_id: str, *, hard_delete: bool = False) -> bool:
        _ = (self, grammar_id, hard_delete)
        raise InputError("grammar missing")

    monkeypatch.setattr(ChatGrammarService, "delete_grammar", _raise_input)

    with pytest.raises(HTTPException) as excinfo:
        await chat_endpoints.delete_chat_grammar("grammar-1", db=chacha_db)

    assert excinfo.value.status_code == 404
    assert excinfo.value.detail == "grammar missing"


@pytest.mark.asyncio
async def test_delete_chat_grammar_preserves_conflict_detail(
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    def _raise_conflict(self, grammar_id: str, *, hard_delete: bool = False) -> bool:
        _ = (self, grammar_id, hard_delete)
        raise ConflictError("grammar is still referenced")

    monkeypatch.setattr(ChatGrammarService, "delete_grammar", _raise_conflict)

    with pytest.raises(HTTPException) as excinfo:
        await chat_endpoints.delete_chat_grammar("grammar-1", db=chacha_db)

    assert excinfo.value.status_code == 409
    assert excinfo.value.detail == "grammar is still referenced"


@pytest.mark.asyncio
async def test_delete_chat_grammar_maps_database_error_to_contextual_500(
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    def _raise_db_error(self, grammar_id: str, *, hard_delete: bool = False) -> bool:
        _ = (self, grammar_id, hard_delete)
        raise CharactersRAGDBError("sqlite backend unavailable")

    monkeypatch.setattr(ChatGrammarService, "delete_grammar", _raise_db_error)

    with pytest.raises(HTTPException) as excinfo:
        await chat_endpoints.delete_chat_grammar("grammar-1", db=chacha_db)

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Failed to delete chat grammar"


@pytest.mark.parametrize(
    ("case_name", "service_method", "handler_factory", "expected_detail", "expected_log"),
    [
        (
            "create",
            "create_grammar",
            lambda db: chat_endpoints.create_chat_grammar(
                ChatGrammarCreate(name="Broken", description="desc", grammar_text='root ::= "broken"'),
                db=db,
            ),
            "Failed to create chat grammar",
            "Error creating chat grammar",
        ),
        (
            "list",
            "list_grammars",
            lambda db: chat_endpoints.list_chat_grammars(
                include_archived=False,
                limit=100,
                offset=0,
                db=db,
            ),
            "Failed to list chat grammars",
            "Error listing chat grammars",
        ),
        (
            "get",
            "get_grammar",
            lambda db: chat_endpoints.get_chat_grammar(
                "grammar-1",
                include_archived=False,
                db=db,
            ),
            "Failed to get chat grammar",
            "Error getting chat grammar",
        ),
        (
            "update",
            "update_grammar",
            lambda db: chat_endpoints.update_chat_grammar(
                "grammar-1",
                ChatGrammarUpdate(description="v2"),
                db=db,
            ),
            "Failed to update chat grammar",
            "Error updating chat grammar",
        ),
        (
            "delete",
            "delete_grammar",
            lambda db: chat_endpoints.delete_chat_grammar("grammar-1", db=db),
            "Failed to delete chat grammar",
            "Error deleting chat grammar",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
@pytest.mark.asyncio
async def test_chat_grammar_handlers_sanitize_unexpected_service_errors(
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
    case_name,
    service_method,
    handler_factory,
    expected_detail,
    expected_log,
):
    class _LoggerStub:
        def __init__(self):
            self.errors = []

        def error(self, message, *args, **kwargs):
            self.errors.append((message, args, kwargs))

    logger = _LoggerStub()
    monkeypatch.setattr(chat_grammar_endpoints, "logger", logger)

    if case_name == "update":
        monkeypatch.setattr(
            ChatGrammarService,
            "get_grammar",
            lambda self, grammar_id, include_archived=False: {
                "id": grammar_id,
                "name": "Grammar",
                "description": "desc",
                "grammar_text": 'root ::= "v1"',
                "is_archived": False,
                "version": 1,
            },
        )

    def _raise_unexpected_error(self, *args, **kwargs):
        _ = (self, args, kwargs)
        raise RuntimeError(f"{case_name} backend unavailable")

    monkeypatch.setattr(ChatGrammarService, service_method, _raise_unexpected_error)

    with pytest.raises(HTTPException) as excinfo:
        await handler_factory(chacha_db)

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == expected_detail
    assert logger.errors == [(expected_log, (), {})]
    logged = repr(logger.errors)
    assert "backend unavailable" not in logged
