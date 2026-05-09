from __future__ import annotations

import datetime

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoints
from tldw_Server_API.app.api.v1.endpoints import chat_dictionaries as chat_dictionary_endpoints
from tldw_Server_API.app.api.v1.schemas.chat_dictionary_schemas import (
    BulkEntryOperation,
    ChatDictionaryCreate,
    ChatDictionaryUpdate,
    DictionaryEntryCreate,
    DictionaryEntryReorderRequest,
    DictionaryEntryUpdate,
    ImportDictionaryJSONRequest,
    ImportDictionaryRequest,
    ProcessTextRequest,
)
from tldw_Server_API.app.core.Character_Chat.chat_dictionary import ChatDictionaryService
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError


@pytest.fixture()
def chacha_db(tmp_path):
    db_path = tmp_path / "ChaChaNotes.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="test-client")
    try:
        yield db
    finally:
        db.close_connection()


def test_chat_dictionary_router_exposes_markdown_export_alias():
    paths = {route.path for route in chat_dictionary_endpoints.router.routes}
    assert "/dictionaries/{dictionary_id}/export" in paths
    assert "/dictionaries/{dictionary_id}/export/markdown" in paths


def test_process_text_dictionary_ids_validation_names_dictionary_ids_field():
    with pytest.raises(ValueError) as exc_info:
        ProcessTextRequest.model_validate({"text": "hello", "dictionary_ids": ["nope"]})

    message = str(exc_info.value)
    assert "dictionary_ids" in message
    assert "included_dictionary_ids" not in message


def test_chat_dictionary_service_initializes_legacy_entries_table_without_sort_order(
    chacha_db: CharactersRAGDB,
):
    with chacha_db.get_connection() as conn:
        conn.execute("DROP TABLE IF EXISTS dictionary_entries")
        conn.execute("DROP TABLE IF EXISTS chat_dictionaries")

        conn.execute(
            """
            CREATE TABLE chat_dictionaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                version INTEGER DEFAULT 1,
                deleted BOOLEAN DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE dictionary_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dictionary_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                content TEXT NOT NULL,
                is_regex BOOLEAN DEFAULT 0,
                probability REAL DEFAULT 1.0,
                max_replacements INTEGER DEFAULT 1,
                group_name TEXT,
                timed_effects TEXT DEFAULT '{"sticky": 0, "cooldown": 0, "delay": 0}',
                enabled BOOLEAN DEFAULT 1,
                case_sensitive BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dictionary_id) REFERENCES chat_dictionaries(id) ON DELETE CASCADE
            )
            """
        )

        conn.execute(
            """
            INSERT INTO chat_dictionaries (name, description, is_active, version, deleted)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("Legacy Dictionary", "Legacy schema test", True, 1, False),
        )
        conn.execute(
            """
            INSERT INTO dictionary_entries
            (dictionary_id, key, content, is_regex, probability, max_replacements, enabled, case_sensitive)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (1, "legacy-key", "legacy-value", False, 1.0, 1, True, True),
        )
        conn.commit()

    service = ChatDictionaryService(chacha_db)
    dictionaries = service.list_dictionaries_with_entry_counts(include_inactive=True)
    assert len(dictionaries) == 1

    with chacha_db.get_connection() as conn:
        column_rows = conn.execute("PRAGMA table_info(dictionary_entries)").fetchall()
        column_names: set[str] = set()
        for row in column_rows:
            if isinstance(row, dict):
                raw_name = row.get("name")
            elif hasattr(row, "_mapping"):
                raw_name = row._mapping.get("name")
            else:
                raw_name = row[1] if len(row) > 1 else None
            if isinstance(raw_name, str):
                column_names.add(raw_name)
        assert "sort_order" in column_names

        index_rows = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'index' AND name = 'idx_dict_entries_sort_order'
            """
        ).fetchall()
        assert len(index_rows) == 1


@pytest.mark.asyncio
async def test_add_dictionary_entry_returns_persisted_fields(chacha_db: CharactersRAGDB):
    service = ChatDictionaryService(chacha_db)
    dictionary_id = service.create_dictionary("Test Dictionary", "Description")

    entry_create = DictionaryEntryCreate(pattern="hello", replacement="world")
    response = await chat_endpoints.add_dictionary_entry(dictionary_id, entry_create, db=chacha_db)

    assert response.dictionary_id == dictionary_id
    assert response.pattern == "hello"
    assert response.replacement == "world"
    assert isinstance(response.created_at, datetime.datetime)
    assert isinstance(response.updated_at, datetime.datetime)

    stored_entries = service.get_entries(dictionary_id=dictionary_id, active_only=False)
    assert len(stored_entries) == 1
    stored_entry = stored_entries[0]
    assert stored_entry["id"] == response.id
    assert stored_entry["pattern"] == "hello"
    assert stored_entry["replacement"] == "world"


@pytest.mark.asyncio
async def test_add_dictionary_entry_maps_prefetch_database_error(
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    def _raise_db_error(self, dictionary_id: int):
        _ = (self, dictionary_id)
        raise CharactersRAGDBError("sqlite backend unavailable")

    monkeypatch.setattr(ChatDictionaryService, "get_dictionary", _raise_db_error)

    with pytest.raises(HTTPException) as exc_info:
        await chat_endpoints.add_dictionary_entry(
            123,
            DictionaryEntryCreate(pattern="alpha", replacement="beta"),
            db=chacha_db,
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to add dictionary entry"


@pytest.mark.asyncio
async def test_timed_effects_round_trip_through_entry_endpoints(
    chacha_db: CharactersRAGDB,
):
    service = ChatDictionaryService(chacha_db)
    dictionary_id = service.create_dictionary("Timed Effects Dictionary", "Description")

    response = await chat_endpoints.add_dictionary_entry(
        dictionary_id,
        DictionaryEntryCreate(
            pattern="pulse",
            replacement="heart-rate",
            timed_effects={"sticky": 17, "cooldown": 8, "delay": 3},
        ),
        db=chacha_db,
    )

    assert response.timed_effects is not None
    assert response.timed_effects.sticky == 17
    assert response.timed_effects.cooldown == 8
    assert response.timed_effects.delay == 3

    entries_response = await chat_endpoints.list_dictionary_entries(
        dictionary_id,
        group=None,
        db=chacha_db,
    )
    assert entries_response.total == 1
    assert entries_response.entries[0].timed_effects is not None
    assert entries_response.entries[0].timed_effects.sticky == 17
    assert entries_response.entries[0].timed_effects.cooldown == 8
    assert entries_response.entries[0].timed_effects.delay == 3

    dictionary_response = await chat_endpoints.get_chat_dictionary(dictionary_id, db=chacha_db)
    assert len(dictionary_response.entries) == 1
    assert dictionary_response.entries[0].timed_effects is not None
    assert dictionary_response.entries[0].timed_effects.sticky == 17
    assert dictionary_response.entries[0].timed_effects.cooldown == 8
    assert dictionary_response.entries[0].timed_effects.delay == 3


@pytest.mark.asyncio
async def test_case_sensitive_defaults_and_explicit_override_compatibility(
    chacha_db: CharactersRAGDB,
):
    service = ChatDictionaryService(chacha_db)
    dictionary_id = service.create_dictionary("Case Sensitivity Dictionary", "Description")

    default_entry = await chat_endpoints.add_dictionary_entry(
        dictionary_id,
        DictionaryEntryCreate(pattern="default-case", replacement="DEFAULT"),
        db=chacha_db,
    )
    explicit_entry = await chat_endpoints.add_dictionary_entry(
        dictionary_id,
        DictionaryEntryCreate(
            pattern="explicit-case",
            replacement="EXPLICIT",
            case_sensitive=False,
        ),
        db=chacha_db,
    )

    assert default_entry.case_sensitive is True
    assert explicit_entry.case_sensitive is False

    entries_response = await chat_endpoints.list_dictionary_entries(
        dictionary_id,
        group=None,
        db=chacha_db,
    )
    cases_by_pattern = {
        entry.pattern: entry.case_sensitive for entry in entries_response.entries
    }
    assert cases_by_pattern["default-case"] is True
    assert cases_by_pattern["explicit-case"] is False


@pytest.mark.asyncio
async def test_get_chat_dictionary_includes_entry_metadata(chacha_db: CharactersRAGDB):
    service = ChatDictionaryService(chacha_db)
    dictionary_id = service.create_dictionary("Metadata Dictionary", "desc")
    service.add_entry(
        dictionary_id,
        pattern="foo",
        replacement="bar",
        probability=0.5,
        group="group-a",
    )

    response = await chat_endpoints.get_chat_dictionary(dictionary_id, db=chacha_db)

    assert response.id == dictionary_id
    assert response.entry_count == 1
    assert len(response.entries) == 1
    entry = response.entries[0]
    assert entry.pattern == "foo"
    assert entry.replacement == "bar"
    assert isinstance(entry.created_at, datetime.datetime)
    assert isinstance(entry.updated_at, datetime.datetime)


@pytest.mark.asyncio
async def test_update_dictionary_entry_returns_latest_state(chacha_db: CharactersRAGDB):
    service = ChatDictionaryService(chacha_db)
    dictionary_id = service.create_dictionary("Update Dictionary", None)
    entry_id = service.add_entry(dictionary_id, pattern="key", replacement="value")

    response = await chat_endpoints.update_dictionary_entry(
        entry_id,
        DictionaryEntryUpdate(group="updated-group", enabled=False, case_sensitive=False),
        db=chacha_db,
    )

    assert response.id == entry_id
    assert response.dictionary_id == dictionary_id
    assert response.group == "updated-group"
    assert response.enabled is False
    assert response.case_sensitive is False
    assert isinstance(response.updated_at, datetime.datetime)


@pytest.mark.asyncio
async def test_update_dictionary_entry_rejects_unsafe_regex_pattern(
    chacha_db: CharactersRAGDB,
):
    service = ChatDictionaryService(chacha_db)
    dictionary_id = service.create_dictionary("Regex Guard Dictionary", None)
    entry_id = service.add_entry(
        dictionary_id,
        pattern="hello.*",
        replacement="hi",
        type="regex",
    )

    with pytest.raises(HTTPException) as exc_info:
        await chat_endpoints.update_dictionary_entry(
            entry_id,
            DictionaryEntryUpdate(pattern="(.+)+"),
            db=chacha_db,
        )

    assert exc_info.value.status_code == 400
    detail = str(exc_info.value.detail).lower()
    assert "dangerous regex pattern" in detail


@pytest.mark.asyncio
async def test_bulk_dictionary_entry_operations_reports_partial_failures(
    chacha_db: CharactersRAGDB,
):
    service = ChatDictionaryService(chacha_db)
    dictionary_id = service.create_dictionary("Bulk Dictionary", None)
    entry_a = service.add_entry(dictionary_id, pattern="a", replacement="A")
    entry_b = service.add_entry(dictionary_id, pattern="b", replacement="B")

    response = await chat_endpoints.bulk_dictionary_entry_operations(
        BulkEntryOperation(
            entry_ids=[entry_a, entry_b, 999999],
            operation="activate",
        ),
        db=chacha_db,
    )

    assert response.success is False
    assert response.affected_count == 2
    assert response.failed_ids == [999999]


@pytest.mark.asyncio
async def test_bulk_dictionary_entry_operations_sets_group(
    chacha_db: CharactersRAGDB,
):
    service = ChatDictionaryService(chacha_db)
    dictionary_id = service.create_dictionary("Bulk Group Dictionary", None)
    entry_id = service.add_entry(dictionary_id, pattern="bp", replacement="blood pressure")

    response = await chat_endpoints.bulk_dictionary_entry_operations(
        BulkEntryOperation(
            entry_ids=[entry_id],
            operation="group",
            group_name="reviewed",
        ),
        db=chacha_db,
    )

    assert response.success is True
    assert response.affected_count == 1
    updated_entry = service.get_entry(entry_id, active_only=False)
    assert updated_entry is not None
    assert updated_entry.get("group") == "reviewed"


@pytest.mark.asyncio
async def test_reorder_dictionary_entries_updates_execution_order(
    chacha_db: CharactersRAGDB,
):
    service = ChatDictionaryService(chacha_db)
    dictionary_id = service.create_dictionary("Reorder Dictionary", None)
    entry_a = service.add_entry(dictionary_id, pattern="a", replacement="A")
    entry_b = service.add_entry(dictionary_id, pattern="b", replacement="B")
    entry_c = service.add_entry(dictionary_id, pattern="c", replacement="C")

    response = await chat_endpoints.reorder_dictionary_entries(
        dictionary_id,
        DictionaryEntryReorderRequest(entry_ids=[entry_c, entry_a, entry_b]),
        db=chacha_db,
    )

    assert response.success is True
    assert response.affected_count == 3
    assert response.entry_ids == [entry_c, entry_a, entry_b]

    ordered_entries = service.get_entries(dictionary_id=dictionary_id, active_only=False)
    assert [int(entry["id"]) for entry in ordered_entries] == [entry_c, entry_a, entry_b]


@pytest.mark.asyncio
async def test_reorder_dictionary_entries_rejects_incomplete_entry_list(
    chacha_db: CharactersRAGDB,
):
    service = ChatDictionaryService(chacha_db)
    dictionary_id = service.create_dictionary("Reorder Validation Dictionary", None)
    entry_a = service.add_entry(dictionary_id, pattern="left", replacement="right")
    _entry_b = service.add_entry(dictionary_id, pattern="up", replacement="down")

    with pytest.raises(HTTPException) as exc_info:
        await chat_endpoints.reorder_dictionary_entries(
            dictionary_id,
            DictionaryEntryReorderRequest(entry_ids=[entry_a]),
            db=chacha_db,
        )

    assert exc_info.value.status_code == 400
    assert "every dictionary entry exactly once" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_bulk_dictionary_entry_operations_sanitizes_unexpected_error(
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    service = ChatDictionaryService(chacha_db)
    dictionary_id = service.create_dictionary("Bulk Fallback Dictionary", None)
    entry_id = service.add_entry(dictionary_id, pattern="fallback", replacement="safe")

    def _raise_unexpected_response_error(*args, **kwargs):
        _ = (args, kwargs)
        raise RuntimeError("bulk response backend unavailable")

    monkeypatch.setattr(
        chat_dictionary_endpoints,
        "BulkOperationResponse",
        _raise_unexpected_response_error,
    )

    with pytest.raises(HTTPException) as exc_info:
        await chat_endpoints.bulk_dictionary_entry_operations(
            BulkEntryOperation(entry_ids=[entry_id], operation="activate"),
            db=chacha_db,
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to perform bulk entry operation"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("service_method", "call_factory", "expected_detail"),
    [
        (
            "reorder_entries",
            lambda dictionary_id, db: chat_endpoints.reorder_dictionary_entries(
                dictionary_id,
                DictionaryEntryReorderRequest(entry_ids=[1]),
                db=db,
            ),
            "Failed to reorder dictionary entries",
        ),
        (
            "list_transform_activity",
            lambda dictionary_id, db: chat_endpoints.list_dictionary_activity(
                dictionary_id,
                limit=10,
                offset=0,
                db=db,
            ),
            "Failed to list dictionary activity",
        ),
        (
            "list_dictionary_versions",
            lambda dictionary_id, db: chat_endpoints.list_dictionary_versions(
                dictionary_id,
                limit=10,
                offset=0,
                db=db,
            ),
            "Failed to list dictionary versions",
        ),
        (
            "get_dictionary_version",
            lambda dictionary_id, db: chat_endpoints.get_dictionary_version(
                dictionary_id,
                1,
                db=db,
            ),
            "Failed to read dictionary revision",
        ),
        (
            "revert_dictionary_to_revision",
            lambda dictionary_id, db: chat_endpoints.revert_dictionary_version(
                dictionary_id,
                1,
                db=db,
            ),
            "Failed to revert dictionary revision",
        ),
        (
            "get_statistics",
            lambda dictionary_id, db: chat_endpoints.get_dictionary_statistics(
                dictionary_id,
                db=db,
            ),
            "Failed to get dictionary statistics",
        ),
    ],
)
async def test_chat_dictionary_tail_handlers_sanitize_unexpected_errors(
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
    service_method: str,
    call_factory,
    expected_detail: str,
):
    service = ChatDictionaryService(chacha_db)
    dictionary_id = service.create_dictionary("Tail Fallback Dictionary", None)

    def _raise_unexpected_error(self, *args, **kwargs):
        _ = (self, args, kwargs)
        raise RuntimeError(f"{service_method} backend unavailable")

    monkeypatch.setattr(ChatDictionaryService, service_method, _raise_unexpected_error)

    with pytest.raises(HTTPException) as exc_info:
        await call_factory(dictionary_id, chacha_db)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == expected_detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case_name", "service_method", "call_factory", "expected_detail"),
    [
        (
            "create_dictionary",
            "create_dictionary",
            lambda dictionary_id, entry_id, db: chat_endpoints.create_chat_dictionary(
                ChatDictionaryCreate(name="Create Fallback Dictionary"),
                db=db,
            ),
            "Failed to create dictionary",
        ),
        (
            "list_dictionaries",
            "list_dictionaries_with_entry_counts",
            lambda dictionary_id, entry_id, db: chat_endpoints.list_chat_dictionaries(
                include_inactive=True,
                include_usage=False,
                db=db,
            ),
            "Failed to list dictionaries",
        ),
        (
            "get_dictionary",
            "get_dictionary",
            lambda dictionary_id, entry_id, db: chat_endpoints.get_chat_dictionary(
                dictionary_id,
                db=db,
            ),
            "Failed to get dictionary",
        ),
        (
            "update_dictionary",
            "update_dictionary",
            lambda dictionary_id, entry_id, db: chat_endpoints.update_chat_dictionary(
                dictionary_id,
                ChatDictionaryUpdate(description="fallback update"),
                db=db,
            ),
            "Failed to update dictionary",
        ),
        (
            "delete_dictionary",
            "delete_dictionary",
            lambda dictionary_id, entry_id, db: chat_endpoints.delete_chat_dictionary(
                dictionary_id,
                hard_delete=False,
                db=db,
            ),
            "Failed to delete dictionary",
        ),
        (
            "add_entry_prefetch",
            "get_dictionary",
            lambda dictionary_id, entry_id, db: chat_endpoints.add_dictionary_entry(
                dictionary_id,
                DictionaryEntryCreate(pattern="prefetch", replacement="safe"),
                db=db,
            ),
            "Failed to add dictionary entry",
        ),
        (
            "add_entry",
            "add_entry",
            lambda dictionary_id, entry_id, db: chat_endpoints.add_dictionary_entry(
                dictionary_id,
                DictionaryEntryCreate(pattern="add", replacement="safe"),
                db=db,
            ),
            "Failed to add dictionary entry",
        ),
        (
            "list_entries",
            "get_entries",
            lambda dictionary_id, entry_id, db: chat_endpoints.list_dictionary_entries(
                dictionary_id,
                group=None,
                db=db,
            ),
            "Failed to list dictionary entries",
        ),
        (
            "update_entry",
            "update_entry",
            lambda dictionary_id, entry_id, db: chat_endpoints.update_dictionary_entry(
                entry_id,
                DictionaryEntryUpdate(replacement="updated fallback"),
                db=db,
            ),
            "Failed to update dictionary entry",
        ),
        (
            "delete_entry",
            "delete_entry",
            lambda dictionary_id, entry_id, db: chat_endpoints.delete_dictionary_entry(
                entry_id,
                db=db,
            ),
            "Failed to delete dictionary entry",
        ),
        (
            "process_text",
            "process_text",
            lambda dictionary_id, entry_id, db: chat_endpoints.process_text_with_dictionaries(
                ProcessTextRequest(text="fallback", dictionary_id=dictionary_id),
                db=db,
            ),
            "Failed to process text",
        ),
        (
            "import_markdown",
            "import_from_markdown",
            lambda dictionary_id, entry_id, db: chat_endpoints.import_dictionary(
                ImportDictionaryRequest(
                    content="# Dictionary",
                    name="Import Fallback Dictionary",
                ),
                db=db,
            ),
            "Failed to import dictionary",
        ),
        (
            "export_markdown",
            "export_to_markdown",
            lambda dictionary_id, entry_id, db: chat_endpoints.export_dictionary(
                dictionary_id,
                db=db,
            ),
            "Failed to export dictionary",
        ),
        (
            "export_json",
            "export_to_json",
            lambda dictionary_id, entry_id, db: chat_endpoints.export_dictionary_json(
                dictionary_id,
                db=db,
            ),
            "Failed to export dictionary JSON",
        ),
        (
            "import_json",
            "import_from_json",
            lambda dictionary_id, entry_id, db: chat_endpoints.import_dictionary_json(
                ImportDictionaryJSONRequest(
                    data={
                        "name": "Import JSON Fallback Dictionary",
                        "entries": [],
                    },
                    activate=True,
                ),
                db=db,
            ),
            "Failed to import dictionary JSON",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
async def test_chat_dictionary_core_handlers_sanitize_unexpected_errors(
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
    case_name: str,
    service_method: str,
    call_factory,
    expected_detail: str,
):
    service = ChatDictionaryService(chacha_db)
    dictionary_id = service.create_dictionary(f"Core Fallback Dictionary {case_name}", None)
    entry_id = service.add_entry(dictionary_id, pattern="core", replacement="safe")

    def _raise_unexpected_error(self, *args, **kwargs):
        _ = (self, args, kwargs)
        raise RuntimeError(f"{case_name} backend unavailable")

    monkeypatch.setattr(ChatDictionaryService, service_method, _raise_unexpected_error)

    with pytest.raises(HTTPException) as exc_info:
        await call_factory(dictionary_id, entry_id, chacha_db)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == expected_detail


@pytest.mark.asyncio
async def test_list_chat_dictionaries_counts_inactive_entries(chacha_db: CharactersRAGDB):
    service = ChatDictionaryService(chacha_db)
    dictionary_id = service.create_dictionary("Inactive Dictionary", None)
    service.add_entry(dictionary_id, pattern="alpha", replacement="beta")
    service.update_dictionary(dictionary_id, is_active=False)

    response = await chat_endpoints.list_chat_dictionaries(include_inactive=True, db=chacha_db)

    # list_chat_dictionaries returns ChatDictionaryResponse models
    dictionaries = {d.id: d for d in response.dictionaries}
    assert dictionary_id in dictionaries
    dict_payload = dictionaries[dictionary_id]
    assert dict_payload.is_active is False
    assert dict_payload.entry_count == 1


@pytest.mark.asyncio
async def test_list_chat_dictionaries_includes_usage_summary(chacha_db: CharactersRAGDB):
    service = ChatDictionaryService(chacha_db)
    dictionary_a = service.create_dictionary("Dictionary A", None)
    dictionary_b = service.create_dictionary("Dictionary B", None)

    character_id = chacha_db.add_character_card({"name": "Dictionary Usage Character"})
    assert character_id is not None

    active_chat_id = chacha_db.add_conversation(
        {
            "character_id": int(character_id),
            "title": "Active Dictionary Chat",
            "state": "in-progress",
        }
    )
    resolved_chat_id = chacha_db.add_conversation(
        {
            "character_id": int(character_id),
            "title": "Resolved Dictionary Chat",
            "state": "resolved",
        }
    )

    assert active_chat_id is not None
    assert resolved_chat_id is not None

    chacha_db.upsert_conversation_settings(
        str(active_chat_id),
        {"chat_dictionary_ids": [dictionary_a, dictionary_b]},
    )
    chacha_db.upsert_conversation_settings(
        str(resolved_chat_id),
        {"chatDictionaryId": dictionary_a},
    )

    response = await chat_endpoints.list_chat_dictionaries(
        include_inactive=True,
        include_usage=True,
        db=chacha_db,
    )

    dictionaries = {d.id: d for d in response.dictionaries}
    dictionary_a_payload = dictionaries[dictionary_a]
    dictionary_b_payload = dictionaries[dictionary_b]

    assert dictionary_a_payload.used_by_chat_count == 2
    assert dictionary_a_payload.used_by_active_chat_count == 1
    assert len(dictionary_a_payload.used_by_chat_refs) >= 1

    assert dictionary_b_payload.used_by_chat_count == 1
    assert dictionary_b_payload.used_by_active_chat_count == 1


@pytest.mark.asyncio
async def test_list_chat_dictionaries_includes_processing_priority(
    chacha_db: CharactersRAGDB,
):
    service = ChatDictionaryService(chacha_db)
    dictionary_b = service.create_dictionary("Beta Dictionary", None)
    dictionary_a = service.create_dictionary("Alpha Dictionary", None)
    dictionary_inactive = service.create_dictionary("Inactive Dictionary", None)
    service.update_dictionary(dictionary_inactive, is_active=False)

    response = await chat_endpoints.list_chat_dictionaries(
        include_inactive=True,
        include_usage=False,
        db=chacha_db,
    )
    dictionaries = {item.id: item for item in response.dictionaries}

    assert dictionaries[dictionary_a].processing_priority == 1
    assert dictionaries[dictionary_b].processing_priority == 2
    assert dictionaries[dictionary_inactive].processing_priority is None


@pytest.mark.asyncio
async def test_process_text_without_dictionary_id_uses_dictionary_priority_order(
    chacha_db: CharactersRAGDB,
):
    service = ChatDictionaryService(chacha_db)
    dictionary_z = service.create_dictionary("Zeta Dictionary", None)
    dictionary_a = service.create_dictionary("Alpha Dictionary", None)

    # Alphabetical dictionary order should run Alpha before Zeta.
    service.add_entry(dictionary_a, pattern="token", replacement="A")
    service.add_entry(dictionary_z, pattern="A", replacement="Z")

    response = await chat_endpoints.process_text_with_dictionaries(
        ProcessTextRequest(
            text="token",
            max_iterations=1,
        ),
        db=chacha_db,
    )

    assert response.processed_text == "Z"
    assert response.replacements == 2


@pytest.mark.asyncio
async def test_create_and_update_dictionary_default_token_budget(
    chacha_db: CharactersRAGDB,
):
    created = await chat_endpoints.create_chat_dictionary(
        ChatDictionaryCreate(
            name="Default Budget Dictionary",
            description="with budget",
            default_token_budget=640,
        ),
        db=chacha_db,
    )
    assert created.default_token_budget == 640

    updated = await chat_endpoints.update_chat_dictionary(
        created.id,
        ChatDictionaryUpdate(default_token_budget=1200),
        db=chacha_db,
    )
    assert updated.default_token_budget == 1200

    cleared = await chat_endpoints.update_chat_dictionary(
        created.id,
        ChatDictionaryUpdate(default_token_budget=None),
        db=chacha_db,
    )
    assert cleared.default_token_budget is None


@pytest.mark.asyncio
async def test_create_and_update_dictionary_metadata_fields(
    chacha_db: CharactersRAGDB,
):
    base_dictionary = await chat_endpoints.create_chat_dictionary(
        ChatDictionaryCreate(
            name="Metadata Base Dictionary",
            description="base dictionary",
        ),
        db=chacha_db,
    )
    created = await chat_endpoints.create_chat_dictionary(
        ChatDictionaryCreate(
            name="Metadata Dictionary",
            description="with metadata",
            category="Medical",
            tags=["abbreviations", "clinical", "abbreviations"],
            included_dictionary_ids=[base_dictionary.id],
        ),
        db=chacha_db,
    )
    assert created.category == "Medical"
    assert created.tags == ["abbreviations", "clinical"]
    assert created.included_dictionary_ids == [base_dictionary.id]

    updated = await chat_endpoints.update_chat_dictionary(
        created.id,
        ChatDictionaryUpdate(
            category="Emergency",
            tags=["urgent", "triage"],
            included_dictionary_ids=[base_dictionary.id],
        ),
        db=chacha_db,
    )
    assert updated.category == "Emergency"
    assert updated.tags == ["urgent", "triage"]
    assert updated.included_dictionary_ids == [base_dictionary.id]

    cleared = await chat_endpoints.update_chat_dictionary(
        created.id,
        ChatDictionaryUpdate(category=None, tags=[], included_dictionary_ids=[]),
        db=chacha_db,
    )
    assert cleared.category is None
    assert cleared.tags == []
    assert cleared.included_dictionary_ids == []

    listed = await chat_endpoints.list_chat_dictionaries(include_inactive=True, db=chacha_db)
    listed_dict = next(item for item in listed.dictionaries if item.id == created.id)
    assert listed_dict.category is None
    assert listed_dict.tags == []


@pytest.mark.asyncio
async def test_process_text_uses_dictionary_default_token_budget_and_records_activity(
    chacha_db: CharactersRAGDB,
):
    service = ChatDictionaryService(chacha_db)
    dictionary_id = service.create_dictionary(
        "Activity Dictionary",
        "tracks activity",
        default_token_budget=250,
    )
    entry_id = service.add_entry(dictionary_id, pattern="term", replacement="TERM")

    response = await chat_endpoints.process_text_with_dictionaries(
        ProcessTextRequest(
            text="term term",
            dictionary_id=dictionary_id,
            chat_id="chat-activity-01",
            max_iterations=2,
        ),
        db=chacha_db,
    )

    assert response.token_budget_used == 250
    assert response.replacements >= 1
    assert entry_id in response.entries_used

    activity = await chat_endpoints.list_dictionary_activity(
        dictionary_id,
        limit=10,
        offset=0,
        db=chacha_db,
    )
    assert activity.dictionary_id == dictionary_id
    assert activity.total >= 1
    assert activity.pagination.total == activity.total
    assert activity.pagination.limit == 10
    assert activity.pagination.offset == 0
    assert activity.pagination.has_more is False
    assert activity.pagination.next_offset is None
    assert activity.has_more is False
    assert activity.next_offset is None
    assert len(activity.events) >= 1

    event = activity.events[0]
    assert event.dictionary_id == dictionary_id
    assert event.chat_id == "chat-activity-01"
    assert entry_id in event.entries_used
    assert event.replacements >= 1
    assert event.original_text_preview.startswith("term")
    assert event.processed_text_preview
    assert isinstance(event.created_at, datetime.datetime)


@pytest.mark.asyncio
async def test_process_text_without_dictionary_id_uses_min_active_default_token_budget(
    chacha_db: CharactersRAGDB,
):
    service = ChatDictionaryService(chacha_db)
    dictionary_a = service.create_dictionary(
        "Alpha Token Budget",
        None,
        default_token_budget=500,
    )
    dictionary_b = service.create_dictionary(
        "Beta Token Budget",
        None,
        default_token_budget=200,
    )
    service.add_entry(dictionary_a, pattern="foo", replacement="bar")
    service.add_entry(dictionary_b, pattern="bar", replacement="baz")

    response = await chat_endpoints.process_text_with_dictionaries(
        ProcessTextRequest(
            text="foo",
            max_iterations=2,
        ),
        db=chacha_db,
    )

    assert response.token_budget_used == 200
    assert response.processed_text == "baz"
    assert response.replacements >= 2


@pytest.mark.asyncio
async def test_dictionary_statistics_exposes_expanded_stage1_fields(
    chacha_db: CharactersRAGDB,
):
    service = ChatDictionaryService(chacha_db)
    dictionary_id = service.create_dictionary("Statistics Dictionary", "desc")
    service.add_entry(dictionary_id, pattern="hello", replacement="hi")
    service.add_entry(
        dictionary_id,
        pattern="pulse",
        replacement="heart-rate",
        probability=0.3,
        timed_effects={"sticky": 15, "cooldown": 0, "delay": 0},
        enabled=False,
    )

    initial_stats = await chat_endpoints.get_dictionary_statistics(dictionary_id, db=chacha_db)
    assert initial_stats.total_entries == 2
    assert initial_stats.enabled_entries == 1
    assert initial_stats.disabled_entries == 1
    assert initial_stats.probabilistic_entries == 1
    assert initial_stats.timed_effect_entries == 1
    assert initial_stats.zero_usage_entries == 2
    assert len(initial_stats.entry_usage) == 2
    assert initial_stats.pattern_conflict_count == 0
    assert initial_stats.pattern_conflicts == []
    assert isinstance(initial_stats.created_at, datetime.datetime)
    assert isinstance(initial_stats.updated_at, datetime.datetime)
    assert initial_stats.total_usage_count == 0
    assert initial_stats.last_used is None

    await chat_endpoints.process_text_with_dictionaries(
        ProcessTextRequest(
            text="hello there",
            dictionary_id=dictionary_id,
        ),
        db=chacha_db,
    )

    usage_stats = await chat_endpoints.get_dictionary_statistics(dictionary_id, db=chacha_db)
    assert usage_stats.total_usage_count >= 1
    assert usage_stats.last_used is not None
    assert isinstance(usage_stats.last_used, datetime.datetime)
    assert usage_stats.zero_usage_entries == 1
    assert len(usage_stats.entry_usage) == 2
    assert usage_stats.pattern_conflict_count == 0
    assert usage_stats.pattern_conflicts == []


@pytest.mark.asyncio
async def test_dictionary_entry_usage_counts_increment_after_processing(
    chacha_db: CharactersRAGDB,
):
    service = ChatDictionaryService(chacha_db)
    dictionary_id = service.create_dictionary("Entry Usage Dictionary", "desc")
    entry_id = service.add_entry(dictionary_id, pattern="term", replacement="TERM")

    entries_before = await chat_endpoints.list_dictionary_entries(
        dictionary_id,
        group=None,
        db=chacha_db,
    )
    before_entry = next(entry for entry in entries_before.entries if entry.id == entry_id)
    assert before_entry.usage_count == 0
    assert before_entry.last_used_at is None

    await chat_endpoints.process_text_with_dictionaries(
        ProcessTextRequest(
            text="term term",
            dictionary_id=dictionary_id,
        ),
        db=chacha_db,
    )

    entries_after = await chat_endpoints.list_dictionary_entries(
        dictionary_id,
        group=None,
        db=chacha_db,
    )
    after_entry = next(entry for entry in entries_after.entries if entry.id == entry_id)
    assert after_entry.usage_count >= 1
    assert after_entry.last_used_at is not None


@pytest.mark.asyncio
async def test_dictionary_usage_updates_when_processing_without_dictionary_id(
    chacha_db: CharactersRAGDB,
):
    service = ChatDictionaryService(chacha_db)
    dictionary_a = service.create_dictionary("Usage A", "desc")
    dictionary_b = service.create_dictionary("Usage B", "desc")
    service.add_entry(dictionary_a, pattern="foo", replacement="FOO")
    service.add_entry(dictionary_b, pattern="bar", replacement="BAR")

    before_a = service.get_usage_statistics(dictionary_a)
    before_b = service.get_usage_statistics(dictionary_b)
    assert before_a["times_used"] == 0
    assert before_a["last_used_at"] is None
    assert before_b["times_used"] == 0
    assert before_b["last_used_at"] is None

    response = await chat_endpoints.process_text_with_dictionaries(
        ProcessTextRequest(text="foo bar"),
        db=chacha_db,
    )

    assert response.processed_text == "FOO BAR"
    assert response.replacements >= 2

    after_a = service.get_usage_statistics(dictionary_a)
    after_b = service.get_usage_statistics(dictionary_b)
    assert after_a["times_used"] >= 1
    assert after_a["last_used_at"] is not None
    assert after_b["times_used"] >= 1
    assert after_b["last_used_at"] is not None


@pytest.mark.asyncio
async def test_dictionary_statistics_reports_pattern_conflicts(
    chacha_db: CharactersRAGDB,
):
    service = ChatDictionaryService(chacha_db)
    dictionary_id = service.create_dictionary("Conflict Dictionary", "desc")
    service.add_entry(dictionary_id, pattern="KCl", replacement="potassium chloride")
    service.add_entry(dictionary_id, pattern="/KC.*/", replacement="kc-regex", type="regex")
    service.add_entry(dictionary_id, pattern="kcl", replacement="kcl-lower")

    stats = await chat_endpoints.get_dictionary_statistics(dictionary_id, db=chacha_db)

    assert stats.pattern_conflict_count >= 2
    assert len(stats.pattern_conflicts) >= 2

    conflict_types = {conflict.conflict_type for conflict in stats.pattern_conflicts}
    assert "literal-regex" in conflict_types
    assert "literal-literal" in conflict_types


@pytest.mark.asyncio
async def test_update_chat_dictionary_returns_409_on_version_conflict(
    chacha_db: CharactersRAGDB,
):
    service = ChatDictionaryService(chacha_db)
    dictionary_id = service.create_dictionary("Versioned Dictionary", "desc")
    baseline = service.get_dictionary(dictionary_id)
    assert baseline is not None
    assert int(baseline["version"]) == 1

    service.update_dictionary(dictionary_id, description="edited in another session")

    with pytest.raises(HTTPException) as exc_info:
        await chat_endpoints.update_chat_dictionary(
            dictionary_id,
            ChatDictionaryUpdate(name="Conflicting Update", version=1),
            db=chacha_db,
        )

    assert exc_info.value.status_code == 409
    detail = str(exc_info.value.detail).lower()
    assert "modified by another session" in detail
    assert "expected version 1" in detail


@pytest.mark.asyncio
async def test_update_chat_dictionary_rejects_include_cycles(
    chacha_db: CharactersRAGDB,
):
    dictionary_a = await chat_endpoints.create_chat_dictionary(
        ChatDictionaryCreate(name="Cycle Dictionary A"),
        db=chacha_db,
    )
    dictionary_b = await chat_endpoints.create_chat_dictionary(
        ChatDictionaryCreate(name="Cycle Dictionary B"),
        db=chacha_db,
    )

    await chat_endpoints.update_chat_dictionary(
        dictionary_a.id,
        ChatDictionaryUpdate(included_dictionary_ids=[dictionary_b.id]),
        db=chacha_db,
    )

    with pytest.raises(HTTPException) as exc_info:
        await chat_endpoints.update_chat_dictionary(
            dictionary_b.id,
            ChatDictionaryUpdate(included_dictionary_ids=[dictionary_a.id]),
            db=chacha_db,
        )

    assert exc_info.value.status_code == 400
    assert "cycle" in str(exc_info.value.detail).lower()


@pytest.mark.asyncio
async def test_process_text_with_dictionary_includes_applies_base_then_child(
    chacha_db: CharactersRAGDB,
):
    base_dictionary = await chat_endpoints.create_chat_dictionary(
        ChatDictionaryCreate(name="Include Base Dictionary"),
        db=chacha_db,
    )
    child_dictionary = await chat_endpoints.create_chat_dictionary(
        ChatDictionaryCreate(
            name="Include Child Dictionary",
            included_dictionary_ids=[base_dictionary.id],
        ),
        db=chacha_db,
    )

    await chat_endpoints.add_dictionary_entry(
        base_dictionary.id,
        DictionaryEntryCreate(pattern="token", replacement="base"),
        db=chacha_db,
    )
    await chat_endpoints.add_dictionary_entry(
        child_dictionary.id,
        DictionaryEntryCreate(pattern="base", replacement="child"),
        db=chacha_db,
    )

    response = await chat_endpoints.process_text_with_dictionaries(
        ProcessTextRequest(
            text="token",
            dictionary_id=child_dictionary.id,
            max_iterations=2,
        ),
        db=chacha_db,
    )

    assert response.processed_text == "child"
    assert response.replacements >= 2


@pytest.mark.asyncio
async def test_export_import_json_preserves_included_dictionary_ids_when_targets_exist(
    chacha_db: CharactersRAGDB,
):
    base_dictionary = await chat_endpoints.create_chat_dictionary(
        ChatDictionaryCreate(name="Roundtrip Base Dictionary"),
        db=chacha_db,
    )
    composed_dictionary = await chat_endpoints.create_chat_dictionary(
        ChatDictionaryCreate(
            name="Roundtrip Composed Dictionary",
            included_dictionary_ids=[base_dictionary.id],
        ),
        db=chacha_db,
    )

    await chat_endpoints.add_dictionary_entry(
        composed_dictionary.id,
        DictionaryEntryCreate(pattern="x", replacement="y"),
        db=chacha_db,
    )

    exported = await chat_endpoints.export_dictionary_json(
        composed_dictionary.id,
        db=chacha_db,
    )
    assert exported.included_dictionary_ids == [base_dictionary.id]

    import_payload = exported.model_dump()
    import_payload["name"] = "Roundtrip Imported Dictionary"
    imported = await chat_endpoints.import_dictionary_json(
        ImportDictionaryJSONRequest(data=import_payload, activate=True),
        db=chacha_db,
    )
    imported_dictionary = await chat_endpoints.get_chat_dictionary(imported.dictionary_id, db=chacha_db)
    assert imported_dictionary.included_dictionary_ids == [base_dictionary.id]


@pytest.mark.asyncio
async def test_dictionary_version_history_endpoints_list_and_read_snapshots(
    chacha_db: CharactersRAGDB,
):
    service = ChatDictionaryService(chacha_db)
    dictionary_id = service.create_dictionary("Version History Dictionary", "initial")
    service.add_entry(dictionary_id, pattern="term", replacement="TERM")

    await chat_endpoints.update_chat_dictionary(
        dictionary_id,
        ChatDictionaryUpdate(description="updated description"),
        db=chacha_db,
    )

    versions_response = await chat_endpoints.list_dictionary_versions(
        dictionary_id,
        limit=20,
        offset=0,
        db=chacha_db,
    )
    assert versions_response.dictionary_id == dictionary_id
    assert versions_response.total >= 3
    assert versions_response.pagination.total == versions_response.total
    assert versions_response.pagination.limit == 20
    assert versions_response.pagination.offset == 0
    assert versions_response.pagination.has_more is False
    assert versions_response.pagination.next_offset is None
    assert versions_response.has_more is False
    assert versions_response.next_offset is None
    assert len(versions_response.versions) >= 3
    latest_revision = versions_response.versions[0].revision
    assert latest_revision >= 1

    detail_response = await chat_endpoints.get_dictionary_version(
        dictionary_id,
        latest_revision,
        db=chacha_db,
    )
    assert detail_response.dictionary_id == dictionary_id
    assert detail_response.revision == latest_revision
    assert detail_response.dictionary.id == dictionary_id
    assert detail_response.dictionary.entry_count == len(detail_response.entries)
    assert any(entry.pattern == "term" for entry in detail_response.entries)


@pytest.mark.asyncio
async def test_revert_dictionary_version_restores_prior_entry_state(
    chacha_db: CharactersRAGDB,
):
    service = ChatDictionaryService(chacha_db)
    dictionary_id = service.create_dictionary("Revertable Dictionary", "baseline")
    service.add_entry(dictionary_id, pattern="alpha", replacement="A")

    first_versions = await chat_endpoints.list_dictionary_versions(
        dictionary_id,
        limit=20,
        offset=0,
        db=chacha_db,
    )
    revision_with_single_entry = max(
        version.revision for version in first_versions.versions if version.entry_count == 1
    )

    service.add_entry(dictionary_id, pattern="beta", replacement="B")
    entries_before_revert = await chat_endpoints.list_dictionary_entries(
        dictionary_id,
        group=None,
        db=chacha_db,
    )
    assert entries_before_revert.total == 2

    revert_response = await chat_endpoints.revert_dictionary_version(
        dictionary_id,
        revision_with_single_entry,
        db=chacha_db,
    )
    assert revert_response.dictionary_id == dictionary_id
    assert revert_response.reverted_to_revision == revision_with_single_entry
    assert revert_response.current_dictionary_version >= 1
    assert revert_response.current_revision > revision_with_single_entry

    entries_after_revert = await chat_endpoints.list_dictionary_entries(
        dictionary_id,
        group=None,
        db=chacha_db,
    )
    assert entries_after_revert.total == 1
    assert entries_after_revert.entries[0].pattern == "alpha"


@pytest.mark.asyncio
async def test_revert_dictionary_version_restores_included_dictionary_ids(
    chacha_db: CharactersRAGDB,
):
    base_dictionary = await chat_endpoints.create_chat_dictionary(
        ChatDictionaryCreate(name="Revert Includes Base Dictionary"),
        db=chacha_db,
    )
    composed_dictionary = await chat_endpoints.create_chat_dictionary(
        ChatDictionaryCreate(
            name="Revert Includes Composed Dictionary",
            included_dictionary_ids=[base_dictionary.id],
        ),
        db=chacha_db,
    )
    await chat_endpoints.update_chat_dictionary(
        composed_dictionary.id,
        ChatDictionaryUpdate(included_dictionary_ids=[]),
        db=chacha_db,
    )
    after_clear = await chat_endpoints.get_chat_dictionary(composed_dictionary.id, db=chacha_db)
    assert after_clear.included_dictionary_ids == []

    versions_response = await chat_endpoints.list_dictionary_versions(
        composed_dictionary.id,
        limit=20,
        offset=0,
        db=chacha_db,
    )
    revision_with_include = None
    for version in versions_response.versions:
        detail = await chat_endpoints.get_dictionary_version(
            composed_dictionary.id,
            version.revision,
            db=chacha_db,
        )
        if detail.dictionary.included_dictionary_ids == [base_dictionary.id]:
            revision_with_include = version.revision
            break

    assert revision_with_include is not None
    await chat_endpoints.revert_dictionary_version(
        composed_dictionary.id,
        revision_with_include,
        db=chacha_db,
    )
    reverted = await chat_endpoints.get_chat_dictionary(composed_dictionary.id, db=chacha_db)
    assert reverted.included_dictionary_ids == [base_dictionary.id]
