from __future__ import annotations

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import chat_dictionaries as chat_dictionary_endpoints
from tldw_Server_API.app.api.v1.schemas.chat_dictionary_schemas import (
    ChatDictionaryCreate,
    ChatDictionaryUpdate,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDBError,
    ConflictError,
    InputError,
)


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self):
        self.error_calls = []
        self.warning_calls = []
        self.exception_calls = []

    def error(self, *args, **kwargs):
        self.error_calls.append((args, kwargs))

    def warning(self, *args, **kwargs):
        self.warning_calls.append((args, kwargs))

    def exception(self, *args, **kwargs):
        self.exception_calls.append((args, kwargs))


_SENSITIVE_MARKERS = (
    "driver failed",
    "/private/tmp/chat-dictionaries.db",
)


def _database_failure() -> CharactersRAGDBError:
    return CharactersRAGDBError("driver failed /private/tmp/chat-dictionaries.db")


def _unexpected_failure() -> RuntimeError:
    return RuntimeError("driver failed /private/tmp/chat-dictionaries.db")


def _patch_endpoint_logger(monkeypatch: pytest.MonkeyPatch) -> _LoggerStub:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(chat_dictionary_endpoints, "logger", logger_stub, raising=True)
    return logger_stub


def _assert_sanitized_log_call(
    logger_stub: _LoggerStub,
    expected_message: str,
    *,
    level: str = "error",
) -> None:
    assert logger_stub.exception_calls == []
    calls = getattr(logger_stub, f"{level}_calls")
    assert calls

    matching_messages = [args[0] for args, _kwargs in calls if args]
    assert expected_message in matching_messages

    rendered_calls = repr(calls)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered_calls


def _patch_service(monkeypatch: pytest.MonkeyPatch, **methods) -> None:
    class _Service:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    for name, method in methods.items():
        setattr(_Service, name, method)

    monkeypatch.setattr(
        chat_dictionary_endpoints,
        "ChatDictionaryService",
        _Service,
        raising=True,
    )


def _build_create_payload() -> ChatDictionaryCreate:
    return ChatDictionaryCreate(name="Dictionary", description="desc")


def _build_update_payload() -> ChatDictionaryUpdate:
    return ChatDictionaryUpdate(name="Updated Dictionary")


def _build_entry_create_payload() -> chat_dictionary_endpoints.DictionaryEntryCreate:
    return chat_dictionary_endpoints.DictionaryEntryCreate(
        pattern="hello",
        replacement="world",
    )


def _build_entry_update_payload() -> chat_dictionary_endpoints.DictionaryEntryUpdate:
    return chat_dictionary_endpoints.DictionaryEntryUpdate(replacement="updated")


@pytest.mark.asyncio
async def test_create_chat_dictionary_maps_conflict_error(monkeypatch: pytest.MonkeyPatch):
    def _create_dictionary(self, **_kwargs):
        raise ConflictError("Dictionary already exists")

    _patch_service(monkeypatch, create_dictionary=_create_dictionary)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.create_chat_dictionary(
            _build_create_payload(),
            db=object(),
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Dictionary already exists"


@pytest.mark.asyncio
async def test_create_chat_dictionary_maps_input_error(monkeypatch: pytest.MonkeyPatch):
    def _create_dictionary(self, **_kwargs):
        raise InputError("Dictionary name is invalid")

    _patch_service(monkeypatch, create_dictionary=_create_dictionary)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.create_chat_dictionary(
            _build_create_payload(),
            db=object(),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Dictionary name is invalid"


@pytest.mark.asyncio
async def test_create_chat_dictionary_maps_database_error(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _create_dictionary(self, **_kwargs):
        raise _database_failure()

    _patch_service(monkeypatch, create_dictionary=_create_dictionary)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.create_chat_dictionary(
            _build_create_payload(),
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to create dictionary"
    _assert_sanitized_log_call(logger_stub, "Error creating dictionary")


@pytest.mark.asyncio
async def test_create_chat_dictionary_sanitizes_unexpected_error_log(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _create_dictionary(self, **_kwargs):
        raise _unexpected_failure()

    _patch_service(monkeypatch, create_dictionary=_create_dictionary)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.create_chat_dictionary(
            _build_create_payload(),
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to create dictionary"
    _assert_sanitized_log_call(logger_stub, "Error creating dictionary")


@pytest.mark.asyncio
async def test_list_chat_dictionaries_maps_database_error(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _list_dictionaries_with_entry_counts(self, **_kwargs):
        raise _database_failure()

    _patch_service(monkeypatch, list_dictionaries_with_entry_counts=_list_dictionaries_with_entry_counts)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.list_chat_dictionaries(
            include_inactive=False,
            include_usage=False,
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list dictionaries"
    _assert_sanitized_log_call(logger_stub, "Error listing dictionaries")


@pytest.mark.asyncio
async def test_list_chat_dictionaries_sanitizes_unexpected_error_log(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _list_dictionaries_with_entry_counts(self, **_kwargs):
        raise _unexpected_failure()

    _patch_service(monkeypatch, list_dictionaries_with_entry_counts=_list_dictionaries_with_entry_counts)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.list_chat_dictionaries(
            include_inactive=False,
            include_usage=False,
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list dictionaries"
    _assert_sanitized_log_call(logger_stub, "Error listing dictionaries")


@pytest.mark.asyncio
async def test_get_chat_dictionary_maps_database_error(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _get_dictionary(self, *_args, **_kwargs):
        raise _database_failure()

    _patch_service(monkeypatch, get_dictionary=_get_dictionary)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.get_chat_dictionary(
            dictionary_id=42,
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to get dictionary"
    _assert_sanitized_log_call(logger_stub, "Error getting dictionary")


@pytest.mark.asyncio
async def test_get_chat_dictionary_sanitizes_unexpected_error_log(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _get_dictionary(self, *_args, **_kwargs):
        raise _unexpected_failure()

    _patch_service(monkeypatch, get_dictionary=_get_dictionary)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.get_chat_dictionary(
            dictionary_id=42,
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to get dictionary"
    _assert_sanitized_log_call(logger_stub, "Error getting dictionary")


@pytest.mark.asyncio
async def test_update_chat_dictionary_maps_conflict_error(monkeypatch: pytest.MonkeyPatch):
    def _update_dictionary(self, *_args, **_kwargs):
        raise ConflictError("Dictionary version conflict")

    _patch_service(monkeypatch, update_dictionary=_update_dictionary)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.update_chat_dictionary(
            dictionary_id=42,
            update=_build_update_payload(),
            db=object(),
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Dictionary version conflict"


@pytest.mark.asyncio
async def test_update_chat_dictionary_maps_input_error(monkeypatch: pytest.MonkeyPatch):
    def _update_dictionary(self, *_args, **_kwargs):
        raise InputError("Included dictionary cannot include itself")

    _patch_service(monkeypatch, update_dictionary=_update_dictionary)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.update_chat_dictionary(
            dictionary_id=42,
            update=_build_update_payload(),
            db=object(),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Included dictionary cannot include itself"


@pytest.mark.asyncio
async def test_update_chat_dictionary_maps_database_error(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _update_dictionary(self, *_args, **_kwargs):
        raise _database_failure()

    _patch_service(monkeypatch, update_dictionary=_update_dictionary)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.update_chat_dictionary(
            dictionary_id=42,
            update=_build_update_payload(),
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to update dictionary"
    _assert_sanitized_log_call(logger_stub, "Error updating dictionary")


@pytest.mark.asyncio
async def test_update_chat_dictionary_sanitizes_unexpected_error_log(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _update_dictionary(self, *_args, **_kwargs):
        raise _unexpected_failure()

    _patch_service(monkeypatch, update_dictionary=_update_dictionary)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.update_chat_dictionary(
            dictionary_id=42,
            update=_build_update_payload(),
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to update dictionary"
    _assert_sanitized_log_call(logger_stub, "Error updating dictionary")


@pytest.mark.asyncio
async def test_delete_chat_dictionary_maps_database_error(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _delete_dictionary(self, *_args, **_kwargs):
        raise _database_failure()

    _patch_service(monkeypatch, delete_dictionary=_delete_dictionary)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.delete_chat_dictionary(
            dictionary_id=42,
            hard_delete=False,
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to delete dictionary"
    _assert_sanitized_log_call(logger_stub, "Error deleting dictionary")


@pytest.mark.asyncio
async def test_delete_chat_dictionary_sanitizes_unexpected_error_log(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _delete_dictionary(self, *_args, **_kwargs):
        raise _unexpected_failure()

    _patch_service(monkeypatch, delete_dictionary=_delete_dictionary)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.delete_chat_dictionary(
            dictionary_id=42,
            hard_delete=False,
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to delete dictionary"
    _assert_sanitized_log_call(logger_stub, "Error deleting dictionary")


@pytest.mark.asyncio
async def test_add_dictionary_entry_maps_input_error(monkeypatch: pytest.MonkeyPatch):
    def _get_dictionary(self, *_args, **_kwargs):
        return {"id": 42}

    def _add_entry(self, *_args, **_kwargs):
        raise InputError("Pattern is invalid")

    _patch_service(
        monkeypatch,
        get_dictionary=_get_dictionary,
        add_entry=_add_entry,
    )

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.add_dictionary_entry(
            dictionary_id=42,
            entry=_build_entry_create_payload(),
            db=object(),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Pattern is invalid"


@pytest.mark.asyncio
async def test_add_dictionary_entry_lookup_maps_database_error(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _get_dictionary(self, *_args, **_kwargs):
        raise _database_failure()

    _patch_service(monkeypatch, get_dictionary=_get_dictionary)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.add_dictionary_entry(
            dictionary_id=42,
            entry=_build_entry_create_payload(),
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to add dictionary entry"
    _assert_sanitized_log_call(logger_stub, "Error retrieving dictionary before adding entry")


@pytest.mark.asyncio
async def test_add_dictionary_entry_lookup_sanitizes_unexpected_error_log(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _get_dictionary(self, *_args, **_kwargs):
        raise _unexpected_failure()

    _patch_service(monkeypatch, get_dictionary=_get_dictionary)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.add_dictionary_entry(
            dictionary_id=42,
            entry=_build_entry_create_payload(),
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to add dictionary entry"
    _assert_sanitized_log_call(logger_stub, "Error retrieving dictionary before adding entry")


@pytest.mark.asyncio
async def test_add_dictionary_entry_maps_database_error(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _get_dictionary(self, *_args, **_kwargs):
        return {"id": 42}

    def _add_entry(self, *_args, **_kwargs):
        raise _database_failure()

    _patch_service(
        monkeypatch,
        get_dictionary=_get_dictionary,
        add_entry=_add_entry,
    )

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.add_dictionary_entry(
            dictionary_id=42,
            entry=_build_entry_create_payload(),
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to add dictionary entry"
    _assert_sanitized_log_call(logger_stub, "Error adding dictionary entry")


@pytest.mark.asyncio
async def test_add_dictionary_entry_sanitizes_unexpected_error_log(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _get_dictionary(self, *_args, **_kwargs):
        return {"id": 42}

    def _add_entry(self, *_args, **_kwargs):
        raise _unexpected_failure()

    _patch_service(
        monkeypatch,
        get_dictionary=_get_dictionary,
        add_entry=_add_entry,
    )

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.add_dictionary_entry(
            dictionary_id=42,
            entry=_build_entry_create_payload(),
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to add dictionary entry"
    _assert_sanitized_log_call(logger_stub, "Error adding dictionary entry")


@pytest.mark.asyncio
async def test_list_dictionary_entries_maps_database_error(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _get_dictionary(self, *_args, **_kwargs):
        raise _database_failure()

    _patch_service(monkeypatch, get_dictionary=_get_dictionary)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.list_dictionary_entries(
            dictionary_id=42,
            group=None,
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list dictionary entries"
    _assert_sanitized_log_call(logger_stub, "Error listing dictionary entries")


@pytest.mark.asyncio
async def test_list_dictionary_entries_sanitizes_unexpected_error_log(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _get_dictionary(self, *_args, **_kwargs):
        return {"id": 42}

    def _get_entries(self, *_args, **_kwargs):
        raise _unexpected_failure()

    _patch_service(
        monkeypatch,
        get_dictionary=_get_dictionary,
        get_entries=_get_entries,
    )

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.list_dictionary_entries(
            dictionary_id=42,
            group=None,
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list dictionary entries"
    _assert_sanitized_log_call(logger_stub, "Error listing dictionary entries")


@pytest.mark.asyncio
async def test_update_dictionary_entry_maps_input_error(monkeypatch: pytest.MonkeyPatch):
    def _update_entry(self, *_args, **_kwargs):
        raise InputError("Pattern is invalid")

    _patch_service(monkeypatch, update_entry=_update_entry)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.update_dictionary_entry(
            entry_id=42,
            update=_build_entry_update_payload(),
            db=object(),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Pattern is invalid"


@pytest.mark.asyncio
async def test_update_dictionary_entry_maps_database_error(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _update_entry(self, *_args, **_kwargs):
        raise _database_failure()

    _patch_service(monkeypatch, update_entry=_update_entry)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.update_dictionary_entry(
            entry_id=42,
            update=_build_entry_update_payload(),
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to update dictionary entry"
    _assert_sanitized_log_call(logger_stub, "Error updating dictionary entry")


@pytest.mark.asyncio
async def test_update_dictionary_entry_sanitizes_unexpected_error_log(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _update_entry(self, *_args, **_kwargs):
        raise _unexpected_failure()

    _patch_service(monkeypatch, update_entry=_update_entry)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.update_dictionary_entry(
            entry_id=42,
            update=_build_entry_update_payload(),
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to update dictionary entry"
    _assert_sanitized_log_call(logger_stub, "Error updating dictionary entry")


@pytest.mark.asyncio
async def test_update_dictionary_entry_regex_precheck_sanitizes_unexpected_error_log(
    monkeypatch: pytest.MonkeyPatch,
):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _get_entry(self, *_args, **_kwargs):
        raise _unexpected_failure()

    _patch_service(monkeypatch, get_entry=_get_entry)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.update_dictionary_entry(
            entry_id=42,
            update=chat_dictionary_endpoints.DictionaryEntryUpdate(type="regex"),
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Unable to validate regex pattern safety"
    _assert_sanitized_log_call(logger_stub, "Error checking existing entry type for regex validation")


@pytest.mark.asyncio
async def test_delete_dictionary_entry_maps_database_error(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _delete_entry(self, *_args, **_kwargs):
        raise _database_failure()

    _patch_service(monkeypatch, delete_entry=_delete_entry)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.delete_dictionary_entry(
            entry_id=42,
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to delete dictionary entry"
    _assert_sanitized_log_call(logger_stub, "Error deleting dictionary entry")


@pytest.mark.asyncio
async def test_delete_dictionary_entry_sanitizes_unexpected_error_log(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _delete_entry(self, *_args, **_kwargs):
        raise _unexpected_failure()

    _patch_service(monkeypatch, delete_entry=_delete_entry)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.delete_dictionary_entry(
            entry_id=42,
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to delete dictionary entry"
    _assert_sanitized_log_call(logger_stub, "Error deleting dictionary entry")


@pytest.mark.asyncio
async def test_process_text_with_dictionaries_maps_input_error(monkeypatch: pytest.MonkeyPatch):
    def _process_text(self, *_args, **_kwargs):
        raise InputError("Token budget is invalid")

    _patch_service(monkeypatch, process_text=_process_text)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.process_text_with_dictionaries(
            request=chat_dictionary_endpoints.ProcessTextRequest(text="hello"),
            db=object(),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Token budget is invalid"


@pytest.mark.asyncio
async def test_process_text_with_dictionaries_maps_database_error(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _process_text(self, *_args, **_kwargs):
        raise _database_failure()

    _patch_service(monkeypatch, process_text=_process_text)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.process_text_with_dictionaries(
            request=chat_dictionary_endpoints.ProcessTextRequest(text="hello"),
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to process text"
    _assert_sanitized_log_call(logger_stub, "Error processing text")


@pytest.mark.asyncio
async def test_process_text_with_dictionaries_sanitizes_unexpected_error_log(
    monkeypatch: pytest.MonkeyPatch,
):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _process_text(self, *_args, **_kwargs):
        raise _unexpected_failure()

    _patch_service(monkeypatch, process_text=_process_text)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.process_text_with_dictionaries(
            request=chat_dictionary_endpoints.ProcessTextRequest(text="hello"),
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to process text"
    _assert_sanitized_log_call(logger_stub, "Error processing text")


@pytest.mark.asyncio
async def test_import_dictionary_maps_conflict_error(monkeypatch: pytest.MonkeyPatch):
    def _import_from_markdown(self, *_args, **_kwargs):
        raise ConflictError("Dictionary already exists")

    _patch_service(monkeypatch, import_from_markdown=_import_from_markdown)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.import_dictionary(
            import_request=chat_dictionary_endpoints.ImportDictionaryRequest(
                name="Imported Dictionary",
                content="# dictionary",
            ),
            db=object(),
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Dictionary already exists"


@pytest.mark.asyncio
async def test_import_dictionary_maps_database_error(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _import_from_markdown(self, *_args, **_kwargs):
        raise _database_failure()

    _patch_service(monkeypatch, import_from_markdown=_import_from_markdown)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.import_dictionary(
            import_request=chat_dictionary_endpoints.ImportDictionaryRequest(
                name="Imported Dictionary",
                content="# dictionary",
            ),
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to import dictionary"
    _assert_sanitized_log_call(logger_stub, "Error importing dictionary")


@pytest.mark.asyncio
async def test_import_dictionary_sanitizes_unexpected_error_log(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _import_from_markdown(self, *_args, **_kwargs):
        raise _unexpected_failure()

    _patch_service(monkeypatch, import_from_markdown=_import_from_markdown)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.import_dictionary(
            import_request=chat_dictionary_endpoints.ImportDictionaryRequest(
                name="Imported Dictionary",
                content="# dictionary",
            ),
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to import dictionary"
    _assert_sanitized_log_call(logger_stub, "Error importing dictionary")


@pytest.mark.asyncio
async def test_export_dictionary_maps_input_error(monkeypatch: pytest.MonkeyPatch):
    def _get_dictionary(self, *_args, **_kwargs):
        return {"id": 42, "name": "Dictionary"}

    def _export_to_markdown(self, *_args, **_kwargs):
        raise InputError("Dictionary cannot be exported")

    _patch_service(
        monkeypatch,
        get_dictionary=_get_dictionary,
        export_to_markdown=_export_to_markdown,
    )

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.export_dictionary(
            dictionary_id=42,
            db=object(),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Dictionary cannot be exported"


@pytest.mark.asyncio
async def test_export_dictionary_maps_database_error(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _get_dictionary(self, *_args, **_kwargs):
        return {"id": 42, "name": "Dictionary"}

    def _export_to_markdown(self, *_args, **_kwargs):
        raise _database_failure()

    _patch_service(
        monkeypatch,
        get_dictionary=_get_dictionary,
        export_to_markdown=_export_to_markdown,
    )

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.export_dictionary(
            dictionary_id=42,
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to export dictionary"
    _assert_sanitized_log_call(logger_stub, "Error exporting dictionary")


@pytest.mark.asyncio
async def test_export_dictionary_sanitizes_unexpected_error_log(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _get_dictionary(self, *_args, **_kwargs):
        return {"id": 42, "name": "Dictionary"}

    def _export_to_markdown(self, *_args, **_kwargs):
        raise _unexpected_failure()

    _patch_service(
        monkeypatch,
        get_dictionary=_get_dictionary,
        export_to_markdown=_export_to_markdown,
    )

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.export_dictionary(
            dictionary_id=42,
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to export dictionary"
    _assert_sanitized_log_call(logger_stub, "Error exporting dictionary")


@pytest.mark.asyncio
async def test_export_dictionary_json_maps_database_error(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _export_to_json(self, *_args, **_kwargs):
        raise _database_failure()

    _patch_service(monkeypatch, export_to_json=_export_to_json)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.export_dictionary_json(
            dictionary_id=42,
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to export dictionary JSON"
    _assert_sanitized_log_call(logger_stub, "Error exporting dictionary JSON")


@pytest.mark.asyncio
async def test_export_dictionary_json_sanitizes_unexpected_error_log(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _export_to_json(self, *_args, **_kwargs):
        raise _unexpected_failure()

    _patch_service(monkeypatch, export_to_json=_export_to_json)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.export_dictionary_json(
            dictionary_id=42,
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to export dictionary JSON"
    _assert_sanitized_log_call(logger_stub, "Error exporting dictionary JSON")


@pytest.mark.asyncio
async def test_import_dictionary_json_maps_conflict_error(monkeypatch: pytest.MonkeyPatch):
    def _import_from_json(self, *_args, **_kwargs):
        raise ConflictError("Dictionary already exists")

    _patch_service(monkeypatch, import_from_json=_import_from_json)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.import_dictionary_json(
            import_request=chat_dictionary_endpoints.ImportDictionaryJSONRequest(
                data={"name": "Imported Dictionary", "entries": []},
            ),
            db=object(),
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Dictionary already exists"


@pytest.mark.asyncio
async def test_import_dictionary_json_maps_database_error(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _import_from_json(self, *_args, **_kwargs):
        raise _database_failure()

    _patch_service(monkeypatch, import_from_json=_import_from_json)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.import_dictionary_json(
            import_request=chat_dictionary_endpoints.ImportDictionaryJSONRequest(
                data={"name": "Imported Dictionary", "entries": []},
            ),
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to import dictionary JSON"
    _assert_sanitized_log_call(logger_stub, "Error importing dictionary JSON")


@pytest.mark.asyncio
async def test_import_dictionary_json_sanitizes_unexpected_error_log(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _import_from_json(self, *_args, **_kwargs):
        raise _unexpected_failure()

    _patch_service(monkeypatch, import_from_json=_import_from_json)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.import_dictionary_json(
            import_request=chat_dictionary_endpoints.ImportDictionaryJSONRequest(
                data={"name": "Imported Dictionary", "entries": []},
            ),
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to import dictionary JSON"
    _assert_sanitized_log_call(logger_stub, "Error importing dictionary JSON")


@pytest.mark.asyncio
async def test_bulk_dictionary_entry_operations_treats_input_error_as_partial_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    def _update_entry(self, entry_id, **_kwargs):
        if entry_id == 2:
            raise InputError("entry failed")
        return True

    _patch_service(monkeypatch, update_entry=_update_entry)

    response = await chat_dictionary_endpoints.bulk_dictionary_entry_operations(
        operation=chat_dictionary_endpoints.BulkEntryOperation(
            entry_ids=[1, 2],
            operation="activate",
        ),
        db=object(),
    )

    assert response.success is False
    assert response.affected_count == 1
    assert response.failed_ids == [2]


@pytest.mark.asyncio
async def test_bulk_dictionary_entry_operations_treats_db_error_as_partial_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _update_entry(self, entry_id, **_kwargs):
        if entry_id == 2:
            raise _database_failure()
        return True

    _patch_service(monkeypatch, update_entry=_update_entry)

    response = await chat_dictionary_endpoints.bulk_dictionary_entry_operations(
        operation=chat_dictionary_endpoints.BulkEntryOperation(
            entry_ids=[1, 2],
            operation="activate",
        ),
        db=object(),
    )

    assert response.success is False
    assert response.affected_count == 1
    assert response.failed_ids == [2]
    _assert_sanitized_log_call(
        logger_stub,
        "Bulk dictionary entry operation failed",
        level="warning",
    )


@pytest.mark.asyncio
async def test_bulk_dictionary_entry_operations_sanitizes_unexpected_error_log(
    monkeypatch: pytest.MonkeyPatch,
):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    class _FailingBulkOperation:
        operation = "activate"
        group_name = None

        @property
        def entry_ids(self):
            raise _unexpected_failure()

    _patch_service(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.bulk_dictionary_entry_operations(
            operation=_FailingBulkOperation(),
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to perform bulk entry operation"
    _assert_sanitized_log_call(logger_stub, "Error performing bulk entry operation")


@pytest.mark.asyncio
async def test_reorder_dictionary_entries_maps_input_error(monkeypatch: pytest.MonkeyPatch):
    def _get_dictionary(self, *_args, **_kwargs):
        return {"id": 42}

    def _reorder_entries(self, *_args, **_kwargs):
        raise InputError("Every dictionary entry exactly once")

    _patch_service(
        monkeypatch,
        get_dictionary=_get_dictionary,
        reorder_entries=_reorder_entries,
    )

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.reorder_dictionary_entries(
            dictionary_id=42,
            reorder_request=chat_dictionary_endpoints.DictionaryEntryReorderRequest(entry_ids=[1]),
            db=object(),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Every dictionary entry exactly once"


@pytest.mark.asyncio
async def test_reorder_dictionary_entries_maps_database_error(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _get_dictionary(self, *_args, **_kwargs):
        return {"id": 42}

    def _reorder_entries(self, *_args, **_kwargs):
        raise _database_failure()

    _patch_service(
        monkeypatch,
        get_dictionary=_get_dictionary,
        reorder_entries=_reorder_entries,
    )

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.reorder_dictionary_entries(
            dictionary_id=42,
            reorder_request=chat_dictionary_endpoints.DictionaryEntryReorderRequest(entry_ids=[1]),
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to reorder dictionary entries"
    _assert_sanitized_log_call(logger_stub, "Error reordering dictionary entries")


@pytest.mark.asyncio
async def test_reorder_dictionary_entries_sanitizes_unexpected_error_log(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _get_dictionary(self, *_args, **_kwargs):
        return {"id": 42}

    def _reorder_entries(self, *_args, **_kwargs):
        raise _unexpected_failure()

    _patch_service(
        monkeypatch,
        get_dictionary=_get_dictionary,
        reorder_entries=_reorder_entries,
    )

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.reorder_dictionary_entries(
            dictionary_id=42,
            reorder_request=chat_dictionary_endpoints.DictionaryEntryReorderRequest(entry_ids=[1]),
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to reorder dictionary entries"
    _assert_sanitized_log_call(logger_stub, "Error reordering dictionary entries")


@pytest.mark.asyncio
async def test_list_dictionary_activity_maps_database_error(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _get_dictionary(self, *_args, **_kwargs):
        return {"id": 42}

    def _list_transform_activity(self, *_args, **_kwargs):
        raise _database_failure()

    _patch_service(
        monkeypatch,
        get_dictionary=_get_dictionary,
        list_transform_activity=_list_transform_activity,
    )

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.list_dictionary_activity(
            dictionary_id=42,
            limit=10,
            offset=0,
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list dictionary activity"
    _assert_sanitized_log_call(logger_stub, "Error listing dictionary activity")


@pytest.mark.asyncio
async def test_list_dictionary_activity_sanitizes_unexpected_error_log(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _get_dictionary(self, *_args, **_kwargs):
        return {"id": 42}

    def _list_transform_activity(self, *_args, **_kwargs):
        raise _unexpected_failure()

    _patch_service(
        monkeypatch,
        get_dictionary=_get_dictionary,
        list_transform_activity=_list_transform_activity,
    )

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.list_dictionary_activity(
            dictionary_id=42,
            limit=10,
            offset=0,
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list dictionary activity"
    _assert_sanitized_log_call(logger_stub, "Error listing dictionary activity")


@pytest.mark.asyncio
async def test_list_dictionary_versions_maps_database_error(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _get_dictionary(self, *_args, **_kwargs):
        return {"id": 42}

    def _list_dictionary_versions(self, *_args, **_kwargs):
        raise _database_failure()

    _patch_service(
        monkeypatch,
        get_dictionary=_get_dictionary,
        list_dictionary_versions=_list_dictionary_versions,
    )

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.list_dictionary_versions(
            dictionary_id=42,
            limit=10,
            offset=0,
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list dictionary versions"
    _assert_sanitized_log_call(logger_stub, "Error listing dictionary versions")


@pytest.mark.asyncio
async def test_list_dictionary_versions_sanitizes_unexpected_error_log(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _get_dictionary(self, *_args, **_kwargs):
        return {"id": 42}

    def _list_dictionary_versions(self, *_args, **_kwargs):
        raise _unexpected_failure()

    _patch_service(
        monkeypatch,
        get_dictionary=_get_dictionary,
        list_dictionary_versions=_list_dictionary_versions,
    )

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.list_dictionary_versions(
            dictionary_id=42,
            limit=10,
            offset=0,
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list dictionary versions"
    _assert_sanitized_log_call(logger_stub, "Error listing dictionary versions")


@pytest.mark.asyncio
async def test_get_dictionary_version_maps_database_error(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _get_dictionary_version(self, *_args, **_kwargs):
        raise _database_failure()

    _patch_service(monkeypatch, get_dictionary_version=_get_dictionary_version)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.get_dictionary_version(
            dictionary_id=42,
            revision=3,
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to read dictionary revision"
    _assert_sanitized_log_call(logger_stub, "Error reading dictionary revision")


@pytest.mark.asyncio
async def test_get_dictionary_version_sanitizes_unexpected_error_log(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _get_dictionary_version(self, *_args, **_kwargs):
        raise _unexpected_failure()

    _patch_service(monkeypatch, get_dictionary_version=_get_dictionary_version)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.get_dictionary_version(
            dictionary_id=42,
            revision=3,
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to read dictionary revision"
    _assert_sanitized_log_call(logger_stub, "Error reading dictionary revision")


@pytest.mark.asyncio
async def test_revert_dictionary_version_maps_conflict_error(monkeypatch: pytest.MonkeyPatch):
    def _revert_dictionary_to_revision(self, *_args, **_kwargs):
        raise ConflictError("Revision conflict")

    _patch_service(monkeypatch, revert_dictionary_to_revision=_revert_dictionary_to_revision)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.revert_dictionary_version(
            dictionary_id=42,
            revision=3,
            db=object(),
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Revision conflict"


@pytest.mark.asyncio
async def test_revert_dictionary_version_maps_not_found_input_error(monkeypatch: pytest.MonkeyPatch):
    def _revert_dictionary_to_revision(self, *_args, **_kwargs):
        raise InputError("Dictionary revision not found")

    _patch_service(monkeypatch, revert_dictionary_to_revision=_revert_dictionary_to_revision)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.revert_dictionary_version(
            dictionary_id=42,
            revision=3,
            db=object(),
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Dictionary revision not found"


@pytest.mark.asyncio
async def test_revert_dictionary_version_maps_validation_input_error(monkeypatch: pytest.MonkeyPatch):
    def _revert_dictionary_to_revision(self, *_args, **_kwargs):
        raise InputError("Revision must be positive")

    _patch_service(monkeypatch, revert_dictionary_to_revision=_revert_dictionary_to_revision)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.revert_dictionary_version(
            dictionary_id=42,
            revision=3,
            db=object(),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Revision must be positive"


@pytest.mark.asyncio
async def test_revert_dictionary_version_maps_database_error(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _revert_dictionary_to_revision(self, *_args, **_kwargs):
        raise _database_failure()

    _patch_service(monkeypatch, revert_dictionary_to_revision=_revert_dictionary_to_revision)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.revert_dictionary_version(
            dictionary_id=42,
            revision=3,
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to revert dictionary revision"
    _assert_sanitized_log_call(logger_stub, "Error reverting dictionary revision")


@pytest.mark.asyncio
async def test_revert_dictionary_version_sanitizes_unexpected_error_log(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _revert_dictionary_to_revision(self, *_args, **_kwargs):
        raise _unexpected_failure()

    _patch_service(monkeypatch, revert_dictionary_to_revision=_revert_dictionary_to_revision)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.revert_dictionary_version(
            dictionary_id=42,
            revision=3,
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to revert dictionary revision"
    _assert_sanitized_log_call(logger_stub, "Error reverting dictionary revision")


@pytest.mark.asyncio
async def test_get_dictionary_statistics_returns_404_when_dictionary_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    def _get_dictionary(self, *_args, **_kwargs):
        return None

    _patch_service(monkeypatch, get_dictionary=_get_dictionary)

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.get_dictionary_statistics(
            dictionary_id=42,
            db=object(),
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Dictionary not found"


@pytest.mark.asyncio
async def test_get_dictionary_statistics_maps_input_error(monkeypatch: pytest.MonkeyPatch):
    def _get_dictionary(self, *_args, **_kwargs):
        return {"id": 42}

    def _get_statistics(self, *_args, **_kwargs):
        raise InputError("Statistics filters are invalid")

    _patch_service(
        monkeypatch,
        get_dictionary=_get_dictionary,
        get_statistics=_get_statistics,
    )

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.get_dictionary_statistics(
            dictionary_id=42,
            db=object(),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Statistics filters are invalid"


@pytest.mark.asyncio
async def test_get_dictionary_statistics_maps_database_error(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _get_dictionary(self, *_args, **_kwargs):
        return {"id": 42}

    def _get_statistics(self, *_args, **_kwargs):
        raise _database_failure()

    _patch_service(
        monkeypatch,
        get_dictionary=_get_dictionary,
        get_statistics=_get_statistics,
    )

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.get_dictionary_statistics(
            dictionary_id=42,
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to get dictionary statistics"
    _assert_sanitized_log_call(logger_stub, "Error getting dictionary statistics")


@pytest.mark.asyncio
async def test_get_dictionary_statistics_sanitizes_unexpected_error_log(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _get_dictionary(self, *_args, **_kwargs):
        return {"id": 42}

    def _get_statistics(self, *_args, **_kwargs):
        raise _unexpected_failure()

    _patch_service(
        monkeypatch,
        get_dictionary=_get_dictionary,
        get_statistics=_get_statistics,
    )

    with pytest.raises(HTTPException) as exc_info:
        await chat_dictionary_endpoints.get_dictionary_statistics(
            dictionary_id=42,
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to get dictionary statistics"
    _assert_sanitized_log_call(logger_stub, "Error getting dictionary statistics")
