"""Dictionary UTC storage timestamps remain unambiguous in API responses."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.endpoints import chat_dictionaries
from tldw_Server_API.app.api.v1.schemas.chat_dictionary_schemas import (
    ChatDictionaryCreate,
    ChatDictionaryResponse,
    ChatDictionaryUpdate,
)
from tldw_Server_API.app.core.Character_Chat.chat_dictionary import ChatDictionaryService
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Utils.iso_datetime import parse_iso_utc

pytestmark = pytest.mark.unit
EXPECTED = datetime(2026, 10, 8, 15, 20, tzinfo=timezone.utc)


@pytest.mark.parametrize("value", [
    "2026-10-08 15:20:00",
    "2026-10-08T15:20:00",
    "2026-10-08T15:20:00Z",
    "2026-10-08T17:20:00+02:00",
    datetime(2026, 10, 8, 15, 20),
    datetime(2026, 10, 8, 17, 20, tzinfo=timezone(timedelta(hours=2))),
    EXPECTED.timestamp(),
])
def test_dictionary_response_timestamps_serialize_as_aware_utc(value):
    response = ChatDictionaryResponse(id=1, name="B03 timestamp", version=1, created_at=value, updated_at=value)

    assert response.created_at == EXPECTED  # nosec B101
    assert response.updated_at == EXPECTED  # nosec B101
    assert response.model_dump(mode="json")["created_at"] == "2026-10-08T15:20:00Z"  # nosec B101
    assert response.model_dump(mode="json")["updated_at"] == "2026-10-08T15:20:00Z"  # nosec B101


@pytest.mark.parametrize("value", [None, "invalid-date", "2026-99-08 15:20:00"])
def test_dictionary_response_rejects_invalid_required_timestamps(value):
    with pytest.raises(ValidationError):
        ChatDictionaryResponse(id=1, name="B03 invalid", version=1, created_at=value, updated_at=value)


@pytest.mark.parametrize("missing", ["created_at", "updated_at"])
def test_dictionary_response_keeps_required_timestamp_fields(missing):
    payload = {"id": 1, "name": "B03 missing", "version": 1, "created_at": EXPECTED, "updated_at": EXPECTED}
    payload.pop(missing)
    with pytest.raises(ValidationError):
        ChatDictionaryResponse(**payload)


@pytest.fixture
def dictionary_db(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "dictionary-timestamps.db"), client_id="timestamp-test", owner_user_id="1")
    try:
        yield db
    finally:
        db.close_all_connections()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["create", "list", "detail", "update"])
async def test_real_dictionary_endpoint_timestamps_preserve_sqlite_utc_instant(dictionary_db, surface):
    created = await chat_dictionaries.create_chat_dictionary(ChatDictionaryCreate(name="B03 newly created"), db=dictionary_db)
    if surface == "create":
        response = created
    elif surface == "list":
        listed = await chat_dictionaries.list_chat_dictionaries(include_inactive=False, include_usage=False, db=dictionary_db)
        response = next(item for item in listed.dictionaries if item.id == created.id)
    elif surface == "detail":
        response = await chat_dictionaries.get_chat_dictionary(created.id, db=dictionary_db)
    else:
        response = await chat_dictionaries.update_chat_dictionary(
            created.id, ChatDictionaryUpdate(description="Just updated", version=created.version), db=dictionary_db,
        )
    stored = ChatDictionaryService(dictionary_db).get_dictionary(created.id)

    for field in ("created_at", "updated_at"):
        assert getattr(response, field) == parse_iso_utc(stored[field])  # nosec B101
        assert response.model_dump(mode="json")[field].endswith("Z")  # nosec B101
