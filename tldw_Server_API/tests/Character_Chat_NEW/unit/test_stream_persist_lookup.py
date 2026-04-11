import pytest

from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions as sessions


class _BatchMetadataDB:
    def __init__(self) -> None:
        self.batch_calls: list[list[str]] = []
        self.single_calls: list[str] = []

    def get_messages_for_conversation(
        self,
        conversation_id: str,
        limit: int = 100,
        offset: int = 0,
        order_by_timestamp: str = "DESC",
    ) -> list[dict[str, object]]:
        assert conversation_id == "chat-1"
        assert limit == 100
        assert offset == 0
        assert order_by_timestamp == "DESC"
        return [
            {"id": "assistant-1", "sender": "Alpha", "deleted": False},
            {"id": "user-1", "sender": "user", "deleted": False},
        ]

    def get_message_metadata_map(self, message_ids: list[str]) -> dict[str, dict[str, object]]:
        self.batch_calls.append(message_ids)
        return {
            "assistant-1": {
                "extra": {
                    "stream_persist_fingerprint": "fingerprint-1",
                    "persist_validation_degraded": False,
                }
            }
        }

    def get_message_metadata(self, message_id: str) -> dict[str, object] | None:
        self.single_calls.append(message_id)
        return None


class _LegacyMetadataDB:
    def __init__(self) -> None:
        self.single_calls: list[str] = []

    def get_messages_for_conversation(
        self,
        conversation_id: str,
        limit: int = 100,
        offset: int = 0,
        order_by_timestamp: str = "DESC",
    ) -> list[dict[str, object]]:
        assert conversation_id == "chat-1"
        assert limit == 100
        assert offset == 0
        assert order_by_timestamp == "DESC"
        return [
            {"id": "assistant-1", "sender": "Alpha", "deleted": False},
            {"id": "assistant-2", "sender": "Beta", "deleted": False},
        ]

    def get_message_metadata(self, message_id: str) -> dict[str, object] | None:
        self.single_calls.append(message_id)
        if message_id == "assistant-2":
            return {"extra": {"stream_persist_fingerprint": "fingerprint-2"}}
        return {"extra": {}}


@pytest.mark.unit
def test_find_existing_stream_persist_message_uses_batch_metadata_lookup() -> None:
    db = _BatchMetadataDB()

    result = sessions._find_existing_stream_persist_message(
        db,
        "chat-1",
        "fingerprint-1",
    )

    assert result == (
        "assistant-1",
        {
            "stream_persist_fingerprint": "fingerprint-1",
            "persist_validation_degraded": False,
        },
    )
    assert db.batch_calls == [["assistant-1"]]
    assert db.single_calls == []


@pytest.mark.unit
def test_find_existing_stream_persist_message_falls_back_to_single_message_lookup() -> None:
    db = _LegacyMetadataDB()

    result = sessions._find_existing_stream_persist_message(
        db,
        "chat-1",
        "fingerprint-2",
    )

    assert result == (
        "assistant-2",
        {"stream_persist_fingerprint": "fingerprint-2"},
    )
    assert db.single_calls == ["assistant-1", "assistant-2"]
