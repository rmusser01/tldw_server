"""Persistence boundary for realtime speech sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class RealtimePersistenceConfig:
    enabled: bool
    conversation_id: str | None
    store_raw_audio: bool = False


class RealtimePersistenceAdapter(Protocol):
    async def write_turn(
        self,
        *,
        conversation_id: str,
        session_id: str,
        turn_index: int,
        user_transcript: str,
        assistant_text: str,
    ) -> None:
        raise NotImplementedError


class NoopRealtimePersistenceAdapter:
    async def write_turn(
        self,
        *,
        conversation_id: str,
        session_id: str,
        turn_index: int,
        user_transcript: str,
        assistant_text: str,
    ) -> None:
        return None


def persistence_config_from_metadata(metadata: dict[str, Any]) -> RealtimePersistenceConfig:
    tldw_metadata = metadata.get("tldw")
    if not isinstance(tldw_metadata, dict):
        return RealtimePersistenceConfig(enabled=False, conversation_id=None)

    conversation_id_value = tldw_metadata.get("conversation_id")
    conversation_id = conversation_id_value if isinstance(conversation_id_value, str) else None
    return RealtimePersistenceConfig(
        enabled=tldw_metadata.get("persist") is True,
        conversation_id=conversation_id,
        store_raw_audio=False,
    )
