from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations import flashcards_module
from tldw_Server_API.app.core.MCP_unified.modules.implementations.flashcards_module import (
    FlashcardsModule,
)


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.debugs.append(message)


class _CloseFailsDB:
    def __init__(self, sensitive_detail: str) -> None:
        self.sensitive_detail = sensitive_detail

    def close_all_connections(self) -> None:
        raise RuntimeError(self.sensitive_detail)

    def list_decks(self, **_kwargs: Any) -> list[dict[str, Any]]:
        return []

    def get_deck(self, _deck_id: int | None) -> dict[str, Any]:
        return {"id": 1, "workspace_id": None}

    def add_deck(self, **_kwargs: Any) -> int:
        return 1

    def list_flashcards(self, **_kwargs: Any) -> list[dict[str, Any]]:
        return []

    def count_flashcards(self, **_kwargs: Any) -> int:
        return 0

    def get_flashcard(self, card_uuid: str | None) -> dict[str, Any]:
        return {"uuid": card_uuid or "card-1", "workspace_id": None}

    def add_flashcard(self, _card_data: dict[str, Any]) -> str:
        return "card-1"

    def add_flashcards_bulk(self, cards_data: list[dict[str, Any]]) -> list[str]:
        return [f"card-{index}" for index, _card in enumerate(cards_data, start=1)]

    def update_flashcard(
        self,
        *,
        card_uuid: str,
        updates: dict[str, Any],
        expected_version: int | None,
        tags: list[str] | None,
    ) -> bool:
        return bool(card_uuid and updates is not None)

    def soft_delete_flashcard(self, card_uuid: str, expected_version: int | None) -> bool:
        return bool(card_uuid and expected_version)

    def review_flashcard(
        self,
        *,
        card_uuid: str,
        rating: int,
        answer_time_ms: int | None,
    ) -> dict[str, Any]:
        return {"card_uuid": card_uuid, "rating": rating, "answer_time_ms": answer_time_ms}

    def set_flashcard_tags(self, card_uuid: str, tags: list[str]) -> bool:
        return bool(card_uuid and tags is not None)

    def get_keywords_for_flashcard(self, _card_uuid: str) -> list[dict[str, str]]:
        return [{"keyword": "safe"}]

    def export_flashcards_csv(self, **_kwargs: Any) -> bytes:
        return b"front,back\n"


def _module_with_failing_close(
    monkeypatch: pytest.MonkeyPatch,
    sensitive_detail: str,
) -> tuple[FlashcardsModule, _LoggerStub]:
    module = FlashcardsModule(ModuleConfig(name="flashcards"))
    logger_stub = _LoggerStub()
    monkeypatch.setattr(flashcards_module, "logger", logger_stub)
    monkeypatch.setattr(module, "_open_db", lambda _context: _CloseFailsDB(sensitive_detail))
    return module, logger_stub


def _assert_close_fallback_log_is_sanitized(
    logger_stub: _LoggerStub,
    sensitive_detail: str,
) -> None:
    assert logger_stub.debugs == ["Failed to close Flashcards DB connections; details redacted"]
    rendered = "\n".join(logger_stub.debugs)
    assert sensitive_detail not in rendered
    assert "/private/" not in rendered
    assert "sk-" not in rendered


@pytest.mark.parametrize(
    "operation",
    [
        lambda module, context: module._list_decks_sync(context, 100, 0, False),
        lambda module, context: module._get_deck_sync(context, 1),
        lambda module, context: module._create_deck_sync(context, {"name": "deck"}),
        lambda module, context: module._list_cards_sync(context, {}),
        lambda module, context: module._get_card_sync(context, "card-1"),
        lambda module, context: module._create_card_sync(
            context,
            {"deck_id": 1, "front": "front", "back": "back"},
        ),
        lambda module, context: module._create_cards_bulk_sync(
            context,
            {"cards": [{"deck_id": 1, "front": "front", "back": "back"}]},
        ),
        lambda module, context: module._update_card_sync(
            context,
            {"card_uuid": "card-1", "updates": {"front": "updated"}},
        ),
        lambda module, context: module._delete_card_sync(
            context,
            {"card_uuid": "card-1", "expected_version": 1},
        ),
        lambda module, context: module._review_card_sync(
            context,
            {"card_uuid": "card-1", "rating": 3},
        ),
        lambda module, context: module._set_tags_sync(
            context,
            {"card_uuid": "card-1", "tags": ["tag"]},
        ),
        lambda module, context: module._get_tags_sync(context, {"card_uuid": "card-1"}),
        lambda module, context: module._export_cards_sync(context, {"format": "csv"}),
    ],
)
def test_flashcards_db_close_fallback_logs_are_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    operation: Callable[[FlashcardsModule, Any], Any],
) -> None:
    sensitive_detail = "close leaked /private/flashcards.db with sk-flashcards-secret"
    module, logger_stub = _module_with_failing_close(monkeypatch, sensitive_detail)

    operation(module, SimpleNamespace(metadata={}))

    _assert_close_fallback_log_is_sanitized(logger_stub, sensitive_detail)
