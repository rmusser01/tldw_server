from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.Character_Chat.world_book_prompt_context import (
    apply_world_book_prompt_context,
    build_recent_world_book_scan_text,
    build_world_book_prompt_context,
)


class _FakeWorldBookService:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def process_context(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.result


def test_build_recent_world_book_scan_text_uses_recent_user_and_assistant_turns() -> None:
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "there"},
        {"role": "tool", "content": "tool output"},
        {"role": "user", "content": "latest"},
    ]

    assert build_recent_world_book_scan_text(messages) == "hello there latest"


def test_world_book_prompt_context_returns_bounded_diagnostics_and_fingerprint() -> None:
    secret_keyword = "clock-secret"
    secret_content = "World prompt secret text"
    service = _FakeWorldBookService(
        {
            "processed_context": secret_content,
            "entries_matched": 2,
            "tokens_used": 12,
            "books_used": 1,
            "entry_ids": [10, 11],
            "token_budget": 100,
            "budget_exhausted": False,
            "skipped_entries_due_to_budget": 3,
            "diagnostics": [
                {
                    "entry_id": 10,
                    "world_book_id": 5,
                    "activation_reason": "keyword_match",
                    "keyword": secret_keyword,
                    "token_cost": 7,
                    "priority": 100,
                    "content_preview": secret_content,
                },
                {
                    "entry_id": 11,
                    "world_book_id": 5,
                    "activation_reason": "keyword_match",
                    "token_cost": 5,
                    "priority": 90,
                    "static_or_pinned": True,
                },
            ],
        }
    )

    context = build_world_book_prompt_context(
        [{"role": "user", "content": "Tell me about the clock"}],
        world_book_service=service,
        character_id=42,
    )
    diagnostics_repr = repr(context.diagnostics)
    legacy_repr = repr(context.legacy_diagnostics)

    assert context.text == f"World info:\n{secret_content}"
    assert context.system_message == {"role": "system", "content": context.text}
    assert context.fingerprint.startswith("prompt-v1:sha256:")
    assert context.estimated_tokens > 0
    assert context.diagnostics["entry_ids"] == [10, 11]
    assert context.diagnostics["world_book_ids"] == [5]
    assert context.diagnostics["included_entry_count"] == 2
    assert context.diagnostics["dropped_entry_count"] == 3
    assert context.diagnostics["static_entry_ids"] == [11]
    assert context.diagnostics["dynamic_entry_ids"] == [10]
    assert secret_keyword not in diagnostics_repr
    assert secret_content not in diagnostics_repr
    assert secret_keyword not in legacy_repr
    assert secret_content not in legacy_repr
    assert service.calls[0]["character_id"] == 42
    assert service.calls[0]["include_diagnostics"] is True


def test_world_book_prompt_context_without_service_or_db_returns_empty_context() -> None:
    context = build_world_book_prompt_context(
        [{"role": "user", "content": "Tell me about the clock"}],
        db=None,
        world_book_service=None,
        character_id=42,
    )

    assert context.text == ""
    assert context.system_message is None
    assert context.estimated_tokens == 0


def test_apply_world_book_prompt_context_preserves_system_message_insertion_order() -> None:
    context = build_world_book_prompt_context(
        [{"role": "user", "content": "clock"}],
        world_book_service=_FakeWorldBookService(
            {
                "processed_context": "clock tower lore",
                "diagnostics": [],
            }
        ),
    )

    messages = apply_world_book_prompt_context(
        [
            {"role": "system", "content": "first"},
            {"role": "system", "content": "second"},
            {"role": "user", "content": "clock"},
        ],
        context,
    )

    assert [message["role"] for message in messages] == [
        "system",
        "system",
        "system",
        "user",
    ]
    assert messages[2] == context.system_message


def test_world_book_prompt_context_fingerprint_is_stable_for_same_inputs() -> None:
    result = {
        "processed_context": "clock tower lore",
        "diagnostics": [{"entry_id": 1, "world_book_id": 2, "token_cost": 4}],
    }

    first = build_world_book_prompt_context(
        [{"role": "user", "content": "clock"}],
        world_book_service=_FakeWorldBookService(result),
    )
    second = build_world_book_prompt_context(
        [{"role": "user", "content": "clock"}],
        world_book_service=_FakeWorldBookService(result),
    )

    assert first.fingerprint == second.fingerprint
    assert first.diagnostics["fingerprint"] == second.diagnostics["fingerprint"]
