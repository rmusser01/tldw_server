from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import pytest

from tldw_Server_API.app.core.Prompt_Management.service_prompts import (
    ResolvedServicePrompt,
    get_service_prompt_definition,
)
from tldw_Server_API.app.core.Writing import note_title as note_title_module
from tldw_Server_API.app.core.Writing.note_title import TitleGenOptions, generate_note_title

pytestmark = pytest.mark.unit


class _RecordingAdapter:
    def __init__(self, result: object) -> None:
        self.result = result
        self.payloads: list[dict[str, object]] = []

    def chat(self, payload: dict[str, object]) -> object:
        self.payloads.append(payload)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _resolved_prompt(*, system: str, title_instruction: str) -> ResolvedServicePrompt:
    return ResolvedServicePrompt(
        definition=get_service_prompt_definition("notes.title.generate"),
        parts=MappingProxyType(
            {
                "system": system,
                "title_instruction": title_instruction,
            }
        ),
        source="user",
        revision="revision-notes-title",
    )


def _enable_llm(monkeypatch: pytest.MonkeyPatch, adapter: _RecordingAdapter) -> None:
    monkeypatch.setattr(
        note_title_module,
        "core_settings",
        {"NOTES_TITLE_LLM_ENABLED": True},
    )
    monkeypatch.setattr(
        note_title_module,
        "get_registry",
        lambda: SimpleNamespace(get_adapter=lambda provider: adapter if provider == "openai" else None),
    )


def test_llm_title_uses_editable_semantics_with_locked_provider_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _RecordingAdapter({"choices": [{"message": {"content": '  "Evocative title"  '}}]})
    _enable_llm(monkeypatch, adapter)
    content = "A" * 2_001

    title = generate_note_title(
        content,
        options=TitleGenOptions(strategy="llm", max_len=64),
        service_prompt=_resolved_prompt(
            system="System {braces} remain literal.",
            title_instruction="Craft an evocative {literal} title",
        ),
    )

    assert title == "Evocative title"
    assert adapter.payloads == [
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Craft an evocative {literal} title no longer than 64 characters for the following note.\n"
                        "Return only the title with no quotes or extra text.\n\n" + ("A" * 2_000)
                    ),
                }
            ],
            "system_message": "System {braces} remain literal.",
            "model": None,
            "temperature": 0.2,
            "max_tokens": 128,
        }
    ]


def test_provider_failure_still_uses_the_existing_heuristic_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _RecordingAdapter(RuntimeError("provider unavailable"))
    _enable_llm(monkeypatch, adapter)

    title = generate_note_title(
        "# Existing heuristic heading\nBody",
        options=TitleGenOptions(strategy="llm_fallback", max_len=80),
        service_prompt=_resolved_prompt(
            system="Custom title system",
            title_instruction="Create a custom title",
        ),
    )

    assert title == "Existing heuristic heading"
    assert len(adapter.payloads) == 1


def test_empty_provider_output_still_uses_the_existing_heuristic_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _RecordingAdapter({"choices": [{"message": {"content": ""}}]})
    _enable_llm(monkeypatch, adapter)

    title = generate_note_title(
        "Existing empty-output fallback",
        options=TitleGenOptions(strategy="llm", max_len=80),
        service_prompt=_resolved_prompt(
            system="Custom title system",
            title_instruction="Create a custom title",
        ),
    )

    assert title == "Existing empty-output fallback"
    assert len(adapter.payloads) == 1


def test_unavailable_provider_still_uses_the_existing_heuristic_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        note_title_module,
        "core_settings",
        {"NOTES_TITLE_LLM_ENABLED": True},
    )
    monkeypatch.setattr(
        note_title_module,
        "get_registry",
        lambda: SimpleNamespace(get_adapter=lambda _provider: None),
    )

    title = generate_note_title(
        "Existing unavailable-provider fallback",
        options=TitleGenOptions(strategy="llm", max_len=80),
        service_prompt=_resolved_prompt(
            system="Custom title system",
            title_instruction="Create a custom title",
        ),
    )

    assert title == "Existing unavailable-provider fallback"


def test_active_llm_strategy_requires_an_owner_scoped_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _RecordingAdapter({"choices": [{"message": {"content": "Must not be used"}}]})
    _enable_llm(monkeypatch, adapter)

    with pytest.raises(RuntimeError, match="Service Prompt resolution is required"):
        generate_note_title(
            "Owner-bound note content",
            options=TitleGenOptions(strategy="llm", max_len=80),
        )

    assert adapter.payloads == []
