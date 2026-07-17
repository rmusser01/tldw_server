from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import translate as translate_module
from tldw_Server_API.app.api.v1.schemas.translate_schemas import TranslateRequest
from tldw_Server_API.app.core.DB_Management.Prompts_DB import (
    DatabaseError,
    ServicePromptOverrideRow,
)
from tldw_Server_API.app.core.LLM_Calls import (
    Summarization_General_Lib as summarization_module,
)
from tldw_Server_API.app.core.Prompt_Management.service_prompts import (
    ServicePromptCorruptOverride,
)

pytestmark = pytest.mark.unit

PACKAGED_SYSTEM = """You are an expert translator. Your task is to provide accurate,
natural-sounding translations that preserve the original meaning, tone, and formatting.
Do not add explanations or notes - only provide the translation."""

PACKAGED_USER = """Translate the following text to French.
Preserve the original formatting, meaning, and tone.
Only output the translation, no explanations, notes, or additional text.

Text to translate:
Hello {target_language} and $&"""


class _PromptDatabase:
    def __init__(self, parts: dict[str, str] | None = None) -> None:
        self.parts = parts
        self.revision = str(uuid4())

    def get_service_prompt_override(self, definition_id: str):
        assert definition_id == "media.text.translation"
        if self.parts is None:
            return None
        return ServicePromptOverrideRow(
            definition_id=definition_id,
            parts_json=json.dumps(self.parts),
            revision=self.revision,
        )


def _request(**overrides: object) -> TranslateRequest:
    values: dict[str, object] = {
        "text": "Hello {target_language} and $&",
        "target_language": "French",
        "source_language": "English",
        "provider": "openai",
        "model": "gpt-test",
    }
    values.update(overrides)
    return TranslateRequest(**values)


@pytest.mark.asyncio
async def test_translation_without_override_is_byte_identical_and_keeps_provider_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analyze_calls: list[dict[str, object]] = []

    def capture_analyze(**kwargs: object) -> str:
        analyze_calls.append(kwargs)
        return " Bonjour "

    monkeypatch.setattr(translate_module, "analyze", capture_analyze)

    response = await translate_module.translate_text(
        _request(),
        current_user=SimpleNamespace(id=1),
        db=_PromptDatabase(),
    )

    assert analyze_calls == [
        {
            "api_name": "openai",
            "input_data": PACKAGED_USER,
            "custom_prompt_arg": None,
            "api_key": None,
            "system_message": PACKAGED_SYSTEM,
            "temp": 0.3,
            "streaming": False,
            "model_override": "gpt-test",
            "input_is_literal_text": True,
        }
    ]
    assert response.translated_text == "Bonjour"
    assert response.detected_source_language == "English"
    assert response.target_language == "French"
    assert response.model_used == "gpt-test"


@pytest.mark.asyncio
async def test_translation_uses_saved_system_and_template_together_on_next_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parts = {
        "system": "CUSTOM SYSTEM",
        "user_template": "Language={target_language}\nPayload={text}",
    }
    db = _PromptDatabase()
    analyze_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        translate_module,
        "analyze",
        lambda **kwargs: analyze_calls.append(kwargs) or "traduit",
    )

    await translate_module.translate_text(
        _request(),
        current_user=SimpleNamespace(id=1),
        db=db,
    )
    db.parts = parts
    await translate_module.translate_text(
        _request(),
        current_user=SimpleNamespace(id=1),
        db=db,
    )

    assert analyze_calls[0]["system_message"] == PACKAGED_SYSTEM
    assert analyze_calls[0]["input_data"] == PACKAGED_USER
    assert analyze_calls[1]["system_message"] == "CUSTOM SYSTEM"
    assert analyze_calls[1]["input_data"] == (
        "Language=French\nPayload=Hello {target_language} and $&"
    )


@pytest.mark.asyncio
async def test_translation_real_analyzer_preserves_json_like_prompt_and_hides_it_from_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "ANALYZER-BODY-MUST-NOT-LEAK"
    output_secret = "ANALYZER-OUTPUT-MUST-NOT-LEAK"
    parts = {
        "system": "CUSTOM SYSTEM",
        "user_template": '{{"title":"{text}","description":"{target_language}"}}',
    }
    dispatch_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def capture_dispatch(*args: object, **kwargs: object) -> str:
        dispatch_calls.append((args, kwargs))
        return output_secret

    monkeypatch.setattr(summarization_module, "_dispatch_to_api", capture_dispatch)
    captured_logs: list[str] = []
    sink_id = summarization_module.logging.add(captured_logs.append, format="{message}")
    try:
        await translate_module.translate_text(
            _request(text=secret),
            current_user=SimpleNamespace(id=1),
            db=_PromptDatabase(parts),
        )
    finally:
        summarization_module.logging.remove(sink_id)

    expected_prompt = f'{{"title":"{secret}","description":"French"}}'
    assert dispatch_calls[0][0][0] == expected_prompt
    assert secret not in "".join(captured_logs)
    assert output_secret not in "".join(captured_logs)


@pytest.mark.asyncio
async def test_translation_real_analyzer_does_not_log_error_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "ANALYZER-ERROR-RESULT-MUST-NOT-LEAK"
    monkeypatch.setattr(
        summarization_module,
        "_dispatch_to_api",
        lambda *_args, **_kwargs: f"Error: {secret}",
    )
    captured_logs: list[str] = []
    sink_id = summarization_module.logging.add(captured_logs.append, format="{message}")
    try:
        with pytest.raises(HTTPException):
            await translate_module.translate_text(
                _request(),
                current_user=SimpleNamespace(id=1),
                db=_PromptDatabase(),
            )
    finally:
        summarization_module.logging.remove(sink_id)

    assert secret not in "".join(captured_logs)


@pytest.mark.asyncio
async def test_translation_real_adapter_exception_is_content_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "ANALYZER-ADAPTER-EXCEPTION-MUST-NOT-LEAK"

    class ExplodingAdapter:
        def chat(self, _request: object, *, timeout: float | None = None) -> object:
            raise RuntimeError(secret)

    class FakeRegistry:
        def get_adapter(self, _provider: str) -> ExplodingAdapter:
            return ExplodingAdapter()

    monkeypatch.setattr(summarization_module, "get_registry", FakeRegistry)
    captured_logs: list[str] = []
    sink_id = summarization_module.logging.add(captured_logs.append, format="{message}")
    try:
        with pytest.raises(HTTPException):
            await translate_module.translate_text(
                _request(),
                current_user=SimpleNamespace(id=1),
                db=_PromptDatabase(),
            )
    finally:
        summarization_module.logging.remove(sink_id)

    assert secret not in "".join(captured_logs)


@pytest.mark.asyncio
async def test_translation_corrupt_override_never_falls_back_or_calls_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CorruptDatabase:
        def get_service_prompt_override(self, definition_id: str):
            return ServicePromptOverrideRow(definition_id, "not-json", str(uuid4()))

    provider_called = False

    def unexpected_provider(**_kwargs: object) -> str:
        nonlocal provider_called
        provider_called = True
        return "unexpected"

    monkeypatch.setattr(translate_module, "analyze", unexpected_provider)

    with pytest.raises(ServicePromptCorruptOverride):
        await translate_module.translate_text(
            _request(),
            current_user=SimpleNamespace(id=1),
            db=CorruptDatabase(),
        )

    assert provider_called is False


@pytest.mark.asyncio
async def test_translation_prompt_store_failure_is_outside_provider_error_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = DatabaseError("prompt store unavailable")

    class FailingDatabase:
        def get_service_prompt_override(self, _definition_id: str):
            raise failure

    monkeypatch.setattr(
        translate_module,
        "analyze",
        lambda **_kwargs: pytest.fail("provider must not be called"),
    )

    with pytest.raises(DatabaseError) as exc_info:
        await translate_module.translate_text(
            _request(),
            current_user=SimpleNamespace(id=1),
            db=FailingDatabase(),
        )

    assert exc_info.value is failure
