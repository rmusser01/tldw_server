from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Chunking.auto_boundary_assistant import (
    AutoChunkBoundaryAssistantRequest,
    AutoChunkBoundaryAssistantResult,
    ChatAutoChunkBoundaryAssistant,
    append_auto_chunking_fallback,
    extract_bounded_text_excerpt,
    parse_boundary_assistant_response,
)

pytestmark = pytest.mark.unit


def _request(**overrides):
    values = {
        "chunk_options": {
            "method": "structure_aware",
            "max_size": 900,
            "overlap": 120,
            "adaptive": False,
            "multi_level": False,
            "language": "en",
        },
        "chunking_plan": {
            "mode": "auto",
            "goal": "balanced",
            "used_llm": False,
            "method": "structure_aware",
            "max_size": 900,
            "overlap": 120,
            "template_name": None,
            "derived_views": ["section_titles"],
            "fallback_reason": None,
            "rationale": "Detected document structure.",
            "profile": {"media_type": "document", "text_length": 2000},
        },
        "media_type": "document",
        "source_name": "notes.md",
        "extracted_text": "# Intro\n\nSome content.\n\n## Details\n\nMore content.",
        "provider": "openai",
        "model": "gpt-test",
        "timeout_sec": 0.5,
    }
    values.update(overrides)
    return AutoChunkBoundaryAssistantRequest(**values)


def test_boundary_assistant_result_types_represent_success_and_fallback():
    success = AutoChunkBoundaryAssistantResult.success(
        chunk_options={"method": "semantic", "max_size": 800, "overlap": 80},
        derived_views=("topic_sections",),
        rationale="Topic shifts are clearer than headings.",
        provider="openai",
        model="gpt-test",
    )
    fallback = AutoChunkBoundaryAssistantResult.fallback(
        reason="ai_assist_invalid_response",
        rationale="Assistant response did not match the strict schema.",
    )

    assert success.used_llm is True
    assert success.fallback_reason is None
    assert success.provider == "openai"
    assert success.model == "gpt-test"
    assert fallback.used_llm is False
    assert fallback.chunk_options is None
    assert fallback.fallback_reason == "ai_assist_invalid_response"


def test_extract_bounded_text_excerpt_limits_context_without_reordering_text():
    text = "a" * 1200 + "TAIL"

    excerpt = extract_bounded_text_excerpt(text, max_chars=64)

    assert excerpt == "a" * 64
    assert "TAIL" not in excerpt


def test_parse_boundary_assistant_response_accepts_strict_bounded_json():
    request = _request()

    result = parse_boundary_assistant_response(
        '{"method":"semantic","max_size":840,"overlap":84,'
        '"derived_views":["topic_sections","outline"],'
        '"rationale":"The document uses topic shifts more than headings."}',
        request=request,
        provider="openai",
        model="gpt-test",
    )

    assert result.used_llm is True
    assert result.chunk_options == {
        "method": "semantic",
        "max_size": 840,
        "overlap": 84,
        "adaptive": False,
        "multi_level": False,
        "language": "en",
    }
    assert result.derived_views == ("topic_sections", "outline")
    assert result.rationale == "The document uses topic shifts more than headings."


@pytest.mark.parametrize(
    ("response_text", "reason_fragment"),
    [
        ("not-json", "not valid JSON"),
        ('{"method":"shell","max_size":840,"overlap":84}', "method"),
        ('{"method":"semantic","max_size":10,"overlap":1}', "max_size"),
        ('{"method":"semantic","max_size":840,"overlap":840}', "overlap"),
        ('{"method":"semantic","max_size":840,"overlap":84,"derived_views":["bad view"]}', "derived_views"),
        ('{"method":"ebook_chapters","max_size":840,"overlap":84}', "ebook_chapters"),
    ],
)
def test_parse_boundary_assistant_response_rejects_invalid_suggestions(response_text, reason_fragment):
    result = parse_boundary_assistant_response(
        response_text,
        request=_request(),
        provider="openai",
        model="gpt-test",
    )

    assert result.used_llm is False
    assert result.fallback_reason == "ai_assist_invalid_response"
    assert reason_fragment in result.rationale


def test_append_auto_chunking_fallback_preserves_deterministic_plan_and_options():
    request = _request()
    plan = dict(request.chunking_plan)
    options = dict(request.chunk_options)

    updated_options, updated_plan = append_auto_chunking_fallback(
        options,
        plan,
        "ai_assist_timeout",
        "Timed out after 0.5 seconds.",
    )

    assert updated_options == options
    assert updated_plan["method"] == plan["method"]
    assert updated_plan["max_size"] == plan["max_size"]
    assert updated_plan["overlap"] == plan["overlap"]
    assert updated_plan["used_llm"] is False
    assert updated_plan["fallback_reason"] == "ai_assist_timeout"
    assert "Timed out after 0.5 seconds." in updated_plan["rationale"]


@pytest.mark.asyncio
async def test_chat_assistant_returns_unavailable_when_provider_cannot_be_resolved():
    assistant = ChatAutoChunkBoundaryAssistant(
        chat_call=lambda **_: pytest.fail("chat call should not run"),
        config_loader=lambda: {},
        registry_getter=lambda: SimpleNamespace(get_adapter=lambda _provider: object()),
        api_key_resolver=lambda _provider, _config=None: None,
        provider_requires_key=lambda _provider: False,
        default_provider=None,
    )

    result = await assistant.refine(_request(provider=None, model=None))

    assert result.used_llm is False
    assert result.fallback_reason == "ai_assist_unavailable"
    assert "provider" in result.rationale


@pytest.mark.asyncio
async def test_chat_assistant_calls_provider_when_available_and_valid():
    calls = []

    async def chat_call(**kwargs):
        calls.append(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"method":"semantic","max_size":820,"overlap":82,'
                            '"derived_views":["topic_sections"],'
                            '"rationale":"Clear topic transitions."}'
                        )
                    }
                }
            ]
        }

    assistant = ChatAutoChunkBoundaryAssistant(
        chat_call=chat_call,
        config_loader=lambda: {"openai_api": {"model": "gpt-config"}},
        registry_getter=lambda: SimpleNamespace(get_adapter=lambda _provider: object()),
        api_key_resolver=lambda _provider, _config=None: "key",
        provider_requires_key=lambda _provider: True,
        default_provider="openai",
    )

    result = await assistant.refine(_request(provider=None, model=None))

    assert result.used_llm is True
    assert result.chunk_options["method"] == "semantic"
    assert result.chunk_options["max_size"] == 820
    assert calls[0]["api_provider"] == "openai"
    assert calls[0]["model"] == "gpt-config"
    assert calls[0]["stream"] is False


@pytest.mark.asyncio
async def test_chat_assistant_runs_availability_checks_off_event_loop_thread():
    main_thread = threading.get_ident()
    loader_threads = []

    async def chat_call(**_kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"method":"semantic","max_size":820,"overlap":82,'
                            '"derived_views":["topic_sections"],'
                            '"rationale":"Clear topic transitions."}'
                        )
                    }
                }
            ]
        }

    def config_loader():
        loader_threads.append(threading.get_ident())
        return {"openai_api": {"model": "gpt-config"}}

    assistant = ChatAutoChunkBoundaryAssistant(
        chat_call=chat_call,
        config_loader=config_loader,
        registry_getter=lambda: SimpleNamespace(get_adapter=lambda _provider: object()),
        api_key_resolver=lambda _provider, _config=None: "key",
        provider_requires_key=lambda _provider: True,
        default_provider="openai",
    )

    result = await assistant.refine(_request(provider=None, model=None))

    assert result.used_llm is True
    assert loader_threads
    assert loader_threads[0] != main_thread


@pytest.mark.asyncio
async def test_chat_assistant_uses_adapter_canonical_provider_for_alias_availability():
    calls = []

    async def chat_call(**kwargs):
        calls.append(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"method":"semantic","max_size":820,"overlap":82,'
                            '"derived_views":["topic_sections"],'
                            '"rationale":"Clear topic transitions."}'
                        )
                    }
                }
            ]
        }

    seen_requires_key = []
    assistant = ChatAutoChunkBoundaryAssistant(
        chat_call=chat_call,
        config_loader=lambda: {"local_llm": {"model": "local-model"}},
        registry_getter=lambda: SimpleNamespace(get_adapter=lambda _provider: SimpleNamespace(name="local-llm")),
        api_key_resolver=lambda _provider, _config=None: None,
        provider_requires_key=lambda provider: seen_requires_key.append(provider) or False,
    )

    result = await assistant.refine(_request(provider="local_llm", model=None))

    assert result.used_llm is True
    assert result.provider == "local-llm"
    assert calls[0]["api_provider"] == "local-llm"
    assert calls[0]["model"] == "local-model"
    assert seen_requires_key == ["local-llm"]


@pytest.mark.asyncio
async def test_chat_assistant_does_not_retry_typeerror_from_api_key_resolver_body():
    def api_key_resolver(_provider, _config=None):
        raise TypeError("inner resolver failure")

    assistant = ChatAutoChunkBoundaryAssistant(
        chat_call=lambda **_: pytest.fail("chat call should not run"),
        config_loader=lambda: {"openai_api": {"model": "gpt-config"}},
        registry_getter=lambda: SimpleNamespace(get_adapter=lambda _provider: object()),
        api_key_resolver=api_key_resolver,
        provider_requires_key=lambda _provider: True,
        default_provider="openai",
    )

    result = await assistant.refine(_request(provider=None, model=None))

    assert result.used_llm is False
    assert result.fallback_reason == "ai_assist_provider_error"
    assert "TypeError" in result.rationale


@pytest.mark.asyncio
async def test_chat_assistant_timeout_falls_back_without_raising():
    async def chat_call(**_kwargs):
        await asyncio.sleep(0.05)
        return "{}"

    assistant = ChatAutoChunkBoundaryAssistant(
        chat_call=chat_call,
        config_loader=lambda: {"openai_api": {"model": "gpt-config"}},
        registry_getter=lambda: SimpleNamespace(get_adapter=lambda _provider: object()),
        api_key_resolver=lambda _provider, _config=None: "key",
        provider_requires_key=lambda _provider: True,
        default_provider="openai",
    )

    result = await assistant.refine(_request(provider=None, model=None, timeout_sec=0.001))

    assert result.used_llm is False
    assert result.fallback_reason == "ai_assist_timeout"


@pytest.mark.asyncio
async def test_chat_assistant_provider_error_falls_back_without_raising():
    async def chat_call(**_kwargs):
        raise RuntimeError("provider exploded")

    assistant = ChatAutoChunkBoundaryAssistant(
        chat_call=chat_call,
        config_loader=lambda: {"openai_api": {"model": "gpt-config"}},
        registry_getter=lambda: SimpleNamespace(get_adapter=lambda _provider: object()),
        api_key_resolver=lambda _provider, _config=None: "key",
        provider_requires_key=lambda _provider: True,
        default_provider="openai",
    )

    result = await assistant.refine(_request(provider=None, model=None))

    assert result.used_llm is False
    assert result.fallback_reason == "ai_assist_provider_error"
    assert "RuntimeError" in result.rationale
