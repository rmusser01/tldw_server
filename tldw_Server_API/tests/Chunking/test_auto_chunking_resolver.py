from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.chunking_options import (
    async_resolve_chunking_for_result,
    async_resolve_chunking_options_and_plan,
    resolve_chunking_options_and_plan,
)

pytestmark = pytest.mark.unit


def _form(**overrides):
    values = {
        "perform_chunking": True,
        "media_type": "document",
        "chunking_mode": None,
        "auto_chunking_goal": "balanced",
        "auto_chunking_use_llm": False,
        "chunk_method": None,
        "chunk_size": 500,
        "chunk_overlap": 200,
        "use_adaptive_chunking": False,
        "use_multi_level_chunking": False,
        "chunk_language": None,
        "transcription_language": None,
        "custom_chapter_pattern": None,
        "enable_contextual_chunking": False,
        "contextual_llm_model": None,
        "context_window_size": None,
        "context_strategy": None,
        "context_token_budget": None,
        "hierarchical_chunking": False,
        "hierarchical_template": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_resolver_returns_no_options_when_chunking_disabled():
    chunk_options, chunking_plan = resolve_chunking_options_and_plan(
        _form(perform_chunking=False, chunking_mode="auto"),
        media_type="document",
    )

    assert chunk_options is None
    assert chunking_plan is None


def test_resolver_preserves_legacy_chunking_options_when_mode_missing():
    chunk_options, chunking_plan = resolve_chunking_options_and_plan(
        _form(chunk_method="words", chunk_size=640, chunk_overlap=64),
        media_type="document",
    )

    assert chunking_plan is None
    assert chunk_options["method"] == "words"
    assert chunk_options["max_size"] == 640
    assert chunk_options["overlap"] == 64


def test_resolver_preserves_manual_chunking_options():
    chunk_options, chunking_plan = resolve_chunking_options_and_plan(
        _form(
            chunking_mode="manual",
            chunk_method="tokens",
            chunk_size=512,
            chunk_overlap=32,
        ),
        media_type="document",
    )

    assert chunking_plan is None
    assert chunk_options["method"] == "tokens"
    assert chunk_options["max_size"] == 512
    assert chunk_options["overlap"] == 32


def test_resolver_auto_ignores_stale_manual_options_and_records_plan():
    chunk_options, chunking_plan = resolve_chunking_options_and_plan(
        _form(
            chunking_mode="auto",
            auto_chunking_goal="qa_search",
            auto_chunking_use_llm=True,
            chunk_method="words",
            chunk_size=333,
            chunk_overlap=1,
        ),
        media_type="document",
        source_name="notes.md",
        extracted_text="# Intro\n\nBody paragraph\n",
    )

    assert chunk_options == {
        "method": "structure_aware",
        "max_size": 700,
        "overlap": 140,
        "adaptive": False,
        "multi_level": False,
        "language": None,
    }
    assert chunking_plan["mode"] == "auto"
    assert chunking_plan["goal"] == "qa_search"
    assert chunking_plan["used_llm"] is False
    assert "ai_assist_unavailable" in chunking_plan["fallback_reason"]
    json.dumps(chunking_plan)


def test_resolver_auto_records_template_fallback_status():
    chunk_options, chunking_plan = resolve_chunking_options_and_plan(
        _form(chunking_mode="auto"),
        media_type="document",
        extracted_text="plain article text",
        template_status="no_match",
        semantic_available=False,
    )

    assert chunk_options["method"] == "sentences"
    assert "semantic_unavailable" in chunking_plan["fallback_reason"]
    assert "template_no_match" in chunking_plan["fallback_reason"]


def test_resolver_accepts_mapping_backed_auto_payload():
    chunk_options, chunking_plan = resolve_chunking_options_and_plan(
        {
            "perform_chunking": True,
            "media_type": "document",
            "chunking_mode": "auto",
            "auto_chunking_goal": "navigation_summary",
            "auto_chunking_use_llm": True,
            "chunk_language": "en",
            "title": "Planning Notes",
            "urls": ["https://example.test/planning.md"],
            "chunk_method": "words",
            "chunk_size": 333,
            "chunk_overlap": 1,
        },
        media_type=None,
        extracted_text="# Overview\n\n- first\n- second\n",
    )

    assert chunk_options["method"] == "structure_aware"
    assert chunk_options["max_size"] == 1400
    assert chunk_options["language"] == "en"
    assert chunking_plan["mode"] == "auto"
    assert chunking_plan["goal"] == "navigation_summary"
    assert chunking_plan["profile"]["title"] == "Planning Notes"
    assert chunking_plan["profile"]["source_name"] == "https://example.test/planning.md"
    assert "outline" in chunking_plan["derived_views"]


@pytest.mark.asyncio
async def test_async_resolver_does_not_call_assistant_without_explicit_opt_in():
    calls = []

    class Assistant:
        async def refine(self, request):
            calls.append(request)
            raise AssertionError("assistant should not be called")

    chunk_options, chunking_plan = await async_resolve_chunking_options_and_plan(
        _form(chunking_mode="auto", auto_chunking_use_llm=False),
        media_type="document",
        extracted_text="# Intro\n\nBody",
        boundary_assistant=Assistant(),
    )

    assert calls == []
    assert chunk_options["method"] == "structure_aware"
    assert chunking_plan["used_llm"] is False
    assert chunking_plan["fallback_reason"] is None


@pytest.mark.asyncio
async def test_async_resolver_applies_valid_opt_in_assistant_result():
    class Assistant:
        async def refine(self, request):
            assert request.media_type == "document"
            assert request.chunking_plan["used_llm"] is False
            assert request.chunking_plan["fallback_reason"] is None
            from tldw_Server_API.app.core.Chunking.auto_boundary_assistant import (
                AutoChunkBoundaryAssistantResult,
            )

            return AutoChunkBoundaryAssistantResult.success(
                chunk_options={
                    "method": "semantic",
                    "max_size": 840,
                    "overlap": 84,
                    "adaptive": False,
                    "multi_level": False,
                    "language": None,
                },
                derived_views=("topic_sections",),
                rationale="Assistant selected topic shifts.",
                provider="openai",
                model="gpt-test",
            )

    chunk_options, chunking_plan = await async_resolve_chunking_options_and_plan(
        _form(chunking_mode="auto", auto_chunking_use_llm=True),
        media_type="document",
        extracted_text="# Intro\n\nBody",
        boundary_assistant=Assistant(),
    )

    assert chunk_options["method"] == "semantic"
    assert chunk_options["max_size"] == 840
    assert chunking_plan["used_llm"] is True
    assert chunking_plan["method"] == "semantic"
    assert chunking_plan["provider"] == "openai"
    assert chunking_plan["model"] == "gpt-test"
    assert chunking_plan["fallback_reason"] is None
    assert chunking_plan["derived_views"] == ["topic_sections"]


@pytest.mark.asyncio
async def test_async_resolver_preserves_deterministic_plan_on_assistant_fallback():
    class Assistant:
        async def refine(self, request):
            from tldw_Server_API.app.core.Chunking.auto_boundary_assistant import (
                AutoChunkBoundaryAssistantResult,
            )

            return AutoChunkBoundaryAssistantResult.fallback(
                reason="ai_assist_timeout",
                rationale="Timed out after 0.5 seconds.",
            )

    baseline_options, baseline_plan = resolve_chunking_options_and_plan(
        _form(chunking_mode="auto", auto_chunking_use_llm=False),
        media_type="document",
        extracted_text="# Intro\n\nBody",
    )
    chunk_options, chunking_plan = await async_resolve_chunking_options_and_plan(
        _form(chunking_mode="auto", auto_chunking_use_llm=True),
        media_type="document",
        extracted_text="# Intro\n\nBody",
        boundary_assistant=Assistant(),
    )

    assert chunk_options == baseline_options
    assert chunking_plan["method"] == baseline_plan["method"]
    assert chunking_plan["max_size"] == baseline_plan["max_size"]
    assert chunking_plan["overlap"] == baseline_plan["overlap"]
    assert chunking_plan["used_llm"] is False
    assert chunking_plan["fallback_reason"] == "ai_assist_timeout"
    assert "Timed out after 0.5 seconds." in chunking_plan["rationale"]


@pytest.mark.asyncio
async def test_async_resolver_preserves_api_name_model_when_provider_is_present():
    seen = {}

    class Assistant:
        async def refine(self, request):
            seen["provider"] = request.provider
            seen["model"] = request.model
            from tldw_Server_API.app.core.Chunking.auto_boundary_assistant import (
                AutoChunkBoundaryAssistantResult,
            )

            return AutoChunkBoundaryAssistantResult.fallback(
                reason="ai_assist_unavailable",
                rationale="test fallback",
            )

    await async_resolve_chunking_options_and_plan(
        _form(
            chunking_mode="auto",
            auto_chunking_use_llm=True,
            api_provider="openai",
            api_name="openai/gpt-4o",
            model_name=None,
        ),
        media_type="document",
        extracted_text="# Intro\n\nBody",
        boundary_assistant=Assistant(),
    )

    assert seen == {"provider": "openai", "model": "gpt-4o"}


@pytest.mark.asyncio
async def test_async_resolve_chunking_for_result_can_reuse_batch_llm_resolution():
    class Assistant:
        async def refine(self, request):
            raise AssertionError("assistant should not be called when reusing batch resolution")

    default_options = {"method": "semantic", "max_size": 820, "overlap": 82}
    default_plan = {
        "mode": "auto",
        "used_llm": True,
        "method": "semantic",
        "max_size": 820,
        "overlap": 82,
    }

    chunk_options, chunking_plan = await async_resolve_chunking_for_result(
        _form(chunking_mode="auto", auto_chunking_use_llm=True),
        {"content": "# Intro\n\nBody"},
        media_type="document",
        default_chunk_options=default_options,
        default_chunking_plan=default_plan,
        boundary_assistant=Assistant(),
        allow_llm_assist=False,
    )

    assert chunk_options == default_options
    assert chunking_plan == default_plan
