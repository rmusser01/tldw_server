"""A workflow step that calls an LLM fails with the user's quota refusal; it does not return an error output (TASK-13638)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.Workflows.adapters.content import generation, image, summarize
from tldw_Server_API.app.core.Workflows.adapters.control import flow, state
from tldw_Server_API.app.core.Workflows.adapters.evaluation import eval as evaluation_step
from tldw_Server_API.app.core.Workflows.adapters.llm import translate
from tldw_Server_API.app.core.Workflows.adapters.research import bibliography
from tldw_Server_API.app.core.Workflows.adapters.text import nlp

pytestmark = pytest.mark.unit

CHAT_CALL = "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async"
ANALYZE = "tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib.analyze"
TEXT = "Plants turn light, water and air into sugar."
GENERATION_CONFIG = {
    "text": TEXT,
    "content": TEXT,
    "excerpt_text": TEXT,
    "topic": "plants",
    "provider": "openai",
    "model": "gpt-4o-mini",
}


def _refusal() -> LLMQuotaExceededError:
    return LLMQuotaExceededError(
        category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "adapter_name",
    [
        "run_flashcard_generate_adapter",
        "run_quiz_generate_adapter",
        "run_outline_generate_adapter",
        "run_glossary_extract_adapter",
        "run_mindmap_generate_adapter",
        "run_slides_generate_adapter",
        "run_report_generate_adapter",
        "run_newsletter_generate_adapter",
        "run_notes_studio_generate_adapter",
        "run_diagram_generate_adapter",
    ],
)
async def test_content_generation_lets_the_refusal_through(adapter_name: str) -> None:
    with patch.object(generation, "perform_chat_api_call_async", AsyncMock(side_effect=_refusal())):
        with pytest.raises(LLMQuotaExceededError):
            await getattr(generation, adapter_name)(dict(GENERATION_CONFIG), {})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("adapter", "config"),
    [
        (nlp.run_keyword_extract_adapter, {"text": TEXT, "method": "llm"}),
        (nlp.run_sentiment_analyze_adapter, {"text": TEXT}),
        (nlp.run_topic_model_adapter, {"text": TEXT}),
        (nlp.run_entity_extract_adapter, {"text": TEXT}),
        (bibliography.run_reference_parse_adapter, {"citation": "Smith J. 2020. Plants. J Bot 1:1."}),
        (bibliography.run_literature_review_adapter, {"papers": [{"title": "Plants", "abstract": TEXT}]}),
    ],
)
async def test_text_and_research_steps_let_the_refusal_through(adapter, config: dict) -> None:
    with patch(CHAT_CALL, AsyncMock(side_effect=_refusal())):
        with pytest.raises(LLMQuotaExceededError):
            await adapter(dict(config), {})


@pytest.mark.asyncio
async def test_image_describe_lets_the_refusal_through() -> None:
    with patch.object(image, "perform_chat_api_call_async", AsyncMock(side_effect=_refusal())):
        with pytest.raises(LLMQuotaExceededError):
            await image.run_image_describe_adapter({"image_url": "https://example.com/plant.png"}, {})


@pytest.mark.asyncio
async def test_summarize_lets_the_refusal_through() -> None:
    with patch.object(summarize, "is_test_mode", return_value=False), patch(ANALYZE, side_effect=_refusal()):
        with pytest.raises(LLMQuotaExceededError):
            await summarize.run_summarize_adapter({"text": TEXT, "api_name": "openai"}, {})


@pytest.mark.asyncio
async def test_translate_does_not_return_the_input_as_a_translation() -> None:
    adapter = MagicMock()
    adapter.achat = AsyncMock(side_effect=_refusal())
    registry = MagicMock()
    registry.get_adapter.return_value = adapter
    with patch.object(translate, "is_test_mode", return_value=False), patch.object(
        translate, "get_registry", return_value=registry
    ):
        with pytest.raises(LLMQuotaExceededError):
            await translate.run_translate_adapter({"input": "Bonjour", "target_lang": "en"}, {})


@pytest.mark.asyncio
async def test_parallel_does_not_drop_a_refused_step() -> None:
    async def refused_step(_config, _context):
        raise _refusal()

    with patch.object(flow, "get_adapter", return_value=refused_step):
        with pytest.raises(LLMQuotaExceededError):
            await flow.run_parallel_adapter({"steps": [{"type": "summarize", "config": {}}]}, {})


@pytest.mark.asyncio
async def test_retry_does_not_retry_a_refused_step() -> None:
    analyze = MagicMock(side_effect=_refusal())
    with patch.object(summarize, "is_test_mode", return_value=False), patch(ANALYZE, analyze), patch.object(
        state.asyncio, "sleep", AsyncMock()
    ):
        with pytest.raises(LLMQuotaExceededError):
            await state.run_retry_adapter(
                {"step_type": "summarize", "step_config": {"text": TEXT, "api_name": "openai"}, "max_retries": 3}, {}
            )
    assert analyze.call_count == 1


@pytest.mark.asyncio
async def test_evaluations_step_lets_the_refusal_through() -> None:
    service = MagicMock()
    service.initialize = AsyncMock()
    service.evaluate_geval = AsyncMock(side_effect=_refusal())
    with patch.object(evaluation_step, "is_test_mode", return_value=False), patch(
        "tldw_Server_API.app.core.Evaluations.unified_evaluation_service.UnifiedEvaluationService",
        MagicMock(return_value=service),
    ):
        with pytest.raises(LLMQuotaExceededError):
            await evaluation_step.run_evaluations_adapter({"action": "geval", "response": TEXT, "context": TEXT}, {"user_id": "1"})
