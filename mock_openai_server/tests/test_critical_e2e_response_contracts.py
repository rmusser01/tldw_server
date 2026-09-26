"""Behavior contracts for deterministic critical-journey mock responses."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from mock_openai_server.mock_openai.config import MockConfig
from mock_openai_server.mock_openai.responses import ResponseManager


_FLASHCARD_SOURCE = """The mitochondria is the powerhouse of the cell.

DNA stands for deoxyribonucleic acid.

Photosynthesis converts light energy into chemical energy.

The human body has 206 bones.

Water boils at 100 degrees Celsius at sea level."""
_PLAYWRIGHT_QUESTION = "What is Playwright? Use the ingested content to answer."
_PLAYWRIGHT_RAG_QUERY = "Playwright"
_PLAYWRIGHT_DOCUMENT = (
    "Playwright is an open-source framework for reliable end-to-end browser testing."
)
_PLAYWRIGHT_SHARED_DB_TOKEN = "task2c-playwright-1735689600000-abc123"  # nosec B105 - public deterministic test token
_PLAYWRIGHT_MISMATCHED_TOKEN = "task2c-playwright-1735689600001-def456"  # nosec B105 - public deterministic test token


def _local_success_config() -> MockConfig:
    repo_root = Path(__file__).resolve().parents[2]
    return MockConfig.from_file(
        repo_root
        / "apps/tldw-frontend/e2e/onboarding-uat/mock-openai/configs/local-success.json"
    )


def _selected_wire_response(
    config: MockConfig,
    messages: list[dict[str, str]],
) -> tuple[str | None, dict[str, Any]]:
    request = {"model": "local-uat-chat", "messages": messages}
    response_file = config.responses["chat_completions"].find_matching_response(request)
    response = ResponseManager(Path(str(config.response_base_dir))).generate_chat_response(
        request,
        response_file,
    )
    return response_file, response.model_dump()


@pytest.mark.unit
def test_flashcard_generation_prompt_selects_cards_the_production_parser_accepts() -> None:
    """A default mock reply must not turn the critical flashcard route into zero cards."""
    from tldw_Server_API.app.core.Workflows.adapters.content import run_flashcard_generate_adapter

    config = _local_success_config()
    selected_files: list[str | None] = []

    async def select_mock_response(**kwargs: Any) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": kwargs["system_message"]},
            *kwargs["messages"],
        ]
        response_file, response = _selected_wire_response(config, messages)
        selected_files.append(response_file)
        return response

    with patch(
        "tldw_Server_API.app.core.Workflows.adapters.content.generation.perform_chat_api_call_async",
        new=select_mock_response,
    ):
        result = asyncio.run(
            run_flashcard_generate_adapter(
                {"text": _FLASHCARD_SOURCE, "num_cards": 1, "card_type": "basic"},
                {},
            )
        )

    assert selected_files == ["chat/flashcards.json"]
    assert result["flashcards"] == [
        {
            "front": "What is the powerhouse of the cell?",
            "back": "The mitochondria is the powerhouse of the cell.",
            "tags": ["biology", "cells"],
            "generation_type": "basic",
            "model_type": "basic",
        }
    ]
    assert result["count"] == 1


@pytest.mark.unit
def test_flashcard_claim_verifier_selects_a_grounded_json_judgment() -> None:
    """The generated card's source-backed claim must clear the unchanged verifier."""
    from tldw_Server_API.app.core.Claims_Extraction.artifact_verification import (
        ArtifactVerificationUnit,
        verify_generated_artifact_against_sources,
    )
    from tldw_Server_API.app.core.RAG.rag_service.types import Document

    config = _local_success_config()
    selected_files: list[str | None] = []

    def select_mock_response(**kwargs: Any) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": kwargs["system_message"]},
            *kwargs["messages_payload"],
        ]
        response_file, response = _selected_wire_response(config, messages)
        selected_files.append(response_file)
        return response

    with patch(
        "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call",
        new=select_mock_response,
    ):
        result = asyncio.run(
            verify_generated_artifact_against_sources(
                artifact_type="flashcards",
                units=[
                    ArtifactVerificationUnit(
                        unit_id="flashcard:1:back",
                        text=_FLASHCARD_SOURCE,
                        claims=["The mitochondria is the powerhouse of the cell."],
                    )
                ],
                source_documents=[
                    Document(
                        id="flashcards-source",
                        content=_FLASHCARD_SOURCE,
                        metadata={"title": "Flashcard generation source"},
                    )
                ],
                generation_provider="openai",
                generation_model="local-uat-chat",
            )
        )

    assert selected_files == ["chat/flashcard-claim-verdict.json"]
    assert result.verdict == "grounded"


@pytest.mark.unit
def test_ingest_search_chat_rag_prompt_selects_grounded_answer() -> None:
    """A concrete ingested document must select the grounded RAG response."""
    from tldw_Server_API.app.core.RAG.rag_service.generation import (
        GenerationConfig,
        LLMGenerator,
    )
    from tldw_Server_API.app.core.RAG.rag_service.types import Document

    config = _local_success_config()
    selected_files: list[str | None] = []

    async def select_mock_response(**kwargs: Any) -> dict[str, Any]:
        response_file, response = _selected_wire_response(config, kwargs["messages"])
        selected_files.append(response_file)
        return response

    context = SimpleNamespace(
        documents=[
            Document(
                id="playwright-ingested-document",
                content=_PLAYWRIGHT_DOCUMENT,
                metadata={
                    "source": "local fixture",
                    "title": "Playwright local fixture",
                },
            )
        ]
    )
    with patch(
        "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
        new=select_mock_response,
    ):
        result = asyncio.run(
            LLMGenerator(GenerationConfig(provider="openai", model="local-uat-chat")).generate(
                context,
                _PLAYWRIGHT_RAG_QUERY,
            )
        )

    assert selected_files == ["chat/playwright-grounded.json"]
    assert result.response == (
        "Playwright is an open-source framework for reliable end-to-end browser testing."
    )


@pytest.mark.unit
def test_local_file_rag_prompt_with_ingested_filename_selects_grounded_answer() -> None:
    """The real local-file RAG prompt must not fall through to source summary."""
    from tldw_Server_API.app.core.RAG.rag_service.generation import (
        GenerationConfig,
        LLMGenerator,
    )
    from tldw_Server_API.app.core.RAG.rag_service.types import Document

    config = _local_success_config()
    selected_files: list[str | None] = []

    async def select_mock_response(**kwargs: Any) -> dict[str, Any]:
        response_file, response = _selected_wire_response(config, kwargs["messages"])
        selected_files.append(response_file)
        return response

    context = SimpleNamespace(
        documents=[
            Document(
                id="playwright-ingested-file",
                content=f"playwright-grounded\n{_PLAYWRIGHT_DOCUMENT}",
                metadata={"source": "media_db", "title": "playwright-grounded"},
            )
        ]
    )
    with patch(
        "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
        new=select_mock_response,
    ):
        result = asyncio.run(
            LLMGenerator(GenerationConfig(provider="openai", model="local-uat-chat")).generate(
                context,
                _PLAYWRIGHT_RAG_QUERY,
            )
        )

    assert selected_files == ["chat/playwright-grounded.json"]
    assert result.response == _PLAYWRIGHT_DOCUMENT


@pytest.mark.unit
def test_unique_playwright_fixture_context_ignores_a_preexisting_playwright_document() -> None:
    """A shared DB fixture must require its generated token, not any Playwright source."""
    from tldw_Server_API.app.core.RAG.rag_service.generation import (
        GenerationConfig,
        LLMGenerator,
    )
    from tldw_Server_API.app.core.RAG.rag_service.types import Document

    config = _local_success_config()
    selected_files: list[str | None] = []

    async def select_mock_response(**kwargs: Any) -> dict[str, Any]:
        response_file, response = _selected_wire_response(config, kwargs["messages"])
        selected_files.append(response_file)
        return response

    preexisting_playwright_document = Document(
        id="preexisting-playwright-document",
        content="Playwright is a pre-existing unrelated document.",
        metadata={"source": "media_db", "title": "preexisting-playwright"},
    )
    seeded_fixture = Document(
        id="task2c-playwright-document",
        content=f"{_PLAYWRIGHT_DOCUMENT}\nFixture token: {_PLAYWRIGHT_SHARED_DB_TOKEN}",
        metadata={"source": "media_db", "title": _PLAYWRIGHT_SHARED_DB_TOKEN},
    )
    # Retrieval must scope the final RAG context to the generated-token fixture.
    context = SimpleNamespace(documents=[seeded_fixture])
    assert preexisting_playwright_document.metadata["title"] != seeded_fixture.metadata["title"]

    with patch(
        "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
        new=select_mock_response,
    ):
        result = asyncio.run(
            LLMGenerator(GenerationConfig(provider="openai", model="local-uat-chat")).generate(
                context,
                f"Playwright {_PLAYWRIGHT_SHARED_DB_TOKEN}",
            )
        )

    assert selected_files == ["chat/playwright-grounded.json"]
    assert result.response == _PLAYWRIGHT_DOCUMENT


@pytest.mark.unit
def test_unique_playwright_stream_context_requires_the_same_title_and_query_token() -> None:
    """The live stream shape must not match when its title/query tokens diverge."""
    from tldw_Server_API.app.core.RAG.rag_service.generation import (
        GenerationConfig,
        LLMGenerator,
    )
    from tldw_Server_API.app.core.RAG.rag_service.types import Document

    config = _local_success_config()
    selected_files: list[str | None] = []

    async def select_mock_response(**kwargs: Any) -> dict[str, Any]:
        response_file, response = _selected_wire_response(config, kwargs["messages"])
        selected_files.append(response_file)
        return response

    context = SimpleNamespace(
        documents=[
            Document(
                id="task2c-playwright-stream-document",
                content=_PLAYWRIGHT_DOCUMENT,
                metadata={"source": "media_db", "title": _PLAYWRIGHT_SHARED_DB_TOKEN},
            )
        ]
    )
    with patch(
        "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
        new=select_mock_response,
    ):
        result = asyncio.run(
            LLMGenerator(GenerationConfig(provider="openai", model="local-uat-chat")).generate(
                context,
                f"Playwright {_PLAYWRIGHT_SHARED_DB_TOKEN}",
            )
        )

    assert selected_files == ["chat/playwright-grounded.json"]
    assert result.response == _PLAYWRIGHT_DOCUMENT


@pytest.mark.unit
def test_unique_playwright_stream_context_rejects_a_mismatched_query_token() -> None:
    """A matching document title alone must not select the grounded response."""
    from tldw_Server_API.app.core.RAG.rag_service.generation import (
        GenerationConfig,
        LLMGenerator,
    )
    from tldw_Server_API.app.core.RAG.rag_service.types import Document

    config = _local_success_config()
    selected_files: list[str | None] = []

    async def select_mock_response(**kwargs: Any) -> dict[str, Any]:
        response_file, response = _selected_wire_response(config, kwargs["messages"])
        selected_files.append(response_file)
        return response

    context = SimpleNamespace(
        documents=[
            Document(
                id="task2c-playwright-mismatched-token-document",
                content=_PLAYWRIGHT_DOCUMENT,
                metadata={"source": "media_db", "title": _PLAYWRIGHT_SHARED_DB_TOKEN},
            )
        ]
    )
    with patch(
        "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
        new=select_mock_response,
    ):
        result = asyncio.run(
            LLMGenerator(GenerationConfig(provider="openai", model="local-uat-chat")).generate(
                context,
                f"Playwright {_PLAYWRIGHT_MISMATCHED_TOKEN}",
            )
        )

    assert selected_files == ["chat/source-summary.json"]
    assert "research note" in result.response


@pytest.mark.unit
def test_multimodal_playwright_user_message_selects_grounded_answer() -> None:
    """The OpenAI text-part shape from the live chat request must dispatch narrowly."""
    config = _local_success_config()
    response_file, response = _selected_wire_response(
        config,
        [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {
                "role": "user",
                "content": [{"type": "text", "text": _PLAYWRIGHT_QUESTION}],
            },
        ],
    )

    assert response_file == "chat/playwright-grounded.json"
    assert "Playwright" in response["choices"][0]["message"]["content"]


@pytest.mark.unit
def test_non_string_multimodal_text_does_not_coerce_into_a_playwright_match() -> None:
    """Malformed text parts must not select a scenario through string coercion."""
    config = _local_success_config()
    response_file, response = _selected_wire_response(
        config,
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": {"nested": "Please summarize this research note."},
                    }
                ],
            },
        ],
    )

    assert response_file == "chat/default.json"
    assert response["choices"][0]["message"]["content"] == (
        "onboarding UAT ready. The mock provider returned a deterministic success response."
    )


@pytest.mark.unit
def test_existing_source_summary_and_default_prompts_keep_their_responses() -> None:
    """Narrow critical scenarios must not shadow the established mock fallbacks."""
    config = _local_success_config()

    source_file, source_response = _selected_wire_response(
        config,
        [{"role": "user", "content": "Please summarize this research note."}],
    )
    default_file, default_response = _selected_wire_response(
        config,
        [{"role": "user", "content": "How should I organize a weekly study plan?"}],
    )

    assert source_file == "chat/source-summary.json"
    assert "research note" in source_response["choices"][0]["message"]["content"]
    assert default_file == "chat/default.json"
    assert default_response["choices"][0]["message"]["content"] == (
        "onboarding UAT ready. The mock provider returned a deterministic success response."
    )
