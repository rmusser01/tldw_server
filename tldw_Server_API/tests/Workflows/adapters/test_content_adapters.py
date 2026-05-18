"""Comprehensive tests for content generation adapters.

This module tests all 15 content adapters:
1. run_summarize_adapter - Summarize text
2. run_citations_adapter - Extract citations
3. run_bibliography_generate_adapter - Generate bibliography
4. run_image_gen_adapter - Generate images
5. run_image_describe_adapter - Describe images
6. run_rerank_adapter - Rerank search results
7. run_flashcard_generate_adapter - Generate flashcards
8. run_quiz_generate_adapter - Generate quiz questions
9. run_outline_generate_adapter - Generate outline
10. run_glossary_extract_adapter - Extract glossary terms
11. run_mindmap_generate_adapter - Generate mindmap
12. run_report_generate_adapter - Generate report
13. run_newsletter_generate_adapter - Generate newsletter
14. run_slides_generate_adapter - Generate slides
15. run_diagram_generate_adapter - Generate diagram
"""

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# Helper fixtures
@pytest.fixture
def base_context() -> Dict[str, Any]:
    """Base context for adapter tests."""
    return {
        "user_id": "1",
        "workflow_id": "test_workflow",
        "step_run_id": "test_step_123",
    }


@pytest.fixture
def sample_documents() -> list:
    """Sample documents for testing adapters that require documents."""
    return [
        {
            "id": "doc_1",
            "content": "Machine learning is a subset of artificial intelligence.",
            "text": "Machine learning is a subset of artificial intelligence.",
            "metadata": {
                "title": "Introduction to ML",
                "author": "John Smith",
                "date": "2024",
            },
            "score": 0.9,
            "source_id": "source_1",
        },
        {
            "id": "doc_2",
            "content": "Deep learning uses neural networks with many layers.",
            "text": "Deep learning uses neural networks with many layers.",
            "metadata": {
                "title": "Deep Learning Basics",
                "author": "Jane Doe",
                "date": "2023",
            },
            "score": 0.85,
            "source_id": "source_2",
        },
        {
            "id": "doc_3",
            "content": "Natural language processing enables computers to understand human language.",
            "text": "Natural language processing enables computers to understand human language.",
            "metadata": {
                "title": "NLP Fundamentals",
                "author": "Bob Wilson",
                "date": "2024",
            },
            "score": 0.8,
            "source_id": "source_3",
        },
    ]


@pytest.fixture
def sample_long_text() -> str:
    """Sample long text for summarization and other content generation tests."""
    return """
    Artificial Intelligence (AI) has transformed the way we interact with technology.
    Machine learning, a subset of AI, enables systems to learn from data and improve
    over time without being explicitly programmed. Deep learning, which uses neural
    networks with many layers, has achieved remarkable results in image recognition,
    natural language processing, and game playing.

    The history of AI dates back to the 1950s when Alan Turing proposed the famous
    Turing Test to determine if a machine could exhibit intelligent behavior. Since
    then, AI has gone through several periods of optimism followed by disappointment,
    often called "AI winters." However, recent advances in computing power and data
    availability have led to a renaissance in AI research and applications.

    Today, AI is used in various domains including healthcare, finance, transportation,
    and entertainment. Self-driving cars use AI to navigate roads safely, while
    virtual assistants like Siri and Alexa use natural language processing to
    understand and respond to user queries. In healthcare, AI helps diagnose diseases
    and develop personalized treatment plans.

    The ethical implications of AI are also being widely discussed. Issues such as
    bias in AI systems, job displacement due to automation, and the potential for
    misuse of AI technology are important considerations for researchers, policymakers,
    and society at large. Ensuring that AI is developed and deployed responsibly is
    crucial for its beneficial integration into our lives.
    """


def mock_chat_response(content: str) -> Dict[str, Any]:
    """Create a mock OpenAI-style chat response."""
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                    "role": "assistant",
                }
            }
        ]
    }


# =============================================================================
# Test: run_summarize_adapter
# =============================================================================


class TestSummarizeAdapter:
    """Tests for the summarize adapter."""

    @pytest.mark.asyncio
    async def test_summarize_adapter_test_mode(self, monkeypatch, base_context, sample_long_text):
        """Test summarize adapter in TEST_MODE returns simulated summary."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.content import run_summarize_adapter

        config = {"text": sample_long_text}
        result = await run_summarize_adapter(config, base_context)

        assert "summary" in result
        assert "text" in result
        assert result.get("simulated") is True
        assert result.get("api_name") == "openai"
        assert len(result["summary"]) > 0

    @pytest.mark.asyncio
    async def test_summarize_adapter_missing_text(self, monkeypatch, base_context):
        """Test summarize adapter with missing text returns error."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.content import run_summarize_adapter

        config = {"text": ""}
        result = await run_summarize_adapter(config, base_context)

        assert result.get("error") == "missing_text"

    @pytest.mark.asyncio
    async def test_summarize_adapter_uses_prev_context(self, monkeypatch, sample_long_text):
        """Test summarize adapter can use text from prev context."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.content import run_summarize_adapter

        config = {}
        context = {
            "user_id": "1",
            "prev": {"text": sample_long_text},
        }
        result = await run_summarize_adapter(config, context)

        assert "summary" in result
        assert result.get("simulated") is True

    @pytest.mark.asyncio
    async def test_summarize_adapter_cancellation(self, base_context, sample_long_text):
        """Test summarize adapter respects cancellation."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_summarize_adapter

        context = {**base_context, "is_cancelled": lambda: True}
        config = {"text": sample_long_text}

        result = await run_summarize_adapter(config, context)
        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_summarize_adapter_with_custom_options(self, monkeypatch, base_context, sample_long_text):
        """Test summarize adapter with custom options."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.content import run_summarize_adapter

        config = {
            "text": sample_long_text,
            "api_name": "anthropic",
            "temperature": 0.5,
            "custom_prompt": "Make it very brief",
        }
        result = await run_summarize_adapter(config, base_context)

        assert result.get("api_name") == "anthropic"
        assert "summary" in result

    @pytest.mark.asyncio
    async def test_summarize_adapter_real_llm_call(self, monkeypatch, base_context, sample_long_text):
        """Test summarize adapter with mocked LLM call (non-test mode)."""
        monkeypatch.delenv("TEST_MODE", raising=False)

        from tldw_Server_API.app.core.Workflows.adapters.content import run_summarize_adapter

        mock_analyze = MagicMock(return_value="This is a summary about AI and its applications.")

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.content.summarize.asyncio.to_thread",
            new_callable=AsyncMock,
            return_value="This is a summary about AI and its applications.",
        ):
            config = {"text": sample_long_text}
            result = await run_summarize_adapter(config, base_context)

            assert "summary" in result
            assert result["summary"] == "This is a summary about AI and its applications."


# =============================================================================
# Test: run_citations_adapter
# =============================================================================


class TestCitationsAdapter:
    """Tests for the citations adapter."""

    @pytest.mark.asyncio
    async def test_citations_adapter_test_mode(self, monkeypatch, base_context, sample_documents):
        """Test citations adapter in TEST_MODE returns simulated citations."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.content import run_citations_adapter

        config = {"documents": sample_documents, "style": "apa"}
        result = await run_citations_adapter(config, base_context)

        assert "citations" in result
        assert "chunk_citations" in result
        assert "inline_markers" in result
        assert "citation_map" in result
        assert result.get("simulated") is True
        assert result.get("style") == "apa"
        assert len(result["citations"]) > 0

    @pytest.mark.asyncio
    async def test_citations_adapter_missing_documents(self, monkeypatch, base_context):
        """Test citations adapter with missing documents returns error."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.content import run_citations_adapter

        config = {}
        result = await run_citations_adapter(config, base_context)

        assert result.get("error") == "missing_documents"
        assert result["citations"] == []

    @pytest.mark.asyncio
    async def test_citations_adapter_different_styles(self, monkeypatch, base_context, sample_documents):
        """Test citations adapter with different citation styles."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.content import run_citations_adapter

        for style in ["apa", "mla", "chicago", "harvard", "ieee"]:
            config = {"documents": sample_documents, "style": style}
            result = await run_citations_adapter(config, base_context)

            assert result.get("style") == style
            assert len(result["citations"]) > 0
            # Each style should have different formatting
            if style == "apa":
                assert "(" in result["citations"][0]  # APA uses parentheses for dates
            elif style == "ieee":
                assert "[" in result["citations"][0]  # IEEE uses brackets

    @pytest.mark.asyncio
    async def test_citations_adapter_max_citations(self, monkeypatch, base_context, sample_documents):
        """Test citations adapter respects max_citations."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.content import run_citations_adapter

        config = {"documents": sample_documents, "max_citations": 2}
        result = await run_citations_adapter(config, base_context)

        assert result.get("count") <= 2

    @pytest.mark.asyncio
    async def test_citations_adapter_cancellation(self, base_context, sample_documents):
        """Test citations adapter respects cancellation."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_citations_adapter

        context = {**base_context, "is_cancelled": lambda: True}
        config = {"documents": sample_documents}

        result = await run_citations_adapter(config, context)
        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_citations_adapter_uses_prev_context(self, monkeypatch, sample_documents):
        """Test citations adapter can use documents from prev context."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.content import run_citations_adapter

        config = {}
        context = {
            "user_id": "1",
            "prev": {"documents": sample_documents},
        }
        result = await run_citations_adapter(config, context)

        assert "citations" in result
        assert len(result["citations"]) > 0


# =============================================================================
# Test: run_bibliography_generate_adapter
# =============================================================================


class TestBibliographyGenerateAdapter:
    """Tests for the bibliography generation adapter."""

    @pytest.mark.asyncio
    async def test_bibliography_generate_valid(self, base_context, sample_documents):
        """Test bibliography generation with valid sources."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_bibliography_generate_adapter

        config = {"sources": sample_documents, "format": "apa", "sort_by": "author"}
        result = await run_bibliography_generate_adapter(config, base_context)

        assert "bibliography" in result
        assert "citations" in result
        assert result.get("format") == "apa"
        assert result.get("count") == len(sample_documents)

    @pytest.mark.asyncio
    async def test_bibliography_generate_missing_sources(self, base_context):
        """Test bibliography generation with missing sources returns error."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_bibliography_generate_adapter

        config = {}
        result = await run_bibliography_generate_adapter(config, base_context)

        assert result.get("error") == "missing_sources"

    @pytest.mark.asyncio
    async def test_bibliography_generate_different_formats(self, base_context, sample_documents):
        """Test bibliography generation with different citation formats."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_bibliography_generate_adapter

        for fmt in ["apa", "mla", "chicago", "harvard", "bibtex"]:
            config = {"sources": sample_documents, "format": fmt}
            result = await run_bibliography_generate_adapter(config, base_context)

            assert result.get("format") == fmt
            assert len(result.get("bibliography", "")) > 0

            # BibTeX has special format
            if fmt == "bibtex":
                assert "@misc{" in result["bibliography"]

    @pytest.mark.asyncio
    async def test_bibliography_generate_sort_by_options(self, base_context, sample_documents):
        """Test bibliography generation with different sort options."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_bibliography_generate_adapter

        for sort_by in ["author", "date", "title"]:
            config = {"sources": sample_documents, "sort_by": sort_by}
            result = await run_bibliography_generate_adapter(config, base_context)

            assert len(result.get("citations", [])) == len(sample_documents)

    @pytest.mark.asyncio
    async def test_bibliography_generate_cancellation(self, base_context, sample_documents):
        """Test bibliography generation respects cancellation."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_bibliography_generate_adapter

        context = {**base_context, "is_cancelled": lambda: True}
        config = {"sources": sample_documents}

        result = await run_bibliography_generate_adapter(config, context)
        assert result.get("__status__") == "cancelled"


# =============================================================================
# Test: run_image_gen_adapter
# =============================================================================


class TestImageGenAdapter:
    """Tests for the image generation adapter."""

    @pytest.mark.asyncio
    async def test_image_gen_adapter_test_mode(self, monkeypatch, base_context):
        """Test image gen adapter in TEST_MODE returns simulated image."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.content import run_image_gen_adapter

        config = {"prompt": "A beautiful sunset over the ocean"}
        result = await run_image_gen_adapter(config, base_context)

        assert "images" in result
        assert result.get("simulated") is True
        assert result.get("count") == 1
        assert len(result["images"]) == 1
        assert "uri" in result["images"][0]
        assert "width" in result["images"][0]
        assert "height" in result["images"][0]

    @pytest.mark.asyncio
    async def test_image_gen_adapter_missing_prompt(self, monkeypatch, base_context):
        """Test image gen adapter with missing prompt returns error."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.content import run_image_gen_adapter

        config = {"prompt": ""}
        result = await run_image_gen_adapter(config, base_context)

        assert result.get("error") == "missing_prompt"

    @pytest.mark.asyncio
    async def test_image_gen_adapter_custom_dimensions(self, monkeypatch, base_context):
        """Test image gen adapter with custom dimensions."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.content import run_image_gen_adapter

        config = {
            "prompt": "A cat",
            "width": 1024,
            "height": 768,
        }
        result = await run_image_gen_adapter(config, base_context)

        assert result["images"][0]["width"] == 1024
        assert result["images"][0]["height"] == 768

    @pytest.mark.asyncio
    async def test_image_gen_adapter_different_formats(self, monkeypatch, base_context):
        """Test image gen adapter with different output formats."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.content import run_image_gen_adapter

        for fmt in ["png", "jpg", "webp"]:
            config = {"prompt": "A dog", "format": fmt}
            result = await run_image_gen_adapter(config, base_context)

            assert result["images"][0]["format"] == fmt

    @pytest.mark.asyncio
    async def test_image_gen_adapter_cancellation(self, base_context):
        """Test image gen adapter respects cancellation."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_image_gen_adapter

        context = {**base_context, "is_cancelled": lambda: True}
        config = {"prompt": "A landscape"}

        result = await run_image_gen_adapter(config, context)
        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_image_gen_adapter_with_negative_prompt(self, monkeypatch, base_context):
        """Test image gen adapter with negative prompt."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.content import run_image_gen_adapter

        config = {
            "prompt": "A beautiful landscape",
            "negative_prompt": "ugly, blurry, low quality",
        }
        result = await run_image_gen_adapter(config, base_context)

        assert "images" in result
        assert result.get("simulated") is True

    @pytest.mark.asyncio
    async def test_image_gen_adapter_prompt_refinement_opt_in(self, monkeypatch, base_context):
        """Test image gen adapter applies prompt refinement when enabled."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.content import run_image_gen_adapter

        config = {"prompt": "cat portrait", "prompt_refinement": True}
        result = await run_image_gen_adapter(config, base_context)

        assert "high detail" in result.get("prompt", "")
        assert result.get("prompt_refinement_mode") == "basic"

    @pytest.mark.asyncio
    async def test_image_gen_adapter_prompt_refinement_opt_out(self, monkeypatch, base_context):
        """Test image gen adapter keeps prompt unchanged when refinement is disabled."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.content import run_image_gen_adapter

        config = {"prompt": "cat portrait", "prompt_refinement": False}
        result = await run_image_gen_adapter(config, base_context)

        assert result.get("prompt") == "cat portrait"
        assert result.get("prompt_refinement_mode") == "off"

    @pytest.mark.asyncio
    async def test_image_gen_adapter_sanitizes_backend_errors(self, monkeypatch, base_context):
        """Test image generation backend failures hide raw exception details."""
        monkeypatch.delenv("TEST_MODE", raising=False)
        monkeypatch.delenv("TLDW_TEST_MODE", raising=False)

        from tldw_Server_API.app.core.Workflows.adapters.content import run_image_gen_adapter

        with (
            patch(
                "tldw_Server_API.app.core.Workflows.adapters.content.image.is_test_mode",
                return_value=False,
            ),
            patch(
                "tldw_Server_API.app.core.Image_Generation.adapter_registry.get_registry",
                side_effect=RuntimeError("image backend exploded at /private/image-cache"),
            ),
        ):
            result = await run_image_gen_adapter({"prompt": "A clean test image"}, base_context)

        assert result == {"error": "image_gen_error"}


# =============================================================================
# Test: run_image_describe_adapter
# =============================================================================


class TestImageDescribeAdapter:
    """Tests for the image description adapter."""

    @pytest.mark.asyncio
    async def test_image_describe_adapter_missing_image(self, base_context):
        """Test image describe adapter with missing image returns error."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_image_describe_adapter

        config = {}
        result = await run_image_describe_adapter(config, base_context)

        assert result.get("error") == "missing_image"

    @pytest.mark.asyncio
    async def test_image_describe_adapter_with_url(self, monkeypatch, base_context):
        """Test image describe adapter with image URL."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_image_describe_adapter

        mock_response = mock_chat_response("This image shows a beautiful sunset over the ocean.")

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.content.image.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            config = {
                "image_url": "https://example.com/image.jpg",
                "prompt": "Describe this image",
            }
            result = await run_image_describe_adapter(config, base_context)

            assert "description" in result
            assert "sunset" in result["description"].lower()

    @pytest.mark.asyncio
    async def test_image_describe_adapter_with_base64(self, monkeypatch, base_context):
        """Test image describe adapter with base64 image."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_image_describe_adapter

        mock_response = mock_chat_response("This is a test image description.")

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.content.image.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            config = {
                "image_base64": "dGVzdGltYWdlZGF0YQ==",  # Base64 encoded test data
                "prompt": "Describe this image",
            }
            result = await run_image_describe_adapter(config, base_context)

            assert "description" in result

    @pytest.mark.asyncio
    async def test_image_describe_adapter_cancellation(self, base_context):
        """Test image describe adapter respects cancellation."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_image_describe_adapter

        context = {**base_context, "is_cancelled": lambda: True}
        config = {"image_url": "https://example.com/image.jpg"}

        result = await run_image_describe_adapter(config, context)
        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_image_describe_adapter_sanitizes_file_read_errors(self, base_context):
        """Test image description file-read failures hide raw path details."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_image_describe_adapter

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.content.image.resolve_workflow_file_path",
            side_effect=RuntimeError("cannot read /private/image-cache/secret.png"),
        ):
            result = await run_image_describe_adapter(
                {"image_path": "secret.png", "prompt": "Describe this image"},
                base_context,
            )

        assert result == {"description": "", "error": "image_read_error"}

    @pytest.mark.asyncio
    async def test_image_describe_adapter_sanitizes_backend_errors(self, base_context):
        """Test image description backend failures hide raw exception details."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_image_describe_adapter

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.content.image.perform_chat_api_call_async",
            new_callable=AsyncMock,
            side_effect=RuntimeError("vision backend exploded at /private/image-cache"),
        ):
            result = await run_image_describe_adapter(
                {"image_url": "https://example.com/image.jpg", "prompt": "Describe this image"},
                base_context,
            )

        assert result == {"description": "", "error": "image_describe_error"}


# =============================================================================
# Test: run_rerank_adapter
# =============================================================================


class TestRerankAdapter:
    """Tests for the rerank adapter."""

    @pytest.mark.asyncio
    async def test_rerank_adapter_test_mode(self, monkeypatch, base_context, sample_documents):
        """Test rerank adapter in TEST_MODE returns simulated reranking."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.content import run_rerank_adapter

        config = {
            "query": "What is machine learning?",
            "documents": sample_documents,
        }
        result = await run_rerank_adapter(config, base_context)

        assert "documents" in result
        assert result.get("simulated") is True
        assert result.get("strategy") == "flashrank"
        # Should have scores in decreasing order
        if len(result["documents"]) > 1:
            assert result["documents"][0]["score"] >= result["documents"][1]["score"]

    @pytest.mark.asyncio
    async def test_rerank_adapter_missing_query(self, monkeypatch, base_context, sample_documents):
        """Test rerank adapter with missing query returns error."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.content import run_rerank_adapter

        config = {"documents": sample_documents}
        result = await run_rerank_adapter(config, base_context)

        assert result.get("error") == "missing_query"

    @pytest.mark.asyncio
    async def test_rerank_adapter_missing_documents(self, monkeypatch, base_context):
        """Test rerank adapter with missing documents returns error."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.content import run_rerank_adapter

        config = {"query": "test query"}
        result = await run_rerank_adapter(config, base_context)

        assert result.get("error") == "missing_documents"

    @pytest.mark.asyncio
    async def test_rerank_adapter_top_k(self, monkeypatch, base_context, sample_documents):
        """Test rerank adapter respects top_k parameter."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.content import run_rerank_adapter

        config = {
            "query": "AI concepts",
            "documents": sample_documents,
            "top_k": 2,
        }
        result = await run_rerank_adapter(config, base_context)

        assert result.get("count") <= 2

    @pytest.mark.asyncio
    async def test_rerank_adapter_cancellation(self, base_context, sample_documents):
        """Test rerank adapter respects cancellation."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_rerank_adapter

        context = {**base_context, "is_cancelled": lambda: True}
        config = {"query": "test", "documents": sample_documents}

        result = await run_rerank_adapter(config, context)
        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_rerank_adapter_different_strategies(self, monkeypatch, base_context, sample_documents):
        """Test rerank adapter with different strategies in test mode."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.content import run_rerank_adapter

        for strategy in ["flashrank", "cross_encoder", "diversity"]:
            config = {
                "query": "What is AI?",
                "documents": sample_documents,
                "strategy": strategy,
            }
            result = await run_rerank_adapter(config, base_context)

            assert result.get("strategy") == strategy


# =============================================================================
# Test: run_flashcard_generate_adapter
# =============================================================================


class TestFlashcardGenerateAdapter:
    """Tests for the flashcard generation adapter."""

    @pytest.mark.asyncio
    async def test_flashcard_generate_valid(self, monkeypatch, base_context, sample_long_text):
        """Test flashcard generation with valid text."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_flashcard_generate_adapter

        mock_flashcards = json.dumps(
            [
                {"front": "What is AI?", "back": "Artificial Intelligence", "tags": ["ai"]},
                {"front": "What is ML?", "back": "Machine Learning", "tags": ["ml"]},
            ]
        )
        mock_response = mock_chat_response(mock_flashcards)

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.content.generation.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            config = {"text": sample_long_text, "num_cards": 5}
            result = await run_flashcard_generate_adapter(config, base_context)

            assert "flashcards" in result
            assert result.get("count") == 2
            assert "front" in result["flashcards"][0]
            assert "back" in result["flashcards"][0]

    @pytest.mark.asyncio
    async def test_flashcard_generate_missing_text(self, base_context):
        """Test flashcard generation with missing text returns error."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_flashcard_generate_adapter

        config = {}
        result = await run_flashcard_generate_adapter(config, base_context)

        assert result.get("error") == "missing_text"

    @pytest.mark.asyncio
    async def test_flashcard_generate_different_types(self, monkeypatch, base_context, sample_long_text):
        """Test flashcard generation with different card types."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_flashcard_generate_adapter

        mock_flashcards = json.dumps([{"front": "Q", "back": "A", "tags": []}])
        mock_response = mock_chat_response(mock_flashcards)

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.content.generation.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            for card_type in ["basic", "cloze", "basic_reverse"]:
                config = {"text": sample_long_text, "card_type": card_type}
                result = await run_flashcard_generate_adapter(config, base_context)

                # Should set model_type on cards
                if result["flashcards"]:
                    assert result["flashcards"][0].get("model_type") == card_type

    @pytest.mark.asyncio
    async def test_flashcard_generate_cancellation(self, base_context, sample_long_text):
        """Test flashcard generation respects cancellation."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_flashcard_generate_adapter

        context = {**base_context, "is_cancelled": lambda: True}
        config = {"text": sample_long_text}

        result = await run_flashcard_generate_adapter(config, context)
        assert result.get("__status__") == "cancelled"


# =============================================================================
# Test: run_quiz_generate_adapter
# =============================================================================


class TestQuizGenerateAdapter:
    """Tests for the quiz generation adapter."""

    @pytest.mark.asyncio
    async def test_quiz_generate_valid(self, monkeypatch, base_context, sample_long_text):
        """Test quiz generation with valid text."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_quiz_generate_adapter

        mock_questions = json.dumps(
            [
                {
                    "question_type": "multiple_choice",
                    "question_text": "What is AI?",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_answer": 0,
                    "explanation": "AI stands for Artificial Intelligence",
                    "points": 1,
                }
            ]
        )
        mock_response = mock_chat_response(mock_questions)

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.content.generation.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            config = {"text": sample_long_text, "num_questions": 5}
            result = await run_quiz_generate_adapter(config, base_context)

            assert "questions" in result
            assert result.get("count") == 1
            assert "question_text" in result["questions"][0]

    @pytest.mark.asyncio
    async def test_quiz_generate_missing_text(self, base_context):
        """Test quiz generation with missing text returns error."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_quiz_generate_adapter

        config = {}
        result = await run_quiz_generate_adapter(config, base_context)

        assert result.get("error") == "missing_text"

    @pytest.mark.asyncio
    async def test_quiz_generate_different_types(self, monkeypatch, base_context, sample_long_text):
        """Test quiz generation with different question types."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_quiz_generate_adapter

        mock_questions = json.dumps([{"question_type": "true_false", "question_text": "Q", "correct_answer": True}])
        mock_response = mock_chat_response(mock_questions)

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.content.generation.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            config = {
                "text": sample_long_text,
                "question_types": ["multiple_choice", "true_false"],
            }
            result = await run_quiz_generate_adapter(config, base_context)

            assert "questions" in result

    @pytest.mark.asyncio
    async def test_quiz_generate_cancellation(self, base_context, sample_long_text):
        """Test quiz generation respects cancellation."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_quiz_generate_adapter

        context = {**base_context, "is_cancelled": lambda: True}
        config = {"text": sample_long_text}

        result = await run_quiz_generate_adapter(config, context)
        assert result.get("__status__") == "cancelled"


# =============================================================================
# Test: run_outline_generate_adapter
# =============================================================================


class TestOutlineGenerateAdapter:
    """Tests for the outline generation adapter."""

    @pytest.mark.asyncio
    async def test_outline_generate_valid(self, monkeypatch, base_context, sample_long_text):
        """Test outline generation with valid text."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_outline_generate_adapter

        mock_outline = json.dumps(
            {
                "sections": [
                    {"title": "Introduction", "level": 1, "subsections": []},
                    {
                        "title": "Main Content",
                        "level": 1,
                        "subsections": [{"title": "Part A", "level": 2, "subsections": []}],
                    },
                ]
            }
        )
        mock_response = mock_chat_response(mock_outline)

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.content.generation.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            config = {"text": sample_long_text, "max_depth": 3}
            result = await run_outline_generate_adapter(config, base_context)

            assert "outline" in result
            assert "sections" in result.get("outline", {})

    @pytest.mark.asyncio
    async def test_outline_generate_missing_text(self, base_context):
        """Test outline generation with missing text returns error."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_outline_generate_adapter

        config = {}
        result = await run_outline_generate_adapter(config, base_context)

        assert result.get("error") == "missing_text"

    @pytest.mark.asyncio
    async def test_outline_generate_cancellation(self, base_context, sample_long_text):
        """Test outline generation respects cancellation."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_outline_generate_adapter

        context = {**base_context, "is_cancelled": lambda: True}
        config = {"text": sample_long_text}

        result = await run_outline_generate_adapter(config, context)
        assert result.get("__status__") == "cancelled"


# =============================================================================
# Test: run_glossary_extract_adapter
# =============================================================================


class TestGlossaryExtractAdapter:
    """Tests for the glossary extraction adapter."""

    @pytest.mark.asyncio
    async def test_glossary_extract_valid(self, monkeypatch, base_context, sample_long_text):
        """Test glossary extraction with valid text."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_glossary_extract_adapter

        mock_glossary = json.dumps(
            [
                {"term": "AI", "definition": "Artificial Intelligence"},
                {"term": "ML", "definition": "Machine Learning"},
            ]
        )
        mock_response = mock_chat_response(mock_glossary)

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.content.generation.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            config = {"text": sample_long_text, "max_terms": 10}
            result = await run_glossary_extract_adapter(config, base_context)

            assert "glossary" in result
            assert result.get("count") == 2
            assert "term" in result["glossary"][0]
            assert "definition" in result["glossary"][0]

    @pytest.mark.asyncio
    async def test_glossary_extract_missing_text(self, base_context):
        """Test glossary extraction with missing text returns error."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_glossary_extract_adapter

        config = {}
        result = await run_glossary_extract_adapter(config, base_context)

        assert result.get("error") == "missing_text"

    @pytest.mark.asyncio
    async def test_glossary_extract_cancellation(self, base_context, sample_long_text):
        """Test glossary extraction respects cancellation."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_glossary_extract_adapter

        context = {**base_context, "is_cancelled": lambda: True}
        config = {"text": sample_long_text}

        result = await run_glossary_extract_adapter(config, context)
        assert result.get("__status__") == "cancelled"


# =============================================================================
# Test: run_mindmap_generate_adapter
# =============================================================================


class TestMindmapGenerateAdapter:
    """Tests for the mindmap generation adapter."""

    @pytest.mark.asyncio
    async def test_mindmap_generate_valid(self, monkeypatch, base_context, sample_long_text):
        """Test mindmap generation with valid text."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_mindmap_generate_adapter

        mock_mindmap = json.dumps(
            {
                "central": "Artificial Intelligence",
                "branches": [
                    {"topic": "Machine Learning", "children": []},
                    {"topic": "Deep Learning", "children": []},
                ],
            }
        )
        mock_response = mock_chat_response(mock_mindmap)

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.content.generation.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            config = {"text": sample_long_text, "max_branches": 5}
            result = await run_mindmap_generate_adapter(config, base_context)

            assert "mindmap" in result
            assert "central" in result.get("mindmap", {})

    @pytest.mark.asyncio
    async def test_mindmap_generate_missing_text(self, base_context):
        """Test mindmap generation with missing text returns error."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_mindmap_generate_adapter

        config = {}
        result = await run_mindmap_generate_adapter(config, base_context)

        assert result.get("error") == "missing_text"

    @pytest.mark.asyncio
    async def test_mindmap_generate_cancellation(self, base_context, sample_long_text):
        """Test mindmap generation respects cancellation."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_mindmap_generate_adapter

        context = {**base_context, "is_cancelled": lambda: True}
        config = {"text": sample_long_text}

        result = await run_mindmap_generate_adapter(config, context)
        assert result.get("__status__") == "cancelled"


# =============================================================================
# Test: run_report_generate_adapter
# =============================================================================


class TestReportGenerateAdapter:
    """Tests for the report generation adapter."""

    @pytest.mark.asyncio
    async def test_report_generate_valid(self, monkeypatch, base_context, sample_long_text):
        """Test report generation with valid content."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_report_generate_adapter

        mock_report = "# AI Report\n\n## Introduction\n\nThis report covers AI topics..."
        mock_response = mock_chat_response(mock_report)

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.content.generation.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            config = {
                "content": sample_long_text,
                "title": "AI Overview",
                "format": "markdown",
            }
            result = await run_report_generate_adapter(config, base_context)

            assert "report" in result
            assert result.get("title") == "AI Overview"
            assert result.get("format") == "markdown"

    @pytest.mark.asyncio
    async def test_report_generate_missing_content(self, base_context):
        """Test report generation with missing content returns error."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_report_generate_adapter

        config = {}
        result = await run_report_generate_adapter(config, base_context)

        assert result.get("error") == "missing_content"

    @pytest.mark.asyncio
    async def test_report_generate_with_sections(self, monkeypatch, base_context, sample_long_text):
        """Test report generation with specified sections."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_report_generate_adapter

        mock_report = "# Report\n\n## Intro\n## Methods\n## Results"
        mock_response = mock_chat_response(mock_report)

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.content.generation.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            config = {
                "content": sample_long_text,
                "sections": ["Introduction", "Methods", "Results"],
            }
            result = await run_report_generate_adapter(config, base_context)

            assert "report" in result

    @pytest.mark.asyncio
    async def test_report_generate_cancellation(self, base_context, sample_long_text):
        """Test report generation respects cancellation."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_report_generate_adapter

        context = {**base_context, "is_cancelled": lambda: True}
        config = {"content": sample_long_text}

        result = await run_report_generate_adapter(config, context)
        assert result.get("__status__") == "cancelled"


# =============================================================================
# Test: run_newsletter_generate_adapter
# =============================================================================


class TestNewsletterGenerateAdapter:
    """Tests for the newsletter generation adapter."""

    @pytest.mark.asyncio
    async def test_newsletter_generate_with_items(self, monkeypatch, base_context):
        """Test newsletter generation with items list."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_newsletter_generate_adapter

        mock_newsletter = "# Weekly Newsletter\n\n## Top Stories\n\n..."
        mock_response = mock_chat_response(mock_newsletter)

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.content.generation.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            config = {
                "items": [
                    {"title": "Item 1", "summary": "Summary 1", "url": "https://example.com/1"},
                    {"title": "Item 2", "summary": "Summary 2", "url": "https://example.com/2"},
                ],
                "title": "Weekly Update",
            }
            result = await run_newsletter_generate_adapter(config, base_context)

            assert "newsletter" in result
            assert result.get("title") == "Weekly Update"

    @pytest.mark.asyncio
    async def test_newsletter_generate_with_content(self, monkeypatch, base_context, sample_long_text):
        """Test newsletter generation with raw content."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_newsletter_generate_adapter

        mock_newsletter = "# Newsletter\n\n..."
        mock_response = mock_chat_response(mock_newsletter)

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.content.generation.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            config = {"content": sample_long_text, "title": "AI Newsletter"}
            result = await run_newsletter_generate_adapter(config, base_context)

            assert "newsletter" in result

    @pytest.mark.asyncio
    async def test_newsletter_generate_missing_content(self, base_context):
        """Test newsletter generation with missing content returns error."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_newsletter_generate_adapter

        config = {}
        result = await run_newsletter_generate_adapter(config, base_context)

        assert result.get("error") == "missing_items_or_content"

    @pytest.mark.asyncio
    async def test_newsletter_generate_cancellation(self, base_context, sample_long_text):
        """Test newsletter generation respects cancellation."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_newsletter_generate_adapter

        context = {**base_context, "is_cancelled": lambda: True}
        config = {"content": sample_long_text}

        result = await run_newsletter_generate_adapter(config, context)
        assert result.get("__status__") == "cancelled"


# =============================================================================
# Test: run_slides_generate_adapter
# =============================================================================


class TestSlidesGenerateAdapter:
    """Tests for the slides generation adapter."""

    @pytest.mark.asyncio
    async def test_slides_generate_valid(self, monkeypatch, base_context, sample_long_text):
        """Test slides generation with valid content."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_slides_generate_adapter

        mock_slides = json.dumps(
            [
                {"slide_number": 1, "title": "Title Slide", "bullets": [], "speaker_notes": "Welcome"},
                {
                    "slide_number": 2,
                    "title": "Overview",
                    "bullets": ["Point 1", "Point 2"],
                    "speaker_notes": "Overview notes",
                },
            ]
        )
        mock_response = mock_chat_response(mock_slides)

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.content.generation.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            config = {
                "content": sample_long_text,
                "title": "AI Presentation",
                "num_slides": 10,
            }
            result = await run_slides_generate_adapter(config, base_context)

            assert "slides" in result
            assert result.get("title") == "AI Presentation"
            assert result.get("slide_count") == 2

    @pytest.mark.asyncio
    async def test_slides_generate_missing_content(self, base_context):
        """Test slides generation with missing content returns error."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_slides_generate_adapter

        config = {}
        result = await run_slides_generate_adapter(config, base_context)

        assert result.get("error") == "missing_content"

    @pytest.mark.asyncio
    async def test_slides_generate_different_styles(self, monkeypatch, base_context, sample_long_text):
        """Test slides generation with different styles."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_slides_generate_adapter

        mock_slides = json.dumps([{"slide_number": 1, "title": "Title", "bullets": []}])
        mock_response = mock_chat_response(mock_slides)

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.content.generation.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            for style in ["professional", "educational", "casual"]:
                config = {"content": sample_long_text, "style": style}
                result = await run_slides_generate_adapter(config, base_context)

                assert "slides" in result

    @pytest.mark.asyncio
    async def test_slides_generate_cancellation(self, base_context, sample_long_text):
        """Test slides generation respects cancellation."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_slides_generate_adapter

        context = {**base_context, "is_cancelled": lambda: True}
        config = {"content": sample_long_text}

        result = await run_slides_generate_adapter(config, context)
        assert result.get("__status__") == "cancelled"


# =============================================================================
# Test: run_diagram_generate_adapter
# =============================================================================


class TestDiagramGenerateAdapter:
    """Tests for the diagram generation adapter."""

    @pytest.mark.asyncio
    async def test_diagram_generate_mermaid(self, monkeypatch, base_context, sample_long_text):
        """Test diagram generation with mermaid format."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_diagram_generate_adapter

        mock_diagram = "flowchart TD\n    A[Start] --> B[Process]\n    B --> C[End]"
        mock_response = mock_chat_response(mock_diagram)

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.content.generation.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            config = {
                "content": sample_long_text,
                "diagram_type": "flowchart",
                "format": "mermaid",
            }
            result = await run_diagram_generate_adapter(config, base_context)

            assert "diagram" in result
            assert result.get("format") == "mermaid"
            assert result.get("diagram_type") == "flowchart"

    @pytest.mark.asyncio
    async def test_diagram_generate_graphviz(self, monkeypatch, base_context, sample_long_text):
        """Test diagram generation with graphviz format."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_diagram_generate_adapter

        mock_diagram = "digraph G {\n    A -> B;\n    B -> C;\n}"
        mock_response = mock_chat_response(mock_diagram)

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.content.generation.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            config = {
                "content": sample_long_text,
                "diagram_type": "flowchart",
                "format": "graphviz",
            }
            result = await run_diagram_generate_adapter(config, base_context)

            assert "diagram" in result
            assert result.get("format") == "graphviz"

    @pytest.mark.asyncio
    async def test_diagram_generate_missing_content(self, base_context):
        """Test diagram generation with missing content returns error."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_diagram_generate_adapter

        config = {}
        result = await run_diagram_generate_adapter(config, base_context)

        assert result.get("error") == "missing_content"

    @pytest.mark.asyncio
    async def test_diagram_generate_different_types(self, monkeypatch, base_context, sample_long_text):
        """Test diagram generation with different diagram types."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_diagram_generate_adapter

        mock_diagram = "graph TD\n    A --> B"
        mock_response = mock_chat_response(mock_diagram)

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.content.generation.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            for diagram_type in ["flowchart", "sequence", "class", "er"]:
                config = {"content": sample_long_text, "diagram_type": diagram_type}
                result = await run_diagram_generate_adapter(config, base_context)

                assert result.get("diagram_type") == diagram_type

    @pytest.mark.asyncio
    async def test_diagram_generate_code_block_cleanup(self, monkeypatch, base_context, sample_long_text):
        """Test diagram generation cleans up markdown code blocks."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_diagram_generate_adapter

        # Response with code block markers
        mock_diagram = "```mermaid\nflowchart TD\n    A --> B\n```"
        mock_response = mock_chat_response(mock_diagram)

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.content.generation.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            config = {"content": sample_long_text}
            result = await run_diagram_generate_adapter(config, base_context)

            # Should have removed the code block markers
            assert "```" not in result.get("diagram", "")

    @pytest.mark.asyncio
    async def test_diagram_generate_cancellation(self, base_context, sample_long_text):
        """Test diagram generation respects cancellation."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_diagram_generate_adapter

        context = {**base_context, "is_cancelled": lambda: True}
        config = {"content": sample_long_text}

        result = await run_diagram_generate_adapter(config, context)
        assert result.get("__status__") == "cancelled"


# =============================================================================
# Integration Tests: Template Rendering
# =============================================================================


class TestContentAdaptersTemplateRendering:
    """Tests for template rendering across content adapters."""

    @pytest.mark.asyncio
    async def test_summarize_template_rendering(self, monkeypatch, base_context):
        """Test that summarize adapter renders templates."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.content import run_summarize_adapter

        context = {
            **base_context,
            "inputs": {"topic": "AI"},
            "prev": {"text": "This is the previous step text about {{inputs.topic}}."},
        }
        config = {"text": "{{prev.text}}"}
        result = await run_summarize_adapter(config, context)

        # The template should have been rendered
        assert "summary" in result

    @pytest.mark.asyncio
    async def test_report_template_rendering(self, monkeypatch, base_context, sample_long_text):
        """Test that report adapter renders template in title."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_report_generate_adapter

        mock_report = "# Report"
        mock_response = mock_chat_response(mock_report)

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.content.generation.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            context = {
                **base_context,
                "inputs": {"report_title": "Quarterly Analysis"},
            }
            config = {
                "content": sample_long_text,
                "title": "{{inputs.report_title}}",
            }
            result = await run_report_generate_adapter(config, context)

            assert "report" in result


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestContentAdaptersErrorHandling:
    """Tests for error handling in content adapters."""

    @pytest.mark.asyncio
    async def test_flashcard_generate_json_parse_error(self, monkeypatch, base_context, sample_long_text):
        """Test flashcard generation handles JSON parse errors gracefully."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_flashcard_generate_adapter

        # Return invalid JSON
        mock_response = mock_chat_response("This is not valid JSON")

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.content.generation.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            config = {"text": sample_long_text}
            result = await run_flashcard_generate_adapter(config, base_context)

            # Should return empty flashcards list, not crash
            assert result.get("flashcards") == []
            assert result.get("count") == 0

    @pytest.mark.asyncio
    async def test_flashcard_generate_sanitizes_backend_errors(self, base_context, sample_long_text):
        """Test flashcard generation hides backend exception details."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_flashcard_generate_adapter

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.content.generation.perform_chat_api_call_async",
            new_callable=AsyncMock,
            side_effect=RuntimeError("flashcard backend exploded at /private/content-cache"),
        ):
            result = await run_flashcard_generate_adapter({"text": sample_long_text}, base_context)

        assert result == {"error": "flashcard_generate_error", "flashcards": [], "count": 0}

    @pytest.mark.asyncio
    async def test_quiz_generate_sanitizes_backend_errors(self, base_context, sample_long_text):
        """Test quiz generation hides backend exception details."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_quiz_generate_adapter

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.content.generation.perform_chat_api_call_async",
            new_callable=AsyncMock,
            side_effect=RuntimeError("quiz backend exploded at /private/content-cache"),
        ):
            result = await run_quiz_generate_adapter({"text": sample_long_text}, base_context)

        assert result == {"error": "quiz_generate_error", "questions": [], "count": 0}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("adapter_name", "config", "expected"),
        [
            (
                "run_outline_generate_adapter",
                {"text": "Sample content"},
                {"error": "outline_generate_error", "outline": {}, "outline_text": ""},
            ),
            (
                "run_glossary_extract_adapter",
                {"text": "Sample content"},
                {"error": "glossary_extract_error", "glossary": [], "count": 0},
            ),
            (
                "run_mindmap_generate_adapter",
                {"text": "Sample content"},
                {"error": "mindmap_generate_error", "mindmap": {}, "mermaid": ""},
            ),
            (
                "run_slides_generate_adapter",
                {"content": "Sample content"},
                {"slides": [], "error": "slides_generate_error"},
            ),
            (
                "run_report_generate_adapter",
                {"content": "Sample content"},
                {"report": "", "error": "report_generate_error"},
            ),
            (
                "run_newsletter_generate_adapter",
                {"content": "Sample content"},
                {"newsletter": "", "error": "newsletter_generate_error"},
            ),
            (
                "run_diagram_generate_adapter",
                {"content": "Sample content", "provider": "mock", "model": "mock-model"},
                {"diagram": "", "error": "diagram_generate_error"},
            ),
        ],
    )
    async def test_content_generation_adapters_sanitize_backend_errors(
        self, base_context, adapter_name, config, expected
    ):
        """Test content generation adapters hide backend exception details."""
        from tldw_Server_API.app.core.Workflows.adapters import content as content_adapters

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.content.generation.perform_chat_api_call_async",
            new_callable=AsyncMock,
            side_effect=RuntimeError("content backend exploded at /private/content-cache"),
        ):
            result = await getattr(content_adapters, adapter_name)(config, base_context)

        assert result == expected

    @pytest.mark.asyncio
    async def test_notes_studio_generate_sanitizes_fallback_warnings(self, base_context):
        """Test Notes Studio fallback hides backend exception details."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_notes_studio_generate_adapter

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.content.generation.perform_chat_api_call_async",
            new_callable=AsyncMock,
            side_effect=RuntimeError("notes backend exploded at /private/content-cache"),
        ):
            result = await run_notes_studio_generate_adapter(
                {
                    "excerpt_text": "Sample study notes.",
                    "source_note_id": "note-1",
                    "provider": "mock",
                    "model": "mock-model",
                },
                base_context,
            )

        assert result["source"] == "deterministic_fallback"
        assert result["warning"] == "notes_studio_generate_warning"
        assert isinstance(result["payload"], dict)

    @pytest.mark.asyncio
    async def test_summarize_adapter_sanitizes_backend_errors(self, monkeypatch, base_context, sample_long_text):
        """Test summarize adapter hides backend exception details."""
        monkeypatch.delenv("TEST_MODE", raising=False)
        from tldw_Server_API.app.core.Workflows.adapters.content import run_summarize_adapter

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.content.summarize.asyncio.to_thread",
            new_callable=AsyncMock,
            side_effect=RuntimeError("summarizer backend exploded at /private/content-cache"),
        ):
            result = await run_summarize_adapter({"text": sample_long_text}, base_context)

        assert result == {"error": "summarize_error"}

    @pytest.mark.asyncio
    async def test_citations_adapter_sanitizes_backend_errors(self, monkeypatch, base_context, sample_documents):
        """Test citations adapter hides backend exception details."""
        monkeypatch.delenv("TEST_MODE", raising=False)
        from tldw_Server_API.app.core.Workflows.adapters.content import run_citations_adapter

        with patch(
            "tldw_Server_API.app.core.RAG.rag_service.citations.CitationGenerator.generate_citations",
            new_callable=AsyncMock,
            side_effect=RuntimeError("citations backend exploded at /private/content-cache"),
        ):
            result = await run_citations_adapter({"documents": sample_documents}, base_context)

        assert result == {"error": "citations_error"}

    @pytest.mark.asyncio
    async def test_rerank_adapter_sanitizes_backend_errors(self, monkeypatch, base_context, sample_documents):
        """Test rerank adapter hides backend exception details."""
        monkeypatch.delenv("TEST_MODE", raising=False)
        from tldw_Server_API.app.core.Workflows.adapters.content import run_rerank_adapter

        with patch(
            "tldw_Server_API.app.core.RAG.rag_service.advanced_reranking.create_reranker",
            side_effect=RuntimeError("rerank backend exploded at /private/content-cache"),
        ):
            result = await run_rerank_adapter(
                {"query": "AI research", "documents": sample_documents},
                base_context,
            )

        assert result == {"error": "rerank_error"}

    @pytest.mark.asyncio
    async def test_slides_generate_json_parse_error(self, monkeypatch, base_context, sample_long_text):
        """Test slides generation handles JSON parse errors gracefully."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_slides_generate_adapter

        # Return invalid JSON
        mock_response = mock_chat_response("Here are some slides...")

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.content.generation.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            config = {"content": sample_long_text}
            result = await run_slides_generate_adapter(config, base_context)

            # Should return error with raw text
            assert result.get("error") == "json_parse_failed"
            assert "raw_text" in result


# =============================================================================
# Context Fallback Tests
# =============================================================================


class TestContentAdaptersContextFallback:
    """Tests for context fallback behavior in content adapters."""

    @pytest.mark.asyncio
    async def test_rerank_uses_query_from_context(self, monkeypatch, sample_documents):
        """Test rerank adapter can use query from prev context."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.content import run_rerank_adapter

        context = {
            "user_id": "1",
            "prev": {
                "query": "machine learning concepts",
                "documents": sample_documents,
            },
        }
        config = {}  # No explicit query or documents
        result = await run_rerank_adapter(config, context)

        assert "documents" in result
        assert result.get("query") == "machine learning concepts"

    @pytest.mark.asyncio
    async def test_newsletter_uses_items_from_context(self, monkeypatch, base_context):
        """Test newsletter adapter can use items from prev context."""
        from tldw_Server_API.app.core.Workflows.adapters.content import run_newsletter_generate_adapter

        mock_newsletter = "# Newsletter"
        mock_response = mock_chat_response(mock_newsletter)

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.content.generation.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            context = {
                **base_context,
                "prev": {
                    "items": [
                        {"title": "News 1", "summary": "Summary 1"},
                    ]
                },
            }
            config = {}
            result = await run_newsletter_generate_adapter(config, context)

            assert "newsletter" in result


# =============================================================================
# Adapter Registration Tests
# =============================================================================


class TestContentAdaptersRegistration:
    """Tests for content adapter registration in the registry."""

    def test_all_content_adapters_registered(self):
        """Verify all 15 content adapters are registered."""
        from tldw_Server_API.app.core.Workflows.adapters import registry

        expected_adapters = [
            "summarize",
            "citations",
            "bibliography_generate",
            "image_gen",
            "image_describe",
            "rerank",
            "flashcard_generate",
            "quiz_generate",
            "outline_generate",
            "glossary_extract",
            "mindmap_generate",
            "report_generate",
            "newsletter_generate",
            "slides_generate",
            "diagram_generate",
        ]

        registered = registry.list_adapters()
        for adapter_name in expected_adapters:
            assert adapter_name in registered, f"Adapter '{adapter_name}' not registered"

    def test_content_adapters_have_correct_category(self):
        """Verify all content adapters are in 'content' category."""
        from tldw_Server_API.app.core.Workflows.adapters import registry

        content_adapters = [
            "summarize",
            "citations",
            "bibliography_generate",
            "image_gen",
            "image_describe",
            "rerank",
            "flashcard_generate",
            "quiz_generate",
            "outline_generate",
            "glossary_extract",
            "mindmap_generate",
            "report_generate",
            "newsletter_generate",
            "slides_generate",
            "diagram_generate",
        ]

        for adapter_name in content_adapters:
            spec = registry.get_spec(adapter_name)
            assert spec is not None, f"Adapter '{adapter_name}' spec not found"
            assert spec.category == "content", f"Adapter '{adapter_name}' has wrong category: {spec.category}"

    def test_content_adapters_have_config_models(self):
        """Verify all content adapters have Pydantic config models."""
        from tldw_Server_API.app.core.Workflows.adapters import registry

        content_adapters = [
            "summarize",
            "citations",
            "bibliography_generate",
            "image_gen",
            "image_describe",
            "rerank",
            "flashcard_generate",
            "quiz_generate",
            "outline_generate",
            "glossary_extract",
            "mindmap_generate",
            "report_generate",
            "newsletter_generate",
            "slides_generate",
            "diagram_generate",
        ]

        for adapter_name in content_adapters:
            spec = registry.get_spec(adapter_name)
            assert spec.config_model is not None, f"Adapter '{adapter_name}' missing config_model"

    def test_content_adapters_are_async(self):
        """Verify all content adapter functions are async."""
        import asyncio

        from tldw_Server_API.app.core.Workflows.adapters import registry

        content_adapters = [
            "summarize",
            "citations",
            "bibliography_generate",
            "image_gen",
            "image_describe",
            "rerank",
            "flashcard_generate",
            "quiz_generate",
            "outline_generate",
            "glossary_extract",
            "mindmap_generate",
            "report_generate",
            "newsletter_generate",
            "slides_generate",
            "diagram_generate",
            "audio_briefing_compose",
        ]

        for adapter_name in content_adapters:
            spec = registry.get_spec(adapter_name)
            assert asyncio.iscoroutinefunction(spec.func), f"Adapter '{adapter_name}' is not async"


# =============================================================================
# Audio Briefing Compose Adapter Tests
# =============================================================================


class TestAudioBriefingComposeAdapter:
    """Tests for run_audio_briefing_compose_adapter."""

    @pytest.fixture
    def sample_items(self):
        return [
            {
                "title": "AI Breakthrough",
                "summary": "New AI model achieves record performance",
                "url": "https://example.com/ai",
            },
            {
                "title": "Climate Report",
                "summary": "Global temperatures continue to rise",
                "url": "https://example.com/climate",
            },
            {
                "title": "Tech Merger",
                "summary": "Two major tech companies announce merger",
                "url": "https://example.com/tech",
            },
        ]

    @pytest.fixture
    def mock_llm_response_multi_voice(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            "[HOST]: Good morning, here is your daily briefing.\n"
                            "[pause]\n"
                            "[REPORTER]: In technology news, a new AI model has achieved record performance "
                            "on standard benchmarks, marking a significant advancement.\n"
                            "[HOST]: Moving on to environmental news.\n"
                            "[REPORTER]: A new climate report shows that global temperatures continue to rise, "
                            "with experts warning about the long-term consequences.\n"
                            "[HOST]: And in business news.\n"
                            "[REPORTER]: Two major technology companies have announced a merger that could "
                            "reshape the industry landscape.\n"
                            "[HOST]: That wraps up today's briefing. Thank you for listening."
                        )
                    }
                }
            ]
        }

    @pytest.fixture
    def mock_llm_response_single_voice(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            "Good morning. Here is your daily briefing. "
                            "In technology news, a new AI model has achieved record performance. "
                            "A new climate report shows rising temperatures. "
                            "Two tech companies have announced a merger. "
                            "That wraps up today's briefing."
                        )
                    }
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_compose_multi_voice_script(self, sample_items, mock_llm_response_multi_voice, base_context):
        """Test multi-voice script composition with section parsing."""
        from tldw_Server_API.app.core.Workflows.adapters.content.audio_briefing import (
            run_audio_briefing_compose_adapter,
        )

        config = {"items": sample_items, "target_audio_minutes": 5, "multi_voice": True}

        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_llm_response_multi_voice,
        ) as mock_llm:
            result = await run_audio_briefing_compose_adapter(config, base_context)

        assert result.get("text")
        assert result.get("script") == result["text"]
        assert isinstance(result["sections"], list)
        assert len(result["sections"]) > 0
        assert result["word_count"] > 0
        assert result["estimated_minutes"] > 0
        assert "voice_assignments" in result

        # Verify sections have voice markers
        voices_used = {s["voice"] for s in result["sections"]}
        assert "HOST" in voices_used
        assert "REPORTER" in voices_used

        # Verify LLM was called with correct params
        mock_llm.assert_called_once()
        call_kwargs = mock_llm.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert "750" in call_kwargs["messages"][0]["content"]  # 5 * 150 = 750 words

    @pytest.mark.asyncio
    async def test_compose_single_voice_script(self, sample_items, mock_llm_response_single_voice, base_context):
        """Test single-voice script produces no voice markers."""
        from tldw_Server_API.app.core.Workflows.adapters.content.audio_briefing import (
            run_audio_briefing_compose_adapter,
        )

        config = {"items": sample_items, "multi_voice": False}

        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_llm_response_single_voice,
        ):
            result = await run_audio_briefing_compose_adapter(config, base_context)

        assert result.get("text")
        assert len(result["sections"]) == 1
        assert result["sections"][0]["voice"] == "HOST"

    @pytest.mark.asyncio
    async def test_compose_voice_map_override(self, sample_items, mock_llm_response_multi_voice, base_context):
        """Test custom voice map overrides defaults."""
        from tldw_Server_API.app.core.Workflows.adapters.content.audio_briefing import (
            run_audio_briefing_compose_adapter,
        )

        custom_map = {"HOST": "am_michael", "REPORTER": "bf_isabella"}
        config = {"items": sample_items, "voice_map": custom_map}

        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_llm_response_multi_voice,
        ):
            result = await run_audio_briefing_compose_adapter(config, base_context)

        assert result["voice_assignments"]["HOST"] == "am_michael"
        assert result["voice_assignments"]["REPORTER"] == "bf_isabella"

    @pytest.mark.asyncio
    async def test_compose_registers_script_artifact(
        self, sample_items, mock_llm_response_multi_voice, base_context, tmp_path, monkeypatch
    ):
        """Test composed briefing script is persisted as a workflow artifact."""
        from tldw_Server_API.app.core.Workflows.adapters.content.audio_briefing import (
            run_audio_briefing_compose_adapter,
        )

        monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
        artifacts: list[dict[str, Any]] = []
        base_context["add_artifact"] = lambda **kwargs: artifacts.append(kwargs)

        config = {
            "items": sample_items,
            "target_audio_minutes": 5,
            "multi_voice": True,
        }

        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_llm_response_multi_voice,
        ):
            result = await run_audio_briefing_compose_adapter(config, base_context)

        script_artifacts = [artifact for artifact in artifacts if artifact["type"] == "audio_script"]
        assert len(script_artifacts) == 1
        artifact = script_artifacts[0]
        assert artifact["mime_type"] == "text/markdown"
        assert artifact["metadata"]["script_artifact"] is True
        assert artifact["metadata"]["voice_assignments"]["HOST"] == "af_bella"
        assert artifact["metadata"]["sections_count"] == len(result["sections"])
        script_path = artifact["uri"].removeprefix("file://")
        assert "Good morning, here is your daily briefing" in Path(script_path).read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_compose_audio_cast_controls_markers_and_voice_assignments(self, sample_items, base_context):
        """Test structured speaker config drives prompt markers and voices."""
        from tldw_Server_API.app.core.Workflows.adapters.content.audio_briefing import (
            run_audio_briefing_compose_adapter,
        )

        mock_response = {
            "choices": [
                {"message": {"content": ("[SPEAKER1]: Welcome to the briefing.\n" "[ANALYST]: Here is the analysis.")}}
            ]
        }
        audio_cast = {
            "speaker_count": 2,
            "speakers": [
                {
                    "id": "speaker1",
                    "label": "Speaker 1",
                    "role": "anchor",
                    "voice": "af_bella",
                },
                {
                    "id": "analyst",
                    "label": "Analyst",
                    "role": "expert commentary",
                    "voice": "am_adam",
                },
            ],
        }
        config = {
            "items": sample_items,
            "multi_voice": True,
            "audio_cast": audio_cast,
        }

        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_llm:
            result = await run_audio_briefing_compose_adapter(config, base_context)

        system_prompt = mock_llm.call_args.kwargs["system_message"]
        assert "[SPEAKER1]" in system_prompt
        assert "[ANALYST]" in system_prompt
        assert "expert commentary" in system_prompt
        assert "REPORTER" not in system_prompt
        assert result["sections"][0]["voice"] == "SPEAKER1"
        assert result["voice_assignments"]["SPEAKER1"] == "af_bella"
        assert result["voice_assignments"]["ANALYST"] == "am_adam"

    @pytest.mark.asyncio
    async def test_compose_empty_items_error(self, base_context):
        """Test empty items returns error."""
        from tldw_Server_API.app.core.Workflows.adapters.content.audio_briefing import (
            run_audio_briefing_compose_adapter,
        )

        config = {"items": []}
        result = await run_audio_briefing_compose_adapter(config, base_context)

        assert result.get("error") == "missing_items"
        assert result["text"] == ""

    @pytest.mark.asyncio
    async def test_compose_items_from_prev(self, sample_items, mock_llm_response_multi_voice, base_context):
        """Test items resolved from prev step output."""
        from tldw_Server_API.app.core.Workflows.adapters.content.audio_briefing import (
            run_audio_briefing_compose_adapter,
        )

        config = {}
        base_context["prev"] = {"results": sample_items}

        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_llm_response_multi_voice,
        ):
            result = await run_audio_briefing_compose_adapter(config, base_context)

        assert result.get("text")
        assert len(result["sections"]) > 0

    @pytest.mark.asyncio
    async def test_compose_cancelled(self, base_context):
        """Test cancellation handling."""
        from tldw_Server_API.app.core.Workflows.adapters.content.audio_briefing import (
            run_audio_briefing_compose_adapter,
        )

        config = {"items": [{"title": "Test", "summary": "Test"}]}
        base_context["is_cancelled"] = lambda: True

        result = await run_audio_briefing_compose_adapter(config, base_context)
        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_compose_system_prompt_includes_word_count(self, sample_items, base_context):
        """Test system prompt includes calculated target word count."""
        from tldw_Server_API.app.core.Workflows.adapters.content.audio_briefing import (
            run_audio_briefing_compose_adapter,
        )

        mock_response = {"choices": [{"message": {"content": "[HOST]: Test script."}}]}
        config = {"items": sample_items, "target_audio_minutes": 8}

        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_llm:
            await run_audio_briefing_compose_adapter(config, base_context)

        call_kwargs = mock_llm.call_args[1]
        # 8 * 150 = 1200 words
        assert "1200" in call_kwargs["system_message"]

    @pytest.mark.asyncio
    async def test_compose_system_prompt_includes_language_rule(self, sample_items, base_context):
        """Test system prompt includes configured output language rule."""
        from tldw_Server_API.app.core.Workflows.adapters.content.audio_briefing import (
            run_audio_briefing_compose_adapter,
        )

        mock_response = {"choices": [{"message": {"content": "[HOST]: Test script."}}]}
        config = {"items": sample_items, "output_language": "es"}

        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_llm:
            await run_audio_briefing_compose_adapter(config, base_context)

        call_kwargs = mock_llm.call_args[1]
        assert "Reply only in es." in call_kwargs["system_message"]

    @pytest.mark.asyncio
    async def test_compose_persona_pre_summarization_preserves_contract(
        self, sample_items, mock_llm_response_multi_voice, base_context
    ):
        """Test persona pre-summarization runs per item and feeds compose prompt."""
        from tldw_Server_API.app.core.Workflows.adapters.content.audio_briefing import (
            run_audio_briefing_compose_adapter,
        )

        persona_rewrites = [
            {"choices": [{"message": {"content": "Persona summary one"}}]},
            {"choices": [{"message": {"content": "Persona summary two"}}]},
            {"choices": [{"message": {"content": "Persona summary three"}}]},
        ]
        config = {
            "items": sample_items,
            "persona_summarize": True,
            "persona_id": "analyst",
            "provider": "openai",
            "model": "gpt-4o-mini",
        }

        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            new_callable=AsyncMock,
            side_effect=persona_rewrites + [mock_llm_response_multi_voice],
        ) as mock_llm:
            result = await run_audio_briefing_compose_adapter(config, base_context)

        assert result.get("error") is None
        assert mock_llm.await_count == 4
        compose_call = mock_llm.call_args_list[-1][1]
        assert "Persona summary one" in compose_call["messages"][0]["content"]
        first_persona_call = mock_llm.call_args_list[0][1]
        assert "analyst" in first_persona_call["system_message"]
        assert "Title:" in first_persona_call["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_compose_persona_pre_summarization_sanitizes_warnings(
        self, sample_items, mock_llm_response_multi_voice, base_context
    ):
        """Test persona pre-summarization warnings hide backend exception details."""
        from tldw_Server_API.app.core.Workflows.adapters.content.audio_briefing import (
            run_audio_briefing_compose_adapter,
        )

        config = {
            "items": sample_items[:1],
            "persona_summarize": True,
            "provider": "openai",
            "model": "gpt-4o-mini",
        }

        with (
            patch(
                "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
                new_callable=AsyncMock,
                side_effect=[
                    RuntimeError("persona backend exploded at /private/audio-cache"),
                    mock_llm_response_multi_voice,
                ],
            ),
            patch("tldw_Server_API.app.core.Workflows.adapters.content.audio_briefing.logger.warning") as mock_warning,
        ):
            result = await run_audio_briefing_compose_adapter(config, base_context)

        assert result.get("error") is None
        mock_warning.assert_called_once_with("Persona pre-summarization failed", exc_info=True)

    @pytest.mark.asyncio
    async def test_compose_strips_reasoning_blocks(self, sample_items, base_context):
        """Test reasoning blocks are stripped from final spoken script."""
        from tldw_Server_API.app.core.Workflows.adapters.content.audio_briefing import (
            run_audio_briefing_compose_adapter,
        )

        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "<think>internal chain of thought</think>\n"
                            "[HOST]: Good morning.\n"
                            "<reasoning>hidden notes</reasoning>\n"
                            "[REPORTER]: Story details."
                        )
                    }
                }
            ]
        }
        config = {"items": sample_items, "multi_voice": True}

        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await run_audio_briefing_compose_adapter(config, base_context)

        assert result.get("error") is None
        assert "<think>" not in result["text"].lower()
        assert "<reasoning>" not in result["text"].lower()
        assert len(result["sections"]) >= 2

    @pytest.mark.asyncio
    async def test_compose_llm_error_handled(self, sample_items, base_context):
        """Test LLM errors are caught without returning raw backend details."""
        from tldw_Server_API.app.core.Workflows.adapters.content.audio_briefing import (
            run_audio_briefing_compose_adapter,
        )

        config = {"items": sample_items}

        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            new_callable=AsyncMock,
            side_effect=RuntimeError("LLM provider down at /private/content-cache"),
        ):
            result = await run_audio_briefing_compose_adapter(config, base_context)

        assert result == {
            "text": "",
            "script": "",
            "sections": [],
            "error": "audio_briefing_compose_error",
        }

    @pytest.mark.asyncio
    async def test_section_parsing_with_pause(self, base_context):
        """Test section parsing handles [pause] markers."""
        from tldw_Server_API.app.core.Workflows.adapters.content.audio_briefing import _parse_sections

        script = (
            "[HOST]: Welcome to the briefing.\n"
            "[pause]\n"
            "[REPORTER]: In breaking news today.\n"
            "[HOST]: That's all for now."
        )
        sections = _parse_sections(script)

        assert len(sections) >= 3
        # [pause] itself should not create a section, it appears inline
        voices = [s["voice"] for s in sections]
        assert "HOST" in voices
        assert "REPORTER" in voices

    @pytest.mark.asyncio
    async def test_default_voice_assignments(self):
        """Test default voice assignments are correct."""
        from tldw_Server_API.app.core.Workflows.adapters.content.audio_briefing import (
            _DEFAULT_VOICE_MAP,
            _resolve_voice_assignments,
        )

        sections = [
            {"voice": "HOST", "text": "Hello"},
            {"voice": "REPORTER", "text": "News"},
        ]

        assignments = _resolve_voice_assignments(sections, None)

        assert assignments["HOST"] == _DEFAULT_VOICE_MAP["HOST"]
        assert assignments["REPORTER"] == _DEFAULT_VOICE_MAP["REPORTER"]

    @pytest.mark.asyncio
    async def test_unknown_voice_gets_fallback(self):
        """Test unknown voice markers get HOST fallback."""
        from tldw_Server_API.app.core.Workflows.adapters.content.audio_briefing import _resolve_voice_assignments

        sections = [
            {"voice": "HOST", "text": "Hello"},
            {"voice": "UNKNOWN_SPEAKER", "text": "Something"},
        ]

        assignments = _resolve_voice_assignments(sections, None)
        assert "UNKNOWN_SPEAKER" in assignments  # Should get a fallback
