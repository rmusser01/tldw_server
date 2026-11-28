import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager


@pytest.mark.unit
def test_contextualization_respects_provider_and_model(chromadb_manager, mock_chroma_client):
    """
    Ensure situate_context and outline generation call analyze() with the configured
    provider and model_override for contextual chunking.
    """
    # Arrange embedding config to enable contextualization with Anthropic
    chromadb_manager.embedding_config.update({
        "enable_contextual_chunking": True,
        "contextual_llm_provider": "anthropic",
        "contextual_llm_model": "claude-3-7-sonnet-20250219",
        "contextual_llm_temperature": 0.33,
        "context_strategy": "outline_window",
        "context_window_size": 100,
        "context_token_budget": 6000,
    })

    content = "Lorem ipsum dolor sit amet, " * 200  # long doc
    chunk_meta = {"start_char": 500, "end_char": 650}
    chunk = {"text": "This is a chunk.", "metadata": chunk_meta}

    # Mock chunking and embeddings
    with patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.chunk_for_embedding') as mock_chunk:
        mock_chunk.return_value = [chunk]
        with patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.create_embeddings_batch') as mock_emb:
            mock_emb.return_value = [[0.1, 0.2, 0.3]]
            # Capture analyze calls
            calls = []

            def _fake_analyze(*args, **kwargs):

                calls.append((args, kwargs))
                return "HEADER"

            with patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.analyze', side_effect=_fake_analyze):
                # Ensure collection exists
                mock_collection = MagicMock()
                mock_collection.count.return_value = 0
                mock_chroma_client.get_or_create_collection.return_value = mock_collection

                chromadb_manager.process_and_store_content(
                    content=content,
                    media_id="mid1",
                    collection_name="ctx_test",
                    file_name="file.txt",
                )

    # Assert at least one call (outline and situate may both be invoked)
    assert len(calls) >= 1
    for args, kwargs in calls:
        # Provider is the first positional argument to analyze()
        assert args[0] == "anthropic"
        # Model override is passed as a kwarg
        assert kwargs.get("model_override") == "claude-3-7-sonnet-20250219"
        # Temperature is the 6th positional argument
        assert pytest.approx(args[5], rel=1e-6) == 0.33


@pytest.mark.unit
def test_a_refusal_gives_an_empty_outline_and_an_empty_context_header(chromadb_manager):
    """Optional enrichment: a refusal is a labelled skip (empty outline, empty header), never an error, so the caller embeds on."""
    from tldw_Server_API.app.core.exceptions import LLMPolicyRefusal

    def refused(*_args, **_kwargs):
        raise LLMPolicyRefusal("Input violates moderation policy")

    with patch("tldw_Server_API.app.core.Embeddings.ChromaDB_Library.analyze", side_effect=refused):
        assert chromadb_manager._build_document_outline("openai", None, "a document", None) == ""
        assert chromadb_manager.situate_context("openai", None, "a document", "a chunk", None) == ""
