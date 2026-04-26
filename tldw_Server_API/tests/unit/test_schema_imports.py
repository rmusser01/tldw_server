"""Smoke tests that verify schema modules import cleanly.

Importing Pydantic schema modules covers their class definitions,
field validators, and model_config blocks — providing broad coverage
for minimal test code.
"""

from __future__ import annotations

import pytest


class TestSchemaImports:
    """Each test imports a schema module, covering its class definitions."""

    def test_media_request_models(self):
        from tldw_Server_API.app.api.v1.schemas import media_request_models
        assert hasattr(media_request_models, "MediaUpdateRequest")

    def test_media_response_models(self):
        from tldw_Server_API.app.api.v1.schemas import media_response_models
        assert hasattr(media_response_models, "MediaDetailResponse")

    def test_chat_request_schemas(self):
        from tldw_Server_API.app.api.v1.schemas import chat_request_schemas
        assert hasattr(chat_request_schemas, "ChatCompletionRequest")

    def test_auth_schemas(self):
        from tldw_Server_API.app.api.v1.schemas import auth_schemas
        assert hasattr(auth_schemas, "TokenResponse")

    def test_rag_schemas(self):
        from tldw_Server_API.app.api.v1.schemas import rag_schemas_simple
        assert hasattr(rag_schemas_simple, "AdvancedSearchRequest")

    def test_evaluation_schemas(self):
        from tldw_Server_API.app.api.v1.schemas import evaluation_schemas_unified
        assert hasattr(evaluation_schemas_unified, "CreateEvaluationRequest")

    def test_kanban_schemas(self):
        from tldw_Server_API.app.api.v1.schemas import kanban_schemas
        assert hasattr(kanban_schemas, "BoardResponse")

    def test_chat_dictionary_schemas(self):
        from tldw_Server_API.app.api.v1.schemas import chat_dictionary_schemas
        assert hasattr(chat_dictionary_schemas, "ChatDictionaryCreate")

    def test_prompt_studio_base(self):
        from tldw_Server_API.app.api.v1.schemas import prompt_studio_base
        assert hasattr(prompt_studio_base, "ErrorResponse")

    def test_personalization_schemas(self):
        from tldw_Server_API.app.api.v1.schemas import personalization
        assert hasattr(personalization, "MemoryCreate")

    def test_notes_schemas(self):
        from tldw_Server_API.app.api.v1.schemas import notes_schemas
        assert hasattr(notes_schemas, "NoteCreate")

    def test_embedding_schemas(self):
        from tldw_Server_API.app.api.v1.schemas import embeddings_models
        assert hasattr(embeddings_models, "CreateEmbeddingRequest")

    def test_watchlist_schemas(self):
        from tldw_Server_API.app.api.v1.schemas import watchlists_schemas
        assert hasattr(watchlists_schemas, "GroupCreateRequest")

    def test_workflow_schemas(self):
        from tldw_Server_API.app.api.v1.schemas import workflows
        assert hasattr(workflows, "WorkflowDefinitionCreate")
