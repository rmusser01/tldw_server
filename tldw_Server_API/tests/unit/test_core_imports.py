"""Smoke tests for core module imports.

These tests verify that critical core modules import cleanly,
exercising their module-level code, class definitions, and
initialization logic. High coverage impact for minimal code.
"""

from __future__ import annotations

import importlib


class TestCoreModuleImports:
    """Import tests for heavily-used core modules."""

    def test_config_module(self):
        from tldw_Server_API.app.core import config

        assert hasattr(config, "settings")
        assert hasattr(config, "loaded_config_data")
        assert hasattr(config, "validate_config")

    def test_config_get_value(self):
        from tldw_Server_API.app.core.config import get_config_value

        # Should return default for a nonexistent key
        result = get_config_value("test_section", "nonexistent_key", "default_val")
        assert result == "default_val"

    def test_auth_settings(self):
        from tldw_Server_API.app.core.AuthNZ import settings as auth_settings

        assert hasattr(auth_settings, "get_settings")

    def test_auth_permissions(self):
        from tldw_Server_API.app.core.AuthNZ import permissions

        assert hasattr(permissions, "MEDIA_DELETE")

    def test_db_media_errors(self):
        from tldw_Server_API.app.core.DB_Management.media_db import errors

        assert issubclass(errors.InputError, Exception)
        assert issubclass(errors.ConflictError, Exception)
        assert issubclass(errors.DatabaseError, Exception)
        assert issubclass(errors.SchemaError, Exception)

    def test_feature_flags(self):
        from tldw_Server_API.app.core.config import route_enabled

        # Should not crash for any route name
        result = route_enabled("nonexistent_route")
        assert isinstance(result, bool)

    def test_http_client(self):
        from tldw_Server_API.app.core import http_client

        assert hasattr(http_client, "HttpxAdapter")

    def test_chunking_exceptions(self):
        from tldw_Server_API.app.core.Chunking import exceptions

        assert hasattr(exceptions, "ChunkingError")

    def test_chat_exceptions(self):
        from tldw_Server_API.app.core.Chat import chat_exceptions

        assert hasattr(chat_exceptions, "ChatModuleException")

    def test_rag_exceptions(self):
        exceptions = importlib.import_module("tldw_Server_API.app.core.RAG.exceptions")
        assert hasattr(exceptions, "RAGError")

    def test_authnz_exceptions(self):
        from tldw_Server_API.app.core.AuthNZ import exceptions

        assert hasattr(exceptions, "AuthenticationError")

    def test_kanban_db_exceptions(self):
        from tldw_Server_API.app.core.DB_Management.Kanban_DB import (
            ConflictError,
            InputError,
            KanbanDBError,
            NotFoundError,
        )

        assert issubclass(ConflictError, KanbanDBError)
        assert issubclass(NotFoundError, KanbanDBError)
        assert issubclass(InputError, ValueError)
