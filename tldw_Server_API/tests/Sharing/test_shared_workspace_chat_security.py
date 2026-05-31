from __future__ import annotations

import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import sharing
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType


class _RepoStub:
    async def get_share(self, share_id: int) -> dict[str, object]:
        assert share_id == 42
        return {
            "id": share_id,
            "owner_user_id": 7,
            "workspace_id": "ws-shared",
            "is_revoked": False,
        }


class _AuditStub:
    async def log(self, *args, **kwargs) -> None:
        raise AssertionError("audit log should not run after pipeline failure")


class _AuditSuccessStub:
    async def log(self, *args, **kwargs) -> None:
        return None


@pytest.mark.asyncio
async def test_chat_with_shared_workspace_hides_internal_pipeline_errors(monkeypatch):
    async def _allow_share_access(*args, **kwargs) -> None:
        return None

    async def _get_owner_chacha_db(_owner_user_id: int) -> SimpleNamespace:
        return SimpleNamespace(db_path="/tmp/chacha.db")

    @contextmanager
    def _managed_media_db_for_owner(_owner_user_id: int):
        yield SimpleNamespace(db_path="/tmp/media.db")

    def _get_media_db_path_for_rag(owner_media: SimpleNamespace) -> str:
        return owner_media.db_path

    async def _failing_pipeline(**kwargs):
        raise RuntimeError("trace=/Users/private/rag-stack.txt")

    fake_logger = MagicMock()
    monkeypatch.setattr(sharing, "_get_repo", lambda: _RepoStub())
    monkeypatch.setattr(sharing, "_get_audit_service", lambda: _AuditStub())
    monkeypatch.setattr(sharing, "_validate_user_has_share_access", _allow_share_access)
    monkeypatch.setattr(sharing, "logger", fake_logger)
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_owner",
        _get_owner_chacha_db,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.DB_Deps.managed_media_db_for_owner",
        _managed_media_db_for_owner,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.DB_Deps.get_media_db_path_for_rag",
        _get_media_db_path_for_rag,
    )
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.RAG.rag_service.unified_pipeline",
        SimpleNamespace(unified_rag_pipeline=_failing_pipeline),
    )

    with pytest.raises(HTTPException) as excinfo:
        await sharing.chat_with_shared_workspace(
            42,
            sharing.SharedChatRequest(query="hello"),
            SimpleNamespace(headers={}, client=None),
            User(
                id=11,
                username="reviewer",
                email="reviewer@example.com",
                password_hash="hash",
            ),
        )

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Chat request failed"
    assert excinfo.value.__cause__ is None
    fake_logger.error.assert_called_once_with("Shared workspace chat failed")


@pytest.mark.asyncio
async def test_chat_with_shared_workspace_redacts_pipeline_error_fields(monkeypatch):
    async def _allow_share_access(*args, **kwargs) -> None:
        return None

    async def _get_owner_chacha_db(_owner_user_id: int) -> SimpleNamespace:
        return SimpleNamespace(db_path="/tmp/chacha.db")

    @contextmanager
    def _managed_media_db_for_owner(_owner_user_id: int):
        yield SimpleNamespace(db_path="/tmp/media.db")

    def _get_media_db_path_for_rag(owner_media: SimpleNamespace) -> str:
        return owner_media.db_path

    async def _leaky_pipeline(**kwargs):
        return {
            "query": kwargs["query"],
            "documents": [],
            "generated_answer": "",
            "error": "trace=/Users/private/rag-stack.txt",
            "errors": [
                "Pipeline error: trace=/Users/private/rag-stack.txt",
                "Retrieval metrics failed: sqlite:/secret/path.db",
            ],
            "metadata": {},
            "timings": {},
        }

    monkeypatch.setattr(sharing, "_get_repo", lambda: _RepoStub())
    monkeypatch.setattr(sharing, "_get_audit_service", lambda: _AuditSuccessStub())
    monkeypatch.setattr(sharing, "_validate_user_has_share_access", _allow_share_access)
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_owner",
        _get_owner_chacha_db,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.DB_Deps.managed_media_db_for_owner",
        _managed_media_db_for_owner,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.DB_Deps.get_media_db_path_for_rag",
        _get_media_db_path_for_rag,
    )
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.RAG.rag_service.unified_pipeline",
        SimpleNamespace(unified_rag_pipeline=_leaky_pipeline),
    )

    result = await sharing.chat_with_shared_workspace(
        42,
        sharing.SharedChatRequest(query="hello"),
        SimpleNamespace(headers={}, client=None),
        User(
            id=11,
            username="reviewer",
            email="reviewer@example.com",
            password_hash="hash",
        ),
    )

    assert result["query"] == "hello"
    assert result["error"] == "Chat request failed"
    assert result["errors"] == ["One or more internal pipeline errors were suppressed."]


@pytest.mark.asyncio
async def test_chat_with_shared_workspace_passes_postgres_media_adapter_when_sqlite_path_is_suppressed(
    monkeypatch,
):
    captured: dict[str, object] = {}

    async def _allow_share_access(*args, **kwargs) -> None:
        return None

    owner_chacha = SimpleNamespace(db_path="/tmp/chacha.db")
    owner_media = SimpleNamespace(
        db_path=":memory:",
        backend_type=BackendType.POSTGRESQL,
    )

    async def _get_owner_chacha_db(_owner_user_id: int) -> SimpleNamespace:
        return owner_chacha

    @contextmanager
    def _managed_media_db_for_owner(_owner_user_id: int):
        yield owner_media

    def _get_media_db_path_for_rag(_owner_media: SimpleNamespace) -> None:
        return None

    async def _capturing_pipeline(**kwargs):
        captured.update(kwargs)
        return {
            "query": kwargs["query"],
            "documents": [],
            "generated_answer": "",
            "errors": [],
            "metadata": {},
            "timings": {},
        }

    monkeypatch.setattr(sharing, "_get_repo", lambda: _RepoStub())
    monkeypatch.setattr(sharing, "_get_audit_service", lambda: _AuditSuccessStub())
    monkeypatch.setattr(sharing, "_validate_user_has_share_access", _allow_share_access)
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_owner",
        _get_owner_chacha_db,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.DB_Deps.managed_media_db_for_owner",
        _managed_media_db_for_owner,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.DB_Deps.get_media_db_path_for_rag",
        _get_media_db_path_for_rag,
    )
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.RAG.rag_service.unified_pipeline",
        SimpleNamespace(unified_rag_pipeline=_capturing_pipeline),
    )

    result = await sharing.chat_with_shared_workspace(
        42,
        sharing.SharedChatRequest(query="hello"),
        SimpleNamespace(headers={}, client=None),
        User(
            id=11,
            username="reviewer",
            email="reviewer@example.com",
            password_hash="hash",
        ),
    )

    assert result["query"] == "hello"
    assert captured["media_db_path"] is None
    assert captured["media_db"] is owner_media
    assert captured["chacha_db"] is owner_chacha


@pytest.mark.asyncio
async def test_chat_with_shared_workspace_releases_owner_session_on_pipeline_failure(monkeypatch):
    events: list[str] = []

    async def _allow_share_access(*args, **kwargs) -> None:
        return None

    async def _get_owner_chacha_db(_owner_user_id: int) -> SimpleNamespace:
        return SimpleNamespace(db_path="/tmp/chacha.db")

    @contextmanager
    def _managed_media_db_for_owner(_owner_user_id: int):
        events.append("enter")
        try:
            yield SimpleNamespace(db_path="/tmp/media.db")
        finally:
            events.append("exit")

    def _get_media_db_path_for_rag(owner_media: SimpleNamespace) -> str:
        return owner_media.db_path

    async def _failing_pipeline(**kwargs):
        raise RuntimeError("trace=/Users/private/rag-stack.txt")

    monkeypatch.setattr(sharing, "_get_repo", lambda: _RepoStub())
    monkeypatch.setattr(sharing, "_get_audit_service", lambda: _AuditStub())
    monkeypatch.setattr(sharing, "_validate_user_has_share_access", _allow_share_access)
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_owner",
        _get_owner_chacha_db,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.DB_Deps.managed_media_db_for_owner",
        _managed_media_db_for_owner,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.DB_Deps.get_media_db_path_for_rag",
        _get_media_db_path_for_rag,
    )
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.RAG.rag_service.unified_pipeline",
        SimpleNamespace(unified_rag_pipeline=_failing_pipeline),
    )

    with pytest.raises(HTTPException):
        await sharing.chat_with_shared_workspace(
            42,
            sharing.SharedChatRequest(query="hello"),
            SimpleNamespace(headers={}, client=None),
            User(
                id=11,
                username="reviewer",
                email="reviewer@example.com",
                password_hash="hash",
            ),
        )

    assert events == ["enter", "exit"]
