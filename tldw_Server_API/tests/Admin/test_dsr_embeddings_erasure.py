"""Tests for GDPR DSR embeddings erasure (Gap 3.1).

Covers:
- _count_embeddings returns correct count (mock ChromaDBManager)
- embeddings is now in _SUPPORTED_CATEGORY_KEYS
- execute_dsr_erasure is callable and handles categories correctly
- update_request_status exists on the repo class
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def _mock_collection():
    """Create a mock ChromaDB collection with a configurable count."""

    def _make(name: str = "test_col", count: int = 42):
        col = MagicMock()
        col.name = name
        col.count.return_value = count
        return col

    return _make


@pytest.fixture()
def _mock_chroma_manager(_mock_collection):
    """Return a mock ChromaDBManager whose list_collections returns one collection."""
    manager = MagicMock()
    manager.list_collections.return_value = [_mock_collection("default", 10)]
    return manager


# ---------------------------------------------------------------------------
# Sub-task 2.1 tests: _count_embeddings & category membership
# ---------------------------------------------------------------------------


class TestCountEmbeddings:
    """Verify that _count_embeddings delegates to ChromaDBManager."""

    def test_embeddings_in_supported_categories(self):
        from tldw_Server_API.app.services.admin_data_subject_requests_service import (
            _SUPPORTED_CATEGORY_KEYS,
        )

        assert "embeddings" in _SUPPORTED_CATEGORY_KEYS

    def test_embeddings_not_in_unsupported_categories(self):
        from tldw_Server_API.app.services.admin_data_subject_requests_service import (
            _UNSUPPORTED_CATEGORY_KEYS,
        )

        assert "embeddings" not in _UNSUPPORTED_CATEGORY_KEYS

    def test_embeddings_category_def_exists(self):
        from tldw_Server_API.app.services.admin_data_subject_requests_service import (
            _CATEGORY_DEFS,
        )

        keys = [entry["key"] for entry in _CATEGORY_DEFS]
        assert "embeddings" in keys

    @pytest.mark.asyncio
    async def test_count_embeddings_with_mock(self, _mock_chroma_manager):
        """_count_embeddings should sum counts across all collections."""
        with patch(
            "tldw_Server_API.app.services.admin_data_subject_requests_service._get_chroma_manager_for_user",
            return_value=_mock_chroma_manager,
        ):
            from tldw_Server_API.app.services.admin_data_subject_requests_service import (
                _count_embeddings,
            )

            result = await _count_embeddings(999)
            assert result == 10

    @pytest.mark.asyncio
    async def test_count_embeddings_multiple_collections(self, _mock_collection):
        """_count_embeddings should sum across multiple collections."""
        manager = MagicMock()
        manager.list_collections.return_value = [
            _mock_collection("col_a", 5),
            _mock_collection("col_b", 15),
            _mock_collection("col_c", 30),
        ]
        with patch(
            "tldw_Server_API.app.services.admin_data_subject_requests_service._get_chroma_manager_for_user",
            return_value=manager,
        ):
            from tldw_Server_API.app.services.admin_data_subject_requests_service import (
                _count_embeddings,
            )

            result = await _count_embeddings(1)
            assert result == 50

    @pytest.mark.asyncio
    async def test_count_embeddings_returns_zero_on_failure(self):
        """_count_embeddings should return 0 when ChromaDBManager fails."""
        with patch(
            "tldw_Server_API.app.services.admin_data_subject_requests_service._get_chroma_manager_for_user",
            side_effect=RuntimeError("no chroma"),
        ):
            from tldw_Server_API.app.services.admin_data_subject_requests_service import (
                _count_embeddings,
            )

            result = await _count_embeddings(1)
            assert result == 0


# ---------------------------------------------------------------------------
# Sub-task 2.2 tests: update_request_status on the repo
# ---------------------------------------------------------------------------


class TestUpdateRequestStatus:
    """Verify update_request_status exists and validates status values."""

    def test_method_exists(self):
        from tldw_Server_API.app.core.AuthNZ.repos.data_subject_requests_repo import (
            AuthnzDataSubjectRequestsRepo,
        )

        assert hasattr(AuthnzDataSubjectRequestsRepo, "update_request_status")
        assert callable(getattr(AuthnzDataSubjectRequestsRepo, "update_request_status", None))

    def test_get_request_by_id_method_exists(self):
        from tldw_Server_API.app.core.AuthNZ.repos.data_subject_requests_repo import (
            AuthnzDataSubjectRequestsRepo,
        )

        assert hasattr(AuthnzDataSubjectRequestsRepo, "get_request_by_id")

    def test_valid_statuses_defined(self):
        from tldw_Server_API.app.core.AuthNZ.repos.data_subject_requests_repo import (
            AuthnzDataSubjectRequestsRepo,
        )

        expected = {"pending", "recorded", "executing", "completed", "failed"}
        assert AuthnzDataSubjectRequestsRepo._VALID_STATUSES == expected


# ---------------------------------------------------------------------------
# Sub-task 2.3 tests: execute_dsr_erasure
# ---------------------------------------------------------------------------


class TestExecuteDsrErasure:
    """Verify execute_dsr_erasure orchestrates category handlers."""

    @pytest.fixture()
    def mock_dsr_repo(self):
        repo = MagicMock()
        repo.update_request_status = AsyncMock(return_value={"id": 1, "status": "completed"})
        return repo

    def test_execute_dsr_erasure_exists(self):
        from tldw_Server_API.app.services.admin_data_subject_requests_service import (
            execute_dsr_erasure,
        )

        assert callable(execute_dsr_erasure)

    @pytest.mark.asyncio
    async def test_execute_dsr_erasure_calls_handlers(self, mock_dsr_repo):
        """execute_dsr_erasure should call the appropriate handler for each category."""
        from tldw_Server_API.app.services import admin_data_subject_requests_service as svc

        mock_media = AsyncMock(return_value=5)
        mock_emb = AsyncMock(return_value=2)
        original = dict(svc._ERASURE_HANDLERS)
        svc._ERASURE_HANDLERS["media_records"] = mock_media
        svc._ERASURE_HANDLERS["embeddings"] = mock_emb
        try:
            result = await svc.execute_dsr_erasure(
                request_id=1,
                user_id=42,
                selected_categories=["media_records", "embeddings"],
                dsr_repo=mock_dsr_repo,
            )

            mock_media.assert_awaited_once_with(42)
            mock_emb.assert_awaited_once_with(42)
            assert result["status"] == "completed"
            assert result["categories"]["media_records"]["deleted_count"] == 5
            assert result["categories"]["embeddings"]["deleted_count"] == 2
            assert not result["errors"]
        finally:
            svc._ERASURE_HANDLERS.update(original)

    @pytest.mark.asyncio
    async def test_execute_dsr_erasure_handles_unknown_category(self, mock_dsr_repo):
        """Unknown categories should be reported as errors."""
        from tldw_Server_API.app.services.admin_data_subject_requests_service import (
            execute_dsr_erasure,
        )

        result = await execute_dsr_erasure(
            request_id=1,
            user_id=42,
            selected_categories=["nonexistent_category"],
            dsr_repo=mock_dsr_repo,
        )

        assert result["status"] == "failed"
        assert "nonexistent_category" in result["errors"]

    @pytest.mark.asyncio
    async def test_execute_dsr_erasure_handles_handler_failure(self, mock_dsr_repo):
        """If a handler raises, the category should be marked as error."""
        from tldw_Server_API.app.services import admin_data_subject_requests_service as svc

        mock_notes = AsyncMock(side_effect=RuntimeError("db locked at /private/dsr.db"))
        original = dict(svc._ERASURE_HANDLERS)
        svc._ERASURE_HANDLERS["notes"] = mock_notes
        try:
            result = await svc.execute_dsr_erasure(
                request_id=1,
                user_id=42,
                selected_categories=["notes"],
                dsr_repo=mock_dsr_repo,
            )

            assert result["status"] == "failed"
            assert "notes" in result["errors"]
            assert result["errors"]["notes"] == "Erasure handler failed"
            assert result["categories"]["notes"]["status"] == "error"
            assert result["categories"]["notes"]["error"] == "Erasure handler failed"
            assert "db locked" not in repr(result)
            assert "/private/dsr.db" not in repr(result)

            final_status_call = mock_dsr_repo.update_request_status.call_args_list[-1]
            assert final_status_call.kwargs["notes"] == (
                "Erasure errors: {'notes': 'Erasure handler failed'}"
            )
        finally:
            svc._ERASURE_HANDLERS.update(original)

    @pytest.mark.asyncio
    async def test_execute_dsr_erasure_status_transitions(self, mock_dsr_repo):
        """The repo should be called with executing, then completed."""
        from tldw_Server_API.app.services import admin_data_subject_requests_service as svc

        mock_media = AsyncMock(return_value=0)
        original = dict(svc._ERASURE_HANDLERS)
        svc._ERASURE_HANDLERS["media_records"] = mock_media
        try:
            from tldw_Server_API.app.services.admin_data_subject_requests_service import (
                execute_dsr_erasure,
            )

            await execute_dsr_erasure(
                request_id=7,
                user_id=1,
                selected_categories=["media_records"],
                dsr_repo=mock_dsr_repo,
            )

            calls = mock_dsr_repo.update_request_status.call_args_list
            assert len(calls) == 2
            assert calls[0].args == (7, "executing")
            assert calls[1].args[0] == 7
            assert calls[1].args[1] == "completed"
        finally:
            svc._ERASURE_HANDLERS.update(original)


# ---------------------------------------------------------------------------
# Sub-task 2.4 tests: coverage_metadata no longer lists embeddings as unsupported
# ---------------------------------------------------------------------------


class TestCoverageMetadata:
    """Verify coverage_metadata reflects embeddings support."""

    def test_no_unsupported_categories(self):
        from tldw_Server_API.app.services.admin_data_subject_requests_service import (
            _coverage_metadata,
        )

        meta = _coverage_metadata(selected_categories=["embeddings"])
        assert meta["unsupported_categories"] == []
        assert meta["unsupported_details"] == {}

    def test_embeddings_in_supported_list(self):
        from tldw_Server_API.app.services.admin_data_subject_requests_service import (
            _coverage_metadata,
        )

        meta = _coverage_metadata(selected_categories=["embeddings"])
        assert "embeddings" in meta["supported_categories"]
