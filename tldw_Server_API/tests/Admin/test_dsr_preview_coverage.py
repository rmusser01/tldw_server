"""Authoritative, selected-category DSR preview coverage regressions."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tldw_Server_API.app.services import admin_data_subject_requests_service as service


@pytest.fixture
def chroma_base(monkeypatch, tmp_path):
    """Use the same configured base as the DSR Chroma manager factory."""
    from tldw_Server_API.app.core.config import settings

    monkeypatch.setitem(settings, "USER_DB_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("TLDW_DB_ALLOWED_BASE_DIRS", str(tmp_path))
    return tmp_path


@pytest.mark.asyncio
async def test_unavailable_manager_with_existing_storage_rejects_preview(monkeypatch, chroma_base):
    (chroma_base / "7" / "chroma_storage").mkdir(parents=True)
    monkeypatch.setattr(service, "_get_chroma_manager_for_user", MagicMock(side_effect=ImportError("no chroma")))

    with pytest.raises(service.DataSubjectRequestCoverageUnavailableError):
        await service._count_embeddings(7)


@pytest.mark.asyncio
async def test_unavailable_optional_manager_with_absent_storage_returns_zero(monkeypatch, chroma_base):
    monkeypatch.setattr(service, "_get_chroma_manager_for_user", MagicMock(side_effect=ImportError("no chroma")))

    assert await service._count_embeddings(7) == 0
    assert not (chroma_base / "7").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_at", ["resolve", "stat"])
async def test_unavailable_manager_with_unknown_storage_rejects_preview(monkeypatch, chroma_base, failure_at):
    monkeypatch.setattr(service, "_get_chroma_manager_for_user", MagicMock(side_effect=ImportError("no chroma")))
    if failure_at == "resolve":
        monkeypatch.setattr(
            service.DatabasePaths, "resolve_user_base_directory", MagicMock(side_effect=ValueError("invalid path"))
        )
    else:
        monkeypatch.setattr(
            service.DatabasePaths, "resolve_user_base_directory", lambda *args, **kwargs: chroma_base / "7"
        )
        original_stat = Path.stat

        def inaccessible_storage(path, *args, **kwargs):
            if path == chroma_base / "7" / "chroma_storage":
                raise PermissionError("private storage")
            return original_stat(path, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", inaccessible_storage)

    with pytest.raises(service.DataSubjectRequestCoverageUnavailableError):
        await service._count_embeddings(7)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_at", ["list", "count"])
async def test_collection_failure_never_returns_zero_or_partial_count(monkeypatch, failure_at):
    manager = MagicMock()
    good_collection = MagicMock()
    good_collection.count.return_value = 5
    bad_collection = MagicMock()
    bad_collection.count.side_effect = RuntimeError("private embedding content")
    manager.list_collections.return_value = [good_collection, bad_collection]
    if failure_at == "list":
        manager.list_collections.side_effect = RuntimeError("private collection name")
    monkeypatch.setattr(service, "_get_chroma_manager_for_user", lambda user_id: manager)

    with pytest.raises(service.DataSubjectRequestCoverageUnavailableError):
        await service._count_embeddings(7)


@pytest.mark.asyncio
async def test_summary_only_queries_selected_categories_in_canonical_order(monkeypatch):
    async def unavailable(user_id):
        raise service.DataSubjectRequestCoverageUnavailableError("unselected store missing")

    async def notes(user_id):
        return 3

    async def embeddings(user_id):
        return 5

    for name in ("_count_media_records", "_count_chat_messages", "_count_audit_events"):
        monkeypatch.setattr(service, name, unavailable)
    monkeypatch.setattr(service, "_count_notes", notes)
    monkeypatch.setattr(service, "_count_embeddings", embeddings)

    assert await service._build_summary_for_user(user_id=7, selected_categories=["embeddings", "notes"]) == [
        {"key": "notes", "label": "Notes", "count": 3},
        {"key": "embeddings", "label": "Vector embeddings", "count": 5},
    ]
