"""Authoritative, selected-category DSR preview coverage regressions."""

from os import stat_result
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.services import admin_data_subject_requests_service as service

pytestmark = pytest.mark.integration


@pytest.fixture
def users_repo() -> MagicMock:
    """Resolve the subject through the public service's injected repository."""
    repo = MagicMock(spec=AuthnzUsersRepo)
    repo.get_user_by_id.return_value = {"id": 7, "username": "subject", "email": "subject@example.test"}
    return repo


@pytest.fixture
def chroma_base(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Use the same configured base as the DSR Chroma manager factory."""
    from tldw_Server_API.app.core.config import settings

    monkeypatch.setitem(settings, "USER_DB_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("TLDW_DB_ALLOWED_BASE_DIRS", str(tmp_path))
    return tmp_path


@pytest.mark.asyncio
async def test_unavailable_manager_with_existing_storage_rejects_preview(
    monkeypatch: pytest.MonkeyPatch, chroma_base: Path, users_repo: MagicMock,
) -> None:
    """Existing embedding storage cannot be reported as empty if its manager is unavailable."""
    (chroma_base / "7" / "chroma_storage").mkdir(parents=True)
    monkeypatch.setattr(service, "_get_chroma_manager_for_user", MagicMock(side_effect=ImportError("no chroma")))

    with pytest.raises(HTTPException) as exc_info:
        await service.preview_data_subject_request(
            requester_identifier="7", categories=["embeddings"], users_repo=users_repo,
        )
    assert (exc_info.value.status_code, exc_info.value.detail) == (500, "requester_data_unavailable")  # nosec B101


@pytest.mark.asyncio
async def test_unavailable_optional_manager_with_absent_storage_returns_zero(
    monkeypatch: pytest.MonkeyPatch, chroma_base: Path, users_repo: MagicMock,
) -> None:
    """Only confirmed absent optional storage yields an authoritative zero count."""
    monkeypatch.setattr(service, "_get_chroma_manager_for_user", MagicMock(side_effect=ImportError("no chroma")))

    preview = await service.preview_data_subject_request(
        requester_identifier="7", categories=["embeddings"], users_repo=users_repo,
    )
    assert preview["counts"] == {"embeddings": 0}
    assert not (chroma_base / "7").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_at", ["resolve", "stat"])
async def test_unavailable_manager_with_unknown_storage_rejects_preview(
    monkeypatch: pytest.MonkeyPatch, chroma_base: Path, failure_at: str, users_repo: MagicMock,
) -> None:
    """Path-policy and permission failures must not become empty previews."""
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

        def inaccessible_storage(path: Path, *args: Any, **kwargs: Any) -> stat_result:
            if path == chroma_base / "7" / "chroma_storage":
                raise PermissionError("private storage")
            return original_stat(path, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", inaccessible_storage)

    with pytest.raises(HTTPException) as exc_info:
        await service.preview_data_subject_request(
            requester_identifier="7", categories=["embeddings"], users_repo=users_repo,
        )
    assert (exc_info.value.status_code, exc_info.value.detail) == (500, "requester_data_unavailable")  # nosec B101


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_at", ["list", "count"])
async def test_collection_failure_never_returns_zero_or_partial_count(
    monkeypatch: pytest.MonkeyPatch, failure_at: str, users_repo: MagicMock,
) -> None:
    """Collection listing/count failures reject the entire public preview."""
    manager = MagicMock()
    good_collection = MagicMock()
    good_collection.count.return_value = 5
    bad_collection = MagicMock()
    bad_collection.count.side_effect = RuntimeError("private embedding content")
    manager.list_collections.return_value = [good_collection, bad_collection]
    if failure_at == "list":
        manager.list_collections.side_effect = RuntimeError("private collection name")
    monkeypatch.setattr(service, "_get_chroma_manager_for_user", lambda user_id: manager)

    with pytest.raises(HTTPException) as exc_info:
        await service.preview_data_subject_request(
            requester_identifier="7", categories=["embeddings"], users_repo=users_repo,
        )
    assert (exc_info.value.status_code, exc_info.value.detail) == (500, "requester_data_unavailable")  # nosec B101


@pytest.mark.asyncio
async def test_summary_only_queries_selected_categories_in_canonical_order(
    monkeypatch: pytest.MonkeyPatch, chroma_base: Path, users_repo: MagicMock,
) -> None:
    """Public previews count selected stores and tolerate missing unselected stores."""
    db = CharactersRAGDB(service.DatabasePaths.get_chacha_db_path(7), client_id="7")
    try:
        for title in ("First", "Second", "Third"):
            db.note_store.add_note(title, "Subject note")
    finally:
        db.close_all_connections()
    manager = MagicMock()
    collection = MagicMock()
    collection.count.return_value = 5
    manager.list_collections.return_value = [collection]
    # Chroma is optional and loaded lazily; isolate only its factory, keeping
    # public preview selection, real notes storage, and selected counters exercised.
    monkeypatch.setattr(service, "_get_chroma_manager_for_user", lambda user_id: manager)

    preview = await service.preview_data_subject_request(
        requester_identifier="7", categories=["embeddings", "notes"], users_repo=users_repo,
    )
    assert preview["summary"] == [
        {"key": "notes", "label": "Notes", "count": 3},
        {"key": "embeddings", "label": "Vector embeddings", "count": 5},
    ]
