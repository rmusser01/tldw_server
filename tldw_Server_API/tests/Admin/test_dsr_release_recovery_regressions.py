"""Regression tests for normalized embedding-storage checks during erasure."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_erasure_rejects_unavailable_chroma_at_normalized_tilde_path(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.config import settings
    from tldw_Server_API.app.services import admin_data_subject_requests_service as service

    configured_base = "~/" + os.path.relpath(tmp_path, Path.home())
    monkeypatch.setitem(settings, "USER_DB_BASE_DIR", configured_base)
    monkeypatch.setenv("TLDW_DB_ALLOWED_BASE_DIRS", str(tmp_path))
    real_storage = (
        service.DatabasePaths.resolve_user_base_directory(7, base_dir_override=configured_base) / "chroma_storage"
    )
    real_storage.mkdir(parents=True)
    sentinel = real_storage / "retained-synthetic-content"
    sentinel.write_text("synthetic embedding content")

    def unavailable_manager(_user_id):
        raise RuntimeError("injected manager failure")

    monkeypatch.setattr(service, "_get_chroma_manager_for_user", unavailable_manager)
    with pytest.raises(service.DataSubjectRequestCoverageUnavailableError):
        await service._count_embeddings(7)
    with pytest.raises(RuntimeError, match="cannot confirm|unavailable|availability"):
        await service._erase_embeddings(7)
    assert sentinel.read_text() == "synthetic embedding content"


@pytest.mark.parametrize("storage_state", ["absent", "existing", "uninspectable", "unresolved"])
async def test_erasure_requires_confirmed_storage_absence(tmp_path, monkeypatch, storage_state):
    from tldw_Server_API.app.core.config import settings
    from tldw_Server_API.app.services import admin_data_subject_requests_service as service

    monkeypatch.setitem(settings, "USER_DB_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("TLDW_DB_ALLOWED_BASE_DIRS", str(tmp_path))
    storage = tmp_path / "7" / "chroma_storage"
    if storage_state == "existing":
        storage.mkdir(parents=True)

    def unavailable_manager(_user_id):
        raise RuntimeError("injected manager failure")

    monkeypatch.setattr(service, "_get_chroma_manager_for_user", unavailable_manager)
    if storage_state == "unresolved":

        def unresolved(*_args, **_kwargs):
            raise ValueError("injected path resolution failure")

        monkeypatch.setattr(service.DatabasePaths, "resolve_user_base_directory", unresolved)
    elif storage_state == "uninspectable":
        original_stat = Path.stat

        def unavailable_stat(path, *args, **kwargs):
            if path == storage:
                raise PermissionError("injected storage inspection failure")
            return original_stat(path, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", unavailable_stat)

    if storage_state == "absent":
        assert await service._erase_embeddings(7) == 0
        assert not storage.exists()
    else:
        with pytest.raises(RuntimeError, match="cannot confirm|unavailable|availability"):
            await service._erase_embeddings(7)
