from __future__ import annotations

import zipfile
from collections.abc import Generator
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_db_pool
from tldw_Server_API.app.api.v1.endpoints import visual_identities
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VisualIdentity_DB import VisualIdentityRepository
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Visual_Identities.jobs import (
    VISUAL_IDENTITIES_DOMAIN,
    VISUAL_IDENTITY_IMPORT_ZIP_JOB_TYPE,
    visual_identity_jobs_queue,
)
from tldw_Server_API.app.services import visual_identity_jobs_worker

pytestmark = pytest.mark.unit


@pytest.fixture
def chacha_db(tmp_path: Path) -> Generator[CharactersRAGDB, None, None]:
    database = CharactersRAGDB(
        str(tmp_path / "ChaChaNotes.db"),
        client_id="visual-identity-worker-test-client",
    )
    yield database
    database.close_connection()


@pytest.fixture
def repo(chacha_db: CharactersRAGDB) -> VisualIdentityRepository:
    return VisualIdentityRepository.initialized(chacha_db)


@pytest.fixture
def storage_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "visual-identities"
    root.mkdir()
    monkeypatch.setattr(
        DatabasePaths,
        "get_user_visual_identities_dir",
        staticmethod(lambda owner_user_id: root / str(owner_user_id)),
    )
    return root


def _user(user_id: int) -> User:
    return User(
        id=user_id,
        username=f"user-{user_id}",
        email=f"user-{user_id}@example.test",
        roles=["user"],
        permissions=[],
    )


def _client(chacha_db: CharactersRAGDB, *, jobs_manager: JobManager) -> TestClient:
    app = FastAPI()
    app.include_router(
        visual_identities.router,
        prefix="/api/v1/visual-identities",
        tags=["visual-identities"],
    )
    app.dependency_overrides[visual_identities.get_request_user] = lambda: _user(1)
    app.dependency_overrides[visual_identities.get_chacha_db_for_user] = lambda: chacha_db
    app.dependency_overrides[visual_identities._job_manager] = lambda: jobs_manager
    app.dependency_overrides[get_db_pool] = lambda: object()
    return TestClient(app)


@pytest.mark.asyncio
async def test_zip_import_job_worker_processes_api_enqueued_draft(
    chacha_db: CharactersRAGDB,
    repo: VisualIdentityRepository,
    storage_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs_manager = JobManager(db_path=tmp_path / "jobs.db")
    client = _client(chacha_db, jobs_manager=jobs_manager)

    response = client.post(
        "/api/v1/visual-identities/imports/zip",
        data={"title": "Worker import", "idempotency_key": "worker-import-1"},
        files={"archive": ("expressions.zip", _zip_bytes_with_png(), "application/zip")},
    )
    assert response.status_code == 202
    draft_id = int(response.json()["draft_id"])
    assert repo.get_draft(draft_id, owner_user_id=1)["status"] == "importing"

    acquired = jobs_manager.acquire_next_job(
        domain=VISUAL_IDENTITIES_DOMAIN,
        queue=visual_identity_jobs_queue(),
        lease_seconds=60,
        worker_id="visual-identity-worker-test",
        owner_user_id="1",
        job_type=VISUAL_IDENTITY_IMPORT_ZIP_JOB_TYPE,
    )
    assert acquired is not None

    async def _get_chacha_db_for_user_id(
        user_id: int,
        *,
        client_id: str | None = None,
    ) -> CharactersRAGDB:
        assert user_id == 1
        assert client_id == "visual-identity-worker-1"
        return chacha_db

    monkeypatch.setattr(
        visual_identity_jobs_worker,
        "get_chacha_db_for_user_id",
        _get_chacha_db_for_user_id,
    )
    monkeypatch.setattr(visual_identity_jobs_worker, "_close_worker_database", lambda db: None)

    result = await visual_identity_jobs_worker.handle_visual_identity_import_zip_job(
        acquired,
        job_manager=jobs_manager,
        storage_root=storage_root / "1",
    )

    assert result["draft_id"] == draft_id
    assert result["status"] == "ready_for_review"
    imported = repo.get_draft(draft_id, owner_user_id=1)
    assert imported["status"] == "ready_for_review"
    assert repo.list_draft_assets(draft_id, owner_user_id=1)[0]["expression_key"] == "neutral"


@pytest.mark.asyncio
async def test_zip_import_job_worker_rejects_owner_payload_mismatch() -> None:
    with pytest.raises(ValueError, match="visual_identity_job_owner_mismatch"):
        await visual_identity_jobs_worker.handle_visual_identity_import_zip_job(
            {
                "id": 1,
                "job_type": VISUAL_IDENTITY_IMPORT_ZIP_JOB_TYPE,
                "owner_user_id": "1",
                "payload": {
                    "owner_user_id": 2,
                    "draft_id": 7,
                    "upload_path": "/tmp/expression.zip",
                    "source_filename": "expression.zip",
                },
            }
        )


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGBA", (8, 8), "purple").save(buffer, format="PNG")
    return buffer.getvalue()


def _zip_bytes_with_png() -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        archive.writestr("neutral.png", _png_bytes())
    return buffer.getvalue()
