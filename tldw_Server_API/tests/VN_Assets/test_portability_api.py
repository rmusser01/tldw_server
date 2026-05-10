from __future__ import annotations

import json
import zipfile
from collections.abc import Generator, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints.vn_assets import (
    _job_manager,
    router as vn_assets_router,
)
from tldw_Server_API.app.api.v1.schemas.vn_asset_schemas import (
    VNAssetPackCreate,
    VNPackExportRequest,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository
from tldw_Server_API.app.core.VN_Assets.jobs import (
    VN_PACK_EXPORT_JOB_TYPE,
    VN_PACK_IMPORT_COMMIT_JOB_TYPE,
    VN_PACK_IMPORT_PREVIEW_JOB_TYPE,
)
from tldw_Server_API.app.core.VN_Assets.portability.constants import (
    CHECKSUMS_PATH,
    MANIFEST_PATH,
    VNPACK_SCHEMA_VERSION,
)
from tldw_Server_API.app.core.VN_Assets.portability.fingerprints import (
    canonical_json_bytes,
    canonical_payload_fingerprint,
    sha256_bytes,
)
from tldw_Server_API.app.core.VN_Assets.service import VNAssetPackService

PNG_BYTES = b"\x89PNG\r\n\x1a\npreview-api-png"


class FakeJobManager:
    def __init__(self) -> None:
        self.jobs: dict[int, dict[str, Any]] = {}
        self.created: list[dict[str, Any]] = []
        self.cancelled: list[tuple[int, str | None]] = []

    def create_job(self, **kwargs: Any) -> dict[str, Any]:
        job_id = len(self.jobs) + 1
        job = {
            "id": job_id,
            "uuid": f"job-{job_id}",
            "status": "queued",
            "result": None,
            **kwargs,
        }
        self.jobs[job_id] = job
        self.created.append(job)
        return job

    def get_job(self, job_id: int) -> dict[str, Any] | None:
        return self.jobs.get(int(job_id))

    def cancel_job(self, job_id: int, *, reason: str | None = None) -> bool:
        job = self.jobs.get(int(job_id))
        if job is None:
            return False
        self.cancelled.append((int(job_id), reason))
        job["status"] = "cancelled"
        return True


@pytest.fixture
def chacha_db(tmp_path) -> Generator[CharactersRAGDB, None, None]:
    database = CharactersRAGDB(str(tmp_path / "ChaChaNotes.db"), client_id="vn-portability-api-test")
    yield database
    database.close_connection()


@pytest.fixture
def character_id(chacha_db: CharactersRAGDB) -> int:
    return chacha_db.add_character_card(
        {
            "name": "Mira",
            "description": "A careful archivist.",
            "personality": "Patient and exacting.",
            "scenario": "Cataloging an orbital library.",
        }
    )


@pytest.fixture
def service(chacha_db: CharactersRAGDB) -> VNAssetPackService:
    return VNAssetPackService(chacha_db, owner_user_id=42)


@pytest.fixture
def repo(chacha_db: CharactersRAGDB) -> VNAssetPacksRepository:
    return VNAssetPacksRepository.initialized(chacha_db)


@pytest.fixture
def pack(service: VNAssetPackService, character_id: int) -> SimpleNamespace:
    created = service.create_pack(VNAssetPackCreate(title="Portable Pack", primary_character_id=character_id))
    return SimpleNamespace(id=created.id)


@pytest.fixture
def current_user_id() -> dict[str, int]:
    return {"value": 42}


@pytest.fixture
def fake_jobs() -> FakeJobManager:
    return FakeJobManager()


@pytest.fixture
def client(
    chacha_db: CharactersRAGDB,
    current_user_id: dict[str, int],
    fake_jobs: FakeJobManager,
) -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(vn_assets_router, prefix="/api/v1/vn")

    async def override_user() -> User:
        user_id = current_user_id["value"]
        return User(id=user_id, username=f"user-{user_id}")

    async def override_chacha_db() -> CharactersRAGDB:
        return chacha_db

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_chacha_db_for_user] = override_chacha_db
    app.dependency_overrides[_job_manager] = lambda: fake_jobs

    with TestClient(app) as test_client:
        yield test_client


def test_export_request_rejects_coerced_booleans() -> None:
    with pytest.raises(ValidationError):
        VNPackExportRequest(strict="true")


def test_vn_asset_job_start_routes_include_rbac_rate_limits() -> None:
    expected = {
        ("POST", "/vn-assets/packs/{pack_id}/export"): "vn_assets.export",
        ("POST", "/vn-assets/import/previews"): "vn_assets.import",
        ("POST", "/vn-assets/import/commit"): "vn_assets.import",
        ("POST", "/vn-assets/packs/{pack_id}/generate"): "vn_assets.generate",
        ("POST", "/vn-assets/packs/{pack_id}/slots/{slot_id}/retry"): "vn_assets.generate",
    }
    routes = {
        (method, route.path): route
        for route in vn_assets_router.routes
        if isinstance(route, APIRoute)
        for method in route.methods
    }

    for key, resource in expected.items():
        route = routes.get(key)
        assert route is not None
        resources = [
            str(limit_resource)
            for dep in route.dependant.dependencies
            if (limit_resource := getattr(dep.call, "_tldw_rate_limit_resource", None))
        ]
        assert resource in resources


def test_start_export_creates_jobs_backed_portability_row(
    client: TestClient,
    fake_jobs: FakeJobManager,
    repo: VNAssetPacksRepository,
    pack: SimpleNamespace,
) -> None:
    response = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack.id}/export",
        json={
            "strict": True,
            "include_full_provenance": False,
            "idempotency_key": "export-create-1",
        },
    )

    assert response.status_code == 202
    body = response.json()
    assert body["job_id"] == "1"
    assert body["operation"] == "export"
    assert body["pack_id"] == pack.id
    assert body["status"] == "queued"
    assert body["stage"] == "queued"

    row = repo.get_portability_job_by_job_id("1", owner_user_id=42)
    assert row is not None
    assert row["status"] == "queued"
    assert fake_jobs.created[0]["job_type"] == VN_PACK_EXPORT_JOB_TYPE
    assert fake_jobs.created[0]["payload"]["pack_id"] == pack.id


def test_start_export_replays_same_idempotency_key_and_conflicts_on_different_payload(
    client: TestClient,
    fake_jobs: FakeJobManager,
    pack: SimpleNamespace,
) -> None:
    payload = {"idempotency_key": "export-pack-1", "strict": True}

    first = client.post(f"/api/v1/vn/vn-assets/packs/{pack.id}/export", json=payload)
    replay = client.post(f"/api/v1/vn/vn-assets/packs/{pack.id}/export", json=payload)
    conflict = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack.id}/export",
        json={"idempotency_key": "export-pack-1", "strict": False},
    )

    assert first.status_code == 202
    assert replay.status_code == 202
    assert replay.json() == first.json()
    assert len(fake_jobs.created) == 1
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "idempotency_key_conflict"


def test_start_export_requires_idempotency_key(
    client: TestClient,
    fake_jobs: FakeJobManager,
    pack: SimpleNamespace,
) -> None:
    response = client.post(f"/api/v1/vn/vn-assets/packs/{pack.id}/export", json={})

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "idempotency_key_required"
    assert fake_jobs.created == []


def test_start_export_rejects_non_owned_pack(
    client: TestClient,
    current_user_id: dict[str, int],
    fake_jobs: FakeJobManager,
    repo: VNAssetPacksRepository,
    pack: SimpleNamespace,
) -> None:
    current_user_id["value"] = 7

    response = client.post(f"/api/v1/vn/vn-assets/packs/{pack.id}/export", json={})

    assert response.status_code == 404
    assert fake_jobs.created == []
    assert repo.get_portability_job_by_job_id("1", owner_user_id=42) is None


def test_export_status_composes_jobs_lifecycle_with_vn_stage(
    client: TestClient,
    fake_jobs: FakeJobManager,
    repo: VNAssetPacksRepository,
    pack: SimpleNamespace,
) -> None:
    started = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack.id}/export",
        json={"idempotency_key": "export-status-1"},
    )
    job_id = started.json()["job_id"]
    fake_jobs.jobs[int(job_id)]["status"] = "processing"
    repo.update_portability_job(
        job_id,
        {"status": "queued", "stage": "writing_archive", "progress": {"asset_count": 3}},
        owner_user_id=42,
    )

    response = client.get(f"/api/v1/vn/vn-assets/portability/exports/{job_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "processing"
    assert body["vn_status"] == "queued"
    assert body["stage"] == "writing_archive"
    assert body["progress"] == {"asset_count": 3}


def test_export_status_reconciles_terminal_jobs_state(
    client: TestClient,
    fake_jobs: FakeJobManager,
    repo: VNAssetPacksRepository,
    pack: SimpleNamespace,
) -> None:
    started = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack.id}/export",
        json={"idempotency_key": "export-status-2"},
    )
    job_id = started.json()["job_id"]
    fake_jobs.jobs[int(job_id)]["status"] = "completed"
    repo.update_portability_job(
        job_id,
        {"status": "processing", "stage": "writing_archive"},
        owner_user_id=42,
    )

    response = client.get(f"/api/v1/vn/vn-assets/portability/exports/{job_id}")

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    reconciled = repo.get_portability_job_by_job_id(job_id, owner_user_id=42)
    assert reconciled is not None
    assert reconciled["status"] == "completed"
    assert reconciled["stage"] == "writing_archive"


def test_export_download_rejects_incomplete_job(
    client: TestClient,
    pack: SimpleNamespace,
) -> None:
    started = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack.id}/export",
        json={"idempotency_key": "export-download-1"},
    )
    job_id = started.json()["job_id"]

    response = client.get(f"/api/v1/vn/vn-assets/portability/exports/{job_id}/download")

    assert response.status_code == 409
    assert response.json()["detail"] == "export_not_completed"


def test_export_cancel_routes_through_jobs_manager(
    client: TestClient,
    fake_jobs: FakeJobManager,
    repo: VNAssetPacksRepository,
    pack: SimpleNamespace,
) -> None:
    started = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack.id}/export",
        json={"idempotency_key": "export-cancel-1"},
    )
    job_id = started.json()["job_id"]

    response = client.post(f"/api/v1/vn/vn-assets/portability/exports/{job_id}/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert fake_jobs.cancelled == [(int(job_id), "vn_pack_export_cancel_requested")]
    row = repo.get_portability_job_by_job_id(job_id, owner_user_id=42)
    assert row is not None
    assert row["status"] == "cancelled"


def test_start_import_preview_creates_jobs_backed_preview_row(
    client: TestClient,
    fake_jobs: FakeJobManager,
    repo: VNAssetPacksRepository,
    tmp_path: Path,
) -> None:
    archive_path = _write_preview_archive(tmp_path / "preview.tldw-vnpack")

    with archive_path.open("rb") as archive_file:
        response = client.post(
            "/api/v1/vn/vn-assets/import/previews",
            data={"idempotency_key": "preview-create-1"},
            files={"archive": ("preview.tldw-vnpack", archive_file, "application/zip")},
        )

    assert response.status_code == 202
    body = response.json()
    assert body["job_id"] == "1"
    assert body["operation"] == "import_preview"
    assert body["preview_id"] >= 1
    assert body["status"] == "queued"
    assert body["stage"] == "queued"

    preview = repo.get_import_preview(body["preview_id"], owner_user_id=42)
    assert preview is not None
    assert preview["status"] == "queued"
    assert Path(preview["archive_path"]).is_file()
    assert Path(preview["archive_path"]).suffix == ".tldw-vnpack"

    portability_job = repo.get_portability_job_by_job_id("1", owner_user_id=42)
    assert portability_job is not None
    assert portability_job["operation"] == "import_preview"
    assert portability_job["preview_id"] == body["preview_id"]
    assert fake_jobs.created[0]["job_type"] == VN_PACK_IMPORT_PREVIEW_JOB_TYPE
    assert fake_jobs.created[0]["payload"]["preview_id"] == body["preview_id"]


def test_start_import_preview_replays_same_idempotency_key_and_conflicts_on_different_archive(
    client: TestClient,
    fake_jobs: FakeJobManager,
    tmp_path: Path,
) -> None:
    archive_path = _write_preview_archive(tmp_path / "preview.tldw-vnpack")
    conflict_archive_path = tmp_path / "preview-conflict.tldw-vnpack"
    conflict_archive_path.write_bytes(archive_path.read_bytes() + b"changed")

    with archive_path.open("rb") as archive_file:
        first = client.post(
            "/api/v1/vn/vn-assets/import/previews",
            data={"idempotency_key": "preview-upload-1"},
            files={"archive": ("preview.tldw-vnpack", archive_file, "application/zip")},
        )
    with archive_path.open("rb") as archive_file:
        replay = client.post(
            "/api/v1/vn/vn-assets/import/previews",
            data={"idempotency_key": "preview-upload-1"},
            files={"archive": ("preview.tldw-vnpack", archive_file, "application/zip")},
        )
    with conflict_archive_path.open("rb") as archive_file:
        conflict = client.post(
            "/api/v1/vn/vn-assets/import/previews",
            data={"idempotency_key": "preview-upload-1"},
            files={"archive": ("preview.tldw-vnpack", archive_file, "application/zip")},
        )

    assert first.status_code == 202
    assert replay.status_code == 202
    assert replay.json() == first.json()
    assert len(fake_jobs.created) == 1
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "idempotency_key_conflict"


def test_start_import_preview_requires_idempotency_key(
    client: TestClient,
    fake_jobs: FakeJobManager,
    tmp_path: Path,
) -> None:
    archive_path = _write_preview_archive(tmp_path / "preview-missing-key.tldw-vnpack")

    with archive_path.open("rb") as archive_file:
        response = client.post(
            "/api/v1/vn/vn-assets/import/previews",
            files={"archive": ("preview.tldw-vnpack", archive_file, "application/zip")},
        )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "idempotency_key_required"
    assert fake_jobs.created == []


def test_get_import_preview_composes_preview_and_jobs_status(
    client: TestClient,
    fake_jobs: FakeJobManager,
    repo: VNAssetPacksRepository,
    tmp_path: Path,
) -> None:
    archive_path = _write_preview_archive(tmp_path / "preview.tldw-vnpack")
    with archive_path.open("rb") as archive_file:
        started = client.post(
            "/api/v1/vn/vn-assets/import/previews",
            data={"idempotency_key": "preview-status-1"},
            files={"archive": ("preview.tldw-vnpack", archive_file, "application/zip")},
        )
    preview_id = started.json()["preview_id"]
    job_id = started.json()["job_id"]
    fake_jobs.jobs[int(job_id)]["status"] = "completed"
    repo.update_import_preview(
        preview_id,
        {
            "status": "completed",
            "archive_sha256": "a" * 64,
            "canonical_payload_fingerprint": "b" * 64,
            "schema_version": VNPACK_SCHEMA_VERSION,
            "bundle_summary": {"item_count": 2},
            "validation_warnings": ["missing_asset_bytes:sprite:key:variant:1"],
            "required_choices": [{"choice_id": "primary_character"}],
        },
        owner_user_id=42,
    )
    repo.update_portability_job(
        job_id,
        {"status": "processing", "stage": "completed"},
        owner_user_id=42,
    )

    response = client.get(f"/api/v1/vn/vn-assets/import/previews/{preview_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["vn_status"] == "completed"
    assert body["stage"] == "completed"
    assert body["bundle_summary"] == {"item_count": 2}
    assert body["validation_warnings"] == ["missing_asset_bytes:sprite:key:variant:1"]
    assert body["required_choices"] == [{"choice_id": "primary_character"}]


def test_import_preview_cancel_routes_through_jobs_manager(
    client: TestClient,
    fake_jobs: FakeJobManager,
    repo: VNAssetPacksRepository,
    tmp_path: Path,
) -> None:
    archive_path = _write_preview_archive(tmp_path / "preview.tldw-vnpack")
    with archive_path.open("rb") as archive_file:
        started = client.post(
            "/api/v1/vn/vn-assets/import/previews",
            data={"idempotency_key": "preview-cancel-1"},
            files={"archive": ("preview.tldw-vnpack", archive_file, "application/zip")},
        )
    preview_id = started.json()["preview_id"]
    job_id = started.json()["job_id"]

    response = client.post(f"/api/v1/vn/vn-assets/import/previews/{preview_id}/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert fake_jobs.cancelled == [(int(job_id), "vn_pack_import_preview_cancel_requested")]
    preview = repo.get_import_preview(preview_id, owner_user_id=42)
    assert preview is not None
    assert preview["status"] == "cancelled"


def test_import_preview_delete_marks_preview_and_removes_uploaded_archive(
    client: TestClient,
    fake_jobs: FakeJobManager,
    repo: VNAssetPacksRepository,
    tmp_path: Path,
) -> None:
    archive_path = _write_preview_archive(tmp_path / "preview.tldw-vnpack")
    with archive_path.open("rb") as archive_file:
        started = client.post(
            "/api/v1/vn/vn-assets/import/previews",
            data={"idempotency_key": "preview-delete-1"},
            files={"archive": ("preview.tldw-vnpack", archive_file, "application/zip")},
        )
    preview_id = started.json()["preview_id"]
    job_id = started.json()["job_id"]
    preview = repo.get_import_preview(preview_id, owner_user_id=42)
    assert preview is not None
    uploaded_archive_path = Path(preview["archive_path"])
    assert uploaded_archive_path.is_file()

    response = client.delete(f"/api/v1/vn/vn-assets/import/previews/{preview_id}")

    assert response.status_code == 204
    assert not uploaded_archive_path.exists()
    deleted_preview = repo.get_import_preview(preview_id, owner_user_id=42)
    assert deleted_preview is not None
    assert deleted_preview["status"] == "deleted"
    assert fake_jobs.cancelled == [(int(job_id), "vn_pack_import_preview_delete_requested")]


def test_start_import_commit_creates_jobs_backed_journal_row(
    client: TestClient,
    fake_jobs: FakeJobManager,
    repo: VNAssetPacksRepository,
    character_id: int,
    tmp_path: Path,
) -> None:
    archive_path = _write_preview_archive(tmp_path / "commit.tldw-vnpack")
    preview = repo.create_import_preview(
        owner_user_id=42,
        job_id="preview-job-1",
        status="completed",
        archive_path=str(archive_path),
        archive_sha256="a" * 64,
        canonical_payload_fingerprint="b" * 64,
        schema_version=VNPACK_SCHEMA_VERSION,
    )

    response = client.post(
        "/api/v1/vn/vn-assets/import/commit",
        json={
            "preview_id": preview["id"],
            "trust_mode": "trusted_restore",
            "target_mode": "create_new",
            "character_action": "link_existing_character",
            "target_character_id": character_id,
            "idempotency_key": "commit-create-1",
        },
    )

    assert response.status_code == 202
    body = response.json()
    assert body["job_id"] == "1"
    assert body["operation"] == "import_commit"
    assert body["preview_id"] == preview["id"]
    assert body["import_id"] >= 1
    assert body["status"] == "queued"
    assert body["stage"] == "queued"

    journal = repo.get_import_journal(body["import_id"], owner_user_id=42)
    assert journal is not None
    assert journal["status"] == "queued"
    assert journal["job_id"] == "1"
    portability_job = repo.get_portability_job_by_job_id("1", owner_user_id=42)
    assert portability_job is not None
    assert portability_job["operation"] == "import_commit"
    assert portability_job["import_id"] == body["import_id"]
    assert fake_jobs.created[0]["job_type"] == VN_PACK_IMPORT_COMMIT_JOB_TYPE
    assert fake_jobs.created[0]["payload"]["import_id"] == body["import_id"]


def test_start_import_commit_replays_same_idempotency_key_and_conflicts_on_different_payload(
    client: TestClient,
    fake_jobs: FakeJobManager,
    repo: VNAssetPacksRepository,
    character_id: int,
    tmp_path: Path,
) -> None:
    archive_path = _write_preview_archive(tmp_path / "commit-idempotent.tldw-vnpack")
    preview = repo.create_import_preview(
        owner_user_id=42,
        job_id="preview-job-1",
        status="completed",
        archive_path=str(archive_path),
        archive_sha256="a" * 64,
        canonical_payload_fingerprint="b" * 64,
        schema_version=VNPACK_SCHEMA_VERSION,
    )
    payload = {
        "preview_id": preview["id"],
        "trust_mode": "trusted_restore",
        "target_mode": "create_new",
        "character_action": "link_existing_character",
        "target_character_id": character_id,
        "idempotency_key": "commit-1",
    }

    first = client.post("/api/v1/vn/vn-assets/import/commit", json=payload)
    replay = client.post("/api/v1/vn/vn-assets/import/commit", json=payload)
    conflict = client.post(
        "/api/v1/vn/vn-assets/import/commit",
        json={**payload, "trust_mode": "untrusted_import"},
    )

    assert first.status_code == 202
    assert replay.status_code == 202
    assert replay.json() == first.json()
    assert len(fake_jobs.created) == 1
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "idempotency_key_conflict"


def test_start_import_commit_releases_idempotency_claim_on_enqueue_failure(
    client: TestClient,
    fake_jobs: FakeJobManager,
    repo: VNAssetPacksRepository,
    character_id: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = _write_preview_archive(tmp_path / "commit-enqueue-failure.tldw-vnpack")
    preview = repo.create_import_preview(
        owner_user_id=42,
        job_id="preview-job-1",
        status="completed",
        archive_path=str(archive_path),
        archive_sha256="a" * 64,
        canonical_payload_fingerprint="b" * 64,
        schema_version=VNPACK_SCHEMA_VERSION,
    )
    payload = {
        "preview_id": preview["id"],
        "trust_mode": "trusted_restore",
        "target_mode": "create_new",
        "character_action": "link_existing_character",
        "target_character_id": character_id,
        "idempotency_key": "commit-enqueue-failure",
    }

    def fail_create_job(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("job enqueue failed")

    monkeypatch.setattr(fake_jobs, "create_job", fail_create_job)
    with pytest.raises(RuntimeError, match="job enqueue failed"):
        client.post("/api/v1/vn/vn-assets/import/commit", json=payload)

    record = repo.get_idempotency_record(
        owner_user_id=42,
        scope="vn_asset_import_commit",
        resource_id=f"preview:{preview['id']}",
        idempotency_key="commit-enqueue-failure",
    )
    assert record is None

    monkeypatch.undo()
    retry = client.post("/api/v1/vn/vn-assets/import/commit", json=payload)

    assert retry.status_code == 202
    assert retry.json()["operation"] == "import_commit"


def test_start_import_commit_requires_idempotency_key(
    client: TestClient,
    fake_jobs: FakeJobManager,
    repo: VNAssetPacksRepository,
    character_id: int,
    tmp_path: Path,
) -> None:
    archive_path = _write_preview_archive(tmp_path / "commit-missing-key.tldw-vnpack")
    preview = repo.create_import_preview(
        owner_user_id=42,
        job_id="preview-job-1",
        status="completed",
        archive_path=str(archive_path),
        archive_sha256="a" * 64,
        canonical_payload_fingerprint="b" * 64,
        schema_version=VNPACK_SCHEMA_VERSION,
    )

    response = client.post(
        "/api/v1/vn/vn-assets/import/commit",
        json={
            "preview_id": preview["id"],
            "trust_mode": "trusted_restore",
            "target_mode": "create_new",
            "character_action": "link_existing_character",
            "target_character_id": character_id,
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "idempotency_key_required"
    assert fake_jobs.created == []


def test_import_commit_status_composes_jobs_lifecycle_with_journal(
    client: TestClient,
    fake_jobs: FakeJobManager,
    repo: VNAssetPacksRepository,
    character_id: int,
    tmp_path: Path,
) -> None:
    archive_path = _write_preview_archive(tmp_path / "commit-status.tldw-vnpack")
    preview = repo.create_import_preview(
        owner_user_id=42,
        job_id="preview-job-1",
        status="completed",
        archive_path=str(archive_path),
        archive_sha256="a" * 64,
        canonical_payload_fingerprint="b" * 64,
        schema_version=VNPACK_SCHEMA_VERSION,
    )
    started = client.post(
        "/api/v1/vn/vn-assets/import/commit",
        json={
            "preview_id": preview["id"],
            "trust_mode": "untrusted_import",
            "target_mode": "create_new",
            "character_action": "link_existing_character",
            "target_character_id": character_id,
            "idempotency_key": "commit-status-1",
        },
    )
    job_id = started.json()["job_id"]
    import_id = started.json()["import_id"]
    fake_jobs.jobs[int(job_id)]["status"] = "processing"
    repo.update_import_journal(
        import_id,
        {"status": "processing", "stage": "creating_items", "id_maps": {"packs": {"100": 200}}},
        owner_user_id=42,
    )
    repo.update_portability_job(
        job_id,
        {"status": "queued", "stage": "creating_items"},
        owner_user_id=42,
    )

    response = client.get(f"/api/v1/vn/vn-assets/portability/imports/{job_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "processing"
    assert body["vn_status"] == "processing"
    assert body["stage"] == "creating_items"
    assert body["id_maps"] == {"packs": {"100": 200}}


def test_import_commit_cancel_routes_through_jobs_manager(
    client: TestClient,
    fake_jobs: FakeJobManager,
    repo: VNAssetPacksRepository,
    character_id: int,
    tmp_path: Path,
) -> None:
    archive_path = _write_preview_archive(tmp_path / "commit-cancel.tldw-vnpack")
    preview = repo.create_import_preview(
        owner_user_id=42,
        job_id="preview-job-1",
        status="completed",
        archive_path=str(archive_path),
        archive_sha256="a" * 64,
        canonical_payload_fingerprint="b" * 64,
        schema_version=VNPACK_SCHEMA_VERSION,
    )
    started = client.post(
        "/api/v1/vn/vn-assets/import/commit",
        json={
            "preview_id": preview["id"],
            "trust_mode": "trusted_restore",
            "target_mode": "create_new",
            "character_action": "link_existing_character",
            "target_character_id": character_id,
            "idempotency_key": "commit-cancel-1",
        },
    )
    job_id = started.json()["job_id"]

    response = client.post(f"/api/v1/vn/vn-assets/portability/imports/{job_id}/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert fake_jobs.cancelled == [(int(job_id), "vn_pack_import_commit_cancel_requested")]


def _write_preview_archive(archive_path: Path) -> Path:
    pack = {
        "pack": {
            "source_pack_id": 10,
            "title": "Preview API Pack",
            "status": "draft",
            "content_rating": "general",
            "primary_character_id": 20,
        }
    }
    slots = {
        "slots": [
            {
                "source_slot_id": 30,
                "asset_type": "sprite",
                "slot_key": "sprite.primary.neutral",
                "required_for_runtime": True,
            }
        ]
    }
    items = {
        "items": [
            {
                "source_item_id": 40,
                "source_slot_id": 30,
                "asset_type": "sprite",
                "slot_key": "sprite.primary.neutral",
                "variant_index": 0,
                "review_status": "approved",
                "asset_bytes_status": "present",
                "asset_path": "assets/items/neutral.png",
                "asset_sha256": sha256_bytes(PNG_BYTES),
                "asset_size_bytes": len(PNG_BYTES),
            }
        ]
    }
    payloads = {
        "metadata/pack.json": canonical_json_bytes(pack),
        "metadata/slots.json": canonical_json_bytes(slots),
        "metadata/items.json": canonical_json_bytes(items),
        "assets/items/neutral.png": PNG_BYTES,
    }
    checksums = {path: sha256_bytes(content) for path, content in sorted(payloads.items())}
    manifest = {
        "schema_version": VNPACK_SCHEMA_VERSION,
        "archive_profile": "backup",
        "pack_title": pack["pack"]["title"],
        "content_rating": "general",
        "canonical_payload_fingerprint": canonical_payload_fingerprint(
            {"pack": pack["pack"], "slots": slots["slots"], "items": items["items"]}
        ),
        "counts": {"slots": 1, "items": 1, "assets_with_bytes": 1},
        "include_images": True,
        "include_character": False,
        "include_world_books": False,
        "provenance_mode": "redacted",
        "encryption": {"encrypted": False, "scheme": None},
        "sections": [{"path": path, "sha256": digest} for path, digest in sorted(checksums.items())],
        "warnings": [],
    }
    payloads[MANIFEST_PATH] = canonical_json_bytes(manifest)
    checksums[MANIFEST_PATH] = sha256_bytes(payloads[MANIFEST_PATH])
    payloads[CHECKSUMS_PATH] = canonical_json_bytes(checksums)

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, content in sorted(payloads.items()):
            archive.writestr(path, content)
    return archive_path
