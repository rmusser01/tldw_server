from __future__ import annotations

import json
import zipfile
from collections.abc import Generator, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import storage_user_files, vn_assets
from tldw_Server_API.app.api.v1.endpoints.vn_assets import (
    _job_manager,
    router as vn_assets_router,
)
from tldw_Server_API.app.api.v1.schemas.storage_schemas import BulkDeleteRequest
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

    def get_job(self, job_id: int, *, owner_user_id: str | None = None) -> dict[str, Any] | None:
        job = self.jobs.get(int(job_id))
        if job and (owner_user_id is None or job.get("owner_user_id") == owner_user_id):
            return job
        return None

    def cancel_job(
        self,
        job_id: int,
        *,
        reason: str | None = None,
        expected_uuid: str | None = None,
        expected_domain: str | None = None,
        expected_job_type: str | None = None,
    ) -> bool:
        job = self.jobs.get(int(job_id))
        if job is None or job["status"] in {"completed", "failed", "cancelled", "quarantined"}:
            return False
        if (
            (expected_uuid is not None and job["uuid"] != expected_uuid)
            or (expected_domain is not None and job["domain"] != expected_domain)
            or (expected_job_type is not None and job["job_type"] != expected_job_type)
        ):
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
    assert first.status_code == 202
    assert client.delete(f"/api/v1/vn/vn-assets/packs/{pack.id}").status_code == 204
    replay = client.post(f"/api/v1/vn/vn-assets/packs/{pack.id}/export", json=payload)
    conflict = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack.id}/export",
        json={"idempotency_key": "export-pack-1", "strict": False},
    )

    assert replay.status_code == 202
    assert replay.json() == first.json()
    assert len(fake_jobs.created) == 1
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "idempotency_key_conflict"


def test_export_checks_captured_account_before_admission(
    client: TestClient, fake_jobs: FakeJobManager, pack: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def principal(_request: Any) -> SimpleNamespace:
        return SimpleNamespace(user_id=42)

    monkeypatch.setattr(auth_deps, "get_auth_principal", principal)
    endpoint = f"/api/v1/vn/vn-assets/packs/{pack.id}/export"
    payload = {"idempotency_key": "export-scoped"}
    stale = client.post(endpoint, json=payload, headers={"X-TLDW-Expected-User-ID": "43"})

    assert stale.status_code == 412
    assert stale.json()["detail"]["code"] == "request_config_scope_changed"
    assert not fake_jobs.created
    assert client.post(endpoint, json=payload, headers={"X-TLDW-Expected-User-ID": "42"}).status_code == 202


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


@pytest.mark.parametrize(
    ("preview_status", "replacement_owner"),
    [("queued", None), ("completed", None), ("queued", "42"), ("queued", "43")],
)
def test_import_preview_delete_marks_preview_and_removes_uploaded_archive(
    client: TestClient,
    fake_jobs: FakeJobManager,
    repo: VNAssetPacksRepository,
    tmp_path: Path,
    preview_status: str,
    replacement_owner: str | None,
    monkeypatch: pytest.MonkeyPatch,
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
    jobs_row = fake_jobs.jobs[int(job_id)]
    jobs_row["status"] = preview_status
    if replacement_owner is not None:
        jobs_row.update(
            owner_user_id=replacement_owner,
            uuid="replacement",
            payload={**jobs_row["payload"], "preview_id": preview_id + 1},
        )
    repo.update_import_preview(preview_id, {"status": preview_status}, owner_user_id=42)
    preview = repo.get_import_preview(preview_id, owner_user_id=42)
    assert preview is not None
    uploaded_archive_path = Path(preview["archive_path"])
    assert uploaded_archive_path.is_file()
    endpoint = f"/api/v1/vn/vn-assets/import/previews/{preview_id}"

    if preview_status == "completed":
        async def fail_unlink(_path: Path) -> None:
            raise OSError("file removal interrupted")

        with monkeypatch.context() as patch:
            patch.setattr(vn_assets, "_remove_file_if_exists", fail_unlink)
            with pytest.raises(OSError, match="file removal interrupted"):
                client.delete(endpoint)
        assert uploaded_archive_path.is_file()
        assert client.get(endpoint).json()["vn_status"] == "deleting"

    response = client.delete(endpoint)

    assert response.status_code == 204
    assert not uploaded_archive_path.exists()
    deleted_preview = repo.get_import_preview(preview_id, owner_user_id=42)
    assert deleted_preview is not None
    assert deleted_preview["status"] == "deleted"
    assert client.delete(endpoint).status_code == 204
    response = client.get(endpoint)
    assert response.json()["vn_status"] == "deleted"
    if replacement_owner is not None:
        assert jobs_row["status"] == "queued"
    elif preview_status == "queued":
        assert fake_jobs.cancelled == [(int(job_id), "vn_pack_import_preview_delete_requested")]


@pytest.mark.parametrize("journal_status", ["queued", "failed"])
def test_preview_delete_keeps_archive_for_active_import(
    client: TestClient,
    fake_jobs: FakeJobManager,
    repo: VNAssetPacksRepository,
    tmp_path: Path,
    journal_status: str,
) -> None:
    archive_path = _write_preview_archive(tmp_path / "active-import.tldw-vnpack")
    with archive_path.open("rb") as archive_file:
        started = client.post(
            "/api/v1/vn/vn-assets/import/previews",
            data={"idempotency_key": "preview-active-import"},
            files={"archive": (archive_path.name, archive_file, "application/zip")},
        )
    preview_id = started.json()["preview_id"]
    repo.update_import_preview(preview_id, {"status": "completed"}, owner_user_id=42)
    fake_jobs.jobs[int(started.json()["job_id"])]["status"] = "completed"
    admitted = client.post(
        "/api/v1/vn/vn-assets/import/commit",
        json={
            "preview_id": preview_id,
            "trust_mode": "trusted_restore",
            "character_action": "import_included_character",
            "idempotency_key": "active-import",
        },
    )
    assert admitted.status_code == 202
    repo.update_import_journal(admitted.json()["import_id"], {"status": journal_status}, owner_user_id=42)

    response = client.delete(f"/api/v1/vn/vn-assets/import/previews/{preview_id}")

    assert response.status_code == 409
    assert response.json()["detail"] == "import_preview_has_active_import"
    preview = repo.get_import_preview(preview_id, owner_user_id=42)
    assert preview["status"] == "completed"
    assert Path(preview["archive_path"]).is_file()


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


@pytest.mark.parametrize(
    ("preview_mutation", "fresh_error"),
    [
        ({"status": "deleted"}, "import_preview_not_completed"),
        ({"expires_at": "2020-01-01T00:00:00+00:00"}, "import_preview_expired"),
    ],
)
def test_start_import_commit_replays_same_idempotency_key_and_conflicts_on_different_payload(
    client: TestClient,
    fake_jobs: FakeJobManager,
    repo: VNAssetPacksRepository,
    character_id: int,
    tmp_path: Path,
    preview_mutation: dict[str, str],
    fresh_error: str,
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
    assert first.status_code == 202
    repo.update_import_preview(preview["id"], preview_mutation, owner_user_id=42)
    replay = client.post("/api/v1/vn/vn-assets/import/commit", json=payload)
    conflict = client.post(
        "/api/v1/vn/vn-assets/import/commit",
        json={**payload, "trust_mode": "untrusted_import"},
    )

    assert replay.status_code == 202
    assert replay.json() == first.json()
    assert len(fake_jobs.created) == 1
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "idempotency_key_conflict"
    different_start = client.post("/api/v1/vn/vn-assets/import/commit", json={**payload, "idempotency_key": "new"})
    assert different_start.status_code == 409
    assert different_start.json()["detail"] == fresh_error


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


def test_import_admission_failure_leaves_no_runnable_job_binding(
    client: TestClient, fake_jobs: FakeJobManager, repo: VNAssetPacksRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    preview = repo.create_import_preview(owner_user_id=42, job_id="preview", status="completed", archive_path="preview.tldw-vnpack")
    payload = {
        "preview_id": preview["id"],
        "trust_mode": "trusted_restore",
        "character_action": "import_included_character",
        "idempotency_key": "import-receipt-publish",
    }
    publish = VNAssetPackService.complete_idempotency_response

    def fail_publication(self: VNAssetPackService, **kwargs: Any) -> None:
        publish(self, **kwargs)
        raise RuntimeError("import receipt readback failed")

    with monkeypatch.context() as patch:
        patch.setattr(VNAssetPackService, "complete_idempotency_response", fail_publication)
        with pytest.raises(RuntimeError, match="import receipt readback failed"):
            client.post("/api/v1/vn/vn-assets/import/commit", json=payload)

    failed_job = fake_jobs.created[0]
    assert repo.get_portability_job_by_job_id(str(failed_job["id"]), owner_user_id=42) is None
    assert client.post("/api/v1/vn/vn-assets/import/commit", json=payload).status_code == 202


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


@pytest.mark.parametrize(
    ("jobs_status", "replacement", "response_status"),
    [
        ("processing", False, "processing"),
        ("completed", False, "completed"),
        ("completed", True, "queued"),
    ],
)
def test_import_commit_status_composes_jobs_lifecycle_with_journal(
    client: TestClient,
    fake_jobs: FakeJobManager,
    repo: VNAssetPacksRepository,
    character_id: int,
    tmp_path: Path,
    jobs_status: str,
    replacement: bool,
    response_status: str,
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
    jobs_row = fake_jobs.jobs[int(job_id)]
    jobs_row["status"] = jobs_status
    if replacement:
        jobs_row["payload"] = {**jobs_row["payload"], "import_id": import_id + 1}
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
    assert body["status"] == response_status
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


@pytest.mark.parametrize("registered", [False, True])
def test_cleanup_interrupted_import_preserves_original_pack_and_removes_expired_preview(
    client: TestClient, fake_jobs: FakeJobManager, repo: VNAssetPacksRepository,
    pack: SimpleNamespace, current_user_id: dict[str, int], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    registered: bool,
) -> None:
    archive_path = _write_preview_archive(tmp_path / "interrupted.tldw-vnpack")
    monkeypatch.setenv("VN_PACK_IMPORT_PREVIEW_STAGING_ROOT", str(tmp_path / "previews"))
    with archive_path.open("rb") as archive:
        started = client.post(
            "/api/v1/vn/vn-assets/import/previews",
            data={"idempotency_key": "interrupted-preview"},
            files={"archive": (archive_path.name, archive, "application/zip")},
        ).json()
    preview_id = started["preview_id"]
    repo.update_import_preview(preview_id, {"status": "completed"}, owner_user_id=42)
    fake_jobs.jobs[int(started["job_id"])]["status"] = "completed"
    admitted = client.post(
        "/api/v1/vn/vn-assets/import/commit",
        json={
            "preview_id": preview_id, "idempotency_key": "interrupted-commit",
            "trust_mode": "trusted_restore", "character_action": "import_included_character",
            "target_pack_id": pack.id, "target_mode": "update_existing",
        },
    )
    assert admitted.status_code == 202
    import_id = admitted.json()["import_id"]
    job_id = admitted.json()["job_id"]
    repo.update_import_preview(preview_id, {"expires_at": "2020-01-01T00:00:00+00:00"}, owner_user_id=42)
    staged = Path(repo.get_import_preview(preview_id, owner_user_id=42)["archive_path"])
    original_slot = repo.create_slot(pack_id=pack.id, asset_type="sprite", slot_key="original")
    original_item = repo.create_item(pack_id=pack.id, slot_id=original_slot["id"], generated_file_id=15)
    extra_slot = repo.create_slot(pack_id=pack.id, asset_type="sprite", slot_key="expanded")
    extra_item = repo.create_item(
        pack_id=pack.id, slot_id=extra_slot["id"], source="imported", review_status="hidden",
        source_context_snapshot={"vnpack_import": {"import_id": import_id, "source_item_id": 40}},
    )
    outputs = tmp_path / "outputs"
    relative_path = f"vn_assets/imports/{import_id}/vn_asset_{extra_item['id']}.png"
    saved = outputs / relative_path
    saved.parent.mkdir(parents=True)
    saved.write_bytes(PNG_BYTES)
    records = {
        "checkpoint_version": 1, "pack_id": pack.id, "slot_ids": [extra_slot["id"]],
        "item_ids": [extra_item["id"]], "generated_file_ids": [], "asset_formats": {str(extra_item["id"]): "png"},
    }
    repo.update_import_journal(import_id, {"status": "processing", "created_records": records}, owner_user_id=42)
    repo.update_item_storage(
        extra_item["id"], generated_file_id=51, storage_ref=relative_path, mime_type="image/png",
        width=1, height=1, bytes=len(PNG_BYTES),
    )
    pack_url = f"/api/v1/vn/vn-assets/packs/{pack.id}"
    assert [item["id"] for item in client.get(pack_url + "/items").json()] == [original_item["id"]]
    assert [slot["id"] for slot in client.get(pack_url + "/slots").json()] == [original_slot["id"]]
    staged_review = client.patch(pack_url + f"/items/{extra_item['id']}/review", json={"review_status": "approved"})
    assert staged_review.status_code == 404

    class Storage:
        record = {
            "id": 51, "user_id": 42, "source_feature": "vn_assets", "source_ref": f"vn_asset_item:{extra_item['id']}",
            "storage_path": relative_path,
        } if registered else None

        async def discard_vn_import_file(self, **kwargs: Any) -> int | None:
            assert kwargs["user_id"] == 42
            assert kwargs["source_ref"] == f"vn_asset_item:{extra_item['id']}"
            assert kwargs["storage_path"] == relative_path
            record = self.record
            if record is None:
                kwargs["remove_bytes"]()
                return None
            kwargs["require_owned_file"](record)
            self.record = None
            raise OSError("cleanup stopped after unregistering")

    storage = Storage()

    async def get_storage() -> Storage:
        return storage

    monkeypatch.setattr(vn_assets, "get_storage_service", get_storage)
    monkeypatch.setattr(vn_assets.DatabasePaths, "get_user_outputs_dir", staticmethod(lambda _: outputs))
    endpoint = f"/api/v1/vn/vn-assets/portability/imports/{import_id}/cleanup"
    current_user_id["value"] = 43
    assert client.post(endpoint).status_code == 404
    current_user_id["value"] = 42
    assert client.post(endpoint).status_code == 409
    assert client.delete(f"/api/v1/vn/vn-assets/import/previews/{preview_id}").status_code == 409
    assert staged.is_file() and saved.is_file()

    fake_jobs.jobs[int(job_id)]["status"] = "failed"
    if registered:
        fake_jobs.jobs.pop(int(job_id))
    job_status = client.get(f"/api/v1/vn/vn-assets/portability/imports/{job_id}").json()
    assert job_status["error_code"] == "import_interrupted"
    if registered:
        with pytest.raises(OSError, match="cleanup stopped"):
            client.post(endpoint)
        assert json.loads(repo.get_import_journal(import_id, owner_user_id=42)["cleanup_status_json"])["status"] == "discarding"
        assert saved.is_file()
    assert client.post(endpoint).json()["cleanup_status"]["status"] == "discarded"
    assert client.post(endpoint).status_code == 200
    assert storage.record is None
    assert not saved.exists() and not staged.exists()
    assert repo.get_item(original_item["id"])["generated_file_id"] == 15
    assert repo.get_item(extra_item["id"]) is None and repo.get_slot(extra_slot["id"]) is None


def test_preview_upload_reaps_expired_archives_only_for_current_owner(
    client: TestClient, repo: VNAssetPacksRepository, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "previews"
    monkeypatch.setenv("VN_PACK_IMPORT_PREVIEW_STAGING_ROOT", str(staging))
    owned_archive = staging / "42" / "expired.tldw-vnpack"
    other_archive = staging / "43" / "expired.tldw-vnpack"
    for owner_id, archive in ((42, owned_archive), (43, other_archive)):
        archive.parent.mkdir(parents=True)
        archive.write_bytes(b"staged backup")
        repo.create_import_preview(
            owner_user_id=owner_id, job_id=f"job-{owner_id}", status="completed", archive_path=str(archive),
            expires_at="2020-01-01T00:00:00+00:00",
        )

    upload = _write_preview_archive(tmp_path / "new.tldw-vnpack")
    with upload.open("rb") as archive:
        response = client.post(
            "/api/v1/vn/vn-assets/import/previews",
            data={"idempotency_key": "new-upload"},
            files={"archive": (upload.name, archive, "application/zip")},
        )

    assert response.status_code == 202
    assert not owned_archive.exists()
    assert other_archive.is_file()
    expired_preview = repo.get_import_preview(1, owner_user_id=42)
    assert expired_preview["status"] == "expired" and expired_preview["archive_path"] == ""


@pytest.mark.asyncio
async def test_generic_storage_delete_waits_for_import_publication(
    repo: VNAssetPacksRepository, monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview = repo.create_import_preview(owner_user_id=42, job_id="preview", status="completed", archive_path="backup")
    journal = repo.create_import_journal(
        owner_user_id=42, preview_id=preview["id"], job_id="restore", status="processing", stage="creating_items",
        trust_mode="trusted_restore", target_mode="create_new",
    )
    record = {"id": 51, "user_id": 42, "source_feature": "vn_assets", "storage_path": f"vn_assets/imports/{journal['id']}/vn_asset_12.png"}
    deleted: list[int] = []

    class Storage:
        async def get_generated_files_repo(self) -> "Storage":
            return self

        async def get_file_by_id(self, _file_id: int) -> dict[str, Any]:
            return record

        async def unregister_generated_file(self, file_id: int, *, hard_delete: bool) -> bool:
            deleted.append(file_id)
            return True

    async def get_db(user_id: int) -> CharactersRAGDB:
        assert user_id == 42
        return repo.db

    monkeypatch.setattr(storage_user_files, "get_chacha_db_for_user_id", get_db)
    user = User(id=42, username="owner")
    storage = Storage()
    with pytest.raises(HTTPException, match="vn_import_file_still_staged") as single:
        await storage_user_files.delete_file(51, user=user, hard_delete=True, service=storage)
    assert single.value.status_code == 409
    with pytest.raises(HTTPException, match="vn_import_file_still_staged") as bulk:
        await storage_user_files.bulk_delete_files(BulkDeleteRequest(file_ids=[51]), user=user, service=storage)
    assert bulk.value.status_code == 409
    assert not deleted

    repo.update_import_journal(
        journal["id"], {"status": "completed", "completed_at": "2026-11-26T10:00:00+00:00"}, owner_user_id=42,
    )
    assert (await storage_user_files.delete_file(51, user=user, hard_delete=True, service=storage)).success
    assert deleted == [51]


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
