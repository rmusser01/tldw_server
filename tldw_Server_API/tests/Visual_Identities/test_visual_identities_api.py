from __future__ import annotations

import hashlib
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
from tldw_Server_API.app.api.v1.router_groups.core import iter_core_router_specs
from tldw_Server_API.app.api.v1.router_registry import register_router_specs
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.VisualIdentity_DB import VisualIdentityRepository
from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository
from tldw_Server_API.app.core.Visual_Identities.storage import (
    resolve_visual_identity_asset_path,
)
from tldw_Server_API.app.core.VN_Assets.storage import (
    SOURCE_FEATURE_VN_ASSETS,
    vn_asset_source_ref,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def chacha_db(tmp_path: Path) -> Generator[CharactersRAGDB, None, None]:
    db_path = tmp_path / "ChaChaNotes.db"
    database = CharactersRAGDB(str(db_path), client_id="visual-identity-api-test-client")
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


@pytest.fixture
def outputs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "outputs"
    root.mkdir()
    monkeypatch.setattr(
        DatabasePaths,
        "get_user_outputs_dir",
        staticmethod(lambda owner_user_id: root / str(owner_user_id)),
    )
    return root


class FakeJobManager:
    def __init__(self) -> None:
        self.created_jobs: list[dict[str, Any]] = []
        self.jobs_by_idempotency_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    def create_job(self, **kwargs: Any) -> dict[str, Any]:
        idempotency_key = str(kwargs.get("idempotency_key") or "")
        key = (
            str(kwargs.get("domain") or ""),
            str(kwargs.get("queue") or ""),
            str(kwargs.get("job_type") or ""),
            idempotency_key,
        )
        if idempotency_key and key in self.jobs_by_idempotency_key:
            return self.jobs_by_idempotency_key[key]
        self.created_jobs.append(kwargs)
        job_id = f"zip-job-{len(self.created_jobs)}"
        job = {"id": job_id, "job_id": job_id}
        if idempotency_key:
            self.jobs_by_idempotency_key[key] = job
        return job


class FakeGeneratedFilesRepo:
    def __init__(self, records: dict[int, dict[str, Any]]) -> None:
        self.records = records
        self.accessed_ids: list[int] = []

    async def get_file_by_id(self, file_id: int) -> dict[str, Any] | None:
        self.accessed_ids.append(file_id)
        return self.records.get(file_id)


def _user(user_id: int) -> User:
    return User(
        id=user_id,
        username=f"user-{user_id}",
        email=f"user-{user_id}@example.test",
        roles=["user"],
        permissions=[],
    )


def _client(
    chacha_db: CharactersRAGDB,
    *,
    user_id: int = 1,
    job_manager: FakeJobManager | None = None,
    files_repo: FakeGeneratedFilesRepo | None = None,
) -> TestClient:
    app = FastAPI()
    app.include_router(
        visual_identities.router,
        prefix="/api/v1/visual-identities",
        tags=["visual-identities"],
    )
    app.dependency_overrides[visual_identities.get_request_user] = lambda: _user(user_id)
    app.dependency_overrides[visual_identities.get_chacha_db_for_user] = lambda: chacha_db
    app.dependency_overrides[get_db_pool] = lambda: object()
    if job_manager is not None:
        app.dependency_overrides[visual_identities._job_manager] = lambda: job_manager
    if files_repo is not None:
        app.dependency_overrides[visual_identities._generated_files_repo] = lambda: files_repo
    return TestClient(app)


def _seed_character(db: CharactersRAGDB, *, name: str = "API Bound Character") -> int:
    with db.transaction() as conn:
        cursor = conn.execute(
            "INSERT INTO character_cards (name, client_id, version) VALUES (?, ?, 1)",
            (name, db.client_id),
        )
    return int(cursor.lastrowid)


def _seed_ready_draft(repo: VisualIdentityRepository, *, owner_user_id: int) -> dict:
    draft = repo.create_draft(
        owner_user_id=owner_user_id,
        title="Reviewable Expression Pack",
        source_kind="zip",
        source_filename="reviewable.zip",
        status="ready_for_review",
        default_expression_key="neutral",
    )
    repo.create_asset(
        owner_user_id=owner_user_id,
        draft_id=draft["id"],
        expression_key="neutral",
        source_filename="neutral.png",
        storage_relpath="packs/draft-1/neutral/neutral.png",
        content_type="image/png",
        bytes=12,
        sha256="abc123",
        width=64,
        height=64,
        preview_relpath="previews/draft-1/neutral/neutral.png",
    )
    return draft


def _create_versioned_pack(
    repo: VisualIdentityRepository,
    *,
    owner_user_id: int,
    title: str,
    assets: tuple[str, ...],
    default_expression_key: str = "neutral",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    pack = repo.create_pack(
        owner_user_id=owner_user_id,
        title=title,
        default_expression_key=default_expression_key,
    )
    version = repo.create_pack_version(
        owner_user_id=owner_user_id,
        pack_id=pack["id"],
        version_number=1,
        manifest={},
        default_expression_key=default_expression_key,
    )
    version_assets: dict[str, dict[str, Any]] = {}
    for expression_key in assets:
        version_assets[expression_key] = repo.create_asset(
            owner_user_id=owner_user_id,
            pack_id=pack["id"],
            pack_version_id=version["id"],
            expression_key=expression_key,
            source_filename=f"{expression_key}.png",
            storage_relpath=f"packs/{pack['id']}/{expression_key}.png",
            content_type="image/png",
            bytes=12,
            sha256=f"sha256-{pack['id']}-{expression_key}",
            width=64,
            height=64,
        )
    return pack, version, version_assets


def _png_bytes(*, color: str = "purple") -> bytes:
    buffer = BytesIO()
    Image.new("RGBA", (8, 8), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _assert_stored_draft_asset(
    repo: VisualIdentityRepository,
    *,
    draft_id: int,
    owner_user_id: int,
    storage_root: Path,
    expected_bytes: bytes,
) -> None:
    """Validate one recorded asset through the public storage resolver."""

    assets = repo.list_draft_assets(draft_id, owner_user_id=owner_user_id)
    assert len(assets) == 1
    asset = assets[0]
    resolved = resolve_visual_identity_asset_path(
        owner_user_id=owner_user_id,
        relpath=asset["storage_relpath"],
    )
    assert resolved.is_relative_to((storage_root / str(owner_user_id)).resolve())
    assert resolved.is_file()
    stored_bytes = resolved.read_bytes()
    assert stored_bytes == expected_bytes
    assert asset["bytes"] == len(stored_bytes)
    assert asset["sha256"] == hashlib.sha256(stored_bytes).hexdigest()


def _zip_bytes_with_png() -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        archive.writestr("neutral.png", _png_bytes())
    return buffer.getvalue()


def test_capabilities_endpoint_reports_supported_formats(chacha_db: CharactersRAGDB) -> None:
    response = _client(chacha_db).get("/api/v1/visual-identities/capabilities")

    assert response.status_code == 200
    payload = response.json()
    assert payload["upload_max_bytes"] > 0
    assert payload["archive_max_bytes"] > payload["upload_max_bytes"]
    assert {"image/png", "image/jpeg", "image/webp", "image/gif"}.issubset(
        set(payload["supported_mime_types"])
    )
    assert isinstance(payload["avif_enabled"], bool)


def test_router_registration_exposes_visual_identity_capabilities(
    chacha_db: CharactersRAGDB,
) -> None:
    specs = [spec for spec in iter_core_router_specs() if spec.route_key == "visual-identities"]
    assert len(specs) == 1
    assert specs[0].prefix == "/api/v1/visual-identities"
    assert specs[0].tags == ("visual-identities",)

    app = FastAPI()
    app.dependency_overrides[visual_identities.get_request_user] = lambda: _user(1)
    app.dependency_overrides[visual_identities.get_chacha_db_for_user] = lambda: chacha_db
    app.dependency_overrides[get_db_pool] = lambda: object()
    register_router_specs(app, specs)

    response = TestClient(app).get("/api/v1/visual-identities/capabilities")

    assert response.status_code == 200


def test_activate_draft_with_character_binds_by_default(
    chacha_db: CharactersRAGDB,
    repo: VisualIdentityRepository,
) -> None:
    character_id = _seed_character(chacha_db)
    draft = _seed_ready_draft(repo, owner_user_id=1)

    response = _client(chacha_db).post(
        f"/api/v1/visual-identities/drafts/{draft['id']}/activate",
        json={"actor_kind": "character", "actor_id": character_id},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "activated"
    assert payload["pack_id"] is not None
    assert payload["pack_version_id"] is not None
    assert payload["binding_id"] is not None
    binding = repo.get_binding_for_actor(
        owner_user_id=1,
        actor_kind="character",
        actor_id=character_id,
    )
    assert binding is not None
    assert binding["pack_id"] == payload["pack_id"]
    assert binding["active_version_id"] == payload["pack_version_id"]


def test_resolve_bound_asset_returns_content_url_and_null_direct_fallback(
    chacha_db: CharactersRAGDB,
    repo: VisualIdentityRepository,
) -> None:
    character_id = _seed_character(chacha_db)
    draft = _seed_ready_draft(repo, owner_user_id=1)
    client = _client(chacha_db)
    activation = client.post(
        f"/api/v1/visual-identities/drafts/{draft['id']}/activate",
        json={"actor_kind": "character", "actor_id": character_id},
    ).json()

    response = client.get(
        "/api/v1/visual-identities/bindings/resolve",
        params={
            "actor_kind": "character",
            "actor_id": character_id,
            "expression_key": "neutral",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["asset_id"] in activation["asset_ids"]
    assert payload["asset_url"] == (
        f"/api/v1/visual-identities/packs/{activation['pack_id']}"
        f"/assets/{payload['asset_id']}/content"
    )
    assert payload["preview_url"] == (
        f"/api/v1/visual-identities/packs/{activation['pack_id']}"
        f"/assets/{payload['asset_id']}/preview"
    )
    assert payload["fallback_reason"] is None


def test_preview_response_is_inline(
    chacha_db: CharactersRAGDB,
    repo: VisualIdentityRepository,
    storage_root: Path,
) -> None:
    character_id = _seed_character(chacha_db)
    draft = _seed_ready_draft(repo, owner_user_id=1)
    preview_path = storage_root / "1" / "previews" / "draft-1" / "neutral" / "neutral.png"
    preview_path.parent.mkdir(parents=True)
    preview_path.write_bytes(_png_bytes())
    client = _client(chacha_db)
    activation = client.post(
        f"/api/v1/visual-identities/drafts/{draft['id']}/activate",
        json={"actor_kind": "character", "actor_id": character_id},
    ).json()

    response = client.get(
        f"/api/v1/visual-identities/packs/{activation['pack_id']}"
        f"/assets/{activation['asset_ids'][0]}/preview"
    )

    assert response.status_code == 200
    assert response.headers["content-disposition"].startswith("inline;")


def test_resolve_endpoint_preserves_existing_query_contract(
    chacha_db: CharactersRAGDB,
    repo: VisualIdentityRepository,
) -> None:
    character_id = _seed_character(chacha_db)
    draft = _seed_ready_draft(repo, owner_user_id=1)
    client = _client(chacha_db)
    client.post(
        f"/api/v1/visual-identities/drafts/{draft['id']}/activate",
        json={"actor_kind": "character", "actor_id": character_id},
    )

    response = client.get(
        "/api/v1/visual-identities/bindings/resolve",
        params={
            "actor_kind": "character",
            "actor_id": character_id,
            "expression_key": "neutral",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["role_id"] is None
    assert payload["role_label"] is None
    assert payload["resolution_source"] == "binding"


def test_resolve_endpoint_accepts_role_override_fields(
    chacha_db: CharactersRAGDB,
    repo: VisualIdentityRepository,
) -> None:
    character_id = _seed_character(chacha_db, name="API Override Character")
    draft = _seed_ready_draft(repo, owner_user_id=1)
    client = _client(chacha_db)
    client.post(
        f"/api/v1/visual-identities/drafts/{draft['id']}/activate",
        json={"actor_kind": "character", "actor_id": character_id},
    )
    override_pack, override_version, override_assets = _create_versioned_pack(
        repo,
        owner_user_id=1,
        title="API Override Pack",
        assets=("happy",),
        default_expression_key="happy",
    )

    response = client.get(
        "/api/v1/visual-identities/bindings/resolve",
        params={
            "actor_kind": "character",
            "actor_id": character_id,
            "expression_key": "happy",
            "role_id": "hero",
            "role_label": "Hero",
            "override_pack_id": override_pack["id"],
            "override_pack_version_id": override_version["id"],
            "allow_override_fallback": "false",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["pack_id"] == override_pack["id"]
    assert payload["pack_version_id"] == override_version["id"]
    assert payload["asset_id"] == override_assets["happy"]["id"]
    assert payload["role_id"] == "hero"
    assert payload["role_label"] == "Hero"
    assert payload["resolution_source"] == "override"
    assert payload["fallback_reason"] is None


def test_resolve_endpoint_reports_override_expression_missing(
    chacha_db: CharactersRAGDB,
    repo: VisualIdentityRepository,
) -> None:
    character_id = _seed_character(chacha_db, name="API Missing Override Character")
    binding_pack, binding_version, _ = _create_versioned_pack(
        repo,
        owner_user_id=1,
        title="API Missing Binding Pack",
        assets=("sad",),
        default_expression_key="missing-default",
    )
    repo.upsert_binding(
        owner_user_id=1,
        actor_kind="character",
        actor_id=character_id,
        pack_id=binding_pack["id"],
        active_version_id=binding_version["id"],
    )
    override_pack, override_version, _ = _create_versioned_pack(
        repo,
        owner_user_id=1,
        title="API Missing Override Pack",
        assets=("angry",),
        default_expression_key="missing-default",
    )

    response = _client(chacha_db).get(
        "/api/v1/visual-identities/bindings/resolve",
        params={
            "actor_kind": "character",
            "actor_id": character_id,
            "expression_key": "sad",
            "override_pack_id": override_pack["id"],
            "override_pack_version_id": override_version["id"],
        },
    )

    assert response.status_code in {409, 422}
    assert response.json()["detail"] == "override_expression_missing"


def test_resolve_endpoint_reports_invalid_actor_without_placeholder_masking(
    chacha_db: CharactersRAGDB,
) -> None:
    response = _client(chacha_db).get(
        "/api/v1/visual-identities/bindings/resolve",
        params={
            "actor_kind": "character",
            "actor_id": 999999,
            "expression_key": "neutral",
        },
    )

    assert response.status_code in {404, 422}
    assert response.json()["detail"] in {
        "visual_identity_actor_kind_invalid",
        "visual_identity_character_not_found",
        "visual_identity_persona_not_found",
    }


def test_resolve_endpoint_reports_cross_user_override_pack(
    chacha_db: CharactersRAGDB,
    repo: VisualIdentityRepository,
) -> None:
    character_id = _seed_character(chacha_db, name="API Cross User Override Character")
    binding_pack, binding_version, _ = _create_versioned_pack(
        repo,
        owner_user_id=1,
        title="API Cross User Binding Pack",
        assets=("happy",),
    )
    repo.upsert_binding(
        owner_user_id=1,
        actor_kind="character",
        actor_id=character_id,
        pack_id=binding_pack["id"],
        active_version_id=binding_version["id"],
    )
    override_pack, override_version, _ = _create_versioned_pack(
        repo,
        owner_user_id=2,
        title="API Cross User Override Pack",
        assets=("happy",),
    )

    response = _client(chacha_db).get(
        "/api/v1/visual-identities/bindings/resolve",
        params={
            "actor_kind": "character",
            "actor_id": character_id,
            "expression_key": "happy",
            "override_pack_id": override_pack["id"],
            "override_pack_version_id": override_version["id"],
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] in {"pack_not_found", "pack_not_owned"}


def test_resolve_endpoint_reports_pack_version_mismatch(
    chacha_db: CharactersRAGDB,
    repo: VisualIdentityRepository,
) -> None:
    character_id = _seed_character(chacha_db, name="API Version Mismatch Character")
    binding_pack, binding_version, _ = _create_versioned_pack(
        repo,
        owner_user_id=1,
        title="API Version Mismatch Binding Pack",
        assets=("happy",),
    )
    repo.upsert_binding(
        owner_user_id=1,
        actor_kind="character",
        actor_id=character_id,
        pack_id=binding_pack["id"],
        active_version_id=binding_version["id"],
    )
    override_pack, _, _ = _create_versioned_pack(
        repo,
        owner_user_id=1,
        title="API Version Mismatch Override Pack",
        assets=("sad",),
    )
    _, other_version, _ = _create_versioned_pack(
        repo,
        owner_user_id=1,
        title="API Version Mismatch Other Pack",
        assets=("happy",),
    )

    response = _client(chacha_db).get(
        "/api/v1/visual-identities/bindings/resolve",
        params={
            "actor_kind": "character",
            "actor_id": character_id,
            "expression_key": "happy",
            "override_pack_id": override_pack["id"],
            "override_pack_version_id": other_version["id"],
        },
    )

    assert response.status_code in {409, 422}
    assert response.json()["detail"] == "pack_version_mismatch"


def test_zip_import_replays_idempotency_and_persists_job_id(
    chacha_db: CharactersRAGDB,
    storage_root: Path,
) -> None:
    job_manager = FakeJobManager()
    client = _client(chacha_db, job_manager=job_manager)
    archive_bytes = _zip_bytes_with_png()

    first = client.post(
        "/api/v1/visual-identities/imports/zip",
        data={"title": "Zip Expressions", "idempotency_key": "zip-import-1"},
        files={"archive": ("expressions.zip", archive_bytes, "application/zip")},
    )
    second = client.post(
        "/api/v1/visual-identities/imports/zip",
        data={"title": "Zip Expressions", "idempotency_key": "zip-import-1"},
        files={"archive": ("expressions.zip", archive_bytes, "application/zip")},
    )

    assert first.status_code == 202
    assert second.status_code == 202
    assert second.json() == first.json()
    assert len(job_manager.created_jobs) == 1
    draft_response = client.get(
        f"/api/v1/visual-identities/drafts/{first.json()['draft_id']}"
    )
    assert draft_response.status_code == 200
    assert draft_response.json()["import_job_id"] == first.json()["import_job_id"]
    assert (storage_root / "1" / "imports" / str(first.json()["draft_id"]) / "expressions.zip").is_file()


def test_zip_import_same_client_key_is_user_scoped_for_jobs(
    chacha_db: CharactersRAGDB,
) -> None:
    job_manager = FakeJobManager()
    archive_bytes = _zip_bytes_with_png()
    first = _client(chacha_db, user_id=1, job_manager=job_manager).post(
        "/api/v1/visual-identities/imports/zip",
        data={"title": "Zip Expressions", "idempotency_key": "shared-key"},
        files={"archive": ("expressions.zip", archive_bytes, "application/zip")},
    )
    second = _client(chacha_db, user_id=2, job_manager=job_manager).post(
        "/api/v1/visual-identities/imports/zip",
        data={"title": "Zip Expressions", "idempotency_key": "shared-key"},
        files={"archive": ("expressions.zip", archive_bytes, "application/zip")},
    )

    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["import_job_id"] != second.json()["import_job_id"]
    assert len(job_manager.created_jobs) == 2


def test_generated_file_asset_import_replays_idempotency(
    chacha_db: CharactersRAGDB,
    repo: VisualIdentityRepository,
    storage_root: Path,
    outputs_root: Path,
) -> None:
    pack = repo.create_pack(owner_user_id=1, title="Generated Expressions")
    source_path = outputs_root / "1" / "image_gen" / "neutral.png"
    source_path.parent.mkdir(parents=True)
    source_bytes = _png_bytes(color="blue")
    source_path.write_bytes(source_bytes)
    files_repo = FakeGeneratedFilesRepo(
        {
            77: {
                "id": 77,
                "user_id": 1,
                "is_deleted": False,
                "file_category": "image",
                "source_feature": "image_gen",
                "storage_path": "image_gen/neutral.png",
                "mime_type": "image/png",
                "original_filename": "neutral.png",
            }
        }
    )
    client = _client(chacha_db, files_repo=files_repo)
    request = {
        "generated_file_id": 77,
        "expression_key": "happy",
        "idempotency_key": "generated-asset-1",
    }

    first = client.post(
        f"/api/v1/visual-identities/packs/{pack['id']}/assets/from-generated-file",
        json=request,
    )
    second = client.post(
        f"/api/v1/visual-identities/packs/{pack['id']}/assets/from-generated-file",
        json=request,
    )
    conflict = client.post(
        f"/api/v1/visual-identities/packs/{pack['id']}/assets/from-generated-file",
        json={**request, "expression_key": "sad"},
    )
    file_conflict = client.post(
        f"/api/v1/visual-identities/packs/{pack['id']}/assets/from-generated-file",
        json={**request, "generated_file_id": 78},
    )

    assert first.status_code == 201
    assert second.status_code == 201
    assert second.json() == first.json()
    assert conflict.status_code == 409
    assert file_conflict.status_code == 409
    assert files_repo.accessed_ids == [77]
    _assert_stored_draft_asset(
        repo,
        draft_id=first.json()["draft_id"],
        owner_user_id=1,
        storage_root=storage_root,
        expected_bytes=source_bytes,
    )


def test_generated_file_asset_import_returns_source_context(
    chacha_db: CharactersRAGDB,
    repo: VisualIdentityRepository,
    storage_root: Path,
    outputs_root: Path,
) -> None:
    pack = repo.create_pack(owner_user_id=1, title="Generated Expressions")
    source_path = outputs_root / "1" / "image_gen" / "happy.png"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(_png_bytes(color="blue"))
    files_repo = FakeGeneratedFilesRepo(
        {
            77: {
                "id": 77,
                "user_id": 1,
                "is_deleted": False,
                "file_category": "image",
                "source_feature": "image_gen",
                "storage_path": "image_gen/happy.png",
                "mime_type": "image/png",
                "original_filename": "happy.png",
            }
        }
    )

    response = _client(chacha_db, files_repo=files_repo).post(
        f"/api/v1/visual-identities/packs/{pack['id']}/assets/from-generated-file",
        json={
            "generated_file_id": 77,
            "expression_key": "happy",
            "source_context": {"source_feature": "image_gen", "generated_file_id": 77},
            "idempotency_key": "generated-context-1",
        },
    )

    assert response.status_code == 201
    assert response.json()["source_context"] == {
        "generated_file_id": 77,
        "source_feature": "image_gen",
    }


def test_generated_file_asset_import_idempotency_uses_canonical_source_context(
    chacha_db: CharactersRAGDB,
    repo: VisualIdentityRepository,
    storage_root: Path,
    outputs_root: Path,
) -> None:
    pack = repo.create_pack(owner_user_id=1, title="Generated Expressions")
    source_path = outputs_root / "1" / "image_gen" / "neutral.png"
    source_path.parent.mkdir(parents=True)
    source_bytes = _png_bytes(color="blue")
    source_path.write_bytes(source_bytes)
    files_repo = FakeGeneratedFilesRepo(
        {
            77: {
                "id": 77,
                "user_id": 1,
                "is_deleted": False,
                "file_category": "image",
                "source_feature": "image_gen",
                "storage_path": "image_gen/neutral.png",
                "mime_type": "image/png",
                "original_filename": "neutral.png",
            }
        }
    )
    client = _client(chacha_db, files_repo=files_repo)
    request = {
        "generated_file_id": 77,
        "expression_key": "happy",
        "idempotency_key": "generated-context-canonical-1",
        "source_context": {
            "source_feature": "image_gen",
            "generated_file_id": 77,
            "metadata": {"rank": 1, "stage": "final"},
        },
    }

    first = client.post(
        f"/api/v1/visual-identities/packs/{pack['id']}/assets/from-generated-file",
        json=request,
    )
    replay = client.post(
        f"/api/v1/visual-identities/packs/{pack['id']}/assets/from-generated-file",
        json={
            **request,
            "source_context": {
                "metadata": {"stage": "final", "rank": 1},
                "generated_file_id": 77,
                "source_feature": "image_gen",
            },
        },
    )
    conflict = client.post(
        f"/api/v1/visual-identities/packs/{pack['id']}/assets/from-generated-file",
        json={
            **request,
            "source_context": {
                "source_feature": "image_gen",
                "generated_file_id": 77,
                "metadata": {"rank": 2, "stage": "final"},
            },
        },
    )

    assert first.status_code == 201
    assert replay.status_code == 201
    assert replay.json() == first.json()
    assert conflict.status_code == 409
    assert files_repo.accessed_ids == [77]
    _assert_stored_draft_asset(
        repo,
        draft_id=first.json()["draft_id"],
        owner_user_id=1,
        storage_root=storage_root,
        expected_bytes=source_bytes,
    )


def test_generated_file_asset_import_records_vn_context_and_rejects_item_mismatch(
    chacha_db: CharactersRAGDB,
    repo: VisualIdentityRepository,
    storage_root: Path,
    outputs_root: Path,
) -> None:
    pack = repo.create_pack(owner_user_id=1, title="VN Generated Expressions")
    source_path = outputs_root / "1" / SOURCE_FEATURE_VN_ASSETS / "maya_happy.png"
    source_path.parent.mkdir(parents=True)
    source_bytes = _png_bytes(color="blue")
    source_path.write_bytes(source_bytes)

    character_id = _seed_character(chacha_db, name="Maya")
    vn_repo = VNAssetPacksRepository.initialized(chacha_db)
    vn_pack = vn_repo.create_pack(
        owner_user_id=1,
        primary_character_id=character_id,
        title="Maya Sprite Pack",
    )
    vn_slot = vn_repo.create_slot(
        pack_id=int(vn_pack["id"]),
        asset_type="sprite",
        slot_key="happy",
        labels={"expression": "happy"},
    )
    vn_item = vn_repo.create_item(
        pack_id=int(vn_pack["id"]),
        slot_id=int(vn_slot["id"]),
        generated_file_id=77,
        storage_ref=f"{SOURCE_FEATURE_VN_ASSETS}/maya_happy.png",
        mime_type="image/png",
        width=8,
        height=8,
        bytes=len(source_bytes),
    )
    item_id = int(vn_item["id"])
    slot_id = int(vn_slot["id"])
    vn_pack_id = int(vn_pack["id"])
    files_repo = FakeGeneratedFilesRepo(
        {
            77: {
                "id": 77,
                "user_id": 1,
                "is_deleted": False,
                "file_category": "image",
                "source_feature": SOURCE_FEATURE_VN_ASSETS,
                "source_ref": vn_asset_source_ref(item_id),
                "storage_path": f"{SOURCE_FEATURE_VN_ASSETS}/maya_happy.png",
                "mime_type": "image/png",
                "original_filename": "maya_happy.png",
            }
        }
    )
    client = _client(chacha_db, files_repo=files_repo)

    response = client.post(
        f"/api/v1/visual-identities/packs/{pack['id']}/assets/from-generated-file",
        json={
            "generated_file_id": 77,
            "expression_key": "happy",
            "source_feature": SOURCE_FEATURE_VN_ASSETS,
            "source_context": {
                "vn_item_id": item_id,
                "vn_pack_id": vn_pack_id,
                "vn_slot_id": slot_id,
                "vn_slot_key": "happy",
                "vn_slot_label": "Happy",
                "vn_asset_type": "sprite",
            },
            "idempotency_key": "vn-generated-context-ok",
        },
    )

    assert response.status_code == 201
    assert response.json()["source_context"] == {
        "filename": "maya_happy.png",
        "generated_file_id": 77,
        "mime_type": "image/png",
        "source_feature": SOURCE_FEATURE_VN_ASSETS,
        "source_ref": vn_asset_source_ref(item_id),
        "vn_asset_type": "sprite",
        "vn_item_id": item_id,
        "vn_pack_id": vn_pack_id,
        "vn_slot_id": slot_id,
        "vn_slot_key": "happy",
        "vn_slot_label": "Happy",
    }
    assert _visual_identity_asset_count(chacha_db, owner_user_id=1) == 1

    item_mismatch_key = "vn-generated-context-item-mismatch"
    item_mismatch = client.post(
        f"/api/v1/visual-identities/packs/{pack['id']}/assets/from-generated-file",
        json={
            "generated_file_id": 77,
            "expression_key": "sad",
            "source_feature": SOURCE_FEATURE_VN_ASSETS,
            "source_context": {"vn_item_id": item_id + 1},
            "idempotency_key": item_mismatch_key,
        },
    )

    assert item_mismatch.status_code == 422
    assert item_mismatch.json()["detail"] == "vn_generated_file_context_mismatch"
    assert _visual_identity_asset_count(chacha_db, owner_user_id=1) == 1
    assert _visual_identity_idempotency_count(
        chacha_db,
        owner_user_id=1,
        idempotency_key=item_mismatch_key,
    ) == 0

    mismatched_contexts = [
        {"vn_item_id": item_id, "vn_pack_id": vn_pack_id + 1},
        {"vn_item_id": item_id, "vn_slot_id": slot_id + 1},
        {"vn_item_id": item_id, "vn_slot_key": "sad"},
        {"vn_item_id": item_id, "vn_asset_type": "background"},
    ]
    for index, source_context in enumerate(mismatched_contexts, start=1):
        idempotency_key = f"vn-generated-context-structural-mismatch-{index}"
        mismatch = client.post(
            f"/api/v1/visual-identities/packs/{pack['id']}/assets/from-generated-file",
            json={
                "generated_file_id": 77,
                "expression_key": "sad",
                "source_feature": SOURCE_FEATURE_VN_ASSETS,
                "source_context": source_context,
                "idempotency_key": idempotency_key,
            },
        )

        assert mismatch.status_code == 422
        assert mismatch.json()["detail"] == "vn_generated_file_context_mismatch"
        assert _visual_identity_asset_count(chacha_db, owner_user_id=1) == 1
        assert _visual_identity_idempotency_count(
            chacha_db,
            owner_user_id=1,
            idempotency_key=idempotency_key,
        ) == 0
    _assert_stored_draft_asset(
        repo,
        draft_id=response.json()["draft_id"],
        owner_user_id=1,
        storage_root=storage_root,
        expected_bytes=source_bytes,
    )


def test_vn_generated_file_import_derives_context_and_replays_without_live_source(
    chacha_db: CharactersRAGDB,
    repo: VisualIdentityRepository,
    storage_root: Path,
    outputs_root: Path,
) -> None:
    pack = repo.create_pack(owner_user_id=1, title="VN Generated Expressions")
    source_path = outputs_root / "1" / SOURCE_FEATURE_VN_ASSETS / "maya_happy.png"
    source_path.parent.mkdir(parents=True)
    source_bytes = _png_bytes(color="blue")
    source_path.write_bytes(source_bytes)

    character_id = _seed_character(chacha_db, name="Maya")
    vn_repo = VNAssetPacksRepository.initialized(chacha_db)
    vn_pack = vn_repo.create_pack(
        owner_user_id=1,
        primary_character_id=character_id,
        title="Maya Sprite Pack",
    )
    vn_slot = vn_repo.create_slot(
        pack_id=int(vn_pack["id"]),
        asset_type="sprite",
        slot_key="happy",
        labels={"expression": "happy"},
    )
    vn_item = vn_repo.create_item(
        pack_id=int(vn_pack["id"]),
        slot_id=int(vn_slot["id"]),
        generated_file_id=77,
        storage_ref=f"{SOURCE_FEATURE_VN_ASSETS}/maya_happy.png",
        mime_type="image/png",
        width=8,
        height=8,
        bytes=len(source_bytes),
    )
    item_id = int(vn_item["id"])
    files_repo = FakeGeneratedFilesRepo(
        {
            77: {
                "id": 77,
                "user_id": 1,
                "is_deleted": False,
                "file_category": "image",
                "source_feature": SOURCE_FEATURE_VN_ASSETS,
                "source_ref": vn_asset_source_ref(item_id),
                "storage_path": f"{SOURCE_FEATURE_VN_ASSETS}/maya_happy.png",
                "mime_type": "image/png",
                "original_filename": "maya_happy.png",
            }
        }
    )
    client = _client(chacha_db, files_repo=files_repo)
    request = {
        "generated_file_id": 77,
        "expression_key": "happy",
        "source_feature": SOURCE_FEATURE_VN_ASSETS,
        "idempotency_key": "vn-generated-context-default-replay",
    }

    first = client.post(
        f"/api/v1/visual-identities/packs/{pack['id']}/assets/from-generated-file",
        json=request,
    )
    files_repo.records.pop(77)
    replay = client.post(
        f"/api/v1/visual-identities/packs/{pack['id']}/assets/from-generated-file",
        json=request,
    )

    assert first.status_code == 201
    assert first.json()["source_context"] == {
        "filename": "maya_happy.png",
        "generated_file_id": 77,
        "mime_type": "image/png",
        "source_feature": SOURCE_FEATURE_VN_ASSETS,
        "source_ref": vn_asset_source_ref(item_id),
        "vn_asset_type": "sprite",
        "vn_item_id": item_id,
        "vn_pack_id": int(vn_pack["id"]),
        "vn_slot_id": int(vn_slot["id"]),
        "vn_slot_key": "happy",
    }
    assert replay.status_code == 201
    assert replay.json() == first.json()
    assert files_repo.accessed_ids == [77]
    assert _visual_identity_asset_count(chacha_db, owner_user_id=1) == 1
    _assert_stored_draft_asset(
        repo,
        draft_id=first.json()["draft_id"],
        owner_user_id=1,
        storage_root=storage_root,
        expected_bytes=source_bytes,
    )


def test_generated_file_asset_import_rejects_invalid_file_without_creating_draft(
    chacha_db: CharactersRAGDB,
    repo: VisualIdentityRepository,
    outputs_root: Path,
) -> None:
    pack = repo.create_pack(owner_user_id=1, title="Generated Expressions")
    corrupt_path = outputs_root / "1" / "image_gen" / "corrupt.png"
    corrupt_path.parent.mkdir(parents=True)
    corrupt_path.write_bytes(b"not-a-real-png")
    files_repo = FakeGeneratedFilesRepo(
        {
            77: {
                "id": 77,
                "user_id": 2,
                "is_deleted": False,
                "file_category": "image",
                "source_feature": "image_gen",
                "storage_path": "image_gen/neutral.png",
                "mime_type": "image/png",
                "original_filename": "neutral.png",
            },
            78: {
                "id": 78,
                "user_id": 1,
                "is_deleted": False,
                "file_category": "image",
                "source_feature": "image_gen",
                "storage_path": "image_gen/corrupt.png",
                "mime_type": "image/png",
                "original_filename": "corrupt.png",
            }
        }
    )
    client = _client(chacha_db, files_repo=files_repo)

    foreign_response = client.post(
        f"/api/v1/visual-identities/packs/{pack['id']}/assets/from-generated-file",
        json={
            "generated_file_id": 77,
            "expression_key": "happy",
            "idempotency_key": "generated-invalid-1",
        },
    )
    corrupt_response = client.post(
        f"/api/v1/visual-identities/packs/{pack['id']}/assets/from-generated-file",
        json={
            "generated_file_id": 78,
            "expression_key": "happy",
            "idempotency_key": "generated-invalid-2",
        },
    )
    draft_count = chacha_db.execute_query(
        "SELECT COUNT(*) FROM visual_identity_pack_drafts WHERE owner_user_id = ?",
        (1,),
    ).fetchone()[0]

    assert foreign_response.status_code == 404
    assert corrupt_response.status_code == 422
    assert draft_count == 0


def _visual_identity_asset_count(db: CharactersRAGDB, *, owner_user_id: int) -> int:
    row = db.execute_query(
        "SELECT COUNT(*) FROM visual_identity_assets WHERE owner_user_id = ?",
        (owner_user_id,),
    ).fetchone()
    return int(row[0])


def _visual_identity_idempotency_count(
    db: CharactersRAGDB,
    *,
    owner_user_id: int,
    idempotency_key: str,
) -> int:
    row = db.execute_query(
        """
        SELECT COUNT(*) FROM visual_identity_idempotency
        WHERE owner_user_id = ? AND idempotency_key = ?
        """,
        (owner_user_id, idempotency_key),
    ).fetchone()
    return int(row[0])


def test_asset_content_requires_owner(
    chacha_db: CharactersRAGDB,
    repo: VisualIdentityRepository,
    storage_root: Path,
) -> None:
    pack = repo.create_pack(owner_user_id=1, title="Owned Asset Pack")
    version = repo.create_pack_version(
        pack_id=pack["id"],
        owner_user_id=1,
        version_number=1,
        manifest={"assets": []},
    )
    active_pack = repo.set_active_version(
        pack_id=pack["id"],
        owner_user_id=1,
        pack_version_id=version["id"],
    )
    relpath = "packs/owned/neutral/asset.png"
    asset_path = storage_root / "1" / relpath
    asset_path.parent.mkdir(parents=True)
    asset_path.write_bytes(b"owned-png-bytes")
    asset = repo.create_asset(
        owner_user_id=1,
        pack_id=active_pack["id"],
        pack_version_id=version["id"],
        expression_key="neutral",
        source_filename="neutral.png",
        storage_relpath=relpath,
        content_type="image/png",
        bytes=15,
        sha256="def456",
        width=64,
        height=64,
    )

    owner_response = _client(chacha_db, user_id=1).get(
        f"/api/v1/visual-identities/packs/{active_pack['id']}/assets/{asset['id']}/content"
    )
    foreign_response = _client(chacha_db, user_id=2).get(
        f"/api/v1/visual-identities/packs/{active_pack['id']}/assets/{asset['id']}/content"
    )

    assert owner_response.status_code == 200
    assert owner_response.content == b"owned-png-bytes"
    assert owner_response.headers["content-type"] == "image/png"
    assert owner_response.headers["cache-control"] == "private, max-age=31536000, immutable"
    assert foreign_response.status_code == 404


def test_invalid_actor_kind_returns_422(chacha_db: CharactersRAGDB) -> None:
    response = _client(chacha_db).get(
        "/api/v1/visual-identities/bindings/resolve",
        params={
            "actor_kind": "scene",
            "actor_id": "7",
            "expression_key": "neutral",
        },
    )

    assert response.status_code == 422
