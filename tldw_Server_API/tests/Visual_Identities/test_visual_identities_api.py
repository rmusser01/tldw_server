from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_db_pool
from tldw_Server_API.app.api.v1.endpoints import visual_identities
from tldw_Server_API.app.api.v1.router_groups.core import iter_core_router_specs
from tldw_Server_API.app.api.v1.router_registry import register_router_specs
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VisualIdentity_DB import VisualIdentityRepository
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

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


def _user(user_id: int) -> User:
    return User(
        id=user_id,
        username=f"user-{user_id}",
        email=f"user-{user_id}@example.test",
        roles=["user"],
        permissions=[],
    )


def _client(chacha_db: CharactersRAGDB, *, user_id: int = 1) -> TestClient:
    app = FastAPI()
    app.include_router(
        visual_identities.router,
        prefix="/api/v1/visual-identities",
        tags=["visual-identities"],
    )
    app.dependency_overrides[visual_identities.get_request_user] = lambda: _user(user_id)
    app.dependency_overrides[visual_identities.get_chacha_db_for_user] = lambda: chacha_db
    app.dependency_overrides[get_db_pool] = lambda: object()
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
    )
    return draft


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
    assert owner_response.headers["cache-control"] == "public, max-age=31536000, immutable"
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
