"""HTTP contracts with real Media storage/PDF children and explicit auth claims."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.endpoints import media
from tldw_Server_API.app.core.AuthNZ.permissions import MEDIA_CREATE, MEDIA_READ
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

DRAFT = {
    "title": "Rowan Observatory",
    "pages": [
        {"heading": "Telescope", "text": "The telescope is named Selene."},
        {"heading": "Tours", "text": "Tours start on Friday at six in the evening."},
    ],
}


@pytest.fixture
def document_client(tmp_path):
    db = MediaDatabase(db_path=tmp_path / "media.db", client_id="1")
    principal = AuthPrincipal(kind="user", user_id=1, permissions=[MEDIA_CREATE, MEDIA_READ])
    app = FastAPI()
    app.include_router(media.router, prefix="/api/v1/media")
    app.dependency_overrides[auth_deps.get_auth_principal] = lambda: principal
    app.dependency_overrides[auth_deps.get_request_user] = lambda: SimpleNamespace(id=1)
    app.dependency_overrides[get_media_db_for_user] = lambda: db

    # Rate storage is outside this HTTP/PDF boundary test; permission/scope checks remain real.
    async def rate_storage_boundary():
        return None

    for route in app.routes:
        if isinstance(route, APIRoute):
            for dependency in route.dependant.dependencies:
                if getattr(dependency.call, "_tldw_rate_limit_resource", None):
                    app.dependency_overrides[dependency.call] = rate_storage_boundary
    try:
        with TestClient(app) as client:
            yield client, db, principal
    finally:
        db.close_connection()


def test_render_route_returns_a_real_complete_pdf(document_client):
    client, _, _ = document_client
    response = client.post("/api/v1/media/documents/render", json=DRAFT, headers={"X-TLDW-Expected-User-ID": "1"})
    assert (
        response.status_code == 200
        and response.headers["content-type"].startswith("application/pdf")
        and response.content.startswith(b"%PDF-")
        and response.content.rstrip().endswith(b"%%EOF")
    )  # nosec B101 # The authenticated route must return complete PDF bytes, not a false success.


def saved_document(document_client, tmp_path, monkeypatch, location="2"):
    import asyncio

    from tldw_Server_API.app.api.v1.endpoints.media.document_annotations import create_annotation
    from tldw_Server_API.app.api.v1.schemas.document_annotations import AnnotationCreate
    from tldw_Server_API.app.core import Storage
    from tldw_Server_API.app.core.Storage.filesystem_storage import FileSystemStorage
    from tldw_Server_API.tests.Media.test_document_pdf_worker import run_worker

    client, db, _ = document_client
    source = run_worker({"operation": "render", "draft": DRAFT}).stdout
    media_id, _, _ = db.add_media_with_keywords(title="Rowan PDF", media_type="pdf", content="Selene", owner_user_id=1)
    storage = FileSystemStorage(tmp_path / "storage")
    key = asyncio.run(storage.store("1", media_id, "rowan.pdf", source, mime_type="application/pdf"))
    db.insert_media_file(
        media_id, "original", key, original_filename="rowan.pdf", mime_type="application/pdf", file_size=len(source)
    )
    asyncio.run(
        create_annotation(
            AnnotationCreate(location=location, text="Friday at six", note="Confirm the tour.", color="blue"),
            media_id=media_id,
            db=db,
            current_user=SimpleNamespace(id=1),
        )
    )
    monkeypatch.setattr(Storage, "get_storage_backend", lambda: storage)
    return client, media_id, storage, key, source


def test_combined_export_uses_saved_annotations_and_preserves_original(document_client, tmp_path, monkeypatch):
    import asyncio

    from tldw_Server_API.tests.Media.test_document_pdf_worker import read_pdf

    client, media_id, storage, key, source = saved_document(document_client, tmp_path, monkeypatch)
    response = client.get(f"/api/v1/media/{media_id}/annotated-file", headers={"X-TLDW-Expected-User-ID": "1"})
    assert response.status_code == 200 and response.headers["content-type"].startswith("application/pdf")  # nosec B101 # The authorized export includes the canonical saved annotation.
    with read_pdf(response.content) as output:
        page = output[1]
        comments = list(page.annots())
        assert (
            len(output) == 2
            and len(comments) == 1
            and "Confirm the tour." in comments[0].info["content"]
            and "Friday at six" in comments[0].info["content"]
        )  # nosec B101 # Export contains the saved note and exact selection on the right page.
    original = asyncio.run(storage.retrieve(key))
    try:
        assert original.read() == source  # nosec B101 # Annotated export never overwrites the preserved source PDF.
    finally:
        original.close()


def test_export_projection_bounds_every_input_field_before_materialization(document_client):
    from datetime import datetime, timezone

    from tldw_Server_API.app.core.DB_Management.media_db.repositories.document_workspace_repository import (
        DocumentWorkspaceRepository,
    )

    _, db, _ = document_client
    repo = DocumentWorkspaceRepository(db)
    now = datetime.now(timezone.utc).isoformat()
    repo.create_annotation(
        annotation_id="bounded-export",
        media_id=1,
        user_id="1",
        location="1" * 100000,
        text="x" * 100000,
        color="blue",
        note="y" * 100000,
        annotation_type="highlight",
        chapter_title=None,
        percentage=None,
        created_at=now,
        updated_at=now,
    )
    row = repo.list_annotations_for_export(media_id=1, user_id="1")[0]
    assert [len(row[field]) for field in ["location", "text", "note"]] == [65, 6001, 6001]  # nosec B101 # Export projections bound database materialization before rejection.


@pytest.mark.parametrize(
    "permissions,expected_user,status", [([], "1", 403), ([MEDIA_READ], "1", 403), ([MEDIA_CREATE], "2", 412)]
)
def test_render_permission_and_owner_refusals_precede_pdf_work(
    document_client, monkeypatch, permissions, expected_user, status
):
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.api.v1.endpoints.media import document_rendering

    client, _, principal = document_client
    principal.permissions = permissions
    render = AsyncMock()
    monkeypatch.setattr(document_rendering, "render_document_pdf", render)
    response = client.post(
        "/api/v1/media/documents/render", json=DRAFT, headers={"X-TLDW-Expected-User-ID": expected_user}
    )
    assert response.status_code == status and render.call_count == 0  # nosec B101 # Actual permission and expected-user guards deny before CPU work.


@pytest.mark.parametrize(
    "error,status", [(TimeoutError("private backend details"), 504), (ValueError("private backend details"), 422)]
)
def test_render_processing_errors_are_fixed_and_private(document_client, monkeypatch, error, status):
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.api.v1.endpoints.media import document_rendering

    client, _, _ = document_client
    monkeypatch.setattr(document_rendering, "render_document_pdf", AsyncMock(side_effect=error))
    response = client.post("/api/v1/media/documents/render", json=DRAFT)
    assert response.status_code == status and "private backend details" not in response.text  # nosec B101 # Internal diagnostics never enter public error details.


def test_render_openapi_advertises_only_pdf_success(document_client):
    client, _, _ = document_client
    schema = client.get("/openapi.json").json()
    media_types = schema["paths"]["/api/v1/media/documents/render"]["post"]["responses"]["200"]["content"]
    assert set(media_types) == {"application/pdf"}  # nosec B101 # Generated clients must not treat PDF bytes as JSON.


def test_export_refuses_an_overlong_location_sentinel(document_client, tmp_path, monkeypatch):
    client, media_id, _, _, _ = saved_document(
        document_client, tmp_path, monkeypatch, location="0" * 64 + "1" + "invalid-tail"
    )
    response = client.get(f"/api/v1/media/{media_id}/annotated-file")
    assert response.status_code == 422  # nosec B101 # A truncated page label is never reinterpreted as a canonical page number.


@pytest.mark.parametrize("field", ["location", "text", "note"])
def test_export_refuses_embedded_nul_instead_of_losing_saved_content(document_client, tmp_path, monkeypatch, field):
    from datetime import datetime, timezone

    from tldw_Server_API.app.core.DB_Management.media_db.repositories.document_workspace_repository import (
        DocumentWorkspaceRepository,
    )

    client, media_id, _, _, _ = saved_document(
        document_client, tmp_path, monkeypatch, location="2\0unread-tail" if field == "location" else "2"
    )
    _, db, _ = document_client
    repo = DocumentWorkspaceRepository(db)
    if field != "location":
        annotation = repo.list_annotations(media_id=media_id, user_id="1")[0]
        repo.update_annotation(
            annotation_id=annotation["id"],
            media_id=media_id,
            user_id="1",
            updated_at=datetime.now(timezone.utc).isoformat(),
            **{field: "Visible prefix\0unread-tail"},
        )
    response = client.get(f"/api/v1/media/{media_id}/annotated-file")
    assert response.status_code == 422  # nosec B101 # SQLite NUL prefixes cannot masquerade as complete canonical annotation data.


from tldw_Server_API.tests.Media.test_document_annotation_sync_replay import replay_store as replay_store


def test_export_projection_preserves_exact_canonical_values_on_both_backends(replay_store):
    from datetime import datetime, timezone

    from tldw_Server_API.app.core.DB_Management.media_db.repositories.document_workspace_repository import (
        DocumentWorkspaceRepository,
    )

    db, media_id, _, _ = replay_store
    repo = DocumentWorkspaceRepository(db)
    now = datetime.now(timezone.utc).isoformat()
    repo.create_annotation(
        annotation_id="export-projection",
        media_id=media_id,
        user_id="401",
        location="2",
        text="Selene — telescope",
        color="blue",
        note="Friday at six",
        annotation_type="highlight",
        chapter_title=None,
        percentage=None,
        created_at=now,
        updated_at=now,
    )
    rows = repo.list_annotations_for_export(media_id=media_id, user_id="401")
    foreign = repo.list_annotations_for_export(media_id=media_id, user_id="402")
    assert (rows, foreign) == (
        [
            {
                "invalid_text": 0,
                "location": "2",
                "text": "Selene — telescope",
                "color": "blue",
                "note": "Friday at six",
                "annotation_type": "highlight",
            }
        ],
        [],
    )  # nosec B101 # Restricted SQL retains exact saved fields and owner isolation.
