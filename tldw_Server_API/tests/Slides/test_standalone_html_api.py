from __future__ import annotations

import asyncio
import hashlib
import json

import pytest
from fastapi import FastAPI, HTTPException, Query
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import (
    get_collections_db_for_user,
)
from tldw_Server_API.app.api.v1.API_Deps.Slides_DB_Deps import (
    get_slides_db_for_user,
)
from tldw_Server_API.app.api.v1.endpoints.slides import (
    _load_version_payload,
    _slides_lifespan,
)
from tldw_Server_API.app.api.v1.endpoints.slides import (
    router as slides_router,
)
from tldw_Server_API.app.api.v1.schemas.slides_schemas import (
    ExportFormat,
    PresentationPatchRequest,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.Slides.slides_db import SlidesDatabase
from tldw_Server_API.app.core.Slides.standalone_html_validator import (
    validate_standalone_html,
)

_ACCEPT = "X-Slides-Accept-Content-Kinds"
_BOTH = {_ACCEPT: "structured_slides,standalone_html"}


class _InlineValidationPool:
    def __init__(self, db: SlidesDatabase) -> None:
        self.db = db
        self.calls: list[str | bytes] = []
        self.closed = False

    async def validate(self, document: str | bytes):
        assert not self.db.get_connection().in_transaction
        self.calls.append(document)
        return validate_standalone_html(document)

    async def close(self) -> None:
        self.closed = True


def _assert_operation_error(response, *, operation: str, content_kind: str) -> None:
    assert response.status_code == 409
    assert response.json() == {
        "detail": "operation_not_supported_for_content_kind",
        "operation": operation,
        "content_kind": content_kind,
    }


def _document(*, title: str = "HTML Deck", text: str = "Visible HTML text") -> str:
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{title}</title><style>.slide{{color:#111}}</style></head>"
        f'<body><section class="slide"><h1>{text}</h1>'
        '<aside class="notes">Hidden note</aside></section>'
        "<script>document.addEventListener('keydown', () => {});</script>"
        "</body></html>"
    )


def _provenance_json() -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "source_kind": "prompt",
            "source_ref": None,
            "source_snapshot_hmac_sha256": "a" * 64,
            "digest_key_id": "slides-generation-v1",
            "source_bytes": 10,
            "provider": "openai",
            "model": "test-model",
            "adapter_id": "openai_official_chat_v1",
            "endpoint_identity": "https://api.openai.com:443/v1/chat/completions",
            "prompt_sha256": "b" * 64,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _create_html(db: SlidesDatabase, *, presentation_id: str = "html"):
    document = _document()
    derived = validate_standalone_html(document)
    return db.create_presentation(
        presentation_id=presentation_id,
        title=derived.title,
        description=None,
        theme="black",
        marp_theme=None,
        settings=None,
        studio_data=None,
        slides="[]",
        slides_text=derived.indexable_text,
        source_type="prompt",
        source_ref=None,
        source_query=None,
        custom_css=None,
        content_kind="standalone_html",
        html_document=document,
        html_sha256=derived.html_sha256,
        html_bytes=derived.html_bytes,
        html_slide_count=derived.slide_count,
        generation_job_uuid=f"job-{presentation_id}",
        generation_provenance_json=_provenance_json(),
    )


def _create_structured(db: SlidesDatabase, *, presentation_id: str = "structured"):
    slides = [
        {
            "order": 0,
            "layout": "title",
            "title": "Structured Deck",
            "content": "",
            "speaker_notes": None,
            "metadata": {},
        }
    ]
    return db.create_presentation(
        presentation_id=presentation_id,
        title="Structured Deck",
        description=None,
        theme="black",
        marp_theme=None,
        settings=None,
        studio_data=None,
        slides=json.dumps(slides),
        slides_text="Structured Deck",
        source_type="manual",
        source_ref=None,
        source_query=None,
        custom_css=None,
    )


class _Collections:
    list_calls = 0

    def get_output_artifact(self, output_id: int):
        return {"id": output_id}

    def resolve_output_storage_path(self, path_value):
        return str(path_value)

    def list_output_artifacts(self, **_kwargs):
        type(self).list_calls += 1
        return [], 0


@pytest.fixture()
def html_client(tmp_path):
    _Collections.list_calls = 0
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="1")
    structured = _create_structured(db)
    html = _create_html(db)
    app = FastAPI()
    validation_pool = _InlineValidationPool(db)
    app.state.standalone_html_validation_pool = validation_pool
    app.include_router(slides_router, prefix="/api/v1", tags=["slides"])

    async def _override_user():
        return User(
            id=1,
            username="tester",
            email=None,
            is_active=True,
            is_admin=True,
        )

    async def _override_principal(request=None):
        principal = AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject="test-user",
            token_type="single_user",  # nosec B106 - test principal type
            jti=None,
            roles=["admin"],
            permissions=[
                "media.create",
                "media.read",
                "media.update",
                "media.delete",
            ],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )
        if request is not None:
            request.state.auth = AuthContext(
                principal=principal,
                ip=None,
                user_agent=None,
                request_id=None,
            )
        return principal

    async def _override_db():
        yield db

    async def _override_collections():
        return _Collections()

    app.dependency_overrides[get_request_user] = _override_user
    app.dependency_overrides[get_auth_principal] = _override_principal
    app.dependency_overrides[get_slides_db_for_user] = _override_db
    app.dependency_overrides[get_collections_db_for_user] = _override_collections

    with TestClient(app) as client:
        yield client, db, structured, html

    assert validation_pool.closed
    assert getattr(app.state, "standalone_html_validation_pool", None) is None
    assert getattr(app.state, "standalone_html_validation_pool_lock", None) is None
    app.dependency_overrides.clear()
    db.close_connection()


@pytest.mark.parametrize(
    "value",
    ["", " ", ",", "structured_slides,", "bad token", "future_kind"],
)
def test_malformed_or_unknown_only_negotiation_is_fixed_400(html_client, value):
    client, _db, _structured, _html = html_client

    response = client.get(
        "/api/v1/slides/presentations",
        headers={_ACCEPT: value},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid_content_kind_header"
    assert _ACCEPT.lower() in response.headers["Vary"].lower()


def test_slides_lifespan_without_html_pool_shuts_down_cleanly():
    app = FastAPI()
    app.include_router(slides_router, prefix="/api/v1")

    with TestClient(app):
        assert getattr(app.state, "standalone_html_validation_pool", None) is None

    assert getattr(app.state, "standalone_html_validation_pool", None) is None
    assert getattr(app.state, "standalone_html_validation_pool_lock", None) is None


def test_slides_lifespan_defers_worker_owned_pool_cleanup_until_composite_shutdown(tmp_path):
    app = FastAPI()
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="1")
    validation_pool = _InlineValidationPool(db)
    app.state.standalone_html_validation_pool = validation_pool
    app.state.standalone_html_validation_pool_worker_owned = True
    app.include_router(slides_router, prefix="/api/v1")

    with TestClient(app):
        pass

    assert validation_pool.closed is False
    assert app.state.standalone_html_validation_pool is validation_pool
    asyncio.run(validation_pool.close())
    db.close_connection()


@pytest.mark.asyncio
async def test_worker_owned_slides_lifespan_does_not_suppress_endpoint_errors():
    app = FastAPI()
    app.state.standalone_html_validation_pool_worker_owned = True

    with pytest.raises(RuntimeError, match="endpoint failed"):
        async with _slides_lifespan(app):
            raise RuntimeError("endpoint failed")


def test_negotiated_validation_errors_match_fastapi_body_and_add_vary(html_client):
    client, _db, structured, _html = html_client
    baseline_app = FastAPI()

    @baseline_app.get("/presentations")
    async def baseline_list(limit: int = Query(50, ge=1, le=200)):
        return {"limit": limit}

    @baseline_app.patch("/presentations/{presentation_id}")
    async def baseline_patch(
        presentation_id: str,
        request: PresentationPatchRequest,
    ):
        return {"id": presentation_id, "request": request.model_dump()}

    with TestClient(baseline_app) as baseline:
        baseline_query = baseline.get("/presentations?limit=0")
        baseline_body = baseline.patch("/presentations/structured", json={"title": []})

    query = client.get("/api/v1/slides/presentations?limit=0")
    body = client.patch(
        f"/api/v1/slides/presentations/{structured.id}",
        json={"title": []},
        headers={"If-Match": 'W/"v1"'},
    )

    assert query.status_code == baseline_query.status_code == 422
    assert body.status_code == baseline_body.status_code == 422
    assert query.json() == baseline_query.json()
    assert body.json() == baseline_body.json()
    assert _ACCEPT.lower() in query.headers["Vary"].lower()
    assert _ACCEPT.lower() in body.headers["Vary"].lower()


def _html_mutation_requests(html_id: str):
    return [
        (
            "PUT",
            f"/api/v1/slides/presentations/{html_id}/html-source",
            {
                "content": _document().encode("utf-8"),
                "headers": {"Content-Type": "application/octet-stream"},
            },
        ),
        ("PUT", f"/api/v1/slides/presentations/{html_id}", {"json": {"title": "No"}}),
        ("PATCH", f"/api/v1/slides/presentations/{html_id}", {"json": {"title": "No"}}),
        ("POST", f"/api/v1/slides/presentations/{html_id}/reorder", {"json": {"order": [0]}}),
        ("DELETE", f"/api/v1/slides/presentations/{html_id}", {}),
        ("POST", f"/api/v1/slides/presentations/{html_id}/restore", {}),
        ("POST", f"/api/v1/slides/presentations/{html_id}/versions/1/restore", {}),
    ]


def test_negotiated_mutations_reject_malformed_kind_before_missing_if_match(html_client):
    client, _db, _structured, html = html_client

    for method, path, kwargs in _html_mutation_requests(html.id):
        kwargs["headers"] = {**kwargs.get("headers", {}), _ACCEPT: "bad token"}
        response = client.request(method, path, **kwargs)

        assert response.status_code == 400, (method, path, response.text)
        assert response.json()["detail"] == "invalid_content_kind_header"
        assert _ACCEPT.lower() in response.headers["Vary"].lower()


def test_negotiated_mutations_reject_unaccepted_html_before_missing_if_match(html_client):
    client, _db, _structured, html = html_client

    for method, path, kwargs in _html_mutation_requests(html.id):
        response = client.request(method, path, **kwargs)

        assert response.status_code == 406, (method, path, response.text)
        assert response.json()["detail"] == "content_kind_not_accepted"
        assert _ACCEPT.lower() in response.headers["Vary"].lower()


def test_list_negotiation_filters_before_pagination_and_returns_source_free_unions(
    html_client,
):
    client, _db, _structured, _html = html_client

    legacy = client.get("/api/v1/slides/presentations?limit=1&offset=0")
    html_only = client.get(
        "/api/v1/slides/presentations?limit=1&offset=0",
        headers={_ACCEPT: " standalone_html ,standalone_html "},
    )
    structured_only = client.get(
        "/api/v1/slides/presentations?limit=1&offset=0",
        headers={_ACCEPT: "structured_slides,future_kind"},
    )
    dual = client.get(
        "/api/v1/slides/presentations?limit=1&offset=0",
        headers={_ACCEPT: "structured_slides, future_kind, standalone_html"},
    )

    assert legacy.status_code == structured_only.status_code == html_only.status_code == dual.status_code == 200
    assert legacy.json()["total"] == 1
    assert set(legacy.json()["presentations"][0]) == {
        "id",
        "title",
        "description",
        "theme",
        "created_at",
        "last_modified",
        "deleted",
        "version",
    }
    assert structured_only.json()["presentations"] == legacy.json()["presentations"]
    assert html_only.json()["total"] == 1
    html_summary = html_only.json()["presentations"][0]
    assert html_summary["content_kind"] == "standalone_html"
    assert html_summary["html_slide_count"] == 1
    assert html_summary["html_bytes"] == len(_document().encode("utf-8"))
    assert html_summary["provenance"] == {
        "source_kind": "prompt",
        "provider": "openai",
        "model": "test-model",
    }
    assert "html_document" not in html_summary and "slides" not in html_summary
    assert dual.json()["total"] == 2
    assert len(dual.json()["presentations"]) == 1
    for response in (legacy, structured_only, html_only, dual):
        assert _ACCEPT.lower() in {item.strip().lower() for item in response.headers["Vary"].split(",")}


def test_structured_version_list_preserves_legacy_title_and_deleted_values(html_client):
    client, db, structured, _html = html_client
    renamed = db.update_presentation(
        presentation_id=structured.id,
        update_fields={"title": "Renamed Structured Deck"},
        expected_version=structured.version,
    )
    deleted = db.soft_delete_presentation(structured.id, renamed.version)

    response = client.get(f"/api/v1/slides/presentations/{structured.id}/versions")

    assert response.status_code == 200, response.text
    assert [(version["version"], version["title"], version["deleted"]) for version in response.json()["versions"]] == [
        (deleted.version, "Renamed Structured Deck", True),
        (renamed.version, "Renamed Structured Deck", False),
        (structured.version, "Structured Deck", False),
    ]


def test_targeted_html_requires_opt_in_before_source_projection(html_client):
    client, db, _structured, html = html_client
    statements: list[str] = []
    db.get_connection().set_trace_callback(statements.append)

    response = client.get(f"/api/v1/slides/presentations/{html.id}")

    assert response.status_code == 406
    assert response.json()["detail"] == "content_kind_not_accepted"
    assert _ACCEPT.lower() in response.headers["Vary"].lower()
    selected = "\n".join(
        statement for statement in statements if statement.lstrip().upper().startswith("SELECT")
    ).lower()
    assert "html_document" not in selected


def test_opted_in_html_detail_is_discriminated_json_with_strong_etag(html_client):
    client, _db, _structured, html = html_client

    response = client.get(f"/api/v1/slides/presentations/{html.id}", headers=_BOTH)

    assert response.status_code == 200
    payload = response.json()
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["ETag"] == '"v1"'
    assert payload["content_kind"] == "standalone_html"
    assert payload["html_document"] == _document()
    assert payload["html_sha256"] == hashlib.sha256(_document().encode("utf-8")).hexdigest()
    assert payload["html_slide_count"] == 1
    assert "slides" not in payload
    assert _ACCEPT.lower() in response.headers["Vary"].lower()


def test_generic_create_and_mutation_reject_standalone_kind(html_client):
    client, _db, structured, html = html_client

    create = client.post(
        "/api/v1/slides/presentations",
        json={
            "title": "No",
            "content_kind": "standalone_html",
            "html_document": _document(),
            "slides": [],
        },
    )
    html_patch = client.patch(
        f"/api/v1/slides/presentations/{html.id}",
        json={"title": "No"},
        headers={**_BOTH, "If-Match": '"v1"'},
    )
    wrong_accept = client.patch(
        f"/api/v1/slides/presentations/{structured.id}",
        json={"title": "No"},
        headers={_ACCEPT: "standalone_html", "If-Match": 'W/"v1"'},
    )

    assert create.status_code == 409
    assert create.json()["detail"] == "standalone_html_creation_requires_generation"
    _assert_operation_error(
        html_patch,
        operation="update",
        content_kind="standalone_html",
    )
    assert wrong_accept.status_code == 406
    assert wrong_accept.json()["detail"] == "content_kind_not_accepted"


def test_html_source_save_validates_derives_and_noops_with_strong_etag(html_client):
    client, _db, _structured, html = html_client
    validation_pool = client.app.state.standalone_html_validation_pool
    changed_document = _document(title="Renamed", text="New searchable content")

    changed = client.put(
        f"/api/v1/slides/presentations/{html.id}/html-source",
        content=changed_document.encode("utf-8"),
        headers={
            **_BOTH,
            "If-Match": '"v1"',
            "Content-Type": "application/octet-stream",
        },
    )
    same = client.put(
        f"/api/v1/slides/presentations/{html.id}/html-source",
        content=changed_document.encode("utf-8"),
        headers={
            **_BOTH,
            "If-Match": '"v2"',
            "Content-Type": "application/octet-stream",
        },
    )

    assert changed.status_code == 200, changed.text
    assert changed.headers["ETag"] == '"v2"'
    assert changed.json()["title"] == "Renamed"
    assert changed.json()["html_bytes"] == len(changed_document.encode("utf-8"))
    assert same.status_code == 200, same.text
    assert same.headers["ETag"] == '"v2"'
    assert same.json()["version"] == 2
    assert validation_pool.calls == [
        changed_document.encode("utf-8"),
        changed_document.encode("utf-8"),
    ]


def test_html_source_errors_preserve_negotiation_vary(html_client):
    client, _db, _structured, html = html_client
    path = f"/api/v1/slides/presentations/{html.id}/html-source"

    wrong_media = client.put(
        path,
        content=_document(),
        headers={**_BOTH, "If-Match": '"v1"', "Content-Type": "text/html"},
    )
    invalid_source = client.put(
        path,
        content=b"not a complete document",
        headers={
            **_BOTH,
            "If-Match": '"v1"',
            "Content-Type": "application/octet-stream",
        },
    )
    stale = client.put(
        path,
        content=_document(),
        headers={
            **_BOTH,
            "If-Match": '"v0"',
            "Content-Type": "application/octet-stream",
        },
    )

    assert wrong_media.status_code == 415
    assert invalid_source.status_code == 422
    assert stale.status_code == 412
    for response in (wrong_media, invalid_source, stale):
        assert _ACCEPT.lower() in response.headers["Vary"].lower()


def test_html_version_list_and_delete_are_source_free(html_client):
    client, db, _structured, html = html_client
    statements: list[str] = []
    db.get_connection().set_trace_callback(statements.append)

    versions = client.get(f"/api/v1/slides/presentations/{html.id}/versions", headers=_BOTH)
    deleted = client.delete(
        f"/api/v1/slides/presentations/{html.id}",
        headers={**_BOTH, "If-Match": '"v1"'},
    )

    assert versions.status_code == 200, versions.text
    assert versions.json()["total"] == 1
    assert "html_document" not in json.dumps(versions.json())
    assert deleted.status_code == 200, deleted.text
    assert set(deleted.json()) == {"id", "content_kind", "deleted_at"}
    assert deleted.json()["content_kind"] == "standalone_html"
    selected = "\n".join(
        statement for statement in statements if statement.lstrip().upper().startswith("SELECT")
    ).lower()
    assert "payload_json" not in selected
    assert "html_document" not in selected


def test_html_reveal_and_render_reject_before_source_or_dispatch(html_client):
    client, db, _structured, html = html_client
    statements: list[str] = []
    db.get_connection().set_trace_callback(statements.append)

    export = client.get(
        f"/api/v1/slides/presentations/{html.id}/export?format=revealjs",
        headers=_BOTH,
    )
    render = client.post(
        f"/api/v1/slides/presentations/{html.id}/render-jobs",
        json={"format": "mp4"},
        headers={**_BOTH, "If-Match": '"v1"'},
    )

    _assert_operation_error(
        export,
        operation="export",
        content_kind="standalone_html",
    )
    _assert_operation_error(
        render,
        operation="render",
        content_kind="standalone_html",
    )
    selected = "\n".join(
        statement for statement in statements if statement.lstrip().upper().startswith("SELECT")
    ).lower()
    assert "html_document" not in selected


def test_html_is_an_explicit_export_format_but_transport_is_deferred():
    assert ExportFormat.HTML.value == "html"


def test_search_negotiation_filters_before_count_and_preserves_legacy_shape(
    html_client,
):
    client, _db, _structured, _html = html_client

    legacy = client.get("/api/v1/slides/presentations/search?q=Deck&limit=1")
    html_only = client.get(
        "/api/v1/slides/presentations/search?q=Deck&limit=1",
        headers={_ACCEPT: "standalone_html"},
    )
    dual = client.get(
        "/api/v1/slides/presentations/search?q=Deck&limit=1",
        headers=_BOTH,
    )

    assert legacy.status_code == html_only.status_code == dual.status_code == 200
    assert legacy.json()["total"] == 1
    assert "content_kind" not in legacy.json()["presentations"][0]
    assert html_only.json()["total"] == 1
    assert html_only.json()["presentations"][0]["content_kind"] == "standalone_html"
    assert dual.json()["total"] == 2
    assert len(dual.json()["presentations"]) == 1
    for response in (legacy, html_only, dual):
        assert _ACCEPT.lower() in response.headers["Vary"].lower()


def test_html_render_artifacts_rejects_before_collection_dispatch(html_client):
    client, _db, _structured, html = html_client

    response = client.get(
        f"/api/v1/slides/presentations/{html.id}/render-artifacts",
        headers=_BOTH,
    )

    _assert_operation_error(
        response,
        operation="render",
        content_kind="standalone_html",
    )
    assert _Collections.list_calls == 0


def test_explicit_null_standalone_fields_and_kind_are_rejected_by_presence(
    html_client,
):
    client, _db, structured, _html = html_client

    create = client.post(
        "/api/v1/slides/presentations",
        json={"title": "No", "slides": [], "html_document": None},
    )
    null_source = client.patch(
        f"/api/v1/slides/presentations/{structured.id}",
        json={"html_document": None},
        headers={**_BOTH, "If-Match": 'W/"v1"'},
    )
    null_kind = client.patch(
        f"/api/v1/slides/presentations/{structured.id}",
        json={"content_kind": None},
        headers={**_BOTH, "If-Match": 'W/"v1"'},
    )

    _assert_operation_error(
        create,
        operation="create",
        content_kind="structured_slides",
    )
    _assert_operation_error(
        null_source,
        operation="update",
        content_kind="structured_slides",
    )
    assert null_kind.status_code == 409
    assert null_kind.json()["detail"] == "content_kind_immutable"


@pytest.mark.parametrize(
    ("method", "payload"),
    [
        ("PUT", {"title": "No", "content_kind": "structured_slides"}),
        ("PATCH", {"content_kind": "structured_slides"}),
    ],
)
def test_html_kind_change_has_stable_immutable_error(html_client, method, payload):
    client, _db, _structured, html = html_client

    response = client.request(
        method,
        f"/api/v1/slides/presentations/{html.id}",
        json=payload,
        headers={**_BOTH, "If-Match": '"v1"'},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "content_kind_immutable"


def test_json_export_is_explicit_and_discriminated_for_opted_in_kinds(html_client):
    client, _db, structured, html = html_client
    validation_pool = client.app.state.standalone_html_validation_pool

    html_export = client.get(
        f"/api/v1/slides/presentations/{html.id}/export?format=json",
        headers=_BOTH,
    )
    structured_export = client.get(
        f"/api/v1/slides/presentations/{structured.id}/export?format=json",
        headers=_BOTH,
    )

    assert html_export.status_code == 200, html_export.text
    assert html_export.headers["content-type"].startswith("application/json")
    assert html_export.json()["content_kind"] == "standalone_html"
    assert html_export.json()["html_document"] == _document()
    assert "slides" not in html_export.json()
    assert structured_export.status_code == 200, structured_export.text
    assert structured_export.json()["content_kind"] == "structured_slides"
    assert "slides" in structured_export.json()
    assert validation_pool.calls == [_document()]


def test_json_export_rejects_corrupt_stored_derived_metadata_after_pool_validation(html_client):
    client, db, _structured, html = html_client
    with db.transaction(immediate=True) as conn:
        conn.execute(
            "UPDATE presentations SET slides_text = ? WHERE id = ?",
            ("forged", html.id),
        )

    response = client.get(
        f"/api/v1/slides/presentations/{html.id}/export?format=json",
        headers=_BOTH,
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "standalone_html_response_invalid"
    assert _ACCEPT.lower() in response.headers["Vary"].lower()
    assert client.app.state.standalone_html_validation_pool.calls == [_document()]


@pytest.mark.parametrize(
    ("column", "corrupt_value"),
    [
        pytest.param("slides", "not-json", id="slides-invalid-json"),
        pytest.param("slides", "{}", id="slides-nonlist"),
        pytest.param("slides", '[{"title":"forged"}]', id="slides-nonempty"),
        pytest.param("slides", "[" + " " * 8192 + "]", id="slides-oversize-empty"),
        pytest.param("generation_job_uuid", None, id="job-uuid-missing"),
        pytest.param("generation_job_uuid", "   ", id="job-uuid-blank"),
        pytest.param("generation_provenance_json", None, id="provenance-missing"),
        pytest.param(
            "generation_provenance_json",
            '{"private":"SECRET-MALFORMED-PROVENANCE"',
            id="provenance-malformed",
        ),
        pytest.param("generation_provenance_json", "[]", id="provenance-nonobject"),
        pytest.param("generation_provenance_json", "{}", id="provenance-empty-object"),
        pytest.param(
            "generation_provenance_json",
            "[" * 1100 + "0" + "]" * 1100,
            id="provenance-recursive",
        ),
        pytest.param(
            "generation_provenance_json",
            json.dumps({"private": "SECRET-OVERSIZE-PROVENANCE" + "x" * 4096}),
            id="provenance-oversize",
        ),
    ],
)
def test_json_export_rejects_corrupt_complete_row_invariant_before_pool_validation(
    html_client,
    column,
    corrupt_value,
):
    client, db, _structured, html = html_client
    with db.transaction(immediate=True) as conn:
        conn.execute(f"UPDATE presentations SET {column} = ? WHERE id = ?", (corrupt_value, html.id))
    validation_pool = client.app.state.standalone_html_validation_pool
    validation_pool.calls.clear()

    response = client.get(
        f"/api/v1/slides/presentations/{html.id}/export?format=json",
        headers=_BOTH,
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "standalone_html_response_invalid"}
    assert "SECRET-" not in response.text
    assert validation_pool.calls == []


def test_negotiated_downstream_http_error_keeps_vary(html_client):
    client, _db, structured, _html = html_client

    response = client.patch(
        f"/api/v1/slides/presentations/{structured.id}",
        json={},
        headers={"If-Match": 'W/"v1"'},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "no_fields_to_update"
    assert _ACCEPT.lower() in response.headers["Vary"].lower()


def test_html_version_restore_uses_interactive_pool_before_atomic_write(html_client):
    client, _db, _structured, html = html_client
    changed_document = _document(title="Changed", text="second")
    changed = client.put(
        f"/api/v1/slides/presentations/{html.id}/html-source",
        content=changed_document.encode("utf-8"),
        headers={
            **_BOTH,
            "If-Match": '"v1"',
            "Content-Type": "application/octet-stream",
        },
    )
    assert changed.status_code == 200, changed.text
    validation_pool = client.app.state.standalone_html_validation_pool
    validation_pool.calls.clear()

    restored = client.post(
        f"/api/v1/slides/presentations/{html.id}/versions/1/restore",
        headers={**_BOTH, "If-Match": '"v2"'},
    )

    assert restored.status_code == 200, restored.text
    assert restored.json()["html_document"] == _document()
    assert validation_pool.calls == [_document()]


def test_html_soft_restore_revalidates_cross_field_consistent_stored_source(html_client):
    client, db, _structured, html = html_client
    deleted = client.delete(
        f"/api/v1/slides/presentations/{html.id}",
        headers={**_BOTH, "If-Match": '"v1"'},
    )
    assert deleted.status_code == 200, deleted.text
    corrupt = _document().replace(
        "document.addEventListener('keydown', () => {});",
        "fetch('/private-data');",
    )
    encoded = corrupt.encode("utf-8")
    with db.transaction(immediate=True) as conn:
        conn.execute(
            """
            UPDATE presentations
            SET html_document = ?, html_sha256 = ?, html_bytes = ?
            WHERE id = ?
            """,
            (corrupt, hashlib.sha256(encoded).hexdigest(), len(encoded), html.id),
        )
    validation_pool = client.app.state.standalone_html_validation_pool
    validation_pool.calls.clear()

    restored = client.post(
        f"/api/v1/slides/presentations/{html.id}/restore",
        headers={**_BOTH, "If-Match": '"v2"'},
    )

    assert restored.status_code == 422, restored.text
    assert validation_pool.calls == [corrupt]
    current = db.get_presentation_by_id(html.id, include_deleted=True)
    assert current.deleted == 1
    assert current.version == 2


def test_html_soft_restore_source_response_sets_private_no_store_headers(html_client):
    client, _db, _structured, html = html_client
    deleted = client.delete(
        f"/api/v1/slides/presentations/{html.id}",
        headers={**_BOTH, "If-Match": '"v1"'},
    )
    assert deleted.status_code == 200, deleted.text

    restored = client.post(
        f"/api/v1/slides/presentations/{html.id}/restore",
        headers={**_BOTH, "If-Match": '"v2"'},
    )

    assert restored.status_code == 200, restored.text
    assert restored.json()["content_kind"] == "standalone_html"
    assert restored.json()["html_document"] == _document()
    assert restored.headers["Cache-Control"] == "private, no-store"
    assert restored.headers["X-Content-Type-Options"] == "nosniff"


def test_structured_soft_restore_keeps_legacy_path_without_pool_validation(html_client):
    client, _db, structured, _html = html_client
    validation_pool = client.app.state.standalone_html_validation_pool
    validation_pool.calls.clear()
    deleted = client.delete(
        f"/api/v1/slides/presentations/{structured.id}",
        headers={"If-Match": 'W/"v1"'},
    )
    assert deleted.status_code == 200, deleted.text

    restored = client.post(
        f"/api/v1/slides/presentations/{structured.id}/restore",
        headers={"If-Match": 'W/"v2"'},
    )

    assert restored.status_code == 200, restored.text
    assert restored.json()["deleted"] is False
    assert "Cache-Control" not in restored.headers
    assert "X-Content-Type-Options" not in restored.headers
    assert validation_pool.calls == []


def test_endpoint_snapshot_decoder_retains_no_source_exception_context():
    sentinel = "SECRET-ENDPOINT-SNAPSHOT"

    with pytest.raises(HTTPException) as exc_info:
        _load_version_payload('{"html_document":"' + sentinel)

    chain = [exc_info.value]
    while chain[-1].__cause__ is not None or chain[-1].__context__ is not None:
        chain.append(chain[-1].__cause__ or chain[-1].__context__)
    assert not any(isinstance(exc, json.JSONDecodeError) for exc in chain)
    assert sentinel not in " ".join(repr(exc) for exc in chain)


@pytest.mark.parametrize(
    ("method", "suffix", "extra_headers"),
    [
        pytest.param("GET", "", {}, id="get"),
        pytest.param("POST", "/restore", {"If-Match": '"v1"'}, id="restore"),
    ],
)
def test_recursive_snapshot_matches_fixed_malformed_payload_mapping(
    html_client,
    method,
    suffix,
    extra_headers,
):
    client, db, _structured, html = html_client
    path = f"/api/v1/slides/presentations/{html.id}/versions/1{suffix}"

    def _replace_snapshot(payload_json: str) -> None:
        with db.transaction(immediate=True) as conn:
            conn.execute(
                "UPDATE presentations_versions SET payload_json = ? " "WHERE presentation_id = ? AND version = 1",
                (payload_json, html.id),
            )

    _replace_snapshot('{"html_document":"malformed')
    baseline = client.request(method, path, headers={**_BOTH, **extra_headers})

    sentinel = "SECRET-RECURSIVE-SNAPSHOT"
    recursive = '{"html_document":"' + sentinel + '","nested":' + "[" * 1100 + "0" + "]" * 1100 + "}"
    assert len(recursive.encode("utf-8")) < 4096
    _replace_snapshot(recursive)

    response = client.request(method, path, headers={**_BOTH, **extra_headers})

    assert baseline.json() == response.json() == {"detail": "version_payload_invalid"}
    assert response.status_code == baseline.status_code
    assert sentinel not in response.text


def test_structured_restore_recomputes_legacy_slide_text_with_image_alt(html_client):
    client, db, _structured, _html = html_client
    created = client.post(
        "/api/v1/slides/presentations",
        json={
            "title": "Legacy image deck",
            "slides": [
                {
                    "order": 0,
                    "layout": "content",
                    "title": "Image slide",
                    "content": "Body",
                    "speaker_notes": "Narration",
                    "metadata": {"images": [{"asset_ref": "output:123", "alt": "Restored cover"}]},
                }
            ],
        },
    )
    assert created.status_code == 201, created.text
    presentation_id = created.json()["id"]

    with db.transaction(immediate=True) as conn:
        version_row = conn.execute(
            """
            SELECT payload_json FROM presentations_versions
            WHERE presentation_id = ? AND version = 1
            """,
            (presentation_id,),
        ).fetchone()
        payload = json.loads(version_row["payload_json"])
        payload.pop("slides_text", None)
        conn.execute(
            """
            UPDATE presentations_versions SET payload_json = ?
            WHERE presentation_id = ? AND version = 1
            """,
            (json.dumps(payload), presentation_id),
        )

    updated = client.patch(
        f"/api/v1/slides/presentations/{presentation_id}",
        json={"title": "Changed"},
        headers={"If-Match": created.headers["ETag"]},
    )
    assert updated.status_code == 200, updated.text

    restored = client.post(
        f"/api/v1/slides/presentations/{presentation_id}/versions/1/restore",
        headers={"If-Match": updated.headers["ETag"]},
    )

    assert restored.status_code == 200, restored.text
    assert "Restored cover" in db.get_presentation_by_id(presentation_id).slides_text


def test_structured_restore_keeps_legacy_missing_version_precedence(html_client):
    client, _db, structured, _html = html_client

    response = client.post(
        f"/api/v1/slides/presentations/{structured.id}/versions/999/restore",
        headers={"If-Match": 'W/"v0"'},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "presentation_version_not_found"
