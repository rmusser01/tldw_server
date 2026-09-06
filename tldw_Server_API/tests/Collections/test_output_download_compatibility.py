"""Characterize inactive generic download routes before descriptor migration."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import outputs
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.tests.Collections.test_reading_output_disposal_routes import client as client
from tldw_Server_API.tests.Collections.test_reading_revision_mutations import db as db

pytestmark = pytest.mark.unit
pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_full_fixtures"]


@pytest.fixture
def artifact(db, tmp_path):
    row = db.create_output_artifact(type_="report", title="Example", format_="md", storage_path="example.md")
    (tmp_path / row.storage_path).write_bytes(b"0123456789")
    return row


def routes(row):
    return [
        ("GET", f"/outputs/{row.id}/download", {}),
        ("GET", "/outputs/download/by-name", {"title": row.title, "format": row.format}),
        ("HEAD", f"/outputs/{row.id}/download", {}),
    ]


@pytest.mark.parametrize(
    "format_, media_type",
    [("md", "text/markdown; charset=utf-8"), ("html", "text/html; charset=utf-8"), ("mp3", "audio/mpeg")],
)
def test_existing_download_headers_and_head(db, client, artifact, format_, media_type):
    row = db.update_output_artifact(artifact.id, format_=format_)
    for method, url, params in routes(row):
        response = client.request(method, url, params=params)
        assert response.status_code == 200
        assert response.headers["content-type"] == media_type
        assert response.headers["content-length"] == "10"
        assert response.content == (b"" if method == "HEAD" else b"0123456789")
        if method == "GET":
            assert response.headers["content-disposition"] == 'attachment; filename="example.md"'
            assert response.headers["accept-ranges"] == "bytes"
            assert response.headers["etag"] and response.headers["last-modified"]
        else:
            assert "etag" not in response.headers and "content-disposition" not in response.headers


@pytest.mark.parametrize(
    "header, status, body",
    [
        ({"Range": "bytes=2-5"}, 206, b"2345"),
        ({"Range": "bytes=-3"}, 206, b"789"),
        ({"Range": "bytes=99-"}, 416, b""),
        ({"Range": "nonsense"}, 400, b"Malformed range header."),
        ({"Range": "bytes=2-5", "If-Range": "stale"}, 200, b"0123456789"),
    ],
)
def test_existing_get_range_behavior(client, artifact, header, status, body):
    for method, url, params in routes(artifact)[:2]:
        response = client.request(method, url, params=params, headers=header)
        assert response.status_code == status
        assert response.content == body


def test_existing_get_validators_control_if_range_but_not_304(client, artifact):
    for method, url, params in routes(artifact)[:2]:
        initial = client.request(method, url, params=params)
        for validator in (initial.headers["etag"], initial.headers["last-modified"]):
            response = client.request(method, url, params=params, headers={"Range": "bytes=2-5", "If-Range": validator})
            assert response.status_code == 206 and response.content == b"2345"
        response = client.request(
            method,
            url,
            params=params,
            headers={"If-None-Match": initial.headers["etag"], "If-Modified-Since": initial.headers["last-modified"]},
        )
        assert response.status_code == 200 and response.content == b"0123456789"


@pytest.mark.parametrize("case", ["missing", "deleted", "foreign", "file_missing"])
def test_existing_inaccessible_downloads(db, tmp_path, client, artifact, case):
    if case == "deleted":
        db.delete_output_artifact(artifact.id)
    elif case == "missing":
        db.delete_output_artifact(artifact.id, hard=True)
    elif case == "foreign":
        client.app.dependency_overrides[outputs.get_collections_db_for_user] = lambda: CollectionsDatabase.from_backend(
            user_id="781", backend=db.backend
        )
    else:
        (tmp_path / artifact.storage_path).unlink()
    for method, url, params in routes(artifact):
        response = client.request(method, url, params=params)
        assert response.status_code == 404
        if method != "HEAD":
            assert response.json() == {"detail": "file_missing" if case == "file_missing" else "output_not_found"}


def test_existing_auth_failure_precedes_output_lookup(client, artifact):
    def unauthorized():
        raise HTTPException(status_code=401, detail="not_authenticated")

    def forbidden():
        pytest.fail("unauthenticated download looked up output")

    client.app.dependency_overrides[outputs.get_request_user] = unauthorized
    client.app.dependency_overrides[outputs.get_collections_db_for_user] = forbidden
    for method, url, params in routes(artifact):
        assert client.request(method, url, params=params).status_code == 401
