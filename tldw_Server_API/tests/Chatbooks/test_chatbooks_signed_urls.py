import hmac
import hashlib
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import chatbooks as chatbooks_mod
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.tests.Chatbooks.test_chatbooks_export_sync import (
    _make_export_payload,
    client_override,
)


def test_signed_download_happy_path(client_override: TestClient, monkeypatch):
    # Enable signed URLs
    monkeypatch.setenv("CHATBOOKS_SIGNED_URLS", "true")
    monkeypatch.setenv("CHATBOOKS_SIGNING_SECRET", "secret123")
    monkeypatch.setenv("CHATBOOKS_ENFORCE_EXPIRY", "true")
    monkeypatch.setenv("CHATBOOKS_URL_TTL_SECONDS", "3600")

    resp = client_override.post("/api/v1/chatbooks/export", json=_make_export_payload(async_mode=False))
    assert resp.status_code == 200, resp.text

    body = resp.json()
    job_id = body.get("job_id")
    download_url = body.get("download_url")
    assert job_id and download_url

    # Valid signed download should succeed
    dresp = client_override.get(download_url)
    assert dresp.status_code == 200
    assert dresp.headers.get("content-type") == "application/zip"
    assert "attachment; filename=" in dresp.headers.get("content-disposition", "")


def test_signed_download_invalid_token(client_override: TestClient, monkeypatch):
    monkeypatch.setenv("CHATBOOKS_SIGNED_URLS", "true")
    monkeypatch.setenv("CHATBOOKS_SIGNING_SECRET", "secret123")
    monkeypatch.setenv("CHATBOOKS_ENFORCE_EXPIRY", "true")
    monkeypatch.setenv("CHATBOOKS_URL_TTL_SECONDS", "3600")

    resp = client_override.post("/api/v1/chatbooks/export", json=_make_export_payload(async_mode=False))
    assert resp.status_code == 200, resp.text

    body = resp.json()
    download_url = body.get("download_url")
    assert download_url

    # Tamper token
    parts = urlparse(download_url)
    qs = parse_qs(parts.query)
    qs["token"] = ["deadbeef"]
    new_qs = urlencode({k: v[0] for k, v in qs.items()})
    bad_url = urlunparse((parts.scheme, parts.netloc, parts.path, parts.params, new_qs, parts.fragment))

    dresp = client_override.get(bad_url)
    assert dresp.status_code == 403
    assert dresp.json().get("detail") == "Invalid signature"


def test_signed_download_expired_exp_param(client_override: TestClient, monkeypatch):
    monkeypatch.setenv("CHATBOOKS_SIGNED_URLS", "true")
    monkeypatch.setenv("CHATBOOKS_SIGNING_SECRET", "secret123")
    monkeypatch.setenv("CHATBOOKS_ENFORCE_EXPIRY", "true")
    monkeypatch.setenv("CHATBOOKS_URL_TTL_SECONDS", "3600")

    resp = client_override.post("/api/v1/chatbooks/export", json=_make_export_payload(async_mode=False))
    assert resp.status_code == 200, resp.text

    body = resp.json()
    job_id = body.get("job_id")
    assert job_id

    # Forge an expired exp param
    exp = 1  # definitely in the past
    msg = f"{job_id}:{exp}".encode("utf-8")
    token = hmac.new(b"secret123", msg, hashlib.sha256).hexdigest()
    url = f"/api/v1/chatbooks/download/{job_id}?exp={exp}&token={token}"

    dresp = client_override.get(url)
    assert dresp.status_code == 410


def test_signed_download_continuation_export_uses_job_backed_url(
    client_override: TestClient,
    monkeypatch,
):
    monkeypatch.setenv("CHATBOOKS_SIGNED_URLS", "true")
    monkeypatch.setenv("CHATBOOKS_SIGNING_SECRET", "secret123")
    monkeypatch.setenv("CHATBOOKS_ENFORCE_EXPIRY", "true")
    monkeypatch.setenv("CHATBOOKS_URL_TTL_SECONDS", "3600")

    continuation_path = DatabasePaths.get_user_chatbooks_exports_dir(1).resolve() / "continued.chatbook"
    continuation_path.write_bytes(b"continued-export")

    async def _fake_continue_export(*_args, **_kwargs):
        return True, "Continuation ready", str(continuation_path)

    monkeypatch.setattr(chatbooks_mod.ChatbookService, "continue_chatbook_export", _fake_continue_export)
    opaque_cursor = "-".join(("cursor", "1"))

    response = client_override.post(
        "/api/v1/chatbooks/export/continue",
        json={
            "export_id": "export-continue-1",
            "continuations": [{"evaluation_id": "eval-1", "continuation_token": opaque_cursor}],
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body.get("job_id"), str) and body["job_id"]
    assert body.get("download_url")
    assert "file_path" not in body

    job_response = client_override.get(f"/api/v1/chatbooks/export/jobs/{body['job_id']}")
    assert job_response.status_code == 200, job_response.text
    job_download_url = job_response.json()["download_url"]
    body_parts = urlparse(body["download_url"])
    job_parts = urlparse(job_download_url)
    assert body_parts.path == job_parts.path == f"/api/v1/chatbooks/download/{body['job_id']}"
    assert {"exp", "token"} <= set(parse_qs(body_parts.query))
    assert {"exp", "token"} <= set(parse_qs(job_parts.query))

    download_response = client_override.get(body["download_url"])
    assert download_response.status_code == 200
    assert download_response.headers.get("content-type") == "application/zip"
    assert download_response.content == b"continued-export"
