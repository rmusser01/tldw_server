from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import llamacpp as lp
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.Jobs.manager import JobManager


def _admin_principal() -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject=None,
        token_type="access",
        jti=None,
        roles=["admin"],
        permissions=[],
        is_admin=True,
        org_ids=[],
        team_ids=[],
    )


class _Logger:
    def error(self, *args: Any, **kwargs: Any) -> None:
        return


class _Manager:
    logger = _Logger()


def _make_app(job_manager: JobManager) -> FastAPI:
    app = FastAPI()
    app.include_router(lp.router, prefix="/api/v1")
    app.state.llm_manager = _Manager()

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
        principal = _admin_principal()
        ip = request.client.host if getattr(request, "client", None) else None
        ua = request.headers.get("User-Agent") if getattr(request, "headers", None) else None
        request_id = request.headers.get("X-Request-ID") if getattr(request, "headers", None) else None
        request.state.auth = AuthContext(principal=principal, ip=ip, user_agent=ua, request_id=request_id)
        return principal

    async def _fake_check_rate_limit() -> None:
        return

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal
    app.dependency_overrides[auth_deps.check_rate_limit] = _fake_check_rate_limit
    app.dependency_overrides[lp.check_rate_limit] = _fake_check_rate_limit
    app.dependency_overrides[lp.get_job_manager] = lambda: job_manager
    return app


def _llamacpp_parser(models_dir: Path, **overrides: str) -> ConfigParser:
    parser = ConfigParser()
    parser.add_section("LlamaCpp")
    values = {
        "enabled": "true",
        "models_dir": str(models_dir),
        "allowed_paths": "",
        "registered_model_paths": "",
        "imported_asset_folders": "",
    }
    values.update(overrides)
    parser["LlamaCpp"] = values
    return parser


def _allow_public_dns(monkeypatch: pytest.MonkeyPatch, llamacpp_acquisition_service: Any) -> None:
    monkeypatch.setattr(
        llamacpp_acquisition_service.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(None, None, None, None, ("93.184.216.34", 0))],
    )


@pytest.mark.unit
def test_download_endpoint_creates_redacted_llamacpp_acquisition_job(monkeypatch, tmp_path: Path) -> None:
    from tldw_Server_API.app.core.Local_LLM import llamacpp_acquisition_service

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(
        llamacpp_acquisition_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir),
    )
    _allow_public_dns(monkeypatch, llamacpp_acquisition_service)
    job_manager = JobManager(db_path=tmp_path / "jobs.db")
    app = _make_app(job_manager)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/llamacpp/assets/downloads",
            json={
                "url": "https://example.com/releases/model.gguf?download=1",
                "expected_size_bytes": 11,
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "queued"
    assert body["operation"] == "download"
    assert body["queue"] == "acquisition"
    assert body["source_label"] == "https://example.com/releases/model.gguf?download=1"
    job = job_manager.get_job(int(body["job_id"]))
    assert job is not None
    assert job["domain"] == "llamacpp"
    assert job["queue"] == "acquisition"
    assert job["job_type"] == "llamacpp_asset_download"
    assert job["payload"]["source_url"] == "https://example.com/releases/model.gguf?download=1"
    assert job["payload"]["destination_path"] == str(models_dir / "model.gguf")


@pytest.mark.unit
def test_download_job_status_list_and_cancel_endpoints(monkeypatch, tmp_path: Path) -> None:
    from tldw_Server_API.app.core.Local_LLM import llamacpp_acquisition_service

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(
        llamacpp_acquisition_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir),
    )
    _allow_public_dns(monkeypatch, llamacpp_acquisition_service)
    job_manager = JobManager(db_path=tmp_path / "jobs.db")
    app = _make_app(job_manager)

    with TestClient(app) as client:
        create_response = client.post(
            "/api/v1/llamacpp/assets/downloads",
            json={"url": "https://example.com/releases/model.gguf"},
        )
        job_id = create_response.json()["job_id"]
        status_response = client.get(f"/api/v1/llamacpp/assets/downloads/{job_id}")
        list_response = client.get("/api/v1/llamacpp/assets/downloads")
        cancel_response = client.delete(f"/api/v1/llamacpp/assets/downloads/{job_id}")

    assert create_response.status_code == 200, create_response.text
    assert status_response.status_code == 200, status_response.text
    assert status_response.json()["job_id"] == job_id
    assert list_response.status_code == 200, list_response.text
    assert [job["job_id"] for job in list_response.json()["jobs"]] == [job_id]
    assert cancel_response.status_code == 200, cancel_response.text
    assert cancel_response.json()["status"] == "cancelled"


@pytest.mark.unit
def test_download_endpoint_rejects_invalid_request(monkeypatch, tmp_path: Path) -> None:
    from tldw_Server_API.app.core.Local_LLM import llamacpp_acquisition_service

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(
        llamacpp_acquisition_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir),
    )
    app = _make_app(JobManager(db_path=tmp_path / "jobs.db"))

    with TestClient(app) as client:
        response = client.post("/api/v1/llamacpp/assets/downloads", json={"url": "file:///tmp/model.gguf"})

    assert response.status_code == 400
    assert "scheme" in response.json()["detail"].lower()


@pytest.mark.unit
@pytest.mark.parametrize(
    "url",
    [
        "https://user:pass@example.com/releases/model.gguf",
        "https://example.com/releases/model.gguf?token=secret&download=1",
    ],
)
def test_download_endpoint_rejects_credentialed_or_secret_urls(monkeypatch, tmp_path: Path, url: str) -> None:
    from tldw_Server_API.app.core.Local_LLM import llamacpp_acquisition_service

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(
        llamacpp_acquisition_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir),
    )
    app = _make_app(JobManager(db_path=tmp_path / "jobs.db"))

    with TestClient(app) as client:
        response = client.post("/api/v1/llamacpp/assets/downloads", json={"url": url})

    assert response.status_code == 400
    assert "credential" in response.json()["detail"].lower() or "secret" in response.json()["detail"].lower()
