import copy
import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from loguru import logger

from tldw_Server_API.app.main import app
from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal


@pytest.fixture
def restore_embedding_settings():
    snapshot = {
        "EMBEDDING_CONFIG": copy.deepcopy(settings.get("EMBEDDING_CONFIG")),
        "ALLOWED_EMBEDDING_PROVIDERS": copy.deepcopy(settings.get("ALLOWED_EMBEDDING_PROVIDERS")),
        "ALLOWED_EMBEDDING_MODELS": copy.deepcopy(settings.get("ALLOWED_EMBEDDING_MODELS")),
    }
    yield
    for key, value in snapshot.items():
        if value is None:
            settings.pop(key, None)
        else:
            settings[key] = value


def _client():


    c = TestClient(app)
    c.cookies.set("csrf_token", "test-csrf")
    return c


def _csrf_headers():


    return {"X-CSRF-Token": "test-csrf"}


class _DummyUser:
    def __init__(self, user_id: int, is_admin: bool):
        self.id = user_id
        self.is_admin = is_admin
        self.is_active = True


async def _override_admin_user():
    return _DummyUser(2, True)


async def _override_regular_user():
    return _DummyUser(1, False)


def _principal_override(user_id: int, is_admin: bool):
    async def _override(request: Request):
        # token_type is a test principal label, not a secret.
        principal = AuthPrincipal(  # nosec B106
            kind="user",
            user_id=user_id,
            api_key_id=None,
            subject=None,
            token_type="access",
            jti=None,
            roles=["admin"] if is_admin else ["user"],
            permissions=["*"] if is_admin else [],
            is_admin=is_admin,
            org_ids=[],
            team_ids=[],
        )
        try:
            request.state.auth = AuthContext(
                principal=principal,
                ip=request.client.host if getattr(request, "client", None) else None,
                user_agent=request.headers.get("User-Agent") if getattr(request, "headers", None) else None,
                request_id=request.headers.get("X-Request-ID") if getattr(request, "headers", None) else None,
            )
        except (AttributeError, ValueError) as exc:
            # Best-effort; setting request.state.auth may fail in some test scenarios.
            logger.debug(f"Failed to set request.state.auth: {exc}")
        return principal
    return _override


def test_list_models_exposes_defaults_and_policy(restore_embedding_settings):


    os.environ["TESTING"] = "true"
    try:
        # Configure defaults and allowlists
        cfg = copy.deepcopy(settings.get("EMBEDDING_CONFIG", {}) or {})
        cfg["default_model_id"] = "text-embedding-3-small"
        cfg["embedding_model"] = "text-embedding-3-small"
        cfg["embedding_provider"] = "openai"
        settings["EMBEDDING_CONFIG"] = cfg
        settings["ALLOWED_EMBEDDING_PROVIDERS"] = ["openai", "huggingface"]
        settings["ALLOWED_EMBEDDING_MODELS"] = ["text-embedding-3-*"]

        client = _client()
        r = client.get("/api/v1/embeddings/models", headers=_csrf_headers())
        assert r.status_code == 200
        j = r.json()
        assert "data" in j and isinstance(j["data"], list)
        # Ensure default present and marked
        defaults = [x for x in j["data"] if x.get("default")]
        assert any(d.get("model") == "text-embedding-3-small" and d.get("provider") == "openai" for d in defaults)
        # Ensure allowlists included
        assert j.get("allowed_providers") is None or isinstance(j.get("allowed_providers"), list)
        assert j.get("allowed_models") is None or isinstance(j.get("allowed_models"), list)
    finally:
        os.environ.pop("TESTING", None)


def test_warmup_requires_admin_and_invokes_backend(restore_embedding_settings):


    os.environ["TESTING"] = "true"
    try:
        # Allow model
        settings["ALLOWED_EMBEDDING_PROVIDERS"] = ["openai"]
        settings["ALLOWED_EMBEDDING_MODELS"] = ["text-embedding-3-small"]

        # Non-admin rejected
        app.dependency_overrides[get_request_user] = _override_regular_user
        app.dependency_overrides[auth_deps.get_auth_principal] = _principal_override(1, False)
        client = _client()
        r_forbidden = client.post(
            "/api/v1/embeddings/models/warmup",
            json={"model": "text-embedding-3-small"},
            headers=_csrf_headers()
        )
        assert r_forbidden.status_code == 403

        app.dependency_overrides.pop(get_request_user, None)
        app.dependency_overrides.pop(auth_deps.get_auth_principal, None)

        # Admin path with backend stub
        app.dependency_overrides[get_request_user] = _override_admin_user
        app.dependency_overrides[auth_deps.get_auth_principal] = _principal_override(2, True)
        client = _client()
        with patch(
            'tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced.create_embeddings_batch_async',
            new=AsyncMock(return_value=[[0.1, 0.2]])
        ):
            r = client.post(
                "/api/v1/embeddings/models/warmup",
                json={"model": "text-embedding-3-small"},
                headers=_csrf_headers()
            )
            assert r.status_code == 200
            assert r.json().get("warmed") is True
    finally:
        os.environ.pop("TESTING", None)
        app.dependency_overrides.pop(get_request_user, None)
        app.dependency_overrides.pop(auth_deps.get_auth_principal, None)


def test_download_requires_admin_and_invokes_backend(restore_embedding_settings):


    os.environ["TESTING"] = "true"
    try:
        # Allow model
        settings["ALLOWED_EMBEDDING_PROVIDERS"] = ["huggingface", "openai"]
        settings["ALLOWED_EMBEDDING_MODELS"] = ["sentence-transformers/all-MiniLM-L6-v2", "text-embedding-3-small"]

        app.dependency_overrides[get_request_user] = _override_admin_user
        app.dependency_overrides[auth_deps.get_auth_principal] = _principal_override(2, True)
        client = _client()
        with patch(
            'tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced.create_embeddings_batch_async',
            new=AsyncMock(return_value=[[0.3, 0.4]])
        ):
            r = client.post("/api/v1/embeddings/models/download", json={"model": "sentence-transformers/all-MiniLM-L6-v2"}, headers=_csrf_headers())
            assert r.status_code == 200
            assert r.json().get("downloaded") is True
    finally:
        os.environ.pop("TESTING", None)
        app.dependency_overrides.pop(get_request_user, None)
        app.dependency_overrides.pop(auth_deps.get_auth_principal, None)


def test_warmup_sanitizes_unexpected_backend_error(restore_embedding_settings):
    os.environ["TESTING"] = "true"
    try:
        settings["ALLOWED_EMBEDDING_PROVIDERS"] = ["openai"]
        settings["ALLOWED_EMBEDDING_MODELS"] = ["text-embedding-3-small"]

        app.dependency_overrides[get_request_user] = _override_admin_user
        app.dependency_overrides[auth_deps.get_auth_principal] = _principal_override(2, True)
        client = _client()

        with patch(
            'tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced.create_embeddings_batch_async',
            new=AsyncMock(side_effect=RuntimeError("backend leaked /private/model-cache path"))
        ):
            r = client.post(
                "/api/v1/embeddings/models/warmup",
                json={"model": "text-embedding-3-small"},
                headers=_csrf_headers(),
            )

        assert r.status_code == 502
        assert r.json()["detail"] == "Warmup failed"
    finally:
        os.environ.pop("TESTING", None)
        app.dependency_overrides.pop(get_request_user, None)
        app.dependency_overrides.pop(auth_deps.get_auth_principal, None)


def test_download_sanitizes_unexpected_backend_error(restore_embedding_settings):
    os.environ["TESTING"] = "true"
    try:
        settings["ALLOWED_EMBEDDING_PROVIDERS"] = ["openai"]
        settings["ALLOWED_EMBEDDING_MODELS"] = ["text-embedding-3-small"]

        app.dependency_overrides[get_request_user] = _override_admin_user
        app.dependency_overrides[auth_deps.get_auth_principal] = _principal_override(2, True)
        client = _client()

        with patch(
            'tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced.create_embeddings_batch_async',
            new=AsyncMock(side_effect=RuntimeError("backend leaked /private/model-cache path"))
        ):
            r = client.post(
                "/api/v1/embeddings/models/download",
                json={"model": "text-embedding-3-small"},
                headers=_csrf_headers(),
            )

        assert r.status_code == 502
        assert r.json()["detail"] == "Download failed"
    finally:
        os.environ.pop("TESTING", None)
        app.dependency_overrides.pop(get_request_user, None)
        app.dependency_overrides.pop(auth_deps.get_auth_principal, None)


def test_warmup_rejects_disallowed_provider_and_model(restore_embedding_settings):


    os.environ["TESTING"] = "true"
    try:
        # Only allow huggingface, and only specific HF model
        settings["ALLOWED_EMBEDDING_PROVIDERS"] = ["huggingface"]
        settings["ALLOWED_EMBEDDING_MODELS"] = ["sentence-transformers/all-MiniLM-L6-v2"]

        app.dependency_overrides[get_request_user] = _override_admin_user
        app.dependency_overrides[auth_deps.get_auth_principal] = _principal_override(2, True)
        client = _client()

        # Disallowed provider (openai)
        r1 = client.post("/api/v1/embeddings/models/warmup", json={"model": "text-embedding-3-small"}, headers=_csrf_headers())
        assert r1.status_code == 403
        assert "not allowed" in r1.text.lower()

        # Disallowed model (HF but not in allowlist)
        r2 = client.post("/api/v1/embeddings/models/warmup", json={"model": "sentence-transformers/all-mpnet-base-v2"}, headers=_csrf_headers())
        assert r2.status_code == 403
        assert "not allowed" in r2.text.lower()
    finally:
        os.environ.pop("TESTING", None)
        app.dependency_overrides.pop(get_request_user, None)
        app.dependency_overrides.pop(auth_deps.get_auth_principal, None)


def test_download_rejects_disallowed_provider_and_model(restore_embedding_settings):


    os.environ["TESTING"] = "true"
    try:
        # Only allow openai and a specific openai model
        settings["ALLOWED_EMBEDDING_PROVIDERS"] = ["openai"]
        settings["ALLOWED_EMBEDDING_MODELS"] = ["text-embedding-3-large"]

        app.dependency_overrides[get_request_user] = _override_admin_user
        app.dependency_overrides[auth_deps.get_auth_principal] = _principal_override(2, True)
        client = _client()

        # Disallowed provider inference (HF)
        r1 = client.post("/api/v1/embeddings/models/download", json={"model": "sentence-transformers/all-MiniLM-L6-v2"}, headers=_csrf_headers())
        assert r1.status_code == 403
        assert "not allowed" in r1.text.lower()

        # Disallowed model (openai model not in allowlist)
        r2 = client.post("/api/v1/embeddings/models/download", json={"model": "text-embedding-3-small"}, headers=_csrf_headers())
        assert r2.status_code == 403
        assert "not allowed" in r2.text.lower()
    finally:
        os.environ.pop("TESTING", None)
        app.dependency_overrides.pop(get_request_user, None)
        app.dependency_overrides.pop(auth_deps.get_auth_principal, None)


def test_list_models_reflects_disallowed_models(restore_embedding_settings):


    os.environ["TESTING"] = "true"
    try:
        # Disallow text-embedding-3-small specifically
        settings["ALLOWED_EMBEDDING_PROVIDERS"] = ["openai"]
        settings["ALLOWED_EMBEDDING_MODELS"] = ["text-embedding-3-large"]
        cfg = copy.deepcopy(settings.get("EMBEDDING_CONFIG", {}) or {})
        cfg["default_model_id"] = "text-embedding-3-small"
        cfg["embedding_model"] = "text-embedding-3-small"
        cfg["embedding_provider"] = "openai"
        settings["EMBEDDING_CONFIG"] = cfg

        client = _client()
        r = client.get("/api/v1/embeddings/models", headers=_csrf_headers())
        assert r.status_code == 200
        data = r.json()["data"]
        # Find entry for small and ensure allowed == False
        smalls = [x for x in data if x.get("model") == "text-embedding-3-small" and x.get("provider") == "openai"]
        assert smalls
        assert all(not x.get("allowed") for x in smalls)
    finally:
        os.environ.pop("TESTING", None)
