import os
import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user


@pytest.fixture
def client():
    previous_auto_download = os.environ.get("AUTO_DOWNLOAD_MODELS")
    os.environ["AUTO_DOWNLOAD_MODELS"] = "false"
    try:
        with TestClient(app) as c:
            c.cookies.set("csrf_token", "x")
            c.headers["X-CSRF-Token"] = "x"
            c.headers["Authorization"] = "Bearer key"
            yield c
    finally:
        if previous_auto_download is None:
            os.environ.pop("AUTO_DOWNLOAD_MODELS", None)
        else:
            os.environ["AUTO_DOWNLOAD_MODELS"] = previous_auto_download


def _override_user(admin=False):


    async def _f():
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
        return User(id=1, username="admin" if admin else "u", email="u@x", is_active=True, is_admin=admin)
    return _f


@pytest.mark.unit
def test_policy_strict_blocks_admin(client):
    os.environ["TESTING"] = "true"
    os.environ["EMBEDDINGS_ENFORCE_POLICY"] = "true"
    os.environ["EMBEDDINGS_ENFORCE_POLICY_STRICT"] = "true"
    try:
        # Disallow openai; allow only hf
        from tldw_Server_API.app.core.config import settings
        original_allowed_providers = settings.get("ALLOWED_EMBEDDING_PROVIDERS")
        original_allowed_models = settings.get("ALLOWED_EMBEDDING_MODELS")
        settings["ALLOWED_EMBEDDING_PROVIDERS"] = ["huggingface"]
        settings["ALLOWED_EMBEDDING_MODELS"] = ["sentence-transformers/all-MiniLM-L6-v2"]

        app.dependency_overrides[get_request_user] = _override_user(admin=True)
        r = client.post(
            "/api/v1/embeddings",
            json={"input": "txt", "model": "text-embedding-3-small"},
            headers={"x-provider": "openai"}
        )
        assert r.status_code == 403
        assert "not allowed" in r.json().get("detail", "").lower()
    finally:
        os.environ.pop("TESTING", None)
        os.environ.pop("EMBEDDINGS_ENFORCE_POLICY", None)
        os.environ.pop("EMBEDDINGS_ENFORCE_POLICY_STRICT", None)
        from tldw_Server_API.app.core.config import settings
        if original_allowed_providers is None:
            settings.pop("ALLOWED_EMBEDDING_PROVIDERS", None)
        else:
            settings["ALLOWED_EMBEDDING_PROVIDERS"] = original_allowed_providers
        if original_allowed_models is None:
            settings.pop("ALLOWED_EMBEDDING_MODELS", None)
        else:
            settings["ALLOWED_EMBEDDING_MODELS"] = original_allowed_models
