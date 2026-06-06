import os
import time
from fastapi.testclient import TestClient
from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.AuthNZ.settings import get_settings


from contextlib import contextmanager


@contextmanager
def _client():
    previous_auto_download = os.environ.get("AUTO_DOWNLOAD_MODELS")
    os.environ["AUTO_DOWNLOAD_MODELS"] = "false"
    try:
        with TestClient(app) as c:
            c.cookies.set("csrf_token", "test-csrf")
            yield c
    finally:
        if previous_auto_download is None:
            os.environ.pop("AUTO_DOWNLOAD_MODELS", None)
        else:
            os.environ["AUTO_DOWNLOAD_MODELS"] = previous_auto_download


def test_embeddings_token_limit_rejected():


     # Ensure test mode rate limiting bypass
    os.environ["TESTING"] = "true"
    try:
        # Preserve existing settings to avoid cross-test contamination
        original_allowed_providers = settings.get("ALLOWED_EMBEDDING_PROVIDERS")
        original_allowed_models = settings.get("ALLOWED_EMBEDDING_MODELS")
        original_model_limits = settings.get("EMBEDDING_MODEL_MAX_TOKENS")
        # Configure strict token limit
        settings["EMBEDDING_MODEL_MAX_TOKENS"] = {"openai:text-embedding-3-small": 1}
        # Allow provider/model
        settings["ALLOWED_EMBEDDING_PROVIDERS"] = ["openai"]
        settings["ALLOWED_EMBEDDING_MODELS"] = ["text-embedding-3-small"]

        with _client() as client:
            payload = {
                "model": "text-embedding-3-small",
                "input": "This sentence surely exceeds a single token."
            }
            # Include required API key for single-user auth
            api_key = get_settings().SINGLE_USER_API_KEY
            r = client.post("/api/v1/embeddings", json=payload, headers={"X-API-KEY": api_key})
            assert r.status_code == 400
            data = r.json()
            assert data.get("error") == "input_too_long"
            assert "details" in data
            assert isinstance(data["details"], list) and len(data["details"]) >= 1
    finally:
        os.environ.pop("TESTING", None)
        # Restore original settings
        if original_allowed_providers is None:
            settings.pop("ALLOWED_EMBEDDING_PROVIDERS", None)
        else:
            settings["ALLOWED_EMBEDDING_PROVIDERS"] = original_allowed_providers
        if original_allowed_models is None:
            settings.pop("ALLOWED_EMBEDDING_MODELS", None)
        else:
            settings["ALLOWED_EMBEDDING_MODELS"] = original_allowed_models
        if original_model_limits is None:
            settings.pop("EMBEDDING_MODEL_MAX_TOKENS", None)
        else:
            settings["EMBEDDING_MODEL_MAX_TOKENS"] = original_model_limits


def test_embeddings_allowlist_rejected():


    os.environ["TESTING"] = "true"
    try:
        original_allowed_providers = settings.get("ALLOWED_EMBEDDING_PROVIDERS")
        original_allowed_models = settings.get("ALLOWED_EMBEDDING_MODELS")
        # Disallow provider/model
        settings["ALLOWED_EMBEDDING_PROVIDERS"] = ["huggingface"]
        settings["ALLOWED_EMBEDDING_MODELS"] = ["sentence-transformers/all-MiniLM-L6-v2"]

        with _client() as client:
            payload = {
                "model": "text-embedding-3-small",
                "input": "short"
            }
            api_key = get_settings().SINGLE_USER_API_KEY
            r = client.post("/api/v1/embeddings", json=payload, headers={"X-API-KEY": api_key})
            assert r.status_code == 403
            assert "not allowed" in r.json().get("detail", "").lower()
    finally:
        os.environ.pop("TESTING", None)
        if original_allowed_providers is None:
            settings.pop("ALLOWED_EMBEDDING_PROVIDERS", None)
        else:
            settings["ALLOWED_EMBEDDING_PROVIDERS"] = original_allowed_providers
        if original_allowed_models is None:
            settings.pop("ALLOWED_EMBEDDING_MODELS", None)
        else:
            settings["ALLOWED_EMBEDDING_MODELS"] = original_allowed_models
