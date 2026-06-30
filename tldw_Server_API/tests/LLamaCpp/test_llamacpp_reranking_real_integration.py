import os
import shutil
import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.tests.LLamaCpp.llamacpp_test_app import llamacpp_test_client


# Ensure rate limiter bypass in tests for determinism
os.environ.setdefault("TEST_MODE", "true")


def _has_llama_embedding() -> bool:

    return shutil.which("llama-embedding") is not None


def _model_path() -> str:

    # Allow test-specific override to avoid touching prod config
    return os.getenv("TEST_QWEN_GGUF_MODEL") or os.getenv("RAG_LLAMA_RERANKER_MODEL") or ""


def _model_available() -> bool:

    m = _model_path()
    return bool(m and os.path.exists(m))


@pytest.fixture(scope="module")
def client():
    settings = get_settings()
    headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
    with llamacpp_test_client(headers=headers) as c:
        yield c


pytestmark = [
    pytest.mark.integration,
    pytest.mark.local_llm_service,
]


def _skip_if_unavailable():

    if not _has_llama_embedding():
        pytest.skip("llama-embedding binary not found on PATH; set RUN with llama.cpp installed")
    if not _model_available():
        pytest.skip("TEST_QWEN_GGUF_MODEL or RAG_LLAMA_RERANKER_MODEL not set to a readable GGUF file")


def test_public_reranking_real_integration(client: TestClient):
    _skip_if_unavailable()
    model = _model_path()
    payload = {
        "model": model,
        "query": "What is panda?",
        "top_n": 2,
        "documents": ["hi", "it is a bear", "The giant panda is a bear species endemic to China."],
    }
    resp = client.post("/v1/reranking", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert isinstance(data.get("results"), list)
    assert len(data["results"]) == 2
    scores = [r.get("score", 0.0) for r in data["results"]]
    assert all(isinstance(s, (int, float)) for s in scores)
    assert all(0.0 <= float(s) <= 1.0 for s in scores)


def test_llamacpp_reranking_real_integration(client: TestClient):
    _skip_if_unavailable()
    model = _model_path()
    payload = {
        "query": "What do llamas eat?",
        "top_k": 2,
        "passages": [
            {"id": "a", "text": "Llamas eat bananas and grass."},
            {"id": "b", "text": "Llamas in pyjamas"},
            {"id": "c", "text": "A bowl of fruit salad"},
        ],
        "model": model,
    }
    resp = client.post("/api/v1/llamacpp/reranking", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert isinstance(data.get("results"), list)
    assert len(data["results"]) == 2
    # Ensure at least the food-related passage ranks high
    top_texts = [r.get("text", "") for r in data["results"]]
    assert any("bananas" in (t or "").lower() or "grass" in (t or "").lower() for t in top_texts)
