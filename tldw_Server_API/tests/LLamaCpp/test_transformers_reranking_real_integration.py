import os
import importlib.util
import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.tests.LLamaCpp.llamacpp_test_app import llamacpp_test_client


def _has_sentence_transformers() -> bool:

    return importlib.util.find_spec("sentence_transformers") is not None


def _xformer_model() -> str:

    return os.getenv("TEST_XFORMERS_RERANKER_MODEL") or os.getenv("RAG_TRANSFORMERS_RERANKER_MODEL") or ""


def _model_available() -> bool:

    # HF id can be remote; tests rely on local cache or network if allowed; skip by default unless explicitly set
    return bool(_xformer_model())


@pytest.fixture(scope="module")
def client():
    settings = get_settings()
    headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
    with llamacpp_test_client(headers=headers) as c:
        yield c


pytestmark = [
    pytest.mark.integration,
    pytest.mark.external_api,
]


def _skip_if_unavailable():

    if not _has_sentence_transformers():
        pytest.skip("sentence_transformers not installed; set TEST_XFORMERS_RERANKER_MODEL to run")
    if not _model_available():
        pytest.skip("TEST_XFORMERS_RERANKER_MODEL or RAG_TRANSFORMERS_RERANKER_MODEL not set")


def test_public_reranking_transformers_real_integration(client: TestClient):
    _skip_if_unavailable()
    payload = {
        "backend": "transformers",
        "model": _xformer_model(),
        "query": "What is panda?",
        "top_n": 2,
        "documents": ["hi", "it is a bear", "The giant panda is a bear species endemic to China."],
    }
    resp = client.post("/v1/reranking", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert isinstance(data.get("results"), list)
    assert len(data["results"]) == 2
