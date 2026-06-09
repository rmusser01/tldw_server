import os
import json
import pytest
from fastapi.testclient import TestClient
from types import SimpleNamespace

from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
from tldw_Server_API.tests.LLamaCpp.llamacpp_test_app import llamacpp_test_client


@pytest.fixture(scope="module")
def client():
    # Ensure a consistent single-user auth configuration for this module,
    # regardless of prior AuthNZ tests that may have switched modes.
    prev_auth_mode = os.environ.get("AUTH_MODE")
    prev_single_key = os.environ.get("SINGLE_USER_API_KEY")
    try:
        os.environ["AUTH_MODE"] = "single_user"
        os.environ["SINGLE_USER_API_KEY"] = os.getenv("SINGLE_USER_TEST_API_KEY", "test-api-key-12345")
        reset_settings()
        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        with llamacpp_test_client(headers=headers) as c:
            yield c
    finally:
        # Restore previous environment and reset settings so other tests see
        # their expected configuration.
        if prev_auth_mode is None:
            os.environ.pop("AUTH_MODE", None)
        else:
            os.environ["AUTH_MODE"] = prev_auth_mode
        if prev_single_key is None:
            os.environ.pop("SINGLE_USER_API_KEY", None)
        else:
            os.environ["SINGLE_USER_API_KEY"] = prev_single_key
        reset_settings()


class _FakeProc:
    def __init__(self, captured):
        self._captured = captured
        self.returncode = 0

    async def communicate(self):
        # Build minimal embeddings array: 1 query + N docs, each 8-dim
        n = self._captured.get("n_docs", 3) + 1
        arr = [[0.1] * 8 for _ in range(n)]
        payload = json.dumps({"embeddings": arr}).encode()
        return payload, b""


@pytest.mark.unit
def test_bge_prefix_applied(monkeypatch, client: TestClient):
    captured = {"args": None, "n_docs": 3}

    async def fake_cpe(*args, **kwargs):
        # Record full args
        captured["args"] = list(args)
        return _FakeProc(captured)

    # Patch the subprocess creator where it is used
    import tldw_Server_API.app.core.RAG.rag_service.advanced_reranking as ar

    monkeypatch.setattr(ar.asyncio, "create_subprocess_exec", fake_cpe)

    payload = {
        "query": "What do llamas eat?",
        "top_k": 2,
        "passages": [
            {"id": "a", "text": "Llamas eat bananas"},
            {"id": "b", "text": "Llamas in pyjamas"},
            {"id": "c", "text": "A bowl of fruit salad"},
        ],
        # Include 'bge' in model path to trigger auto prefixes
        "model": "/models/bge-small-en-v1.5.gguf",
    }
    resp = client.post("/api/v1/llamacpp/reranking", json=payload)
    assert resp.status_code == 200, resp.text
    args = captured["args"]
    # Flatten to string for easier checking
    flat = " ".join(map(str, args or []))
    assert "query: What do llamas eat?" in flat
    assert "passage: Llamas eat bananas" in flat
