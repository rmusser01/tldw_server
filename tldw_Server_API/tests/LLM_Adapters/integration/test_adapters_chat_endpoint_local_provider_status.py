"""A local provider's upstream 401, 403 and 429 get the chat answer a remote provider's get.

The transport is stubbed on the adapter class, no socket is opened, and the blocker fails
the test if a non-loopback DNS lookup or connect is attempted.
"""

from __future__ import annotations

import httpx
import pytest

# Ensure chat fixtures (client/auth) are registered as pytest fixtures
from tldw_Server_API.tests._plugins import chat_fixtures as _chat_pl  # noqa: F401

AUTH = (502, "provider_authentication_failed")
UNAVAILABLE = (502, "provider_unavailable")


@pytest.fixture(autouse=True)
def _env(monkeypatch, no_live_network):
    monkeypatch.setenv("STREAMS_UNIFIED", "1")
    monkeypatch.setenv("LOGURU_LEVEL", "ERROR")
    monkeypatch.delenv("TEST_MODE", raising=False)


def _status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://127.0.0.1:8080/v1/chat/completions")
    response = httpx.Response(status, request=request, content=b'{"error":{"message":"upstream secret"}}')
    return httpx.HTTPStatusError("upstream secret", request=request, response=response)


class _Response:
    def __init__(self, status: int):
        self._status = status

    def raise_for_status(self):
        raise _status_error(self._status)

    def close(self):
        pass


class _StreamContext:
    def __init__(self, status: int):
        self._response = _Response(status)

    def __enter__(self):
        return self._response

    def __exit__(self, *_exc):
        return False


def _post(authenticated_client, monkeypatch, status: int, stream: bool):
    from tldw_Server_API.app.core.LLM_Calls.providers import local_adapters as local_mod

    monkeypatch.setattr(local_mod.LlamaCppAdapter, "http_fetcher", staticmethod(lambda **_k: _Response(status)))
    monkeypatch.setattr(
        local_mod.LlamaCppAdapter, "http_streamer", staticmethod(lambda **_k: _StreamContext(status))
    )
    return authenticated_client.post(
        "/api/v1/chat/completions",
        json={
            "api_provider": "llama.cpp",
            "model": "local-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": stream,
        },
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("status", "expected"),
    [(401, AUTH), (403, AUTH), (429, (429, None)), (404, (400, None))],
    ids=["unauthorized", "forbidden", "rate-limit", "unpulled-model"],
)
def test_local_provider_reply_not_streamed(monkeypatch, authenticated_client, status, expected):
    response = _post(authenticated_client, monkeypatch, status, stream=False)

    assert response.status_code == expected[0], response.text
    detail = response.json()["detail"]
    if expected[1] is not None:
        assert detail["error_code"] == expected[1]
    elif status == 429:
        assert detail == "Rate limit exceeded. Please retry."
    else:
        assert detail == "Invalid request."
    assert "upstream secret" not in response.text


@pytest.mark.integration
@pytest.mark.parametrize(
    ("status", "expected"),
    [(401, AUTH), (403, AUTH), (429, UNAVAILABLE), (404, UNAVAILABLE)],
    ids=["unauthorized", "forbidden", "rate-limit", "unpulled-model"],
)
def test_local_provider_reply_streamed(monkeypatch, authenticated_client, status, expected):
    response = _post(authenticated_client, monkeypatch, status, stream=True)

    assert response.status_code == expected[0], response.text
    assert response.json()["detail"]["error_code"] == expected[1]
    assert "upstream secret" not in response.text
