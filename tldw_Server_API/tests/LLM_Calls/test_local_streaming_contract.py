import json
from typing import Iterable, Union

import pytest

from tldw_Server_API.app.core.LLM_Calls.providers.local_adapters import (
    _chat_with_openai_compatible_local_server,
)
from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope


class DummyResponse:
    """Minimal synchronous response stub for httpx streaming tests."""

    def __init__(self, lines: Iterable[Union[str, bytes]]):
        self._lines = list(lines)
        self._close_calls = 0

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self):
        for line in self._lines:
            yield line

    def iter_text(self):
        for line in self._lines:
            text = line.decode("utf-8", errors="replace") if isinstance(line, bytes) else str(line)
            yield text

    def close(self) -> None:
        self._close_calls += 1

    @property
    def close_calls(self) -> int:
        return self._close_calls


class DummyStreamContext:
    """Context manager that mimics httpx.Client.stream(...)."""

    def __init__(self, response: DummyResponse):
        self._response = response

    def __enter__(self) -> DummyResponse:
        return self._response

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._response.close()
        # Propagate exceptions so the caller can handle them.
        return False


@pytest.mark.unit
def test_local_streaming_normalizes_sse_and_closes_client():
    # Simulate a local provider that emits a raw text line and an SSE-formatted line,
    # without sending a [DONE] sentinel. The helper should normalize the output and
    # append the final sentinel for us.
    raw_lines = [
        "event: stream-start",
        b"Hello",  # plain data without SSE framing
        "id: chunk-1",
        'data: {"choices":[{"delta":{"content":" world"}}]}',  # already framed
        "retry: 0",
    ]
    dummy_response = DummyResponse(raw_lines)

    client_closed = {"called": False}

    def fake_stream(method, url, headers=None, json=None, timeout=None):
        assert method == "POST"
        assert json["stream"] is True
        return DummyStreamContext(dummy_response)

    def fake_close():
        client_closed["called"] = True

    def checked_streamer(**kwargs):
        assert kwargs["configured_endpoint"].matches(kwargs["url"])
        return fake_stream(
            kwargs["method"],
            kwargs["url"],
            headers=kwargs.get("headers"),
            json=kwargs.get("json"),
            timeout=kwargs.get("timeout"),
        )

    generator = _chat_with_openai_compatible_local_server(
        api_base_url="http://fake.local",
        model_name="local-model",
        input_data=[{"role": "user", "content": "Ping"}],
        streaming=True,
        provider_name="TestLocalProvider",
        http_streamer=checked_streamer,
        configured_endpoint_scope=ConfiguredEndpointScope.from_url("http://fake.local"),
    )

    chunks = list(generator)

    # Expect two payload chunks plus the appended [DONE] sentinel.
    assert len(chunks) == 3

    for chunk in chunks[:-1]:
        assert chunk.startswith("data: ")
        assert chunk.endswith("\n\n")
        payload = json.loads(chunk[len("data: ") : -2])
        assert "choices" in payload
        assert payload["choices"][0]["delta"]["content"] in {"Hello", " world"}

    assert chunks[-1] == "data: [DONE]\n\n"
    assert dummy_response.close_calls >= 1
    assert client_closed["called"] is False
