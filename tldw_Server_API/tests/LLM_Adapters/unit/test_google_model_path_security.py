"""Credentialed Google model-path boundary regressions."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

pytestmark = pytest.mark.unit


class _Response:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {
            "candidates": [
                {
                    "content": {"parts": [{"text": "ok"}]},
                    "finishReason": "STOP",
                }
            ]
        }

    def iter_lines(self):
        yield 'data: {"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}'
        yield "data: [DONE]"


class _StreamContext:
    def __enter__(self) -> _Response:
        return _Response()

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del exc_type, exc, traceback
        return False


class _Client:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del exc_type, exc, traceback
        return False

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]):
        del url, headers, json
        return _Response()

    def stream(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> _StreamContext:
        del method, url, headers, json
        return _StreamContext()


def _request(model: str, api_key: str = "trusted-google-key") -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": "hi"}],
        "model": model,
        "api_key": api_key,
        "base_url": "https://google.example/v1beta",
    }


@pytest.mark.parametrize("operation", ["chat", "stream"])
@pytest.mark.parametrize(
    "model",
    [
        "../files",
        "../../v1beta/files#",
        "org\\model",
        "%2e%2e/files",
        "model?key=attacker",
        "model#fragment",
    ],
)
def test_google_rejects_unsafe_model_paths_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    model: str,
) -> None:
    import tldw_Server_API.app.core.LLM_Calls.providers.google_adapter as google_module
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatBadRequestError
    from tldw_Server_API.app.core.LLM_Calls.providers.google_adapter import GoogleAdapter

    monkeypatch.setattr(
        google_module,
        "http_client_factory",
        lambda **_kwargs: pytest.fail("unsafe model must fail before HTTP dispatch"),
    )

    with pytest.raises(ChatBadRequestError, match="model identifier") as exc_info:
        result = getattr(GoogleAdapter(), operation)(_request(model))
        if operation == "stream":
            list(result)

    assert model not in str(exc_info.value)
    assert "trusted-google-key" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("operation", "suffix"),
    [
        ("chat", ":generateContent"),
        ("stream", ":streamGenerateContent?alt=sse"),
    ],
)
def test_google_encodes_valid_model_path_segments(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    suffix: str,
) -> None:
    import tldw_Server_API.app.core.LLM_Calls.providers.google_adapter as google_module
    from tldw_Server_API.app.core.LLM_Calls.providers.google_adapter import GoogleAdapter

    calls: list[str] = []

    class _CaptureClient(_Client):
        def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]):
            del headers, json
            calls.append(url)
            return _Response()

        def stream(
            self,
            method: str,
            url: str,
            *,
            headers: dict[str, str],
            json: dict[str, Any],
        ) -> _StreamContext:
            del method, headers, json
            calls.append(url)
            return _StreamContext()

    monkeypatch.setattr(google_module, "http_client_factory", lambda **_kwargs: _CaptureClient())
    result = getattr(GoogleAdapter(), operation)(_request("org/mødel"))
    if operation == "stream":
        list(result)

    assert calls == [f"https://google.example/v1beta/models/org/m%C3%B8del{suffix}"]


@pytest.mark.concurrent
def test_concurrent_google_model_routes_keep_url_and_key_paired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_Server_API.app.core.LLM_Calls.providers.google_adapter as google_module
    from tldw_Server_API.app.core.LLM_Calls.providers.google_adapter import GoogleAdapter

    calls: list[tuple[str, str]] = []
    lock = threading.Lock()
    both_arrived = threading.Event()
    release = threading.Event()

    class _GatedClient(_Client):
        def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]):
            del json
            with lock:
                calls.append((url, headers["x-goog-api-key"]))
                if len(calls) == 2:
                    both_arrived.set()
            if not release.wait(10):
                raise TimeoutError("concurrent Google calls were not released")
            return _Response()

    monkeypatch.setattr(google_module, "http_client_factory", lambda **_kwargs: _GatedClient())
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(GoogleAdapter().chat, _request("org/model-alpha", "key-alpha")),
            executor.submit(GoogleAdapter().chat, _request("org/model-beta", "key-beta")),
        ]
        try:
            assert both_arrived.wait(10)
        finally:
            release.set()
        assert all(future.result(timeout=10)["object"] == "chat.completion" for future in futures)

    assert len(calls) == 2
    assert set(calls) == {
        ("https://google.example/v1beta/models/org/model-alpha:generateContent", "key-alpha"),
        ("https://google.example/v1beta/models/org/model-beta:generateContent", "key-beta"),
    }


@pytest.mark.concurrent
def test_concurrent_invalid_google_model_cannot_affect_legitimate_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_Server_API.app.core.LLM_Calls.providers.google_adapter as google_module
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatBadRequestError
    from tldw_Server_API.app.core.LLM_Calls.providers.google_adapter import GoogleAdapter

    calls: list[tuple[str, str]] = []
    arrived = threading.Event()
    release = threading.Event()

    class _GatedClient(_Client):
        def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]):
            del json
            calls.append((url, headers["x-goog-api-key"]))
            arrived.set()
            if not release.wait(10):
                raise TimeoutError("legitimate Google call was not released")
            return _Response()

    monkeypatch.setattr(google_module, "http_client_factory", lambda **_kwargs: _GatedClient())
    with ThreadPoolExecutor(max_workers=2) as executor:
        legitimate = executor.submit(
            GoogleAdapter().chat,
            _request("org/legitimate-model", "legitimate-key"),
        )
        try:
            assert arrived.wait(10)
            malicious = executor.submit(
                GoogleAdapter().chat,
                _request("../../credential-admin", "must-not-dispatch"),
            )
            with pytest.raises(ChatBadRequestError):
                malicious.result(timeout=5)
        finally:
            release.set()
        assert legitimate.result(timeout=10)["object"] == "chat.completion"

    assert calls == [
        (
            "https://google.example/v1beta/models/org/legitimate-model:generateContent",
            "legitimate-key",
        )
    ]


@pytest.mark.concurrent
def test_concurrent_control_header_cannot_affect_legitimate_google_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_Server_API.app.core.LLM_Calls.providers.google_adapter as google_module
    from tldw_Server_API.app.core.LLM_Calls.providers.google_adapter import GoogleAdapter

    calls: list[tuple[str, str, str]] = []
    arrived = threading.Event()
    release = threading.Event()

    class _GatedClient(_Client):
        def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]):
            del json
            calls.append(
                (url, headers["x-goog-api-key"], headers["X-Provider-Extension"])
            )
            arrived.set()
            if not release.wait(10):
                raise TimeoutError("legitimate Google header call was not released")
            return _Response()

    monkeypatch.setattr(google_module, "http_client_factory", lambda **_kwargs: _GatedClient())
    legitimate_request = _request("org/legitimate-model", "legitimate-key") | {
        "extra_headers": {"X-Provider-Extension": "legitimate-value"}
    }
    malicious_request = _request("org/malicious-model", "malicious-key") | {
        "extra_headers": {
            "X-Provider-Extension": "attacker\r\nAuthorization: Bearer stolen"
        }
    }
    with ThreadPoolExecutor(max_workers=2) as executor:
        legitimate = executor.submit(GoogleAdapter().chat, legitimate_request)
        try:
            assert arrived.wait(10)
            malicious = executor.submit(GoogleAdapter().chat, malicious_request)
            with pytest.raises(ValueError, match="header value") as exc_info:
                malicious.result(timeout=5)
            assert "attacker" not in str(exc_info.value)
            assert "stolen" not in str(exc_info.value)
        finally:
            release.set()
        assert legitimate.result(timeout=10)["object"] == "chat.completion"

    assert calls == [
        (
            "https://google.example/v1beta/models/org/legitimate-model:generateContent",
            "legitimate-key",
            "legitimate-value",
        )
    ]
