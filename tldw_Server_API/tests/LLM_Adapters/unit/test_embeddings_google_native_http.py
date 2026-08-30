import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import httpx
import pytest


@pytest.mark.unit
def test_google_embeddings_adapter_native_http_single(monkeypatch):
    monkeypatch.setenv("LLM_EMBEDDINGS_NATIVE_HTTP_GOOGLE", "1")

    from tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter import (
        GoogleEmbeddingsAdapter,
    )

    adapter = GoogleEmbeddingsAdapter()

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"embedding": {"values": [0.5, 0.6]}}

    def _fake_post(url, params=None, json=None, headers=None, **kwargs):  # noqa: ANN001, ARG001
        return _Resp()

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        def post(self, url, params=None, json=None, headers=None, **kwargs):  # noqa: ANN001, ARG001
            return _fake_post(url, params=params, json=json, headers=headers, **kwargs)

    def _fake_create_client(*args, **kwargs):  # noqa: ANN001, ARG001
        return _FakeClient()

    with patch(
        "tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter.create_client",
        _fake_create_client,
    ):
        out = adapter.embed({"input": "hello", "model": "text-embedding-004", "api_key": "g"})
        assert isinstance(out, dict)
        assert out.get("data") and out["data"][0]["embedding"] == [0.5, 0.6]


@pytest.mark.unit
def test_google_embeddings_adapter_native_http_multi(monkeypatch):
    monkeypatch.setenv("LLM_EMBEDDINGS_NATIVE_HTTP_GOOGLE", "1")

    from tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter import (
        GoogleEmbeddingsAdapter,
    )

    adapter = GoogleEmbeddingsAdapter()

    seq = [
        {"embedding": {"values": [0.1, 0.2]}},
        {"embedding": {"values": [0.3, 0.4]}},
    ]
    calls = {"i": 0}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            i = calls["i"]
            calls["i"] += 1
            return seq[i]

    def _fake_post(url, params=None, json=None, headers=None, **kwargs):  # noqa: ANN001, ARG001
        return _Resp()

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        def post(self, url, params=None, json=None, headers=None, **kwargs):  # noqa: ANN001, ARG001
            return _fake_post(url, params=params, json=json, headers=headers, **kwargs)

    def _fake_create_client(*args, **kwargs):  # noqa: ANN001, ARG001
        return _FakeClient()

    with patch(
        "tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter.create_client",
        _fake_create_client,
    ):
        out = adapter.embed({"input": ["a", "b"], "model": "text-embedding-004", "api_key": "g"})
        assert isinstance(out, dict)
        embs = [d["embedding"] for d in out.get("data", [])]
        assert embs == [[0.1, 0.2], [0.3, 0.4]]
        assert calls["i"] == 2


@pytest.mark.unit
def test_resolved_google_list_uses_one_pinned_batch_request_without_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_EMBEDDINGS_NATIVE_HTTP_GOOGLE", raising=False)
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "https://ambient-attacker.example/v1")

    import tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter as module
    from tldw_Server_API.app.core.AuthNZ.byok_config import (
        runtime_base_url_override_provenance,
    )
    from tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter import (
        GoogleEmbeddingsAdapter,
    )

    client_options: list[dict[str, object]] = []
    requests: list[tuple[str, dict[str, object]]] = []

    class Response:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "embeddings": [
                    {"values": [0.1, 0.2]},
                    {"values": [0.3, 0.4]},
                ]
            }

    class Client:
        def __enter__(self) -> "Client":
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        def post(self, url: str, **kwargs: object) -> Response:
            requests.append((url, kwargs))
            return Response()

    def create_client(**kwargs: object) -> Client:
        client_options.append(kwargs)
        return Client()

    monkeypatch.setattr(module, "create_client", create_client)

    result = GoogleEmbeddingsAdapter().embed(
        {
            "input": ["note one", "note two"],
            "model": "text-embedding-004",
            "api_key": "trusted-key",
            "base_url": "https://pinned-google.example/v1",
            "credentials_resolved": True,
            "_runtime_base_url_override": runtime_base_url_override_provenance(),
        }
    )

    assert [item["embedding"] for item in result["data"]] == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]
    assert client_options == [{"timeout": 60.0, "follow_redirects": False}]
    assert requests == [
        (
            "https://pinned-google.example/v1/models/text-embedding-004:batchEmbedContents",
            {
                "headers": {
                    "Content-Type": "application/json",
                    "x-goog-api-key": "trusted-key",
                },
                "json": {
                    "requests": [
                        {
                            "model": "models/text-embedding-004",
                            "content": {"parts": [{"text": "note one"}]},
                        },
                        {
                            "model": "models/text-embedding-004",
                            "content": {"parts": [{"text": "note two"}]},
                        },
                    ]
                },
            },
        )
    ]
    assert "ambient-attacker" not in repr(requests)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status_code", "location"),
    [
        (307, "https://cross-origin.example/receive"),
        (308, "https://cross-origin.example/receive"),
        (307, "https://pinned-google.example/v1/redirected"),
    ],
)
def test_resolved_google_request_rejects_redirect_without_replaying_note_text(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    location: str,
) -> None:
    monkeypatch.delenv("LLM_EMBEDDINGS_NATIVE_HTTP_GOOGLE", raising=False)

    import tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter as module
    from tldw_Server_API.app.core.AuthNZ.byok_config import (
        runtime_base_url_override_provenance,
    )
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatProviderError
    from tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter import (
        GoogleEmbeddingsAdapter,
    )

    seen: list[tuple[str, str, bytes]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.host, request.url.path, request.content))
        if len(seen) == 1:
            return httpx.Response(status_code, headers={"location": location}, json={})
        return httpx.Response(
            200,
            json={
                "embeddings": [
                    {"values": [0.1, 0.2]},
                    {"values": [0.3, 0.4]},
                ]
            },
        )

    def create_client(**kwargs: object) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(module, "create_client", create_client)

    with pytest.raises(ChatProviderError):
        GoogleEmbeddingsAdapter().embed(
            {
                "input": ["note text must stay pinned", "second note text"],
                "model": "text-embedding-004",
                "api_key": "trusted-key",
                "base_url": "https://pinned-google.example/v1",
                "credentials_resolved": True,
                "_runtime_base_url_override": runtime_base_url_override_provenance(),
            }
        )

    assert [host for host, _path, _body in seen] == ["pinned-google.example"]
    assert seen[0][1].endswith(":batchEmbedContents")
    assert b"note text must stay pinned" in seen[0][2]


@pytest.mark.unit
def test_resolved_google_batch_error_never_fans_out_to_per_input_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_EMBEDDINGS_NATIVE_HTTP_GOOGLE", raising=False)

    import tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter as module
    from tldw_Server_API.app.core.AuthNZ.byok_config import (
        runtime_base_url_override_provenance,
    )
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatProviderError
    from tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter import (
        GoogleEmbeddingsAdapter,
    )

    urls: list[str] = []

    class Response:
        status_code = 500

        def raise_for_status(self) -> None:
            raise RuntimeError("raw-google-batch-error")

    class Client:
        def __enter__(self) -> "Client":
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        def post(self, url: str, **_kwargs: object) -> Response:
            urls.append(url)
            return Response()

    monkeypatch.setattr(module, "create_client", lambda **_kwargs: Client())

    with pytest.raises(ChatProviderError) as exc_info:
        GoogleEmbeddingsAdapter().embed(
            {
                "input": ["first note", "second note"],
                "model": "text-embedding-004",
                "api_key": "trusted-key",
                "base_url": "https://pinned-google.example/v1",
                "credentials_resolved": True,
                "_runtime_base_url_override": runtime_base_url_override_provenance(),
            }
        )

    assert urls == [
        "https://pinned-google.example/v1/models/text-embedding-004:batchEmbedContents"
    ]
    assert "raw-google-batch-error" not in str(exc_info.value)


@pytest.mark.unit
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
def test_google_embeddings_reject_unsafe_model_paths_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
) -> None:
    monkeypatch.setenv("LLM_EMBEDDINGS_NATIVE_HTTP_GOOGLE", "1")

    import tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter as module
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatBadRequestError
    from tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter import (
        GoogleEmbeddingsAdapter,
    )

    monkeypatch.setattr(
        module,
        "create_client",
        lambda **_kwargs: pytest.fail("unsafe model must fail before HTTP dispatch"),
    )

    with pytest.raises(ChatBadRequestError, match="model identifier") as exc_info:
        GoogleEmbeddingsAdapter().embed(
            {"input": "hi", "model": model, "api_key": "trusted-key"}
        )

    assert model not in str(exc_info.value)
    assert "trusted-key" not in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.concurrent
def test_concurrent_google_embedding_model_routes_keep_url_and_key_paired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_EMBEDDINGS_NATIVE_HTTP_GOOGLE", "1")

    import tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter as module
    from tldw_Server_API.app.core.AuthNZ.byok_config import (
        runtime_base_url_override_provenance,
    )
    from tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter import (
        GoogleEmbeddingsAdapter,
    )

    calls: list[tuple[str, str, str]] = []
    lock = threading.Lock()
    both_arrived = threading.Event()
    release = threading.Event()

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"embedding": {"values": [0.1, 0.2]}}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> bool:
            del exc_type, exc, traceback
            return False

        def post(self, url, params=None, headers=None, json=None, **_kwargs):
            assert params is None
            assert "key=" not in url
            with lock:
                calls.append(
                    (
                        url,
                        headers["x-goog-api-key"],
                        json["content"]["parts"][0]["text"],
                    )
                )
                if len(calls) == 2:
                    both_arrived.set()
            if not release.wait(10):
                raise TimeoutError("concurrent Google embedding calls were not released")
            return _Resp()

    monkeypatch.setattr(module, "create_client", lambda **_kwargs: _Client())
    adapter = GoogleEmbeddingsAdapter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                adapter.embed,
                {
                    "input": "alpha",
                    "model": "model-alpha",
                    "api_key": "key-alpha",
                    "base_url": "https://google-alpha.example/v1",
                    "credentials_resolved": True,
                    "_runtime_base_url_override": runtime_base_url_override_provenance(),
                },
            ),
            executor.submit(
                adapter.embed,
                {
                    "input": "beta",
                    "model": "model-beta",
                    "api_key": "key-beta",
                    "base_url": "https://google-beta.example/v1",
                    "credentials_resolved": True,
                    "_runtime_base_url_override": runtime_base_url_override_provenance(),
                },
            ),
        ]
        try:
            assert both_arrived.wait(10)
        finally:
            release.set()
        assert all(future.result(timeout=10)["data"] for future in futures)

    assert len(calls) == 2
    assert set(calls) == {
        (
            "https://google-alpha.example/v1/models/model-alpha:embedContent",
            "key-alpha",
            "alpha",
        ),
        (
            "https://google-beta.example/v1/models/model-beta:embedContent",
            "key-beta",
            "beta",
        ),
    }


@pytest.mark.unit
@pytest.mark.concurrent
def test_concurrent_invalid_google_embedding_cannot_affect_legitimate_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_EMBEDDINGS_NATIVE_HTTP_GOOGLE", "1")

    import tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter as module
    from tldw_Server_API.app.core.AuthNZ.byok_config import (
        runtime_base_url_override_provenance,
    )
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatBadRequestError
    from tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter import (
        GoogleEmbeddingsAdapter,
    )

    calls = []
    arrived = threading.Event()
    release = threading.Event()

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"embedding": {"values": [0.1, 0.2]}}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, url, params=None, headers=None, json=None, **_kwargs):
            calls.append((url, params, headers["x-goog-api-key"], json))
            arrived.set()
            if not release.wait(10):
                raise TimeoutError("legitimate Google embedding call was not released")
            return _Resp()

    monkeypatch.setattr(module, "create_client", lambda **_kwargs: _Client())
    provenance = runtime_base_url_override_provenance()
    adapter = GoogleEmbeddingsAdapter()
    legitimate = {
        "input": "legitimate",
        "model": "embedding-model",
        "api_key": "legitimate-key",
        "base_url": "https://legitimate-google.example/v1",
        "credentials_resolved": True,
        "_runtime_base_url_override": provenance,
    }
    malicious = legitimate | {
        "model": "../../credential-admin",
        "api_key": "must-not-dispatch",
        "base_url": "https://malicious-google.example/v1",
    }
    with ThreadPoolExecutor(max_workers=2) as executor:
        legitimate_future = executor.submit(adapter.embed, legitimate)
        try:
            assert arrived.wait(10)
            malicious_future = executor.submit(adapter.embed, malicious)
            with pytest.raises(ChatBadRequestError):
                malicious_future.result(timeout=5)
        finally:
            release.set()
        assert legitimate_future.result(timeout=10)["data"]

    assert len(calls) == 1
    assert calls[0][0] == (
        "https://legitimate-google.example/v1/models/embedding-model:embedContent"
    )
    assert calls[0][2] == "legitimate-key"


@pytest.mark.unit
@pytest.mark.parametrize(
    "model",
    ["gemini-embedding-001", "models/gemini-embedding-001"],
)
def test_google_embeddings_uses_one_models_prefix_and_header_auth(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
) -> None:
    monkeypatch.setenv("LLM_EMBEDDINGS_NATIVE_HTTP_GOOGLE", "1")
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "https://google.example/v1")

    import tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter as module
    from tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter import (
        GoogleEmbeddingsAdapter,
    )

    calls = []

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"embedding": {"values": [0.1, 0.2]}}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> bool:
            del exc_type, exc, traceback
            return False

        def post(self, url, params=None, headers=None, json=None, **_kwargs):
            calls.append((url, params, headers, json))
            return _Resp()

    monkeypatch.setattr(module, "create_client", lambda **_kwargs: _Client())
    assert GoogleEmbeddingsAdapter().embed(
        {"input": "hello", "model": model, "api_key": "trusted-key"}
    )["data"]

    assert len(calls) == 1
    url, params, headers, _payload = calls[0]
    assert url == "https://google.example/v1/models/gemini-embedding-001:embedContent"
    assert params is None
    assert headers == {"Content-Type": "application/json", "x-goog-api-key": "trusted-key"}
    assert "trusted-key" not in url


@pytest.mark.unit
@pytest.mark.parametrize("status_code", [401, 403])
def test_resolved_google_auth_failure_never_retries_with_query_key(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    monkeypatch.delenv("LLM_EMBEDDINGS_NATIVE_HTTP_GOOGLE", raising=False)
    monkeypatch.setenv("GOOGLE_EMBEDDINGS_QUERY_KEY_FALLBACK", "1")

    import tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter as module
    from tldw_Server_API.app.core.AuthNZ.byok_config import (
        runtime_base_url_override_provenance,
    )
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatProviderError
    from tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter import (
        GoogleEmbeddingsAdapter,
    )

    calls: list[tuple[dict[str, str] | None, dict[str, str]]] = []

    class Response:
        def raise_for_status(self) -> None:
            raise RuntimeError("raw-resolved-google-auth-error")

        @property
        def status_code(self) -> int:
            return status_code

    class Client:
        def __enter__(self) -> "Client":
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        def post(
            self,
            _url: str,
            *,
            params: dict[str, str] | None = None,
            headers: dict[str, str],
            **_kwargs: object,
        ) -> Response:
            calls.append((params, dict(headers)))
            return Response()

    monkeypatch.setattr(module, "create_client", lambda **_kwargs: Client())

    with pytest.raises(ChatProviderError):
        GoogleEmbeddingsAdapter().embed(
            {
                "input": "note text",
                "model": "text-embedding-004",
                "api_key": "trusted-key",
                "base_url": "https://google-compatible.example/v1",
                "credentials_resolved": True,
                "_runtime_base_url_override": runtime_base_url_override_provenance(),
            }
        )

    assert calls == [
        (
            None,
            {
                "Content-Type": "application/json",
                "x-goog-api-key": "trusted-key",
            },
        )
    ]


@pytest.mark.unit
@pytest.mark.parametrize("first_status", [401, 403])
def test_google_custom_endpoint_query_key_fallback_requires_opt_in_and_auth_failure(
    monkeypatch: pytest.MonkeyPatch,
    first_status: int,
) -> None:
    monkeypatch.setenv("LLM_EMBEDDINGS_NATIVE_HTTP_GOOGLE", "1")
    monkeypatch.setenv("GOOGLE_EMBEDDINGS_QUERY_KEY_FALLBACK", "1")
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "https://google-compatible.example/v1")

    import tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter as module
    from tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter import (
        GoogleEmbeddingsAdapter,
    )

    calls = []

    class _Resp:
        def __init__(self, status_code):
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code} raw-body-sentinel")

        def json(self):
            return {"embedding": {"values": [0.1, 0.2]}}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, url, params=None, headers=None, json=None, **_kwargs):
            calls.append((url, params, dict(headers or {}), json))
            return _Resp(first_status if len(calls) == 1 else 200)

    monkeypatch.setattr(module, "create_client", lambda **_kwargs: _Client())

    result = GoogleEmbeddingsAdapter().embed(
        {"input": "hello", "model": "embedding-model", "api_key": "trusted-key"}
    )

    assert result["data"]
    assert len(calls) == 2
    assert calls[0][1] is None
    assert calls[0][2]["x-goog-api-key"] == "trusted-key"
    assert calls[1][1] == {"key": "trusted-key"}
    assert "x-goog-api-key" not in calls[1][2]
    assert all("trusted-key" not in call[0] for call in calls)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("fallback_enabled", "base_url", "status_code"),
    [
        (False, "https://google-compatible.example/v1", 401),
        (True, "https://generativelanguage.googleapis.com/v1", 401),
        (True, "https://google-compatible.example/v1", 400),
        (True, "https://google-compatible.example/v1", 429),
        (True, "https://google-compatible.example/v1", 500),
    ],
)
def test_google_query_key_fallback_never_runs_outside_explicit_custom_auth_case(
    monkeypatch: pytest.MonkeyPatch,
    fallback_enabled: bool,
    base_url: str,
    status_code: int,
) -> None:
    monkeypatch.setenv("LLM_EMBEDDINGS_NATIVE_HTTP_GOOGLE", "1")
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", base_url)
    if fallback_enabled:
        monkeypatch.setenv("GOOGLE_EMBEDDINGS_QUERY_KEY_FALLBACK", "1")
    else:
        monkeypatch.delenv("GOOGLE_EMBEDDINGS_QUERY_KEY_FALLBACK", raising=False)

    import tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter as module
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatProviderError
    from tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter import (
        GoogleEmbeddingsAdapter,
    )

    calls = []

    class _Resp:
        def raise_for_status(self):
            raise RuntimeError(f"HTTP {status_code} raw-body-sentinel")

        @property
        def status_code(self):
            return status_code

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, url, params=None, headers=None, json=None, **_kwargs):
            calls.append((url, params, dict(headers or {}), json))
            return _Resp()

    monkeypatch.setattr(module, "create_client", lambda **_kwargs: _Client())

    with pytest.raises(ChatProviderError) as exc_info:
        GoogleEmbeddingsAdapter().embed(
            {"input": "hello", "model": "embedding-model", "api_key": "trusted-key"}
        )

    assert len(calls) == 1
    assert calls[0][1] is None
    assert calls[0][2]["x-goog-api-key"] == "trusted-key"
    assert "raw-body-sentinel" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.unit
@pytest.mark.concurrent
def test_concurrent_resolved_google_never_retries_while_unresolved_custom_call_can(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolved auth stays header-only beside an ordinary custom retry."""
    monkeypatch.setenv("LLM_EMBEDDINGS_NATIVE_HTTP_GOOGLE", "1")
    monkeypatch.setenv("GOOGLE_EMBEDDINGS_QUERY_KEY_FALLBACK", "1")
    monkeypatch.setenv(
        "GOOGLE_GEMINI_BASE_URL",
        "https://google-compatible.example/v1",
    )

    import tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter as module
    from tldw_Server_API.app.core.AuthNZ.byok_config import (
        runtime_base_url_override_provenance,
    )
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatProviderError
    from tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter import (
        GoogleEmbeddingsAdapter,
    )

    calls: list[tuple[str, dict[str, str] | None, dict[str, str], str]] = []
    lock = threading.Lock()
    both_initial_arrived = threading.Event()
    release = threading.Event()

    class _Response:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError("raw-google-auth-body-sentinel")

        def json(self) -> dict[str, object]:
            return {"embedding": {"values": [0.1, 0.2]}}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, url, params=None, headers=None, json=None, **_kwargs):
            label = json["content"]["parts"][0]["text"]
            call = (url, params, dict(headers or {}), label)
            with lock:
                calls.append(call)
                initial_labels = {item[3] for item in calls if item[1] is None}
                if initial_labels == {"official", "custom"}:
                    both_initial_arrived.set()
            if params is None and not release.wait(10):
                raise TimeoutError("concurrent Google auth calls were not released")
            return _Response(200 if params is not None else 401)

    monkeypatch.setattr(module, "create_client", lambda **_kwargs: _Client())
    provenance = runtime_base_url_override_provenance()
    adapter = GoogleEmbeddingsAdapter()

    def _invoke(
        label: str,
        key: str,
        *,
        resolved_base_url: str | None = None,
    ):
        request = {
            "input": label,
            "model": f"embedding-{label}",
            "api_key": key,
        }
        if resolved_base_url is not None:
            request.update(
                {
                    "base_url": resolved_base_url,
                    "credentials_resolved": True,
                    "_runtime_base_url_override": provenance,
                }
            )
        return adapter.embed(request)

    with ThreadPoolExecutor(max_workers=2) as executor:
        official = executor.submit(
            _invoke,
            "official",
            "official-secret-key",
            resolved_base_url="https://generativelanguage.googleapis.com./v1",
        )
        custom = executor.submit(
            _invoke,
            "custom",
            "custom-secret-key",
        )
        try:
            assert both_initial_arrived.wait(10)
        finally:
            release.set()
        with pytest.raises(ChatProviderError):
            official.result(timeout=10)
        assert custom.result(timeout=10)["data"]

    official_calls = [call for call in calls if call[3] == "official"]
    custom_calls = [call for call in calls if call[3] == "custom"]
    assert len(official_calls) == 1
    assert official_calls[0][1] is None
    assert official_calls[0][2]["x-goog-api-key"] == "official-secret-key"
    assert "official-secret-key" not in official_calls[0][0]
    assert len(custom_calls) == 2
    assert custom_calls[0][1] is None
    assert custom_calls[1][1] == {"key": "custom-secret-key"}
