import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import httpx
import pytest


@pytest.mark.unit
def test_huggingface_embeddings_adapter_native_http_single(monkeypatch):
    monkeypatch.setenv("LLM_EMBEDDINGS_NATIVE_HTTP_HUGGINGFACE", "1")

    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_embeddings_adapter import (
        HuggingFaceEmbeddingsAdapter,
    )

    adapter = HuggingFaceEmbeddingsAdapter()

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            # HF may return [[...]] for single input
            return [[0.1, 0.2]]

    def _fake_post(url, headers=None, json=None, **kwargs):  # noqa: ANN001, ARG001
        return _Resp()

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        def post(self, url, headers=None, json=None, **kwargs):  # noqa: ANN001, ARG001
            return _fake_post(url, headers=headers, json=json, **kwargs)

    def _fake_create_client(*args, **kwargs):  # noqa: ANN001, ARG001
        return _FakeClient()

    with patch(
        "tldw_Server_API.app.core.LLM_Calls.providers.huggingface_embeddings_adapter.create_client",
        _fake_create_client,
    ):
        out = adapter.embed({"input": "hi", "model": "sentence-transformers/all-MiniLM-L6-v2", "api_key": "k"})
        assert isinstance(out, dict)
        assert out.get("data") and out["data"][0]["embedding"] == [0.1, 0.2]


@pytest.mark.unit
def test_huggingface_embeddings_accepts_default_style_leading_model_slash(monkeypatch):
    monkeypatch.setenv("LLM_EMBEDDINGS_NATIVE_HTTP_HUGGINGFACE", "1")

    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_embeddings_adapter as module
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_embeddings_adapter import (
        HuggingFaceEmbeddingsAdapter,
    )

    calls = []

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return [[0.1, 0.2]]

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, url, **_kwargs):
            calls.append(url)
            return _Resp()

    monkeypatch.setattr(module, "create_client", lambda **_kwargs: _Client())

    result = HuggingFaceEmbeddingsAdapter().embed(
        {"input": "hello", "model": "/Qwen/Qwen3-235B-A22B", "api_key": "key"}
    )

    assert result["data"]
    assert calls == [
        "https://api-inference.huggingface.co/models/Qwen/Qwen3-235B-A22B"
    ]


@pytest.mark.unit
def test_huggingface_embeddings_adapter_native_http_multi(monkeypatch):
    monkeypatch.setenv("LLM_EMBEDDINGS_NATIVE_HTTP_HUGGINGFACE", "1")

    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_embeddings_adapter import (
        HuggingFaceEmbeddingsAdapter,
    )

    adapter = HuggingFaceEmbeddingsAdapter()

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return [[0.1, 0.2], [0.3, 0.4]]

    def _fake_post(url, headers=None, json=None, **kwargs):  # noqa: ANN001, ARG001
        return _Resp()

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        def post(self, url, headers=None, json=None, **kwargs):  # noqa: ANN001, ARG001
            return _fake_post(url, headers=headers, json=json, **kwargs)

    def _fake_create_client(*args, **kwargs):  # noqa: ANN001, ARG001
        return _FakeClient()

    with patch(
        "tldw_Server_API.app.core.LLM_Calls.providers.huggingface_embeddings_adapter.create_client",
        _fake_create_client,
    ):
        out = adapter.embed(
            {"input": ["a", "b"], "model": "sentence-transformers/all-MiniLM-L6-v2", "api_key": "k"}
        )
        assert isinstance(out, dict)
        embs = [d["embedding"] for d in out.get("data", [])]
        assert embs == [[0.1, 0.2], [0.3, 0.4]]


@pytest.mark.unit
def test_resolved_huggingface_request_forces_pinned_native_client_without_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_EMBEDDINGS_NATIVE_HTTP_HUGGINGFACE", raising=False)
    monkeypatch.setenv("HUGGINGFACE_INFERENCE_BASE_URL", "https://ambient-attacker.example/models")

    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_embeddings_adapter as module
    from tldw_Server_API.app.core.AuthNZ.byok_config import (
        runtime_base_url_override_provenance,
    )
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_embeddings_adapter import (
        HuggingFaceEmbeddingsAdapter,
    )

    client_options: list[dict[str, object]] = []
    urls: list[str] = []

    class Response:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> list[list[float]]:
            return [[0.1, 0.2]]

    class Client:
        def __enter__(self) -> "Client":
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        def post(self, url: str, **_kwargs: object) -> Response:
            urls.append(url)
            return Response()

    def create_client(**kwargs: object) -> Client:
        client_options.append(kwargs)
        return Client()

    monkeypatch.setattr(module, "create_client", create_client)

    result = HuggingFaceEmbeddingsAdapter().embed(
        {
            "input": ["note text"],
            "model": "org/model",
            "api_key": "trusted-key",
            "base_url": "https://pinned-hf.example/models",
            "credentials_resolved": True,
            "_runtime_base_url_override": runtime_base_url_override_provenance(),
        }
    )

    assert result["data"]
    assert client_options == [{"timeout": 60.0, "follow_redirects": False}]
    assert urls == ["https://pinned-hf.example/models/org/model"]
    assert "ambient-attacker" not in repr(urls)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status_code", "location"),
    [
        (307, "https://cross-origin.example/receive"),
        (308, "https://cross-origin.example/receive"),
        (307, "https://pinned-hf.example/models/redirected"),
    ],
)
def test_resolved_huggingface_request_rejects_redirect_without_replaying_note_text(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    location: str,
) -> None:
    monkeypatch.delenv("LLM_EMBEDDINGS_NATIVE_HTTP_HUGGINGFACE", raising=False)

    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_embeddings_adapter as module
    from tldw_Server_API.app.core.AuthNZ.byok_config import (
        runtime_base_url_override_provenance,
    )
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatProviderError
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_embeddings_adapter import (
        HuggingFaceEmbeddingsAdapter,
    )

    seen: list[tuple[str, bytes]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.host, request.content))
        if len(seen) == 1:
            return httpx.Response(status_code, headers={"location": location}, json={})
        return httpx.Response(200, json=[[0.1, 0.2]])

    def create_client(**kwargs: object) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(module, "create_client", create_client)

    with pytest.raises(ChatProviderError):
        HuggingFaceEmbeddingsAdapter().embed(
            {
                "input": ["note text must stay pinned"],
                "model": "org/model",
                "api_key": "trusted-key",
                "base_url": "https://pinned-hf.example/models",
                "credentials_resolved": True,
                "_runtime_base_url_override": runtime_base_url_override_provenance(),
            }
        )

    assert [host for host, _body in seen] == ["pinned-hf.example"]
    assert b"note text must stay pinned" in seen[0][1]


@pytest.mark.unit
@pytest.mark.parametrize(
    "model",
    [
        "../admin",
        "org/../../api/whoami-v2#",
        "org\\model",
        "org/%2e%2e/admin",
        "org/model?alt=admin",
        "org/model#fragment",
        "//org/model",
    ],
)
def test_huggingface_embeddings_reject_unsafe_model_paths_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
) -> None:
    monkeypatch.setenv("LLM_EMBEDDINGS_NATIVE_HTTP_HUGGINGFACE", "1")

    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_embeddings_adapter as module
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatBadRequestError
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_embeddings_adapter import (
        HuggingFaceEmbeddingsAdapter,
    )

    monkeypatch.setattr(
        module,
        "create_client",
        lambda **_kwargs: pytest.fail("unsafe model must fail before HTTP dispatch"),
    )

    with pytest.raises(ChatBadRequestError, match="model identifier") as exc_info:
        HuggingFaceEmbeddingsAdapter().embed(
            {"input": "hi", "model": model, "api_key": "trusted-key"}
        )

    assert model not in str(exc_info.value)
    assert "trusted-key" not in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.concurrent
def test_concurrent_huggingface_embedding_model_routes_keep_url_and_key_paired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_EMBEDDINGS_NATIVE_HTTP_HUGGINGFACE", "1")

    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_embeddings_adapter as module
    from tldw_Server_API.app.core.AuthNZ.byok_config import (
        runtime_base_url_override_provenance,
    )
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_embeddings_adapter import (
        HuggingFaceEmbeddingsAdapter,
    )

    calls: list[tuple[str, str, str]] = []
    lock = threading.Lock()
    both_arrived = threading.Event()
    release = threading.Event()

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return [[0.1, 0.2]]

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> bool:
            del exc_type, exc, traceback
            return False

        def post(self, url, headers=None, json=None, **_kwargs):
            with lock:
                calls.append((url, headers["Authorization"], json["inputs"]))
                if len(calls) == 2:
                    both_arrived.set()
            if not release.wait(10):
                raise TimeoutError("concurrent Hugging Face embedding calls were not released")
            return _Resp()

    monkeypatch.setattr(module, "create_client", lambda **_kwargs: _Client())
    adapter = HuggingFaceEmbeddingsAdapter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                adapter.embed,
                {
                    "input": "alpha",
                    "model": "org/model-alpha",
                    "api_key": "key-alpha",
                    "base_url": "https://hf-alpha.example/models",
                    "credentials_resolved": True,
                    "_runtime_base_url_override": runtime_base_url_override_provenance(),
                },
            ),
            executor.submit(
                adapter.embed,
                {
                    "input": "beta",
                    "model": "org/model-beta",
                    "api_key": "key-beta",
                    "base_url": "https://hf-beta.example/models",
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
            "https://hf-alpha.example/models/org/model-alpha",
            "Bearer key-alpha",
            "alpha",
        ),
        (
            "https://hf-beta.example/models/org/model-beta",
            "Bearer key-beta",
            "beta",
        ),
    }


@pytest.mark.unit
@pytest.mark.concurrent
def test_concurrent_invalid_huggingface_embedding_cannot_affect_legitimate_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_EMBEDDINGS_NATIVE_HTTP_HUGGINGFACE", "1")

    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_embeddings_adapter as module
    from tldw_Server_API.app.core.AuthNZ.byok_config import (
        runtime_base_url_override_provenance,
    )
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatBadRequestError
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_embeddings_adapter import (
        HuggingFaceEmbeddingsAdapter,
    )

    calls = []
    arrived = threading.Event()
    release = threading.Event()

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return [[0.1, 0.2]]

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, url, headers=None, json=None, **_kwargs):
            calls.append((url, headers["Authorization"], json["inputs"]))
            arrived.set()
            if not release.wait(10):
                raise TimeoutError("legitimate Hugging Face embedding call was not released")
            return _Resp()

    monkeypatch.setattr(module, "create_client", lambda **_kwargs: _Client())
    provenance = runtime_base_url_override_provenance()
    adapter = HuggingFaceEmbeddingsAdapter()
    legitimate = {
        "input": "legitimate",
        "model": "/Qwen/Qwen3-235B-A22B",
        "api_key": "legitimate-key",
        "base_url": "https://legitimate-hf.example/models",
        "credentials_resolved": True,
        "_runtime_base_url_override": provenance,
    }
    malicious = legitimate | {
        "model": "//credential-admin",
        "api_key": "must-not-dispatch",
        "base_url": "https://malicious-hf.example/models",
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

    assert calls == [
        (
            "https://legitimate-hf.example/models/Qwen/Qwen3-235B-A22B",
            "Bearer legitimate-key",
            "legitimate",
        )
    ]


@pytest.mark.unit
def test_huggingface_embedding_provider_failure_has_no_raw_cause_or_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_EMBEDDINGS_NATIVE_HTTP_HUGGINGFACE", "1")

    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_embeddings_adapter as module
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatProviderError
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_embeddings_adapter import (
        HuggingFaceEmbeddingsAdapter,
    )

    sentinel = "raw-huggingface-provider-key-endpoint-body"

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, *_args, **_kwargs):
            raise RuntimeError(sentinel)

    monkeypatch.setattr(module, "create_client", lambda **_kwargs: _Client())

    with pytest.raises(ChatProviderError) as exc_info:
        HuggingFaceEmbeddingsAdapter().embed(
            {
                "input": "hello",
                "model": "org/model",
                "api_key": "trusted-key",
            }
        )

    assert sentinel not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.unit
def test_resolved_keyless_huggingface_request_defers_to_local_fallback_before_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolved keyless HF requests must not turn local execution into remote HTTP."""
    monkeypatch.setenv("LLM_EMBEDDINGS_NATIVE_HTTP_HUGGINGFACE", "1")

    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_embeddings_adapter as module
    from tldw_Server_API.app.core.LLM_Calls.providers.base import (
        EmbeddingsAdapterUnavailableError,
    )
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_embeddings_adapter import (
        HuggingFaceEmbeddingsAdapter,
    )

    monkeypatch.setattr(
        module,
        "create_client",
        lambda **_kwargs: pytest.fail("keyless local HF must not dispatch remote HTTP"),
    )

    with pytest.raises(EmbeddingsAdapterUnavailableError):
        HuggingFaceEmbeddingsAdapter().embed(
            {
                "input": "hello",
                "model": "org/local-model",
                "api_key": None,
                "credentials_resolved": True,
            }
        )


@pytest.mark.unit
def test_resolved_remote_huggingface_endpoint_without_key_fails_before_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured remote HF endpoint must not silently dispatch anonymously."""
    monkeypatch.setenv("LLM_EMBEDDINGS_NATIVE_HTTP_HUGGINGFACE", "1")

    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_embeddings_adapter as module
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_embeddings_adapter import (
        HuggingFaceEmbeddingsAdapter,
    )

    monkeypatch.setattr(
        module,
        "create_client",
        lambda **_kwargs: pytest.fail("missing remote credentials must fail before HTTP"),
    )

    with pytest.raises(ChatConfigurationError):
        HuggingFaceEmbeddingsAdapter().embed(
            {
                "input": "hello",
                "model": "org/remote-model",
                "api_key": None,
                "base_url": "https://hf.example/models",
                "credentials_resolved": True,
            }
        )
