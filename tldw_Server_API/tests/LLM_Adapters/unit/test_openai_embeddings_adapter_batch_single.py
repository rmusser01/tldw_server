import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest


@pytest.mark.unit
def test_openai_embeddings_adapter_uses_batch_helper_for_list(monkeypatch):
    monkeypatch.delenv("LLM_EMBEDDINGS_NATIVE_HTTP_OPENAI", raising=False)

    from tldw_Server_API.app.core.LLM_Calls.providers.openai_embeddings_adapter import (
        OpenAIEmbeddingsAdapter,
    )

    def _fake_batch(texts, model, app_config=None, dimensions=None):  # noqa: ANN001, ARG001
        assert texts == ["a", "b"]
        return [[0.1], [0.2]]

    def _fail_single(*_args, **_kwargs):  # noqa: ANN001
        raise AssertionError("single helper should not be called for batch input")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.chat_calls.get_openai_embeddings_batch",
        _fake_batch,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.chat_calls.get_openai_embeddings",
        _fail_single,
    )

    adapter = OpenAIEmbeddingsAdapter()
    out = adapter.embed({"input": ["a", "b"], "model": "text-embedding-3-small", "app_config": {}})
    assert [item["embedding"] for item in out.get("data", [])] == [[0.1], [0.2]]


@pytest.mark.unit
def test_openai_embeddings_adapter_default_path_restores_upstream_index_order(
    monkeypatch,
):
    monkeypatch.delenv("LLM_EMBEDDINGS_NATIVE_HTTP_OPENAI", raising=False)

    from tldw_Server_API.app.core.LLM_Calls.providers.openai_embeddings_adapter import (
        OpenAIEmbeddingsAdapter,
    )

    class _Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            }

    class _Session:
        def post(self, *_args, **_kwargs):
            return _Response()

        def close(self):
            return None

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.chat_calls.create_session_with_retries",
        lambda **_kwargs: _Session(),
    )

    result = OpenAIEmbeddingsAdapter().embed(
        {
            "input": ["first", "second"],
            "model": "text-embedding-3-small",
            "app_config": {"openai_api": {"api_key": "test-key"}},
        }
    )

    assert result["data"] == [
        {"index": 0, "embedding": [1.0, 0.0]},
        {"index": 1, "embedding": [0.0, 1.0]},
    ]


@pytest.mark.unit
def test_openai_embeddings_adapter_uses_single_helper_for_scalar(monkeypatch):
    monkeypatch.delenv("LLM_EMBEDDINGS_NATIVE_HTTP_OPENAI", raising=False)

    from tldw_Server_API.app.core.LLM_Calls.providers.openai_embeddings_adapter import (
        OpenAIEmbeddingsAdapter,
    )

    def _fail_batch(*_args, **_kwargs):  # noqa: ANN001
        raise AssertionError("batch helper should not be called for scalar input")

    def _fake_single(text, model, app_config=None, dimensions=None):  # noqa: ANN001, ARG001
        assert text == "hi"
        return [0.5, 0.6]

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.chat_calls.get_openai_embeddings_batch",
        _fail_batch,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.chat_calls.get_openai_embeddings",
        _fake_single,
    )

    adapter = OpenAIEmbeddingsAdapter()
    out = adapter.embed({"input": "hi", "model": "text-embedding-3-small", "app_config": {}})
    assert out.get("data", [])[0]["embedding"] == [0.5, 0.6]


@pytest.mark.unit
def test_resolved_openai_request_forces_pinned_native_transport_without_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_EMBEDDINGS_NATIVE_HTTP_OPENAI", raising=False)
    monkeypatch.setenv("OPENAI_API_BASE_URL", "https://ambient-attacker.example/v1")

    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core.AuthNZ.byok_config import (
        runtime_base_url_override_provenance,
    )
    from tldw_Server_API.app.core.LLM_Calls.providers.openai_embeddings_adapter import (
        OpenAIEmbeddingsAdapter,
    )

    calls: list[dict[str, object]] = []

    class Response:
        status_code = 200

        def json(self) -> dict[str, object]:
            return {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}

    def fetch(**kwargs: object) -> Response:
        calls.append(kwargs)
        return Response()

    monkeypatch.setattr(http_client, "fetch", fetch)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.chat_calls.get_openai_embeddings_batch",
        lambda *_args, **_kwargs: pytest.fail("resolved request used legacy transport"),
    )

    result = OpenAIEmbeddingsAdapter().embed(
        {
            "input": ["note text"],
            "model": "text-embedding-3-small",
            "api_key": "trusted-key",
            "base_url": "https://pinned-openai.example/v1",
            "credentials_resolved": True,
            "_runtime_base_url_override": runtime_base_url_override_provenance(),
        }
    )

    assert result["data"]
    assert calls[0]["url"] == "https://pinned-openai.example/v1/embeddings"
    assert calls[0]["allow_redirects"] is False
    assert "ambient-attacker" not in repr(calls)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status_code", "location"),
    [
        (307, "https://cross-origin.example/receive"),
        (308, "https://cross-origin.example/receive"),
        (307, "https://pinned-openai.example/v1/redirected"),
    ],
)
def test_resolved_openai_request_rejects_redirect_without_replaying_note_text(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    location: str,
) -> None:
    monkeypatch.setenv("LLM_EMBEDDINGS_NATIVE_HTTP_OPENAI", "1")

    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core.AuthNZ.byok_config import (
        runtime_base_url_override_provenance,
    )
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatProviderError
    from tldw_Server_API.app.core.LLM_Calls.providers.openai_embeddings_adapter import (
        OpenAIEmbeddingsAdapter,
    )

    seen: list[tuple[str, bytes]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.host, request.content))
        if len(seen) == 1:
            return httpx.Response(status_code, headers={"location": location}, json={})
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]},
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
    )
    monkeypatch.setattr(http_client, "_get_httpx_client", lambda **_kwargs: client)
    monkeypatch.setattr(http_client, "_validate_egress_or_raise", lambda *_args, **_kwargs: None)

    try:
        with pytest.raises(ChatProviderError):
            OpenAIEmbeddingsAdapter().embed(
                {
                    "input": ["note text must stay pinned"],
                    "model": "text-embedding-3-small",
                    "api_key": "trusted-key",
                    "base_url": "https://pinned-openai.example/v1",
                    "credentials_resolved": True,
                    "_runtime_base_url_override": runtime_base_url_override_provenance(),
                }
            )
    finally:
        client.close()

    assert [host for host, _body in seen] == ["pinned-openai.example"]
    assert b"note text must stay pinned" in seen[0][1]


@pytest.mark.unit
@pytest.mark.parametrize("status_code", [401, 403])
def test_openai_native_embedding_preserves_bounded_auth_status_without_raw_cause(
    monkeypatch,
    status_code,
):
    monkeypatch.setenv("LLM_EMBEDDINGS_NATIVE_HTTP_OPENAI", "1")

    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAuthenticationError
    from tldw_Server_API.app.core.LLM_Calls.providers.openai_embeddings_adapter import (
        OpenAIEmbeddingsAdapter,
    )

    sentinel = f"raw-openai-{status_code}-key-endpoint-body"

    class _UpstreamError(RuntimeError):
        def __init__(self):
            super().__init__(sentinel)
            self.status_code = status_code

    class _Response:
        def __init__(self):
            self.status_code = status_code

        def raise_for_status(self):
            raise _UpstreamError()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.http_client.fetch",
        lambda **_kwargs: _Response(),
    )

    with pytest.raises(ChatAuthenticationError) as exc_info:
        OpenAIEmbeddingsAdapter().embed(
            {
                "input": "hello",
                "model": "text-embedding-3-small",
                "api_key": "trusted-key",
            }
        )

    assert exc_info.value.status_code == status_code
    assert sentinel not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.unit
@pytest.mark.concurrent
def test_concurrent_openai_native_embedding_routes_keep_endpoint_key_and_model_paired(
    monkeypatch,
):
    monkeypatch.setenv("LLM_EMBEDDINGS_NATIVE_HTTP_OPENAI", "1")

    from tldw_Server_API.app.core.AuthNZ.byok_config import (
        runtime_base_url_override_provenance,
    )
    from tldw_Server_API.app.core.LLM_Calls.providers.openai_embeddings_adapter import (
        OpenAIEmbeddingsAdapter,
    )

    calls = []
    lock = threading.Lock()
    both_arrived = threading.Event()
    release = threading.Event()

    class _Response:
        status_code = 200

        def json(self):
            return {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}

    def _fetch(**kwargs):
        with lock:
            calls.append(
                (
                    kwargs["url"],
                    kwargs["headers"]["Authorization"],
                    kwargs["json"]["model"],
                    kwargs["json"]["input"],
                    kwargs["headers"]["OpenAI-Organization"],
                    kwargs["headers"]["OpenAI-Project"],
                )
            )
            if len(calls) == 2:
                both_arrived.set()
        if not release.wait(10):
            raise TimeoutError("concurrent OpenAI embedding calls were not released")
        return _Response()

    monkeypatch.setattr("tldw_Server_API.app.core.http_client.fetch", _fetch)
    adapter = OpenAIEmbeddingsAdapter()
    provenance = runtime_base_url_override_provenance()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                adapter.embed,
                {
                    "input": "alpha",
                    "model": "model-alpha",
                    "api_key": "key-alpha",
                    "base_url": "https://openai-alpha.example/v1",
                    "credentials_resolved": True,
                    "_runtime_base_url_override": provenance,
                    "app_config": {
                        "openai_api": {
                            "org_id": "org-alpha",
                            "project_id": "project-alpha",
                        }
                    },
                },
            ),
            executor.submit(
                adapter.embed,
                {
                    "input": "beta",
                    "model": "model-beta",
                    "api_key": "key-beta",
                    "base_url": "https://openai-beta.example/v1",
                    "credentials_resolved": True,
                    "_runtime_base_url_override": provenance,
                    "app_config": {
                        "openai_api": {
                            "org_id": "org-beta",
                            "project_id": "project-beta",
                        }
                    },
                },
            ),
        ]
        try:
            assert both_arrived.wait(10)
        finally:
            release.set()
        assert all(future.result(timeout=10)["data"] for future in futures)

    assert set(calls) == {
        (
            "https://openai-alpha.example/v1/embeddings",
            "Bearer key-alpha",
            "model-alpha",
            "alpha",
            "org-alpha",
            "project-alpha",
        ),
        (
            "https://openai-beta.example/v1/embeddings",
            "Bearer key-beta",
            "model-beta",
            "beta",
            "org-beta",
            "project-beta",
        ),
    }


@pytest.mark.unit
@pytest.mark.concurrent
def test_concurrent_openai_legacy_embedding_routes_keep_endpoint_key_model_and_input_paired(
    monkeypatch,
):
    """The native-disabled adapter must carry each trusted endpoint into legacy HTTP."""
    monkeypatch.delenv("LLM_EMBEDDINGS_NATIVE_HTTP_OPENAI", raising=False)
    monkeypatch.setenv("OPENAI_API_BASE_URL", "https://ambient-attacker.example/v1")

    from tldw_Server_API.app.core.LLM_Calls.providers.openai_embeddings_adapter import (
        OpenAIEmbeddingsAdapter,
    )

    calls: list[tuple[str, str, str, str]] = []
    lock = threading.Lock()
    both_arrived = threading.Event()
    release = threading.Event()

    def _legacy_single(text, model, app_config=None, dimensions=None):
        del dimensions
        config = dict((app_config or {}).get("openai_api") or {})
        call = (config["api_base_url"], config["api_key"], model, text)
        with lock:
            calls.append(call)
            if len(calls) == 2:
                both_arrived.set()
        if not release.wait(10):
            raise TimeoutError("concurrent legacy OpenAI embedding calls were not released")
        return [0.1, 0.2]

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.chat_calls.get_openai_embeddings",
        _legacy_single,
    )
    adapter = OpenAIEmbeddingsAdapter()

    def _invoke(label: str) -> dict[str, object]:
        return adapter.embed(
            {
                "input": label,
                "model": f"model-{label}",
                "app_config": {
                    "openai_api": {
                        "api_base_url": f"https://openai-{label}.example/v1",
                        "api_key": f"key-{label}",
                        "organization": f"org-{label}",
                    }
                },
            }
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        alpha = executor.submit(_invoke, "alpha")
        beta = executor.submit(_invoke, "beta")
        try:
            assert both_arrived.wait(10)
        finally:
            release.set()
        assert alpha.result(timeout=10)["data"]
        assert beta.result(timeout=10)["data"]

    assert set(calls) == {
        (
            "https://openai-alpha.example/v1",
            "key-alpha",
            "model-alpha",
            "alpha",
        ),
        (
            "https://openai-beta.example/v1",
            "key-beta",
            "model-beta",
            "beta",
        ),
    }
    assert "ambient-attacker" not in repr(calls)


@pytest.mark.unit
@pytest.mark.concurrent
def test_concurrent_openai_legacy_auth_failures_are_status_preserving_and_detached(
    monkeypatch,
):
    """Legacy transport failures must reach OAuth policy without raw exception chains."""
    monkeypatch.delenv("LLM_EMBEDDINGS_NATIVE_HTTP_OPENAI", raising=False)

    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAuthenticationError
    from tldw_Server_API.app.core.LLM_Calls.providers.openai_embeddings_adapter import (
        OpenAIEmbeddingsAdapter,
    )

    lock = threading.Lock()
    arrivals = 0
    both_arrived = threading.Event()
    release = threading.Event()

    class _RawLegacyAuthError(RuntimeError):
        def __init__(self, status_code: int, sentinel: str) -> None:
            super().__init__(sentinel)
            self.status_code = status_code

    def _legacy_single(text, model, app_config=None, dimensions=None):
        del model, app_config, dimensions
        nonlocal arrivals
        with lock:
            arrivals += 1
            if arrivals == 2:
                both_arrived.set()
        if not release.wait(10):
            raise TimeoutError("concurrent legacy auth failures were not released")
        status = 401 if text == "alpha" else 403
        raise _RawLegacyAuthError(status, f"raw-legacy-{text}-secret")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.chat_calls.get_openai_embeddings",
        _legacy_single,
    )
    adapter = OpenAIEmbeddingsAdapter()

    def _invoke(label: str) -> ChatAuthenticationError:
        try:
            adapter.embed(
                {
                    "input": label,
                    "model": f"model-{label}",
                    "app_config": {
                        "openai_api": {
                            "api_base_url": f"https://openai-{label}.example/v1",
                            "api_key": f"key-{label}",
                        }
                    },
                }
            )
        except ChatAuthenticationError as error:
            return error
        raise AssertionError("legacy authentication failure was not normalized")

    with ThreadPoolExecutor(max_workers=2) as executor:
        alpha_future = executor.submit(_invoke, "alpha")
        beta_future = executor.submit(_invoke, "beta")
        try:
            assert both_arrived.wait(10)
        finally:
            release.set()
        alpha = alpha_future.result(timeout=10)
        beta = beta_future.result(timeout=10)

    assert (alpha.status_code, beta.status_code) == (401, 403)
    for error in (alpha, beta):
        assert "raw-legacy" not in str(error)
        assert error.__cause__ is None
        assert error.__context__ is None
