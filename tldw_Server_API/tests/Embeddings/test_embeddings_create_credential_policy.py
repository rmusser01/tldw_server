from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from loguru import logger

from tldw_Server_API.app.core.Embeddings import async_embeddings
from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create as ec

SECRET = "upstream-secret-body"
EXPLICIT_KEY = "runtime-explicit-key-must-not-leak"
EXPLICIT_ENDPOINT = "https://runtime-endpoint-must-not-leak.example/v1"


def _config(tmp_path, provider, **model_fields):
    model_id = f"{provider}:test-model"
    return model_id, {
        "openai_api": {"api_key": "server-config-key"},
        "embedding_config": {
            "default_model_id": model_id,
            "model_storage_base_dir": str(tmp_path),
            "models": {
                model_id: {
                    "provider": provider,
                    "model_name_or_path": "test-model",
                    **model_fields,
                }
            },
        },
    }


def _capture_full_loguru_output():
    messages = []
    sink_id = logger.add(messages.append, backtrace=True, diagnose=True)
    return messages, sink_id


@pytest.mark.unit
def test_explicit_openai_key_overrides_model_and_server_keys(monkeypatch, tmp_path):
    monkeypatch.setattr(ec, "_EMBEDDINGS_STORAGE_ALLOWLIST_ROOT", tmp_path.resolve())
    monkeypatch.setenv("OPENAI_API_BASE_URL", "https://hostile-env.example/v1")
    model_id, config = _config(tmp_path, "openai", api_key="model-spec-key")
    seen = []

    class Response:
        status_code = 200

        def json(self):
            return {"data": [{"embedding": [0.1, 0.2]}]}

        def close(self):
            return None

    def fake_fetch(**kwargs):
        seen.append(kwargs)
        return Response()

    monkeypatch.setattr("tldw_Server_API.app.core.http_client.fetch", fake_fetch)
    monkeypatch.setattr(
        ec,
        "get_openai_embeddings_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("explicit credentials must bypass the legacy helper")
        ),
    )

    result = ec.create_embeddings_batch(
        ["hello"],
        config,
        model_id,
        api_key_override="explicit-key",
        credentials_resolved=True,
    )

    assert result == [[0.1, 0.2]]
    assert seen[0]["url"] == "https://api.openai.com/v1/embeddings"
    assert seen[0]["headers"]["Authorization"] == "Bearer explicit-key"
    assert seen[0]["json"] == {"input": ["hello"], "model": "test-model"}
    assert seen[0]["retry"].attempts == 1
    assert seen[0]["sensitive_observability"] is True
    assert config["openai_api"]["api_key"] == "server-config-key"


@pytest.mark.unit
def test_explicit_openai_auth_failure_is_sanitized_and_not_retried(monkeypatch, tmp_path):
    monkeypatch.setattr(ec, "_EMBEDDINGS_STORAGE_ALLOWLIST_ROOT", tmp_path.resolve())
    model_id, config = _config(tmp_path, "openai", api_key="model-spec-key")
    seen = []
    log_messages = []
    sink_id = logger.add(log_messages.append, format="{message}")

    class Response:
        status_code = 401

        def json(self):
            return {"error": {"message": SECRET}}

        def close(self):
            return None

    def fake_fetch(**kwargs):
        seen.append(kwargs)
        return Response()

    monkeypatch.setattr("tldw_Server_API.app.core.http_client.fetch", fake_fetch)
    monkeypatch.setattr(
        ec,
        "get_openai_embeddings_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("explicit credentials must bypass the legacy helper")
        ),
    )

    try:
        with pytest.raises(async_embeddings.EmbeddingProviderError) as exc_info:
            ec.create_embeddings_batch(
                ["hello"],
                config,
                model_id,
                api_key_override="explicit-key",
                base_url_override="https://explicit.example/v1",
                credentials_resolved=True,
            )
    finally:
        logger.remove(sink_id)

    assert len(seen) == 1
    assert seen[0]["url"] == "https://explicit.example/v1/embeddings"
    assert seen[0]["headers"]["Authorization"] == "Bearer explicit-key"
    assert seen[0]["retry"].attempts == 1
    assert exc_info.value.code == "authentication"
    assert exc_info.value.provider == "openai"
    assert exc_info.value.status_code == 401
    assert exc_info.value.__cause__ is None
    assert SECRET not in str(exc_info.value)
    assert SECRET not in repr(exc_info.value)
    assert SECRET not in "".join(log_messages)


@pytest.mark.unit
@pytest.mark.parametrize("status_code", [429, 500])
def test_explicit_openai_transient_http_failure_is_sanitized_and_not_retried(
    monkeypatch,
    tmp_path,
    status_code,
):
    monkeypatch.setattr(ec, "_EMBEDDINGS_STORAGE_ALLOWLIST_ROOT", tmp_path.resolve())
    model_id, config = _config(tmp_path, "openai")
    calls = 0
    logs, sink_id = _capture_full_loguru_output()

    class Response:
        def __init__(self):
            self.status_code = status_code

        def json(self):
            return {"error": {"message": SECRET}}

        def close(self):
            return None

    def fake_fetch(**kwargs):
        nonlocal calls
        calls += 1
        leaked_headers = kwargs["headers"]
        leaked_endpoint = kwargs["url"]
        assert leaked_headers["Authorization"] == f"Bearer {EXPLICIT_KEY}"
        assert leaked_endpoint == f"{EXPLICIT_ENDPOINT}/embeddings"
        return Response()

    monkeypatch.setattr("tldw_Server_API.app.core.http_client.fetch", fake_fetch)

    try:
        with pytest.raises(async_embeddings.EmbeddingProviderError) as exc_info:
            ec.create_embeddings_batch(
                ["hello"],
                config,
                model_id,
                api_key_override=EXPLICIT_KEY,
                base_url_override=EXPLICIT_ENDPOINT,
                credentials_resolved=True,
            )
    finally:
        logger.remove(sink_id)

    output = "".join(logs)
    assert calls == 1
    assert exc_info.value.code == "provider_failure"
    assert exc_info.value.status_code == status_code
    assert exc_info.value.__cause__ is None
    assert EXPLICIT_KEY not in output
    assert EXPLICIT_ENDPOINT not in output
    assert SECRET not in output


@pytest.mark.unit
def test_explicit_openai_network_failure_is_sanitized_and_not_retried(monkeypatch, tmp_path):
    monkeypatch.setattr(ec, "_EMBEDDINGS_STORAGE_ALLOWLIST_ROOT", tmp_path.resolve())
    model_id, config = _config(tmp_path, "openai")
    calls = 0
    logs, sink_id = _capture_full_loguru_output()

    def fake_fetch(**kwargs):
        nonlocal calls
        calls += 1
        leaked_headers = kwargs["headers"]
        leaked_endpoint = kwargs["url"]
        raise ConnectionError(
            f"network failed at {leaked_endpoint} with {leaked_headers['Authorization']} and {SECRET}"
        )

    monkeypatch.setattr("tldw_Server_API.app.core.http_client.fetch", fake_fetch)

    try:
        with pytest.raises(async_embeddings.EmbeddingProviderError) as exc_info:
            ec.create_embeddings_batch(
                ["hello"],
                config,
                model_id,
                api_key_override=EXPLICIT_KEY,
                base_url_override=EXPLICIT_ENDPOINT,
                credentials_resolved=True,
            )
    finally:
        logger.remove(sink_id)

    output = "".join(logs)
    assert calls == 1
    assert exc_info.value.code == "provider_failure"
    assert exc_info.value.__cause__ is None
    assert EXPLICIT_KEY not in output
    assert EXPLICIT_ENDPOINT not in output
    assert SECRET not in output


@pytest.mark.unit
def test_explicit_missing_openai_key_does_not_use_configured_keys(monkeypatch, tmp_path):
    monkeypatch.setattr(ec, "_EMBEDDINGS_STORAGE_ALLOWLIST_ROOT", tmp_path.resolve())
    model_id, config = _config(tmp_path, "openai", api_key="model-spec-key")
    monkeypatch.setattr(
        ec,
        "get_openai_embeddings_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must fail before call")),
    )

    with pytest.raises(ValueError, match="credential"):
        ec.create_embeddings_batch(
            ["hello"],
            config,
            model_id,
            api_key_override=" ",
            credentials_resolved=True,
        )


@pytest.mark.unit
def test_explicit_local_api_uses_per_call_key_and_url(monkeypatch, tmp_path):
    monkeypatch.setattr(ec, "_EMBEDDINGS_STORAGE_ALLOWLIST_ROOT", tmp_path.resolve())
    model_id, config = _config(
        tmp_path,
        "local_api",
        api_key="model-spec-key",
        api_url="http://configured.example/embeddings",
    )
    seen = []

    class Response:
        status_code = 200

        def json(self):
            return {"embeddings": [[0.3, 0.4]]}

    def fake_fetch(**kwargs):
        seen.append(kwargs)
        return Response()

    monkeypatch.setattr("tldw_Server_API.app.core.http_client.fetch", fake_fetch)

    result = ec.create_embedding(
        "hello",
        config,
        model_id,
        api_key_override="local-call-key",
        base_url_override="http://explicit.example/embeddings",
        credentials_resolved=True,
    )

    assert result == [0.3, 0.4]
    assert seen[0]["url"] == "http://explicit.example/embeddings"
    assert seen[0]["headers"]["Authorization"] == "Bearer local-call-key"
    assert seen[0]["sensitive_observability"] is True
    assert config["embedding_config"]["models"][model_id]["api_key"] == "model-spec-key"


@pytest.mark.unit
@pytest.mark.concurrent
def test_explicit_concurrent_local_api_calls_keep_same_endpoint_keys_isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(ec, "_EMBEDDINGS_STORAGE_ALLOWLIST_ROOT", tmp_path.resolve())
    model_id, config = _config(
        tmp_path,
        "local_api",
        api_url="http://configured.example/embeddings",
    )
    endpoint = "http://shared.example/embeddings"
    calls: list[tuple[str, str, str]] = []
    lock = threading.Lock()
    both_arrived = threading.Event()
    release = threading.Event()

    class Response:
        status_code = 200

        def __init__(self, marker: float):
            self.marker = marker

        def json(self):
            return {"embeddings": [[self.marker]]}

        def close(self):
            return None

    def fake_fetch(**kwargs):
        authorization = kwargs["headers"].get("Authorization", "")
        text = kwargs["json"]["texts"][0]
        with lock:
            calls.append((authorization, kwargs["url"], text))
            if len(calls) == 2:
                both_arrived.set()
        if not release.wait(5):
            raise TimeoutError("concurrent local embedding calls did not release")
        return Response(1.0 if authorization == "Bearer key-alpha" else 2.0)

    monkeypatch.setattr("tldw_Server_API.app.core.http_client.fetch", fake_fetch)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                ec.create_embedding,
                label,
                config,
                model_id,
                api_key_override=f"key-{label}",
                base_url_override=endpoint,
                credentials_resolved=True,
            )
            for label in ("alpha", "beta")
        ]
        assert both_arrived.wait(5)
        release.set()
        results = [future.result(timeout=5) for future in futures]

    assert results == [[1.0], [2.0]]
    assert set(calls) == {
        ("Bearer key-alpha", endpoint, "alpha"),
        ("Bearer key-beta", endpoint, "beta"),
    }


@pytest.mark.unit
@pytest.mark.parametrize("embedding", [[SECRET], [10**10000]])
def test_explicit_local_api_rejects_malformed_embedding(monkeypatch, tmp_path, embedding):
    monkeypatch.setattr(ec, "_EMBEDDINGS_STORAGE_ALLOWLIST_ROOT", tmp_path.resolve())
    model_id, config = _config(
        tmp_path,
        "local_api",
        api_url="http://configured.example/embeddings",
    )

    response = SimpleNamespace(
        status_code=200,
        json=lambda: {"embeddings": [embedding]},
        close=lambda: None,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.http_client.fetch",
        lambda **_kwargs: response,
    )

    with pytest.raises(async_embeddings.EmbeddingProviderError) as exc_info:
        ec.create_embeddings_batch(
            ["hello"],
            config,
            model_id,
            api_key_override=EXPLICIT_KEY,
            base_url_override=EXPLICIT_ENDPOINT,
            credentials_resolved=True,
        )

    assert exc_info.value.code == "provider_failure"
    assert SECRET not in str(exc_info.value)


@pytest.mark.unit
def test_explicit_local_api_missing_endpoint_preserves_configuration_error(monkeypatch, tmp_path):
    monkeypatch.setattr(ec, "_EMBEDDINGS_STORAGE_ALLOWLIST_ROOT", tmp_path.resolve())
    model_id, config = _config(
        tmp_path,
        "local_api",
        api_url="http://configured.example/embeddings",
    )

    with pytest.raises(async_embeddings.EmbeddingEndpointError):
        ec.create_embeddings_batch(
            ["hello"],
            config,
            model_id,
            api_key_override=None,
            base_url_override=None,
            credentials_resolved=True,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("response", "expected_code", "expected_status"),
    [
        (SimpleNamespace(status_code=401), "authentication", 401),
        (SimpleNamespace(status_code=500), "provider_failure", 500),
        (SimpleNamespace(status_code=200), "provider_failure", None),
    ],
)
def test_explicit_local_api_failure_is_sanitized_and_not_retried(
    monkeypatch,
    tmp_path,
    response,
    expected_code,
    expected_status,
):
    monkeypatch.setattr(ec, "_EMBEDDINGS_STORAGE_ALLOWLIST_ROOT", tmp_path.resolve())
    model_id, config = _config(
        tmp_path,
        "local_api",
        api_url="http://configured.example/embeddings",
    )
    calls = 0
    logs, sink_id = _capture_full_loguru_output()

    response.json = lambda: {"error": SECRET}
    response.raise_for_status = lambda: (_ for _ in ()).throw(
        RuntimeError(f"{SECRET} {EXPLICIT_KEY} {EXPLICIT_ENDPOINT}")
    )
    response.close = lambda: None

    def fake_fetch(**kwargs):
        nonlocal calls
        calls += 1
        leaked_headers = kwargs["headers"]
        leaked_endpoint = kwargs["url"]
        assert leaked_headers["Authorization"] == f"Bearer {EXPLICIT_KEY}"
        assert leaked_endpoint == EXPLICIT_ENDPOINT
        return response

    monkeypatch.setattr("tldw_Server_API.app.core.http_client.fetch", fake_fetch)

    try:
        with pytest.raises(async_embeddings.EmbeddingProviderError) as exc_info:
            ec.create_embeddings_batch(
                ["hello"],
                config,
                model_id,
                api_key_override=EXPLICIT_KEY,
                base_url_override=EXPLICIT_ENDPOINT,
                credentials_resolved=True,
            )
    finally:
        logger.remove(sink_id)

    output = "".join(logs)
    assert calls == 1
    assert exc_info.value.code == expected_code
    assert exc_info.value.status_code == expected_status
    assert exc_info.value.__cause__ is None
    assert EXPLICIT_KEY not in output
    assert EXPLICIT_ENDPOINT not in output
    assert SECRET not in output


@pytest.mark.unit
def test_legacy_openai_transient_failure_retains_retry_policy(monkeypatch, tmp_path):
    monkeypatch.setattr(ec, "_EMBEDDINGS_STORAGE_ALLOWLIST_ROOT", tmp_path.resolve())
    model_id, config = _config(tmp_path, "openai")
    calls = 0
    sleeps = []

    class RetryableLegacyError(RuntimeError):
        status_code = 500

    def fail_legacy(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RetryableLegacyError("legacy retry remains unchanged")

    monkeypatch.setattr(ec, "get_openai_embeddings_batch", fail_legacy)
    monkeypatch.setattr(ec.time, "sleep", sleeps.append)

    with pytest.raises(RetryableLegacyError):
        ec.create_embeddings_batch(["hello"], config, model_id)

    assert calls == 4
    assert sleeps == [1, 2, 4]
