from __future__ import annotations

import pytest
from loguru import logger

from tldw_Server_API.app.core.Embeddings import async_embeddings
from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create as ec


SECRET = "upstream-secret-body"


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
    assert config["embedding_config"]["models"][model_id]["api_key"] == "model-spec-key"
