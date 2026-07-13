from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create as ec


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

    def fake_openai(texts, *, model, app_config, dimensions):
        seen.append(app_config)
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(ec, "get_openai_embeddings_batch", fake_openai)

    result = ec.create_embeddings_batch(
        ["hello"],
        config,
        model_id,
        api_key_override="explicit-key",
        credentials_resolved=True,
    )

    assert result == [[0.1, 0.2]]
    assert seen[0]["openai_api"]["api_key"] == "explicit-key"
    assert seen[0]["openai_api"]["api_base_url"] == "https://api.openai.com/v1"
    assert config["openai_api"]["api_key"] == "server-config-key"


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
