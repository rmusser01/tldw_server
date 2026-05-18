from __future__ import annotations

from tldw_Server_API.app.core.Setup import setup_manager
from tldw_Server_API.app.core.Setup.readiness_service import preview_readiness_selection


def test_preview_returns_config_updates_and_install_plan_without_writing(monkeypatch):
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("preview must not write config")

    monkeypatch.setattr(setup_manager, "update_config", fail_if_called)

    preview = preview_readiness_selection(
        {
            "profile_id": "local_balanced",
            "lanes": {
                "chat": {"mode": "skip"},
                "embeddings_rag": {
                    "provider": "huggingface",
                    "model": "Qwen/Qwen3-Embedding-0.6B",
                },
                "speech": {"bundle_id": "cpu_local", "resource_profile": "balanced"},
            },
        }
    )

    assert called is False
    assert preview["operation_required"] is True
    assert "restart_required" in preview["overlays"]
    assert preview["lanes"]["chat"]["status"] == "skipped"
    assert preview["config_updates"]["Embeddings"]["embedding_provider"] == "huggingface"
    assert preview["config_updates"]["Embeddings"]["embedding_model"] == "Qwen/Qwen3-Embedding-0.6B"
    assert "Qwen/Qwen3-Embedding-0.6B" in preview["install_plan"]["embeddings"]["huggingface"]
    assert preview["install_plan"]["stt"][0]["engine"] == "faster_whisper"
    assert preview["install_plan"]["tts"][0]["engine"] == "kokoro"


def test_preview_never_echoes_hosted_provider_secret():
    preview = preview_readiness_selection(
        {
            "profile_id": "advanced_custom",
            "lanes": {
                "chat": {
                    "mode": "hosted",
                    "provider": "openai",
                    "api_key": "sk-sensitive",
                    "model": "gpt-4.1-mini",
                }
            },
        }
    )

    assert "sk-sensitive" not in str(preview)
    assert preview["config_updates"]["API"]["default_api"] == "openai"
    assert preview["config_updates"]["API"]["openai_model"] == "gpt-4.1-mini"
    assert "openai_api_key" not in preview["config_updates"]["API"]
    assert preview["secret_fields"] == [
        {"section": "API", "key": "openai_api_key", "provider": "openai", "state": "submitted"}
    ]


def test_trusted_custom_hf_requires_acknowledgement():
    preview = preview_readiness_selection(
        {
            "profile_id": "advanced_custom",
            "lanes": {
                "embeddings_rag": {
                    "provider": "huggingface",
                    "model": "custom/requires-trust",
                    "trusted_custom_model": True,
                    "trusted_custom_model_acknowledged": False,
                }
            },
        }
    )

    assert preview["lanes"]["embeddings_rag"]["status"] == "blocked"
    assert "trusted custom Hugging Face model acknowledgement is required" in preview["lanes"][
        "embeddings_rag"
    ]["blockers"]
    assert "custom/requires-trust" not in preview["install_plan"]["embeddings"]["custom"]


def test_acknowledged_trusted_custom_hf_enters_custom_install_plan():
    preview = preview_readiness_selection(
        {
            "profile_id": "advanced_custom",
            "lanes": {
                "embeddings_rag": {
                    "provider": "huggingface",
                    "model": "custom/allowed-with-ack",
                    "trusted_custom_model": True,
                    "trusted_custom_model_acknowledged": True,
                }
            },
        }
    )

    assert preview["lanes"]["embeddings_rag"]["status"] == "previewed"
    assert "custom/allowed-with-ack" in preview["install_plan"]["embeddings"]["custom"]


def test_local_chat_preview_only_emits_existing_config_keys():
    preview = preview_readiness_selection(
        {
            "profile_id": "advanced_custom",
            "lanes": {
                "chat": {
                    "mode": "local",
                    "provider": "llama",
                    "endpoint": "http://127.0.0.1:8080/completion",
                    "model": "local-model-name",
                }
            },
        }
    )

    assert preview["config_updates"]["API"]["default_api"] == "llama"
    assert preview["config_updates"]["Local-API"] == {
        "llama_api_IP": "http://127.0.0.1:8080/completion"
    }
