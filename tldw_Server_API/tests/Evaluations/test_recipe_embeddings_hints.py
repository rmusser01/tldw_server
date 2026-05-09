from __future__ import annotations

from tldw_Server_API.app.api.v1.schemas.evaluation_schemas_unified import RunStatus
from tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints import (
    build_embedding_recipe_apply_preview,
    build_embedding_recipe_candidate_hints,
)


def test_candidate_hints_include_current_model_and_policy_status(monkeypatch) -> None:
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_simplified_embeddings_config",
        lambda: {
            "default_provider": "openai",
            "default_model": "text-embedding-3-small",
            "providers": [{"name": "openai", "models": ["text-embedding-3-small"], "api_key": "sk-test"}],
        },
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_allowed_embedding_providers",
        lambda: ["openai"],
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_allowed_embedding_models",
        lambda: ["text-embedding-3-*"],
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.should_enforce_embedding_policy",
        lambda user=None: True,
    )

    result = build_embedding_recipe_candidate_hints(user=None)

    assert result["current"]["provider"] == "openai"
    assert result["current"]["model"] == "text-embedding-3-small"
    assert result["candidates"][0]["status"] == "ready"
    assert result["candidates"][0]["default"] is True


def test_candidate_hints_mark_disallowed_provider(monkeypatch) -> None:
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_simplified_embeddings_config",
        lambda: {
            "default_provider": "openai",
            "default_model": "text-embedding-3-small",
            "providers": [{"name": "huggingface", "models": ["BAAI/bge-small-en-v1.5"]}],
        },
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_allowed_embedding_providers",
        lambda: ["openai"],
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_allowed_embedding_models",
        lambda: None,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.should_enforce_embedding_policy",
        lambda user=None: True,
    )

    result = build_embedding_recipe_candidate_hints(user=None)

    assert result["candidates"][0]["status"] == "disallowed_provider"
    assert "not allowed" in result["candidates"][0]["status_reason"].lower()


def test_candidate_hints_mark_disallowed_model(monkeypatch) -> None:
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_simplified_embeddings_config",
        lambda: {
            "default_provider": "openai",
            "default_model": "text-embedding-3-small",
            "providers": [{"name": "openai", "models": ["legacy-embedding"], "api_key": "sk-test"}],
        },
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_allowed_embedding_providers",
        lambda: ["openai"],
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_allowed_embedding_models",
        lambda: ["text-embedding-3-*"],
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.should_enforce_embedding_policy",
        lambda user=None: True,
    )

    result = build_embedding_recipe_candidate_hints(user=None)

    assert result["candidates"][0]["status"] == "disallowed_model"
    assert "not allowed" in result["candidates"][0]["status_reason"].lower()


def test_candidate_hints_ignore_allowlists_when_policy_not_enforced(monkeypatch) -> None:
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_simplified_embeddings_config",
        lambda: {
            "default_provider": "huggingface",
            "default_model": "BAAI/bge-small-en-v1.5",
            "providers": [{"name": "huggingface", "models": ["BAAI/bge-small-en-v1.5"]}],
        },
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_allowed_embedding_providers",
        lambda: ["openai"],
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_allowed_embedding_models",
        lambda: ["text-embedding-3-*"],
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.should_enforce_embedding_policy",
        lambda user=None: False,
    )

    result = build_embedding_recipe_candidate_hints(user=None)

    assert result["candidates"][0]["status"] == "ready"


def test_candidate_hints_only_support_exact_or_trailing_star_model_patterns(monkeypatch) -> None:
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_simplified_embeddings_config",
        lambda: {
            "default_provider": "openai",
            "default_model": "text-embedding-3-small",
            "providers": [{"name": "openai", "models": ["text-embedding-3-small"], "api_key": "sk-test"}],
        },
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_allowed_embedding_providers",
        lambda: ["openai"],
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_allowed_embedding_models",
        lambda: ["text-embedding-?-small"],
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.should_enforce_embedding_policy",
        lambda user=None: True,
    )

    result = build_embedding_recipe_candidate_hints(user=None)

    assert result["candidates"][0]["status"] == "disallowed_model"


def test_candidate_hints_mark_missing_key_for_remote_provider(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_simplified_embeddings_config",
        lambda: {
            "default_provider": "openai",
            "default_model": "text-embedding-3-small",
            "providers": [{"name": "openai", "models": ["text-embedding-3-small"], "api_key": None}],
        },
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_allowed_embedding_providers",
        lambda: None,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_allowed_embedding_models",
        lambda: None,
    )

    result = build_embedding_recipe_candidate_hints(user=None)

    assert result["candidates"][0]["status"] == "missing_key"


def test_apply_preview_resolves_slot_to_copy_config(monkeypatch) -> None:
    class FakeService:
        def get_report(self, _run_id):
            return {
                "run": {
                    "run_id": "recipe-run-1",
                    "recipe_id": "embeddings_model_selection",
                    "status": RunStatus.COMPLETED,
                    "metadata": {},
                },
                "recommendation_slots": {
                    "best_overall": {
                        "candidate_run_id": "arm-1",
                        "metadata": {
                            "provider": "openai",
                            "model": "text-embedding-3-small",
                            "apply_eligible": True,
                            "apply_warnings": ["Existing indexes may need rebuild."],
                        },
                    }
                },
            }

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_current_embedding_config",
        lambda: {"provider": "huggingface", "model": "Qwen/Qwen3-Embedding-0.6B"},
    )

    preview = build_embedding_recipe_apply_preview(
        FakeService(),
        run_id="recipe-run-1",
        slot_name="best_overall",
        live_apply_supported=False,
    )

    assert preview["apply_eligible"] is True
    assert preview["apply_available"] is False
    assert preview["proposed"]["provider"] == "openai"
    assert preview["copy_config"]["Embeddings"]["embedding_model"] == "text-embedding-3-small"


def test_apply_preview_blocks_candidate_run_id_mismatch() -> None:
    class FakeService:
        def get_report(self, _run_id):
            return {
                "run": {
                    "run_id": "recipe-run-1",
                    "recipe_id": "embeddings_model_selection",
                    "status": "completed",
                    "metadata": {},
                },
                "recommendation_slots": {
                    "best_overall": {
                        "candidate_run_id": "arm-1",
                        "metadata": {
                            "provider": "openai",
                            "model": "text-embedding-3-small",
                            "apply_eligible": True,
                        },
                    }
                },
            }

    preview = build_embedding_recipe_apply_preview(
        FakeService(),
        run_id="recipe-run-1",
        slot_name="best_overall",
        candidate_run_id="arm-2",
    )

    assert preview["apply_eligible"] is False
    assert "candidate_run_id" in preview["blocked_reason"]
