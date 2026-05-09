from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tldw_Server_API.app.api.v1.schemas.evaluation_schemas_unified import RunStatus
from tldw_Server_API.app.core.Evaluations.recipes import embeddings_recipe_hints
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
            "default_model": "legacy-embedding",
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


def test_candidate_hints_include_current_model_when_provider_models_drift(monkeypatch) -> None:
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_simplified_embeddings_config",
        lambda: {
            "default_provider": "huggingface",
            "default_model": "BAAI/bge-small-en-v1.5",
            "providers": [],
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

    assert result["candidates"][0]["provider"] == "huggingface"
    assert result["candidates"][0]["model"] == "BAAI/bge-small-en-v1.5"
    assert result["candidates"][0]["default"] is True


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


def test_apply_preview_blocks_slot_without_candidate_run_id(monkeypatch) -> None:
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
                        "candidate_run_id": None,
                        "metadata": {
                            "provider": "openai",
                            "model": "text-embedding-3-small",
                            "apply_eligible": True,
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
    )

    assert preview["apply_eligible"] is False
    assert "candidate_run_id" in preview["blocked_reason"]


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


def test_apply_recommendation_updates_only_embedding_defaults_and_audits(monkeypatch) -> None:
    monkeypatch.delenv("EMBEDDINGS_DEFAULT_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDINGS_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDINGS_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDINGS_MODEL", raising=False)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_current_embedding_config",
        lambda: {"provider": "huggingface", "model": "Qwen/Qwen3-Embedding-0.6B"},
    )
    update_config_spy = MagicMock(return_value=Path("/tmp/config.txt.pre-setup-test.bak"))
    monkeypatch.setattr(
        embeddings_recipe_hints,
        "setup_manager",
        SimpleNamespace(update_config=update_config_spy),
        raising=False,
    )

    class FakeDB:
        def __init__(self) -> None:
            self.metadata: dict[str, object] | None = None

        def update_recipe_run(self, run_id: str, *, metadata: dict[str, object]) -> bool:
            assert run_id == "recipe-run-1"
            self.metadata = metadata
            return True

    class FakeService:
        def __init__(self) -> None:
            self.db = FakeDB()
            self.run = SimpleNamespace(metadata={"owner_user_id": "user_123"})

        def get_run(self, _run_id: str):
            return self.run

        def get_report(self, _run_id: str):
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
                        },
                    }
                },
            }

    service = FakeService()

    result = embeddings_recipe_hints.apply_embedding_recipe_recommendation(
        service,
        run_id="recipe-run-1",
        slot_name="best_overall",
        candidate_run_id="arm-1",
        confirmed_provider="openai",
        confirmed_model="text-embedding-3-small",
    )

    assert update_config_spy.call_args.args[0] == {
        "Embeddings": {
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
        }
    }
    assert update_config_spy.call_args.kwargs == {"create_backup": True}
    assert result["applied"] is True
    assert result["backup_path"] == "/tmp/config.txt.pre-setup-test.bak"
    assert result["audit_ref"] == "embedding_recipe_apply_audit"
    audit = service.db.metadata["embedding_recipe_apply_audit"]
    assert audit["slot"] == "best_overall"
    assert audit["candidate_run_id"] == "arm-1"
    assert audit["previous"] == {
        "provider": "huggingface",
        "model": "Qwen/Qwen3-Embedding-0.6B",
    }
    assert audit["proposed"] == {
        "provider": "openai",
        "model": "text-embedding-3-small",
    }
    assert audit["new"] == audit["proposed"]
    assert audit["backup_path"] == "/tmp/config.txt.pre-setup-test.bak"
    assert audit["status"] == "applied"
    assert audit["timestamp"]


def test_apply_recommendation_requires_pending_audit_before_config_mutation(monkeypatch) -> None:
    monkeypatch.delenv("EMBEDDINGS_DEFAULT_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDINGS_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDINGS_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDINGS_MODEL", raising=False)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_current_embedding_config",
        lambda: {"provider": "huggingface", "model": "Qwen/Qwen3-Embedding-0.6B"},
    )
    update_config_spy = MagicMock(return_value=Path("/tmp/config.txt.pre-setup-test.bak"))
    monkeypatch.setattr(
        embeddings_recipe_hints,
        "setup_manager",
        SimpleNamespace(update_config=update_config_spy),
        raising=False,
    )

    class FailingDB:
        def update_recipe_run(self, run_id: str, *, metadata: dict[str, object]) -> bool:
            assert run_id == "recipe-run-1"
            assert metadata["embedding_recipe_apply_audit"]["status"] == "pending"
            return False

    class FakeService:
        def __init__(self) -> None:
            self.db = FailingDB()
            self.run = SimpleNamespace(metadata={"owner_user_id": "user_123"})

        def get_run(self, _run_id: str):
            return self.run

        def get_report(self, _run_id: str):
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
                        },
                    }
                },
            }

    with pytest.raises(RuntimeError, match="audit metadata"):
        embeddings_recipe_hints.apply_embedding_recipe_recommendation(
            FakeService(),
            run_id="recipe-run-1",
            slot_name="best_overall",
            candidate_run_id="arm-1",
            confirmed_provider="openai",
            confirmed_model="text-embedding-3-small",
        )

    update_config_spy.assert_not_called()


def test_apply_recommendation_refuses_env_override_without_mutation(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDINGS_MODEL", "env-selected-model")
    update_config_spy = MagicMock()
    monkeypatch.setattr(
        embeddings_recipe_hints,
        "setup_manager",
        SimpleNamespace(update_config=update_config_spy),
        raising=False,
    )

    class FakeService:
        def get_report(self, _run_id: str):
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
                        },
                    }
                },
            }

    with pytest.raises(ValueError, match="environment variable"):
        embeddings_recipe_hints.apply_embedding_recipe_recommendation(
            FakeService(),
            run_id="recipe-run-1",
            slot_name="best_overall",
            candidate_run_id="arm-1",
            confirmed_provider="openai",
            confirmed_model="text-embedding-3-small",
        )

    update_config_spy.assert_not_called()
