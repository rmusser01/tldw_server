from __future__ import annotations

from typing import Any

import pytest

from pydantic import BaseModel

from tldw_Server_API.app.api.v1.schemas.evaluation_schemas_unified import RunStatus
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_single_user_instance
from tldw_Server_API.app.core.DB_Management.Evaluations_DB import EvaluationsDatabase
from tldw_Server_API.app.core.Evaluations.recipe_runs_service import (
    RECIPE_RUN_REUSE_ENTITY_TYPE,
    RecipeDefinitionNotLaunchableError,
    RecipeRunNotFoundError,
    RecipeRunsService,
)
from tldw_Server_API.app.core.Evaluations.recipes.base import RecipeDefinition
from tldw_Server_API.app.core.Evaluations.recipes.registry import RecipeRegistry
from tldw_Server_API.app.core.Evaluations.recipes.dataset_snapshot import build_dataset_content_hash


def _inline_dataset() -> list[dict[str, Any]]:
    return [
        {
            "input": "What is 2 + 2?",
            "expected": "4",
            "metadata": {"sample_id": "math-1"},
        },
        {
            "input": "What color is the sky on a clear day?",
            "expected": "blue",
            "metadata": {"sample_id": "sky-1"},
        },
    ]


def _run_config() -> dict[str, Any]:
    return {
        "candidate_model_ids": [
            "openai:gpt-4.1-mini",
            "ollama:llama3.1:8b",
        ],
        "judge_config": {
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "temperature": 0.0,
        },
        "prompts": {
            "system": "Grade the candidates carefully.",
            "user": "Prefer grounded and concise answers.",
        },
        "weights": {
            "quality": 0.7,
            "cost": 0.2,
            "latency": 0.1,
        },
        "comparison_mode": "pairwise",
        "source_normalization": {
            "strip_citations": True,
            "normalize_whitespace": True,
        },
        "context_policy": {
            "mode": "recipe_default",
            "allow_missing_context": False,
        },
        "execution_policy": {
            "max_parallel_candidates": 2,
            "capture_raw_judgments": True,
        },
    }


def _rag_dataset() -> list[dict[str, Any]]:
    return [
        {
            "sample_id": "q-1",
            "query": "find the architecture note",
            "targets": {
                "relevant_media_ids": [{"id": 10, "grade": 3}],
                "relevant_note_ids": [{"id": "note-7", "grade": 2}],
            },
        }
    ]


def _rag_run_config() -> dict[str, Any]:
    return {
        "candidate_creation_mode": "manual",
        "corpus_scope": {
            "sources": ["media_db", "notes"],
            "media_ids": [10],
            "note_ids": ["note-7"],
            "indexing_fixed": True,
        },
        "candidates": [
            {
                "candidate_id": "baseline",
                "retrieval_config": {
                    "search_mode": "hybrid",
                    "top_k": 5,
                    "hybrid_alpha": 0.7,
                    "enable_reranking": False,
                },
                "indexing_config": {"chunking_preset": "fixed_index"},
            },
            {
                "candidate_id": "rerank",
                "retrieval_config": {
                    "search_mode": "hybrid",
                    "top_k": 5,
                    "hybrid_alpha": 0.7,
                    "enable_reranking": True,
                    "reranking_strategy": "cross_encoder",
                    "rerank_top_k": 3,
                },
                "indexing_config": {"chunking_preset": "fixed_index"},
            },
        ],
    }


def _rag_answer_quality_dataset() -> list[dict[str, Any]]:
    return [
        {
            "sample_id": "aq-1",
            "query": "What is the capital of France?",
            "expected_behavior": "answer",
            "reference_answer": "Paris is the capital of France.",
            "inline_contexts": [
                {
                    "source": "knowledge_base",
                    "text": "Paris is the capital and most populous city of France.",
                }
            ],
        }
    ]


def _rag_answer_quality_run_config(prompt_variant: str = "default") -> dict[str, Any]:
    return {
        "evaluation_mode": "fixed_context",
        "supervision_mode": "mixed",
        "candidate_dimensions": [
            "generation_model",
            "prompt_variant",
            "formatting_citation_mode",
        ],
        "weights": {
            "grounding": 0.4,
            "answer_relevance": 0.3,
            "format_style": 0.2,
            "abstention_behavior": 0.1,
        },
        "grounding_threshold": 0.7,
        "context_snapshot_ref": "context-snapshot-1",
        "candidates": [
            {
                "candidate_id": "openai-gpt-4.1-mini",
                "provider": "openai",
                "model": "gpt-4.1-mini",
                "prompt_variant": prompt_variant,
                "formatting_citation_mode": "citations",
                "is_local": False,
                "cost_usd": 0.05,
            },
            {
                "candidate_id": "ollama-llama3.1-8b",
                "provider": "ollama",
                "model": "llama3.1:8b",
                "prompt_variant": "direct",
                "formatting_citation_mode": "plain",
                "is_local": True,
                "cost_usd": 0.0,
            },
        ],
        "prompts": {
            "system": "Answer with citations only when grounded.",
            "user": "Use the provided context and abstain if support is missing.",
        },
        "judge_config": {"provider": "openai", "model": "gpt-4.1-mini", "temperature": 0.0},
        "execution_policy": {"temperature": 0.1},
    }


def _service(tmp_path) -> tuple[EvaluationsDatabase, RecipeRunsService, str]:
    db = EvaluationsDatabase(str(tmp_path / "evaluations.db"))
    user_id = get_single_user_instance().id_str
    return db, RecipeRunsService(db=db, user_id=user_id), user_id


def _mark_recipe_run_completed(db: EvaluationsDatabase, run_id: str) -> None:
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE evaluation_recipe_runs
            SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE run_id = ?
            """,
            (RunStatus.COMPLETED.value, run_id),
        )
        conn.commit()


def _set_reuse_mapping(
    db: EvaluationsDatabase,
    *,
    reuse_hash: str,
    user_id: str,
    run_id: str,
) -> None:
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE idempotency_keys
            SET entity_id = ?
            WHERE user_id = ? AND entity_type = ? AND idempotency_key = ?
            """,
            (run_id, user_id, RECIPE_RUN_REUSE_ENTITY_TYPE, reuse_hash),
        )
        conn.commit()


def test_recipe_service_lists_and_fetches_builtin_manifests(tmp_path) -> None:
    _, service, _ = _service(tmp_path)

    manifests = service.list_manifests()

    assert {manifest.recipe_id for manifest in manifests} >= {
        "embeddings_model_selection",
        "summarization_quality",
    }

    manifest = service.get_manifest("summarization_quality")

    assert manifest.recipe_id == "summarization_quality"
    assert manifest.recipe_version == "1"


def test_recipe_service_validates_rag_retrieval_tuning_dataset(tmp_path) -> None:
    _, service, _ = _service(tmp_path)

    result = service.validate_dataset(
        "rag_retrieval_tuning",
        dataset=_rag_dataset(),
        run_config=_rag_run_config(),
    )

    assert result["valid"] is True
    assert result["dataset_mode"] == "labeled"
    assert result["corpus_scope"]["sources"] == ["media_db", "notes"]


def test_recipe_service_preserves_embeddings_guided_validation_config(tmp_path) -> None:
    _, service, _ = _service(tmp_path)
    dataset = [{"query_id": "q-1", "input": "alpha query"}]
    run_config = {
        "comparison_mode": "embedding_only",
        "guided_source_labeling": True,
        "source_id_contract": "media_id",
        "candidates": [],
    }

    result = service.validate_dataset(
        "embeddings_model_selection",
        dataset=dataset,
        run_config=run_config,
    )

    assert result["valid"] is True
    assert result["dataset_mode"] == "unlabeled"
    assert any("expected sources" in warning for warning in result["warnings"])
    with pytest.raises(ValueError, match="candidates"):
        service.create_run(
            "embeddings_model_selection",
            dataset=dataset,
            run_config=run_config,
        )


def test_recipe_runs_ignore_unapproved_synthetic_drafts(tmp_path) -> None:
    _, service, _ = _service(tmp_path)

    result = service.validate_dataset(
        "rag_retrieval_tuning",
        dataset=[
            {
                "sample_id": "synthetic-draft-1",
                "query": "find the architecture note",
                "targets": {
                    "relevant_media_ids": [{"id": 10, "grade": 3}],
                },
                "metadata": {
                    "synthetic_eval": {
                        "sample_id": "synthetic-draft-1",
                        "review_state": "draft",
                        "promotion_state": "not_promoted",
                        "provenance": "synthetic_from_corpus",
                    }
                },
            }
        ],
        run_config=_rag_run_config(),
    )

    assert result["valid"] is False
    assert result["sample_count"] == 0
    assert any("Dataset must contain at least one sample." in error for error in result["errors"])


def test_recipe_service_creates_rag_retrieval_tuning_run(tmp_path) -> None:
    _, service, _ = _service(tmp_path)

    record = service.create_run(
        "rag_retrieval_tuning",
        dataset=_rag_dataset(),
        run_config=_rag_run_config(),
    )

    assert record.status is RunStatus.PENDING
    assert record.metadata["run_config"]["corpus_scope"]["sources"] == ["media_db", "notes"]
    assert record.metadata["recipe_validation"]["corpus_scope"]["media_ids"] == ["10"]


def test_recipe_service_validates_dataset_shape_and_labeling_mode(tmp_path) -> None:
    _, service, _ = _service(tmp_path)

    result = service.validate_dataset(
        "summarization_quality",
        dataset=[
            {"input": "valid prompt", "expected": "summary"},
            {"input": "missing label partner"},
        ],
    )

    assert result["valid"] is False
    assert result["dataset_mode"] == "mixed"
    assert any("consistent labeling mode" in error for error in result["errors"])


def test_recipe_service_creates_parent_run_and_normalized_report_shell(tmp_path) -> None:
    _, service, _ = _service(tmp_path)
    dataset = _inline_dataset()
    run_config = _run_config()

    record = service.create_run(
        "summarization_quality",
        dataset=dataset,
        run_config=run_config,
    )

    assert record.status is RunStatus.PENDING
    assert record.dataset_content_hash == build_dataset_content_hash(dataset)
    assert record.metadata["run_config"] == run_config
    assert "reuse_hash" in record.metadata
    assert record.metadata["owner_user_id"]

    fetched = service.get_run(record.run_id)
    assert fetched is not None
    assert fetched.run_id == record.run_id
    assert fetched.metadata["run_config"] == run_config

    report = service.get_report(record.run_id)

    assert report.run.run_id == record.run_id
    assert set(report.recommendation_slots) == {
        "best_overall",
        "best_quality",
        "best_cheap",
        "best_local",
    }
    for slot in report.recommendation_slots.values():
        assert slot.candidate_run_id is None
        assert slot.reason_code is not None


def test_build_reuse_hash_for_rag_retrieval_tuning_includes_corpus_scope(tmp_path) -> None:
    _, service, _ = _service(tmp_path)
    dataset = _rag_dataset()
    run_config_a = _rag_run_config()
    run_config_b = {
        **_rag_run_config(),
        "corpus_scope": {
            **_rag_run_config()["corpus_scope"],
            "media_ids": [10, 11],
        },
    }

    reuse_hash_a = service.build_reuse_hash(
        "rag_retrieval_tuning",
        dataset=dataset,
        run_config=run_config_a,
    )
    reuse_hash_b = service.build_reuse_hash(
        "rag_retrieval_tuning",
        dataset=dataset,
        run_config=run_config_b,
    )

    assert reuse_hash_a != reuse_hash_b


def test_recipe_service_uses_embeddings_recipe_validation_and_normalization(tmp_path) -> None:
    _, service, _ = _service(tmp_path)
    dataset = [
        {
            "query_id": "q-1",
            "input": "find the alpha document",
            "expected_ids": ["1"],
        },
    ]
    run_config = {
        "comparison_mode": "embedding_only",
        "candidates": [
            {"model": "local:bge-small", "provider": "local", "is_local": True},
            {"model": "openai:text-embedding-3-small", "provider": "openai", "is_local": False},
        ],
    }

    validation = service.validate_dataset(
        "embeddings_model_selection",
        dataset=dataset,
    )

    assert validation["valid"] is True
    assert validation["dataset_mode"] == "labeled"
    assert validation["sample_count"] == 1

    reuse_hash_a = service.build_reuse_hash(
        "embeddings_model_selection",
        dataset=dataset,
        run_config=run_config,
    )
    reuse_hash_b = service.build_reuse_hash(
        "embeddings_model_selection",
        dataset=dataset,
        run_config={
            "comparison_mode": "embedding_only",
            "candidates": list(reversed(run_config["candidates"])),
        },
    )
    record = service.create_run(
        "embeddings_model_selection",
        dataset=dataset,
        run_config=run_config,
    )

    assert reuse_hash_a == reuse_hash_b
    assert record.metadata["run_config"]["comparison_mode"] == "embedding_only"
    assert [candidate["model"] for candidate in record.metadata["run_config"]["candidates"]] == [
        "local:bge-small",
        "openai:text-embedding-3-small",
    ]
    assert record.metadata["inline_dataset"] == dataset
    assert record.metadata["review_sample"] == {
        "required": False,
        "sample_size": 0,
        "sample_query_ids": [],
    }


def test_recipe_service_builds_embeddings_report_from_stored_recipe_inputs(tmp_path) -> None:
    db, service, _ = _service(tmp_path)
    dataset = [
        {"query_id": "q-1", "input": "alpha", "expected_ids": ["1"]},
        {"query_id": "q-2", "input": "beta", "expected_ids": ["2"]},
    ]
    record = service.create_run(
        "embeddings_model_selection",
        dataset=dataset,
        run_config={
            "comparison_mode": "embedding_only",
            "candidates": [
                {"model": "openai:text-embedding-3-small", "provider": "openai"},
                {"model": "local:bge-small", "provider": "local", "is_local": True},
            ],
        },
    )

    db.update_recipe_run(
        record.run_id,
        metadata={
            **record.metadata,
            "candidate_results": [
                {
                    "candidate_id": "arm-1",
                    "candidate_run_id": "child-run-1",
                    "model": "openai:text-embedding-3-small",
                    "provider": "openai",
                    "query_results": [
                        {
                            "ranked_ids": ["doc-1"],
                            "expected_ids": ["doc-1"],
                            "latency_ms": 100.0,
                        },
                        {
                            "ranked_ids": ["doc-2"],
                            "expected_ids": ["doc-2"],
                            "latency_ms": 105.0,
                        },
                    ],
                },
                {
                    "candidate_id": "arm-2",
                    "candidate_run_id": "child-run-2",
                    "model": "local:bge-small",
                    "provider": "local",
                    "is_local": True,
                    "query_results": [
                        {
                            "ranked_ids": ["doc-1"],
                            "expected_ids": ["doc-1"],
                            "latency_ms": 95.0,
                        },
                        {
                            "ranked_ids": ["doc-9"],
                            "expected_ids": ["doc-2"],
                            "latency_ms": 96.0,
                        },
                    ],
                },
            ],
        },
    )

    report = service.get_report(record.run_id)

    assert report.confidence_summary is not None
    assert report.recommendation_slots["best_overall"].candidate_run_id == "child-run-1"
    assert report.run.metadata["recipe_report"]["best_overall"]["candidate_id"] == "arm-1"


def test_recipe_service_builds_summarization_report_from_stored_recipe_inputs(tmp_path) -> None:
    db, service, _ = _service(tmp_path)
    record = service.create_run(
        "summarization_quality",
        dataset=_inline_dataset(),
        run_config=_run_config(),
    )

    db.update_recipe_run(
        record.run_id,
        metadata={
            **record.metadata,
            "recipe_report_inputs": {
                "dataset_mode": "labeled",
                "review_sample": {"required": False, "sample_size": 0, "sample_ids": []},
                "weights": {"grounding": 0.5, "coverage": 0.3, "usefulness": 0.2},
                "candidate_results": [
                    {
                        "candidate_id": "cand-openai",
                        "candidate_run_id": "cand-openai",
                        "provider": "openai",
                        "model": "gpt-4.1-mini",
                        "cost_usd": 0.01,
                        "sample_results": [
                            {
                                "sample_id": "math-1",
                                "metrics": {
                                    "grounding": 0.92,
                                    "coverage": 0.88,
                                    "usefulness": 0.80,
                                },
                                "latency_ms": 110.0,
                            },
                            {
                                "sample_id": "sky-1",
                                "metrics": {
                                    "grounding": 0.89,
                                    "coverage": 0.86,
                                    "usefulness": 0.81,
                                },
                                "latency_ms": 115.0,
                            },
                        ],
                    },
                    {
                        "candidate_id": "cand-local",
                        "candidate_run_id": "cand-local",
                        "provider": "ollama",
                        "model": "llama3.1:8b",
                        "is_local": True,
                        "cost_usd": 0.0,
                        "sample_results": [
                            {
                                "sample_id": "math-1",
                                "metrics": {
                                    "grounding": 0.80,
                                    "coverage": 0.77,
                                    "usefulness": 0.79,
                                },
                                "latency_ms": 80.0,
                            },
                            {
                                "sample_id": "sky-1",
                                "metrics": {
                                    "grounding": 0.78,
                                    "coverage": 0.76,
                                    "usefulness": 0.78,
                                },
                                "latency_ms": 82.0,
                            },
                        ],
                    },
                ],
            },
        },
    )

    report = service.get_report(record.run_id)

    assert report.confidence_summary is not None
    assert report.recommendation_slots["best_overall"].candidate_run_id == "cand-openai"
    assert report.recommendation_slots["best_local"].candidate_run_id == "cand-local"
    assert report.run.metadata["recipe_report"]["best_overall"]["candidate_id"] == "cand-openai"


def test_recipe_service_builds_rag_report_from_stored_recipe_inputs(tmp_path) -> None:
    db, service, _ = _service(tmp_path)
    record = service.create_run(
        "rag_retrieval_tuning",
        dataset=_rag_dataset(),
        run_config=_rag_run_config(),
    )

    db.update_recipe_run(
        record.run_id,
        metadata={
            **record.metadata,
            "recipe_report_inputs": {
                "dataset_mode": "labeled",
                "review_sample": {"required": False, "sample_size": 0, "sample_ids": []},
                "corpus_scope": _rag_run_config()["corpus_scope"],
                "candidate_results": [
                    {
                        "candidate_id": "baseline",
                        "candidate_run_id": "baseline",
                        "latency_ms": 120.0,
                        "metrics": {
                            "pre_rerank_recall_at_k": 0.50,
                            "post_rerank_ndcg_at_k": 0.55,
                            "first_pass_recall_score": 0.50,
                            "post_rerank_quality_score": 0.55,
                        },
                    },
                    {
                        "candidate_id": "rerank",
                        "candidate_run_id": "rerank",
                        "is_local": True,
                        "cost_usd": 0.0,
                        "latency_ms": 95.0,
                        "metrics": {
                            "pre_rerank_recall_at_k": 0.75,
                            "post_rerank_ndcg_at_k": 0.82,
                            "first_pass_recall_score": 0.75,
                            "post_rerank_quality_score": 0.82,
                        },
                    },
                ],
            },
        },
    )

    report = service.get_report(record.run_id)

    assert report.confidence_summary is not None
    assert report.recommendation_slots["best_overall"].candidate_run_id == "rerank"
    assert report.recommendation_slots["best_local"].candidate_run_id == "rerank"
    assert report.run.metadata["recipe_report"]["best_overall"]["candidate_id"] == "rerank"


def test_recipe_service_preserves_rag_answer_quality_candidates_and_reports_from_inputs(
    tmp_path,
) -> None:
    db, service, _ = _service(tmp_path)
    dataset = _rag_answer_quality_dataset()
    run_config = _rag_answer_quality_run_config(prompt_variant="default")

    record = service.create_run(
        "rag_answer_quality",
        dataset=dataset,
        run_config=run_config,
    )

    assert record.metadata["run_config"]["candidates"][0]["prompt_variant"] == "default"

    hash_a = service.build_reuse_hash(
        "rag_answer_quality",
        dataset=dataset,
        run_config=run_config,
    )
    hash_b = service.build_reuse_hash(
        "rag_answer_quality",
        dataset=dataset,
        run_config=_rag_answer_quality_run_config(prompt_variant="concise"),
    )

    assert hash_a != hash_b

    db.update_recipe_run(
        record.run_id,
        metadata={
            **record.metadata,
            "recipe_report_inputs": {
                "dataset_mode": "labeled",
                "review_sample": {"required": False, "sample_size": 0, "sample_ids": []},
                "evaluation_mode": "fixed_context",
                "supervision_mode": "mixed",
                "context_snapshot_ref": "context-snapshot-1",
                "grounding_threshold": 0.7,
                "weights": {
                    "grounding": 0.4,
                    "answer_relevance": 0.3,
                    "format_style": 0.2,
                    "abstention_behavior": 0.1,
                },
                "candidate_results": [
                    {
                        "candidate_id": "openai-gpt-4.1-mini",
                        "candidate_run_id": "openai-gpt-4.1-mini",
                        "provider": "openai",
                        "model": "gpt-4.1-mini",
                        "prompt_variant": "default",
                        "formatting_citation_mode": "citations",
                        "is_local": False,
                        "cost_usd": 0.05,
                        "sample_results": [
                            {
                                "sample_id": "aq-1",
                                "query": "What is the capital of France?",
                                "answer": "Paris is the capital of France.",
                                "reference_answer": "Paris is the capital of France.",
                                "metrics": {
                                    "grounding": 0.92,
                                    "answer_relevance": 0.91,
                                    "format_style": 0.88,
                                    "abstention_behavior": 0.93,
                                },
                                "latency_ms": 110.0,
                            }
                        ],
                    },
                    {
                        "candidate_id": "ollama-llama3.1-8b",
                        "candidate_run_id": "ollama-llama3.1-8b",
                        "provider": "ollama",
                        "model": "llama3.1:8b",
                        "prompt_variant": "direct",
                        "formatting_citation_mode": "plain",
                        "is_local": True,
                        "cost_usd": 0.0,
                        "sample_results": [
                            {
                                "sample_id": "aq-1",
                                "query": "What is the capital of France?",
                                "answer": "It appears to be Paris, based on the context.",
                                "reference_answer": "Paris is the capital of France.",
                                "metrics": {
                                    "grounding": 0.75,
                                    "answer_relevance": 0.74,
                                    "format_style": 0.80,
                                    "abstention_behavior": 0.76,
                                },
                                "latency_ms": 120.0,
                            }
                        ],
                    },
                ],
            },
        },
    )

    report = service.get_report(record.run_id)

    assert report.recommendation_slots["best_overall"].candidate_run_id == "openai-gpt-4.1-mini"
    assert report.recommendation_slots["best_quality"].candidate_run_id == "openai-gpt-4.1-mini"
    assert report.recommendation_slots["best_local"].candidate_run_id == "ollama-llama3.1-8b"
    assert report.run.metadata["recipe_report"]["best_overall"]["candidate_id"] == "openai-gpt-4.1-mini"


def test_build_reuse_hash_for_rag_answer_quality_includes_prompt_and_judge_settings(tmp_path) -> None:
    _, service, _ = _service(tmp_path)
    dataset = _rag_answer_quality_dataset()
    base = _rag_answer_quality_run_config()
    changed_prompts = {
        **_rag_answer_quality_run_config(),
        "prompts": {
            **_rag_answer_quality_run_config()["prompts"],
            "system": "Use terse grounded answers.",
        },
    }
    changed_judge = {
        **_rag_answer_quality_run_config(),
        "judge_config": {"provider": "openai", "model": "gpt-4.1"},
    }

    base_hash = service.build_reuse_hash(
        "rag_answer_quality",
        dataset=dataset,
        run_config=base,
    )
    prompt_hash = service.build_reuse_hash(
        "rag_answer_quality",
        dataset=dataset,
        run_config=changed_prompts,
    )
    judge_hash = service.build_reuse_hash(
        "rag_answer_quality",
        dataset=dataset,
        run_config=changed_judge,
    )

    assert base_hash != prompt_hash
    assert base_hash != judge_hash


def test_build_reuse_hash_for_live_rag_answer_quality_includes_retrieval_settings(tmp_path) -> None:
    _, service, _ = _service(tmp_path)
    dataset = _rag_answer_quality_dataset()
    base = {
        **_rag_answer_quality_run_config(),
        "evaluation_mode": "live_end_to_end",
        "retrieval_baseline_ref": "baseline-run-42",
        "search_mode": "hybrid",
        "top_k": 5,
    }
    changed = {
        **base,
        "top_k": 12,
    }

    base_hash = service.build_reuse_hash(
        "rag_answer_quality",
        dataset=dataset,
        run_config=base,
    )
    changed_hash = service.build_reuse_hash(
        "rag_answer_quality",
        dataset=dataset,
        run_config=changed,
    )

    assert base_hash != changed_hash


def test_recipe_service_rejects_rag_answer_quality_runs_without_candidates(tmp_path) -> None:
    _, service, _ = _service(tmp_path)

    with pytest.raises(ValueError, match="run_config.candidates"):
        service.create_run(
            "rag_answer_quality",
            dataset=_rag_answer_quality_dataset(),
            run_config={
                "evaluation_mode": "fixed_context",
                "supervision_mode": "rubric",
                "context_snapshot_ref": "context-snapshot-1",
                "candidates": [],
            },
        )


def test_recipe_service_reuses_completed_run_unless_force_rerun(tmp_path) -> None:
    db, service, user_id = _service(tmp_path)
    dataset = _inline_dataset()
    run_config = _run_config()
    created = service.create_run(
        "summarization_quality",
        dataset=dataset,
        run_config=run_config,
    )
    reuse_hash = created.metadata["reuse_hash"]

    assert (
        db.lookup_idempotency(
            RECIPE_RUN_REUSE_ENTITY_TYPE,
            reuse_hash,
            user_id,
        )
        == created.run_id
    )

    _mark_recipe_run_completed(db, created.run_id)

    reused = service.create_run(
        "summarization_quality",
        dataset=dataset,
        run_config=run_config,
    )

    assert reused.run_id == created.run_id
    assert reused.status is RunStatus.COMPLETED
    assert (
        db.lookup_idempotency(
            RECIPE_RUN_REUSE_ENTITY_TYPE,
            reuse_hash,
            user_id,
        )
        == created.run_id
    )

    forced = service.create_run(
        "summarization_quality",
        dataset=dataset,
        run_config=run_config,
        force_rerun=True,
    )

    assert forced.run_id != created.run_id
    assert forced.status is RunStatus.PENDING


def test_recipe_service_repairs_stale_reuse_mapping_to_latest_completed_run(tmp_path) -> None:
    db, service, user_id = _service(tmp_path)
    dataset = _inline_dataset()
    run_config = _run_config()

    first_run = service.create_run(
        "summarization_quality",
        dataset=dataset,
        run_config=run_config,
    )
    reuse_hash = first_run.metadata["reuse_hash"]
    db.record_idempotency(
        RECIPE_RUN_REUSE_ENTITY_TYPE,
        reuse_hash,
        first_run.run_id,
        user_id,
    )

    forced_run = service.create_run(
        "summarization_quality",
        dataset=dataset,
        run_config=run_config,
        force_rerun=True,
    )
    _mark_recipe_run_completed(db, forced_run.run_id)
    _set_reuse_mapping(
        db,
        reuse_hash=reuse_hash,
        user_id=user_id,
        run_id=first_run.run_id,
    )

    reused = service.create_run(
        "summarization_quality",
        dataset=dataset,
        run_config=run_config,
    )

    assert reused.run_id == forced_run.run_id
    assert reused.status is RunStatus.COMPLETED
    assert (
        db.lookup_idempotency(
            RECIPE_RUN_REUSE_ENTITY_TYPE,
            reuse_hash,
            user_id,
        )
        == forced_run.run_id
    )


def test_recipe_service_does_not_reuse_completed_run_from_other_user(tmp_path) -> None:
    db, service, _ = _service(tmp_path)
    dataset = _inline_dataset()
    run_config = _run_config()

    other_user_service = RecipeRunsService(db=db, user_id="other-user")
    other_user_run = other_user_service.create_run(
        "summarization_quality",
        dataset=dataset,
        run_config=run_config,
    )
    _mark_recipe_run_completed(db, other_user_run.run_id)

    created = service.create_run(
        "summarization_quality",
        dataset=dataset,
        run_config=run_config,
    )

    assert created.run_id != other_user_run.run_id
    assert created.status is RunStatus.PENDING


def test_recipe_service_reuses_legacy_completed_run_without_owner_in_single_user_mode(tmp_path) -> None:
    db, service, _ = _service(tmp_path)
    dataset = _inline_dataset()
    run_config = _run_config()
    reuse_hash = service.build_reuse_hash(
        "summarization_quality",
        dataset=dataset,
        run_config=run_config,
    )

    legacy_run_id = db.create_recipe_run(
        recipe_id="summarization_quality",
        recipe_version="1",
        status=RunStatus.COMPLETED,
        dataset_content_hash=build_dataset_content_hash(dataset),
        metadata={
            "run_config": run_config,
            "reuse_hash": reuse_hash,
        },
    )

    reused = service.create_run(
        "summarization_quality",
        dataset=dataset,
        run_config=run_config,
    )

    assert reused.run_id == legacy_run_id
    assert reused.status is RunStatus.COMPLETED


def test_recipe_service_create_run_validates_dataset_once_per_request(tmp_path) -> None:
    class _CountingRecipe(RecipeDefinition):
        recipe_id = "counting_recipe"
        recipe_version = "1"

        def __init__(self) -> None:
            self.validation_calls = 0

        def get_manifest(self):
            class _Manifest(BaseModel):
                recipe_id: str = "counting_recipe"
                recipe_version: str = "1"
                name: str = "Counting Recipe"
                description: str = "Counts validation calls."
                launchable: bool = True
                supported_modes: list[str] = ["labeled"]
                tags: list[str] = []
                capabilities: dict[str, Any] = {}
                default_run_config: dict[str, Any] = {}

            return _Manifest()

        def validate_dataset(self, dataset: list[dict[str, Any]], *, run_config: dict[str, Any] | None = None):
            del dataset, run_config
            self.validation_calls += 1
            return {
                "valid": True,
                "errors": [],
                "dataset_mode": "labeled",
                "sample_count": 1,
                "review_sample": {"required": False, "sample_size": 0, "sample_ids": []},
            }

    db = EvaluationsDatabase(str(tmp_path / "evaluations.db"))
    recipe = _CountingRecipe()
    service = RecipeRunsService(
        db=db,
        user_id=get_single_user_instance().id_str,
        recipe_registry=RecipeRegistry(recipes=(recipe,)),
    )

    record = service.create_run(
        "counting_recipe",
        dataset=[{"input": "hello", "expected": "world"}],
        run_config={
            "candidate_model_ids": ["openai:gpt-4.1-mini"],
            "comparison_mode": "leaderboard",
            "weights": {"quality": 1.0},
        },
    )

    assert record.status is RunStatus.PENDING
    assert recipe.validation_calls == 1


def test_recipe_service_rejects_non_numeric_weight_values_with_field_name(tmp_path) -> None:
    _, service, _ = _service(tmp_path)

    with pytest.raises(ValueError, match=r"run_config\.weights\.quality must be numeric"):
        service.create_run(
            "summarization_quality",
            dataset=_inline_dataset(),
            run_config={
                **_run_config(),
                "weights": {"quality": "not-a-number"},
            },
        )


def test_recipe_service_get_report_ignores_invalid_confidence_summary_payload(
    tmp_path,
    monkeypatch,
) -> None:
    db, service, _ = _service(tmp_path)
    record = service.create_run(
        "summarization_quality",
        dataset=_inline_dataset(),
        run_config=_run_config(),
    )
    recipe = service.recipe_registry.get_recipe("summarization_quality")
    original_build_report = recipe.build_report

    def _invalid_build_report(**kwargs):
        report = dict(original_build_report(**kwargs))
        report["confidence_summary"] = {"confidence": "bad"}
        return report

    monkeypatch.setattr(recipe, "build_report", _invalid_build_report)

    db.update_recipe_run(
        record.run_id,
        metadata={
            **record.metadata,
            "recipe_report_inputs": {
                "dataset_mode": "labeled",
                "review_sample": {"required": False, "sample_size": 0, "sample_ids": []},
                "weights": {"grounding": 0.5, "coverage": 0.3, "usefulness": 0.2},
                "candidate_results": [
                    {
                        "candidate_id": "cand-openai",
                        "candidate_run_id": "cand-openai",
                        "provider": "openai",
                        "model": "gpt-4.1-mini",
                        "sample_results": [
                            {
                                "sample_id": "sample-1",
                                "metrics": {
                                    "grounding": 0.8,
                                    "coverage": 0.8,
                                    "usefulness": 0.8,
                                },
                                "latency_ms": 100.0,
                            }
                        ],
                    }
                ],
            },
        },
    )

    report = service.get_report(record.run_id)

    assert report.confidence_summary is None


def test_find_latest_completed_run_by_reuse_hash_does_not_refetch_full_record(tmp_path, monkeypatch) -> None:
    db, service, _ = _service(tmp_path)
    created = service.create_run(
        "summarization_quality",
        dataset=_inline_dataset(),
        run_config=_run_config(),
    )
    _mark_recipe_run_completed(db, created.run_id)

    reuse_hash = created.metadata["reuse_hash"]

    def _unexpected_get_recipe_run(_run_id: str):
        raise AssertionError("reuse-hash lookup should not refetch the full recipe run record")

    monkeypatch.setattr(db, "get_recipe_run", _unexpected_get_recipe_run)

    record = service._find_latest_completed_run_by_reuse_hash(reuse_hash)

    assert record is not None
    assert record.run_id == created.run_id
    assert record.status is RunStatus.COMPLETED


def test_recipe_service_redacts_sensitive_run_config_in_public_metadata(tmp_path) -> None:
    db, service, _ = _service(tmp_path)
    record = service.create_run(
        "rag_answer_quality",
        dataset=_rag_answer_quality_dataset(),
        run_config={
            **_rag_answer_quality_run_config(prompt_variant="default"),
            "candidate_api_keys": {
                "openai": "sk-live-secret",
                "ollama": "ollama-local-secret",
            },
            "judge_config": {
                "provider": "openai",
                "model": "gpt-4.1-mini",
                "api_key": "sk-judge-secret",
            },
        },
    )

    assert record.metadata["run_config"]["judge_config"]["api_key"] == "[REDACTED]"
    assert "run_config_internal" not in record.metadata

    raw_record = db.get_recipe_run(record.run_id)
    assert raw_record is not None
    assert raw_record.metadata["run_config_internal"]["judge_config"]["api_key"] == "sk-judge-secret"


def test_recipe_service_get_run_rejects_other_users_run(tmp_path) -> None:
    db, _, _ = _service(tmp_path)
    other_user_service = RecipeRunsService(db=db, user_id="other-user")
    record = other_user_service.create_run(
        "summarization_quality",
        dataset=_inline_dataset(),
        run_config=_run_config(),
    )

    current_user_service = RecipeRunsService(
        db=db,
        user_id=get_single_user_instance().id_str,
    )

    with pytest.raises(RecipeRunNotFoundError):
        current_user_service.get_run(record.run_id)
