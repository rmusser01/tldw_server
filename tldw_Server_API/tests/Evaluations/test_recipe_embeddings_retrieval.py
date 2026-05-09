from __future__ import annotations

from tldw_Server_API.app.core.Evaluations.recipes.embeddings_retrieval import (
    EmbeddingsRetrievalRecipe,
)


def test_manifest_advertises_guided_media_scoped_rag_flow() -> None:
    recipe = EmbeddingsRetrievalRecipe()
    source_labeling = recipe.manifest.capabilities["source_labeling"]
    candidate_discovery = recipe.manifest.capabilities["candidate_discovery"]
    apply_target = recipe.manifest.capabilities["apply_target"]

    assert recipe.manifest.capabilities["guided_ui"] is True
    assert source_labeling["supported"] is True
    assert source_labeling["source_id_contract"] == {
        "kind": "media_id",
        "type": "integer",
    }
    assert candidate_discovery["endpoint"].endswith(
        "/recipes/embeddings_model_selection/candidates"
    )
    assert set(candidate_discovery["runnable_statuses"]) == {
        "ready",
        "missing_key",
        "disallowed_provider",
        "disallowed_model",
        "quota_risk",
        "unknown",
    }
    assert apply_target["preview_supported"] is True
    assert apply_target["live_apply_supported"] is False
    assert apply_target["config_keys"] == ["embedding_provider", "embedding_model"]
    assert recipe.manifest.default_run_config["comparison_mode"] == "embedding_only"
    assert recipe.manifest.default_run_config["top_k"] == 10


def test_labeled_dataset_requires_query_ids_and_expected_ids() -> None:
    recipe = EmbeddingsRetrievalRecipe()

    result = recipe.validate_dataset(
        [
            {
                "query_id": "q-1",
                "input": "alpha query",
                "expected_ids": ["1", "2"],
            },
            {
                "input": "missing query id",
                "expected_ids": [],
            },
        ]
    )

    assert result["valid"] is False
    assert result["dataset_mode"] == "mixed"
    assert any("query_id" in error for error in result["errors"])
    assert any("expected_ids" in error for error in result["errors"])


def test_unlabeled_dataset_reserves_review_sample() -> None:
    recipe = EmbeddingsRetrievalRecipe()
    dataset = [
        {"query_id": f"q-{index}", "input": f"query {index}"}
        for index in range(12)
    ]

    result = recipe.validate_dataset(dataset)

    assert result["valid"] is True
    assert result["dataset_mode"] == "unlabeled"
    assert result["review_sample"]["required"] is True
    assert result["review_sample"]["sample_size"] == 3
    assert result["review_sample"]["sample_query_ids"] == ["q-0", "q-1", "q-2"]


def test_validation_warns_when_guided_queries_have_no_expected_sources() -> None:
    recipe = EmbeddingsRetrievalRecipe()

    result = recipe.validate_dataset(
        [{"query_id": "q-1", "input": "alpha query"}],
        run_config={"guided_source_labeling": True},
    )

    assert result["valid"] is True
    assert result["dataset_mode"] == "unlabeled"
    assert any("expected sources" in warning for warning in result["warnings"])


def test_validation_rejects_non_integer_expected_ids_for_media_scoped_guided_flow() -> None:
    recipe = EmbeddingsRetrievalRecipe()

    result = recipe.validate_dataset(
        [{"query_id": "q-1", "input": "alpha query", "expected_ids": ["chunk-123"]}],
        run_config={"source_id_contract": "media_id"},
    )

    assert result["valid"] is False
    assert any("media id" in error.lower() for error in result["errors"])


def test_recipe_supports_embedding_only_and_retrieval_stack_modes() -> None:
    recipe = EmbeddingsRetrievalRecipe()

    embedding_only = recipe.normalize_run_config(
        {
            "comparison_mode": "embedding_only",
            "candidates": [{"model": "openai:text-embedding-3-small"}],
        }
    )
    retrieval_stack = recipe.normalize_run_config(
        {
            "comparison_mode": "retrieval_stack",
            "candidates": [{"model": "local:bge-small", "reranker": "flashrank"}],
        }
    )

    assert embedding_only["comparison_mode"] == "embedding_only"
    assert retrieval_stack["comparison_mode"] == "retrieval_stack"
    assert embedding_only["candidates"][0]["model"] == "openai:text-embedding-3-small"


def test_build_report_emits_recommendation_slots_and_confidence_inputs() -> None:
    recipe = EmbeddingsRetrievalRecipe()

    report = recipe.build_report(
        dataset_mode="labeled",
        review_sample={"required": False, "sample_size": 0, "sample_query_ids": []},
        candidate_results=[
            {
                "candidate_id": "m1",
                "model": "openai:text-embedding-3-small",
                "provider": "openai",
                "is_local": False,
                "cost_usd": 0.12,
                "query_results": [
                    {
                        "ranked_ids": ["doc-1", "doc-3"],
                        "expected_ids": ["doc-1"],
                        "latency_ms": 110.0,
                    },
                    {
                        "ranked_ids": ["doc-2", "doc-4"],
                        "expected_ids": ["doc-2"],
                        "latency_ms": 125.0,
                    },
                ],
            },
            {
                "candidate_id": "m2",
                "model": "local:bge-small",
                "provider": "local",
                "is_local": True,
                "cost_usd": 0.0,
                "query_results": [
                    {
                        "ranked_ids": ["doc-1", "doc-4"],
                        "expected_ids": ["doc-1"],
                        "latency_ms": 85.0,
                    },
                    {
                        "ranked_ids": ["doc-4", "doc-2"],
                        "expected_ids": ["doc-2"],
                        "latency_ms": 90.0,
                    },
                ],
            },
        ],
    )

    candidates_by_id = {
        candidate["candidate_id"]: candidate for candidate in report["candidates"]
    }

    assert report["best_overall"]["candidate_id"] == "m1"
    assert report["best_cheap"]["candidate_id"] == "m2"
    assert report["best_local"]["candidate_id"] == "m2"
    assert report["confidence_summary"]["sample_count"] == 2
    assert report["confidence_summary"]["spread"] > 0.0
    assert "winner_margin" in report["confidence_inputs"]
    assert report["recommendation_slots"]["best_overall"]["candidate_run_id"] == "m1"
    assert report["recommendation_slots"]["best_overall"]["reason_code"] == "highest_quality_score"
    assert report["recommendation_slots"]["best_overall"]["metadata"]["provider"] == "openai"
    assert report["recommendation_slots"]["best_overall"]["metadata"]["model"] == "text-embedding-3-small"
    assert report["recommendation_slots"]["best_overall"]["metadata"]["apply_eligible"] is True
    assert "apply_warnings" in report["recommendation_slots"]["best_overall"]["metadata"]
    assert report["recommendation_slots"]["best_local"]["metadata"]["candidate_id"] == "m2"
    assert candidates_by_id["m1"]["metrics"]["recall_at_k"] > 0.0
    assert candidates_by_id["m1"]["metrics"]["quality_score"] > candidates_by_id["m2"]["metrics"]["quality_score"]


def test_build_report_accepts_abtest_result_rows_and_uses_conservative_sample_count() -> None:
    recipe = EmbeddingsRetrievalRecipe()

    report = recipe.build_report(
        dataset_mode="unlabeled",
        review_sample={"required": True, "sample_size": 2, "sample_query_ids": ["q-1", "q-2"]},
        candidate_results=[
            {
                "candidate_id": "arm-1",
                "candidate_run_id": "child-run-1",
                "model": "openai:text-embedding-3-small",
                "provider": "openai",
                "results": [
                    {
                        "query_id": "q-1",
                        "ranked_ids": ["doc-1", "doc-2"],
                        "metrics_json": {"recall_at_k": 1.0, "mrr": 1.0, "ndcg": 1.0},
                        "latency_ms": 111.0,
                    },
                    {
                        "query_id": "q-2",
                        "ranked_ids": ["doc-3", "doc-4"],
                        "metrics_json": {"recall_at_k": 0.5, "mrr": 0.5, "ndcg": 0.6},
                        "latency_ms": 120.0,
                    },
                ],
                "metrics": {"recall_at_k": 0.75, "mrr": 0.75, "ndcg": 0.8},
            },
            {
                "candidate_id": "arm-2",
                "candidate_run_id": "child-run-2",
                "model": "local:bge-small",
                "provider": "local",
                "is_local": True,
                "results": [
                    {
                        "query_id": "q-1",
                        "ranked_ids": ["doc-1", "doc-9"],
                        "metrics_json": {"recall_at_k": 1.0, "mrr": 1.0, "ndcg": 1.0},
                        "latency_ms": 90.0,
                    },
                ],
                "metrics": {"recall_at_k": 1.0, "mrr": 1.0, "ndcg": 1.0},
            },
        ],
    )

    assert report["best_overall"]["candidate_id"] == "arm-2"
    assert report["best_local"]["candidate_id"] == "arm-2"
    assert report["best_cheap"] is None
    assert report["recommendation_slots"]["best_overall"]["candidate_run_id"] == "child-run-2"
    assert report["recommendation_slots"]["best_cheap"]["reason_code"] == "not_available"
    assert report["confidence_summary"]["sample_count"] == 1
