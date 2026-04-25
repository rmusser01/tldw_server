from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.schemas.evaluation_schemas_unified import RunStatus
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_single_user_instance
from tldw_Server_API.app.core.DB_Management.Evaluations_DB import EvaluationsDatabase
from tldw_Server_API.app.core.Evaluations.recipe_runs_jobs import (
    RECIPE_RUN_JOB_DOMAIN,
    RECIPE_RUN_JOB_TYPE,
    build_recipe_run_idempotency_key,
    build_recipe_run_job_payload,
    enqueue_recipe_run,
    recipe_run_queue,
)
from tldw_Server_API.app.core.Evaluations.recipe_runs_jobs_worker import (
    RecipeRunJobError,
    handle_recipe_run_job,
    start_recipe_run_jobs_worker,
)
from tldw_Server_API.app.core.Evaluations.recipe_runs_service import RecipeRunsService


def _inline_dataset() -> list[dict[str, Any]]:
    return [
        {
            "input": "Summarize the notes.",
            "expected": "A short summary.",
        }
    ]


def _run_config() -> dict[str, Any]:
    return {
        "candidate_model_ids": ["openai:gpt-4.1-mini"],
        "judge_config": {"provider": "openai", "model": "gpt-4.1-mini"},
        "prompts": {"system": "Grade carefully."},
        "weights": {"quality": 1.0},
        "comparison_mode": "leaderboard",
        "source_normalization": {},
        "context_policy": {"mode": "recipe_default"},
        "execution_policy": {"max_parallel_candidates": 1},
    }


def _embeddings_dataset() -> list[dict[str, Any]]:
    return [
        {
            "query_id": "q-1",
            "input": "find alpha",
            "expected_ids": ["1"],
        },
        {
            "query_id": "q-2",
            "input": "find beta",
            "expected_ids": ["2"],
        },
    ]


def _embeddings_run_config() -> dict[str, Any]:
    return {
        "comparison_mode": "embedding_only",
        "candidates": [
            {"provider": "openai", "model": "text-embedding-3-small"},
            {"provider": "local", "model": "bge-small", "is_local": True},
        ],
        "media_ids": [1, 2],
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
            }
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


def _rag_answer_quality_run_config() -> dict[str, Any]:
    return {
        "evaluation_mode": "fixed_context",
        "supervision_mode": "mixed",
        "context_snapshot_ref": "context-snapshot-1",
        "weights": {
            "grounding": 0.4,
            "answer_relevance": 0.3,
            "format_style": 0.2,
            "abstention_behavior": 0.1,
        },
        "grounding_threshold": 0.7,
        "candidates": [
            {
                "candidate_id": "openai:gpt-4.1-mini::default::citations",
                "provider": "openai",
                "model": "gpt-4.1-mini",
                "prompt_variant": "default",
                "formatting_citation_mode": "citations",
                "is_local": False,
                "cost_usd": 0.05,
            },
            {
                "candidate_id": "ollama:llama3.1:8b::direct::plain",
                "provider": "ollama",
                "model": "llama3.1:8b",
                "prompt_variant": "direct",
                "formatting_citation_mode": "plain",
                "is_local": True,
                "cost_usd": 0.0,
            },
        ],
        "judge_config": {"provider": "openai", "model": "gpt-4.1-mini"},
    }


def _rag_answer_quality_live_run_config() -> dict[str, Any]:
    return {
        "evaluation_mode": "live_end_to_end",
        "supervision_mode": "mixed",
        "retrieval_baseline_ref": "baseline-run-42",
        "weights": {
            "grounding": 0.4,
            "answer_relevance": 0.3,
            "format_style": 0.2,
            "abstention_behavior": 0.1,
        },
        "grounding_threshold": 0.7,
        "candidates": [
            {
                "candidate_id": "openai:gpt-4.1-mini::default::citations",
                "provider": "openai",
                "model": "gpt-4.1-mini",
                "prompt_variant": "default",
                "formatting_citation_mode": "citations",
                "is_local": False,
                "cost_usd": 0.05,
            }
        ],
        "judge_config": {"provider": "openai", "model": "gpt-4.1-mini"},
    }


def _rag_answer_quality_dataset_without_contexts() -> list[dict[str, Any]]:
    return [
        {
            "sample_id": "aq-1",
            "query": "What is the capital of France?",
            "expected_behavior": "answer",
            "reference_answer": "Paris is the capital of France.",
        }
    ]


def _service(tmp_path) -> tuple[EvaluationsDatabase, RecipeRunsService, str]:
    db = EvaluationsDatabase(str(tmp_path / "evaluations.db"))
    user_id = get_single_user_instance().id_str
    return db, RecipeRunsService(db=db, user_id=user_id), user_id


def test_enqueue_recipe_run_uses_jobs_contract(tmp_path) -> None:
    _, service, user_id = _service(tmp_path)
    record = service.create_run(
        "summarization_quality",
        dataset=_inline_dataset(),
        run_config=_run_config(),
    )
    captured: dict[str, Any] = {}

    class _JobManager:
        def create_job(self, **kwargs):
            captured.update(kwargs)
            return {"id": 123}

    job_id = enqueue_recipe_run(record, owner_user_id=user_id, job_manager=_JobManager())

    assert job_id == "123"
    assert captured["domain"] == RECIPE_RUN_JOB_DOMAIN
    assert captured["queue"] == recipe_run_queue()
    assert captured["job_type"] == RECIPE_RUN_JOB_TYPE
    assert captured["owner_user_id"] == user_id
    assert captured["payload"] == build_recipe_run_job_payload(
        run_id=record.run_id,
        recipe_id=record.recipe_id,
        owner_user_id=user_id,
    )
    assert captured["idempotency_key"] == build_recipe_run_idempotency_key(run_id=record.run_id)


def test_handle_recipe_run_job_marks_run_completed_with_normalized_report(tmp_path) -> None:
    db, service, user_id = _service(tmp_path)
    record = service.create_run(
        "summarization_quality",
        dataset=_inline_dataset(),
        run_config=_run_config(),
    )
    import tldw_Server_API.app.core.Evaluations.recipe_runs_jobs_worker as worker

    def _fake_execute(*, record, db, user_id, service):
        del db, user_id, service
        return {
            "child_run_ids": [],
            "metadata": {
                "candidate_results": [
                    {
                        "candidate_id": "openai:gpt-4.1-mini",
                        "candidate_run_id": "openai:gpt-4.1-mini",
                        "provider": "openai",
                        "model": "gpt-4.1-mini",
                        "sample_results": [
                            {
                                "sample_id": "sample-0",
                                "metrics": {
                                    "consistency": 4.8,
                                    "relevance": 4.5,
                                    "coherence": 4.2,
                                    "fluency": 2.9,
                                },
                                "latency_ms": 100.0,
                            }
                        ],
                    }
                ],
                "recipe_report_inputs": {
                    "dataset_mode": record.metadata["dataset_mode"],
                    "review_sample": record.metadata["review_sample"],
                    "weights": {"grounding": 0.5, "coverage": 0.3, "usefulness": 0.2},
                    "candidate_results": [
                        {
                            "candidate_id": "openai:gpt-4.1-mini",
                            "candidate_run_id": "openai:gpt-4.1-mini",
                            "provider": "openai",
                            "model": "gpt-4.1-mini",
                            "sample_results": [
                                {
                                    "sample_id": "sample-0",
                                    "metrics": {
                                        "consistency": 4.8,
                                        "relevance": 4.5,
                                        "coherence": 4.2,
                                        "fluency": 2.9,
                                    },
                                    "latency_ms": 100.0,
                                }
                            ],
                        }
                    ],
                },
            },
        }

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(worker, "_execute_summarization_recipe_run", _fake_execute)

    try:
        result = handle_recipe_run_job(
            {
                "id": "job-42",
                "payload": build_recipe_run_job_payload(
                    run_id=record.run_id,
                    recipe_id=record.recipe_id,
                    owner_user_id=user_id,
                ),
            },
            db=db,
            user_id=user_id,
        )
    finally:
        monkeypatch.undo()

    refreshed = db.get_recipe_run(record.run_id)

    assert result["status"] == "completed"
    assert result["run_id"] == record.run_id
    assert result["job_id"] == "job-42"
    assert refreshed is not None
    assert refreshed.status is RunStatus.COMPLETED
    assert set(refreshed.recommendation_slots) == {
        "best_overall",
        "best_quality",
        "best_cheap",
        "best_local",
    }
    assert refreshed.metadata["jobs"]["job_id"] == "job-42"
    assert refreshed.metadata["jobs"]["worker_state"] == "completed"


def test_handle_recipe_run_job_executes_rag_retrieval_tuning_and_persists_report(tmp_path) -> None:
    db, service, user_id = _service(tmp_path)
    record = service.create_run(
        "rag_retrieval_tuning",
        dataset=_rag_dataset(),
        run_config=_rag_run_config(),
    )
    import tldw_Server_API.app.core.Evaluations.recipe_runs_jobs_worker as worker

    captured: dict[str, Any] = {}

    def _fake_execute(*, record, db, user_id, service):
        del db, service
        captured["run_id"] = record.run_id
        captured["user_id"] = user_id
        captured["inline_dataset"] = record.metadata.get("inline_dataset")
        return {
            "child_run_ids": ["candidate-rerank"],
            "metadata": {
                "candidate_results": [
                    {
                        "candidate_id": "candidate-rerank",
                        "candidate_run_id": "candidate-rerank",
                        "is_local": True,
                        "metrics": {
                            "pre_rerank_recall_at_k": 0.75,
                            "post_rerank_ndcg_at_k": 0.82,
                            "first_pass_recall_score": 0.75,
                            "post_rerank_quality_score": 0.82,
                        },
                    }
                ],
                "recipe_report_inputs": {
                    "dataset_mode": record.metadata["dataset_mode"],
                    "review_sample": record.metadata["review_sample"],
                    "corpus_scope": record.metadata["run_config"]["corpus_scope"],
                    "candidate_results": [
                        {
                            "candidate_id": "candidate-rerank",
                            "candidate_run_id": "candidate-rerank",
                            "is_local": True,
                            "metrics": {
                                "pre_rerank_recall_at_k": 0.75,
                                "post_rerank_ndcg_at_k": 0.82,
                                "first_pass_recall_score": 0.75,
                                "post_rerank_quality_score": 0.82,
                            },
                        }
                    ],
                },
            },
        }

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(worker, "_execute_rag_retrieval_tuning_recipe_run", _fake_execute)

    try:
        result = handle_recipe_run_job(
            {
                "id": "job-rag-1",
                "payload": build_recipe_run_job_payload(
                    run_id=record.run_id,
                    recipe_id=record.recipe_id,
                    owner_user_id=user_id,
                ),
            },
            db=db,
            user_id=user_id,
        )
    finally:
        monkeypatch.undo()

    refreshed = db.get_recipe_run(record.run_id)

    assert result["status"] == "completed"
    assert captured["run_id"] == record.run_id
    assert captured["inline_dataset"] == _rag_dataset()
    assert refreshed is not None
    assert refreshed.status is RunStatus.COMPLETED
    assert refreshed.child_run_ids == ["candidate-rerank"]
    assert refreshed.metadata["recipe_report_inputs"]["corpus_scope"]["sources"] == ["media_db", "notes"]
    assert refreshed.metadata["jobs"]["worker_state"] == "completed"
    assert refreshed.recommendation_slots["best_overall"].candidate_run_id == "candidate-rerank"


def test_handle_recipe_run_job_executes_rag_answer_quality_fixed_context_and_persists_report(
    tmp_path,
    monkeypatch,
) -> None:
    db, service, user_id = _service(tmp_path)
    dataset_id = db.create_dataset(
        name="rag-answer-quality",
        samples=_rag_answer_quality_dataset(),
        created_by=user_id,
    )
    record = service.create_run(
        "rag_answer_quality",
        dataset_id=dataset_id,
        run_config=_rag_answer_quality_run_config(),
    )

    import tldw_Server_API.app.core.Evaluations.recipe_runs_jobs_worker as worker
    import tldw_Server_API.app.core.Evaluations.recipes.rag_answer_quality_execution as execution

    def _fake_analyze(
        provider: str,
        prompt: str,
        user_prompt: str,
        *,
        api_key=None,
        system_message=None,
        temp=None,
        model_override=None,
    ) -> str:
        del prompt, user_prompt, api_key, system_message, temp
        if provider == "openai" and model_override == "gpt-4.1-mini":
            return "Paris is the capital of France. [1]"
        return "I cannot answer from the provided context."

    def _fake_run_geval(**kwargs):
        summary = kwargs.get("summary") or ""
        if "[1]" in summary:
            return {
                "metrics": {
                    "consistency": 4.9,
                    "relevance": 4.8,
                    "coherence": 4.7,
                    "fluency": 4.6,
                }
            }
        return {
            "metrics": {
                "consistency": 2.4,
                "relevance": 2.3,
                "coherence": 2.2,
                "fluency": 2.1,
            }
        }

    monkeypatch.setattr(execution, "analyze", _fake_analyze)
    monkeypatch.setattr(execution, "run_geval", _fake_run_geval)

    result = worker.handle_recipe_run_job(
        {
            "id": "job-rag-answer-quality",
            "payload": build_recipe_run_job_payload(
                run_id=record.run_id,
                recipe_id=record.recipe_id,
                owner_user_id=user_id,
            ),
        },
        db=db,
        user_id=user_id,
    )

    refreshed = db.get_recipe_run(record.run_id)

    assert result["status"] == "completed"
    assert refreshed is not None
    assert refreshed.status is RunStatus.COMPLETED
    assert len(refreshed.metadata["candidate_results"]) == 2
    assert refreshed.metadata["candidate_results"][0]["sample_results"][0]["answer"]
    assert refreshed.metadata["candidate_results"][1]["sample_results"][0]["failure_labels"] == [
        "missed_answer",
        "bad_abstention",
    ]
    assert refreshed.metadata["recipe_report_inputs"]["context_snapshot_ref"] == "context-snapshot-1"
    assert refreshed.metadata["recipe_report"]["best_overall"]["candidate_id"] == "openai:gpt-4.1-mini::default::citations"
    assert refreshed.metadata["recipe_report"]["failure_examples"][0]["candidate_id"] == "ollama:llama3.1:8b::direct::plain"
    assert refreshed.metadata["recipe_report"]["failure_examples"][0]["failure_labels"] == [
        "missed_answer",
        "bad_abstention",
    ]
    assert refreshed.recommendation_slots["best_overall"].candidate_run_id == "openai:gpt-4.1-mini::default::citations"
    assert refreshed.metadata["jobs"]["worker_state"] == "completed"


def test_handle_recipe_run_job_executes_rag_answer_quality_live_mode_with_frozen_retrieval(
    tmp_path,
    monkeypatch,
) -> None:
    db, service, user_id = _service(tmp_path)
    retrieval_record = service.create_run(
        "rag_retrieval_tuning",
        dataset=_rag_dataset(),
        run_config=_rag_run_config(),
    )
    db.update_recipe_run(
        retrieval_record.run_id,
        status=RunStatus.COMPLETED,
        recommendation_slots={
            "best_overall": {
                "candidate_run_id": "baseline",
                "reason_code": "highest_retrieval_quality",
                "metadata": {},
            }
        },
        metadata={
            **retrieval_record.metadata,
            "recipe_report": {
                "best_overall": {"candidate_id": "baseline"},
            },
        },
    )
    dataset_id = db.create_dataset(
        name="rag-answer-quality-live",
        samples=_rag_answer_quality_dataset(),
        created_by=user_id,
    )
    record = service.create_run(
        "rag_answer_quality",
        dataset_id=dataset_id,
        run_config={
            **_rag_answer_quality_live_run_config(),
            "retrieval_baseline_ref": retrieval_record.run_id,
        },
    )

    import tldw_Server_API.app.core.Evaluations.recipe_runs_jobs_worker as worker
    import tldw_Server_API.app.core.Evaluations.recipes.rag_answer_quality_execution as execution

    captured_request: dict[str, Any] = {}

    async def _fake_unified_rag_pipeline(**kwargs):
        captured_request.update(kwargs)
        return {
            "documents": [
                {
                    "source": "knowledge_base",
                    "text": "Paris is the capital and most populous city of France.",
                }
            ],
            "metadata": {
                "pre_rerank_documents": [
                    {
                        "source": "knowledge_base",
                        "text": "Paris is the capital and most populous city of France.",
                    }
                ],
                "reranked_documents": [
                    {
                        "source": "knowledge_base",
                        "text": "Paris is the capital and most populous city of France.",
                    }
                ],
            },
        }

    def _fake_analyze(
        provider: str,
        prompt: str,
        user_prompt: str,
        *,
        api_key=None,
        system_message=None,
        temp=None,
        model_override=None,
    ) -> str:
        del prompt, user_prompt, api_key, system_message, temp
        if provider == "openai":
            assert model_override == "gpt-4.1-mini"
            return "Paris is the capital of France. [1]"
        assert provider == "ollama"
        assert model_override == "llama3.1:8b"
        return "It appears to be Paris, based on the context."

    def _fake_run_geval(**kwargs):
        del kwargs
        return {
            "metrics": {
                "consistency": 4.9,
                "relevance": 4.8,
                "coherence": 4.7,
                "fluency": 4.6,
            }
        }

    monkeypatch.setattr(execution, "unified_rag_pipeline", _fake_unified_rag_pipeline)
    monkeypatch.setattr(execution, "analyze", _fake_analyze)
    monkeypatch.setattr(execution, "run_geval", _fake_run_geval)

    result = worker.handle_recipe_run_job(
        {
            "id": "job-rag-answer-quality-live",
            "payload": build_recipe_run_job_payload(
                run_id=record.run_id,
                recipe_id=record.recipe_id,
                owner_user_id=user_id,
            ),
        },
        db=db,
        user_id=user_id,
    )

    refreshed = db.get_recipe_run(record.run_id)

    assert result["status"] == "completed"
    assert refreshed is not None
    assert refreshed.status is RunStatus.COMPLETED
    assert captured_request["search_mode"] == "hybrid"
    assert captured_request["top_k"] == 5
    assert captured_request["enable_reranking"] is False
    assert captured_request["expand_query"] is False
    assert captured_request["enable_intent_routing"] is False
    assert captured_request["enable_prf"] is False
    assert captured_request["enable_hyde"] is False
    assert captured_request["enable_post_verification"] is False
    assert captured_request["adaptive_rerun_on_low_confidence"] is False
    assert captured_request["adaptive_hybrid_weights"] is False
    assert refreshed.metadata["recipe_report_inputs"]["retrieval_baseline_ref"] == retrieval_record.run_id
    assert refreshed.metadata["recipe_report_inputs"]["retrieval_preset_hash"]
    assert refreshed.metadata["candidate_results"][0]["sample_results"][0]["retrieved_contexts"] == [
        {
            "source": "knowledge_base",
            "text": "Paris is the capital and most populous city of France.",
        }
    ]
    sample_hash = refreshed.metadata["candidate_results"][0]["sample_results"][0]["retrieval_preset_hash"]
    assert refreshed.metadata["candidate_results"][0]["retrieval_preset_hash"] == sample_hash
    assert refreshed.metadata["recipe_report_inputs"]["retrieval_preset_hash"] == sample_hash
    assert refreshed.metadata["recipe_report"]["retrieval_baseline_ref"] == retrieval_record.run_id
    assert refreshed.metadata["recipe_report"]["retrieval_preset_hash"] == sample_hash
    assert refreshed.metadata["jobs"]["worker_state"] == "completed"


def test_handle_recipe_run_job_fails_rag_answer_quality_live_mode_when_baseline_cannot_be_resolved(
    tmp_path,
    monkeypatch,
) -> None:
    db, service, user_id = _service(tmp_path)
    dataset_id = db.create_dataset(
        name="rag-answer-quality-live-missing-baseline",
        samples=_rag_answer_quality_dataset(),
        created_by=user_id,
    )
    record = service.create_run(
        "rag_answer_quality",
        dataset_id=dataset_id,
        run_config=_rag_answer_quality_live_run_config(),
    )

    import tldw_Server_API.app.core.Evaluations.recipe_runs_jobs_worker as worker
    import tldw_Server_API.app.core.Evaluations.recipes.rag_answer_quality_execution as execution

    async def _unexpected_unified_rag_pipeline(**kwargs):
        raise AssertionError(f"live retrieval should not execute when baseline resolution fails: {kwargs}")

    monkeypatch.setattr(execution, "unified_rag_pipeline", _unexpected_unified_rag_pipeline)

    with pytest.raises(RecipeRunJobError, match="retrieval_baseline_ref") as exc_info:
        worker.handle_recipe_run_job(
            {
                "id": "job-rag-answer-quality-live-missing-baseline",
                "payload": build_recipe_run_job_payload(
                    run_id=record.run_id,
                    recipe_id=record.recipe_id,
                    owner_user_id=user_id,
                ),
            },
            db=db,
            user_id=user_id,
        )
    assert exc_info.value.retryable is False

    refreshed = db.get_recipe_run(record.run_id)

    assert refreshed is not None
    assert refreshed.status is RunStatus.FAILED
    assert refreshed.metadata["jobs"]["error"] == "recipe_run_failed"
    assert refreshed.metadata["jobs"]["error_type"] == "ValueError"


def test_handle_recipe_run_job_executes_rag_answer_quality_fixed_mode_with_run_level_contexts(
    tmp_path,
    monkeypatch,
) -> None:
    db, service, user_id = _service(tmp_path)
    record = service.create_run(
        "rag_answer_quality",
        dataset=_rag_answer_quality_dataset_without_contexts(),
        run_config={
            **_rag_answer_quality_run_config(),
            "inline_contexts": [
                {
                    "source": "knowledge_base",
                    "text": "Paris is the capital and most populous city of France.",
                }
            ],
        },
    )

    import tldw_Server_API.app.core.Evaluations.recipe_runs_jobs_worker as worker
    import tldw_Server_API.app.core.Evaluations.recipes.rag_answer_quality_execution as execution

    def _fake_analyze(
        provider: str,
        prompt: str,
        user_prompt: str,
        *,
        api_key=None,
        system_message=None,
        temp=None,
        model_override=None,
    ) -> str:
        del prompt, user_prompt, api_key, system_message, temp
        if provider == "openai":
            assert model_override == "gpt-4.1-mini"
            return "Paris is the capital of France. [1]"
        assert provider == "ollama"
        assert model_override == "llama3.1:8b"
        return "It appears to be Paris, based on the context."

    def _fake_run_geval(**kwargs):
        del kwargs
        return {
            "metrics": {
                "consistency": 4.9,
                "relevance": 4.8,
                "coherence": 4.7,
                "fluency": 4.6,
            }
        }

    monkeypatch.setattr(execution, "analyze", _fake_analyze)
    monkeypatch.setattr(execution, "run_geval", _fake_run_geval)

    result = worker.handle_recipe_run_job(
        {
            "id": "job-rag-answer-quality-fixed-run-level-contexts",
            "payload": build_recipe_run_job_payload(
                run_id=record.run_id,
                recipe_id=record.recipe_id,
                owner_user_id=user_id,
            ),
        },
        db=db,
        user_id=user_id,
    )

    refreshed = db.get_recipe_run(record.run_id)

    assert result["status"] == "completed"
    assert refreshed is not None
    assert refreshed.status is RunStatus.COMPLETED
    assert refreshed.metadata["candidate_results"][0]["sample_results"][0]["retrieved_contexts"] == [
        {
            "source": "knowledge_base",
            "text": "Paris is the capital and most populous city of France.",
        }
    ]


def test_normalize_rag_targets_preserves_chunk_labels_for_scoring() -> None:
    import tldw_Server_API.app.core.Evaluations.recipe_runs_jobs_worker as worker

    normalized = worker._normalize_rag_targets(
        {
            "targets": {
                "relevant_chunk_ids": [{"id": "chunk-1", "grade": 2}],
            }
        }
    )

    assert normalized["chunks"]["chunk-1"] == 2
    assert worker._grade_rag_document(
        {
            "id": "chunk-1",
            "metadata": {"chunk_id": "chunk-1"},
        },
        normalized,
    ) == 2


def test_extract_ranked_documents_prefers_pipeline_snapshots_when_present() -> None:
    import tldw_Server_API.app.core.Evaluations.recipe_runs_jobs_worker as worker

    first_pass_docs, reranked_docs = worker._extract_ranked_documents(
        {
            "documents": [{"id": "fallback-doc"}],
            "metadata": {
                "pre_rerank_documents": [{"id": "pre-doc"}],
                "reranked_documents": [{"id": "post-doc"}],
            },
        }
    )

    assert [doc["id"] for doc in first_pass_docs] == ["pre-doc"]
    assert [doc["id"] for doc in reranked_docs] == ["post-doc"]


def test_handle_recipe_run_job_marks_run_failed_when_report_building_errors(tmp_path) -> None:
    db, service, user_id = _service(tmp_path)
    record = service.create_run(
        "summarization_quality",
        dataset=_inline_dataset(),
        run_config=_run_config(),
    )

    class _BrokenService:
        def get_run(self, run_id: str):
            assert run_id == record.run_id
            return service.get_run(run_id)

        def get_report(self, run_id: str):
            assert run_id == record.run_id
            raise RuntimeError("boom")

    import tldw_Server_API.app.core.Evaluations.recipe_runs_jobs_worker as worker

    def _fake_execute(*, record, db, user_id, service):
        del db, user_id, service
        return {
            "child_run_ids": [],
            "metadata": {
                "candidate_results": [
                    {
                        "candidate_id": "openai:gpt-4.1-mini",
                        "candidate_run_id": "openai:gpt-4.1-mini",
                        "provider": "openai",
                        "model": "gpt-4.1-mini",
                        "sample_results": [
                            {
                                "sample_id": "sample-0",
                                "metrics": {
                                    "consistency": 4.8,
                                    "relevance": 4.5,
                                    "coherence": 4.2,
                                    "fluency": 2.9,
                                },
                                "latency_ms": 100.0,
                            }
                        ],
                    }
                ],
                "recipe_report_inputs": {
                    "dataset_mode": record.metadata["dataset_mode"],
                    "review_sample": record.metadata["review_sample"],
                    "weights": {"grounding": 0.5, "coverage": 0.3, "usefulness": 0.2},
                    "candidate_results": [],
                },
            },
        }

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(worker, "_execute_summarization_recipe_run", _fake_execute)

    try:
        with pytest.raises(RecipeRunJobError, match="boom") as exc_info:
            handle_recipe_run_job(
                {
                    "id": "job-99",
                    "payload": build_recipe_run_job_payload(
                        run_id=record.run_id,
                        recipe_id=record.recipe_id,
                        owner_user_id=user_id,
                    ),
                },
                db=db,
                user_id=user_id,
                service=_BrokenService(),
            )
        assert exc_info.value.retryable is False
    finally:
        monkeypatch.undo()

    refreshed = db.get_recipe_run(record.run_id)
    assert refreshed is not None
    assert refreshed.status is RunStatus.FAILED
    assert refreshed.metadata["jobs"]["worker_state"] == "failed"
    assert refreshed.metadata["jobs"]["error"] == "recipe_run_failed"
    assert refreshed.metadata["jobs"]["error_type"] == "RuntimeError"


def test_handle_recipe_run_job_marks_retryable_failures_pending(tmp_path) -> None:
    db, service, user_id = _service(tmp_path)
    record = service.create_run(
        "summarization_quality",
        dataset=_inline_dataset(),
        run_config=_run_config(),
    )

    import tldw_Server_API.app.core.Evaluations.recipe_runs_jobs_worker as worker

    def _fake_execute(*, record, db, user_id, service):
        del record, db, user_id, service
        raise TimeoutError("temporary upstream timeout")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(worker, "_execute_summarization_recipe_run", _fake_execute)

    try:
        with pytest.raises(RecipeRunJobError, match="temporary upstream timeout") as exc_info:
            handle_recipe_run_job(
                {
                    "id": "job-100",
                    "payload": build_recipe_run_job_payload(
                        run_id=record.run_id,
                        recipe_id=record.recipe_id,
                        owner_user_id=user_id,
                    ),
                },
                db=db,
                user_id=user_id,
                service=service,
            )
        assert exc_info.value.retryable is True
    finally:
        monkeypatch.undo()

    refreshed = db.get_recipe_run(record.run_id)
    assert refreshed is not None
    assert refreshed.status is RunStatus.PENDING
    assert refreshed.metadata["jobs"]["worker_state"] == "retrying"
    assert refreshed.metadata["jobs"]["error"] == "recipe_run_retrying"
    assert refreshed.metadata["jobs"]["retryable"] is True


def test_handle_recipe_run_job_executes_summarization_recipe_and_persists_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    db, service, user_id = _service(tmp_path)
    dataset = [
        {
            "input": "OpenAI released a smaller model for fast summarization workloads.",
            "expected": "OpenAI released a small model for summarization.",
            "metadata": {"sample_id": "news-1"},
        },
        {
            "input": "Local models remain attractive for private deployments with lower direct cost.",
            "expected": "Local models can be cheaper and more private.",
            "metadata": {"sample_id": "news-2"},
        },
    ]
    record = service.create_run(
        "summarization_quality",
        dataset=dataset,
        run_config={
            **_run_config(),
            "candidate_model_ids": [
                "openai:gpt-4.1-mini",
                "ollama:llama3.1:8b",
            ],
            "weights": {"grounding": 0.5, "coverage": 0.3, "usefulness": 0.2},
        },
    )

    import tldw_Server_API.app.core.Evaluations.recipe_runs_jobs_worker as worker

    def _fake_generate_summary(*, provider: str, model: str, source_text: str, run_config: dict[str, Any]) -> str:
        del run_config
        return f"{provider}:{model} summary for {source_text[:18]}"

    def _fake_score_summary(
        *,
        source_text: str,
        summary: str,
        run_config: dict[str, Any],
        reference_summary: str | None,
    ) -> dict[str, Any]:
        del source_text, run_config, reference_summary
        if summary.startswith("openai:"):
            return {
                "metrics": {
                    "consistency": 4.8,
                    "relevance": 4.5,
                    "coherence": 4.3,
                    "fluency": 2.9,
                },
                "average_score": 0.89,
                "assessment": "High quality",
            }
        return {
            "metrics": {
                "consistency": 4.1,
                "relevance": 4.0,
                "coherence": 4.0,
                "fluency": 2.8,
            },
            "average_score": 0.80,
            "assessment": "Good quality",
        }

    monkeypatch.setattr(worker, "_generate_summary_for_candidate", _fake_generate_summary)
    monkeypatch.setattr(worker, "_score_summary_with_geval", _fake_score_summary)

    result = worker.handle_recipe_run_job(
        {
            "id": "job-summary",
            "payload": build_recipe_run_job_payload(
                run_id=record.run_id,
                recipe_id=record.recipe_id,
                owner_user_id=user_id,
            ),
        },
        db=db,
        user_id=user_id,
    )

    refreshed = db.get_recipe_run(record.run_id)

    assert result["status"] == "completed"
    assert refreshed is not None
    assert refreshed.status is RunStatus.COMPLETED
    assert refreshed.child_run_ids == []
    assert len(refreshed.metadata["candidate_results"]) == 2
    assert refreshed.metadata["candidate_results"][0]["sample_results"][0]["summary"].startswith("openai:")
    assert refreshed.metadata["recipe_report_inputs"]["weights"] == {
        "grounding": 0.5,
        "coverage": 0.3,
        "usefulness": 0.2,
    }
    assert refreshed.metadata["recipe_report"]["best_overall"]["candidate_id"] == "openai:gpt-4.1-mini"
    assert refreshed.recommendation_slots["best_overall"].candidate_run_id == "openai:gpt-4.1-mini"
    assert refreshed.metadata["jobs"]["worker_state"] == "completed"


def test_handle_recipe_run_job_executes_embeddings_recipe_and_persists_child_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    db, service, user_id = _service(tmp_path)
    dataset = _embeddings_dataset()
    record = service.create_run(
        "embeddings_model_selection",
        dataset=dataset,
        run_config=_embeddings_run_config(),
    )

    import tldw_Server_API.app.core.Evaluations.recipe_runs_jobs_worker as worker

    captured: dict[str, Any] = {}

    def _fake_execute(*, record, db, user_id, service):
        del db, service
        captured["run_id"] = record.run_id
        captured["inline_dataset"] = record.metadata.get("inline_dataset")
        captured["user_id"] = user_id
        return {
            "child_run_ids": ["arm-1"],
            "metadata": {
                "abtest": {"test_id": "abtest-child-1"},
                "candidate_results": [
                    {
                        "candidate_id": "arm-1",
                        "candidate_run_id": "arm-1",
                        "model": "text-embedding-3-small",
                        "provider": "openai",
                        "query_results": [
                            {
                                "ranked_ids": ["1"],
                                "expected_ids": ["1"],
                                "latency_ms": 80.0,
                            },
                            {
                                "ranked_ids": ["2"],
                                "expected_ids": ["2"],
                                "latency_ms": 90.0,
                            },
                        ],
                    }
                ],
                "recipe_report_inputs": {
                    "dataset_mode": "labeled",
                    "review_sample": {
                        "required": False,
                        "sample_size": 0,
                        "sample_query_ids": [],
                    },
                    "candidate_results": [
                        {
                            "candidate_id": "arm-1",
                            "candidate_run_id": "arm-1",
                            "model": "text-embedding-3-small",
                            "provider": "openai",
                            "query_results": [
                                {
                                    "ranked_ids": ["1"],
                                    "expected_ids": ["1"],
                                    "latency_ms": 80.0,
                                },
                                {
                                    "ranked_ids": ["2"],
                                    "expected_ids": ["2"],
                                    "latency_ms": 90.0,
                                },
                            ],
                        }
                    ],
                },
            },
        }

    monkeypatch.setattr(
        worker,
        "_execute_embeddings_recipe_run",
        _fake_execute,
        raising=False,
    )

    result = worker.handle_recipe_run_job(
        {
            "id": "job-embeddings",
            "payload": build_recipe_run_job_payload(
                run_id=record.run_id,
                recipe_id=record.recipe_id,
                owner_user_id=user_id,
            ),
        },
        db=db,
        user_id=user_id,
    )

    refreshed = db.get_recipe_run(record.run_id)

    assert result["status"] == "completed"
    assert captured["run_id"] == record.run_id
    assert captured["inline_dataset"] == dataset
    assert captured["user_id"] == user_id
    assert refreshed is not None
    assert refreshed.status is RunStatus.COMPLETED
    assert refreshed.child_run_ids == ["arm-1"]
    assert refreshed.metadata["abtest"]["test_id"] == "abtest-child-1"
    assert refreshed.metadata["candidate_results"]
    assert refreshed.metadata["recipe_report_inputs"]["candidate_results"]
    assert refreshed.metadata["recipe_report"]["best_overall"]["candidate_id"] == "arm-1"
    assert refreshed.metadata["jobs"]["worker_state"] == "completed"
    assert refreshed.recommendation_slots["best_overall"].candidate_run_id == "arm-1"


@pytest.mark.asyncio
async def test_start_recipe_run_jobs_worker_respects_enable_flag(monkeypatch) -> None:
    monkeypatch.delenv("EVALUATIONS_RECIPE_RUN_JOBS_WORKER_ENABLED", raising=False)
    monkeypatch.delenv("EVALS_RECIPE_RUN_JOBS_WORKER_ENABLED", raising=False)

    task = await start_recipe_run_jobs_worker()

    assert task is None


@pytest.mark.asyncio
async def test_start_recipe_run_jobs_worker_returns_task_when_enabled(monkeypatch) -> None:
    started = asyncio.Event()
    finished = asyncio.Event()

    async def _fake_run(_stop_event: asyncio.Event | None = None):
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            finished.set()
            raise

    monkeypatch.setenv("EVALUATIONS_RECIPE_RUN_JOBS_WORKER_ENABLED", "true")
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipe_runs_jobs_worker.run_recipe_run_jobs_worker",
        _fake_run,
    )

    task = await start_recipe_run_jobs_worker()
    assert task is not None
    await asyncio.wait_for(started.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


@pytest.mark.asyncio
async def test_run_recipe_run_jobs_worker_stops_sdk_when_stop_event_is_set(monkeypatch) -> None:
    import tldw_Server_API.app.core.Evaluations.recipe_runs_jobs_worker as worker

    stop_event = asyncio.Event()
    observed = {"stopped": False}

    class _FakeWorkerSDK:
        def __init__(self, _jm, _cfg) -> None:
            pass

        def stop(self) -> None:
            observed["stopped"] = True

        async def run(self, *, handler) -> None:
            while not observed["stopped"]:
                await asyncio.sleep(0)

    monkeypatch.setattr(worker, "WorkerSDK", _FakeWorkerSDK)

    task = asyncio.create_task(worker.run_recipe_run_jobs_worker(stop_event))
    await asyncio.sleep(0)
    stop_event.set()
    await asyncio.wait_for(task, timeout=1)

    assert observed["stopped"] is True


@pytest.mark.asyncio
async def test_start_recipe_run_jobs_worker_forwards_caller_owned_stop_event(monkeypatch) -> None:
    import tldw_Server_API.app.core.Evaluations.recipe_runs_jobs_worker as worker

    observed: dict[str, object] = {}
    stop_event = asyncio.Event()

    async def _fake_run(stop_event_arg: asyncio.Event | None = None) -> None:
        observed["stop_event"] = stop_event_arg
        await stop_event_arg.wait()

    monkeypatch.setenv("EVALUATIONS_RECIPE_RUN_JOBS_WORKER_ENABLED", "true")
    monkeypatch.setattr(worker, "run_recipe_run_jobs_worker", _fake_run)

    task = await worker.start_recipe_run_jobs_worker(stop_event=stop_event)
    assert task is not None
    await asyncio.sleep(0)

    assert observed["stop_event"] is stop_event

    stop_event.set()
    await asyncio.wait_for(task, timeout=1)
