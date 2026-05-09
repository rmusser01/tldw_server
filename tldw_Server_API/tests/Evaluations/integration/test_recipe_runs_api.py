from __future__ import annotations

import os
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.schemas.evaluation_recipe_schemas import RecommendationSlot
from tldw_Server_API.app.api.v1.schemas.evaluation_schemas_unified import RunStatus
from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_recipes import (
    get_recipe_run_job_enqueuer,
)
from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_unified import (
    router as evaluations_unified_router,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_single_user_instance
from tldw_Server_API.app.core.DB_Management.Evaluations_DB import EvaluationsDatabase
from tldw_Server_API.app.core.Evaluations.recipe_runs_service import (
    RECIPE_RUN_REUSE_ENTITY_TYPE,
    RecipeRunsService,
)
from tldw_Server_API.app.core.Evaluations.recipes.dataset_snapshot import build_dataset_content_hash
from tldw_Server_API.app.core.exceptions import RecipeEnqueueError

pytestmark = [pytest.mark.integration]


@pytest.fixture(autouse=True)
def _override_recipe_run_enqueue_dependency():
    from tldw_Server_API.app.main import app

    def _noop_enqueue(record, *, owner_user_id=None, job_manager=None):
        del record, owner_user_id, job_manager
        return "job-noop"

    app.dependency_overrides[get_recipe_run_job_enqueuer] = lambda: _noop_enqueue
    yield
    app.dependency_overrides.pop(get_recipe_run_job_enqueuer, None)


@pytest.fixture(autouse=True)
def _enable_recipe_worker_by_default(monkeypatch) -> None:
    monkeypatch.setenv("EVALUATIONS_RECIPE_RUN_JOBS_WORKER_ENABLED", "true")


def _inline_dataset() -> list[dict[str, Any]]:
    return [
        {
            "input": "Summarize the meeting notes.",
            "expected": "A concise summary of the meeting notes.",
        }
    ]


def _run_config() -> dict[str, Any]:
    return {
        "candidate_model_ids": [
            "openai:gpt-4.1-mini",
            "local:mistral-small",
        ],
        "judge_config": {
            "provider": "openai",
            "model": "gpt-4.1-mini",
        },
        "prompts": {
            "system": "Compare model outputs.",
        },
        "weights": {
            "quality": 0.8,
            "cost": 0.2,
        },
        "comparison_mode": "leaderboard",
        "source_normalization": {
            "strip_citations": True,
        },
        "context_policy": {
            "mode": "strict",
        },
        "execution_policy": {
            "max_parallel_candidates": 2,
        },
    }


def _embeddings_run_config() -> dict[str, Any]:
    return {
        "comparison_mode": "embedding_only",
        "candidates": [
            {"provider": "openai", "model": "text-embedding-3-small"},
            {"provider": "local", "model": "bge-small", "is_local": True},
        ],
        "media_ids": [1, 2],
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


def test_evaluations_unified_router_registers_recipe_routes_before_eval_id_routes() -> None:
    paths = [route.path for route in evaluations_unified_router.routes if hasattr(route, "path")]

    assert "/evaluations/recipes" in paths
    assert "/evaluations/recipe-runs/{run_id}" in paths
    assert paths.index("/evaluations/recipes") < paths.index("/evaluations/{eval_id}")


@pytest.mark.asyncio
async def test_recipe_manifest_endpoints(async_api_client, auth_headers) -> None:
    list_response = await async_api_client.get(
        "/api/v1/evaluations/recipes",
        headers=auth_headers,
    )

    assert list_response.status_code == 200
    manifests = list_response.json()
    assert any(item["recipe_id"] == "summarization_quality" for item in manifests)
    rag_manifest = next(
        item for item in manifests if item["recipe_id"] == "rag_retrieval_tuning"
    )
    assert rag_manifest["launchable"] is True
    assert rag_manifest["capabilities"]["corpus_sources"] == ["media_db", "notes"]
    assert rag_manifest["default_run_config"]["candidate_creation_mode"] == "auto_sweep"

    detail_response = await async_api_client.get(
        "/api/v1/evaluations/recipes/rag_retrieval_tuning",
        headers=auth_headers,
    )

    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["recipe_id"] == "rag_retrieval_tuning"
    assert detail["launchable"] is True
    assert detail["capabilities"]["candidate_creation_modes"] == ["auto_sweep", "manual"]
    assert detail["default_run_config"]["corpus_scope"]["sources"] == ["media_db", "notes"]


@pytest.mark.asyncio
async def test_recipe_launch_readiness_endpoint_reports_worker_disabled_by_default(
    async_api_client,
    auth_headers,
    monkeypatch,
) -> None:
    monkeypatch.delenv("EVALUATIONS_RECIPE_RUN_JOBS_WORKER_ENABLED", raising=False)
    monkeypatch.delenv("EVALS_RECIPE_RUN_JOBS_WORKER_ENABLED", raising=False)

    response = await async_api_client.get(
        "/api/v1/evaluations/recipes/summarization_quality/launch-readiness",
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["recipe_id"] == "summarization_quality"
    assert body["ready"] is False
    assert body["can_enqueue_runs"] is False
    assert body["can_reuse_completed_runs"] is True
    assert body["runtime_checks"]["recipe_run_worker_enabled"] is False
    assert "recipe worker is not running" in body["message"]


@pytest.mark.asyncio
async def test_recipe_launch_readiness_endpoint_reports_worker_enabled(
    async_api_client,
    auth_headers,
    monkeypatch,
) -> None:
    monkeypatch.setenv("EVALUATIONS_RECIPE_RUN_JOBS_WORKER_ENABLED", "true")

    response = await async_api_client.get(
        "/api/v1/evaluations/recipes/summarization_quality/launch-readiness",
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["can_enqueue_runs"] is True
    assert body["runtime_checks"]["recipe_run_worker_enabled"] is True


@pytest.mark.asyncio
async def test_recipe_launch_readiness_endpoint_reports_rag_worker_disabled_state(
    async_api_client,
    auth_headers,
    monkeypatch,
) -> None:
    monkeypatch.delenv("EVALUATIONS_RECIPE_RUN_JOBS_WORKER_ENABLED", raising=False)
    monkeypatch.delenv("EVALS_RECIPE_RUN_JOBS_WORKER_ENABLED", raising=False)

    response = await async_api_client.get(
        "/api/v1/evaluations/recipes/rag_retrieval_tuning/launch-readiness",
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["recipe_id"] == "rag_retrieval_tuning"
    assert body["ready"] is False
    assert body["can_enqueue_runs"] is False
    assert body["can_reuse_completed_runs"] is True
    assert "recipe worker is not running" in body["message"]


@pytest.mark.asyncio
async def test_recipe_validate_dataset_endpoint_rejects_invalid_payload_shape(
    async_api_client,
    auth_headers,
) -> None:
    response = await async_api_client.post(
        "/api/v1/evaluations/recipes/summarization_quality/validate-dataset",
        json={"dataset": {"input": "not-a-list"}},
        headers=auth_headers,
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_recipe_validate_dataset_endpoint_returns_errors(async_api_client, auth_headers) -> None:
    response = await async_api_client.post(
        "/api/v1/evaluations/recipes/summarization_quality/validate-dataset",
        json={
            "dataset": [
                {"input": "valid prompt", "expected": "summary"},
                {"input": "missing label partner"},
            ]
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert body["errors"]


@pytest.mark.asyncio
async def test_recipe_validate_dataset_endpoint_returns_404_for_unknown_recipe(
    async_api_client,
    auth_headers,
) -> None:
    response = await async_api_client.post(
        "/api/v1/evaluations/recipes/not_a_real_recipe/validate-dataset",
        json={"dataset": _inline_dataset()},
        headers=auth_headers,
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Recipe not found"


@pytest.mark.asyncio
async def test_recipe_validate_dataset_endpoint_accepts_rag_retrieval_tuning(
    async_api_client,
    auth_headers,
) -> None:
    response = await async_api_client.post(
        "/api/v1/evaluations/recipes/rag_retrieval_tuning/validate-dataset",
        json={"dataset": _rag_dataset(), "run_config": _rag_run_config()},
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["dataset_mode"] == "labeled"
    assert body["corpus_scope"]["sources"] == ["media_db", "notes"]


@pytest.mark.asyncio
async def test_recipe_run_create_metadata_and_report_endpoints(async_api_client, auth_headers) -> None:
    payload = {
        "dataset": _inline_dataset(),
        "run_config": _run_config(),
    }

    create_response = await async_api_client.post(
        "/api/v1/evaluations/recipes/summarization_quality/runs",
        json=payload,
        headers=auth_headers,
    )

    assert create_response.status_code == 202
    created = create_response.json()
    assert created["status"] == "pending"
    assert created["metadata"]["run_config"] == payload["run_config"]
    assert created["metadata"]["owner_user_id"]

    run_id = created["run_id"]

    metadata_response = await async_api_client.get(
        f"/api/v1/evaluations/recipe-runs/{run_id}",
        headers=auth_headers,
    )

    assert metadata_response.status_code == 200
    assert metadata_response.json()["run_id"] == run_id

    report_response = await async_api_client.get(
        f"/api/v1/evaluations/recipe-runs/{run_id}/report",
        headers=auth_headers,
    )

    assert report_response.status_code == 200
    report = report_response.json()
    assert report["run"]["run_id"] == run_id
    assert set(report["recommendation_slots"]) == {
        "best_overall",
        "best_quality",
        "best_cheap",
        "best_local",
    }
    for slot in report["recommendation_slots"].values():
        assert slot["candidate_run_id"] is None
        assert slot["reason_code"] is not None


@pytest.mark.asyncio
async def test_recipe_run_create_endpoint_returns_404_for_unknown_recipe(
    async_api_client,
    auth_headers,
) -> None:
    response = await async_api_client.post(
        "/api/v1/evaluations/recipes/not_a_real_recipe/runs",
        json={
            "dataset": _inline_dataset(),
            "run_config": _run_config(),
        },
        headers=auth_headers,
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Recipe not found"


@pytest.mark.asyncio
async def test_recipe_run_create_endpoint_accepts_rag_retrieval_tuning(
    async_api_client,
    auth_headers,
) -> None:
    response = await async_api_client.post(
        "/api/v1/evaluations/recipes/rag_retrieval_tuning/runs",
        json={
            "dataset": _rag_dataset(),
            "run_config": _rag_run_config(),
        },
        headers=auth_headers,
    )

    assert response.status_code == 202
    body = response.json()
    assert body["recipe_id"] == "rag_retrieval_tuning"
    assert body["metadata"]["run_config"]["corpus_scope"]["sources"] == ["media_db", "notes"]
    assert body["metadata"]["inline_dataset"] == _rag_dataset()


@pytest.mark.asyncio
async def test_embeddings_recipe_run_create_persists_inline_dataset(async_api_client, auth_headers) -> None:
    payload = {
        "dataset": _embeddings_dataset(),
        "run_config": _embeddings_run_config(),
    }

    response = await async_api_client.post(
        "/api/v1/evaluations/recipes/embeddings_model_selection/runs",
        json=payload,
        headers=auth_headers,
    )

    assert response.status_code == 202
    body = response.json()
    assert body["metadata"]["inline_dataset"] == payload["dataset"]
    assert body["metadata"]["run_config"]["media_ids"] == [1, 2]


@pytest.mark.asyncio
async def test_embeddings_recipe_candidates_endpoint_returns_current_model(
    async_api_client,
    auth_headers,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_recipes.build_embedding_recipe_candidate_hints",
        lambda *, user: {
            "recipe_id": "embeddings_model_selection",
            "current": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "is_local": False,
                "default": True,
                "status": "ready",
            },
            "candidates": [
                {
                    "provider": "openai",
                    "model": "text-embedding-3-small",
                    "is_local": False,
                    "default": True,
                    "status": "ready",
                }
            ],
            "warnings": [],
        },
    )

    response = await async_api_client.get(
        "/api/v1/evaluations/recipes/embeddings_model_selection/candidates",
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["recipe_id"] == "embeddings_model_selection"
    assert body["current"]["provider"] == "openai"
    assert body["candidates"][0]["model"] == "text-embedding-3-small"


@pytest.mark.asyncio
async def test_embeddings_recipe_apply_preview_returns_copy_config_for_completed_run(
    async_api_client,
    auth_headers,
    monkeypatch,
) -> None:
    db = EvaluationsDatabase(os.environ["EVALUATIONS_TEST_DB_PATH"])
    user_id = get_single_user_instance().id_str
    run_id = db.create_recipe_run(
        recipe_id="embeddings_model_selection",
        recipe_version="1",
        status=RunStatus.COMPLETED,
        dataset_content_hash=build_dataset_content_hash(_embeddings_dataset()),
        metadata={"owner_user_id": user_id},
        recommendation_slots={
            "best_overall": RecommendationSlot(
                candidate_run_id="arm-1",
                metadata={
                    "provider": "openai",
                    "model": "text-embedding-3-small",
                    "apply_eligible": True,
                    "apply_warnings": ["Existing indexes may need rebuild."],
                },
            )
        },
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_current_embedding_config",
        lambda: {"provider": "huggingface", "model": "Qwen/Qwen3-Embedding-0.6B"},
    )

    response = await async_api_client.post(
        f"/api/v1/evaluations/recipe-runs/{run_id}/apply-preview",
        json={"slot_name": "best_overall", "candidate_run_id": "arm-1"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["apply_eligible"] is True
    assert body["apply_available"] is False
    assert body["copy_config"]["Embeddings"] == {
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
    }
    assert body["reindex_required"] is True


@pytest.mark.asyncio
async def test_embeddings_recipe_apply_preview_rejects_non_embeddings_recipe_run(
    async_api_client,
    auth_headers,
) -> None:
    db = EvaluationsDatabase(os.environ["EVALUATIONS_TEST_DB_PATH"])
    run_id = db.create_recipe_run(
        recipe_id="summarization_quality",
        recipe_version="1",
        status=RunStatus.COMPLETED,
        dataset_content_hash=build_dataset_content_hash(_inline_dataset()),
        metadata={"owner_user_id": get_single_user_instance().id_str},
    )

    response = await async_api_client.post(
        f"/api/v1/evaluations/recipe-runs/{run_id}/apply-preview",
        json={"slot_name": "best_overall"},
        headers=auth_headers,
    )

    assert response.status_code == 409
    assert "not 'embeddings_model_selection'" in response.json()["detail"]


@pytest.mark.asyncio
async def test_recipe_run_create_endpoint_enqueues_pending_job(
    async_api_client,
    auth_headers,
) -> None:
    from tldw_Server_API.app.main import app

    captured: dict[str, Any] = {}

    def _capture_enqueue(record, *, owner_user_id=None, job_manager=None):
        del job_manager
        captured["run_id"] = record.run_id
        captured["owner_user_id"] = owner_user_id
        return "job-123"

    app.dependency_overrides[get_recipe_run_job_enqueuer] = lambda: _capture_enqueue

    response = await async_api_client.post(
        "/api/v1/evaluations/recipes/summarization_quality/runs",
        json={
            "dataset": _inline_dataset(),
            "run_config": _run_config(),
        },
        headers=auth_headers,
    )

    assert response.status_code == 202
    body = response.json()
    assert captured["run_id"] == body["run_id"]
    assert captured["owner_user_id"] == get_single_user_instance().id_str


@pytest.mark.asyncio
async def test_recipe_run_create_endpoint_marks_run_failed_when_enqueue_fails(
    async_api_client,
    auth_headers,
) -> None:
    from tldw_Server_API.app.main import app

    def _raise_enqueue(record, *, owner_user_id=None, job_manager=None):
        del record, owner_user_id, job_manager
        raise RecipeEnqueueError()

    app.dependency_overrides[get_recipe_run_job_enqueuer] = lambda: _raise_enqueue

    response = await async_api_client.post(
        "/api/v1/evaluations/recipes/summarization_quality/runs",
        json={
            "dataset": _inline_dataset(),
            "run_config": _run_config(),
        },
        headers=auth_headers,
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "recipe_run_enqueue_failed"

    db = EvaluationsDatabase(os.environ["EVALUATIONS_TEST_DB_PATH"])
    user_id = get_single_user_instance().id_str
    reuse_hash = RecipeRunsService(db=db, user_id=user_id).build_reuse_hash(
        "summarization_quality",
        dataset=_inline_dataset(),
        run_config=_run_config(),
    )
    failed_run_id = db.lookup_idempotency(
        RECIPE_RUN_REUSE_ENTITY_TYPE,
        reuse_hash,
        user_id,
    )
    assert failed_run_id is not None
    failed_run = db.get_recipe_run(failed_run_id)
    assert failed_run is not None
    assert failed_run.status is RunStatus.FAILED
    assert failed_run.metadata["jobs"]["worker_state"] == "enqueue_failed"
    assert failed_run.metadata["jobs"]["error"] == "recipe_run_enqueue_failed"
    assert failed_run.metadata["jobs"]["error_message"] == "Failed to enqueue recipe run."


@pytest.mark.asyncio
async def test_recipe_run_create_endpoint_rejects_new_pending_run_when_worker_disabled(
    async_api_client,
    auth_headers,
    monkeypatch,
) -> None:
    monkeypatch.delenv("EVALUATIONS_RECIPE_RUN_JOBS_WORKER_ENABLED", raising=False)
    monkeypatch.delenv("EVALS_RECIPE_RUN_JOBS_WORKER_ENABLED", raising=False)

    response = await async_api_client.post(
        "/api/v1/evaluations/recipes/summarization_quality/runs",
        json={
            "dataset": _inline_dataset(),
            "run_config": _run_config(),
        },
        headers=auth_headers,
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "recipe_run_worker_disabled"

    db = EvaluationsDatabase(os.environ["EVALUATIONS_TEST_DB_PATH"])
    user_id = get_single_user_instance().id_str
    reuse_hash = RecipeRunsService(db=db, user_id=user_id).build_reuse_hash(
        "summarization_quality",
        dataset=_inline_dataset(),
        run_config=_run_config(),
    )
    failed_run_id = db.lookup_idempotency(
        RECIPE_RUN_REUSE_ENTITY_TYPE,
        reuse_hash,
        user_id,
    )
    assert failed_run_id is not None
    failed_run = db.get_recipe_run(failed_run_id)
    assert failed_run is not None
    assert failed_run.status is RunStatus.FAILED
    assert failed_run.metadata["jobs"]["worker_state"] == "disabled"
    assert failed_run.metadata["jobs"]["error"] == "recipe_run_worker_disabled"


@pytest.mark.asyncio
async def test_recipe_run_endpoint_reuses_completed_run_unless_forced(async_api_client, auth_headers) -> None:
    db_path = os.environ["EVALUATIONS_TEST_DB_PATH"]
    db = EvaluationsDatabase(db_path)
    user_id = get_single_user_instance().id_str
    dataset = _inline_dataset()
    run_config = _run_config()
    create_response = await async_api_client.post(
        "/api/v1/evaluations/recipes/summarization_quality/runs",
        json={
            "dataset": dataset,
            "run_config": run_config,
        },
        headers=auth_headers,
    )

    assert create_response.status_code == 202
    created = create_response.json()
    reuse_hash = created["metadata"]["reuse_hash"]

    assert (
        db.lookup_idempotency(
            RECIPE_RUN_REUSE_ENTITY_TYPE,
            reuse_hash,
            user_id,
        )
        == created["run_id"]
    )

    _mark_recipe_run_completed(db, created["run_id"])

    response = await async_api_client.post(
        "/api/v1/evaluations/recipes/summarization_quality/runs",
        json={
            "dataset": dataset,
            "run_config": run_config,
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    reused = response.json()
    assert reused["run_id"] == created["run_id"]
    assert reused["status"] == "completed"
    assert (
        db.lookup_idempotency(
            RECIPE_RUN_REUSE_ENTITY_TYPE,
            reuse_hash,
            user_id,
        )
        == created["run_id"]
    )

    forced_response = await async_api_client.post(
        "/api/v1/evaluations/recipes/summarization_quality/runs",
        json={
            "dataset": dataset,
            "run_config": run_config,
            "force_rerun": True,
        },
        headers=auth_headers,
    )

    assert forced_response.status_code == 202
    forced = forced_response.json()
    assert forced["run_id"] != created["run_id"]
    assert forced["status"] == "pending"


@pytest.mark.asyncio
async def test_recipe_run_endpoint_repairs_stale_reuse_mapping_to_latest_completed_run(
    async_api_client,
    auth_headers,
) -> None:
    db_path = os.environ["EVALUATIONS_TEST_DB_PATH"]
    db = EvaluationsDatabase(db_path)
    user_id = get_single_user_instance().id_str
    dataset = _inline_dataset()
    run_config = _run_config()

    first_create_response = await async_api_client.post(
        "/api/v1/evaluations/recipes/summarization_quality/runs",
        json={
            "dataset": dataset,
            "run_config": run_config,
        },
        headers=auth_headers,
    )

    assert first_create_response.status_code == 202
    first_run = first_create_response.json()
    reuse_hash = first_run["metadata"]["reuse_hash"]
    db.record_idempotency(
        RECIPE_RUN_REUSE_ENTITY_TYPE,
        reuse_hash,
        first_run["run_id"],
        user_id,
    )

    forced_response = await async_api_client.post(
        "/api/v1/evaluations/recipes/summarization_quality/runs",
        json={
            "dataset": dataset,
            "run_config": run_config,
            "force_rerun": True,
        },
        headers=auth_headers,
    )

    assert forced_response.status_code == 202
    forced_run = forced_response.json()
    _mark_recipe_run_completed(db, forced_run["run_id"])
    _set_reuse_mapping(
        db,
        reuse_hash=reuse_hash,
        user_id=user_id,
        run_id=first_run["run_id"],
    )

    reused_response = await async_api_client.post(
        "/api/v1/evaluations/recipes/summarization_quality/runs",
        json={
            "dataset": dataset,
            "run_config": run_config,
        },
        headers=auth_headers,
    )

    assert reused_response.status_code == 200
    reused = reused_response.json()
    assert reused["run_id"] == forced_run["run_id"]
    assert reused["status"] == "completed"
    assert (
        db.lookup_idempotency(
            RECIPE_RUN_REUSE_ENTITY_TYPE,
            reuse_hash,
            user_id,
        )
        == forced_run["run_id"]
    )


@pytest.mark.asyncio
async def test_recipe_run_endpoint_does_not_reuse_completed_run_from_other_user(
    async_api_client,
    auth_headers,
) -> None:
    db_path = os.environ["EVALUATIONS_TEST_DB_PATH"]
    db = EvaluationsDatabase(db_path)
    dataset = _inline_dataset()
    run_config = _run_config()
    current_user_id = get_single_user_instance().id_str
    reuse_hash = RecipeRunsService(db=db, user_id=current_user_id).build_reuse_hash(
        "summarization_quality",
        dataset=dataset,
        run_config=run_config,
    )

    other_user_run_id = db.create_recipe_run(
        recipe_id="summarization_quality",
        recipe_version="1",
        status=RunStatus.COMPLETED,
        dataset_content_hash=build_dataset_content_hash(dataset),
        metadata={
            "run_config": run_config,
            "reuse_hash": reuse_hash,
            "owner_user_id": "other-user",
        },
    )
    _mark_recipe_run_completed(db, other_user_run_id)

    reused_response = await async_api_client.post(
        "/api/v1/evaluations/recipes/summarization_quality/runs",
        json={
            "dataset": dataset,
            "run_config": run_config,
        },
        headers=auth_headers,
    )

    assert reused_response.status_code == 202
    reused = reused_response.json()
    assert reused["run_id"] != other_user_run_id
    assert reused["metadata"]["owner_user_id"] == current_user_id


@pytest.mark.asyncio
async def test_recipe_run_endpoint_reuses_legacy_completed_run_without_owner_in_single_user_mode(
    async_api_client,
    auth_headers,
) -> None:
    db_path = os.environ["EVALUATIONS_TEST_DB_PATH"]
    db = EvaluationsDatabase(db_path)
    dataset = _inline_dataset()
    run_config = _run_config()
    reuse_hash = RecipeRunsService(
        db=db,
        user_id=get_single_user_instance().id_str,
    ).build_reuse_hash(
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

    response = await async_api_client.post(
        "/api/v1/evaluations/recipes/summarization_quality/runs",
        json={
            "dataset": dataset,
            "run_config": run_config,
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    reused = response.json()
    assert reused["run_id"] == legacy_run_id
    assert reused["status"] == "completed"
