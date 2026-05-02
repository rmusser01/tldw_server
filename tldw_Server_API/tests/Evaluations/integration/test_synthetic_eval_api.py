from __future__ import annotations

import pytest

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_single_user_instance
from tldw_Server_API.app.core.Evaluations.synthetic_eval_service import (
    get_synthetic_eval_service_for_user,
)

pytestmark = [pytest.mark.integration]


def _retrieval_generation_payload() -> dict:
    return {
        "recipe_kind": "rag_retrieval_tuning",
        "corpus_scope": {"sources": ["media_db", "notes"]},
        "real_examples": [
            {
                "sample_id": "real-1",
                "source_kind": "media_db",
                "query_intent": "lookup",
                "difficulty": "straightforward",
                "sample_payload": {"query": "Existing real retrieval query"},
                "provenance": "real",
            }
        ],
        "seed_examples": [],
        "target_sample_count": 2,
    }


def _answer_quality_generation_payload() -> dict:
    return {
        "recipe_kind": "rag_answer_quality",
        "corpus_scope": {"sources": ["media_db", "notes"]},
        "context_snapshot_ref": "context-1",
        "retrieval_baseline_ref": "baseline-1",
        "reference_answer": "The rollout finished on Friday, but beta access remained limited.",
        "real_examples": [
            {
                "sample_id": "real-aq-1",
                "source_kind": "notes",
                "query_intent": "comparison",
                "difficulty": "distractor-heavy",
                "sample_payload": {
                    "query": "Should this answer hedge?",
                    "expected_behavior": "hedge",
                },
                "provenance": "real",
            }
        ],
        "seed_examples": [],
        "target_sample_count": 2,
    }


@pytest.mark.asyncio
async def test_synthetic_queue_filters_by_recipe_kind(async_api_client, auth_headers) -> None:
    retrieval_response = await async_api_client.post(
        "/api/v1/evaluations/synthetic/drafts/generate",
        json=_retrieval_generation_payload(),
        headers=auth_headers,
    )
    assert retrieval_response.status_code == 200

    answer_quality_response = await async_api_client.post(
        "/api/v1/evaluations/synthetic/drafts/generate",
        json=_answer_quality_generation_payload(),
        headers=auth_headers,
    )
    assert answer_quality_response.status_code == 200

    response = await async_api_client.get(
        "/api/v1/evaluations/synthetic/queue",
        params={"recipe_kind": "rag_answer_quality"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["data"]
    assert body["total"] == 2
    assert body["pagination"] == {
        "mode": "offset",
        "limit": 50,
        "offset": 0,
        "total": 2,
        "has_more": False,
        "next_offset": None,
    }
    assert body["has_more"] is False
    assert body["next_offset"] is None
    assert all(item["recipe_kind"] == "rag_answer_quality" for item in body["data"])


@pytest.mark.asyncio
async def test_synthetic_review_and_promotion_flow(async_api_client, auth_headers) -> None:
    user_id = get_single_user_instance().id_str
    generation_response = await async_api_client.post(
        "/api/v1/evaluations/synthetic/drafts/generate",
        json=_answer_quality_generation_payload(),
        headers=auth_headers,
    )
    assert generation_response.status_code == 200
    sample_id = generation_response.json()["samples"][0]["sample_id"]

    review_response = await async_api_client.post(
        f"/api/v1/evaluations/synthetic/queue/{sample_id}/review",
        json={
            "action": "approve",
            "reviewer_id": "reviewer-1",
            "notes": "Looks realistic enough.",
        },
        headers=auth_headers,
    )
    assert review_response.status_code == 200
    assert review_response.json()["resulting_review_state"] == "approved"
    assert review_response.json()["reviewer_id"] == user_id

    promote_response = await async_api_client.post(
        "/api/v1/evaluations/synthetic/promotions",
        json={
            "sample_ids": [sample_id],
            "dataset_name": "approved synthetic answer-quality set",
            "dataset_description": "Reviewed synthetic samples for answer quality.",
            "promoted_by": "malicious-reviewer",
        },
        headers=auth_headers,
    )

    assert promote_response.status_code == 200
    body = promote_response.json()
    assert body["dataset_id"].startswith("dataset_")
    assert body["dataset_snapshot_ref"]
    assert len(body["promotion_ids"]) == 1
    assert body["sample_count"] == 1
    service = get_synthetic_eval_service_for_user(user_id)
    promotion_record = service.repository.get_promotion(body["promotion_ids"][0])
    assert promotion_record is not None
    assert promotion_record["promoted_by"] == user_id


@pytest.mark.asyncio
async def test_synthetic_queue_filters_by_generation_batch_id(async_api_client, auth_headers) -> None:
    retrieval_response = await async_api_client.post(
        "/api/v1/evaluations/synthetic/drafts/generate",
        json=_retrieval_generation_payload(),
        headers=auth_headers,
    )
    assert retrieval_response.status_code == 200
    generation_batch_id = retrieval_response.json()["generation_batch_id"]

    other_response = await async_api_client.post(
        "/api/v1/evaluations/synthetic/drafts/generate",
        json=_answer_quality_generation_payload(),
        headers=auth_headers,
    )
    assert other_response.status_code == 200

    response = await async_api_client.get(
        "/api/v1/evaluations/synthetic/queue",
        params={"generation_batch_id": generation_batch_id},
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["data"]
    assert body["total"] == 2
    assert body["pagination"] == {
        "mode": "offset",
        "limit": 50,
        "offset": 0,
        "total": 2,
        "has_more": False,
        "next_offset": None,
    }
    assert body["has_more"] is False
    assert body["next_offset"] is None
    assert all(
        item["sample_metadata"]["generation_batch_id"] == generation_batch_id
        for item in body["data"]
    )


@pytest.mark.asyncio
async def test_generate_endpoint_returns_batch_id_and_anchor_metadata(async_api_client, auth_headers) -> None:
    response = await async_api_client.post(
        "/api/v1/evaluations/synthetic/drafts/generate",
        json=_answer_quality_generation_payload(),
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["generation_batch_id"]
    assert body["samples"][0]["sample_metadata"]["generation_batch_id"] == body["generation_batch_id"]
    assert body["samples"][0]["sample_metadata"]["generation_metadata"]["retrieval_baseline_ref"] == "baseline-1"
    assert body["samples"][0]["sample_payload"]["context_snapshot_ref"] == "context-1"
    assert body["samples"][0]["sample_payload"]["retrieval_baseline_ref"] == "baseline-1"
    assert body["samples"][0]["sample_payload"]["reference_answer"] == "The rollout finished on Friday, but beta access remained limited."


@pytest.mark.asyncio
async def test_promote_endpoint_requires_sample_ids(async_api_client, auth_headers) -> None:
    response = await async_api_client.post(
        "/api/v1/evaluations/synthetic/promotions",
        json={"dataset_name": "missing samples"},
        headers=auth_headers,
    )

    assert response.status_code == 422
