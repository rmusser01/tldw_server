from __future__ import annotations

import pytest

from tldw_Server_API.app.api.v1.schemas.synthetic_eval_schemas import (
    SyntheticEvalProvenance,
    SyntheticEvalReviewActionType,
    SyntheticEvalReviewState,
)
from tldw_Server_API.app.core.DB_Management.Evaluations_DB import EvaluationsDatabase
from tldw_Server_API.app.core.Evaluations.synthetic_eval_generation import (
    SyntheticEvalCorpusScope,
    SyntheticEvalStructuredTuple,
)
from tldw_Server_API.app.core.Evaluations.synthetic_eval_repository import SyntheticEvalRepository
from tldw_Server_API.app.core.Evaluations.synthetic_eval_service import (
    SyntheticEvalGenerationService,
    SyntheticEvalWorkflowService,
)


def _repository(tmp_path) -> SyntheticEvalRepository:
    db = EvaluationsDatabase(str(tmp_path / "evaluations.db"))
    return SyntheticEvalRepository(db)


def _service(tmp_path) -> SyntheticEvalGenerationService:
    db = EvaluationsDatabase(str(tmp_path / "evaluations.db"))
    repository = SyntheticEvalRepository(db)
    return SyntheticEvalGenerationService(repository=repository)


def _workflow_service(tmp_path) -> SyntheticEvalWorkflowService:
    db = EvaluationsDatabase(str(tmp_path / "evaluations.db"))
    repository = SyntheticEvalRepository(db)
    return SyntheticEvalWorkflowService(db=db, repository=repository)


def test_evaluations_sqlite_connections_enable_foreign_keys(tmp_path) -> None:
    db = EvaluationsDatabase(str(tmp_path / "evaluations.db"))

    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys")
        row = cursor.fetchone()

    assert row[0] == 1  # nosec B101


def test_repository_persists_draft_samples_with_provenance_and_review_state(tmp_path) -> None:
    repository = _repository(tmp_path)

    repository.create_draft_sample(
        sample_id="sample-1",
        recipe_kind="rag_retrieval_tuning",
        sample_payload={
            "query": "What does the policy say?",
            "targets": {"relevant_media_ids": [42]},
        },
        provenance=SyntheticEvalProvenance.SYNTHETIC_FROM_CORPUS.value,
        review_state=SyntheticEvalReviewState.DRAFT.value,
        source_kind="media_db",
    )

    row = repository.get_draft_sample("sample-1")

    assert row is not None  # nosec B101
    assert row["provenance"] == "synthetic_from_corpus"  # nosec B101
    assert row["review_state"] == "draft"  # nosec B101
    assert row["sample_payload"]["query"] == "What does the policy say?"  # nosec B101


def test_repository_records_review_action_history(tmp_path) -> None:
    repository = _repository(tmp_path)

    repository.create_draft_sample(
        sample_id="sample-2",
        recipe_kind="rag_answer_quality",
        sample_payload={
            "query": "Should we answer this?",
            "expected_behavior": "abstain",
        },
        provenance=SyntheticEvalProvenance.SYNTHETIC_FROM_CORPUS.value,
        review_state=SyntheticEvalReviewState.DRAFT.value,
        source_kind="notes",
    )

    repository.record_review_action(
        sample_id="sample-2",
        action=SyntheticEvalReviewActionType.EDIT.value,
        reviewer_id="reviewer-a",
        notes="Tighten the framing.",
    )
    repository.record_review_action(
        sample_id="sample-2",
        action=SyntheticEvalReviewActionType.EDIT_AND_APPROVE.value,
        reviewer_id="reviewer-a",
        notes="Approved after edit.",
    )

    history = repository.list_review_actions("sample-2")

    assert history[-1]["action"] == "edit_and_approve"  # nosec B101


def test_repository_marks_separately_edited_samples_as_human_edited_when_approved(tmp_path) -> None:
    repository = _repository(tmp_path)

    repository.create_draft_sample(
        sample_id="sample-3",
        recipe_kind="rag_retrieval_tuning",
        sample_payload={"query": "Does this need an edit?"},
        provenance=SyntheticEvalProvenance.SYNTHETIC_FROM_CORPUS.value,
        review_state=SyntheticEvalReviewState.DRAFT.value,
    )

    repository.record_review_action(
        sample_id="sample-3",
        action=SyntheticEvalReviewActionType.EDIT.value,
        reviewer_id="reviewer-b",
        notes="Refine the query wording.",
    )
    repository.record_review_action(
        sample_id="sample-3",
        action=SyntheticEvalReviewActionType.APPROVE.value,
        reviewer_id="reviewer-b",
        notes="Approved after edit.",
    )

    row = repository.get_draft_sample("sample-3")

    assert row is not None  # nosec B101
    assert row["provenance"] == "synthetic_human_edited"  # nosec B101
    assert row["review_state"] == "approved"  # nosec B101


def test_repository_rejects_review_actions_for_missing_samples(tmp_path) -> None:
    repository = _repository(tmp_path)

    with pytest.raises(ValueError, match="sample does not exist"):
        repository.record_review_action(
            sample_id="missing-sample",
            action=SyntheticEvalReviewActionType.EDIT.value,
            reviewer_id="reviewer-c",
        )


def test_repository_rejects_promotions_for_missing_samples(tmp_path) -> None:
    repository = _repository(tmp_path)

    with pytest.raises(ValueError, match="sample does not exist"):
        repository.record_promotion(sample_id="missing-sample")


def test_repository_requires_approved_review_state_before_promotion(tmp_path) -> None:
    repository = _repository(tmp_path)

    repository.create_draft_sample(
        sample_id="sample-4",
        recipe_kind="rag_retrieval_tuning",
        sample_payload={"query": "Needs approval first."},
        provenance=SyntheticEvalProvenance.SYNTHETIC_FROM_CORPUS.value,
        review_state=SyntheticEvalReviewState.DRAFT.value,
    )

    with pytest.raises(ValueError, match="approved"):
        repository.record_promotion(sample_id="sample-4")


def test_repository_preserves_real_edited_provenance_after_edit_and_approve(tmp_path) -> None:
    repository = _repository(tmp_path)

    repository.create_draft_sample(
        sample_id="sample-5",
        recipe_kind="rag_retrieval_tuning",
        sample_payload={"query": "Keep the real provenance."},
        provenance=SyntheticEvalProvenance.REAL_EDITED.value,
        review_state=SyntheticEvalReviewState.DRAFT.value,
    )

    repository.record_review_action(
        sample_id="sample-5",
        action=SyntheticEvalReviewActionType.EDIT.value,
        reviewer_id="reviewer-d",
        notes="Minor phrasing change.",
    )
    repository.record_review_action(
        sample_id="sample-5",
        action=SyntheticEvalReviewActionType.APPROVE.value,
        reviewer_id="reviewer-d",
        notes="Approved after edit.",
    )

    row = repository.get_draft_sample("sample-5")

    assert row is not None  # nosec B101
    assert row["provenance"] == "real_edited"  # nosec B101
    assert row["review_state"] == "approved"  # nosec B101


def test_repository_filters_draft_samples_by_generation_batch_id(tmp_path) -> None:
    repository = _repository(tmp_path)

    repository.create_draft_sample(
        sample_id="sample-batch-a",
        recipe_kind="rag_retrieval_tuning",
        sample_payload={"query": "Batch A"},
        provenance=SyntheticEvalProvenance.SYNTHETIC_FROM_CORPUS.value,
        review_state=SyntheticEvalReviewState.DRAFT.value,
        sample_metadata={"generation_batch_id": "batch-a"},
    )
    repository.create_draft_sample(
        sample_id="sample-batch-b",
        recipe_kind="rag_retrieval_tuning",
        sample_payload={"query": "Batch B"},
        provenance=SyntheticEvalProvenance.SYNTHETIC_FROM_CORPUS.value,
        review_state=SyntheticEvalReviewState.DRAFT.value,
        sample_metadata={"generation_batch_id": "batch-b"},
    )

    rows = repository.list_draft_samples(generation_batch_id="batch-a")

    assert [row["sample_id"] for row in rows] == ["sample-batch-a"]  # nosec B101


def test_service_filters_queue_by_generation_batch_id(tmp_path) -> None:
    service = _workflow_service(tmp_path)

    service.repository.create_draft_sample(
        sample_id="sample-service-a",
        recipe_kind="rag_retrieval_tuning",
        sample_payload={"query": "Batch A"},
        provenance=SyntheticEvalProvenance.SYNTHETIC_FROM_CORPUS.value,
        review_state=SyntheticEvalReviewState.DRAFT.value,
        sample_metadata={"generation_batch_id": "batch-a"},
    )
    service.repository.create_draft_sample(
        sample_id="sample-service-b",
        recipe_kind="rag_retrieval_tuning",
        sample_payload={"query": "Batch B"},
        provenance=SyntheticEvalProvenance.SYNTHETIC_FROM_CORPUS.value,
        review_state=SyntheticEvalReviewState.DRAFT.value,
        sample_metadata={"generation_batch_id": "batch-b"},
    )

    queue = service.list_queue(generation_batch_id="batch-a")

    assert [row["sample_id"] for row in queue["data"]] == ["sample-service-a"]  # nosec B101


def test_service_reports_true_queue_total_before_pagination(tmp_path) -> None:
    service = _workflow_service(tmp_path)

    for sample_id in ("sample-total-a", "sample-total-b", "sample-total-c"):
        service.repository.create_draft_sample(
            sample_id=sample_id,
            recipe_kind="rag_answer_quality",
            sample_payload={"query": sample_id},
            provenance=SyntheticEvalProvenance.SYNTHETIC_FROM_CORPUS.value,
            review_state=SyntheticEvalReviewState.DRAFT.value,
            sample_metadata={"generation_batch_id": "batch-total"},
        )

    queue = service.list_queue(generation_batch_id="batch-total", limit=2, offset=0)

    page_sample_ids = {row["sample_id"] for row in queue["data"]}
    assert len(page_sample_ids) == 2  # nosec B101
    assert page_sample_ids <= {"sample-total-a", "sample-total-b", "sample-total-c"}  # nosec B101
    assert queue["total"] == 3  # nosec B101


def test_generation_prefers_real_examples_before_seed_and_synthetic_fill(tmp_path) -> None:
    service = _service(tmp_path)
    generation_calls: list[dict[str, object]] = []

    def _fake_generate_structured_tuples(*, coverage_gaps, corpus_scope, corpus_examples, seed_examples, target_count):
        generation_calls.append(
            {
                "coverage_gaps": coverage_gaps,
                "corpus_scope": corpus_scope,
                "corpus_examples": corpus_examples,
                "seed_examples": seed_examples,
                "target_count": target_count,
            }
        )
        return [
            SyntheticEvalStructuredTuple(
                source_kind="media_db",
                query_intent="lookup",
                difficulty="straightforward",
                query="What does the policy say?",
                failure_mode="missing_relevant_span",
                target_payload={"relevant_media_ids": [1]},
                provenance_hint="synthetic_from_corpus",
            )
        ]

    result = service.generate_draft_batch(
        recipe_kind="rag_retrieval_tuning",
        corpus_scope=SyntheticEvalCorpusScope(sources=("media_db", "notes")),
        real_examples=[
            {
                "sample_id": "real-1",
                "source_kind": "media_db",
                "query_intent": "lookup",
                "difficulty": "straightforward",
                "sample_payload": {"query": "Existing real query"},
                "provenance": SyntheticEvalProvenance.REAL.value,
            }
        ],
        seed_examples=[
            {
                "sample_id": "seed-1",
                "source_kind": "notes",
                "query_intent": "comparison",
                "difficulty": "distractor-heavy",
                "sample_payload": {"query": "Seeded query"},
            }
        ],
        target_sample_count=3,
        tuple_generator=_fake_generate_structured_tuples,
    )

    assert generation_calls  # nosec B101
    assert result.source_breakdown["real"] == 1  # nosec B101
    assert result.source_breakdown["seed_examples"] == 1  # nosec B101
    assert result.source_breakdown["synthetic_from_corpus"] == 1  # nosec B101
    assert [sample["provenance"] for sample in result.samples] == [  # nosec B101
        "real",
        "synthetic_from_seed_examples",
        "synthetic_from_corpus",
    ]


def test_generation_batch_response_preserves_generation_batch_id_and_scope_metadata(tmp_path) -> None:
    service = _service(tmp_path)

    result = service.generate_draft_batch(
        recipe_kind="rag_retrieval_tuning",
        corpus_scope={
            "sources": ["media_db"],
            "media_ids": [10],
            "note_ids": ["note-7"],
            "indexing_fixed": True,
        },
        target_sample_count=1,
        tuple_generator=lambda **kwargs: [
            SyntheticEvalStructuredTuple(
                source_kind="media_db",
                query_intent="lookup",
                difficulty="straightforward",
                query="What does the policy say?",
                failure_mode="missing_relevant_span",
                target_payload={"relevant_media_ids": [10]},
                provenance_hint="synthetic_from_corpus",
            )
        ],
    )

    sample_metadata = result.samples[0]["sample_metadata"]

    assert result.generation_batch_id  # nosec B101
    assert sample_metadata["generation_batch_id"] == result.generation_batch_id  # nosec B101
    assert sample_metadata["corpus_scope"]["media_ids"] == [10]  # nosec B101
    assert sample_metadata["corpus_scope"]["note_ids"] == ["note-7"]  # nosec B101
    assert sample_metadata["corpus_scope"]["indexing_fixed"] is True  # nosec B101


def test_generation_batch_response_preserves_answer_quality_anchor_metadata(tmp_path) -> None:
    service = _service(tmp_path)

    result = service.generate_draft_batch(
        recipe_kind="rag_answer_quality",
        corpus_scope={"sources": ["notes"]},
        context_snapshot_ref="context-1",
        retrieval_baseline_ref="baseline-1",
        reference_answer="The answer should hedge because context is insufficient.",
        target_sample_count=1,
        tuple_generator=lambda **kwargs: [
            SyntheticEvalStructuredTuple(
                source_kind="notes",
                query_intent="comparison",
                difficulty="distractor-heavy",
                query="Should this answer hedge?",
                failure_mode="comparison_confusion",
                target_payload={"expected_behavior": "hedge"},
                provenance_hint="synthetic_from_corpus",
            )
        ],
    )

    sample_metadata = result.samples[0]["sample_metadata"]
    sample_payload = result.samples[0]["sample_payload"]

    assert result.generation_batch_id  # nosec B101
    assert sample_metadata["generation_batch_id"] == result.generation_batch_id  # nosec B101
    assert sample_metadata["generation_metadata"]["context_snapshot_ref"] == "context-1"  # nosec B101
    assert sample_metadata["generation_metadata"]["retrieval_baseline_ref"] == "baseline-1"  # nosec B101
    assert sample_payload["context_snapshot_ref"] == "context-1"  # nosec B101
    assert sample_payload["retrieval_baseline_ref"] == "baseline-1"  # nosec B101
    assert sample_payload["reference_answer"] == "The answer should hedge because context is insufficient."  # nosec B101


def test_generation_treats_seed_examples_as_seed_provenance_by_default(tmp_path) -> None:
    service = _service(tmp_path)

    result = service.generate_draft_batch(
        recipe_kind="rag_retrieval_tuning",
        corpus_scope=SyntheticEvalCorpusScope(sources=("media_db", "notes")),
        real_examples=[
            {
                "sample_id": "real-2",
                "source_kind": "media_db",
                "query_intent": "lookup",
                "difficulty": "straightforward",
                "sample_payload": {"query": "Existing real query"},
                "provenance": SyntheticEvalProvenance.REAL.value,
            }
        ],
        seed_examples=[
            {
                "sample_id": "seed-2",
                "source_kind": "notes",
                "query_intent": "comparison",
                "difficulty": "distractor-heavy",
                "sample_payload": {"query": "Implicit seed query"},
            }
        ],
        target_sample_count=2,
        tuple_generator=lambda **kwargs: [],
    )

    assert result.source_breakdown["real"] == 1  # nosec B101
    assert result.source_breakdown["seed_examples"] == 1  # nosec B101
    assert [sample["provenance"] for sample in result.samples] == [  # nosec B101
        "real",
        "synthetic_from_seed_examples",
    ]


def test_generation_is_stratified_across_media_notes_intent_and_difficulty(tmp_path) -> None:
    service = _service(tmp_path)

    result = service.generate_draft_batch(
        recipe_kind="rag_answer_quality",
        corpus_scope=SyntheticEvalCorpusScope(sources=("media_db", "notes")),
        real_examples=[
            {
                "sample_id": "real-media-1",
                "source_kind": "media_db",
                "query_intent": "lookup",
                "difficulty": "straightforward",
                "sample_payload": {"query": "Media lookup"},
                "provenance": SyntheticEvalProvenance.REAL.value,
            },
            {
                "sample_id": "real-notes-1",
                "source_kind": "notes",
                "query_intent": "synthesis",
                "difficulty": "multi-source",
                "sample_payload": {"query": "Notes synthesis"},
                "provenance": SyntheticEvalProvenance.REAL.value,
            },
        ],
        seed_examples=[],
        target_sample_count=4,
        tuple_generator=lambda **kwargs: [
            SyntheticEvalStructuredTuple(
                source_kind="media_db",
                query_intent="comparison",
                difficulty="distractor-heavy",
                query="What conflicts with the source?",
                failure_mode="confused_sources",
                target_payload={"expected_behavior": "hedge"},
                provenance_hint="synthetic_from_corpus",
            ),
            SyntheticEvalStructuredTuple(
                source_kind="notes",
                query_intent="ambiguous / underspecified",
                difficulty="abstention-worthy",
                query="Is this answerable from the notes alone?",
                failure_mode="requires_abstention",
                target_payload={"expected_behavior": "abstain"},
                provenance_hint="synthetic_from_corpus",
            ),
        ],
    )

    assert "media_db" in result.coverage["sources"]  # nosec B101
    assert "notes" in result.coverage["sources"]  # nosec B101
    assert {"lookup", "synthesis", "comparison"} <= set(result.coverage["query_intents"])  # nosec B101
    assert {"straightforward", "multi-source", "distractor-heavy", "abstention-worthy"} <= set(result.coverage["difficulties"])  # nosec B101
    assert result.source_breakdown["real"] == 2  # nosec B101
    assert result.source_breakdown["synthetic_from_corpus"] == 2  # nosec B101
