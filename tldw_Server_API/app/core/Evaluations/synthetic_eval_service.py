"""Service layer for synthetic evaluation draft generation."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from tldw_Server_API.app.api.v1.schemas.synthetic_eval_schemas import (
    SyntheticEvalProvenance,
    SyntheticEvalReviewState,
)
from tldw_Server_API.app.core.DB_Management.Evaluations_DB import EvaluationsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Evaluations.synthetic_eval_generation import (
    SyntheticEvalCorpusScope,
    SyntheticEvalGenerationPlan,
    SyntheticEvalStructuredTuple,
    convert_structured_tuples_to_draft_samples,
    generate_structured_tuples,
    ingest_examples,
    resolve_corpus_scope,
    summarize_coverage,
)
from tldw_Server_API.app.core.Evaluations.recipes.dataset_snapshot import (
    build_dataset_snapshot_ref,
)
from tldw_Server_API.app.core.Evaluations.synthetic_eval_repository import (
    SyntheticEvalRepository,
)


@dataclass(slots=True)
class SyntheticEvalGenerationResult:
    """Structured result for a draft generation batch."""

    generation_batch_id: str | None = None
    samples: list[dict[str, Any]] = field(default_factory=list)
    source_breakdown: dict[str, int] = field(default_factory=dict)
    coverage: dict[str, list[str]] = field(default_factory=dict)
    missing_coverage: dict[str, list[str]] = field(default_factory=dict)
    structured_tuples: list[SyntheticEvalStructuredTuple] = field(default_factory=list)
    corpus_scope: SyntheticEvalCorpusScope | None = None


@dataclass(slots=True)
class SyntheticEvalPromotionResult:
    """Result of promoting reviewed samples into a dataset."""

    dataset_id: str
    dataset_snapshot_ref: str
    promotion_ids: list[str] = field(default_factory=list)
    sample_count: int = 0


class SyntheticEvalGenerationService:
    """Orchestrate corpus scope resolution, generation, and persistence."""

    def __init__(
        self,
        *,
        repository: SyntheticEvalRepository,
        tuple_generator: Callable[..., Sequence[SyntheticEvalStructuredTuple]] | None = None,
    ) -> None:
        self.repository = repository
        self.tuple_generator = tuple_generator or generate_structured_tuples

    def generate_draft_batch(
        self,
        *,
        recipe_kind: str,
        corpus_scope: SyntheticEvalCorpusScope | Sequence[str] | None = None,
        generation_metadata: dict[str, Any] | None = None,
        context_snapshot_ref: str | None = None,
        retrieval_baseline_ref: str | None = None,
        reference_answer: str | None = None,
        real_examples: Sequence[dict[str, Any]] | None = None,
        seed_examples: Sequence[dict[str, Any]] | None = None,
        target_sample_count: int = 0,
        tuple_generator: Callable[..., Sequence[SyntheticEvalStructuredTuple]] | None = None,
        created_by: str | None = None,
    ) -> SyntheticEvalGenerationResult:
        """Generate a mixed batch using real, seed, and synthetic examples."""

        resolved_scope = resolve_corpus_scope(corpus_scope)
        generation_batch_id = f"synth_gen_{uuid.uuid4().hex[:12]}"
        resolved_generation_metadata = dict(generation_metadata or {})
        context_snapshot_ref = self._resolve_generation_field(
            explicit_value=context_snapshot_ref,
            metadata=resolved_generation_metadata,
            metadata_key="context_snapshot_ref",
        )
        retrieval_baseline_ref = self._resolve_generation_field(
            explicit_value=retrieval_baseline_ref,
            metadata=resolved_generation_metadata,
            metadata_key="retrieval_baseline_ref",
        )
        reference_answer = self._resolve_generation_field(
            explicit_value=reference_answer,
            metadata=resolved_generation_metadata,
            metadata_key="reference_answer",
        )
        anchor_fields: dict[str, Any] = {}
        if context_snapshot_ref is not None:
            resolved_generation_metadata.setdefault("context_snapshot_ref", context_snapshot_ref)
            anchor_fields["context_snapshot_ref"] = context_snapshot_ref
        if retrieval_baseline_ref is not None:
            resolved_generation_metadata.setdefault("retrieval_baseline_ref", retrieval_baseline_ref)
            anchor_fields["retrieval_baseline_ref"] = retrieval_baseline_ref
        if reference_answer is not None:
            resolved_generation_metadata.setdefault("reference_answer", reference_answer)
            anchor_fields["reference_answer"] = reference_answer
        plan: SyntheticEvalGenerationPlan = ingest_examples(
            real_examples=real_examples,
            seed_examples=seed_examples,
            corpus_scope=resolved_scope,
        )
        synthetic_needed = max(
            0,
            int(target_sample_count) - self._count_examples(plan.examples),
        )
        generator = tuple_generator or self.tuple_generator
        structured_tuples = list(
            generator(
                coverage_gaps=plan.missing_coverage,
                corpus_scope=resolved_scope,
                corpus_examples=[example for example in plan.examples if example["example_category"] == "real"],
                seed_examples=[example for example in plan.examples if example["example_category"] == "seed_examples"],
                target_count=synthetic_needed,
            )
        )
        synthetic_samples = convert_structured_tuples_to_draft_samples(
            recipe_kind=recipe_kind,
            structured_tuples=structured_tuples,
            corpus_scope=resolved_scope,
            generation_batch_id=generation_batch_id,
            generation_metadata=resolved_generation_metadata,
            sample_payload_overrides=anchor_fields,
        )

        persisted_samples: list[dict[str, Any]] = []
        for example in self._ordered_examples(plan.examples) + synthetic_samples:
            provenance = self._normalize_provenance(example.get("provenance"))
            sample_id = str(example.get("sample_id") or self._build_sample_id(recipe_kind, provenance, len(persisted_samples) + 1))
            sample_payload = dict(example.get("sample_payload") or {})
            for field_name, field_value in anchor_fields.items():
                sample_payload.setdefault(field_name, field_value)
            draft_row = self.repository.create_draft_sample(
                sample_id=sample_id,
                recipe_kind=example.get("recipe_kind") or recipe_kind,
                sample_payload=sample_payload,
                provenance=provenance,
                review_state=example.get("review_state") or SyntheticEvalReviewState.DRAFT.value,
                sample_metadata=self._build_sample_metadata(
                    example=example,
                    recipe_kind=recipe_kind,
                    created_by=created_by,
                    corpus_scope=resolved_scope,
                    generation_batch_id=generation_batch_id,
                    generation_metadata=resolved_generation_metadata,
                ),
                source_kind=example.get("source_kind"),
                created_by=created_by,
            )
            persisted_samples.append(draft_row)

        coverage = summarize_coverage(persisted_samples)
        source_breakdown = self._build_source_breakdown(persisted_samples)
        return SyntheticEvalGenerationResult(
            generation_batch_id=generation_batch_id,
            samples=persisted_samples,
            source_breakdown=source_breakdown,
            coverage={
                "sources": coverage.sources,
                "query_intents": coverage.query_intents,
                "difficulties": coverage.difficulties,
                "failure_modes": coverage.failure_modes,
            },
            missing_coverage=plan.missing_coverage,
            structured_tuples=structured_tuples,
            corpus_scope=resolved_scope,
        )

    def _ordered_examples(self, examples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        real_examples = [example for example in examples if self._normalize_provenance(example.get("provenance")) in {SyntheticEvalProvenance.REAL.value, SyntheticEvalProvenance.REAL_EDITED.value}]
        seed_examples = [example for example in examples if self._normalize_provenance(example.get("provenance")) == SyntheticEvalProvenance.SYNTHETIC_FROM_SEED_EXAMPLES.value]
        ordered = real_examples + seed_examples
        return ordered

    def _build_source_breakdown(self, samples: Sequence[dict[str, Any]]) -> dict[str, int]:
        breakdown = {
            "real": 0,
            "seed_examples": 0,
            "synthetic_from_corpus": 0,
        }
        for sample in samples:
            provenance = self._normalize_provenance(sample.get("provenance"))
            if provenance in {SyntheticEvalProvenance.REAL.value, SyntheticEvalProvenance.REAL_EDITED.value}:
                breakdown["real"] += 1
            elif provenance == SyntheticEvalProvenance.SYNTHETIC_FROM_SEED_EXAMPLES.value:
                breakdown["seed_examples"] += 1
            elif provenance == SyntheticEvalProvenance.SYNTHETIC_FROM_CORPUS.value:
                breakdown["synthetic_from_corpus"] += 1
        return breakdown

    def _build_sample_metadata(
        self,
        *,
        example: dict[str, Any],
        recipe_kind: str,
        created_by: str | None,
        corpus_scope: SyntheticEvalCorpusScope,
        generation_batch_id: str,
        generation_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = dict(example.get("sample_metadata") or {})
        metadata.setdefault("recipe_kind", recipe_kind)
        metadata.setdefault("source_kind", example.get("source_kind"))
        metadata.setdefault("query_intent", example.get("query_intent"))
        metadata.setdefault("difficulty", example.get("difficulty"))
        metadata.setdefault("created_by", created_by)
        metadata.setdefault("generation_stage", "generation_service")
        metadata.setdefault("generation_batch_id", generation_batch_id)
        metadata.setdefault("corpus_scope", corpus_scope.to_metadata_dict())
        metadata.setdefault("generation_metadata", dict(generation_metadata))
        return metadata

    def _build_sample_id(self, recipe_kind: str, provenance: str, index: int) -> str:
        safe_provenance = provenance.replace(" ", "_")
        return f"{recipe_kind}-{safe_provenance}-{index}"

    def _count_examples(self, examples: Sequence[dict[str, Any]]) -> int:
        return len(self._ordered_examples(examples))

    def _normalize_provenance(self, provenance: Any) -> str:
        if isinstance(provenance, SyntheticEvalProvenance):
            return provenance.value
        if provenance == "seed_examples":
            return SyntheticEvalProvenance.SYNTHETIC_FROM_SEED_EXAMPLES.value
        return str(provenance or SyntheticEvalProvenance.SYNTHETIC_FROM_CORPUS.value)

    def _resolve_generation_field(
        self,
        *,
        explicit_value: Any,
        metadata: dict[str, Any],
        metadata_key: str,
    ) -> Any:
        """Resolve a field from explicit input or request metadata."""

        if explicit_value is not None:
            if isinstance(explicit_value, str) and not explicit_value.strip():
                pass
            else:
                return explicit_value
        value = metadata.get(metadata_key)
        if value is not None:
            if isinstance(value, str) and not value.strip():
                return None
            return value
        return None


class SyntheticEvalWorkflowService:
    """Shared generation, review, and promotion workflow for synthetic eval data."""

    def __init__(
        self,
        *,
        db: EvaluationsDatabase,
        user_id: str | None = None,
        repository: SyntheticEvalRepository | None = None,
        generation_service: SyntheticEvalGenerationService | None = None,
    ) -> None:
        self.db = db
        self.user_id = (user_id or "").strip()
        self.repository = repository or SyntheticEvalRepository(db)
        self.generation_service = generation_service or SyntheticEvalGenerationService(
            repository=self.repository
        )

    def generate_draft_batch(self, **kwargs: Any) -> SyntheticEvalGenerationResult:
        """Proxy through to the shared generation service."""

        if "created_by" not in kwargs:
            kwargs["created_by"] = self.user_id or None
        return self.generation_service.generate_draft_batch(**kwargs)

    def list_queue(
        self,
        *,
        recipe_kind: str | None = None,
        review_state: str | None = None,
        source_kind: str | None = None,
        generation_batch_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Return a filtered synthetic review queue."""

        rows = self.repository.list_draft_samples(
            recipe_kind=recipe_kind,
            review_state=review_state,
            source_kind=source_kind,
            generation_batch_id=generation_batch_id,
            limit=limit,
            offset=offset,
        )
        total = self.repository.count_draft_samples(
            recipe_kind=recipe_kind,
            review_state=review_state,
            source_kind=source_kind,
            generation_batch_id=generation_batch_id,
        )
        return {
            "data": rows,
            "total": total,
        }

    def review_sample(
        self,
        sample_id: str,
        *,
        action: str,
        reviewer_id: str | None = None,
        notes: str | None = None,
        action_payload: dict[str, Any] | None = None,
        resulting_review_state: str | None = None,
    ) -> dict[str, Any]:
        """Apply a review action to a draft sample."""

        return self.repository.record_review_action(
            sample_id=sample_id,
            action=action,
            reviewer_id=reviewer_id or self.user_id or None,
            notes=notes,
            action_payload=action_payload,
            resulting_review_state=resulting_review_state,
        )

    def promote_samples(
        self,
        *,
        sample_ids: list[str],
        dataset_name: str,
        dataset_description: str | None = None,
        dataset_metadata: dict[str, Any] | None = None,
        promoted_by: str | None = None,
        promotion_reason: str | None = None,
    ) -> SyntheticEvalPromotionResult:
        """Promote approved review items into a normal dataset."""

        if not sample_ids:
            raise ValueError("sample_ids must contain at least one sample.")

        samples = self.repository.get_draft_samples(sample_ids)
        recipe_kinds = {str(sample.get("recipe_kind") or "").strip() for sample in samples}
        if "" in recipe_kinds:
            raise ValueError("all promoted samples must include a recipe_kind")
        if len(recipe_kinds) != 1:
            raise ValueError("all promoted samples must share the same recipe_kind")

        promoted_samples = [self._to_dataset_sample(sample) for sample in samples]
        metadata = dict(dataset_metadata or {})
        metadata.setdefault(
            "synthetic_eval",
            {
                "recipe_kind": next(iter(recipe_kinds)),
                "sample_ids": list(sample_ids),
                "promotion_state": "promoted",
            },
        )

        owner = promoted_by or self.user_id or None
        dataset_id = self.db.create_dataset(
            name=dataset_name,
            description=dataset_description,
            samples=promoted_samples,
            created_by=owner,
            metadata=metadata,
        )
        dataset_row = self.db.get_dataset(dataset_id, created_by=owner)
        if not dataset_row:
            raise ValueError(f"Promoted dataset '{dataset_id}' could not be loaded.")
        created_value = dataset_row.get("created") or dataset_row.get("created_at") or "unknown"
        dataset_snapshot_ref = build_dataset_snapshot_ref(dataset_id, created_value)

        promotion_ids: list[str] = []
        for sample_id in sample_ids:
            promotion = self.repository.record_promotion(
                sample_id=sample_id,
                dataset_id=dataset_id,
                dataset_snapshot_ref=dataset_snapshot_ref,
                promoted_by=owner,
                promotion_reason=promotion_reason,
                promotion_metadata={"dataset_name": dataset_name},
            )
            promotion_ids.append(str(promotion.get("promotion_id") or ""))

        return SyntheticEvalPromotionResult(
            dataset_id=dataset_id,
            dataset_snapshot_ref=dataset_snapshot_ref,
            promotion_ids=promotion_ids,
            sample_count=len(promoted_samples),
        )

    def _to_dataset_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        payload = dict(sample.get("sample_payload") or {})
        metadata = dict(sample.get("sample_metadata") or {})
        synthetic_eval_metadata = {
            "sample_id": sample.get("sample_id"),
            "recipe_kind": sample.get("recipe_kind"),
            "provenance": sample.get("provenance"),
            "review_state": sample.get("review_state"),
            "promotion_state": "promoted",
        }

        promoted_sample: dict[str, Any] = dict(payload)
        promoted_sample.setdefault("sample_id", sample.get("sample_id"))

        if sample.get("recipe_kind") == "rag_retrieval_tuning":
            targets: dict[str, Any] = {}
            for key in (
                "relevant_media_ids",
                "relevant_note_ids",
                "relevant_chunk_ids",
                "relevant_spans",
            ):
                if key in promoted_sample:
                    targets[key] = promoted_sample.pop(key)
            if targets:
                promoted_sample["targets"] = targets

        sample_level_metadata = dict(promoted_sample.get("metadata") or {})
        sample_level_metadata.update({key: value for key, value in metadata.items() if value is not None})
        sample_level_metadata["synthetic_eval"] = synthetic_eval_metadata
        promoted_sample["metadata"] = sample_level_metadata
        return promoted_sample


def get_synthetic_eval_service_for_user(user_id: str | int | None) -> SyntheticEvalWorkflowService:
    """Build a synthetic-eval workflow service bound to the appropriate evaluations DB."""

    db_path = os.getenv("EVALUATIONS_TEST_DB_PATH")
    if not db_path:
        db_path = str(DatabasePaths.get_evaluations_db_path(user_id))
    db = EvaluationsDatabase(db_path)
    return SyntheticEvalWorkflowService(
        db=db,
        user_id=str(user_id) if user_id is not None else None,
    )


__all__ = [
    "SyntheticEvalGenerationResult",
    "SyntheticEvalGenerationService",
    "SyntheticEvalPromotionResult",
    "SyntheticEvalWorkflowService",
    "get_synthetic_eval_service_for_user",
]
