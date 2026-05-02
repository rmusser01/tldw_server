"""Repository helpers for synthetic evaluation draft persistence."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from tldw_Server_API.app.api.v1.schemas.synthetic_eval_schemas import (
    SyntheticEvalDraftSampleCreate,
    SyntheticEvalPromotionCreate,
    SyntheticEvalProvenance,
    SyntheticEvalReviewActionCreate,
    SyntheticEvalReviewActionType,
    SyntheticEvalReviewState,
)
from tldw_Server_API.app.core.DB_Management.Evaluations_DB import EvaluationsDatabase


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(sep=" ")


def _json_dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _json_load(value: Any) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, (dict, list, int, float, bool)):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return value


def _row_to_dict(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    return dict(row)


def _is_synthetic_origin(provenance: str | SyntheticEvalProvenance) -> bool:
    value = provenance.value if isinstance(provenance, SyntheticEvalProvenance) else str(provenance)
    return value.startswith("synthetic_")


def _draft_sample_filter_clause(
    *,
    recipe_kind: str | None = None,
    review_state: str | SyntheticEvalReviewState | None = None,
    source_kind: str | None = None,
    generation_batch_id: str | None = None,
) -> tuple[str, list[Any]]:
    query = " WHERE 1=1"
    params: list[Any] = []
    if recipe_kind:
        query += " AND recipe_kind = ?"
        params.append(recipe_kind)
    if review_state:
        normalized_state = (
            review_state.value
            if isinstance(review_state, SyntheticEvalReviewState)
            else str(review_state)
        )
        query += " AND review_state = ?"
        params.append(normalized_state)
    if source_kind:
        query += " AND source_kind = ?"
        params.append(source_kind)
    if generation_batch_id:
        query += " AND CAST(sample_metadata_json AS TEXT) LIKE ?"
        params.append(f'%\"generation_batch_id\":{_json_dumps(generation_batch_id)}%')
    return query, params


class SyntheticEvalRepository:
    """Persistence helper for synthetic evaluation draft samples."""

    def __init__(self, db: EvaluationsDatabase) -> None:
        self._db = db

    def create_draft_sample(
        self,
        *,
        sample_id: str,
        recipe_kind: str,
        sample_payload: dict[str, Any],
        provenance: str | SyntheticEvalProvenance,
        review_state: str | SyntheticEvalReviewState = SyntheticEvalReviewState.DRAFT,
        sample_metadata: dict[str, Any] | None = None,
        source_kind: str | None = None,
        created_by: str | None = None,
    ) -> dict[str, Any]:
        """Insert or replace a synthetic draft sample."""

        create_payload = SyntheticEvalDraftSampleCreate(
            sample_id=sample_id,
            recipe_kind=recipe_kind,
            provenance=SyntheticEvalProvenance(provenance),
            review_state=SyntheticEvalReviewState(review_state),
            sample_payload=sample_payload,
            sample_metadata=sample_metadata or {},
            source_kind=source_kind,
            created_by=created_by,
        )
        now = _utcnow_iso()
        with self._db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO synthetic_eval_draft_samples (
                    sample_id,
                    recipe_kind,
                    provenance,
                    review_state,
                    sample_payload_json,
                    sample_metadata_json,
                    source_kind,
                    created_by,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sample_id) DO UPDATE SET
                    recipe_kind = excluded.recipe_kind,
                    provenance = excluded.provenance,
                    review_state = excluded.review_state,
                    sample_payload_json = excluded.sample_payload_json,
                    sample_metadata_json = excluded.sample_metadata_json,
                    source_kind = excluded.source_kind,
                    created_by = excluded.created_by,
                    updated_at = excluded.updated_at
                """,
                (
                    create_payload.sample_id,
                    create_payload.recipe_kind,
                    create_payload.provenance.value,
                    create_payload.review_state.value,
                    _json_dumps(create_payload.sample_payload),
                    _json_dumps(create_payload.sample_metadata),
                    create_payload.source_kind,
                    create_payload.created_by,
                    now,
                    now,
                ),
            )
            conn.commit()
        return self.get_draft_sample(sample_id) or {}

    def get_draft_sample(self, sample_id: str) -> dict[str, Any] | None:
        """Fetch a persisted synthetic draft sample."""

        with self._db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM synthetic_eval_draft_samples WHERE sample_id = ?",
                (sample_id,),
            )
            row = cursor.fetchone()
        if not row:
            return None
        payload = _row_to_dict(row)
        payload["sample_payload"] = _json_load(payload.pop("sample_payload_json", None)) or {}
        payload["sample_metadata"] = _json_load(payload.pop("sample_metadata_json", None)) or {}
        payload.pop("review_summary_json", None)
        return payload

    def _filtered_draft_samples(
        self,
        *,
        recipe_kind: str | None = None,
        review_state: str | SyntheticEvalReviewState | None = None,
        source_kind: str | None = None,
        generation_batch_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return matching synthetic draft samples."""

        where_clause, params = _draft_sample_filter_clause(
            recipe_kind=recipe_kind,
            review_state=review_state,
            source_kind=source_kind,
            generation_batch_id=generation_batch_id,
        )
        # where_clause is assembled from fixed predicates and bound parameters.
        query = "SELECT * FROM synthetic_eval_draft_samples" + where_clause  # nosec B608
        query += " ORDER BY created_at DESC, sample_id DESC"
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([max(1, int(limit)), max(0, int(offset))])

        with self._db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            payload = _row_to_dict(row)
            payload["sample_payload"] = _json_load(payload.pop("sample_payload_json", None)) or {}
            payload["sample_metadata"] = _json_load(payload.pop("sample_metadata_json", None)) or {}
            payload.pop("review_summary_json", None)
            results.append(payload)

        return results

    def list_draft_samples(
        self,
        *,
        recipe_kind: str | None = None,
        review_state: str | SyntheticEvalReviewState | None = None,
        source_kind: str | None = None,
        generation_batch_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List persisted synthetic draft samples with simple filters."""

        results = self._filtered_draft_samples(
            recipe_kind=recipe_kind,
            review_state=review_state,
            source_kind=source_kind,
            generation_batch_id=generation_batch_id,
            limit=limit,
            offset=offset,
        )
        return results

    def count_draft_samples(
        self,
        *,
        recipe_kind: str | None = None,
        review_state: str | SyntheticEvalReviewState | None = None,
        source_kind: str | None = None,
        generation_batch_id: str | None = None,
    ) -> int:
        """Count matching synthetic draft samples before pagination."""

        where_clause, params = _draft_sample_filter_clause(
            recipe_kind=recipe_kind,
            review_state=review_state,
            source_kind=source_kind,
            generation_batch_id=generation_batch_id,
        )
        # where_clause is assembled from fixed predicates and bound parameters.
        query = "SELECT COUNT(*) AS total FROM synthetic_eval_draft_samples" + where_clause  # nosec B608

        with self._db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
        if not row:
            return 0
        try:
            row_dict = _row_to_dict(row)
        except TypeError:
            row_dict = {}
        value = row_dict.get("total") if row_dict else row[0]
        return int(value or 0)

    def get_draft_samples(self, sample_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch multiple draft samples in caller-provided order."""

        ordered: list[dict[str, Any]] = []
        for sample_id in sample_ids:
            ordered.append(self.require_draft_sample(sample_id))
        return ordered

    def require_draft_sample(self, sample_id: str) -> dict[str, Any]:
        """Return a sample or raise when it does not exist."""

        sample = self.get_draft_sample(sample_id)
        if sample is None:
            raise ValueError("sample does not exist")
        return sample

    def _sample_has_edit_history(self, sample_id: str) -> bool:
        """Return True when the sample has already been edited by a reviewer."""

        with self._db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT 1
                FROM synthetic_eval_review_actions
                WHERE sample_id = ? AND action IN ('edit', 'edit_and_approve')
                LIMIT 1
                """,
                (sample_id,),
            )
            return cursor.fetchone() is not None

    def record_review_action(
        self,
        *,
        sample_id: str,
        action: str | SyntheticEvalReviewActionType,
        reviewer_id: str | None = None,
        notes: str | None = None,
        action_payload: dict[str, Any] | None = None,
        resulting_review_state: str | SyntheticEvalReviewState | None = None,
    ) -> dict[str, Any]:
        """Append a review action and update the sample's current review state."""

        normalized_action = SyntheticEvalReviewActionType(action)
        sample = self.require_draft_sample(sample_id)
        if resulting_review_state is None:
            if normalized_action in {
                SyntheticEvalReviewActionType.EDIT_AND_APPROVE,
                SyntheticEvalReviewActionType.APPROVE,
            }:
                resulting_review_state = SyntheticEvalReviewState.APPROVED
            elif normalized_action is SyntheticEvalReviewActionType.REJECT:
                resulting_review_state = SyntheticEvalReviewState.REJECTED
            elif normalized_action is SyntheticEvalReviewActionType.REQUEST_CHANGES:
                resulting_review_state = SyntheticEvalReviewState.IN_REVIEW
            elif normalized_action is SyntheticEvalReviewActionType.EDIT:
                resulting_review_state = SyntheticEvalReviewState.EDITED
            else:  # pragma: no cover - exhaustive future guard
                resulting_review_state = SyntheticEvalReviewState.IN_REVIEW

        now = _utcnow_iso()
        action_id = f"synth_action_{uuid.uuid4().hex[:12]}"
        action_record = SyntheticEvalReviewActionCreate(
            sample_id=sample_id,
            action=normalized_action,
            reviewer_id=reviewer_id,
            notes=notes,
            action_payload=action_payload or {},
            resulting_review_state=SyntheticEvalReviewState(resulting_review_state),
        )

        with self._db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO synthetic_eval_review_actions (
                    action_id,
                    sample_id,
                    action,
                    reviewer_id,
                    notes,
                    action_payload_json,
                    resulting_review_state,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    action_id,
                    action_record.sample_id,
                    action_record.action.value,
                    action_record.reviewer_id,
                    action_record.notes,
                    _json_dumps(action_record.action_payload),
                    action_record.resulting_review_state.value if action_record.resulting_review_state else None,
                    now,
                ),
            )
            cursor.execute(
                """
                UPDATE synthetic_eval_draft_samples
                SET review_state = ?, updated_at = ?
                WHERE sample_id = ?
                """,
                (
                    action_record.resulting_review_state.value if action_record.resulting_review_state else SyntheticEvalReviewState.IN_REVIEW.value,
                    now,
                    sample_id,
                ),
            )
            if normalized_action in {
                SyntheticEvalReviewActionType.EDIT_AND_APPROVE,
                SyntheticEvalReviewActionType.APPROVE,
            } and self._sample_has_edit_history(sample_id) and _is_synthetic_origin(sample.get("provenance", "")):
                cursor.execute(
                    """
                    UPDATE synthetic_eval_draft_samples
                    SET provenance = ?, updated_at = ?
                    WHERE sample_id = ?
                    """,
                    (
                        SyntheticEvalProvenance.SYNTHETIC_HUMAN_EDITED.value,
                        now,
                        sample_id,
                    ),
                )
            conn.commit()

        return self.get_review_action(action_id) or {}

    def get_review_action(self, action_id: str) -> dict[str, Any] | None:
        """Fetch a single review action history row."""

        with self._db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM synthetic_eval_review_actions WHERE action_id = ?",
                (action_id,),
            )
            row = cursor.fetchone()
        if not row:
            return None
        payload = _row_to_dict(row)
        payload["action_payload"] = _json_load(payload.pop("action_payload_json", None)) or {}
        return payload

    def list_review_actions(self, sample_id: str) -> list[dict[str, Any]]:
        """Return review history in chronological order."""

        with self._db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM synthetic_eval_review_actions
                WHERE sample_id = ?
                ORDER BY created_at ASC, action_id ASC
                """,
                (sample_id,),
            )
            rows = cursor.fetchall()
        history: list[dict[str, Any]] = []
        for row in rows:
            payload = _row_to_dict(row)
            payload["action_payload"] = _json_load(payload.pop("action_payload_json", None)) or {}
            history.append(payload)
        return history

    def record_promotion(
        self,
        *,
        sample_id: str,
        dataset_id: str | None = None,
        dataset_snapshot_ref: str | None = None,
        promoted_by: str | None = None,
        promotion_reason: str | None = None,
        promotion_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist a dataset promotion record for an approved draft sample."""

        sample = self.require_draft_sample(sample_id)
        if sample.get("review_state") != SyntheticEvalReviewState.APPROVED.value:
            raise ValueError("sample must be approved before promotion")

        create_payload = SyntheticEvalPromotionCreate(
            sample_id=sample_id,
            dataset_id=dataset_id,
            dataset_snapshot_ref=dataset_snapshot_ref,
            promoted_by=promoted_by,
            promotion_reason=promotion_reason,
            promotion_metadata=promotion_metadata or {},
        )
        promotion_id = f"synth_promo_{uuid.uuid4().hex[:12]}"
        now = _utcnow_iso()

        with self._db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO synthetic_eval_promotions (
                    promotion_id,
                    sample_id,
                    dataset_id,
                    dataset_snapshot_ref,
                    promoted_by,
                    promotion_reason,
                    promotion_metadata_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    promotion_id,
                    create_payload.sample_id,
                    create_payload.dataset_id,
                    create_payload.dataset_snapshot_ref,
                    create_payload.promoted_by,
                    create_payload.promotion_reason,
                    _json_dumps(create_payload.promotion_metadata),
                    now,
                ),
            )
            cursor.execute(
                """
                UPDATE synthetic_eval_draft_samples
                SET review_state = ?, updated_at = ?
                WHERE sample_id = ?
                """,
                (SyntheticEvalReviewState.APPROVED.value, now, sample_id),
            )
            conn.commit()

        return self.get_promotion(promotion_id) or {}

    def get_promotion(self, promotion_id: str) -> dict[str, Any] | None:
        """Fetch a single promotion record."""

        with self._db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM synthetic_eval_promotions WHERE promotion_id = ?",
                (promotion_id,),
            )
            row = cursor.fetchone()
        if not row:
            return None
        payload = _row_to_dict(row)
        payload["promotion_metadata"] = _json_load(payload.pop("promotion_metadata_json", None)) or {}
        return payload


__all__ = ["SyntheticEvalRepository"]
