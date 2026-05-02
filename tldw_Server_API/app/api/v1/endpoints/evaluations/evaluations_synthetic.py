"""Shared synthetic evaluation workflow endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_auth import (
    get_eval_request_user,
    require_eval_permissions,
    sanitize_error_message,
    verify_api_key,
)
from tldw_Server_API.app.api.v1.schemas.synthetic_eval_schemas import (
    SyntheticEvalGenerationRequest,
    SyntheticEvalGenerationResponse,
    SyntheticEvalPromotionRequest,
    SyntheticEvalPromotionResponse,
    SyntheticEvalQueueResponse,
    SyntheticEvalReviewActionRecord,
    SyntheticEvalReviewRequest,
)
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User
from tldw_Server_API.app.core.AuthNZ.permissions import EVALS_MANAGE, EVALS_READ
from tldw_Server_API.app.core.Evaluations.synthetic_eval_service import (
    get_synthetic_eval_service_for_user,
)

synthetic_router = APIRouter()


def _service_for_user(current_user: User):
    stable_user_id = getattr(current_user, "id_str", None) or str(current_user.id)
    return get_synthetic_eval_service_for_user(stable_user_id)


@synthetic_router.post(
    "/synthetic/drafts/generate",
    response_model=SyntheticEvalGenerationResponse,
    dependencies=[Depends(require_eval_permissions(EVALS_MANAGE))],
)
async def generate_synthetic_drafts(
    payload: SyntheticEvalGenerationRequest,
    user_ctx: str = Depends(verify_api_key),
    current_user: User = Depends(get_eval_request_user),
):
    del user_ctx
    service = _service_for_user(current_user)
    try:
        result = service.generate_draft_batch(
            recipe_kind=payload.recipe_kind,
            corpus_scope=payload.corpus_scope,
            generation_metadata=payload.generation_metadata,
            context_snapshot_ref=payload.context_snapshot_ref,
            retrieval_baseline_ref=payload.retrieval_baseline_ref,
            reference_answer=payload.reference_answer,
            real_examples=payload.real_examples,
            seed_examples=payload.seed_examples,
            target_sample_count=payload.target_sample_count,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=sanitize_error_message(exc, "generating synthetic drafts"),
        ) from exc

    corpus_scope = result.corpus_scope
    return SyntheticEvalGenerationResponse(
        generation_batch_id=result.generation_batch_id,
        samples=result.samples,
        source_breakdown=result.source_breakdown,
        coverage=result.coverage,
        missing_coverage=result.missing_coverage,
        corpus_scope=corpus_scope.to_metadata_dict() if corpus_scope else {},
    )


@synthetic_router.get(
    "/synthetic/queue",
    response_model=SyntheticEvalQueueResponse,
    dependencies=[Depends(require_eval_permissions(EVALS_READ))],
)
async def list_synthetic_queue(
    recipe_kind: str | None = Query(default=None),
    review_state: str | None = Query(default=None),
    source_kind: str | None = Query(default=None),
    generation_batch_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    user_ctx: str = Depends(verify_api_key),
    current_user: User = Depends(get_eval_request_user),
):
    del user_ctx
    service = _service_for_user(current_user)
    queue = service.list_queue(
        recipe_kind=recipe_kind,
        review_state=review_state,
        source_kind=source_kind,
        generation_batch_id=generation_batch_id,
        limit=limit,
        offset=offset,
    )
    return SyntheticEvalQueueResponse.model_validate(
        {
            **queue,
            "pagination": build_offset_pagination_meta(
                total=queue["total"],
                limit=limit,
                offset=offset,
                count=len(queue["data"]),
            ),
        }
    )


@synthetic_router.post(
    "/synthetic/queue/{sample_id}/review",
    response_model=SyntheticEvalReviewActionRecord,
    dependencies=[Depends(require_eval_permissions(EVALS_MANAGE))],
)
async def review_synthetic_sample(
    sample_id: str,
    payload: SyntheticEvalReviewRequest,
    user_ctx: str = Depends(verify_api_key),
    current_user: User = Depends(get_eval_request_user),
):
    del user_ctx
    service = _service_for_user(current_user)
    try:
        record = service.review_sample(
            sample_id,
            action=payload.action.value,
            reviewer_id=getattr(current_user, "id_str", None) or str(current_user.id),
            notes=payload.notes,
            action_payload=payload.action_payload,
            resulting_review_state=(
                payload.resulting_review_state.value
                if payload.resulting_review_state is not None
                else None
            ),
        )
        return SyntheticEvalReviewActionRecord.model_validate(record)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=sanitize_error_message(exc, "reviewing synthetic draft"),
        ) from exc


@synthetic_router.post(
    "/synthetic/promotions",
    response_model=SyntheticEvalPromotionResponse,
    dependencies=[Depends(require_eval_permissions(EVALS_MANAGE))],
)
async def promote_synthetic_samples(
    payload: SyntheticEvalPromotionRequest,
    user_ctx: str = Depends(verify_api_key),
    current_user: User = Depends(get_eval_request_user),
):
    del user_ctx
    service = _service_for_user(current_user)
    try:
        result = service.promote_samples(
            sample_ids=payload.sample_ids,
            dataset_name=payload.dataset_name,
            dataset_description=payload.dataset_description,
            dataset_metadata=payload.dataset_metadata,
            promoted_by=getattr(current_user, "id_str", None) or str(current_user.id),
            promotion_reason=payload.promotion_reason,
        )
        return SyntheticEvalPromotionResponse(
            dataset_id=result.dataset_id,
            dataset_snapshot_ref=result.dataset_snapshot_ref,
            promotion_ids=result.promotion_ids,
            sample_count=result.sample_count,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=sanitize_error_message(exc, "promoting synthetic drafts"),
        ) from exc
