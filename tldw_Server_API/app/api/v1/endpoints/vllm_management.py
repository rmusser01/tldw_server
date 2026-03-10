from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit, require_roles
from tldw_Server_API.app.api.v1.schemas.vllm_management import (
    VLLMDefaultRouteRequest,
    VLLMDefaultRouteResponse,
    VLLMDeleteResponse,
    VLLMInstanceCreateRequest,
    VLLMInstanceEnvelope,
    VLLMInstanceListResponse,
    VLLMInstanceRecordResponse,
    VLLMInstanceUpdateRequest,
)
from tldw_Server_API.app.core.VLLM_Management import (
    VLLMInstanceRepository,
    get_default_vllm_instance_repository,
)

router = APIRouter()


def _resolve_vllm_repository() -> VLLMInstanceRepository:
    return get_default_vllm_instance_repository()


def _serialize_instance(record: object) -> VLLMInstanceRecordResponse:
    return VLLMInstanceRecordResponse.model_validate(record)


def _sync_default_route(
    *,
    repository: VLLMInstanceRepository,
    instance_id: str,
    routing_policy: dict[str, object] | None,
) -> None:
    if not isinstance(routing_policy, dict) or "is_default" not in routing_policy:
        return
    if bool(routing_policy.get("is_default")):
        repository.set_default_instance(instance_id)
        return
    if repository.get_default_instance_id() == instance_id:
        repository.set_default_instance(None)


@router.post(
    "/llm/providers/vllm/instances",
    response_model=VLLMInstanceEnvelope,
    status_code=201,
    dependencies=[Depends(check_rate_limit), Depends(require_roles("admin"))],
)
async def create_vllm_instance(
    payload: VLLMInstanceCreateRequest,
    repository: VLLMInstanceRepository = Depends(_resolve_vllm_repository),
) -> VLLMInstanceEnvelope:
    record = repository.create_instance(payload.to_domain())
    _sync_default_route(repository=repository, instance_id=record.instance_id, routing_policy=payload.routing_policy)
    return VLLMInstanceEnvelope(instance=_serialize_instance(record))


@router.get(
    "/llm/providers/vllm/instances",
    response_model=VLLMInstanceListResponse,
    dependencies=[Depends(check_rate_limit), Depends(require_roles("admin"))],
)
async def list_vllm_instances(
    repository: VLLMInstanceRepository = Depends(_resolve_vllm_repository),
) -> VLLMInstanceListResponse:
    records = [_serialize_instance(record) for record in repository.list_instances()]
    return VLLMInstanceListResponse(
        default_instance_id=repository.get_default_instance_id(),
        instances=records,
    )


@router.get(
    "/llm/providers/vllm/instances/{instance_id}",
    response_model=VLLMInstanceEnvelope,
    dependencies=[Depends(check_rate_limit), Depends(require_roles("admin"))],
)
async def get_vllm_instance(
    instance_id: str,
    repository: VLLMInstanceRepository = Depends(_resolve_vllm_repository),
) -> VLLMInstanceEnvelope:
    record = repository.get_instance(instance_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Managed vLLM instance '{instance_id}' was not found")
    return VLLMInstanceEnvelope(instance=_serialize_instance(record))


@router.patch(
    "/llm/providers/vllm/instances/{instance_id}",
    response_model=VLLMInstanceEnvelope,
    dependencies=[Depends(check_rate_limit), Depends(require_roles("admin"))],
)
async def update_vllm_instance(
    instance_id: str,
    payload: VLLMInstanceUpdateRequest,
    repository: VLLMInstanceRepository = Depends(_resolve_vllm_repository),
) -> VLLMInstanceEnvelope:
    patch = payload.to_patch()
    try:
        record = repository.update_instance(instance_id, patch)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    _sync_default_route(
        repository=repository,
        instance_id=instance_id,
        routing_policy=patch.get("routing_policy") if isinstance(patch.get("routing_policy"), dict) else None,
    )
    return VLLMInstanceEnvelope(instance=_serialize_instance(record))


@router.delete(
    "/llm/providers/vllm/instances/{instance_id}",
    response_model=VLLMDeleteResponse,
    dependencies=[Depends(check_rate_limit), Depends(require_roles("admin"))],
)
async def delete_vllm_instance(
    instance_id: str,
    force: bool = Query(default=False),
    repository: VLLMInstanceRepository = Depends(_resolve_vllm_repository),
) -> VLLMDeleteResponse:
    record = repository.get_instance(instance_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Managed vLLM instance '{instance_id}' was not found")
    if not force and (record.desired_state != "stopped" or record.observed_state != "stopped"):
        raise HTTPException(
            status_code=409,
            detail="Managed vLLM instance must be stopped before deletion unless force=true",
        )
    deleted = repository.delete_instance(instance_id)
    return VLLMDeleteResponse(deleted=deleted, instance_id=instance_id)


@router.post(
    "/llm/providers/vllm/default",
    response_model=VLLMDefaultRouteResponse,
    dependencies=[Depends(check_rate_limit), Depends(require_roles("admin"))],
)
async def set_default_vllm_instance(
    payload: VLLMDefaultRouteRequest,
    repository: VLLMInstanceRepository = Depends(_resolve_vllm_repository),
) -> VLLMDefaultRouteResponse:
    try:
        repository.set_default_instance(payload.instance_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return VLLMDefaultRouteResponse(default_instance_id=repository.get_default_instance_id())
