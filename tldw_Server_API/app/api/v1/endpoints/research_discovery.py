"""API routes for standalone research discovery."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.schemas.research_discovery_schemas import (
    ResearchDiscoverySearchRequest,
    ResearchDiscoverySearchResponse,
    ResearchSourceListResponse,
    ResearchSourceResponse,
)
from tldw_Server_API.app.core.Research.discovery import (
    ResearchDiscoveryService,
    default_source_catalog,
)

router = APIRouter(tags=["research-discovery"])

_VALIDATION_VALUE_ERROR_PREFIXES = (
    "source_selection_over_cap",
    "research_discovery_fallback_disabled",
    "research_discovery_no_runnable_sources",
    "research_discovery_query_contains_unsafe_url",
    "research_discovery_filters_contain_unsafe_url",
)


def get_research_discovery_service() -> ResearchDiscoveryService:
    """Return the default research discovery service."""
    return ResearchDiscoveryService()


@router.get(
    "/sources",
    response_model=ResearchSourceListResponse,
    summary="List research discovery sources",
)
async def list_research_discovery_sources() -> ResearchSourceListResponse:
    """List the default research discovery source catalog."""
    catalog = default_source_catalog()
    return ResearchSourceListResponse(
        catalog_version=catalog.catalog_version,
        sources=[ResearchSourceResponse.model_validate(source) for source in catalog.list_sources()],
    )


@router.post(
    "/discovery/search",
    response_model=ResearchDiscoverySearchResponse,
    summary="Search configured research discovery sources",
)
async def search_research_discovery(
    payload: ResearchDiscoverySearchRequest,
    current_user: User = Depends(get_request_user),
    service: ResearchDiscoveryService = Depends(get_research_discovery_service),
) -> ResearchDiscoverySearchResponse:
    """Run standalone research discovery through the shared discovery service."""
    try:
        response = await service.search(
            owner_user_id=str(current_user.id),
            query=payload.query,
            source_ids=payload.source_ids,
            categories=payload.categories,
            per_source_limit=payload.per_source_limit,
            total_limit=payload.total_limit,
            fallback_policy=payload.fallback_policy,
            filters=payload.filters,
        )
    except ValueError as exc:
        _raise_value_error_http_exception(exc)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from None
    except RuntimeError as exc:
        _raise_runtime_error_http_exception(exc)
    return ResearchDiscoverySearchResponse.model_validate(response)


def _raise_value_error_http_exception(exc: ValueError) -> None:
    detail = str(exc)
    if _is_validation_value_error(detail):
        raise HTTPException(status_code=422, detail=detail) from None
    raise HTTPException(status_code=400, detail=detail) from None


def _raise_runtime_error_http_exception(exc: RuntimeError) -> None:
    detail = str(exc)
    if detail.startswith("research_discovery_total_timeout"):
        raise HTTPException(status_code=504, detail=detail) from None
    if detail.startswith("research_discovery_all_sources_failed"):
        raise HTTPException(status_code=502, detail=detail) from None
    raise exc


def _is_validation_value_error(detail: str) -> bool:
    if detail.startswith(_VALIDATION_VALUE_ERROR_PREFIXES):
        return True
    return "fallback" in detail or "policy" in detail
