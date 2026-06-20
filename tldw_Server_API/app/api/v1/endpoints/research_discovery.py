"""API routes for standalone research discovery."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit, rbac_rate_limit
from tldw_Server_API.app.api.v1.schemas.research_discovery_schemas import (
    ResearchDiscoverySearchRequest,
    ResearchDiscoverySearchResponse,
    ResearchSourceListResponse,
    ResearchSourceResponse,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.Research.discovery import (
    ResearchDiscoveryService,
    default_source_catalog,
)
from tldw_Server_API.app.core.exceptions import (
    ResearchDiscoveryBadRequestError,
    ResearchDiscoveryTimeoutError,
    ResearchDiscoveryUpstreamError,
    ResearchDiscoveryValidationError,
)

router = APIRouter(tags=["research-discovery"])


def get_research_discovery_service() -> ResearchDiscoveryService:
    """Return the default research discovery service."""
    return ResearchDiscoveryService()


@router.get(
    "/sources",
    response_model=ResearchSourceListResponse,
    summary="List research discovery sources",
    dependencies=[
        Depends(check_rate_limit),
        Depends(rbac_rate_limit("research.discovery.sources")),
    ],
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
    dependencies=[
        Depends(check_rate_limit),
        Depends(rbac_rate_limit("research.discovery.search")),
    ],
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
    except ResearchDiscoveryValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.public_detail) from None
    except ResearchDiscoveryBadRequestError as exc:
        raise HTTPException(status_code=400, detail=exc.public_detail) from None
    except ResearchDiscoveryTimeoutError as exc:
        raise HTTPException(status_code=504, detail=exc.public_detail) from None
    except ResearchDiscoveryUpstreamError as exc:
        raise HTTPException(status_code=502, detail=exc.public_detail) from None
    return ResearchDiscoverySearchResponse.model_validate(response)
