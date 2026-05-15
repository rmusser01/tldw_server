"""VN platform capabilities endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

from tldw_Server_API.app.api.v1.schemas.vn_capabilities_schemas import VNCapabilitiesResponse
from tldw_Server_API.app.core.VN_Platform.capabilities import build_vn_capabilities

router = APIRouter(prefix="/vn-capabilities", tags=["vn-capabilities"])


@router.get("", response_model=VNCapabilitiesResponse, operation_id="get_vn_capabilities")
def get_vn_capabilities(request: Request) -> VNCapabilitiesResponse:
    """Return route-aware capability metadata for VN clients."""
    return VNCapabilitiesResponse.model_validate(build_vn_capabilities(request.app.routes))
