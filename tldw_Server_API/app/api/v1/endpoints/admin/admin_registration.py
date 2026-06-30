from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    get_auth_principal,
    get_db_transaction,
)
from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    RegistrationCodeListResponse,
    RegistrationCodeRequest,
    RegistrationCodeResponse,
    RegistrationSettingsResponse,
    RegistrationSettingsUpdateRequest,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.services import admin_registration_service

router = APIRouter()


def _get_emit_admin_audit_event():
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod

    return admin_mod._emit_admin_audit_event


@router.get("/registration-settings", response_model=RegistrationSettingsResponse)
async def get_registration_settings() -> RegistrationSettingsResponse:
    return await admin_registration_service.get_registration_settings()


@router.post("/registration-settings", response_model=RegistrationSettingsResponse)
async def update_registration_settings(
    payload: RegistrationSettingsUpdateRequest,
) -> RegistrationSettingsResponse:
    return await admin_registration_service.update_registration_settings(payload)


@router.post("/registration-codes", response_model=RegistrationCodeResponse)
async def create_registration_code(
    request: RegistrationCodeRequest,
    http_request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Any = Depends(get_db_transaction),
) -> RegistrationCodeResponse:
    response, audit_info = await admin_registration_service.create_registration_code(
        request=request,
        principal=principal,
        db=db,
    )
    if audit_info:
        try:
            await _get_emit_admin_audit_event()(http_request, principal, **audit_info)
        except Exception:
            logger.debug("Audit emission failed for registration code creation")
    return response


@router.get("/registration-codes", response_model=RegistrationCodeListResponse)
async def list_registration_codes(
    include_expired: bool = Query(False),
    db: Any = Depends(get_db_transaction),
) -> RegistrationCodeListResponse:
    return await admin_registration_service.list_registration_codes(include_expired, db)


@router.delete("/registration-codes/{code_id}")
async def delete_registration_code(
    code_id: int,
    http_request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Any = Depends(get_db_transaction),
) -> dict[str, str]:
    response, audit_info = await admin_registration_service.delete_registration_code(code_id, db)
    if audit_info:
        try:
            await _get_emit_admin_audit_event()(http_request, principal, **audit_info)
        except Exception:
            logger.debug("Audit emission failed for registration code deletion")
    return response
