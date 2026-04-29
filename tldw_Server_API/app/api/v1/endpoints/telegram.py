from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    RequireRole,
    check_rate_limit,
    get_auth_principal,
)
from tldw_Server_API.app.api.v1.endpoints.telegram_support import (
    telegram_admin_get_bot_impl,
    telegram_admin_list_linked_actors_impl,
    telegram_admin_put_bot_impl,
    telegram_admin_revoke_linked_actor_impl,
    telegram_admin_start_link_impl,
    telegram_webhook_impl,
)
from tldw_Server_API.app.api.v1.schemas.telegram_schemas import (
    TelegramBotConfigResponse,
    TelegramBotConfigUpdate,
    TelegramLinkedActorListResponse,
    TelegramLinkedActorRevokeResponse,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.services.telegram_execution_identity_service import (
    get_telegram_execution_identity_service,
)

router = APIRouter(prefix="/telegram", tags=["telegram"])


@router.put("/admin/bot", dependencies=[Depends(RequireRole("admin"))], response_model=TelegramBotConfigResponse)
async def telegram_admin_put_bot(
    request: Request,
    payload: TelegramBotConfigUpdate,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> TelegramBotConfigResponse:
    return await telegram_admin_put_bot_impl(principal=principal, payload=payload, request=request)


@router.get("/admin/bot", dependencies=[Depends(RequireRole("admin"))], response_model=TelegramBotConfigResponse)
async def telegram_admin_get_bot(
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> TelegramBotConfigResponse:
    return await telegram_admin_get_bot_impl(principal=principal, request=request)


@router.post("/admin/link/start", dependencies=[Depends(RequireRole("admin"))])
async def telegram_admin_start_link(
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
):
    return await telegram_admin_start_link_impl(principal=principal, request=request)


@router.get("/admin/links", dependencies=[Depends(RequireRole("admin"))], response_model=TelegramLinkedActorListResponse)
async def telegram_admin_list_links(
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> TelegramLinkedActorListResponse:
    return await telegram_admin_list_linked_actors_impl(principal=principal, request=request)


@router.delete(
    "/admin/links/{link_id}",
    dependencies=[Depends(RequireRole("admin"))],
    response_model=TelegramLinkedActorRevokeResponse,
)
async def telegram_admin_revoke_link(
    link_id: int,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> TelegramLinkedActorRevokeResponse:
    return await telegram_admin_revoke_linked_actor_impl(link_id=link_id, principal=principal, request=request)


@router.post("/webhook", dependencies=[Depends(check_rate_limit)])
async def telegram_webhook(request: Request):
    return await telegram_webhook_impl(
        request=request,
        get_execution_identity_service=get_telegram_execution_identity_service,
    )
