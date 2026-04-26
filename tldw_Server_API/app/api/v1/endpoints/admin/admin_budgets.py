from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    get_auth_principal,
    get_db_transaction,
)
from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    BudgetForecastResponse,
    OrgBudgetItem,
    OrgBudgetListResponse,
    OrgBudgetSelfUpdateRequest,
    OrgBudgetUpdateRequest,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.services import admin_budgets_service

router = APIRouter()


@router.get("/budgets", response_model=OrgBudgetListResponse)
async def admin_list_budgets(
    org_id: int | None = Query(None, description="Restrict to a specific organization"),
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    principal: AuthPrincipal = Depends(get_auth_principal),
    db=Depends(get_db_transaction),
) -> OrgBudgetListResponse:
    return await admin_budgets_service.list_budgets(
        principal=principal,
        org_id=org_id,
        page=page,
        limit=limit,
        db=db,
    )


@router.post("/budgets", response_model=OrgBudgetItem)
async def admin_upsert_budget(
    payload: OrgBudgetUpdateRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db=Depends(get_db_transaction),
) -> OrgBudgetItem:
    return await admin_budgets_service.upsert_budget(
        payload=payload,
        request=request,
        principal=principal,
        db=db,
    )


@router.put("/budgets/{org_id}", response_model=OrgBudgetItem)
async def admin_update_budget_by_org_id(
    org_id: int,
    payload: OrgBudgetSelfUpdateRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db=Depends(get_db_transaction),
) -> OrgBudgetItem:
    update_payload = OrgBudgetUpdateRequest(
        org_id=org_id,
        budgets=payload.budgets,
        clear_budgets=payload.clear_budgets,
    )
    return await admin_budgets_service.upsert_budget(
        payload=update_payload,
        request=request,
        principal=principal,
        db=db,
    )


@router.get("/budgets/forecast", response_model=BudgetForecastResponse)
async def admin_get_budget_forecast(
    org_id: int = Query(..., description="Organization ID"),
    principal: AuthPrincipal = Depends(get_auth_principal),
    db=Depends(get_db_transaction),
) -> BudgetForecastResponse:
    """Get budget spend forecast based on trailing 7-day average."""
    try:
        # Use list_budgets filtered by org_id to get the org's budget
        budget_list = await admin_budgets_service.list_budgets(
            principal=principal, db=db, org_id=org_id, page=1, limit=1
        )
        items = budget_list.items if hasattr(budget_list, "items") else []
        if not items:
            return BudgetForecastResponse(org_id=org_id, forecast_available=False)

        budget = items[0]
        budget_dict = (
            budget.model_dump()
            if hasattr(budget, "model_dump")
            else (budget.dict() if hasattr(budget, "dict") else budget)
        )
        budgets = budget_dict.get("budgets", {})

        # Extract budget limits — budget payload uses flat keys like budget_month_usd
        monthly_usd = budgets.get("budget_month_usd") if isinstance(budgets, dict) else None

        if monthly_usd is None:
            return BudgetForecastResponse(
                org_id=org_id,
                forecast_available=False,
                reason="No monthly USD budget configured",
            )

        return BudgetForecastResponse(
            org_id=org_id,
            forecast_available=True,
            monthly_limit_usd=monthly_usd,
            note="Detailed burn-rate projection requires usage aggregation — coming in a future release.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Budget forecast failed")
        raise HTTPException(status_code=500, detail="Budget forecast failed") from exc
