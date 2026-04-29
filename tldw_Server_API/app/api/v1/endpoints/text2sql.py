"""Standalone Text2SQL API endpoint."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit, get_request_user, rbac_rate_limit, RequirePermission, TokenScopeGuard, User

from tldw_Server_API.app.api.v1.API_Deps.billing_deps import require_within_limit
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.schemas.text2sql_schemas import (
    Text2SQLRequest,
    Text2SQLResponse,
)
from tldw_Server_API.app.core.AuthNZ.permissions import (
    SQL_READ,
    SQL_TARGET_ANY,
)
from tldw_Server_API.app.core.Billing.enforcement import LimitCategory
from tldw_Server_API.app.core.Text2SQL.executor import SqliteReadOnlyExecutor
from tldw_Server_API.app.core.Text2SQL.service import Text2SQLCoreService
from tldw_Server_API.app.core.Text2SQL.source_registry import normalize_source
from tldw_Server_API.app.core.Text2SQL.sql_guard import SqlPolicyViolation

router = APIRouter(prefix="/text2sql", tags=["text2sql"])


class _PassThroughSqlGenerator:
    """Temporary SQL generator: accepts explicit SQL only (fail-closed)."""

    async def generate(self, *, query: str, target_id: str) -> dict[str, str]:
        _ = target_id
        text = str(query).strip()
        if not text:
            raise ValueError("Query must not be empty")
        if not text.lower().startswith(("select", "with")):
            raise ValueError("sql_generation_failed: provide SQL beginning with SELECT or WITH")
        return {"sql": text}


def _resolve_internal_target(target_id: str, media_db: Any) -> tuple[str, str]:
    try:
        normalized = normalize_source(target_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "invalid_source", "message": str(exc)},
        ) from exc

    if normalized != "media_db":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "invalid_source",
                "message": f"Unsupported target_id '{target_id}' for standalone text2sql endpoint",
            },
        )

    return normalized, str(media_db.db_path)


def _shape_rows(columns: list[str], rows: list[Any]) -> list[dict[str, Any]]:
    shaped: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            shaped.append({str(key): value for key, value in row.items()})
            continue
        if isinstance(row, (list, tuple)):
            record: dict[str, Any] = {}
            for index, column in enumerate(columns):
                record[str(column)] = row[index] if index < len(row) else None
            shaped.append(record)
            continue
        shaped.append({"value": row})
    return shaped


def _connector_acl_allows(current_user: User, target_id: str) -> bool:
    """Fail-closed ACL hook for SQL connector targets."""
    permissions = {str(p) for p in (getattr(current_user, "permissions", []) or [])}
    if SQL_TARGET_ANY in permissions or f"sql.target:{target_id}" in permissions:
        return True
    return False


@router.post(
    "/query",
    response_model=Text2SQLResponse,
    dependencies=[
        Depends(check_rate_limit),
        Depends(rbac_rate_limit("text2sql.query")),
        Depends(RequirePermission(SQL_READ)),
        Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="text2sql.query", count_as="call")),
        Depends(require_within_limit(LimitCategory.RAG_QUERIES_DAY, 1)),
    ],
)
async def query_text2sql(
    request: Text2SQLRequest,
    current_user: User = Depends(get_request_user),
    media_db: Any = Depends(get_media_db_for_user),
) -> Text2SQLResponse:
    """Execute a read-only SQL query through Text2SQL policy guardrails."""
    target_id, db_path = _resolve_internal_target(request.target_id, media_db)
    if not _connector_acl_allows(current_user, target_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "unauthorized_target",
                "message": f"Target '{target_id}' is not authorized for this user",
            },
        )

    service = Text2SQLCoreService(
        generator=_PassThroughSqlGenerator(),
        executor=SqliteReadOnlyExecutor(db_path),
    )

    try:
        result = await service.generate_and_execute(
            query=request.query,
            target_id=target_id,
            timeout_ms=request.timeout_ms,
            max_rows=request.max_rows,
        )
    except SqlPolicyViolation as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "sql_policy_violation", "message": str(exc)},
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "sql_generation_failed", "message": str(exc)},
        ) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={"code": "sql_timeout", "message": "SQL execution timed out"},
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "sql_execution_failed", "message": "SQL execution failed"},
        ) from exc

    sql_text = str(result.get("sql", ""))
    if not request.include_sql:
        sql_text = ""

    columns = [str(value) for value in result.get("columns", [])]
    rows = _shape_rows(columns, list(result.get("rows", [])))

    return Text2SQLResponse(
        sql=sql_text,
        columns=columns,
        rows=rows,
        row_count=int(result.get("row_count", len(rows))),
        duration_ms=int(result.get("duration_ms", 0)),
        target_id=str(result.get("target_id", target_id)),
        guardrail=result.get("guardrail", {}),
        truncated=bool(result.get("truncated", False)),
    )
