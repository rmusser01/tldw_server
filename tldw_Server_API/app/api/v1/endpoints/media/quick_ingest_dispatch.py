"""Admit a persistent Quick Ingest source at the receiving transport boundary."""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.routing import APIRoute
from pydantic import BaseModel

from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import get_collections_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.usage_quota_deps import MediaAnalysisQuotaDeniedBeforeSave
from tldw_Server_API.app.api.v1.endpoints.media import add, ingest_jobs, process_web_scraping
from tldw_Server_API.app.api.v1.endpoints.media.collections import _item_response
from tldw_Server_API.app.api.v1.schemas.media_collections import (
    MediaCollectionItemResponse,
    PlanKey,
    QuickIngestDispatchState,
    positive_quick_ingest_media_id,
)
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.exceptions import BadRequestError


@dataclass(frozen=True)
class PlanAttempt:
    plan_key: str
    source_key: str
    retry_count: int
    idempotency_key: str
    db: CollectionsDatabase


class DispatchReceipt(BaseModel):
    dispatched: bool
    dispatch_state: QuickIngestDispatchState
    item: MediaCollectionItemResponse
    result: Any = None


def _attempt(
    plan_key: PlanKey,
    source_key: PlanKey,
    retry_count: int = Query(..., ge=0),
    idempotency_key: str = Query(..., min_length=1, max_length=255),
    db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> PlanAttempt:
    return PlanAttempt(plan_key, source_key, retry_count, idempotency_key, db)


def _validate_source(item: Any, transport: str, inputs: dict[str, Any]) -> None:
    if transport == "web":
        payload = inputs["payload"]
        valid = (
            payload.mode == "persist"
            and payload.scrape_method.lower() in {"individual", "individual urls"}
            and payload.url_input.strip() == item.source_url
            and not item.source_url.startswith("local://")
            and len(payload.url_input.splitlines()) == 1
        )
    else:
        urls = [url.strip() for url in inputs["form_data"].urls or [] if url.strip()]
        files = inputs["files"] or []
        if item.source_url.startswith("local://"):
            valid = not urls and len(files) == 1 and files[0].filename == unquote(item.source_url[8:])
        else:
            valid = not files and urls == [item.source_url]
    if not valid:
        raise HTTPException(409, "The submitted source does not match the collection plan.")


def _outcome(transport: str, result: Any) -> dict[str, Any] | None:
    """Project only a single authoritative Job or Media identity for this source."""
    if not isinstance(result, dict):
        return None
    if transport == "jobs":
        jobs = result.get("jobs")
        if isinstance(jobs, list) and len(jobs) == 1 and isinstance(jobs[0], dict):
            job_id = positive_quick_ingest_media_id(jobs[0].get("id"))
            if job_id:
                return {"status": "processing", "latest_job_id": str(job_id)}
        return None
    row = result
    if transport == "add":
        rows = result.get("results")
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
            return None
        row = rows[0]
    ids = row.get("media_ids", [])
    if not isinstance(ids, list):
        return None
    candidates = [*ids, row.get("db_id"), row.get("media_id")]
    keys = {positive_quick_ingest_media_id(key) for key in candidates if key is not None}
    if None in keys or len(keys) > 1:
        return None
    status = str(row.get("status", "")).lower()
    if status in {"duplicate", "skipped", "skipped_existing"}:
        return {"status": "skipped_existing", "media_id": next(iter(keys), None)}
    if keys:
        return {"status": "completed", "media_id": next(iter(keys))}
    if (
        transport == "web"
        and row.get("stored_articles") == 0
        and row.get("total_articles") == 1
        and isinstance(row.get("extraction_failures"), list)
        and len(row["extraction_failures"]) == 1
    ):
        return {"status": "failed"}
    if status in {"error", "failed", "persist-error"}:
        return {"status": "failed"}
    return None


async def receive(
    attempt: PlanAttempt,
    transport: str,
    inputs: dict[str, Any],
    operation: Callable[..., Awaitable[Any]],
) -> dict[str, Any]:
    """Consume one reservation, run the original handler, and commit its exact receipt."""
    db = attempt.db
    slot = (attempt.plan_key, attempt.source_key)
    identity = {"retry_count": attempt.retry_count, "idempotency_key": attempt.idempotency_key, "transport": transport}
    try:
        item = db.get_quick_ingest_source(*slot)
        if (item.retry_count, item.idempotency_key) != (attempt.retry_count, attempt.idempotency_key):
            raise HTTPException(409, "The source attempt has changed.")
        _validate_source(item, transport, inputs)
        won, item = db.begin_quick_ingest_dispatch(*slot, **identity)
    except KeyError as exc:
        raise HTTPException(404, "Collection plan source not found.") from exc
    except ValueError as exc:
        raise HTTPException(409, "The source attempt has changed.") from exc

    def receipt(dispatched: bool, row: Any, result: Any = None) -> dict[str, Any]:
        return {
            "dispatched": dispatched,
            "dispatch_state": row.dispatch_state,
            "item": _item_response(row),
            "result": result,
        }

    def finish(**outcome: Any) -> Any:
        try:
            return db.finish_quick_ingest_dispatch(*slot, **identity, **outcome)
        except KeyError as exc:
            raise HTTPException(404, "Collection plan source not found.") from exc
        except ValueError as exc:
            raise HTTPException(409, "The source attempt changed before the result was recorded.") from exc

    if not won:
        if (item.retry_count, item.idempotency_key) != (attempt.retry_count, attempt.idempotency_key):
            raise HTTPException(409, "The source attempt changed before dispatch.")
        return receipt(False, item)
    if transport == "jobs":
        # Bind the Job to the server's item and logical key, including after queue routing changes.
        inputs.update(
            media_collection_id=str(item.collection_id),
            collection_id=None,
            media_collection_item_id=str(item.id),
            planned_item_ids=None,
            idempotency_key=item.idempotency_key,
            idempotency_keys=None,
            collections_db=None,
        )
    try:
        raw = await operation(**inputs)
    except MediaAnalysisQuotaDeniedBeforeSave:
        finish(status="failed", error_summary="Analysis quota reached before ingestion started.")
        raise
    except HTTPException as exc:
        if transport == "jobs" and isinstance(exc.__cause__, BadRequestError):
            if exc.status_code == 429 and "concurrent job limit" in str(exc.__cause__).lower():
                finish(status="fallback_ready")
                raise HTTPException(429, {"code": "plan_jobs_quota_rejected", "message": str(exc.detail)}) from exc
            if exc.status_code == 400:
                finish(status="failed")
        raise

    result = raw.model_dump(mode="json") if isinstance(raw, BaseModel) else raw
    if isinstance(raw, Response):
        result = json.loads(raw.body)
        response = inputs.get("response")
        if isinstance(response, Response):
            for key, value in raw.headers.items():
                if key.lower() not in {"content-length", "content-type"}:
                    response.headers[key] = value
    if isinstance(raw, ingest_jobs.UnpublishedUploadValidationResponse):
        error = "The upload was rejected before a job was created. Check the file and retry."
        if "Uploaded file content is empty." in result["errors"]:
            error = "The uploaded file was empty."
        return receipt(True, finish(status="failed", error_summary=error), result)
    outcome = _outcome(transport, result)
    if outcome is None:
        raise HTTPException(502, "The source result could not be confirmed. Check the collection plan before retrying.")
    return receipt(True, finish(**outcome), result)


router = APIRouter()
_PREFIX = "/collections/quick-ingest-plans/{plan_key}/sources/{source_key}/dispatch"


def _register(transport: str, legacy_router: APIRouter, path: str) -> None:
    """Reuse the original typed body, multipart fields and access checks for each planned transport."""
    original = next(route for route in legacy_router.routes if isinstance(route, APIRoute) and route.path == path)
    signature = inspect.signature(original.endpoint, eval_str=True)

    async def dispatch(*, plan_attempt: PlanAttempt, **inputs: Any) -> dict[str, Any]:
        return await receive(plan_attempt, transport, inputs, original.endpoint)

    # Keep FastAPI's validated inputs and dependencies identical to the unplanned route.
    dispatch.__signature__ = signature.replace(  # type: ignore[attr-defined]
        parameters=[
            *signature.parameters.values(),
            inspect.Parameter(
                "plan_attempt",
                inspect.Parameter.KEYWORD_ONLY,
                default=Depends(_attempt),
                annotation=PlanAttempt,
            ),
        ],
        return_annotation=DispatchReceipt,
    )
    router.add_api_route(
        f"{_PREFIX}/{transport}",
        dispatch,
        methods=["POST"],
        dependencies=original.dependencies,
        response_model=DispatchReceipt,
        name=f"dispatch_planned_quick_ingest_{transport}",
    )


_register("jobs", ingest_jobs.router, "/ingest/jobs")
_register("web", process_web_scraping.router, "/process-web-scraping")
_register("add", add.router, "/add")
