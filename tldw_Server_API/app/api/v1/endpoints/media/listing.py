import json
from collections.abc import Mapping
from math import ceil
from typing import Any, Optional

from fastapi import (
    APIRouter,
    Body,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from loguru import logger
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, rbac_rate_limit, RequirePermission, User

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import (
    get_media_db_for_user,
    try_get_media_db_for_user,
)
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_page_pagination_meta
from tldw_Server_API.app.api.v1.schemas.media_request_models import SearchRequest
from tldw_Server_API.app.api.v1.schemas.media_response_models import (
    MediaListItem,
    MediaListResponse,
    PaginationInfo,
)
from tldw_Server_API.app.api.v1.utils.cache import generate_etag, is_not_modified
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.core.AuthNZ.permissions import MEDIA_DELETE
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.media_db.api import (
    fetch_keywords_for_media_batch,
    get_paginated_files,
    get_paginated_trash_files,
    search_media,
)
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    DatabaseError,
    InputError,
)
from tldw_Server_API.app.core.Utils.metadata_utils import normalize_safe_metadata

from .....core.DB_Management.media_db.legacy_maintenance import (
    permanently_delete_item,
)

router = APIRouter(tags=["Media Management"])
_NOT_MODIFIED_OPENAPI_RESPONSE = {
    status.HTTP_304_NOT_MODIFIED: {
        "description": "Media listing not modified (ETag match).",
    },
}

_MEDIA_LISTING_COERCE_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)

_MEDIA_LISTING_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)


try:
    HTTP_422_UNPROCESSABLE = status.HTTP_422_UNPROCESSABLE_CONTENT
except AttributeError:  # Starlette < 0.27
    HTTP_422_UNPROCESSABLE = status.HTTP_422_UNPROCESSABLE_ENTITY


def _is_test_mode() -> bool:
    try:
        from tldw_Server_API.app.core.testing import is_test_mode as _is_test_mode_impl

        return bool(_is_test_mode_impl())
    except _MEDIA_LISTING_NONCRITICAL_EXCEPTIONS:
        return False


_SEARCH_RATE_LIMIT = "600/minute" if _is_test_mode() else "30/minute"

_EMAIL_MEDIA_SEARCH_DELEGATION_MODES = {"opt_in", "auto_email"}


def _normalize_media_types(media_types: list[str] | None) -> list[str]:
    return [str(media_type).strip().lower() for media_type in (media_types or []) if str(media_type).strip()]


def _parse_csv_values(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return [value.strip() for value in str(raw_value).split(",") if value.strip()]


def _as_optional_str(raw_value: Any) -> str | None:
    """Return a stripped string value for whitelisted source-picker fields."""
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    return value or None


def _as_optional_bool(raw_value: Any) -> bool | None:
    """Coerce common stored boolean encodings without exposing unknown values."""
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, int):
        return bool(raw_value)
    if isinstance(raw_value, str):
        value = raw_value.strip().lower()
        if value in {"1", "true", "yes", "y"}:
            return True
        if value in {"0", "false", "no", "n"}:
            return False
    return None


def _as_metadata_mapping(raw_value: Any) -> dict[str, Any]:
    """Parse safe_metadata into a mapping for field-by-field allowlisting."""
    if isinstance(raw_value, Mapping):
        return dict(raw_value)
    if not isinstance(raw_value, str) or not raw_value.strip():
        return {}
    try:
        parsed = json.loads(raw_value)
    except (TypeError, ValueError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _first_present(record: Mapping[str, Any], metadata: Mapping[str, Any], *keys: str) -> Any:
    """Find the first non-empty value across row fields and safe metadata."""
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
        value = metadata.get(key)
        if value not in (None, ""):
            return value
    return None


def _workspace_metadata(
    record: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    """Extract workspace identity from accepted flat or nested metadata shapes."""
    workspace_record = record.get("workspace")
    workspace_metadata = metadata.get("workspace")
    workspace = workspace_record if isinstance(workspace_record, Mapping) else workspace_metadata
    workspace = workspace if isinstance(workspace, Mapping) else {}

    workspace_id = _as_optional_str(
        _first_present(record, metadata, "workspace_id", "workspaceId") or workspace.get("id")
    )
    workspace_name = _as_optional_str(
        _first_present(record, metadata, "workspace_name", "workspaceName") or workspace.get("name")
    )
    return workspace_id, workspace_name


def _source_picker_metadata(record: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return only source-picker-safe metadata fields for list/search payloads."""
    if record is None:
        return {}

    safe_metadata = _as_metadata_mapping(record.get("safe_metadata"))
    workspace_id, workspace_name = _workspace_metadata(record, safe_metadata)
    source_metadata: dict[str, Any] = {}

    status_value = _as_optional_str(
        _first_present(record, safe_metadata, "status", "processing_status", "chunking_status")
    )
    if status_value:
        source_metadata["status"] = status_value

    created_at = _as_optional_str(
        _first_present(record, safe_metadata, "created_at", "ingestion_date", "imported_at")
    )
    if created_at:
        source_metadata["created_at"] = created_at

    updated_at = _as_optional_str(_first_present(record, safe_metadata, "updated_at", "last_modified"))
    if updated_at:
        source_metadata["updated_at"] = updated_at

    if workspace_id:
        source_metadata["workspace_id"] = workspace_id
    if workspace_name:
        source_metadata["workspace_name"] = workspace_name

    bool_fields = {
        "workspace_artifact": ("workspace_artifact", "is_workspace_artifact"),
        "is_generated": ("is_generated", "generated"),
        "test_artifact": ("test_artifact", "is_test"),
    }
    for output_key, input_keys in bool_fields.items():
        value = _as_optional_bool(_first_present(record, safe_metadata, *input_keys))
        if value is not None:
            source_metadata[output_key] = value

    for output_key in ("artifact_kind", "source_kind"):
        value = _as_optional_str(_first_present(record, safe_metadata, output_key))
        if value:
            source_metadata[output_key] = value

    kind_value = _as_optional_str(_first_present(record, safe_metadata, "kind"))
    if kind_value and "artifact_kind" not in source_metadata:
        source_metadata["artifact_kind"] = kind_value

    return source_metadata


@router.get(
    "/keywords",
    summary="List media keywords",
)
async def list_media_keywords(
    query: str | None = Query(None, description="Optional substring filter for keyword suggestions."),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of keywords to return."),
    db: Any = Depends(get_media_db_for_user),
) -> dict[str, list[str]]:
    try:
        keywords = db.fetch_all_keywords()
    except (DatabaseError, InputError) as exc:
        logger.error("Failed to list media keywords")
        raise map_db_error_to_http(
            exc,
            default_detail="Failed to load media keywords",
        ) from exc

    normalized_query = str(query or "").strip().lower()
    if normalized_query:
        keywords = [keyword for keyword in keywords if normalized_query in str(keyword).lower()]

    return {"keywords": keywords[:limit]}


def _should_delegate_media_search_to_email(
    *,
    search_params: SearchRequest,
    email_operator_enabled: bool,
) -> bool:
    explicit_mode = str(search_params.email_query_mode or "").strip().lower()
    media_types = _normalize_media_types(search_params.media_types)
    email_only_media_scope = bool(media_types) and all(media_type == "email" for media_type in media_types)

    if explicit_mode == "operators":
        if not email_operator_enabled:
            raise HTTPException(
                status_code=HTTP_422_UNPROCESSABLE,
                detail="email_query_mode='operators' is disabled by server configuration.",
            )
        if not email_only_media_scope:
            raise HTTPException(
                status_code=HTTP_422_UNPROCESSABLE,
                detail="email_query_mode='operators' requires media_types=['email'].",
            )
        return True

    if explicit_mode == "legacy":
        return False

    delegation_mode = str(settings.get("EMAIL_MEDIA_SEARCH_DELEGATION_MODE", "opt_in") or "").strip().lower()
    if delegation_mode not in _EMAIL_MEDIA_SEARCH_DELEGATION_MODES:
        logger.warning(
            "Invalid EMAIL_MEDIA_SEARCH_DELEGATION_MODE='{}'; falling back to 'opt_in'.",
            delegation_mode or "<empty>",
        )
        delegation_mode = "opt_in"

    return bool(delegation_mode == "auto_email" and email_operator_enabled and email_only_media_scope)


@router.get(
    "/",
    summary="List Media (slash)",
    responses=_NOT_MODIFIED_OPENAPI_RESPONSE,
)
async def list_media_endpoint(
    request: Request,
    response: Response,
    current_user: User = Depends(get_request_user),
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    results_per_page: int = Query(10, ge=1, description="Items per page"),
    include_keywords: bool = Query(
        False,
        description="Include associated keywords for each media item.",
    ),
    db: Any = Depends(get_media_db_for_user),
    if_none_match: Optional[str] = Header(None),
) -> dict[str, Any]:
    """
    Return paginated list of active media items (basic fields only).

    Preserves existing TEST_MODE diagnostics and response shape while
    adding ETag support based on a deterministic serialization.
    """
    try:
        # TEST_MODE diagnostics (headers + log messages)
        try:
            if _is_test_mode():
                db_path = getattr(db, "db_path_str", getattr(db, "db_path", "?"))
                headers = getattr(request, "headers", {}) or {}
                logger.warning(
                    "TEST_MODE: list_media db_path={} user_id={} auth_headers="
                    "{{'X-API-KEY': {{'present': {}}}}, 'Authorization': {{'present': {}}}}}",
                    db_path,
                    getattr(current_user, "id", "?"),
                    bool(headers.get("X-API-KEY")),
                    bool(headers.get("authorization")),
                )
        except _MEDIA_LISTING_NONCRITICAL_EXCEPTIONS:
            pass

        rows, total_pages, current_page, total_items = get_paginated_files(
            db,
            page=page,
            results_per_page=results_per_page,
        )

        # Additional TEST_MODE summary + headers
        try:
            if _is_test_mode():
                logger.warning(
                    "TEST_MODE: list_media summary page={} rpp={} total_items={} rows_returned={}",
                    page,
                    results_per_page,
                    total_items,
                    len(rows or []),
                )
                if response is not None:
                    db_path = getattr(db, "db_path_str", getattr(db, "db_path", "?"))
                    try:
                        response.headers["X-TLDW-DB-Path"] = str(db_path)
                        response.headers["X-TLDW-List-Total"] = str(int(total_items))
                    except _MEDIA_LISTING_COERCE_EXCEPTIONS:
                        pass
        except _MEDIA_LISTING_NONCRITICAL_EXCEPTIONS:
            pass

        # Build base items and collect IDs for keyword lookup
        base_items: list[dict[str, Any]] = []
        media_ids: list[int] = []
        skipped_count = 0
        for r in rows or []:
            row_record = r if isinstance(r, Mapping) else None
            rid_raw = r["id"] if isinstance(r, Mapping) else r[0]
            title = r["title"] if isinstance(r, Mapping) else r[1]
            rtype = r["type"] if isinstance(r, Mapping) else r[2]
            try:
                rid = int(rid_raw)
            except (TypeError, ValueError):
                # Skip rows with invalid IDs rather than failing the entire listing
                logger.error("Skipping media row with invalid id")
                skipped_count += 1
                continue
            media_ids.append(rid)
            base_items.append(
                {
                    "id": rid,
                    "title": str(title),
                    "type": str(rtype),
                    **_source_picker_metadata(row_record),
                }
            )

        # Optionally fetch keywords for all media items on this page in a single batch.
        # keywords_available has three states:
        #   None  -> keywords not requested (omitted from response)
        #   True  -> keywords successfully retrieved or no items to fetch
        #   False -> keyword retrieval failed (graceful degradation)
        keywords_map: dict[int, list[str]] = {}
        keywords_available: Optional[bool] = None
        if include_keywords and media_ids:
            try:
                keywords_map = fetch_keywords_for_media_batch(
                    db,
                    media_ids,
                )
                keywords_available = True
            except (TypeError, InputError, DatabaseError) as exc:
                # Log and degrade gracefully for known keyword lookup failures
                logger.error(
                    "Error fetching keywords for media list page={} rpp={}: {}",
                    page,
                    results_per_page,
                    exc,
                    exc_info=True,
                )
                keywords_map = {}
                # Surface failure via a coarse availability flag so clients can
                # distinguish "no keywords" from "keyword lookup failed".
                keywords_available = False
            except Exception as exc:  # noqa: BLE001  # pragma: no cover - unexpected failures
                # Preserve graceful degradation for unexpected errors while
                # still logging with full context for observability.
                logger.error(
                    "Unexpected error fetching keywords for media list page={} rpp={}: {}",
                    page,
                    results_per_page,
                    exc,
                    exc_info=True,
                )
                keywords_map = {}
                keywords_available = False
        elif include_keywords:
            # Keywords were requested but there were no media_ids to look up;
            # treat this as a successful (no-op) lookup.
            keywords_available = True

        # Build response items, including keywords only when requested
        items: list[dict[str, Any]] = []
        for item in base_items:
            mid = item["id"]
            base_payload: dict[str, Any] = {
                **item,
                "url": f"/api/v1/media/{mid}",
            }
            if include_keywords:
                base_payload["keywords"] = keywords_map.get(mid, [])
            items.append(base_payload)

        pagination_info = PaginationInfo(
            results_per_page=int(results_per_page),
            total_items=int(total_items),
            **build_page_pagination_meta(
                page=int(current_page),
                per_page=int(results_per_page),
                total=int(total_items),
                total_pages=int(total_pages),
            ).model_dump(),
        )

        payload: dict[str, Any] = {
            "items": items,
            "pagination": pagination_info.model_dump(),
        }

        if include_keywords and keywords_available is not None:
            payload["keywords_available"] = keywords_available

        if skipped_count > 0:
            payload["skipped_count"] = skipped_count

        etag = generate_etag(payload)
        response.headers["ETag"] = etag
        if is_not_modified(etag, if_none_match):
            response.status_code = status.HTTP_304_NOT_MODIFIED
            return {}

        return payload
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error listing media")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list media",
        ) from exc


@router.get(
    "/trash",
    summary="List trashed media items",
    responses=_NOT_MODIFIED_OPENAPI_RESPONSE,
)
async def list_media_trash_endpoint(
    request: Request,
    response: Response,
    current_user: User = Depends(get_request_user),
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    results_per_page: int = Query(10, ge=1, description="Items per page"),
    include_keywords: bool = Query(
        False,
        description="Include associated keywords for each media item.",
    ),
    db: Any = Depends(get_media_db_for_user),
    if_none_match: Optional[str] = Header(None),
) -> dict[str, Any]:
    """
    Return paginated list of trashed media items (basic fields only).
    """
    try:
        try:
            if _is_test_mode():
                db_path = getattr(db, "db_path_str", getattr(db, "db_path", "?"))
                headers = getattr(request, "headers", {}) or {}
                logger.warning(
                    "TEST_MODE: list_media_trash db_path={} user_id={} auth_headers="
                    "{{'X-API-KEY': {{'present': {}}}}, 'Authorization': {{'present': {}}}}}",
                    db_path,
                    getattr(current_user, "id", "?"),
                    bool(headers.get("X-API-KEY")),
                    bool(headers.get("authorization")),
                )
        except _MEDIA_LISTING_NONCRITICAL_EXCEPTIONS:
            pass

        rows, total_pages, current_page, total_items = get_paginated_trash_files(
            db,
            page=page,
            results_per_page=results_per_page,
        )

        try:
            if _is_test_mode():
                logger.warning(
                    "TEST_MODE: list_media_trash summary page={} rpp={} total_items={} rows_returned={}",
                    page,
                    results_per_page,
                    total_items,
                    len(rows or []),
                )
                if response is not None:
                    db_path = getattr(db, "db_path_str", getattr(db, "db_path", "?"))
                    try:
                        response.headers["X-TLDW-DB-Path"] = str(db_path)
                        response.headers["X-TLDW-List-Total"] = str(int(total_items))
                    except _MEDIA_LISTING_COERCE_EXCEPTIONS:
                        pass
        except _MEDIA_LISTING_NONCRITICAL_EXCEPTIONS:
            pass

        base_items: list[dict[str, Any]] = []
        media_ids: list[int] = []
        skipped_count = 0
        for r in rows or []:
            rid_raw = r["id"] if isinstance(r, dict) else r[0]
            title = r["title"] if isinstance(r, dict) else r[1]
            rtype = r["type"] if isinstance(r, dict) else r[2]
            try:
                rid = int(rid_raw)
            except (TypeError, ValueError):
                logger.error("Skipping trashed media row with invalid id")
                skipped_count += 1
                continue
            media_ids.append(rid)
            base_items.append(
                {
                    "id": rid,
                    "title": str(title),
                    "type": str(rtype),
                }
            )

        keywords_map: dict[int, list[str]] = {}
        keywords_available: Optional[bool] = None
        if include_keywords and media_ids:
            try:
                keywords_map = fetch_keywords_for_media_batch(
                    db,
                    media_ids,
                )
                keywords_available = True
            except (TypeError, InputError, DatabaseError) as exc:
                logger.error(
                    "Error fetching keywords for media trash list page={} rpp={}: {}",
                    page,
                    results_per_page,
                    exc,
                    exc_info=True,
                )
                keywords_map = {}
                keywords_available = False
            except _MEDIA_LISTING_NONCRITICAL_EXCEPTIONS as exc:  # pragma: no cover
                logger.error(
                    "Unexpected error fetching keywords for media trash list page={} rpp={}: {}",
                    page,
                    results_per_page,
                    exc,
                    exc_info=True,
                )
                keywords_map = {}
                keywords_available = False
        elif include_keywords:
            keywords_available = True

        items: list[dict[str, Any]] = []
        for item in base_items:
            mid = item["id"]
            base_payload: dict[str, Any] = {
                "id": mid,
                "title": item["title"],
                "type": item["type"],
                "url": f"/api/v1/media/{mid}",
            }
            if include_keywords:
                base_payload["keywords"] = keywords_map.get(mid, [])
            items.append(base_payload)

        pagination_info = PaginationInfo(
            results_per_page=int(results_per_page),
            total_items=int(total_items),
            **build_page_pagination_meta(
                page=int(current_page),
                per_page=int(results_per_page),
                total=int(total_items),
                total_pages=int(total_pages),
            ).model_dump(),
        )

        payload: dict[str, Any] = {
            "items": items,
            "pagination": pagination_info.model_dump(),
        }

        if include_keywords and keywords_available is not None:
            payload["keywords_available"] = keywords_available

        if skipped_count > 0:
            payload["skipped_count"] = skipped_count

        etag = generate_etag(payload)
        response.headers["ETag"] = etag
        if is_not_modified(etag, if_none_match):
            response.status_code = status.HTTP_304_NOT_MODIFIED
            return {}

        return payload
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error listing trashed media")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list trashed media",
        ) from exc


@router.post(
    "/trash/empty",
    summary="Empty media trash",
    dependencies=[
        Depends(RequirePermission(MEDIA_DELETE)),
        Depends(rbac_rate_limit("media.delete")),
    ],
)
async def empty_media_trash_endpoint(
    response: Response,
    db: Any = Depends(get_media_db_for_user),
    current_user: User = Depends(get_request_user),
) -> dict[str, Any]:
    """
    Permanently delete all items currently in trash.
    """
    try:
        cursor = db.execute_query("SELECT id FROM Media WHERE deleted = 0 AND is_trash = 1")
        rows = cursor.fetchall()
        media_ids = [row["id"] for row in rows] if rows else []

        deleted_count = 0
        failed_ids: list[int] = []
        for media_id in media_ids:
            try:
                deleted = permanently_delete_item(db, int(media_id))
                if deleted:
                    deleted_count += 1
                else:
                    failed_ids.append(int(media_id))
            except _MEDIA_LISTING_NONCRITICAL_EXCEPTIONS as exc:
                logger.error(
                    "Error permanently deleting trashed media {}: {}",
                    media_id,
                    exc,
                    exc_info=True,
                )
                failed_ids.append(int(media_id))

        remaining_count = -1
        try:
            count_cursor = db.execute_query(
                "SELECT COUNT(*) AS total_items FROM Media WHERE deleted = 0 AND is_trash = 1"
            )
            count_row = count_cursor.fetchone()
            remaining_count = count_row["total_items"] if count_row else 0
        except _MEDIA_LISTING_NONCRITICAL_EXCEPTIONS:
            pass

        logger.warning(
            "User {} emptied trash: deleted_count={} failed_count={} remaining_count={}",
            getattr(current_user, "id", "?"),
            deleted_count,
            len(failed_ids),
            remaining_count,
        )

        return {
            "deleted_count": deleted_count,
            "failed_count": len(failed_ids),
            "failed_ids": failed_ids,
            "remaining_count": remaining_count,
        }
    except Exception as exc:
        logger.error("Error emptying media trash")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to empty media trash",
        ) from exc


@router.get(
    "/metadata-search",
    summary="Search media by safe metadata",
    responses=_NOT_MODIFIED_OPENAPI_RESPONSE,
)
async def search_by_metadata(
    request: Request,
    response: Response,
    filters: Optional[str] = Query(
        None,
        description="JSON list of {field, op, value}",
    ),
    field: Optional[str] = Query(
        None,
        description="Single filter field",
    ),
    op: Optional[str] = Query(
        "icontains",
        description="Operator: eq|contains|icontains|startswith|endswith",
    ),
    value: Optional[str] = Query(
        None,
        description="Single filter value",
    ),
    match_mode: str = Query("all", description="all|any"),
    group_by_media: bool = Query(True),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    q: Optional[str] = Query(
        None,
        description="Optional text query against title/safe metadata",
    ),
    media_types: Optional[str] = Query(
        None,
        description="Optional comma-separated media types",
    ),
    must_have: Optional[str] = Query(
        None,
        description="Optional comma-separated required keywords",
    ),
    must_not_have: Optional[str] = Query(
        None,
        description="Optional comma-separated excluded keywords",
    ),
    date_start: Optional[str] = Query(
        None,
        description="Optional lower date bound (ISO 8601 string)",
    ),
    date_end: Optional[str] = Query(
        None,
        description="Optional upper date bound (ISO 8601 string)",
    ),
    sort_by: Optional[str] = Query(
        None,
        description="Optional sort override: date_desc|date_asc|title_asc|title_desc",
    ),
    db: Any = Depends(get_media_db_for_user),
    if_none_match: Optional[str] = Header(None),
) -> dict[str, Any]:
    """
    Search media items based on version safe_metadata fields and identifier indices.

    Mirrors the legacy implementation while adding basic ETag support.
    """
    try:
        flt_list: list[dict[str, Any]] = []
        import json as _json

        if filters:
            try:
                parsed = _json.loads(filters)
                if isinstance(parsed, list):
                    for f in parsed:
                        if isinstance(f, dict) and "field" in f and "value" in f:
                            flt_list.append(
                                {
                                    "field": f["field"],
                                    "op": f.get("op", "icontains"),
                                    "value": f["value"],
                                }
                            )
            except Exception as je:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid 'filters' JSON: {je}",
                ) from je
        elif field and value is not None:
            flt_list.append(
                {
                    "field": field,
                    "op": op or "icontains",
                    "value": value,
                }
            )

        # Normalize identifier filters where applicable (doi/pmid/pmcid/arxiv_id)
        norm_fields = {
            "doi",
            "pmid",
            "pmcid",
            "arxiv_id",
            "DOI",
            "PMID",
            "PMCID",
            "arXiv",
            "ArXiv",
        }
        canonical_order = ("doi", "pmid", "pmcid", "arxiv_id", "s2_paper_id")
        normalized_filters: list[dict[str, Any]] = []
        for f in flt_list or []:
            try:
                fld = f.get("field")
                if fld in norm_fields:
                    norm = normalize_safe_metadata({fld: f.get("value")})
                    key = next(
                        (k for k in canonical_order if k in norm),
                        (fld or "").lower(),
                    )
                    val = norm.get(key, f.get("value"))
                    normalized_filters.append(
                        {
                            "field": key,
                            "op": f.get("op", "icontains"),
                            "value": val,
                        }
                    )
                else:
                    normalized_filters.append(f)
            except ValueError as ve:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(ve),
                ) from ve

        optional_search_kwargs: dict[str, Any] = {}

        text_query = (q or "").strip()
        if text_query:
            optional_search_kwargs["text_query"] = text_query

        normalized_media_types = [value.lower() for value in _parse_csv_values(media_types)]
        if normalized_media_types:
            optional_search_kwargs["media_types"] = normalized_media_types

        must_have_keywords = _parse_csv_values(must_have)
        if must_have_keywords:
            optional_search_kwargs["must_have_keywords"] = must_have_keywords

        must_not_have_keywords = _parse_csv_values(must_not_have)
        if must_not_have_keywords:
            optional_search_kwargs["must_not_have_keywords"] = must_not_have_keywords

        normalized_date_start = (date_start or "").strip()
        if normalized_date_start:
            optional_search_kwargs["date_start"] = normalized_date_start

        normalized_date_end = (date_end or "").strip()
        if normalized_date_end:
            optional_search_kwargs["date_end"] = normalized_date_end

        normalized_sort_by = (sort_by or "").strip().lower()
        if normalized_sort_by:
            optional_search_kwargs["sort_by"] = normalized_sort_by

        rows, total = db.search_by_safe_metadata(
            filters=normalized_filters or None,
            match_all=(match_mode.lower() == "all"),
            page=page,
            per_page=per_page,
            group_by_media=group_by_media,
            **optional_search_kwargs,
        )

        for r in rows:
            sm = r.get("safe_metadata")
            if isinstance(sm, str):
                try:
                    r["safe_metadata"] = _json.loads(sm)
                except _MEDIA_LISTING_COERCE_EXCEPTIONS:
                    r["safe_metadata"] = None

        total_pages = (total + per_page - 1) // per_page
        payload: dict[str, Any] = {
            "results": rows,
            "pagination": build_page_pagination_meta(
                page=page,
                per_page=per_page,
                total=total,
                total_pages=total_pages,
            ).model_dump(),
        }

        etag = generate_etag(payload)
        response.headers["ETag"] = etag
        if is_not_modified(etag, if_none_match):
            response.status_code = status.HTTP_304_NOT_MODIFIED
            return {}

        return payload
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Metadata search error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error performing metadata search",
        ) from exc


async def _validate_identifier_query(
    doi: Optional[str] = Query(None),
    pmid: Optional[str] = Query(None),
    pmcid: Optional[str] = Query(None),
    arxiv_id: Optional[str] = Query(None),
    s2_paper_id: Optional[str] = Query(None),
) -> bool:
    """
    Early validation for /by-identifier to ensure malformed IDs return 400 before auth/DB.

    Uses normalize_safe_metadata which raises ValueError for invalid DOI/PMID/PMCID.
    """
    raw: dict[str, Any] = {}
    if doi is not None:
        raw["doi"] = doi
    if pmid is not None:
        raw["pmid"] = pmid
    if pmcid is not None:
        raw["pmcid"] = pmcid
    if arxiv_id is not None:
        raw["arxiv_id"] = arxiv_id
    if s2_paper_id is not None:
        raw["s2_paper_id"] = s2_paper_id

    try:
        if raw:
            normalize_safe_metadata(raw)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Provide at least one identifier",
            )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        ) from ve
    return True


@router.get(
    "/by-identifier",
    summary="Find media by standard identifier (DOI/PMID/PMCID/arXiv/S2)",
    dependencies=[Depends(_validate_identifier_query)],
    responses=_NOT_MODIFIED_OPENAPI_RESPONSE,
)
async def get_by_identifier(
    request: Request,
    response: Response,
    doi: Optional[str] = Query(None),
    pmid: Optional[str] = Query(None),
    pmcid: Optional[str] = Query(None),
    arxiv_id: Optional[str] = Query(None),
    s2_paper_id: Optional[str] = Query(None),
    group_by_media: bool = Query(True),
    db: Optional[Any] = Depends(try_get_media_db_for_user),
    if_none_match: Optional[str] = Header(None),
) -> dict[str, Any]:
    """
    Quick lookup by canonical identifiers. Returns latest matching version per media by default.
    """
    try:
        flt_list: list[dict[str, Any]] = []
        raw_filters: list[dict[str, Any]] = []
        if doi:
            raw_filters.append({"field": "doi", "op": "eq", "value": doi})
        if pmid:
            raw_filters.append({"field": "pmid", "op": "eq", "value": pmid})
        if pmcid:
            raw_filters.append({"field": "pmcid", "op": "eq", "value": pmcid})
        if arxiv_id:
            raw_filters.append({"field": "arxiv_id", "op": "eq", "value": arxiv_id})
        if s2_paper_id:
            raw_filters.append({"field": "s2_paper_id", "op": "eq", "value": s2_paper_id})

        for f in raw_filters:
            try:
                if f["field"] != "s2_paper_id":
                    norm = normalize_safe_metadata({f["field"]: f["value"]})
                else:
                    norm = {f["field"]: f["value"]}
                canonical_order = ("doi", "pmid", "pmcid", "arxiv_id", "s2_paper_id")
                key = next(
                    (k for k in canonical_order if k in norm),
                    (f["field"] or "").lower(),
                )
                val = norm.get(key, f["value"])
                flt_list.append({"field": key, "op": f["op"], "value": val})
            except ValueError as ve:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(ve),
                ) from ve

        if not flt_list:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Provide at least one identifier",
            )
        if db is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Media DB initialization failed",
            )

        rows, total = db.search_by_safe_metadata(
            filters=flt_list,
            match_all=True,
            page=1,
            per_page=50,
            group_by_media=group_by_media,
        )

        import json as _json

        for r in rows:
            sm = r.get("safe_metadata")
            if isinstance(sm, str):
                try:
                    r["safe_metadata"] = _json.loads(sm)
                except _MEDIA_LISTING_COERCE_EXCEPTIONS:
                    r["safe_metadata"] = None

        payload: dict[str, Any] = {"results": rows, "total": total}
        etag = generate_etag(payload)
        response.headers["ETag"] = etag
        if is_not_modified(etag, if_none_match):
            response.status_code = status.HTTP_304_NOT_MODIFIED
            return {}

        return payload
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Identifier lookup error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error in identifier lookup",
        ) from exc


@router.post(
    "/search",
    status_code=status.HTTP_200_OK,
    summary="Search Media Items",
    response_model=MediaListResponse,
    responses=_NOT_MODIFIED_OPENAPI_RESPONSE,
)
async def search_media_items(
    request: Request,
    search_params: SearchRequest = Body(...),
    page: int = Query(1, ge=1, description="Page number"),
    results_per_page: int = Query(
        10,
        ge=1,
        le=100,
        description="Results per page",
    ),
    db: Any = Depends(get_media_db_for_user),
    if_none_match: Optional[str] = Header(None),
) -> Response:
    """
    Search across media items based on various criteria.

    Preserves the legacy response envelope while using centralized
    ETag helpers for conditional responses.
    """
    try:
        query_text_for_match: Optional[str] = None
        if search_params.exact_phrase:
            query_text_for_match = f'"{search_params.exact_phrase.strip()}"'
        elif search_params.query:
            query_text_for_match = search_params.query.strip()

        email_operator_enabled = bool(settings.get("EMAIL_OPERATOR_SEARCH_ENABLED", True))
        use_email_operator_bridge = _should_delegate_media_search_to_email(
            search_params=search_params,
            email_operator_enabled=email_operator_enabled,
        )
        if use_email_operator_bridge:
            email_rows, total_items = db.search_email_messages(
                query=query_text_for_match,
                include_deleted=False,
                limit=results_per_page,
                offset=(page - 1) * results_per_page,
            )
            items_data: list[dict[str, Any]] = []
            for row in email_rows:
                media_id_raw = row.get("media_id")
                try:
                    media_id = int(media_id_raw)
                except (TypeError, ValueError):
                    continue
                title_value = row.get("media_title") or row.get("subject") or f"Email {media_id}"
                items_data.append(
                    {
                        "id": media_id,
                        "title": str(title_value),
                        "type": "email",
                    }
                )
        else:
            items_data, total_items = search_media(
                db,
                search_query=query_text_for_match,
                search_fields=search_params.fields,
                media_types=search_params.media_types,
                date_range=search_params.date_range,
                must_have_keywords=search_params.must_have,
                must_not_have_keywords=search_params.must_not_have,
                sort_by=search_params.sort_by,
                boost_fields=search_params.boost_fields,
                page=page,
                results_per_page=results_per_page,
                include_trash=False,
                include_deleted=False,
            )

        formatted_items = [
            MediaListItem(
                id=item["id"],
                title=item["title"],
                type=item["type"],
                url=f"/api/v1/media/{item['id']}",
                **_source_picker_metadata(item),
            )
            for item in items_data
        ]

        total_pages = ceil(total_items / results_per_page) if results_per_page > 0 and total_items > 0 else 0

        pagination_info = PaginationInfo(
            results_per_page=results_per_page,
            total_items=total_items,
            **build_page_pagination_meta(
                page=page,
                per_page=results_per_page,
                total=total_items,
                total_pages=total_pages,
            ).model_dump(),
        )

        try:
            response_obj = MediaListResponse(
                items=formatted_items,
                pagination=pagination_info,
            )

            payload_dict = response_obj.model_dump(exclude_none=True)
            payload_dict["results"] = payload_dict.get("items", [])

            etag = generate_etag(payload_dict)
            if is_not_modified(etag, if_none_match):
                return Response(
                    status_code=status.HTTP_304_NOT_MODIFIED,
                    headers={"ETag": etag},
                )

            import json

            response_json = json.dumps(payload_dict)
            return Response(
                content=response_json,
                media_type="application/json",
                headers={"ETag": etag},
            )
        except Exception as ve:
            logger.debug(
                "Data causing validation error in search: items_count={}, " "pagination={}",
                len(formatted_items),
                pagination_info.model_dump_json(indent=2) if pagination_info else "None",
            )
            logger.error(
                f"Pydantic validation error creating MediaListResponse for search: {ve}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error: Response creation failed.",
            ) from ve
    except InputError as exc:
        logger.warning(
            f"Invalid email operator query for media search bridge: {exc}",
            exc_info=True,
        )
        raise map_db_error_to_http(exc) from exc
    except ValueError as ve:
        logger.warning(
            f"Invalid parameters for media search: {ve}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE,
            detail=str(ve),
        ) from ve
    except DatabaseError as exc:
        logger.error("Database error during media search")
        raise map_db_error_to_http(
            exc,
            default_detail="A database error occurred during the search.",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            f"Unexpected error in search_media_items endpoint: {exc}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal server error occurred.",
        ) from exc


__all__ = ["router"]
