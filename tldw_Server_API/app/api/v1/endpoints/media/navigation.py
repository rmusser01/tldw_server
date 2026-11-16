"""Media navigation tree endpoint for chapter/section browsing."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user, rbac_rate_limit
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.schemas.media_navigation_schemas import (
    MediaNavigationContentQueryParams,
    MediaNavigationContentResponse,
    MediaNavigationQueryParams,
    MediaNavigationResponse,
    MediaNavigationStats,
    MediaNavigationTarget,
)
from tldw_Server_API.app.api.v1.utils.cache import cache_response, get_cached_response
from tldw_Server_API.app.api.v1.utils.media_navigation_policy import (
    MEDIA_NAVIGATION_ROUTE_POLICY,
)
from tldw_Server_API.app.core.Media.navigation import (
    MAX_OUTLINE_FILE_SIZE as MAX_OUTLINE_FILE_SIZE,
)
from tldw_Server_API.app.core.Media.navigation import (
    NAVIGATION_SOURCE_PRIORITY as NAVIGATION_SOURCE_PRIORITY,
)
from tldw_Server_API.app.core.Media.navigation import (
    MediaNavigationDb as MediaNavigationDb,
)
from tldw_Server_API.app.core.Media.navigation import (
    _build_content_variants as _build_content_variants,
)
from tldw_Server_API.app.core.Media.navigation import (
    _build_navigation_cache_key as _build_navigation_cache_key,
)
from tldw_Server_API.app.core.Media.navigation import (
    _build_navigation_content_cache_key as _build_navigation_content_cache_key,
)
from tldw_Server_API.app.core.Media.navigation import (
    _build_transcript_node_title as _build_transcript_node_title,
)
from tldw_Server_API.app.core.Media.navigation import (
    _chunk_path_to_node_id as _chunk_path_to_node_id,
)
from tldw_Server_API.app.core.Media.navigation import (
    _clean_navigation_title as _clean_navigation_title,
)
from tldw_Server_API.app.core.Media.navigation import (
    _coerce_segment_seconds as _coerce_segment_seconds,
)
from tldw_Server_API.app.core.Media.navigation import (
    _coerce_transcript_payload as _coerce_transcript_payload,
)
from tldw_Server_API.app.core.Media.navigation import (
    _compute_navigation_version as _compute_navigation_version,
)
from tldw_Server_API.app.core.Media.navigation import (
    _contains_html_tag_linear as _contains_html_tag_linear,
)
from tldw_Server_API.app.core.Media.navigation import (
    _contains_markdown_hint as _contains_markdown_hint,
)
from tldw_Server_API.app.core.Media.navigation import (
    _derive_content_span as _derive_content_span,
)
from tldw_Server_API.app.core.Media.navigation import (
    _detect_intrinsic_format as _detect_intrinsic_format,
)
from tldw_Server_API.app.core.Media.navigation import (
    _extract_chunk_metadata_nodes as _extract_chunk_metadata_nodes,
)
from tldw_Server_API.app.core.Media.navigation import (
    _extract_document_structure_nodes as _extract_document_structure_nodes,
)
from tldw_Server_API.app.core.Media.navigation import (
    _extract_generated_chunk_nodes as _extract_generated_chunk_nodes,
)
from tldw_Server_API.app.core.Media.navigation import (
    _extract_generated_fallback_nodes as _extract_generated_fallback_nodes,
)
from tldw_Server_API.app.core.Media.navigation import (
    _extract_generated_heading_nodes as _extract_generated_heading_nodes,
)
from tldw_Server_API.app.core.Media.navigation import (
    _extract_generated_toc_nodes as _extract_generated_toc_nodes,
)
from tldw_Server_API.app.core.Media.navigation import (
    _extract_pdf_outline_nodes as _extract_pdf_outline_nodes,
)
from tldw_Server_API.app.core.Media.navigation import (
    _extract_transcript_segment_nodes as _extract_transcript_segment_nodes,
)
from tldw_Server_API.app.core.Media.navigation import (
    _filter_navigation_nodes as _filter_navigation_nodes,
)
from tldw_Server_API.app.core.Media.navigation import (
    _find_navigation_node as _find_navigation_node,
)
from tldw_Server_API.app.core.Media.navigation import (
    _get_media_text as _get_media_text,
)
from tldw_Server_API.app.core.Media.navigation import (
    _html_to_plain as _html_to_plain,
)
from tldw_Server_API.app.core.Media.navigation import (
    _infer_generated_toc_level as _infer_generated_toc_level,
)
from tldw_Server_API.app.core.Media.navigation import (
    _is_noisy_navigation_title as _is_noisy_navigation_title,
)
from tldw_Server_API.app.core.Media.navigation import (
    _is_plausible_generated_toc_title as _is_plausible_generated_toc_title,
)
from tldw_Server_API.app.core.Media.navigation import (
    _is_sparse_pdf_outline as _is_sparse_pdf_outline,
)
from tldw_Server_API.app.core.Media.navigation import (
    _looks_like_multi_entry_start as _looks_like_multi_entry_start,
)
from tldw_Server_API.app.core.Media.navigation import (
    _markdown_to_html as _markdown_to_html,
)
from tldw_Server_API.app.core.Media.navigation import (
    _materialize_navigation_nodes as _materialize_navigation_nodes,
)
from tldw_Server_API.app.core.Media.navigation import (
    _node_sort_key as _node_sort_key,
)
from tldw_Server_API.app.core.Media.navigation import (
    _normalize_section_path as _normalize_section_path,
)
from tldw_Server_API.app.core.Media.navigation import (
    _normalize_transcript_segments as _normalize_transcript_segments,
)
from tldw_Server_API.app.core.Media.navigation import (
    _parse_chunk_metadata as _parse_chunk_metadata,
)
from tldw_Server_API.app.core.Media.navigation import (
    _parse_colon_time_to_seconds as _parse_colon_time_to_seconds,
)
from tldw_Server_API.app.core.Media.navigation import (
    _plain_to_html as _plain_to_html,
)
from tldw_Server_API.app.core.Media.navigation import (
    _preorder_with_path_labels as _preorder_with_path_labels,
)
from tldw_Server_API.app.core.Media.navigation import (
    _resolve_requested_content as _resolve_requested_content,
)
from tldw_Server_API.app.core.Media.navigation import (
    _sanitize_navigation_path_parts as _sanitize_navigation_path_parts,
)
from tldw_Server_API.app.core.Media.navigation import (
    _select_source_nodes as _select_source_nodes,
)
from tldw_Server_API.app.core.Media.navigation import (
    _serialize_nodes_for_version as _serialize_nodes_for_version,
)
from tldw_Server_API.app.core.Media.navigation import (
    _slice_text as _slice_text,
)
from tldw_Server_API.app.core.Media.navigation import (
    _strip_asterisk_emphasis as _strip_asterisk_emphasis,
)
from tldw_Server_API.app.core.Media.navigation import (
    _strip_html_tags_linear as _strip_html_tags_linear,
)
from tldw_Server_API.app.core.Media.navigation import (
    _strip_underscore_emphasis as _strip_underscore_emphasis,
)
from tldw_Server_API.app.core.Media.navigation import (
    _tighten_navigation_punctuation as _tighten_navigation_punctuation,
)
from tldw_Server_API.app.core.Media.navigation import (
    _to_float as _to_float,
)
from tldw_Server_API.app.core.Media.navigation import (
    _to_int as _to_int,
)
from tldw_Server_API.app.core.Media.navigation import (
    _truncate_multi_entry_title as _truncate_multi_entry_title,
)
from tldw_Server_API.app.core.Media.navigation import get_media_page_map, load_navigation_document
from tldw_Server_API.app.core.Media.page_provenance import page_spans_for_range

router = APIRouter(tags=["Document Workspace"])


@router.get(
    "/{media_id:int}/navigation",
    status_code=status.HTTP_200_OK,
    summary="Get Media Navigation Tree",
    response_model=MediaNavigationResponse,
    dependencies=[Depends(rbac_rate_limit(MEDIA_NAVIGATION_ROUTE_POLICY.rate_limit_resource))],
    responses={
        200: {"description": "Navigation tree retrieved"},
        404: {"description": "Media item not found"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Server error"},
    },
)
async def get_media_navigation(
    media_id: int = Path(..., description="The ID of the media item"),
    params: MediaNavigationQueryParams = Depends(),
    db: MediaNavigationDb = Depends(get_media_db_for_user),
    current_user: User = Depends(get_request_user),
) -> MediaNavigationResponse:
    """Return normalized chapter/section navigation for a media item."""
    logger.debug(
        "Fetching media navigation for media_id={}, user_id={}, max_depth={}, max_nodes={}, parent_id={}",
        media_id,
        getattr(current_user, "id", "?"),
        params.max_depth,
        params.max_nodes,
        params.parent_id,
    )

    try:
        media = db.get_media_by_id(media_id, include_deleted=False, include_trash=False)
    except Exception as exc:
        logger.error("Database error fetching media for navigation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while fetching media item",
        ) from exc

    if not media:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Media not found or is inactive/trashed",
        )

    media = load_navigation_document(media_id, media, db)
    cache_key = _build_navigation_cache_key(media_id, media, params)
    cached = get_cached_response(cache_key)
    if cached is not None:
        _etag, payload = cached
        try:
            return MediaNavigationResponse.model_validate(payload)
        except (TypeError, ValueError):
            logger.debug("Ignoring invalid cached navigation payload")

    raw_nodes, source_order_used = await _select_source_nodes(
        media_id=media_id,
        db=db,
        media=media,
        include_generated_fallback=bool(params.include_generated_fallback),
    )
    all_nodes = _materialize_navigation_nodes(raw_nodes)
    filtered_nodes, node_count, truncated, max_depth_seen = _filter_navigation_nodes(all_nodes, params)

    response = MediaNavigationResponse(
        media_id=media_id,
        available=bool(all_nodes),
        navigation_version=_compute_navigation_version(
            media_id=media_id,
            media=media,
            source_order_used=source_order_used,
            nodes=all_nodes,
        ),
        source_order_used=source_order_used,
        nodes=filtered_nodes,
        stats=MediaNavigationStats(
            returned_node_count=len(filtered_nodes),
            node_count=node_count,
            max_depth=max_depth_seen,
            truncated=truncated,
        ),
    )

    cache_response(cache_key, response.model_dump(mode="json"), media_id=media_id)
    return response


@router.get(
    "/{media_id:int}/navigation/{node_id}/content",
    status_code=status.HTTP_200_OK,
    summary="Get Media Navigation Node Content",
    response_model=MediaNavigationContentResponse,
    dependencies=[Depends(rbac_rate_limit(MEDIA_NAVIGATION_ROUTE_POLICY.rate_limit_resource))],
    responses={
        200: {"description": "Node content retrieved"},
        404: {"description": "Media item or navigation node not found"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Server error"},
    },
)
async def get_media_navigation_content(
    media_id: int = Path(..., description="The ID of the media item"),
    node_id: str = Path(..., min_length=1, description="Navigation node ID"),
    params: MediaNavigationContentQueryParams = Depends(),
    db: MediaNavigationDb = Depends(get_media_db_for_user),
    current_user: User = Depends(get_request_user),
) -> MediaNavigationContentResponse:
    """Return content payload for a selected navigation node."""
    logger.debug(
        "Fetching media navigation content for media_id={}, node_id={}, user_id={}, format={}, include_alternates={}",
        media_id,
        node_id,
        getattr(current_user, "id", "?"),
        params.format,
        params.include_alternates,
    )

    try:
        media = db.get_media_by_id(media_id, include_deleted=False, include_trash=False)
    except Exception as exc:
        logger.error("Database error fetching media for navigation content")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while fetching media item",
        ) from exc

    if not media:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Media not found or is inactive/trashed",
        )

    media = load_navigation_document(media_id, media, db)
    raw_nodes, source_order_used = await _select_source_nodes(
        media_id=media_id,
        db=db,
        media=media,
        include_generated_fallback=False,
    )
    all_nodes = _materialize_navigation_nodes(raw_nodes)
    navigation_version = _compute_navigation_version(
        media_id=media_id,
        media=media,
        source_order_used=source_order_used,
        nodes=all_nodes,
    )

    if params.navigation_version is not None and params.navigation_version != navigation_version:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error_code": "NAVIGATION_VERSION_CHANGED", "navigation_version": navigation_version},
        )

    node = _find_navigation_node(all_nodes, node_id)
    if node is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": "NAVIGATION_NODE_NOT_FOUND",
                "message": f"Node '{node_id}' not found in current navigation set.",
                "media_id": media_id,
                "node_id": node_id,
                "navigation_version": navigation_version,
            },
        )

    cache_key = _build_navigation_content_cache_key(
        media_id=media_id,
        media=media,
        node_id=node_id,
        navigation_version=navigation_version,
        params=params,
    )
    cached = get_cached_response(cache_key)
    if cached is not None:
        _etag, payload = cached
        try:
            return MediaNavigationContentResponse.model_validate(payload)
        except (TypeError, ValueError):
            logger.debug("Ignoring invalid cached navigation content payload")

    full_text = _get_media_text(media_id=media_id, media=media, db=db)
    span = _derive_content_span(
        node=node,
        all_nodes=all_nodes,
        media=media,
        db=db,
        media_id=media_id,
        content_length=len(full_text),
        content=full_text,
    )
    selected_text = _slice_text(full_text, span[0], span[1]) if span is not None else ""

    variants, intrinsic_formats = _build_content_variants(selected_text)
    resolved_format, resolved_content = _resolve_requested_content(
        requested_format=params.format,
        variants=variants,
        intrinsic_formats=intrinsic_formats,
    )

    alternate_content = None
    if params.include_alternates:
        alt_map = {fmt: variants[fmt] for fmt in intrinsic_formats if fmt != resolved_format and fmt in variants}
        alternate_content = alt_map or None

    response = MediaNavigationContentResponse(
        media_id=media_id,
        node_id=node.id,
        title=node.title,
        content_format=resolved_format,
        available_formats=intrinsic_formats,
        content=resolved_content,
        alternate_content=alternate_content,
        navigation_version=navigation_version,
        resolved_start_char=span[0] if span is not None else None,
        resolved_end_char=span[1] if span is not None else None,
        page_spans=page_spans_for_range(get_media_page_map(media, full_text, db), *span) if span else [],
        coverage_status="exact" if span is not None else "unavailable",
        target=MediaNavigationTarget(
            target_type=node.target_type,
            target_start=node.target_start,
            target_end=node.target_end,
            target_href=node.target_href,
        ),
    )

    cache_response(cache_key, response.model_dump(mode="json"), media_id=media_id)
    return response
