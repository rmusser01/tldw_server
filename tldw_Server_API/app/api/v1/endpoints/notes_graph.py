import asyncio
import uuid as _uuid_mod
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from loguru import logger
from starlette.responses import Response

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    RequirePermission,
    TokenScopeGuard,
    User,
    enforce_rbac_rate_limit,
    get_db_pool,
    get_request_user,
    principal_has_admin_bypass_claims,
    rbac_rate_limit,
)
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints.notes_sync_errors import notes_sync_http_error
from tldw_Server_API.app.api.v1.schemas.notes_graph import (
    EdgeType,
    GraphFormat,
    NoteGraphRequest,
    NoteLinkCreate,
    NoteLinkRestore,
    NoteLinkUpdate,
)
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.core.AuthNZ.permissions import (
    NOTES_GRAPH_ADMIN,
    NOTES_GRAPH_READ,
    NOTES_GRAPH_SUGGEST,
    NOTES_GRAPH_WRITE,
    SYSTEM_CONFIGURE,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.DB_Management.chacha.note_link_store import NotesLink
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Notes_Graph.formatters import to_cytoscape
from tldw_Server_API.app.core.Notes_Graph.graph_cache import GraphCache
from tldw_Server_API.app.core.Notes_Graph.graph_service import (
    NOTES_GRAPH_ENABLED,
    GraphProjectionNotReadyError,
    NoteGraphService,
    decode_notes_link_cursor,
    encode_notes_link_cursor,
    notes_link_cursor_binding,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_api import (
    load_semantic_settings,
    resolve_semantic_capabilities,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_projector import (
    SemanticGraphProjector,
    SemanticProjectionError,
    build_projection_vector_store,
)
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.notes_link_coordinator import (
    NotesLinkDatasetConflictError,
    NotesLinkNotReadyError,
    NotesLinkPreflightError,
    NotesLinkResourceNotFoundError,
    NotesLinkSyncInactiveDatasetError,
    NotesLinkVersionConflictError,
    resolve_notes_link_coordinator,
    resolve_notes_link_dataset_authority,
)


def _semantic_edge_requested(request: Request) -> bool:
    return any(
        part.strip() == EdgeType.semantic.value
        for raw_value in request.query_params.getlist("edge_types")
        for part in raw_value.split(",")
    )


def _semantic_graph_query_present(request: Request) -> bool:
    return _semantic_edge_requested(request) or any(
        key in request.query_params
        for key in ("semantic_top_k", "semantic_threshold")
    )


def _invalid_semantic_graph_request() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail={
            "error_code": "notes_semantic_invalid_request",
            "message": "The semantic graph request is invalid.",
        },
    )


def _is_semantic_graph_validation_error(exc: RequestValidationError) -> bool:
    semantic_fields = {"edge_types", "semantic_top_k", "semantic_threshold"}
    for error in exc.errors():
        location = error.get("loc", ())
        if any(part in semantic_fields for part in location):
            return True
        if "semantic" in str(error.get("msg", "")).lower():
            return True
    return False


def _invalid_edge_type(request: Request) -> str | None:
    valid = {edge_type.value for edge_type in EdgeType}
    for raw_value in request.query_params.getlist("edge_types"):
        for part in raw_value.split(","):
            candidate = part.strip()
            if candidate and candidate not in valid:
                return candidate
    return None


class _NotesGraphRoute(APIRoute):
    """Map semantic graph validation failures to the stable feature contract."""

    def get_route_handler(self) -> Callable[[Request], Any]:
        original = super().get_route_handler()

        async def handler(request: Request) -> Response:
            try:
                return await original(request)
            except RequestValidationError as exc:
                if request.method == "GET" and self.path.endswith("/graph"):
                    if (
                        _semantic_graph_query_present(request)
                        and _is_semantic_graph_validation_error(exc)
                    ):
                        raise _invalid_semantic_graph_request() from exc
                    invalid_edge_type = _invalid_edge_type(request)
                    if invalid_edge_type is not None:
                        valid = [edge_type.value for edge_type in EdgeType]
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=(
                                f"Invalid edge_type: '{invalid_edge_type}'. "
                                f"Valid: {valid}"
                            ),
                        ) from exc
                raise

        return handler


router = APIRouter(route_class=_NotesGraphRoute)
_GRAPH_CACHE = GraphCache()


async def _enforce_graph_request_rate_limit(
    request: Request,
    db_pool: Any = Depends(get_db_pool),
) -> None:
    """Charge exactly one ordinary or semantic graph-read resource."""

    resource = (
        "notes.graph.semantic.read"
        if _semantic_edge_requested(request)
        else "notes.graph.read"
    )
    await enforce_rbac_rate_limit(request, resource, db_pool)


_enforce_graph_request_rate_limit._tldw_rate_limit_resources = (
    "notes.graph.read",
    "notes.graph.semantic.read",
)


def _build_semantic_graph_projector(
    *,
    owner_user_id: str,
    dataset_id: str,
    db: CharactersRAGDB,
    graph_service: NoteGraphService,
) -> SemanticGraphProjector:
    """Build a projector whose physical vector backend remains lazy."""

    settings = load_semantic_settings()

    async def vector_store_factory() -> object:
        config = await asyncio.to_thread(
            db.note_semantic_store.get_configuration,
            dataset_id,
        )
        backend_name = str(getattr(config, "vector_backend", "") or "")
        return await build_projection_vector_store(
            db=db,
            owner_user_id=owner_user_id,
            backend_name=backend_name,
            settings=settings,
        )

    return SemanticGraphProjector(
        owner_user_id=owner_user_id,
        dataset_id=dataset_id,
        db=db,
        graph_service=graph_service,
        cache=_GRAPH_CACHE,
        vector_store_factory=vector_store_factory,
        capability_resolver=lambda: resolve_semantic_capabilities(
            db,
            settings=settings,
        ),
        settings=settings,
    )


def _semantic_projection_http_error(exc: SemanticProjectionError) -> HTTPException:
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    message = "The semantic relationship is no longer available."
    if exc.code == "notes_semantic_conversion_owner_mismatch":
        status_code = status.HTTP_404_NOT_FOUND
    elif exc.code in {
        "notes_semantic_conversion_generation_stale",
        "notes_semantic_conversion_pair_mismatch",
        "notes_semantic_cursor_mismatch",
    }:
        status_code = status.HTTP_409_CONFLICT
    if exc.code == "notes_semantic_cursor_mismatch":
        message = "The semantic graph cursor is stale or mismatched."
    return HTTPException(
        status_code=status_code,
        detail={
            "error_code": exc.code,
            "message": message,
        },
    )


async def _audit_semantic_conversion(
    *,
    actor_user_id: str,
    source_note_id: str,
    target_note_id: str,
    generation_id: str,
    result: str,
) -> None:
    """Emit a bounded content-free conversion audit record."""

    from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import (
        get_or_create_audit_service_for_user_id_optional,
    )
    from tldw_Server_API.app.core.Audit.unified_audit_service import (
        AuditContext,
        AuditEventCategory,
        AuditEventType,
    )

    audit_service = await get_or_create_audit_service_for_user_id_optional(
        actor_user_id
    )
    await audit_service.log_event(
        event_type=AuditEventType.DATA_UPDATE,
        category=AuditEventCategory.DATA_MODIFICATION,
        context=AuditContext(user_id=actor_user_id),
        resource_type="notes_semantic_relationship",
        resource_id=source_note_id,
        action="notes_semantic.manual_conversion",
        result=result,
        metadata={
            "target_note_id": target_note_id,
            "generation_id": generation_id,
        },
    )
    await audit_service.flush(raise_on_failure=True)


def _link_response(link: NotesLink) -> dict[str, object]:
    """Return the complete public representation of one explicit link."""

    properties = dict(link.properties)
    metadata = dict(properties)
    if link.label is not None:
        metadata["label"] = link.label
    return {
        "id": link.edge_id,
        "edge_id": link.edge_id,
        "user_id": link.owner_user_id,
        "from_note_id": link.source_note_id,
        "to_note_id": link.target_note_id,
        "type": link.type,
        "directed": link.directed,
        "weight": link.weight,
        "label": link.label,
        "properties": properties,
        "metadata": metadata,
        "created_at": link.created_at,
        "last_modified": link.last_modified,
        "created_by": link.created_by,
        "version": link.version,
        "deleted": link.deleted,
        "deleted_at": link.deleted_at,
    }


def _link_summary(link: NotesLink) -> dict[str, object]:
    """Return the bounded public list shape without arbitrary metadata."""

    return {
        "id": link.edge_id,
        "edge_id": link.edge_id,
        "from_note_id": link.source_note_id,
        "to_note_id": link.target_note_id,
        "type": link.type,
        "directed": link.directed,
        "weight": link.weight,
        "label": link.label,
        "created_at": link.created_at,
        "last_modified": link.last_modified,
        "created_by": link.created_by,
        "version": link.version,
        "deleted": link.deleted,
        "deleted_at": link.deleted_at,
    }


def _graph_dataset_key(*, user_id: str, dataset_id: str | None) -> str:
    authority = resolve_notes_link_dataset_authority(
        user_id=user_id,
        dataset_id=dataset_id,
    )
    if authority is None:
        return f"legacy:{user_id}"
    return authority[1].dataset_id


def _link_error(exc: Exception) -> HTTPException:
    if isinstance(exc, NotesLinkResourceNotFoundError):
        return HTTPException(
            status_code=404,
            detail={"error_code": exc.error_code, "message": "The Notes link was not found."},
        )
    if isinstance(exc, NotesLinkVersionConflictError):
        return HTTPException(
            status_code=409,
            detail={
                "error_code": exc.error_code,
                "message": "The Notes link has changed; refresh and retry.",
            },
        )
    if isinstance(exc, NotesLinkNotReadyError):
        return HTTPException(
            status_code=409,
            detail={
                "error_code": exc.error_code,
                "message": "Notes link Sync is not ready for writes.",
                "state": exc.state,
            },
        )
    if isinstance(exc, NotesLinkPreflightError):
        return HTTPException(
            status_code=409,
            detail={
                "error_code": exc.error_code,
                "message": "The Notes link conflicts with canonical state.",
            },
        )
    if isinstance(exc, (NotesLinkDatasetConflictError, NotesLinkSyncInactiveDatasetError)):
        return HTTPException(
            status_code=409,
            detail={
                "error_code": exc.error_code,
                "message": "The requested Notes Sync dataset cannot authorize this link operation.",
            },
        )
    if isinstance(exc, SyncStoreError):
        return notes_sync_http_error(exc)
    if isinstance(exc, ConflictError):
        return map_db_error_to_http(exc, conflict_detail="duplicate manual link")
    if isinstance(exc, (InputError, CharactersRAGDBError)):
        return map_db_error_to_http(exc, default_detail="Notes link operation failed")
    return HTTPException(status_code=500, detail="Notes link operation failed")


def _has_existing_manual_link(
    db: CharactersRAGDB,
    *,
    source_note_id: str,
    target_note_id: str,
) -> bool:
    """Confirm that a duplicate conversion already has an authoritative link."""

    expected_pair = {source_note_id, target_note_id}
    return any(
        link.type == "manual"
        and not link.deleted
        and {link.source_note_id, link.target_note_id} == expected_pair
        for link in db.notes_link_store.list_for_notes(
            [source_note_id, target_note_id]
        )
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_note_id(raw_id: Optional[str]) -> str:
    if raw_id is None:
        raise HTTPException(status_code=400, detail="note_id is required")
    text = str(raw_id).strip()
    if not text:
        raise HTTPException(status_code=400, detail="note_id is required")
    if ":" in text:
        prefix, value = text.split(":", 1)
        if prefix != "note" or not value:
            raise HTTPException(status_code=400, detail="note_id must be a raw UUID or note:<uuid>")
        try:
            _uuid_mod.UUID(value)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid UUID format: {value}") from None
        return value
    return text


def _normalize_edge_id(raw_id: Optional[str]) -> str:
    if raw_id is None:
        raise HTTPException(status_code=400, detail="edge_id is required")
    text = str(raw_id).strip()
    if not text:
        raise HTTPException(status_code=400, detail="edge_id is required")
    if ":" in text:
        prefix, value = text.split(":", 1)
        if prefix not in {"e", "edge"} or not value:
            raise HTTPException(status_code=400, detail="edge_id must be a raw UUID or e:<uuid>")
        try:
            _uuid_mod.UUID(value)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid UUID format: {value}") from None
        return value
    return text


def _can_use_heavy_graph_limits(user: User) -> bool:
    roles = {str(role).strip().lower() for role in (getattr(user, "roles", []) or [])}
    legacy_role = str(getattr(user, "role", "") or "").strip().lower()
    if legacy_role:
        roles.add(legacy_role)
    permissions = {str(perm).strip().lower() for perm in (getattr(user, "permissions", []) or [])}
    admin_permissions = {"*", SYSTEM_CONFIGURE.lower(), NOTES_GRAPH_ADMIN.lower()}
    return bool(
        getattr(user, "is_admin", False)
        or getattr(user, "is_superuser", False)
        or "admin" in roles
        or bool(permissions & admin_permissions)
    )


def _enforce_heavy_graph_permission(req: NoteGraphRequest, current_user: User) -> bool:
    if not req.allow_heavy:
        return False
    if _can_use_heavy_graph_limits(current_user):
        return True
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=f"{NOTES_GRAPH_ADMIN} permission is required for allow_heavy graph requests",
    )


def _suggestions_authorized(principal: AuthPrincipal | None) -> bool:
    """Return request-local suggestion authority from verified principal claims."""

    if principal_has_admin_bypass_claims(principal):
        return True
    permissions = {
        str(permission).strip()
        for permission in (getattr(principal, "permissions", None) or ())
    }
    return NOTES_GRAPH_SUGGEST in permissions


def _manual_link_authorized(principal: AuthPrincipal | None) -> bool:
    """Return request-local manual-link authority from verified claims."""

    if principal_has_admin_bypass_claims(principal):
        return True
    permissions = {
        str(permission).strip()
        for permission in (getattr(principal, "permissions", None) or ())
    }
    return NOTES_GRAPH_WRITE in permissions


@router.get(
    "/graph",
    summary="Fetch a graph of notes and related entities",
    description=(
        "Returns a bounded subgraph of notes, tags, and sources based on filters.\n\n"
        "- Honors enum edge_types: manual, wikilink, backlink, tag_membership, source_membership.\n"
        "- Uses BFS with deterministic ordering; see Docs/Design/Graphing-Notes-PRD.md §21 for cursor details.\n\n"
        "Example response (default format) matches the NoteGraphResponse schema.\n\n"
        "Cytoscape example (when format=cytoscape) is documented in Docs/Design/Graphing-Notes-PRD.md (§9, §14)."
    ),
    tags=["notes", "notes-graph"],
    openapi_extra={
        "x-codeSamples": [
            {
                "lang": "bash",
                "label": "curl",
                "source": "curl -H 'Authorization: Bearer <token>' 'http://127.0.0.1:8000/api/v1/notes/graph?center_note_id=note:123&radius=1&edge_types=manual,wikilink,tag_membership&max_nodes=200'",
            },
            {
                "lang": "python",
                "label": "urllib",
                "source": "import json\nfrom urllib.parse import urlencode\nfrom urllib.request import Request, urlopen\n\nparams = {\n    \"center_note_id\": \"note:123\",\n    \"radius\": 1,\n    \"edge_types\": \"manual,wikilink,tag_membership\",\n}\nurl = \"http://127.0.0.1:8000/api/v1/notes/graph?\" + urlencode(params)\nreq = Request(url, headers={\"Authorization\": \"Bearer <token>\"})\nwith urlopen(req) as resp:\n    print(json.load(resp))",
            },
        ]
    },
)
async def get_notes_graph(
    req: Annotated[NoteGraphRequest, Query()],
    current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(_enforce_graph_request_rate_limit),
    principal: AuthPrincipal = Depends(RequirePermission(NOTES_GRAPH_READ)),
    ___: None = Depends(TokenScopeGuard("notes", require_if_present=True, endpoint_id="notes.graph.read")),
):
    """Return a bounded subgraph of notes, tags, and sources."""
    if not NOTES_GRAPH_ENABLED():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Notes graph is disabled")
    try:
        if getattr(req, "center_note_id", None):
            req.center_note_id = _normalize_note_id(req.center_note_id)
        heavy_limits_allowed = _enforce_heavy_graph_permission(req, current_user)
        dataset_key = _graph_dataset_key(
            user_id=str(current_user.id_str),
            dataset_id=req.dataset_id,
        )
        service = NoteGraphService(
            user_id=str(current_user.id_str),
            dataset_id=dataset_key,
            db=db,
            cache=_GRAPH_CACHE,
            allow_heavy_limits=heavy_limits_allowed,
        )
        graph = service.generate_graph(req)
        if req.semantic_requested:
            projector = _build_semantic_graph_projector(
                owner_user_id=str(current_user.id_str),
                dataset_id=dataset_key,
                db=db,
                graph_service=service,
            )
            graph = await projector.project(req, graph, user=current_user)
        graph = graph.model_copy(
            update={
                "suggestions_authorized": _suggestions_authorized(principal),
                "manual_link_authorized": _manual_link_authorized(principal),
            }
        )
        if req.format == GraphFormat.cytoscape:
            formatted = to_cytoscape(graph)
            formatted.update(
                {
                    "active_note_count": graph.active_note_count,
                    "all_notes_note_cap": graph.all_notes_note_cap,
                    "all_notes_eligible": graph.all_notes_eligible,
                    "suggestions_authorized": graph.suggestions_authorized,
                    "manual_link_authorized": graph.manual_link_authorized,
                }
            )
            return formatted
        return graph
    except HTTPException:
        raise
    except GraphProjectionNotReadyError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": "notes_graph_projection_not_ready",
                "message": str(e),
            },
        ) from e
    except SemanticProjectionError as exc:
        raise _semantic_projection_http_error(exc) from exc
    except SyncStoreError as e:
        raise _link_error(e) from e
    except (InputError, CharactersRAGDBError) as e:
        raise map_db_error_to_http(e, default_detail="Graph fetch failed") from e
    except Exception as e:
        logger.error("notes.graph.read failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Graph fetch failed") from e


@router.get(
    "/graph/orphans",
    summary="List live Notes without note-to-note relationships",
    tags=["notes", "notes-graph"],
)
async def list_orphan_notes(
    dataset_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None),
    current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("notes.graph.read")),
    __: None = Depends(RequirePermission(NOTES_GRAPH_READ)),
    ___: None = Depends(TokenScopeGuard("notes", require_if_present=True, endpoint_id="notes.graph.read")),
) -> dict[str, object]:
    """Return a revision-bound keyset page; tags and sources do not disqualify notes."""

    try:
        owner_id = str(current_user.id_str)
        dataset_key = _graph_dataset_key(user_id=owner_id, dataset_id=dataset_id)
        service = NoteGraphService(
            user_id=owner_id,
            dataset_id=dataset_key,
            db=db,
            cache=_GRAPH_CACHE,
        )
        notes, has_more, next_cursor = service.list_orphans(
            limit=limit,
            cursor=cursor,
        )
        return {
            "notes": [note.model_dump(mode="json") for note in notes],
            "has_more": has_more,
            "next_cursor": next_cursor,
        }
    except GraphProjectionNotReadyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": "notes_graph_projection_not_ready",
                "message": str(exc),
            },
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - use the established graph/link mapping.
        raise _link_error(exc) from exc


@router.get(
    "/{note_id}/neighbors",
    summary="Fetch neighbors for a note",
    description=(
        "Returns a radius=1 ego network for the given note. Uses the same filters, limits, and ordering as /graph.\n"
        "See Docs/Design/Graphing-Notes-PRD.md (§9, §10, §21)."
    ),
    tags=["notes", "notes-graph"],
    openapi_extra={
        "x-codeSamples": [
            {
                "lang": "bash",
                "label": "curl",
                "source": "curl -H 'Authorization: Bearer <token>' 'http://127.0.0.1:8000/api/v1/notes/note:123/neighbors?edge_types=manual,backlink'",
            }
        ]
    },
)
async def get_note_neighbors(
    note_id: str,
    request: Request,
    req: NoteGraphRequest = Depends(),
    current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("notes.graph.read")),
    __: None = Depends(RequirePermission(NOTES_GRAPH_READ)),
    ___: None = Depends(TokenScopeGuard("notes", require_if_present=True, endpoint_id="notes.graph.read")),
):
    """Return a radius=1 ego network for the given note."""
    if not NOTES_GRAPH_ENABLED():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Notes graph is disabled")
    try:
        # Re-parse edge_types from query params (FastAPI Depends may not handle list[Enum] well)
        raw_et = request.query_params.getlist("edge_types")
        if raw_et:
            parsed = []
            for val in raw_et:
                for part in val.split(","):
                    part = part.strip()
                    if part:
                        try:
                            parsed.append(EdgeType(part))
                        except ValueError:
                            raise HTTPException(
                                status_code=400,
                                detail=f"Invalid edge_type: '{part}'. Valid: {[e.value for e in EdgeType]}",
                            ) from None
            req.edge_types = parsed or None
        normalized_note_id = _normalize_note_id(note_id)
        req.center_note_id = normalized_note_id
        req.radius = 1
        heavy_limits_allowed = _enforce_heavy_graph_permission(req, current_user)
        dataset_key = _graph_dataset_key(
            user_id=str(current_user.id_str),
            dataset_id=req.dataset_id,
        )
        service = NoteGraphService(
            user_id=str(current_user.id_str),
            dataset_id=dataset_key,
            db=db,
            cache=_GRAPH_CACHE,
            allow_heavy_limits=heavy_limits_allowed,
        )
        graph = service.generate_graph(req)
        if req.format == GraphFormat.cytoscape:
            return to_cytoscape(graph)
        return graph
    except HTTPException:
        raise
    except GraphProjectionNotReadyError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": "notes_graph_projection_not_ready",
                "message": str(e),
            },
        ) from e
    except SyncStoreError as e:
        raise _link_error(e) from e
    except (InputError, CharactersRAGDBError) as e:
        raise map_db_error_to_http(e, default_detail="Graph fetch failed") from e
    except Exception as e:
        logger.error("notes.graph.neighbors failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Graph fetch failed") from e


@router.post(
    "/{note_id}/links",
    summary="Create a manual link between notes",
    description=(
        "Creates a manual link from the given note to another note. Undirected by default (directed=false).\n"
        "See Docs/Design/Graphing-Notes-PRD.md (§8, §9, §10)."
    ),
    tags=["notes", "notes-graph"],
    responses={
        200: {
            "description": "Creation result",
            "content": {
                "application/json": {
                    "example": {
                        "status": "created",
                        "edge": {"edge_id": "...", "from_note_id": "123", "to_note_id": "456"},
                    }
                }
            },
        }
    },
    openapi_extra={
        "x-codeSamples": [
            {
                "lang": "bash",
                "label": "curl",
                "source": "curl -X POST -H 'Authorization: Bearer <token>' -H 'Content-Type: application/json' \\\n+  -d '{\"to_note_id\":\"note:456\",\"directed\":false,\"weight\":1.0}' \\\n+  'http://127.0.0.1:8000/api/v1/notes/note:123/links'",
            }
        ]
    },
)
async def create_manual_link(
    note_id: str,
    link: NoteLinkCreate = Body(
        ..., example={"to_note_id": "note:456", "directed": False, "weight": 1.0}
    ),
    current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("notes.graph.write")),
    __: None = Depends(RequirePermission(NOTES_GRAPH_WRITE)),
    ___: None = Depends(TokenScopeGuard("notes", require_if_present=True, endpoint_id="notes.graph.write")),
) -> dict[str, Any]:
    """
    Create a manual link in the user's ChaChaNotes DB. Populates created_by.
    """
    from_note_id = _normalize_note_id(note_id)
    to_note_id = _normalize_note_id(link.to_note_id)

    try:
        directed = bool(link.directed)
        weight = link.weight if link.weight is not None else 1.0
        conversion = link.semantic_conversion
        if conversion is not None:
            owner_user_id = str(current_user.id_str)
            dataset_key = _graph_dataset_key(
                user_id=owner_user_id,
                dataset_id=link.dataset_id,
            )
            service = NoteGraphService(
                user_id=owner_user_id,
                dataset_id=dataset_key,
                db=db,
                cache=_GRAPH_CACHE,
            )
            projector = _build_semantic_graph_projector(
                owner_user_id=owner_user_id,
                dataset_id=dataset_key,
                db=db,
                graph_service=service,
            )
            try:
                await projector.validate_conversion(
                    source_note_id=from_note_id,
                    target_note_id=to_note_id,
                    generation_id=conversion.generation_id,
                )
            except SemanticProjectionError as exc:
                raise _semantic_projection_http_error(exc) from exc
            directed = False
            weight = 1.0
        coordinator = resolve_notes_link_coordinator(
            user_id=str(current_user.id_str),
            note_db=db,
            dataset_id=link.dataset_id,
        )
        if coordinator is not None:
            edge = coordinator.create(
                source_note_id=from_note_id,
                target_note_id=to_note_id,
                directed=directed,
                weight=weight,
                label=link.label,
                properties=link.properties or {},
                idempotency_key=link.idempotency_key,
            )
            edge_response = _link_response(edge)
        else:
            legacy_metadata = dict(link.properties or {})
            if link.label is not None:
                legacy_metadata["label"] = link.label
            edge_response = db.create_manual_note_edge(
                user_id=str(current_user.id_str),
                from_note_id=from_note_id,
                to_note_id=to_note_id,
                directed=directed,
                weight=weight,
                metadata=legacy_metadata,
                created_by=f"user:{current_user.id_str}",
            )
            stored_edge_id = str(edge_response.get("edge_id") or "")
            if stored_edge_id and hasattr(db, "notes_link_store"):
                stored = db.notes_link_store.get(stored_edge_id)
                if stored is not None:
                    edge_response = _link_response(stored)
        if conversion is not None:
            try:
                await _audit_semantic_conversion(
                    actor_user_id=str(current_user.id_str),
                    source_note_id=from_note_id,
                    target_note_id=to_note_id,
                    generation_id=conversion.generation_id,
                    result="created",
                )
            except Exception:  # noqa: BLE001 - the link is already authoritative.
                logger.warning("Notes semantic conversion audit emission failed")
        return {"status": "created", "edge": edge_response}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - map the closed link error contract.
        if link.semantic_conversion is not None and isinstance(
            exc,
            (ConflictError, NotesLinkPreflightError),
        ):
            try:
                manual_link_exists = _has_existing_manual_link(
                    db,
                    source_note_id=from_note_id,
                    target_note_id=to_note_id,
                )
            except (CharactersRAGDBError, InputError):
                manual_link_exists = False
            if manual_link_exists:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail={
                        "error_code": (
                            "notes_semantic_conversion_manual_link_exists"
                        ),
                        "message": "A manual Notes link already exists.",
                    },
                ) from exc
        raise _link_error(exc) from exc


@router.get(
    "/links",
    summary="List explicit manual Notes links",
    tags=["notes", "notes-graph"],
)
async def list_manual_links(
    dataset_id: str | None = Query(default=None),
    include_deleted: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None),
    current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("notes.graph.read")),
    __: None = Depends(RequirePermission(NOTES_GRAPH_READ)),
    ___: None = Depends(TokenScopeGuard("notes", require_if_present=True, endpoint_id="notes.graph.read")),
) -> dict[str, object]:
    """List a bounded keyset page of live owner-scoped explicit links."""

    try:
        owner_id = str(current_user.id_str)
        dataset_key = _graph_dataset_key(
            user_id=owner_id,
            dataset_id=dataset_id,
        )
        binding = notes_link_cursor_binding(
            db=db,
            dataset_key=dataset_key,
            include_deleted=include_deleted,
            limit=limit,
        )
        after_edge_id = decode_notes_link_cursor(cursor, expected=binding)
        links = db.notes_link_store.list_page(
            after_edge_id=after_edge_id,
            limit=limit + 1,
            include_deleted_links=include_deleted,
            include_deleted_endpoints=False,
        )
        has_more = len(links) > limit
        page = links[:limit]
        next_cursor = None
        if has_more and page:
            next_cursor = encode_notes_link_cursor(
                payload={**binding, "last_id": page[-1].edge_id}
            )
        return {
            "links": [_link_summary(link) for link in page],
            "has_more": has_more,
            "next_cursor": next_cursor,
        }
    except Exception as exc:  # noqa: BLE001 - map the closed link error contract.
        raise _link_error(exc) from exc


@router.get(
    "/links/{edge_id}",
    summary="Fetch one explicit manual Notes link",
    tags=["notes", "notes-graph"],
)
async def get_manual_link(
    edge_id: str,
    dataset_id: str | None = Query(default=None),
    current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("notes.graph.read")),
    __: None = Depends(RequirePermission(NOTES_GRAPH_READ)),
    ___: None = Depends(TokenScopeGuard("notes", require_if_present=True, endpoint_id="notes.graph.read")),
) -> dict[str, object]:
    """Return one owner-scoped explicit link, including tombstone state."""

    try:
        normalized_edge_id = _normalize_edge_id(edge_id)
        _graph_dataset_key(
            user_id=str(current_user.id_str),
            dataset_id=dataset_id,
        )
        link = db.notes_link_store.get(normalized_edge_id)
        if link is None:
            raise NotesLinkResourceNotFoundError()
        return _link_response(link)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - map the closed link error contract.
        raise _link_error(exc) from exc


@router.patch(
    "/links/{edge_id}",
    summary="Update one explicit manual Notes link",
    tags=["notes", "notes-graph"],
)
async def update_manual_link(
    edge_id: str,
    mutation: NoteLinkUpdate,
    current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("notes.graph.write")),
    __: None = Depends(RequirePermission(NOTES_GRAPH_WRITE)),
    ___: None = Depends(TokenScopeGuard("notes", require_if_present=True, endpoint_id="notes.graph.write")),
) -> dict[str, object]:
    """Update mutable link presentation fields with optimistic concurrency."""

    try:
        normalized_edge_id = _normalize_edge_id(edge_id)
        coordinator = resolve_notes_link_coordinator(
            user_id=str(current_user.id_str),
            note_db=db,
            dataset_id=mutation.dataset_id,
        )
        if coordinator is not None and mutation.expected_version is None:
            raise HTTPException(
                status_code=status.HTTP_428_PRECONDITION_REQUIRED,
                detail={
                    "error_code": "notes_link_expected_version_required",
                    "message": "expected_version is required while Notes Sync is active.",
                },
            )
        current = (
            coordinator.get(normalized_edge_id)
            if coordinator is not None
            else db.notes_link_store.get(normalized_edge_id)
        )
        if current is None:
            raise NotesLinkResourceNotFoundError()
        weight = mutation.weight if "weight" in mutation.model_fields_set else current.weight
        label = mutation.label if "label" in mutation.model_fields_set else current.label
        properties = (
            mutation.properties
            if "properties" in mutation.model_fields_set
            else dict(current.properties)
        )
        if coordinator is not None:
            updated = coordinator.update(
                edge_id=normalized_edge_id,
                expected_version=int(mutation.expected_version),
                weight=float(weight),
                label=label,
                properties=properties or {},
                idempotency_key=mutation.idempotency_key,
            )
        else:
            timestamp = _utc_now()
            updated = db.notes_link_store.upsert(
                edge_id=normalized_edge_id,
                payload={
                    "source_note_id": current.source_note_id,
                    "target_note_id": current.target_note_id,
                    "type": current.type,
                    "directed": current.directed,
                    "weight": float(weight),
                    "label": label,
                    "properties": properties or {},
                    "created_at": current.created_at,
                    "last_modified": timestamp,
                    "created_by": current.created_by,
                },
                expected_version=(mutation.expected_version or current.version),
            ).link
        return _link_response(updated)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - map the closed link error contract.
        raise _link_error(exc) from exc


@router.delete(
    "/links/{edge_id}",
    summary="Delete a manual link",
    description=(
        "Deletes a manual link by id.\n"
        "See Docs/Design/Graphing-Notes-PRD.md (§9)."
    ),
    tags=["notes", "notes-graph"],
    responses={
        200: {
            "description": "Deletion result",
            "content": {
                "application/json": {
                    "example": {"deleted": True, "edge_id": "e:1"}
                }
            },
        }
    },
    openapi_extra={
        "x-codeSamples": [
            {
                "lang": "bash",
                "label": "curl",
                "source": "curl -X DELETE -H 'Authorization: Bearer <token>' 'http://127.0.0.1:8000/api/v1/notes/links/e:1'",
            }
        ]
    },
)
async def delete_manual_link(
    edge_id: str,
    dataset_id: str | None = Query(default=None),
    expected_version: int | None = Query(default=None, ge=1),
    idempotency_key: str | None = Query(default=None, min_length=1, max_length=128),
    reason: str | None = Query(default=None, max_length=256),
    current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("notes.graph.write")),
    __: None = Depends(RequirePermission(NOTES_GRAPH_WRITE)),
    ___: None = Depends(TokenScopeGuard("notes", require_if_present=True, endpoint_id="notes.graph.write")),
) -> dict[str, Any]:
    """
    Delete a manual link by id for the current user.
    """
    try:
        normalized_edge_id = _normalize_edge_id(edge_id)
        coordinator = resolve_notes_link_coordinator(
            user_id=str(current_user.id_str),
            note_db=db,
            dataset_id=dataset_id,
        )
        if coordinator is not None:
            if expected_version is None:
                raise HTTPException(
                    status_code=status.HTTP_428_PRECONDITION_REQUIRED,
                    detail={
                        "error_code": "notes_link_expected_version_required",
                        "message": "expected_version is required while Notes Sync is active.",
                    },
                )
            deleted = coordinator.tombstone(
                edge_id=normalized_edge_id,
                expected_version=expected_version,
                reason=reason,
                idempotency_key=idempotency_key,
            )
            return {
                "deleted": deleted.deleted,
                "edge_id": normalized_edge_id,
                "edge": _link_response(deleted),
            }
        deleted = db.delete_manual_note_edge(
            user_id=str(current_user.id_str),
            edge_id=normalized_edge_id,
        )
        return {"deleted": bool(deleted), "edge_id": normalized_edge_id}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - map the closed link error contract.
        raise _link_error(exc) from exc


@router.post(
    "/links/{edge_id}/restore",
    summary="Restore one tombstoned explicit Notes link",
    tags=["notes", "notes-graph"],
)
async def restore_manual_link(
    edge_id: str,
    mutation: NoteLinkRestore,
    current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("notes.graph.write")),
    __: None = Depends(RequirePermission(NOTES_GRAPH_WRITE)),
    ___: None = Depends(TokenScopeGuard("notes", require_if_present=True, endpoint_id="notes.graph.write")),
) -> dict[str, object]:
    """Restore a durable explicit link without changing its identity."""

    try:
        normalized_edge_id = _normalize_edge_id(edge_id)
        coordinator = resolve_notes_link_coordinator(
            user_id=str(current_user.id_str),
            note_db=db,
            dataset_id=mutation.dataset_id,
        )
        if coordinator is not None and mutation.expected_version is None:
            raise HTTPException(
                status_code=status.HTTP_428_PRECONDITION_REQUIRED,
                detail={
                    "error_code": "notes_link_expected_version_required",
                    "message": "expected_version is required while Notes Sync is active.",
                },
            )
        current = (
            coordinator.get(normalized_edge_id)
            if coordinator is not None
            else db.notes_link_store.get(normalized_edge_id)
        )
        if current is None:
            raise NotesLinkResourceNotFoundError()
        if coordinator is not None:
            restored = coordinator.restore(
                edge_id=normalized_edge_id,
                expected_version=int(mutation.expected_version),
                idempotency_key=mutation.idempotency_key,
            )
        else:
            timestamp = _utc_now()
            restored = db.notes_link_store.restore(
                edge_id=normalized_edge_id,
                payload={
                    "source_note_id": current.source_note_id,
                    "target_note_id": current.target_note_id,
                    "type": current.type,
                    "directed": current.directed,
                    "weight": current.weight,
                    "label": current.label,
                    "properties": dict(current.properties),
                    "created_at": current.created_at,
                    "last_modified": timestamp,
                    "created_by": current.created_by,
                },
                expected_version=(mutation.expected_version or current.version),
                allow_deleted_endpoints=True,
            ).link
        return {"restored": True, "edge": _link_response(restored)}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - map the closed link error contract.
        raise _link_error(exc) from exc
