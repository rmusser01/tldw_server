import uuid as _uuid_mod
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from loguru import logger
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, rbac_rate_limit, RequirePermission, TokenScopeGuard, User

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.api.v1.schemas.notes_graph import (
    EdgeType,
    GraphFormat,
    NoteGraphRequest,
    NoteLinkCreate,
)
from tldw_Server_API.app.core.AuthNZ.permissions import NOTES_GRAPH_READ, NOTES_GRAPH_WRITE
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Notes_Graph.formatters import to_cytoscape
from tldw_Server_API.app.core.Notes_Graph.graph_service import (
    NOTES_GRAPH_ENABLED,
    NoteGraphService,
)

router = APIRouter()


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
    request: Request,
    req: NoteGraphRequest = Depends(),
    current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("notes.graph.read")),
    __: None = Depends(RequirePermission(NOTES_GRAPH_READ)),
    ___: None = Depends(TokenScopeGuard("notes", require_if_present=True, endpoint_id="notes.graph.read")),
):
    """Return a bounded subgraph of notes, tags, and sources."""
    if not NOTES_GRAPH_ENABLED():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Notes graph is disabled")
    try:
        # FastAPI Depends() may not propagate edge_types correctly; re-parse from query
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
        if getattr(req, "center_note_id", None):
            req.center_note_id = _normalize_note_id(req.center_note_id)
        service = NoteGraphService(user_id=str(current_user.id_str), db=db)
        graph = service.generate_graph(req)
        if req.format == GraphFormat.cytoscape:
            return to_cytoscape(graph)
        return graph
    except HTTPException:
        raise
    except (InputError, CharactersRAGDBError) as e:
        raise map_db_error_to_http(e, default_detail="Graph fetch failed") from e
    except Exception as e:
        logger.error("notes.graph.read failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Graph fetch failed") from e


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
        service = NoteGraphService(user_id=str(current_user.id_str), db=db)
        graph = service.generate_graph(req)
        if req.format == GraphFormat.cytoscape:
            return to_cytoscape(graph)
        return graph
    except HTTPException:
        raise
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
        principal = f"user:{current_user.id_str}"
        edge = db.create_manual_note_edge(
            user_id=str(current_user.id_str),
            from_note_id=from_note_id,
            to_note_id=to_note_id,
            directed=bool(link.directed),
            weight=link.weight if link.weight is not None else 1.0,
            metadata=link.metadata,
            created_by=principal,
        )
        return {"status": "created", "edge": edge}
    except ConflictError as e:
        raise map_db_error_to_http(e, conflict_detail="duplicate manual link") from e
    except (InputError, CharactersRAGDBError) as e:
        raise map_db_error_to_http(e, default_detail="Link creation failed") from e


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
        deleted = db.delete_manual_note_edge(user_id=str(current_user.id_str), edge_id=normalized_edge_id)
        return {"deleted": bool(deleted), "edge_id": normalized_edge_id}
    except (InputError, CharactersRAGDBError) as e:
        raise map_db_error_to_http(e, default_detail="Link deletion failed") from e
