"""Authenticated acceptance and observation of process-owned Buddy work."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.schemas.buddy_turns import BuddyTurn, BuddyTurnCreate, BuddyTurnList
from tldw_Server_API.app.core.Buddy import turns
from tldw_Server_API.app.core.Buddy.service import BuddyService
from tldw_Server_API.app.core.Buddy.turns import BuddyTurnRuntime
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.exceptions import (
    BuddyConfigurationError,
    BuddyConflictError,
    BuddyNotFoundError,
    BuddyQueueFullError,
    BuddyRuntimeBusyError,
)


@asynccontextmanager
async def _lifespan(app: Any) -> AsyncIterator[None]:
    """Close process-owned work when the composed router shuts down."""
    yield
    runtime = getattr(app.state, "buddy_turn_runtime", None)
    if runtime is not None:
        await runtime.close()


router = APIRouter(prefix="/turns", lifespan=_lifespan)
ClientSlot = Annotated[str, Query(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")]


def _service(
    user: User = Depends(get_request_user), db: CharactersRAGDB = Depends(get_chacha_db_for_user)
) -> BuddyService:
    if user.id is None:
        raise HTTPException(401, "Authentication required")
    return BuddyService(db, str(user.id))


@contextmanager
def _errors() -> Iterator[None]:
    """Map expected Buddy domain failures without hiding implementation errors."""
    try:
        yield
    except BuddyNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except BuddyConflictError as exc:
        raise HTTPException(409, str(exc)) from exc
    except BuddyQueueFullError as exc:
        raise HTTPException(429, str(exc)) from exc
    except BuddyRuntimeBusyError as exc:
        raise HTTPException(503, str(exc), headers={"Retry-After": "5"}) from exc
    except BuddyConfigurationError as exc:
        raise HTTPException(422, str(exc)) from exc


def _response(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in BuddyTurn.model_fields}


@router.post("", response_model=BuddyTurn, status_code=202)
async def accept_turn(
    body: BuddyTurnCreate,
    request: Request,
    client_slot: ClientSlot = "default",
    service: BuddyService = Depends(_service),
) -> dict[str, Any]:
    """Retain work before returning; browser disconnect is never Stop."""
    runtime = getattr(request.app.state, "buddy_turn_runtime", None)
    if runtime is None or runtime.closed:
        runtime = BuddyTurnRuntime(request.app)
        request.app.state.buddy_turn_runtime = runtime
    # Forward only admission context, retaining trusted-proxy handling and CSRF.
    headers = {
        name: ",".join(request.headers.getlist(name))
        for name in (
            "authorization",
            "x-api-key",
            "api-key",
            "cookie",
            "x-csrf-token",
            "origin",
            "host",
            "user-agent",
            "forwarded",
            "x-forwarded-for",
            "x-real-ip",
            "x-forwarded-host",
            "x-forwarded-proto",
            "x-forwarded-port",
        )
        if request.headers.getlist(name)
    }
    client = (request.client.host, request.client.port) if request.client else None
    with _errors():
        return _response(await runtime.accept(service, client_slot, body, headers, client))


@router.get("", response_model=BuddyTurnList)
async def list_turns(
    client_slot: ClientSlot = "default",
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: Literal["active"] | None = Query(None),
    service: BuddyService = Depends(_service),
) -> dict[str, Any]:
    """Read the principal's ledger even after a routing preference is detached."""
    with _errors():
        rows = await turns.list_turns(service, client_slot, limit, offset, active_only=status == "active")
        return {"turns": [_response(row) for row in rows], "limit": limit, "offset": offset}


@router.get("/{turn_id}", response_model=BuddyTurn)
async def get_turn(turn_id: str, service: BuddyService = Depends(_service)) -> dict[str, Any]:
    """Return one turn's safe status fields for the authenticated principal."""
    with _errors():
        return _response(await turns.get_turn(service, turn_id))


@router.post("/{turn_id}/stop", response_model=BuddyTurn)
async def stop_turn(turn_id: str, request: Request, service: BuddyService = Depends(_service)) -> dict[str, Any]:
    """Revoke publication without retrying or claiming to undo provider effects."""
    with _errors():
        runtime = getattr(request.app.state, "buddy_turn_runtime", None)
        return _response(await turns.stop_turn(service, turn_id, runtime))
