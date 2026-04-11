# character_memory.py
"""
API endpoints for cross-session character memory.

Provides CRUD for persistent, categorized memories scoped to (user, character)
pairs, plus a manual LLM-based extraction trigger.

Memories are stored in the existing ``persona_memory_entries`` table, keyed by
a lightweight ``persona_profiles`` row created on first use (convention:
``id = "char:{character_id}"``).
"""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.schemas.character_memory_schemas import (
    CharacterMemoryArchiveRequest,
    CharacterMemoryCreate,
    CharacterMemoryExtractRequest,
    CharacterMemoryExtractResponse,
    CharacterMemoryListResponse,
    CharacterMemoryResponse,
    CharacterMemoryUpdate,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    InputError,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _persona_id_for_character(character_id: str) -> str:
    """Deterministic persona_id for a character."""
    return f"char:{character_id}"


def _character_id_from_persona(persona_id: str) -> str:
    """Extract character_id from a ``char:{id}`` persona_id."""
    if persona_id.startswith("char:"):
        return persona_id[5:]
    return persona_id


def _ids_match(left: Any, right: Any) -> bool:
    """Compare ownership identifiers using int coercion first, then normalized strings."""
    try:
        return int(left) == int(right)
    except (TypeError, ValueError):
        return str(left).strip() == str(right).strip()


def get_or_create_character_persona_profile(
    db: CharactersRAGDB,
    character_id: str,
    character_name: str,
    user_id: str,
) -> str:
    """Ensure a persona_profiles row exists for this (user, character) pair.

    Creates a minimal profile if absent.  Returns the ``persona_id``.
    """
    persona_id = _persona_id_for_character(character_id)
    existing = db.get_persona_profile(persona_id, user_id=user_id)
    if existing:
        return persona_id

    try:
        db.create_persona_profile({
            "id": persona_id,
            "user_id": user_id,
            "name": f"char_memory:{character_id}",
            "origin_character_id": int(character_id) if str(character_id).isdigit() else None,
            "origin_character_name": character_name,
            "mode": "persistent_scoped",
            "is_active": True,
        })
    except Exception as exc:
        # Race condition: another request may have created it
        existing = db.get_persona_profile(persona_id, user_id=user_id)
        if existing:
            return persona_id
        logger.error("Failed to create persona profile for character {}: {}", character_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize character memory storage.",
        ) from exc

    return persona_id


def _fetch_memory_by_id(
    db: CharactersRAGDB, user_id: str, persona_id: str, memory_id: str, character_id: str,
) -> CharacterMemoryResponse:
    """Fetch a single memory entry by ID or raise 404."""
    row = db.get_persona_memory_entry_by_id(
        entry_id=memory_id, user_id=user_id, persona_id=persona_id,
    )
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")
    return _row_to_response(row, character_id)


def _row_to_response(row: dict[str, Any], character_id: str) -> CharacterMemoryResponse:
    """Convert a persona_memory_entries row dict to API response."""
    return CharacterMemoryResponse(
        id=row["id"],
        character_id=character_id,
        memory_type=row.get("memory_type", "manual"),
        content=row.get("content", ""),
        salience=float(row.get("salience", 0.0)),
        source_conversation_id=row.get("source_conversation_id"),
        archived=bool(row.get("archived", False)),
        created_at=row.get("created_at", ""),
        last_modified=row.get("last_modified", ""),
    )


def _resolve_character(db: CharactersRAGDB, character_id: str) -> dict[str, Any]:
    """Load the character card or raise 404."""
    try:
        cid = int(character_id)
    except (TypeError, ValueError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="character_id must be an integer")
    card = db.get_character_card_by_id(cid)
    if not card:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Character {character_id} not found")
    return card


# ---------------------------------------------------------------------------
# CRUD endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/{character_id}/memories",
    response_model=CharacterMemoryListResponse,
    summary="List character memories",
)
async def list_character_memories(
    character_id: str = Path(...),
    memory_type: str | None = Query(None, description="Filter by category"),
    include_archived: bool = Query(False),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    _resolve_character(db, character_id)
    persona_id = _persona_id_for_character(character_id)
    user_id = str(current_user.id)

    rows = db.list_persona_memory_entries(
        user_id=user_id,
        persona_id=persona_id,
        memory_type=memory_type,
        include_archived=include_archived,
        include_deleted=False,
        limit=limit,
        offset=offset,
    )
    total_count = db.count_persona_memory_entries(
        user_id=user_id,
        persona_id=persona_id,
        memory_type=memory_type,
        include_archived=include_archived,
        include_deleted=False,
    )
    memories = [_row_to_response(r, character_id) for r in rows]
    return CharacterMemoryListResponse(memories=memories, total=total_count)


@router.post(
    "/{character_id}/memories",
    response_model=CharacterMemoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a character memory",
)
async def create_character_memory(
    character_id: str = Path(...),
    body: CharacterMemoryCreate = ...,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    card = _resolve_character(db, character_id)
    user_id = str(current_user.id)
    char_name = card.get("name", "Unknown")

    persona_id = get_or_create_character_persona_profile(
        db, character_id, char_name, user_id,
    )

    try:
        entry_id = db.add_persona_memory_entry({
            "persona_id": persona_id,
            "user_id": user_id,
            "memory_type": body.memory_type,
            "content": body.content,
            "salience": body.salience,
        })
    except InputError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return _fetch_memory_by_id(db, user_id, persona_id, entry_id, character_id)


# ---------------------------------------------------------------------------
# Manual extraction endpoint (must be before /{memory_id} routes)
# ---------------------------------------------------------------------------

@router.post(
    "/{character_id}/memories/extract",
    response_model=CharacterMemoryExtractResponse,
    summary="Extract memories from a chat session via LLM",
)
async def extract_character_memories_endpoint(
    character_id: str = Path(...),
    body: CharacterMemoryExtractRequest = ...,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    from tldw_Server_API.app.core.Character_Chat.modules.character_memory_extraction import (
        extract_character_memories,
    )

    card = _resolve_character(db, character_id)
    user_id = str(current_user.id)
    char_name = card.get("name", "Unknown")
    conversation = db.get_conversation_by_id(body.chat_id)
    if not conversation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found")
    stored_client_id = conversation.get("client_id")
    request_user_id = current_user.id
    if not _ids_match(stored_client_id, request_user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not your chat session")
    conversation_character_id = str(conversation.get("character_id") or "").strip()
    requested_character_id = str(card.get("id") or character_id).strip()
    if conversation_character_id != requested_character_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chat session must belong to the requested character",
        )

    user_name = conversation.get("user_name", "User")
    persona_id = get_or_create_character_persona_profile(db, character_id, char_name, user_id)

    # Load messages
    messages = db.get_messages_for_conversation(body.chat_id, limit=body.message_limit, offset=0) or []
    messages = [m for m in messages if not m.get("deleted")]
    if not messages:
        return CharacterMemoryExtractResponse(extracted=0, skipped_duplicates=0, memories=[])

    # Load existing memories for dedup
    existing = db.list_persona_memory_entries(
        user_id=user_id, persona_id=persona_id, include_archived=True, include_deleted=False, limit=1000,
    )

    # Resolve provider/model
    api_endpoint = body.provider or "openai"
    model = body.model

    # Run extraction in thread — returns ExtractionResult with dedup stats
    extraction = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: extract_character_memories(
            messages=messages,
            char_name=char_name,
            user_name=user_name,
            existing_memories=existing,
            api_endpoint=api_endpoint,
            model=model,
        ),
    )

    # Persist unique memories
    created: list[CharacterMemoryResponse] = []
    for mem in extraction.unique:
        try:
            entry_id = db.add_persona_memory_entry({
                "persona_id": persona_id,
                "user_id": user_id,
                "memory_type": mem["category"],
                "content": mem["content"],
                "salience": mem["salience"],
                "source_conversation_id": body.chat_id,
            })
            created.append(CharacterMemoryResponse(
                id=entry_id,
                character_id=character_id,
                memory_type=mem["category"],
                content=mem["content"],
                salience=mem["salience"],
                source_conversation_id=body.chat_id,
                archived=False,
                created_at="",
                last_modified="",
            ))
        except Exception as exc:
            logger.warning("Failed to persist extracted memory: {}", exc)

    return CharacterMemoryExtractResponse(
        extracted=len(created),
        skipped_duplicates=extraction.duplicates_skipped,
        memories=created,
    )


# ---------------------------------------------------------------------------
# Memory item endpoints (with {memory_id} path param)
# ---------------------------------------------------------------------------

@router.patch(
    "/{character_id}/memories/{memory_id}",
    response_model=CharacterMemoryResponse,
    summary="Update a character memory",
)
async def update_character_memory(
    character_id: str = Path(...),
    memory_id: str = Path(...),
    body: CharacterMemoryUpdate = ...,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    _resolve_character(db, character_id)
    persona_id = _persona_id_for_character(character_id)
    user_id = str(current_user.id)

    update_data: dict[str, Any] = {}
    if body.content is not None:
        update_data["content"] = body.content
    if body.memory_type is not None:
        update_data["memory_type"] = body.memory_type
    if body.salience is not None:
        update_data["salience"] = body.salience

    if not update_data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields to update")

    try:
        ok = db.update_persona_memory_entry(
            entry_id=memory_id,
            user_id=user_id,
            persona_id=persona_id,
            update_data=update_data,
        )
    except InputError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")

    return _fetch_memory_by_id(db, user_id, persona_id, memory_id, character_id)


@router.delete(
    "/{character_id}/memories/{memory_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Soft-delete a character memory",
)
async def delete_character_memory(
    character_id: str = Path(...),
    memory_id: str = Path(...),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    _resolve_character(db, character_id)
    persona_id = _persona_id_for_character(character_id)
    user_id = str(current_user.id)

    try:
        ok = db.update_persona_memory_entry(
            entry_id=memory_id,
            user_id=user_id,
            persona_id=persona_id,
            update_data={"deleted": True},
        )
    except InputError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")


@router.post(
    "/{character_id}/memories/{memory_id}/archive",
    response_model=CharacterMemoryResponse,
    summary="Archive or unarchive a character memory",
)
async def archive_character_memory(
    character_id: str = Path(...),
    memory_id: str = Path(...),
    body: CharacterMemoryArchiveRequest = ...,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    _resolve_character(db, character_id)
    persona_id = _persona_id_for_character(character_id)
    user_id = str(current_user.id)

    ok = db.set_persona_memory_archived(
        entry_id=memory_id,
        user_id=user_id,
        persona_id=persona_id,
        archived=body.archived,
    )
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")

    return _fetch_memory_by_id(db, user_id, persona_id, memory_id, character_id)
