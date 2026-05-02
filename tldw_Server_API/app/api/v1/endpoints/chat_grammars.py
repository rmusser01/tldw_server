from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.api.v1.schemas.chat_grammar_schemas import (
    ChatGrammarCreate,
    ChatGrammarListResponse,
    ChatGrammarResponse,
    ChatGrammarUpdate,
)
from tldw_Server_API.app.core.Character_Chat.chat_grammar import ChatGrammarService
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)

router = APIRouter()


def _grammar_to_response(grammar_data: dict[str, Any]) -> ChatGrammarResponse:
    return ChatGrammarResponse(**grammar_data)


@router.post(
    "/grammars",
    response_model=ChatGrammarResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a saved chat grammar",
    description="Create a user-scoped reusable GBNF grammar for llama.cpp chat sessions.",
    tags=["chat-grammars"],
)
async def create_chat_grammar(
    grammar: ChatGrammarCreate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> ChatGrammarResponse:
    """Create a saved llama.cpp grammar for the current user."""
    service = ChatGrammarService(db)
    try:
        grammar_id = service.create_grammar(
            name=grammar.name,
            description=grammar.description,
            grammar_text=grammar.grammar_text,
        )
        created = service.get_grammar(grammar_id, include_archived=True)
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to create chat grammar") from exc
    except Exception as exc:
        logger.error("Error creating chat grammar")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create chat grammar",
        ) from exc

    if not created:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Grammar created but could not be retrieved",
        )
    return _grammar_to_response(created)


@router.get(
    "/grammars",
    response_model=ChatGrammarListResponse,
    summary="List saved chat grammars",
    description="List user-scoped saved GBNF grammars. Use include_archived to include archived items.",
    tags=["chat-grammars"],
)
async def list_chat_grammars(
    include_archived: bool = Query(False, description="Include archived saved grammars"),
    limit: int = Query(100, ge=1, le=500, description="Maximum grammars to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> ChatGrammarListResponse:
    """List saved llama.cpp grammars visible to the current user."""
    try:
        service = ChatGrammarService(db)
        grammars = service.list_grammars(
            include_archived=include_archived,
            limit=limit,
            offset=offset,
        )
        total = service.count_grammars(include_archived=include_archived)
    except CharactersRAGDBError as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to list chat grammars") from exc
    except Exception as exc:
        logger.error("Error listing chat grammars")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list chat grammars",
        ) from exc

    return ChatGrammarListResponse(
        items=[_grammar_to_response(grammar) for grammar in grammars],
        total=total,
        pagination=build_offset_pagination_meta(
            total=total,
            offset=offset,
            limit=limit,
            count=len(grammars),
        ),
    )


@router.get(
    "/grammars/{grammar_id}",
    response_model=ChatGrammarResponse,
    summary="Get a saved chat grammar",
    description="Retrieve a saved GBNF grammar by identifier.",
    tags=["chat-grammars"],
)
async def get_chat_grammar(
    grammar_id: str,
    include_archived: bool = Query(False, description="Allow archived grammars to be returned"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> ChatGrammarResponse:
    """Fetch one saved llama.cpp grammar owned by the current user."""
    try:
        service = ChatGrammarService(db)
        grammar = service.get_grammar(grammar_id, include_archived=include_archived)
    except (InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to get chat grammar") from exc
    except Exception as exc:
        logger.error("Error getting chat grammar")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get chat grammar",
        ) from exc

    if not grammar:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Grammar not found")
    return _grammar_to_response(grammar)


@router.patch(
    "/grammars/{grammar_id}",
    response_model=ChatGrammarResponse,
    summary="Update a saved chat grammar",
    description="Update grammar metadata or grammar text for a saved user-scoped grammar.",
    tags=["chat-grammars"],
)
async def update_chat_grammar(
    grammar_id: str,
    update: ChatGrammarUpdate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> ChatGrammarResponse:
    """Update a saved llama.cpp grammar owned by the current user."""
    service = ChatGrammarService(db)
    try:
        current = service.get_grammar(grammar_id, include_archived=True)
        if not current:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Grammar not found")

        update_payload = update.model_dump(exclude_unset=True)
        expected_version = update_payload.pop("version", None)
        updated = service.update_grammar(
            grammar_id,
            update_payload,
            expected_version=expected_version if expected_version is not None else current["version"],
        )
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to update chat grammar") from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error updating chat grammar")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update chat grammar",
        ) from exc

    return _grammar_to_response(updated)


@router.delete(
    "/grammars/{grammar_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a saved chat grammar",
    description="Soft-delete a saved user-scoped grammar unless hard_delete=true is supplied.",
    tags=["chat-grammars"],
)
async def delete_chat_grammar(
    grammar_id: str,
    hard_delete: bool = Query(False, description="Permanently delete instead of soft delete"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> Response:
    """Delete a saved llama.cpp grammar owned by the current user."""
    service = ChatGrammarService(db)
    try:
        deleted = service.delete_grammar(grammar_id, hard_delete=hard_delete)
    except InputError as exc:
        raise map_db_error_to_http(
            exc,
            input_status_code=status.HTTP_404_NOT_FOUND,
        ) from exc
    except (ConflictError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to delete chat grammar") from exc
    except Exception as exc:
        logger.error("Error deleting chat grammar")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete chat grammar",
        ) from exc

    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Grammar not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
