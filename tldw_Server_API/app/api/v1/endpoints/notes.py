# app/api/v1/endpoints/notes.py
import logging
from typing import List, Optional, Dict, Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
    Body,
    Header  # Keep Header for expected_version
)
from loguru import logger  # Using loguru as in your chat example

# Local Imports from your project structure
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (  # Corrected import path if needed
    CharactersRAGDB, InputError, ConflictError, CharactersRAGDBError
)

# Schemas for notes (assuming this path is correct)
from tldw_Server_API.app.api.v1.schemas.notes_schemas import (
    NoteCreate, NoteUpdate, NoteResponse,
    KeywordCreate, KeywordResponse,
    NoteKeywordLinkResponse, KeywordsForNoteResponse, NotesForKeywordResponse,
    DetailResponse
)
# Dependency to get user-specific ChaChaNotes_DB instance
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

# We don't need get_current_user_id explicitly if get_chacha_db_for_user handles user context

router = APIRouter()


# --- Helper for Exception Handling (largely the same) ---
def handle_db_errors(e: Exception, entity_type: str = "resource"):
    if isinstance(e, InputError):
        logger.warning(f"Input error for {entity_type}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    elif isinstance(e, ConflictError):
        logger.warning(
            f"Conflict error for {entity_type} (ID: {e.entity_id if hasattr(e, 'entity_id') else 'N/A'}): {e}")
        user_message = str(e)
        if e.entity and e.entity_id:
            user_message = f"A conflict occurred with {e.entity} (ID: {e.entity_id}). It might have been modified or deleted, or a unique constraint was violated."
        elif "version mismatch" in str(e).lower():
            user_message = "The resource has been modified since you last fetched it. Please refresh and try again."
        elif "already exists" in str(e).lower():
            user_message = f"A {entity_type} with the provided identifier already exists."
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=user_message)
    elif isinstance(e, CharactersRAGDBError):
        logger.error(f"Database error for {entity_type}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"A database error occurred while processing your request for {entity_type}.")
    # ValueError might be raised by ChaChaNotes_DB for invalid inputs not caught by InputError
    elif isinstance(e, ValueError):
        logger.warning(f"Value error for {entity_type}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    else:
        logger.error(f"Unexpected DB service error for {entity_type}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"An unexpected error occurred while processing your request for {entity_type}.")


# --- Notes Endpoints ---
@router.post(
    "/",
    response_model=NoteResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new note",
    tags=["Notes"]
)
async def create_note(
        note_in: NoteCreate,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)  # Use the user-specific DB instance
):
    try:
        # The user context (user_id) is implicitly handled by `get_chacha_db_for_user`
        # The `db` instance is already specific to the authenticated user.
        logger.info(f"User (via DB instance client_id: {db.client_id}) creating note: Title='{note_in.title[:30]}...'")
        note_id = db.add_note(
            title=note_in.title,
            content=note_in.content,
            note_id=note_in.id  # Pass optional client-provided ID
        )
        if note_id is None:  # Should be caught by exceptions
            raise CharactersRAGDBError("Note creation failed to return an ID.")

        created_note_data = db.get_note_by_id(note_id=note_id)
        if not created_note_data:
            logger.error(
                f"Failed to retrieve note '{note_id}' immediately after creation for user (DB client_id: {db.client_id}).")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Note created but could not be retrieved.")
        logger.info(f"Note '{note_id}' created successfully for user (DB client_id: {db.client_id}).")
        return created_note_data  # Pydantic will convert dict to NoteResponse
    except Exception as e:
        handle_db_errors(e, "note")


@router.get(
    "/{note_id}",
    response_model=NoteResponse,
    summary="Get a specific note by ID",
    tags=["Notes"],
    responses={status.HTTP_404_NOT_FOUND: {"model": DetailResponse}}
)
async def get_note(
        note_id: str,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    try:
        logger.debug(f"User (DB client_id: {db.client_id}) fetching note: ID='{note_id}'")
        note_data = db.get_note_by_id(note_id=note_id)
        if not note_data:
            logger.warning(f"Note ID '{note_id}' not found for user (DB client_id: {db.client_id}).")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Note not found")
        return note_data
    except Exception as e:
        handle_db_errors(e, "note")


@router.get(
    "/",
    response_model=List[NoteResponse],
    summary="List all notes for the current user",
    tags=["Notes"]
)
async def list_notes(
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        limit: int = Query(100, ge=1, le=1000, description="Number of notes to return"),
        offset: int = Query(0, ge=0, description="Offset for pagination")
):
    try:
        logger.debug(f"User (DB client_id: {db.client_id}) listing notes: limit={limit}, offset={offset}")
        notes_data = db.list_notes(limit=limit, offset=offset)
        return notes_data
    except Exception as e:
        handle_db_errors(e, "notes list")


@router.put(
    "/{note_id}",
    response_model=NoteResponse,
    summary="Update an existing note",
    tags=["Notes"],
    responses={
        status.HTTP_404_NOT_FOUND: {"model": DetailResponse},
        status.HTTP_409_CONFLICT: {"model": DetailResponse}
    }
)
async def update_note(
        note_id: str,
        note_in: NoteUpdate,
        expected_version: int = Header(..., description="The expected version of the note for optimistic locking"),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    update_data = note_in.model_dump(exclude_unset=True)
    if not update_data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields provided for update.")
    try:
        logger.info(
            f"User (DB client_id: {db.client_id}) updating note: ID='{note_id}', Version={expected_version}, DataKeys={list(update_data.keys())}")
        success = db.update_note(
            note_id=note_id,
            update_data=update_data,
            expected_version=expected_version
        )
        if not success:
            raise CharactersRAGDBError("Note update reported non-success without specific exception.")

        updated_note_data = db.get_note_by_id(note_id=note_id)
        if not updated_note_data:
            logger.error(f"Note '{note_id}' not found after successful update for user (DB client_id: {db.client_id}).")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Note not found after update.")
        logger.info(
            f"Note '{note_id}' updated successfully for user (DB client_id: {db.client_id}) to version {updated_note_data['version']}.")
        return updated_note_data
    except Exception as e:
        handle_db_errors(e, "note")


@router.delete(
    "/{note_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Soft-delete a note",
    tags=["Notes"],
    responses={
        status.HTTP_404_NOT_FOUND: {"model": DetailResponse},
        status.HTTP_409_CONFLICT: {"model": DetailResponse}
    }
)
async def delete_note(
        note_id: str,
        expected_version: int = Header(..., description="The expected version of the note for optimistic locking"),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    try:
        logger.info(
            f"User (DB client_id: {db.client_id}) soft-deleting note: ID='{note_id}', Version={expected_version}")
        success = db.soft_delete_note(
            note_id=note_id,
            expected_version=expected_version
        )
        if not success:
            raise CharactersRAGDBError("Note soft delete reported non-success without specific exception.")
        logger.info(
            f"Note '{note_id}' soft-deleted successfully (or was already deleted) for user (DB client_id: {db.client_id}).")
        return  # FastAPI handles 204 No Content
    except Exception as e:
        handle_db_errors(e, "note")


@router.get(
    "/search/",
    response_model=List[NoteResponse],
    summary="Search notes for the current user",
    tags=["Notes"]
)
async def search_notes_endpoint(  # Renamed to avoid conflict with imported search_notes
        query: str = Query(..., min_length=1, description="Search term for notes"),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        limit: int = Query(10, ge=1, le=100, description="Number of results to return")
):
    try:
        logger.debug(f"User (DB client_id: {db.client_id}) searching notes: query='{query}', limit={limit}")
        notes_data = db.search_notes(search_term=query, limit=limit)
        return notes_data
    except Exception as e:
        handle_db_errors(e, "notes search")


# --- Keyword Endpoints (related to Notes) ---
@router.post(
    "/keywords/",
    response_model=KeywordResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new keyword",
    tags=["Keywords (for Notes)"]
)
async def create_keyword(
        keyword_in: KeywordCreate,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    try:
        logger.info(f"User (DB client_id: {db.client_id}) creating keyword: Text='{keyword_in.keyword}'")
        keyword_id = db.add_keyword(keyword_text=keyword_in.keyword)
        if keyword_id is None:
            raise CharactersRAGDBError("Keyword creation failed to return an ID.")

        created_keyword_data = db.get_keyword_by_id(keyword_id=keyword_id)
        if not created_keyword_data:
            logger.error(
                f"Failed to retrieve keyword '{keyword_id}' after creation for user (DB client_id: {db.client_id}).")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Keyword created but could not be retrieved.")
        logger.info(f"Keyword '{keyword_id}' created successfully for user (DB client_id: {db.client_id}).")
        return created_keyword_data
    except Exception as e:
        handle_db_errors(e, "keyword")


@router.get(
    "/keywords/{keyword_id}",
    response_model=KeywordResponse,
    summary="Get a keyword by its ID",
    tags=["Keywords (for Notes)"],
    responses={status.HTTP_404_NOT_FOUND: {"model": DetailResponse}}
)
async def get_keyword(
        keyword_id: int,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    try:
        logger.debug(f"User (DB client_id: {db.client_id}) fetching keyword by ID: {keyword_id}")
        keyword_data = db.get_keyword_by_id(keyword_id=keyword_id)
        if not keyword_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Keyword not found")
        return keyword_data
    except Exception as e:
        handle_db_errors(e, "keyword")


@router.get(
    "/keywords/text/{keyword_text}",
    response_model=KeywordResponse,
    summary="Get a keyword by its text content",
    tags=["Keywords (for Notes)"],
    responses={status.HTTP_404_NOT_FOUND: {"model": DetailResponse}}
)
async def get_keyword_by_text(
        keyword_text: str,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    try:
        logger.debug(f"User (DB client_id: {db.client_id}) fetching keyword by text: '{keyword_text}'")
        keyword_data = db.get_keyword_by_text(keyword_text=keyword_text)
        if not keyword_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Keyword not found")
        return keyword_data
    except Exception as e:
        handle_db_errors(e, "keyword")


@router.get(
    "/keywords/",
    response_model=List[KeywordResponse],
    summary="List all keywords for the current user",
    tags=["Keywords (for Notes)"]
)
async def list_keywords_endpoint(  # Renamed to avoid conflict
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0)
):
    try:
        logger.debug(f"User (DB client_id: {db.client_id}) listing keywords: limit={limit}, offset={offset}")
        keywords_data = db.list_keywords(limit=limit, offset=offset)
        return keywords_data
    except Exception as e:
        handle_db_errors(e, "keywords list")


@router.delete(
    "/keywords/{keyword_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Soft-delete a keyword",
    tags=["Keywords (for Notes)"],
    responses={
        status.HTTP_404_NOT_FOUND: {"model": DetailResponse},
        status.HTTP_409_CONFLICT: {"model": DetailResponse}
    }
)
async def delete_keyword(
        keyword_id: int,
        expected_version: int = Header(..., description="The expected version of the keyword for optimistic locking"),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    try:
        logger.info(
            f"User (DB client_id: {db.client_id}) soft-deleting keyword: ID='{keyword_id}', Version={expected_version}")
        success = db.soft_delete_keyword(
            keyword_id=keyword_id,
            expected_version=expected_version
        )
        if not success:
            raise CharactersRAGDBError("Keyword soft delete reported non-success without specific exception.")
        logger.info(
            f"Keyword '{keyword_id}' soft-deleted successfully (or was already deleted) for user (DB client_id: {db.client_id}).")
        return
    except Exception as e:
        handle_db_errors(e, "keyword")


@router.get(
    "/keywords/search/",
    response_model=List[KeywordResponse],
    summary="Search keywords for the current user",
    tags=["Keywords (for Notes)"]
)
async def search_keywords_endpoint(  # Renamed
        query: str = Query(..., min_length=1, description="Search term for keywords"),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        limit: int = Query(10, ge=1, le=100)
):
    try:
        logger.debug(f"User (DB client_id: {db.client_id}) searching keywords: query='{query}', limit={limit}")
        keywords_data = db.search_keywords(search_term=query, limit=limit)
        return keywords_data
    except Exception as e:
        handle_db_errors(e, "keywords search")


# --- Note-Keyword Linking Endpoints ---
@router.post(
    "/{note_id}/keywords/{keyword_id}",
    response_model=NoteKeywordLinkResponse,
    summary="Link a note to a keyword",
    tags=["Notes Linking"],
    responses={status.HTTP_404_NOT_FOUND: {"model": DetailResponse}}
)
async def link_note_to_keyword_endpoint(
        note_id: str,
        keyword_id: int,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    try:
        logger.info(f"User (DB client_id: {db.client_id}) linking note '{note_id}' to keyword '{keyword_id}'")
        # Check if note and keyword exist in the user's DB
        note_data = db.get_note_by_id(note_id)
        if not note_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Note with ID '{note_id}' not found.")
        keyword_data = db.get_keyword_by_id(keyword_id)
        if not keyword_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Keyword with ID '{keyword_id}' not found.")

        success = db.link_note_to_keyword(note_id=note_id, keyword_id=keyword_id)
        msg = "Note linked to keyword successfully." if success else "Link already exists or was created."
        return NoteKeywordLinkResponse(success=True, message=msg)  # True even if already exists
    except HTTPException:
        raise
    except Exception as e:
        handle_db_errors(e, "note-keyword link")


@router.delete(
    "/{note_id}/keywords/{keyword_id}",
    response_model=NoteKeywordLinkResponse,
    summary="Unlink a note from a keyword",
    tags=["Notes Linking"]
)
async def unlink_note_from_keyword_endpoint(
        note_id: str,
        keyword_id: int,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    try:
        logger.info(f"User (DB client_id: {db.client_id}) unlinking note '{note_id}' from keyword '{keyword_id}'")
        success = db.unlink_note_from_keyword(note_id=note_id, keyword_id=keyword_id)
        msg = "Note unlinked from keyword successfully." if success else "Link not found or no action taken."
        return NoteKeywordLinkResponse(success=success, message=msg)
    except Exception as e:
        handle_db_errors(e, "note-keyword unlink")


@router.get(
    "/{note_id}/keywords/",
    response_model=KeywordsForNoteResponse,
    summary="Get all keywords linked to a note",
    tags=["Notes Linking"],
    responses={status.HTTP_404_NOT_FOUND: {"model": DetailResponse}}
)
async def get_keywords_for_note_endpoint(
        note_id: str,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    try:
        logger.debug(f"User (DB client_id: {db.client_id}) fetching keywords for note '{note_id}'")
        note_check = db.get_note_by_id(note_id=note_id)
        if not note_check:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Note with ID '{note_id}' not found.")

        keywords_list = db.get_keywords_for_note(note_id=note_id)
        return KeywordsForNoteResponse(note_id=note_id, keywords=keywords_list)
    except HTTPException:
        raise
    except Exception as e:
        handle_db_errors(e, "keywords for note")


@router.get(
    "/keywords/{keyword_id}/notes/",
    response_model=NotesForKeywordResponse,
    summary="Get all notes linked to a keyword",
    tags=["Notes Linking"],
    responses={status.HTTP_404_NOT_FOUND: {"model": DetailResponse}}
)
async def get_notes_for_keyword_endpoint(
        keyword_id: int,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0)
):
    try:
        logger.debug(f"User (DB client_id: {db.client_id}) fetching notes for keyword '{keyword_id}'")
        keyword_check = db.get_keyword_by_id(keyword_id=keyword_id)
        if not keyword_check:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Keyword with ID '{keyword_id}' not found.")

        notes_list = db.get_notes_for_keyword(keyword_id=keyword_id, limit=limit, offset=offset)
        return NotesForKeywordResponse(keyword_id=keyword_id, notes=notes_list)
    except HTTPException:
        raise
    except Exception as e:
        handle_db_errors(e, "notes for keyword")

#
# --- End of Notes and Keywords Endpoints ---
########################################################################################################################
