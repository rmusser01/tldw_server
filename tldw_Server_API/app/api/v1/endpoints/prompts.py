# tldw_Server_API/app/api/v1/endpoints/prompts.py
#
#
# Imports
import logging
import os
import base64
import sqlite3
from typing import List, Optional, Union, Tuple
#
# 3rd-party imports
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Body,
    status,
    Header,
    File,
    UploadFile
)
from starlette.responses import FileResponse # For serving exported files
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.v1_endpoint_deps import verify_token
#
# Local Imports
from tldw_Server_API.app.core.Prompt_Management.Prompts_Interop import (
    db_export_prompts_formatted, # Using the standalone function from interop
    db_export_prompt_keywords_to_csv,
    db_view_prompt_keywords_markdown
)
from tldw_Server_API.app.core.DB_Management.Prompts_DB import (
    DatabaseError,
    SchemaError,
    InputError,
    ConflictError,
    PromptsDatabase
)
from tldw_Server_API.app.api.v1.API_Deps.Prompts_DB_Deps import get_prompts_db_for_user
from tldw_Server_API.app.api.v1.schemas import prompt_schemas as schemas
# For auth, assuming similar setup to chat.py
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
from tldw_Server_API.app.core.config import settings
#
# DB Mgmt
from tldw_Server_API.app.services.ephemeral_store import ephemeral_storage
#from tldw_Server_API.app.core.DB_Management.DB_Manager import DBManager
#
#
#######################################################################################################################
#
# Functions:

router = APIRouter()

# --- Sync Log Endpoints ---
@router.get(
    "/sync-log",
    response_model=List[schemas.SyncLogEntryResponse],
    summary="Get sync log entries (admin/debug)",
    dependencies=[Depends(verify_token)] # Should be admin-only
)
async def get_sync_log(
    since_change_id: int = Query(0, ge=0),
    limit: Optional[int] = Query(100, ge=1, le=1000),
    Token: str = Header(None, description="Bearer token for authentication."),
    db: PromptsDatabase = Depends(get_prompts_db_for_user) # User specific sync log
):
    # Add admin role check here if you have role-based auth
    # if not current_user.is_admin:
    #     raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    try:
        entries = db.get_sync_log_entries(since_change_id=since_change_id, limit=limit)
        return [schemas.SyncLogEntryResponse(**entry) for entry in entries]
    except DatabaseError as e:
        logger.error(f"Database error fetching sync log: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error.")




# --- Search Endpoints ---
@router.post(
    "/search",
    response_model=schemas.PromptSearchResponse,
    summary="Search prompts",
    dependencies=[Depends(verify_token)]
)
async def search_all_prompts(
    search_query: str = Query(..., min_length=1, description="Search term(s)"),
    search_fields: Optional[List[str]] = Query(None, description="Fields to search: name, author, details, system_prompt, user_prompt, keywords"),
    page: int = Query(1, ge=1),
    results_per_page: int = Query(20, ge=1, le=100),
    include_deleted: bool = Query(False),
    Token: str = Header(None, description="Bearer token for authentication."),
    db: PromptsDatabase = Depends(get_prompts_db_for_user)
):
    try:
        results_list, total_matches = db.search_prompts(
            search_query=search_query,
            search_fields=search_fields,
            page=page,
            results_per_page=results_per_page,
            include_deleted=include_deleted
        )
        # Convert dicts to PromptSearchResultItem
        items = [schemas.PromptSearchResultItem(**item) for item in results_list]
        return schemas.PromptSearchResponse(
            items=items,
            total_matches=total_matches,
            page=page,
            per_page=results_per_page
        )
    except ValueError as e: # Bad page/per_page
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except DatabaseError as e:
        logger.error(f"Database error searching prompts: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error during search.")


# === Keyword Endpoints ===
@router.post(
    "/keywords/",
    response_model=schemas.KeywordResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a new keyword",
    dependencies=[Depends(verify_token)]
)
async def create_keyword(
    keyword_data: schemas.KeywordCreate,
    Token: str = Header(None, description="Bearer token for authentication."),
    db: PromptsDatabase = Depends(get_prompts_db_for_user)
):
    try:
        # Step 1: Check if an active keyword with this normalized text already exists.
        # The new DB method handles normalization internally.
        existing_active_keyword = db.get_active_keyword_by_text(keyword_data.keyword_text)

        if existing_active_keyword:
            # If it exists and is active, this endpoint should return a conflict.
            normalized_text = db._normalize_keyword(keyword_data.keyword_text) # For error message
            raise ConflictError(f"Keyword '{normalized_text}' already exists and is active.")

        # Step 2: If not actively existing, proceed to add (which might create or undelete).
        # db.add_keyword is "get or create or undelete".
        kw_id, kw_uuid = db.add_keyword(keyword_data.keyword_text)

        if not kw_id or not kw_uuid: # Should be rare if db.add_keyword is robust
            logger.error(f"db.add_keyword failed to return ID/UUID for '{keyword_data.keyword_text}' after pre-check.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create or retrieve keyword.")

        # Fetch the full details of the (potentially newly created or undeleted) keyword for the response.
        # To do this properly, we might need a get_keyword_by_id or get_keyword_by_uuid
        # For now, constructing from what we have. Prompts_DB.add_keyword normalizes.
        final_keyword_text = db._normalize_keyword(keyword_data.keyword_text)

        return schemas.KeywordResponse(
            id=kw_id,
            uuid=kw_uuid,
            keyword_text=final_keyword_text
        )
    except InputError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ConflictError as e: # Catches the ConflictError from our explicit check
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except DatabaseError as e:
        logger.error(f"Database error creating keyword: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error.")
    except Exception as e:
        logger.error(f"Unexpected error creating keyword: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {str(e)}")


@router.get(
    "/keywords/",
    response_model=List[str], # Just a list of keyword strings
    summary="List all active keywords",
    dependencies=[Depends(verify_token)]
)
async def list_all_keywords(
    db: PromptsDatabase = Depends(get_prompts_db_for_user)
):
    try:
        return db.fetch_all_keywords(include_deleted=False)
    except DatabaseError as e:
        logger.error(f"Database error listing keywords: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error.")


@router.delete(
    "/keywords/{keyword_text}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Soft delete a keyword",
    dependencies=[Depends(verify_token)]
)
async def delete_keyword(
    keyword_text: str,
    Token: str = Header(None, description="Bearer token for authentication."),
    db: PromptsDatabase = Depends(get_prompts_db_for_user)
):
    try:
        success = db.soft_delete_keyword(keyword_text)
        if not success:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Keyword not found or already deleted.")
        return None
    except InputError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ConflictError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except DatabaseError as e:
        logger.error(f"Database error deleting keyword '{keyword_text}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error.")


# === Export Endpoints ===

@router.get(
    "/export",
    response_model=schemas.ExportResponse, # Returns message and base64 content
    summary="Export prompts to CSV or Markdown (as base64 string)",
    dependencies=[Depends(verify_token)]
)
async def export_prompts_api(
    export_format: str = Query("csv", enum=["csv", "markdown"]),
    filter_keywords: Optional[List[str]] = Query(None),
    include_system: bool = Query(True),
    include_user: bool = Query(True),
    include_details: bool = Query(True),
    include_author: bool = Query(True),
    include_associated_keywords: bool = Query(True),
    markdown_template_name: Optional[str] = Query("Basic Template"),
    Token: str = Header(None, description="Bearer token for authentication."),
    db: PromptsDatabase = Depends(get_prompts_db_for_user)
):
    try:
        # Use the standalone function from prompts_interop (or Prompts_DB_v2)
        # It needs the db_instance.
        status_msg, file_path_or_content = db_export_prompts_formatted(
            db_instance=db, # Pass the user-specific DB instance
            export_format=export_format,
            filter_keywords=filter_keywords,
            include_system=include_system,
            include_user=include_user,
            include_details=include_details,
            include_author=include_author,
            include_associated_keywords=include_associated_keywords,
            markdown_template_name=markdown_template_name
        )

        if file_path_or_content == "None" or not os.path.exists(file_path_or_content):
            if "No prompts found" in status_msg:
                 return schemas.ExportResponse(message=status_msg, file_content_b64=None)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Export failed: {status_msg}")

        with open(file_path_or_content, "rb") as f:
            file_bytes = f.read()
        file_b64 = base64.b64encode(file_bytes).decode('utf-8')

        # Clean up the temporary file
        try:
            os.remove(file_path_or_content)
        except OSError as e_remove:
            logger.warning(f"Could not remove temporary export file {file_path_or_content}: {e_remove}")

        return schemas.ExportResponse(message=status_msg, file_content_b64=file_b64)

    except ValueError as e: # Invalid export format etc.
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except DatabaseError as e:
        logger.error(f"Database error during export: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error during export.")
    except Exception as e:
        logger.error(f"Unexpected error during export: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error during export: {str(e)}")


@router.get(
    "/keywords/export-csv",
    response_model=schemas.ExportResponse,
    summary="Export all prompt keywords with associations to CSV (as base64 string)",
    dependencies=[Depends(verify_token)]
)
async def export_keywords_api(
    Token: str = Header(None, description="Bearer token for authentication."),
    db: PromptsDatabase = Depends(get_prompts_db_for_user)
):
    try:
        status_msg, file_path = db_export_prompt_keywords_to_csv(db_instance=db)
        if file_path == "None" or not os.path.exists(file_path):
            if "Successfully exported 0 active prompt keywords" in status_msg or "No active keywords found" in status_msg : # Adjusted condition for empty export
                 return schemas.ExportResponse(message=status_msg, file_content_b64=None)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Keyword export failed: {status_msg}")

        with open(file_path, "rb") as f:
            file_bytes = f.read()
        file_b64 = base64.b64encode(file_bytes).decode('utf-8')
        try: os.remove(file_path)
        except OSError: pass
        return schemas.ExportResponse(message=status_msg, file_content_b64=file_b64)
    except Exception as e:
        logger.error(f"Unexpected error during keyword export: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error during keyword export: {str(e)}")


# === Prompt Endpoints ===

@router.post(
    "/",
    response_model=schemas.PromptResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new prompt",
    dependencies=[Depends(verify_token)] # Apply token verification
)
async def create_prompt(
    prompt_data: schemas.PromptCreate,
    Token: str = Header(None, description="Bearer token for authentication."),
    db: PromptsDatabase = Depends(get_prompts_db_for_user)
):
    try:
        # The db.add_prompt method with overwrite=False should raise ConflictError
        # if the name already exists and is active (as per our DB layer modification).
        p_id, p_uuid, db_message = db.add_prompt(  # db_message is returned by add_prompt on success
            name=prompt_data.name,
            author=prompt_data.author,
            details=prompt_data.details,
            system_prompt=prompt_data.system_prompt,
            user_prompt=prompt_data.user_prompt,
            keywords=prompt_data.keywords,
            overwrite=False  # For a POST/create, we don't want to overwrite.
        )
        # If add_prompt successfully created or undeleted (if that's its logic for overwrite=False and deleted=True)
        # then p_id and p_uuid will be set.

        # The 'msg' variable was causing the NameError.
        # db.add_prompt returns (id, uuid, message_string)
        # We can use db_message for logging if needed.

        if not p_id or not p_uuid:  # Should ideally not be hit if add_prompt raises on failure
            logger.error(
                f"Failed to create prompt '{prompt_data.name}', add_prompt returned: {p_id}, {p_uuid}, {db_message}")
            # If db_message has specific error info from add_prompt, use it.
            detail_msg = f"Failed to create prompt: {db_message}" if db_message else "Failed to create prompt (unknown DB issue)."
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail_msg)

        created_prompt_dict = db.fetch_prompt_details(p_uuid)  # Fetch by UUID to be sure
        if not created_prompt_dict:
            logger.error(f"Could not fetch newly created prompt by UUID {p_uuid}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Prompt created but could not be retrieved.")

        # Ensure 'deleted' field is populated if the schema expects it
        if 'deleted' not in created_prompt_dict and schemas.PromptResponse.model_fields.get('deleted'):
            created_prompt_dict['deleted'] = False  # Default for new prompts

        return schemas.PromptResponse(**created_prompt_dict)

    except InputError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ConflictError as e:  # This is expected if name exists and overwrite=False
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except DatabaseError as e:
        logger.error(f"Database error creating prompt: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Database error during prompt creation.")
    except Exception as e:  # Catch-all for other unexpected errors
        logger.error(f"Unexpected error creating prompt: {e}", exc_info=True)
        # Avoid leaking the raw 'msg' variable if it was a NameError
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")

@router.get(
    "/",
    response_model=schemas.PaginatedPromptsResponse,
    summary="List all prompts (paginated)",
    dependencies=[Depends(verify_token)]
)
async def list_all_prompts(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page"),
    include_deleted: bool = Query(False, description="Include soft-deleted prompts"),
    Token: str = Header(None, description="Bearer token for authentication."),
    db: PromptsDatabase = Depends(get_prompts_db_for_user)
):
    try:
        items_dict_list, total_pages, current_page, total_items = db.list_prompts(
            page=page, per_page=per_page, include_deleted=include_deleted
        )
        # Convert list of dicts to list of PromptBriefResponse
        brief_items = [schemas.PromptBriefResponse(**item) for item in items_dict_list]
        return schemas.PaginatedPromptsResponse(
            items=brief_items,
            total_pages=total_pages,
            current_page=current_page,
            total_items=total_items
        )
    except ValueError as e: # For bad page/per_page from DB layer
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except DatabaseError as e:
        logger.error(f"Database error listing prompts: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error listing prompts.")


@router.get(
    "/{prompt_identifier}",
    response_model=schemas.PromptResponse,
    summary="Get a specific prompt by ID, UUID, or Name",
    dependencies=[Depends(verify_token)]
)
async def get_prompt(
    prompt_identifier: Union[int, str], # Path param will be string, FastAPI can convert to int if possible
    include_deleted: bool = Query(False, description="Include if soft-deleted"),
    Token: str = Header(None, description="Bearer token for authentication."),
    db: PromptsDatabase = Depends(get_prompts_db_for_user)
):
    try:
        # Attempt to convert to int if it looks like an ID
        processed_identifier: Union[int, str] = prompt_identifier
        try:
            processed_identifier = int(prompt_identifier)
        except ValueError:
            pass # Keep as string if not an int (name or UUID)

        prompt_details = db.fetch_prompt_details(processed_identifier, include_deleted=include_deleted)
        if not prompt_details:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prompt not found.")
        return schemas.PromptResponse(**prompt_details)
    except DatabaseError as e:
        logger.error(f"Database error getting prompt '{prompt_identifier}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error.")


@router.put(
    "/{prompt_identifier}",
    response_model=schemas.PromptResponse,
    summary="Update an existing prompt (or create if name matches and overwrite=true logic used)",
    dependencies=[Depends(verify_token)]
)
async def update_prompt(
    prompt_identifier: Union[int, str],
    prompt_data: schemas.PromptCreate, # Using PromptCreate for full replacement, or PromptUpdate for partial
    Token: str = Header(None, description="Bearer token for authentication."),
    db: PromptsDatabase = Depends(get_prompts_db_for_user)
):
    # This uses add_prompt with overwrite=True logic.
    # For a true PATCH, you'd need a different DB method.
    # The prompt_identifier is used to ensure we are updating the one intended if name changes.
    try:
        # 1. Resolve identifier to actual prompt ID
        target_prompt_dict = db.fetch_prompt_details(prompt_identifier,
                                                     include_deleted=True)  # Allow updating soft-deleted
        if not target_prompt_dict:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Prompt with identifier '{prompt_identifier}' not found.")

        prompt_id_to_update = target_prompt_dict['id']

        # 2. Call the new update method
        # Convert Pydantic model to dict, excluding unset to allow partial-like updates if some fields are optional
        update_payload_dict = prompt_data.model_dump(
            exclude_unset=False)  # exclude_unset=False means all fields are included

        updated_prompt_uuid, msg = db.update_prompt_by_id(prompt_id_to_update, update_payload_dict)

        if not updated_prompt_uuid:
            # This case should be rare if fetch_prompt_details found it, unless db.update_prompt_by_id returns None for "no changes"
            logger.error(
                f"Update for prompt identifier '{prompt_identifier}' (ID: {prompt_id_to_update}) resulted in no UUID: {msg}")
            # Determine appropriate HTTP status based on msg
            if "not found" in msg.lower():  # Should have been caught by fetch_prompt_details
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=msg)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Prompt update failed: {msg}")

        # Fetch the fully updated prompt to return
        final_updated_prompt = db.fetch_prompt_details(updated_prompt_uuid)  # Fetch by UUID
        if not final_updated_prompt:
            logger.error(f"Could not retrieve prompt by UUID {updated_prompt_uuid} after update.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Prompt updated but could not be retrieved.")

        if 'deleted' not in final_updated_prompt and hasattr(schemas.PromptResponse, 'deleted'):
            final_updated_prompt['deleted'] = False

        return schemas.PromptResponse(**final_updated_prompt)

    except InputError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ConflictError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except DatabaseError as e:
        logger.error(f"Database error updating prompt '{prompt_identifier}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Database error during prompt update.")
    except HTTPException:  # Re-raise
        raise
    except Exception as e:
        logger.error(f"Unexpected error updating prompt '{prompt_identifier}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred during prompt update.")


@router.delete(
    "/{prompt_identifier}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Soft delete a prompt",
    dependencies=[Depends(verify_token)]
)
async def delete_prompt(
    prompt_identifier: Union[int, str],
    Token: str = Header(None, description="Bearer token for authentication."),
    db: PromptsDatabase = Depends(get_prompts_db_for_user)
):
    try:
        processed_identifier: Union[int, str] = prompt_identifier
        try: processed_identifier = int(prompt_identifier)
        except ValueError: pass

        success = db.soft_delete_prompt(processed_identifier)
        if not success:
            # Could be not found or already deleted, DB layer logs warning
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prompt not found or already deleted.")
        return None # HTTP 204 returns no body
    except ConflictError as e: # If version mismatch during delete
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except DatabaseError as e:
        logger.error(f"Database error deleting prompt '{prompt_identifier}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error.")


#
# End of prompts.py
#######################################################################################################################
