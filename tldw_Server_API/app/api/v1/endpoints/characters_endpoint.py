# characters.py
# Description:
#
# Imports
import base64
import json
import pathlib
from typing import List, Union, Any, Dict, Optional
#
# Third-party Libraries
from fastapi import HTTPException, Depends, Query, UploadFile, File, APIRouter, Path as FastAPIPath
from loguru import logger
from pydantic import BaseModel, Field, field_validator
from pydantic import ValidationInfo
from pydantic_core.core_schema import FieldValidationInfo
from starlette import status
#
# Local Imports
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.schemas.character_schemas import CharacterResponse, CharacterImportResponse, \
    CharacterCreate, CharacterUpdate, DeletionResponse
from tldw_Server_API.app.core.Character_Chat.Character_Chat_Lib import import_and_save_character_from_file, \
    search_characters_by_query_text, delete_character_from_db, get_character_details, update_existing_character_details, \
    create_new_character_from_data
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError, InputError, CharactersRAGDBError
#
#######################################################################################################################
#
# Functions:



# --- Router ---
router = APIRouter()


# --- Helper Functions (Keep _convert_db_char_to_response_model as is) ---
def _convert_db_char_to_response_model(char_dict_from_db: Dict[str, Any]) -> CharacterResponse:
    response_data = char_dict_from_db.copy()
    if response_data.get('image') and isinstance(response_data['image'], bytes):
        try:
            response_data['image_base64'] = base64.b64encode(response_data['image']).decode('utf-8')
            response_data['image_present'] = True
        except Exception as e:
            logger.error(f"Error encoding image for char {response_data.get('id')}: {e}")
            response_data['image_base64'] = None;
            response_data['image_present'] = False
    else:
        response_data['image_base64'] = None
        response_data['image_present'] = bool(
            response_data.get('image') and isinstance(response_data.get('image'), bytes))
    for field_name in ["alternate_greetings", "tags", "extensions"]:
        value = response_data.get(field_name)
        if isinstance(value, str):  # Already deserialized by DB layer if stored as JSON text
            pass  # Should be Python objects now from DB layer
    response_data.pop('image', None)
    return CharacterResponse.model_validate(response_data)


# --- API Endpoints ---

@router.post("/import", response_model=CharacterImportResponse, summary="Import character card file",
             tags=["Characters"], status_code=status.HTTP_201_CREATED)
async def import_character_from_file_endpoint(
        character_file: UploadFile = File(..., description="Character card file (PNG, WEBP, JSON, MD)."),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    try:
        file_content_bytes = await character_file.read()
        if not file_content_bytes:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

        logger.info(f"API: Attempting to import character from file: {character_file.filename}")

        # import_and_save_character_from_file now raises ConflictError if name exists
        # and InputError/CharactersRAGDBError for other issues.
        char_id = import_and_save_character_from_file(db, file_content_bytes)

        # If no exception, char_id should be valid
        imported_char_db = db.get_character_card_by_id(char_id)  # type: ignore
        if not imported_char_db:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Character imported but could not be retrieved.")

        response_char_model = _convert_db_char_to_response_model(imported_char_db)
        return CharacterImportResponse(
            message=f"Character '{response_char_model.name}' processed successfully (ID: {char_id}).",
            character=response_char_model
        )
    except ConflictError as e:  # Character with same name already exists
        # The library function might return the ID of the existing char if we want that behavior.
        # For now, let's assume ConflictError means it tried to add but failed.
        # If import_and_save_character_from_file returns existing ID on conflict, API status code could be 200.
        # The current lib function returns the existing ID on conflict.
        logger.warning(f"Conflict during import: {e}")
        # Try to retrieve the conflicting character if the error message provides enough info or if the lib returned an ID
        # This part needs careful alignment with how `import_and_save_character_from_file` signals "already exists"
        # If it returns the existing ID, then the initial `char_id` would be that.
        existing_char_id_from_conflict = None
        if hasattr(e, 'entity_id') and isinstance(e.entity_id, int):  # If ConflictError has the ID
            existing_char_id_from_conflict = e.entity_id
        elif isinstance(e.entity_id, str):  # If entity_id is the name
            existing_char_obj = db.get_character_card_by_name(e.entity_id)
            if existing_char_obj: existing_char_id_from_conflict = existing_char_obj['id']

        if existing_char_id_from_conflict:
            existing_char_db = db.get_character_card_by_id(existing_char_id_from_conflict)
            if existing_char_db:
                return CharacterImportResponse(
                    message=f"Character '{existing_char_db['name']}' already exists (ID: {existing_char_id_from_conflict}). Details provided.",
                    character=_convert_db_char_to_response_model(existing_char_db)
                )  # Consider HTTP 200 OK for this case

        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))

    except (InputError, CharactersRAGDBError) as e:
        logger.error(f"Error during character import: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during character import: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred during file import.")
    finally:
        await character_file.close()


@router.get("/", response_model=List[CharacterResponse], summary="List characters", tags=["Characters"])
async def list_all_characters(  # Renamed from list_characters to avoid conflict with Python's list
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0)
):
    try:
        # Using the interop library function that calls db.list_character_cards
        # but get_character_list_for_ui returns simplified data. We need full data here.
        raw_cards = db.list_character_cards(limit=limit, offset=offset)  # Direct DB call for full data
        return [_convert_db_char_to_response_model(card) for card in raw_cards]
    except CharactersRAGDBError as e:
        logger.error(f"DB error listing characters: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error listing characters: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")


@router.post("/", response_model=CharacterResponse, status_code=status.HTTP_201_CREATED, summary="Create character",
             tags=["Characters"])
async def create_new_character_endpoint(
        character_data: CharacterCreate,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    try:
        # The Pydantic model CharacterCreate ensures 'name' is present.
        # The interop function create_new_character_from_data handles image_base64 etc.
        # and calls db.add_character_card.
        # It will raise ConflictError if name exists, InputError for bad data.

        # Convert Pydantic model to dict for the interop library function
        payload_dict = character_data.model_dump(exclude_unset=False)  # include all fields

        char_id = create_new_character_from_data(db, payload_dict)

        if not char_id:  # Should be caught by exceptions in lib layer
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Failed to create character (no ID returned).")

        created_char_db = get_character_details(db, char_id)  # Use interop get
        if not created_char_db:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Failed to retrieve character after creation.")
        return _convert_db_char_to_response_model(created_char_db)
    except (InputError, ConflictError) as e:  # Propagated from lib
        status_code = status.HTTP_400_BAD_REQUEST if isinstance(e, InputError) else status.HTTP_409_CONFLICT
        logger.warning(f"Error creating character: {e} (Status: {status_code})")
        raise HTTPException(status_code=status_code, detail=str(e))
    except CharactersRAGDBError as e:
        logger.error(f"DB error creating character: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating character: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")


@router.get("/{character_id}", response_model=CharacterResponse, summary="Get character by ID", tags=["Characters"])
async def get_character_by_id_endpoint(  # Renamed from get_character
        character_id: int = FastAPIPath(..., description="ID of the character.", gt=0),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    try:
        char_db = get_character_details(db, character_id)  # Use interop get
        if not char_db:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Character with ID {character_id} not found.")
        return _convert_db_char_to_response_model(char_db)
    except CharactersRAGDBError as e:
        logger.error(f"DB error getting character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")


@router.put("/{character_id}", response_model=CharacterResponse, summary="Update character", tags=["Characters"])
async def update_character_endpoint(  # Renamed from update_character
        update_data: CharacterUpdate,
        character_id: int = FastAPIPath(..., description="ID of the character to update.", gt=0),
        expected_version: int = Query(...,
                                      description="Expected current version of the character for optimistic locking."),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    try:
        # Check if character exists before attempting update, to provide 404 early
        # The lib function update_existing_character_details might also do this or rely on DB layer
        current_char_for_check = get_character_details(db, character_id)
        if not current_char_for_check:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Character with ID {character_id} not found for update.")
        # Validate expected_version against actual current version
        if current_char_for_check['version'] != expected_version:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                                detail=f"Version mismatch. Expected {expected_version}, found {current_char_for_check['version']}. Please refresh and try again.")

        payload_dict = update_data.model_dump(exclude_unset=True)  # Only include fields that were set

        success = update_existing_character_details(db, character_id, payload_dict, expected_version)

        if not success:  # Should be caught by specific exceptions from lib
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Failed to update character (unexpected boolean failure).")

        updated_char_db = get_character_details(db, character_id)
        if not updated_char_db:  # Should not happen if update was successful
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Failed to retrieve character after update.")
        return _convert_db_char_to_response_model(updated_char_db)

    except (InputError, ConflictError) as e:
        status_code = status.HTTP_400_BAD_REQUEST if isinstance(e, InputError) else status.HTTP_409_CONFLICT
        logger.warning(f"Error updating character {character_id}: {e} (Status: {status_code})")
        raise HTTPException(status_code=status_code, detail=str(e))
    except CharactersRAGDBError as e:
        logger.error(f"DB error updating character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error updating character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")


@router.delete("/{character_id}", response_model=DeletionResponse, summary="Delete character", tags=["Characters"])
async def delete_character_endpoint(  # Renamed from delete_character
        character_id: int = FastAPIPath(..., description="ID of the character to delete.", gt=0),
        expected_version: int = Query(...,
                                      description="Expected current version of the character for optimistic locking."),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    try:
        # Check existence and get name for response message before delete attempt
        char_to_delete = get_character_details(db, character_id)
        if not char_to_delete:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Character with ID {character_id} not found for deletion.")

        # Validate expected_version here before calling lib, for clearer HTTP 409
        if char_to_delete['version'] != expected_version:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                                detail=f"Version mismatch for deletion. Expected {expected_version}, found {char_to_delete['version']}. Please refresh.")

        char_name = char_to_delete.get('name', 'N/A')
        success = delete_character_from_db(db, character_id, expected_version)

        if not success:  # Should be caught by specific exceptions from lib
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Failed to delete character (unexpected boolean failure).")

        return DeletionResponse(
            message=f"Character '{char_name}' (ID: {character_id}) soft-deleted.",
            character_id=character_id
        )
    except ConflictError as e:  # From lib (e.g. if somehow version changed between API check and lib call, or FK issue)
        logger.warning(f"Conflict error deleting character {character_id}: {e}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except CharactersRAGDBError as e:
        logger.error(f"DB error deleting character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error deleting character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")


@router.get("/search/", response_model=List[CharacterResponse], summary="Search characters", tags=["Characters"])
async def search_characters_endpoint(
        query: str = Query(..., description="Search term for character name, description, etc."),
        limit: int = Query(10, ge=1, le=100),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """
    Searches for characters based on a query string.
    The search is performed against FTS-indexed fields in the database.
    """
    if not query.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Search query cannot be empty.")
    try:
        results_db = search_characters_by_query_text(db, query, limit=limit)
        return [_convert_db_char_to_response_model(card) for card in results_db]
    except CharactersRAGDBError as e:
        logger.error(f"DB error searching characters for '{query}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error searching characters for '{query}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")

#
# End of characters.py
#######################################################################################################################
