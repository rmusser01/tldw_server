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
from fastapi import HTTPException, Depends, Query, UploadFile, File, APIRouter, Path
from loguru import logger
from pydantic import BaseModel, Field, field_validator
from pydantic import ValidationInfo
from pydantic_core.core_schema import FieldValidationInfo
from starlette import status
#
# Local Imports
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.core.Character_Chat.Character_Chat_Lib import import_and_save_character_from_file
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError, InputError, CharactersRAGDBError
#
#######################################################################################################################
#
# Functions:
# --- Pydantic Schemas ---

class CharacterBase(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = Field(None, examples=["A brave knight"])
    personality: Optional[str] = Field(None, examples=["Stoic and honorable"])
    scenario: Optional[str] = Field(None, examples=["Guarding the ancient ruins"])
    system_prompt: Optional[str] = Field(None, examples=["You are a helpful character."])
    post_history_instructions: Optional[str] = None
    first_message: Optional[str] = Field(None, examples=["Greetings, traveler!"])
    message_example: Optional[str] = Field(None, examples=["<START>\nUSER: Hello\nASSISTANT: Hi there!\n<END>"])
    creator_notes: Optional[str] = None
    alternate_greetings: Optional[Union[List[str], str]] = Field(None,
                                                                 description="List of strings or a JSON string representation of a list.",
                                                                 examples=[["Hello!", "Good day!"]])
    tags: Optional[Union[List[str], str]] = Field(None,
                                                  description="List of strings or a JSON string representation of a list.",
                                                  examples=[["fantasy", "knight"]])
    creator: Optional[str] = None
    character_version: Optional[str] = None
    extensions: Optional[Union[Dict[str, Any], str]] = Field(None,
                                                             description="Dictionary or a JSON string representation of a dictionary.")
    image_base64: Optional[str] = Field(None,
                                        description="Base64 encoded image string (without 'data:image/...;base64,' prefix).")

    @field_validator("alternate_greetings", "tags", "extensions", mode="before")
    @classmethod
    def parse_json_string(cls, value: Any, info: ValidationInfo) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON string for field '{info.field_name}': {value[:100]}...")
                if info.field_name in ["alternate_greetings", "tags"]:
                    return []  # Return empty list for tag-like fields
                if info.field_name == "extensions":
                    return {}  # Return empty dict for extensions
                # For other fields or if strict parsing is needed, you might raise a ValueError
                # raise ValueError(f"Invalid JSON string for {info.field_name}")
                return value  # Or keep as string if it might be non-JSON intentionally
        return value


class CharacterCreate(CharacterBase):
    name: str = Field(..., examples=["Sir Gideon"])


class CharacterUpdate(CharacterBase):
    # All fields are optional for update.
    # To remove an image, pass image_base64 as None or an empty string.
    pass


class CharacterResponse(CharacterBase):
    id: int
    version: int
    image_present: bool = False  # Indicates if an image is stored for the character in the DB

    model_config = {
        "from_attributes": True  # Allow creating from ORM models or dicts
    }


class CharacterImportResponse(BaseModel):
    message: str
    character: Optional[CharacterResponse] = None


class DeletionResponse(BaseModel):
    message: str
    character_id: int


# --- Router ---
router = APIRouter()


# --- Helper Functions ---

def _convert_db_char_to_response_model(char_dict_from_db: Dict[str, Any]) -> CharacterResponse:
    """Converts a character dictionary from the DB to a CharacterResponse model."""
    try:
        response_data = char_dict_from_db.copy()

        if response_data.get('image') and isinstance(response_data['image'], bytes):
            try:
                response_data['image_base64'] = base64.b64encode(response_data['image']).decode('utf-8')
                response_data['image_present'] = True
            except Exception as e:
                logger.error(f"Error encoding image for character {response_data.get('id')}: {e}")
                response_data['image_base64'] = None
                response_data['image_present'] = False
        else:
            response_data['image_base64'] = None
            # image_present should be False if image is None or not bytes
            response_data['image_present'] = bool(
                response_data.get('image') and isinstance(response_data.get('image'), bytes))

        # Ensure JSON fields (alternate_greetings, tags, extensions) are Python objects
        # The DB layer should ideally load these as Python objects if column type is JSON.
        # If they are stored as text and are JSON strings, this parsing is needed.
        for field_name in ["alternate_greetings", "tags", "extensions"]:
            value = response_data.get(field_name)
            if isinstance(value, str):
                try:
                    response_data[field_name] = json.loads(value)
                except json.JSONDecodeError:
                    logger.warning(
                        f"DB data for {field_name} in char {response_data.get('id')} is not valid JSON: {value[:100]}...")
                    # Set defaults based on expected type if parsing fails
                    if field_name in ["alternate_greetings", "tags"]:
                        response_data[field_name] = []
                    elif field_name == "extensions":
                        response_data[field_name] = {}

        # Remove the raw 'image' bytes field before passing to Pydantic model if it exists
        response_data.pop('image', None)

        return CharacterResponse.model_validate(response_data)
    except StopIteration as si: # ADD THIS EXCEPT
        logger.error(f"****** StopIteration CAUGHT INSIDE _convert_db_char_to_response_model ******: {si}", exc_info=True)
        raise # Re-raise to see if it's the one caught by the endpoint
    except Exception as ex_inner: # ADD THIS EXCEPT
        logger.error(f"****** Other Exception in _convert_db_char_to_response_model ******: Type: {type(ex_inner)}, Msg: {str(ex_inner)}", exc_info=True)
        raise

def _prepare_char_data_for_db(char_input_data: Union[CharacterCreate, CharacterUpdate], is_update: bool = False) -> \
Dict[str, Any]:
    """Prepares character data from Pydantic model for DB insertion/update."""
    if is_update:
        db_data = char_input_data.model_dump(exclude_unset=True)
    else:
        db_data = char_input_data.model_dump()

    # Handle image_base64: convert to bytes for 'image' field, or set 'image' to None
    if 'image_base64' in db_data:  # Check if key exists, even if value is None
        base64_str = db_data.pop('image_base64')  # Remove from dict, process its value
        if base64_str and isinstance(base64_str, str):
            try:
                # Robustly remove data URL prefix if present, common in web inputs
                if ',' in base64_str and base64_str.startswith("data:image"):
                    base64_str = base64_str.split(',', 1)[1]
                db_data['image'] = base64.b64decode(base64_str, validate=True)
                logger.debug("Image successfully decoded.")
            except Exception as e:
                logger.error(f"IMAGE DECODE FAILED for character '{getattr(char_input_data, 'name', 'N/A')}': {e}. RAISING HTTP 400.")
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image_base64 data: {e}")
        else:  # image_base64 was None or empty string
            logger.debug("image_base64 was None, empty, or not a string. Setting image to None.")
            db_data['image'] = None
    elif not is_update and 'image_base64' not in db_data:  # For create, if not provided at all
        logger.debug("image_base64 not in input for create. Setting image to None.")
        db_data['image'] = None

    # Pydantic model's field_validator should have already converted incoming JSON strings
    # for alternate_greetings, tags, extensions into Python lists/dicts.
    # The CharactersRAGDB methods are responsible for converting these Python objects
    # to JSON strings if the DB column type requires it (e.g., TEXT).
    return db_data


# --- API Endpoints ---

@router.post(
    "/import",  # Renamed from /import_image for clarity
    response_model=CharacterImportResponse,
    summary="Import a character from a character card file (PNG, WEBP, JSON, MD).",
    status_code=status.HTTP_201_CREATED,  # Default, might change to 200 if character already exists
    tags=["Characters"],
    responses={
        status.HTTP_200_OK: {"description": "Character already existed and was retrieved."},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid file format, content, or missing required data."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Server-side error during import."},
    }
)
async def import_character_from_file(  # Renamed function for clarity
        character_file: UploadFile = File(...,
                                          description="Character card file (e.g., PNG with 'chara' metadata, JSON, or Markdown with character data)."),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """
    Imports a character from a file. Supports:
    - PNG/WEBP images with embedded 'chara' metadata.
    - JSON files (.json) defining the character.
    - Markdown files (.md) with YAML frontmatter or JSON code blocks.

    If a character with the same name already exists, its details are returned.
    """
    try:
        file_content_bytes = await character_file.read()
        if not file_content_bytes:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

        # The library function import_and_save_character_from_file handles various input types.
        # It expects bytes, BytesIO, or a file path. We pass bytes.
        # It also handles logging internally.
        # It returns the character_id (int) if successful (new or existing), or None on failure.
        logger.info(
            f"Attempting to import character from file: {character_file.filename} (Type: {character_file.content_type}, Size: {len(file_content_bytes)} bytes)")

        # Pass the raw bytes to the library function
        # The library's `import_and_save_character_from_file` is expected to handle
        # ConflictError by returning the existing character's ID.
        char_id = import_and_save_character_from_file(db, file_content_bytes)

        if char_id is None:
            # The library function logs errors, so we raise a generic error here if it fails.
            # Specific errors like file format issues should ideally be raised from the library
            # or translated here if the library provides more context.
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Failed to import character from file. Check server logs for details. The file might be invalid, not a character card, or data could be missing/corrupt.")

        # If char_id is returned, fetch the character to build the response
        imported_char_db = db.get_character_card_by_id(char_id)
        if not imported_char_db:
            # This case should be rare if char_id was valid
            logger.error(f"Character ID {char_id} returned from import, but character not found in DB.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Character imported but could not be retrieved.")

        response_char_model = _convert_db_char_to_response_model(imported_char_db)

        # Determine if the character was newly created or already existed
        # This is a bit tricky without knowing if add_character_card or get_character_by_name was hit inside the lib due to conflict.
        # For now, assume success means the character is available.
        # A more robust way would be for the lib to return a status tuple (char_id, was_created)

        # Simplified message. The status code (201 vs 200) could also differentiate.
        # However, since the library handles conflict by returning existing ID,
        # we can treat it as "resource identified/available".
        message = f"Character '{response_char_model.name}' processed successfully."

        # Heuristic: if current version is 1 and creation time is recent, it's likely new.
        # For now, let's just return 201 for simplicity if an ID is retrieved.
        # Or, more simply, always return 200 if char_id is not None after the call.
        # If the library handles conflict by returning existing ID, it's not strictly a "creation".
        # Let's use 200 if it's known to exist, 201 if new. The lib doesn't easily give this info.
        # So, let's use a generic success message and rely on the returned data.

        return CharacterImportResponse(
            message=message,
            character=response_char_model
        )

    except HTTPException:  # Re-raise HTTPExceptions directly
        raise
    except CharactersRAGDBError as e:  # Catch DB errors from get_character_card_by_id after import
        logger.error(f"Database error after character import: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during character import: {e}", exc_info=True)
        # Redact potentially sensitive file content from general error messages
        detail_msg = "An unexpected error occurred during file import."
        if isinstance(e, (UnicodeDecodeError, json.JSONDecodeError)):  # More specific for common file issues
            detail_msg = "File content is not valid UTF-8 or contains malformed JSON."
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail_msg)
    finally:
        await character_file.close()


@router.get("/", response_model=List[CharacterResponse], summary="List all characters", tags=["Characters"])
async def list_characters(
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0)
):
    try:
        raw_cards = db.list_character_cards(limit=limit, offset=offset)
        return [_convert_db_char_to_response_model(card) for card in raw_cards]
    except CharactersRAGDBError as e:
        logger.error(f"Database error listing characters: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error listing characters: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")


@router.post(
    "/",
    response_model=CharacterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new character from payload",
    tags=["Characters"]
)
async def create_character(
        character_data: CharacterCreate,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    try:
        existing_char = db.get_character_card_by_name(character_data.name)
        if existing_char:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Character with name '{character_data.name}' already exists (ID: {existing_char['id']})."
            )

        db_ready_data = _prepare_char_data_for_db(character_data, is_update=False)  # This can raise HTTPException(400)
        char_id = db.add_character_card(db_ready_data)
        if not char_id:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Failed to create character in database.")

        created_char_db = db.get_character_card_by_id(char_id)
        if not created_char_db:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Failed to retrieve character after creation.")
        return _convert_db_char_to_response_model(created_char_db)
    except HTTPException:  # ADD THIS
        raise  # RE-RAISE HTTPExceptions directly
    except InputError as e:
        logger.warning(f"Input error creating character: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ConflictError as e:
        logger.warning(f"Conflict error creating character: {e}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except CharactersRAGDBError as e:
        logger.error(f"Database error creating character: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error creating character: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")



@router.get("/{character_id}", response_model=CharacterResponse, summary="Get a specific character by ID",
            tags=["Characters"])
async def get_character(
        character_id: int = Path(..., description="The ID of the character to retrieve.", gt=0),  # Use fastapi.Path
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """
    Retrieves the details of a specific character by their unique ID.
    The data returned is the raw character information as stored in the database,
    suitable for display or editing.
    """
    try:
        char_db = db.get_character_card_by_id(character_id)
        if not char_db: # Corrected: Check for None if character not found
            logger.info(f"Character with ID {character_id} not found in get_character endpoint.")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Character with ID {character_id} not found.")
        return _convert_db_char_to_response_model(char_db)
    except CharactersRAGDBError as e:
        logger.error(f"Database error getting character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {e}")
    except HTTPException: # Re-raise HTTPExceptions if they were raised intentionally
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")


@router.put("/{character_id}", response_model=CharacterResponse, summary="Update an existing character",
            tags=["Characters"])
async def update_character(
        update_data: CharacterUpdate,
        character_id: int = Path(..., description="The ID of the character to update.", gt=0),  # Use fastapi.Path
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """
    Updates an existing character's information.
    Only the fields provided in the request body will be updated.
    Requires the character's current version for optimistic concurrency control.
    If the character name is changed, it checks for conflicts with existing names.
    """
    try:
        current_char_db = db.get_character_card_by_id(character_id)
        if not current_char_db:  # Corrected: Check for None
            logger.info(f"Character with ID {character_id} not found for update.")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Character with ID {character_id} not found for update.")

        db_ready_update_data = _prepare_char_data_for_db(update_data, is_update=True)

        if not db_ready_update_data:
            logger.info(f"No updatable fields provided for character {character_id}. Returning current data.")
            return _convert_db_char_to_response_model(current_char_db)

        new_name = db_ready_update_data.get('name')
        if new_name is not None and new_name != current_char_db.get('name'):
            existing_char_with_new_name = db.get_character_card_by_name(new_name)
            if existing_char_with_new_name and existing_char_with_new_name.get('id') != character_id:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Another character with name '{new_name}' already exists (ID: {existing_char_with_new_name['id']})."
                )

        # Corrected: Changed parameter name from 'data' to 'card_data'
        success = db.update_character_card(
            character_id=character_id,
            card_data=db_ready_update_data,
            expected_version=current_char_db['version']
        )
        if not success:
            # This path assumes update_character_card returns False on logical failure
            # but doesn't raise an exception for it (e.g. version mismatch handled by return False).
            # If it raises ConflictError for version mismatch, that will be caught below.
            logger.error(
                f"Update for character {character_id} returned false from DB layer without raising an exception for a non-conflict issue.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Failed to update character in database (unexpected).")

        updated_char_db = db.get_character_card_by_id(character_id)
        if not updated_char_db:
            logger.error(f"Character {character_id} not found after supposedly successful update.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Failed to retrieve character after update.")

        return _convert_db_char_to_response_model(updated_char_db)
    except HTTPException:
        raise
    except InputError as e:  # From DB layer
        logger.warning(f"Input error updating character {character_id}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ConflictError as e:  # From DB layer (e.g. version mismatch)
        logger.warning(f"Conflict error updating character {character_id}: {e}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except CharactersRAGDBError as e:  # General DB errors
        logger.error(f"Database error updating character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {e}")
    except Exception as e:  # Catch-all for other unexpected errors
        logger.error(
            f"Unexpected error updating character {character_id}. Type: {type(e)}, Message: '{str(e)}', Args: {e.args}",
            exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"An unexpected error occurred: {type(e).__name__}")


@router.delete("/{character_id}", response_model=DeletionResponse, summary="Delete a character", tags=["Characters"])
async def delete_character(
        character_id: int = Path(..., description="The ID of the character to delete.", gt=0),  # Use fastapi.Path
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """
    Deletes a character by their unique ID.
    This operation uses the character's current version for optimistic concurrency.
    Note: The `CharactersRAGDB` class must implement a `delete_character_card` method
    for this endpoint to function correctly. Deletion might be restricted by database
    foreign key constraints if the character is associated with other data.
    """
    try:
        char_to_delete = db.get_character_card_by_id(character_id)
        if not char_to_delete:  # Corrected: Check for None
            logger.info(f"Character with ID {character_id} not found for deletion.")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Character with ID {character_id} not found for deletion.")

        logger.info(
            f"Attempting to delete character ID: {character_id} (Name: '{char_to_delete.get('name', 'N/A')}') with version {char_to_delete['version']}.")
        logger.warning(
            f"Ensure database foreign key constraints for 'conversations.character_id' are set appropriately (e.g., ON DELETE SET NULL or CASCADE) if characters with active conversations need to be deleted."
        )

        # Corrected: Check if the soft_delete_character_card method exists before calling
        if not hasattr(db, 'soft_delete_character_card'):
            logger.error(
                "`CharactersRAGDB` class does not have a `soft_delete_character_card` method. Deletion cannot proceed.")
            raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED,
                                detail="Character deletion functionality is not available on the server.")

        success = db.soft_delete_character_card(character_id, expected_version=char_to_delete['version'])

        if not success:
            # Similar to update, this assumes `delete_character_card` returns False for logical non-conflict failures.
            # If it raises ConflictError for version mismatch, it will be caught below.
            logger.error(
                f"Delete for character {character_id} returned false from DB layer without raising an exception for a non-conflict issue.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Failed to delete character (unexpected).")

        return DeletionResponse(
            message=f"Character with ID {character_id} ('{char_to_delete.get('name', 'N/A')}') marked as deleted successfully.",
            character_id=character_id)

    except HTTPException:
        raise
    except ConflictError as e:  # From DB layer (e.g. version mismatch or FK constraint)
        logger.warning(
            f"Conflict error deleting character {character_id}: {e}. "
            f"This might be due to a version mismatch or because the character is still referenced by other data (e.g., conversations) "
            f"and database constraints prevent deletion."
        )
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail=f"Cannot delete character: {e}. It might be in use or a concurrent modification occurred.")
    except CharactersRAGDBError as e:  # General DB errors
        logger.error(f"Database error deleting character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {e}")
    except Exception as e:  # Catch-all for other unexpected errors
        logger.error(
            f"Unexpected error deleting character {character_id}. Type: {type(e)}, Message: '{str(e)}', Args: {e.args}",
            exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"An unexpected error occurred: {type(e).__name__}")

#
# End of characters.py
#######################################################################################################################
