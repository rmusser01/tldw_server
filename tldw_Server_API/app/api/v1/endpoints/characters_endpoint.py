# characters.py
# Description:
#
# Imports
import base64
import json
import os
import pathlib
import struct
import zlib
from datetime import datetime
from typing import Any, Literal, Optional

#
# Third-party Libraries
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi import Path as FastAPIPath
from fastapi.responses import JSONResponse, Response
from loguru import logger
from starlette import status

# Constants for file upload validation
MAX_CHARACTER_FILE_SIZE = 10 * 1024 * 1024  # 10MB max file size
ALLOWED_EXTENSIONS = frozenset({".png", ".webp", ".jpeg", ".jpg", ".json", ".yaml", ".yml", ".txt", ".md"})


def _format_allowed_extensions() -> str:
    """Return supported import extensions in stable sorted order for API errors/docs."""
    return ", ".join(sorted(ALLOWED_EXTENSIONS))

def _detect_mime_type(data: bytes) -> Optional[str]:
    """
    Detect MIME type from file magic bytes.

    Returns the detected MIME type or None if unknown.
    This is more reliable than extension-based detection for security.
    """
    if len(data) < 12:
        return None

    # Check PNG
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return 'image/png'

    # Check WebP (RIFF....WEBP)
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'image/webp'

    # Check JPEG (various signatures)
    if data[:2] == b'\xff\xd8':
        return 'image/jpeg'

    # Check for JSON (starts with { or [, possibly with BOM or whitespace)
    stripped = data.lstrip(b'\xef\xbb\xbf \t\n\r')  # Strip BOM and whitespace
    if stripped and stripped[0:1] in (b'{', b'['):
        return 'application/json'

    # Check for YAML/Markdown (text files - check for printable ASCII)
    try:
        # Check first 100 bytes for text-like content
        sample = data[:100].decode('utf-8', errors='strict')
        # If it decodes as valid UTF-8 and contains printable chars, likely text
        if sample.isprintable() or '\n' in sample or '\r' in sample:
            return 'text/plain'
    except (UnicodeDecodeError, AttributeError):
        pass

    return None


def _validate_file_type(data: bytes, filename: Optional[str]) -> tuple[bool, str, Optional[str]]:
    """
    Validate file type via both magic bytes and extension.

    Returns:
        Tuple of (is_valid, error_message, detected_type) where detected_type is
        one of: "image", "json", "yaml", "text".
    """
    ext = None
    if filename:
        ext = pathlib.Path(filename).suffix.lower()

    # Check extension first
    if ext and ext not in ALLOWED_EXTENSIONS:
        return (
            False,
            f"File extension '{ext}' not allowed. Allowed: {_format_allowed_extensions()}",
            None,
        )

    # Detect MIME type from content
    detected_mime = _detect_mime_type(data)

    # For image files, validate magic bytes match extension claim
    if ext in ('.png', '.webp', '.jpeg', '.jpg'):
        expected_mimes = {
            '.png': 'image/png',
            '.webp': 'image/webp',
            '.jpeg': 'image/jpeg',
            '.jpg': 'image/jpeg',
        }
        expected = expected_mimes.get(ext)
        if expected:
            if detected_mime is None:
                return False, f"File content missing or invalid magic bytes for extension {ext}", None
            if detected_mime != expected:
                return False, f"File content doesn't match extension. Extension: {ext}, detected: {detected_mime}", None

    # Determine file type for processing
    if detected_mime in ('image/png', 'image/webp', 'image/jpeg'):
        return True, "", "image"

    if detected_mime == 'application/json':
        return True, "", "json"

    if ext in ('.json',):
        return True, "", "json"
    if ext in ('.yaml', '.yml'):
        return True, "", "yaml"
    if ext in ('.txt', '.md'):
        return True, "", "text"

    if detected_mime == 'text/plain':
        return True, "", "text"

    return False, "Could not determine file type", None
#
# Local Imports
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.schemas.character_schemas import (
    CharacterWorldBookDetachDeletionResponse,
    CharacterCreate,
    CharacterExemplarDeletionResponse,
    CharacterExemplarIn,
    CharacterExemplarResponse,
    CharacterExemplarSearchRequest,
    CharacterExemplarSearchResponse,
    CharacterExemplarSelectionConfig,
    CharacterExemplarSelectionDebug,
    CharacterExemplarSelectionDebugRequest,
    CharacterExemplarUpdate,
    CharacterImportResponse,
    CharacterListQueryResponse,
    CharacterRevertRequest,
    CharacterTagOperationRequest,
    CharacterTagOperationResponse,
    CharacterResponse,
    CharacterUpdate,
    CharacterVersionDiffField,
    CharacterVersionDiffResponse,
    CharacterVersionEntry,
    CharacterVersionListResponse,
    DeletionResponse,
    WorldBookDeletionResponse,
    WorldBookEntryDeletionResponse,
)
from tldw_Server_API.app.api.v1.schemas.world_book_schemas import (
    BulkEntryOperation,
    BulkOperationResponse,
    CharacterWorldBookAttachment,
    CharacterWorldBookResponse,
    EntryListResponse,
    ProcessContextRequest,
    ProcessContextResponse,
    WorldBookCreate,
    WorldBookEntryCreate,
    WorldBookEntryResponse,
    WorldBookEntryUpdate,
    WorldBookExport,
    WorldBookImportRequest,
    WorldBookImportResponse,
    WorldBookListResponse,
    WorldBookRuntimeConfig,
    WorldBookResponse,
    WorldBookStatistics,
    WorldBookUpdate,
    WorldBookWithEntries,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.Character_Chat.Character_Chat_Lib_facade import (
    create_new_character_from_data,
    delete_character_from_db,
    get_character_details,
    import_and_save_character_from_file,
    restore_character_from_db,
    search_characters_by_query_text,
    update_existing_character_details,
)
from tldw_Server_API.app.core.Character_Chat.character_limits import get_character_limits
from tldw_Server_API.app.core.Character_Chat.character_rate_limiter import get_character_rate_limiter
from tldw_Server_API.app.core.Character_Chat.constants import MAX_RECURSIVE_DEPTH
from tldw_Server_API.app.core.Character_Chat.modules.persona_exemplar_selector import (
    PersonaExemplarSelectorConfig,
    select_character_exemplars,
)
from tldw_Server_API.app.core.Character_Chat.modules.persona_exemplar_embeddings import (
    score_exemplars_with_embeddings,
    upsert_character_exemplar_embeddings,
    delete_character_exemplar_embeddings,
)
from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
    RestoreWindowExpiredError,
)

_CHARACTERS_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
    UnicodeError,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)

#
#######################################################################################################################
#
# Functions:



# --- Router ---
router = APIRouter()

_EXEMPLAR_SEARCH_HYBRID_CANDIDATE_CAP = 200
_EXEMPLAR_SEARCH_HYBRID_MIN_POOL = 40
_EXEMPLAR_SEARCH_HYBRID_VECTOR_WEIGHT = 0.55
_EXEMPLAR_SEARCH_HYBRID_LEXICAL_WEIGHT = 0.45
_CHARACTERS_RESTORE_RETENTION_ENV_KEY = "CHARACTERS_RESTORE_RETENTION_DAYS"
_CHARACTERS_RESTORE_RETENTION_DEFAULT_DAYS = 30
_CHARACTER_REVERT_FIELDS = (
    "name",
    "description",
    "personality",
    "scenario",
    "system_prompt",
    "post_history_instructions",
    "first_message",
    "message_example",
    "creator_notes",
    "alternate_greetings",
    "tags",
    "creator",
    "character_version",
    "extensions",
)


def _get_characters_restore_retention_days() -> int:
    """Read and validate restore retention policy from environment."""
    raw = os.getenv(
        _CHARACTERS_RESTORE_RETENTION_ENV_KEY,
        str(_CHARACTERS_RESTORE_RETENTION_DEFAULT_DAYS),
    )
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid {} value {!r}; using default {} days.",
            _CHARACTERS_RESTORE_RETENTION_ENV_KEY,
            raw,
            _CHARACTERS_RESTORE_RETENTION_DEFAULT_DAYS,
        )
        return _CHARACTERS_RESTORE_RETENTION_DEFAULT_DAYS

    if parsed < 1:
        logger.warning(
            "Non-positive {} value {}; using default {} days.",
            _CHARACTERS_RESTORE_RETENTION_ENV_KEY,
            parsed,
            _CHARACTERS_RESTORE_RETENTION_DEFAULT_DAYS,
        )
        return _CHARACTERS_RESTORE_RETENTION_DEFAULT_DAYS

    return parsed


def _build_character_revert_payload(
    snapshot_payload: dict[str, Any],
    current_character: dict[str, Any],
) -> dict[str, Any]:
    if not any(field in snapshot_payload for field in _CHARACTER_REVERT_FIELDS):
        raise InputError(
            "Target version does not contain a full character snapshot and cannot be reverted."
        )

    payload: dict[str, Any] = {}
    for field in _CHARACTER_REVERT_FIELDS:
        if field in snapshot_payload:
            payload[field] = snapshot_payload.get(field)
            continue
        if field in current_character:
            payload[field] = current_character.get(field)

    if not payload.get("name"):
        raise InputError("Target version snapshot is missing a valid character name.")
    return payload


def _build_character_version_entry(entry: dict[str, Any]) -> CharacterVersionEntry:
    raw_version = int(entry.get("version") or 0)
    normalized_version = raw_version if raw_version >= 1 else 1
    return CharacterVersionEntry(
        change_id=int(entry.get("change_id") or 0),
        version=normalized_version,
        operation=str(entry.get("operation") or "update"),
        timestamp=entry.get("timestamp"),
        client_id=entry.get("client_id"),
        payload=entry.get("payload")
        if isinstance(entry.get("payload"), dict)
        else {},
    )


def _normalize_snapshot_comparison_value(value: Any) -> str:
    if value is None:
        return "__null__"
    if isinstance(value, str):
        return f"str:{value}"
    if isinstance(value, (int, float, bool)):
        return f"scalar:{value}"
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return str(value)


def _build_character_version_diff_fields(
    from_payload: dict[str, Any],
    to_payload: dict[str, Any],
) -> list[CharacterVersionDiffField]:
    changed_fields: list[CharacterVersionDiffField] = []
    for field in _CHARACTER_REVERT_FIELDS:
        old_value = from_payload.get(field)
        new_value = to_payload.get(field)
        if _normalize_snapshot_comparison_value(old_value) == _normalize_snapshot_comparison_value(new_value):
            continue
        changed_fields.append(
            CharacterVersionDiffField(
                field=field,
                old_value=old_value,
                new_value=new_value,
            )
        )
    return changed_fields


# --- Helper Functions (Keep _convert_db_char_to_response_model as is) ---
def _convert_db_char_to_response_model(
        char_dict_from_db: dict[str, Any],
        *,
        include_image_base64: bool = True
) -> CharacterResponse:
    response_data = char_dict_from_db.copy()
    if response_data.get('image') and isinstance(response_data['image'], bytes):
        if include_image_base64:
            try:
                response_data['image_base64'] = base64.b64encode(response_data['image']).decode('utf-8')
                response_data['image_present'] = True
            except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"Error encoding image for char {response_data.get('id')}: {e}")
                response_data['image_base64'] = None
                response_data['image_present'] = False
        else:
            response_data['image_base64'] = None
            response_data['image_present'] = True
    else:
        response_data['image_base64'] = None
        response_data['image_present'] = bool(
            response_data.get('image') and isinstance(response_data.get('image'), bytes))
    if response_data.get("updated_at") is None:
        response_data["updated_at"] = response_data.get("last_modified")
    if response_data.get("last_modified") is None:
        response_data["last_modified"] = response_data.get("updated_at")
    response_data.pop('image', None)
    return CharacterResponse.model_validate(response_data)


def _build_conflict_import_response(
    error: ConflictError,
    db: CharactersRAGDB,
) -> Optional[CharacterImportResponse]:
    existing_id: Optional[int] = None
    try:
        entity_id = getattr(error, "entity_id", None)
        if isinstance(entity_id, int):
            existing_id = entity_id
        elif isinstance(entity_id, str) and entity_id.strip():
            existing_char_obj = db.get_character_card_by_name(entity_id)
            if existing_char_obj:
                try:
                    existing_id = int(existing_char_obj.get("id"))
                except (TypeError, ValueError) as exc:
                    logger.debug(f"Invalid conflict character id from name lookup: {exc}")
    except (CharactersRAGDBError, ValueError, TypeError) as exc:
        logger.debug(f"Failed to resolve conflict character id: {exc}")
        return None
    if not existing_id:
        return None
    existing_char_db = db.get_character_card_by_id(existing_id)
    if not existing_char_db:
        return None
    existing_name = existing_char_db.get("name", "Unknown")
    return CharacterImportResponse(
        id=existing_id,
        name=existing_name,
        message=(
            f"Character '{existing_name}' already exists (ID: {existing_id}). "
            "Details provided."
        ),
        character=_convert_db_char_to_response_model(existing_char_db),
    )


def _coerce_string_list(value: Any) -> list[str]:
    """Normalize API/DB mixed list payloads to a list[str]."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        except _CHARACTERS_NONCRITICAL_EXCEPTIONS:
            pass
        return [value] if value.strip() else []
    return [str(value)] if str(value).strip() else []


def _flatten_character_exemplar_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten exemplar API payload structure into DB fields."""
    flat: dict[str, Any] = {}

    if 'text' in payload:
        flat['text'] = payload.get('text')
    if 'novelty_hint' in payload:
        flat['novelty_hint'] = payload.get('novelty_hint')
    if 'length_tokens' in payload:
        flat['length_tokens'] = payload.get('length_tokens')

    source = payload.get('source')
    if isinstance(source, dict):
        if 'type' in source:
            flat['source_type'] = source.get('type')
        if 'url_or_id' in source:
            flat['source_url_or_id'] = source.get('url_or_id')
        if 'date' in source:
            flat['source_date'] = source.get('date')

    labels = payload.get('labels')
    if isinstance(labels, dict):
        if 'emotion' in labels:
            flat['emotion'] = labels.get('emotion')
        if 'scenario' in labels:
            flat['scenario'] = labels.get('scenario')
        if 'rhetorical' in labels:
            flat['rhetorical'] = labels.get('rhetorical')
        if 'register' in labels:
            flat['register'] = labels.get('register')

    safety = payload.get('safety')
    if isinstance(safety, dict):
        if 'allowed' in safety:
            flat['safety_allowed'] = safety.get('allowed')
        if 'blocked' in safety:
            flat['safety_blocked'] = safety.get('blocked')

    rights = payload.get('rights')
    if isinstance(rights, dict):
        if 'public_figure' in rights:
            flat['rights_public_figure'] = rights.get('public_figure')
        if 'notes' in rights:
            flat['rights_notes'] = rights.get('notes')

    return flat


def _is_permission_denied_db_error(error: Exception) -> bool:
    """Best-effort matcher for DB/auth scope violations surfaced as DB errors."""
    message = str(error).strip().lower()
    if not message:
        return False
    permission_markers = (
        "permission denied",
        "forbidden",
        "not authorized",
        "not authorised",
        "insufficient privilege",
        "insufficient permissions",
    )
    return any(marker in message for marker in permission_markers)


def _convert_db_exemplar_to_response_model(exemplar_dict_from_db: dict[str, Any]) -> CharacterExemplarResponse:
    """Convert DB exemplar row shape to API response shape."""
    source_type = exemplar_dict_from_db.get('source_type') or 'other'
    novelty_hint = exemplar_dict_from_db.get('novelty_hint') or 'unknown'
    emotion = exemplar_dict_from_db.get('emotion') or 'other'
    scenario = exemplar_dict_from_db.get('scenario') or 'other'

    response_data = {
        'id': exemplar_dict_from_db.get('id'),
        'character_id': exemplar_dict_from_db.get('character_id'),
        'text': exemplar_dict_from_db.get('text'),
        'source': {
            'type': source_type,
            'url_or_id': exemplar_dict_from_db.get('source_url_or_id'),
            'date': exemplar_dict_from_db.get('source_date'),
        },
        'novelty_hint': novelty_hint,
        'labels': {
            'emotion': emotion,
            'scenario': scenario,
            'rhetorical': _coerce_string_list(exemplar_dict_from_db.get('rhetorical')),
            'register': exemplar_dict_from_db.get('register'),
        },
        'safety': {
            'allowed': _coerce_string_list(exemplar_dict_from_db.get('safety_allowed')),
            'blocked': _coerce_string_list(exemplar_dict_from_db.get('safety_blocked')),
        },
        'rights': {
            'public_figure': bool(exemplar_dict_from_db.get('rights_public_figure', True)),
            'notes': exemplar_dict_from_db.get('rights_notes'),
        },
        'length_tokens': exemplar_dict_from_db.get('length_tokens'),
        'created_at': exemplar_dict_from_db.get('created_at'),
        'updated_at': exemplar_dict_from_db.get('updated_at'),
    }

    return CharacterExemplarResponse.model_validate(response_data)


def _resolve_exemplar_embedding_user_id(db: CharactersRAGDB) -> str | None:
    user_id = str(getattr(db, "client_id", "") or "").strip()
    return user_id or None


def _sync_exemplar_embeddings_best_effort(
    *,
    db: CharactersRAGDB,
    character_id: int,
    exemplars: list[dict[str, Any]],
) -> None:
    user_id = _resolve_exemplar_embedding_user_id(db)
    if not user_id or not exemplars:
        return
    try:
        upsert_character_exemplar_embeddings(
            user_id=user_id,
            character_id=character_id,
            exemplars=exemplars,
        )
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        logger.warning(
            "Failed to sync exemplar embeddings for character {} user {} with non-fatal base exception: {}",
            character_id,
            user_id,
            exc,
        )
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(
            "Failed to sync exemplar embeddings for character {} user {}: {}",
            character_id,
            user_id,
            exc,
        )


def _delete_exemplar_embeddings_best_effort(
    *,
    db: CharactersRAGDB,
    character_id: int,
    exemplar_ids: list[str],
) -> None:
    user_id = _resolve_exemplar_embedding_user_id(db)
    if not user_id or not exemplar_ids:
        return
    try:
        delete_character_exemplar_embeddings(
            user_id=user_id,
            character_id=character_id,
            exemplar_ids=exemplar_ids,
        )
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        logger.warning(
            "Failed to delete exemplar embeddings for character {} user {} with non-fatal base exception: {}",
            character_id,
            user_id,
            exc,
        )
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(
            "Failed to delete exemplar embeddings for character {} user {}: {}",
            character_id,
            user_id,
            exc,
        )


def _build_lexical_rank_scores(exemplars: list[dict[str, Any]]) -> dict[str, float]:
    """Build deterministic lexical rank scores in [0, 1] for hybrid re-ranking."""
    if not exemplars:
        return {}

    denom = max(1, len(exemplars) - 1)
    scores: dict[str, float] = {}
    for idx, item in enumerate(exemplars):
        exemplar_id = str(item.get("id") or "").strip()
        if not exemplar_id:
            continue
        scores[exemplar_id] = round(1.0 - (idx / denom), 6)
    return scores


def _as_sortable_timestamp(value: Any) -> str:
    """Normalize DB timestamp-ish values for stable descending sort keys."""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value or "")


def _search_character_exemplars_hybrid_best_effort(
    *,
    db: CharactersRAGDB,
    character_id: int,
    search_request: CharacterExemplarSearchRequest,
) -> tuple[list[dict[str, Any]], int]:
    """Run best-effort lexical+embedding hybrid search with safe lexical fallback."""
    normalized_query = str(search_request.query or "").strip()
    if not normalized_query or not search_request.use_embedding_scores:
        return db.search_character_exemplars(
            character_id,
            query=search_request.query,
            emotion=search_request.filter.emotion,
            scenario=search_request.filter.scenario,
            rhetorical=search_request.filter.rhetorical,
            limit=search_request.limit,
            offset=search_request.offset,
        )

    base_window = max(1, int(search_request.limit) + int(search_request.offset))
    candidate_pool_size = min(
        _EXEMPLAR_SEARCH_HYBRID_CANDIDATE_CAP,
        max(_EXEMPLAR_SEARCH_HYBRID_MIN_POOL, base_window * 4),
    )

    lexical_candidates, _ = db.search_character_exemplars(
        character_id,
        query=normalized_query,
        emotion=search_request.filter.emotion,
        scenario=search_request.filter.scenario,
        rhetorical=search_request.filter.rhetorical,
        limit=candidate_pool_size,
        offset=0,
    )
    lexical_scores = _build_lexical_rank_scores(lexical_candidates)

    candidate_map: dict[str, dict[str, Any]] = {}
    for item in lexical_candidates:
        exemplar_id = str(item.get("id") or "").strip()
        if exemplar_id:
            candidate_map[exemplar_id] = item

    if len(candidate_map) < candidate_pool_size:
        try:
            listed = db.list_character_exemplars(character_id, limit=candidate_pool_size, offset=0)
            for item in listed:
                exemplar_id = str(item.get("id") or "").strip()
                if exemplar_id and exemplar_id not in candidate_map:
                    candidate_map[exemplar_id] = item
                if len(candidate_map) >= candidate_pool_size:
                    break
        except CharactersRAGDBError as exc:
            logger.warning("Hybrid exemplar search list backfill failed for character {}: {}", character_id, exc)

    all_candidates = list(candidate_map.values())
    if not all_candidates:
        return [], 0

    user_id = _resolve_exemplar_embedding_user_id(db)
    if not user_id:
        total = len(lexical_candidates)
        return lexical_candidates[search_request.offset:search_request.offset + search_request.limit], total

    try:
        embedding_scores = score_exemplars_with_embeddings(
            normalized_query,
            all_candidates,
            user_id=user_id,
            character_id=character_id,
            model_id_override=search_request.embedding_model_id,
        )
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning("Hybrid exemplar search embedding scoring failed for character {}: {}", character_id, exc)
        total = len(lexical_candidates)
        return lexical_candidates[search_request.offset:search_request.offset + search_request.limit], total
    if not embedding_scores:
        total = len(lexical_candidates)
        return lexical_candidates[search_request.offset:search_request.offset + search_request.limit], total

    ranked: list[dict[str, Any]] = []
    for item in all_candidates:
        exemplar_id = str(item.get("id") or "").strip()
        if not exemplar_id:
            continue
        lexical_score = float(lexical_scores.get(exemplar_id, 0.0))
        vector_score = float(embedding_scores.get(exemplar_id, 0.0))
        if lexical_score <= 0.0 and vector_score <= 0.0:
            continue
        hybrid_score = (
            _EXEMPLAR_SEARCH_HYBRID_VECTOR_WEIGHT * vector_score
            + _EXEMPLAR_SEARCH_HYBRID_LEXICAL_WEIGHT * lexical_score
        )
        ranked.append(
            {
                "item": item,
                "hybrid_score": round(hybrid_score, 6),
                "vector_score": round(vector_score, 6),
                "lexical_score": round(lexical_score, 6),
                "updated_at": _as_sortable_timestamp(item.get("updated_at") or item.get("created_at")),
            }
        )

    ranked.sort(
        key=lambda entry: (
            entry["hybrid_score"],
            entry["vector_score"],
            entry["lexical_score"],
            entry["updated_at"],
        ),
        reverse=True,
    )

    sorted_items = [entry["item"] for entry in ranked]
    total = len(sorted_items)
    return sorted_items[search_request.offset:search_request.offset + search_request.limit], total


# --- API Endpoints ---

@router.post("/import", response_model=CharacterImportResponse,
             summary="Import character card", tags=["characters"],
             status_code=status.HTTP_201_CREATED)
async def import_character_endpoint(
        character_file: UploadFile = File(
            ...,
            description="Character card file (PNG, WEBP, JPEG, JSON, YAML, YML, MD, TXT).",
        ),
        allow_image_only: bool = Form(False),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        current_user: User = Depends(get_request_user),
):
    """
    Import a character card from a file.

    Supports:
    - Image files (PNG, WEBP, JPEG) with embedded character data
    - JSON files (including Character Card V3 format)
    - YAML files
    - Markdown and plain text files (including text containing JSON)

    For JSON data, you can upload a .json file or a text file containing JSON.
    """
    try:
        try:
            limits = get_character_limits()
            max_import_size_mb = int(limits.max_import_size_mb)
            max_import_bytes = max_import_size_mb * 1024 * 1024
        except _CHARACTERS_NONCRITICAL_EXCEPTIONS:
            max_import_bytes = MAX_CHARACTER_FILE_SIZE
            max_import_size_mb = MAX_CHARACTER_FILE_SIZE // (1024 * 1024)

        # Pre-size check using content-length header (if available)
        # This prevents loading very large files into memory
        if character_file.size is not None and character_file.size > max_import_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum allowed size is {max_import_size_mb}MB"
            )

        # Read file with size limit
        file_content_bytes = await character_file.read()
        if not file_content_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty"
            )

        # Post-read size check (in case content-length was not accurate)
        if len(file_content_bytes) > max_import_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum allowed size is {max_import_size_mb}MB"
            )

        # Validate file type via magic bytes and extension
        is_valid, error_msg, detected_type = _validate_file_type(
            file_content_bytes, character_file.filename
        )
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )

        # Check rate limits
        rate_limiter = get_character_rate_limiter()
        await rate_limiter.check_rate_limit(current_user.id, "character_import")
        rate_limiter.check_import_size(len(file_content_bytes))

        # Check character count limit. The helper expects the current count
        # (before this import) and rejects when current_count >= max_characters.
        existing_chars = db.list_character_cards(limit=10000)
        await rate_limiter.check_character_limit(current_user.id, len(existing_chars))

        logger.info(f"API: Importing character from file: {character_file.filename}")

        # Use the detected type from validation
        file_type_validated = detected_type

        success, message, char_id = import_and_save_character_from_file(
            db,
            file_content=file_content_bytes,
            file_type=file_type_validated,
            file_name=character_file.filename,
            allow_image_only=allow_image_only
        )

        if not success or not char_id:
            if message == "missing_character_data":
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "code": "missing_character_data",
                        "message": (
                            "No character data detected in image metadata. "
                            "Continue to create an image-only character?"
                        ),
                        "can_import_image_only": True
                    }
                )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message or "Failed to import character"
            )

        # Retrieve the imported character
        imported_char = db.get_character_card_by_id(char_id)
        if not imported_char:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Character imported but could not be retrieved"
            )

        logger.info(f"Character '{imported_char.get('name', 'Unknown')}' imported successfully (ID: {char_id})")

        return CharacterImportResponse(
            id=char_id,
            name=imported_char.get('name', 'Unknown'),
            message=f"Character '{imported_char.get('name', 'Unknown')}' imported successfully",
            character=_convert_db_char_to_response_model(imported_char)
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
        conflict_response = _build_conflict_import_response(e, db)
        if conflict_response:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content=conflict_response.model_dump()
            )

        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e

    except (InputError, CharactersRAGDBError) as e:
        logger.error(f"Error during character import: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except HTTPException:
        raise
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error during character import: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred during file import.") from e
    finally:
        await character_file.close()



@router.get("/", response_model=list[CharacterResponse], summary="List characters", tags=["characters"])
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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error listing characters: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.") from e


@router.get("/query", response_model=CharacterListQueryResponse, summary="Query characters", tags=["characters"])
async def query_characters(
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        page: int = Query(1, ge=1),
        page_size: int = Query(25, ge=1, le=100),
        query: Optional[str] = Query(None, description="Search term across name/description/prompt fields"),
        tags: list[str] = Query([], description="Filter by tags"),
        match_all_tags: bool = Query(False, description="Require all tags instead of any tag"),
        creator: Optional[str] = Query(None, description="Filter by creator"),
        has_conversations: Optional[bool] = Query(None, description="Filter by conversation existence"),
        favorite_only: bool = Query(False, description="Return only favorited characters"),
        created_from: Optional[str] = Query(None, description="Created-at lower bound (ISO timestamp)"),
        created_to: Optional[str] = Query(None, description="Created-at upper bound (ISO timestamp)"),
        updated_from: Optional[str] = Query(None, description="Updated-at lower bound (ISO timestamp)"),
        updated_to: Optional[str] = Query(None, description="Updated-at upper bound (ISO timestamp)"),
        include_deleted: bool = Query(False, description="Include soft-deleted characters in query results"),
        deleted_only: bool = Query(False, description="Return only soft-deleted characters"),
        sort_by: Literal[
            "name",
            "creator",
            "created_at",
            "updated_at",
            "last_used_at",
            "conversation_count"
        ] = Query("name"),
        sort_order: Literal["asc", "desc"] = Query("asc"),
        include_image_base64: bool = Query(False, description="Include image_base64 payloads in list results")
):
    try:
        offset = (page - 1) * page_size
        raw_cards, total = db.query_character_cards(
            query=query,
            tags=tags,
            match_all_tags=match_all_tags,
            creator=creator,
            has_conversations=has_conversations,
            favorite_only=favorite_only,
            created_from=created_from,
            created_to=created_to,
            updated_from=updated_from,
            updated_to=updated_to,
            include_deleted=include_deleted,
            deleted_only=deleted_only,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=page_size,
            offset=offset
        )
        items = [
            _convert_db_char_to_response_model(
                card,
                include_image_base64=include_image_base64
            )
            for card in raw_cards
        ]
        return CharacterListQueryResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            has_more=(offset + len(items)) < total
        )
    except CharactersRAGDBError as e:
        logger.error(f"DB error querying characters: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error querying characters: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.") from e


@router.get("/rate-limit-status", summary="Get rate limit status", tags=["characters"])
async def get_rate_limit_status(
    current_user: User = Depends(get_request_user)
):
    """Get current rate limit usage statistics for the authenticated user."""
    rate_limiter = get_character_rate_limiter()
    stats = await rate_limiter.get_usage_stats(current_user.id)
    return stats


@router.post("/", response_model=CharacterResponse, status_code=status.HTTP_201_CREATED, summary="Create character",
             tags=["characters"])
async def create_new_character_endpoint(
        character_data: CharacterCreate,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        current_user: User = Depends(get_request_user)
):
    try:
        # Check rate limits
        rate_limiter = get_character_rate_limiter()
        await rate_limiter.check_rate_limit(current_user.id, "character_create")

        # Check character count limit. The helper expects the current count
        # (before this create) and rejects when current_count >= max_characters.
        existing_chars = db.list_character_cards(limit=10000)
        await rate_limiter.check_character_limit(current_user.id, len(existing_chars))

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
        raise HTTPException(status_code=status_code, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.error(f"DB error creating character: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except HTTPException:
        raise
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error creating character: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.") from e


@router.get("/filter", response_model=list[CharacterResponse],
            summary="Filter characters by tags", tags=["characters"])
async def filter_characters_by_tags(
    tags: list[str] = Query([], description="List of tags to filter by"),
    match_all: bool = Query(False, description="Require all tags (AND) vs any tag (OR)"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """
    Filter characters by tags.

    Args:
        tags: List of tags to filter by
        match_all: If True, require all tags; if False, match any tag
        limit: Maximum results
        offset: Pagination offset
        db: Database instance

    Returns:
        List of characters matching the tag criteria
    """
    try:
        # If no tags specified, return all characters
        if not tags:
            results = db.list_character_cards(limit=limit, offset=offset)
            return [_convert_db_char_to_response_model(char) for char in results]

        # Get all characters (we'll filter in memory for now)
        all_characters = db.list_character_cards(limit=1000, offset=0)

        filtered = []
        for char in all_characters:
            char_tags = char.get('tags', [])
            if isinstance(char_tags, str):
                import json
                try:
                    char_tags = json.loads(char_tags)
                except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
                    logger.debug(f"Failed to decode character tags JSON; skipping tags. error={e}")
                    char_tags = []

            if not char_tags:
                continue

            # Check tag matching
            if match_all:
                # All specified tags must be present
                if all(tag in char_tags for tag in tags):
                    filtered.append(char)
            else:
                # Any specified tag must be present
                if any(tag in char_tags for tag in tags):
                    filtered.append(char)

        # Apply pagination
        paginated = filtered[offset:offset+limit]

        return [_convert_db_char_to_response_model(char) for char in paginated]

    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error filtering characters by tags: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while filtering characters"
        ) from e


@router.post(
    "/tags/operations",
    response_model=CharacterTagOperationResponse,
    summary="Manage character tags in bulk",
    tags=["characters"],
)
async def manage_character_tags_endpoint(
    request: CharacterTagOperationRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
):
    """Apply rename, merge, or delete operations to character tags."""
    try:
        result = db.manage_character_tags(
            operation=request.operation,
            source_tag=request.source_tag,
            target_tag=request.target_tag,
        )
        return CharacterTagOperationResponse.model_validate(result)
    except InputError as e:
        logger.warning(f"Invalid tag operation request: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.error(f"DB error applying tag operation: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error applying tag operation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while updating character tags",
        ) from e


@router.post(
    "/{character_id:int}/exemplars",
    response_model=CharacterExemplarResponse | list[CharacterExemplarResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Create character exemplar(s)",
    tags=["characters"],
)
async def create_character_exemplars_endpoint(
    character_id: int = FastAPIPath(..., description="Character ID", gt=0),
    exemplar_payload: CharacterExemplarIn | list[CharacterExemplarIn] = ...,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
):
    """Create one or multiple exemplars for a character."""
    try:
        if not db.get_character_card_by_id(character_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Character with ID {character_id} not found.",
            )

        if isinstance(exemplar_payload, list):
            if not exemplar_payload:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Exemplar payload list cannot be empty.",
                )
            created_items: list[CharacterExemplarResponse] = []
            created_rows_for_sync: list[dict[str, Any]] = []
            for exemplar in exemplar_payload:
                flattened = _flatten_character_exemplar_payload(exemplar.model_dump(exclude_none=False))
                created = db.add_character_exemplar(character_id, flattened)
                created_items.append(_convert_db_exemplar_to_response_model(created))
                created_rows_for_sync.append(created)
            _sync_exemplar_embeddings_best_effort(
                db=db,
                character_id=character_id,
                exemplars=created_rows_for_sync,
            )
            return created_items

        flattened = _flatten_character_exemplar_payload(exemplar_payload.model_dump(exclude_none=False))
        created = db.add_character_exemplar(character_id, flattened)
        _sync_exemplar_embeddings_best_effort(
            db=db,
            character_id=character_id,
            exemplars=[created],
        )
        return _convert_db_exemplar_to_response_model(created)
    except (InputError, ConflictError) as e:
        status_code = status.HTTP_400_BAD_REQUEST if isinstance(e, InputError) else status.HTTP_409_CONFLICT
        raise HTTPException(status_code=status_code, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.error(f"DB error creating exemplar(s) for character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except HTTPException:
        raise
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error creating exemplar(s) for character {character_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while creating exemplars.",
        ) from e


@router.get(
    "/{character_id:int}/exemplars/{exemplar_id}",
    response_model=CharacterExemplarResponse,
    summary="Get character exemplar by ID",
    tags=["characters"],
)
async def get_character_exemplar_endpoint(
    character_id: int = FastAPIPath(..., description="Character ID", gt=0),
    exemplar_id: str = FastAPIPath(..., description="Exemplar ID"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
):
    """Fetch a single exemplar for a character."""
    try:
        exemplar = db.get_character_exemplar_by_id(character_id, exemplar_id)
        if not exemplar:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Exemplar '{exemplar_id}' not found for character {character_id}.",
            )
        return _convert_db_exemplar_to_response_model(exemplar)
    except CharactersRAGDBError as e:
        logger.error(f"DB error fetching exemplar {exemplar_id} for character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except HTTPException:
        raise
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(
            f"Unexpected error fetching exemplar {exemplar_id} for character {character_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while fetching exemplar.",
        ) from e


@router.put(
    "/{character_id:int}/exemplars/{exemplar_id}",
    response_model=CharacterExemplarResponse,
    summary="Update character exemplar",
    tags=["characters"],
)
async def update_character_exemplar_endpoint(
    update_data: CharacterExemplarUpdate,
    character_id: int = FastAPIPath(..., description="Character ID", gt=0),
    exemplar_id: str = FastAPIPath(..., description="Exemplar ID"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
):
    """Update a character exemplar."""
    try:
        if not db.get_character_card_by_id(character_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Character with ID {character_id} not found.",
            )
        flattened = _flatten_character_exemplar_payload(update_data.model_dump(exclude_unset=True))
        updated = db.update_character_exemplar(character_id, exemplar_id, flattened)
        if not updated:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Exemplar '{exemplar_id}' not found for character {character_id}.",
            )
        _sync_exemplar_embeddings_best_effort(
            db=db,
            character_id=character_id,
            exemplars=[updated],
        )
        return _convert_db_exemplar_to_response_model(updated)
    except InputError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.error(f"DB error updating exemplar {exemplar_id} for character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except HTTPException:
        raise
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(
            f"Unexpected error updating exemplar {exemplar_id} for character {character_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while updating exemplar.",
        ) from e


@router.delete(
    "/{character_id:int}/exemplars/{exemplar_id}",
    response_model=CharacterExemplarDeletionResponse,
    summary="Delete character exemplar",
    tags=["characters"],
)
async def delete_character_exemplar_endpoint(
    character_id: int = FastAPIPath(..., description="Character ID", gt=0),
    exemplar_id: str = FastAPIPath(..., description="Exemplar ID"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
):
    """Soft-delete a character exemplar."""
    try:
        existing = db.get_character_exemplar_by_id(character_id, exemplar_id, include_deleted=True)
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Exemplar '{exemplar_id}' not found for character {character_id}.",
            )

        deleted = db.soft_delete_character_exemplar(character_id, exemplar_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete exemplar '{exemplar_id}'.",
            )
        _delete_exemplar_embeddings_best_effort(
            db=db,
            character_id=character_id,
            exemplar_ids=[exemplar_id],
        )

        return CharacterExemplarDeletionResponse(
            message=f"Exemplar '{exemplar_id}' soft-deleted for character {character_id}.",
            character_id=character_id,
            exemplar_id=exemplar_id,
        )
    except CharactersRAGDBError as e:
        logger.error(f"DB error deleting exemplar {exemplar_id} for character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except HTTPException:
        raise
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(
            f"Unexpected error deleting exemplar {exemplar_id} for character {character_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while deleting exemplar.",
        ) from e


@router.post(
    "/{character_id:int}/exemplars/search",
    response_model=CharacterExemplarSearchResponse,
    summary="Search character exemplars",
    tags=["characters"],
)
async def search_character_exemplars_endpoint(
    search_request: CharacterExemplarSearchRequest,
    character_id: int = FastAPIPath(..., description="Character ID", gt=0),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
):
    """Search exemplars for a character using query + labels filters."""
    try:
        if not db.get_character_card_by_id(character_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Character with ID {character_id} not found.",
            )

        results, total = _search_character_exemplars_hybrid_best_effort(
            db=db,
            character_id=character_id,
            search_request=search_request,
        )
        return CharacterExemplarSearchResponse(
            items=[_convert_db_exemplar_to_response_model(item) for item in results],
            total=total,
        )
    except InputError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.error(f"DB error searching exemplars for character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except HTTPException:
        raise
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error searching exemplars for character {character_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while searching exemplars.",
        ) from e


@router.post(
    "/{character_id:int}/exemplars/select/debug",
    response_model=CharacterExemplarSelectionDebug,
    summary="Debug exemplar selection",
    tags=["characters"],
)
async def select_character_exemplars_debug_endpoint(
    request: CharacterExemplarSelectionDebugRequest,
    character_id: int = FastAPIPath(..., description="Character ID", gt=0),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
):
    """Return selected exemplars and scoring metadata for debug workflows."""
    try:
        if not db.get_character_card_by_id(character_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Character with ID {character_id} not found.",
            )

        selection_config: CharacterExemplarSelectionConfig = request.selection_config
        selector_config = PersonaExemplarSelectorConfig(
            budget_tokens=selection_config.budget_tokens,
            max_exemplar_tokens=selection_config.max_exemplar_tokens,
            mmr_lambda=selection_config.mmr_lambda,
        )
        embedding_callback = None
        if selection_config.use_embedding_scores:
            embedding_model_id = selection_config.embedding_model_id
            embedding_user_id = _resolve_exemplar_embedding_user_id(db)

            def _embedding_callback(user_turn: str, candidates: list[dict[str, Any]]) -> dict[str, float]:
                return score_exemplars_with_embeddings(
                    user_turn,
                    candidates,
                    user_id=embedding_user_id,
                    character_id=character_id,
                    model_id_override=embedding_model_id,
                )

            embedding_callback = _embedding_callback

        selected_result = select_character_exemplars(
            db=db,
            character_id=character_id,
            user_turn=request.user_turn,
            config=selector_config,
            embedding_score_fn=embedding_callback,
        )

        return CharacterExemplarSelectionDebug(
            selected=[_convert_db_exemplar_to_response_model(item) for item in selected_result.selected],
            budget_tokens=selected_result.budget_tokens_used,
            coverage=selected_result.coverage,
            scores=selected_result.scores,
        )
    except (InputError, ValueError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.error(f"DB error running exemplar debug selection for character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except HTTPException:
        raise
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(
            f"Unexpected error running exemplar debug selection for character {character_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while selecting exemplars.",
        ) from e



# --- World Book List (deduplicated, defined before /{character_id}) ---


@router.get("/world-books", response_model=WorldBookListResponse, summary="List world books", tags=["World Books"])
async def list_world_books(
        include_disabled: bool = Query(False, description="Include disabled world books"),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """List all world books for the user."""
    try:
        service = WorldBookService(db)
        books = service.list_world_books(include_disabled=include_disabled)

        # Add entry counts in one query to avoid N+1 lookups.
        book_ids = [
            int(book["id"])
            for book in books
            if isinstance(book.get("id"), int)
        ]
        entry_counts = service.get_entry_counts_for_world_books(book_ids)
        for book in books:
            try:
                world_book_id = int(book.get("id"))
            except (TypeError, ValueError):
                world_book_id = -1
            book["entry_count"] = int(entry_counts.get(world_book_id, 0))

        # Convert to response models (filter unexpected fields)
        allowed_keys = {
            'id', 'name', 'description', 'scan_depth', 'token_budget',
            'recursive_scanning', 'enabled', 'created_at', 'last_modified',
            'version', 'entry_count'
        }
        filtered = []
        for book in books:
            filtered_dict = {k: book.get(k) for k in allowed_keys if k in book}
            filtered.append(WorldBookResponse(**filtered_dict))
        book_responses = filtered

        # Calculate statistics
        enabled_count = sum(1 for b in book_responses if b.enabled)
        disabled_count = len(book_responses) - enabled_count

        return WorldBookListResponse(
            world_books=book_responses,
            total=len(book_responses),
            enabled_count=enabled_count,
            disabled_count=disabled_count
        )

    except CharactersRAGDBError as e:
        logger.error(f"DB error listing world books: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error listing world books: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred") from e


@router.get(
    "/world-books/config",
    response_model=WorldBookRuntimeConfig,
    summary="Get world book runtime config",
    tags=["World Books"],
)
async def get_world_book_runtime_config():
    """Expose runtime constants used by world-book authoring UIs."""
    return WorldBookRuntimeConfig(max_recursive_depth=MAX_RECURSIVE_DEPTH)


@router.get("/{character_id:int}", response_model=CharacterResponse, summary="Get character by ID", tags=["characters"])
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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except HTTPException:
        raise
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error getting character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.") from e


@router.get(
    "/{character_id:int}/versions/diff",
    response_model=CharacterVersionDiffResponse,
    summary="Diff two character versions",
    tags=["characters"],
)
async def get_character_versions_diff_endpoint(
    character_id: int = FastAPIPath(..., description="ID of the character.", gt=0),
    from_version: int = Query(..., ge=1, description="Baseline version number."),
    to_version: int = Query(..., ge=1, description="Comparison version number."),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
):
    try:
        current_char = get_character_details(db, character_id)
        if not current_char:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Character with ID {character_id} not found.",
            )
        if from_version == to_version:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="from_version and to_version must be different.",
            )

        history_entries = db.get_character_version_history(character_id, limit=500)
        from_entry = next(
            (
                entry
                for entry in history_entries
                if int(entry.get("version") or -1) == from_version
            ),
            None,
        )
        to_entry = next(
            (
                entry
                for entry in history_entries
                if int(entry.get("version") or -1) == to_version
            ),
            None,
        )

        if not from_entry:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Version {from_version} not found for character {character_id}.",
            )
        if not to_entry:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Version {to_version} not found for character {character_id}.",
            )

        from_payload = from_entry.get("payload")
        to_payload = to_entry.get("payload")
        if not isinstance(from_payload, dict) or not isinstance(to_payload, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="One or both version payloads are missing or malformed.",
            )

        changed_fields = _build_character_version_diff_fields(from_payload, to_payload)
        return CharacterVersionDiffResponse(
            character_id=character_id,
            from_entry=_build_character_version_entry(from_entry),
            to_entry=_build_character_version_entry(to_entry),
            changed_fields=changed_fields,
            changed_count=len(changed_fields),
        )
    except CharactersRAGDBError as e:
        logger.error(
            f"DB error getting character version diff for {character_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except HTTPException:
        raise
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(
            f"Unexpected error getting character version diff for {character_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.") from e


@router.get(
    "/{character_id:int}/versions",
    response_model=CharacterVersionListResponse,
    summary="Get character version history",
    tags=["characters"],
)
async def get_character_versions_endpoint(
    character_id: int = FastAPIPath(..., description="ID of the character.", gt=0),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of versions to return."),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
):
    try:
        current_char = get_character_details(db, character_id)
        if not current_char:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Character with ID {character_id} not found.",
            )

        raw_entries = db.get_character_version_history(character_id, limit=limit)
        items = [_build_character_version_entry(entry) for entry in raw_entries]
        return CharacterVersionListResponse(items=items, total=len(items))
    except CharactersRAGDBError as e:
        logger.error(f"DB error getting character versions for {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except HTTPException:
        raise
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error getting character versions for {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.") from e


@router.post(
    "/{character_id:int}/revert",
    response_model=CharacterResponse,
    summary="Revert character to a previous version snapshot",
    tags=["characters"],
)
async def revert_character_to_version_endpoint(
    revert_request: CharacterRevertRequest,
    character_id: int = FastAPIPath(..., description="ID of the character to revert.", gt=0),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
):
    try:
        current_char = get_character_details(db, character_id)
        if not current_char:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Character with ID {character_id} not found.",
            )

        target_version = int(revert_request.target_version)
        history_entries = db.get_character_version_history(character_id, limit=500)
        target_entry = next(
            (
                entry
                for entry in history_entries
                if int(entry.get("version") or -1) == target_version
            ),
            None,
        )
        if not target_entry:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Version {target_version} not found for character {character_id}.",
            )

        snapshot_payload = target_entry.get("payload")
        if not isinstance(snapshot_payload, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Target version payload is missing or malformed.",
            )

        revert_payload = _build_character_revert_payload(snapshot_payload, current_char)
        expected_version = int(current_char.get("version") or 0)
        if expected_version <= 0:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Current character version is invalid. Refresh and try again.",
            )

        success = update_existing_character_details(
            db,
            character_id,
            revert_payload,
            expected_version,
        )
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to revert character (unexpected boolean failure).",
            )

        updated_char = get_character_details(db, character_id)
        if not updated_char:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Character reverted but could not be retrieved.",
            )
        return _convert_db_char_to_response_model(updated_char)
    except InputError as e:
        logger.warning(f"Validation error reverting character {character_id}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except ConflictError as e:
        logger.warning(f"Version conflict reverting character {character_id}: {e}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.error(f"DB error reverting character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except HTTPException:
        raise
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error reverting character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.") from e


@router.put("/{character_id:int}", response_model=CharacterResponse, summary="Update character", tags=["characters"])
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
        raise HTTPException(status_code=status_code, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.error(f"DB error updating character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except HTTPException:
        raise
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error updating character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.") from e


@router.delete("/{character_id:int}", response_model=DeletionResponse, summary="Delete character", tags=["characters"])
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
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.error(f"DB error deleting character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except HTTPException:
        raise
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error deleting character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.") from e


@router.post("/{character_id:int}/restore", response_model=CharacterResponse, summary="Restore deleted character", tags=["characters"])
async def restore_character_endpoint(
        character_id: int = FastAPIPath(..., description="ID of the character to restore.", gt=0),
        expected_version: int = Query(...,
                                      description="Expected current version of the character for optimistic locking."),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """
    Restore a soft-deleted character.

    This endpoint undoes a soft delete, making the character visible and usable again.
    The expected_version must match the current version of the soft-deleted character.
    Restore eligibility is enforced by `CHARACTERS_RESTORE_RETENTION_DAYS` (default: 30).
    """
    try:
        retention_days = _get_characters_restore_retention_days()
        success = restore_character_from_db(
            db,
            character_id,
            expected_version,
            retention_days=retention_days,
        )

        if not success:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Failed to restore character (unexpected boolean failure).")

        # Retrieve the restored character to return full details
        restored_char = get_character_details(db, character_id)
        if not restored_char:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Character restored but could not be retrieved.")

        logger.info(f"Character '{restored_char.get('name', 'Unknown')}' (ID: {character_id}) restored successfully")

        return _convert_db_char_to_response_model(restored_char)

    except RestoreWindowExpiredError as e:
        logger.info(
            "Restore denied for character {} with reason=restore_window_expired "
            "(deleted_at={}, restore_expires_at={}, retention_days={}).",
            character_id,
            e.deleted_at_iso,
            e.restore_expires_at_iso,
            e.retention_days,
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Restore window expired for character ID {character_id}. "
                f"This character was deleted at {e.deleted_at_iso} and could only be restored until "
                f"{e.restore_expires_at_iso} UTC."
            ),
        ) from e
    except ConflictError as e:
        logger.warning(f"Conflict error restoring character {character_id}: {e}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.error(f"DB error restoring character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except HTTPException:
        raise
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error restoring character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.") from e


@router.get("/search/", response_model=list[CharacterResponse], summary="Search characters", tags=["characters"])
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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error searching characters for '{query}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.") from e

# --- World Book Endpoints ---

@router.post("/world-books", response_model=WorldBookResponse, status_code=status.HTTP_201_CREATED,
             summary="Create a world book", tags=["World Books"])
async def create_world_book(
        world_book: WorldBookCreate,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Create a new world book for the user."""
    try:
        service = WorldBookService(db)
        world_book_id = service.create_world_book(
            name=world_book.name,
            description=world_book.description,
            scan_depth=world_book.scan_depth,
            token_budget=world_book.token_budget,
            recursive_scanning=world_book.recursive_scanning,
            enabled=world_book.enabled
        )

        created_book = service.get_world_book(world_book_id)
        if not created_book:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve world book after creation"
            )

        # Add entry count
        entries = service.get_entries(world_book_id, enabled_only=False)
        created_book['entry_count'] = len(entries)

        return WorldBookResponse(**created_book)

    except InputError as e:
        logger.warning(f"Input error creating world book: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except ConflictError as e:
        logger.warning(f"Conflict creating world book: {e}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.error(f"DB error creating world book: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error creating world book: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred") from e



@router.get("/world-books/{world_book_id}", response_model=WorldBookWithEntries,
            summary="Get world book with entries", tags=["World Books"])
async def get_world_book(
        world_book_id: int = FastAPIPath(..., description="World book ID", gt=0),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Get a world book with all its entries."""
    try:
        service = WorldBookService(db)
        book = service.get_world_book(world_book_id)

        if not book:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"World book with ID {world_book_id} not found"
            )

        # Get entries
        entries = service.get_entries(world_book_id, enabled_only=False)
        book['entry_count'] = len(entries)

        # Convert entries to response models
        entry_responses = []
        for entry in entries:
            if hasattr(entry, "to_api_dict"):
                entry_dict = entry.to_api_dict()
            elif hasattr(entry, "_d"):
                entry_dict = dict(entry._d)
            elif hasattr(entry, "to_dict"):
                entry_dict = entry.to_dict()
            else:
                entry_dict = dict(entry)
            entry_dict['created_at'] = entry_dict.get('created_at') or book.get('created_at')
            entry_dict['last_modified'] = entry_dict.get('last_modified') or book.get('last_modified')
            entry_responses.append(WorldBookEntryResponse(**entry_dict))

        return WorldBookWithEntries(
            **book,
            entries=entry_responses
        )

    except HTTPException:
        raise
    except CharactersRAGDBError as e:
        logger.error(f"DB error getting world book {world_book_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error getting world book {world_book_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred") from e


@router.put("/world-books/{world_book_id}", response_model=WorldBookResponse,
            summary="Update world book", tags=["World Books"])
async def update_world_book(
        world_book_id: int,
        update_data: WorldBookUpdate,
        expected_version: Optional[int] = Query(
            None,
            description="Expected current version of the world book for optimistic locking."
        ),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Update a world book."""
    try:
        service = WorldBookService(db)

        # Check if exists
        existing = service.get_world_book(world_book_id)
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"World book with ID {world_book_id} not found"
            )
        if expected_version is not None and existing.get("version") != expected_version:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Version mismatch. Expected {expected_version}, "
                    f"found {existing.get('version')}. Please refresh and try again."
                )
            )

        # Update
        success = service.update_world_book(
            world_book_id=world_book_id,
            name=update_data.name,
            description=update_data.description,
            scan_depth=update_data.scan_depth,
            token_budget=update_data.token_budget,
            recursive_scanning=update_data.recursive_scanning,
            enabled=update_data.enabled,
            expected_version=expected_version
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update world book"
            )

        # Get updated book
        updated_book = service.get_world_book(world_book_id)
        entries = service.get_entries(world_book_id, enabled_only=False)
        updated_book['entry_count'] = len(entries)

        return WorldBookResponse(**updated_book)

    except HTTPException:
        raise
    except ConflictError as e:
        logger.warning(f"Conflict updating world book {world_book_id}: {e}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.error(f"DB error updating world book {world_book_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error updating world book {world_book_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred") from e


@router.delete("/world-books/{world_book_id}", response_model=WorldBookDeletionResponse,
               summary="Delete world book", tags=["World Books"])
async def delete_world_book(
        world_book_id: int = FastAPIPath(..., description="World book ID", gt=0),
        hard_delete: bool = Query(False, description="Permanently delete (default is soft delete)"),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Delete a world book."""
    try:
        service = WorldBookService(db)

        # Check if exists and get name
        book = service.get_world_book(world_book_id)
        if not book:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"World book with ID {world_book_id} not found"
            )

        book_name = book['name']
        success = service.delete_world_book(world_book_id, hard_delete=hard_delete)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete world book"
            )

        delete_type = "permanently deleted" if hard_delete else "soft-deleted"
        return WorldBookDeletionResponse(
            message=f"World book '{book_name}' (ID: {world_book_id}) {delete_type}",
            world_book_id=world_book_id,
        )

    except HTTPException:
        raise
    except CharactersRAGDBError as e:
        logger.error(f"DB error deleting world book {world_book_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error deleting world book {world_book_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred") from e


# --- World Book Entry Endpoints ---

def _merge_entry_appendable_metadata(
        metadata: Optional[dict[str, Any]],
        appendable: Optional[bool],
) -> Optional[dict[str, Any]]:
    merged = dict(metadata or {})
    if appendable is not None:
        merged["appendable"] = bool(appendable)
    return merged


def _normalize_entry_group(group: Optional[str]) -> Optional[str]:
    if group is None:
        return None
    normalized = str(group).strip()
    return normalized or None


def _merge_entry_group_metadata(
        metadata: Optional[dict[str, Any]],
        group: Optional[str],
) -> Optional[dict[str, Any]]:
    merged = dict(metadata or {})
    normalized_group = _normalize_entry_group(group)
    if group is None:
        return merged or None
    if normalized_group is None:
        merged.pop("group", None)
    else:
        merged["group"] = normalized_group
    return merged or None


def _merge_entry_recursive_metadata(
        metadata: Optional[dict[str, Any]],
        recursive_scanning: Optional[bool],
) -> Optional[dict[str, Any]]:
    merged = dict(metadata or {})
    if recursive_scanning is None:
        return merged or None
    merged["recursive_scanning"] = bool(recursive_scanning)
    return merged or None

@router.post("/world-books/{world_book_id}/entries", response_model=WorldBookEntryResponse,
             status_code=status.HTTP_201_CREATED, summary="Add entry to world book", tags=["World Books"])
async def add_world_book_entry(
        world_book_id: int,
        entry: WorldBookEntryCreate,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Add an entry to a world book."""
    try:
        service = WorldBookService(db)

        # Check if world book exists
        book = service.get_world_book(world_book_id)
        if not book:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"World book with ID {world_book_id} not found"
            )

        entry_metadata = _merge_entry_group_metadata(
            _merge_entry_appendable_metadata(entry.metadata, entry.appendable),
            entry.group,
        )
        entry_metadata = _merge_entry_recursive_metadata(
            entry_metadata,
            entry.recursive_scanning,
        )

        entry_id = service.add_entry(
            world_book_id=world_book_id,
            keywords=entry.keywords,
            content=entry.content,
            priority=entry.priority,
            enabled=entry.enabled,
            case_sensitive=entry.case_sensitive,
            regex_match=entry.regex_match,
            whole_word_match=entry.whole_word_match,
            metadata=entry_metadata
        )

        # Get the created entry by id (avoid scanning all book entries).
        created_entry = service.get_entry(entry_id)

        if not created_entry:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve entry after creation"
            )

        if hasattr(created_entry, 'to_api_dict'):
            entry_dict = created_entry.to_api_dict()
        elif hasattr(created_entry, '_d'):
            entry_dict = dict(created_entry._d)
        else:
            entry_dict = dict(created_entry)
        entry_dict['created_at'] = entry_dict.get('created_at') or book.get('created_at')
        entry_dict['last_modified'] = entry_dict.get('last_modified') or book.get('last_modified')

        return WorldBookEntryResponse(**entry_dict)

    except HTTPException:
        raise
    except InputError as e:
        logger.warning(f"Input error adding entry to world book {world_book_id}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.error(f"DB error adding entry to world book {world_book_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error adding entry to world book {world_book_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred") from e


@router.get("/world-books/{world_book_id}/entries", response_model=EntryListResponse,
            summary="List world book entries", tags=["World Books"])
async def list_world_book_entries(
        world_book_id: int = FastAPIPath(..., description="World book ID", gt=0),
        enabled_only: bool = Query(False, description="Only show enabled entries"),
        group: Optional[str] = Query(None, description="Optional entry group/category filter"),
        appendable: Optional[bool] = Query(None, description="Optional appendable flag filter"),
        recursive_scanning: Optional[bool] = Query(
            None,
            description="Optional recursive-source flag filter",
        ),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """List all entries in a world book."""
    try:
        service = WorldBookService(db)

        # Check if world book exists
        book = service.get_world_book(world_book_id)
        if not book:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"World book with ID {world_book_id} not found"
            )

        entries = service.get_entries(
            world_book_id,
            enabled_only=enabled_only,
            group=group,
            appendable=appendable,
            recursive_scanning=recursive_scanning,
        )

        # Convert to response models
        entry_responses = []
        for entry in entries:
            if hasattr(entry, 'to_api_dict'):
                entry_dict = entry.to_api_dict()
            elif hasattr(entry, '_d'):
                entry_dict = dict(entry._d)
            else:
                entry_dict = dict(entry)
            entry_dict['created_at'] = entry_dict.get('created_at') or book.get('created_at')
            entry_dict['last_modified'] = entry_dict.get('last_modified') or book.get('last_modified')
            entry_responses.append(WorldBookEntryResponse(**entry_dict))

        return EntryListResponse(
            entries=entry_responses,
            total=len(entry_responses),
            world_book_id=world_book_id
        )

    except HTTPException:
        raise
    except CharactersRAGDBError as e:
        logger.error(f"DB error listing entries for world book {world_book_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error listing entries for world book {world_book_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred") from e


@router.put("/world-books/entries/{entry_id}", response_model=WorldBookEntryResponse,
            summary="Update world book entry", tags=["World Books"])
async def update_world_book_entry(
        entry_id: int,
        update_data: WorldBookEntryUpdate,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Update a world book entry."""
    try:
        service = WorldBookService(db)

        entry_metadata = update_data.metadata
        if (
            update_data.appendable is not None
            or update_data.group is not None
            or update_data.recursive_scanning is not None
        ):
            if entry_metadata is None:
                existing_entry = service.get_entry(entry_id)
                if existing_entry is not None:
                    if hasattr(existing_entry, 'get'):
                        entry_metadata = existing_entry.get('metadata') or {}
                    elif hasattr(existing_entry, '_d'):
                        entry_metadata = getattr(existing_entry, '_d', {}).get('metadata') or {}
            entry_metadata = _merge_entry_appendable_metadata(entry_metadata, update_data.appendable)
            entry_metadata = _merge_entry_group_metadata(entry_metadata, update_data.group)
            entry_metadata = _merge_entry_recursive_metadata(
                entry_metadata,
                update_data.recursive_scanning,
            )

        success = service.update_entry(
            entry_id=entry_id,
            keywords=update_data.keywords,
            content=update_data.content,
            priority=update_data.priority,
            enabled=update_data.enabled,
            case_sensitive=update_data.case_sensitive,
            regex_match=update_data.regex_match,
            whole_word_match=update_data.whole_word_match,
            metadata=entry_metadata
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Entry with ID {entry_id} not found"
            )

        # Get updated entry by id (avoid scanning all entries).
        updated_entry = service.get_entry(entry_id)

        if not updated_entry:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve entry after update"
            )

        # Get the world book for timestamps
        # Resolve world book id and entry dict from hybrid entry
        wb_id_resolved = getattr(updated_entry, 'world_book_id', None)
        if not wb_id_resolved and hasattr(updated_entry, 'get'):
            wb_id_resolved = updated_entry.get('world_book_id')
        book = service.get_world_book(wb_id_resolved)
        if hasattr(updated_entry, 'to_api_dict'):
            entry_dict = updated_entry.to_api_dict()
        elif hasattr(updated_entry, '_d'):
            entry_dict = dict(updated_entry._d)
        else:
            entry_dict = dict(updated_entry)
        entry_dict['created_at'] = entry_dict.get('created_at') or (book['created_at'] if book else datetime.utcnow())
        entry_dict['last_modified'] = entry_dict.get('last_modified') or (book['last_modified'] if book else datetime.utcnow())

        return WorldBookEntryResponse(**entry_dict)

    except HTTPException:
        raise
    except InputError as e:
        logger.warning(f"Input error updating entry {entry_id}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.error(f"DB error updating entry {entry_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error updating entry {entry_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred") from e


@router.delete("/world-books/entries/{entry_id}", response_model=WorldBookEntryDeletionResponse,
               summary="Delete world book entry", tags=["World Books"])
async def delete_world_book_entry(
        entry_id: int = FastAPIPath(..., description="Entry ID", gt=0),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Delete a world book entry."""
    try:
        service = WorldBookService(db)

        success = service.delete_entry(entry_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Entry with ID {entry_id} not found"
            )

        return WorldBookEntryDeletionResponse(
            message=f"World book entry (ID: {entry_id}) deleted",
            entry_id=entry_id,
        )

    except HTTPException:
        raise
    except CharactersRAGDBError as e:
        logger.error(f"DB error deleting entry {entry_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error deleting entry {entry_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred") from e


# --- Character-World Book Association Endpoints ---

@router.post("/{character_id:int}/world-books", response_model=CharacterWorldBookResponse,
             status_code=status.HTTP_200_OK, summary="Attach world book to character", tags=["World Books"])
async def attach_world_book_to_character(
        character_id: int,
        attachment: CharacterWorldBookAttachment,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Attach a world book to a character."""
    try:
        # Check if character exists
        char = get_character_details(db, character_id)
        if not char:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Character with ID {character_id} not found"
            )

        service = WorldBookService(db)

        # Check if world book exists
        book = service.get_world_book(attachment.world_book_id)
        if not book:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"World book with ID {attachment.world_book_id} not found"
            )

        success = service.attach_to_character(
            character_id=character_id,
            world_book_id=attachment.world_book_id,
            enabled=attachment.enabled,
            priority=attachment.priority
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to attach world book to character"
            )

        # Get entry count
        entries = service.get_entries(attachment.world_book_id, enabled_only=False)
        book['entry_count'] = len(entries)

        return CharacterWorldBookResponse(
            **book,
            world_book_id=book.get('id', attachment.world_book_id),
            attachment_enabled=attachment.enabled,
            attachment_priority=attachment.priority
        )

    except HTTPException:
        raise
    except CharactersRAGDBError as e:
        if _is_permission_denied_db_error(e):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to modify character world-book attachments",
            ) from e
        logger.error(f"DB error attaching world book {attachment.world_book_id} to character {character_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error attaching world book to character: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred") from e


@router.delete("/{character_id:int}/world-books/{world_book_id:int}", response_model=CharacterWorldBookDetachDeletionResponse,
               summary="Detach world book from character", tags=["World Books"])
async def detach_world_book_from_character(
        character_id: int = FastAPIPath(..., description="Character ID", gt=0),
        world_book_id: int = FastAPIPath(..., description="World book ID", gt=0),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Detach a world book from a character."""
    try:
        service = WorldBookService(db)

        success = service.detach_from_character(world_book_id, character_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Attachment between character {character_id} and world book {world_book_id} not found"
            )

        return CharacterWorldBookDetachDeletionResponse(
            message=f"World book {world_book_id} detached from character {character_id}",
            world_book_id=world_book_id,
            detached_from_character_id=character_id,
        )

    except HTTPException:
        raise
    except CharactersRAGDBError as e:
        if _is_permission_denied_db_error(e):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to modify character world-book attachments",
            ) from e
        logger.error(f"DB error detaching world book {world_book_id} from character {character_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error detaching world book from character: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred") from e


@router.get("/{character_id:int}/world-books", response_model=list[CharacterWorldBookResponse],
            summary="List character's world books", tags=["World Books"])
async def get_character_world_books(
        character_id: int = FastAPIPath(..., description="Character ID", gt=0),
        enabled_only: bool = Query(True, description="Only show enabled attachments"),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Get all world books attached to a character."""
    try:
        # Check if character exists
        char = get_character_details(db, character_id)
        if not char:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Character with ID {character_id} not found"
            )

        service = WorldBookService(db)
        books = service.get_character_world_books(character_id, enabled_only=enabled_only)
        book_ids = [
            int(book["id"])
            for book in books
            if isinstance(book.get("id"), int)
        ]
        entry_counts = service.get_entry_counts_for_world_books(book_ids)

        # Add entry counts and convert to response models
        response_books = []
        for book in books:
            # Build response dict with explicit world_book_id alias
            book_dict = dict(book)
            try:
                world_book_id = int(book_dict.get("id"))
            except (TypeError, ValueError):
                world_book_id = -1
            book_dict['entry_count'] = int(entry_counts.get(world_book_id, 0))
            book_dict['world_book_id'] = book_dict.get('id')
            response_books.append(CharacterWorldBookResponse(**book_dict))

        return response_books

    except HTTPException:
        raise
    except CharactersRAGDBError as e:
        if _is_permission_denied_db_error(e):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to access character world-book attachments",
            ) from e
        logger.error(f"DB error getting world books for character {character_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error getting character's world books: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred") from e


# --- Processing Endpoints ---

@router.post("/world-books/process", response_model=ProcessContextResponse,
             summary="Process text with world info", tags=["World Books"])
async def process_context_with_world_info(
        request: ProcessContextRequest,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Process text to find and inject relevant world info."""
    try:
        service = WorldBookService(db)

        result = service.process_context(
            text=request.text,
            world_book_ids=request.world_book_ids,
            character_id=request.character_id,
            scan_depth=request.scan_depth,
            token_budget=request.token_budget,
            recursive_scanning=request.recursive_scanning,
            include_diagnostics=True,
        )

        if isinstance(result, dict):
            return ProcessContextResponse(
                injected_content=result.get("processed_context", ""),
                entries_matched=int(result.get("entries_matched", 0)),
                tokens_used=int(result.get("tokens_used", 0)),
                books_used=int(result.get("books_used", 0)),
                entry_ids=[int(e) for e in result.get("entry_ids", [])],
                token_budget=int(result.get("token_budget", request.token_budget)),
                budget_exhausted=bool(result.get("budget_exhausted", False)),
                skipped_entries_due_to_budget=int(
                    result.get("skipped_entries_due_to_budget", 0)
                ),
                diagnostics=list(result.get("diagnostics") or []),
            )

        if isinstance(result, list):
            injected_content = "\n\n".join(
                entry.get("content", "") for entry in result if isinstance(entry, dict)
            )
            entry_ids = [
                int(entry.get("id"))
                for entry in result
                if isinstance(entry, dict) and entry.get("id") is not None
            ]
            books_used = {
                entry.get("world_book_id")
                for entry in result
                if isinstance(entry, dict) and entry.get("world_book_id") is not None
            }
            # Best-effort token estimate mirrors service-level counting for consistency.
            tokens_used = sum(
                service.count_tokens(entry.get("content", ""))  # type: ignore[arg-type]
                for entry in result
                if isinstance(entry, dict)
            )
            return ProcessContextResponse(
                injected_content=injected_content,
                entries_matched=len(result),
                tokens_used=tokens_used,
                books_used=len(books_used),
                entry_ids=entry_ids,
                token_budget=request.token_budget,
                budget_exhausted=tokens_used >= request.token_budget
                if request.token_budget
                else False,
                skipped_entries_due_to_budget=0,
                diagnostics=[
                    {
                        "entry_id": entry.get("id"),
                        "world_book_id": entry.get("world_book_id"),
                        "activation_reason": "regex_match"
                        if bool(entry.get("regex_match"))
                        else "keyword_match",
                        "keyword": (entry.get("keywords") or [None])[0],
                        "token_cost": int(service.count_tokens(entry.get("content", ""))),
                        "priority": int(entry.get("priority") or 0),
                        "regex_match": bool(entry.get("regex_match")),
                        "content_preview": str(entry.get("content", ""))[:240],
                        "depth_level": None,
                    }
                    for entry in result
                    if isinstance(entry, dict)
                ],
            )

        logger.error(
            'WorldBookService.process_context returned unexpected type {}',
            type(result).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected world book processing result.",
        )

    except CharactersRAGDBError as e:
        logger.error(f"DB error processing context: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error processing context: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred") from e


# --- Import/Export Endpoints ---

@router.post("/world-books/import", response_model=WorldBookImportResponse,
             status_code=status.HTTP_201_CREATED, summary="Import world book", tags=["World Books"])
async def import_world_book(
        import_data: WorldBookImportRequest,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Import a world book from external format."""
    try:
        service = WorldBookService(db)

        world_book_id = service.import_world_book(
            data={
                "world_book": import_data.world_book,
                "entries": import_data.entries
            },
            merge_on_conflict=import_data.merge_on_conflict
        )

        # Get the imported/merged book
        book = service.get_world_book(world_book_id)
        if not book:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve world book after import"
            )

        return WorldBookImportResponse(
            world_book_id=world_book_id,
            name=book['name'],
            entries_imported=len(import_data.entries),
            merged=import_data.merge_on_conflict and book['version'] > 1
        )

    except InputError as e:
        logger.warning(f"Input error importing world book: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except ConflictError as e:
        logger.warning(f"Conflict importing world book: {e}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.error(f"DB error importing world book: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error importing world book: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred") from e


@router.get("/world-books/{world_book_id}/export", response_model=WorldBookExport,
            summary="Export world book", tags=["World Books"])
async def export_world_book(
        world_book_id: int = FastAPIPath(..., description="World book ID", gt=0),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Export a world book to external format."""
    try:
        service = WorldBookService(db)

        export_data = service.export_world_book(world_book_id)

        return WorldBookExport(
            world_book=export_data['world_book'],
            entries=export_data['entries'],
            export_date=datetime.utcnow(),
            format_version="1.0"
        )

    except InputError as e:
        logger.warning(f"World book {world_book_id} not found for export: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.error(f"DB error exporting world book {world_book_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error exporting world book {world_book_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred") from e


# --- Statistics Endpoint ---

@router.get("/world-books/{world_book_id}/statistics", response_model=WorldBookStatistics,
            summary="Get world book statistics", tags=["World Books"])
async def get_world_book_statistics(
        world_book_id: int = FastAPIPath(..., description="World book ID", gt=0),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Get statistics for a world book."""
    try:
        service = WorldBookService(db)

        book = service.get_world_book(world_book_id)
        if not book:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"World book with ID {world_book_id} not found"
            )

        entries = service.get_entries(world_book_id, enabled_only=False)

        # Calculate statistics
        enabled_entries = sum(1 for e in entries if e.enabled)
        disabled_entries = len(entries) - enabled_entries
        total_keywords = sum(len(e.keywords) for e in entries)
        regex_entries = sum(1 for e in entries if e.regex_match)
        case_sensitive_entries = sum(1 for e in entries if e.case_sensitive)

        priorities = [e.priority for e in entries]
        average_priority = sum(priorities) / len(priorities) if priorities else 0.0

        total_content_length = sum(len(e.content) for e in entries)
        estimated_tokens = service.count_tokens(" ".join(e.content for e in entries))

        return WorldBookStatistics(
            world_book_id=world_book_id,
            name=book['name'],
            total_entries=len(entries),
            enabled_entries=enabled_entries,
            disabled_entries=disabled_entries,
            total_keywords=total_keywords,
            regex_entries=regex_entries,
            case_sensitive_entries=case_sensitive_entries,
            average_priority=average_priority,
            total_content_length=total_content_length,
            estimated_tokens=estimated_tokens
        )

    except HTTPException:
        raise
    except CharactersRAGDBError as e:
        logger.error(f"DB error getting statistics for world book {world_book_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error getting world book statistics: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred") from e


# --- Bulk Operations Endpoint ---

@router.post("/world-books/entries/bulk", response_model=BulkOperationResponse,
             summary="Bulk operations on entries", tags=["World Books"])
async def bulk_entry_operations(
        operation: BulkEntryOperation,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """Perform bulk operations on world book entries."""
    try:
        service = WorldBookService(db)

        affected_count = 0
        failed_ids = []

        for entry_id in operation.entry_ids:
            try:
                if operation.operation == "delete":
                    success = service.delete_entry(entry_id)
                elif operation.operation == "enable":
                    success = service.update_entry(entry_id, enabled=True)
                elif operation.operation == "disable":
                    success = service.update_entry(entry_id, enabled=False)
                elif operation.operation == "set_priority" and operation.priority is not None:
                    success = service.update_entry(entry_id, priority=operation.priority)
                else:
                    success = False

                if success:
                    affected_count += 1
                else:
                    failed_ids.append(entry_id)

            except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
                logger.warning(f"Failed to perform {operation.operation} on entry {entry_id}: {e}")
                failed_ids.append(entry_id)

        message = f"Operation '{operation.operation}' completed: {affected_count} entries affected"
        if failed_ids:
            message += f", {len(failed_ids)} failed"

        return BulkOperationResponse(
            success=len(failed_ids) == 0,
            affected_count=affected_count,
            failed_ids=failed_ids,
            message=message
        )

    except CharactersRAGDBError as e:
        logger.error(f"DB error performing bulk operation: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error performing bulk operation: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred") from e


# ========================================================================
# Additional Endpoints: Tag Filtering and Export
# ========================================================================


def _encode_png_with_chara_metadata(
    image_data: Optional[bytes],
    card_json: str,
) -> bytes:
    """Create a PNG file with character card JSON in a tEXt chunk.

    If *image_data* is valid PNG bytes the character data is injected into it;
    otherwise a minimal 1x1 transparent PNG is generated as the carrier.
    """
    chara_b64 = base64.b64encode(card_json.encode("utf-8")).decode("ascii")

    # Build tEXt chunk: keyword NUL text
    keyword = b"chara"
    text_data = keyword + b"\x00" + chara_b64.encode("ascii")
    chunk_type = b"tEXt"
    chunk_length = struct.pack(">I", len(text_data))
    chunk_crc = struct.pack(">I", zlib.crc32(chunk_type + text_data) & 0xFFFFFFFF)
    text_chunk = chunk_length + chunk_type + text_data + chunk_crc

    if image_data and image_data[:8] == b"\x89PNG\r\n\x1a\n":
        # Insert tEXt chunk before the IEND chunk (last 12 bytes of a valid PNG).
        iend_pos = image_data.rfind(b"IEND")
        if iend_pos >= 4:
            iend_start = iend_pos - 4  # length field is 4 bytes before type
            return image_data[:iend_start] + text_chunk + image_data[iend_start:]

    # Fallback: generate a minimal 1x1 transparent PNG.
    png_header = b"\x89PNG\r\n\x1a\n"

    # IHDR chunk
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)  # 1x1 RGBA
    ihdr_type = b"IHDR"
    ihdr = (
        struct.pack(">I", len(ihdr_data))
        + ihdr_type
        + ihdr_data
        + struct.pack(">I", zlib.crc32(ihdr_type + ihdr_data) & 0xFFFFFFFF)
    )

    # IDAT chunk (1x1 transparent pixel: filter byte + 4 zero bytes)
    raw_data = zlib.compress(b"\x00\x00\x00\x00\x00")
    idat_type = b"IDAT"
    idat = (
        struct.pack(">I", len(raw_data))
        + idat_type
        + raw_data
        + struct.pack(">I", zlib.crc32(idat_type + raw_data) & 0xFFFFFFFF)
    )

    # IEND chunk
    iend_type = b"IEND"
    iend = struct.pack(">I", 0) + iend_type + struct.pack(">I", zlib.crc32(iend_type) & 0xFFFFFFFF)

    return png_header + ihdr + text_chunk + idat + iend


@router.get("/{character_id:int}/export", response_model=None,
            summary="Export character in various formats", tags=["characters"])
async def export_character(
    character_id: int = FastAPIPath(..., description="Character ID to export", gt=0),
    format: str = Query("v3", description="Export format (v3, v2, json, png)"),
    include_world_books: bool = Query(False, description="Include associated world books"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """
    Export a character in various formats.

    Args:
        character_id: ID of character to export
        format: Export format (v3 for Character Card V3, v2 for V2, json for raw)
        include_world_books: Whether to include world book data
        db: Database instance

    Returns:
        Character data in requested format

    Raises:
        HTTPException: 404 if character not found
    """
    try:
        def _as_export_str(value: Any, default: str = "") -> str:
            if value is None:
                return default
            if isinstance(value, str):
                return value
            return str(value)

        def _as_export_str_list(value: Any) -> list[str]:
            if not isinstance(value, list):
                return []
            return [str(item) for item in value if item is not None]

        # Get character
        character = get_character_details(db, character_id)
        if not character:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Character with ID {character_id} not found"
            )

        get_character_world_books_fn = getattr(db, "get_character_world_books", None)
        character_world_books: list[dict[str, Any]] = []
        if (format == "v2" or include_world_books) and callable(get_character_world_books_fn):
            result = get_character_world_books_fn(character_id)
            if isinstance(result, list):
                character_world_books = result

        # Build export data based on format
        if format == "v3":
            # Character Card V3 format
            export_data = {
                "spec": "chara_card_v3",
                "spec_version": "3.0",
                "data": {
                    "name": character.get('name'),
                    "description": character.get('description'),
                    "personality": character.get('personality'),
                    "scenario": character.get('scenario'),
                    "first_mes": character.get('first_message'),
                    "mes_example": character.get('message_example'),
                    "creator_notes": character.get('creator_notes'),
                    "system_prompt": character.get('system_prompt'),
                    "post_history_instructions": character.get('post_history_instructions'),
                    "alternate_greetings": character.get('alternate_greetings', []),
                    "tags": character.get('tags', []),
                    "creator": character.get('creator'),
                    "character_version": character.get('character_version', "1.0"),
                    "extensions": character.get('extensions', {})
                }
            }
        elif format == "v2":
            # Character Card V2 format
            extensions = character.get("extensions")
            if not isinstance(extensions, dict):
                extensions = {}
            if character_world_books:
                extensions = extensions.copy()
                extensions.setdefault(
                    "world_book_links",
                    [
                        wb["world_book_id"]
                        for wb in character_world_books
                        if isinstance(wb.get("world_book_id"), int)
                    ],
                )

            export_data = {
                "spec": "chara_card_v2",
                "spec_version": "2.0",
                "data": {
                    "name": _as_export_str(character.get("name")),
                    "description": _as_export_str(character.get("description")),
                    "personality": _as_export_str(character.get("personality")),
                    "scenario": _as_export_str(character.get("scenario")),
                    "first_mes": _as_export_str(character.get("first_message")),
                    "mes_example": _as_export_str(character.get("message_example")),
                    "creator_notes": _as_export_str(character.get("creator_notes")),
                    "system_prompt": _as_export_str(character.get("system_prompt")),
                    "post_history_instructions": _as_export_str(character.get("post_history_instructions")),
                    "alternate_greetings": _as_export_str_list(character.get("alternate_greetings")),
                    "tags": _as_export_str_list(character.get("tags")),
                    "creator": _as_export_str(character.get("creator")),
                    "character_version": _as_export_str(character.get("character_version"), default="1.0"),
                    "extensions": extensions,
                }
            }
        else:
            # Raw JSON format
            export_data = character

        # Add world books if requested
        if include_world_books:
            if character_world_books:
                world_books = character_world_books
            elif callable(get_character_world_books_fn):
                fetched = get_character_world_books_fn(character_id)
                world_books = fetched if isinstance(fetched, list) else []
            else:
                world_books = []
            if world_books:
                export_data["world_books"] = []
                for wb in world_books:
                    wb_data = db.get_world_book(wb['world_book_id'])
                    if wb_data:
                        entries = db.get_world_book_entries(wb['world_book_id'])
                        wb_data['entries'] = entries or []
                        export_data["world_books"].append(wb_data)

        # Add character image if present
        if character.get('image'):
            import base64
            encoded_image = base64.b64encode(character['image']).decode('utf-8')
            if format == "v2":
                export_data["data"]["char_image"] = encoded_image
            export_data["character_image"] = encoded_image

        logger.info(f"Exported character {character_id} in format {format}")

        # PNG export: embed V2 card JSON in PNG tEXt metadata chunk.
        if format == "png":
            # Build V2 card JSON for embedding (standard interchange format).
            v2_data: dict[str, Any] = {
                "spec": "chara_card_v2",
                "spec_version": "2.0",
                "data": {
                    "name": _as_export_str(character.get("name")),
                    "description": _as_export_str(character.get("description")),
                    "personality": _as_export_str(character.get("personality")),
                    "scenario": _as_export_str(character.get("scenario")),
                    "first_mes": _as_export_str(character.get("first_message")),
                    "mes_example": _as_export_str(character.get("message_example")),
                    "creator_notes": _as_export_str(character.get("creator_notes")),
                    "system_prompt": _as_export_str(character.get("system_prompt")),
                    "post_history_instructions": _as_export_str(character.get("post_history_instructions")),
                    "alternate_greetings": _as_export_str_list(character.get("alternate_greetings")),
                    "tags": _as_export_str_list(character.get("tags")),
                    "creator": _as_export_str(character.get("creator")),
                    "character_version": _as_export_str(character.get("character_version"), default="1.0"),
                    "extensions": character.get("extensions") if isinstance(character.get("extensions"), dict) else {},
                },
            }
            card_json_str = json.dumps(v2_data, ensure_ascii=False)

            image_bytes = character.get("image") if isinstance(character.get("image"), bytes) else None
            png_bytes = _encode_png_with_chara_metadata(image_bytes, card_json_str)

            safe_name = (character.get("name") or "character").replace(" ", "_")[:50]
            return Response(
                content=png_bytes,
                media_type="image/png",
                headers={
                    "Content-Disposition": f'attachment; filename="{safe_name}.png"',
                },
            )

        return export_data

    except HTTPException:
        raise
    except _CHARACTERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error exporting character {character_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while exporting character"
        ) from e


#
# End of characters.py
#######################################################################################################################
