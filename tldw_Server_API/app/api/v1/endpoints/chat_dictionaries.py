from __future__ import annotations

import datetime
import re
import time
import warnings
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.api.v1.schemas.chat_dictionary_schemas import (
    BulkEntryOperation,
    BulkOperationResponse,
    ChatDictionaryCreate,
    ChatDictionaryResponse,
    ChatDictionaryUpdate,
    ChatDictionaryWithEntries,
    DictionaryEntryCreate,
    DictionaryEntryReorderRequest,
    DictionaryEntryReorderResponse,
    DictionaryEntryResponse,
    DictionaryEntryUpdate,
    DictionaryActivityListResponse,
    DictionaryVersionDetailResponse,
    DictionaryVersionListResponse,
    DictionaryVersionRevertResponse,
    DictionaryListResponse,
    DictionaryStatistics,
    EntryListResponse,
    ExportDictionaryJSONResponse,
    ExportDictionaryResponse,
    ImportDictionaryJSONRequest,
    ImportDictionaryRequest,
    ImportDictionaryResponse,
    ProcessTextRequest,
    ProcessTextResponse,
    validate_regex_pattern_safety,
)
from tldw_Server_API.app.api.v1.utils.datetime_utils import coerce_datetime, parse_timed_effects
from tldw_Server_API.app.core.Character_Chat.chat_dictionary import (
    ChatDictionaryService,
    TokenBudgetExceededWarning,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)

router = APIRouter()


def _entry_dict_to_response(
    entry_data: dict[str, Any],
    fallback_dictionary_id: int | None = None,
) -> DictionaryEntryResponse:
    dictionary_id = entry_data.get("dictionary_id") or fallback_dictionary_id
    if dictionary_id is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Dictionary ID missing for entry.",
        )

    pattern = entry_data.get("pattern") or entry_data.get("key") or ""
    replacement = entry_data.get("replacement") or entry_data.get("content") or ""
    probability = entry_data.get("probability", 1.0)
    try:
        probability = float(probability)
    except (TypeError, ValueError):
        probability = 1.0

    max_replacements = entry_data.get("max_replacements", 0)
    try:
        max_replacements = int(max_replacements or 0)
    except (TypeError, ValueError):
        max_replacements = 0

    entry_type = entry_data.get("type")
    if not entry_type:
        entry_type = "regex" if bool(entry_data.get("is_regex")) else "literal"

    enabled = bool(entry_data.get("enabled", entry_data.get("is_enabled", 1)))
    case_sensitive = bool(entry_data.get("case_sensitive", entry_data.get("is_case_sensitive", 1)))

    return DictionaryEntryResponse(
        id=int(entry_data.get("id")),
        dictionary_id=int(dictionary_id),
        pattern=pattern,
        replacement=replacement,
        probability=probability,
        group=entry_data.get("group") or entry_data.get("group_name"),
        timed_effects=parse_timed_effects(entry_data.get("timed_effects")),
        max_replacements=max_replacements,
        type=entry_type,
        enabled=enabled,
        case_sensitive=case_sensitive,
        priority=(
            int(entry_data.get("sort_order"))
            if entry_data.get("sort_order") is not None
            else None
        ),
        usage_count=int(entry_data.get("usage_count", 0) or 0),
        last_used_at=_coerce_optional_datetime(entry_data.get("last_used_at")),
        created_at=coerce_datetime(entry_data.get("created_at")),
        updated_at=coerce_datetime(entry_data.get("updated_at")),
    )


def _coerce_optional_datetime(value: Any) -> datetime.datetime | None:
    """Best-effort conversion for optional datetime fields."""
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00").replace(" ", "T")
        try:
            return datetime.datetime.fromisoformat(normalized)
        except ValueError:
            for fmt in (
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
            ):
                try:
                    return datetime.datetime.strptime(normalized, fmt)
                except ValueError:
                    continue
            return None
    return None

def _entry_has_timed_effects(entry_data: dict[str, Any]) -> bool:
    timed_effects = parse_timed_effects(entry_data.get("timed_effects"))
    if not timed_effects:
        return False
    return any(
        int(getattr(timed_effects, key, 0) or 0) > 0
        for key in ("sticky", "cooldown", "delay")
    )


def _entry_pattern(entry_data: dict[str, Any]) -> str:
    return str(entry_data.get("pattern") or entry_data.get("key") or "")


def _entry_type(entry_data: dict[str, Any]) -> str:
    entry_type = str(entry_data.get("type") or "").strip().lower()
    if entry_type in {"literal", "regex"}:
        return entry_type
    pattern = _entry_pattern(entry_data)
    if pattern.startswith("/") and pattern.rfind("/") > 0:
        return "regex"
    return "literal"


def _parse_regex_pattern(raw_pattern: str) -> tuple[str, int]:
    """Parse /pattern/flags syntax into pattern body + Python flags."""
    if not raw_pattern:
        return "", 0
    pattern_body = raw_pattern
    flag_string = ""

    if raw_pattern.startswith("/") and raw_pattern.rfind("/") > 0:
        last_slash = raw_pattern.rfind("/")
        pattern_body = raw_pattern[1:last_slash]
        flag_string = raw_pattern[last_slash + 1 :]

    flags = 0
    if "i" in flag_string:
        flags |= re.IGNORECASE
    if "m" in flag_string:
        flags |= re.MULTILINE
    if "s" in flag_string:
        flags |= re.DOTALL
    if "x" in flag_string:
        flags |= re.VERBOSE

    return pattern_body, flags


def _compile_entry_regex(entry_data: dict[str, Any]) -> re.Pattern[str] | None:
    pattern_body, flags = _parse_regex_pattern(_entry_pattern(entry_data))
    if not pattern_body:
        return None
    try:
        return re.compile(pattern_body, flags)
    except re.error:
        return None


def _regex_literal_prefix(raw_pattern: str) -> str:
    """Extract a simple literal prefix from a regex body when possible."""
    pattern_body, _ = _parse_regex_pattern(raw_pattern)
    if not pattern_body:
        return ""

    prefix_chars: list[str] = []
    escaped = False
    for char in pattern_body:
        if escaped:
            prefix_chars.append(char)
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char.isalnum() or char in {" ", "-", "_"}:
            prefix_chars.append(char)
            continue
        break
    return "".join(prefix_chars).strip()


def _regex_seed_samples(raw_pattern: str) -> list[str]:
    """Generate lightweight sample strings for overlap heuristics."""
    pattern_body, _ = _parse_regex_pattern(raw_pattern)
    tokens = re.findall(r"[A-Za-z0-9]{2,}", pattern_body)
    seeds: list[str] = ["sample", "test", "kcl", "kc123", "doctor"]
    for token in tokens[:4]:
        seeds.extend([token, token.lower(), token.upper(), f"x{token}y"])
    # Preserve insertion order while deduplicating
    return list(dict.fromkeys(seeds))


def _build_conflict(
    entry_a: dict[str, Any],
    entry_b: dict[str, Any],
    *,
    conflict_type: str,
    severity: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "entry_id_a": int(entry_a.get("id")),
        "entry_id_b": int(entry_b.get("id")),
        "pattern_a": _entry_pattern(entry_a),
        "pattern_b": _entry_pattern(entry_b),
        "type_a": _entry_type(entry_a),
        "type_b": _entry_type(entry_b),
        "conflict_type": conflict_type,
        "severity": severity,
        "reason": reason,
    }


def _detect_pair_conflict(entry_a: dict[str, Any], entry_b: dict[str, Any]) -> dict[str, Any] | None:
    type_a = _entry_type(entry_a)
    type_b = _entry_type(entry_b)
    pattern_a = _entry_pattern(entry_a)
    pattern_b = _entry_pattern(entry_b)

    if not pattern_a or not pattern_b:
        return None

    # literal-literal overlap
    if type_a == "literal" and type_b == "literal":
        normalized_a = pattern_a.casefold()
        normalized_b = pattern_b.casefold()
        if normalized_a == normalized_b:
            return _build_conflict(
                entry_a,
                entry_b,
                conflict_type="literal-literal",
                severity="high",
                reason="Both literal entries match the same text and may shadow one another.",
            )
        if normalized_a in normalized_b or normalized_b in normalized_a:
            return _build_conflict(
                entry_a,
                entry_b,
                conflict_type="literal-literal",
                severity="medium",
                reason="One literal is contained in the other, so processing order can change output.",
            )
        return None

    # literal-regex overlap
    if {type_a, type_b} == {"literal", "regex"}:
        literal_entry = entry_a if type_a == "literal" else entry_b
        regex_entry = entry_a if type_a == "regex" else entry_b
        literal_pattern = _entry_pattern(literal_entry)
        compiled_regex = _compile_entry_regex(regex_entry)
        if not compiled_regex:
            return None
        try:
            if compiled_regex.search(literal_pattern):
                full_match = compiled_regex.fullmatch(literal_pattern) is not None
                return _build_conflict(
                    entry_a,
                    entry_b,
                    conflict_type="literal-regex",
                    severity="high" if full_match else "medium",
                    reason=(
                        "Regex pattern fully matches a literal pattern."
                        if full_match
                        else "Regex pattern overlaps with a literal pattern and may trigger on the same input."
                    ),
                )
        except re.error:
            return None
        return None

    # regex-regex overlap
    if type_a == "regex" and type_b == "regex":
        body_a, flags_a = _parse_regex_pattern(pattern_a)
        body_b, flags_b = _parse_regex_pattern(pattern_b)
        if body_a == body_b and flags_a == flags_b:
            return _build_conflict(
                entry_a,
                entry_b,
                conflict_type="regex-regex",
                severity="high",
                reason="Regex entries are identical and likely redundant.",
            )

        prefix_a = _regex_literal_prefix(pattern_a)
        prefix_b = _regex_literal_prefix(pattern_b)
        if prefix_a and prefix_b:
            normalized_prefix_a = prefix_a.casefold()
            normalized_prefix_b = prefix_b.casefold()
            if (
                normalized_prefix_a.startswith(normalized_prefix_b)
                or normalized_prefix_b.startswith(normalized_prefix_a)
            ):
                return _build_conflict(
                    entry_a,
                    entry_b,
                    conflict_type="regex-regex",
                    severity="low",
                    reason="Regex entries share a literal prefix and may overlap on similar text.",
                )

        regex_a = _compile_entry_regex(entry_a)
        regex_b = _compile_entry_regex(entry_b)
        if not regex_a or not regex_b:
            return None
        samples = _regex_seed_samples(pattern_a) + _regex_seed_samples(pattern_b)
        for sample in samples[:16]:
            try:
                if regex_a.search(sample) and regex_b.search(sample):
                    return _build_conflict(
                        entry_a,
                        entry_b,
                        conflict_type="regex-regex",
                        severity="low",
                        reason=f"Both regex entries match representative sample '{sample}'.",
                    )
            except re.error:
                return None
        return None

    return None


def _analyze_pattern_conflicts(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    severity_weight = {"high": 3, "medium": 2, "low": 1}

    indexed_entries = [entry for entry in entries if entry.get("id") is not None]
    max_pairs = 2400
    inspected_pairs = 0

    for idx, entry_a in enumerate(indexed_entries):
        for entry_b in indexed_entries[idx + 1 :]:
            inspected_pairs += 1
            if inspected_pairs > max_pairs:
                break
            conflict = _detect_pair_conflict(entry_a, entry_b)
            if conflict:
                conflicts.append(conflict)
        if inspected_pairs > max_pairs:
            break

    conflicts.sort(
        key=lambda item: (
            severity_weight.get(str(item.get("severity")), 0),
            int(item.get("entry_id_a", 0)),
            int(item.get("entry_id_b", 0)),
        ),
        reverse=True,
    )
    return conflicts[:50]


@router.post(
    "/dictionaries",
    response_model=ChatDictionaryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new chat dictionary",
    description="Create a dictionary used for pattern-based text replacements in chat messages.",
    tags=["chat-dictionaries"],
)
async def create_chat_dictionary(
    dictionary: ChatDictionaryCreate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> ChatDictionaryResponse:
    """
    Create a new chat dictionary for pattern-based text replacements.
    """
    service = ChatDictionaryService(db)
    try:
        dict_id = service.create_dictionary(
            name=dictionary.name,
            description=dictionary.description,
            default_token_budget=dictionary.default_token_budget,
            category=dictionary.category,
            tags=dictionary.tags,
            included_dictionary_ids=dictionary.included_dictionary_ids,
        )
        dict_data = service.get_dictionary(dict_id)
        entries = service.get_entries(dictionary_id=dict_id) if dict_data else []
    except (ConflictError, InputError, CharactersRAGDBError) as e:
        if isinstance(e, CharactersRAGDBError) and not isinstance(e, (ConflictError, InputError)):
            logger.error("Error creating dictionary")
        raise map_db_error_to_http(e, default_detail="Failed to create dictionary") from e
    except Exception as e:
        logger.error("Error creating dictionary")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create dictionary",
        ) from e

    if not dict_data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Dictionary created but could not be retrieved",
        )

    dict_data["entry_count"] = len(entries)
    return ChatDictionaryResponse(**dict_data)


@router.get(
    "/dictionaries",
    response_model=DictionaryListResponse,
    summary="List all chat dictionaries",
    description="List dictionaries for the current user. Use include_inactive to show inactive ones.",
    tags=["chat-dictionaries"],
)
async def list_chat_dictionaries(
    include_inactive: bool = Query(False, description="Include inactive dictionaries"),
    include_usage: bool = Query(False, description="Include chat usage summary per dictionary"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> DictionaryListResponse:
    """List all chat dictionaries for the current user."""
    try:
        service = ChatDictionaryService(db)
        dictionaries = service.list_dictionaries_with_entry_counts(include_inactive=include_inactive)
        if include_usage and dictionaries:
            usage_summary = service.get_dictionary_usage_summary(
                dictionary_ids=[int(item.get("id")) for item in dictionaries if item.get("id") is not None],
            )
            for item in dictionaries:
                dictionary_id_raw = item.get("id")
                if dictionary_id_raw is None:
                    continue
                dictionary_id = int(dictionary_id_raw)
                usage = usage_summary.get(dictionary_id) or {}
                item["used_by_chat_count"] = int(usage.get("used_by_chat_count", 0))
                item["used_by_active_chat_count"] = int(usage.get("used_by_active_chat_count", 0))
                item["used_by_chat_refs"] = usage.get("used_by_chat_refs", [])

        active_priority_rows = sorted(
            (
                row for row in dictionaries
                if bool(row.get("is_active", True))
            ),
            key=lambda row: (
                str(row.get("name") or "").strip().lower(),
                int(row.get("id") or 0),
            ),
        )
        priority_by_dictionary_id = {
            int(row.get("id")): index + 1
            for index, row in enumerate(active_priority_rows)
            if row.get("id") is not None
        }
        for item in dictionaries:
            dictionary_id = item.get("id")
            if dictionary_id is None:
                continue
            item["processing_priority"] = priority_by_dictionary_id.get(int(dictionary_id))

        active_count = sum(1 for d in dictionaries if d.get("is_active", True))
        inactive_count = len(dictionaries) - active_count

        dict_responses = [ChatDictionaryResponse(**d) for d in dictionaries]

        return DictionaryListResponse(
            dictionaries=dict_responses,
            total=len(dictionaries),
            active_count=active_count,
            inactive_count=inactive_count,
        )
    except CharactersRAGDBError as e:
        logger.error("Error listing dictionaries")
        raise map_db_error_to_http(e, default_detail="Failed to list dictionaries") from e
    except Exception as e:
        logger.error("Error listing dictionaries")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list dictionaries",
        ) from e


@router.get(
    "/dictionaries/{dictionary_id}",
    response_model=ChatDictionaryWithEntries,
    summary="Get dictionary with entries",
    description="Retrieve a dictionary and all its entries by ID.",
    tags=["chat-dictionaries"],
)
async def get_chat_dictionary(
    dictionary_id: int,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> ChatDictionaryWithEntries:
    """Get a specific dictionary with all its entries."""
    service = ChatDictionaryService(db)
    try:
        dict_data = service.get_dictionary(dictionary_id)
        entries = service.get_entries(dictionary_id=dictionary_id, active_only=False) if dict_data else []
    except HTTPException:
        raise
    except CharactersRAGDBError as e:
        logger.error("Error getting dictionary")
        raise map_db_error_to_http(e, default_detail="Failed to get dictionary") from e
    except Exception as e:
        logger.error("Error getting dictionary")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get dictionary",
        ) from e

    if not dict_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dictionary not found")

    dict_data["entry_count"] = len(entries)

    entry_responses = [
        _entry_dict_to_response(entry_dict, fallback_dictionary_id=dictionary_id) for entry_dict in entries
    ]

    return ChatDictionaryWithEntries(
        **dict_data,
        entries=entry_responses,
    )


@router.put(
    "/dictionaries/{dictionary_id}",
    response_model=ChatDictionaryResponse,
    summary="Update a dictionary",
    description="Update dictionary metadata such as name, description, and active status.",
    tags=["chat-dictionaries"],
)
async def update_chat_dictionary(
    dictionary_id: int,
    update: ChatDictionaryUpdate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> ChatDictionaryResponse:
    """Update a dictionary's metadata."""
    service = ChatDictionaryService(db)
    try:
        success = service.update_dictionary(
            dictionary_id,
            name=update.name,
            description=update.description,
            category=update.category,
            tags=update.tags,
            included_dictionary_ids=update.included_dictionary_ids,
            is_active=update.is_active,
            default_token_budget=update.default_token_budget,
            update_category=("category" in update.model_fields_set),
            update_tags=("tags" in update.model_fields_set),
            update_included_dictionary_ids=("included_dictionary_ids" in update.model_fields_set),
            update_default_token_budget=("default_token_budget" in update.model_fields_set),
            expected_version=update.version,
        )

        dict_data = service.get_dictionary(dictionary_id) if success else None
        entries = service.get_entries(dictionary_id=dictionary_id) if success else []
    except (ConflictError, InputError, CharactersRAGDBError) as e:
        if isinstance(e, CharactersRAGDBError) and not isinstance(e, (ConflictError, InputError)):
            logger.error("Error updating dictionary")
        raise map_db_error_to_http(e, default_detail="Failed to update dictionary") from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error updating dictionary")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update dictionary",
        ) from e

    if not success or not dict_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dictionary not found")

    dict_data["entry_count"] = len(entries)

    return ChatDictionaryResponse(**dict_data)


@router.delete(
    "/dictionaries/{dictionary_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a dictionary",
    description="Delete a dictionary and its entries.",
    tags=["chat-dictionaries"],
)
async def delete_chat_dictionary(
    dictionary_id: int,
    hard_delete: bool = Query(False, description="Permanently delete instead of soft delete"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> Response:
    """Delete a dictionary (soft delete by default)."""
    service = ChatDictionaryService(db)
    try:
        success = service.delete_dictionary(dictionary_id, hard_delete=hard_delete)
    except HTTPException:
        raise
    except CharactersRAGDBError as e:
        logger.error("Error deleting dictionary")
        raise map_db_error_to_http(e, default_detail="Failed to delete dictionary") from e
    except Exception as e:
        logger.error("Error deleting dictionary")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete dictionary",
        ) from e

    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dictionary not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/dictionaries/{dictionary_id}/entries",
    response_model=DictionaryEntryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add entry to dictionary",
    description="Add a pattern/replacement entry to a dictionary.",
    tags=["chat-dictionaries"],
)
async def add_dictionary_entry(
    dictionary_id: int,
    entry: DictionaryEntryCreate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> DictionaryEntryResponse:
    """
    Add a new entry to a dictionary.
    """
    service = ChatDictionaryService(db)
    try:
        dict_data = service.get_dictionary(dictionary_id)
    except CharactersRAGDBError as e:
        logger.error("Error retrieving dictionary before adding entry")
        raise map_db_error_to_http(e, default_detail="Failed to add dictionary entry") from e
    except Exception as e:
        logger.error("Error retrieving dictionary before adding entry")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add dictionary entry",
        ) from e

    if not dict_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dictionary not found")

    timed_effects_dict = entry.timed_effects.model_dump() if entry.timed_effects else None

    try:
        entry_id = service.add_entry(
            dictionary_id,
            pattern=entry.pattern,
            replacement=entry.replacement,
            probability=entry.probability,
            group=entry.group,
            timed_effects=timed_effects_dict,
            max_replacements=entry.max_replacements,
            type=entry.type,
            enabled=entry.enabled,
            case_sensitive=entry.case_sensitive,
        )

        created_entries = service.get_entries(dictionary_id=dictionary_id, active_only=False)
        entry_data = next((item for item in created_entries if item.get("id") == entry_id), None)
    except (InputError, CharactersRAGDBError) as e:
        if isinstance(e, CharactersRAGDBError) and not isinstance(e, InputError):
            logger.error("Error adding dictionary entry")
        raise map_db_error_to_http(e, default_detail="Failed to add dictionary entry") from e
    except Exception as e:
        logger.error("Error adding dictionary entry")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add dictionary entry",
        ) from e

    if not entry_data:
        entry_data = {
            "id": entry_id,
            "dictionary_id": dictionary_id,
            "pattern": entry.pattern,
            "replacement": entry.replacement,
            "probability": entry.probability,
            "group": entry.group,
            "timed_effects": entry.timed_effects.model_dump() if entry.timed_effects else None,
            "max_replacements": entry.max_replacements,
            "type": entry.type,
            "enabled": entry.enabled,
            "case_sensitive": entry.case_sensitive,
            "created_at": datetime.datetime.now(datetime.timezone.utc),
            "updated_at": datetime.datetime.now(datetime.timezone.utc),
        }

    return _entry_dict_to_response(entry_data, fallback_dictionary_id=dictionary_id)


@router.get(
    "/dictionaries/{dictionary_id}/entries",
    response_model=EntryListResponse,
    summary="List dictionary entries",
    description="List entries for a dictionary.",
    tags=["chat-dictionaries"],
)
async def list_dictionary_entries(
    dictionary_id: int,
    group: str | None = Query(None, description="Filter by group"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> EntryListResponse:
    """List all entries in a dictionary, optionally filtered by group."""
    service = ChatDictionaryService(db)
    try:
        dict_data = service.get_dictionary(dictionary_id)
        entries = service.get_entries(dictionary_id=dictionary_id, group=group, active_only=False) if dict_data else []
    except HTTPException:
        raise
    except CharactersRAGDBError as e:
        logger.error("Error listing dictionary entries")
        raise map_db_error_to_http(e, default_detail="Failed to list dictionary entries") from e
    except Exception as e:
        logger.error("Error listing dictionary entries")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list dictionary entries",
        ) from e

    if not dict_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dictionary not found")

    entry_responses = [
        _entry_dict_to_response(entry_dict, fallback_dictionary_id=dictionary_id) for entry_dict in entries
    ]

    return EntryListResponse(
        entries=entry_responses,
        total=len(entries),
        dictionary_id=dictionary_id,
        group=group,
    )


@router.put(
    "/dictionaries/entries/{entry_id}",
    response_model=DictionaryEntryResponse,
    summary="Update dictionary entry",
    description="Update entry fields such as replacement, enabled, group, case sensitivity, and probability.",
    tags=["chat-dictionaries"],
)
async def update_dictionary_entry(
    entry_id: int,
    update: DictionaryEntryUpdate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> DictionaryEntryResponse:
    """Update a dictionary entry."""
    service = ChatDictionaryService(db)

    # Security: Validate regex pattern safety when updates could enable regex matching.
    # The Pydantic validator only validates when BOTH type and pattern are provided.
    # We need to check existing values when only one field changes.
    existing_dict_id = None
    if (
        (update.pattern is not None and update.type is None)
        or (update.type == "regex" and update.pattern is None)
    ):
        # Pattern is being updated without explicit type, or type is being set to regex.
        try:
            existing_entry = service.get_entry(entry_id, active_only=False)
            if not existing_entry:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Entry not found for validation",
                )
            existing_dict_id = existing_entry.get("dictionary_id")
            if existing_dict_id is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Entry not found for validation",
                )
            existing_type = existing_entry.get("type")
            if not existing_type:
                # Fallback for legacy is_regex field
                existing_type = "regex" if existing_entry.get("is_regex") else "literal"
            if update.pattern is not None and update.type is None:
                if existing_type == "regex":
                    # Validate the new pattern for ReDoS safety
                    try:
                        validate_regex_pattern_safety(update.pattern)
                    except ValueError as e:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=str(e)
                        ) from e
            elif update.type == "regex":
                existing_pattern = existing_entry.get("pattern") or existing_entry.get("key") or ""
                try:
                    validate_regex_pattern_safety(existing_pattern)
                except ValueError as e:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=str(e)
                    ) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error checking existing entry type for regex validation")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unable to validate regex pattern safety",
            ) from e

    timed_effects_dict = update.timed_effects.model_dump() if update.timed_effects else None
    try:
        success = service.update_entry(
            entry_id,
            pattern=update.pattern,
            replacement=update.replacement,
            probability=update.probability,
            group=update.group,
            timed_effects=timed_effects_dict,
            max_replacements=update.max_replacements,
            type=update.type,
            enabled=update.enabled,
            case_sensitive=update.case_sensitive,
        )

        if success:
            dictionary_id_for_entry = (
                existing_dict_id
                if existing_dict_id is not None
                else service.get_entry_dictionary_id(entry_id)
            )
        else:
            dictionary_id_for_entry = None
        refreshed_entries = (
            service.get_entries(dictionary_id=dictionary_id_for_entry, active_only=False)
            if dictionary_id_for_entry is not None
            else []
        )
        updated_entry = next((item for item in refreshed_entries if item.get("id") == entry_id), None)
    except (InputError, CharactersRAGDBError) as e:
        if isinstance(e, CharactersRAGDBError) and not isinstance(e, InputError):
            logger.error("Error updating dictionary entry")
        raise map_db_error_to_http(e, default_detail="Failed to update dictionary entry") from e
    except Exception as e:
        logger.error("Error updating dictionary entry")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update dictionary entry",
        ) from e

    if not success or dictionary_id_for_entry is None or not updated_entry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Entry not found")

    return _entry_dict_to_response(updated_entry, fallback_dictionary_id=dictionary_id_for_entry)


@router.delete(
    "/dictionaries/entries/{entry_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete dictionary entry",
    description="Delete a single dictionary entry by ID.",
    tags=["chat-dictionaries"],
)
async def delete_dictionary_entry(
    entry_id: int,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> Response:
    """Delete a dictionary entry."""
    service = ChatDictionaryService(db)
    try:
        success = service.delete_entry(entry_id)
    except HTTPException:
        raise
    except CharactersRAGDBError as e:
        logger.error("Error deleting dictionary entry")
        raise map_db_error_to_http(e, default_detail="Failed to delete dictionary entry") from e
    except Exception as e:
        logger.error("Error deleting dictionary entry")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete dictionary entry",
        ) from e

    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Entry not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/dictionaries/entries/bulk",
    response_model=BulkOperationResponse,
    summary="Bulk operations on dictionary entries",
    description="Perform delete/activate/deactivate/group operations on multiple dictionary entries.",
    tags=["chat-dictionaries"],
)
async def bulk_dictionary_entry_operations(
    operation: BulkEntryOperation,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> BulkOperationResponse:
    """Perform bulk operations on dictionary entries with partial-failure reporting."""
    service = ChatDictionaryService(db)
    try:
        affected_count = 0
        failed_ids: list[int] = []

        for entry_id in operation.entry_ids:
            try:
                if operation.operation == "delete":
                    success = service.delete_entry(entry_id)
                elif operation.operation == "activate":
                    success = service.update_entry(entry_id, enabled=True)
                elif operation.operation == "deactivate":
                    success = service.update_entry(entry_id, enabled=False)
                elif operation.operation == "group":
                    success = service.update_entry(entry_id, group=operation.group_name)
                else:
                    success = False

                if success:
                    affected_count += 1
                else:
                    failed_ids.append(entry_id)
            except Exception as e:
                logger.warning("Bulk dictionary entry operation failed")
                failed_ids.append(entry_id)

        message = (
            f"Operation '{operation.operation}' completed: {affected_count} entries affected"
        )
        if failed_ids:
            message += f", {len(failed_ids)} failed"

        return BulkOperationResponse(
            success=len(failed_ids) == 0,
            affected_count=affected_count,
            failed_ids=failed_ids,
            message=message,
        )
    except InputError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except Exception as e:
        logger.error("Error performing bulk entry operation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform bulk entry operation",
        ) from e


@router.put(
    "/dictionaries/{dictionary_id}/entries/reorder",
    response_model=DictionaryEntryReorderResponse,
    summary="Reorder dictionary entries",
    description="Persist a new execution order for all entries in a dictionary.",
    tags=["chat-dictionaries"],
)
async def reorder_dictionary_entries(
    dictionary_id: int,
    reorder_request: DictionaryEntryReorderRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> DictionaryEntryReorderResponse:
    """Reorder entries for a dictionary using a full ordered list of entry IDs."""
    service = ChatDictionaryService(db)
    try:
        dict_data = service.get_dictionary(dictionary_id)
        if not dict_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dictionary not found")

        affected_count = service.reorder_entries(dictionary_id, reorder_request.entry_ids)
        ordered_entries = service.get_entries(dictionary_id=dictionary_id, active_only=False)
        ordered_entry_ids = [
            int(entry.get("id"))
            for entry in ordered_entries
            if entry.get("id") is not None
        ]

        return DictionaryEntryReorderResponse(
            success=True,
            dictionary_id=dictionary_id,
            affected_count=affected_count,
            entry_ids=ordered_entry_ids,
            message=f"Reordered {affected_count} entries.",
        )
    except HTTPException:
        raise
    except (InputError, CharactersRAGDBError) as e:
        if isinstance(e, CharactersRAGDBError) and not isinstance(e, InputError):
            logger.error("Error reordering dictionary entries")
        raise map_db_error_to_http(e, default_detail="Failed to reorder dictionary entries") from e
    except Exception as e:
        logger.error("Error reordering dictionary entries")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reorder dictionary entries",
        ) from e


@router.post(
    "/dictionaries/process",
    response_model=ProcessTextResponse,
    summary="Process text through dictionaries",
    description="Apply active dictionaries to the provided text and return transformed text and statistics.",
    tags=["chat-dictionaries"],
)
async def process_text_with_dictionaries(
    request: ProcessTextRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> ProcessTextResponse:
    """
    Process text through active dictionaries to apply replacements.
    """
    try:
        service = ChatDictionaryService(db)

        start_time = time.time()
        dictionary_ids = request.dictionary_ids
        if dictionary_ids is not None and request.dictionary_id is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Use either dictionary_id or dictionary_ids, not both.",
            )

        if dictionary_ids is not None:
            for dictionary_id in dictionary_ids:
                if service.get_dictionary(dictionary_id=dictionary_id) is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Dictionary {dictionary_id} not found.",
                    )

            processed_text = request.text
            replacements = 0
            iterations = 0
            entries_used: list[int] = []
            token_budget_exceeded = False
            remaining_token_budget = request.token_budget
            token_budget_used = 0 if request.token_budget is not None else None

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")

                for dictionary_id in dictionary_ids:
                    processed_text, stats = service.process_text(
                        processed_text,
                        dictionary_id=dictionary_id,
                        group=request.group,
                        max_iterations=request.max_iterations,
                        token_budget=remaining_token_budget,
                        return_stats=True,
                        chat_id=request.chat_id,
                    )

                    replacements += stats.get("replacements", 0)
                    iterations += stats.get("iterations", 0)
                    step_budget_used = stats.get("token_budget_used")
                    if isinstance(step_budget_used, (int, float)):
                        normalized_step_budget_used = int(step_budget_used)
                        token_budget_used = (
                            normalized_step_budget_used
                            if token_budget_used is None
                            else token_budget_used + normalized_step_budget_used
                        )
                        if remaining_token_budget is not None:
                            remaining_token_budget = max(
                                remaining_token_budget - normalized_step_budget_used,
                                0,
                            )
                    for entry_id in stats.get("entries_used", []):
                        if entry_id not in entries_used:
                            entries_used.append(entry_id)
                    if stats.get("token_budget_exceeded", False):
                        token_budget_exceeded = True
                        break

                warning_exceeded_budget = any(
                    issubclass(warning.category, TokenBudgetExceededWarning)
                    for warning in caught
                )
                if warning_exceeded_budget:
                    token_budget_exceeded = True

            processing_time_ms = (time.time() - start_time) * 1000

            return ProcessTextResponse(
                original_text=request.text,
                processed_text=processed_text,
                replacements=replacements,
                iterations=iterations,
                entries_used=entries_used,
                token_budget_exceeded=token_budget_exceeded,
                token_budget_used=token_budget_used,
                processing_time_ms=processing_time_ms,
            )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            processed_text, stats = service.process_text(
                request.text,
                dictionary_id=request.dictionary_id,
                group=request.group,
                max_iterations=request.max_iterations,
                token_budget=request.token_budget,
                return_stats=True,
                chat_id=request.chat_id,
            )

            token_budget_exceeded = any(issubclass(warning.category, TokenBudgetExceededWarning) for warning in caught)
            if token_budget_exceeded:
                stats["token_budget_exceeded"] = True

        processing_time_ms = (time.time() - start_time) * 1000

        return ProcessTextResponse(
            original_text=request.text,
            processed_text=processed_text,
            replacements=stats.get("replacements", 0),
            iterations=stats.get("iterations", 0),
            entries_used=stats.get("entries_used", []),
            token_budget_exceeded=stats.get("token_budget_exceeded", False),
            token_budget_used=stats.get("token_budget_used"),
            processing_time_ms=processing_time_ms,
        )
    except HTTPException:
        raise
    except (InputError, CharactersRAGDBError) as e:
        if isinstance(e, CharactersRAGDBError) and not isinstance(e, InputError):
            logger.error("Error processing text")
        raise map_db_error_to_http(e, default_detail="Failed to process text") from e
    except Exception as e:
        logger.error("Error processing text")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process text",
        ) from e


@router.post(
    "/dictionaries/import",
    response_model=ImportDictionaryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Import dictionary from markdown",
    description="Create a dictionary and entries from a markdown representation.",
    tags=["chat-dictionaries"],
)
async def import_dictionary(
    import_request: ImportDictionaryRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> ImportDictionaryResponse:
    """
    Import a dictionary from markdown format.
    """
    try:
        service = ChatDictionaryService(db)

        dict_id = service.import_from_markdown(import_request.content, import_request.name)

        entries = service.get_entries(dictionary_id=dict_id, active_only=False)
        groups = list({e.get("group") for e in entries if e.get("group")})

        if import_request.activate:
            service.update_dictionary(dict_id, is_active=True)

        return ImportDictionaryResponse(
            dictionary_id=dict_id,
            name=import_request.name,
            entries_imported=len(entries),
            groups_created=groups,
        )
    except (InputError, ConflictError, CharactersRAGDBError) as e:
        if isinstance(e, CharactersRAGDBError) and not isinstance(e, (ConflictError, InputError)):
            logger.error("Error importing dictionary")
        raise map_db_error_to_http(e, default_detail="Failed to import dictionary") from e
    except Exception as e:
        logger.error("Error importing dictionary")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to import dictionary",
        ) from e


@router.get(
    "/dictionaries/{dictionary_id}/export/markdown",
    response_model=ExportDictionaryResponse,
    include_in_schema=False,
    tags=["chat-dictionaries"],
)
@router.get(
    "/dictionaries/{dictionary_id}/export",
    response_model=ExportDictionaryResponse,
    summary="Export dictionary to markdown",
    description="Export a dictionary and entries to a markdown representation.",
    tags=["chat-dictionaries"],
)
async def export_dictionary(
    dictionary_id: int,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> ExportDictionaryResponse:
    """Export a dictionary to markdown format."""
    service = ChatDictionaryService(db)
    try:
        dict_data = service.get_dictionary(dictionary_id)
        content = service.export_to_markdown(dictionary_id) if dict_data else None
        entries = service.get_entries(dictionary_id=dictionary_id, active_only=False) if dict_data else []
    except (InputError, CharactersRAGDBError) as e:
        if isinstance(e, CharactersRAGDBError) and not isinstance(e, InputError):
            logger.error("Error exporting dictionary")
        raise map_db_error_to_http(e, default_detail="Failed to export dictionary") from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error exporting dictionary")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export dictionary",
        ) from e

    if not dict_data or content is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dictionary not found")

    groups = list({e.get("group") for e in entries if e.get("group")})

    return ExportDictionaryResponse(
        name=dict_data["name"],
        content=content,
        entry_count=len(entries),
        group_count=len(groups),
    )


@router.get(
    "/dictionaries/{dictionary_id}/export/json",
    response_model=ExportDictionaryJSONResponse,
    summary="Export dictionary to JSON",
    description="Export a dictionary and entries to a JSON representation.",
    tags=["chat-dictionaries"],
)
async def export_dictionary_json(
    dictionary_id: int,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> ExportDictionaryJSONResponse:
    """Export a dictionary to JSON."""
    try:
        service = ChatDictionaryService(db)
        data = service.export_to_json(dictionary_id)
        return ExportDictionaryJSONResponse(**data)
    except (InputError, CharactersRAGDBError) as e:
        if isinstance(e, CharactersRAGDBError) and not isinstance(e, InputError):
            logger.error("Error exporting dictionary JSON")
        raise map_db_error_to_http(e, default_detail="Failed to export dictionary JSON") from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error exporting dictionary JSON")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export dictionary JSON",
        ) from e


@router.post(
    "/dictionaries/import/json",
    response_model=ImportDictionaryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Import dictionary from JSON",
    description="Create a dictionary and entries from a JSON payload.",
    tags=["chat-dictionaries"],
)
async def import_dictionary_json(
    import_request: ImportDictionaryJSONRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> ImportDictionaryResponse:
    """Import a dictionary and entries from JSON."""
    try:
        service = ChatDictionaryService(db)
        dict_id = service.import_from_json(import_request.data)
        entries = service.get_entries(dictionary_id=dict_id, active_only=False)
        if import_request.activate:
            service.update_dictionary(dict_id, is_active=True)
        name = import_request.data.get("name") or service.get_dictionary(dict_id).get("name", "Imported")
        return ImportDictionaryResponse(
            dictionary_id=dict_id,
            name=name,
            entries_imported=len(entries),
            groups_created=list({e.get("group") for e in entries if e.get("group")}),
        )
    except (InputError, ConflictError, CharactersRAGDBError) as e:
        if isinstance(e, CharactersRAGDBError) and not isinstance(e, (ConflictError, InputError)):
            logger.error("Error importing dictionary JSON")
        raise map_db_error_to_http(e, default_detail="Failed to import dictionary JSON") from e
    except Exception as e:
        logger.error("Error importing dictionary JSON")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to import dictionary JSON",
        ) from e


@router.get(
    "/dictionaries/{dictionary_id}/activity",
    response_model=DictionaryActivityListResponse,
    summary="List dictionary activity events",
    description="Return recent transformation activity for a dictionary.",
    tags=["chat-dictionaries"],
)
async def list_dictionary_activity(
    dictionary_id: int,
    limit: int = Query(20, ge=1, le=100, description="Maximum number of events to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> DictionaryActivityListResponse:
    """Return paginated recent transformation activity for a dictionary."""
    service = ChatDictionaryService(db)
    try:
        dict_data = service.get_dictionary(dictionary_id)
        if not dict_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dictionary not found")

        events, total = service.list_transform_activity(
            dictionary_id,
            limit=limit,
            offset=offset,
        )
        return DictionaryActivityListResponse(
            dictionary_id=dictionary_id,
            events=events,
            total=total,
            limit=limit,
            offset=offset,
            pagination=build_offset_pagination_meta(
                total=total,
                offset=offset,
                limit=limit,
                count=len(events),
            ),
        )
    except HTTPException:
        raise
    except CharactersRAGDBError as e:
        logger.error("Error listing dictionary activity")
        raise map_db_error_to_http(e, default_detail="Failed to list dictionary activity") from e
    except Exception as e:
        logger.error("Error listing dictionary activity")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list dictionary activity",
        ) from e


@router.get(
    "/dictionaries/{dictionary_id}/versions",
    response_model=DictionaryVersionListResponse,
    summary="List dictionary version history",
    description="Return paginated revision history snapshots for a dictionary.",
    tags=["chat-dictionaries"],
)
async def list_dictionary_versions(
    dictionary_id: int,
    limit: int = Query(20, ge=1, le=100, description="Maximum number of versions to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> DictionaryVersionListResponse:
    """Return paginated dictionary version history."""
    service = ChatDictionaryService(db)
    try:
        dict_data = service.get_dictionary(dictionary_id)
        if not dict_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dictionary not found")

        versions, total = service.list_dictionary_versions(
            dictionary_id,
            limit=limit,
            offset=offset,
        )
        return DictionaryVersionListResponse(
            dictionary_id=dictionary_id,
            versions=[
                {
                    **version,
                    "created_at": coerce_datetime(version.get("created_at")),
                }
                for version in versions
            ],
            total=total,
            limit=limit,
            offset=offset,
            pagination=build_offset_pagination_meta(
                total=total,
                offset=offset,
                limit=limit,
                count=len(versions),
            ),
        )
    except HTTPException:
        raise
    except CharactersRAGDBError as e:
        logger.error("Error listing dictionary versions")
        raise map_db_error_to_http(e, default_detail="Failed to list dictionary versions") from e
    except Exception as e:
        logger.error("Error listing dictionary versions")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list dictionary versions",
        ) from e


@router.get(
    "/dictionaries/{dictionary_id}/versions/{revision}",
    response_model=DictionaryVersionDetailResponse,
    summary="Get dictionary version snapshot",
    description="Return metadata and entry payload for a specific dictionary revision.",
    tags=["chat-dictionaries"],
)
async def get_dictionary_version(
    dictionary_id: int,
    revision: int,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> DictionaryVersionDetailResponse:
    """Get a specific dictionary revision snapshot."""
    service = ChatDictionaryService(db)
    try:
        snapshot = service.get_dictionary_version(dictionary_id, revision)
        if not snapshot:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dictionary revision not found")

        entries_payload = snapshot.get("entries") if isinstance(snapshot.get("entries"), list) else []
        entry_responses = [
            _entry_dict_to_response(entry_data, fallback_dictionary_id=dictionary_id)
            for entry_data in entries_payload
            if isinstance(entry_data, dict)
        ]

        dictionary_payload = dict(snapshot.get("dictionary") or {})
        dictionary_payload.setdefault("id", int(dictionary_id))
        dictionary_payload.setdefault("entry_count", len(entry_responses))
        dictionary_payload.setdefault("used_by_chat_count", 0)
        dictionary_payload.setdefault("used_by_active_chat_count", 0)
        dictionary_payload.setdefault("used_by_chat_refs", [])
        dictionary_payload["entry_count"] = len(entry_responses)
        if dictionary_payload.get("created_at") is None:
            dictionary_payload["created_at"] = datetime.datetime.now(datetime.timezone.utc)
        if dictionary_payload.get("updated_at") is None:
            dictionary_payload["updated_at"] = datetime.datetime.now(datetime.timezone.utc)
        if dictionary_payload.get("version") is None:
            dictionary_payload["version"] = 1

        return DictionaryVersionDetailResponse(
            dictionary_id=dictionary_id,
            revision=int(snapshot.get("revision") or revision),
            source_dictionary_version=snapshot.get("source_dictionary_version"),
            change_type=str(snapshot.get("change_type") or "unspecified"),
            summary=snapshot.get("summary"),
            created_at=coerce_datetime(snapshot.get("created_at")),
            dictionary=ChatDictionaryResponse(**dictionary_payload),
            entries=entry_responses,
        )
    except HTTPException:
        raise
    except CharactersRAGDBError as e:
        logger.error("Error reading dictionary revision")
        raise map_db_error_to_http(e, default_detail="Failed to read dictionary revision") from e
    except Exception as e:
        logger.error("Error reading dictionary revision")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read dictionary revision",
        ) from e


@router.post(
    "/dictionaries/{dictionary_id}/versions/{revision}/revert",
    response_model=DictionaryVersionRevertResponse,
    summary="Revert dictionary to a previous revision",
    description="Restore dictionary metadata and entries from a specific version-history revision.",
    tags=["chat-dictionaries"],
)
async def revert_dictionary_version(
    dictionary_id: int,
    revision: int,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> DictionaryVersionRevertResponse:
    """Revert dictionary to a previous version snapshot."""
    service = ChatDictionaryService(db)
    try:
        result = service.revert_dictionary_to_revision(dictionary_id, revision)
        return DictionaryVersionRevertResponse(**result)
    except ConflictError as e:
        raise map_db_error_to_http(
            e,
            default_detail="Failed to revert dictionary revision",
        ) from e
    except InputError as e:
        raise map_db_error_to_http(
            e,
            default_detail="Failed to revert dictionary revision",
            not_found_substrings=("not found",),
        ) from e
    except CharactersRAGDBError as e:
        logger.error("Error reverting dictionary revision")
        raise map_db_error_to_http(e, default_detail="Failed to revert dictionary revision") from e
    except Exception as e:
        logger.error("Error reverting dictionary revision")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revert dictionary revision",
        ) from e


@router.get(
    "/dictionaries/{dictionary_id}/statistics",
    response_model=DictionaryStatistics,
    summary="Get dictionary statistics",
    description="Return counts, groups, usage metrics, and averages for the specified dictionary.",
    tags=["chat-dictionaries"],
)
async def get_dictionary_statistics(
    dictionary_id: int,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> DictionaryStatistics:
    """Get statistics for a dictionary."""
    service = ChatDictionaryService(db)
    try:
        dict_data = service.get_dictionary(dictionary_id)
        stats = service.get_statistics(dictionary_id) if dict_data else {}
        entries = service.get_entries(dictionary_id=dictionary_id, active_only=False) if dict_data else []
        usage_stats = service.get_usage_statistics(dictionary_id) if dict_data else {}
    except HTTPException:
        raise
    except (InputError, CharactersRAGDBError) as e:
        if isinstance(e, CharactersRAGDBError) and not isinstance(e, InputError):
            logger.error("Error getting dictionary statistics")
        raise map_db_error_to_http(e, default_detail="Failed to get dictionary statistics") from e
    except Exception as e:
        logger.error("Error getting dictionary statistics")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get dictionary statistics",
        ) from e

    if not dict_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dictionary not found")

    regex_count = int(stats.get("regex_entries", 0))
    total_entries = int(stats.get("total_entries", len(entries)))
    literal_count = int(stats.get("literal_entries", total_entries - regex_count))
    groups = sorted({str(e.get("group")).strip() for e in entries if str(e.get("group") or "").strip()})
    avg_probability = sum(float(e.get("probability", 1.0)) for e in entries) / len(entries) if entries else 0.0
    enabled_entries = sum(1 for entry in entries if bool(entry.get("enabled", True)))
    disabled_entries = max(total_entries - enabled_entries, 0)
    probabilistic_entries = int(stats.get("probabilistic_entries", 0))
    timed_effect_entries = sum(1 for entry in entries if _entry_has_timed_effects(entry))
    entry_usage = [
        {
            "entry_id": int(entry.get("id")),
            "pattern": str(entry.get("pattern") or entry.get("key") or ""),
            "usage_count": int(entry.get("usage_count", 0) or 0),
            "last_used_at": _coerce_optional_datetime(entry.get("last_used_at")),
        }
        for entry in entries
        if entry.get("id") is not None
    ]
    entry_usage.sort(
        key=lambda item: (item.get("usage_count", 0), int(item.get("entry_id", 0))),
        reverse=True,
    )
    zero_usage_entries = sum(1 for item in entry_usage if int(item.get("usage_count", 0) or 0) == 0)
    pattern_conflicts = _analyze_pattern_conflicts(entries)

    return DictionaryStatistics(
        dictionary_id=dictionary_id,
        name=dict_data["name"],
        total_entries=total_entries,
        regex_entries=regex_count,
        literal_entries=literal_count,
        enabled_entries=enabled_entries,
        disabled_entries=disabled_entries,
        probabilistic_entries=probabilistic_entries,
        timed_effect_entries=timed_effect_entries,
        groups=groups,
        average_probability=avg_probability,
        created_at=coerce_datetime(dict_data.get("created_at")),
        updated_at=coerce_datetime(dict_data.get("updated_at")),
        zero_usage_entries=zero_usage_entries,
        entry_usage=entry_usage,
        pattern_conflict_count=len(pattern_conflicts),
        pattern_conflicts=pattern_conflicts,
        total_usage_count=int(usage_stats.get("times_used", 0) or 0),
        last_used=_coerce_optional_datetime(usage_stats.get("last_used_at")),
    )
