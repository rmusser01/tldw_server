"""Backend aggregation for VN Play session setup options."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.schemas.vn_asset_schemas import (
    VNAssetPackResponse,
    VNAssetReadinessResponse,
)
from tldw_Server_API.app.api.v1.schemas.vn_play_schemas import (
    VNPlayMode,
    VNPlaySetupAssetPackOption,
    VNPlaySetupCharacterOption,
    VNPlaySetupCompatibility,
    VNPlaySetupDefaults,
    VNPlaySetupEmptyState,
    VNPlaySetupOptionsResponse,
    VNPlaySetupPagination,
    VNPlaySetupPaginationSet,
    VNPlaySetupTrustLevel,
    VNPlaySetupTrustSource,
    VNPlaySetupWarning,
    VNPlaySetupWarningSeverity,
    VNPlaySetupWarningSummary,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.VN_Assets.service import VNAssetPackService

DEFAULT_SETUP_LIMIT = 25
MAX_SETUP_LIMIT = 100
CONTENT_RATING_ORDER = {
    "general": 0,
    "suggestive": 1,
    "mature": 2,
    "violent": 3,
}
REQUIRED_ASSET_READINESS_ERROR_CODES = {
    "missing_required_asset",
    "missing_required_assets",
    "missing_required_runtime_asset",
    "missing_required_slot",
    "required_slot_not_ready",
}
SEVERITY_ORDER: dict[VNPlaySetupWarningSeverity, int] = {
    "info": 0,
    "warning": 1,
    "high_risk": 2,
}
WARNING_MESSAGES = {
    "pack_character_mismatch": "This pack was generated for a different primary character.",
    "pack_not_ready": "This pack is not runtime-ready yet.",
    "pack_has_readiness_errors": "This pack has readiness errors.",
    "pack_missing_required_assets": "This pack is missing required runtime assets.",
    "content_rating_mismatch": "This pack content rating differs from the requested session rating.",
    "pack_untrusted_import": "This pack was last committed from an untrusted import.",
    "pack_deleted_or_archived": "This pack is hidden from normal use.",
    "readiness_unavailable": "Readiness could not be computed for this pack.",
}


def build_vn_play_setup_options(
    *,
    db: CharactersRAGDB,
    owner_user_id: int,
    mode: VNPlayMode | None = None,
    character_query: str | None = None,
    pack_query: str | None = None,
    character_limit: int = DEFAULT_SETUP_LIMIT,
    character_offset: int = 0,
    pack_limit: int = DEFAULT_SETUP_LIMIT,
    pack_offset: int = 0,
    selected_character_id: int | None = None,
    content_rating: str | None = "general",
) -> VNPlaySetupOptionsResponse:
    """Return setup selectors with server-computed pack readiness and warnings."""
    character_limit = _bounded_limit(character_limit)
    pack_limit = _bounded_limit(pack_limit)
    character_offset = max(0, int(character_offset))
    pack_offset = max(0, int(pack_offset))
    requested_rating = _normalized_rating(content_rating or "general")

    character_rows, character_total = db.query_character_setup_options(
        query=character_query,
        include_deleted=False,
        limit=character_limit,
        offset=character_offset,
    )
    characters = [_character_option(row) for row in character_rows]

    selected_character = _selected_character_option(db, selected_character_id)
    asset_service = VNAssetPackService(db, owner_user_id=owner_user_id)
    pack_rows, pack_has_more = asset_service.list_packs_for_setup(
        query=pack_query,
        limit=pack_limit,
        offset=pack_offset,
    )
    provenance_by_pack_id, provenance_lookup_failed = _latest_import_provenance(
        asset_service,
        pack_rows,
    )

    asset_packs = [
        _asset_pack_option(
            asset_service=asset_service,
            pack=pack,
            selected_character=selected_character,
            requested_rating=requested_rating,
            provenance=provenance_by_pack_id.get(pack.id),
            provenance_lookup_failed=provenance_lookup_failed,
        )
        for pack in pack_rows
    ]
    asset_packs = _sort_pack_options(asset_packs)

    return VNPlaySetupOptionsResponse(
        characters=characters,
        selected_character=selected_character,
        asset_packs=asset_packs,
        defaults=_setup_defaults(
            mode=mode,
            content_rating=requested_rating,
            selected_character=selected_character,
            characters=characters,
            asset_packs=asset_packs,
        ),
        pagination=VNPlaySetupPaginationSet(
            characters=VNPlaySetupPagination(
                limit=character_limit,
                offset=character_offset,
                has_more=(character_offset + len(characters)) < character_total,
                total=character_total,
            ),
            asset_packs=VNPlaySetupPagination(
                limit=pack_limit,
                offset=pack_offset,
                has_more=pack_has_more,
                total=None,
            ),
        ),
        empty_states=_empty_states(
            characters=characters,
            character_total=character_total,
            character_query=character_query,
            character_offset=character_offset,
            asset_packs=asset_packs,
            pack_query=pack_query,
            pack_offset=pack_offset,
            selected_character_id=selected_character_id,
            selected_character=selected_character,
        ),
        generated_at=_utc_now_iso(),
    )


def _bounded_limit(value: int) -> int:
    """Clamp client-provided setup page sizes to the supported API bounds."""
    return max(1, min(MAX_SETUP_LIMIT, int(value)))


def _selected_character_option(
    db: CharactersRAGDB,
    selected_character_id: int | None,
) -> VNPlaySetupCharacterOption | None:
    """Return the selected character selector row even when it is off-page."""
    if selected_character_id is None:
        return None
    row = db.get_character_setup_option_by_id(int(selected_character_id))
    if row is None:
        return None
    return _character_option(row)


def _character_option(row: dict[str, Any]) -> VNPlaySetupCharacterOption:
    """Serialize a lightweight character row into selector-safe API shape."""
    has_image = (
        bool(row.get("has_image"))
        if "has_image" in row
        else bool(row.get("image") or row.get("image_base64"))
    )
    return VNPlaySetupCharacterOption(
        id=int(row["id"]),
        name=str(row.get("name") or f"Character {row['id']}"),
        description_preview=_preview_text(row.get("description")),
        tags=_string_list(row.get("tags")),
        favorite=_character_favorite(row.get("extensions")),
        deleted=bool(row.get("deleted", False)),
        has_image=has_image,
    )


def _asset_pack_option(
    *,
    asset_service: VNAssetPackService,
    pack: VNAssetPackResponse,
    selected_character: VNPlaySetupCharacterOption | None,
    requested_rating: str,
    provenance: dict[str, Any] | None,
    provenance_lookup_failed: bool,
) -> VNPlaySetupAssetPackOption:
    """Serialize pack setup state with readiness, compatibility, and warnings."""
    readiness = _readiness_for_pack(asset_service, pack.id)
    warnings: list[VNPlaySetupWarning] = []
    if readiness is None:
        ready = False
        readiness_status = "unknown"
        readiness_warnings: list[str] = []
        readiness_errors: list[str] = []
        warnings.append(_warning("readiness_unavailable", "warning"))
    else:
        ready = bool(readiness.ready)
        readiness_status = str(readiness.status)
        readiness_warnings = [str(item) for item in readiness.warnings]
        readiness_errors = [str(item) for item in readiness.errors]
        if not ready:
            warnings.append(_warning("pack_not_ready", "high_risk"))
        if readiness_errors:
            warnings.append(_warning("pack_has_readiness_errors", "high_risk"))
        if _missing_required_assets(readiness_errors):
            warnings.append(_warning("pack_missing_required_assets", "high_risk"))

    compatibility = _compatibility(pack, selected_character)
    if compatibility.status == "different_character":
        warnings.append(_warning("pack_character_mismatch", "high_risk"))

    rating_warning = _content_rating_warning(pack.content_rating, requested_rating)
    if rating_warning is not None:
        warnings.append(rating_warning)

    trust_level, trust_source = _trust_for_provenance(
        provenance,
        lookup_failed=provenance_lookup_failed,
    )
    if trust_level == "untrusted_import":
        warnings.append(_warning("pack_untrusted_import", "warning"))

    if pack.deleted or str(pack.status).lower() in {"archived", "deleted", "hidden"}:
        warnings.append(_warning("pack_deleted_or_archived", "high_risk"))

    warning_summary = _warning_summary(warnings)
    recommended = (
        ready
        and compatibility.status in {"compatible", "unknown"}
        and not warning_summary.requires_acknowledgement
    )
    return VNPlaySetupAssetPackOption(
        id=pack.id,
        title=pack.title,
        primary_character_id=pack.primary_character_id,
        content_rating=pack.content_rating,
        status=pack.status,
        trust_level=trust_level,
        trust_source=trust_source,
        ready=ready,
        readiness_status=readiness_status,
        readiness_warnings=readiness_warnings,
        readiness_errors=readiness_errors,
        compatibility=compatibility,
        warning_summary=warning_summary,
        recommended=recommended,
    )


def _latest_import_provenance(
    asset_service: VNAssetPackService,
    packs: list[VNAssetPackResponse],
) -> tuple[dict[int, dict[str, Any]], bool]:
    """Return latest completed import provenance for listed packs when available."""
    pack_ids = [pack.id for pack in packs]
    if not pack_ids:
        return {}, False
    try:
        return (
            asset_service.repo.latest_completed_import_provenance_by_pack_ids(
                owner_user_id=asset_service.owner_user_id,
                pack_ids=pack_ids,
            ),
            False,
        )
    except Exception as exc:
        logger.warning("Failed to derive VN setup import provenance: {}", exc)
        return {}, True


def _readiness_for_pack(
    asset_service: VNAssetPackService,
    pack_id: int,
) -> VNAssetReadinessResponse | None:
    """Return pack readiness, degrading to ``None`` when the check fails."""
    try:
        return asset_service.get_readiness(pack_id)
    except Exception as exc:
        logger.warning("Failed to compute VN setup readiness for pack {}: {}", pack_id, exc)
        return None


def _compatibility(
    pack: VNAssetPackResponse,
    selected_character: VNPlaySetupCharacterOption | None,
) -> VNPlaySetupCompatibility:
    """Derive selected-character compatibility for one asset pack."""
    if selected_character is None:
        return VNPlaySetupCompatibility(status="unknown", reason_codes=["no_selected_character"])
    if pack.primary_character_id == selected_character.id:
        return VNPlaySetupCompatibility(status="compatible", reason_codes=[])
    return VNPlaySetupCompatibility(
        status="different_character",
        reason_codes=["pack_character_mismatch"],
    )


def _content_rating_warning(
    pack_rating: str | None,
    requested_rating: str | None,
) -> VNPlaySetupWarning | None:
    """Return a warning when pack and requested content ratings differ."""
    normalized_pack = _normalized_rating(pack_rating)
    normalized_requested = _normalized_rating(requested_rating)
    if normalized_pack == normalized_requested:
        return None

    pack_rank = CONTENT_RATING_ORDER.get(normalized_pack)
    requested_rank = CONTENT_RATING_ORDER.get(normalized_requested)
    severity: VNPlaySetupWarningSeverity
    if pack_rank is None or requested_rank is None or pack_rank > requested_rank:
        severity = "high_risk"
    else:
        severity = "warning"
    return _warning("content_rating_mismatch", severity)


def _trust_for_provenance(
    provenance: dict[str, Any] | None,
    *,
    lookup_failed: bool,
) -> tuple[VNPlaySetupTrustLevel, VNPlaySetupTrustSource]:
    """Map import-journal provenance into setup trust level and source labels."""
    if lookup_failed:
        return "unknown", "unknown"
    if provenance is None:
        return "local", "local_pack"
    trust_mode = str(provenance.get("trust_mode") or "").strip()
    if trust_mode in {"trusted_restore", "untrusted_import"}:
        return trust_mode, "latest_import_journal"
    return "unknown", "unknown"


def _warning(
    code: str,
    severity: VNPlaySetupWarningSeverity,
) -> VNPlaySetupWarning:
    """Build a warning payload with acknowledgement semantics from severity."""
    return VNPlaySetupWarning(
        code=code,
        severity=severity,
        message=WARNING_MESSAGES.get(code, code),
        requires_acknowledgement=severity == "high_risk",
    )


def _warning_summary(
    warnings: list[VNPlaySetupWarning],
) -> VNPlaySetupWarningSummary:
    """Summarize warning severity and acknowledgement requirements."""
    if not warnings:
        return VNPlaySetupWarningSummary()
    highest = max(warnings, key=lambda warning: SEVERITY_ORDER[warning.severity]).severity
    return VNPlaySetupWarningSummary(
        highest_severity=highest,
        requires_acknowledgement=any(warning.requires_acknowledgement for warning in warnings),
        warnings=warnings,
    )


def _setup_defaults(
    *,
    mode: VNPlayMode | None,
    content_rating: str,
    selected_character: VNPlaySetupCharacterOption | None,
    characters: list[VNPlaySetupCharacterOption],
    asset_packs: list[VNPlaySetupAssetPackOption],
) -> VNPlaySetupDefaults:
    """Choose conservative setup defaults from selected and recommended options."""
    default_character_id = selected_character.id if selected_character is not None else None
    if default_character_id is None and len(characters) == 1:
        default_character_id = characters[0].id

    recommended_pack_ids = [pack.id for pack in asset_packs if pack.recommended]
    default_pack_id = recommended_pack_ids[0] if len(recommended_pack_ids) == 1 else None
    return VNPlaySetupDefaults(
        mode=mode,
        character_id=default_character_id,
        asset_pack_id=default_pack_id,
        content_rating=content_rating,
    )


def _empty_states(
    *,
    characters: list[VNPlaySetupCharacterOption],
    character_total: int,
    character_query: str | None,
    character_offset: int,
    asset_packs: list[VNPlaySetupAssetPackOption],
    pack_query: str | None,
    pack_offset: int,
    selected_character_id: int | None,
    selected_character: VNPlaySetupCharacterOption | None,
) -> list[VNPlaySetupEmptyState]:
    """Build scoped empty-state hints for selector pages and filters."""
    states: list[VNPlaySetupEmptyState] = []
    if character_total == 0:
        states.append(
            VNPlaySetupEmptyState(
                code="no_characters",
                scope="filter" if _has_query(character_query) else "global",
                message="No available characters were found.",
            )
        )
    elif not characters and character_offset > 0:
        states.append(
            VNPlaySetupEmptyState(
                code="no_characters",
                scope="page",
                message="No characters were found on this page.",
            )
        )

    if selected_character_id is not None and selected_character is None:
        states.append(
            VNPlaySetupEmptyState(
                code="selected_character_not_found",
                scope="global",
                message="The selected character is not available.",
            )
        )

    if not asset_packs:
        if pack_offset > 0:
            scope = "page"
        elif _has_query(pack_query):
            scope = "filter"
        else:
            scope = "global"
        states.append(
            VNPlaySetupEmptyState(
                code="no_asset_packs",
                scope=scope,
                message="No VN asset packs were found.",
            )
        )
        return states

    if not any(pack.ready for pack in asset_packs):
        states.append(
            VNPlaySetupEmptyState(
                code="no_ready_packs",
                scope="page",
                message="No ready packs were found in this page of results.",
            )
        )
    if selected_character is not None and not any(
        pack.compatibility.status == "compatible" for pack in asset_packs
    ):
        states.append(
            VNPlaySetupEmptyState(
                code="no_compatible_packs",
                scope="page",
                message="No compatible packs were found in this page of results.",
            )
        )
    return states


def _sort_pack_options(
    packs: list[VNPlaySetupAssetPackOption],
) -> list[VNPlaySetupAssetPackOption]:
    """Sort packs with recommended and compatible options first without reordering ties."""
    def sort_key(indexed_pack: tuple[int, VNPlaySetupAssetPackOption]) -> tuple[int, int]:
        index, pack = indexed_pack
        if pack.recommended:
            group = 0
        elif pack.compatibility.status == "compatible":
            group = 1
        elif pack.compatibility.status == "unknown":
            group = 2
        else:
            group = 3
        return group, index

    return [pack for _, pack in sorted(enumerate(packs), key=sort_key)]


def _missing_required_assets(readiness_errors: list[str]) -> bool:
    """Return true only for structured missing-required readiness error codes."""
    for error in readiness_errors:
        code = str(error).split(":", 1)[0].strip().lower()
        if code in REQUIRED_ASSET_READINESS_ERROR_CODES:
            return True
    return False


def _normalized_rating(value: str | None) -> str:
    """Normalize empty or mixed-case content ratings for comparisons."""
    normalized = (value or "general").strip().lower()
    return normalized or "general"


def _preview_text(value: Any, *, max_length: int = 160) -> str | None:
    """Return compact preview text whose final length stays within ``max_length``."""
    if value is None:
        return None
    text = " ".join(str(value).split())
    if not text:
        return None
    if len(text) <= max_length:
        return text
    if max_length <= 3:
        return text[:max_length]
    return text[: max_length - 3].rstrip() + "..."


def _string_list(value: Any) -> list[str]:
    """Normalize stored JSON, iterable, or scalar tag values into strings."""
    if value is None:
        return []
    raw_values: list[Any]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            raw_values = parsed if isinstance(parsed, list) else [value]
        except (TypeError, ValueError, json.JSONDecodeError):
            raw_values = [value]
    elif isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        raw_values = [value]
    return [tag for item in raw_values if (tag := str(item).strip())]


def _character_favorite(extensions: Any) -> bool:
    """Extract the setup favorite flag from known extension locations."""
    if extensions is None:
        return False
    parsed = extensions
    if isinstance(extensions, str):
        try:
            parsed = json.loads(extensions)
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
    if not isinstance(parsed, dict):
        return False
    nested = parsed.get("tldw")
    if isinstance(nested, dict) and nested.get("favorite") is not None:
        return bool(nested.get("favorite"))
    return bool(parsed.get("favorite", False))


def _has_query(value: str | None) -> bool:
    """Return whether a query parameter contains non-whitespace text."""
    return bool((value or "").strip())


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in API-friendly ISO-8601 form."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
