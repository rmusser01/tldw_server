"""Backend aggregation for VN Play session setup options."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Mapping

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
    VNPlaySetupScriptVersionOption,
    VNPlaySetupTrustLevel,
    VNPlaySetupTrustSource,
    VNPlaySetupWarning,
    VNPlaySetupWarningSeverity,
    VNPlaySetupWarningSummary,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNPolicy_DB import VNProfileSnapshotRepository
from tldw_Server_API.app.core.DB_Management.VNScripts_DB import VNScriptsRepository
from tldw_Server_API.app.core.VN_Assets.service import VNAssetPackService
from tldw_Server_API.app.core.VN_Policy.service import evaluate_character_safety_definition
from tldw_Server_API.app.core.VN_Scripts.service import _character_safety_status

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
    "script_pack_unavailable": "The script's asset pack could not be loaded.",
    "policy_snapshot_unavailable": "The script policy snapshot could not be loaded.",
    "generation_profile_snapshot_unavailable": "The script generation profile snapshot could not be loaded.",
    "generation_profile_snapshot_missing": "The script references a generation profile snapshot that is not pinned.",
    "generation_profile_output_schema_unavailable": "The generation profile does not support one of this script's generated output schemas.",
    "character_safety_missing": "Character safety metadata is missing.",
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
    script_versions = _script_version_options(
        db=db,
        asset_service=asset_service,
        owner_user_id=owner_user_id,
        selected_character=selected_character,
        requested_rating=requested_rating,
    ) if mode == "scripted_story" else []

    return VNPlaySetupOptionsResponse(
        characters=characters,
        selected_character=selected_character,
        asset_packs=asset_packs,
        script_versions=script_versions,
        defaults=_setup_defaults(
            mode=mode,
            content_rating=requested_rating,
            selected_character=selected_character,
            characters=characters,
            asset_packs=asset_packs,
            script_versions=script_versions,
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
            script_versions=script_versions,
            mode=mode,
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


def _script_version_options(
    *,
    db: CharactersRAGDB,
    asset_service: VNAssetPackService,
    owner_user_id: int,
    selected_character: VNPlaySetupCharacterOption | None,
    requested_rating: str,
) -> list[VNPlaySetupScriptVersionOption]:
    """Return latest published script versions with runtime readiness hints."""
    script_repo = VNScriptsRepository.initialized(db)
    profile_snapshots = VNProfileSnapshotRepository.initialized(db)
    versions = script_repo.list_latest_versions_for_setup(
        owner_user_id=owner_user_id,
        limit=DEFAULT_SETUP_LIMIT,
        offset=0,
    )
    options: list[VNPlaySetupScriptVersionOption] = []
    for version in versions:
        pack = _pack_for_script_version(asset_service, int(version["asset_pack_id"]))
        warnings: list[VNPlaySetupWarning] = []
        pack_ready = False
        compatible = True
        if pack is None:
            warnings.append(_warning("script_pack_unavailable", "high_risk"))
        else:
            pack_option = _asset_pack_option(
                asset_service=asset_service,
                pack=pack,
                selected_character=selected_character,
                requested_rating=requested_rating,
                provenance=None,
                provenance_lookup_failed=False,
            )
            pack_ready = pack_option.ready
            compatible = pack_option.compatibility.status in {"compatible", "unknown"}
            warnings.extend(pack_option.warning_summary.warnings)

        policy_blocked = False
        policy_snapshot = profile_snapshots.get_profile_snapshot(
            int(version["policy_snapshot_id"]),
            owner_user_id=owner_user_id,
        )
        if policy_snapshot is None:
            warnings.append(_warning("policy_snapshot_unavailable", "high_risk"))
            policy_blocked = True
        else:
            decision = evaluate_character_safety_definition(
                profile_definition=policy_snapshot["definition"],
                policy_profile_id=str(policy_snapshot["profile_id"]),
                content_rating=str(version.get("content_rating") or requested_rating),
                metadata_status=_script_character_safety_status(db, pack),
            )
            policy_blocked = bool(decision.get("blocked"))
            warnings.extend(_policy_warnings(decision))

        generation_snapshot_id = _optional_int(version.get("generation_profile_snapshot_id"))
        generation_snapshot = None
        if generation_snapshot_id is not None:
            generation_snapshot = profile_snapshots.get_profile_snapshot(
                generation_snapshot_id,
                owner_user_id=owner_user_id,
            )
        generation_definition = _profile_definition(generation_snapshot) if generation_snapshot else {}
        generation_requirements = _script_generation_requirements(version.get("program"))
        generation_blocked = generation_snapshot is None
        if generation_snapshot_id is None:
            warnings.append(_warning("generation_profile_snapshot_missing", "high_risk"))
        elif generation_snapshot is None:
            warnings.append(_warning("generation_profile_snapshot_unavailable", "high_risk"))
        generation_warnings, generation_requirements_blocked = _generation_profile_warnings(
            profile_snapshots=profile_snapshots,
            owner_user_id=owner_user_id,
            version=version,
            requirements=generation_requirements,
        )
        warnings.extend(generation_warnings)
        generation_blocked = generation_blocked or generation_requirements_blocked

        warning_summary = _warning_summary(_dedupe_warnings(warnings))
        ready = pack_ready and not policy_blocked and not generation_blocked
        options.append(
            VNPlaySetupScriptVersionOption(
                id=int(version["id"]),
                script_id=int(version["script_id"]),
                title=str(version.get("title") or f"Script {version['script_id']}"),
                version_number=int(version["version_number"]),
                label=version.get("label"),
                asset_pack_id=int(version["asset_pack_id"]),
                manifest_snapshot_id=int(version["manifest_snapshot_id"]),
                policy_snapshot_id=int(version["policy_snapshot_id"]),
                generation_profile_snapshot_id=generation_snapshot_id,
                policy_profile_id=str(version.get("policy_profile_id") or ""),
                generation_profile_id=str(version.get("generation_profile_id") or ""),
                generation_profile_key="default",
                generation_profile_snapshot_immutable=True,
                provider_class=_optional_string(
                    generation_definition.get("provider_class")
                    or generation_definition.get("deployment_class")
                    or generation_definition.get("provider")
                ),
                max_automatic_generation_batch_count=_optional_int(
                    generation_definition.get("automatic_generation_batch_cap")
                    or generation_definition.get("max_automatic_generation_batch")
                    or 1
                ),
                moderation_required=_optional_bool(
                    generation_definition.get("moderation_required", False)
                ),
                estimated_cost_class=_optional_string(
                    generation_definition.get("estimated_cost_class")
                ),
                supported_output_schemas=_supported_generation_output_schemas(
                    generation_definition
                ),
                dynamic_choice_support="choice_set" in generation_requirements["output_schemas"],
                scene_update_support="scene_update" in generation_requirements["output_schemas"],
                confirmation_required=(
                    generation_requirements["requires_confirmation"]
                    or bool(
                        generation_definition.get("requires_user_confirm")
                        or generation_definition.get("confirmation_required")
                    )
                ),
                content_rating=str(version.get("content_rating") or "general"),
                ready=ready,
                warning_summary=warning_summary,
                recommended=ready and compatible and not warning_summary.requires_acknowledgement,
            )
        )
    return _sort_script_version_options(options)


def _pack_for_script_version(
    asset_service: VNAssetPackService,
    pack_id: int,
) -> VNAssetPackResponse | None:
    try:
        return asset_service.get_pack(pack_id)
    except Exception as exc:
        logger.warning("Failed to load VN script setup pack {}: {}", pack_id, exc)
        return None


def _script_character_safety_status(
    db: CharactersRAGDB,
    pack: VNAssetPackResponse | None,
) -> str:
    if pack is None:
        return "missing"
    character = db.get_character_card_by_id(pack.primary_character_id)
    if not isinstance(character, Mapping):
        return "missing"
    return _character_safety_status(character)


def _policy_warnings(decision: Mapping[str, Any]) -> list[VNPlaySetupWarning]:
    warnings: list[VNPlaySetupWarning] = []
    reasons = decision.get("reasons")
    if not isinstance(reasons, list):
        return warnings
    for reason in reasons:
        if not isinstance(reason, Mapping):
            continue
        code = str(reason.get("code") or "policy_warning")
        severity = _policy_warning_severity(reason)
        warnings.append(
            VNPlaySetupWarning(
                code=code,
                severity=severity,
                message=str(reason.get("message") or WARNING_MESSAGES.get(code) or code),
                requires_acknowledgement=bool(reason.get("requires_acknowledgement")),
            )
        )
    return warnings


def _policy_warning_severity(reason: Mapping[str, Any]) -> VNPlaySetupWarningSeverity:
    raw_severity = str(reason.get("severity") or "").strip().lower()
    if raw_severity == "error":
        return "high_risk"
    if raw_severity == "warning":
        return "warning"
    return "info"


def _profile_definition(snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
    if snapshot is None:
        return {}
    definition = snapshot.get("definition")
    return dict(definition) if isinstance(definition, Mapping) else {}


def _script_generation_requirements(program: Any) -> dict[str, Any]:
    """Summarize generated-output requirements from a published script program."""
    output_schemas: set[str] = set()
    profile_keys: set[str] = set()
    schemas_by_profile: dict[str, set[str]] = {}
    requires_confirmation = False
    if not isinstance(program, Mapping):
        return {
            "output_schemas": output_schemas,
            "profile_keys": {"default"},
            "schemas_by_profile": {"default": set()},
            "requires_confirmation": False,
        }
    labels = program.get("labels")
    if not isinstance(labels, Mapping):
        return {
            "output_schemas": output_schemas,
            "profile_keys": {"default"},
            "schemas_by_profile": {"default": set()},
            "requires_confirmation": False,
        }
    for raw_ops in labels.values():
        if not isinstance(raw_ops, list):
            continue
        for opcode in raw_ops:
            if not isinstance(opcode, Mapping) or opcode.get("op") != "generate":
                continue
            profile_key = opcode.get("profile_key", "default")
            profile_key_text = str(profile_key) if isinstance(profile_key, str) else "default"
            profile_keys.add(profile_key_text)
            if opcode.get("requires_user_confirm") is True:
                requires_confirmation = True
            is_literal_generation = (
                isinstance(opcode.get("narrative_text"), str)
                or isinstance(opcode.get("regeneration_text"), str)
            )
            output_schema = opcode.get("output_schema")
            if output_schema is None and not is_literal_generation:
                output_schema = "narrative_dialogue"
            if isinstance(output_schema, str):
                output_schemas.add(output_schema)
                schemas_by_profile.setdefault(profile_key_text, set()).add(output_schema)
    return {
        "output_schemas": output_schemas,
        "profile_keys": profile_keys or {"default"},
        "schemas_by_profile": schemas_by_profile or {"default": set()},
        "requires_confirmation": requires_confirmation,
    }


def _generation_profile_warnings(
    *,
    profile_snapshots: VNProfileSnapshotRepository,
    owner_user_id: int,
    version: Mapping[str, Any],
    requirements: Mapping[str, Any],
) -> tuple[list[VNPlaySetupWarning], bool]:
    """Return generation profile readiness warnings for all generated script outputs."""
    warnings: list[VNPlaySetupWarning] = []
    blocked = False
    snapshot_ids = _generation_snapshot_ids(version)
    schemas_by_profile = requirements.get("schemas_by_profile", {})
    if not isinstance(schemas_by_profile, Mapping):
        schemas_by_profile = {}
    for profile_key in sorted(str(key) for key in requirements.get("profile_keys", {"default"})):
        snapshot_id = snapshot_ids.get(profile_key)
        if snapshot_id is None:
            warnings.append(_warning("generation_profile_snapshot_missing", "high_risk"))
            blocked = True
            continue
        snapshot = profile_snapshots.get_profile_snapshot(snapshot_id, owner_user_id=owner_user_id)
        if snapshot is None:
            warnings.append(_warning("generation_profile_snapshot_unavailable", "high_risk"))
            blocked = True
            continue
        definition = _profile_definition(snapshot)
        supported = set(_supported_generation_output_schemas(definition))
        required_schemas = {
            str(schema)
            for schema in (
                schemas_by_profile.get(profile_key, set())
                if isinstance(schemas_by_profile, Mapping)
                else set()
            )
        }
        if required_schemas and not required_schemas.issubset(supported):
            warnings.append(
                _warning("generation_profile_output_schema_unavailable", "high_risk")
            )
            blocked = True
    return warnings, blocked


def _generation_snapshot_ids(version: Mapping[str, Any]) -> dict[str, int]:
    snapshot_ids: dict[str, int] = {}
    default_snapshot_id = _optional_int(version.get("generation_profile_snapshot_id"))
    if default_snapshot_id is not None:
        snapshot_ids["default"] = default_snapshot_id
    raw_map = version.get("generation_profile_snapshots")
    if isinstance(raw_map, Mapping):
        for key, value in raw_map.items():
            try:
                snapshot_ids[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
    return snapshot_ids


def _supported_generation_output_schemas(definition: Mapping[str, Any]) -> list[str]:
    supported = definition.get("supported_output_schemas", definition.get("allowed_output_schemas"))
    if isinstance(supported, list):
        return sorted({str(schema) for schema in supported if str(schema).strip()})
    if bool(definition.get("supports_structured_output")):
        return ["choice_set", "narrative_dialogue", "scene_update"]
    return ["narrative_dialogue"]


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


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
    script_versions: list[VNPlaySetupScriptVersionOption],
) -> VNPlaySetupDefaults:
    """Choose conservative setup defaults from selected and recommended options."""
    default_character_id = selected_character.id if selected_character is not None else None
    if default_character_id is None and len(characters) == 1:
        default_character_id = characters[0].id

    recommended_pack_ids = [pack.id for pack in asset_packs if pack.recommended]
    default_pack_id = recommended_pack_ids[0] if len(recommended_pack_ids) == 1 else None
    default_script = _default_script_version(script_versions)
    if default_pack_id is None and default_script is not None:
        default_pack_id = default_script.asset_pack_id
    return VNPlaySetupDefaults(
        mode=mode,
        character_id=default_character_id,
        asset_pack_id=default_pack_id,
        script_id=default_script.script_id if default_script is not None else None,
        script_version_id=default_script.id if default_script is not None else None,
        policy_profile_id=default_script.policy_profile_id if default_script is not None else None,
        generation_profile_id=(
            default_script.generation_profile_id if default_script is not None else None
        ),
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
    script_versions: list[VNPlaySetupScriptVersionOption],
    mode: VNPlayMode | None,
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
        if mode == "scripted_story" and not script_versions:
            states.append(
                VNPlaySetupEmptyState(
                    code="no_script_versions",
                    scope="global",
                    message="No published VN scripts were found.",
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
    if mode == "scripted_story" and not script_versions:
        states.append(
            VNPlaySetupEmptyState(
                code="no_script_versions",
                scope="global",
                message="No published VN scripts were found.",
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


def _sort_script_version_options(
    script_versions: list[VNPlaySetupScriptVersionOption],
) -> list[VNPlaySetupScriptVersionOption]:
    """Sort script versions with ready and recommended options first."""
    def sort_key(indexed_script: tuple[int, VNPlaySetupScriptVersionOption]) -> tuple[int, int]:
        index, script_version = indexed_script
        if script_version.recommended:
            group = 0
        elif script_version.ready:
            group = 1
        else:
            group = 2
        return group, index

    return [
        script_version
        for _, script_version in sorted(enumerate(script_versions), key=sort_key)
    ]


def _default_script_version(
    script_versions: list[VNPlaySetupScriptVersionOption],
) -> VNPlaySetupScriptVersionOption | None:
    """Choose a default script without hiding required acknowledgement state."""
    recommended = [script for script in script_versions if script.recommended]
    if len(recommended) == 1:
        return recommended[0]
    ready = [script for script in script_versions if script.ready]
    if len(ready) == 1:
        return ready[0]
    return None


def _dedupe_warnings(
    warnings: list[VNPlaySetupWarning],
) -> list[VNPlaySetupWarning]:
    """Deduplicate warning codes while preserving first occurrence order."""
    seen: set[str] = set()
    deduped: list[VNPlaySetupWarning] = []
    for warning in warnings:
        if warning.code in seen:
            continue
        seen.add(warning.code)
        deduped.append(warning)
    return deduped


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
