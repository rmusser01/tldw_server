from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, cast

from tldw_Server_API.app.core.RPG.errors import RPGValidationError

RulesPackSourceType = Literal["media_item", "media_collection"]
RULES_PACK_SOURCE_TYPES = {"media_item", "media_collection"}


@dataclass(frozen=True, slots=True)
class RulesPackRef:
    ref_id: str
    source_type: RulesPackSourceType
    source_id: int
    display_name: str
    enabled: bool
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RulesPackRefReplacementResult:
    refs: list[RulesPackRef]
    version: int
    replayed: bool = False


def normalize_rules_pack_ref_payloads(
    payloads: list[dict[str, Any]],
    existing_refs: list[dict[str, Any]],
    now: datetime,
) -> list[RulesPackRef]:
    existing_by_ref_id = _existing_refs_by_ref_id(existing_refs)
    normalized: list[RulesPackRef] = []
    seen_ref_ids: set[str] = set()

    for payload in payloads:
        source_type = _normalize_source_type(payload.get("source_type"))
        source_id = _normalize_source_id(payload.get("source_id"))
        ref_id = f"{source_type}:{source_id}"
        if ref_id in seen_ref_ids:
            raise RPGValidationError("duplicate_rules_pack_ref")
        seen_ref_ids.add(ref_id)

        display_name = str(payload.get("display_name") or "").strip() or ref_id
        metadata = payload.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise RPGValidationError("invalid_rules_pack_ref_metadata")

        existing = existing_by_ref_id.get(ref_id)
        normalized.append(
            RulesPackRef(
                ref_id=ref_id,
                source_type=source_type,
                source_id=source_id,
                display_name=display_name,
                enabled=payload.get("enabled", True) is not False,
                created_at=existing.created_at if existing is not None else now,
                updated_at=now,
                metadata=dict(metadata),
            )
        )

    return normalized


def rules_pack_ref_to_dict(ref: RulesPackRef) -> dict[str, Any]:
    return {
        "ref_id": ref.ref_id,
        "source_type": ref.source_type,
        "source_id": ref.source_id,
        "display_name": ref.display_name,
        "enabled": ref.enabled,
        "created_at": _format_utc(ref.created_at),
        "updated_at": _format_utc(ref.updated_at),
        "metadata": dict(ref.metadata),
    }


def rules_pack_ref_from_dict(data: dict[str, Any]) -> RulesPackRef:
    source_type = data.get("source_type")
    source_id = data.get("source_id")
    if source_type is None or source_id is None:
        ref_id_value = data.get("ref_id")
        if isinstance(ref_id_value, str) and ":" in ref_id_value:
            source_type, raw_source_id = ref_id_value.split(":", 1)
            source_id = int(raw_source_id)

    source_type = _normalize_source_type(source_type)
    source_id = _normalize_source_id(source_id)
    ref_id = f"{source_type}:{source_id}"
    metadata = data.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise RPGValidationError("invalid_rules_pack_ref_metadata")

    return RulesPackRef(
        ref_id=ref_id,
        source_type=source_type,
        source_id=source_id,
        display_name=str(data.get("display_name") or "").strip() or ref_id,
        enabled=data.get("enabled", True) is not False,
        created_at=_parse_datetime(data.get("created_at")),
        updated_at=_parse_datetime(data.get("updated_at")),
        metadata=dict(metadata),
    )


def _existing_refs_by_ref_id(existing_refs: list[dict[str, Any]]) -> dict[str, RulesPackRef]:
    refs: dict[str, RulesPackRef] = {}
    for existing_ref in existing_refs:
        try:
            ref = rules_pack_ref_from_dict(existing_ref)
        except (RPGValidationError, TypeError, ValueError):
            continue
        refs[ref.ref_id] = ref
    return refs


def _normalize_source_type(source_type: Any) -> RulesPackSourceType:
    if source_type not in RULES_PACK_SOURCE_TYPES:
        raise RPGValidationError("invalid_rules_pack_ref_source_type")
    return cast(RulesPackSourceType, source_type)


def _normalize_source_id(source_id: Any) -> int:
    if isinstance(source_id, bool) or not isinstance(source_id, int) or source_id <= 0:
        raise RPGValidationError("invalid_rules_pack_ref_source_id")
    return source_id


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        parsed = datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    utc_value = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    return utc_value.isoformat().replace("+00:00", "Z")
