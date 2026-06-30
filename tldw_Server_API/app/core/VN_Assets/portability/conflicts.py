"""Conflict and identity helpers for VN pack import previews."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def slot_identity(slot: Mapping[str, Any]) -> str:
    """Return the v1 update identity for a VN asset slot."""
    return f"{slot.get('asset_type')}:{slot.get('slot_key')}"


def source_item_fingerprint(item: Mapping[str, Any]) -> str | None:
    """Return an explicit source item fingerprint when the bundle provides one."""
    for key in ("source_item_fingerprint", "item_fingerprint"):
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def item_identity_candidates(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return ordered v1 item identity candidates from strongest to weakest."""
    candidates: list[dict[str, Any]] = []
    explicit_fingerprint = source_item_fingerprint(item)
    if explicit_fingerprint:
        candidates.append(
            {
                "kind": "source_item_fingerprint",
                "value": explicit_fingerprint,
                "confidence": "strong",
            }
        )
    checksum = item.get("asset_sha256") or item.get("checksum")
    if checksum:
        candidates.append(
            {
                "kind": "slot_checksum",
                "slot_identity": slot_identity(item),
                "value": str(checksum),
                "confidence": "strong",
            }
        )
    if item.get("variant_index") not in (None, ""):
        candidates.append(
            {
                "kind": "slot_variant_index",
                "slot_identity": slot_identity(item),
                "value": int(item["variant_index"]),
                "confidence": "ambiguous",
            }
        )
    return candidates


def detect_conflicts(
    *,
    repo: Any | None,
    owner_user_id: int,
    manifest: Mapping[str, Any],
    pack: Mapping[str, Any],
    slots: list[Mapping[str, Any]],
    items: list[Mapping[str, Any]],
    character: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Detect deterministic import conflicts without choosing destructive actions."""
    conflicts: list[dict[str, Any]] = []
    conflicts.extend(_pack_conflicts(repo=repo, owner_user_id=owner_user_id, manifest=manifest, pack=pack))
    conflicts.extend(_character_conflicts(repo=repo, character=character))

    slot_identities = [slot_identity(slot) for slot in slots]
    duplicate_slots = sorted(_duplicates(slot_identities))
    for identity in duplicate_slots:
        conflicts.append(
            {
                "conflict_id": _stable_id("slot", "duplicate", identity),
                "kind": "slot_duplicate",
                "severity": "blocking",
                "identity": identity,
                "message": "Duplicate slot identity in archive.",
                "allowed_actions": ["fail_import"],
            }
        )

    item_candidate_counts: dict[str, int] = {}
    for item in items:
        for candidate in item_identity_candidates(item):
            key = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
            item_candidate_counts[key] = item_candidate_counts.get(key, 0) + 1
    for key, count in sorted(item_candidate_counts.items()):
        if count > 1:
            candidate = json.loads(key)
            conflicts.append(
                {
                    "conflict_id": _stable_id("item", "duplicate-candidate", key),
                    "kind": "item_duplicate_identity",
                    "severity": "blocking",
                    "identity": candidate,
                    "count": count,
                    "message": "Multiple imported items share the same identity candidate.",
                    "allowed_actions": ["fail_import", "manual_resolve"],
                }
            )

    return sorted(conflicts, key=lambda conflict: str(conflict["conflict_id"]))


def build_update_existing_plan(
    *,
    repo: Any | None,
    owner_user_id: int,
    manifest: Mapping[str, Any],
    pack: Mapping[str, Any],
    slots: list[Mapping[str, Any]],
    items: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build non-destructive update candidates for local packs."""
    if repo is None:
        return {"allowed": False, "candidate_packs": []}
    candidates = [
        _build_update_candidate(repo=repo, target_pack=target_pack, slots=slots, items=items)
        for target_pack in _candidate_update_packs(
            repo=repo,
            owner_user_id=owner_user_id,
            manifest=manifest,
            pack=pack,
        )
    ]
    return {
        "allowed": bool(candidates),
        "candidate_packs": sorted(candidates, key=lambda candidate: int(candidate["target_pack_id"])),
    }


def _pack_conflicts(
    *,
    repo: Any | None,
    owner_user_id: int,
    manifest: Mapping[str, Any],
    pack: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if repo is None:
        return []
    list_packs = getattr(repo, "list_packs", None)
    if not callable(list_packs):
        return []
    imported_title = _normalized_text(pack.get("title") or manifest.get("pack_title"))
    source_fingerprint = manifest.get("canonical_payload_fingerprint")
    conflicts: list[dict[str, Any]] = []
    for local_pack in list_packs(owner_user_id=owner_user_id):
        signals: list[str] = []
        if imported_title and _normalized_text(local_pack.get("title")) == imported_title:
            signals.append("pack_title")
        if not signals:
            continue
        conflicts.append(
            {
                "conflict_id": _stable_id("pack", local_pack.get("id"), imported_title, source_fingerprint),
                "kind": "pack_candidate",
                "severity": "review",
                "local_pack_id": int(local_pack["id"]),
                "signals": signals,
                "allowed_actions": ["create_new", "update_existing", "fail_on_conflict"],
            }
        )
    return conflicts


def _candidate_update_packs(
    *,
    repo: Any,
    owner_user_id: int,
    manifest: Mapping[str, Any],
    pack: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    list_packs = getattr(repo, "list_packs", None)
    if not callable(list_packs):
        return []
    imported_title = _normalized_text(pack.get("title") or manifest.get("pack_title"))
    if not imported_title:
        return []
    return [
        local_pack
        for local_pack in list_packs(owner_user_id=owner_user_id)
        if _normalized_text(local_pack.get("title")) == imported_title
    ]


def _build_update_candidate(
    *,
    repo: Any,
    target_pack: Mapping[str, Any],
    slots: list[Mapping[str, Any]],
    items: list[Mapping[str, Any]],
) -> dict[str, Any]:
    target_pack_id = int(target_pack["id"])
    local_slots = repo.list_slots(target_pack_id)
    local_slots_by_identity = {slot_identity(slot): slot for slot in local_slots}
    imported_slot_to_local: dict[int, Mapping[str, Any]] = {}
    matched_slots: list[dict[str, Any]] = []
    added_slots: list[dict[str, Any]] = []
    diffs: list[dict[str, Any]] = []

    for slot in slots:
        identity = slot_identity(slot)
        local_slot = local_slots_by_identity.get(identity)
        source_slot_id = _int_or_none(slot.get("source_slot_id"))
        if local_slot is None:
            added_slots.append({"source_slot_id": source_slot_id, "identity": identity})
            continue
        if source_slot_id is not None:
            imported_slot_to_local[source_slot_id] = local_slot
        matched_slots.append(
            {
                "source_slot_id": source_slot_id,
                "local_slot_id": int(local_slot["id"]),
                "identity": identity,
            }
        )
        slot_diff = _slot_metadata_diff(target_pack_id=target_pack_id, imported=slot, local=local_slot)
        if slot_diff is not None:
            diffs.append(slot_diff)

    item_plan = _plan_items(
        repo=repo,
        target_pack_id=target_pack_id,
        items=items,
        imported_slot_to_local=imported_slot_to_local,
        local_slots_by_identity=local_slots_by_identity,
    )
    diffs.extend(item_plan["diffs"])
    return {
        "target_pack_id": target_pack_id,
        "matched_slots": matched_slots,
        "added_slots": added_slots,
        "matched_items": item_plan["matched_items"],
        "added_items": item_plan["added_items"],
        "diffs": sorted(diffs, key=lambda diff: str(diff["diff_id"])),
        "requires_confirmation": any(bool(diff.get("requires_confirmation")) for diff in diffs),
        "blocked": any(str(diff.get("severity")) == "blocking" for diff in diffs),
    }


def _slot_metadata_diff(
    *,
    target_pack_id: int,
    imported: Mapping[str, Any],
    local: Mapping[str, Any],
) -> dict[str, Any] | None:
    changed_fields: list[str] = []
    comparisons = {
        "labels": (_normal_mapping(imported.get("labels")), _loads_json(local.get("labels_json"), {})),
        "prompt_template": (imported.get("prompt_template"), local.get("prompt_template")),
        "negative_prompt_template": (
            imported.get("negative_prompt_template"),
            local.get("negative_prompt_template"),
        ),
        "variant_count": (_int_or_none(imported.get("variant_count")), _int_or_none(local.get("variant_count"))),
        "width": (_int_or_none(imported.get("width")), _int_or_none(local.get("width"))),
        "height": (_int_or_none(imported.get("height")), _int_or_none(local.get("height"))),
        "required_for_runtime": (
            bool(imported.get("required_for_runtime", True)),
            bool(local.get("required_for_runtime", True)),
        ),
    }
    for field_name, (imported_value, local_value) in comparisons.items():
        if imported_value != local_value:
            changed_fields.append(field_name)
    if not changed_fields:
        return None
    source_slot_id = _int_or_none(imported.get("source_slot_id"))
    return {
        "diff_id": _stable_id("slot_metadata_diff", target_pack_id, slot_identity(imported), changed_fields),
        "kind": "slot_metadata_diff",
        "severity": "review",
        "requires_confirmation": True,
        "source_slot_id": source_slot_id,
        "local_slot_id": int(local["id"]),
        "identity": slot_identity(imported),
        "fields": changed_fields,
    }


def _plan_items(
    *,
    repo: Any,
    target_pack_id: int,
    items: list[Mapping[str, Any]],
    imported_slot_to_local: Mapping[int, Mapping[str, Any]],
    local_slots_by_identity: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    local_items = repo.list_items(target_pack_id)
    local_items_by_slot = _local_items_by_slot_identity(local_items, local_slots_by_identity)
    fingerprint_index = _index_local_items_by_fingerprint(local_items)
    checksum_index = _index_local_items_by_checksum(local_items, local_slots_by_identity)
    variant_index = _index_local_items_by_variant(local_items, local_slots_by_identity)
    matched_items: list[dict[str, Any]] = []
    added_items: list[dict[str, Any]] = []
    diffs: list[dict[str, Any]] = []

    for item in items:
        source_item_id = _int_or_none(item.get("source_item_id"))
        slot = _local_slot_for_imported_item(
            item=item,
            imported_slot_to_local=imported_slot_to_local,
            local_slots_by_identity=local_slots_by_identity,
        )
        if slot is None:
            added_items.append(
                {
                    "source_item_id": source_item_id,
                    "source_slot_id": _int_or_none(item.get("source_slot_id")),
                }
            )
            continue
        slot_key = slot_identity(item)
        match = _match_item(
            item=item,
            slot_key=slot_key,
            fingerprint_index=fingerprint_index,
            checksum_index=checksum_index,
            variant_index=variant_index,
        )
        if match["kind"] == "matched":
            local_item = match["local_item"]
            matched_items.append(
                {
                    "source_item_id": source_item_id,
                    "local_item_id": int(local_item["id"]),
                    "source_slot_id": _int_or_none(item.get("source_slot_id")),
                    "local_slot_id": int(slot["id"]),
                    "match_kind": match["match_kind"],
                }
            )
            if item.get("asset_bytes_status") == "missing" and _local_item_has_bytes(local_item):
                diffs.append(
                    {
                        "diff_id": _stable_id("item_missing_bytes_skipped", target_pack_id, source_item_id),
                        "kind": "item_missing_bytes_skipped",
                        "severity": "info",
                        "requires_confirmation": False,
                        "source_item_id": source_item_id,
                        "local_item_id": int(local_item["id"]),
                        "message": "Imported missing-byte item will not replace local byte-backed item.",
                    }
                )
            continue
        if match["kind"] == "duplicate":
            diffs.append(
                {
                    "diff_id": _stable_id("item_duplicate_match", target_pack_id, source_item_id, match["match_kind"]),
                    "kind": "item_duplicate_match",
                    "severity": "blocking",
                    "requires_confirmation": True,
                    "source_item_id": source_item_id,
                    "match_kind": match["match_kind"],
                    "local_item_ids": [int(local_item["id"]) for local_item in match["local_items"]],
                }
            )
            continue
        if match["kind"] == "ambiguous":
            local_matches = local_items_by_slot.get(slot_key, [])
            diffs.append(
                {
                    "diff_id": _stable_id("item_variant_index_ambiguous", target_pack_id, source_item_id),
                    "kind": "item_variant_index_ambiguous",
                    "severity": "review",
                    "requires_confirmation": True,
                    "source_item_id": source_item_id,
                    "source_slot_id": _int_or_none(item.get("source_slot_id")),
                    "variant_index": _int_or_none(item.get("variant_index")),
                    "candidate_local_item_ids": [int(local_item["id"]) for local_item in local_matches],
                }
            )
            continue
        added_items.append(
            {
                "source_item_id": source_item_id,
                "source_slot_id": _int_or_none(item.get("source_slot_id")),
            }
        )

    return {"matched_items": matched_items, "added_items": added_items, "diffs": diffs}


def _match_item(
    *,
    item: Mapping[str, Any],
    slot_key: str,
    fingerprint_index: Mapping[str, list[Mapping[str, Any]]],
    checksum_index: Mapping[tuple[str, str], list[Mapping[str, Any]]],
    variant_index: Mapping[tuple[str, int], list[Mapping[str, Any]]],
) -> dict[str, Any]:
    fingerprint = source_item_fingerprint(item)
    if fingerprint:
        match = _match_index(fingerprint_index.get(fingerprint, []), "source_item_fingerprint")
        if match["kind"] != "none":
            return match
    checksum = item.get("asset_sha256") or item.get("checksum")
    if checksum:
        match = _match_index(checksum_index.get((slot_key, str(checksum)), []), "slot_checksum")
        if match["kind"] != "none":
            return match
    if item.get("variant_index") not in (None, ""):
        matches = variant_index.get((slot_key, int(item["variant_index"])), [])
        if matches:
            return {"kind": "ambiguous", "local_items": list(matches), "match_kind": "slot_variant_index"}
    return {"kind": "none"}


def _match_index(matches: list[Mapping[str, Any]], match_kind: str) -> dict[str, Any]:
    if len(matches) == 1:
        return {"kind": "matched", "local_item": matches[0], "match_kind": match_kind}
    if len(matches) > 1:
        return {"kind": "duplicate", "local_items": list(matches), "match_kind": match_kind}
    return {"kind": "none"}


def _local_slot_for_imported_item(
    *,
    item: Mapping[str, Any],
    imported_slot_to_local: Mapping[int, Mapping[str, Any]],
    local_slots_by_identity: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    if item.get("source_slot_id") not in (None, ""):
        local_slot = imported_slot_to_local.get(int(item["source_slot_id"]))
        if local_slot is not None:
            return local_slot
    return local_slots_by_identity.get(slot_identity(item))


def _local_items_by_slot_identity(
    local_items: list[Mapping[str, Any]],
    local_slots_by_identity: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    slot_identity_by_id = {int(slot["id"]): identity for identity, slot in local_slots_by_identity.items()}
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for item in local_items:
        identity = slot_identity_by_id.get(int(item["slot_id"]))
        if identity is not None:
            grouped.setdefault(identity, []).append(item)
    return grouped


def _index_local_items_by_fingerprint(local_items: list[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    index: dict[str, list[Mapping[str, Any]]] = {}
    for item in local_items:
        for fingerprint in _local_item_fingerprints(item):
            index.setdefault(fingerprint, []).append(item)
    return index


def _index_local_items_by_checksum(
    local_items: list[Mapping[str, Any]],
    local_slots_by_identity: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    slot_identity_by_id = {int(slot["id"]): identity for identity, slot in local_slots_by_identity.items()}
    index: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for item in local_items:
        identity = slot_identity_by_id.get(int(item["slot_id"]))
        if identity is None:
            continue
        for checksum in _local_item_checksums(item):
            index.setdefault((identity, checksum), []).append(item)
    return index


def _index_local_items_by_variant(
    local_items: list[Mapping[str, Any]],
    local_slots_by_identity: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, int], list[Mapping[str, Any]]]:
    slot_identity_by_id = {int(slot["id"]): identity for identity, slot in local_slots_by_identity.items()}
    index: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for item in local_items:
        identity = slot_identity_by_id.get(int(item["slot_id"]))
        if identity is None or item.get("variant_index") in (None, ""):
            continue
        index.setdefault((identity, int(item["variant_index"])), []).append(item)
    return index


def _character_conflicts(*, repo: Any | None, character: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if repo is None or character is None:
        return []
    character_name = _normalized_text(character.get("name"))
    if not character_name:
        return []
    db = getattr(repo, "db", None)
    get_by_name = getattr(db, "get_character_card_by_name", None)
    if not callable(get_by_name):
        return []
    local_character = get_by_name(str(character.get("name") or ""))
    if not local_character:
        return []
    return [
        {
            "conflict_id": _stable_id("character", character_name, local_character.get("id")),
            "kind": "character_candidate",
            "severity": "review",
            "signals": ["normalized_name"],
            "local_character_id": int(local_character["id"]),
            "source_character_name": character.get("name"),
            "allowed_actions": ["import_included_character", "link_existing_character"],
        }
    ]


def _local_item_fingerprints(item: Mapping[str, Any]) -> list[str]:
    return _local_item_values(
        item,
        keys=("source_item_fingerprint", "item_fingerprint"),
    )


def _local_item_checksums(item: Mapping[str, Any]) -> list[str]:
    return _local_item_values(
        item,
        keys=("source_asset_sha256", "asset_sha256", "checksum"),
    )


def _local_item_values(item: Mapping[str, Any], *, keys: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for field_name in ("source_context_snapshot_json", "backend_metadata_json"):
        payload = _loads_json(item.get(field_name), {})
        for value in _values_from_payload(payload, keys=keys):
            if value not in values:
                values.append(value)
    return values


def _values_from_payload(payload: Mapping[str, Any], *, keys: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            values.append(str(value))
    nested = payload.get("vnpack_import")
    if isinstance(nested, Mapping):
        for key in keys:
            value = nested.get(key)
            if value not in (None, ""):
                values.append(str(value))
    return values


def _local_item_has_bytes(item: Mapping[str, Any]) -> bool:
    return bool(item.get("generated_file_id") or item.get("storage_ref"))


def _duplicates(values: list[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _normal_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _loads_json(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _stable_id(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
