"""First-run MCP tool pack catalog and policy generation helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

CATALOG_VERSION = "2026-07-04.v1"
CONFIRMATION_VERSION = "2026-07-04.v1"
SETUP_ORIGIN = "first_run_mcp_tools"
PROFILE_DISPLAY_NAME = "First-run default"
VALIDATION_STATES = {
    "not_run",
    "built_in_passed",
    "external_discovered",
    "external_tool_passed",
    "no_safe_external_tool",
    "external_discovery_incomplete",
    "failed",
    "skipped",
}

_PACKS: tuple[dict[str, Any], ...] = (
    {
        "pack_id": "research",
        "label": "Research",
        "purpose": "Search and inspect saved knowledge, media, prompts, and MCP catalogs.",
        "default_selected": True,
        "module_targets": ["knowledge", "media", "prompts", "mcp_discovery"],
        "tool_patterns": [
            "knowledge.search",
            "knowledge.get",
            "media.search",
            "media.get",
            "prompts.search",
            "prompts.get",
            "mcp.catalogs.list",
            "mcp.modules.list",
            "mcp.tools.list",
        ],
        "sample_validation_candidates": ["mcp.tools.list"],
    },
    {
        "pack_id": "learning",
        "label": "Learning",
        "purpose": "Review quizzes, flashcards, and media without changing study data.",
        "default_selected": True,
        "module_targets": ["quizzes", "flashcards", "media"],
        "tool_patterns": [
            "quizzes.list",
            "quizzes.get",
            "quizzes.questions.list",
            "quizzes.attempts.list",
            "quizzes.attempts.get",
            "flashcards.decks.list",
            "flashcards.decks.get",
            "flashcards.list",
            "flashcards.get",
            "flashcards.tags.get",
            "media.search",
            "media.get",
        ],
        "sample_validation_candidates": [],
    },
    {
        "pack_id": "writing",
        "label": "Writing",
        "purpose": "Search prompts and notes without creating or editing content.",
        "default_selected": True,
        "module_targets": ["prompts", "notes"],
        "tool_patterns": [
            "prompts.search",
            "prompts.get",
            "notes.search",
            "notes.get",
            "notes.tags.list",
            "notes.tasks.list",
            "notes.tasks.get",
        ],
        "sample_validation_candidates": [],
    },
    {
        "pack_id": "media_library",
        "label": "Media Library",
        "purpose": "Search and inspect media records.",
        "default_selected": True,
        "module_targets": ["media"],
        "tool_patterns": [
            "media.search",
            "media.get",
        ],
        "sample_validation_candidates": [],
    },
    {
        "pack_id": "personal_knowledge",
        "label": "Personal Knowledge",
        "purpose": "Search saved notes, prompts, and knowledge records.",
        "default_selected": True,
        "module_targets": ["notes", "prompts", "knowledge"],
        "tool_patterns": [
            "notes.search",
            "notes.get",
            "notes.tags.list",
            "prompts.search",
            "prompts.get",
            "knowledge.search",
            "knowledge.get",
        ],
        "sample_validation_candidates": [],
    },
)

_ADD_ONS: tuple[dict[str, Any], ...] = (
    {
        "addon_id": "external_network_read",
        "label": "External network reads",
        "default_selected": False,
        "requirement": "Explicit opt-in; only read-only tools may be added.",
        "strong_confirmation": False,
    },
    {
        "addon_id": "local_file_read",
        "label": "Local file reads",
        "default_selected": False,
        "requirement": "Explicit opt-in; server-local path semantics apply.",
        "strong_confirmation": False,
    },
    {
        "addon_id": "workspace_write",
        "label": "Writes and updates",
        "default_selected": False,
        "requirement": "Strong confirmation; generated policy must enumerate writable tools.",
        "strong_confirmation": True,
    },
    {
        "addon_id": "destructive_actions",
        "label": "Delete/destructive actions",
        "default_selected": False,
        "requirement": "Strong confirmation; never implied by pack selection.",
        "strong_confirmation": True,
    },
    {
        "addon_id": "process_run_command",
        "label": "Process or command execution",
        "default_selected": False,
        "requirement": "Strong confirmation; never implied by pack selection.",
        "strong_confirmation": True,
    },
)
_PACKS_BY_ID = {pack["pack_id"]: pack for pack in _PACKS}
_STRONG_ADD_ON_IDS = {
    addon["addon_id"] for addon in _ADD_ONS if addon["strong_confirmation"]
}
_SAFE_MCP_DISCOVERY_TOOLS = {
    "mcp.catalogs.list",
    "mcp.modules.list",
    "mcp.tools.list",
}
_READ_FORBIDDEN_CAPABILITIES = {
    "filesystem.write",
    "filesystem.delete",
    "process.execute",
}


def build_mcp_tools_catalog(
    *,
    tool_entries: Sequence[Mapping[str, Any]],
    selected_pack_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return the static first-run MCP pack catalog with current availability."""

    entries_by_name = _entries_by_name(tool_entries)
    registry_available = bool(entries_by_name)
    packs = [
        _catalog_pack(pack, entries_by_name, registry_available=registry_available)
        for pack in _PACKS
    ]
    for pack_id in _unknown_pack_ids(selected_pack_ids or []):
        packs.append(
            {
                "pack_id": pack_id,
                "label": f"Legacy choice: {pack_id}",
                "purpose": "Unavailable pack from an older first-run catalog.",
                "default_selected": False,
                "available": False,
                "legacy": True,
                "module_targets": [],
                "tool_patterns": [],
                "available_tools": [],
                "unavailable_tools": [],
                "add_on_ids": [],
                "sample_validation_candidates": [],
                "catalog_version": CATALOG_VERSION,
            }
        )

    return {
        "catalog_version": CATALOG_VERSION,
        "confirmation_version": CONFIRMATION_VERSION,
        "packs": packs,
        "add_ons": [dict(addon) for addon in _ADD_ONS],
        "validation_states": sorted(VALIDATION_STATES),
    }


def generate_first_run_policy(
    *,
    selected_pack_ids: Sequence[str],
    selected_addon_ids: Sequence[str],
    confirmed_addon_ids: Sequence[str],
    confirmation_version: str | None,
    setup_instance_id: str,
    tool_entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Generate a strict first-run MCP Hub policy document."""

    selected_pack_ids = _unique_strings(selected_pack_ids)
    selected_addon_ids = _unique_strings(selected_addon_ids)
    confirmed_addon_ids = _unique_strings(confirmed_addon_ids)
    _validate_strong_addons(
        selected_addon_ids=selected_addon_ids,
        confirmed_addon_ids=confirmed_addon_ids,
        confirmation_version=confirmation_version,
    )

    entries_by_name = _entries_by_name(tool_entries)
    registry_available = bool(entries_by_name)
    allowed_tools: list[str] = []
    capabilities: list[str] = []

    for pack_id in selected_pack_ids:
        pack = _PACKS_BY_ID.get(pack_id)
        if not pack:
            continue
        for tool_name in pack["tool_patterns"]:
            entry = entries_by_name.get(tool_name)
            if registry_available and not _is_safe_default_read_tool(entry):
                continue
            allowed_tools.append(tool_name)

    if "external_network_read" in selected_addon_ids:
        capabilities.append("network.external")
        allowed_tools.extend(
            entry["tool_name"]
            for entry in entries_by_name.values()
            if _is_safe_external_read_tool(entry)
        )

    if "local_file_read" in selected_addon_ids:
        local_file_tools = [
            entry["tool_name"]
            for entry in entries_by_name.values()
            if _is_safe_local_file_read_tool(entry)
        ]
        if local_file_tools:
            capabilities.append("filesystem.read")
            allowed_tools.extend(local_file_tools)

    if _strong_addon_enabled(
        "workspace_write",
        selected_addon_ids=selected_addon_ids,
        confirmed_addon_ids=confirmed_addon_ids,
    ):
        allowed_tools.extend(
            entry["tool_name"]
            for entry in entries_by_name.values()
            if _is_workspace_write_tool(entry)
        )

    if _strong_addon_enabled(
        "destructive_actions",
        selected_addon_ids=selected_addon_ids,
        confirmed_addon_ids=confirmed_addon_ids,
    ):
        destructive_tools = [
            entry
            for entry in entries_by_name.values()
            if _is_destructive_tool(entry)
        ]
        allowed_tools.extend(entry["tool_name"] for entry in destructive_tools)
        if any(_as_bool(entry.get("uses_filesystem")) for entry in destructive_tools):
            capabilities.append("filesystem.delete")

    if _strong_addon_enabled(
        "process_run_command",
        selected_addon_ids=selected_addon_ids,
        confirmed_addon_ids=confirmed_addon_ids,
    ):
        process_tools = [
            entry
            for entry in entries_by_name.values()
            if _is_process_tool(entry)
        ]
        allowed_tools.extend(entry["tool_name"] for entry in process_tools)
        if process_tools:
            capabilities.append("process.execute")

    allowed_tools = _unique_strings(allowed_tools)
    capabilities = _unique_strings(capabilities)
    policy_hash = _generated_policy_hash(
        allowed_tools=allowed_tools,
        capabilities=capabilities,
        selected_pack_ids=[pack_id for pack_id in selected_pack_ids if pack_id in _PACKS_BY_ID],
        selected_addon_ids=selected_addon_ids,
    )
    return {
        "allowed_tools": allowed_tools,
        "capabilities": capabilities,
        "first_run_mcp_tools": {
            "setup_origin": SETUP_ORIGIN,
            "setup_instance_id": setup_instance_id,
            "catalog_version": CATALOG_VERSION,
            "confirmation_version": confirmation_version,
            "selected_pack_ids": selected_pack_ids,
            "legacy_pack_ids": _unknown_pack_ids(selected_pack_ids),
            "selected_addon_ids": selected_addon_ids,
            "generated_policy_hash": policy_hash,
            "last_generated_hash": policy_hash,
        },
    }


def _catalog_pack(
    pack: Mapping[str, Any],
    entries_by_name: Mapping[str, Mapping[str, Any]],
    *,
    registry_available: bool,
) -> dict[str, Any]:
    tools = list(pack["tool_patterns"])
    available_tools: list[dict[str, Any]] = []
    unavailable_tools: list[dict[str, Any]] = []
    for tool_name in tools:
        entry = entries_by_name.get(tool_name)
        if registry_available and not _is_safe_default_read_tool(entry):
            unavailable_tools.append({"tool_name": tool_name, "available": False})
            continue
        available_tools.append({"tool_name": tool_name, "available": True})

    return {
        **dict(pack),
        "available": True,
        "legacy": False,
        "add_on_ids": [addon["addon_id"] for addon in _ADD_ONS],
        "available_tools": available_tools,
        "unavailable_tools": unavailable_tools,
        "catalog_version": CATALOG_VERSION,
    }


def _entries_by_name(
    tool_entries: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {
        str(entry.get("tool_name") or "").strip(): entry
        for entry in tool_entries
        if str(entry.get("tool_name") or "").strip()
    }


def _unknown_pack_ids(pack_ids: Sequence[str]) -> list[str]:
    return [pack_id for pack_id in _unique_strings(pack_ids) if pack_id not in _PACKS_BY_ID]


def _unique_strings(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _validate_strong_addons(
    *,
    selected_addon_ids: Sequence[str],
    confirmed_addon_ids: Sequence[str],
    confirmation_version: str | None,
) -> None:
    selected_strong_addons = _STRONG_ADD_ON_IDS.intersection(selected_addon_ids)
    if not selected_strong_addons:
        return
    confirmed = set(confirmed_addon_ids)
    missing = selected_strong_addons.difference(confirmed)
    if missing or confirmation_version != CONFIRMATION_VERSION:
        raise ValueError(
            "Strong first-run MCP add-on requires current confirmation: "
            + ", ".join(sorted(selected_strong_addons))
        )


def _strong_addon_enabled(
    addon_id: str,
    *,
    selected_addon_ids: Sequence[str],
    confirmed_addon_ids: Sequence[str],
) -> bool:
    return addon_id in selected_addon_ids and addon_id in confirmed_addon_ids


def _is_safe_default_read_tool(entry: Mapping[str, Any] | None) -> bool:
    if entry is None:
        return False
    if str(entry.get("tool_name") or "").strip() in _SAFE_MCP_DISCOVERY_TOOLS:
        return _is_non_mutating_internal_tool(entry)
    return (
        _is_low_risk_read_only(entry)
        and not _uses_external_network(entry)
        and not _as_bool(entry.get("uses_filesystem"))
        and not _is_process_tool(entry)
    )


def _is_safe_external_read_tool(entry: Mapping[str, Any]) -> bool:
    return (
        _is_low_risk_read_only(entry)
        and _uses_external_network(entry)
        and not _as_bool(entry.get("uses_filesystem"))
        and not _is_process_tool(entry)
    )


def _is_safe_local_file_read_tool(entry: Mapping[str, Any]) -> bool:
    return (
        _is_low_risk_read_only(entry)
        and _as_bool(entry.get("uses_filesystem"))
        and _as_bool(entry.get("path_boundable"))
        and not _uses_external_network(entry)
        and not _is_process_tool(entry)
    )


def _is_workspace_write_tool(entry: Mapping[str, Any]) -> bool:
    return (
        _as_bool(entry.get("mutates_state"))
        and not _is_destructive_tool(entry)
        and not _as_bool(entry.get("uses_filesystem"))
        and not _uses_external_network(entry)
        and not _is_process_tool(entry)
    )


def _is_destructive_tool(entry: Mapping[str, Any]) -> bool:
    tool_name = str(entry.get("tool_name") or "").lower()
    capabilities = set(_str_list(entry.get("capabilities")))
    return (
        _as_bool(entry.get("destructive"))
        or "filesystem.delete" in capabilities
        or ".delete" in tool_name
        or tool_name.endswith("_delete")
        or ".remove" in tool_name
        or tool_name.endswith("_remove")
    )


def _is_process_tool(entry: Mapping[str, Any]) -> bool:
    capabilities = set(_str_list(entry.get("capabilities")))
    return _as_bool(entry.get("uses_processes")) or "process.execute" in capabilities


def _uses_external_network(entry: Mapping[str, Any]) -> bool:
    capabilities = set(_str_list(entry.get("capabilities")))
    return (
        _as_bool(entry.get("uses_network"))
        or "network.external" in capabilities
        or "external.network" in capabilities
    )


def _is_low_risk_read_only(entry: Mapping[str, Any]) -> bool:
    return (
        str(entry.get("risk_class") or "").strip().lower() == "low"
        and not _as_bool(entry.get("mutates_state"))
        and not _is_destructive_tool(entry)
        and not _has_any_capability(entry, _READ_FORBIDDEN_CAPABILITIES)
    )


def _is_non_mutating_internal_tool(entry: Mapping[str, Any]) -> bool:
    return (
        not _as_bool(entry.get("mutates_state"))
        and not _is_destructive_tool(entry)
        and not _has_any_capability(entry, _READ_FORBIDDEN_CAPABILITIES)
        and not _uses_external_network(entry)
        and not _as_bool(entry.get("uses_filesystem"))
        and not _is_process_tool(entry)
    )


def _has_any_capability(entry: Mapping[str, Any], candidates: set[str]) -> bool:
    return bool(set(_str_list(entry.get("capabilities"))).intersection(candidates))


def _str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item).strip() for item in value if str(item or "").strip()]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return False


def _generated_policy_hash(
    *,
    allowed_tools: Sequence[str],
    capabilities: Sequence[str],
    selected_pack_ids: Sequence[str],
    selected_addon_ids: Sequence[str],
) -> str:
    payload = {
        "allowed_tools": _unique_strings(allowed_tools),
        "capabilities": _unique_strings(capabilities),
        "selected_pack_ids": _unique_strings(selected_pack_ids),
        "selected_addon_ids": _unique_strings(selected_addon_ids),
        "catalog_version": CATALOG_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
