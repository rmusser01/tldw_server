"""Validate and execute owner-materialized dictionary input transformations."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from typing import Any

from tldw_Server_API.app.core.Character_Chat import chat_dictionary
from tldw_Server_API.app.core.Character_Chat.character_behavior_snapshot import DEFAULT_MAX_SNAPSHOT_BYTES
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError

DICTIONARY_SETTING_KEYS = frozenset({"chat_dictionary_ids", "chat_dictionary_id"})
MAX_FROZEN_DICTIONARIES = 64
MAX_FROZEN_DICTIONARY_ENTRIES = 10_000
_ENTRY_KEYS = frozenset({"id", "dictionary_id", "key", "content", "is_regex", "probability", "max_replacements",
                         "group_name", "timed_effects", "enabled", "case_sensitive", "created_at"})


def _positive_id(value: Any) -> int:
    if isinstance(value, str) and value.strip().isdigit():
        try:
            value = int(value.strip())
        except ValueError as exc:
            raise InputError("Assigned dictionary IDs must be positive integers.") from exc
    if type(value) is not int or value <= 0:
        raise InputError("Assigned dictionary IDs must be positive integers.")
    return value


def assigned_dictionary_ids(settings: Mapping[str, Any]) -> list[int]:
    """Resolve the canonical list, legacy singleton, and supported context writer."""
    if not isinstance(settings, Mapping):
        raise InputError("Assigned dictionary settings must be an object.")
    context = settings.get("conversationContext")
    if isinstance(context, Mapping) and "chat_dictionary_ids" in context:
        raw = context["chat_dictionary_ids"]
    elif "chat_dictionary_ids" in settings:
        raw = settings["chat_dictionary_ids"]
    elif settings.get("chat_dictionary_id") is not None:
        raw = [settings["chat_dictionary_id"]]
    else:
        return []
    if not isinstance(raw, list) or len(raw) > MAX_FROZEN_DICTIONARIES:
        raise InputError("Assigned dictionary IDs require a bounded list.")
    return list(dict.fromkeys(_positive_id(value) for value in raw))


def without_dictionary_assignment(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only representable assignment controls for legacy behavior preflight."""
    if not isinstance(settings, Mapping):
        raise InputError("Assigned dictionary controls must be an object.")
    result = deepcopy(dict(settings))
    for key in DICTIONARY_SETTING_KEYS:
        result.pop(key, None)
    context = result.get("conversationContext")
    if isinstance(context, dict):
        context.pop("chat_dictionary_ids", None)
        if context.get("world_book_ids") == []:
            context.pop("world_book_ids")
        if not context:
            result.pop("conversationContext")
    return result


def validate_frozen_dictionary_plan(value: Any, root_ids: list[int]) -> dict[str, Any]:
    """Refuse malformed executable state instead of defaulting or reading live data."""
    if not isinstance(value, dict) or set(value) not in ({"root_ids", "passes"}, {"root_ids", "passes", "catalog_owner"}):
        raise InputError("Invalid frozen dictionary plan.")
    if "catalog_owner" in value and (
        not isinstance(value["catalog_owner"], str) or not value["catalog_owner"]
        or value["catalog_owner"] != value["catalog_owner"].strip()
    ):
        raise InputError("Invalid frozen dictionary catalog owner.")
    if (not isinstance(root_ids, list) or len(root_ids) > MAX_FROZEN_DICTIONARIES
            or any(type(item) is not int or item <= 0 for item in root_ids) or len(root_ids) != len(set(root_ids))
            or value["root_ids"] != root_ids or any(type(item) is not int for item in value["root_ids"])
            or not isinstance(value["passes"], list) or len(value["passes"]) != len(root_ids)):
        raise InputError("Frozen dictionary assignment does not match saved IDs.")
    entry_count = 0
    for root_id, step in zip(root_ids, value["passes"]):
        if (not isinstance(step, dict) or set(step) != {"root_id", "sources", "token_budget", "entries"}
                or type(step["root_id"]) is not int or step["root_id"] != root_id):
            raise InputError("Invalid frozen dictionary pass.")
        budget = step["token_budget"]
        if budget is not None and (type(budget) is not int or budget <= 0):
            raise InputError("Invalid frozen dictionary budget.")
        sources = step["sources"]
        if not isinstance(sources, list) or len(sources) > MAX_FROZEN_DICTIONARIES:
            raise InputError("Invalid frozen dictionary sources.")
        source_ids = set()
        for source in sources:
            if (not isinstance(source, dict) or set(source) != {"id", "version"}
                    or type(source["id"]) is not int or type(source["version"]) is not int
                    or source["id"] <= 0 or source["version"] <= 0 or source["id"] in source_ids):
                raise InputError("Invalid frozen dictionary source binding.")
            source_ids.add(source["id"])
        if sources and sources[-1]["id"] != root_id:
            raise InputError("Frozen dictionary include order has no assigned root.")
        entries = step["entries"]
        if not isinstance(entries, list):
            raise InputError("Invalid frozen dictionary entries.")
        entry_count += len(entries)
        if entry_count > MAX_FROZEN_DICTIONARY_ENTRIES:
            raise InputError("Too many frozen dictionary entries.")
        entry_ids = set()
        for entry in entries:
            if (not isinstance(entry, dict) or set(entry) != _ENTRY_KEYS
                    or type(entry["id"]) is not int or entry["id"] <= 0 or entry["id"] in entry_ids
                    or type(entry["dictionary_id"]) is not int or entry["dictionary_id"] not in source_ids
                    or not isinstance(entry["key"], str) or not entry["key"] or not isinstance(entry["content"], str)
                    or type(entry["max_replacements"]) is not int or entry["max_replacements"] < 0
                    or any(type(entry[key]) is not bool for key in ("is_regex", "enabled", "case_sensitive"))
                    or type(entry["probability"]) is not float or not math.isfinite(entry["probability"])
                    or not 0 <= entry["probability"] <= 1
                    or entry["group_name"] is not None and not isinstance(entry["group_name"], str)):
                raise InputError("Invalid frozen dictionary entry.")
            effects = entry["timed_effects"]
            if (not isinstance(effects, dict) or set(effects) - {"sticky", "cooldown", "delay"}
                    or any(type(item) is not int or item < 0 for item in effects.values())):
                raise InputError("Unsupported frozen dictionary timed effects.")
            try:
                datetime.fromisoformat(entry["created_at"].replace("Z", "+00:00"))
                compiled = chat_dictionary.ChatDictionaryEntry.from_dict(entry)
            except (AttributeError, TypeError, ValueError, chat_dictionary.re.error) as exc:
                raise InputError("Invalid frozen dictionary entry configuration.") from exc
            if entry["is_regex"] and not compiled.is_regex:
                raise InputError("Invalid or unsafe frozen dictionary regex.")
            entry_ids.add(entry["id"])
    encoded = json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > DEFAULT_MAX_SNAPSHOT_BYTES:
        raise InputError("Frozen dictionary plan exceeds the snapshot byte limit.")
    return deepcopy(value)


def transform_frozen_user_content(messages: list[dict[str, Any]], plan: dict[str, Any]) -> list[Any]:
    """Transform only new user text; return detached content with images preserved."""
    plan = validate_frozen_dictionary_plan(plan, plan["root_ids"])

    def process(text: str) -> str:
        for step in plan["passes"]:
            entries = [chat_dictionary.ChatDictionaryEntry.from_dict(entry) for entry in step["entries"] if entry["enabled"]]
            text, stats = chat_dictionary.process_dictionary_entries(
                text, entries, max_iterations=5, token_budget=step["token_budget"], count_tokens=chat_dictionary._count_tokens,
            )
            if stats["token_budget_exceeded"]:
                break
        return text

    result = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = deepcopy(message.get("content"))
        if isinstance(content, str):
            content = process(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    raise InputError("Dictionary content parts must be objects.")
                if part.get("type") == "text":
                    if not isinstance(part.get("text"), str):
                        raise InputError("Dictionary text parts require text.")
                    part["text"] = process(part["text"])
        else:
            raise InputError("Dictionary input requires user text content.")
        result.append(content)
    return result
