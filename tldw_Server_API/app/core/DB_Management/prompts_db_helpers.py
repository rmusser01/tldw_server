"""Pure helper functions for prompt database record handling."""

import json
import re
import unicodedata
from contextlib import suppress
from typing import Any, Optional


def serialize_prompt_definition(prompt_definition: Any) -> Optional[str]:
    """Serialize structured prompt definition payloads for storage."""
    if prompt_definition is None:
        return None
    if isinstance(prompt_definition, str):
        return prompt_definition
    return json.dumps(prompt_definition, sort_keys=True)


def deserialize_prompt_record(prompt_data: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Hydrate a prompt row dict with a decoded ``prompt_definition`` field."""
    if not prompt_data:
        return prompt_data

    record = dict(prompt_data)
    record["prompt_format"] = record.get("prompt_format") or "legacy"

    prompt_definition_payload = record.pop("prompt_definition_json", None)
    if prompt_definition_payload is None:
        record["prompt_definition"] = None
        return record

    if isinstance(prompt_definition_payload, dict):
        record["prompt_definition"] = prompt_definition_payload
        return record

    if isinstance(prompt_definition_payload, str) and prompt_definition_payload.strip():
        try:
            record["prompt_definition"] = json.loads(prompt_definition_payload)
        except json.JSONDecodeError:
            record["prompt_definition"] = prompt_definition_payload
    else:
        record["prompt_definition"] = None
    return record


def build_structured_prompt_searchable_text(prompt_definition: Any) -> str:
    """Build searchable text from structured prompt variables and blocks."""
    if prompt_definition is None:
        return ""

    definition_payload = prompt_definition
    if isinstance(prompt_definition, str):
        with suppress(TypeError, ValueError, json.JSONDecodeError):
            definition_payload = json.loads(prompt_definition)

    if not isinstance(definition_payload, dict):
        return ""

    parts: list[str] = []

    variables = definition_payload.get("variables")
    if isinstance(variables, list):
        for variable in variables:
            if not isinstance(variable, dict):
                continue
            for key in ("name", "label", "description"):
                value = variable.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())

    blocks = definition_payload.get("blocks")
    if isinstance(blocks, list):
        for block in blocks:
            if not isinstance(block, dict) or block.get("enabled") is False:
                continue
            for key in ("name", "role", "content"):
                value = block.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())

    normalized_parts: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if part in seen:
            continue
        normalized_parts.append(part)
        seen.add(part)
    return "\n".join(normalized_parts)


def normalize_keyword(keyword: str) -> str:
    """Normalize keyword while preserving case for round-trip display/export."""
    normalized = keyword.strip()
    return re.sub(r"\s+", " ", normalized).strip()


def normalize_text_for_search(val: Any) -> str:
    """Normalize text for robust case-insensitive search comparisons."""
    normalized = "" if val is None else str(val)
    normalized = normalized.replace("İ", "I").replace("ı", "i")
    normalized = normalized.casefold()
    normalized = unicodedata.normalize("NFKD", normalized)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
