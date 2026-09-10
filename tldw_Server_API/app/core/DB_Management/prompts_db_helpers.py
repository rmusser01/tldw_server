"""Pure helper functions for prompt database record handling."""

import json
import re
import unicodedata
from collections.abc import Mapping
from contextlib import suppress
from typing import Any, Optional

from pydantic import TypeAdapter, ValidationError

from tldw_Server_API.app.core.Prompt_Management.structured_prompts.models import (
    PromptDefinition,
    SingleTextRecipeDefinitionV2,
    parse_prompt_definition,
)
from tldw_Server_API.app.core.Prompt_Management.structured_prompts.single_text_renderer import (
    render_single_text_recipe_template,
)
from tldw_Server_API.app.core.Prompt_Management.structured_prompts.validator import validate_prompt_definition

_RUNTIME_VALUE_KEYS = frozenset({"runtime_values", "variable_values", "resolved_values"})
_LEGACY_SCHEMA_VERSION = TypeAdapter(int)


def reject_recipe_runtime_values(value: Any) -> None:
    """Reject structural ephemeral-value keys, never words inside authored text."""
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, Mapping):
            if _RUNTIME_VALUE_KEYS.intersection(item):
                raise ValueError("invalid_recipe_runtime_values")
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)


def parse_stored_prompt_definition(value: Any) -> PromptDefinition | SingleTextRecipeDefinitionV2:
    """Parse/validate a storage definition without leaking recipe content in errors."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError) as error:
            raise ValueError("invalid_prompt_definition") from error
    if isinstance(value, (PromptDefinition, SingleTextRecipeDefinitionV2)):
        value = value.model_dump()
    if not isinstance(value, Mapping):
        raise ValueError("invalid_prompt_definition")
    reject_recipe_runtime_values(value)
    # Preserve the original v1 int-field acceptance only when it selects v1.
    try:
        version = _LEGACY_SCHEMA_VERSION.validate_python(value.get("schema_version", 1))
    except ValidationError:
        version = None
    if version == 1:
        value = {**value, "schema_version": 1}
    else:
        issues = validate_prompt_definition(value)
        if issues:
            raise ValueError(issues[0].code)
    try:
        definition = parse_prompt_definition(value)
    except ValidationError as error:
        if version == 1 and "definition_kind" not in value:
            raise ValueError(f"Invalid prompt_definition: {error}") from error
        raise ValueError("invalid_prompt_definition") from error
    issues = validate_prompt_definition(definition)
    if issues:
        raise ValueError(issues[0].message if version == 1 else issues[0].code)
    return definition


def prepare_recipe_storage_fields(prompt_format: str, schema_version: Any, definition: Any) -> dict[str, Any]:
    """Validate definitions at DB ingestion, compiling only v2 authored snapshots.

    V1 bytes and legacy fields remain caller-owned as before; v2 snapshots are
    always derived so a direct DB caller cannot persist a filled instance there.
    """
    if definition is None:
        if schema_version not in (None, 1):
            raise ValueError("invalid_prompt_definition")
        return {}
    parsed = parse_stored_prompt_definition(definition)
    if isinstance(parsed, PromptDefinition):
        if schema_version not in (None, 1):
            raise ValueError("prompt_schema_version must match prompt_definition.schema_version.")
        return {}
    if prompt_format != "structured":
        raise ValueError("invalid_recipe_prompt_format")
    if schema_version != parsed.schema_version:
        raise ValueError("prompt_schema_version must match prompt_definition.schema_version.")
    snapshot = render_single_text_recipe_template(parsed).legacy
    return {
        "prompt_definition": parsed.model_dump(),
        "prompt_schema_version": parsed.schema_version,
        "system_prompt": snapshot.system_prompt,
        "user_prompt": snapshot.user_prompt,
    }


def serialize_prompt_definition(prompt_definition: Any) -> Optional[str]:
    """Serialize structured prompt definition payloads for storage."""
    if prompt_definition is None:
        return None
    if isinstance(prompt_definition, str):
        with suppress(json.JSONDecodeError):
            reject_recipe_runtime_values(json.loads(prompt_definition))
        return prompt_definition
    reject_recipe_runtime_values(prompt_definition)
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

    # Recipes are searched through their compiled authored legacy snapshot.
    # Do not add variable declarations or block labels to the v1 FTS detail text.
    if definition_payload.get("schema_version") == 2:
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
