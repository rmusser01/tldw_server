"""Pure helper functions for prompt database record handling."""

import json
import re
import unicodedata
from collections.abc import Iterator, Mapping
from contextlib import suppress
from typing import Any, Optional

from pydantic import BaseModel, TypeAdapter, ValidationError

from tldw_Server_API.app.core.Prompt_Management.structured_prompts.legacy_renderer import render_legacy_snapshot
from tldw_Server_API.app.core.Prompt_Management.structured_prompts.models import (
    SINGLE_TEXT_RECIPE_LIMITS,
    PromptDefinition,
    PromptLegacySnapshot,
    SingleTextRecipeDefinitionV2,
    parse_prompt_definition,
)
from tldw_Server_API.app.core.Prompt_Management.structured_prompts.single_text_renderer import (
    render_single_text_recipe_template,
)
from tldw_Server_API.app.core.Prompt_Management.structured_prompts.validator import validate_prompt_definition

_RUNTIME_VALUE_KEYS = frozenset({"runtime_values", "variable_values", "resolved_values"})
_RECIPE_MARKER_KEYS = frozenset({"definition_kind", "assembly_mode", "target_role", "render_format", "section_key"})
_PROMPT_ENVELOPE_FIELDS = frozenset({"prompt_definition", "prompt_schema_version", "prompt_format"})
_DEFINITION_IDENTITY_KEYS = _RECIPE_MARKER_KEYS | {"schema_version", "assembly_config", "blocks"}
_LEGACY_SCHEMA_VERSION = TypeAdapter(int)
# Bound every save envelope/definition walk, including direct Python callers.
# Repeated containers are rejected (not skipped): JSON would expand aliases.
PROMPT_STRUCTURE_MAX_DEPTH = 32
PROMPT_STRUCTURE_MAX_NODES = 10000
LEGACY_PROMPT_TEXT_LIMIT = 20000


def validate_prompt_text_fields(system_prompt: Any, user_prompt: Any, *, is_recipe: bool = False) -> None:
    """Bound transport/storage text without including authored content in errors."""
    limit = SINGLE_TEXT_RECIPE_LIMITS["max_rendered_output_length"] if is_recipe else LEGACY_PROMPT_TEXT_LIMIT
    for value in (system_prompt, user_prompt):
        if value is not None and (not isinstance(value, str) or len(value) > limit):
            raise ValueError("invalid_prompt_text")


def render_v1_authored_snapshot(definition: PromptDefinition) -> PromptLegacySnapshot:
    """Keep the original authored v1 snapshot derivation shared with API saves."""
    messages = [
        {"role": block.role, "content": block.content}
        for block in sorted(definition.blocks, key=lambda item: item.order)
        if block.enabled
    ]
    return render_legacy_snapshot(messages, definition)


def _walk_prompt_containers(value: Any) -> Iterator[Any]:
    """Visit a JSON-shaped tree with bounded work; reject cycles and all aliases."""
    stack = [(iter((value,)), 0)]
    seen: set[int] = set()
    nodes = 0
    while stack:
        iterator, depth = stack[-1]
        try:
            item = next(iterator)
        except StopIteration:
            stack.pop()
            continue
        nodes += 1
        if nodes > PROMPT_STRUCTURE_MAX_NODES:
            raise ValueError("invalid_prompt_definition_structure")
        if isinstance(item, BaseModel):
            children = vars(item).values()
        elif isinstance(item, Mapping):
            children = item.values()
        elif isinstance(item, (list, tuple)):
            children = item
        elif item is None or isinstance(item, (str, int, float, bool)):
            continue
        else:
            raise ValueError("invalid_prompt_definition")
        if id(item) in seen or depth > PROMPT_STRUCTURE_MAX_DEPTH or len(children) > PROMPT_STRUCTURE_MAX_NODES - nodes:
            raise ValueError("invalid_prompt_definition_structure")
        seen.add(id(item))
        yield item
        stack.append((iter(children), depth + 1))


def reject_recipe_runtime_values(value: Any) -> None:
    """Reject structural ephemeral-value keys, never words inside authored text."""
    for item in _walk_prompt_containers(value):
        if isinstance(item, Mapping):
            if any(key in item for key in _RUNTIME_VALUE_KEYS):
                raise ValueError("invalid_recipe_runtime_values")


def has_recipe_markers(value: Any) -> bool:
    """Recognize recipe-only keys regardless of their values or missing identity."""
    return any(
        isinstance(item, Mapping) and any(key in item for key in _RECIPE_MARKER_KEYS | _PROMPT_ENVELOPE_FIELDS)
        for item in _walk_prompt_containers(value)
    )


def has_prompt_identity(value: Mapping) -> bool:
    """Recognize schema/transport identity without interpreting authored strings."""
    return any(key in value for key in _DEFINITION_IDENTITY_KEYS | {"prompt_definition", "prompt_schema_version"}) or (
        any(key in value and value[key] != "legacy" for key in ("format", "prompt_format"))
    )


def reject_malformed_prompt_envelope(value: Any) -> None:
    """Reject root/batch shape errors before framework validation can echo input."""
    if not isinstance(value, Mapping):
        raise ValueError("invalid_prompt_definition")
    if "prompts" in value and (
        not isinstance(value["prompts"], list) or any(not isinstance(item, Mapping) for item in value["prompts"])
    ):
        raise ValueError("invalid_prompt_definition")


def reject_misplaced_prompt_identity(value: Mapping, *, allow_preview_variables: bool = False) -> None:
    """Only prompt_definition may carry schema identity in a prompt envelope."""
    reject_malformed_prompt_envelope(value)
    allowed_fields = set(_PROMPT_ENVELOPE_FIELDS)
    if allow_preview_variables:
        allowed_fields.add("variables")
    outside_definition = {key: item for key, item in value.items() if key not in allowed_fields}
    for item in _walk_prompt_containers(outside_definition):
        if isinstance(item, Mapping) and has_prompt_identity(item):
            raise ValueError("invalid_prompt_definition")


def reject_legacy_prompt_identity(value: Any) -> None:
    """Fail closed on unsupported identity anywhere in a legacy-only envelope."""
    reject_malformed_prompt_envelope(value)
    for item in _walk_prompt_containers(value):
        if isinstance(item, Mapping) and has_prompt_identity(item):
            raise ValueError("structured_prompt_not_supported_on_legacy_route")


def parse_stored_prompt_definition(
    value: Any, *, schema_version: Any = None
) -> PromptDefinition | SingleTextRecipeDefinitionV2:
    """Parse/validate a storage definition without leaking recipe content in errors."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except RecursionError as error:
            raise ValueError("invalid_prompt_definition_structure") from error
        except (ValueError, TypeError) as error:
            raise ValueError("invalid_prompt_definition") from error
    reject_recipe_runtime_values(value)
    if isinstance(value, (PromptDefinition, SingleTextRecipeDefinitionV2)):
        try:
            value = value.model_dump()
        except (ValueError, TypeError, RecursionError) as error:
            raise ValueError("invalid_prompt_definition") from error
    if not isinstance(value, Mapping):
        raise ValueError("invalid_prompt_definition")
    reject_recipe_runtime_values(value)
    recipe_like = has_recipe_markers(value)
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
    # Non-v1 validation above has content-free codes (including future versions).
    # Resolve the outer identity before any value-rich v1 parse/semantic errors.
    if schema_version is not None and schema_version != version:
        raise ValueError("prompt_schema_version must match prompt_definition.schema_version.")
    try:
        definition = parse_prompt_definition(value)
    except ValidationError as error:
        if version == 1 and not recipe_like:
            raise ValueError(f"Invalid prompt_definition: {error}") from error
        raise ValueError("invalid_prompt_definition") from error
    issues = validate_prompt_definition(definition)
    if issues:
        raise ValueError(issues[0].message if version == 1 else issues[0].code)
    return definition


def prepare_recipe_storage_fields(
    prompt_format: str,
    schema_version: Any,
    definition: Any,
    *,
    system_prompt: Any = None,
    user_prompt: Any = None,
) -> dict[str, Any]:
    """Validate definitions at DB ingestion, compiling only v2 authored snapshots.

    V1 bytes and legacy fields remain caller-owned as before; v2 snapshots are
    always derived so a direct DB caller cannot persist a filled instance there.
    """
    if definition is None:
        validate_prompt_text_fields(system_prompt, user_prompt)
        if schema_version not in (None, 1):
            raise ValueError("invalid_prompt_definition")
        return {}
    parsed = parse_stored_prompt_definition(definition, schema_version=schema_version)
    validate_prompt_text_fields(system_prompt, user_prompt, is_recipe=True)
    if isinstance(parsed, PromptDefinition):
        if schema_version not in (None, 1):
            raise ValueError("prompt_schema_version must match prompt_definition.schema_version.")
        # API-authored v1 snapshots may exceed the old request-field bound.
        # Verify exact derivation rather than trusting a caller-provided flag.
        oversized = {
            key: value
            for key, value in {"system_prompt": system_prompt, "user_prompt": user_prompt}.items()
            if value is not None and len(value) > LEGACY_PROMPT_TEXT_LIMIT
        }
        if oversized:
            snapshot = render_v1_authored_snapshot(parsed)
            if any(value != getattr(snapshot, key) for key, value in oversized.items()):
                raise ValueError("invalid_prompt_text")
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
        try:
            reject_recipe_runtime_values(json.loads(prompt_definition))
        except json.JSONDecodeError:
            pass  # Preserve the existing standalone opaque-string serializer contract.
        except RecursionError as error:
            raise ValueError("invalid_prompt_definition_structure") from error
        return prompt_definition
    reject_recipe_runtime_values(prompt_definition)
    try:
        return json.dumps(prompt_definition, sort_keys=True)
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError("invalid_prompt_definition") from error


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
