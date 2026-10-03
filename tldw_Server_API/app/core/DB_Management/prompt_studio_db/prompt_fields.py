"""Normalise the stored prompt fields for a legacy or structured prompt record."""

from __future__ import annotations

from typing import Any, Optional

from tldw_Server_API.app.core.DB_Management.Prompts_DB import InputError
from tldw_Server_API.app.core.DB_Management.prompts_db_helpers import (
    parse_stored_prompt_definition,
    prepare_recipe_storage_fields,
)
from tldw_Server_API.app.core.Prompt_Management.structured_prompts import (
    PromptDefinition,
    render_legacy_snapshot,
)


def _render_definition_legacy_fields(definition: PromptDefinition) -> tuple[str, str]:
    messages = [
        {"role": block.role, "content": block.content}
        for block in sorted(definition.blocks, key=lambda item: item.order)
        if block.enabled
    ]
    legacy = render_legacy_snapshot(messages, definition)
    return legacy.system_prompt, legacy.user_prompt


def _prepare_prompt_record_fields(
    *,
    prompt_format: Optional[str],
    prompt_schema_version: Optional[int],
    prompt_definition: Optional[Any],
    system_prompt: Optional[str],
    user_prompt: Optional[str],
    current_prompt: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    effective_format = prompt_format or (
        current_prompt.get("prompt_format") if current_prompt else "legacy"
    ) or "legacy"

    if effective_format == "structured":
        effective_schema_version = prompt_schema_version
        if effective_schema_version is None and current_prompt:
            effective_schema_version = current_prompt.get("prompt_schema_version")
        effective_definition = prompt_definition
        if effective_definition is None and current_prompt:
            effective_definition = current_prompt.get("prompt_definition")

        if effective_schema_version is None:
            raise InputError("Structured prompts require prompt_schema_version.")  # noqa: TRY003
        if not isinstance(effective_definition, dict):
            raise InputError("Structured prompts require prompt_definition.")  # noqa: TRY003

        try:
            definition = parse_stored_prompt_definition(
                effective_definition,
                schema_version=effective_schema_version,
            )
        except ValueError as exc:
            raise InputError(str(exc)) from exc  # noqa: TRY003

        definition_schema_version = int(definition.schema_version)
        if int(effective_schema_version) != definition_schema_version:
            raise InputError(
                "prompt_schema_version must match prompt_definition.schema_version."
            )  # noqa: TRY003

        if not isinstance(definition, PromptDefinition):
            try:
                return {
                    "prompt_format": "structured",
                    **prepare_recipe_storage_fields(
                        "structured",
                        definition_schema_version,
                        definition,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                    ),
                }
            except ValueError as exc:
                raise InputError(str(exc)) from exc  # noqa: TRY003

        derived_system_prompt, derived_user_prompt = _render_definition_legacy_fields(definition)
        return {
            "prompt_format": "structured",
            "prompt_schema_version": definition_schema_version,
            "prompt_definition": definition.model_dump(),
            "system_prompt": derived_system_prompt,
            "user_prompt": derived_user_prompt,
        }

    if prompt_definition is not None:
        raise InputError("Legacy prompts cannot include prompt_definition.")  # noqa: TRY003
    if prompt_schema_version is not None:
        raise InputError("Legacy prompts cannot include prompt_schema_version.")  # noqa: TRY003

    return {
        "prompt_format": "legacy",
        "prompt_schema_version": None,
        "prompt_definition": None,
        "system_prompt": (
            system_prompt
            if system_prompt is not None
            else current_prompt.get("system_prompt") if current_prompt else None
        ),
        "user_prompt": (
            user_prompt
            if user_prompt is not None
            else current_prompt.get("user_prompt") if current_prompt else None
        ),
    }
