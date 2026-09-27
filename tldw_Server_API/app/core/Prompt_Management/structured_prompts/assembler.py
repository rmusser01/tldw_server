import json
import re
from collections.abc import Mapping
from typing import Any

from pydantic import TypeAdapter, ValidationError

from .legacy_renderer import render_legacy_snapshot
from .models import (
    PromptAssemblyResult,
    PromptDefinition,
    PromptVariableDefinition,
    SingleTextRecipeDefinitionV2,
    parse_prompt_definition,
)
from .single_text_renderer import (
    SingleTextRecipeRenderError,
    SingleTextRecipeRenderResult,
    render_single_text_recipe,
)
from .validator import validate_prompt_definition

_TEMPLATE_VARIABLE_PATTERN = re.compile(r"{{\s*([a-zA-Z0-9_]+)\s*}}")
_LEGACY_VERSION_ADAPTER = TypeAdapter(int)


class StructuredPromptAssemblyError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        variable_name: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.variable_name = variable_name


def assemble_prompt_definition(
    definition: dict[str, Any] | PromptDefinition | SingleTextRecipeDefinitionV2,
    variables: Mapping[str, Any] | None,
    *,
    extras: Mapping[str, Any] | None = None,
) -> PromptAssemblyResult | SingleTextRecipeRenderResult:
    prompt_definition = _coerce_definition(definition)
    if isinstance(prompt_definition, SingleTextRecipeDefinitionV2):
        try:
            return render_single_text_recipe(prompt_definition, variables)
        except SingleTextRecipeRenderError as error:
            raise StructuredPromptAssemblyError(error.code, str(error), variable_name=error.variable_name) from error

    validation_issues = validate_prompt_definition(prompt_definition)
    if validation_issues:
        first_issue = validation_issues[0]
        raise StructuredPromptAssemblyError(
            "invalid_prompt_definition",
            first_issue.message,
        )

    resolved_variables = _resolve_variables(prompt_definition, variables or {})
    rendered_messages = _render_block_messages(prompt_definition, resolved_variables)
    messages = _insert_fixed_sections(rendered_messages, extras or {})
    legacy = render_legacy_snapshot(messages, prompt_definition)
    return PromptAssemblyResult(messages=messages, legacy=legacy)


def _coerce_definition(
    definition: dict[str, Any] | PromptDefinition | SingleTextRecipeDefinitionV2,
) -> PromptDefinition | SingleTextRecipeDefinitionV2:
    if isinstance(definition, (PromptDefinition, SingleTextRecipeDefinitionV2)):
        return definition
    if "schema_version" not in definition:
        return PromptDefinition.model_validate(definition)
    # The original v1 model used an int field. Preserve its coercion only when
    # it selects version 1; v2/future identity still goes through the strict union.
    try:
        legacy_version = _LEGACY_VERSION_ADAPTER.validate_python(definition["schema_version"])
    except ValidationError:
        return parse_prompt_definition(definition)
    if legacy_version == 1:
        return PromptDefinition.model_validate({**definition, "schema_version": 1})
    return parse_prompt_definition(definition)


def _resolve_variables(
    definition: PromptDefinition,
    variables: Mapping[str, Any],
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for variable in definition.variables:
        resolved[variable.name] = _resolve_variable(variable, variables)
    return resolved


def _resolve_variable(
    variable: PromptVariableDefinition,
    variables: Mapping[str, Any],
) -> Any:
    if variable.name in variables:
        return variables[variable.name]
    if variable.default_value is not None:
        return variable.default_value
    if variable.required:
        raise StructuredPromptAssemblyError(
            "missing_required_variable",
            f"Missing required variable: {variable.name}",
            variable_name=variable.name,
        )
    return ""


def _render_block_messages(
    definition: PromptDefinition,
    resolved_variables: Mapping[str, Any],
) -> list[dict[str, str]]:
    ordered_blocks = sorted(definition.blocks, key=lambda block: block.order)
    rendered_messages: list[dict[str, str]] = []

    for block in ordered_blocks:
        if not block.enabled:
            continue
        rendered_messages.append(
            {
                "role": block.role,
                "content": _render_block_content(block.content, block.is_template, resolved_variables),
            }
        )

    return rendered_messages


def _render_block_content(
    content: str,
    is_template: bool,
    resolved_variables: Mapping[str, Any],
) -> str:
    if not is_template:
        return content

    def replace_variable(match: re.Match[str]) -> str:
        variable_name = match.group(1)
        if variable_name not in resolved_variables:
            raise StructuredPromptAssemblyError(
                "unknown_variable_reference",
                f"Unknown variable reference: {variable_name}",
                variable_name=variable_name,
            )
        value = resolved_variables[variable_name]
        return "" if value is None else str(value)

    return _TEMPLATE_VARIABLE_PATTERN.sub(replace_variable, content)


def _insert_fixed_sections(
    rendered_messages: list[dict[str, str]],
    extras: Mapping[str, Any],
) -> list[dict[str, str]]:
    inserted_messages = _render_module_messages(extras.get("modules_config"))
    inserted_messages.extend(_render_example_messages(extras.get("few_shot_examples")))

    if not inserted_messages:
        return list(rendered_messages)

    first_user_index = next(
        (index for index, message in enumerate(rendered_messages) if message["role"] == "user"),
        len(rendered_messages),
    )
    return rendered_messages[:first_user_index] + inserted_messages + rendered_messages[first_user_index:]


def _render_module_messages(modules_config: Any) -> list[dict[str, str]]:
    if not isinstance(modules_config, list):
        return []

    rendered_modules: list[dict[str, str]] = []
    for module in modules_config:
        if not isinstance(module, Mapping):
            continue
        if module.get("enabled") is False:
            continue

        module_type = str(module.get("type") or "unknown")
        config = module.get("config")
        if isinstance(config, Mapping) and config:
            config_text = ", ".join(f"{key}={_stringify_module_value(value)}" for key, value in sorted(config.items()))
            content = f"Module {module_type}: {config_text}"
        else:
            content = f"Module {module_type}"

        rendered_modules.append({"role": "developer", "content": content})

    return rendered_modules


def _stringify_module_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _render_example_messages(few_shot_examples: Any) -> list[dict[str, str]]:
    if not isinstance(few_shot_examples, list):
        return []

    rendered_examples: list[dict[str, str]] = []
    for example in few_shot_examples:
        if not isinstance(example, Mapping):
            continue

        example_inputs = example.get("inputs", example.get("input"))
        example_outputs = example.get("outputs", example.get("output"))

        if example_inputs is not None:
            rendered_examples.append(
                {
                    "role": "user",
                    "content": f"Example input: {_json_text(example_inputs)}",
                }
            )
        if example_outputs is not None:
            rendered_examples.append(
                {
                    "role": "assistant",
                    "content": f"Example output: {_json_text(example_outputs)}",
                }
            )

    return rendered_examples


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True)
