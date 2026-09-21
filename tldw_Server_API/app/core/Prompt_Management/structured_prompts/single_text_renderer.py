"""Deterministic single-field rendering; runtime values are never persisted."""

import re
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel

from .legacy_renderer import render_single_text_legacy_snapshot
from .models import SINGLE_TEXT_RECIPE_LIMITS, PromptLegacySnapshot, SingleTextRecipeDefinitionV2
from .validator import validate_prompt_definition

_TEMPLATE_VARIABLE_PATTERN = re.compile(r"{{\s*([a-zA-Z0-9_]+)\s*}}")


class SingleTextRecipeRenderResult(BaseModel):
    """A tagged single string, intentionally without a messages field."""

    definition_kind: Literal["single_text_recipe"] = "single_text_recipe"
    rendered_text: str
    legacy: PromptLegacySnapshot


class SingleTextRecipeRenderError(ValueError):
    """Stable error codes without echoing authored or runtime content."""

    def __init__(self, code: str, *, variable_name: str | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.variable_name = variable_name


def render_single_text_recipe_template(definition: SingleTextRecipeDefinitionV2) -> SingleTextRecipeRenderResult:
    """Format authored template text for storage, never a filled runtime instance.

    Validate the original declarations/references first, then reuse the exact
    bounded formatter with substitution disabled on copies. Required variables
    need no defaults to save, and authored token whitespace remains untouched.
    """
    issues = validate_prompt_definition(definition)
    if issues:
        raise SingleTextRecipeRenderError(issues[0].code)
    template = definition.model_copy(
        update={
            "variables": [],
            "blocks": [block.model_copy(update={"is_template": False}) for block in definition.blocks],
        }
    )
    return render_single_text_recipe(template)


def render_single_text_recipe(
    definition: SingleTextRecipeDefinitionV2,
    runtime_values: Mapping[str, Any] | None = None,
) -> SingleTextRecipeRenderResult:
    """Render a validated recipe with string-only values and code-point bounds.

    A null/omitted default means absent; an explicit runtime null is invalid.
    Unknown runtime keys are ignored. Required declarations are resolved even
    when unused, matching v1. Substitution is single-pass in template content
    only, never labels, keys, separators, or substituted values.
    """
    issues = validate_prompt_definition(definition)
    if issues:
        raise SingleTextRecipeRenderError(issues[0].code)

    runtime_values = runtime_values or {}
    resolved: dict[str, str] = {}
    for variable in definition.variables:
        if variable.name in runtime_values:
            value = runtime_values[variable.name]
        elif variable.default_value is not None:
            value = variable.default_value
        elif variable.required:
            raise SingleTextRecipeRenderError("missing_required_variable", variable_name=variable.name)
        else:
            value = ""
        if not isinstance(value, str):
            raise SingleTextRecipeRenderError("invalid_variable_value", variable_name=variable.name)
        resolved[variable.name] = value

    config = definition.assembly_config
    rendered: list[str] = []
    output_length = 0
    # Python's stable sort preserves original indexes for equal orders.
    for block in sorted(definition.blocks, key=lambda block: block.order):
        if not block.enabled:
            continue
        content = block.content
        content_length = len(content)
        if block.is_template:
            content_length += sum(
                len(resolved[match.group(1)]) - len(match.group(0))
                for match in _TEMPLATE_VARIABLE_PATTERN.finditer(content)
            )
        prefix, suffix = "", ""
        if config.render_format == "xml":
            prefix, suffix = f"<{block.section_key}>", f"</{block.section_key}>"
        elif config.render_format == "markdown":
            prefix = f"## {block.name}\n\n"
        output_length += content_length + len(prefix) + len(suffix) + (len(config.block_separator) if rendered else 0)
        if output_length > SINGLE_TEXT_RECIPE_LIMITS["max_rendered_output_length"]:
            raise SingleTextRecipeRenderError("rendered_output_too_large")
        # Only allocate the substituted content after the exact expansion fits.
        if block.is_template:
            content = _TEMPLATE_VARIABLE_PATTERN.sub(lambda match: resolved[match.group(1)], content)
        if config.render_format == "xml" and suffix in content:
            raise SingleTextRecipeRenderError("closing_tag_collision")
        rendered.append(prefix + content + suffix)

    text = config.block_separator.join(rendered)
    return SingleTextRecipeRenderResult(
        rendered_text=text,
        legacy=render_single_text_legacy_snapshot(text, config.target_role),
    )
