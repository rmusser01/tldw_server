import re
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from .models import PromptDefinition, SingleTextRecipeDefinitionV2, ValidationIssue, normalize_recipe_schema_version

SUPPORTED_SCHEMA_VERSION = 1
VALID_BLOCK_ROLES = {"system", "developer", "user", "assistant"}
_TEMPLATE_VARIABLE_PATTERN = re.compile(r"{{\s*([a-zA-Z0-9_]+)\s*}}")
_XML_SECTION_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, (PromptDefinition, SingleTextRecipeDefinitionV2)):
        return value.model_dump()
    if isinstance(value, Mapping):
        return value
    return {}


def validate_prompt_definition(
    definition: dict[str, Any] | PromptDefinition | SingleTextRecipeDefinitionV2,
) -> list[ValidationIssue]:
    """Return stable structural/semantic issues without modifying either schema."""
    payload = normalize_recipe_schema_version(_as_mapping(definition))
    issues: list[ValidationIssue] = []
    declared_variable_names: set[str] = set()

    version = payload.get("schema_version")
    if type(version) is not int or version not in (SUPPORTED_SCHEMA_VERSION, 2):
        issues.append(
            ValidationIssue(
                code="unsupported_schema_version",
                message=f"Unsupported schema version: {payload.get('schema_version')!r}",
                path="schema_version",
            )
        )
        return issues

    if version == 2:
        issues = _validate_recipe_structure(payload)
        if issues:
            return issues

    variables = payload.get("variables", [])
    if isinstance(variables, list):
        seen_variable_names: set[str] = set()
        for index, variable in enumerate(variables):
            if not isinstance(variable, Mapping):
                continue
            name = str(variable.get("name") or "").strip()
            if not name:
                continue
            if name in seen_variable_names:
                issues.append(
                    ValidationIssue(
                        code="duplicate_variable_name",
                        message=f"Duplicate variable name: {name}",
                        path=f"variables[{index}].name",
                    )
                )
                break
            seen_variable_names.add(name)
            declared_variable_names.add(name)

    blocks = payload.get("blocks", [])
    if isinstance(blocks, list):
        seen_block_ids: set[str] = set()
        for index, block in enumerate(blocks):
            if not isinstance(block, Mapping):
                continue

            block_id = str(block.get("id") or "").strip()
            if block_id:
                if block_id in seen_block_ids:
                    issues.append(
                        ValidationIssue(
                            code="duplicate_block_id",
                            message=f"Duplicate block id: {block_id}",
                            path=f"blocks[{index}].id",
                        )
                    )
                    break
                seen_block_ids.add(block_id)

            role = block.get("role")
            if role not in VALID_BLOCK_ROLES or (version == 2 and role != payload["assembly_config"]["target_role"]):
                issues.append(
                    ValidationIssue(
                        code="invalid_block_role",
                        message=f"Invalid block role: {role!r}",
                        path=f"blocks[{index}].role",
                    )
                )
                break

            if block.get("is_template") is True:
                content = str(block.get("content") or "")
                for match in _TEMPLATE_VARIABLE_PATTERN.finditer(content):
                    variable_name = match.group(1)
                    if variable_name not in declared_variable_names:
                        issues.append(
                            ValidationIssue(
                                code="unknown_variable_reference",
                                message=f"Unknown variable reference: {variable_name}",
                                path=f"blocks[{index}].content",
                            )
                        )
                        break
                if issues:
                    break

    if version == 2 and not issues:
        issues.extend(_validate_recipe_section_keys(payload))
    return issues


def _validate_recipe_structure(payload: Mapping[str, Any]) -> list[ValidationIssue]:
    """Validate exact recipe identity before bounded Pydantic field validation."""
    config = payload.get("assembly_config")
    config = config if isinstance(config, Mapping) else {}
    discriminants = [
        (payload, "definition_kind", {"single_text_recipe"}, "definition_kind"),
        (payload, "format", {"structured"}, "format"),
        (config, "assembly_mode", {"single_text"}, "assembly_config.assembly_mode"),
        (config, "target_role", {"system", "user"}, "assembly_config.target_role"),
        (config, "render_format", {"xml", "markdown", "freeform"}, "assembly_config.render_format"),
    ]
    for source, key, allowed, path in discriminants:
        value = source.get(key)
        if not isinstance(value, str) or value not in allowed:
            return [ValidationIssue(code=f"invalid_{key}", message=f"Invalid recipe {key}.", path=path)]
    try:
        SingleTextRecipeDefinitionV2.model_validate(payload)
    except ValidationError as error:
        issues = []
        for detail in error.errors(include_input=False, include_url=False):
            location = detail["loc"]
            path = "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in location).lstrip(".")
            code = detail["type"]
            if len(location) == 3 and location[0] == "blocks" and location[-1] == "role":
                code = "invalid_block_role"
            issues.append(ValidationIssue(code=code, message=detail["msg"], path=path))
        return issues
    return []


def _validate_recipe_section_keys(payload: Mapping[str, Any]) -> list[ValidationIssue]:
    """Require valid unique keys only for enabled XML-style sections."""
    if payload["assembly_config"]["render_format"] != "xml":
        return []
    seen_keys: set[str] = set()
    for index, block in enumerate(payload.get("blocks", [])):
        if not block.get("enabled", True):
            continue
        key = block.get("section_key")
        path = f"blocks[{index}].section_key"
        if not isinstance(key, str) or not _XML_SECTION_KEY_PATTERN.fullmatch(key):
            return [
                ValidationIssue(
                    code="invalid_section_key", message="Enabled XML blocks require a valid section key.", path=path
                )
            ]
        if key in seen_keys:
            return [
                ValidationIssue(
                    code="duplicate_section_key", message="Enabled XML section keys must be unique.", path=path
                )
            ]
        seen_keys.add(key)
    return []
