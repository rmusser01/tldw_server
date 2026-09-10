"""Schema-v2 recipe identity, bounds, and immutable validation contracts."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.core.Prompt_Management import structured_prompts
from tldw_Server_API.app.core.Prompt_Management.structured_prompts.validator import (
    validate_prompt_definition,
)


def valid_recipe() -> dict:
    """Return an independently editable recipe with all required discriminants."""
    return {
        "schema_version": 2,
        "format": "structured",
        "definition_kind": "single_text_recipe",
        "assembly_config": {
            "assembly_mode": "single_text",
            "target_role": "system",
            "render_format": "xml",
            "block_separator": "\n\n",
        },
        "variables": [{"name": "audience", "required": True, "default_value": ""}],
        "blocks": [
            {
                "id": "objective",
                "name": "Objective",
                "section_key": "objective",
                "role": "system",
                "content": "Explain {{audience}}",
                "order": 10,
                "enabled": True,
                "is_template": True,
            }
        ],
    }


@pytest.mark.parametrize("target", ["system", "user"])
@pytest.mark.parametrize("render_format", ["xml", "markdown", "freeform"])
def test_accepts_each_target_and_render_format(target: str, render_format: str) -> None:
    payload = valid_recipe()
    payload["assembly_config"].update(target_role=target, render_format=render_format)
    payload["blocks"][0]["role"] = target
    assert validate_prompt_definition(payload) == []


@pytest.mark.parametrize(
    "path,code",
    [
        (("schema_version",), "unsupported_schema_version"),
        (("format",), "invalid_format"),
        (("definition_kind",), "invalid_definition_kind"),
        (("assembly_config", "assembly_mode"), "invalid_assembly_mode"),
        (("assembly_config", "target_role"), "invalid_target_role"),
        (("assembly_config", "render_format"), "invalid_render_format"),
    ],
)
@pytest.mark.parametrize("missing", [True, False])
def test_requires_exact_discriminants(path: tuple, code: str, missing: bool) -> None:
    payload = valid_recipe()
    parent = payload if len(path) == 1 else payload[path[0]]
    if missing:
        parent.pop(path[-1])
    else:
        parent[path[-1]] = "unsupported"
    issues = validate_prompt_definition(payload)
    assert (issues[0].code, issues[0].path) == (code, ".".join(path))


@pytest.mark.parametrize("path", [(), ("assembly_config",), ("blocks", 0), ("variables", 0)])
def test_rejects_extra_fields_at_every_level(path: tuple) -> None:
    payload = valid_recipe()
    parent = payload
    for key in path:
        parent = parent[key]
    parent["runtime_values"] = {"audience": "private draft"}
    issues = validate_prompt_definition(payload)
    prefix = {
        (): "",
        ("assembly_config",): "assembly_config.",
        ("blocks", 0): "blocks[0].",
        ("variables", 0): "variables[0].",
    }[path]
    assert (issues[0].code, issues[0].path) == ("extra_forbidden", prefix + "runtime_values")


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("role", ["user", "assistant", "developer", "tool"])
def test_all_block_roles_must_match_target(role: str, enabled: bool) -> None:
    payload = valid_recipe()
    payload["blocks"][0].update(role=role, enabled=enabled)
    issues = validate_prompt_definition(payload)
    assert (issues[0].code, issues[0].path) == ("invalid_block_role", "blocks[0].role")


@pytest.mark.parametrize(
    "collection,code,path",
    [
        ("blocks", "duplicate_block_id", "blocks[1].id"),
        ("variables", "duplicate_variable_name", "variables[1].name"),
    ],
)
def test_rejects_duplicate_identifiers(collection: str, code: str, path: str) -> None:
    payload = valid_recipe()
    payload[collection].append(deepcopy(payload[collection][0]))
    issues = validate_prompt_definition(payload)
    assert (issues[0].code, issues[0].path) == (code, path)


@pytest.mark.parametrize("section_key", [None, "", "bad key", "1key", "a:b", "é", "a\n", "</x>"])
def test_enabled_xml_blocks_require_conservative_section_keys(section_key: str | None) -> None:
    payload = valid_recipe()
    payload["blocks"][0]["section_key"] = section_key
    issues = validate_prompt_definition(payload)
    assert (issues[0].code, issues[0].path) == ("invalid_section_key", "blocks[0].section_key")


@pytest.mark.parametrize("section_key", ["_", "A", "a.b-c_0"])
def test_accepts_conservative_xml_keys_without_rewriting(section_key: str) -> None:
    payload = valid_recipe()
    payload["blocks"][0]["section_key"] = section_key
    assert validate_prompt_definition(payload) == []
    assert payload["blocks"][0]["section_key"] == section_key


def test_enabled_xml_section_keys_are_unique() -> None:
    payload = valid_recipe()
    payload["blocks"].append({**payload["blocks"][0], "id": "other"})
    issues = validate_prompt_definition(payload)
    assert (issues[0].code, issues[0].path) == ("duplicate_section_key", "blocks[1].section_key")


@pytest.mark.parametrize("render_format,enabled", [("xml", False), ("markdown", True), ("freeform", True)])
def test_section_keys_are_optional_outside_enabled_xml(render_format: str, enabled: bool) -> None:
    payload = valid_recipe()
    payload["assembly_config"]["render_format"] = render_format
    payload["blocks"][0].pop("section_key")
    payload["blocks"][0]["enabled"] = enabled
    assert validate_prompt_definition(payload) == []


@pytest.mark.parametrize("collection", ["blocks", "variables"])
@pytest.mark.parametrize("count,valid", [(100, True), (101, False)])
def test_definition_collection_limits(collection: str, count: int, valid: bool) -> None:
    payload = valid_recipe()
    item = payload[collection][0]
    payload[collection] = [
        {
            **item,
            **(
                {"id": f"b{i}", "section_key": f"b{i}"}
                if collection == "blocks"
                else {"name": "audience" if i == 0 else f"v{i}"}
            ),
        }
        for i in range(count)
    ]
    issues = validate_prompt_definition(payload)
    if valid:
        assert issues == []
    else:
        assert (issues[0].code, issues[0].path) == ("too_long", collection)


@pytest.mark.parametrize(
    "path,limit",
    [
        (("blocks", 0, "id"), 128),
        (("blocks", 0, "name"), 200),
        (("blocks", 0, "section_key"), 128),
        (("blocks", 0, "content"), 20000),
        (("variables", 0, "name"), 128),
        (("variables", 0, "label"), 200),
        (("assembly_config", "block_separator"), 20),
    ],
)
@pytest.mark.parametrize("extra", [0, 1])
def test_string_limits_at_boundary(path: tuple, limit: int, extra: int) -> None:
    payload = valid_recipe()
    payload["blocks"][0]["is_template"] = False
    parent = payload
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = "x" * (limit + extra)
    issues = validate_prompt_definition(payload)
    if extra:
        expected_path = f"{path[0]}[0].{path[2]}" if len(path) == 3 else ".".join(path)
        assert (issues[0].code, issues[0].path) == ("string_too_long", expected_path)
    else:
        assert issues == []


@pytest.mark.parametrize("name", ["has space", " x", "x ", "x-y", "é", ""])
def test_variable_names_must_be_resolvable_by_template_grammar(name: str) -> None:
    payload = valid_recipe()
    payload["variables"][0]["name"] = name
    issues = validate_prompt_definition(payload)
    assert issues[0].path == "variables[0].name"


def test_rejects_unknown_template_variable() -> None:
    payload = valid_recipe()
    payload["variables"] = []
    issues = validate_prompt_definition(payload)
    assert (issues[0].code, issues[0].path) == ("unknown_variable_reference", "blocks[0].content")


@pytest.mark.parametrize("invalid", [True, False])
def test_validation_never_mutates_payload(invalid: bool) -> None:
    payload = valid_recipe()
    if invalid:
        payload["blocks"][0]["section_key"] = "not valid"
    original = deepcopy(payload)
    validate_prompt_definition(payload)
    assert payload == original


def test_parser_preserves_v2_identity_and_validates_model_inputs() -> None:
    payload = valid_recipe()
    parsed = structured_prompts.parse_prompt_definition(payload)
    assert isinstance(parsed, structured_prompts.SingleTextRecipeDefinitionV2)
    assert validate_prompt_definition(parsed) == []
    assert parsed.model_dump(exclude_unset=True) == payload


def test_v2_cannot_parse_as_v1_even_after_removing_recipe_fields() -> None:
    with pytest.raises(ValidationError):
        structured_prompts.PromptDefinition.model_validate({"schema_version": 2})


@pytest.mark.parametrize("version", [3, 99, "2", 2.0, True, None])
def test_parser_rejects_unknown_or_noninteger_schema_versions(version: object) -> None:
    payload = valid_recipe()
    payload["schema_version"] = version
    with pytest.raises(ValidationError):
        structured_prompts.parse_prompt_definition(payload)


def test_blank_recipe_is_valid() -> None:
    payload = valid_recipe()
    payload.update(blocks=[], variables=[])
    assert validate_prompt_definition(payload) == []


@pytest.mark.parametrize("field,value", [("enabled", "false"), ("is_template", "true"), ("order", "10")])
def test_recipe_does_not_coerce_fields_before_semantic_validation(field: str, value: str) -> None:
    payload = valid_recipe()
    payload["blocks"][0][field] = value
    issues = validate_prompt_definition(payload)
    assert len(issues) == 1
    assert issues[0].path == f"blocks[0].{field}"


def test_recipe_model_rejects_float_schema_version() -> None:
    payload = valid_recipe()
    payload["schema_version"] = 2.0
    with pytest.raises(ValidationError):
        structured_prompts.SingleTextRecipeDefinitionV2.model_validate(payload)
