from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tldw_Server_API.app.core.VN_Scripts.authoring_catalog import list_authoring_catalog
from tldw_Server_API.app.core.VN_Scripts.validator import (
    forbidden_generation_routing_keys,
    known_script_ops,
    supported_generation_output_schemas,
)


def _operation_ids(catalog: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(operation["op"] for operation in catalog["operations"])


def _assert_object_schemas_are_closed(schema: Mapping[str, Any], path: str = "$") -> None:
    if schema.get("type") == "object":
        assert schema.get("additionalProperties") is False, path
    for key in ("properties", "patternProperties", "$defs", "definitions"):
        children = schema.get(key)
        if isinstance(children, Mapping):
            for child_name, child_schema in children.items():
                if isinstance(child_schema, Mapping):
                    _assert_object_schemas_are_closed(child_schema, f"{path}.{key}.{child_name}")
    items = schema.get("items")
    if isinstance(items, Mapping):
        _assert_object_schemas_are_closed(items, f"{path}.items")
    for key in ("oneOf", "anyOf", "allOf"):
        variants = schema.get(key)
        if isinstance(variants, list):
            for index, variant in enumerate(variants):
                if isinstance(variant, Mapping):
                    _assert_object_schemas_are_closed(variant, f"{path}.{key}[{index}]")


def _forbidden_keys_present(value: Any) -> set[str]:
    forbidden = set(forbidden_generation_routing_keys()) | {
        "raw_prompt",
        "prompt",
        "provider_config",
        "validation_codes",
    }
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in forbidden:
                found.add(str(key))
            found.update(_forbidden_keys_present(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_forbidden_keys_present(child))
    return found


def test_catalog_operations_exactly_match_validator_known_ops() -> None:
    catalog = list_authoring_catalog()

    assert catalog["schema_version"] == "vn_script_authoring_catalog.v1"
    assert catalog["program_schema_version"] == "vn_script_program.v1"
    assert _operation_ids(catalog) == known_script_ops()


def test_catalog_is_preview_safe_and_includes_canonical_capability_tokens() -> None:
    catalog = list_authoring_catalog()

    assert catalog["capability_tokens"] == [
        "script_authoring_catalog",
        "scripted_generation",
        "scripted_generation.output_schema.choice_set",
        "scripted_generation.output_schema.scene_update",
        "scripted_generation.user_confirmation",
    ]
    assert catalog["generation_output_schemas"] == list(supported_generation_output_schemas())
    assert set(catalog["operation_categories"]) == {"story", "branching", "visuals", "audio", "generation", "state"}
    assert _forbidden_keys_present(catalog) == set()


def test_catalog_includes_required_v1_snippets_with_parameters_schema() -> None:
    catalog = list_authoring_catalog()

    snippet_ids = {snippet["id"] for snippet in catalog["snippets"]}

    assert {
        "narration",
        "dialogue",
        "authored_choice",
        "generated_choice_set",
        "scene_update_generation",
        "confirm_gated_generation",
        "set_background",
        "show_sprite",
        "play_bgm",
        "set_variable",
        "ending",
    }.issubset(snippet_ids)
    for snippet in catalog["snippets"]:
        assert "parameters_schema" in snippet
        assert "parameter_schema" not in snippet


def test_generated_choice_snippet_exposes_handler_label_not_opcode_target() -> None:
    catalog = list_authoring_catalog()
    snippet = next(snippet for snippet in catalog["snippets"] if snippet["id"] == "generated_choice_set")

    schema = snippet["parameters_schema"]
    default_parameters = snippet["default_parameters"]

    assert schema["required"] == ["handler_label"]
    assert "handler_label" in schema["properties"]
    assert "on_generated_choice" not in schema["properties"]
    assert "handler_label" in default_parameters
    assert "on_generated_choice" not in default_parameters
    assert "handler_label" in str(snippet["preview"])
    assert "on_generated_choice" not in str(snippet["preview"])


def test_catalog_json_contains_no_validation_codes_or_routing_secrets() -> None:
    catalog = list_authoring_catalog()

    serialized = str(catalog).lower()
    assert "validation_codes" not in serialized
    assert "api_key" not in serialized
    assert "provider_config" not in serialized
    assert "raw prompt" not in serialized
    assert "raw_prompt" not in serialized
    for key in forbidden_generation_routing_keys():
        assert key not in _forbidden_keys_present(catalog)


def test_snippet_parameter_schemas_forbid_extra_object_fields_recursively() -> None:
    catalog = list_authoring_catalog()

    assert catalog["snippets"]
    for snippet in catalog["snippets"]:
        assert snippet["schema_version"] == "vn_script_program.v1"
        _assert_object_schemas_are_closed(snippet["parameters_schema"], f"$.snippets.{snippet['id']}.parameters_schema")


def test_catalog_payloads_are_isolated_between_calls() -> None:
    first = list_authoring_catalog()
    first["operations"][0]["label"] = "mutated"
    first["snippets"][0]["parameters_schema"]["properties"]["unexpected"] = {"type": "string"}
    first["limits"]["max_title_length"] = 1

    second = list_authoring_catalog()

    assert second["operations"][0]["label"] != "mutated"
    assert "unexpected" not in second["snippets"][0]["parameters_schema"]["properties"]
    assert second["limits"]["max_title_length"] != 1
