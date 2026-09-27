"""Shared, hand-derived rendering fixtures: no production helper builds expectations."""

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.core.Prompt_Management.structured_prompts import assembler, single_text_renderer
from tldw_Server_API.app.core.Prompt_Management.structured_prompts.models import parse_prompt_definition

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[3] / "Docs/fixtures/single-text-recipes"
RENDER_CASES = json.loads((FIXTURES / "render-cases.json").read_text())
ERROR_CASES = json.loads((FIXTURES / "error-cases.json").read_text())


@pytest.mark.parametrize(
    ("format_name", "expected"),
    [
        ("xml", "<task>  {{ topic }}  </task>"),
        ("markdown", "## Task\n\n  {{ topic }}  "),
        ("freeform", "  {{ topic }}  "),
    ],
)
def test_authored_template_snapshot_preserves_tokens_without_resolving_required_values(format_name, expected):
    payload = {
        "schema_version": 2,
        "format": "structured",
        "definition_kind": "single_text_recipe",
        "assembly_config": {"assembly_mode": "single_text", "target_role": "user", "render_format": format_name},
        "variables": [{"name": "topic", "required": True}],
        "blocks": [
            {
                "id": "task",
                "name": "Task",
                "section_key": "task",
                "role": "user",
                "content": "  {{ topic }}  ",
                "order": 0,
                "is_template": True,
            }
        ],
    }
    definition = parse_prompt_definition(payload)
    before = definition.model_dump()
    render = getattr(single_text_renderer, "render_single_text_recipe_template", None)
    assert callable(render), "Persistence needs an authored-template rendering path"
    result = render(definition)
    assert result.rendered_text == expected
    assert result.legacy.model_dump() == {"system_prompt": "", "user_prompt": expected}
    assert definition.model_dump() == before


def runtime_values(case):
    """Expand compact repeat descriptors without using renderer logic."""
    values = copy.deepcopy(case.get("runtimeValues", {}))
    for name, repeat in case.get("runtime_repeats", {}).items():
        values[name] = repeat["text"] * repeat["count"]
    return values


def expected_result(case):
    """Expand independently specified output strings and their target snapshot."""
    if "expected_repeat" in case:
        repeat = case["expected_repeat"]
        text = repeat["prefix"] + repeat["text"] * repeat["count"] + repeat["suffix"]
        role = case["definition"]["assembly_config"]["target_role"]
        legacy = {"system_prompt": text if role == "system" else "", "user_prompt": text if role == "user" else ""}
    else:
        text, legacy = case["expected_text"], case["expected_legacy"]
    return {"definition_kind": "single_text_recipe", "rendered_text": text, "legacy": legacy}


@pytest.mark.parametrize("case", RENDER_CASES, ids=lambda case: case["name"])
def test_shared_success_cases_dispatch_to_exact_single_text_result(case):
    definition = copy.deepcopy(case["definition"])
    values = runtime_values(case)
    original = copy.deepcopy((definition, values))
    result = assembler.assemble_prompt_definition(definition, values)
    assert result.model_dump() == expected_result(case)
    assert (definition, values) == original


@pytest.mark.parametrize("case", RENDER_CASES, ids=lambda case: case["name"])
def test_shared_success_cases_render_parsed_model_directly(case):
    render = getattr(assembler, "render_single_text_recipe", None)
    assert callable(render), "The v2 renderer must be available independently of v1 assembly"
    result = render(parse_prompt_definition(case["definition"]), runtime_values(case))
    assert result.model_dump() == expected_result(case)


@pytest.mark.parametrize("case", ERROR_CASES, ids=lambda case: case["name"])
def test_shared_errors_are_stable_and_do_not_mutate_inputs(case):
    definition = copy.deepcopy(case["definition"])
    values = runtime_values(case)
    original = copy.deepcopy((definition, values))
    with pytest.raises(ValueError) as error:
        assembler.assemble_prompt_definition(definition, values)
    code = error.value.errors()[0]["type"] if isinstance(error.value, ValidationError) else error.value.code
    assert code == case["expected_error_code"]
    assert (definition, values) == original


def test_recipe_never_inserts_v1_extras_or_fabricates_messages():
    case = RENDER_CASES[0]
    result = assembler.assemble_prompt_definition(
        case["definition"],
        None,
        extras={"modules_config": [{"type": "hidden"}], "few_shot_examples": [{"input": "hidden"}]},
    )
    assert result.model_dump() == expected_result(case)


@pytest.mark.parametrize("order", [10**100, -(10**100), "1", 1.0, True])
def test_v1_keeps_unbounded_and_coercive_block_orders(order):
    definition = {
        "schema_version": 1,
        "blocks": [{"id": "a", "name": "A", "role": "user", "order": order, "content": "legacy"}],
    }
    assert assembler.assemble_prompt_definition(definition, None).messages == [{"role": "user", "content": "legacy"}]


def test_v1_keeps_message_result_template_coercion_and_whitespace():
    definition = {
        "schema_version": 1,
        "variables": [{"name": "x", "required": True}],
        "blocks": [{"id": "a", "name": "A", "role": "user", "order": 0, "content": " {{x}} ", "is_template": True}],
    }
    assert assembler.assemble_prompt_definition(definition, {"x": True}).model_dump() == {
        "messages": [{"role": "user", "content": " True "}],
        "legacy": {"system_prompt": "", "user_prompt": " True "},
    }


def test_v1_keeps_implicit_default_version():
    assert assembler.assemble_prompt_definition({}, None).model_dump() == {
        "messages": [],
        "legacy": {"system_prompt": "", "user_prompt": ""},
    }


@pytest.mark.parametrize("version", [1, 1.0, True, "1", "1.0", "01", "+1", b"1"])
def test_v1_keeps_existing_literal_version_coercion(version):
    assert assembler.assemble_prompt_definition({"schema_version": version}, None).model_dump() == {
        "messages": [],
        "legacy": {"system_prompt": "", "user_prompt": ""},
    }


@pytest.mark.parametrize("version", [2, 2.0, "2", 3, "3", False, 1.5])
def test_other_explicit_versions_never_enter_the_v1_assembler(version):
    with pytest.raises(ValueError):
        assembler.assemble_prompt_definition({"schema_version": version}, None)


@pytest.mark.parametrize("use_default", [False, True])
def test_output_budget_is_checked_before_materializing_repeated_values(monkeypatch, use_default):
    """A valid 250-character template must not allocate a 500k-code-point expansion."""
    definition = copy.deepcopy(RENDER_CASES[0]["definition"])
    definition["blocks"] = [
        {
            "id": "a",
            "name": "A",
            "role": "system",
            "order": 0,
            "content": "{{x}}" * 50,
            "is_template": True,
        }
    ]
    value = "😀" * 10000
    definition["variables"] = [{"name": "x", "default_value": value if use_default else None}]
    pattern = single_text_renderer._TEMPLATE_VARIABLE_PATTERN

    class AllocationGuard:
        finditer = pattern.finditer

        @staticmethod
        def sub(*args, **kwargs):
            raise AssertionError("Oversized substitution was materialized")

    monkeypatch.setattr(single_text_renderer, "_TEMPLATE_VARIABLE_PATTERN", AllocationGuard())
    with pytest.raises(ValueError) as error:
        assembler.assemble_prompt_definition(definition, {} if use_default else {"x": value})
    assert error.value.code == "rendered_output_too_large"
