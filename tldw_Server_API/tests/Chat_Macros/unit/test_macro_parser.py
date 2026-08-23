from pathlib import Path

import pytest

from tldw_Server_API.app.core.Chat_Macros.exceptions import MacroValidationError
from tldw_Server_API.app.core.Chat_Macros.models import MacroArgSpec
from tldw_Server_API.app.core.Chat_Macros.parser import (
    load_macro_definition,
    normalize_structured_macro_args,
    parse_macro_args,
)

pytestmark = pytest.mark.unit


BUILTIN_WRAPUP_PATH = (
    Path(__file__).parents[3]
    / "app"
    / "core"
    / "Chat_Macros"
    / "builtin"
    / "wrapup"
    / "MACRO.yaml"
)


def wrapup_args_spec() -> dict[str, MacroArgSpec]:
    """Return the representative wrapup argument contract used by parser tests."""
    return {
        "preset": MacroArgSpec(type="string", default="general"),
        "keep_forks": MacroArgSpec(type="boolean", default=False, aliases=["keep-forks"]),
        "mode": MacroArgSpec(type="string", default="background"),
        "output_profile": MacroArgSpec(type="string", default="default", aliases=["output-profile"]),
        "question": MacroArgSpec(type="string", repeated=True, default=[]),
    }


def test_builtin_wrapup_loads_and_validates():
    macro = load_macro_definition(BUILTIN_WRAPUP_PATH.read_text())
    assert macro.command == "wrapup"
    assert [step.output for step in macro.steps if step.type == "branch_prompt"] == [
        "summary",
        "decisions",
        "action_items",
        "open_questions",
    ]


def test_non_empty_tool_permissions_rejected():
    raw = (
        "schema_version: 1\n"
        "name: bad\n"
        "command: bad\n"
        "permissions:\n"
        "  tool_calls: [shell]\n"
        "steps: []\n"
    )
    with pytest.raises(MacroValidationError, match="tool"):
        load_macro_definition(raw)


def test_non_empty_skill_permissions_rejected():
    raw = (
        "schema_version: 1\n"
        "name: bad\n"
        "command: bad\n"
        "permissions:\n"
        "  skills: [python]\n"
        "steps: []\n"
    )
    with pytest.raises(MacroValidationError, match="skills"):
        load_macro_definition(raw)


def test_repeated_arg_default_must_be_list():
    raw = (
        "schema_version: 1\n"
        "name: bad\n"
        "command: bad\n"
        "args:\n"
        "  question:\n"
        "    type: string\n"
        "    repeated: true\n"
        "    default: nope\n"
        "steps: []\n"
    )
    with pytest.raises(MacroValidationError, match="default"):
        load_macro_definition(raw)


def test_arg_default_must_match_declared_scalar_type():
    raw = (
        "schema_version: 1\n"
        "name: bad\n"
        "command: bad\n"
        "args:\n"
        "  keep_forks:\n"
        "    type: boolean\n"
        "    default: maybe\n"
        "steps: []\n"
    )
    with pytest.raises(MacroValidationError, match="default"):
        load_macro_definition(raw)


def test_arg_alias_collision_rejected():
    raw = (
        "schema_version: 1\n"
        "name: bad\n"
        "command: bad\n"
        "args:\n"
        "  output_profile:\n"
        "    type: string\n"
        "    aliases: [mode]\n"
        "  mode:\n"
        "    type: string\n"
        "steps: []\n"
    )
    with pytest.raises(MacroValidationError, match="duplicate"):
        load_macro_definition(raw)


def test_parse_slash_args_normalizes_aliases_and_repeated_questions():
    spec = wrapup_args_spec()
    args = parse_macro_args(
        '--preset dev_handoff --keep-forks --output-profile compact '
        '--question "What changed?" --question "What is next?"',
        spec,
    )
    assert args["keep_forks"] is True
    assert args["output_profile"] == "compact"
    assert args["question"] == ["What changed?", "What is next?"]


def test_explicit_repeated_values_replace_declared_defaults() -> None:
    spec = {"tag": MacroArgSpec(type="string", repeated=True, default=["seed"])}

    assert parse_macro_args("--tag first --tag second", spec) == {
        "tag": ["first", "second"]
    }


def test_repeated_value_limit_applies_to_every_repeated_argument() -> None:
    spec = {"tag": MacroArgSpec(type="string", repeated=True)}

    with pytest.raises(MacroValidationError, match="too many tag arguments"):
        parse_macro_args("--tag one --tag two", spec, max_repeated_values=1)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "Infinity"])
def test_number_arguments_reject_non_finite_values(value: str) -> None:
    spec = {"threshold": MacroArgSpec(type="number")}

    with pytest.raises(MacroValidationError, match="invalid numeric"):
        parse_macro_args(f"--threshold {value}", spec)


def test_parse_slash_args_rejects_duplicate_non_repeated_arg():
    spec = wrapup_args_spec()
    with pytest.raises(MacroValidationError, match="duplicate"):
        parse_macro_args("--mode foreground --mode background", spec)


def test_parse_slash_args_rejects_duplicate_alias_and_canonical_arg():
    spec = wrapup_args_spec()
    with pytest.raises(MacroValidationError, match="duplicate"):
        parse_macro_args("--output-profile compact --output_profile full", spec)


def test_normalize_structured_args_applies_defaults_and_validates_values():
    macro = load_macro_definition(BUILTIN_WRAPUP_PATH.read_text())

    args = normalize_structured_macro_args(
        macro,
        {"keep_forks": True, "question": ["What changed?"]},
    )

    assert args["preset"] == "general"
    assert args["keep_forks"] is True
    assert args["question"] == ["What changed?"]
    with pytest.raises(MacroValidationError, match="unknown macro argument"):
        normalize_structured_macro_args(macro, {"unknown": "value"})
    with pytest.raises(MacroValidationError, match="invalid type"):
        normalize_structured_macro_args(macro, {"keep_forks": "yes"})


def test_macro_execution_rejects_unsupported_modes_and_strategies() -> None:
    base = (
        "schema_version: 1\n"
        "name: bad\n"
        "command: bad\n"
        "execution:\n"
    )

    with pytest.raises(MacroValidationError, match="mode_default"):
        load_macro_definition(base + "  mode_default: foreground\n")
    with pytest.raises(MacroValidationError, match="branch_strategy"):
        load_macro_definition(base + "  branch_strategy: unknown\n")
    with pytest.raises(MacroValidationError, match="partial_failure"):
        load_macro_definition(base + "  partial_failure: ignore_everything\n")


def test_merge_and_post_result_consumes_must_reference_previous_outputs():
    raw = (
        "schema_version: 1\n"
        "name: bad\n"
        "command: bad\n"
        "steps:\n"
        "  - id: merge\n"
        "    type: merge\n"
        "    consumes: [missing]\n"
        "    output: final\n"
        "permissions:\n"
        "  tool_calls: []\n"
        "  skills: []\n"
    )
    with pytest.raises(MacroValidationError, match="missing"):
        load_macro_definition(raw)


def test_prompt_step_output_can_be_consumed_by_post_result():
    raw = (
        "schema_version: 1\n"
        "name: ok\n"
        "command: ok\n"
        "steps:\n"
        "  - id: prompt\n"
        "    type: prompt\n"
        "    output: answer\n"
        "    prompt: Say hi.\n"
        "  - id: post\n"
        "    type: post_result\n"
        "    consumes: [answer]\n"
    )
    macro = load_macro_definition(raw)
    assert macro.steps[0].type == "prompt"
