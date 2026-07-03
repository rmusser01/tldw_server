from pathlib import Path

import pytest

from tldw_Server_API.app.core.Chat_Macros.exceptions import MacroValidationError
from tldw_Server_API.app.core.Chat_Macros.models import MacroArgSpec
from tldw_Server_API.app.core.Chat_Macros.parser import load_macro_definition, parse_macro_args


BUILTIN_WRAPUP_PATH = (
    Path(__file__).parents[3]
    / "app"
    / "core"
    / "Chat_Macros"
    / "builtin"
    / "wrapup"
    / "MACRO.yaml"
)


def WrapupArgsSpec() -> dict[str, MacroArgSpec]:
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


def test_non_empty_tool_or_skill_permissions_rejected():
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


def test_parse_slash_args_normalizes_aliases_and_repeated_questions():
    spec = WrapupArgsSpec()
    args = parse_macro_args(
        '--preset dev_handoff --keep-forks --output-profile compact '
        '--question "What changed?" --question "What is next?"',
        spec,
    )
    assert args["keep_forks"] is True
    assert args["output_profile"] == "compact"
    assert args["question"] == ["What changed?", "What is next?"]


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
