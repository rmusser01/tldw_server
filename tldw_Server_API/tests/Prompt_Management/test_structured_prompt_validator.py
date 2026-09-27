# test_structured_prompt_validator.py
# Unit tests for structured prompt definition validation

import pytest

from tldw_Server_API.app.core.Prompt_Management.structured_prompts.validator import (
    validate_prompt_definition,
)

pytestmark = pytest.mark.unit


def _make_definition(**overrides):
    definition = {
        "schema_version": 1,
        "format": "structured",
        "variables": [
            {
                "name": "topic",
                "label": "Topic",
                "required": True,
                "input_type": "textarea",
            }
        ],
        "blocks": [
            {
                "id": "identity",
                "name": "Identity",
                "role": "system",
                "content": "You are precise.",
                "enabled": True,
                "order": 10,
                "is_template": False,
            },
            {
                "id": "task",
                "name": "Task",
                "role": "user",
                "content": "Summarize {{topic}}",
                "enabled": True,
                "order": 20,
                "is_template": True,
            },
        ],
        "assembly_config": {
            "legacy_system_roles": ["system", "developer"],
            "legacy_user_roles": ["user"],
            "block_separator": "\n\n",
        },
    }
    definition.update(overrides)
    return definition


def test_validator_accepts_valid_definition():
    errors = validate_prompt_definition(_make_definition())

    assert errors == []


def test_discriminated_parser_preserves_v1_compatibility():
    from tldw_Server_API.app.core.Prompt_Management import structured_prompts

    payload = _make_definition()
    parsed = structured_prompts.parse_prompt_definition(payload)
    assert isinstance(parsed, structured_prompts.PromptDefinition)
    assert isinstance(parsed, structured_prompts.MultiMessagePromptDefinitionV1)
    assert parsed.model_dump(exclude_unset=True) == payload
    assert validate_prompt_definition(parsed) == []


def test_validator_rejects_duplicate_variable_names():
    definition = _make_definition(
        variables=[
            {"name": "topic", "required": True, "input_type": "textarea"},
            {"name": "topic", "required": False, "input_type": "text"},
        ]
    )

    errors = validate_prompt_definition(definition)

    assert [error.code for error in errors] == ["duplicate_variable_name"]


def test_validator_rejects_duplicate_block_ids():
    definition = _make_definition(
        blocks=[
            {
                "id": "task",
                "name": "Identity",
                "role": "system",
                "content": "You are precise.",
                "enabled": True,
                "order": 10,
                "is_template": False,
            },
            {
                "id": "task",
                "name": "Task",
                "role": "user",
                "content": "Summarize {{topic}}",
                "enabled": True,
                "order": 20,
                "is_template": True,
            },
        ]
    )

    errors = validate_prompt_definition(definition)

    assert [error.code for error in errors] == ["duplicate_block_id"]


def test_validator_rejects_invalid_block_role():
    definition = _make_definition(
        blocks=[
            {
                "id": "task",
                "name": "Task",
                "role": "tool",
                "content": "Summarize {{topic}}",
                "enabled": True,
                "order": 20,
                "is_template": True,
            }
        ]
    )

    errors = validate_prompt_definition(definition)

    assert [error.code for error in errors] == ["invalid_block_role"]


def test_validator_rejects_unsupported_schema_version():
    errors = validate_prompt_definition(_make_definition(schema_version=99))

    assert [error.code for error in errors] == ["unsupported_schema_version"]


def test_validator_rejects_unknown_variable_references():
    definition = _make_definition(
        variables=[],
        blocks=[
            {
                "id": "task",
                "name": "Task",
                "role": "user",
                "content": "Summarize {{missing_topic}}",
                "enabled": True,
                "order": 20,
                "is_template": True,
            }
        ],
    )

    errors = validate_prompt_definition(definition)

    assert [error.code for error in errors] == ["unknown_variable_reference"]
