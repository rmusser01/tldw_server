import pytest

from tldw_Server_API.app.core.DB_Management.prompts_db_helpers import (
    parse_stored_prompt_definition,
)
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import InputError, PromptStudioDatabase

pytestmark = pytest.mark.integration


def _make_prompt_definition_payload(task_text: str = "Evaluate {{input}}") -> dict:
    return {
        "schema_version": 1,
        "format": "structured",
        "variables": [
            {
                "name": "input",
                "label": "Input",
                "required": True,
                "input_type": "textarea",
            }
        ],
        "blocks": [
            {
                "id": "identity",
                "name": "Identity",
                "role": "system",
                "content": "You are a careful evaluator.",
                "enabled": True,
                "order": 10,
                "is_template": False,
            },
            {
                "id": "task",
                "name": "Task",
                "role": "user",
                "content": task_text,
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


def _make_recipe_definition_payload(
    task_text: str = "Explain {{topic}} for {{audience}}.",
) -> dict:
    return {
        "schema_version": 2,
        "format": "structured",
        "definition_kind": "single_text_recipe",
        "variables": [
            {
                "name": "topic",
                "label": "Topic",
                "required": True,
                "default_value": None,
                "input_type": "text",
            },
            {
                "name": "audience",
                "label": "Audience",
                "required": False,
                "default_value": "developers",
                "input_type": "text",
            },
        ],
        "blocks": [
            {
                "id": "objective",
                "name": "Objective",
                "section_key": "objective",
                "role": "user",
                "kind": "objective",
                "content": task_text,
                "enabled": True,
                "order": 10,
                "is_template": True,
            }
        ],
        "assembly_config": {
            "assembly_mode": "single_text",
            "target_role": "user",
            "render_format": "markdown",
            "block_separator": "\n\n",
        },
    }


def test_prompt_studio_update_creates_new_structured_prompt_version(
    isolated_db: PromptStudioDatabase,
):
    project = isolated_db.create_project(name="Structured Prompt Project", user_id="test-user")
    created = isolated_db.create_prompt(
        project_id=project["id"],
        name="Structured Evaluator",
        version_number=1,
        prompt_format="structured",
        prompt_schema_version=1,
        prompt_definition=_make_prompt_definition_payload(),
        client_id="test-client",
    )

    updated = isolated_db.create_prompt_version(
        created["id"],
        change_description="Adjust instructions",
        prompt_definition=_make_prompt_definition_payload("Evaluate {{input}} carefully."),
        client_id="test-client",
    )

    assert updated["version_number"] == 2
    assert updated["prompt_format"] == "structured"
    assert updated["prompt_definition"]["blocks"][1]["content"] == "Evaluate {{input}} carefully."


def test_prompt_studio_rejects_schema_version_mismatch(
    isolated_db: PromptStudioDatabase,
):
    project = isolated_db.create_project(name="Structured Prompt Project", user_id="test-user")

    with pytest.raises(InputError, match="must match prompt_definition.schema_version"):
        isolated_db.create_prompt(
            project_id=project["id"],
            name="Structured Evaluator",
            version_number=1,
            prompt_format="structured",
            prompt_schema_version=999,
            prompt_definition=_make_prompt_definition_payload(),
            client_id="test-client",
        )


def test_prompt_studio_recipe_create_update_and_version_history_preserve_identity(
    isolated_db: PromptStudioDatabase,
):
    project = isolated_db.create_project(name="Recipe Project", user_id="test-user")
    original = _make_recipe_definition_payload()
    canonical_original = parse_stored_prompt_definition(original).model_dump()

    created = isolated_db.create_prompt(
        project_id=project["id"],
        name="User Recipe",
        prompt_format="structured",
        prompt_schema_version=2,
        prompt_definition=original,
        client_id="test-client",
    )

    assert created["prompt_schema_version"] == 2
    assert created["prompt_definition"] == canonical_original
    assert created["system_prompt"] == ""
    assert created["user_prompt"] == "## Objective\n\nExplain {{topic}} for {{audience}}."

    revised = _make_recipe_definition_payload("Compare {{topic}} for {{audience}}.")
    canonical_revised = parse_stored_prompt_definition(revised).model_dump()
    updated = isolated_db.create_prompt_version(
        created["id"],
        change_description="Use comparison wording",
        prompt_definition=revised,
        client_id="test-client",
    )

    assert updated["version_number"] == 2
    assert updated["prompt_schema_version"] == 2
    assert updated["prompt_definition"] == canonical_revised
    assert updated["user_prompt"] == "## Objective\n\nCompare {{topic}} for {{audience}}."

    history = isolated_db.list_prompt_versions(project["id"], "User Recipe")
    assert [item["prompt_schema_version"] for item in history] == [2, 2]
    assert [item["prompt_definition"] for item in history] == [canonical_revised, canonical_original]


@pytest.mark.parametrize("runtime_key", ["runtime_values", "variable_values", "resolved_values"])
def test_prompt_studio_recipe_storage_rejects_runtime_value_maps(
    isolated_db: PromptStudioDatabase,
    runtime_key: str,
):
    project = isolated_db.create_project(name="Recipe Guard Project", user_id="test-user")
    definition = _make_recipe_definition_payload()
    definition[runtime_key] = {"topic": "private"}

    with pytest.raises(InputError, match="invalid_recipe_runtime_values"):
        isolated_db.create_prompt(
            project_id=project["id"],
            name="Unsafe Recipe",
            prompt_format="structured",
            prompt_schema_version=2,
            prompt_definition=definition,
            client_id="test-client",
        )

    assert isolated_db.list_prompt_versions(project["id"], "Unsafe Recipe") == []
