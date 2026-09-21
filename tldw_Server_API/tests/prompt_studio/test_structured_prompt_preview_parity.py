import pytest

from tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_prompts import (
    preview_prompt,
)
from tldw_Server_API.app.api.v1.schemas.prompt_studio_project import (
    StructuredPromptPreviewRequest,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.prompt_executor import (
    PromptExecutor,
)


def _make_prompt_definition_payload() -> dict:
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
                "content": "Evaluate {{input}}",
                "enabled": True,
                "order": 20,
                "is_template": True,
            },
        ],
    }


def _make_recipe_definition_payload() -> dict:
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
            }
        ],
        "blocks": [
            {
                "id": "objective",
                "name": "Objective",
                "section_key": "objective",
                "role": "system",
                "kind": "objective",
                "content": "Explain {{topic}}.",
                "enabled": True,
                "order": 10,
                "is_template": True,
            }
        ],
        "assembly_config": {
            "assembly_mode": "single_text",
            "target_role": "system",
            "render_format": "xml",
            "block_separator": "\n\n",
        },
    }


@pytest.mark.asyncio
async def test_structured_prompt_preview_matches_executor_assembly(isolated_db):
    project = isolated_db.create_project(
        name="Structured Preview Project",
        user_id="test-user",
    )
    signature = isolated_db.create_signature(
        project_id=project["id"],
        name="Structured Preview Signature",
        input_schema=[{"name": "input", "type": "string"}],
        output_schema=[{"name": "answer", "type": "string"}],
    )
    prompt = isolated_db.create_prompt(
        project_id=project["id"],
        signature_id=signature["id"],
        name="Structured Preview Prompt",
        prompt_format="structured",
        prompt_schema_version=1,
        prompt_definition=_make_prompt_definition_payload(),
        few_shot_examples=[
            {
                "inputs": {"input": "Indexes"},
                "outputs": {"answer": "Use the covering index."},
            }
        ],
        modules_config=[
            {"type": "style_rules", "enabled": True, "config": {"tone": "concise"}}
        ],
    )

    preview_response = await preview_prompt(
        StructuredPromptPreviewRequest(
            project_id=project["id"],
            signature_id=signature["id"],
            prompt_format="structured",
            prompt_schema_version=1,
            prompt_definition=_make_prompt_definition_payload(),
            few_shot_examples=[
                {
                    "inputs": {"input": "Indexes"},
                    "outputs": {"answer": "Use the covering index."},
                }
            ],
            modules_config=[
                {"type": "style_rules", "enabled": True, "config": {"tone": "concise"}}
            ],
            variables={"input": "SQLite FTS"},
        ),
        db=isolated_db,
        user_context={
            "user_id": "test-user",
            "client_id": "test-client",
            "is_admin": False,
        },
    )

    preview_data = preview_response.data
    assert preview_data is not None

    executor = PromptExecutor(isolated_db)
    prompt_record = isolated_db.get_prompt(prompt["id"])
    signature_record = isolated_db.get_signature(signature["id"])
    execution_request = executor._build_structured_prompt_request(
        prompt_record,
        signature_record,
        {"input": "SQLite FTS"},
    )

    assert preview_data.assembled_messages == execution_request["assembled_messages"]
    assert [message["role"] for message in preview_data.assembled_messages] == [
        "system",
        "developer",
        "user",
        "assistant",
        "user",
    ]
    assert preview_data.assembled_messages[1]["content"] == "Module style_rules: tone=concise"
    assert "Evaluate SQLite FTS" in preview_data.legacy_user_prompt
    assert "Please format your response as JSON" in preview_data.legacy_user_prompt


@pytest.mark.asyncio
async def test_recipe_preview_returns_exact_single_text_without_messages(isolated_db):
    project = isolated_db.create_project(name="Recipe Preview Project", user_id="test-user")

    preview_response = await preview_prompt(
        StructuredPromptPreviewRequest(
            project_id=project["id"],
            prompt_format="structured",
            prompt_schema_version=2,
            prompt_definition=_make_recipe_definition_payload(),
            variables={"topic": "SQLite FTS"},
        ),
        db=isolated_db,
        security_config=None,
        user_context={
            "user_id": "test-user",
            "client_id": "test-client",
            "is_admin": False,
        },
    )

    preview_data = preview_response.data
    assert preview_data is not None
    assert preview_data.rendered_text == "<objective>Explain SQLite FTS.</objective>"
    assert preview_data.assembled_messages == []
    assert preview_data.legacy_system_prompt == preview_data.rendered_text
    assert preview_data.legacy_user_prompt == ""
