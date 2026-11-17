"""Saving templates validates their shape and length before runtime inputs exist."""

from copy import deepcopy

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.prompt_studio import prompt_studio_prompts as endpoint
from tldw_Server_API.app.api.v1.schemas.prompt_studio_base import SecurityConfig
from tldw_Server_API.app.api.v1.schemas.prompt_studio_project import PromptCreate, PromptUpdate
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.prompt_executor import PromptExecutor
from tldw_Server_API.app.core.Prompt_Management.structured_prompts import (
    PromptDefinition,
    StructuredPromptAssemblyError,
)

pytestmark = pytest.mark.integration
OWNER = {"user_id": "1", "is_admin": False}


def definition(default=None):
    return {
        "schema_version": 1,
        "format": "structured",
        "variables": [{"name": "source", "required": True, "default_value": default}],
        "blocks": [
            {
                "id": "source",
                "name": "Source",
                "role": "user",
                "order": 10,
                "content": "Source {{source}}",
                "is_template": True,
            }
        ],
    }


def fields(prompt_format):
    return (
        {"prompt_format": "structured", "prompt_schema_version": 1, "prompt_definition": definition()}
        if prompt_format == "structured"
        else {"user_prompt": "Source {{source}}"}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt_format", ["legacy", "structured"])
@pytest.mark.parametrize("action", ["create", "version"])
async def test_owned_save_retains_required_template_without_execution_inputs(isolated_db, prompt_format, action):
    project = isolated_db.create_project(name="Required template authoring", user_id="1")
    authored = fields(prompt_format)
    if action == "create":
        result = await endpoint.create_prompt(
            PromptCreate(project_id=project["id"], name="Required source", **authored),
            db=isolated_db,
            security_config=SecurityConfig(),
            user_context=OWNER,
        )
    else:
        original = isolated_db.create_prompt(project_id=project["id"], name="Original", **authored)
        result = await endpoint.update_prompt(
            original["id"],
            PromptUpdate(name="Required source revision", change_description="Required template revision", **authored),
            db=isolated_db,
            security_config=SecurityConfig(),
            user_context=OWNER,
        )
        assert isolated_db.get_prompt(original["id"])["name"] == "Original"  # nosec B101 - pytest immutable revision assertion
        assert result.data.version_number == 2  # nosec B101 - pytest actual version persistence assertion
    saved = isolated_db.get_prompt(result.data.id)
    assert saved["user_prompt"] == "Source {{source}}"  # nosec B101 - pytest authored template persistence assertion
    if prompt_format == "structured":
        assert saved["prompt_definition"]["variables"][0]["required"] is True  # nosec B101 - pytest required metadata preservation assertion


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["create", "version"])
async def test_owned_save_still_rejects_oversized_required_default(isolated_db, action):
    project = isolated_db.create_project(name="Required default length", user_id="1")
    payload = {"prompt_format": "structured", "prompt_schema_version": 1, "prompt_definition": definition("x" * 140)}
    original = isolated_db.create_prompt(project_id=project["id"], name="Original", **fields("structured"))
    before = isolated_db.list_prompts(project["id"])
    with pytest.raises(HTTPException) as failure:
        if action == "create":
            await endpoint.create_prompt(
                PromptCreate(project_id=project["id"], name="Oversized", **payload),
                db=isolated_db,
                security_config=SecurityConfig(max_prompt_length=100),
                user_context=OWNER,
            )
        else:
            await endpoint.update_prompt(
                original["id"],
                PromptUpdate(change_description="Oversized", **payload),
                db=isolated_db,
                security_config=SecurityConfig(max_prompt_length=100),
                user_context=OWNER,
            )
    assert failure.value.status_code == 400  # nosec B101 - pytest retained length rejection assertion
    assert "maximum length" in failure.value.detail  # nosec B101 - pytest length guidance assertion
    assert isolated_db.list_prompts(project["id"]) == before  # nosec B101 - pytest no partial write assertion


def test_save_validation_does_not_mutate_required_metadata():
    authored = PromptDefinition.model_validate(definition())
    original = deepcopy(authored)
    endpoint._validate_prompt_content(definition=authored, extras={}, security_config=SecurityConfig())
    assert authored == original  # nosec B101 - pytest authored schema immutability assertion


def test_save_validation_still_counts_signature_output():
    authored = PromptDefinition.model_validate(definition())
    with pytest.raises(HTTPException) as failure:
        endpoint._validate_prompt_content(
            definition=authored,
            extras={},
            security_config=SecurityConfig(max_prompt_length=100),
            signature={"input_schema": [], "output_schema": [{"name": "x" * 140, "type": "string"}]},
        )
    assert failure.value.status_code == 400  # nosec B101 - pytest retained signature length assertion


def test_save_validation_still_enforces_total_length(monkeypatch):
    monkeypatch.setattr(PromptExecutor, "MAX_TOTAL_PROMPT_LENGTH", 5)
    with pytest.raises(HTTPException) as failure:
        endpoint._validate_prompt_content(
            definition=PromptDefinition.model_validate(definition()),
            extras={},
            security_config=SecurityConfig(),
        )
    assert "maximum total length" in failure.value.detail  # nosec B101 - pytest total-length boundary assertion


def test_save_validation_still_rejects_undeclared_variables():
    authored = definition()
    authored["blocks"][0]["content"] = "{{other}}"
    with pytest.raises(StructuredPromptAssemblyError) as failure:
        endpoint._validate_prompt_content(
            definition=PromptDefinition.model_validate(authored),
            extras={},
            security_config=SecurityConfig(),
        )
    assert failure.value.code == "invalid_prompt_definition"  # nosec B101 - pytest structural validation assertion
