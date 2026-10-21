"""Stale immutable edits must remain recoverable on both supported databases."""

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_prompts import update_prompt
from tldw_Server_API.app.api.v1.schemas.prompt_studio_base import SecurityConfig
from tldw_Server_API.app.api.v1.schemas.prompt_studio_project import PromptUpdate
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase
from tldw_Server_API.tests.prompt_studio.test_structured_prompt_versions import (
    _make_prompt_definition_payload,
    _make_recipe_definition_payload,
)


@pytest.fixture(params=["sqlite", "postgres"])
def conflict_db(request, tmp_path):
    if request.param == "sqlite":
        yield request.getfixturevalue("isolated_db")
        return
    backend = request.getfixturevalue("pg_restricted_backend")
    db = PromptStudioDatabase(str(tmp_path / "placeholder.sqlite"), "c02-conflict", backend=backend)
    try:
        yield db
    finally:
        db.close_connection()
        db.close()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("schema_version", [1, 2])
async def test_stale_prompt_version_conflict_preserves_revisions_and_allows_retry(conflict_db, schema_version):
    db = conflict_db
    project = db.create_project(name="C02 conflict project", user_id="c02-owner")
    make_definition = _make_prompt_definition_payload if schema_version == 1 else _make_recipe_definition_payload
    original = db.create_prompt(
        project_id=project["id"],
        name="C02 conflict prompt",
        prompt_format="structured",
        prompt_schema_version=schema_version,
        prompt_definition=make_definition("Original instruction."),
    )
    user = {"user_id": "c02-owner", "is_admin": False, "client_id": "c02-test"}
    security = SecurityConfig(enable_rate_limiting=False)

    async def save(parent_id, text, description):
        return await update_prompt(
            prompt_id=parent_id,
            updates=PromptUpdate(
                prompt_format="structured",
                prompt_schema_version=schema_version,
                prompt_definition=make_definition(text),
                change_description=description,
            ),
            db=db,
            security_config=security,
            user_context=user,
        )

    newer_response = await save(original["id"], "First owner revision.", "First edit")
    newer_id = newer_response.data.id
    newer = db.get_prompt(newer_id)
    before = db.list_prompt_versions(project["id"], original["name"])
    with pytest.raises(HTTPException) as conflict:
        await save(original["id"], "Second unsaved revision.", "Stale edit")

    assert conflict.value.status_code == 409  # nosec B101 - Pytest regression assertion, not a runtime security boundary.
    assert conflict.value.detail == (  # nosec B101 - Pytest regression assertion, not a runtime security boundary.
        "Prompt version already exists. Reopen the latest version and reapply your changes."
    )
    assert db.get_prompt(original["id"]) == original  # nosec B101 - Pytest regression assertion, not a runtime security boundary.
    assert db.get_prompt(newer_id) == newer  # nosec B101 - Pytest regression assertion, not a runtime security boundary.
    assert db.list_prompt_versions(project["id"], original["name"]) == before  # nosec B101 - Pytest regression assertion, not a runtime security boundary.

    retry = await save(newer_id, "First and second revisions combined.", "Reapplied edit")
    assert retry.data.version_number == 3  # nosec B101 - Pytest regression assertion, not a runtime security boundary.
    assert retry.data.parent_version_id == newer_id  # nosec B101 - Pytest regression assertion, not a runtime security boundary.
    assert "First and second revisions combined." in retry.data.user_prompt  # nosec B101 - Pytest regression assertion, not a runtime security boundary.
    assert db.get_prompt(original["id"]) == original  # nosec B101 - Pytest regression assertion, not a runtime security boundary.
    assert db.get_prompt(newer_id) == newer  # nosec B101 - Pytest regression assertion, not a runtime security boundary.
    assert [row["version_number"] for row in db.list_prompt_versions(project["id"], original["name"])] == [3, 2, 1]  # nosec B101 - Pytest regression assertion, not a runtime security boundary.
