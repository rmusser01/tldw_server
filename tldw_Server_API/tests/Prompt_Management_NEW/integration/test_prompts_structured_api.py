"""
Integration tests for structured prompt support in the regular Prompts API.
"""

from copy import deepcopy

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(name="test_client")
def _prompt_test_client(test_env_vars):
    """Exercise the real app/router without unrelated background-service lifespans."""
    from fastapi.testclient import TestClient

    from tldw_Server_API.app.main import app

    client = TestClient(app)
    yield client
    client.close()


@pytest.fixture(autouse=True)
def isolated_prompt_database(test_client, tmp_path):
    """Use real per-test storage; the shared app fixture otherwise shares records."""
    from tldw_Server_API.app.api.v1.API_Deps.Prompts_DB_Deps import get_prompts_db_for_user
    from tldw_Server_API.app.core.DB_Management.Prompts_DB import PromptsDatabase

    db = PromptsDatabase(tmp_path / "prompts.db", client_id="recipe-api-test")
    overrides = test_client.app.dependency_overrides
    previous = overrides.get(get_prompts_db_for_user)
    overrides[get_prompts_db_for_user] = lambda: db
    yield db
    if previous is None:
        overrides.pop(get_prompts_db_for_user, None)
    else:
        overrides[get_prompts_db_for_user] = previous
    db.close_connection()


def _recipe_payload(target="system"):
    return {
        "name": f"Recipe {target}",
        "prompt_format": "structured",
        "prompt_schema_version": 2,
        "prompt_definition": {
            "schema_version": 2,
            "format": "structured",
            "definition_kind": "single_text_recipe",
            "assembly_config": {
                "assembly_mode": "single_text",
                "target_role": target,
                "render_format": "xml",
            },
            "variables": [{"name": "topic", "required": True, "default_value": "starter"}],
            "blocks": [
                {
                    "id": "task",
                    "name": "Task",
                    "section_key": "task",
                    "role": target,
                    "content": "Explain {{topic}}",
                    "order": 1,
                    "is_template": True,
                }
            ],
        },
    }


@pytest.mark.parametrize("target", ["system", "user"])
def test_recipe_crud_preview_and_runtime_nonpersistence(test_client, auth_headers, target):
    payload = _recipe_payload(target)
    original = deepcopy(payload)
    created = test_client.post("/api/v1/prompts", json=payload, headers=auth_headers)
    assert created.status_code == 201, created.text
    saved = created.json()
    assert saved[f"{target}_prompt"] == "<task>Explain {{topic}}</task>"
    assert saved[f"{'user' if target == 'system' else 'system'}_prompt"] == ""
    url = f"/api/v1/prompts/{saved['id']}"
    assert test_client.get(url, headers=auth_headers).json() == saved
    preview = test_client.post(
        "/api/v1/prompts/preview",
        json={**payload, "variables": {"topic": "PRIVATE"}},
        headers=auth_headers,
    )
    assert preview.status_code == 200, preview.text
    assert preview.json() == {
        "prompt_format": "structured",
        "prompt_schema_version": 2,
        "assembled_messages": [],
        "rendered_text": "<task>Explain PRIVATE</task>",
        "legacy_system_prompt": "<task>Explain PRIVATE</task>" if target == "system" else "",
        "legacy_user_prompt": "<task>Explain PRIVATE</task>" if target == "user" else "",
    }
    assert test_client.get(url, headers=auth_headers).json() == saved
    assert payload == original
    payload["prompt_definition"]["blocks"][0]["content"] = "Updated {{topic}}"
    payload.pop("prompt_format")
    updated = test_client.put(url, json=payload, headers=auth_headers)
    assert updated.status_code == 200, updated.text
    assert updated.json()[f"{target}_prompt"] == "<task>Updated {{topic}}</task>"
    assert updated.json()["version"] == saved["version"] + 1
    deleted = test_client.delete(url, headers=auth_headers)
    assert deleted.status_code in (200, 204), deleted.text
    assert test_client.get(url, headers=auth_headers).status_code == 404


@pytest.mark.parametrize("key", ["runtime_values", "variable_values", "resolved_values"])
@pytest.mark.parametrize("location", ["envelope", "definition", "block", "nested_default"])
def test_recipe_runtime_maps_rejected_before_create_or_update(test_client, auth_headers, key, location):
    # A good legacy record makes overwrite protection observable even before v2 CRUD works.
    good = test_client.post(
        "/api/v1/prompts", json={"name": "Keep me", "system_prompt": "original"}, headers=auth_headers
    ).json()
    payload = _recipe_payload()
    payload["name"] = "Keep me"
    definition = payload["prompt_definition"]
    places = {
        "envelope": payload,
        "definition": definition,
        "block": definition["blocks"][0],
        "nested_default": definition["variables"][0],
    }
    if location == "nested_default":
        places[location]["default_value"] = [{"nested": {key: {"topic": "SECRET_SENTINEL"}}}]
    else:
        places[location][key] = {"topic": "SECRET_SENTINEL"}
    for method, url in [("post", "/api/v1/prompts"), ("put", f"/api/v1/prompts/{good['id']}")]:
        response = getattr(test_client, method)(url, json=payload, headers=auth_headers)
        assert response.status_code == 400, response.text
        assert response.json()["detail"] == "invalid_recipe_runtime_values"
        assert "SECRET_SENTINEL" not in response.text
    assert test_client.get(f"/api/v1/prompts/{good['id']}", headers=auth_headers).json() == good


@pytest.mark.parametrize("path", ["/api/v1/prompts/import", "/api/v1/prompts/create"])
def test_legacy_ingestion_cannot_silently_drop_embedded_runtime_maps(test_client, auth_headers, path):
    item = {"name": "Unpersistable", "content": "safe", "prompt_definition": {"runtime_values": {"x": "PRIVATE"}}}
    body = {"prompts": [item]} if path.endswith("import") else item
    response = test_client.post(path, json=body, headers=auth_headers)
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "invalid_recipe_runtime_values"


@pytest.mark.parametrize(
    ("failure", "code"),
    [
        ("future", "unsupported_schema_version"),
        ("mismatch", "must match"),
        ("role", "invalid_block_role"),
        ("xml", "invalid_section_key"),
        ("collision", "closing_tag_collision"),
    ],
)
def test_recipe_invalid_save_never_changes_existing_record(test_client, auth_headers, failure, code):
    good = test_client.post(
        "/api/v1/prompts", json={"name": "Stable", "user_prompt": "untouched"}, headers=auth_headers
    ).json()
    payload = _recipe_payload()
    payload["name"] = "Stable"
    definition = payload["prompt_definition"]
    if failure == "future":
        definition["schema_version"] = 99
    elif failure == "mismatch":
        payload["prompt_schema_version"] = 1
    elif failure == "role":
        definition["blocks"][0]["role"] = "user"
    elif failure == "xml":
        definition["blocks"][0]["section_key"] = "bad key"
    elif failure == "collision":
        definition["blocks"][0]["content"] = "</task>"
    url = f"/api/v1/prompts/{good['id']}"
    response = test_client.put(url, json=payload, headers=auth_headers)
    assert response.status_code == 400, response.text
    assert code in response.json()["detail"]
    assert test_client.get(url, headers=auth_headers).json() == good


@pytest.mark.parametrize("target", ["system", "user"])
def test_recipe_authored_runtime_words_and_large_snapshot_round_trip(test_client, auth_headers, target):
    payload = _recipe_payload(target)
    definition = payload["prompt_definition"]
    definition["variables"][0]["default_value"] = "runtime_values variable_values resolved_values"
    definition["blocks"][0]["content"] = "{{topic}}" + "x" * 19980
    definition["blocks"].append({**definition["blocks"][0], "id": "extra", "section_key": "extra"})
    response = test_client.post("/api/v1/prompts", json=payload, headers=auth_headers)
    assert response.status_code == 201, response.text
    assert len(response.json()[f"{target}_prompt"]) > 20000
    assert (
        response.json()["prompt_definition"]["variables"][0]["default_value"]
        == "runtime_values variable_values resolved_values"
    )


@pytest.mark.parametrize("target", ["system", "user"])
def test_recipe_save_preserves_unfilled_template_but_preview_requires_values(test_client, auth_headers, target):
    payload = _recipe_payload(target)
    payload["prompt_definition"]["variables"][0].pop("default_value")
    payload["prompt_definition"]["blocks"][0]["content"] = "Explain {{ topic }}"
    saved = test_client.post("/api/v1/prompts", json=payload, headers=auth_headers)
    assert saved.status_code == 201, saved.text
    assert saved.json()[f"{target}_prompt"] == "<task>Explain {{ topic }}</task>"
    preview = test_client.post("/api/v1/prompts/preview", json=payload, headers=auth_headers)
    assert preview.status_code == 400, preview.text
    assert preview.json()["detail"] == "missing_required_variable"


def test_recipe_capability_advertises_centralized_limits_without_enablement(test_client, auth_headers):
    from tldw_Server_API.app.core.Prompt_Management.structured_prompts.models import SINGLE_TEXT_RECIPE_LIMITS

    response = test_client.get("/api/v1/prompts/capabilities", headers=auth_headers)
    assert response.status_code == 200, response.text
    assert response.json()["single_text_recipe_v2"] == {"supported": False, "limits": dict(SINGLE_TEXT_RECIPE_LIMITS)}


def test_recipe_preview_openapi_exposes_optional_rendered_text(test_client):
    schema = test_client.app.openapi()["components"]["schemas"]["StructuredPromptPreviewResponse"]
    assert "rendered_text" in schema["properties"]
    assert "rendered_text" not in schema.get("required", [])


@pytest.mark.parametrize("target", ["system", "user"])
@pytest.mark.parametrize(
    ("value", "code"),
    [
        ("</task>", "closing_tag_collision"),
        (None, "invalid_variable_value"),
        ("x" * 100000, "rendered_output_too_large"),
    ],
)
def test_recipe_preview_rejects_invalid_runtime_output_without_saving(test_client, auth_headers, target, value, code):
    payload = _recipe_payload(target)
    saved = test_client.post("/api/v1/prompts", json=payload, headers=auth_headers)
    assert saved.status_code == 201, saved.text
    preview = test_client.post(
        "/api/v1/prompts/preview", json={**payload, "variables": {"topic": value}}, headers=auth_headers
    )
    assert preview.status_code == 400, preview.text
    assert preview.json()["detail"] == code
    assert test_client.get(f"/api/v1/prompts/{saved.json()['id']}", headers=auth_headers).json() == saved.json()


def test_recipe_save_rejects_authored_output_exceeding_total_limit(test_client, auth_headers):
    payload = _recipe_payload()
    block = payload["prompt_definition"]["blocks"][0]
    payload["prompt_definition"]["blocks"] = [
        {**block, "id": f"b{i}", "section_key": f"b{i}", "content": "x" * 20000} for i in range(6)
    ]
    response = test_client.post("/api/v1/prompts", json=payload, headers=auth_headers)
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "rendered_output_too_large"


def _make_prompt_definition_payload() -> dict:
    return {
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


def test_create_structured_prompt_persists_definition_and_format(test_client, auth_headers):
    payload = {
        "name": "Structured Summarizer",
        "author": "integration-test",
        "prompt_format": "structured",
        "prompt_schema_version": 1,
        "prompt_definition": _make_prompt_definition_payload(),
        "keywords": ["summary"],
    }

    create_response = test_client.post(
        "/api/v1/prompts",
        json=payload,
        headers=auth_headers,
    )

    assert create_response.status_code == 201, create_response.text
    created_body = create_response.json()
    assert created_body["prompt_format"] == "structured"
    assert created_body["prompt_schema_version"] == 1
    assert created_body["prompt_definition"]["schema_version"] == 1

    prompt_id = created_body["id"]
    get_response = test_client.get(
        f"/api/v1/prompts/{prompt_id}",
        headers=auth_headers,
    )

    assert get_response.status_code == 200, get_response.text
    fetched_body = get_response.json()
    assert fetched_body["prompt_format"] == "structured"
    assert fetched_body["prompt_schema_version"] == 1
    assert fetched_body["prompt_definition"]["blocks"][1]["content"] == "Summarize {{topic}}"


def test_update_structured_prompt_preserves_format_when_prompt_format_is_omitted(
    test_client,
    auth_headers,
):
    create_response = test_client.post(
        "/api/v1/prompts",
        json={
            "name": "Structured Evaluator",
            "author": "integration-test",
            "prompt_format": "structured",
            "prompt_schema_version": 1,
            "prompt_definition": _make_prompt_definition_payload(),
            "keywords": ["evaluate"],
        },
        headers=auth_headers,
    )

    assert create_response.status_code == 201, create_response.text
    created_body = create_response.json()
    updated_definition = _make_prompt_definition_payload()
    updated_definition["blocks"][1]["content"] = "Evaluate {{topic}} carefully."

    update_response = test_client.put(
        f"/api/v1/prompts/{created_body['id']}",
        json={
            "name": created_body["name"],
            "author": created_body["author"],
            "prompt_schema_version": 1,
            "prompt_definition": updated_definition,
            "keywords": created_body["keywords"],
        },
        headers=auth_headers,
    )

    assert update_response.status_code == 200, update_response.text
    updated_body = update_response.json()
    assert updated_body["prompt_format"] == "structured"
    assert updated_body["prompt_definition"]["blocks"][1]["content"] == "Evaluate {{topic}} carefully."


def test_preview_prompt_returns_assembled_messages_and_legacy_snapshot(test_client, auth_headers):
    response = test_client.post(
        "/api/v1/prompts/preview",
        json={
            "prompt_format": "structured",
            "prompt_schema_version": 1,
            "prompt_definition": _make_prompt_definition_payload(),
            "variables": {"topic": "SQLite FTS"},
        },
        headers=auth_headers,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["assembled_messages"] == [
        {"role": "system", "content": "You are precise."},
        {"role": "user", "content": "Summarize SQLite FTS"},
    ]
    assert body["legacy_system_prompt"] == "You are precise."
    assert body["legacy_user_prompt"] == "Summarize SQLite FTS"
    assert set(body) == {
        "prompt_format",
        "prompt_schema_version",
        "assembled_messages",
        "legacy_system_prompt",
        "legacy_user_prompt",
    }


def test_preview_prompt_rejects_missing_required_variable_with_client_error(
    test_client,
    auth_headers,
):
    response = test_client.post(
        "/api/v1/prompts/preview",
        json={
            "prompt_format": "structured",
            "prompt_schema_version": 1,
            "prompt_definition": _make_prompt_definition_payload(),
            "variables": {},
        },
        headers=auth_headers,
    )

    assert response.status_code == 400, response.text
    assert "Missing required variable: topic" in response.json()["detail"]


def test_convert_prompt_returns_structured_definition_with_normalized_variables(test_client, auth_headers):
    response = test_client.post(
        "/api/v1/prompts/convert",
        json={
            "system_prompt": "Be precise about {topic}.",
            "user_prompt": "Summarize $topic against <baseline> in {{style}}.",
        },
        headers=auth_headers,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["prompt_format"] == "structured"
    assert body["prompt_schema_version"] == 1
    assert body["extracted_variables"] == ["topic", "baseline", "style"]
    assert body["prompt_definition"]["blocks"][0]["content"] == "Be precise about {{topic}}."
    assert body["prompt_definition"]["blocks"][1]["content"] == (
        "Summarize {{topic}} against {{baseline}} in {{style}}."
    )


def test_create_structured_prompt_rejects_unknown_variable_references(test_client, auth_headers):
    invalid_definition = _make_prompt_definition_payload()
    invalid_definition["variables"] = []

    response = test_client.post(
        "/api/v1/prompts",
        json={
            "name": "Broken Structured Prompt",
            "author": "integration-test",
            "prompt_format": "structured",
            "prompt_schema_version": 1,
            "prompt_definition": invalid_definition,
        },
        headers=auth_headers,
    )

    assert response.status_code == 400, response.text
    assert "Unknown variable reference" in response.json()["detail"]


def test_create_structured_prompt_rejects_schema_version_mismatch(test_client, auth_headers):
    response = test_client.post(
        "/api/v1/prompts",
        json={
            "name": "Mismatched Structured Prompt",
            "author": "integration-test",
            "prompt_format": "structured",
            "prompt_schema_version": 999,
            "prompt_definition": _make_prompt_definition_payload(),
        },
        headers=auth_headers,
    )

    assert response.status_code == 400, response.text
    assert "must match prompt_definition.schema_version" in response.json()["detail"]
