"""Real router/SQLite checks for recipe trust boundaries (auth is out of scope)."""

import json
from copy import deepcopy

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from loguru import logger

from tldw_Server_API.app.api.v1.endpoints import prompts
from tldw_Server_API.app.core.DB_Management.Prompts_DB import PromptsDatabase

pytestmark = pytest.mark.integration


def recipe(target="system"):
    return {
        "name": "Recipe",
        "prompt_format": "structured",
        "prompt_schema_version": 2,
        "prompt_definition": {
            "schema_version": 2,
            "format": "structured",
            "definition_kind": "single_text_recipe",
            "assembly_config": {
                "assembly_mode": "single_text",
                "target_role": target,
                "render_format": "freeform",
                "block_separator": "",
            },
            "blocks": [
                {
                    "id": "task",
                    "name": "Task",
                    "role": target,
                    "order": 0,
                    "section_key": "task",
                    "content": "Authored text",
                }
            ],
        },
    }


@pytest.fixture
def boundary_client(tmp_path):
    db = PromptsDatabase(tmp_path / "prompts.db", client_id="boundary-test")
    app = FastAPI()
    app.include_router(prompts.router, prefix="/api/v1/prompts")
    app.dependency_overrides[prompts.verify_prompts_user] = lambda: True
    app.dependency_overrides[prompts.get_prompts_db_for_user] = lambda: db
    client = TestClient(app, raise_server_exceptions=False)
    yield client, db
    client.close()
    db.close_connection()


def db_state(db):
    return (
        [dict(row) for row in db.execute_query("SELECT * FROM Prompts ORDER BY id").fetchall()],
        db.get_sync_log_entries(),
    )


@pytest.mark.parametrize(
    "identity",
    ["v2", "future", "mistagged", "kind_only", "version_only", "json_definition", "config_only", "format_only"],
)
def test_legacy_import_rejects_entire_structured_batch_before_mutation(boundary_client, identity):
    client, db = boundary_client
    db.add_prompt("Existing", None, None, system_prompt="Keep")
    before = db_state(db)
    item = {**recipe(), "content": "Details", "system_prompt": "FILLED_INSTANCE_SENTINEL"}
    if identity == "future":
        item["prompt_schema_version"] = item["prompt_definition"]["schema_version"] = 99
    elif identity == "mistagged":
        item["prompt_schema_version"] = item["prompt_definition"]["schema_version"] = 1
    elif identity == "kind_only":
        item = {"name": "Recipe", "content": "Details", "definition_kind": "single_text_recipe"}
    elif identity == "version_only":
        item = {"name": "Recipe", "content": "Details", "prompt_schema_version": 99}
    elif identity == "config_only":
        item = {"name": "Recipe", "content": "Details", "assembly_config": {"assembly_mode": "single_text"}}
    elif identity == "format_only":
        item = {"name": "Recipe", "content": "Details", "format": "structured"}
    elif identity == "json_definition":
        item["prompt_definition"] = json.dumps(item["prompt_definition"])
    body = {"prompts": [{"name": "Earlier item", "content": "Never partially import"}, item]}
    original = deepcopy(body)
    response = client.post("/api/v1/prompts/import", json=body)
    assert response.status_code == 400, response.text
    assert response.json() == {"detail": "structured_prompt_not_supported_on_legacy_route"}
    assert db_state(db) == before
    assert body == original


@pytest.mark.parametrize("marker", ["section_key", "assembly_mode", "target_role", "render_format", "definition_kind"])
@pytest.mark.parametrize("value", ["SECRET_KEY", {"secret": "SECRET_KEY"}, ["SECRET_KEY"]])
@pytest.mark.parametrize("operation", ["create", "update", "preview"])
def test_recipe_marker_errors_are_content_free_and_never_500(boundary_client, marker, value, operation):
    client, db = boundary_client
    prompt_id, _, _ = db.add_prompt("Good", None, None, system_prompt="Keep")
    before = db_state(db)
    payload = recipe()
    definition = payload["prompt_definition"]
    definition["schema_version"] = payload["prompt_schema_version"] = 1
    definition.pop("definition_kind")
    definition["assembly_config"] = {}
    definition["blocks"][0].pop("section_key")
    if marker == "section_key":
        definition["blocks"][0][marker] = value
    elif marker == "definition_kind":
        definition[marker] = value
    else:
        definition["assembly_config"][marker] = value
    messages = []
    sink = logger.add(messages.append, format="{message}")
    try:
        if operation == "update":
            response = client.put(f"/api/v1/prompts/{prompt_id}", json=payload)
        else:
            response = client.post(
                "/api/v1/prompts/preview" if operation == "preview" else "/api/v1/prompts", json=payload
            )
    finally:
        logger.remove(sink)
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "invalid_prompt_definition"
    assert "SECRET_KEY" not in response.text
    assert "SECRET_KEY" not in "".join(str(message) for message in messages)
    assert db_state(db) == before


@pytest.mark.parametrize("identity", ["v2", "future", "mistagged", "json_definition"])
def test_legacy_create_rejects_structured_identity_without_echoing_input(boundary_client, identity):
    client, db = boundary_client
    payload = recipe()
    payload["prompt_definition"]["blocks"][0]["content"] = "SECRET_KEY"
    if identity in {"future", "mistagged"}:
        payload["prompt_schema_version"] = payload["prompt_definition"]["schema_version"] = (
            99 if identity == "future" else 1
        )
    if identity == "json_definition":
        payload["prompt_definition"] = json.dumps(payload["prompt_definition"])
    before = db_state(db)
    response = client.post("/api/v1/prompts/create", json=payload)
    assert response.status_code == 400, response.text
    assert response.json() == {"detail": "structured_prompt_not_supported_on_legacy_route"}
    assert "SECRET_KEY" not in response.text
    assert db_state(db) == before


@pytest.mark.parametrize("field", ["prompt_schema_version", "prompt_format", "prompt_definition"])
@pytest.mark.parametrize("path", ["/api/v1/prompts", "/api/v1/prompts/preview"])
def test_recipe_envelope_model_errors_do_not_echo_input(boundary_client, field, path):
    client, db = boundary_client
    payload = recipe()
    payload[field] = ["SECRET_KEY"]
    response = client.post(path, json=payload)
    assert response.status_code == 400, response.text
    assert "SECRET_KEY" not in response.text
    assert db_state(db) == ([], [])


def sized_recipe(target, size):
    payload = recipe(target)
    block = payload["prompt_definition"]["blocks"][0]
    payload["prompt_definition"]["blocks"] = [
        {**block, "id": f"block{i}", "section_key": f"block{i}", "order": i, "content": "x" * min(20000, size - offset)}
        for i, offset in enumerate(range(0, size, 20000))
    ]
    return payload


@pytest.mark.parametrize("target", ["system", "user"])
@pytest.mark.parametrize("size", [40002, 100000])
def test_large_recipe_snapshot_roundtrip_create_get_put_preview(boundary_client, target, size):
    client, db = boundary_client
    payload = sized_recipe(target, size)
    created = client.post("/api/v1/prompts", json=payload)
    assert created.status_code == 201, created.text
    record = created.json()
    field = f"{target}_prompt"
    assert record[field] == "x" * size
    fetched = client.get(f"/api/v1/prompts/{record['id']}")
    assert fetched.json()[field] == record[field]
    updated = client.put(f"/api/v1/prompts/{record['id']}", json=record)
    assert updated.status_code == 200, updated.text
    assert updated.json()[field] == record[field]
    before = db_state(db)
    preview = client.post("/api/v1/prompts/preview", json=updated.json())
    assert preview.status_code == 200, preview.text
    assert preview.json()["rendered_text"] == record[field]
    assert db_state(db) == before


@pytest.mark.parametrize("target", ["system", "user"])
@pytest.mark.parametrize("operation", ["create", "update", "preview"])
@pytest.mark.parametrize("overflow", ["derived", "supplied", "wrong_type"])
def test_recipe_snapshot_overflow_and_types_are_sanitized_without_mutation(
    boundary_client, target, operation, overflow
):
    client, db = boundary_client
    prompt_id, _, _ = db.add_prompt("Good", None, None, system_prompt="Keep")
    before = db_state(db)
    payload = sized_recipe(target, 100001 if overflow == "derived" else 20001)
    if overflow != "derived":
        payload[f"{target}_prompt"] = ("SECRET_KEY" + "x" * 100000) if overflow == "supplied" else ["SECRET_KEY"]
    messages = []
    sink = logger.add(messages.append, format="{message}")
    try:
        response = (
            client.put(f"/api/v1/prompts/{prompt_id}", json=payload)
            if operation == "update"
            else client.post("/api/v1/prompts/preview" if operation == "preview" else "/api/v1/prompts", json=payload)
        )
    finally:
        logger.remove(sink)
    assert response.status_code == 400, response.text
    assert "SECRET_KEY" not in response.text + "".join(str(message) for message in messages)
    assert db_state(db) == before


@pytest.mark.parametrize("target", ["system", "user"])
@pytest.mark.parametrize("kind", ["legacy", "v1"])
@pytest.mark.parametrize("size", [20000, 20001])
@pytest.mark.parametrize("operation", ["create", "update", "preview"])
def test_old_text_request_limit_preserved_and_sanitized(boundary_client, target, kind, size, operation):
    client, db = boundary_client
    prompt_id, _, _ = db.add_prompt("Good", None, None, system_prompt="Keep")
    before = db_state(db)
    payload = {"name": "New", f"{target}_prompt": "SECRET_KEY" + "x" * (size - 10)}
    if kind == "v1":
        payload.update(
            prompt_format="structured",
            prompt_schema_version=1,
            prompt_definition={
                "schema_version": 1,
                "format": "structured",
                "blocks": [{"id": "task", "name": "Task", "role": target, "content": "Authored", "order": 0}],
            },
        )
    response = (
        client.put(f"/api/v1/prompts/{prompt_id}", json=payload)
        if operation == "update"
        else client.post("/api/v1/prompts/preview" if operation == "preview" else "/api/v1/prompts", json=payload)
    )
    if size == 20000:
        assert response.status_code == (201 if operation == "create" else 200), response.text
    else:
        assert response.status_code == 400, response.text
        assert "SECRET_KEY" not in response.text
        assert db_state(db) == before


@pytest.mark.parametrize("target", ["system", "user"])
@pytest.mark.parametrize("operation", ["create", "overwrite", "update"])
@pytest.mark.parametrize("kind", ["legacy", "v1", "v2"])
def test_direct_db_text_limit_rejection_preserves_record(boundary_client, target, operation, kind):
    _, db = boundary_client
    prompt_id, _, _ = db.add_prompt("Good", None, None, system_prompt="Keep")
    before = db_state(db)
    payload = {f"{target}_prompt": "SECRET_KEY" + "x" * (100000 if kind == "v2" else 20000)}
    if kind != "legacy":
        structured = recipe(target)
        if kind == "v1":
            structured["prompt_schema_version"] = 1
            structured["prompt_definition"] = {"schema_version": 1, "format": "structured", "blocks": []}
        payload.update({key: value for key, value in structured.items() if key != "name"})
    with pytest.raises(prompts.InputError, match="^invalid_prompt_text$"):
        if operation == "update":
            db.update_prompt_by_id(prompt_id, payload)
        else:
            db.add_prompt(
                "Good" if operation == "overwrite" else "New", None, None, overwrite=operation == "overwrite", **payload
            )
    assert db_state(db) == before


@pytest.mark.parametrize("target", ["system", "user"])
def test_v1_derived_snapshot_above_request_bound_remains_compatible(boundary_client, target):
    client, _ = boundary_client
    payload = sized_recipe(target, 40000)
    payload["prompt_schema_version"] = payload["prompt_definition"]["schema_version"] = 1
    payload["prompt_definition"].pop("definition_kind")
    payload["prompt_definition"]["assembly_config"] = {"block_separator": ""}
    for block in payload["prompt_definition"]["blocks"]:
        block.pop("section_key")
    created = client.post("/api/v1/prompts", json=payload)
    assert created.status_code == 201, created.text
    assert created.json()[f"{target}_prompt"] == "x" * 40000
    updated = client.put(f"/api/v1/prompts/{created.json()['id']}", json=payload)
    assert updated.status_code == 200, updated.text
    assert updated.json()[f"{target}_prompt"] == "x" * 40000


def test_openapi_transport_bounds_match_centralized_recipe_snapshot_limit(boundary_client):
    from tldw_Server_API.app.core.Prompt_Management.structured_prompts.models import SINGLE_TEXT_RECIPE_LIMITS

    client, _ = boundary_client
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    for name in ("PromptCreate", "PromptResponse", "StructuredPromptPreviewRequest"):
        for field in ("system_prompt", "user_prompt"):
            text_schema = next(item for item in schemas[name]["properties"][field]["anyOf"] if item["type"] == "string")
            assert text_schema["maxLength"] == SINGLE_TEXT_RECIPE_LIMITS["max_rendered_output_length"]
