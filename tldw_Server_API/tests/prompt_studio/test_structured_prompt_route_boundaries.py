from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps import (
    get_prompt_studio_db,
    get_prompt_studio_user,
    get_security_config,
)
from tldw_Server_API.app.api.v1.endpoints.prompt_studio import prompt_studio_prompts
from tldw_Server_API.app.api.v1.schemas.prompt_studio_base import SecurityConfig
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase

pytestmark = pytest.mark.integration


def _recipe_definition(marker: str = "route-boundary") -> dict:
    return {
        "schema_version": 2,
        "format": "structured",
        "definition_kind": "single_text_recipe",
        "variables": [
            {
                "name": "topic",
                "label": "Topic",
                "description": "",
                "required": False,
                "default_value": f"default-{marker}",
                "input_type": "text",
            }
        ],
        "blocks": [
            {
                "id": "objective",
                "name": "Objective",
                "section_key": "objective",
                "role": "user",
                "kind": "objective",
                "content": f"content-{marker} {{{{topic}}}}",
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


def _create_payload(project_id: int, marker: str = "route-boundary") -> dict:
    return {
        "project_id": project_id,
        "name": f"name-{marker}",
        "prompt_format": "structured",
        "prompt_schema_version": 2,
        "prompt_definition": _recipe_definition(marker),
        "system_prompt": f"snapshot-system-{marker}",
        "user_prompt": f"snapshot-user-{marker}",
    }


def _v1_definition() -> dict:
    return {
        "schema_version": 1,
        "format": "structured",
        "variables": [],
        "blocks": [],
    }


@pytest.fixture
def route_client(
    isolated_db: PromptStudioDatabase,
) -> Iterator[tuple[TestClient, PromptStudioDatabase, int]]:
    project = isolated_db.create_project(name="Boundary Project", user_id="test-user")
    app = FastAPI()
    app.include_router(prompt_studio_prompts.router)

    async def override_db():
        try:
            yield isolated_db
        finally:
            isolated_db.close_connection()

    app.dependency_overrides[get_prompt_studio_db] = override_db
    app.dependency_overrides[get_prompt_studio_user] = lambda: {
        "user_id": "test-user",
        "is_admin": False,
        "client_id": "boundary-test",
    }
    app.dependency_overrides[get_security_config] = lambda: SecurityConfig(
        enable_rate_limiting=False,
    )
    with TestClient(app) as client:
        yield client, isolated_db, project["id"]


def test_route_client_releases_request_thread_sqlite_handle(route_client, monkeypatch) -> None:
    client, db, project_id = route_client
    main_thread = threading.get_ident()
    db_path = db._impl.db_path_str
    request_connections = []
    original_connect = sqlite3.connect

    def tracked_connect(database, *args, **kwargs):
        connection = original_connect(database, *args, **kwargs)
        if threading.get_ident() != main_thread and str(database) == db_path:
            request_connections.append(connection)
        return connection

    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", tracked_connect)
        response = client.get(f"/api/v1/prompt-studio/prompts/list/{project_id}")

    assert response.status_code == 200
    assert request_connections
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        request_connections[-1].execute("SELECT 1")


def _mutation_state(db: PromptStudioDatabase) -> tuple[tuple[tuple, ...], tuple[tuple, ...]]:
    conn = db.get_connection()
    prompt_rows = tuple(
        tuple(row) for row in conn.execute("SELECT * FROM prompt_studio_prompts ORDER BY id").fetchall()
    )
    sync_rows = tuple(tuple(row) for row in conn.execute("SELECT * FROM sync_log ORDER BY change_id").fetchall())
    return prompt_rows, sync_rows


@pytest.mark.parametrize("route", ["", "/create"])
@pytest.mark.parametrize("runtime_key", ["runtime_values", "variable_values", "resolved_values"])
def test_prompt_studio_create_routes_reject_raw_runtime_maps_before_mutation(
    route_client: tuple[TestClient, PromptStudioDatabase, int],
    route: str,
    runtime_key: str,
) -> None:
    client, db, project_id = route_client
    marker = f"SECRET_CREATE_{route or 'simple'}_{runtime_key}"
    payload = _create_payload(project_id, marker)
    payload[runtime_key] = {"topic": marker}
    before = _mutation_state(db)
    captured: list[str] = []
    sink_id = logger.add(captured.append, format="{message}")
    try:
        response = client.post(f"/api/v1/prompt-studio/prompts{route}", json=payload)
    finally:
        logger.remove(sink_id)

    assert response.status_code == 400
    assert response.json() == {"detail": "invalid_recipe_runtime_values"}
    assert _mutation_state(db) == before
    assert marker not in "\n".join(captured)


@pytest.mark.parametrize("runtime_key", ["runtime_values", "variable_values", "resolved_values"])
def test_prompt_studio_update_route_rejects_raw_runtime_maps_before_mutation(
    route_client: tuple[TestClient, PromptStudioDatabase, int],
    runtime_key: str,
) -> None:
    client, db, project_id = route_client
    created = db.create_prompt(
        project_id=project_id,
        name="Existing",
        prompt_format="structured",
        prompt_schema_version=2,
        prompt_definition=_recipe_definition("existing"),
        client_id="boundary-test",
    )
    marker = f"SECRET_UPDATE_{runtime_key}"
    payload = {
        "name": "Updated",
        "change_description": "Boundary update",
        "prompt_format": "structured",
        "prompt_schema_version": 2,
        "prompt_definition": _recipe_definition(marker),
        runtime_key: {"topic": marker},
    }
    before = _mutation_state(db)
    captured: list[str] = []
    sink_id = logger.add(captured.append, format="{message}")
    try:
        response = client.put(
            f"/api/v1/prompt-studio/prompts/update/{created['id']}",
            json=payload,
        )
    finally:
        logger.remove(sink_id)

    assert response.status_code == 400
    assert response.json() == {"detail": "invalid_recipe_runtime_values"}
    assert _mutation_state(db) == before
    assert marker not in "\n".join(captured)


@pytest.mark.parametrize("runtime_key", ["runtime_values", "variable_values", "resolved_values"])
@pytest.mark.parametrize("operation", ["create", "update"])
def test_prompt_studio_save_routes_reject_runtime_maps_inside_prompt_definition(
    route_client: tuple[TestClient, PromptStudioDatabase, int],
    runtime_key: str,
    operation: str,
) -> None:
    client, db, project_id = route_client
    marker = f"SECRET_DEFINITION_{operation}_{runtime_key}"
    payload = _create_payload(project_id, marker)
    payload["prompt_definition"]["variables"][0]["default_value"] = {
        "domain": [{runtime_key: {"topic": marker}}]
    }
    before = _mutation_state(db)
    captured: list[str] = []
    sink_id = logger.add(captured.append, format="{message}")
    try:
        if operation == "create":
            response = client.post("/api/v1/prompt-studio/prompts/create", json=payload)
        else:
            created = db.create_prompt(
                project_id=project_id,
                name="Existing Nested Guard",
                client_id="boundary-test",
            )
            payload.pop("project_id")
            payload["change_description"] = "Nested boundary update"
            before = _mutation_state(db)
            response = client.put(
                f"/api/v1/prompt-studio/prompts/update/{created['id']}",
                json=payload,
            )
    finally:
        logger.remove(sink_id)

    assert response.status_code == 400
    assert response.json() == {"detail": "invalid_recipe_runtime_values"}
    assert _mutation_state(db) == before
    assert marker not in "\n".join(captured)


@pytest.mark.parametrize("runtime_key", ["runtime_values", "variable_values", "resolved_values"])
@pytest.mark.parametrize("location", ["inputs", "outputs", "config"])
@pytest.mark.parametrize("operation", ["create", "update"])
def test_prompt_studio_v1_arbitrary_maps_may_use_runtime_key_names(
    route_client: tuple[TestClient, PromptStudioDatabase, int],
    runtime_key: str,
    location: str,
    operation: str,
) -> None:
    client, db, project_id = route_client
    marker = f"V1_DOMAIN_{operation}_{location}_{runtime_key}"
    payload = {
        "project_id": project_id,
        "name": "V1 arbitrary maps",
        "prompt_format": "structured",
        "prompt_schema_version": 1,
        "prompt_definition": _v1_definition(),
        "system_prompt": "",
        "user_prompt": "",
    }
    if location == "config":
        payload["modules_config"] = [
            {
                "type": "domain-data",
                "enabled": True,
                "config": {runtime_key: marker},
            }
        ]
    else:
        payload["few_shot_examples"] = [
            {
                "inputs": {runtime_key: marker} if location == "inputs" else {},
                "outputs": {runtime_key: marker} if location == "outputs" else {},
            }
        ]

    if operation == "update":
        existing = db.create_prompt(
            project_id=project_id,
            name="V1 arbitrary maps",
            prompt_format="structured",
            prompt_schema_version=1,
            prompt_definition=_v1_definition(),
            client_id="boundary-test",
        )
        prompt_id = existing["id"]
        payload.pop("project_id")
        payload["change_description"] = "Preserve v1 arbitrary map"
    before = _mutation_state(db)
    captured: list[str] = []
    sink_id = logger.add(captured.append, format="{message}")
    try:
        if operation == "create":
            response = client.post("/api/v1/prompt-studio/prompts/create", json=payload)
        else:
            response = client.put(
                f"/api/v1/prompt-studio/prompts/update/{prompt_id}",
                json=payload,
            )
    finally:
        logger.remove(sink_id)

    assert response.status_code == (201 if operation == "create" else 200)
    assert _mutation_state(db) != before
    data = response.json()["data"]
    if location == "config":
        assert data["modules_config"][0]["config"][runtime_key] == marker
    else:
        assert data["few_shot_examples"][0][location][runtime_key] == marker
    assert marker not in "\n".join(captured)


def test_prompt_studio_runtime_words_in_authored_strings_and_preview_variables_are_valid(
    route_client: tuple[TestClient, PromptStudioDatabase, int],
) -> None:
    client, _db, project_id = route_client
    definition = _recipe_definition("runtime_values variable_values resolved_values")
    create = client.post(
        "/api/v1/prompt-studio/prompts/create",
        json={
            **_create_payload(project_id, "authored-control"),
            "prompt_definition": definition,
        },
    )
    assert create.status_code == 201

    preview = client.post(
        "/api/v1/prompt-studio/prompts/preview",
        json={
            "project_id": project_id,
            "prompt_format": "structured",
            "prompt_schema_version": 2,
            "prompt_definition": definition,
            "variables": {
                "topic": "runtime_values variable_values resolved_values",
                "runtime_values": "ephemeral",
            },
        },
    )
    assert preview.status_code == 200


@pytest.mark.parametrize("operation", ["create", "update", "preview"])
@pytest.mark.parametrize("valid", [True, False])
def test_prompt_studio_route_logs_never_include_prompt_authored_content(
    route_client: tuple[TestClient, PromptStudioDatabase, int],
    operation: str,
    valid: bool,
) -> None:
    client, db, project_id = route_client
    marker = f"UNIQUE_LOG_SECRET_{operation}_{valid}"
    definition = _recipe_definition(marker)
    if not valid:
        definition["blocks"][0]["role"] = "system"
    captured: list[str] = []
    sink_id = logger.add(captured.append, format="{message}")
    try:
        if operation == "create":
            response = client.post(
                "/api/v1/prompt-studio/prompts/create",
                json={
                    **_create_payload(project_id, marker),
                    "prompt_definition": definition,
                },
            )
        elif operation == "update":
            original = db.create_prompt(
                project_id=project_id,
                name="Log Existing",
                prompt_format="structured",
                prompt_schema_version=2,
                prompt_definition=_recipe_definition("existing-log-record"),
                client_id="boundary-test",
            )
            response = client.put(
                f"/api/v1/prompt-studio/prompts/update/{original['id']}",
                json={
                    "name": f"name-{marker}",
                    "change_description": "Log boundary update",
                    "prompt_format": "structured",
                    "prompt_schema_version": 2,
                    "prompt_definition": definition,
                    "system_prompt": f"snapshot-system-{marker}",
                    "user_prompt": f"snapshot-user-{marker}",
                },
            )
        else:
            response = client.post(
                "/api/v1/prompt-studio/prompts/preview",
                json={
                    "project_id": project_id,
                    "prompt_format": "structured",
                    "prompt_schema_version": 2,
                    "prompt_definition": definition,
                    "variables": {"topic": marker},
                    "system_prompt": f"snapshot-system-{marker}",
                    "user_prompt": f"snapshot-user-{marker}",
                },
            )
    finally:
        logger.remove(sink_id)

    expected_status = (201 if operation == "create" else 200) if valid else 400
    assert response.status_code == expected_status
    assert marker not in "\n".join(captured)
