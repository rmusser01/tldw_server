"""
Integration tests for structured prompt search indexing.
"""

import pytest

from .test_prompts_structured_api import _prompt_test_client, isolated_prompt_database  # noqa: F401

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("target", ["system", "user"])
def test_recipe_search_uses_authored_snapshot_and_coexists_with_v1(test_client, auth_headers, target):
    from .test_prompts_structured_api import _recipe_payload

    payload = _recipe_payload(target)
    payload["name"] = "Sharedtitle Recipe"
    definition = payload["prompt_definition"]
    definition["variables"][0]["default_value"] = "Privatedefaulttoken"
    definition["variables"][0]["label"] = "Unindexedvariablelabel"
    definition["blocks"][0]["name"] = "Unindexedblocklabel"
    definition["blocks"][0]["content"] = "Snapshotsentinel {{topic}}"
    definition["blocks"].append(
        {**definition["blocks"][0], "id": "disabled", "enabled": False, "content": "Unindexeddisabled"}
    )
    created = test_client.post("/api/v1/prompts", json=payload, headers=auth_headers)
    assert created.status_code == 201, created.text
    v1 = test_client.post(
        "/api/v1/prompts",
        json={
            "name": "Sharedtitle V1",
            "prompt_format": "structured",
            "prompt_schema_version": 1,
            "prompt_definition": _make_prompt_definition_payload(),
        },
        headers=auth_headers,
    )
    assert v1.status_code == 201, v1.text
    for term, count in [
        ("Sharedtitle", 2),
        ("Snapshotsentinel", 1),
        ("Privatedefaulttoken", 0),
        ("Unindexedvariablelabel", 0),
        ("Unindexedblocklabel", 0),
        ("Unindexeddisabled", 0),
    ]:
        response = test_client.post("/api/v1/prompts/search", params={"search_query": term}, headers=auth_headers)
        assert response.status_code == 200, response.text
        assert response.json()["total_matches"] == count, (term, response.json())
    assert test_client.get(f"/api/v1/prompts/{v1.json()['id']}", headers=auth_headers).json() == v1.json()


def _make_prompt_definition_payload() -> dict:
    return {
        "schema_version": 1,
        "format": "structured",
        "variables": [
            {
                "name": "text",
                "label": "Text",
                "required": True,
                "input_type": "textarea",
            }
        ],
        "blocks": [
            {
                "id": "identity",
                "name": "Identity",
                "role": "system",
                "content": "You are a classifier.",
                "enabled": True,
                "order": 10,
                "is_template": False,
            },
            {
                "id": "task",
                "name": "Task",
                "role": "user",
                "content": "Classify {{text}} by sentiment.",
                "enabled": True,
                "order": 20,
                "is_template": True,
            },
            {
                "id": "rubric",
                "name": "Rubric",
                "role": "assistant",
                "content": "Use the mauveflint verdict label when sentiment is mixed.",
                "enabled": True,
                "order": 30,
                "is_template": False,
            },
        ],
    }


def test_structured_prompt_search_indexes_enabled_block_content(
    test_client,
    auth_headers,
):
    create_response = test_client.post(
        "/api/v1/prompts",
        json={
            "name": "Structured Classifier",
            "author": "integration-test",
            "prompt_format": "structured",
            "prompt_schema_version": 1,
            "prompt_definition": _make_prompt_definition_payload(),
            "keywords": ["classification"],
        },
        headers=auth_headers,
    )

    assert create_response.status_code == 201, create_response.text

    search_response = test_client.post(
        "/api/v1/prompts/search",
        params={"search_query": "mauveflint"},
        headers=auth_headers,
    )

    assert search_response.status_code == 200, search_response.text
    body = search_response.json()
    assert body["total_matches"] == 1
    assert body["items"][0]["name"] == "Structured Classifier"
