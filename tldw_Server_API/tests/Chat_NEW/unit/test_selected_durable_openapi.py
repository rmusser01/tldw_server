"""Check the generated route contract without starting services or replacing them."""

import pytest
from fastapi import FastAPI

from tldw_Server_API.app.api.v1.endpoints import character_messages, chat

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def selected_openapi():
    app = FastAPI()
    app.include_router(chat.router, prefix="/api/v1/chat")
    app.include_router(character_messages.router, prefix="/api/v1")
    return app.openapi()


def resolve(spec, schema):
    while "$ref" in schema:
        schema = spec["components"]["schemas"][schema["$ref"].rsplit("/", 1)[1]]
    return schema


def test_selected_durable_operation_advertises_exact_v1(selected_openapi):
    operation = selected_openapi["paths"]["/api/v1/chat/completions"]["post"]
    assert operation.get("x-tldw-selected-durable-turn") == {
        "version": 1,
        "history": "h1_single_input_v1",
        "result": "rag_source_v1",
        "request_digest": "history_context_wire_v1",
        "recovery_read": "protected_live_v1",
        "inference_guarantee": "multiple_results_possible",
    }


def test_protected_result_schema_advertises_enforced_uuid(selected_openapi):
    result = selected_openapi["components"]["schemas"]["HistoryResultV1"]
    assert result["properties"]["result_message_id"].get("format") == "uuid"


def test_source_metadata_total_chunks_advertises_positive_safe_integer(selected_openapi):
    metadata = [
        schema for name, schema in selected_openapi["components"]["schemas"].items()
        if name.startswith("SourceMetadataV1")
    ]
    assert metadata
    for schema in metadata:
        total = schema["properties"]["total_chunks"]
        assert {key: total[key] for key in ("type", "minimum", "maximum")} == {
            "type": "integer", "minimum": 1, "maximum": 9007199254740991,
        }
        assert "total_chunks" not in schema.get("required", [])


@pytest.mark.parametrize("path", ["/api/v1/messages/{message_id}", "/api/v1/chats/{chat_id}/messages"])
def test_recovery_operations_advertise_protected_read(selected_openapi, path):
    operation = selected_openapi["paths"][path]["get"]
    assert operation.get("x-tldw-history-recovery-read") == {"version": 1, "projection": "protected_live_v1"}


def test_list_response_reaches_recovery_dto_without_filtering_legacy_formats(selected_openapi):
    operation = selected_openapi["paths"]["/api/v1/chats/{chat_id}/messages"]["get"]
    schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
    wrapper = resolve(selected_openapi, schema)
    message = resolve(selected_openapi, wrapper["properties"]["messages"]["items"])
    assert "tldw_history_recovery_v1" in message["properties"]
    queries = {item["name"]: item["schema"] for item in operation["parameters"] if item["in"] == "query"}
    assert {key: queries["limit"][key] for key in ("minimum", "maximum", "default")} == {
        "minimum": 1,
        "maximum": 200,
        "default": 50,
    }
    assert queries["offset"]["minimum"] == 0
    route = next(
        route
        for route in character_messages.router.routes
        if route.path == "/chats/{chat_id}/messages" and "GET" in route.methods
    )
    assert route.response_model is None
