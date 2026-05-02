from __future__ import annotations

from tldw_Server_API.app.api.v1.schemas.monitoring_schemas import MarkReadResponse


def test_mark_read_response_schema_describes_minimal_mutation_contract() -> None:
    schema = MarkReadResponse.model_json_schema()
    assert set(schema["properties"]) == {"status", "id", "item"}
    assert set(schema["required"]) == {"status", "id", "item"}

    status_description = schema["properties"]["status"]["description"].lower()
    id_description = schema["properties"]["id"]["description"].lower()
    item_description = schema["properties"]["item"]["description"].lower()

    assert "mutation succeeded" in status_description
    assert "authoritative merged state" in status_description
    assert "runtime alert row id" in id_description
    assert "authoritative merged alert state" in item_description
