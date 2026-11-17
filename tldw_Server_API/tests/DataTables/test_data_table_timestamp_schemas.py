"""Public timestamp contracts across native PostgreSQL and SQLite values."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.data_tables_schemas import DataTableJobStatus, DataTableSummary

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "model,fields,payload",
    [
        (
            DataTableSummary,
            ("created_at", "updated_at", "last_modified"),
            {
                "uuid": "table",
                "name": "Rowan",
                "prompt": "Keep facts",
                "status": "ready",
                "row_count": 3,
            },
        ),
        (
            DataTableJobStatus,
            ("created_at", "started_at", "completed_at", "cancelled_at"),
            {
                "id": 1,
                "uuid": "job",
                "status": "queued",
                "job_type": "data_table_generate",
                "owner_user_id": "401",
                "cancellation_reason": None,
                "progress_percent": None,
                "progress_message": None,
                "result": None,
                "error_message": None,
            },
        ),
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        datetime(2026, 10, 27, 15, 16, 17, 123456, timezone(timedelta(hours=5, minutes=30))),
        datetime(2026, 10, 27, 15, 16, 17, 123456),
        "2026-10-27T15:16:17.123456Z",
        None,
    ],
)
def test_timestamp_fields_preserve_instants_strings_and_null(model, fields, payload, value):
    response = model.model_validate({**payload, **dict.fromkeys(fields, value)})
    expected = value.isoformat() if isinstance(value, datetime) else value
    assert {name: response.model_dump(mode="json")[name] for name in fields} == dict.fromkeys(
        fields, expected
    )  # nosec B101 -- exact public timestamp representation.


@pytest.mark.parametrize(
    "model,fields,payload",
    [
        (
            DataTableSummary,
            ("created_at", "updated_at", "last_modified"),
            {
                "uuid": "table",
                "name": "Rowan",
                "prompt": "Keep facts",
                "status": "ready",
                "row_count": 3,
            },
        ),
        (
            DataTableJobStatus,
            ("created_at", "started_at", "completed_at", "cancelled_at"),
            {
                "id": 1,
                "uuid": "job",
                "status": "queued",
                "job_type": "data_table_generate",
                "owner_user_id": "401",
                "cancellation_reason": None,
                "progress_percent": None,
                "progress_message": None,
                "result": None,
                "error_message": None,
            },
        ),
    ],
)
def test_timestamp_fields_keep_invalid_numbers_rejected_and_openapi_strings(model, fields, payload):
    with pytest.raises(ValidationError):
        model.model_validate({**payload, **dict.fromkeys(fields, 42)})
    properties = model.model_json_schema()["properties"]
    assert all(
        properties[name]["anyOf"] == [{"type": "string"}, {"type": "null"}] for name in fields
    )  # nosec B101 -- validators retain the existing OpenAPI string/null contract.
