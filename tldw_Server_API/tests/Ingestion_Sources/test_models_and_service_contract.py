from __future__ import annotations

import inspect

import pytest


@pytest.mark.unit
def test_create_source_normalizes_enums_and_defaults():
    from tldw_Server_API.app.core.Ingestion_Sources.service import normalize_source_payload

    payload = normalize_source_payload(
        {
            "source_type": "local_directory",
            "sink_type": "media",
            "policy": "canonical",
            "enabled": None,
        }
    )

    assert payload["source_type"] == "local_directory"
    assert payload["sink_type"] == "media"
    assert payload["policy"] == "canonical"
    assert payload["enabled"] is True


@pytest.mark.unit
def test_normalize_source_payload_rejects_string_boolean_enabled():
    from tldw_Server_API.app.core.Ingestion_Sources.service import normalize_source_payload
    from tldw_Server_API.app.core.exceptions import IngestionSourceValidationError

    with pytest.raises(IngestionSourceValidationError, match="enabled must be a boolean"):
        normalize_source_payload(
            {
                "source_type": "local_directory",
                "sink_type": "media",
                "enabled": "false",
            }
        )


@pytest.mark.unit
def test_normalize_source_payload_rejects_string_boolean_schedule_enabled():
    from tldw_Server_API.app.core.Ingestion_Sources.service import normalize_source_payload
    from tldw_Server_API.app.core.exceptions import IngestionSourceValidationError

    with pytest.raises(IngestionSourceValidationError, match="schedule_enabled must be a boolean"):
        normalize_source_payload(
            {
                "source_type": "local_directory",
                "sink_type": "media",
                "schedule_enabled": "true",
            }
        )


@pytest.mark.unit
def test_ingestion_source_schema_models_have_docstrings():
    from tldw_Server_API.app.api.v1.schemas.ingestion_sources import (
        IngestionSourceCreateRequest,
        IngestionSourceItemResponse,
        IngestionSourcePatchRequest,
        IngestionSourceResponse,
        IngestionSourceSyncTriggerResponse,
    )

    missing = [
        schema.__name__
        for schema in (
            IngestionSourceCreateRequest,
            IngestionSourcePatchRequest,
            IngestionSourceResponse,
            IngestionSourceItemResponse,
            IngestionSourceSyncTriggerResponse,
        )
        if not schema.__doc__ or not schema.__doc__.strip()
    ]

    assert missing == []
