from __future__ import annotations

import pytest

from tldw_Server_API.app.api.v1.endpoints.monitoring import _decode_alert_metadata


@pytest.mark.parametrize(
    ("raw_metadata", "expected_metadata"),
    [
        ("", None),
        ("null", None),
        ('{"host": "api-1"}', {"host": "api-1"}),
        ('["tag-a", "tag-b"]', {"value": ["tag-a", "tag-b"]}),
        ("not-json", {"raw": "not-json"}),
    ],
)
def test_decode_alert_metadata_normalizes_runtime_database_values(
    raw_metadata: object,
    expected_metadata: object,
) -> None:
    decoded = _decode_alert_metadata({"id": 1, "metadata": raw_metadata})

    assert decoded["metadata"] == expected_metadata  # nosec B101
