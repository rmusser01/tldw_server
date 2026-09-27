import pytest

from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint


@pytest.mark.unit
def test_normalize_message_timestamp_uses_fixed_utc_milliseconds():
    assert (
        chat_endpoint._normalize_message_timestamp("2026-08-23T04:28:54Z")
        == "2026-08-23T04:28:54.000Z"
    )
    assert (
        chat_endpoint._normalize_message_timestamp("2026-08-23T04:28:54.179123+00:00")
        == "2026-08-23T04:28:54.179Z"
    )
