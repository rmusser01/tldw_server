"""Audio pagination cursors reject tampered input instead of silently repairing it.

Both decoders used a bare urlsafe_b64decode, which DISCARDS out-of-alphabet characters,
so "<valid cursor>!!!!" decoded to the valid position. Both routes already map ValueError
to 400 "Invalid cursor" (Docs/API-related/Pagination_Cursors.md).
"""

import pytest

from tldw_Server_API.app.api.v1.endpoints.audio.audio_history import (
    _decode_cursor as decode_history_cursor,
    _encode_cursor as encode_history_cursor,
)
from tldw_Server_API.app.api.v1.endpoints.audio.audio_jobs import (
    _decode_audio_jobs_cursor,
    _encode_audio_jobs_cursor,
)

pytestmark = pytest.mark.unit


def test_history_cursor_roundtrips_and_rejects_junk() -> None:
    cursor = encode_history_cursor("2026-01-01T00:00:00+00:00", 7)
    assert decode_history_cursor(cursor) == ("2026-01-01T00:00:00+00:00", 7)
    with pytest.raises(ValueError):
        decode_history_cursor(cursor + "!!!!")


def test_audio_jobs_cursor_roundtrips_and_rejects_junk() -> None:
    cursor = _encode_audio_jobs_cursor("2026-01-01T00:00:00+00:00", 7)
    created_at, job_id = _decode_audio_jobs_cursor(cursor)
    assert (created_at.year, job_id) == (2026, 7)
    with pytest.raises(ValueError):
        _decode_audio_jobs_cursor(cursor + "!!!!")
