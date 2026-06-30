from __future__ import annotations

from io import BytesIO

import pytest

from tldw_Server_API.app.core.VN_Platform.idempotency import (
    canonical_multipart_payload_hash,
    canonical_payload_hash,
    stream_sha256,
)

pytestmark = pytest.mark.unit


def test_canonical_payload_hash_is_order_independent() -> None:
    left = canonical_payload_hash({"b": 2, "a": {"z": True, "y": ["x"]}})
    right = canonical_payload_hash({"a": {"y": ["x"], "z": True}, "b": 2})

    assert left == right


def test_canonical_payload_hash_changes_when_payload_changes() -> None:
    first = canonical_payload_hash({"slot_id": 1, "variant_count": 2})
    second = canonical_payload_hash({"slot_id": 1, "variant_count": 3})

    assert first != second


def test_stream_sha256_reads_file_like_stream_from_current_position() -> None:
    stream = BytesIO(b"prefix-vn-asset-bytes")
    stream.seek(len(b"prefix-"))

    digest = stream_sha256(stream)

    assert digest == "812e2fa75f9e1a34a8e72c691d1b448549f1c6d1ca81808bd9845a6cf03f38de"
    assert stream.tell() == len(b"prefix-vn-asset-bytes")


def test_canonical_multipart_payload_hash_includes_fields_and_file_digest() -> None:
    first = canonical_multipart_payload_hash(
        {"slot_id": "1", "review_status": "draft"},
        file_sha256="abc123",
        filename="sprite.png",
        content_type="image/png",
    )
    second = canonical_multipart_payload_hash(
        {"review_status": "draft", "slot_id": "1"},
        file_sha256="abc123",
        filename="sprite.png",
        content_type="image/png",
    )
    changed = canonical_multipart_payload_hash(
        {"review_status": "draft", "slot_id": "1"},
        file_sha256="changed",
        filename="sprite.png",
        content_type="image/png",
    )

    assert first == second
    assert first != changed
