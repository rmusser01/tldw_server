import base64

import pytest

from tldw_Server_API.app.core.Utils import chunked_image_processor as cip


@pytest.mark.asyncio
async def test_decode_base64_image_chunked_handles_boundary():
    data = b"abcdefghijklmnopqrstuvwxyz0123456789"
    b64 = base64.b64encode(data).decode("ascii")

    parts = []
    async for chunk in cip.decode_base64_image_chunked(b64, chunk_size=7):
        parts.append(chunk)

    assert b"".join(parts) == data


@pytest.mark.asyncio
async def test_streaming_image_processor_preserves_bytes_without_pil(monkeypatch):
    monkeypatch.setattr(cip, "PIL_AVAILABLE", False)

    data = b"not-really-an-image"
    b64 = base64.b64encode(data).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"

    processor = cip.StreamingImageProcessor(max_memory_mb=1)
    ok, image_data, mime_type, error = await processor.process_image_url(
        data_url,
        max_size_bytes=len(data) + 10,
    )

    assert ok is True
    assert image_data == data
    assert mime_type == "image/png"
    assert error == ""


@pytest.mark.asyncio
async def test_streaming_image_processor_rejects_non_image_mime_type():
    data = b"plain text payload"
    b64 = base64.b64encode(data).decode("ascii")
    data_url = f"data:text/plain;base64,{b64}"

    processor = cip.StreamingImageProcessor(max_memory_mb=1)
    ok, image_data, mime_type, error = await processor.process_image_url(
        data_url,
        max_size_bytes=len(data) + 10,
    )

    assert ok is False
    assert image_data is None
    assert mime_type == "text/plain"
    assert "Unsupported image MIME type" in error


@pytest.mark.asyncio
@pytest.mark.skipif(not cip.PIL_AVAILABLE, reason="PIL not available")
async def test_streaming_image_processor_rejects_invalid_image_payload():
    # Valid base64, but not a parseable image payload.
    data = b"not-an-image" * 1000
    b64 = base64.b64encode(data).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"

    processor = cip.StreamingImageProcessor(max_memory_mb=2)
    ok, image_data, mime_type, error = await processor.process_image_url(
        data_url,
        max_size_bytes=len(data) + 100,
    )

    assert ok is False
    assert image_data is None
    assert mime_type == "image/png"
    assert "Invalid image format" in error


@pytest.mark.asyncio
async def test_process_image_chunked_rejects_original_pixel_count_before_resize(monkeypatch):
    class OversizedImage:
        width = cip.MAX_IMAGE_DIMENSION + 1
        height = cip.MAX_IMAGE_DIMENSION + 1
        mode = "RGB"

        def thumbnail(self, *_args, **_kwargs):
            raise AssertionError("thumbnail should not run before pixel limit validation")

    monkeypatch.setattr(cip, "PIL_AVAILABLE", True)
    monkeypatch.setattr(cip.Image, "open", lambda _payload: OversizedImage())

    with pytest.raises(ValueError, match="Image too large"):
        async for _chunk in cip.process_image_chunked(b"fake", "image/png"):
            pass
