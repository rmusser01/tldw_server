"""Saved chat reads reject compressed images before allocating oversized bitmaps."""

import base64
import io
import struct
import zlib

import pytest
from fastapi import HTTPException
from PIL import Image, ImageFile

from tldw_Server_API.app.api.v1.utils.chat_message_images import _complete_message_images
from tldw_Server_API.app.core.Utils import chunked_image_processor as processor
from tldw_Server_API.app.core.Utils import image_validation


def _png(size: tuple[int, int]) -> bytes:
    """Create compact complete pixels without a large RGB test allocation."""
    output = io.BytesIO()
    with Image.new("1", size) as image:
        image.save(output, format="PNG")
    return output.getvalue()


def _apng(png: bytes) -> bytes:
    """Add one valid animated frame with eager background disposal."""
    width, height = struct.unpack_from(">II", png, 16)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    animation = chunk(b"acTL", struct.pack(">II", 1, 0))
    frame = chunk(b"fcTL", struct.pack(">IIIIIHHBB", 0, width, height, 0, 0, 1, 100, 1, 0))
    return png[:33] + animation + frame + png[33:]


def _ico(frame: bytes) -> bytes:
    """Wrap an embedded raster whose actual dimensions need separate validation."""
    return struct.pack("<HHHBBBBHHII", 0, 1, 1, 0, 0, 0, 0, 1, 1, len(frame), 22) + frame


def _oversized_payload(container: str) -> tuple[bytes, str]:
    """Exercise each eager codec, including a forged small GIF canvas."""
    size = (4097, 4096)
    if container.startswith("gif"):
        output = io.BytesIO()
        with Image.new("P", size) as image:
            image.save(output, format="GIF", disposal=2, transparency=0)
        data = bytearray(output.getvalue())
        if container == "gif-frame":
            struct.pack_into("<HH", data, 6, 16, 16)
        return bytes(data), "image/gif"
    if container == "ico-dib":
        dib = struct.pack("<IiiHHIIiiII", 40, size[0], size[1] * 2, 1, 1, 0, 0, 0, 0, 2, 0)
        return _ico(dib + b"\x00\x00\x00\x00\xff\xff\xff\x00"), "image/x-icon"
    png = _png(size)
    if "apng" in container:
        png = _apng(png)
    return (_ico(png), "image/x-icon") if container.startswith("ico") else (png, "image/png")


@pytest.mark.parametrize("container", ["png", "apng", "gif", "gif-frame", "ico", "ico-apng", "ico-dib"])
def test_saved_image_pixel_limit_precedes_decode(monkeypatch: pytest.MonkeyPatch, container: str) -> None:
    """Neither decoding nor animation/container bitmap allocation may precede the cap."""
    data, mime = _oversized_payload(container)
    assert len(data) < 5 * 1024 * 1024
    assert processor.MAX_IMAGE_PIXELS < 4097 * 4096

    def reject_decode(*args: object, **kwargs: object) -> None:
        pytest.fail("Oversized saved image reached pixel decoding or allocation")

    monkeypatch.setattr(ImageFile.ImageFile, "load", reject_decode)
    monkeypatch.setattr(Image.core, "fill", reject_decode)
    monkeypatch.setattr(Image.core, "new", reject_decode)
    with pytest.raises(HTTPException) as error:
        _complete_message_images({"image_data": data, "image_mime_type": mime})
    assert error.value.status_code == 413


@pytest.mark.parametrize("container", ["PNG", "APNG", "GIF", "ICO", "DIB-ICO", "JPEG", "WEBP", "BMP"])
def test_saved_image_accepts_complete_small_raster(container: str) -> None:
    """Supported ordinary images, animated images, and both ICO frame formats remain readable."""
    output = io.BytesIO()
    with Image.new("RGBA" if container in {"ICO", "DIB-ICO"} else "RGB", (16, 16), "red") as image:
        image.save(
            output, format="ICO" if container == "DIB-ICO" else ("PNG" if container == "APNG" else container),
            sizes=[(16, 16)], bitmap_format="bmp" if container == "DIB-ICO" else "png", disposal=2,
        )
    data = _apng(output.getvalue()) if container == "APNG" else output.getvalue()
    mime = {"ICO": "image/x-icon", "DIB-ICO": "image/x-icon", "APNG": "image/png",
            "JPEG": "image/jpeg"}.get(container, f"image/{container.lower()}")
    assert _complete_message_images({"image_data": data, "image_mime_type": mime}) == [
        f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    ]


def test_saved_image_accepts_exact_pixel_ceiling() -> None:
    """The shared ceiling is inclusive and preserves a complete valid boundary image."""
    data = _png((4096, 4096))
    assert _complete_message_images({"image_data": data, "image_mime_type": "image/png"}) == [
        f"data:image/png;base64,{base64.b64encode(data).decode('ascii')}"
    ]


@pytest.mark.parametrize("consumer", ["base64", "upload", "process", "stream", "url"])
@pytest.mark.asyncio
async def test_core_consumers_reject_eager_pixels_without_allocating(
    monkeypatch: pytest.MonkeyPatch, consumer: str,
) -> None:
    """Core validators keep their public rejection contracts at the same eager-open boundary."""
    data, mime = _oversized_payload("apng")
    encoded = base64.b64encode(data).decode("ascii")

    def reject_decode(*args: object, **kwargs: object) -> None:
        pytest.fail("Oversized core image reached pixel decoding or allocation")

    monkeypatch.setattr(ImageFile.ImageFile, "load", reject_decode)
    monkeypatch.setattr(Image.core, "fill", reject_decode)
    if consumer == "base64":
        assert image_validation.safe_decode_base64_image(encoded, mime) is None
    elif consumer == "upload":
        assert image_validation.validate_uploaded_image_bytes(data, mime)[0] is False
    elif consumer == "process":
        with pytest.raises(ValueError, match="Image too large"):
            async for _ in processor.process_image_chunked(data, mime):
                pass
    elif consumer == "stream":
        async def chunks():
            yield data
        assert (await processor.validate_and_process_image_stream(chunks(), mime, len(data) + 1))[0] is False
    else:
        assert (await processor.StreamingImageProcessor().process_image_url(
            f"data:{mime};base64,{encoded}", len(data) + 1,
        ))[0] is False


@pytest.mark.parametrize("container", ["png", "gif", "ico"])
def test_truncated_eager_metadata_retains_controlled_invalid_image_error(container: str) -> None:
    """Every prefix before the required metadata is complete fails with HTTP 409."""
    if container == "png":
        data, mime, metadata_end = _png((16, 16)), "image/png", 33
    elif container == "ico":
        data, mime = _ico(_png((16, 16))), "image/x-icon"
        metadata_end = len(data)  # The declared frame must fit in the ICO container.
    else:
        output = io.BytesIO()
        with Image.new("P", (16, 16)) as image:
            image.save(output, format="GIF", disposal=2, transparency=0)
        data, mime = output.getvalue(), "image/gif"
        metadata_end = data.index(b"\x2c") + 10  # First image descriptor and its nine-byte body.
    for length in range(12, metadata_end):
        with pytest.raises(HTTPException) as error:
            _complete_message_images({"image_data": data[:length], "image_mime_type": mime})
        assert error.value.status_code == 409


@pytest.mark.parametrize("icon", [False, True], ids=["png", "ico-png"])
def test_duplicate_png_canvas_header_is_invalid_before_allocation(monkeypatch, icon: bool) -> None:
    """Pillow must not replace a small first canvas with a larger duplicate header."""
    data = _apng(_png((4097, 4096)))
    data = data[:8] + _png((16, 16))[8:33] + data[8:]
    if icon:
        data = _ico(data)

    def reject_allocate(*args, **kwargs):
        pytest.fail("Duplicate PNG header reached pixel allocation")

    monkeypatch.setattr(Image.core, "fill", reject_allocate)
    monkeypatch.setattr(Image.core, "new", reject_allocate)
    monkeypatch.setattr(ImageFile.ImageFile, "load", reject_allocate)
    with pytest.raises(HTTPException) as error:
        _complete_message_images({"image_data": data, "image_mime_type": "image/x-icon" if icon else "image/png"})
    assert error.value.status_code == 409


@pytest.mark.parametrize("container", ["lossy", "lossless", "animated"])
def test_webp_canvas_limit_precedes_native_decoder(monkeypatch, container: str) -> None:
    """The native WebP constructor allocates its canvas before Python obtains dimensions."""
    from PIL import WebPImagePlugin

    output = io.BytesIO()
    with Image.new("RGB", (16, 16), "red") as first, Image.new("RGB", (16, 16), "blue") as second:
        first.save(output, format="WEBP", lossless=container == "lossless",
                   save_all=container == "animated", append_images=[second] if container == "animated" else [])
    data = bytearray(output.getvalue())
    if container == "animated":
        data[24:27] = (4097 - 1).to_bytes(3, "little")
        data[27:30] = (4096 - 1).to_bytes(3, "little")
    elif container == "lossless":
        bits = struct.unpack_from("<I", data, 21)[0]
        struct.pack_into("<I", data, 21, (bits & 0xF0000000) | (4097 - 1) | ((4096 - 1) << 14))
    else:
        struct.pack_into("<HH", data, 26, 4097, 4096)

    def reject_decoder(*args, **kwargs):
        pytest.fail("Oversized WebP reached its allocating native decoder")

    monkeypatch.setattr(WebPImagePlugin._webp, "WebPAnimDecoder", reject_decoder)
    with pytest.raises(HTTPException) as error:
        _complete_message_images({"image_data": bytes(data), "image_mime_type": "image/webp"})
    assert error.value.status_code == 413


@pytest.mark.parametrize("bitmap_format", ["png", "bmp"])
def test_saved_image_preserves_default_selection_of_multi_frame_icon(bitmap_format: str) -> None:
    """Metadata bounds match the one default frame that Pillow will actually load."""
    output = io.BytesIO()
    with Image.new("RGBA", (64, 64), "red") as image:
        image.save(output, format="ICO", sizes=[(16, 16), (32, 32), (64, 64)], bitmap_format=bitmap_format)
    data = output.getvalue()
    assert _complete_message_images({"image_data": data, "image_mime_type": "image/x-icon"}) == [
        f"data:image/x-icon;base64,{base64.b64encode(data).decode('ascii')}"
    ]
