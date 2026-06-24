import base64

import pytest

from tldw_Server_API.app.core.Image_Generation.adapters import image_format_utils
from tldw_Server_API.app.core.Image_Generation.exceptions import ImageGenerationError


class _FakeStreamResponse:
    def __init__(self, *, headers, chunks):
        self.status_code = 200
        self.headers = headers
        self.url = "https://cdn.example/image.png"
        self._chunks = chunks
        self.closed = False
        self.content_accessed = False

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        self.closed = True
        return False

    @property
    def content(self):
        self.content_accessed = True
        raise AssertionError("streaming fetch should not access response.content")

    def iter_bytes(self):
        yield from self._chunks


class _FakeStreamClient:
    def __init__(self, response):
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        return False

    def stream(self, *_args, **_kwargs):
        return self.response


def _patch_stream_client(monkeypatch, response):
    monkeypatch.setattr(image_format_utils, "_validate_egress_or_raise", lambda _url: None)
    monkeypatch.setattr(image_format_utils, "create_client", lambda **_kwargs: _FakeStreamClient(response))


def test_decode_data_url_allows_base64_whitespace():
    encoded = base64.b64encode(b"image-bytes").decode("ascii")
    spaced = f"{encoded[:4]}\n {encoded[4:]}"

    content, content_type = image_format_utils.decode_data_url(f"data:image/png;base64,{spaced}")

    assert content == b"image-bytes"
    assert content_type == "image/png"


def test_fetch_image_bytes_rejects_content_length_over_max(monkeypatch):
    response = _FakeStreamResponse(
        headers={
            "content-length": "10",
            "content-type": "image/png",
        },
        chunks=[b"oversized"],
    )
    _patch_stream_client(monkeypatch, response)

    with pytest.raises(ImageGenerationError, match="too large"):
        image_format_utils.fetch_image_bytes("https://cdn.example/image.png", timeout=1, max_bytes=3)

    assert response.closed is True
    assert response.content_accessed is False


def test_fetch_image_bytes_rejects_stream_over_max_without_content_length(monkeypatch):
    response = _FakeStreamResponse(
        headers={"content-type": "image/png"},
        chunks=[b"12", b"34"],
    )
    _patch_stream_client(monkeypatch, response)

    with pytest.raises(ImageGenerationError, match="too large"):
        image_format_utils.fetch_image_bytes("https://cdn.example/image.png", timeout=1, max_bytes=3)

    assert response.closed is True
    assert response.content_accessed is False


def test_decode_base64_image_rejects_content_over_max():
    encoded = base64.b64encode(b"1234").decode("ascii")

    with pytest.raises(ImageGenerationError, match="too large"):
        image_format_utils.decode_base64_image(encoded, max_bytes=3)


def test_validate_and_convert_image_output_rejects_unknown_png_bytes():
    with pytest.raises(ImageGenerationError, match="invalid image content"):
        image_format_utils.validate_and_convert_image_output(
            b"hello",
            "image/png",
            "png",
            max_bytes=4_000_000,
        )
