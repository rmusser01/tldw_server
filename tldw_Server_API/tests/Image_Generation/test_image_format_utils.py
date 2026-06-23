import base64

import pytest

from tldw_Server_API.app.core.Image_Generation.adapters import image_format_utils
from tldw_Server_API.app.core.Image_Generation.exceptions import ImageGenerationError


def test_fetch_image_bytes_rejects_content_length_over_max(monkeypatch):
    class FakeResponse:
        status_code = 200
        headers = {
            "content-length": "10",
            "content-type": "image/png",
        }

        @property
        def content(self):
            raise AssertionError("oversized responses should be rejected before content is read")

        def close(self):
            return None

    monkeypatch.setattr(image_format_utils, "fetch", lambda **_kwargs: FakeResponse())

    with pytest.raises(ImageGenerationError, match="too large"):
        image_format_utils.fetch_image_bytes("https://cdn.example/image.png", timeout=1, max_bytes=3)


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
