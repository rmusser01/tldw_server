import os
import pytest


@pytest.mark.unit
def test_points_backend_available_returns_bool():
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.points_reader import (
        PointsReaderBackend,
    )
    assert isinstance(PointsReaderBackend.available(), bool)


@pytest.mark.unit
def test_points_backend_sglang_mock(monkeypatch):
    # Force SGLang mode and stub requests.post
    monkeypatch.setenv("POINTS_MODE", "sglang")
    monkeypatch.setenv("POINTS_SGLANG_URL", "http://127.0.0.1:9999/v1/chat/completions")
    monkeypatch.setenv("POINTS_SGLANG_MODEL", "WePoints")

    class DummyResp:
        status_code = 200
        text = "{\"choices\":[{\"message\":{\"content\":\"MOCK_TEXT\"}}]}"

        def raise_for_status(self):

            return None

        def json(self):

            import json as _json
            return _json.loads(self.text)

    # Patch the correct call site used by points backend (http_client.fetch_json)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.http_client.fetch_json",
        lambda **kwargs: {
            "choices": [{"message": {"content": "MOCK_TEXT"}}]
        },
    )

    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.registry import (
        get_backend,
    )

    backend = get_backend("points")
    assert backend is not None

    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        b"\x00\x00\x00\x0bIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\x0d\n\x2d\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    out = backend.ocr_image(png_bytes, lang="eng")
    assert isinstance(out, str) and out == "MOCK_TEXT"


@pytest.mark.unit
def test_points_transformers_requires_wepoints(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.points_reader import (
        PointsReaderBackend,
    )

    monkeypatch.setenv("POINTS_MODE", "transformers")

    def fake_find_spec(name):
        if name in {"transformers", "torch"}:
            return object()
        if name == "wepoints":
            return None
        raise AssertionError(f"unexpected module probe: {name}")

    monkeypatch.setattr("importlib.util.find_spec", fake_find_spec)

    assert PointsReaderBackend.available() is False


@pytest.mark.unit
def test_points_transformers_missing_optional_dependency_returns_empty(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends import (
        points_reader,
    )

    monkeypatch.setenv("POINTS_MODE", "transformers")
    monkeypatch.setattr(points_reader.PointsReaderBackend, "available", classmethod(lambda cls: True))
    monkeypatch.setattr(
        points_reader,
        "_ocr_via_transformers",
        lambda image_path, prompt: (_ for _ in ()).throw(ModuleNotFoundError("No module named 'wepoints'")),
    )

    out = points_reader.PointsReaderBackend().ocr_image(b"not-a-real-image", lang="eng")

    assert out == ""


@pytest.mark.unit
def test_points_sglang_import_error_does_not_fall_through_to_transformers(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends import (
        points_reader,
    )

    monkeypatch.setenv("POINTS_MODE", "sglang")
    monkeypatch.setenv("POINTS_SGLANG_URL", "http://127.0.0.1:9999/v1/chat/completions")

    def fail_sglang(image_path, prompt):
        raise ModuleNotFoundError("No module named 'requests'")

    def fail_transformers(image_path, prompt):
        pytest.fail("SGLang import failures must not fall through to transformers")

    monkeypatch.setattr(points_reader, "_ocr_via_sglang", fail_sglang)
    monkeypatch.setattr(points_reader, "_ocr_via_transformers", fail_transformers)

    with pytest.raises(ModuleNotFoundError):
        points_reader.PointsReaderBackend().ocr_image(b"not-a-real-image", lang="eng")
