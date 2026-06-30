from types import ModuleType
import sys

import pytest

from tldw_Server_API.app.api.v1.endpoints import ocr as ocr_endpoint


class _LoggerStub:
    def __init__(self):
        self.errors = []

    def error(self, *args, **kwargs):
        self.errors.append((args, kwargs))


def _assert_sanitized_error_log(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.errors
    args, kwargs = logger_stub.errors[-1]
    rendered = " ".join(str(arg) for arg in args)

    assert args == (expected_message,)
    assert "points loader exploded" not in rendered
    assert "/private/models/points" not in rendered
    assert "points loader exploded" not in str(kwargs)
    assert "/private/models/points" not in str(kwargs)


@pytest.mark.unit
def test_preload_points_transformers_sanitizes_loader_failure(monkeypatch):
    module_name = (
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.points_reader"
    )
    fake_points_reader = ModuleType(module_name)

    def _fail_load_transformers():
        raise RuntimeError("points loader exploded at /private/models/points")

    fake_points_reader._load_transformers = _fail_load_transformers
    monkeypatch.setitem(sys.modules, module_name, fake_points_reader)
    logger_stub = _LoggerStub()
    monkeypatch.setattr(ocr_endpoint, "logging", logger_stub)

    result = ocr_endpoint.preload_points_transformers()

    assert result == {"status": "error", "error": "POINTS OCR preload failed"}
    assert "points loader exploded" not in str(result)
    assert "/private/models/points" not in str(result)
    _assert_sanitized_error_log(logger_stub, "POINTS transformers preload failed")
