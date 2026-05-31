import importlib

import pytest


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str) -> None:
        self.warnings.append(str(message))


def test_query_features_treats_test_mode_y_as_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_MODE", "y")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")
    monkeypatch.delenv("ALLOW_NLTK_DOWNLOADS", raising=False)

    module = importlib.import_module("tldw_Server_API.app.core.RAG.rag_service.query_features")
    module = importlib.reload(module)

    assert module._TEST_MODE is True
    assert module._ALLOW_NLTK_DOWNLOADS is False


def test_nltk_download_error_warning_omits_resource_and_exception_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("tldw_Server_API.app.core.RAG.rag_service.query_features")
    logger_stub = _LoggerStub()

    def fail_download(_resource: str, quiet: bool) -> bool:
        raise RuntimeError("download failed for /private/nltk-data/punkt?token=secret-token")

    monkeypatch.setattr(module, "logger", logger_stub)
    monkeypatch.setattr(module.nltk, "download", fail_download)

    ok = module._download_with_timeout("/private/resource?token=secret-token", timeout_s=1)

    assert ok is False
    assert logger_stub.warnings == ["NLTK download error; continuing without resource"]
    joined = "\n".join(logger_stub.warnings)
    assert "/private/" not in joined
    assert "secret-token" not in joined
    assert "download failed" not in joined
