import pytest

from tldw_Server_API.app.api.v1.endpoints.media import ingest_jobs


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        self.debugs.append((message, args, kwargs))


def test_cleanup_dir_failure_log_is_sanitized(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(ingest_jobs, "logger", logger_stub)

    def _raise_cleanup(path: str, *, ignore_errors: bool = False) -> None:
        assert path == "/private/ingest/leaked"
        assert ignore_errors is True
        raise RuntimeError("cleanup exploded at /private/ingest/leaked")

    monkeypatch.setattr(ingest_jobs.shutil, "rmtree", _raise_cleanup)

    ingest_jobs._cleanup_dir("/private/ingest/leaked")

    assert logger_stub.debugs == [("Failed to cleanup media ingest temp dir", (), {})]
    rendered = " ".join([logger_stub.debugs[0][0], *(str(arg) for arg in logger_stub.debugs[0][1])])
    assert "/private/ingest/leaked" not in rendered
    assert "exploded" not in rendered
