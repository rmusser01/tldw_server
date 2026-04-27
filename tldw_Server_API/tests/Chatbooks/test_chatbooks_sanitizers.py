import pytest


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []
        self.warnings: list[str] = []

    def debug(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.debugs.append(message)

    def warning(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.warnings.append(message)


def test_safe_increment_metric_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import chatbooks

    def _raise_increment_counter(*_args, **_kwargs):
        raise RuntimeError("chatbook metrics exploded at /private/chatbooks.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(chatbooks, "increment_counter", _raise_increment_counter)
    monkeypatch.setattr(chatbooks, "logger", logger_stub)

    chatbooks._safe_increment_metric(
        "chatbooks.secret.metric",
        labels={"user_id": "secret-user"},
        error_context="private-export-context",
    )

    assert logger_stub.debugs == ["metrics increment failed"]
    assert "chatbooks.secret.metric" not in str(logger_stub.debugs)
    assert "secret-user" not in str(logger_stub.debugs)
    assert "private-export-context" not in str(logger_stub.debugs)
    assert "chatbook metrics exploded" not in str(logger_stub.debugs)
    assert "/private/chatbooks.db" not in str(logger_stub.debugs)


def test_persist_completed_sync_export_job_failure_logs_are_sanitized(monkeypatch, tmp_path):
    from tldw_Server_API.app.api.v1.endpoints import chatbooks

    export_file = tmp_path / "secret-export.zip"
    export_file.write_bytes(b"zip")

    class _FailingService:
        export_dir = tmp_path

        def _get_export_expiry(self, now):
            return now

        def _get_download_expiry(self, now, expires_at):
            return expires_at

        def _build_download_url(self, job_id, download_expires_at):
            return f"/api/v1/chatbooks/download/{job_id}"

        def _save_export_job(self, job):
            raise RuntimeError("chatbook job store leaked /private/chatbooks-jobs.db")

    def _failing_unlink(self):
        raise RuntimeError("cleanup leaked /private/secret-export.zip")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(chatbooks, "logger", logger_stub)
    monkeypatch.setattr(chatbooks.Path, "unlink", _failing_unlink)

    with pytest.raises(chatbooks.HTTPException) as exc_info:
        chatbooks._persist_completed_sync_export_job(
            _FailingService(),
            user_id="secret-user",
            chatbook_name="private-chatbook",
            output_path=export_file,
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Export completed but failed to persist job metadata"
    assert logger_stub.warnings == [
        "Failed to persist completed export job for sync path",
        "Failed to remove export archive after job persistence failure",
    ]
    assert "secret-user" not in str(logger_stub.warnings)
    assert "private-chatbook" not in str(logger_stub.warnings)
    assert "chatbook job store leaked" not in str(logger_stub.warnings)
    assert "cleanup leaked" not in str(logger_stub.warnings)
    assert "/private/" not in str(logger_stub.warnings)
