import pytest


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.exceptions: list[str] = []

    def debug(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.debugs.append(message)

    def warning(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.warnings.append(message)

    def error(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.errors.append(message)

    def exception(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.exceptions.append(message)


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


@pytest.mark.asyncio
async def test_chatbook_job_read_failure_logs_are_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import chatbooks

    class _FailingJobService:
        def count_export_jobs(self):
            raise RuntimeError("export list leaked /private/chatbooks-export-jobs.db")

        def get_export_job(self, job_id):
            raise RuntimeError(f"export get leaked {job_id} /private/chatbooks-export-job.db")

        def count_import_jobs(self):
            raise RuntimeError("import list leaked /private/chatbooks-import-jobs.db")

        def get_import_job(self, job_id):
            raise RuntimeError(f"import get leaked {job_id} /private/chatbooks-import-job.db")

    user = chatbooks.User(id=1, username="tester", email=None, is_active=True)
    service = _FailingJobService()
    logger_stub = _LoggerStub()
    monkeypatch.setattr(chatbooks, "logger", logger_stub)

    with pytest.raises(chatbooks.HTTPException) as export_list_exc:
        await chatbooks.list_export_jobs(request=None, service=service, user=user)
    with pytest.raises(chatbooks.HTTPException) as export_get_exc:
        await chatbooks.get_export_job("secret-export-job", service=service, user=user)
    with pytest.raises(chatbooks.HTTPException) as import_list_exc:
        await chatbooks.list_import_jobs(request=None, service=service, user=user)
    with pytest.raises(chatbooks.HTTPException) as import_get_exc:
        await chatbooks.get_import_job("secret-import-job", service=service, user=user)

    assert export_list_exc.value.status_code == 500
    assert export_list_exc.value.detail == "An error occurred while retrieving export jobs"
    assert export_get_exc.value.status_code == 500
    assert export_get_exc.value.detail == "An error occurred while retrieving the export job"
    assert import_list_exc.value.status_code == 500
    assert import_list_exc.value.detail == "An error occurred while retrieving import jobs"
    assert import_get_exc.value.status_code == 500
    assert import_get_exc.value.detail == "An error occurred while retrieving the import job"
    assert logger_stub.errors == [
        "Failed to list chatbook export jobs",
        "Failed to get chatbook export job",
        "Failed to list chatbook import jobs",
        "Failed to get chatbook import job",
    ]
    assert logger_stub.exceptions == []
    assert "secret-" not in str(logger_stub.errors)
    assert "tester" not in str(logger_stub.errors)
    assert "/private/" not in str(logger_stub.errors)


@pytest.mark.asyncio
async def test_chatbook_job_mutation_failure_logs_are_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import chatbooks

    class _FailingMutationService:
        def cleanup_expired_exports(self):
            raise RuntimeError("cleanup leaked /private/chatbooks-cleanup.db")

        def cancel_export_job(self, job_id):
            raise RuntimeError(f"cancel export leaked {job_id} /private/chatbooks-export-cancel.db")

        def cancel_import_job(self, job_id):
            raise RuntimeError(f"cancel import leaked {job_id} /private/chatbooks-import-cancel.db")

        def delete_export_job(self, job_id):
            raise RuntimeError(f"remove export leaked {job_id} /private/chatbooks-export-remove.db")

        def delete_import_job(self, job_id):
            raise RuntimeError(f"remove import leaked {job_id} /private/chatbooks-import-remove.db")

    class _AuditService:
        async def log_event(self, *args, **kwargs):
            return None

    user = chatbooks.User(id=1, username="tester", email=None, is_active=True)
    service = _FailingMutationService()
    audit_service = _AuditService()
    logger_stub = _LoggerStub()
    monkeypatch.setattr(chatbooks, "logger", logger_stub)

    with pytest.raises(chatbooks.HTTPException) as cleanup_exc:
        await chatbooks.cleanup_expired_exports(service=service, user=user, audit_service=audit_service)
    with pytest.raises(chatbooks.HTTPException) as cancel_export_exc:
        await chatbooks.cancel_export_job(
            "secret-export-cancel",
            request=None,
            service=service,
            user=user,
            audit_service=audit_service,
        )
    with pytest.raises(chatbooks.HTTPException) as cancel_import_exc:
        await chatbooks.cancel_import_job(
            "secret-import-cancel",
            request=None,
            service=service,
            user=user,
            audit_service=audit_service,
        )
    with pytest.raises(chatbooks.HTTPException) as remove_export_exc:
        await chatbooks.remove_export_job(
            "secret-export-remove",
            request=None,
            service=service,
            user=user,
            audit_service=audit_service,
        )
    with pytest.raises(chatbooks.HTTPException) as remove_import_exc:
        await chatbooks.remove_import_job(
            "secret-import-remove",
            request=None,
            service=service,
            user=user,
            audit_service=audit_service,
        )

    assert cleanup_exc.value.status_code == 500
    assert cleanup_exc.value.detail == "An error occurred while cleaning up expired exports"
    assert cancel_export_exc.value.status_code == 500
    assert cancel_export_exc.value.detail == "An error occurred while cancelling the export job"
    assert cancel_import_exc.value.status_code == 500
    assert cancel_import_exc.value.detail == "An error occurred while cancelling the import job"
    assert remove_export_exc.value.status_code == 500
    assert remove_export_exc.value.detail == "An error occurred while removing the export job"
    assert remove_import_exc.value.status_code == 500
    assert remove_import_exc.value.detail == "An error occurred while removing the import job"
    assert logger_stub.errors == [
        "Failed to clean up expired chatbook exports",
        "Failed to cancel chatbook export job",
        "Failed to cancel chatbook import job",
        "Failed to remove chatbook export job",
        "Failed to remove chatbook import job",
    ]
    assert logger_stub.exceptions == []
    assert "secret-" not in str(logger_stub.errors)
    assert "tester" not in str(logger_stub.errors)
    assert "/private/" not in str(logger_stub.errors)


@pytest.mark.asyncio
async def test_chatbook_download_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import chatbooks

    class _FailingDownloadService:
        _jobs_backend = "core"

        def get_export_job(self, job_id):
            raise RuntimeError(f"download leaked {job_id} /private/chatbooks-download.db")

    user = chatbooks.User(id=1, username="tester", email=None, is_active=True)
    logger_stub = _LoggerStub()
    monkeypatch.setattr(chatbooks, "logger", logger_stub)

    with pytest.raises(chatbooks.HTTPException) as exc_info:
        await chatbooks.download_chatbook(
            job_id="123e4567-e89b-12d3-a456-426614174000",
            request=None,
            service=_FailingDownloadService(),
            user=user,
            audit_service=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "An error occurred while downloading the file"
    assert logger_stub.errors == ["Failed to download chatbook"]
    assert logger_stub.exceptions == []
    assert "123e4567" not in str(logger_stub.errors)
    assert "tester" not in str(logger_stub.errors)
    assert "/private/" not in str(logger_stub.errors)
