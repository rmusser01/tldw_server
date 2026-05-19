from unittest.mock import MagicMock

import pytest

from tldw_Server_API.app.core.External_Sources import connectors_service as svc
from tldw_Server_API.app.core.Security import crypto


pytestmark = pytest.mark.unit


def _assert_sanitized_debug(fake_logger: MagicMock, expected_message: str) -> None:
    fake_logger.debug.assert_called_once_with(expected_message)
    rendered = repr(fake_logger.debug.call_args)
    assert "connectors crypto exploded" not in rendered
    assert "/private/connectors.db" not in rendered


def _assert_sanitized_error(fake_logger: MagicMock, expected_message: str) -> None:
    fake_logger.error.assert_called_once_with(expected_message)
    rendered = repr(fake_logger.error.call_args)
    assert "connectors backend exploded" not in rendered
    assert "/private/connectors.db" not in rendered


def _assert_sanitized_warning(fake_logger: MagicMock, expected_message: str) -> None:
    fake_logger.warning.assert_called_once_with(expected_message)
    rendered = repr(fake_logger.warning.call_args)
    assert "connectors backend exploded" not in rendered
    assert "/private/connectors.db" not in rendered


def test_protect_oauth_state_metadata_sanitizes_encryption_fallback_log(monkeypatch):
    fake_logger = MagicMock()

    def _fail_encrypt(_metadata):
        raise RuntimeError("connectors crypto exploded /private/connectors.db")

    monkeypatch.setattr(svc, "logger", fake_logger)
    monkeypatch.setattr(crypto, "encrypt_json_blob", _fail_encrypt)

    result = svc._protect_oauth_state_metadata({"nonce": "state-token"})

    assert result == {"nonce": "state-token"}
    _assert_sanitized_debug(
        fake_logger,
        "Failed to encrypt oauth state metadata",
    )


def test_unprotect_oauth_state_metadata_sanitizes_decryption_fallback_log(monkeypatch):
    fake_logger = MagicMock()

    def _fail_decrypt(_metadata):
        raise RuntimeError("connectors crypto exploded /private/connectors.db")

    monkeypatch.setattr(svc, "logger", fake_logger)
    monkeypatch.setattr(crypto, "decrypt_json_blob", _fail_decrypt)

    result = svc._unprotect_oauth_state_metadata({"_enc": "aesgcm:v1", "ct": "encrypted"})

    assert result == {}
    _assert_sanitized_debug(
        fake_logger,
        "Failed to decrypt oauth state metadata",
    )


@pytest.mark.asyncio
async def test_ensure_tables_sanitizes_failure_log_and_reraises(monkeypatch):
    fake_logger = MagicMock()
    failure = RuntimeError("connectors backend exploded /private/connectors.db")

    async def _fail_ensure_tables(_db, *, is_postgres: bool):
        assert is_postgres is False
        raise failure

    monkeypatch.setattr(svc, "logger", fake_logger)
    monkeypatch.setattr(svc, "ensure_connectors_tables", _fail_ensure_tables)

    with pytest.raises(RuntimeError) as exc_info:
        await svc._ensure_tables(object())

    assert exc_info.value is failure
    _assert_sanitized_error(
        fake_logger,
        "Failed to ensure connector tables",
    )


@pytest.mark.asyncio
async def test_create_import_job_sanitizes_job_manager_fallback_log(monkeypatch):
    class _Pool:
        pass

    class _FailingJobManager:
        def __init__(self):
            raise RuntimeError("connectors backend exploded /private/connectors.db")

    async def _get_db_pool():
        return _Pool()

    import tldw_Server_API.app.core.AuthNZ.database as auth_database
    import tldw_Server_API.app.core.Jobs.manager as jobs_manager

    fake_logger = MagicMock()
    monkeypatch.setattr(svc, "logger", fake_logger)
    monkeypatch.setattr(auth_database, "get_db_pool", _get_db_pool)
    monkeypatch.setattr(jobs_manager, "JobManager", _FailingJobManager)

    result = await svc.create_import_job(user_id=7, source_id=42, job_type="import")

    assert result["source_id"] == 42
    assert result["type"] == "import"
    assert result["status"] == "queued"
    assert result["progress_pct"] == 0
    assert result["counts"] == {"processed": 0, "skipped": 0, "failed": 0}
    _assert_sanitized_warning(
        fake_logger,
        "Failed to create connectors job via JobManager",
    )
