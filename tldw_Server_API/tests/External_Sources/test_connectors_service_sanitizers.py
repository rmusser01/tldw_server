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

    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.delenv("CONNECTORS_REQUIRE_TOKEN_ENCRYPTION", raising=False)
    monkeypatch.delenv("tldw_production", raising=False)
    monkeypatch.setattr(svc, "logger", fake_logger)
    monkeypatch.setattr(crypto, "encrypt_json_blob", _fail_encrypt)

    result = svc._protect_oauth_state_metadata({"nonce": "state-token"})

    assert result == {"nonce": "state-token"}
    _assert_sanitized_debug(
        fake_logger,
        "Failed to encrypt oauth state metadata",
    )


def test_protect_oauth_state_metadata_requires_encryption_in_multi_user(monkeypatch):
    fake_logger = MagicMock()

    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.delenv("CONNECTORS_REQUIRE_TOKEN_ENCRYPTION", raising=False)
    monkeypatch.delenv("tldw_production", raising=False)
    monkeypatch.delenv("WORKFLOWS_ARTIFACT_ENC_KEY", raising=False)
    monkeypatch.setattr(svc, "logger", fake_logger)
    monkeypatch.setattr(crypto, "encrypt_json_blob", lambda _metadata: None)

    with pytest.raises(RuntimeError, match="Connector secret encryption is required"):
        svc._protect_oauth_state_metadata({"oauth_token_secret": "temp-secret"})

    fake_logger.warning.assert_called_once_with("Connector secret encryption is required but unavailable")


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
async def test_consume_oauth_state_consumes_postgres_row_atomically(monkeypatch):
    class _FakePg:
        _is_sqlite = False

        def __init__(self):
            self.delete_rows = [
                {
                    "state": "state-token",
                    "provider": "zotero",
                    "metadata": {"nonce": "only-once"},
                    "created_at": "2026-05-01 00:00:00",
                }
            ]
            self.delete_calls = 0
            self.select_calls = 0

        async def fetchrow(self, sql, *args):
            sql_head = " ".join(str(sql).split()).upper()
            if sql_head.startswith("SELECT"):
                self.select_calls += 1
                return {
                    "state": "state-token",
                    "provider": "zotero",
                    "metadata": {"nonce": "stale-select"},
                    "created_at": "2026-05-01 00:00:00",
                }
            if sql_head.startswith("DELETE"):
                self.delete_calls += 1
                return self.delete_rows.pop(0) if self.delete_rows else None
            raise AssertionError(f"unexpected query: {sql}")

        async def execute(self, *args, **kwargs):
            return None

    async def _noop_ensure_tables(_db):
        return None

    db = _FakePg()
    monkeypatch.setattr(svc, "_ensure_tables", _noop_ensure_tables)

    first = await svc.consume_oauth_state(
        db,
        user_id=7,
        provider="zotero",
        state="state-token",
    )
    second = await svc.consume_oauth_state(
        db,
        user_id=7,
        provider="zotero",
        state="state-token",
    )

    assert first is not False
    assert first["nonce"] == "only-once"
    assert second is False
    assert db.delete_calls == 2
    assert db.select_calls == 0


@pytest.mark.asyncio
async def test_create_import_job_sanitizes_job_manager_failure_log(monkeypatch):
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

    with pytest.raises(RuntimeError, match="Failed to create connectors job"):
        await svc.create_import_job(user_id=7, source_id=42, job_type="import")

    _assert_sanitized_warning(
        fake_logger,
        "Failed to create connectors job via JobManager",
    )
