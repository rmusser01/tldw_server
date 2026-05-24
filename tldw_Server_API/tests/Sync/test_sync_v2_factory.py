from __future__ import annotations

from tldw_Server_API.app.core.Sync.v2.factory import _sync_v2_settings_from_env


def test_sync_v2_factory_keeps_blob_transfer_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("SYNC_V2_ENABLE_BLOB_TRANSFER", raising=False)

    settings = _sync_v2_settings_from_env()

    assert settings.supports_attachments is False


def test_sync_v2_factory_enables_blob_transfer_from_env(monkeypatch) -> None:
    monkeypatch.setenv("SYNC_V2_ENABLE_BLOB_TRANSFER", "true")
    monkeypatch.setenv("SYNC_V2_MAX_BLOB_BYTES", "4096")
    monkeypatch.setenv("SYNC_V2_MAX_CHUNK_BYTES", "1024")
    monkeypatch.setenv("SYNC_V2_MAX_ACTIVE_BLOB_UPLOADS", "3")
    monkeypatch.setenv("SYNC_V2_USER_BLOB_QUOTA_BYTES", "8192")

    settings = _sync_v2_settings_from_env()

    assert settings.supports_attachments is True
    assert settings.max_blob_bytes == 4096
    assert settings.max_chunk_bytes == 1024
    assert settings.max_active_blob_uploads == 3
    assert settings.user_blob_quota_bytes == 8192
