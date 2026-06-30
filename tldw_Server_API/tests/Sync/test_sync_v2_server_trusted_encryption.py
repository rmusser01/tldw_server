from __future__ import annotations

from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
    server_trusted_encryption_status_from_env,
)


def test_server_trusted_ready_requires_enabled_covered_mode() -> None:
    status = server_trusted_encryption_status_from_config(
        mode="managed_storage",
        server_trusted_enabled=True,
        auth_mode="multi_user",
    )

    assert status.ready is True
    assert status.encryption["policy"] == "server_trusted_v1"
    assert status.encryption["attestation"]["configured"] is True
    assert status.encryption["attestation"]["mode"] == "managed_storage"
    assert status.warnings == []


def test_server_trusted_is_not_ready_without_explicit_attestation() -> None:
    status = server_trusted_encryption_status_from_config(
        mode=None,
        server_trusted_enabled=False,
        auth_mode="multi_user",
    )

    assert status.ready is False
    assert status.encryption["attestation"]["configured"] is False
    assert status.encryption["attestation"]["mode"] is None
    assert status.warnings[0]["code"] == "sync_encryption_attestation_required"


def test_development_unencrypted_reports_posture_without_claiming_ready() -> None:
    status = server_trusted_encryption_status_from_config(
        mode="development_unencrypted",
        server_trusted_enabled=True,
        auth_mode="single_user",
    )

    assert status.ready is False
    assert status.encryption["attestation"]["configured"] is True
    assert status.encryption["attestation"]["mode"] == "development_unencrypted"
    assert status.encryption["attestation"]["development"] is True
    assert status.warnings[0]["code"] == "sync_development_unencrypted"


def test_server_trusted_env_config_is_deterministic(monkeypatch) -> None:
    monkeypatch.setenv("SYNC_V2_AT_REST_ENCRYPTION_MODE", "encrypted_volume")
    monkeypatch.setenv("SYNC_V2_SERVER_TRUSTED_ENABLED", "true")
    monkeypatch.setenv("AUTH_MODE", "multi_user")

    status = server_trusted_encryption_status_from_env()

    assert status.ready is True
    assert status.encryption["attestation"]["mode"] == "encrypted_volume"
