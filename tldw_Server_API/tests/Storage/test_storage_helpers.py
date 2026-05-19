"""Tests for storage endpoint helper conversions and compatibility seams."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.api.v1.endpoints import storage_helpers
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.generated_files_repo import FILE_CATEGORY_VOICE_CLONE


def _principal(
    *,
    roles: list[str] | None = None,
    permissions: list[str] | None = None,
    is_admin: bool = False,
) -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        roles=roles or [],
        permissions=permissions or [],
        is_admin=is_admin,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "principal",
    [
        _principal(roles=["admin"]),
        _principal(permissions=["*"]),
        _principal(permissions=["system.configure"]),
        _principal(is_admin=True),
    ],
)
def test_principal_is_storage_admin_preserves_accepted_admin_claims(principal: AuthPrincipal) -> None:
    """Storage admin helper accepts the legacy and claim-first admin forms."""
    assert storage_helpers._principal_is_storage_admin(principal) is True


@pytest.mark.unit
def test_principal_is_storage_admin_rejects_non_admin_claims() -> None:
    """Storage admin helper rejects principals without accepted admin claims."""
    principal = _principal(roles=["user"], permissions=["storage.read"])

    assert storage_helpers._principal_is_storage_admin(principal) is False


@pytest.mark.unit
def test_parse_datetime_handles_z_suffix_and_naive_strings() -> None:
    """Datetime helper preserves parsed values and defaults naive values to UTC."""
    z_value = storage_helpers._parse_datetime("2026-01-02T03:04:05Z")
    naive_value = storage_helpers._parse_datetime("2026-01-02T03:04:05")

    assert z_value == datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert naive_value == datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


@pytest.mark.unit
def test_parse_datetime_returns_none_for_invalid_values() -> None:
    """Datetime helper preserves the endpoint fallback for invalid values."""
    assert storage_helpers._parse_datetime(None) is None
    assert storage_helpers._parse_datetime("not-a-date") is None
    assert storage_helpers._parse_datetime(object()) is None


@pytest.mark.unit
def test_to_generated_file_applies_defaults_and_parses_timestamps() -> None:
    """Generated-file helper converts sparse records with endpoint-compatible defaults."""
    generated_file = storage_helpers._to_generated_file(
        {
            "id": 42,
            "uuid": "file-uuid",
            "user_id": 7,
            "filename": "result.png",
            "storage_path": "images/result.png",
            "file_category": "image",
            "source_feature": "image_gen",
            "created_at": "2026-01-02T03:04:05Z",
            "updated_at": "2026-01-02T04:05:06Z",
            "accessed_at": "invalid",
        }
    )

    assert generated_file.id == 42
    assert generated_file.uuid == "file-uuid"
    assert generated_file.user_id == 7
    assert generated_file.filename == "result.png"
    assert generated_file.storage_path == "images/result.png"
    assert generated_file.file_size_bytes == 0
    assert generated_file.tags == []
    assert generated_file.is_deleted is False
    assert generated_file.created_at == datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert generated_file.updated_at == datetime(2026, 1, 2, 4, 5, 6, tzinfo=timezone.utc)
    assert generated_file.accessed_at is None


@pytest.mark.unit
def test_resolve_storage_base_dir_selects_voice_or_output_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Storage-base helper preserves voice clone routing and default output routing."""
    voices_dir = Path("/tmp/test-voices")
    outputs_dir = Path("/tmp/test-outputs")
    monkeypatch.setattr(storage_helpers.DatabasePaths, "get_user_voices_dir", lambda user_id: voices_dir)
    monkeypatch.setattr(storage_helpers.DatabasePaths, "get_user_outputs_dir", lambda user_id: outputs_dir)

    assert (
        storage_helpers._resolve_storage_base_dir(7, {"file_category": FILE_CATEGORY_VOICE_CLONE})
        == voices_dir
    )
    assert storage_helpers._resolve_storage_base_dir(7, {"file_category": "image"}) == outputs_dir


@pytest.mark.unit
def test_to_quota_status_applies_endpoint_defaults() -> None:
    """Quota helper converts sparse quota data with endpoint-compatible defaults."""
    quota = storage_helpers._to_quota_status({"quota_mb": 1000, "used_mb": 125.5})

    assert quota.quota_mb == 1000
    assert quota.used_mb == 125.5
    assert quota.remaining_mb is None
    assert quota.usage_pct == 0.0
    assert quota.at_soft_limit is False
    assert quota.at_hard_limit is False
    assert quota.has_quota is False
