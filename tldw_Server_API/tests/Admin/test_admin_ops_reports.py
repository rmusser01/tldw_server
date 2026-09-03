"""Tests for report schedules and digest preferences in admin system ops."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from fastapi import HTTPException


def _configure_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Any, Path]:
    """Redirect the system-ops JSON store to an isolated temporary path."""
    from tldw_Server_API.app.services import admin_system_ops_service

    store_path = tmp_path / "system_ops.json"
    monkeypatch.setattr(admin_system_ops_service, "_STORE_PATH", store_path)
    return admin_system_ops_service, store_path


class TestReportScheduleCreate:
    """Tests for create_report_schedule service function."""

    def test_create_report_schedule_basic(self, monkeypatch, tmp_path):
        """Create a schedule with valid frequency, format, and recipients."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        schedule = service.create_report_schedule(
            frequency="weekly",
            recipients=["admin@example.com"],
            report_format="html",
            enabled=True,
        )

        assert schedule["frequency"] == "weekly"
        assert schedule["recipients"] == ["admin@example.com"]
        assert schedule["format"] == "html"
        assert schedule["enabled"] is True
        assert schedule["last_sent_at"] is None

    def test_create_report_schedule_invalid_frequency(self, monkeypatch, tmp_path):
        """Invalid frequency raises ValueError('invalid_frequency')."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="invalid_frequency"):
            service.create_report_schedule(
                frequency="hourly",
                recipients=["admin@example.com"],
            )

    def test_create_report_schedule_invalid_format(self, monkeypatch, tmp_path):
        """Invalid format raises ValueError('invalid_format')."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="invalid_format"):
            service.create_report_schedule(
                frequency="weekly",
                recipients=["admin@example.com"],
                report_format="pdf",
            )

    def test_create_report_schedule_empty_recipients(self, monkeypatch, tmp_path):
        """Empty recipients list raises ValueError('recipients_required')."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="recipients_required"):
            service.create_report_schedule(
                frequency="weekly",
                recipients=[],
            )

    def test_create_report_schedule_invalid_recipient_email(self, monkeypatch, tmp_path):
        """Recipient without @ raises ValueError('invalid_recipient_email')."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="invalid_recipient_email"):
            service.create_report_schedule(
                frequency="weekly",
                recipients=["not-an-email"],
            )

    def test_create_report_schedule_all_frequencies(self, monkeypatch, tmp_path):
        """All valid frequencies are accepted: daily, weekly, monthly."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        for freq in ("daily", "weekly", "monthly"):
            schedule = service.create_report_schedule(
                frequency=freq,
                recipients=["admin@example.com"],
            )
            assert schedule["frequency"] == freq

    def test_create_report_schedule_json_format(self, monkeypatch, tmp_path):
        """JSON format is accepted."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        schedule = service.create_report_schedule(
            frequency="weekly",
            recipients=["admin@example.com"],
            report_format="json",
        )

        assert schedule["format"] == "json"


# ═══════════════════════════════════════════════════════════════════════════
# 8. Report Schedule List / Update / Delete
# ═══════════════════════════════════════════════════════════════════════════


class TestReportScheduleListUpdateDelete:
    """Tests for list, update, and delete of report schedules."""

    def test_list_report_schedules_empty(self, monkeypatch, tmp_path):
        """No schedules yields empty list."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        schedules = service.list_report_schedules()
        assert schedules == []

    def test_list_report_schedules_returns_all(self, monkeypatch, tmp_path):
        """Multiple schedules are returned."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        service.create_report_schedule(
            frequency="weekly",
            recipients=["a@b.com"],
        )
        service.create_report_schedule(
            frequency="daily",
            recipients=["c@d.com"],
        )

        schedules = service.list_report_schedules()
        assert len(schedules) == 2

    def test_update_report_schedule_frequency(self, monkeypatch, tmp_path):
        """Updating the frequency works."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        schedule = service.create_report_schedule(
            frequency="weekly",
            recipients=["admin@example.com"],
        )

        updated = service.update_report_schedule(
            schedule_id=schedule["id"],
            frequency="monthly",
        )

        assert updated["frequency"] == "monthly"

    def test_update_report_schedule_recipients(self, monkeypatch, tmp_path):
        """Updating recipients replaces the list."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        schedule = service.create_report_schedule(
            frequency="weekly",
            recipients=["old@example.com"],
        )

        updated = service.update_report_schedule(
            schedule_id=schedule["id"],
            recipients=["new1@example.com", "new2@example.com"],
        )

        assert updated["recipients"] == ["new1@example.com", "new2@example.com"]

    def test_update_report_schedule_toggle_enabled(self, monkeypatch, tmp_path):
        """Toggling enabled flag works."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        schedule = service.create_report_schedule(
            frequency="weekly",
            recipients=["admin@example.com"],
            enabled=True,
        )

        updated = service.update_report_schedule(
            schedule_id=schedule["id"],
            enabled=False,
        )

        assert updated["enabled"] is False

    def test_update_report_schedule_not_found(self, monkeypatch, tmp_path):
        """Updating a nonexistent schedule raises ValueError('not_found')."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="not_found"):
            service.update_report_schedule(
                schedule_id="nonexistent",
                frequency="daily",
            )

    def test_update_report_schedule_invalid_frequency(self, monkeypatch, tmp_path):
        """Updating with invalid frequency raises ValueError."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        schedule = service.create_report_schedule(
            frequency="weekly",
            recipients=["admin@example.com"],
        )

        with pytest.raises(ValueError, match="invalid_frequency"):
            service.update_report_schedule(
                schedule_id=schedule["id"],
                frequency="biweekly",
            )

    def test_delete_report_schedule(self, monkeypatch, tmp_path):
        """Deleting a schedule removes it and returns the deleted record."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        schedule = service.create_report_schedule(
            frequency="weekly",
            recipients=["admin@example.com"],
        )

        deleted = service.delete_report_schedule(schedule_id=schedule["id"])

        assert deleted["id"] == schedule["id"]

        # Should be gone from the list
        schedules = service.list_report_schedules()
        assert len(schedules) == 0

    def test_delete_report_schedule_not_found(self, monkeypatch, tmp_path):
        """Deleting a nonexistent schedule raises ValueError('not_found')."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="not_found"):
            service.delete_report_schedule(schedule_id="nonexistent")

    def test_mark_report_schedule_sent(self, monkeypatch, tmp_path):
        """mark_report_schedule_sent updates last_sent_at."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        schedule = service.create_report_schedule(
            frequency="weekly",
            recipients=["admin@example.com"],
        )

        assert schedule["last_sent_at"] is None

        marked = service.mark_report_schedule_sent(schedule_id=schedule["id"])

        assert marked["last_sent_at"] is not None

    def test_mark_report_schedule_sent_not_found(self, monkeypatch, tmp_path):
        """Marking a nonexistent schedule raises ValueError('not_found')."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="not_found"):
            service.mark_report_schedule_sent(schedule_id="nonexistent")


class TestAdminOpsScheduleErrorSanitization:
    """Direct endpoint tests for narrow report-schedule fallback sanitization."""

    @pytest.mark.asyncio
    async def test_get_report_schedules_sanitizes_backend_error(self, monkeypatch):
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops

        monkeypatch.setattr(
            "tldw_Server_API.app.api.v1.endpoints.admin.admin_ops._require_platform_admin",
            lambda _: None,
        )

        def _raise_schedules() -> list[dict[str, Any]]:
            raise OSError("report schedules backend exploded")

        monkeypatch.setattr(admin_ops, "svc_list_report_schedules", _raise_schedules)

        with pytest.raises(HTTPException) as exc_info:
            await admin_ops.get_report_schedules(principal=mock.MagicMock())

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to list report schedules"


class TestAdminOpsDigestErrorSanitization:
    """Direct endpoint tests for digest preference fallback sanitization."""

    @pytest.mark.asyncio
    async def test_get_digest_preference_sanitizes_backend_error(self, monkeypatch):
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops

        def _raise_pref(*, user_id: str) -> dict[str, Any] | None:
            _ = user_id
            raise OSError("digest preference backend exploded")

        monkeypatch.setattr(admin_ops, "svc_get_digest_preference", _raise_pref)

        with pytest.raises(HTTPException) as exc_info:
            await admin_ops.get_digest_preference(
                principal=mock.MagicMock(user_id="user_42"),
            )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to get digest preference"

    @pytest.mark.asyncio
    async def test_set_digest_preference_sanitizes_backend_error(self, monkeypatch):
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops

        class _Request:
            async def json(self) -> dict[str, Any]:
                return {"email": "user42@example.com", "frequency": "weekly"}

        def _raise_pref(*, user_id: str, email: str, frequency: str) -> dict[str, Any]:
            _ = (user_id, email, frequency)
            raise OSError("digest preference write exploded")

        monkeypatch.setattr(admin_ops, "svc_set_digest_preference", _raise_pref)

        with pytest.raises(HTTPException) as exc_info:
            await admin_ops.set_digest_preference(
                request=_Request(),
                principal=mock.MagicMock(user_id="user_42"),
            )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to set digest preference"


# ═══════════════════════════════════════════════════════════════════════════
# 9. Digest Preference Get / Set — per-user scoping
# ═══════════════════════════════════════════════════════════════════════════


class TestDigestPreferences:
    """Tests for get_digest_preference and set_digest_preference."""

    def test_set_and_get_digest_preference(self, monkeypatch, tmp_path):
        """Setting a preference and retrieving it returns the same values."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        pref = service.set_digest_preference(
            user_id="user_42",
            email="user42@example.com",
            frequency="weekly",
        )

        assert pref["user_id"] == "user_42"
        assert pref["email"] == "user42@example.com"
        assert pref["frequency"] == "weekly"
        assert pref["enabled"] is True

        retrieved = service.get_digest_preference(user_id="user_42")
        assert retrieved is not None
        assert retrieved["user_id"] == "user_42"
        assert retrieved["frequency"] == "weekly"

    def test_set_digest_preference_updates_existing(self, monkeypatch, tmp_path):
        """Setting a preference for an existing user updates it in place."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        service.set_digest_preference(
            user_id="user_42",
            email="user42@example.com",
            frequency="weekly",
        )

        updated = service.set_digest_preference(
            user_id="user_42",
            email="user42@example.com",
            frequency="daily",
        )

        assert updated["frequency"] == "daily"

        # Should still be only one entry
        retrieved = service.get_digest_preference(user_id="user_42")
        assert retrieved["frequency"] == "daily"

    def test_digest_preference_per_user_scoped(self, monkeypatch, tmp_path):
        """Different users have independent digest preferences."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        service.set_digest_preference(
            user_id="user_1",
            email="user1@example.com",
            frequency="daily",
        )
        service.set_digest_preference(
            user_id="user_2",
            email="user2@example.com",
            frequency="weekly",
        )

        pref_1 = service.get_digest_preference(user_id="user_1")
        pref_2 = service.get_digest_preference(user_id="user_2")

        assert pref_1["frequency"] == "daily"
        assert pref_2["frequency"] == "weekly"

    def test_set_digest_preference_off_disables(self, monkeypatch, tmp_path):
        """Setting frequency to 'off' sets enabled=False."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        pref = service.set_digest_preference(
            user_id="user_42",
            email="user42@example.com",
            frequency="off",
        )

        assert pref["frequency"] == "off"
        assert pref["enabled"] is False

    def test_set_digest_preference_invalid_email(self, monkeypatch, tmp_path):
        """Invalid email raises ValueError('invalid_email')."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="invalid_email"):
            service.set_digest_preference(
                user_id="user_42",
                email="not-an-email",
                frequency="weekly",
            )

    def test_set_digest_preference_invalid_frequency(self, monkeypatch, tmp_path):
        """Invalid frequency raises ValueError('invalid_frequency')."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="invalid_frequency"):
            service.set_digest_preference(
                user_id="user_42",
                email="user42@example.com",
                frequency="hourly",
            )


# ═══════════════════════════════════════════════════════════════════════════
# 10. Digest Preference Default — returns None (off) for unknown user
# ═══════════════════════════════════════════════════════════════════════════


class TestDigestPreferenceDefault:
    """Tests for default digest preference behavior."""

    def test_unknown_user_returns_none(self, monkeypatch, tmp_path):
        """Getting preference for unknown user returns None."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        pref = service.get_digest_preference(user_id="unknown_user_999")

        assert pref is None

    def test_empty_store_returns_none(self, monkeypatch, tmp_path):
        """Fresh store returns None for any user."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        assert service.get_digest_preference(user_id="any_user") is None
        assert service.get_digest_preference(user_id="") is None
