"""Tests for user invitation CRUD in admin_system_ops_service.

Exercises create, list, resend, revoke, and accept invitation functions
with the JSON store redirected to a temp directory for isolation.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Shared fixture: isolate the JSON store to a temp directory
# ---------------------------------------------------------------------------


def _configure_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Redirect the system-ops JSON store to *tmp_path* and return the module."""
    from tldw_Server_API.app.services import admin_system_ops_service as svc

    store_path = tmp_path / "system_ops.json"
    monkeypatch.setattr(svc, "_STORE_PATH", store_path)
    return svc


# ===========================================================================
# 1. create_invitation()
# ===========================================================================


class TestCreateInvitation:
    """Tests for the create_invitation service function."""

    def test_create_basic(self, monkeypatch, tmp_path):
        """Creating an invitation returns all expected fields."""
        svc = _configure_store(monkeypatch, tmp_path)
        inv = svc.create_invitation(email="alice@example.com", role="user", invited_by="admin")
        assert inv["email"] == "alice@example.com"
        assert inv["role"] == "user"
        assert inv["status"] == "pending"
        assert inv["invited_by"] == "admin"
        assert isinstance(inv["token"], str) and len(inv["token"]) > 0
        assert inv["id"]
        assert inv["expires_at"]
        assert inv["accepted_at"] is None

    def test_create_normalizes_email(self, monkeypatch, tmp_path):
        """Email is lowercased and stripped."""
        svc = _configure_store(monkeypatch, tmp_path)
        inv = svc.create_invitation(email="  Alice@EXAMPLE.COM  ")
        assert inv["email"] == "alice@example.com"

    def test_create_invalid_email_raises(self, monkeypatch, tmp_path):
        """Invalid email raises ValueError."""
        svc = _configure_store(monkeypatch, tmp_path)
        with pytest.raises(ValueError, match="invalid_email"):
            svc.create_invitation(email="not-an-email")

    def test_create_invalid_role_raises(self, monkeypatch, tmp_path):
        """Invalid role raises ValueError."""
        svc = _configure_store(monkeypatch, tmp_path)
        with pytest.raises(ValueError, match="invalid_role"):
            svc.create_invitation(email="a@b.com", role="superadmin")

    def test_create_generates_unique_token(self, monkeypatch, tmp_path):
        """Each invitation gets a unique token."""
        svc = _configure_store(monkeypatch, tmp_path)
        inv1 = svc.create_invitation(email="a@example.com")
        inv2 = svc.create_invitation(email="b@example.com")
        assert inv1["token"] != inv2["token"]

    def test_create_expiry_defaults_to_7_days(self, monkeypatch, tmp_path):
        """Default expiry is 7 days from now."""
        svc = _configure_store(monkeypatch, tmp_path)
        inv = svc.create_invitation(email="a@example.com")
        expires_dt = datetime.fromisoformat(inv["expires_at"].replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        # Should be roughly 7 days in the future (within 1 minute tolerance)
        diff = (expires_dt - now).total_seconds()
        assert 6 * 86400 < diff < 8 * 86400

    def test_create_custom_expiry(self, monkeypatch, tmp_path):
        """Custom expiry_days is respected."""
        svc = _configure_store(monkeypatch, tmp_path)
        inv = svc.create_invitation(email="a@example.com", expiry_days=30)
        expires_dt = datetime.fromisoformat(inv["expires_at"].replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        diff = (expires_dt - now).total_seconds()
        assert 29 * 86400 < diff < 31 * 86400

    def test_create_duplicate_pending_raises(self, monkeypatch, tmp_path):
        """Duplicate pending invitation to the same email raises ValueError."""
        svc = _configure_store(monkeypatch, tmp_path)
        svc.create_invitation(email="same@example.com")
        with pytest.raises(ValueError, match="duplicate_pending_invitation"):
            svc.create_invitation(email="same@example.com")

    def test_create_duplicate_allowed_after_revoke(self, monkeypatch, tmp_path):
        """Creating an invitation after revoking the first succeeds."""
        svc = _configure_store(monkeypatch, tmp_path)
        inv1 = svc.create_invitation(email="same@example.com")
        svc.revoke_invitation(invitation_id=inv1["id"])
        inv2 = svc.create_invitation(email="same@example.com")
        assert inv2["status"] == "pending"
        assert inv2["id"] != inv1["id"]


# ===========================================================================
# 2. list_invitations()
# ===========================================================================


class TestListInvitations:
    """Tests for listing and filtering invitations."""

    def test_list_empty(self, monkeypatch, tmp_path):
        """Empty store returns empty list."""
        svc = _configure_store(monkeypatch, tmp_path)
        assert svc.list_invitations() == []

    def test_list_returns_all(self, monkeypatch, tmp_path):
        """All created invitations are returned."""
        svc = _configure_store(monkeypatch, tmp_path)
        svc.create_invitation(email="a@example.com")
        svc.create_invitation(email="b@example.com")
        result = svc.list_invitations()
        assert len(result) == 2

    def test_list_sorted_newest_first(self, monkeypatch, tmp_path):
        """Invitations are sorted by created_at descending."""
        svc = _configure_store(monkeypatch, tmp_path)
        inv1 = svc.create_invitation(email="first@example.com")
        inv2 = svc.create_invitation(email="second@example.com")
        result = svc.list_invitations()
        # The second created should appear first (newest)
        assert result[0]["email"] == "second@example.com"

    def test_list_filter_by_status(self, monkeypatch, tmp_path):
        """Filtering by status works."""
        svc = _configure_store(monkeypatch, tmp_path)
        inv1 = svc.create_invitation(email="a@example.com")
        inv2 = svc.create_invitation(email="b@example.com")
        svc.revoke_invitation(invitation_id=inv1["id"])

        pending = svc.list_invitations(status="pending")
        assert len(pending) == 1
        assert pending[0]["email"] == "b@example.com"

        revoked = svc.list_invitations(status="revoked")
        assert len(revoked) == 1
        assert revoked[0]["email"] == "a@example.com"

    def test_list_auto_expires_past_invitations(self, monkeypatch, tmp_path):
        """Pending invitations past their expiry_at are returned as expired."""
        svc = _configure_store(monkeypatch, tmp_path)

        # Manually seed a past-due invitation
        past_dt = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        store_data = svc._default_store()
        store_data["invitations"] = [
            {
                "id": "inv_old",
                "email": "expired@example.com",
                "role": "user",
                "status": "pending",
                "token": "tok123",
                "expires_at": past_dt,
                "created_at": (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
                "invited_by": None,
            }
        ]
        svc._save_store(store_data)

        result = svc.list_invitations()
        assert len(result) == 1
        assert result[0]["status"] == "expired"

    def test_list_filter_expired_returns_auto_expired(self, monkeypatch, tmp_path):
        """status=expired filter includes auto-expired invitations."""
        svc = _configure_store(monkeypatch, tmp_path)
        past_dt = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        store_data = svc._default_store()
        store_data["invitations"] = [
            {
                "id": "inv_exp",
                "email": "expired@example.com",
                "role": "user",
                "status": "pending",
                "token": "tok456",
                "expires_at": past_dt,
                "created_at": (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
                "invited_by": None,
            }
        ]
        svc._save_store(store_data)

        expired_list = svc.list_invitations(status="expired")
        assert len(expired_list) == 1
        assert expired_list[0]["id"] == "inv_exp"


# ===========================================================================
# 3. resend_invitation()
# ===========================================================================


class TestResendInvitation:
    """Tests for the resend_invitation service function."""

    def test_resend_regenerates_token(self, monkeypatch, tmp_path):
        """Resending an invitation generates a new token."""
        svc = _configure_store(monkeypatch, tmp_path)
        inv = svc.create_invitation(email="a@example.com")
        old_token = inv["token"]
        updated = svc.resend_invitation(invitation_id=inv["id"])
        assert updated["token"] != old_token

    def test_resend_extends_expiry(self, monkeypatch, tmp_path):
        """Resending extends the expiry date."""
        svc = _configure_store(monkeypatch, tmp_path)
        inv = svc.create_invitation(email="a@example.com", expiry_days=1)
        old_expires = datetime.fromisoformat(inv["expires_at"].replace("Z", "+00:00"))
        updated = svc.resend_invitation(invitation_id=inv["id"])
        new_expires = datetime.fromisoformat(updated["expires_at"].replace("Z", "+00:00"))
        # New expiry should be further in the future
        assert new_expires > old_expires

    def test_resend_increments_count(self, monkeypatch, tmp_path):
        """Each resend increments resend_count."""
        svc = _configure_store(monkeypatch, tmp_path)
        inv = svc.create_invitation(email="a@example.com")
        assert inv.get("resend_count", 0) == 0
        r1 = svc.resend_invitation(invitation_id=inv["id"])
        assert r1["resend_count"] == 1
        r2 = svc.resend_invitation(invitation_id=inv["id"])
        assert r2["resend_count"] == 2

    def test_resend_rate_limit_at_3(self, monkeypatch, tmp_path):
        """Resending more than 3 times raises resend_limit_reached."""
        svc = _configure_store(monkeypatch, tmp_path)
        inv = svc.create_invitation(email="a@example.com")
        for _ in range(3):
            svc.resend_invitation(invitation_id=inv["id"])
        with pytest.raises(ValueError, match="resend_limit_reached"):
            svc.resend_invitation(invitation_id=inv["id"])

    def test_resend_not_found_raises(self, monkeypatch, tmp_path):
        """Resending a nonexistent invitation raises not_found."""
        svc = _configure_store(monkeypatch, tmp_path)
        with pytest.raises(ValueError, match="not_found"):
            svc.resend_invitation(invitation_id="ghost")

    def test_resend_revoked_raises_not_pending(self, monkeypatch, tmp_path):
        """Resending a revoked invitation raises not_pending."""
        svc = _configure_store(monkeypatch, tmp_path)
        inv = svc.create_invitation(email="a@example.com")
        svc.revoke_invitation(invitation_id=inv["id"])
        with pytest.raises(ValueError, match="not_pending"):
            svc.resend_invitation(invitation_id=inv["id"])

    def test_resend_resets_email_status(self, monkeypatch, tmp_path):
        """Resending resets email_sent and email_error."""
        svc = _configure_store(monkeypatch, tmp_path)
        inv = svc.create_invitation(email="a@example.com")
        svc.update_invitation_email_status(
            invitation_id=inv["id"], email_sent=True, email_error=None,
        )
        updated = svc.resend_invitation(invitation_id=inv["id"])
        assert updated["email_sent"] is False
        assert updated["email_error"] is None

    def test_resend_sets_last_resent_at(self, monkeypatch, tmp_path):
        """Resending populates last_resent_at timestamp."""
        svc = _configure_store(monkeypatch, tmp_path)
        inv = svc.create_invitation(email="a@example.com")
        updated = svc.resend_invitation(invitation_id=inv["id"])
        assert updated["last_resent_at"] is not None


# ===========================================================================
# 4. revoke_invitation()
# ===========================================================================


class TestRevokeInvitation:
    """Tests for the revoke_invitation service function."""

    def test_revoke_pending_succeeds(self, monkeypatch, tmp_path):
        """A pending invitation can be revoked."""
        svc = _configure_store(monkeypatch, tmp_path)
        inv = svc.create_invitation(email="a@example.com")
        revoked = svc.revoke_invitation(invitation_id=inv["id"])
        assert revoked["status"] == "revoked"

    def test_revoke_accepted_raises(self, monkeypatch, tmp_path):
        """An accepted invitation cannot be revoked."""
        svc = _configure_store(monkeypatch, tmp_path)
        inv = svc.create_invitation(email="a@example.com")
        svc.accept_invitation(invitation_id=inv["id"])
        with pytest.raises(ValueError, match="not_pending"):
            svc.revoke_invitation(invitation_id=inv["id"])

    def test_revoke_already_revoked_raises(self, monkeypatch, tmp_path):
        """A revoked invitation cannot be revoked again."""
        svc = _configure_store(monkeypatch, tmp_path)
        inv = svc.create_invitation(email="a@example.com")
        svc.revoke_invitation(invitation_id=inv["id"])
        with pytest.raises(ValueError, match="not_pending"):
            svc.revoke_invitation(invitation_id=inv["id"])

    def test_revoke_not_found_raises(self, monkeypatch, tmp_path):
        """Revoking a nonexistent invitation raises not_found."""
        svc = _configure_store(monkeypatch, tmp_path)
        with pytest.raises(ValueError, match="not_found"):
            svc.revoke_invitation(invitation_id="ghost")


# ===========================================================================
# 5. accept_invitation()
# ===========================================================================


class TestAcceptInvitation:
    """Tests for the accept_invitation service function."""

    def test_accept_pending_succeeds(self, monkeypatch, tmp_path):
        """A pending invitation transitions to accepted."""
        svc = _configure_store(monkeypatch, tmp_path)
        inv = svc.create_invitation(email="a@example.com")
        accepted = svc.accept_invitation(invitation_id=inv["id"])
        assert accepted["status"] == "accepted"
        assert accepted["accepted_at"] is not None

    def test_accept_revoked_raises(self, monkeypatch, tmp_path):
        """A revoked invitation cannot be accepted."""
        svc = _configure_store(monkeypatch, tmp_path)
        inv = svc.create_invitation(email="a@example.com")
        svc.revoke_invitation(invitation_id=inv["id"])
        with pytest.raises(ValueError, match="not_pending"):
            svc.accept_invitation(invitation_id=inv["id"])

    def test_accept_already_accepted_raises(self, monkeypatch, tmp_path):
        """An already accepted invitation cannot be accepted again."""
        svc = _configure_store(monkeypatch, tmp_path)
        inv = svc.create_invitation(email="a@example.com")
        svc.accept_invitation(invitation_id=inv["id"])
        with pytest.raises(ValueError, match="not_pending"):
            svc.accept_invitation(invitation_id=inv["id"])

    def test_accept_not_found_raises(self, monkeypatch, tmp_path):
        """Accepting a nonexistent invitation raises not_found."""
        svc = _configure_store(monkeypatch, tmp_path)
        with pytest.raises(ValueError, match="not_found"):
            svc.accept_invitation(invitation_id="ghost")
