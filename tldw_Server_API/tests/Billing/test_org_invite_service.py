"""
Tests for organization invite service.

Test Strategy
=============
This module tests the OrgInviteService at the unit level using mocked
repository dependencies. Tests verify business logic without database access.

Test Groups
-----------
1. **InviteValidationResult tests**: Verify the `is_valid` property correctly
    reflects invite status (VALID, EXPIRED, EXHAUSTED, REVOKED, NOT_FOUND).

2. **RedemptionResult tests**: Verify result dataclass captures success/failure
    states and idempotent redemption (was_already_member).

3. **OrgInviteService tests**: Core service logic including:
    - **Creation validation**: Role restrictions (no owner grants), max_uses bounds,
        expiry_days bounds, team-org membership verification.
    - **Validation states**: All five invite states are correctly identified.
    - **Redemption flows**: Invalid codes, already-member idempotency, new member
        addition, team-specific invites.
    - **Management operations**: Revoke, list, preview, cleanup.

Mocking Approach
----------------
- `mock_invites_repo`: Mocks AuthnzOrgInvitesRepo for invite CRUD operations
- `mock_orgs_repo`: Mocks AuthnzOrgsTeamsRepo for membership operations
- Both are injected via constructor to avoid database pool initialization

For integration tests with real database, see test_billing_endpoints_integration.py.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from tldw_Server_API.app.services.org_invite_service import (
    OrgInviteService,
    InviteStatus,
    InviteValidationResult,
    RedemptionResult,
    get_invite_service,
)
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings


class TestInviteValidationResult:
    """Tests for InviteValidationResult dataclass."""

    def test_is_valid_true_when_valid(self):

        """is_valid should be True when status is VALID."""
        result = InviteValidationResult(
            status=InviteStatus.VALID,
            invite={"id": 1},
        )
        assert result.is_valid is True

    def test_is_valid_false_when_expired(self):

        """is_valid should be False when status is EXPIRED."""
        result = InviteValidationResult(
            status=InviteStatus.EXPIRED,
            invite={"id": 1},
            message="Expired",
        )
        assert result.is_valid is False

    def test_is_valid_false_when_exhausted(self):

        """is_valid should be False when status is EXHAUSTED."""
        result = InviteValidationResult(
            status=InviteStatus.EXHAUSTED,
            invite={"id": 1},
        )
        assert result.is_valid is False

    def test_is_valid_false_when_revoked(self):

        """is_valid should be False when status is REVOKED."""
        result = InviteValidationResult(
            status=InviteStatus.REVOKED,
            invite={"id": 1},
        )
        assert result.is_valid is False

    def test_is_valid_false_when_not_found(self):

        """is_valid should be False when status is NOT_FOUND."""
        result = InviteValidationResult(
            status=InviteStatus.NOT_FOUND,
        )
        assert result.is_valid is False


class TestRedemptionResult:
    """Tests for RedemptionResult dataclass."""

    def test_successful_redemption(self):

        """Successful redemption should have success=True."""
        result = RedemptionResult(
            success=True,
            org_id=1,
            org_name="Test Org",
            role="member",
        )
        assert result.success is True
        assert result.was_already_member is False

    def test_already_member_redemption(self):

        """Already member redemption should be marked."""
        result = RedemptionResult(
            success=True,
            org_id=1,
            was_already_member=True,
        )
        assert result.success is True
        assert result.was_already_member is True

    def test_failed_redemption(self):

        """Failed redemption should have success=False."""
        result = RedemptionResult(
            success=False,
            message="Invite expired",
        )
        assert result.success is False


class TestOrgInviteService:
    """Tests for OrgInviteService class."""

    @pytest.fixture
    def mock_invites_repo(self):
        """Create a mock invites repository."""
        return AsyncMock()

    @pytest.fixture
    def mock_orgs_repo(self):
        """Create a mock orgs/teams repository."""
        return AsyncMock()

    @pytest.fixture
    def service(self, mock_invites_repo, mock_orgs_repo):
        """Create a service with mocked dependencies."""
        return OrgInviteService(
            db_pool=MagicMock(),
            invites_repo=mock_invites_repo,
            orgs_repo=mock_orgs_repo,
        )

    @pytest.mark.asyncio
    async def test_create_invite_success(self, service, mock_invites_repo):
        """Creating an invite should succeed with valid params."""
        mock_invites_repo.create_invite.return_value = {
            "id": 1,
            "code": "ABC123XYZ",
            "org_id": 1,
            "role_to_grant": "member",
        }

        invite = await service.create_invite(
            org_id=1,
            created_by=100,
            max_uses=5,
            expiry_days=7,
        )

        assert invite["code"] == "ABC123XYZ"
        mock_invites_repo.create_invite.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_invite_invalid_role_raises(self, service):
        """Creating an invite with invalid role should raise ValueError."""
        with pytest.raises(ValueError, match="role_to_grant"):
            await service.create_invite(
                org_id=1,
                created_by=100,
                role_to_grant="owner",  # Invalid - can't grant owner via invite
            )

    @pytest.mark.asyncio
    async def test_create_invite_invalid_max_uses_raises(self, service):
        """Creating an invite with invalid max_uses should raise ValueError."""
        with pytest.raises(ValueError, match="max_uses"):
            await service.create_invite(
                org_id=1,
                created_by=100,
                max_uses=0,  # Invalid - must be >= 1
            )

        with pytest.raises(ValueError, match="max_uses"):
            await service.create_invite(
                org_id=1,
                created_by=100,
                max_uses=1001,  # Invalid - must be <= 1000
            )

    @pytest.mark.asyncio
    async def test_create_invite_invalid_expiry_days_raises(self, service):
        """Creating an invite with invalid expiry_days should raise ValueError."""
        with pytest.raises(ValueError, match="expiry_days"):
            await service.create_invite(
                org_id=1,
                created_by=100,
                expiry_days=0,  # Invalid - must be >= 1
            )

        with pytest.raises(ValueError, match="expiry_days"):
            await service.create_invite(
                org_id=1,
                created_by=100,
                expiry_days=366,  # Invalid - must be <= 365
            )

    @pytest.mark.asyncio
    async def test_create_invite_team_wrong_org_raises(self, service, mock_orgs_repo):
        """Creating a team invite for wrong org should raise ValueError."""
        mock_orgs_repo.get_team.return_value = {"id": 77, "org_id": 999}  # Different org

        with pytest.raises(ValueError, match="does not belong"):
            await service.create_invite(
                org_id=1,
                created_by=100,
                team_id=77,
            )

    @pytest.mark.asyncio
    async def test_validate_invite_not_found(self, service, mock_invites_repo):
        """Validating nonexistent invite should return NOT_FOUND."""
        mock_invites_repo.get_invite_by_code.return_value = None

        result = await service.validate_invite("BADCODE")

        assert result.status == InviteStatus.NOT_FOUND
        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_validate_invite_revoked(self, service, mock_invites_repo):
        """Validating revoked invite should return REVOKED."""
        mock_invites_repo.get_invite_by_code.return_value = {
            "id": 1,
            "is_active": False,  # Revoked
        }

        result = await service.validate_invite("REVOKEDCODE")

        assert result.status == InviteStatus.REVOKED
        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_validate_invite_expired(self, service, mock_invites_repo):
        """Validating expired invite should return EXPIRED."""
        yesterday = (datetime.utcnow() - timedelta(days=1)).isoformat()
        mock_invites_repo.get_invite_by_code.return_value = {
            "id": 1,
            "is_active": True,
            "expires_at": yesterday,
            "uses_count": 0,
            "max_uses": 5,
        }

        result = await service.validate_invite("EXPIREDCODE")

        assert result.status == InviteStatus.EXPIRED
        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_validate_invite_exhausted(self, service, mock_invites_repo):
        """Validating exhausted invite should return EXHAUSTED."""
        tomorrow = (datetime.utcnow() + timedelta(days=1)).isoformat()
        mock_invites_repo.get_invite_by_code.return_value = {
            "id": 1,
            "is_active": True,
            "expires_at": tomorrow,
            "uses_count": 5,
            "max_uses": 5,  # At limit
        }

        result = await service.validate_invite("USEDUPCODE")

        assert result.status == InviteStatus.EXHAUSTED
        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_validate_invite_valid(self, service, mock_invites_repo):
        """Validating valid invite should return VALID."""
        tomorrow = (datetime.utcnow() + timedelta(days=1)).isoformat()
        mock_invites_repo.get_invite_by_code.return_value = {
            "id": 1,
            "is_active": True,
            "expires_at": tomorrow,
            "uses_count": 2,
            "max_uses": 5,
        }

        result = await service.validate_invite("VALIDCODE")

        assert result.status == InviteStatus.VALID
        assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_redeem_invite_invalid_code(self, service, mock_invites_repo):
        """Redeeming invalid invite should fail."""
        mock_invites_repo.get_invite_by_code.return_value = None

        result = await service.redeem_invite(code="BADCODE", user_id=1)

        assert result.success is False

    @pytest.mark.asyncio
    async def test_redeem_invite_already_member(self, service, mock_invites_repo, mock_orgs_repo):
        """Redeeming invite when already member should succeed (idempotent)."""
        tomorrow = (datetime.utcnow() + timedelta(days=1)).isoformat()
        mock_invites_repo.get_invite_by_code.return_value = {
            "id": 1,
            "org_id": 1,
            "org_name": "Test Org",
            "is_active": True,
            "expires_at": tomorrow,
            "uses_count": 0,
            "max_uses": 5,
            "role_to_grant": "member",
        }
        mock_orgs_repo.get_org_member.return_value = {"user_id": 100, "role": "member"}
        mock_invites_repo.has_user_redeemed.return_value = True

        result = await service.redeem_invite(code="VALIDCODE", user_id=100)

        assert result.success is True
        assert result.was_already_member is True
        mock_orgs_repo.add_org_member.assert_not_called()  # Should not try to add again

    @pytest.mark.asyncio
    async def test_redeem_invite_new_member(self, service, mock_invites_repo, mock_orgs_repo):
        """Redeeming invite as new member should add to org."""
        tomorrow = (datetime.utcnow() + timedelta(days=1)).isoformat()
        mock_invites_repo.get_invite_by_code.return_value = {
            "id": 1,
            "org_id": 1,
            "org_name": "Test Org",
            "is_active": True,
            "expires_at": tomorrow,
            "uses_count": 0,
            "max_uses": 5,
            "role_to_grant": "member",
        }
        mock_orgs_repo.get_org_member.return_value = None  # Not a member yet

        result = await service.redeem_invite(code="VALIDCODE", user_id=100)

        assert result.success is True
        assert result.was_already_member is False
        mock_orgs_repo.add_org_member.assert_called_once_with(org_id=1, user_id=100, role="member")
        mock_invites_repo.record_redemption.assert_called_once()
        mock_invites_repo.increment_uses_count.assert_called_once()

    @pytest.mark.asyncio
    async def test_redeem_invite_domain_allowlist_mismatch(self, service, mock_invites_repo, mock_orgs_repo):
        """Redeeming invite with mismatched domain should fail."""
        tomorrow = (datetime.utcnow() + timedelta(days=1)).isoformat()
        mock_invites_repo.get_invite_by_code.return_value = {
            "id": 1,
            "org_id": 1,
            "org_name": "Test Org",
            "is_active": True,
            "expires_at": tomorrow,
            "uses_count": 0,
            "max_uses": 5,
            "role_to_grant": "member",
            "allowed_email_domain": "example.com",
        }
        mock_orgs_repo.get_org_member.return_value = None

        result = await service.redeem_invite(
            code="DOMAINCODE",
            user_id=100,
            user_email="user@other.com",
        )

        assert result.success is False
        assert "allowed email domain" in (result.message or "").lower()
        mock_orgs_repo.add_org_member.assert_not_called()
        reset_settings()

    @pytest.mark.asyncio
    async def test_redeem_invite_domain_allowlist_missing_email_blocks(
        self, service, mock_invites_repo, mock_orgs_repo, monkeypatch
    ):
        """Redeeming invite without email should fail when fallback is disabled."""
        monkeypatch.delenv("ORG_INVITE_ALLOW_MISSING_EMAIL", raising=False)
        reset_settings()
        tomorrow = (datetime.utcnow() + timedelta(days=1)).isoformat()
        mock_invites_repo.get_invite_by_code.return_value = {
            "id": 1,
            "org_id": 1,
            "org_name": "Test Org",
            "is_active": True,
            "expires_at": tomorrow,
            "uses_count": 0,
            "max_uses": 5,
            "role_to_grant": "member",
            "allowed_email_domain": "example.com",
        }
        mock_orgs_repo.get_org_member.return_value = None

        result = await service.redeem_invite(
            code="DOMAINMISS",
            user_id=100,
            user_email=None,
        )

        assert result.success is False
        assert "allowed email domain" in (result.message or "").lower()
        mock_orgs_repo.add_org_member.assert_not_called()

    @pytest.mark.asyncio
    async def test_redeem_invite_domain_allowlist_missing_email_fallback(
        self, service, mock_invites_repo, mock_orgs_repo, monkeypatch
    ):
        """Redeeming invite without email should succeed when fallback is enabled."""
        monkeypatch.setenv("ORG_INVITE_ALLOW_MISSING_EMAIL", "true")
        reset_settings()
        tomorrow = (datetime.utcnow() + timedelta(days=1)).isoformat()
        mock_invites_repo.get_invite_by_code.return_value = {
            "id": 1,
            "org_id": 1,
            "org_name": "Test Org",
            "is_active": True,
            "expires_at": tomorrow,
            "uses_count": 0,
            "max_uses": 5,
            "role_to_grant": "member",
            "allowed_email_domain": "example.com",
        }
        mock_orgs_repo.get_org_member.return_value = None

        result = await service.redeem_invite(
            code="DOMAINFALLBACK",
            user_id=100,
            user_email=None,
        )

        assert result.success is True
        mock_orgs_repo.add_org_member.assert_called_once_with(org_id=1, user_id=100, role="member")
        monkeypatch.delenv("ORG_INVITE_ALLOW_MISSING_EMAIL", raising=False)
        reset_settings()

    @pytest.mark.asyncio
    async def test_redeem_invite_domain_allowlist_match(self, service, mock_invites_repo, mock_orgs_repo):
        """Redeeming invite with matching domain should succeed."""
        tomorrow = (datetime.utcnow() + timedelta(days=1)).isoformat()
        mock_invites_repo.get_invite_by_code.return_value = {
            "id": 1,
            "org_id": 1,
            "org_name": "Test Org",
            "is_active": True,
            "expires_at": tomorrow,
            "uses_count": 0,
            "max_uses": 5,
            "role_to_grant": "member",
            "allowed_email_domain": "example.com",
        }
        mock_orgs_repo.get_org_member.return_value = None

        result = await service.redeem_invite(
            code="DOMAINOK",
            user_id=100,
            user_email="User@Example.com",
        )

        assert result.success is True
        mock_orgs_repo.add_org_member.assert_called_once_with(org_id=1, user_id=100, role="member")

    @pytest.mark.asyncio
    async def test_redeem_invite_org_member_add_error_is_sanitized(
        self,
        service,
        mock_invites_repo,
        mock_orgs_repo,
    ):
        """Org membership failures should not expose repository internals."""
        tomorrow = (datetime.utcnow() + timedelta(days=1)).isoformat()
        mock_invites_repo.get_invite_by_code.return_value = {
            "id": 1,
            "org_id": 1,
            "org_name": "Test Org",
            "is_active": True,
            "expires_at": tomorrow,
            "uses_count": 0,
            "max_uses": 5,
            "role_to_grant": "member",
        }
        mock_orgs_repo.get_org_member.return_value = None
        mock_orgs_repo.add_org_member.side_effect = RuntimeError(
            "org repo failed at /private/orgs.db"
        )

        result = await service.redeem_invite(code="VALIDCODE", user_id=100)

        assert result.success is False
        assert result.message == "Failed to join organization"
        assert "org repo failed" not in (result.message or "")
        assert "/private/orgs.db" not in (result.message or "")
        mock_invites_repo.record_redemption.assert_not_called()
        mock_invites_repo.increment_uses_count.assert_not_called()

    @pytest.mark.asyncio
    async def test_redeem_invite_with_team(self, service, mock_invites_repo, mock_orgs_repo):
        """Redeeming team invite should add to both org and team."""
        tomorrow = (datetime.utcnow() + timedelta(days=1)).isoformat()
        mock_invites_repo.get_invite_by_code.return_value = {
            "id": 1,
            "org_id": 1,
            "org_name": "Test Org",
            "team_id": 77,
            "team_name": "Dev Team",
            "is_active": True,
            "expires_at": tomorrow,
            "uses_count": 0,
            "max_uses": 5,
            "role_to_grant": "member",
        }
        mock_orgs_repo.get_org_member.return_value = None

        result = await service.redeem_invite(code="TEAMCODE", user_id=100)

        assert result.success is True
        assert result.team_id == 77
        mock_orgs_repo.add_org_member.assert_called_once()
        mock_orgs_repo.add_team_member.assert_called_once_with(team_id=77, user_id=100, role="member")

    @pytest.mark.asyncio
    async def test_revoke_invite_success(self, service, mock_invites_repo):
        """Revoking invite should deactivate it."""
        mock_invites_repo.deactivate_invite.return_value = True

        result = await service.revoke_invite(invite_id=1, org_id=1)

        assert result is True
        mock_invites_repo.deactivate_invite.assert_called_once_with(1, 1)

    @pytest.mark.asyncio
    async def test_revoke_invite_not_found(self, service, mock_invites_repo):
        """Revoking nonexistent invite should return False."""
        mock_invites_repo.deactivate_invite.return_value = False

        result = await service.revoke_invite(invite_id=999, org_id=1)

        assert result is False

    @pytest.mark.asyncio
    async def test_list_org_invites(self, service, mock_invites_repo):
        """Listing org invites should return invites and count."""
        mock_invites_repo.list_org_invites.return_value = (
            [{"id": 1}, {"id": 2}],
            2,
        )

        invites, count = await service.list_org_invites(org_id=1)

        assert len(invites) == 2
        assert count == 2

    @pytest.mark.asyncio
    async def test_preview_invite_valid(self, service, mock_invites_repo):
        """Preview of valid invite should return public info."""
        tomorrow = (datetime.utcnow() + timedelta(days=1)).isoformat()
        mock_invites_repo.get_invite_by_code.return_value = {
            "id": 1,
            "org_name": "Test Org",
            "org_slug": "test-org",
            "team_name": None,
            "role_to_grant": "member",
            "is_active": True,
            "expires_at": tomorrow,
            "uses_count": 0,
            "max_uses": 5,
        }

        preview = await service.preview_invite("VALIDCODE")

        assert preview is not None
        assert preview["org_name"] == "Test Org"
        assert preview["is_valid"] is True
        assert preview["status"] == "valid"

    @pytest.mark.asyncio
    async def test_preview_invite_not_found(self, service, mock_invites_repo):
        """Preview of nonexistent invite should return None."""
        mock_invites_repo.get_invite_by_code.return_value = None

        preview = await service.preview_invite("BADCODE")

        assert preview is None

    @pytest.mark.asyncio
    async def test_cleanup_expired_invites(self, service, mock_invites_repo):
        """Cleanup should deactivate expired invites."""
        mock_invites_repo.cleanup_expired_invites.return_value = 3

        count = await service.cleanup_expired_invites()

        assert count == 3
        mock_invites_repo.cleanup_expired_invites.assert_called_once_with(deactivate_only=True)
