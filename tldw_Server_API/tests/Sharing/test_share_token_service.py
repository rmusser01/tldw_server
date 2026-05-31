"""Unit tests for ShareTokenService."""
from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Sharing.share_token_service import ShareTokenService

pytestmark = pytest.mark.unit


@pytest.fixture
def token_service(repo):
    return ShareTokenService(repo)


@pytest.mark.asyncio
async def test_generate_token(token_service):
    result = await token_service.generate_token(
        resource_type="workspace",
        resource_id="ws-1",
        owner_user_id=1,
    )
    assert "raw_token" in result
    assert len(result["raw_token"]) > 20
    assert result["resource_type"] == "workspace"
    assert result["resource_id"] == "ws-1"
    assert result["use_count"] == 0


@pytest.mark.asyncio
async def test_validate_token_success(token_service):
    result = await token_service.generate_token(
        resource_type="workspace",
        resource_id="ws-1",
        owner_user_id=1,
    )
    raw = result["raw_token"]
    validated = await token_service.validate_token(raw)
    assert validated is not None
    assert validated["resource_id"] == "ws-1"


@pytest.mark.asyncio
async def test_validate_token_invalid(token_service):
    validated = await token_service.validate_token("not-a-valid-token-at-all-1234567890")
    assert validated is None


@pytest.mark.asyncio
async def test_validate_token_expired(token_service):
    result = await token_service.generate_token(
        resource_type="workspace",
        resource_id="ws-1",
        owner_user_id=1,
        expires_at="2020-01-01T00:00:00+00:00",
    )
    validated = await token_service.validate_token(result["raw_token"])
    assert validated is None


@pytest.mark.asyncio
async def test_validate_token_max_uses_exceeded(token_service):
    result = await token_service.generate_token(
        resource_type="workspace",
        resource_id="ws-1",
        owner_user_id=1,
        max_uses=1,
    )
    raw = result["raw_token"]
    # Use the token once
    await token_service.use_token(result["id"])
    validated = await token_service.validate_token(raw)
    assert validated is None


@pytest.mark.asyncio
async def test_password_protection(token_service):
    result = await token_service.generate_token(
        resource_type="chatbook",
        resource_id="cb-1",
        owner_user_id=1,
        password="secret123",
    )
    raw = result["raw_token"]
    validated = await token_service.validate_token(raw)
    assert validated is not None
    assert validated["is_password_protected"] is True

    # Correct password
    assert await token_service.verify_password(validated, "secret123") is True
    # Wrong password
    assert await token_service.verify_password(validated, "wrong") is False


@pytest.mark.asyncio
async def test_no_password_always_passes(token_service):
    result = await token_service.generate_token(
        resource_type="workspace",
        resource_id="ws-1",
        owner_user_id=1,
    )
    validated = await token_service.validate_token(result["raw_token"])
    assert await token_service.verify_password(validated, "anything") is True


@pytest.mark.asyncio
async def test_use_token_increments(token_service, repo):
    result = await token_service.generate_token(
        resource_type="workspace",
        resource_id="ws-1",
        owner_user_id=1,
    )
    await token_service.use_token(result["id"])
    await token_service.use_token(result["id"])
    fetched = await repo.get_token(result["id"])
    assert fetched["use_count"] == 2


@pytest.mark.asyncio
async def test_claim_token_use_honors_max_uses(token_service, repo):
    result = await token_service.generate_token(
        resource_type="workspace",
        resource_id="ws-1",
        owner_user_id=1,
        max_uses=1,
    )
    assert await token_service.claim_token_use(result["id"]) is True
    assert await token_service.claim_token_use(result["id"]) is False

    fetched = await repo.get_token(result["id"])
    assert fetched["use_count"] == 1


@pytest.mark.asyncio
async def test_revoke_token(token_service):
    result = await token_service.generate_token(
        resource_type="workspace",
        resource_id="ws-1",
        owner_user_id=1,
    )
    await token_service.revoke_token(result["id"])
    # Revoked tokens are excluded from find_tokens_by_prefix
    validated = await token_service.validate_token(result["raw_token"])
    assert validated is None


@pytest.mark.asyncio
async def test_list_tokens_strips_sensitive(token_service):
    await token_service.generate_token(
        resource_type="workspace",
        resource_id="ws-1",
        owner_user_id=1,
        password="secret",
    )
    tokens = await token_service.list_tokens(1)
    assert len(tokens) == 1
    assert "token_hash" not in tokens[0]
    assert "password_hash" not in tokens[0]


@pytest.mark.asyncio
async def test_revoke_tokens_for_resource(token_service):
    await token_service.generate_token(
        resource_type="workspace", resource_id="ws-1", owner_user_id=1,
    )
    await token_service.generate_token(
        resource_type="workspace", resource_id="ws-1", owner_user_id=1,
    )
    await token_service.revoke_tokens_for_resource("workspace", "ws-1", 1)
    tokens = await token_service.list_tokens(1)
    # All should be revoked
    assert all(t.get("is_revoked") or t.get("revoked_at") is not None for t in tokens)
