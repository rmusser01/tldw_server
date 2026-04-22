"""Tests for the ACP webhook trigger system.

Covers:
1. GitHub HMAC signature verification (valid + invalid)
2. Slack signature verification (valid + replay attack)
3. Generic HMAC signature verification
4. Per-trigger rate limiting
5. Secret encryption/decryption roundtrip (Fernet and base64 fallback)
6. End-to-end handle_webhook -> acp_run submission
7. Disabled trigger rejection
8. Full CRUD lifecycle
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.triggers import (
    ACPTriggerManager,
    TriggerConfig,
    TriggerSecretManager,
    WebhookVerifier,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_github_signature(payload: bytes, secret: str) -> str:
    """Produce a valid GitHub X-Hub-Signature-256 header value."""
    return "sha256=" + hmac.new(
        secret.encode("utf-8"), payload, hashlib.sha256
    ).hexdigest()


def _make_slack_signature(payload: bytes, secret: str, timestamp: str) -> str:
    """Produce a valid Slack X-Slack-Signature header value."""
    sig_basestring = f"v0:{timestamp}:{payload.decode('utf-8')}"
    return "v0=" + hmac.new(
        secret.encode("utf-8"),
        sig_basestring.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _make_generic_signature(payload: bytes, secret: str, timestamp: str | None = None) -> str:
    """Produce a valid generic X-Webhook-Signature header value.

    When *timestamp* is provided the signed payload is ``<ts>.<body>``
    (matching the verifier's replay-protection scheme).
    """
    if timestamp is not None:
        signed_payload = f"{timestamp}.".encode("utf-8") + payload
    else:
        signed_payload = payload
    return hmac.new(
        secret.encode("utf-8"), signed_payload, hashlib.sha256
    ).hexdigest()


class FakeDB:
    """In-memory stand-in for ACPSessionsDB webhook trigger methods."""

    def __init__(self) -> None:
        self._triggers: dict[str, dict[str, Any]] = {}

    def create_webhook_trigger(self, **kwargs: Any) -> None:
        self._triggers[kwargs["trigger_id"]] = kwargs

    def list_webhook_triggers(self, owner_user_id: int) -> list[dict[str, Any]]:
        return [
            dict(t) for t in self._triggers.values()
            if t["owner_user_id"] == owner_user_id
        ]

    def get_webhook_trigger(self, trigger_id: str) -> dict[str, Any] | None:
        return dict(self._triggers[trigger_id]) if trigger_id in self._triggers else None

    def update_webhook_trigger(self, trigger_id: str, updates: dict[str, Any]) -> bool:
        if trigger_id not in self._triggers:
            return False
        self._triggers[trigger_id].update(updates)
        return True

    def delete_webhook_trigger(self, trigger_id: str) -> bool:
        if trigger_id in self._triggers:
            del self._triggers[trigger_id]
            return True
        return False


@pytest.fixture()
def secret_mgr() -> TriggerSecretManager:
    """A TriggerSecretManager with a fixed key for deterministic tests."""
    return TriggerSecretManager(encryption_key="test-key-for-unit-tests")


@pytest.fixture()
def db() -> FakeDB:
    return FakeDB()


@pytest.fixture()
def mgr(db: FakeDB, secret_mgr: TriggerSecretManager) -> ACPTriggerManager:
    return ACPTriggerManager(db=db, secret_manager=secret_mgr)


# ---------------------------------------------------------------------------
# 1. GitHub signature verification -- valid
# ---------------------------------------------------------------------------


def test_verify_github_valid_signature():
    secret = "gh-secret-123"
    payload = b'{"action": "push", "ref": "refs/heads/main"}'
    sig = _make_github_signature(payload, secret)
    v = WebhookVerifier()
    assert v.verify_github(payload, sig, secret, delivery_id="d-1") is True


# ---------------------------------------------------------------------------
# 2. GitHub signature verification -- invalid
# ---------------------------------------------------------------------------


def test_verify_github_invalid_signature():
    secret = "gh-secret-123"
    payload = b'{"action": "push"}'
    bad_sig = "sha256=0000000000000000000000000000000000000000000000000000000000000000"
    v = WebhookVerifier()
    assert v.verify_github(payload, bad_sig, secret) is False


def test_verify_github_tampered_payload():
    secret = "gh-secret-123"
    payload = b'{"action": "push"}'
    sig = _make_github_signature(payload, secret)
    v = WebhookVerifier()
    tampered = b'{"action": "delete"}'
    assert v.verify_github(tampered, sig, secret) is False


def test_verify_github_replay_rejected():
    """Same delivery_id is rejected on second use."""
    secret = "gh-secret-123"
    payload = b'{"action": "push"}'
    sig = _make_github_signature(payload, secret)
    v = WebhookVerifier()
    assert v.verify_github(payload, sig, secret, delivery_id="dup-1") is True
    assert v.verify_github(payload, sig, secret, delivery_id="dup-1") is False


# ---------------------------------------------------------------------------
# 3. Slack signature verification -- valid
# ---------------------------------------------------------------------------


def test_verify_slack_valid_signature():
    secret = "slack-signing-secret"
    payload = b"token=abc&team_id=T123"
    timestamp = str(int(time.time()))
    sig = _make_slack_signature(payload, secret, timestamp)

    assert WebhookVerifier.verify_slack(payload, timestamp, sig, secret) is True


# ---------------------------------------------------------------------------
# 4. Slack signature verification -- replay attack
# ---------------------------------------------------------------------------


def test_verify_slack_replay_attack():
    secret = "slack-signing-secret"
    payload = b"token=abc"
    # Timestamp 10 minutes ago -- beyond the 5 minute window
    old_timestamp = str(int(time.time()) - 600)
    sig = _make_slack_signature(payload, secret, old_timestamp)

    assert WebhookVerifier.verify_slack(payload, old_timestamp, sig, secret) is False


def test_verify_slack_invalid_timestamp():
    """Non-numeric timestamp should fail gracefully."""
    secret = "slack-signing-secret"
    payload = b"data"
    assert WebhookVerifier.verify_slack(payload, "not-a-number", "v0=abc", secret) is False


# ---------------------------------------------------------------------------
# 5. Rate limit blocks after max
# ---------------------------------------------------------------------------


def test_rate_limit_blocks_after_max(mgr: ACPTriggerManager):
    trigger_id = "rate-test-1"
    max_per_minute = 5

    # First 5 should pass
    for i in range(max_per_minute):
        assert mgr.check_rate_limit(trigger_id, max_per_minute=max_per_minute) is True, \
            f"Request {i+1} should be within limit"

    # 6th should be blocked
    assert mgr.check_rate_limit(trigger_id, max_per_minute=max_per_minute) is False


def test_rate_limit_resets_after_window(mgr: ACPTriggerManager):
    """Timestamps older than 60s are pruned, freeing up capacity."""
    trigger_id = "rate-test-2"
    max_per_minute = 2

    # Manually inject old timestamps
    old_time = time.time() - 61
    mgr._rate_limits[trigger_id] = [old_time, old_time]

    # Should still pass because old entries get pruned
    assert mgr.check_rate_limit(trigger_id, max_per_minute=max_per_minute) is True


# ---------------------------------------------------------------------------
# 6. Secret encrypt/decrypt roundtrip
# ---------------------------------------------------------------------------


def test_secret_encrypt_decrypt_roundtrip(secret_mgr: TriggerSecretManager):
    original = "my-webhook-secret-456"
    encrypted = secret_mgr.encrypt(original)
    assert encrypted != original, "Encrypted value should differ from plaintext"

    decrypted = secret_mgr.decrypt(encrypted)
    assert decrypted == original


def test_secret_manager_no_key_raises():
    """TriggerSecretManager without a key should raise ValueError."""
    with pytest.raises(ValueError, match="ACP_TRIGGER_ENCRYPTION_KEY"):
        TriggerSecretManager()


def test_secret_manager_generate_key():
    """generate_key() returns a usable Fernet key."""
    key = TriggerSecretManager.generate_key()
    assert isinstance(key, str)
    # Key should be usable to create a new manager
    mgr = TriggerSecretManager(encryption_key=key)
    original = "roundtrip-test"
    assert mgr.decrypt(mgr.encrypt(original)) == original


# ---------------------------------------------------------------------------
# 7. End-to-end: handle_webhook submits acp_run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_webhook_submits_acp_run(
    mgr: ACPTriggerManager,
    secret_mgr: TriggerSecretManager,
):
    webhook_secret = "e2e-secret"

    # Create a trigger
    trigger_id = mgr.create_trigger(
        name="E2E Test",
        source_type="generic",
        secret=webhook_secret,
        owner_user_id=42,
        agent_config={"cwd": "/workspace", "agent_type": "coding"},
        prompt_template="Event: {event_type}\nPayload: {payload}",
    )

    payload = b'{"ref": "main", "commits": [{"message": "fix bug"}]}'
    timestamp = str(int(time.time()))
    sig = _make_generic_signature(payload, webhook_secret, timestamp=timestamp)

    # Mock the scheduler submission
    with patch.object(mgr, "_submit_acp_run", new_callable=AsyncMock) as mock_submit:
        mock_submit.return_value = "task-abc-123"

        result = await mgr.handle_webhook(
            trigger_id=trigger_id,
            payload_body=payload,
            headers={
                "x-webhook-signature": sig,
                "x-webhook-timestamp": timestamp,
            },
        )

    assert result["status"] == "accepted"
    assert result["task_id"] == "task-abc-123"

    # Verify the acp_run payload
    call_args = mock_submit.call_args
    acp_payload = call_args[0][0]
    assert acp_payload["user_id"] == 42
    assert acp_payload["cwd"] == "/workspace"
    assert acp_payload["agent_type"] == "coding"
    assert "Event: webhook" in acp_payload["prompt"]
    assert "fix bug" in acp_payload["prompt"]


@pytest.mark.asyncio
async def test_handle_webhook_github_provider(
    mgr: ACPTriggerManager,
    secret_mgr: TriggerSecretManager,
):
    """GitHub source_type uses X-Hub-Signature-256 and X-GitHub-Event headers."""
    webhook_secret = "gh-test-secret"

    trigger_id = mgr.create_trigger(
        name="GitHub Trigger",
        source_type="github",
        secret=webhook_secret,
        owner_user_id=1,
        prompt_template="GitHub {event_type}: {payload}",
    )

    payload = b'{"action": "opened", "number": 42}'
    sig = _make_github_signature(payload, webhook_secret)

    with patch.object(mgr, "_submit_acp_run", new_callable=AsyncMock) as mock_submit:
        mock_submit.return_value = "task-gh-1"

        result = await mgr.handle_webhook(
            trigger_id=trigger_id,
            payload_body=payload,
            headers={
                "X-Hub-Signature-256": sig,
                "X-GitHub-Event": "pull_request",
                "X-GitHub-Delivery": "delivery-gh-e2e-1",
            },
        )

    assert result["status"] == "accepted"
    acp_payload = mock_submit.call_args[0][0]
    assert "GitHub pull_request" in acp_payload["prompt"]


# ---------------------------------------------------------------------------
# 8. Disabled trigger returns error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_webhook_disabled_trigger(
    mgr: ACPTriggerManager,
    secret_mgr: TriggerSecretManager,
):
    trigger_id = mgr.create_trigger(
        name="Disabled",
        source_type="generic",
        secret="some-secret",
        owner_user_id=1,
        enabled=False,
    )

    result = await mgr.handle_webhook(
        trigger_id=trigger_id,
        payload_body=b"test",
        headers={"x-webhook-signature": "doesn't matter"},
    )

    assert result["status"] == "rejected"
    assert result["error"] == "trigger_disabled"


@pytest.mark.asyncio
async def test_handle_webhook_unknown_trigger(mgr: ACPTriggerManager):
    result = await mgr.handle_webhook(
        trigger_id="nonexistent-id",
        payload_body=b"test",
        headers={},
    )

    assert result["status"] == "rejected"
    assert result["error"] == "trigger_not_found"


@pytest.mark.asyncio
async def test_handle_webhook_invalid_signature(
    mgr: ACPTriggerManager,
    secret_mgr: TriggerSecretManager,
):
    trigger_id = mgr.create_trigger(
        name="Bad Sig Test",
        source_type="generic",
        secret="correct-secret",
        owner_user_id=1,
    )

    result = await mgr.handle_webhook(
        trigger_id=trigger_id,
        payload_body=b"data",
        headers={
            "x-webhook-signature": "wrong-signature",
            "x-webhook-timestamp": str(int(time.time())),
        },
    )

    assert result["status"] == "rejected"
    assert result["error"] == "verification_failed"


@pytest.mark.asyncio
async def test_handle_webhook_submission_failure_hides_exception_details(
    mgr: ACPTriggerManager,
) -> None:
    trigger_id = mgr.create_trigger(
        name="Submission Failure",
        source_type="generic",
        secret="submit-secret",
        owner_user_id=7,
    )
    payload = b'{"event":"push"}'
    timestamp = str(int(time.time()))
    sig = _make_generic_signature(payload, "submit-secret", timestamp=timestamp)

    with patch.object(mgr, "_submit_acp_run", new_callable=AsyncMock) as mock_submit:
        mock_submit.side_effect = RuntimeError("database password=super-secret")

        result = await mgr.handle_webhook(
            trigger_id=trigger_id,
            payload_body=payload,
            headers={
                "x-webhook-signature": sig,
                "x-webhook-timestamp": timestamp,
            },
        )

    assert result == {"error": "submission_failed", "status": "error"}


# ---------------------------------------------------------------------------
# 9. Trigger CRUD lifecycle
# ---------------------------------------------------------------------------


def test_trigger_crud_lifecycle(
    mgr: ACPTriggerManager,
    secret_mgr: TriggerSecretManager,
):
    # Create
    trigger_id = mgr.create_trigger(
        name="Lifecycle Test",
        source_type="github",
        secret="lifecycle-secret",
        owner_user_id=99,
        agent_config={"cwd": "/app"},
        prompt_template="Handle: {payload}",
        enabled=True,
    )
    assert trigger_id is not None

    # List
    triggers = mgr.list_triggers(user_id=99)
    assert len(triggers) == 1
    assert triggers[0]["name"] == "Lifecycle Test"
    assert triggers[0]["source_type"] == "github"
    # Secret should NOT be in list results
    assert "secret_encrypted" not in triggers[0]

    # Get
    trigger = mgr.get_trigger(trigger_id)
    assert trigger is not None
    assert trigger["name"] == "Lifecycle Test"
    assert trigger["owner_user_id"] == 99
    assert trigger["agent_config"] == {"cwd": "/app"}
    assert trigger["prompt_template"] == "Handle: {payload}"
    assert trigger["enabled"] is True
    # get_trigger includes secret_encrypted for internal use
    assert "secret_encrypted" in trigger

    # Update
    ok = mgr.update_trigger(
        trigger_id,
        name="Updated Name",
        source_type="slack",
        enabled=False,
        agent_config={"cwd": "/new"},
        prompt_template="New template: {payload}",
    )
    assert ok is True

    updated = mgr.get_trigger(trigger_id)
    assert updated is not None
    assert updated["name"] == "Updated Name"
    assert updated["source_type"] == "slack"
    assert updated["enabled"] is False
    assert updated["agent_config"] == {"cwd": "/new"}
    assert updated["prompt_template"] == "New template: {payload}"

    # Update secret
    ok = mgr.update_trigger(trigger_id, secret="new-secret")
    assert ok is True
    updated = mgr.get_trigger(trigger_id)
    decrypted = secret_mgr.decrypt(updated["secret_encrypted"])
    assert decrypted == "new-secret"

    # Delete
    ok = mgr.delete_trigger(trigger_id)
    assert ok is True
    assert mgr.get_trigger(trigger_id) is None

    # List should be empty now
    assert mgr.list_triggers(user_id=99) == []

    # Delete again should return False
    assert mgr.delete_trigger(trigger_id) is False


def test_list_triggers_filters_by_user(
    mgr: ACPTriggerManager,
    secret_mgr: TriggerSecretManager,
):
    """list_triggers only returns triggers for the specified user."""
    mgr.create_trigger(name="User1 Trigger", source_type="generic",
                        secret="s1", owner_user_id=1)
    mgr.create_trigger(name="User2 Trigger", source_type="generic",
                        secret="s2", owner_user_id=2)

    user1_triggers = mgr.list_triggers(user_id=1)
    user2_triggers = mgr.list_triggers(user_id=2)

    assert len(user1_triggers) == 1
    assert user1_triggers[0]["name"] == "User1 Trigger"
    assert len(user2_triggers) == 1
    assert user2_triggers[0]["name"] == "User2 Trigger"


def test_update_trigger_no_changes(mgr: ACPTriggerManager):
    """update_trigger with no kwargs returns False."""
    trigger_id = mgr.create_trigger(
        name="No-op", source_type="generic", secret="s", owner_user_id=1,
    )
    assert mgr.update_trigger(trigger_id) is False


# ---------------------------------------------------------------------------
# 10. Generic HMAC verification
# ---------------------------------------------------------------------------


def test_verify_generic_valid():
    secret = "generic-secret"
    payload = b"some data"
    timestamp = str(int(time.time()))
    sig = _make_generic_signature(payload, secret, timestamp=timestamp)
    assert WebhookVerifier.verify_generic(payload, sig, secret, timestamp=timestamp) is True


def test_verify_generic_invalid():
    timestamp = str(int(time.time()))
    assert WebhookVerifier.verify_generic(b"data", "badsig", "secret", timestamp=timestamp) is False


def test_verify_generic_missing_timestamp():
    """Generic webhook without timestamp is rejected."""
    secret = "generic-secret"
    payload = b"some data"
    sig = _make_generic_signature(payload, secret)
    assert WebhookVerifier.verify_generic(payload, sig, secret, timestamp=None) is False


def test_verify_generic_replay_attack():
    """Generic webhook with old timestamp is rejected."""
    secret = "generic-secret"
    payload = b"some data"
    old_timestamp = str(int(time.time()) - 600)
    sig = _make_generic_signature(payload, secret, timestamp=old_timestamp)
    assert WebhookVerifier.verify_generic(payload, sig, secret, timestamp=old_timestamp) is False


# ---------------------------------------------------------------------------
# 11. TriggerConfig dataclass
# ---------------------------------------------------------------------------


def test_trigger_config_to_dict():
    cfg = TriggerConfig(
        id="t1",
        name="Test",
        source_type="github",
        secret_encrypted="enc:abc",
        owner_user_id=1,
        agent_config={"cwd": "/x"},
        prompt_template="tmpl",
    )
    d = cfg.to_dict()
    assert d["id"] == "t1"
    assert d["name"] == "Test"
    assert d["source_type"] == "github"
    assert d["agent_config"] == {"cwd": "/x"}
    assert d["enabled"] is True
    # secret_encrypted should NOT be in to_dict output
    assert "secret_encrypted" not in d


# ---------------------------------------------------------------------------
# 12. Prompt template rendering
# ---------------------------------------------------------------------------


def test_render_prompt_with_placeholders():
    template = "Event: {event_type}\nData: {payload}"
    payload = b'{"key": "value"}'
    headers = {"x-github-event": "push"}

    result = ACPTriggerManager._render_prompt(template, payload, headers)
    assert "Event: push" in result
    assert '{"key": "value"}' in result


def test_render_prompt_default_event_type():
    """When no event header is present, event_type defaults to 'webhook'."""
    template = "Type: {event_type}"
    result = ACPTriggerManager._render_prompt(template, b"data", {})
    assert "Type: webhook" in result


def test_render_prompt_unknown_placeholders():
    """Extra placeholders that aren't supported should not crash."""
    template = "Event: {event_type}, Unknown: {unknown}"
    result = ACPTriggerManager._render_prompt(template, b"data", {})
    # Should at least contain the event_type substitution
    assert "Event: webhook" in result


# ---------------------------------------------------------------------------
# 13. DB integration (ACPSessionsDB webhook_triggers table)
# ---------------------------------------------------------------------------


def test_acp_sessions_db_webhook_trigger_crud(tmp_path):
    """Test webhook trigger CRUD directly against ACPSessionsDB."""
    from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB

    db_path = str(tmp_path / "test_triggers.db")
    db = ACPSessionsDB(db_path=db_path)

    # Create
    db.create_webhook_trigger(
        trigger_id="t-1",
        name="DB Test",
        source_type="github",
        secret_encrypted="enc:secret",
        owner_user_id=42,
        agent_config_json='{"cwd": "/work"}',
        prompt_template="test: {payload}",
        enabled=True,
    )

    # Get
    trigger = db.get_webhook_trigger("t-1")
    assert trigger is not None
    assert trigger["name"] == "DB Test"
    assert trigger["source_type"] == "github"
    assert trigger["owner_user_id"] == 42
    assert trigger["agent_config_json"] == '{"cwd": "/work"}'
    assert trigger["prompt_template"] == "test: {payload}"
    assert trigger["enabled"] == 1

    # List
    triggers = db.list_webhook_triggers(42)
    assert len(triggers) == 1
    assert triggers[0]["id"] == "t-1"

    # List for different user
    assert db.list_webhook_triggers(999) == []

    # Update
    ok = db.update_webhook_trigger("t-1", {"name": "Updated", "enabled": 0})
    assert ok is True

    updated = db.get_webhook_trigger("t-1")
    assert updated["name"] == "Updated"
    assert updated["enabled"] == 0

    # Delete
    ok = db.delete_webhook_trigger("t-1")
    assert ok is True
    assert db.get_webhook_trigger("t-1") is None

    # Delete non-existent
    assert db.delete_webhook_trigger("t-1") is False

    db.close()


def test_acp_sessions_db_schema_version(tmp_path):
    """Verify the schema version matches the current ACP sessions DB migration level."""
    from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB, _SCHEMA_VERSION

    db_path = str(tmp_path / "test_version.db")
    db = ACPSessionsDB(db_path=db_path)
    conn = db._get_conn()
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == _SCHEMA_VERSION
    db.close()
