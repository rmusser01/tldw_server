"""A Slack command mapped to a locked tldw user queues nothing; other actor ids are untouched.

The Slack fixtures have no AuthNZ database, so ``is_user_security_locked`` is replaced in ``account_lock``
(read at call time by the chatops policy helper) with a plain function that answers for the locked id only.
"""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.AuthNZ import account_lock
from tldw_Server_API.tests.Slack.test_slack_policy_hardening import (  # noqa: F401  (fixtures)
    _post_command,
    slack_policy_client,
)

LOCKED_ID = 77


@pytest.fixture(autouse=True)
def _only_77_is_locked(monkeypatch):
    async def _is_locked(user_id: int) -> bool:
        return user_id == LOCKED_ID

    monkeypatch.setattr(account_lock, "is_user_security_locked", _is_locked)


def _map(client, slack_user_id: str, mapped_to: str) -> None:
    put = client.put(
        "/api/v1/slack/admin/policy",
        json={"team_id": "T1", "user_mappings": {slack_user_id: mapped_to}, "allowed_commands": ["ask"]},
    )
    assert put.status_code == 200


def test_slack_command_mapped_to_a_locked_user_queues_nothing(slack_policy_client) -> None:
    client, jm = slack_policy_client
    _map(client, "U1", str(LOCKED_ID))

    response = _post_command(client, team_id="T1", user_id="U1", text="ask hello", trigger_id="lock-1")

    assert response.status_code == 403
    assert response.json() == {"ok": False, "error": "account_locked", "message": "This account is locked"}
    assert jm._jobs == {}


def test_slack_command_mapped_to_an_unlocked_user_still_queues(slack_policy_client) -> None:
    client, jm = slack_policy_client
    _map(client, "U1", "78")

    response = _post_command(client, team_id="T1", user_id="U1", text="ask hello", trigger_id="lock-2")

    assert response.json()["status"] == "queued"
    assert [job["owner_user_id"] for job in jm._jobs.values()] == ["78"]


def test_slack_actor_that_is_not_a_tldw_user_id_is_untouched(slack_policy_client) -> None:
    client, jm = slack_policy_client

    response = _post_command(client, team_id="T1", user_id="U123", text="ask hello", trigger_id="lock-3")

    assert response.json()["status"] == "queued"
    assert [job["owner_user_id"] for job in jm._jobs.values()] == ["U123"]
