"""A Discord interaction mapped to a locked tldw user queues nothing; other actor ids are untouched.

The Discord fixtures have no AuthNZ database, so ``is_user_security_locked`` is replaced in ``account_lock``
(read at call time by the chatops policy helper) with a plain function that answers for the locked id only.
"""

from __future__ import annotations

import pytest

from tldw_Server_API.app.api.v1.utils.chatops.policy import locked_actor_error
from tldw_Server_API.app.core.AuthNZ import account_lock
from tldw_Server_API.tests.Discord.test_discord_policy_hardening import (  # noqa: F401  (fixtures)
    _post_interaction,
    discord_policy_client,
)

LOCKED_ID = 77


@pytest.fixture(autouse=True)
def _only_77_is_locked(monkeypatch):
    async def _is_locked(user_id: int) -> bool:
        return user_id == LOCKED_ID

    monkeypatch.setattr(account_lock, "is_user_security_locked", _is_locked)


def _map(client, discord_user_id: str, mapped_to: str) -> None:
    put = client.put(
        "/api/v1/discord/admin/policy",
        json={"guild_id": "G1", "user_mappings": {discord_user_id: mapped_to}, "allowed_commands": ["ask"]},
    )
    assert put.status_code == 200


def _ask(client, signer, discord_user_id: str, interaction_id: str):
    return _post_interaction(
        client, signer, interaction_id=interaction_id, guild_id="G1", user_id=discord_user_id,
        command_name="ask", command_input="hello",
    )


def test_discord_interaction_mapped_to_a_locked_user_queues_nothing(discord_policy_client) -> None:
    client, signer, jm = discord_policy_client
    _map(client, "U1", str(LOCKED_ID))

    response = _ask(client, signer, "U1", "lock-1")

    assert response.status_code == 403
    assert response.json() == {"ok": False, "error": "account_locked", "message": "This account is locked"}
    assert jm._jobs == {}


def test_discord_interaction_mapped_to_an_unlocked_user_still_queues(discord_policy_client) -> None:
    client, signer, jm = discord_policy_client
    _map(client, "U1", "78")

    response = _ask(client, signer, "U1", "lock-2")

    assert response.json()["status"] == "queued"
    assert [job["owner_user_id"] for job in jm._jobs.values()] == ["78"]


def test_discord_actor_that_is_not_a_tldw_user_id_is_untouched(discord_policy_client) -> None:
    client, signer, jm = discord_policy_client

    response = _ask(client, signer, "U123", "lock-3")

    assert response.json()["status"] == "queued"
    assert [job["owner_user_id"] for job in jm._jobs.values()] == ["U123"]


@pytest.mark.parametrize(
    ("actor_id", "looked_up"),
    [("123456789012345678", []), ("0", []), ("78", [78])],
    ids=["discord-snowflake", "zero", "user-id"],
)
async def test_only_an_actor_id_in_the_user_id_range_is_looked_up(monkeypatch, actor_id, looked_up) -> None:
    """An unmapped Discord actor's snowflake does not fit users.id (int4 on PostgreSQL): no lookup, no 500."""
    calls: list[int] = []

    async def _record(user_id: int) -> bool:
        calls.append(user_id)
        return False

    monkeypatch.setattr(account_lock, "is_user_security_locked", _record)
    assert await locked_actor_error(actor_id) is None
    assert calls == looked_up
