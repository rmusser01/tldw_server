"""A webhook trigger whose owner is security-locked submits nothing.

The trigger tests have no AuthNZ database, so ``is_user_security_locked`` is replaced in ``account_lock``
(read at call time by ``handle_webhook``) with a plain function that answers for the locked id only.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest

from tldw_Server_API.app.core.AuthNZ import account_lock
from tldw_Server_API.app.core.Agent_Client_Protocol.triggers import ACPTriggerManager
from tldw_Server_API.tests.Agent_Client_Protocol.test_webhook_triggers import (  # noqa: F401  (fixtures)
    _make_generic_signature,
    db,
    mgr,
    secret_mgr,
)

LOCKED_ID = 42


async def _fire(mgr: ACPTriggerManager, owner_user_id: int, monkeypatch):
    async def _is_locked(user_id: int) -> bool:
        return user_id == LOCKED_ID

    monkeypatch.setattr(account_lock, "is_user_security_locked", _is_locked)
    trigger_id = mgr.create_trigger(name="t", source_type="generic", secret="s3cret", owner_user_id=owner_user_id)
    payload = b'{"ref": "main"}'
    timestamp = str(int(time.time()))
    headers = {
        "x-webhook-signature": _make_generic_signature(payload, "s3cret", timestamp=timestamp),
        "x-webhook-timestamp": timestamp,
    }
    with patch.object(mgr, "_submit_acp_run", new_callable=AsyncMock) as submit:
        submit.return_value = "task-1"
        result = await mgr.handle_webhook(trigger_id=trigger_id, payload_body=payload, headers=headers)
    return result, submit


@pytest.mark.asyncio
async def test_acp_trigger_webhook_of_a_locked_owner_submits_nothing(mgr, monkeypatch) -> None:
    result, submit = await _fire(mgr, LOCKED_ID, monkeypatch)

    assert result == {"error": "account_locked", "status": "rejected"}
    assert submit.await_count == 0


@pytest.mark.asyncio
async def test_acp_trigger_webhook_of_an_unlocked_owner_still_submits(mgr, monkeypatch) -> None:
    result, submit = await _fire(mgr, 43, monkeypatch)

    assert result == {"task_id": "task-1", "status": "accepted"}
    assert submit.await_count == 1
