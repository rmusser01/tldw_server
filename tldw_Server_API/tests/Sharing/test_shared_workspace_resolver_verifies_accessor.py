"""The shared-workspace resolver must verify the accessor, not trust it.

resolve() took accessor_user_id, looked the share up by primary key with
get_share, and echoed the accessor straight back into the returned context
without ever checking that this accessor was a recipient. Any authenticated
caller reaching this code could open any non-revoked share by guessing its id.

The correct primitive was already next door: get_active_share_for_user applies
the recipient, team and org membership checks. The live access service uses it;
the resolver did not.
"""

import pytest

from tldw_Server_API.app.core.Sharing.shared_workspace_resolver import (
    SharedWorkspaceDBResolver,
)

pytestmark = pytest.mark.unit

SHARE = {
    "id": 1,
    "workspace_id": "ws-1",
    "owner_user_id": 7,
    "access_level": "view_chat",
    "allow_clone": False,
}


class _Repo:
    """Grants the share only to the accounts the share is actually shared with."""

    def __init__(self, permitted: set[int]):
        self._permitted = permitted
        self.pk_lookups = 0

    async def get_share(self, share_id):
        self.pk_lookups += 1
        return dict(SHARE)

    async def get_active_share_for_user(self, share_id, user_id):
        return dict(SHARE) if user_id in self._permitted else None


async def _resolve(repo, accessor):
    return await SharedWorkspaceDBResolver(repo).resolve(
        1,
        accessor,
        source_chacha_db=object(),
        source_media_db=object(),
        conversation_chacha_db=object(),
    )


async def test_a_recipient_can_resolve_the_share():
    repo = _Repo(permitted={9})

    ctx = await _resolve(repo, 9)

    assert ctx.owner_user_id == 7
    assert ctx.accessor_user_id == 9


async def test_a_stranger_cannot_resolve_the_share():
    """The regression: this returned a full context to any caller."""
    repo = _Repo(permitted={9})

    with pytest.raises(PermissionError):
        await _resolve(repo, 999)


async def test_the_primary_key_lookup_is_no_longer_the_gate():
    """get_share must not be what decides access."""
    repo = _Repo(permitted={9})

    with pytest.raises(PermissionError):
        await _resolve(repo, 999)

    assert repo.pk_lookups == 0, "authorization must not go through get_share"
