"""SQLite storage invariants for provider credentials (TASK-13950.17 PR A1, characterization)."""

import pytest

from tldw_Server_API.tests.AuthNZ_SQLite.test_byok_endpoints_sqlite import _setup_byok_sqlite
from tldw_Server_API.tests.helpers.provider_credential_invariants import (
    assert_org_secret_encrypted_and_scope_bound,
    assert_pre_move_envelope_decrypts,
    assert_user_secret_encrypted_and_owner_scoped,
)

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_user_secret_is_encrypted_at_rest_and_owner_scoped_sqlite(tmp_path, monkeypatch) -> None:
    state = await _setup_byok_sqlite(tmp_path, monkeypatch)
    await assert_user_secret_encrypted_and_owner_scoped(
        state["pool"], owner_id=int(state["user"]["id"]), other_id=int(state["admin"]["id"])
    )


@pytest.mark.asyncio
async def test_org_secret_is_encrypted_at_rest_and_scope_bound_sqlite(tmp_path, monkeypatch) -> None:
    state = await _setup_byok_sqlite(tmp_path, monkeypatch)
    org_id = int(state["org"]["id"])
    await assert_org_secret_encrypted_and_scope_bound(state["pool"], org_id=org_id, other_org_id=org_id + 1)


@pytest.mark.asyncio
async def test_a_pre_move_envelope_still_decrypts(tmp_path, monkeypatch) -> None:
    await _setup_byok_sqlite(tmp_path, monkeypatch)
    assert_pre_move_envelope_decrypts()
