from __future__ import annotations

import pytest


async def _make_workspace_provider_installations_repo(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.repos import get_workspace_provider_installations_repo as get_repo
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()
    return await get_repo()


@pytest.mark.asyncio
async def test_upsert_and_list_workspace_installations_round_trip(tmp_path, monkeypatch, fresh_services):
    repo = await _make_workspace_provider_installations_repo(tmp_path, monkeypatch)

    await repo.upsert_installation(
        org_id=1,
        provider="slack",
        external_id="T123",
        display_name="Acme Slack",
        installed_by_user_id=7,
        disabled=False,
    )

    rows = await repo.list_installations(org_id=1, provider="slack")
    if rows[0]["external_id"] != "T123":
        raise AssertionError(f"unexpected external_id: {rows[0]['external_id']}")
    if rows[0]["installed_by_user_id"] != 7:
        raise AssertionError(f"unexpected installed_by_user_id: {rows[0]['installed_by_user_id']}")


@pytest.mark.asyncio
async def test_disable_and_delete_workspace_installation(tmp_path, monkeypatch, fresh_services):
    repo = await _make_workspace_provider_installations_repo(tmp_path, monkeypatch)

    await repo.upsert_installation(
        org_id=1,
        provider="slack",
        external_id="T999",
        display_name="Acme Slack",
        installed_by_user_id=7,
        disabled=False,
    )

    disabled = await repo.set_disabled(
        org_id=1,
        provider="slack",
        external_id="T999",
        disabled=True,
    )
    if not disabled:
        raise AssertionError("expected set_disabled() to report a change")

    rows = await repo.list_installations(org_id=1, provider="slack")
    if not rows[0]["disabled"]:
        raise AssertionError("expected disabled installation to remain visible")

    deleted = await repo.delete_installation(org_id=1, provider="slack", external_id="T999")
    if not deleted:
        raise AssertionError("expected delete_installation() to report a change")

    remaining = await repo.list_installations(org_id=1, provider="slack")
    if remaining:
        raise AssertionError(f"expected no remaining rows, found {remaining}")
