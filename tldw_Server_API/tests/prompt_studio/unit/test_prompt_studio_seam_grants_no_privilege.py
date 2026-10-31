"""The Prompt Studio test hook must inject identity, never privilege.

get_prompt_studio_user has a branch, reachable outside TEST_MODE, that builds a
user context from a patchable stub. It set is_admin=True and permissions
["all"]. require_project_access returns True for any admin before it compares
project ownership, so anything making that stub return a dict became a full
cross-account bypass.

The stub returns None in production today, so this was latent rather than live.
It is a seam on the production path, and the fix is to stop it conferring
privilege rather than to rely on the stub staying inert.
"""

import pytest

from tldw_Server_API.app.api.v1.API_Deps import prompt_studio_deps as deps

pytestmark = pytest.mark.unit


class _State:
    rg_policy_id = None
    user_context = None


class _Request:
    state = _State()
    url = type("U", (), {"path": "/api/v1/prompt-studio/projects"})()
    headers: dict = {}


async def test_the_patchable_hook_confers_identity_but_not_admin(monkeypatch):
    """The regression: this branch handed out is_admin=True to any caller."""
    monkeypatch.setenv("TEST_MODE", "false")
    monkeypatch.setenv("TLDW_TEST_MODE", "false")
    monkeypatch.setattr(deps, "get_current_active_user", lambda: {"id": "123"})

    ctx = await deps.get_prompt_studio_user(_Request())

    assert ctx["user_id"] == "123"
    assert ctx["is_admin"] is False
    assert ctx["permissions"] == []


async def test_production_stub_is_still_inert(monkeypatch):
    """The default hook returns None, so this branch does not fire at all."""
    assert deps.get_current_active_user() is None
