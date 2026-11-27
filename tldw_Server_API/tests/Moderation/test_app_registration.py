"""The app registers the Moderation gateway hooks at startup and removes them at shutdown (TASK-13950.27 PR C)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import tldw_Server_API.app.core.Moderation.gateway_hooks as hooks_mod
from tldw_Server_API.app.core.LLM_Calls import gateway

pytestmark = pytest.mark.integration


def _moderation_pairs() -> int:
    return sum(1 for hook in gateway._POLICY_HOOKS if hooks_mod.pre_call in tuple(hook) or hooks_mod.post_call in tuple(hook))


def test_app_start_stop_start_leaves_one_registration(monkeypatch):
    from tldw_Server_API.app.main import app

    monkeypatch.setenv("PRIVILEGE_METADATA_VALIDATE_ON_STARTUP", "0")
    for _ in range(2):
        with TestClient(app):
            assert _moderation_pairs() == 1
        assert _moderation_pairs() == 0
