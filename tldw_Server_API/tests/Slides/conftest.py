"""Explicit policy inputs for mocked standalone-provider boundaries."""

import pytest


@pytest.fixture
def standalone_provider_egress(monkeypatch: pytest.MonkeyPatch) -> None:
    """Admit only the official hosts whose HTTP traffic these fixtures mock."""
    monkeypatch.setenv("WORKFLOWS_EGRESS_ALLOWLIST", "api.openai.com,api.anthropic.com")
