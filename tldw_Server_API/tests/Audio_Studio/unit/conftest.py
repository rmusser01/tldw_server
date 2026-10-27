"""Shared fixtures for Audio Studio unit tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def ace_step_dns(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Resolve the test ACE-Step hosts as production DNS would, under the production egress policy.

    The ACE-Step adapter scopes each request to its operator-allowlisted origin, and a scoped
    request resolves the host and pins the answer even under pytest. Tests may add or change
    answers in the returned mapping; an unknown host does not resolve. Private and link-local
    addresses are blocked for unscoped requests.
    """
    from tldw_Server_API.app.core.Security import egress

    answers = {"ace.example.test": "93.184.216.34", "other.example.test": "93.184.216.35"}
    monkeypatch.setattr(
        egress,
        "_resolve_host_ips",
        lambda host, **_kwargs: [answers[host]] if host in answers else [],
    )
    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "true")
    return answers
