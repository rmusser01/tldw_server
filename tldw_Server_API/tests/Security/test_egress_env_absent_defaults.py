"""Env-absent egress policy defaults — the exact #2590/egress-bypass gap (RA6).

The suite's conftest force-sets ``WORKFLOWS_EGRESS_BLOCK_PRIVATE=false`` and
``WORKFLOWS_EGRESS_ALLOWED_PORTS=*``, so no test exercises the *secure*
no-override defaults. The existing private-IP tests explicitly SET
``WORKFLOWS_EGRESS_BLOCK_PRIVATE=true`` rather than removing it, so a hardening
pass that flips the ``os.getenv(..., "true")`` fallback would not be caught.

These tests scrub every egress env var and assert the real-deployment defaults:
private IPs blocked, only ports 80/443 allowed, non-http(s) schemes rejected.
``resolved_ips_override`` avoids real DNS.
"""
from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Security import egress

pytestmark = pytest.mark.unit

# every egress env var the conftest (or a deployment) might set
_EGRESS_ENV_VARS = (
    egress.BLOCK_PRIVATE_ENV,
    egress.ALLOWED_PORTS_ENV,
    egress.ALLOWLIST_ENV,
    egress.DENYLIST_ENV,
    egress.GLOBAL_ALLOWLIST_ENV,
    egress.GLOBAL_DENYLIST_ENV,
    egress.WEBHOOK_ALLOWLIST_ENV,
    egress.WEBHOOK_DENYLIST_ENV,
    egress.PROFILENAME,
    "ENVIRONMENT",
    "APP_ENV",
    "ENV",
)


def _scrub_egress_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _EGRESS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_private_ip_blocked_by_default_with_no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The core #2590 guard: with NO egress env vars set, a URL resolving to a
    private/loopback IP must be denied. ``block_private_override`` is deliberately
    NOT passed — the point is to exercise the ``getenv(..., "true")`` default.
    """
    _scrub_egress_env(monkeypatch)
    result = egress.evaluate_url_policy(
        "https://internal.example.com/steal",
        resolved_ips_override=["127.0.0.1"],
    )
    assert result.allowed is False, "private IP was allowed with no egress env set (#2590 class)"
    assert "private" in (result.reason or "").lower()


def test_link_local_and_metadata_ip_blocked_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cloud-metadata SSRF target (169.254.169.254) must be blocked by default."""
    _scrub_egress_env(monkeypatch)
    result = egress.evaluate_url_policy(
        "http://metadata.example/latest/meta-data",
        resolved_ips_override=["169.254.169.254"],
    )
    assert result.allowed is False


def test_public_ip_allowed_by_default_in_non_prod(monkeypatch: pytest.MonkeyPatch) -> None:
    """A public IP on port 443 should pass with no env (permissive non-prod
    profile) — proving the block is targeted at private ranges, not a blanket deny.
    """
    _scrub_egress_env(monkeypatch)
    result = egress.evaluate_url_policy(
        "https://example.com/ok",
        resolved_ips_override=["93.184.216.34"],
    )
    assert result.allowed is True, f"public URL wrongly denied by default: {result.reason}"


def test_non_standard_port_rejected_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no WORKFLOWS_EGRESS_ALLOWED_PORTS, only 80/443 are allowed."""
    _scrub_egress_env(monkeypatch)
    result = egress.evaluate_url_policy(
        "https://example.com:8080/ok",
        resolved_ips_override=["93.184.216.34"],
    )
    assert result.allowed is False
    assert "port" in (result.reason or "").lower()


def test_non_http_scheme_rejected_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-http(s) schemes (file://, gopher://, ...) are rejected regardless of env."""
    _scrub_egress_env(monkeypatch)
    result = egress.evaluate_url_policy("file:///etc/passwd")
    assert result.allowed is False


def test_block_private_env_default_is_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """Directly pin the secure fallback of the private-IP gate."""
    monkeypatch.delenv(egress.BLOCK_PRIVATE_ENV, raising=False)
    assert egress._should_block_private_env() is True
