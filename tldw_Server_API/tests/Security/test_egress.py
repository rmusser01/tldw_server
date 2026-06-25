import os

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.core.Security import egress
from tldw_Server_API.app.core.Security.url_validation import assert_url_safe


pytestmark = pytest.mark.unit


def _always_public(host: str):
    return True, ["203.0.113.10"]


class TestEgressPolicy:
    def test_allowlist_enforces_exact_and_subdomain_matches(self, monkeypatch):
        monkeypatch.setenv("WORKFLOWS_EGRESS_ALLOWLIST", "example.com")
        monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "false")
        monkeypatch.setattr(egress, "_resolve_and_check_private", _always_public)

        assert not egress.is_url_allowed("https://badexample.com")
        assert egress.is_url_allowed("https://example.com")
        assert egress.is_url_allowed("https://sub.example.com")

        with pytest.raises(HTTPException) as exc:
            assert_url_safe("https://badexample.com/resource")
        assert exc.value.status_code == 400
        assert "allowlist" in exc.value.detail.lower()

    def test_ipv4_mapped_ipv6_is_blocked(self, monkeypatch):

        monkeypatch.delenv("WORKFLOWS_EGRESS_ALLOWLIST", raising=False)
        monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "true")

        url = "http://[::ffff:127.0.0.1]/"
        assert not egress.is_url_allowed(url)

        with pytest.raises(HTTPException) as exc:
            assert_url_safe(url)
        assert "private" in exc.value.detail.lower()

    def test_invalid_port_is_rejected(self):

        res = egress.evaluate_url_policy("http://example.com:bad/path")
        assert res.allowed is False
        assert "port" in (res.reason or "").lower()

    def test_resolved_ips_override_blocks_private_targets(self, monkeypatch):
        monkeypatch.setenv("WORKFLOWS_EGRESS_PROFILE", "permissive")
        monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "true")
        monkeypatch.delenv("WORKFLOWS_EGRESS_ALLOWLIST", raising=False)
        monkeypatch.delenv("WORKFLOWS_EGRESS_DENYLIST", raising=False)
        monkeypatch.delenv("EGRESS_ALLOWLIST", raising=False)
        monkeypatch.delenv("EGRESS_DENYLIST", raising=False)

        res = egress.evaluate_url_policy(
            "https://example.com/path",
            resolved_ips_override=["127.0.0.1"],
        )
        assert res.allowed is False
        assert "private" in (res.reason or "").lower()

    def test_evaluate_url_policy_exposes_resolved_ips(self, monkeypatch):
        monkeypatch.setenv("WORKFLOWS_EGRESS_PROFILE", "permissive")
        monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "true")
        monkeypatch.delenv("WORKFLOWS_EGRESS_ALLOWLIST", raising=False)
        monkeypatch.delenv("WORKFLOWS_EGRESS_DENYLIST", raising=False)
        monkeypatch.delenv("EGRESS_ALLOWLIST", raising=False)
        monkeypatch.delenv("EGRESS_DENYLIST", raising=False)
        monkeypatch.setattr(egress, "_resolve_and_check_private", lambda _host: (True, ["93.184.216.34"]))

        res = egress.evaluate_url_policy("https://example.com/resource")
        assert res.allowed is True
        assert res.resolved_ips == ("93.184.216.34",)

    def test_pinned_resolution_rejects_dns_drift(self, monkeypatch):
        monkeypatch.setenv("WORKFLOWS_EGRESS_PROFILE", "permissive")
        monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "true")
        monkeypatch.delenv("WORKFLOWS_EGRESS_ALLOWLIST", raising=False)
        monkeypatch.delenv("WORKFLOWS_EGRESS_DENYLIST", raising=False)
        monkeypatch.delenv("EGRESS_ALLOWLIST", raising=False)
        monkeypatch.delenv("EGRESS_DENYLIST", raising=False)
        monkeypatch.setattr(egress, "_resolve_and_check_private", lambda _host: (True, ["93.184.216.35"]))

        res = egress.evaluate_url_policy(
            "https://example.com/resource",
            pinned_resolved_ips=["93.184.216.34"],
        )

        assert res.allowed is False
        assert "changed" in (res.reason or "").lower()

    def test_resolve_host_ips_does_not_mutate_global_socket_timeout(self, monkeypatch):
        calls: list[object] = []

        monkeypatch.setattr(egress.socket, "setdefaulttimeout", lambda value: calls.append(value))
        monkeypatch.setattr(
            egress.socket,
            "getaddrinfo",
            lambda *_args, **_kwargs: [
                (egress.socket.AF_INET, egress.socket.SOCK_STREAM, 0, "", ("93.184.216.34", 443)),
            ],
        )

        assert egress._resolve_host_ips("example.com") == ["93.184.216.34"]
        assert calls == []

    def test_getaddrinfo_uses_bounded_executor(self, monkeypatch):
        seen: dict[str, object] = {}
        expected = [
            (egress.socket.AF_INET, egress.socket.SOCK_STREAM, 0, "", ("93.184.216.34", 443)),
        ]

        class _FakeFuture:
            def result(self, timeout):
                seen["timeout"] = timeout
                return expected

        class _FakeExecutor:
            _max_workers = 2

            def submit(self, func, *args, **kwargs):
                seen["submitted"] = (func, args, kwargs)
                return _FakeFuture()

        monkeypatch.setattr(egress, "_DNS_RESOLVER_EXECUTOR", _FakeExecutor())

        assert egress._getaddrinfo_with_timeout("example.com", timeout_s=0.25) == expected
        assert seen["timeout"] == 0.25
        assert seen["submitted"]
