import types

import pytest

import tldw_Server_API.app.core.http_client as hc


pytestmark = pytest.mark.unit


def test_validate_egress_treats_tldw_test_mode_y_as_test_context(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TESTING", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "y")

    from tldw_Server_API.app.core.Security import egress as egress_mod

    captured: dict[str, object] = {}

    def _fake_policy(url: str, *, block_private_override=None):
        captured["url"] = url
        captured["block_private_override"] = block_private_override
        return types.SimpleNamespace(allowed=True, reason=None)

    monkeypatch.setattr(egress_mod, "evaluate_url_policy", _fake_policy)

    hc._validate_egress_or_raise("https://example.com/path")
    assert captured["block_private_override"] is False


def test_validate_egress_keeps_ip_policy_for_literal_ip_in_test_context(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TESTING", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "y")

    from tldw_Server_API.app.core.Security import egress as egress_mod

    captured: dict[str, object] = {}

    def _fake_policy(url: str, *, block_private_override=None):
        captured["url"] = url
        captured["block_private_override"] = block_private_override
        return types.SimpleNamespace(allowed=True, reason=None)

    monkeypatch.setattr(egress_mod, "evaluate_url_policy", _fake_policy)

    hc._validate_egress_or_raise("http://127.0.0.1/")
    assert captured["block_private_override"] is None


def test_fetch_simple_redirect_flags_accept_y(monkeypatch):
    class _DummyResp:
        def __init__(self, status_code: int, url: str, headers: dict[str, str] | None = None, text: str = "ok"):
            self.status_code = status_code
            self.url = url
            self.headers = headers or {}
            self.text = text

    calls: list[str] = []

    class _DummyClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def request(self, method, url, headers=None, cookies=None, follow_redirects=None):  # noqa: ARG002
            calls.append(url)
            if url == "https://a.test/start":
                return _DummyResp(302, url, headers={"Location": "https://b.test/next"})
            return _DummyResp(200, url, text="done")

    monkeypatch.setattr(hc, "_is_url_allowed", lambda url: True)
    monkeypatch.setattr(hc, "_resolve_httpx", lambda: types.SimpleNamespace(Client=_DummyClient))
    monkeypatch.setenv("HTTP_ALLOW_REDIRECTS", "y")
    monkeypatch.setenv("HTTP_ALLOW_CROSS_HOST_REDIRECTS", "y")

    resp = hc.fetch("https://a.test/start", backend="httpx")
    assert calls == ["https://a.test/start", "https://b.test/next"]
    assert resp["status"] == 200
    assert resp["url"] == "https://b.test/next"


def test_validate_egress_reuses_dns_pin_cache_for_same_host(monkeypatch):
    from tldw_Server_API.app.core.Security import egress as egress_mod

    calls: list[object] = []

    def _fake_policy(url: str, *, block_private_override=None, resolved_ips_override=None, pinned_resolved_ips=None):
        calls.append(pinned_resolved_ips)
        if pinned_resolved_ips is None:
            return types.SimpleNamespace(
                allowed=True,
                reason=None,
                resolved_ips=("93.184.216.34", "93.184.216.35"),
            )
        return types.SimpleNamespace(
            allowed=True,
            reason=None,
            resolved_ips=tuple(pinned_resolved_ips),
        )

    monkeypatch.setattr(egress_mod, "evaluate_url_policy", _fake_policy)

    dns_pin_cache: dict[str, tuple[str, ...]] = {}
    hc._validate_egress_or_raise("https://example.com/path-a", dns_pin_cache=dns_pin_cache)
    hc._validate_egress_or_raise("https://example.com/path-b", dns_pin_cache=dns_pin_cache)

    assert calls == [None, ("93.184.216.34", "93.184.216.35")]
    assert dns_pin_cache["example.com"] == ("93.184.216.34", "93.184.216.35")
