import threading

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.core.exceptions import EgressPolicyError
from tldw_Server_API.app.core.Security import egress
from tldw_Server_API.app.core.Security.url_validation import assert_url_safe


class _CapturedLogger:
    """Capture Loguru bind/opt calls and messages for egress log assertions."""

    def __init__(self) -> None:
        self.current_bound: dict[str, object] = {}
        self.warning_logs: list[tuple[dict[str, object], str]] = []
        self.debug_logs: list[tuple[dict[str, object], str]] = []
        self.opt_kwargs: list[dict[str, object]] = []

    def bind(self, **kwargs: object) -> "_CapturedLogger":
        """Store fields from the latest logger.bind call."""
        self.current_bound = dict(kwargs)
        return self

    def opt(self, **kwargs: object) -> "_CapturedLogger":
        """Store options from logger.opt calls."""
        self.opt_kwargs.append(dict(kwargs))
        return self

    def warning(self, message: str, *_args: object, **_kwargs: object) -> None:
        """Record a warning with the current bound fields."""
        self.warning_logs.append((dict(self.current_bound), message))

    def debug(self, message: str, *_args: object, **_kwargs: object) -> None:
        """Record a debug message with the current bound fields."""
        self.debug_logs.append((dict(self.current_bound), message))


def _always_public(host: str):
    return True, ["203.0.113.10"]


@pytest.mark.unit
def test_platform_webhook_policy_composes_all_global_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def _capture(url: str, **kwargs: object) -> egress.URLPolicyResult:
        observed["url"] = url
        observed.update(kwargs)
        return egress.URLPolicyResult(True)

    monkeypatch.setenv(egress.GLOBAL_ALLOWLIST_ENV, "global.example")
    monkeypatch.setenv(egress.ALLOWLIST_ENV, "workflow.example")
    monkeypatch.setenv(egress.WEBHOOK_ALLOWLIST_ENV, "*.webhook.example")
    monkeypatch.setenv(egress.GLOBAL_DENYLIST_ENV, "blocked-global.example")
    monkeypatch.setenv(egress.DENYLIST_ENV, "blocked-workflow.example")
    monkeypatch.setenv(egress.WEBHOOK_DENYLIST_ENV, "*.blocked-webhook.example")
    monkeypatch.setattr(egress, "evaluate_url_policy", _capture)

    result = egress.evaluate_platform_webhook_url_policy(
        "https://receiver.example/private?token=secret"
    )

    assert result.allowed is True
    assert observed == {
        "url": "https://receiver.example/private?token=secret",
        "allowlist": [
            "global.example",
            "workflow.example",
            "webhook.example",
        ],
        "denylist": [
            "blocked-global.example",
            "blocked-workflow.example",
            "blocked-webhook.example",
        ],
        "block_private_override": True,
        "sensitive_observability": True,
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("url", "resolved_ips"),
    [
        ("https://127.0.0.1/hook", None),
        ("https://receiver.example/hook", ["10.0.0.7"]),
    ],
)
def test_platform_webhook_policy_blocks_private_targets_when_ambient_false(
    monkeypatch: pytest.MonkeyPatch,
    url: str,
    resolved_ips: list[str] | None,
) -> None:
    monkeypatch.setenv(egress.BLOCK_PRIVATE_ENV, "false")
    monkeypatch.setenv(egress.PROFILENAME, "permissive")
    for env_name in (
        egress.GLOBAL_ALLOWLIST_ENV,
        egress.ALLOWLIST_ENV,
        egress.WEBHOOK_ALLOWLIST_ENV,
        egress.GLOBAL_DENYLIST_ENV,
        egress.DENYLIST_ENV,
        egress.WEBHOOK_DENYLIST_ENV,
    ):
        monkeypatch.delenv(env_name, raising=False)
    if resolved_ips is not None:
        monkeypatch.setattr(
            egress,
            "_resolve_host_ips",
            lambda *_args, **_kwargs: resolved_ips,
        )

    result = egress.evaluate_platform_webhook_url_policy(url)

    assert result.allowed is False
    assert result.reason_code == "address_forbidden"


@pytest.mark.unit
def test_platform_webhook_policy_preserves_deny_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(egress.GLOBAL_ALLOWLIST_ENV, "receiver.example")
    monkeypatch.setenv(egress.WEBHOOK_ALLOWLIST_ENV, "receiver.example")
    monkeypatch.setenv(egress.WEBHOOK_DENYLIST_ENV, "receiver.example")
    monkeypatch.setenv(egress.BLOCK_PRIVATE_ENV, "false")

    result = egress.evaluate_platform_webhook_url_policy(
        "https://receiver.example/hook"
    )

    assert result.allowed is False
    assert result.reason_code == "host_denied"


class TestEgressPolicy:
    @pytest.fixture(autouse=True)
    def _scoped_policy_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Keep scoped-policy cases independent of suite-level egress relaxations."""
        monkeypatch.setenv(egress.BLOCK_PRIVATE_ENV, "true")
        monkeypatch.setenv(egress.ALLOWED_PORTS_ENV, "80,443")
        monkeypatch.setenv(egress.PROFILENAME, "permissive")
        monkeypatch.delenv(egress.ALLOWLIST_ENV, raising=False)
        monkeypatch.delenv(egress.DENYLIST_ENV, raising=False)
        monkeypatch.delenv(egress.GLOBAL_ALLOWLIST_ENV, raising=False)
        monkeypatch.delenv(egress.GLOBAL_DENYLIST_ENV, raising=False)

    @pytest.mark.unit
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

    @pytest.mark.unit
    def test_ipv4_mapped_ipv6_is_blocked(self, monkeypatch):

        monkeypatch.delenv("WORKFLOWS_EGRESS_ALLOWLIST", raising=False)
        monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "true")

        url = "http://[::ffff:127.0.0.1]/"
        assert not egress.is_url_allowed(url)

        with pytest.raises(HTTPException) as exc:
            assert_url_safe(url)
        assert "private" in exc.value.detail.lower()

    @pytest.mark.unit
    def test_invalid_port_is_rejected(self):

        res = egress.evaluate_url_policy("http://example.com:bad/path")
        assert res.allowed is False
        assert "port" in (res.reason or "").lower()

    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.parametrize(
        ("configured_url", "request_url", "resolved_ip"),
        [
            ("http://127.0.0.1:11434", "http://127.0.0.1:11434/api/tags", "127.0.0.1"),
            ("http://llama.lan:11434", "http://llama.lan:11434/v1/models", "192.168.1.20"),
            ("http://docker:11434", "http://docker:11434/v1/models", "127.0.0.11"),
            ("http://overlay:11434", "http://overlay:11434/v1/models", "100.64.0.7"),
            ("http://[fd12:3456::10]:11434", "http://[fd12:3456::10]:11434/v1/models", "fd12:3456::10"),
            ("https://public.example:9443", "https://public.example:9443/v1/models", "8.8.8.8"),
        ],
    )
    @pytest.mark.unit
    def test_configured_scope_allows_approved_addresses_on_its_exact_port(
        self,
        configured_url: str,
        request_url: str,
        resolved_ip: str,
    ) -> None:
        scope = egress.ConfiguredEndpointScope.from_url(configured_url)

        result = egress.evaluate_url_policy(
            request_url,
            configured_endpoint=scope,
            resolved_ips_override=[resolved_ip],
        )

        assert result == egress.URLPolicyResult(True, None, (resolved_ip,), None)

    @pytest.mark.parametrize(
        ("request_url", "reason_code"),
        [
            ("https://llama.lan:11434/v1/models", "origin_mismatch"),
            ("http://other.lan:11434/v1/models", "origin_mismatch"),
            ("http://llama.lan:11435/v1/models", "origin_mismatch"),
            ("http://user:pass@llama.lan:11434/v1/models", "userinfo_not_allowed"),
        ],
    )
    @pytest.mark.unit
    def test_configured_scope_rejects_origin_and_userinfo_mismatches(
        self,
        request_url: str,
        reason_code: str,
    ) -> None:
        scope = egress.ConfiguredEndpointScope.from_url("http://llama.lan:11434")

        result = egress.evaluate_url_policy(
            request_url,
            configured_endpoint=scope,
            resolved_ips_override=["192.168.1.20"],
        )

        assert result.allowed is False
        assert result.reason_code == reason_code

    @pytest.mark.parametrize(
        "resolved_ip",
        [
            "169.254.1.1",  # link-local
            "224.0.0.1",  # multicast
            "0.0.0.0",  # unspecified/current network
            "192.0.2.1",  # documentation
            "198.18.0.1",  # benchmarking
            "192.88.99.1",  # deprecated 6to4 relay anycast
            "64:ff9b::808:808",  # IPv4/IPv6 translation
            "2001:3::1",  # AMT special-use range reported as global
            "3fff::1",  # IPv6 documentation
            "fec0::1",  # deprecated IPv6 site-local
            "240.0.0.1",  # reserved
            "::ffff:192.168.1.20",  # IPv4-mapped IPv6
        ],
    )
    @pytest.mark.unit
    def test_configured_scope_rejects_nonordinary_special_use_addresses(
        self,
        resolved_ip: str,
    ) -> None:
        scope = egress.ConfiguredEndpointScope.from_url("http://llama.lan:11434")

        result = egress.evaluate_url_policy(
            "http://llama.lan:11434/v1/models",
            configured_endpoint=scope,
            resolved_ips_override=[resolved_ip],
        )

        assert result.allowed is False
        assert result.reason_code == "address_forbidden"

    @pytest.mark.parametrize(
        "metadata_ip",
        [
            "169.254.169.254",
            "169.254.170.2",
            "169.254.170.23",
            "100.100.100.200",
            "168.63.129.16",
            "fd00:ec2::254",
        ],
    )
    @pytest.mark.unit
    def test_configured_scope_rejects_metadata_endpoints(self, metadata_ip: str) -> None:
        scope = egress.ConfiguredEndpointScope.from_url("http://llama.lan:11434")

        result = egress.evaluate_url_policy(
            "http://llama.lan:11434/v1/models",
            configured_endpoint=scope,
            resolved_ips_override=[metadata_ip],
        )

        assert result.allowed is False
        assert result.reason_code == "address_forbidden"

    @pytest.mark.unit
    def test_configured_scope_rejects_mixed_dns_answers(self) -> None:
        scope = egress.ConfiguredEndpointScope.from_url("http://llama.lan:11434")

        result = egress.evaluate_url_policy(
            "http://llama.lan:11434/v1/models",
            configured_endpoint=scope,
            resolved_ips_override=["192.168.1.20", "169.254.169.254"],
        )

        assert result.allowed is False
        assert result.resolved_ips == ("192.168.1.20", "169.254.169.254")
        assert result.reason_code == "address_forbidden"

    @pytest.mark.unit
    def test_configured_scope_global_denylist_retains_precedence(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(egress.GLOBAL_DENYLIST_ENV, "llama.lan")
        scope = egress.ConfiguredEndpointScope.from_url("http://llama.lan:11434")

        result = egress.evaluate_url_policy(
            "http://llama.lan:11434/v1/models",
            configured_endpoint=scope,
            resolved_ips_override=["192.168.1.20"],
        )

        assert result.allowed is False
        assert result.reason_code == "host_denied"

    @pytest.mark.parametrize(
        ("request_host", "denylisted_host"),
        [
            ("fd12:3456:0:0:0:0:0:10", "fd12:3456::10"),
            ("fd12:3456::10", "fd12:3456:0:0:0:0:0:10"),
        ],
    )
    @pytest.mark.unit
    def test_configured_scope_denylist_compares_canonical_ip_literals(
        self,
        monkeypatch: pytest.MonkeyPatch,
        request_host: str,
        denylisted_host: str,
    ) -> None:
        monkeypatch.setenv(egress.GLOBAL_DENYLIST_ENV, denylisted_host)
        request_url = f"http://[{request_host}]:11434/v1/models"
        scope = egress.ConfiguredEndpointScope.from_url(request_url)

        result = egress.evaluate_url_policy(
            request_url,
            configured_endpoint=scope,
            resolved_ips_override=[request_host],
        )

        assert result.allowed is False
        assert result.reason_code == "host_denied"

    @pytest.mark.unit
    def test_configured_scope_satisfies_strict_profile_without_global_allowlist(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(egress.PROFILENAME, "strict")
        scope = egress.ConfiguredEndpointScope.from_url("http://llama.lan:11434")

        result = egress.evaluate_url_policy(
            "http://llama.lan:11434/v1/models",
            configured_endpoint=scope,
            resolved_ips_override=["192.168.1.20"],
        )

        assert result.allowed is True

    @pytest.mark.unit
    def test_configured_scope_always_resolves_despite_private_block_relaxations(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(egress.BLOCK_PRIVATE_ENV, "false")
        monkeypatch.setenv("TESTING", "true")
        monkeypatch.setattr(egress, "_resolve_host_ips", lambda _host: ["169.254.169.254"])
        scope = egress.ConfiguredEndpointScope.from_url("http://llama.lan:11434")

        result = egress.evaluate_url_policy(
            "http://llama.lan:11434/v1/models",
            configured_endpoint=scope,
            block_private_override=False,
        )

        assert result.allowed is False
        assert result.reason_code == "address_forbidden"

    @pytest.mark.unit
    def test_configured_scope_rejects_unresolved_and_changed_dns(self) -> None:
        scope = egress.ConfiguredEndpointScope.from_url("http://llama.lan:11434")

        unresolved = egress.evaluate_url_policy(
            "http://llama.lan:11434/v1/models",
            configured_endpoint=scope,
            resolved_ips_override=[],
        )
        changed = egress.evaluate_url_policy(
            "http://llama.lan:11434/v1/models",
            configured_endpoint=scope,
            resolved_ips_override=["192.168.1.21"],
            pinned_resolved_ips=["192.168.1.20"],
        )

        assert unresolved.reason_code == "dns_unresolved"
        assert changed.reason_code == "dns_changed"

    @pytest.mark.parametrize(
        ("configured_url", "equivalent_url"),
        [
            ("https://b\u00fccher.example", "https://xn--bcher-kva.example.:443/v1/models"),
            ("http://llama.lan", "http://llama.lan.:80/v1/models"),
            (
                "http://[fd12:3456:0:0:0:0:0:10]",
                "http://[fd12:3456::10]:80/v1/models",
            ),
        ],
    )
    @pytest.mark.unit
    def test_configured_scope_matches_canonical_equivalent_origins(
        self,
        configured_url: str,
        equivalent_url: str,
    ) -> None:
        scope = egress.ConfiguredEndpointScope.from_url(configured_url)

        assert scope.matches(equivalent_url) is True

    @pytest.mark.unit
    def test_url_policy_result_third_positional_argument_remains_resolved_ips(self) -> None:
        result = egress.URLPolicyResult(True, None, ("192.168.1.20",))

        assert result.resolved_ips == ("192.168.1.20",)
        assert result.reason_code is None

    @pytest.mark.parametrize(
        ("url", "reason_code"),
        [
            ("http://example.com:bad", "invalid_url"),
            ("file:///etc/passwd", "unsupported_scheme"),
            ("https://example.com:9443", "port_not_allowed"),
        ],
    )
    @pytest.mark.unit
    def test_unscoped_policy_failures_expose_stable_reason_codes(
        self,
        url: str,
        reason_code: str,
    ) -> None:
        result = egress.evaluate_url_policy(url, resolved_ips_override=["8.8.8.8"])

        assert result.allowed is False
        assert result.reason_code == reason_code

    @pytest.mark.unit
    def test_unscoped_permissive_url_with_userinfo_keeps_legacy_behavior(self) -> None:
        result = egress.evaluate_url_policy(
            "https://legacy-user:legacy-pass@example.com/resource",
            block_private_override=False,
        )

        assert result.allowed is True

    @pytest.mark.unit
    def test_unscoped_disallowed_port_precedes_denylist_like_legacy_policy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(egress.GLOBAL_DENYLIST_ENV, "blocked.example.com")

        result = egress.evaluate_url_policy(
            "https://blocked.example.com:9443/resource",
            block_private_override=False,
        )

        assert result.allowed is False
        assert result.reason_code == "port_not_allowed"

    @pytest.mark.unit
    def test_egress_policy_error_reason_code_is_optional(self) -> None:
        legacy = EgressPolicyError("message")
        coded = EgressPolicyError("message", reason_code="dns_unresolved")

        assert str(legacy) == "message"
        assert legacy.reason_code is None
        assert str(coded) == "message"
        assert coded.reason_code == "dns_unresolved"

    @pytest.mark.unit
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

    @pytest.mark.unit
    def test_public_resolver_forwards_timeout_and_deduplicates(self, monkeypatch):
        calls: list[tuple[str, float]] = []

        def _resolve(host: str, timeout_s: float = 2.0) -> list[tuple]:
            calls.append((host, timeout_s))
            return [
                (egress.socket.AF_INET, egress.socket.SOCK_STREAM, 0, "", ("8.8.8.8", 443)),
                (egress.socket.AF_INET, egress.socket.SOCK_STREAM, 0, "", ("8.8.8.8", 443)),
                (
                    egress.socket.AF_INET6,
                    egress.socket.SOCK_STREAM,
                    0,
                    "",
                    ("2001:4860:4860::8888", 443, 0, 0),
                ),
            ]

        monkeypatch.setattr(egress, "_getaddrinfo_with_timeout", _resolve)

        assert egress.resolve_host_ips("dns.example", timeout_s=0.25) == (
            "8.8.8.8",
            "2001:4860:4860::8888",
        )
        assert calls == [("dns.example", 0.25)]

    @pytest.mark.parametrize(
        "malformed",
        (
            (egress.socket.AF_INET, egress.socket.SOCK_STREAM, 0, "", ()),
            (egress.socket.AF_INET, egress.socket.SOCK_STREAM, 0, "", (123, 443)),
        ),
    )
    @pytest.mark.unit
    def test_public_resolver_rejects_mixed_valid_and_malformed_answers(
        self,
        monkeypatch: pytest.MonkeyPatch,
        malformed: tuple[object, ...],
    ) -> None:
        monkeypatch.setattr(
            egress,
            "_getaddrinfo_with_timeout",
            lambda *_args, **_kwargs: [
                (egress.socket.AF_INET, egress.socket.SOCK_STREAM, 0, "", ("8.8.8.8", 443)),
                malformed,
            ],
        )

        assert egress.resolve_host_ips("dns.example") == ()

    @pytest.mark.unit
    def test_dns_timeout_limits_outstanding_resolver_threads(
        self: "TestEgressPolicy",
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """DNS resolution fails closed when all resolver worker slots are occupied."""
        release = threading.Event()
        started = 0
        started_lock = threading.Lock()
        resolver_slots = threading.BoundedSemaphore(2)
        captured_logger = _CapturedLogger()

        def _blocked_getaddrinfo(*_args: object, **_kwargs: object) -> list[object]:
            nonlocal started
            with started_lock:
                started += 1
            release.wait(timeout=5)
            return []

        monkeypatch.setattr(egress, "_DNS_RESOLVER_SLOTS", resolver_slots, raising=False)
        monkeypatch.setattr(egress.socket, "getaddrinfo", _blocked_getaddrinfo)
        monkeypatch.setattr(egress, "logger", captured_logger)

        try:
            results = [
                egress._getaddrinfo_with_timeout(
                    f"example-{idx}.invalid",
                    timeout_s=0.01,
                )
                for idx in range(5)
            ]
            assert results == [[], [], [], [], []]
            assert started <= 2
            assert any(
                fields.get("event") == "dns_resolver_slots_exhausted" and fields.get("host") == "example-2.invalid"
                for fields, _message in captured_logger.warning_logs
            )
        finally:
            release.set()

    @pytest.mark.unit
    def test_dns_timeout_logs_resolver_errors(
        self: "TestEgressPolicy",
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Resolver exceptions are logged with structured host and exception fields."""
        captured_logger = _CapturedLogger()

        def _failing_getaddrinfo(*_args: object, **_kwargs: object) -> list[object]:
            raise OSError("resolver failed")

        monkeypatch.setattr(egress.socket, "getaddrinfo", _failing_getaddrinfo)
        monkeypatch.setattr(egress, "logger", captured_logger)

        assert egress._getaddrinfo_with_timeout("example.invalid", timeout_s=0.5) == []
        assert any(
            fields.get("event") == "dns_resolver_error"
            and fields.get("host") == "example.invalid"
            and fields.get("exception_type") == "OSError"
            for fields, _message in captured_logger.debug_logs
        )

    @pytest.mark.unit
    def test_sensitive_dns_failure_redacts_structured_host(
        self: "TestEgressPolicy",
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Sensitive endpoint DNS failures retain taxonomy without the real host."""
        captured_logger = _CapturedLogger()
        private_host = "credential-derived.private.example"

        def _failing_getaddrinfo(*_args: object, **_kwargs: object) -> list[object]:
            raise OSError("resolver failed")

        monkeypatch.setattr(egress.socket, "getaddrinfo", _failing_getaddrinfo)
        monkeypatch.setattr(egress, "logger", captured_logger)

        assert egress._getaddrinfo_with_timeout(
            private_host,
            timeout_s=0.5,
            sensitive_observability=True,
        ) == []
        assert any(
            fields.get("event") == "dns_resolver_error"
            and fields.get("host") == "sensitive_endpoint"
            and fields.get("exception_type") == "OSError"
            for fields, _message in captured_logger.debug_logs
        )
        assert private_host not in repr(captured_logger.debug_logs)

    @pytest.mark.unit
    def test_sensitive_dns_worker_contains_unexpected_exception(
        self: "TestEgressPolicy",
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unexpected resolver failures never reach the process-wide thread hook."""
        captured_logger = _CapturedLogger()
        private_host = "credential-derived.private.example"
        hook_details: list[str] = []

        def _failing_getaddrinfo(*_args: object, **_kwargs: object) -> list[object]:
            raise RuntimeError(f"resolver exploded for {private_host}")

        def _capture_thread_exception(args: threading.ExceptHookArgs) -> None:
            hook_details.append(f"{type(args.exc_value).__name__}: {args.exc_value}")

        monkeypatch.setattr(egress.socket, "getaddrinfo", _failing_getaddrinfo)
        monkeypatch.setattr(egress.threading, "excepthook", _capture_thread_exception)
        monkeypatch.setattr(egress, "logger", captured_logger)

        assert egress._getaddrinfo_with_timeout(
            private_host,
            timeout_s=0.5,
            sensitive_observability=True,
        ) == []
        assert hook_details == []
        assert private_host not in repr(hook_details)
        assert private_host not in repr(captured_logger.debug_logs)
        assert any(
            fields.get("event") == "dns_resolver_error"
            and fields.get("host") == "sensitive_endpoint"
            and fields.get("exception_type") == "RuntimeError"
            for fields, _message in captured_logger.debug_logs
        )

    @pytest.mark.unit
    def test_dns_slot_wait_rejects_nan_config(
        self: "TestEgressPolicy",
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """NaN DNS slot wait configuration falls back to the safe default."""
        captured_logger = _CapturedLogger()
        monkeypatch.setenv(egress.DNS_RESOLVER_SLOT_WAIT_SECONDS_ENV, "nan")
        monkeypatch.setattr(egress, "logger", captured_logger)

        assert egress._dns_slot_wait_seconds(1.0) == egress._DNS_RESOLVER_SLOT_WAIT_SECONDS_DEFAULT
        assert any(
            fields.get("event") == "invalid_egress_dns_config"
            and fields.get("env_var") == egress.DNS_RESOLVER_SLOT_WAIT_SECONDS_ENV
            and fields.get("reason") == "not_finite_or_negative"
            for fields, _message in captured_logger.warning_logs
        )

    @pytest.mark.unit
    def test_dns_timeout_budget_subtracts_slot_wait(
        self: "TestEgressPolicy",
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The resolver worker join only receives the remaining timeout budget."""
        captured_logger = _CapturedLogger()
        join_timeouts: list[float | None] = []

        class _FakeSlots:
            def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
                return True

            def release(self) -> None:
                return None

        class _FakeThread:
            def __init__(self, target: object, daemon: bool) -> None:
                self._target = target
                self._daemon = daemon

            def start(self) -> None:
                return None

            def join(self, timeout: float | None = None) -> None:
                join_timeouts.append(timeout)

            def is_alive(self) -> bool:
                return True

        class _FakeTime:
            @staticmethod
            def monotonic() -> float:
                return next(times)

        times = iter([100.0, 100.03, 100.04, 100.05])
        monkeypatch.setattr(egress, "_DNS_RESOLVER_SLOTS", _FakeSlots(), raising=False)
        monkeypatch.setattr(egress.threading, "Thread", _FakeThread)
        monkeypatch.setattr(egress, "time", _FakeTime())
        monkeypatch.setattr(egress, "logger", captured_logger)

        assert egress._getaddrinfo_with_timeout("example.invalid", timeout_s=0.1) == []
        assert len(join_timeouts) == 1
        assert join_timeouts[0] == pytest.approx(0.06)
        assert any(
            fields.get("event") == "dns_resolver_timeout" and fields.get("host") == "example.invalid"
            for fields, _message in captured_logger.warning_logs
        )
