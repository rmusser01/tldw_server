import threading

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.core.Security import egress
from tldw_Server_API.app.core.Security.url_validation import assert_url_safe

pytestmark = pytest.mark.unit


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
                fields.get("event") == "dns_resolver_slots_exhausted"
                and fields.get("host") == "example-2.invalid"
                for fields, _message in captured_logger.warning_logs
            )
        finally:
            release.set()

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

        times = iter([100.0, 100.03, 100.04, 100.05])
        monkeypatch.setattr(egress, "_DNS_RESOLVER_SLOTS", _FakeSlots(), raising=False)
        monkeypatch.setattr(egress.threading, "Thread", _FakeThread)
        monkeypatch.setattr(egress.time, "monotonic", lambda: next(times))
        monkeypatch.setattr(egress, "logger", captured_logger)

        assert egress._getaddrinfo_with_timeout("example.invalid", timeout_s=0.1) == []
        assert len(join_timeouts) == 1
        assert join_timeouts[0] == pytest.approx(0.06)
        assert any(
            fields.get("event") == "dns_resolver_timeout"
            and fields.get("host") == "example.invalid"
            for fields, _message in captured_logger.warning_logs
        )
