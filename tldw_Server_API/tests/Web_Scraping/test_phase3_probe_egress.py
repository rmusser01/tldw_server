from __future__ import annotations

import ast
import asyncio
import importlib
import importlib.util
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

import tldw_Server_API.app.core.Web_Scraping.policy.adapters as policy_adapters
import tldw_Server_API.app.core.Web_Scraping.runtime as runtime
import tldw_Server_API.app.core.Web_Scraping.runtime.policy as runtime_policy
from tldw_Server_API.app.core.Security import egress
from tldw_Server_API.app.core.Security.egress import URLPolicyResult
from tldw_Server_API.app.core.Web_Scraping.runtime import PolicyDecision, RuntimeRequestContext

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_POLICY_PATH = _ROOT / "app/core/Web_Scraping/runtime/policy.py"
_POLICY_ADAPTERS_PATH = _ROOT / "app/core/Web_Scraping/policy/adapters.py"
_FACADE_MODULE = "tldw_Server_API.app.core.Web_Scraping.preflight.facade"


def _required_attribute(module: Any, name: str) -> Any:
    assert hasattr(module, name), f"Task 3 contract {name} is missing"
    return getattr(module, name)


def _guard_type() -> type[Any]:
    return _required_attribute(policy_adapters, "DefaultProbeEgressGuard")


def _decision_type() -> type[Any]:
    return _required_attribute(runtime_policy, "ProbeEgressDecision")


class _BoundLogger:
    def __init__(self) -> None:
        self.events: list[tuple[dict[str, object], str]] = []
        self._bound: dict[str, object] = {}

    def bind(self, **values: object) -> _BoundLogger:
        bound = _BoundLogger()
        bound.events = self.events
        bound._bound = dict(values)
        return bound

    def warning(self, message: str) -> None:
        self.events.append((self._bound, message))


class _FakePolicyChecker:
    def __init__(self, decision: PolicyDecision) -> None:
        self.decision = decision
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def decide(self, url: str, **kwargs: object) -> PolicyDecision:
        self.calls.append((url, dict(kwargs)))
        return self.decision


def _policy_decision(*, allowed: bool = True) -> PolicyDecision:
    return PolicyDecision(
        allowed=allowed,
        mode="strict",
        reason="allowed" if allowed else "policy_denied",
        stage="pre_fetch",
        source="article_extract",
    )


def test_probe_egress_runtime_contract_is_public_immutable_and_narrow() -> None:
    decision_type = _decision_type()
    guard_protocol = _required_attribute(runtime_policy, "ProbeEgressGuard")

    assert runtime.ProbeEgressDecision is decision_type
    assert runtime.ProbeEgressGuard is guard_protocol
    assert [field.name for field in fields(decision_type)] == [
        "allowed",
        "reason",
        "resolved_ips",
    ]
    decision = decision_type(allowed=True, reason="allowed")
    with pytest.raises(FrozenInstanceError):
        decision.reason = "changed"


@pytest.mark.asyncio
async def test_probe_guard_delegates_once_and_allowed_decision_uses_stable_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        egress,
        "evaluate_url_policy",
        lambda url: calls.append(url)
        or URLPolicyResult(
            True,
            "raw reason must not escape",
            ("93.184.216.34",),
            "unknown_allowed_code",
        ),
    )

    decision = await _guard_type()().decide(
        "https://example.com/private?token=secret",
        context=RuntimeRequestContext(source="preflight", stage="preflight_subrequest"),
    )

    assert calls == ["https://example.com/private?token=secret"]
    assert decision == _decision_type()(
        allowed=True,
        reason="allowed",
        resolved_ips=("93.184.216.34",),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason_code",
    [
        "invalid_url",
        "unsupported_scheme",
        "userinfo_not_allowed",
        "port_not_allowed",
        "host_denied",
        "origin_mismatch",
        "dns_unresolved",
        "address_forbidden",
        "dns_changed",
    ],
)
async def test_probe_guard_allows_only_known_central_reason_codes(
    monkeypatch: pytest.MonkeyPatch,
    reason_code: str,
) -> None:
    monkeypatch.setattr(
        egress,
        "evaluate_url_policy",
        lambda _url: URLPolicyResult(False, "mutable human text", (), reason_code),
    )

    decision = await _guard_type()().decide(
        "https://example.com",
        context=RuntimeRequestContext(),
    )

    assert decision.reason == reason_code


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("legacy_reason", "expected"),
    [
        ("Invalid URL", "invalid_url"),
        ("Unsupported URL scheme", "unsupported_scheme"),
        ("URL userinfo is not allowed", "userinfo_not_allowed"),
        ("URL must include a hostname", "invalid_url"),
        ("Invalid URL port", "invalid_url"),
        ("Host in denylist", "host_denied"),
        ("URL origin does not match configured endpoint", "origin_mismatch"),
        ("No allowlist configured (strict)", "host_denied"),
        ("Host not in allowlist", "host_denied"),
        ("Host could not be resolved", "dns_unresolved"),
        ("URL resolves to a forbidden address", "address_forbidden"),
        ("URL resolves to a private or reserved address", "address_forbidden"),
        ("DNS resolution changed since policy check", "dns_changed"),
    ],
)
async def test_probe_guard_maps_only_known_legacy_reason_strings_when_code_is_absent(
    monkeypatch: pytest.MonkeyPatch,
    legacy_reason: str,
    expected: str,
) -> None:
    old_style_result = SimpleNamespace(
        allowed=False,
        reason=legacy_reason,
        resolved_ips=(),
    )
    monkeypatch.setattr(egress, "evaluate_url_policy", lambda _url: old_style_result)

    decision = await _guard_type()().decide(
        "https://example.com",
        context=RuntimeRequestContext(),
    )

    assert decision.reason == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        URLPolicyResult(False, "Invalid URL", (), "future_secret_code"),
        SimpleNamespace(
            allowed=False,
            reason="future reason containing secret-token",
            resolved_ips=(),
        ),
    ],
)
async def test_probe_guard_maps_unknown_reasons_to_other(
    monkeypatch: pytest.MonkeyPatch,
    result: object,
) -> None:
    monkeypatch.setattr(egress, "evaluate_url_policy", lambda _url: result)

    decision = await _guard_type()().decide(
        "https://example.com",
        context=RuntimeRequestContext(),
    )

    assert decision.reason == "other"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        SimpleNamespace(
            allowed=False,
            reason="Invalid URL",
            resolved_ips=(),
            reason_code=[],
        ),
        SimpleNamespace(allowed=False, reason=[], resolved_ips=()),
    ],
)
async def test_probe_guard_maps_malformed_reason_values_to_other(
    monkeypatch: pytest.MonkeyPatch,
    result: object,
) -> None:
    monkeypatch.setattr(egress, "evaluate_url_policy", lambda _url: result)

    decision = await _guard_type()().decide(
        "https://example.com",
        context=RuntimeRequestContext(),
    )

    assert decision.reason == "other"


@pytest.mark.asyncio
async def test_probe_guard_denial_log_contains_only_sanitized_context_and_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_logger = _BoundLogger()
    monkeypatch.setattr(policy_adapters, "logger", fake_logger, raising=False)
    monkeypatch.setattr(
        egress,
        "evaluate_url_policy",
        lambda _url: URLPolicyResult(
            False,
            "private/path?token=secret#fragment",
            (),
            "unknown-secret-code",
        ),
    )

    decision = await _guard_type()().decide(
        "https://user:password@example.com/private/path?token=secret#fragment",
        context=RuntimeRequestContext(source="preflight", stage="preflight_subrequest"),
    )

    assert decision.reason == "other"
    assert fake_logger.events == [
        (
            {
                "source": "preflight",
                "stage": "preflight_subrequest",
                "host": "example.com",
            },
            "Probe egress policy denied target",
        )
    ]
    rendered = repr(fake_logger.events)
    for secret in (
        "user",
        "password",
        "private",
        "token",
        "secret",
        "fragment",
        "unknown-secret-code",
    ):
        assert secret not in rendered


@pytest.mark.asyncio
async def test_probe_guard_evaluator_exception_fails_closed_and_sanitizes_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_logger = _BoundLogger()
    monkeypatch.setattr(policy_adapters, "logger", fake_logger, raising=False)

    def raise_policy_error(_url: str) -> URLPolicyResult:
        raise RuntimeError("https://user:password@example.com/private?token=secret")

    monkeypatch.setattr(egress, "evaluate_url_policy", raise_policy_error)

    decision = await _guard_type()().decide(
        "https://user:password@example.com/private?token=secret#fragment",
        context=RuntimeRequestContext(source="preflight", stage="preflight_subrequest"),
    )

    assert decision == _decision_type()(allowed=False, reason="policy_error")
    assert fake_logger.events == [
        (
            {
                "source": "preflight",
                "stage": "preflight_subrequest",
                "host": "example.com",
            },
            "Probe egress policy evaluation failed",
        )
    ]
    rendered = repr(fake_logger.events)
    for secret in ("user", "password", "private", "token", "secret", "fragment", "RuntimeError"):
        assert secret not in rendered


@pytest.mark.asyncio
async def test_probe_guard_propagates_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def cancel_to_thread(*_args: object, **_kwargs: object) -> object:
        raise asyncio.CancelledError

    _guard_type()
    adapter_asyncio = _required_attribute(policy_adapters, "asyncio")
    monkeypatch.setattr(adapter_asyncio, "to_thread", cancel_to_thread)

    with pytest.raises(asyncio.CancelledError):
        await _guard_type()().decide(
            "https://example.com",
            context=RuntimeRequestContext(),
        )


@pytest.mark.asyncio
async def test_probe_guard_copies_resolved_ips_into_immutable_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved_ips = ["93.184.216.34"]
    raw = SimpleNamespace(
        allowed=True,
        reason=None,
        resolved_ips=resolved_ips,
        reason_code=None,
    )
    monkeypatch.setattr(egress, "evaluate_url_policy", lambda _url: raw)

    decision = await _guard_type()().decide(
        "https://example.com",
        context=RuntimeRequestContext(),
    )
    resolved_ips.append("203.0.113.1")

    assert decision.resolved_ips == ("93.184.216.34",)
    with pytest.raises(FrozenInstanceError):
        decision.resolved_ips = ()


def test_reason_code_and_legacy_reason_boundaries_are_immutable() -> None:
    reason_codes = _required_attribute(policy_adapters, "_ALLOWED_REASON_CODES")
    legacy_reasons = _required_attribute(policy_adapters, "_LEGACY_REASON_MAP")

    assert isinstance(reason_codes, frozenset)
    assert isinstance(legacy_reasons, MappingProxyType)
    with pytest.raises(AttributeError):
        reason_codes.add("future_code")
    with pytest.raises(TypeError):
        legacy_reasons["future reason"] = "future_code"


def test_runtime_policy_and_probe_adapter_preserve_import_direction() -> None:
    runtime_tree = ast.parse(_RUNTIME_POLICY_PATH.read_text(encoding="utf-8"))
    adapter_tree = ast.parse(_POLICY_ADAPTERS_PATH.read_text(encoding="utf-8"))

    def imported_modules(tree: ast.AST) -> set[str]:
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.add(node.module or "")
        return modules

    runtime_imports = imported_modules(runtime_tree)
    assert not any(
        forbidden in module.lower()
        for module in runtime_imports
        for forbidden in ("security.egress", "preflight", "policy.adapters", "robots")
    )
    assert not any("robots" in module.lower() for module in imported_modules(adapter_tree))


@pytest.mark.asyncio
async def test_evaluate_target_uses_scrape_policy_once_with_exact_inputs() -> None:
    assert importlib.util.find_spec(_FACADE_MODULE) is not None, "Task 3 facade is missing"
    facade = importlib.import_module(_FACADE_MODULE)
    evaluate_target = _required_attribute(facade, "evaluate_target")
    request_context = RuntimeRequestContext(source="article_extract", stage="pre_fetch")
    config = {"web_scraper": {}}
    checker = _FakePolicyChecker(_policy_decision())

    target = await evaluate_target(
        "https://example.com/article",
        respect_robots=True,
        user_agent="UA",
        request_context=request_context,
        config=config,
        policy_checker=checker,
    )

    assert checker.calls == [
        (
            "https://example.com/article",
            {
                "respect_robots": True,
                "user_agent": "UA",
                "context": request_context,
                "config": config,
            },
        )
    ]
    assert target.url == "https://example.com/article"
    assert target.decision is checker.decision
    assert target.request_context is request_context


def test_evaluate_target_has_no_probe_guard_or_concrete_adapter_dependency() -> None:
    assert importlib.util.find_spec(_FACADE_MODULE) is not None, "Task 3 facade is missing"
    facade_spec = importlib.util.find_spec(_FACADE_MODULE)
    assert facade_spec is not None and facade_spec.origin is not None
    facade_source = Path(facade_spec.origin).read_text(encoding="utf-8")

    assert "ProbeEgressGuard" not in facade_source
    assert "DefaultProbeEgressGuard" not in facade_source
    assert "policy.adapters" not in facade_source


@pytest.mark.asyncio
async def test_evaluate_target_propagates_policy_cancellation() -> None:
    assert importlib.util.find_spec(_FACADE_MODULE) is not None, "Task 3 facade is missing"
    facade = importlib.import_module(_FACADE_MODULE)
    evaluate_target = _required_attribute(facade, "evaluate_target")

    class _CancellingPolicyChecker:
        async def decide(self, *_args: object, **_kwargs: object) -> PolicyDecision:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await evaluate_target(
            "https://example.com/article",
            respect_robots=False,
            user_agent=None,
            request_context=RuntimeRequestContext(),
            config=None,
            policy_checker=_CancellingPolicyChecker(),
        )
