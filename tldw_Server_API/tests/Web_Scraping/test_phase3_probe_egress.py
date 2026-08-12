from __future__ import annotations

import ast
import asyncio
import importlib
import importlib.util
import json
import subprocess
import sys
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
_POLICY_PROBE_PATH = _ROOT / "app/core/Web_Scraping/policy/probe.py"
_POLICY_MODULE = "tldw_Server_API.app.core.Web_Scraping.policy"
_PROBE_MODULE = f"{_POLICY_MODULE}.probe"
_FACADE_MODULE = "tldw_Server_API.app.core.Web_Scraping.preflight.facade"


def _required_attribute(module: Any, name: str) -> Any:
    assert hasattr(module, name), f"Task 3 contract {name} is missing"
    return getattr(module, name)


def _guard_type() -> type[Any]:
    policy_package = importlib.import_module(_POLICY_MODULE)
    return _required_attribute(policy_package, "DefaultProbeEgressGuard")


def _probe_implementation_module() -> Any:
    if importlib.util.find_spec(_PROBE_MODULE) is None:
        return policy_adapters
    return importlib.import_module(_PROBE_MODULE)


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


def test_probe_egress_decision_copies_directly_supplied_resolved_ips() -> None:
    resolved_ips = ["93.184.216.34"]

    decision = _decision_type()(
        allowed=True,
        reason="allowed",
        resolved_ips=resolved_ips,
    )
    resolved_ips.append("203.0.113.1")

    assert decision.resolved_ips == ("93.184.216.34",)
    assert isinstance(decision.resolved_ips, tuple)


def test_public_probe_import_does_not_load_scrape_or_http_stacks() -> None:
    script = f"""
import json
import sys

from {_POLICY_MODULE} import DefaultProbeEgressGuard

forbidden_exact = {{
    'tldw_Server_API.app.core.Web_Scraping.outbound_policy',
    'tldw_Server_API.app.core.Web_Scraping.filters',
    'tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib',
    'tldw_Server_API.app.core.http_client',
}}
forbidden_prefixes = (
    'tldw_Server_API.app.core.Web_Scraping.Web_Scraping_Lib',
    'tldw_Server_API.app.core.Metrics',
)
loaded = sorted(
    name
    for name in sys.modules
    if name in forbidden_exact or name.startswith(forbidden_prefixes)
)
print(json.dumps(loaded))
raise SystemExit(1 if loaded else 0)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_ROOT.parent,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.stdout.strip(), completed.stderr
    loaded = json.loads(completed.stdout.strip().splitlines()[-1])
    assert completed.returncode == 0, loaded
    assert loaded == []


def test_lazy_public_names_are_discoverable_without_loading_forbidden_stacks() -> None:
    script = f"""
import json
import sys

import {_POLICY_MODULE} as policy_package
import tldw_Server_API.app.core.Web_Scraping.runtime as runtime_package

policy_first = policy_package.__dir__()
policy_second = policy_package.__dir__()
policy_builtin = dir(policy_package)
runtime_first = runtime_package.__dir__()
runtime_second = runtime_package.__dir__()
runtime_builtin = dir(runtime_package)
forbidden_exact = {{
    'tldw_Server_API.app.core.Web_Scraping.outbound_policy',
    'tldw_Server_API.app.core.Web_Scraping.filters',
    'tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib',
    'tldw_Server_API.app.core.http_client',
}}
forbidden_prefixes = (
    'tldw_Server_API.app.core.Web_Scraping.Web_Scraping_Lib',
    'tldw_Server_API.app.core.Metrics',
)
result = {{
    'policy_defines_dir': '__dir__' in policy_package.__dict__,
    'policy_builtin_matches': policy_builtin == policy_first,
    'policy_exact': policy_first == sorted(
        set(policy_package.__dict__) | set(policy_package.__all__)
    ),
    'policy_fresh': policy_first is not policy_second,
    'policy_missing': sorted(set(policy_package.__all__) - set(policy_first)),
    'runtime_defines_dir': '__dir__' in runtime_package.__dict__,
    'runtime_builtin_matches': runtime_builtin == runtime_first,
    'runtime_exact': runtime_first == sorted(
        set(runtime_package.__dict__) | set(runtime_package.__all__)
    ),
    'runtime_fresh': runtime_first is not runtime_second,
    'runtime_missing': sorted(set(runtime_package.__all__) - set(runtime_first)),
    'forbidden': sorted(
        name
        for name in sys.modules
        if name in forbidden_exact or name.startswith(forbidden_prefixes)
    ),
}}
print(json.dumps(result))
raise SystemExit(0 if all((
    result['policy_defines_dir'],
    result['policy_builtin_matches'],
    result['policy_exact'],
    result['policy_fresh'],
    not result['policy_missing'],
    result['runtime_defines_dir'],
    result['runtime_builtin_matches'],
    result['runtime_exact'],
    result['runtime_fresh'],
    not result['runtime_missing'],
    not result['forbidden'],
)) else 1)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_ROOT.parent,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.stdout.strip(), completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert completed.returncode == 0, result
    assert result == {
        "policy_defines_dir": True,
        "policy_builtin_matches": True,
        "policy_exact": True,
        "policy_fresh": True,
        "policy_missing": [],
        "runtime_defines_dir": True,
        "runtime_builtin_matches": True,
        "runtime_exact": True,
        "runtime_fresh": True,
        "runtime_missing": [],
        "forbidden": [],
    }


def test_probe_exports_preserve_direct_and_lazy_scrape_checker_compatibility() -> None:
    assert _POLICY_PROBE_PATH.exists(), "narrow Task 3 probe module is missing"
    probe_module = importlib.import_module(_PROBE_MODULE)
    policy_package = importlib.import_module(_POLICY_MODULE)

    assert policy_package.DefaultProbeEgressGuard is probe_module.DefaultProbeEgressGuard
    assert policy_adapters.DefaultProbeEgressGuard is probe_module.DefaultProbeEgressGuard
    assert policy_package.DefaultWebOutboundPolicyChecker is policy_adapters.DefaultWebOutboundPolicyChecker


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


class _UnhashableString(str):
    __hash__ = None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        SimpleNamespace(
            allowed=False,
            reason="Invalid URL",
            resolved_ips=(),
            reason_code=_UnhashableString("invalid_url"),
        ),
        SimpleNamespace(
            allowed=False,
            reason=_UnhashableString("Invalid URL"),
            resolved_ips=(),
        ),
    ],
)
async def test_probe_guard_rejects_non_builtin_string_reason_values(
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
    monkeypatch.setattr(
        _probe_implementation_module(),
        "logger",
        fake_logger,
        raising=False,
    )
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
    monkeypatch.setattr(
        _probe_implementation_module(),
        "logger",
        fake_logger,
        raising=False,
    )

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
@pytest.mark.parametrize(
    ("source", "stage", "expected_source", "expected_stage"),
    [
        ("enhanced_scrape", "fetch", "enhanced_scrape", "fetch"),
        ("secretToken123", "tenant987", "web_scraping", "runtime"),
        ("preflight" * 100, "pre_fetch" * 100, "web_scraping", "runtime"),
    ],
)
async def test_probe_guard_logs_only_approved_low_cardinality_context_labels(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    stage: str,
    expected_source: str,
    expected_stage: str,
) -> None:
    probe_module = _probe_implementation_module()
    fake_logger = _BoundLogger()
    monkeypatch.setattr(probe_module, "logger", fake_logger, raising=False)
    monkeypatch.setattr(
        egress,
        "evaluate_url_policy",
        lambda _url: URLPolicyResult(False, "Invalid URL", (), "invalid_url"),
    )

    await _guard_type()().decide(
        "https://example.com/private?token=secret",
        context=RuntimeRequestContext(source=source, stage=stage),
    )

    assert fake_logger.events == [
        (
            {
                "source": expected_source,
                "stage": expected_stage,
                "host": "example.com",
            },
            "Probe egress policy denied target",
        )
    ]
    rendered = repr(fake_logger.events)
    assert "secretToken123" not in rendered
    assert "tenant987" not in rendered


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://user:password@Example.COM/private?token=secret#fragment", "example.com"),
        ("https://b\N{LATIN SMALL LETTER U WITH DIAERESIS}cher.example/path", "xn--bcher-kva.example"),
        ("https://[2001:0db8::1]:443/path", "2001:db8::1"),
        ("https://192.0.2.1:443/path", "192.0.2.1"),
        ("https://example.com./path?token=secret#fragment", "example.com"),
        ("https://example.com\\secret/path?token=secret", "unknown"),
        ("https://[fe80::1%25secret-zone]/path", "unknown"),
        ("https://example.com:secret/path", "unknown"),
        ("https://example.com:99999/path", "unknown"),
        ("https://exa_mple.com/path", "unknown"),
        ("https://-bad.example/path", "unknown"),
        ("https://user:password@/private?token=secret#fragment", "unknown"),
        ("https://secret-token.example.com\\@safe.example/private", "unknown"),
    ],
)
def test_probe_log_host_is_canonical_or_unknown(url: str, expected: str) -> None:
    from tldw_Server_API.app.core.Web_Scraping import observability

    probe_module = _probe_implementation_module()
    assert probe_module.sanitized_host is observability.sanitized_host

    label = probe_module.sanitized_host(url)

    assert label == expected
    if expected == "unknown":
        for secret in ("secret", "password", "private", "token", "zone"):
            assert secret not in label


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com\nsecret/path",
        "https://example.com\rsecret/path",
        "https://example.com\tsecret/path",
        "\x01https://example.com/path",
        "https://example.com/path\x7fsecret",
    ],
)
def test_probe_log_host_rejects_raw_c0_controls_and_del(url: str) -> None:
    from tldw_Server_API.app.core.Web_Scraping.observability import sanitized_host

    assert sanitized_host(url) == "unknown"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://xn--a.example/path", "unknown"),
        ("https://xn--0.example/path", "unknown"),
        ("https://xn--example.example/path", "unknown"),
        ("https://b\N{LATIN SMALL LETTER U WITH DIAERESIS}cher.example/path", "xn--bcher-kva.example"),
        ("https://xn--bcher-kva.example/path", "xn--bcher-kva.example"),
    ],
)
def test_probe_log_host_requires_canonical_idna_alabels(
    url: str,
    expected: str,
) -> None:
    from tldw_Server_API.app.core.Web_Scraping.observability import sanitized_host

    assert sanitized_host(url) == expected


@pytest.mark.asyncio
async def test_probe_guard_propagates_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def cancel_to_thread(*_args: object, **_kwargs: object) -> object:
        raise asyncio.CancelledError

    _guard_type()
    adapter_asyncio = _required_attribute(_probe_implementation_module(), "asyncio")
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
    probe_module = _probe_implementation_module()
    reason_codes = _required_attribute(probe_module, "_ALLOWED_REASON_CODES")
    legacy_reasons = _required_attribute(probe_module, "_LEGACY_REASON_MAP")
    source_labels = _required_attribute(probe_module, "_SOURCE_LABEL_MAP")
    stage_labels = _required_attribute(probe_module, "_STAGE_LABEL_MAP")

    assert isinstance(reason_codes, frozenset)
    assert isinstance(legacy_reasons, MappingProxyType)
    assert isinstance(source_labels, MappingProxyType)
    assert isinstance(stage_labels, MappingProxyType)
    assert dict(source_labels) == {
        label: label
        for label in (
            "article_extract",
            "characterization",
            "enhanced_scrape",
            "preflight",
            "test",
            "web_scraping",
        )
    }
    assert dict(stage_labels) == {
        label: label
        for label in (
            "fetch",
            "pre_fetch",
            "preflight",
            "preflight_subrequest",
            "runtime",
        )
    }
    with pytest.raises(AttributeError):
        reason_codes.add("future_code")
    with pytest.raises(TypeError):
        legacy_reasons["future reason"] = "future_code"
    with pytest.raises(TypeError):
        source_labels["tenant-secret"] = "tenant-secret"
    with pytest.raises(TypeError):
        stage_labels["tenant-stage"] = "tenant-stage"


def test_runtime_policy_and_probe_adapter_preserve_import_direction() -> None:
    assert _POLICY_PROBE_PATH.exists(), "narrow Task 3 probe module is missing"
    runtime_tree = ast.parse(_RUNTIME_POLICY_PATH.read_text(encoding="utf-8"))
    adapter_tree = ast.parse(_POLICY_ADAPTERS_PATH.read_text(encoding="utf-8"))
    probe_tree = ast.parse(_POLICY_PROBE_PATH.read_text(encoding="utf-8"))

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
    probe_imports = imported_modules(probe_tree)
    assert not any(
        forbidden in module.lower()
        for module in probe_imports
        for forbidden in (
            "outbound_policy",
            "filters",
            "robots",
            "http_client",
            "metrics",
            "policy.adapters",
        )
    )


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
    facade_tree = ast.parse(facade_source)
    evaluate_nodes = [
        node for node in facade_tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "evaluate_target"
    ]
    assert len(evaluate_nodes) == 1

    referenced_names = {node.id for node in ast.walk(evaluate_nodes[0]) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(evaluate_nodes[0]) if isinstance(node, ast.Attribute)
    }
    locally_imported_names = {
        imported_name
        for node in ast.walk(evaluate_nodes[0])
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
        for imported_name in (alias.name.rsplit(".", 1)[-1], alias.asname)
        if imported_name is not None
    }
    forbidden_dependencies = {
        "ProbeEgressGuard",
        "DefaultProbeEgressGuard",
        "DefaultWebOutboundPolicyChecker",
        "GuardedHttpProbe",
        "GuardedPlaywrightBrowserProbe",
        "GuardedExternalToolProbe",
    }
    assert forbidden_dependencies.isdisjoint(referenced_names | locally_imported_names)

    top_level_concrete_dependencies = forbidden_dependencies - {"ProbeEgressGuard"}
    top_level_imported_names = {
        imported_name
        for node in facade_tree.body
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
        for imported_name in (alias.name.rsplit(".", 1)[-1], alias.asname)
        if imported_name is not None
    }
    assert top_level_concrete_dependencies.isdisjoint(top_level_imported_names)


def test_facade_import_and_injected_context_do_not_load_concrete_stacks() -> None:
    script = f"""
import json
import sys

from {_FACADE_MODULE} import PreflightAdapterOverrides, build_execution_context, evaluate_target
from tldw_Server_API.app.core.Web_Scraping.preflight.options import PreflightOptions
from tldw_Server_API.app.core.Web_Scraping.preflight.target import PreflightTarget
from tldw_Server_API.app.core.Web_Scraping.runtime import PolicyDecision, RuntimeRequestContext

forbidden_exact = {{
    'tldw_Server_API.app.core.Web_Scraping.outbound_policy',
    'tldw_Server_API.app.core.Security.egress',
    'tldw_Server_API.app.core.http_client',
    'tldw_Server_API.app.core.Web_Scraping.preflight.adapters.http',
    'tldw_Server_API.app.core.Web_Scraping.preflight.adapters.browser',
    'tldw_Server_API.app.core.Web_Scraping.preflight.adapters.external_tools',
}}
forbidden_prefixes = (
    'tldw_Server_API.app.core.Metrics',
    'playwright',
    'curl_cffi',
)

def loaded_forbidden():
    return sorted(
        name
        for name in sys.modules
        if name in forbidden_exact or name.startswith(forbidden_prefixes)
    )

initial = loaded_forbidden()
request_context = RuntimeRequestContext(source='test', stage='preflight')
target = PreflightTarget(
    url='https://example.com',
    decision=PolicyDecision(
        allowed=True,
        mode='test',
        reason='allowed',
        stage='preflight',
        source='test',
    ),
    request_context=request_context,
)
injected = object()
context = build_execution_context(
    target,
    PreflightOptions(),
    policy_checker=injected,
    injected_adapters=PreflightAdapterOverrides(
        http=injected,
        browser=injected,
        external_tools=injected,
        egress_guard=injected,
    ),
)
after_injected = loaded_forbidden()
default_context = build_execution_context(target, PreflightOptions())
default_types = [
    type(default_context.policy_checker).__name__,
    type(default_context.egress_guard).__name__,
    type(default_context.http).__name__,
    type(default_context.browser).__name__,
    type(default_context.external_tools).__name__,
]
required_default_modules = sorted(forbidden_exact)
default_loaded = sorted(name for name in sys.modules if name in forbidden_exact)
result = {{
    'initial': initial,
    'after_injected': after_injected,
    'default_types': default_types,
    'default_modules_present': all(name in default_loaded for name in required_default_modules),
}}
print(json.dumps(result))
raise SystemExit(0 if (
    evaluate_target is not None
    and not initial
    and not after_injected
    and result['default_modules_present']
) else 1)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_ROOT.parent,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.stdout.strip(), completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert completed.returncode == 0, result
    assert result == {
        "initial": [],
        "after_injected": [],
        "default_types": [
            "DefaultWebOutboundPolicyChecker",
            "DefaultProbeEgressGuard",
            "GuardedHttpProbe",
            "GuardedPlaywrightBrowserProbe",
            "GuardedExternalToolProbe",
        ],
        "default_modules_present": True,
    }


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
