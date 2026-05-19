import importlib
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


def test_web_outbound_policy_mode_reads_config_when_env_unset(monkeypatch):
    monkeypatch.delenv("WEB_OUTBOUND_POLICY_MODE", raising=False)

    config_mod = importlib.import_module("tldw_Server_API.app.core.config")

    class _ConfigStub:
        def get(self, section, option, fallback=None):
            if section == "Web-Scraper" and option == "web_outbound_policy_mode":
                return "strict"
            return fallback

    monkeypatch.setattr(
        config_mod,
        "load_comprehensive_config",
        lambda: _ConfigStub(),
        raising=False,
    )

    assert config_mod.web_outbound_policy_mode() == "strict"


def test_web_outbound_policy_mode_reads_legacy_web_scraping_section_when_needed(monkeypatch):
    monkeypatch.delenv("WEB_OUTBOUND_POLICY_MODE", raising=False)

    config_mod = importlib.import_module("tldw_Server_API.app.core.config")

    class _ConfigStub:
        def has_section(self, section):
            return section == "Web-Scraping"

        def get(self, section, option, fallback=None):
            if section == "Web-Scraping" and option == "web_outbound_policy_mode":
                return "strict"
            return fallback

    monkeypatch.setattr(
        config_mod,
        "load_comprehensive_config",
        lambda: _ConfigStub(),
        raising=False,
    )

    assert config_mod.web_outbound_policy_mode() == "strict"


def test_web_outbound_policy_sync_denies_provider_request(monkeypatch):
    monkeypatch.delenv("WEB_OUTBOUND_POLICY_MODE", raising=False)
    policy = importlib.import_module(
        "tldw_Server_API.app.core.Web_Scraping.outbound_policy"
    )

    monkeypatch.setattr(
        policy,
        "evaluate_url_policy",
        lambda _url: SimpleNamespace(allowed=False, reason="deny_test"),
        raising=False,
    )

    decision = policy.decide_web_outbound_policy_sync(
        "https://example.com/search",
        respect_robots=False,
        source="websearch_provider",
        stage="provider_request",
    )

    assert decision.allowed is False
    assert decision.mode == "compat"
    assert decision.reason == "deny_test"
    assert decision.stage == "provider_request"
    assert decision.source == "websearch_provider"


def test_web_outbound_policy_sync_emits_decision_metric(monkeypatch):
    monkeypatch.delenv("WEB_OUTBOUND_POLICY_MODE", raising=False)
    policy = importlib.import_module(
        "tldw_Server_API.app.core.Web_Scraping.outbound_policy"
    )

    metric_calls: list[tuple[str, float, dict[str, str]]] = []

    monkeypatch.setattr(
        policy,
        "evaluate_url_policy",
        lambda _url: SimpleNamespace(allowed=False, reason="Port not allowed: 12345"),
        raising=False,
    )
    monkeypatch.setattr(
        policy,
        "increment_counter",
        lambda name, value=1, labels=None: metric_calls.append(
            (name, value, dict(labels or {}))
        ),
        raising=False,
    )

    decision = policy.decide_web_outbound_policy_sync(
        "https://example.com/search",
        respect_robots=False,
        source="websearch_provider",
        stage="provider_request",
    )

    assert decision.allowed is False
    assert metric_calls == [
        (
            "web_outbound_policy_decisions_total",
            1,
            {
                "mode": "compat",
                "source": "websearch_provider",
                "stage": "provider_request",
                "outcome": "blocked",
                "reason": "port_not_allowed",
            },
        )
    ]


@pytest.mark.asyncio
async def test_web_outbound_policy_compat_allows_when_robots_fetch_errors(monkeypatch):
    monkeypatch.setenv("WEB_OUTBOUND_POLICY_MODE", "compat")
    policy = importlib.import_module(
        "tldw_Server_API.app.core.Web_Scraping.outbound_policy"
    )

    monkeypatch.setattr(
        policy,
        "evaluate_url_policy",
        lambda _url: SimpleNamespace(allowed=True, reason="allowed"),
        raising=False,
    )

    async def check_result(self, _url, *, skip_egress_check, fail_open):
        assert skip_egress_check is True
        assert fail_open is True
        return SimpleNamespace(allowed=True, status="unreachable")

    monkeypatch.setattr(policy.RobotsFilter, "check", check_result, raising=False)
    monkeypatch.setattr(
        policy.RobotsFilter,
        "allowed",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("use check() instead of allowed()")),
        raising=False,
    )

    decision = await policy.decide_web_outbound_policy(
        "https://example.com/page",
        respect_robots=True,
        user_agent="UA",
        source="article_extract",
        stage="pre_fetch",
    )

    assert decision.allowed is True
    assert decision.mode == "compat"
    assert decision.reason == "robots_unreachable_allowed"
    assert decision.stage == "pre_fetch"
    assert decision.source == "article_extract"


@pytest.mark.asyncio
async def test_web_outbound_policy_strict_blocks_when_robots_fetch_errors(monkeypatch):
    monkeypatch.setenv("WEB_OUTBOUND_POLICY_MODE", "strict")
    policy = importlib.import_module(
        "tldw_Server_API.app.core.Web_Scraping.outbound_policy"
    )

    monkeypatch.setattr(
        policy,
        "evaluate_url_policy",
        lambda _url: SimpleNamespace(allowed=True, reason="allowed"),
        raising=False,
    )

    async def check_result(self, _url, *, skip_egress_check, fail_open):
        assert skip_egress_check is True
        assert fail_open is False
        return SimpleNamespace(allowed=False, status="unreachable")

    monkeypatch.setattr(policy.RobotsFilter, "check", check_result, raising=False)
    monkeypatch.setattr(
        policy.RobotsFilter,
        "allowed",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("use check() instead of allowed()")),
        raising=False,
    )

    decision = await policy.decide_web_outbound_policy(
        "https://example.com/page",
        respect_robots=True,
        user_agent="UA",
        source="article_extract",
        stage="pre_fetch",
    )

    assert decision.allowed is False
    assert decision.mode == "strict"
    assert decision.reason == "robots_unreachable"
    assert decision.stage == "pre_fetch"
    assert decision.source == "article_extract"


@pytest.mark.asyncio
async def test_web_outbound_policy_strict_blocks_when_robots_endpoint_returns_5xx(monkeypatch):
    monkeypatch.setenv("WEB_OUTBOUND_POLICY_MODE", "strict")
    policy = importlib.import_module(
        "tldw_Server_API.app.core.Web_Scraping.outbound_policy"
    )
    filters = importlib.import_module(
        "tldw_Server_API.app.core.Web_Scraping.filters"
    )
    article_lib = importlib.import_module(
        "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib"
    )

    monkeypatch.setattr(
        policy,
        "evaluate_url_policy",
        lambda _url: SimpleNamespace(allowed=True, reason="allowed"),
        raising=False,
    )
    monkeypatch.setattr(
        filters,
        "http_fetch",
        lambda **_kwargs: {"status": 503, "text": ""},
        raising=False,
    )
    monkeypatch.setattr(
        article_lib,
        "http_fetch",
        lambda **_kwargs: {"status": 503, "text": ""},
        raising=False,
    )

    decision = await policy.decide_web_outbound_policy(
        "https://example.com/page",
        respect_robots=True,
        user_agent="UA",
        source="article_extract",
        stage="pre_fetch",
        robots_filter=filters.RobotsFilter(user_agent="UA"),
    )

    assert decision.allowed is False
    assert decision.mode == "strict"
    assert decision.reason == "robots_unreachable"


@pytest.mark.asyncio
async def test_web_outbound_policy_reuses_existing_egress_decision_for_robots_check(monkeypatch):
    monkeypatch.setenv("WEB_OUTBOUND_POLICY_MODE", "compat")
    policy = importlib.import_module(
        "tldw_Server_API.app.core.Web_Scraping.outbound_policy"
    )
    filters = importlib.import_module(
        "tldw_Server_API.app.core.Web_Scraping.filters"
    )
    egress = importlib.import_module(
        "tldw_Server_API.app.core.Security.egress"
    )

    monkeypatch.setattr(
        policy,
        "evaluate_url_policy",
        lambda _url: SimpleNamespace(allowed=True, reason="allowed"),
        raising=False,
    )
    monkeypatch.setattr(
        egress,
        "evaluate_url_policy",
        lambda _url: (_ for _ in ()).throw(AssertionError("unexpected second egress evaluation")),
        raising=False,
    )
    monkeypatch.setattr(
        filters,
        "http_fetch",
        lambda **_kwargs: {
            "status": 200,
            "text": "User-agent: *\nAllow: /\n",
        },
        raising=False,
    )

    decision = await policy.decide_web_outbound_policy(
        "https://example.com/page",
        respect_robots=True,
        user_agent="UA",
        source="article_extract",
        stage="pre_fetch",
        robots_filter=filters.RobotsFilter(user_agent="UA"),
    )

    assert decision.allowed is True
    assert decision.mode == "compat"
    assert decision.reason == "allowed"


@pytest.mark.asyncio
async def test_web_outbound_policy_blocks_internal_robots_check_errors_and_sanitizes_log(monkeypatch):
    monkeypatch.setenv("WEB_OUTBOUND_POLICY_MODE", "compat")
    policy = importlib.import_module(
        "tldw_Server_API.app.core.Web_Scraping.outbound_policy"
    )

    monkeypatch.setattr(
        policy,
        "evaluate_url_policy",
        lambda _url: SimpleNamespace(allowed=True, reason="allowed"),
        raising=False,
    )

    async def raise_internal_error(self, _url, *, skip_egress_check, fail_open):
        assert skip_egress_check is True
        assert fail_open is True
        raise RuntimeError("secret-token")

    class _BoundLogger:
        def __init__(self):
            self.bound = {}
            self.message = None

        def bind(self, **kwargs):
            self.bound = dict(kwargs)
            return self

        def debug(self, message):
            self.message = message

    fake_logger = _BoundLogger()
    monkeypatch.setattr(policy.RobotsFilter, "check", raise_internal_error, raising=False)
    monkeypatch.setattr(policy, "logger", fake_logger, raising=False)

    decision = await policy.decide_web_outbound_policy(
        "https://example.com/private?token=secret-token",
        respect_robots=True,
        user_agent="UA",
        source="article_extract",
        stage="pre_fetch",
    )

    assert decision.allowed is False
    assert decision.reason == "robots_check_error_internal"
    assert fake_logger.bound["sanitized_url"] == "example.com/private"
    assert fake_logger.bound["error_type"] == "RuntimeError"
    assert fake_logger.message == "Web outbound policy robots check failed"
