"""Contract tests for immutable Phase 4C article request models."""

from __future__ import annotations

from dataclasses import fields
from types import MappingProxyType

import pytest

from tldw_Server_API.app.core.Web_Scraping.orchestration.article_models import (
    PUBLIC_FAILURE_CODES,
    ArticleFailure,
    ArticleLimits,
    ArticlePlan,
    DirectBrowserProfile,
    article_failure_result,
)
from tldw_Server_API.app.core.Web_Scraping.scraper_router import ScrapePlan


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ({}, (16_777_216, 67_108_864)),
        ({"web_scraper_max_article_bytes": "not-a-number"}, (16_777_216, 67_108_864)),
        ({"web_scraper_max_article_bytes": True}, (16_777_216, 67_108_864)),
        ({"web_scraper_max_article_bytes": 0}, (16_777_216, 67_108_864)),
        ({"web_scraper_max_browser_transfer_bytes": -1}, (16_777_216, 67_108_864)),
        ({"web_scraper_max_article_bytes": 1.5}, (16_777_216, 67_108_864)),
        ({"web_scraper_max_browser_transfer_bytes": "2.0"}, (16_777_216, 67_108_864)),
        ({"web_scraper_max_article_bytes": "9" * 1_000}, (16_777_216, 67_108_864)),
        ({"web_scraper_max_article_bytes": "9" * 5_000}, (16_777_216, 67_108_864)),
        ({"web_scraper_max_article_bytes": 1_073_741_825}, (16_777_216, 67_108_864)),
        ({"web_scraper_max_browser_transfer_bytes": "1073741825"}, (16_777_216, 67_108_864)),
    ],
)
def test_article_limits_fall_back_when_configured_values_are_not_positive_integers(
    configured: dict[str, object],
    expected: tuple[int, int],
) -> None:
    limits = ArticleLimits.from_mapping(configured)

    assert (limits.max_article_bytes, limits.max_browser_transfer_bytes) == expected


def test_article_limits_use_positive_integer_config_values() -> None:
    limits = ArticleLimits.from_mapping(
        {
            "web_scraper_max_article_bytes": "1024",
            "web_scraper_max_browser_transfer_bytes": 4096,
        }
    )

    assert limits == ArticleLimits(max_article_bytes=1024, max_browser_transfer_bytes=4096)


def test_article_limits_accept_the_explicit_configuration_ceiling() -> None:
    limits = ArticleLimits(
        max_article_bytes=1_073_741_824,
        max_browser_transfer_bytes="1073741824",
    )

    assert limits == ArticleLimits(
        max_article_bytes=1_073_741_824,
        max_browser_transfer_bytes=1_073_741_824,
    )


def test_full_loaded_config_uses_raw_limits_and_legacy_route_values() -> None:
    routing_plan = ScrapePlan(url="https://example.com/article", domain="example.com")
    loaded_config = {
        "web_scraper": {
            "web_scraper_default_backend": "curl",
            "web_scraper_retry_count": "4",
            "web_scraper_retry_timeout": "8",
            "web_scraper_stealth_playwright": "true",
        },
        "Web-Scraping": {
            "web_scraper_max_article_bytes": "1234",
            "web_scraper_max_browser_transfer_bytes": "5678",
            "stealth_wait_ms": "90",
        },
    }

    plan = ArticlePlan.from_routing_plan(routing_plan, loaded_config)

    assert plan.backend == "curl"
    assert plan.limits == ArticleLimits(max_article_bytes=1234, max_browser_transfer_bytes=5678)
    assert plan.browser.retries == 4
    assert plan.browser.timeout_ms == 8000
    assert plan.browser.stealth_enabled is True
    assert plan.browser.stealth_wait_ms == 90


def test_article_plan_snapshots_lightweight_and_browser_inputs_without_cross_leaking() -> None:
    caller_cookie = {"name": "session", "value": "caller-value", "metadata": {"path": ["/"]}}
    routing_plan = ScrapePlan(
        url="https://example.com/article",
        domain="example.com",
        backend="curl",
        handler="tldw_Server_API.app.core.Web_Scraping.handlers:handle_generic_html",
        extra_headers={"X-Plan": "plan-header"},
        cookies={"plan": "plan-cookie"},
        proxies={"https": "http://proxy.invalid"},
        strategy_order=["jsonld", "trafilatura"],
        schema_rules={"headline": {"selector": "h1"}},
    )
    config = {
        "web_scraper_max_article_bytes": 1024,
        "web_scraper_max_browser_transfer_bytes": 4096,
        "web_scraper_retry_count": 2,
        "web_scraper_retry_timeout": 9,
        "web_scraper_stealth_playwright": True,
        "STEALTH_WAIT_MS": 45,
    }

    plan = ArticlePlan.from_routing_plan(routing_plan, config, [caller_cookie])
    caller_cookie["metadata"]["path"].append("/changed")
    routing_plan.extra_headers["X-Plan"] = "changed"
    routing_plan.cookies["plan"] = "changed"
    routing_plan.proxies["https"] = "http://changed.invalid"
    routing_plan.strategy_order.append("llm")

    assert plan.headers["X-Plan"] == "plan-header"
    assert plan.cookies == {"plan": "plan-cookie"}
    assert plan.proxies == {"https": "http://proxy.invalid"}
    assert plan.strategy_order == ("jsonld", "trafilatura")
    assert plan.schema_rules == {"headline": {"selector": "h1"}}
    assert plan.limits == ArticleLimits(max_article_bytes=1024, max_browser_transfer_bytes=4096)
    assert plan.browser.custom_cookies[0]["metadata"]["path"] == ("/",)
    assert isinstance(plan.browser.custom_cookies[0], MappingProxyType)
    assert {field.name for field in fields(DirectBrowserProfile)} == {
        "user_agent",
        "custom_cookies",
        "retries",
        "timeout_ms",
        "stealth_enabled",
        "stealth_wait_ms",
        "viewport_width",
        "viewport_height",
    }
    assert plan.browser.retries == 2
    assert plan.browser.timeout_ms == 9000
    assert plan.browser.stealth_enabled is True
    assert plan.browser.stealth_wait_ms == 45
    assert (plan.browser.viewport_width, plan.browser.viewport_height) == (1280, 720)


def test_direct_browser_profile_freezes_mutable_cookie_sets_and_byte_buffers() -> None:
    cookie = {
        "name": "session",
        "value": bytearray(b"value"),
        "metadata": {
            "scopes": {"read"},
            "payload": memoryview(bytearray(b"payload")),
        },
    }

    profile = DirectBrowserProfile(
        user_agent="agent",
        custom_cookies=(cookie,),
        retries=1,
        timeout_ms=2,
        stealth_enabled=False,
        stealth_wait_ms=3,
    )
    cookie["value"][0] = ord("X")
    cookie["metadata"]["scopes"].add("write")
    cookie["metadata"]["payload"][0] = ord("X")

    assert profile.custom_cookies[0]["value"] == b"value"
    assert profile.custom_cookies[0]["metadata"]["scopes"] == frozenset({"read"})
    assert profile.custom_cookies[0]["metadata"]["payload"] == b"payload"


def test_direct_browser_profile_does_not_apply_response_budget_ceiling_to_viewport() -> None:
    profile = DirectBrowserProfile(
        user_agent="agent",
        custom_cookies=(),
        retries=1,
        timeout_ms=2,
        stealth_enabled=False,
        stealth_wait_ms=3,
        viewport_width=1_073_741_825,
        viewport_height="1073741825",
    )

    assert (profile.viewport_width, profile.viewport_height) == (
        1_073_741_825,
        1_073_741_825,
    )


def test_public_failure_codes_match_the_approved_contract() -> None:
    assert (
        frozenset(
            {
                "policy_error",
                "regex_invalid",
                "regex_too_large",
                "regex_timeout",
                "selector_invalid",
                "provider_error",
                "fetch_error",
                "browser_error",
                "browser_transport_unavailable",
                "response_too_large",
                "extraction_error",
            }
        )
        == PUBLIC_FAILURE_CODES
    )


@pytest.mark.parametrize(
    "code",
    ["policy_error", "fetch_error", "browser_error", "response_too_large", "extraction_error"],
)
def test_article_failure_results_expose_only_the_stable_public_code(code: str) -> None:
    failure = ArticleFailure(code, "fetch")

    assert str(failure) == code
    assert failure.code == code
    assert failure.stage == "fetch"
    assert article_failure_result(failure) == {
        "title": "N/A",
        "author": "N/A",
        "date": "N/A",
        "content": "",
        "extraction_successful": False,
        "error": code,
    }


def test_unknown_article_failure_code_is_sanitized_without_a_url_or_error_text() -> None:
    result = article_failure_result(ArticleFailure("transport https://token@example.com/path", "fetch"))

    assert result["error"] == "extraction_error"
    assert "url" not in result
    assert "token" not in str(result)


def test_browser_transport_failure_snapshots_exact_bounded_capability() -> None:
    capability: dict[str, object] = {
        "name": "safe_browser_transport",
        "available": False,
        "configured_mode": "auto",
        "effective_mode": "disabled",
        "dns_peer_attested": False,
        "reason": "browser_transport_unattested",
    }
    failure = ArticleFailure(
        "browser_transport_unavailable",
        "browser_transport_unattested",
        capability=capability,
    )
    capability["reason"] = "secret-mutated-reason"

    assert article_failure_result(failure) == {
        "title": "N/A",
        "author": "N/A",
        "date": "N/A",
        "content": "",
        "extraction_successful": False,
        "error": "browser_transport_unavailable",
        "capability": {
            "name": "safe_browser_transport",
            "available": False,
            "configured_mode": "auto",
            "effective_mode": "disabled",
            "dns_peer_attested": False,
            "reason": "browser_transport_unattested",
        },
    }


def test_invalid_browser_transport_capability_is_replaced_fail_closed() -> None:
    result = article_failure_result(
        ArticleFailure(
            "browser_transport_unavailable",
            "browser_transport_unattested",
            capability={
                "name": "safe_browser_transport",
                "available": False,
                "configured_mode": ["secret"],
                "effective_mode": "disabled",
                "dns_peer_attested": False,
                "reason": "browser_transport_unattested",
            },
        )
    )

    assert result["capability"] == {
        "name": "safe_browser_transport",
        "available": False,
        "configured_mode": "disabled",
        "effective_mode": "disabled",
        "dns_peer_attested": False,
        "reason": "browser_transport_config_invalid",
    }
    assert "secret" not in str(result)
