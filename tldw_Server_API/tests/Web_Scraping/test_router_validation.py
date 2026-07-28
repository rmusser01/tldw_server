from collections import UserDict

import pytest

from tldw_Server_API.app.core.Web_Scraping.scraper_router import (
    DEFAULT_HANDLER,
    ScraperRouter,
)


class _Unstringable:
    def __str__(self) -> str:
        raise ValueError("must not escape router normalization")


def _resolve_both(rule):
    rules = {"domains": {"example.com": rule}}
    direct = ScraperRouter(rules).resolve("https://example.com/path")
    validated = ScraperRouter(ScraperRouter.validate_rules(rules)).resolve("https://example.com/path")
    return direct, validated


def test_validate_rules_normalizes_and_drops_invalid():
    raw = {
        "domains": {
            "invalid": {"backend": "curl", "unknown": True},  # no dot or wildcard
            "example.com": {
                "backend": "bogus",
                "handler": "tldw_Server_API.app.core.Web_Scraping.handlers:handle_generic_html",
                "url_patterns": [".*\\?ok=1$", "["],  # second is invalid regex
                "extra_headers": {"Referer": "https://google.com"},
                "cookies": [{"k": "v"}],  # wrong shape
                "unknown_key": 123,
            },
            "*.sub.example.com": {
                "backend": "curl",
                "url_patterns": [".*"],
            },
        }
    }

    cleaned = ScraperRouter.validate_rules(raw)
    assert "invalid" not in cleaned.get("domains", {})

    ex = cleaned["domains"]["example.com"]
    # backend normalized to 'auto'
    assert ex["backend"] == "auto"
    # unknown keys dropped; invalid regex removed
    assert ex.get("unknown_key") is None
    assert ex["url_patterns"] == [".*\\?ok=1$"]
    # cookies normalized to map
    assert ex["cookies"] == {}

    sub = cleaned["domains"]["*.sub.example.com"]
    assert sub["backend"] == "curl"
    assert sub["url_patterns"] == [".*"]


@pytest.mark.parametrize("value", [None, 123, [], "invalid", object()])
def test_validate_rules_accepts_any_non_mapping_object(value):
    assert ScraperRouter.validate_rules(value) == {"domains": {}}


def test_validate_rules_accepts_mapping_implementations():
    raw = UserDict(
        {
            "domains": UserDict(
                {
                    "example.com": UserDict(
                        {
                            "backend": "curl",
                            "extra_headers": UserDict({"X-Test": 1}),
                        }
                    )
                }
            )
        }
    )

    cleaned = ScraperRouter.validate_rules(raw)

    assert cleaned == {
        "domains": {
            "example.com": {
                "backend": "curl",
                "extra_headers": {"X-Test": "1"},
            }
        }
    }


def test_malformed_rule_values_normalize_without_widening_types():
    rule = {
        "backend": ["curl"],
        "handler": 123,
        "extra_headers": 123,
        "cookies": ["not", "a", "mapping"],
        "proxies": object(),
        "ua_profile": ["firefox_120_win"],
        "impersonate": {"value": "firefox120"},
        "strategy_order": ["schema", 123, {}, "llm"],
        "schema_rules": ["invalid"],
        "schema": {"title": {"selector": "h1"}},
        "llm_settings": "invalid",
        "llm": {"provider": "openai"},
        "regex_settings": 123,
        "regex": {"mask_pii": True},
        "cluster_settings": [],
        "cluster": {"cluster_linkage": "complete"},
        "respect_robots": False,
    }

    cleaned = ScraperRouter.validate_rules({"domains": {"example.com": rule}})
    normalized = cleaned["domains"]["example.com"]

    assert normalized["backend"] == "auto"
    assert normalized["handler"] == DEFAULT_HANDLER
    assert normalized["extra_headers"] == {}
    assert normalized["cookies"] == {}
    assert normalized["proxies"] == {}
    assert "ua_profile" not in normalized
    assert "impersonate" not in normalized
    assert normalized["strategy_order"] == ["schema", "llm"]
    assert "schema_rules" not in normalized
    assert normalized["schema"] == {"title": {"selector": "h1"}}
    assert "llm_settings" not in normalized
    assert normalized["llm"] == {"provider": "openai"}


def test_malformed_rule_values_have_validated_and_direct_plan_parity():
    rule = {
        "backend": ["curl"],
        "handler": 123,
        "extra_headers": 123,
        "cookies": ["not", "a", "mapping"],
        "proxies": object(),
        "ua_profile": ["firefox_120_win"],
        "impersonate": {"value": "firefox120"},
        "strategy_order": ["schema", 123, {}, "llm"],
        "schema_rules": ["invalid"],
        "schema": {"title": {"selector": "h1"}},
        "llm_settings": "invalid",
        "llm": {"provider": "openai"},
        "regex_settings": 123,
        "regex": {"mask_pii": True},
        "cluster_settings": [],
        "cluster": {"cluster_linkage": "complete"},
        "respect_robots": False,
    }

    direct, validated = _resolve_both(rule)

    assert direct == validated
    assert direct.backend == "auto"
    assert direct.handler == DEFAULT_HANDLER
    assert direct.ua_profile == "chrome_120_win"
    assert direct.impersonate == "chrome120"
    assert direct.extra_headers == {}
    assert direct.cookies == {}
    assert direct.proxies == {}
    assert direct.strategy_order == ["schema", "llm"]
    assert direct.schema_rules == {"title": {"selector": "h1"}}
    assert direct.llm_settings == {"provider": "openai"}
    assert direct.regex_settings == {"mask_pii": True}
    assert direct.cluster_settings == {"cluster_linkage": "complete"}
    assert direct.respect_robots is False


def test_mapping_entries_normalize_consistently_in_validated_and_direct_plans():
    invalid = _Unstringable()
    mixed = UserDict(
        {
            "X-Text": "value",
            7: 9,
            "X-List": ["value"],
            invalid: "ignored-key",
            "ignored-value": invalid,
        }
    )
    rule = {
        "backend": " CURL ",
        "extra_headers": mixed,
        "cookies": mixed,
        "proxies": mixed,
    }

    direct, validated = _resolve_both(rule)

    expected = {"X-Text": "value", "7": "9", "X-List": "['value']"}
    assert direct == validated
    assert direct.backend == "curl"
    assert direct.extra_headers == expected
    assert direct.cookies == expected
    assert direct.proxies == expected


def test_valid_scalar_ua_values_have_validated_and_direct_plan_parity():
    direct, validated = _resolve_both(
        {
            "backend": "PLAYWRIGHT",
            "handler": DEFAULT_HANDLER,
            "ua_profile": 123,
            "impersonate": 456,
        }
    )

    assert direct == validated
    assert direct.backend == "playwright"
    assert direct.handler == DEFAULT_HANDLER
    assert direct.ua_profile == "123"
    assert direct.impersonate == "456"
