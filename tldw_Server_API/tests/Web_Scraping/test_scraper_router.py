from collections import UserDict

import pytest

from tldw_Server_API.app.core.Web_Scraping.scraper_router import (
    DEFAULT_HANDLER,
    ScraperRouter,
    _match_domain_rule,
    _validate_handler,
)


def test_router_precedence_exact_over_wildcard_and_patterns():
    rules = {
        "domains": {
            "example.com": {
                "backend": "curl",
                "handler": "tldw_Server_API.app.core.Web_Scraping.handlers:handle_generic_html",
                "ua_profile": "chrome_120_win",
                "impersonate": "chrome120",
            },
            "*.example.com": {
                "backend": "httpx",
                "handler": "tldw_Server_API.app.core.Web_Scraping.handlers:handle_generic_html",
                "ua_profile": "firefox_120_win",
                "url_patterns": [".*\\?output=1$"],
            },
        }
    }
    router = ScraperRouter(rules, ua_mode="fixed")

    # Exact domain wins
    plan1 = router.resolve("https://example.com/article?id=1")
    assert plan1.domain == "example.com"
    assert plan1.backend == "curl"
    assert plan1.ua_profile == "chrome_120_win"

    # Wildcard applies for subdomain when no exact rule exists
    plan2 = router.resolve("https://sub.example.com/post?output=1")
    assert plan2.domain == "sub.example.com"
    assert plan2.backend == "httpx"
    assert plan2.ua_profile == "firefox_120_win"

    # Wildcard pattern present but not matching -> falls back to default plan
    plan3 = router.resolve("https://sub.example.com/post?id=2")
    assert plan3.backend == "auto"  # default


def test_handler_allowlist_blocks_unknown():
    rules = {
        "domains": {
            "evil.example": {
                "backend": "curl",
                "handler": "os.system:rm -rf /",  # should be denied
            }
        }
    }
    router = ScraperRouter(rules)
    plan = router.resolve("https://evil.example/")
    # Should fall back to safe handler
    assert plan.handler.startswith("tldw_Server_API.app.core.Web_Scraping.handlers:")


def test_router_proxies_parsed():
    rules = {
        "domains": {
            "proxied.example": {
                "backend": "curl",
                "proxies": {"http": "http://localhost:8080", "https": "http://localhost:8080"},
            }
        }
    }
    router = ScraperRouter(ScraperRouter.validate_rules(rules))
    plan = router.resolve("https://proxied.example/path")
    assert plan.proxies.get("http").startswith("http://")


@pytest.mark.parametrize(
    "rules",
    [
        ["not", "a", "mapping"],
        {"domains": ["not", "a", "mapping"]},
        {"domains": {"example.com": ["not", "a", "mapping"]}},
    ],
    ids=["top-level", "domains", "matched-rule"],
)
def test_direct_router_non_mapping_boundaries_return_default_plan(rules):
    plan = ScraperRouter(rules).resolve("https://example.com/path")

    assert plan.backend == "auto"
    assert plan.handler == DEFAULT_HANDLER
    assert plan.extra_headers == {}
    assert plan.cookies == {}
    assert plan.proxies == {}


def test_match_domain_rule_skips_non_string_keys_and_non_mapping_wildcard_rules():
    valid_rule = UserDict({"backend": "curl"})
    rules = UserDict(
        {
            "domains": UserDict(
                {
                    123: {"backend": "playwright"},
                    "*.example.com": ["not", "a", "mapping"],
                    "*.sub.example.com": valid_rule,
                }
            )
        }
    )

    match = _match_domain_rule("deep.sub.example.com", rules)

    assert match == ("*.sub.example.com", valid_rule)


def test_non_mapping_exact_rule_does_not_fall_through_to_wildcard():
    rules = {
        "domains": {
            "example.com": ["not", "a", "mapping"],
            "*.example.com": {"backend": "curl"},
        }
    }

    assert _match_domain_rule("example.com", rules) is None


def test_validate_handler_ignores_non_string_allowlist_entries():
    allowlist = [123, None, "approved.handlers:", {"invalid": "prefix"}]

    assert _validate_handler("approved.handlers:extract", allowlist) == ("approved.handlers:extract")
    assert _validate_handler(123, allowlist) == DEFAULT_HANDLER
    assert _validate_handler("approved.handlers:extract", object()) == DEFAULT_HANDLER


def test_direct_router_handler_allowlist_with_non_string_entries_is_safe():
    rules = {
        "domains": {
            "example.com": {
                "handler": "approved.handlers:extract",
            }
        }
    }

    plan = ScraperRouter(
        rules,
        handler_allowlist=[123, "approved.handlers:"],
    ).resolve("https://example.com/path")

    assert plan.handler == "approved.handlers:extract"


def test_direct_router_falsy_non_list_pattern_constraint_fails_open():
    router = ScraperRouter(
        {
            "domains": {
                "example.com": {
                    "backend": "curl",
                    "url_patterns": {},
                }
            }
        }
    )

    assert router.resolve("https://example.com/path").backend == "auto"
