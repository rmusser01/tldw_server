from collections import UserDict

import pytest

from tldw_Server_API.app.core.Web_Scraping import scraper_router as scraper_router_module
from tldw_Server_API.app.core.Web_Scraping.safe_regex import SafeRegexResult
from tldw_Server_API.app.core.Web_Scraping.scraper_router import (
    DEFAULT_HANDLER,
    DEFAULT_HANDLER_ALLOWLIST,
    ScraperRouter,
    _match_domain_rule,
    _validate_handler,
)


class _HandlerStringSubclass(str):
    def startswith(self, prefix, *args):
        return True


class _PrefixStringSubclass(str):
    pass


class _DomainStringSubclass(str):
    def startswith(self, prefix, *args):
        raise RuntimeError("caller-owned domain startswith must not run")


class _AllowlistBoolTrap(list):
    def __bool__(self) -> bool:
        raise RuntimeError("custom allowlist bool must not run")


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


def _has_int_digit_limit_behavior() -> bool:
    try:
        str(10**5000)
    except ValueError:
        return True
    return False


_INT_DIGIT_LIMIT_ACTIVE = _has_int_digit_limit_behavior()


@pytest.mark.skipif(
    not _INT_DIGIT_LIMIT_ACTIVE,
    reason="interpreter does not enforce a decimal digit limit for int-to-string conversion",
)
@pytest.mark.parametrize("bad_setting", ["ua_profile", "impersonate"])
def test_direct_router_drops_only_digit_limited_scalar_values(bad_setting):
    huge = 10**5000
    rule = {
        "backend": "curl",
        "ua_profile": "firefox_120_win",
        "impersonate": "firefox120",
        "extra_headers": {
            "X-Ordinary": "header-value",
            "X-Integer": 7,
            huge: "drop-huge-key",
            "X-Huge-Value": huge,
        },
        "cookies": {
            "session": "cookie-value",
            huge: "drop-huge-key",
            "drop-cookie-value": huge,
        },
        "proxies": {
            "http": "http://proxy.local",
            huge: "drop-huge-key",
            "drop-proxy-value": huge,
        },
    }
    rule[bad_setting] = huge

    plan = ScraperRouter({"domains": {"example.com": rule}}).resolve("https://example.com/path")

    assert plan.backend == "curl"
    if bad_setting == "ua_profile":
        assert plan.ua_profile == "chrome_120_win"
        assert plan.impersonate == "firefox120"
    else:
        assert plan.ua_profile == "firefox_120_win"
        assert plan.impersonate == "firefox120"
    assert plan.extra_headers == {"X-Ordinary": "header-value", "X-Integer": "7"}
    assert plan.cookies == {"session": "cookie-value"}
    assert plan.proxies == {"http": "http://proxy.local"}


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


def test_domain_key_string_subclasses_fail_safely_without_string_hooks():
    domain = _DomainStringSubclass("*.example.com")
    rules = {"domains": {domain: {"backend": "curl"}}}

    assert ScraperRouter.validate_rules(rules) == {"domains": {}}
    assert ScraperRouter(rules).resolve("https://sub.example.com/path").backend == "auto"


def test_validate_handler_ignores_non_string_allowlist_entries():
    allowlist = [123, None, "approved.handlers:", {"invalid": "prefix"}]

    assert _validate_handler("approved.handlers:extract", allowlist) == ("approved.handlers:extract")
    assert _validate_handler(123, allowlist) == DEFAULT_HANDLER
    assert _validate_handler("approved.handlers:extract", object()) == DEFAULT_HANDLER


@pytest.mark.parametrize(
    ("handler", "allowlist"),
    [
        ("math:sqrt", [""]),
        ("math:sqrt", [_PrefixStringSubclass("math:")]),
        (_HandlerStringSubclass("math:sqrt"), list(DEFAULT_HANDLER_ALLOWLIST)),
        ("math:sqrt", _AllowlistBoolTrap(["math:"])),
    ],
    ids=["empty-prefix", "prefix-subclass", "handler-subclass", "container-bool"],
)
def test_validate_handler_rejects_protocol_subclasses_and_empty_prefixes(
    handler,
    allowlist,
):
    assert _validate_handler(handler, allowlist) == DEFAULT_HANDLER


@pytest.mark.parametrize(
    "allowlist",
    [
        [""],
        [_PrefixStringSubclass("math:")],
        _AllowlistBoolTrap(["math:"]),
    ],
    ids=["empty-prefix", "prefix-subclass", "container-bool"],
)
def test_router_canonicalizes_unsafe_allowlists_to_safe_tuple(allowlist):
    rules = {"domains": {"example.com": {"handler": "math:sqrt"}}}

    router = ScraperRouter(rules, handler_allowlist=allowlist)
    plan = router.resolve("https://example.com/path")

    assert type(router.allowlist) is tuple
    assert router.allowlist == tuple(DEFAULT_HANDLER_ALLOWLIST)
    assert plan.handler == DEFAULT_HANDLER


def test_router_canonicalizes_mixed_allowlist_without_subclass_prefixes():
    rules = {
        "domains": {
            "example.com": {
                "handler": "approved.handlers:extract",
            }
        }
    }

    router = ScraperRouter(
        rules,
        handler_allowlist=[
            "",
            123,
            _PrefixStringSubclass("math:"),
            "approved.handlers:",
        ],
    )
    plan = router.resolve("https://example.com/path")

    assert type(router.allowlist) is tuple
    assert router.allowlist == ("approved.handlers:",)
    assert plan.handler == "approved.handlers:extract"
    assert type(plan.handler) is str


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


def test_direct_rule_applies_match_at_url_pattern_position_33():
    patterns = [rf"/never-{index}$" for index in range(32)] + [r"/target$"]
    router = ScraperRouter(
        {
            "domains": {
                "example.com": {
                    "backend": "curl",
                    "url_patterns": patterns,
                }
            }
        }
    )

    plan = router.resolve("https://example.com/target")

    assert plan.backend == "curl"


def test_direct_rule_checks_later_match_after_prior_search_time(
    monkeypatch: pytest.MonkeyPatch,
):
    state = {"now": 0.0}
    calls: list[tuple[str, float]] = []

    def fake_monotonic() -> float:
        return state["now"]

    def fake_search(pattern, _value, *, limits):
        calls.append((pattern, limits.timeout_s))
        state["now"] += min(0.040, limits.timeout_s)
        return SafeRegexResult(matched=pattern == "match-later")

    monkeypatch.setattr(scraper_router_module, "_monotonic", fake_monotonic, raising=False)
    monkeypatch.setattr(scraper_router_module, "search_untrusted", fake_search)
    patterns = ["no-match-1", "no-match-2", "no-match-3", "match-later"]
    router = ScraperRouter(
        {
            "domains": {
                "example.com": {
                    "backend": "curl",
                    "url_patterns": patterns,
                }
            }
        }
    )

    plan = router.resolve("https://example.com/target")

    assert plan.backend == "curl"
    assert calls == [(pattern, 0.100) for pattern in patterns]
