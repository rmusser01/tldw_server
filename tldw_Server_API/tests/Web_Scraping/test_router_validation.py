from collections import UserDict
from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from tldw_Server_API.app.core.Web_Scraping.scraper_router import (
    DEFAULT_HANDLER,
    ScraperRouter,
)


class _Unstringable:
    def __str__(self) -> str:
        raise ValueError("must not escape router normalization")


class _SecretValue:
    def __str__(self) -> str:
        return "router-review-secret"


class _StringSubclass(str):
    pass


class _IntegerSubclass(int):
    pass


class _EqBomb:
    def __init__(self, collision: str) -> None:
        self._collision = collision

    def __hash__(self) -> int:
        return hash(self._collision)

    def __eq__(self, other: object) -> bool:
        raise RuntimeError("hostile key equality must not run")

    def __repr__(self) -> str:
        return "<eqbomb-router-secret>"


class _BackendStringTrap(str):
    def strip(self, chars=None):
        raise RuntimeError("caller-owned strip must not run")

    def lower(self):
        raise RuntimeError("caller-owned lower must not run")


class _BoolTrap:
    def __bool__(self) -> bool:
        raise RuntimeError("caller-owned bool must not run")


class _FalsyBoolTrap:
    def __bool__(self) -> bool:
        return False


class _HookTrapMapping(Mapping):
    def __init__(self, data: dict[Any, Any]) -> None:
        self._data = data

    def __getitem__(self, key: Any) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: Any, default: Any = None) -> Any:
        raise RuntimeError("custom get must not run")

    def items(self):
        raise RuntimeError("custom items must not run")


class _LyingHookMapping(_HookTrapMapping):
    def __init__(self, data: dict[Any, Any], lie: dict[Any, Any]) -> None:
        super().__init__(data)
        self._lie = lie

    def get(self, key: Any, default: Any = None) -> Any:
        return self._lie.get(key, default)

    def items(self):
        return self._lie.items()


class _IterationTrapMapping(_HookTrapMapping):
    def __iter__(self) -> Iterator[Any]:
        raise RuntimeError("custom iteration must fail closed")

    def get(self, key: Any, default: Any = None) -> Any:
        return self._data.get(key, default)

    def items(self):
        return self._data.items()


class _ConversionTrapMapping(_IterationTrapMapping):
    def __iter__(self) -> Iterator[Any]:
        return iter(self._data)

    def keys(self):
        raise RuntimeError("custom mapping conversion must fail closed")


class _RaisingPatternList(list):
    def __bool__(self) -> bool:
        raise RuntimeError("custom list bool must not run")

    def __getitem__(self, key):
        if isinstance(key, slice):
            raise RuntimeError("custom list slicing must not run")
        return super().__getitem__(key)


class _FalsyPatternList(list):
    def __bool__(self) -> bool:
        return False


class _SliceTrapPatternList(list):
    def __getitem__(self, key):
        if isinstance(key, slice):
            raise RuntimeError("custom list slicing must not run")
        return super().__getitem__(key)


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


def test_hostile_domains_key_is_ignored_at_top_level_without_equality():
    rules = {
        _EqBomb("domains"): {
            "example.com": {"backend": "playwright"},
        }
    }

    assert ScraperRouter.validate_rules(rules) == {"domains": {}}

    plan = ScraperRouter(rules).resolve("https://example.com/path")

    assert plan.backend == "auto"
    assert "eqbomb-router-secret" not in repr(plan)


def test_hostile_backend_key_is_ignored_in_direct_and_validated_rules():
    rule = {
        _EqBomb("backend"): "playwright",
        "handler": DEFAULT_HANDLER,
    }
    rules = {"domains": {"example.com": rule}}

    cleaned = ScraperRouter.validate_rules(rules)
    direct = ScraperRouter(rules).resolve("https://example.com/path")
    validated = ScraperRouter(cleaned).resolve("https://example.com/path")

    assert direct == validated
    assert direct.backend == "auto"
    assert direct.handler == DEFAULT_HANDLER
    assert all(type(key) is str for key in cleaned["domains"]["example.com"])


def test_hostile_schema_rules_key_cannot_block_safe_settings_alias():
    settings = {
        _EqBomb("backend"): "eqbomb-router-secret",
        "title": {"selector": "h1"},
    }
    rule = {
        _EqBomb("schema_rules"): {"evil": True},
        "backend": "curl",
        "schema": settings,
    }

    direct, validated = _resolve_both(rule)

    expected = {"title": {"selector": "h1"}}
    assert direct == validated
    assert direct.backend == "curl"
    assert direct.schema_rules == expected
    assert all(type(key) is str for key in direct.schema_rules)
    assert "eqbomb-router-secret" not in repr(direct)


def test_scalar_maps_reject_hostile_keys_without_repr_leakage():
    mixed = UserDict(
        {
            _EqBomb("backend"): "eqbomb-router-secret",
            "X-Test": 7,
        }
    )

    direct, validated = _resolve_both(
        {
            "extra_headers": mixed,
            "cookies": mixed,
            "proxies": mixed,
        }
    )

    assert direct == validated
    assert direct.extra_headers == {"X-Test": "7"}
    assert direct.cookies == {"X-Test": "7"}
    assert direct.proxies == {"X-Test": "7"}
    assert "eqbomb-router-secret" not in repr(direct)


def test_exact_string_keys_work_alongside_hostile_config_keys():
    rules = UserDict(
        {
            _EqBomb("backend"): "ignored",
            "domains": UserDict(
                {
                    "example.com": UserDict(
                        {
                            _EqBomb("schema_rules"): {"evil": True},
                            "backend": "curl",
                            "extra_headers": {"X-Test": 7},
                        }
                    )
                }
            ),
        }
    )

    cleaned = ScraperRouter.validate_rules(rules)
    direct = ScraperRouter(rules).resolve("https://example.com/path")
    validated = ScraperRouter(cleaned).resolve("https://example.com/path")

    assert direct == validated
    assert direct.backend == "curl"
    assert direct.extra_headers == {"X-Test": "7"}


def test_mapping_get_and_items_hooks_are_not_used_after_snapshot():
    rule = _HookTrapMapping({"backend": "curl", "extra_headers": {"X-Test": 1}})
    rules = _HookTrapMapping({"domains": _HookTrapMapping({"example.com": rule})})

    cleaned = ScraperRouter.validate_rules(rules)
    direct = ScraperRouter(rules).resolve("https://example.com/path")
    validated = ScraperRouter(cleaned).resolve("https://example.com/path")

    assert direct == validated
    assert direct.backend == "curl"
    assert direct.extra_headers == {"X-Test": "1"}


def test_nested_mapping_items_hook_is_not_used_after_snapshot():
    rules = {"domains": {"example.com": _HookTrapMapping({"backend": "curl", "extra_headers": {"X-Test": 1}})}}

    cleaned = ScraperRouter.validate_rules(rules)
    direct = ScraperRouter(rules).resolve("https://example.com/path")
    validated = ScraperRouter(cleaned).resolve("https://example.com/path")

    assert direct == validated
    assert direct.backend == "curl"
    assert direct.extra_headers == {"X-Test": "1"}


def test_lying_mapping_get_and_items_cannot_widen_snapshotted_rules():
    actual_rule = {"backend": "curl"}
    lying_rule = {"backend": "playwright", "handler": "math:sqrt"}
    rules = _LyingHookMapping(
        {"domains": {"example.com": _LyingHookMapping(actual_rule, lying_rule)}},
        {"domains": {"example.com": lying_rule}},
    )

    cleaned = ScraperRouter.validate_rules(rules)
    direct = ScraperRouter(rules).resolve("https://example.com/path")
    validated = ScraperRouter(cleaned).resolve("https://example.com/path")

    assert direct == validated
    assert direct.backend == "curl"
    assert direct.handler == DEFAULT_HANDLER


@pytest.mark.parametrize(
    "rules",
    [
        _IterationTrapMapping({"domains": {"example.com": {"backend": "curl"}}}),
        _ConversionTrapMapping({"domains": {"example.com": {"backend": "curl"}}}),
    ],
    ids=["iteration", "dict-conversion"],
)
def test_failed_mapping_snapshots_return_empty_or_default(rules):
    assert ScraperRouter.validate_rules(rules) == {"domains": {}}

    plan = ScraperRouter(rules).resolve("https://example.com/path")

    assert plan.backend == "auto"
    assert plan.handler == DEFAULT_HANDLER


@pytest.mark.parametrize(
    "patterns",
    [
        _RaisingPatternList([".*"]),
        _FalsyPatternList([".*"]),
        _SliceTrapPatternList([".*"]),
    ],
    ids=["raising-bool", "lying-bool", "raising-slice"],
)
def test_url_pattern_list_subclasses_fail_closed_without_hooks(patterns):
    rules = {
        "domains": {
            "example.com": {
                "backend": "curl",
                "url_patterns": patterns,
            }
        }
    }

    assert ScraperRouter.validate_rules(rules) == {"domains": {}}
    assert ScraperRouter(rules).resolve("https://example.com/path").backend == "auto"


def test_exact_empty_url_pattern_list_remains_an_unconditional_constraint():
    rule = {"backend": "curl", "url_patterns": []}

    direct, validated = _resolve_both(rule)

    assert direct == validated
    assert direct.backend == "curl"


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
    secret = _SecretValue()
    mixed = UserDict(
        {
            "X-Text": "value",
            7: 9,
            "X-Bool": False,
            1.5: 2.5,
            "X-List": ["value"],
            "X-Bytes": b"secret",
            "X-Object": secret,
            _StringSubclass("X-Subclass-Key"): "bad",
            "X-Subclass-Value": _StringSubclass("bad"),
            _IntegerSubclass(8): 10,
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

    expected = {
        "X-Text": "value",
        "7": "9",
        "X-Bool": "False",
        "1.5": "2.5",
    }
    assert direct == validated
    assert direct.backend == "curl"
    assert type(direct.backend) is str
    assert direct.extra_headers == expected
    assert direct.cookies == expected
    assert direct.proxies == expected
    assert all(type(key) is str for key in direct.extra_headers)
    assert all(type(value) is str for value in direct.extra_headers.values())
    assert "router-review-secret" not in repr(direct)


def test_backend_and_scalar_subclasses_fail_safely_to_canonical_defaults():
    direct, validated = _resolve_both(
        {
            "backend": _BackendStringTrap("curl"),
            "ua_profile": _StringSubclass("firefox_120_win"),
            "impersonate": _StringSubclass("firefox120"),
            "strategy_order": _RaisingPatternList(["schema"]),
        }
    )

    assert direct == validated
    assert direct.backend == "auto"
    assert type(direct.backend) is str
    assert direct.ua_profile == "chrome_120_win"
    assert type(direct.ua_profile) is str
    assert direct.impersonate == "chrome120"
    assert type(direct.impersonate) is str
    assert direct.strategy_order is None


@pytest.mark.parametrize("malformed", [_BoolTrap(), _FalsyBoolTrap(), [], object()])
def test_malformed_robots_values_use_defaults_without_truthiness_hooks(malformed):
    rule = {"respect_robots": malformed}
    rules = {"domains": {"example.com": rule}}

    cleaned = ScraperRouter.validate_rules(rules)
    direct = ScraperRouter(
        rules,
        default_respect_robots=_BoolTrap(),
    ).resolve("https://example.com/path")
    validated = ScraperRouter(
        cleaned,
        default_respect_robots=_BoolTrap(),
    ).resolve("https://example.com/path")

    assert "respect_robots" not in cleaned["domains"]["example.com"]
    assert direct == validated
    assert direct.respect_robots is True
    assert type(direct.respect_robots) is bool


@pytest.mark.parametrize(
    ("value", "expected"),
    [(False, False), (True, True), (0, False), (1, True), ("", False), ("false", True)],
)
def test_safe_builtin_robots_values_preserve_boolean_compatibility(value, expected):
    direct, validated = _resolve_both({"respect_robots": value})

    assert direct == validated
    assert direct.respect_robots is expected


def test_settings_aliases_fail_over_after_safe_mapping_snapshots():
    nested = UserDict({"selector": "h1"})
    invalid_primary = _ConversionTrapMapping({"ignored": True})
    valid_alias = _HookTrapMapping({"title": nested})
    rule = {
        "schema_rules": invalid_primary,
        "schema": valid_alias,
        "llm_settings": valid_alias,
        "regex_settings": invalid_primary,
        "regex": valid_alias,
    }

    direct, validated = _resolve_both(rule)

    assert direct == validated
    assert direct.schema_rules == {"title": nested}
    assert direct.llm_settings == {"title": nested}
    assert direct.regex_settings == {"title": nested}
    assert type(direct.schema_rules) is dict
    assert type(direct.llm_settings) is dict
    assert type(direct.regex_settings) is dict


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
