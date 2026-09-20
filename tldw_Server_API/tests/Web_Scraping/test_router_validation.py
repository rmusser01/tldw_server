from collections import UserDict
from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from tldw_Server_API.app.core.Web_Scraping import scraper_router as scraper_router_module
from tldw_Server_API.app.core.Web_Scraping.safe_regex import SafeRegexResult
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
        self.eq_calls = 0
        self.repr_calls = 0

    def __hash__(self) -> int:
        return hash(self._collision)

    def __eq__(self, other: object) -> bool:
        self.eq_calls += 1
        raise RuntimeError("hostile key equality must not run")

    def __repr__(self) -> str:
        self.repr_calls += 1
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
        return self._data.items()


class _LyingHookMapping(_HookTrapMapping):
    def __init__(self, data: dict[Any, Any], lie: dict[Any, Any]) -> None:
        super().__init__(data)
        self._lie = lie

    def get(self, key: Any, default: Any = None) -> Any:
        return self._lie.get(key, default)

    def items(self):
        return self._data.items()


class _ItemsSubstitutionMapping(_HookTrapMapping):
    def __init__(self, data: dict[Any, Any], substitution: dict[Any, Any]) -> None:
        super().__init__(data)
        self._substitution = substitution
        self.items_calls = 0

    def items(self):
        self.items_calls += 1
        return self._substitution.items()


class _ItemsSubstitutionDict(dict):
    def __init__(self, data: dict[Any, Any], substitution: dict[Any, Any]) -> None:
        super().__init__(data)
        self._substitution = substitution
        self.items_calls = 0

    def items(self):
        self.items_calls += 1
        return self._substitution.items()


class _ItemsIterationTrapMapping(_HookTrapMapping):

    def get(self, key: Any, default: Any = None) -> Any:
        return self._data.get(key, default)

    def items(self):
        yielded = False
        for item in self._data.items():
            if yielded:
                raise RuntimeError("custom items iteration must fail closed")
            yielded = True
            yield item
        raise RuntimeError("custom items iteration must fail closed")


class _ItemsAcquisitionTrapMapping(_HookTrapMapping):
    def items(self):
        raise RuntimeError("custom items acquisition must fail closed")


class _ItemsUnpackingTrapMapping(_HookTrapMapping):
    def items(self):
        yield from self._data.items()
        yield ("malformed", "entry", "shape")


class _CanonicalIterationTrapMapping(_HookTrapMapping):
    def __iter__(self) -> Iterator[Any]:
        yielded = False
        for key in self._data:
            if yielded:
                raise RuntimeError("canonical iteration must fail closed")
            yielded = True
            yield key
        raise RuntimeError("canonical iteration must fail closed")


class _CanonicalLookupTrapMapping(_HookTrapMapping):
    def __init__(self, data: dict[Any, Any], fail_key: Any) -> None:
        super().__init__(data)
        self._fail_key = fail_key

    def __getitem__(self, key: Any) -> Any:
        if key == self._fail_key:
            raise RuntimeError("canonical lookup must fail closed")
        return self._data[key]


class _PairMapping(Mapping):
    def __init__(self, pairs: list[tuple[Any, Any]]) -> None:
        self._pairs = list(pairs)

    def __getitem__(self, key: Any) -> Any:
        for stored_key, value in self._pairs:
            if key is stored_key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[Any]:
        return (key for key, _value in self._pairs)

    def __len__(self) -> int:
        return len(self._pairs)

    def items(self):
        return iter(self._pairs)


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


def _has_int_digit_limit_behavior() -> bool:
    try:
        str(10**5000)
    except ValueError:
        return True
    return False


_INT_DIGIT_LIMIT_ACTIVE = _has_int_digit_limit_behavior()


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

    assert direct.backend == "curl"
    assert validated.backend == "curl"
    assert direct.extra_headers == {"X-Test": 7}
    assert validated.extra_headers == {"X-Test": "7"}


def test_overridden_items_cannot_substitute_canonical_mapping_values():
    headers = _ItemsSubstitutionMapping(
        {"X-Test": 7},
        {"X-Test": "substituted"},
    )
    settings = _ItemsSubstitutionMapping(
        {"title": {"selector": "h1"}},
        {"title": {"selector": "script"}},
    )
    rule = _ItemsSubstitutionMapping(
        {
            "backend": "curl",
            "respect_robots": True,
            "extra_headers": headers,
            "schema": settings,
        },
        {
            "backend": "playwright",
            "respect_robots": False,
            "extra_headers": {"X-Test": "substituted"},
            "schema": {"title": {"selector": "script"}},
        },
    )
    domains = _ItemsSubstitutionMapping(
        {"example.com": rule},
        {
            "example.com": {
                "backend": "playwright",
                "respect_robots": False,
                "extra_headers": {"X-Test": "substituted"},
                "schema": {"title": {"selector": "script"}},
            }
        },
    )
    rules = _ItemsSubstitutionMapping(
        {"domains": domains},
        {
            "domains": {
                "example.com": {
                    "backend": "playwright",
                    "respect_robots": False,
                    "extra_headers": {"X-Test": "substituted"},
                    "schema": {"title": {"selector": "script"}},
                }
            }
        },
    )

    cleaned = ScraperRouter.validate_rules(rules)
    direct = ScraperRouter(rules).resolve("https://example.com/path")
    validated = ScraperRouter(cleaned).resolve("https://example.com/path")

    assert direct == validated
    assert direct.backend == "curl"
    assert direct.respect_robots is True
    assert direct.extra_headers == {"X-Test": "7"}
    assert direct.schema_rules == {"title": {"selector": "h1"}}
    assert all(mapping.items_calls == 0 for mapping in (rules, domains, rule, headers, settings))


def test_dict_subclass_items_override_cannot_substitute_canonical_values():
    rule = _ItemsSubstitutionDict(
        {"backend": "curl", "respect_robots": True},
        {"backend": "playwright", "respect_robots": False},
    )

    direct, validated = _resolve_both(rule)

    assert direct == validated
    assert direct.backend == "curl"
    assert direct.respect_robots is True
    assert rule.items_calls == 0


@pytest.mark.parametrize("mapping_type", [dict, UserDict], ids=["dict", "userdict"])
def test_standard_mapping_types_preserve_canonical_values(mapping_type):
    rule = mapping_type(
        {
            "backend": "curl",
            "respect_robots": False,
            "extra_headers": mapping_type({"X-Test": 7}),
            "schema": mapping_type({"title": {"selector": "h1"}}),
        }
    )

    direct, validated = _resolve_both(rule)

    assert direct.backend == "curl"
    assert direct.respect_robots is False
    assert validated.backend == "curl"
    assert validated.respect_robots is False
    expected_headers = {"X-Test": 7} if mapping_type is dict else {"X-Test": "7"}
    assert direct.extra_headers == expected_headers
    assert validated.extra_headers == {"X-Test": "7"}
    assert direct.schema_rules == {"title": {"selector": "h1"}}
    assert validated.schema_rules == {"title": {"selector": "h1"}}


def test_same_hash_domains_key_survives_pair_backed_mapping_without_hooks():
    bomb = _EqBomb("domains")
    rules = _PairMapping(
        [
            (bomb, {"example.com": {"backend": "playwright"}}),
            ("domains", {"example.com": {"backend": "curl"}}),
        ]
    )

    cleaned = ScraperRouter.validate_rules(rules)
    direct = ScraperRouter(rules).resolve("https://example.com/path")
    validated = ScraperRouter(cleaned).resolve("https://example.com/path")

    assert direct == validated
    assert direct.backend == "curl"
    assert bomb.eq_calls == 0
    assert bomb.repr_calls == 0


def test_same_hash_backend_key_survives_direct_and_validated_rule_snapshots():
    bomb = _EqBomb("backend")
    rule = _PairMapping(
        [
            (bomb, "playwright"),
            ("backend", "curl"),
        ]
    )

    direct, validated = _resolve_both(rule)

    assert direct == validated
    assert direct.backend == "curl"
    assert bomb.eq_calls == 0
    assert bomb.repr_calls == 0


def test_same_hash_header_key_survives_scalar_map_snapshot():
    bomb = _EqBomb("X-Test")
    mixed = _PairMapping(
        [
            (bomb, "eqbomb-router-secret"),
            ("X-Test", 7),
        ]
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
    assert bomb.eq_calls == 0
    assert bomb.repr_calls == 0


def test_same_hash_settings_key_survives_schema_mapping_snapshot():
    bomb = _EqBomb("title")
    settings = _PairMapping(
        [
            (bomb, {"selector": "script"}),
            ("title", {"selector": "h1"}),
        ]
    )

    direct, validated = _resolve_both({"schema": settings})

    assert direct == validated
    assert direct.schema_rules == {"title": {"selector": "h1"}}
    assert bomb.eq_calls == 0
    assert bomb.repr_calls == 0


def test_mapping_get_hook_is_not_used_after_snapshot():
    rule = _HookTrapMapping({"backend": "curl", "extra_headers": {"X-Test": 1}})
    rules = _HookTrapMapping({"domains": _HookTrapMapping({"example.com": rule})})

    cleaned = ScraperRouter.validate_rules(rules)
    direct = ScraperRouter(rules).resolve("https://example.com/path")
    validated = ScraperRouter(cleaned).resolve("https://example.com/path")

    assert direct.backend == "curl"
    assert validated.backend == "curl"
    assert direct.extra_headers == {"X-Test": 1}
    assert validated.extra_headers == {"X-Test": "1"}


def test_nested_mapping_items_are_snapshotted_once():
    rules = {"domains": {"example.com": _HookTrapMapping({"backend": "curl", "extra_headers": {"X-Test": 1}})}}

    cleaned = ScraperRouter.validate_rules(rules)
    direct = ScraperRouter(rules).resolve("https://example.com/path")
    validated = ScraperRouter(cleaned).resolve("https://example.com/path")

    assert direct.backend == "curl"
    assert validated.backend == "curl"
    assert direct.extra_headers == {"X-Test": 1}
    assert validated.extra_headers == {"X-Test": "1"}


def test_lying_mapping_get_cannot_widen_snapshotted_rules():
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
        _ItemsIterationTrapMapping({"domains": {"example.com": {"backend": "curl"}}}),
        _ItemsAcquisitionTrapMapping({"domains": {"example.com": {"backend": "curl"}}}),
        _ItemsUnpackingTrapMapping({"domains": {"example.com": {"backend": "curl"}}}),
    ],
    ids=["items-iteration", "items-acquisition", "items-unpacking"],
)
def test_overridden_items_failures_do_not_replace_canonical_config_state(rules):
    cleaned = ScraperRouter.validate_rules(rules)
    direct = ScraperRouter(rules).resolve("https://example.com/path")
    validated = ScraperRouter(cleaned).resolve("https://example.com/path")

    assert direct == validated
    assert direct.backend == "curl"


@pytest.mark.parametrize(
    "mixed",
    [
        _ItemsIterationTrapMapping({"X-Test": 7}),
        _ItemsAcquisitionTrapMapping({"X-Test": 7}),
        _ItemsUnpackingTrapMapping({"X-Test": 7}),
    ],
    ids=["items-iteration", "items-acquisition", "items-unpacking"],
)
def test_overridden_items_failures_do_not_replace_canonical_scalar_map_state(mixed):
    direct, validated = _resolve_both({"extra_headers": mixed})

    assert direct == validated
    assert direct.extra_headers == {"X-Test": "7"}


@pytest.mark.parametrize(
    "rules",
    [
        _CanonicalIterationTrapMapping(
            {
                "domains": {"example.com": {"backend": "curl"}},
                "ignored": True,
            }
        ),
        _CanonicalLookupTrapMapping(
            {
                "domains": {"example.com": {"backend": "curl"}},
                "failure": True,
            },
            "failure",
        ),
    ],
    ids=["iteration", "lookup"],
)
def test_canonical_config_snapshot_failures_discard_partial_state(rules):
    assert ScraperRouter.validate_rules(rules) == {"domains": {}}

    plan = ScraperRouter(rules).resolve("https://example.com/path")

    assert plan.backend == "auto"
    assert plan.handler == DEFAULT_HANDLER


@pytest.mark.parametrize(
    "mixed",
    [
        _CanonicalIterationTrapMapping({"X-Test": 7, "X-Ignored": 8}),
        _CanonicalLookupTrapMapping({"X-Test": 7, "X-Failure": 8}, "X-Failure"),
    ],
    ids=["iteration", "lookup"],
)
def test_canonical_scalar_map_snapshot_failures_discard_partial_state(mixed):
    direct, validated = _resolve_both({"extra_headers": mixed})

    assert direct == validated
    assert direct.extra_headers == {}


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


def test_validated_rule_applies_match_at_url_pattern_position_33():
    patterns = [rf"/never-{index}$" for index in range(32)] + [r"/target$"]
    rules = {
        "domains": {
            "example.com": {
                "backend": "curl",
                "url_patterns": patterns,
            }
        }
    }

    cleaned = ScraperRouter.validate_rules(rules)
    plan = ScraperRouter(cleaned).resolve("https://example.com/target")

    assert cleaned["domains"]["example.com"]["url_patterns"] == patterns
    assert plan.backend == "curl"


def test_validation_checks_later_pattern_after_prior_search_time(
    monkeypatch: pytest.MonkeyPatch,
):
    state = {"now": 0.0}
    calls: list[tuple[str, float]] = []

    def fake_monotonic() -> float:
        return state["now"]

    def fake_search(pattern, _value, *, limits):
        calls.append((pattern, limits.timeout_s))
        state["now"] += min(0.040, limits.timeout_s)
        code = "regex_invalid" if pattern.startswith("rejected") else None
        return SafeRegexResult(matched=False, code=code)

    monkeypatch.setattr(scraper_router_module, "_monotonic", fake_monotonic, raising=False)
    monkeypatch.setattr(scraper_router_module, "search_untrusted", fake_search)
    patterns = ["rejected-1", "rejected-2", "rejected-3", "survivor"]

    cleaned = ScraperRouter.validate_rules({"domains": {"example.com": {"backend": "curl", "url_patterns": patterns}}})

    assert cleaned["domains"]["example.com"]["url_patterns"] == ["survivor"]
    assert calls == [(pattern, 0.100) for pattern in patterns]


def test_malformed_builtin_rule_values_preserve_predecessor_validation_output():
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
    assert normalized["handler"] == 123
    assert normalized["extra_headers"] == {}
    assert normalized["cookies"] == {}
    assert normalized["proxies"] == {}
    assert normalized["ua_profile"] == ["firefox_120_win"]
    assert normalized["impersonate"] == {"value": "firefox120"}
    assert normalized["strategy_order"] == ["schema", "llm"]
    assert normalized["schema_rules"] == {}
    assert normalized["schema"] == {"title": {"selector": "h1"}}
    assert normalized["llm_settings"] == {}
    assert normalized["llm"] == {"provider": "openai"}
    assert normalized["regex_settings"] == {}
    assert normalized["regex"] == {"mask_pii": True}
    assert normalized["cluster_settings"] == {}
    assert normalized["cluster"] == {"cluster_linkage": "complete"}


def test_malformed_rule_values_raise_predecessor_handler_error_first():
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

    rules = {"domains": {"example.com": rule}}
    with pytest.raises(AttributeError):
        ScraperRouter(rules).resolve("https://example.com/path")

    cleaned = ScraperRouter.validate_rules(rules)
    with pytest.raises(AttributeError):
        ScraperRouter(cleaned).resolve("https://example.com/path")


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
    assert direct.backend == " CURL "
    assert validated.backend == "curl"
    assert type(direct.backend) is str
    assert type(validated.backend) is str
    assert direct.extra_headers == expected
    assert validated.extra_headers == expected
    assert direct.cookies == expected
    assert validated.cookies == expected
    assert direct.proxies == expected
    assert validated.proxies == expected
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


@pytest.mark.parametrize("malformed", [_BoolTrap(), _FalsyBoolTrap(), object()])
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
    [
        (False, False),
        (True, True),
        (0, False),
        (1, True),
        ("", False),
        ("false", True),
        ([], False),
        ({}, False),
        ([1], True),
        ({"value": 1}, True),
    ],
)
def test_safe_builtin_robots_values_preserve_boolean_compatibility(value, expected):
    direct, validated = _resolve_both({"respect_robots": value})

    assert direct == validated
    assert direct.respect_robots is expected


def test_settings_aliases_fail_over_after_safe_mapping_snapshots():
    nested = UserDict({"selector": "h1"})
    invalid_primary = _CanonicalIterationTrapMapping({"ignored": True})
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


def test_valid_scalar_ua_values_preserve_predecessor_path_specific_results():
    direct, validated = _resolve_both(
        {
            "backend": "PLAYWRIGHT",
            "handler": DEFAULT_HANDLER,
            "ua_profile": 123,
            "impersonate": 456,
        }
    )

    assert direct.backend == "PLAYWRIGHT"
    assert validated.backend == "playwright"
    assert direct.handler == DEFAULT_HANDLER
    assert validated.handler == DEFAULT_HANDLER
    assert direct.ua_profile == "123"
    assert validated.ua_profile == "123"
    assert direct.impersonate == 456
    assert validated.impersonate == 456


@pytest.mark.skipif(
    not _INT_DIGIT_LIMIT_ACTIVE,
    reason="interpreter does not enforce a decimal digit limit for int-to-string conversion",
)
@pytest.mark.parametrize("bad_setting", ["ua_profile", "impersonate"])
def test_validated_router_drops_only_digit_limited_scalar_values(bad_setting):
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

    cleaned = ScraperRouter.validate_rules({"domains": {"example.com": rule}})
    plan = ScraperRouter(cleaned).resolve("https://example.com/path")

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
