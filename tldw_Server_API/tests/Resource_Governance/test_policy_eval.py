"""Shared policy evaluation: the decisions both governor backends must agree on."""

from collections.abc import Callable, Mapping
from typing import Any

import pytest

from tldw_Server_API.app.core.Resource_Governance import policy_eval
from tldw_Server_API.app.core.Resource_Governance.policy_eval import (
    BUILTIN_DEFAULT_POLICY,
    clamp_token_units,
    effective_policy,
    scope_pairs,
)

pytestmark = [pytest.mark.unit, pytest.mark.rate_limit]


def _getter(policies: Mapping[str, Any]) -> Callable[[str], Any]:
    return lambda pid: policies.get(pid)


def test_known_policy_is_returned_unchanged():
    pol = {"requests": {"rpm": 5}, "scopes": ["user"]}
    assert effective_policy(_getter({"p": pol}), "p") == pol


def test_unknown_policy_falls_back_to_default():
    default = {"requests": {"rpm": 7}, "scopes": ["user"]}
    assert effective_policy(_getter({"default": default}), "typo.policy") == default


def test_missing_default_falls_back_to_builtin():
    assert effective_policy(_getter({}), "anything") == BUILTIN_DEFAULT_POLICY


def test_policy_without_requests_inherits_default_requests():
    default = {"requests": {"rpm": 9, "burst": 2.0}}
    pol = {"tokens": {"per_min": 100}, "scopes": ["user"]}
    out = effective_policy(_getter({"p": pol, "default": default}), "p")
    assert out["requests"] == {"rpm": 9, "burst": 2.0}
    assert out["tokens"] == {"per_min": 100}


def test_fractional_rpm_capacity_is_raised_to_one_unit():
    src = {"requests": {"rpm": 0.5, "burst": 1.0}, "scopes": ["user"]}
    out = effective_policy(_getter({"p": src}), "p")
    assert out["requests"] == {"rpm": 0.5, "burst": 2.0}
    assert src["requests"] == {"rpm": 0.5, "burst": 1.0}  # the store's dict is not mutated


def test_fractional_rpm_with_enough_burst_is_unchanged():
    pol = {"requests": {"rpm": 0.3, "burst": 10.0}}
    assert effective_policy(_getter({"p": pol}), "p")["requests"] == {"rpm": 0.3, "burst": 10.0}


def test_getter_errors_are_treated_as_unknown():
    def boom(_pid):
        raise RuntimeError("store down")

    assert effective_policy(boom, "p") == BUILTIN_DEFAULT_POLICY


def test_unknown_policy_is_logged_once(monkeypatch):
    from loguru import logger

    monkeypatch.setattr(policy_eval, "_warned_unknown", set())
    seen = []
    sink = logger.add(lambda m: seen.append(str(m)), level="ERROR")
    try:
        getter = _getter({"default": {"requests": {"rpm": 1}}})
        effective_policy(getter, "typo")
        effective_policy(getter, "typo")
    finally:
        logger.remove(sink)
    assert len([m for m in seen if "'typo'" in m]) == 1


def test_scope_pairs_always_include_the_entity_bucket():
    assert scope_pairs({"scopes": ["user", "api_key"]}, "ip", "1.2.3.4") == [("ip", "1.2.3.4")]


def test_scope_pairs_add_global_only_when_listed():
    assert scope_pairs({"scopes": ["global", "user"]}, "user", "1") == [("global", "*"), ("user", "1")]
    assert scope_pairs({"scopes": ["user"]}, "user", "1") == [("user", "1")]


def test_scope_pairs_default_scopes_are_global_plus_entity():
    assert scope_pairs({}, "user", "1") == [("global", "*"), ("user", "1")]


def test_clamp_caps_oversized_token_reservation_at_capacity():
    pol = {"tokens": {"per_min": 100, "burst": 1.5}}
    cats = {"tokens": {"units": 1000}, "requests": {"units": 1}}
    assert clamp_token_units(pol, cats, capacity_includes_burst=True)["tokens"]["units"] == 150
    assert clamp_token_units(pol, cats, capacity_includes_burst=False)["tokens"]["units"] == 100


def test_clamp_leaves_fitting_and_unbounded_reservations_alone():
    cats = {"tokens": {"units": 50}}
    assert clamp_token_units({"tokens": {"per_min": 100}}, cats, capacity_includes_burst=True) == cats
    assert clamp_token_units({}, {"tokens": {"units": 10**9}}, capacity_includes_burst=True) == {"tokens": {"units": 10**9}}


def test_malformed_store_value_is_treated_as_unknown():
    assert effective_policy(lambda pid: "oops", "p") == BUILTIN_DEFAULT_POLICY


def test_store_lookup_error_is_logged_once_and_falls_back(monkeypatch):
    from loguru import logger

    monkeypatch.setattr(policy_eval, "_warned_lookup_errors", set(), raising=False)
    monkeypatch.setattr(policy_eval, "_warned_unknown", set())

    def boom(_pid: str) -> None:
        raise RuntimeError("store down")

    seen: list[str] = []
    sink = logger.add(lambda m: seen.append(str(m)), level="ERROR")
    try:
        results = [effective_policy(boom, "p"), effective_policy(boom, "p")]
    finally:
        logger.remove(sink)
    assert results == [BUILTIN_DEFAULT_POLICY, BUILTIN_DEFAULT_POLICY]
    lookup_errors = [m for m in seen if "'p'" in m and "RuntimeError" in m]
    assert len(lookup_errors) == 1, seen
