"""Unit tests for PolicyConditions and evaluate_conditions."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.policy_conditions import (
    DelegationCondition,
    PolicyConditions,
    evaluate_conditions,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 4, 7, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Empty / is_empty
# ---------------------------------------------------------------------------


def test_empty_conditions_pass():
    """Empty conditions (all None) should always evaluate to True."""
    cond = PolicyConditions()
    assert cond.is_empty()
    assert evaluate_conditions(cond, now=_NOW) is True


def test_non_empty_conditions_is_empty_false():
    cond = PolicyConditions(valid_from=_NOW)
    assert not cond.is_empty()


# ---------------------------------------------------------------------------
# Time window
# ---------------------------------------------------------------------------


def test_valid_time_window_passes():
    cond = PolicyConditions(
        valid_from=_NOW - timedelta(hours=1),
        valid_until=_NOW + timedelta(hours=1),
    )
    assert evaluate_conditions(cond, now=_NOW) is True


def test_expired_time_window_fails():
    cond = PolicyConditions(
        valid_from=_NOW - timedelta(hours=2),
        valid_until=_NOW - timedelta(hours=1),
    )
    assert evaluate_conditions(cond, now=_NOW) is False


def test_future_time_window_fails():
    cond = PolicyConditions(
        valid_from=_NOW + timedelta(hours=1),
        valid_until=_NOW + timedelta(hours=2),
    )
    assert evaluate_conditions(cond, now=_NOW) is False


def test_valid_from_only_passes_when_after():
    cond = PolicyConditions(valid_from=_NOW - timedelta(hours=1))
    assert evaluate_conditions(cond, now=_NOW) is True


def test_valid_from_only_fails_when_before():
    cond = PolicyConditions(valid_from=_NOW + timedelta(hours=1))
    assert evaluate_conditions(cond, now=_NOW) is False


def test_valid_until_only_passes_when_before():
    cond = PolicyConditions(valid_until=_NOW + timedelta(hours=1))
    assert evaluate_conditions(cond, now=_NOW) is True


def test_valid_until_only_fails_when_after():
    cond = PolicyConditions(valid_until=_NOW - timedelta(hours=1))
    assert evaluate_conditions(cond, now=_NOW) is False


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def test_label_match_passes():
    cond = PolicyConditions(required_labels={"env": "prod", "team": "alpha"})
    labels = {"env": "prod", "team": "alpha", "extra": "ignored"}
    assert evaluate_conditions(cond, resource_labels=labels, now=_NOW) is True


def test_label_mismatch_fails():
    cond = PolicyConditions(required_labels={"env": "prod"})
    labels = {"env": "staging"}
    assert evaluate_conditions(cond, resource_labels=labels, now=_NOW) is False


def test_label_missing_fails():
    cond = PolicyConditions(required_labels={"env": "prod"})
    assert evaluate_conditions(cond, resource_labels={}, now=_NOW) is False
    assert evaluate_conditions(cond, resource_labels=None, now=_NOW) is False


def test_required_labels_non_mapping_deserializes_to_none() -> None:
    cond = PolicyConditions.from_dict({"required_labels": ["not", "a", "mapping"]})
    assert cond.required_labels is None
    assert cond.is_empty()


# ---------------------------------------------------------------------------
# Source IPs
# ---------------------------------------------------------------------------


def test_source_ip_exact_match_passes() -> None:
    cond = PolicyConditions(source_ips=["10.0.0.7"])
    assert evaluate_conditions(cond, source_ip="10.0.0.7", now=_NOW) is True


def test_source_ip_cidr_match_passes() -> None:
    cond = PolicyConditions(source_ips=["10.0.0.0/24"])
    assert evaluate_conditions(cond, source_ip="10.0.0.42", now=_NOW) is True


def test_source_ip_missing_context_fails() -> None:
    cond = PolicyConditions(source_ips=["10.0.0.0/24"])
    assert evaluate_conditions(cond, source_ip=None, now=_NOW) is False


def test_source_ip_outside_allowlist_fails() -> None:
    cond = PolicyConditions(source_ips=["10.0.0.0/24"])
    assert evaluate_conditions(cond, source_ip="192.168.1.10", now=_NOW) is False


# ---------------------------------------------------------------------------
# Delegation
# ---------------------------------------------------------------------------


def test_delegation_match_passes():
    cond = PolicyConditions(
        delegation=DelegationCondition(principal_type="user", principal_id="u-42"),
    )
    assert evaluate_conditions(cond, ancestry_chain=["u-42", "u-1"], now=_NOW) is True


def test_delegation_mismatch_fails():
    cond = PolicyConditions(
        delegation=DelegationCondition(principal_type="agent", principal_id="a-99"),
    )
    assert evaluate_conditions(cond, ancestry_chain=["u-42", "u-1"], now=_NOW) is False


def test_delegation_empty_ancestry_fails():
    cond = PolicyConditions(
        delegation=DelegationCondition(principal_type="user", principal_id="u-42"),
    )
    assert evaluate_conditions(cond, ancestry_chain=[], now=_NOW) is False
    assert evaluate_conditions(cond, ancestry_chain=None, now=_NOW) is False


# ---------------------------------------------------------------------------
# Combined conditions
# ---------------------------------------------------------------------------


def test_combined_conditions_all_pass():
    cond = PolicyConditions(
        valid_from=_NOW - timedelta(hours=1),
        valid_until=_NOW + timedelta(hours=1),
        required_labels={"env": "prod"},
        delegation=DelegationCondition(principal_type="user", principal_id="u-42"),
    )
    assert evaluate_conditions(
        cond,
        resource_labels={"env": "prod"},
        ancestry_chain=["u-42"],
        now=_NOW,
    ) is True


def test_combined_conditions_time_fails():
    """Time window expired, even though labels and delegation match."""
    cond = PolicyConditions(
        valid_from=_NOW - timedelta(hours=2),
        valid_until=_NOW - timedelta(hours=1),
        required_labels={"env": "prod"},
        delegation=DelegationCondition(principal_type="user", principal_id="u-42"),
    )
    assert evaluate_conditions(
        cond,
        resource_labels={"env": "prod"},
        ancestry_chain=["u-42"],
        now=_NOW,
    ) is False


def test_combined_conditions_label_fails():
    """Labels mismatch, even though time and delegation match."""
    cond = PolicyConditions(
        valid_from=_NOW - timedelta(hours=1),
        valid_until=_NOW + timedelta(hours=1),
        required_labels={"env": "prod"},
        delegation=DelegationCondition(principal_type="user", principal_id="u-42"),
    )
    assert evaluate_conditions(
        cond,
        resource_labels={"env": "staging"},
        ancestry_chain=["u-42"],
        now=_NOW,
    ) is False


# ---------------------------------------------------------------------------
# Serialization roundtrips
# ---------------------------------------------------------------------------


def test_to_dict_from_dict_roundtrip():
    cond = PolicyConditions(
        valid_from=_NOW,
        valid_until=_NOW + timedelta(hours=2),
        source_ips=["10.0.0.1"],
        required_labels={"env": "prod"},
        delegation=DelegationCondition(principal_type="agent", principal_id="a-7"),
    )
    d = cond.to_dict()
    restored = PolicyConditions.from_dict(d)
    assert restored.valid_from == cond.valid_from
    assert restored.valid_until == cond.valid_until
    assert restored.source_ips == cond.source_ips
    assert restored.required_labels == cond.required_labels
    assert restored.delegation is not None
    assert restored.delegation.principal_type == "agent"
    assert restored.delegation.principal_id == "a-7"


def test_to_json_from_json_roundtrip():
    cond = PolicyConditions(
        valid_from=_NOW,
        required_labels={"tier": "premium"},
    )
    raw = cond.to_json()
    restored = PolicyConditions.from_json(raw)
    assert restored.valid_from == cond.valid_from
    assert restored.required_labels == cond.required_labels
    assert restored.delegation is None


def test_from_dict_none_yields_empty():
    assert PolicyConditions.from_dict(None).is_empty()


def test_from_dict_empty_dict_yields_empty():
    assert PolicyConditions.from_dict({}).is_empty()


def test_from_json_none_yields_empty():
    assert PolicyConditions.from_json(None).is_empty()


def test_from_json_invalid_yields_empty():
    assert PolicyConditions.from_json("not-json!!").is_empty()


def test_is_empty_source_ips_only():
    """source_ips alone makes it non-empty (even though not evaluated)."""
    cond = PolicyConditions(source_ips=["192.168.1.1"])
    assert not cond.is_empty()


def test_delegation_from_dict_missing_fields():
    """DelegationCondition.from_dict returns None when required fields are missing."""
    assert DelegationCondition.from_dict(None) is None
    assert DelegationCondition.from_dict({}) is None
    assert DelegationCondition.from_dict({"principal_type": "user"}) is None
    assert DelegationCondition.from_dict({"principal_id": "u-1"}) is None


def test_empty_conditions_to_dict_is_empty_dict():
    cond = PolicyConditions()
    assert cond.to_dict() == {}
