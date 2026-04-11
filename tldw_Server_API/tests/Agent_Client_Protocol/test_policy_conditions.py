"""Tests for PolicyConditions and evaluate_conditions."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.policy_conditions import (
    DelegationCondition,
    PolicyConditions,
    evaluate_conditions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime(2026, 4, 7, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# 1. Empty conditions → passes
# ---------------------------------------------------------------------------

def test_empty_conditions_pass():
    cond = PolicyConditions()
    assert evaluate_conditions(cond, now=_now()) is True


# ---------------------------------------------------------------------------
# 2. Valid time window → passes
# ---------------------------------------------------------------------------

def test_valid_time_window_passes():
    cond = PolicyConditions(
        valid_from=_now() - timedelta(hours=1),
        valid_until=_now() + timedelta(hours=1),
    )
    assert evaluate_conditions(cond, now=_now()) is True


# ---------------------------------------------------------------------------
# 3. Expired time window → fails
# ---------------------------------------------------------------------------

def test_expired_time_window_fails():
    cond = PolicyConditions(
        valid_from=_now() - timedelta(hours=2),
        valid_until=_now() - timedelta(hours=1),
    )
    assert evaluate_conditions(cond, now=_now()) is False


# ---------------------------------------------------------------------------
# 4. Future time window → fails
# ---------------------------------------------------------------------------

def test_future_time_window_fails():
    cond = PolicyConditions(
        valid_from=_now() + timedelta(hours=1),
        valid_until=_now() + timedelta(hours=2),
    )
    assert evaluate_conditions(cond, now=_now()) is False


# ---------------------------------------------------------------------------
# 5. Label match → passes
# ---------------------------------------------------------------------------

def test_label_match_passes():
    cond = PolicyConditions(required_labels={"env": "prod"})
    assert evaluate_conditions(
        cond, resource_labels={"env": "prod", "team": "sre"}, now=_now()
    ) is True


# ---------------------------------------------------------------------------
# 6. Label mismatch → fails
# ---------------------------------------------------------------------------

def test_label_mismatch_fails():
    cond = PolicyConditions(required_labels={"env": "prod"})
    assert evaluate_conditions(
        cond, resource_labels={"env": "staging"}, now=_now()
    ) is False


# ---------------------------------------------------------------------------
# 7. Missing label → fails
# ---------------------------------------------------------------------------

def test_missing_label_fails():
    cond = PolicyConditions(required_labels={"env": "prod"})
    assert evaluate_conditions(
        cond, resource_labels={}, now=_now()
    ) is False


# ---------------------------------------------------------------------------
# 8. Delegation with matching ancestry → passes
# ---------------------------------------------------------------------------

def test_delegation_matching_ancestry_passes():
    cond = PolicyConditions(
        delegation=DelegationCondition(principal_type="user", principal_id="user-42"),
    )
    assert evaluate_conditions(
        cond, ancestry_chain=["user-42", "agent-1", "agent-2"], now=_now()
    ) is True


# ---------------------------------------------------------------------------
# 9. Delegation with non-matching ancestry → fails
# ---------------------------------------------------------------------------

def test_delegation_non_matching_ancestry_fails():
    cond = PolicyConditions(
        delegation=DelegationCondition(principal_type="user", principal_id="user-99"),
    )
    assert evaluate_conditions(
        cond, ancestry_chain=["user-42", "agent-1"], now=_now()
    ) is False


# ---------------------------------------------------------------------------
# 10. Delegation with empty ancestry → fails
# ---------------------------------------------------------------------------

def test_delegation_empty_ancestry_fails():
    cond = PolicyConditions(
        delegation=DelegationCondition(principal_type="user", principal_id="user-42"),
    )
    assert evaluate_conditions(cond, ancestry_chain=[], now=_now()) is False
    assert evaluate_conditions(cond, ancestry_chain=None, now=_now()) is False


# ---------------------------------------------------------------------------
# 11. Combined conditions (time + labels) → both must pass
# ---------------------------------------------------------------------------

def test_combined_time_and_labels_both_must_pass():
    cond = PolicyConditions(
        valid_from=_now() - timedelta(hours=1),
        valid_until=_now() + timedelta(hours=1),
        required_labels={"env": "prod"},
    )
    # Both pass
    assert evaluate_conditions(
        cond, resource_labels={"env": "prod"}, now=_now()
    ) is True

    # Time passes, labels fail
    assert evaluate_conditions(
        cond, resource_labels={"env": "staging"}, now=_now()
    ) is False

    # Labels pass, time fails (expired)
    cond_expired = PolicyConditions(
        valid_from=_now() - timedelta(hours=2),
        valid_until=_now() - timedelta(hours=1),
        required_labels={"env": "prod"},
    )
    assert evaluate_conditions(
        cond_expired, resource_labels={"env": "prod"}, now=_now()
    ) is False


# ---------------------------------------------------------------------------
# 12. PolicyConditions.to_dict() → from_dict() roundtrip
# ---------------------------------------------------------------------------

def test_policy_conditions_dict_roundtrip():
    cond = PolicyConditions(
        valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
        valid_until=datetime(2026, 12, 31, tzinfo=timezone.utc),
        source_ips=["10.0.0.0/8"],
        required_labels={"env": "prod", "tier": "1"},
        delegation=DelegationCondition(principal_type="agent", principal_id="agent-7"),
    )
    d = cond.to_dict()
    restored = PolicyConditions.from_dict(d)

    assert restored.valid_from == cond.valid_from
    assert restored.valid_until == cond.valid_until
    assert restored.source_ips == cond.source_ips
    assert restored.required_labels == cond.required_labels
    assert restored.delegation is not None
    assert restored.delegation.principal_type == "agent"
    assert restored.delegation.principal_id == "agent-7"


# ---------------------------------------------------------------------------
# 13. PolicyConditions.to_json() → from_json() roundtrip
# ---------------------------------------------------------------------------

def test_policy_conditions_json_roundtrip():
    cond = PolicyConditions(
        valid_from=datetime(2026, 6, 15, 8, 0, 0, tzinfo=timezone.utc),
        required_labels={"team": "platform"},
    )
    j = cond.to_json()
    restored = PolicyConditions.from_json(j)

    assert restored.valid_from == cond.valid_from
    assert restored.required_labels == cond.required_labels
    assert restored.delegation is None


def test_from_json_with_none_or_empty():
    assert PolicyConditions.from_json(None).is_empty()
    assert PolicyConditions.from_json("").is_empty()


def test_from_dict_with_none():
    assert PolicyConditions.from_dict(None).is_empty()


# ---------------------------------------------------------------------------
# 14. DelegationCondition.to_dict() → from_dict() roundtrip
# ---------------------------------------------------------------------------

def test_delegation_condition_dict_roundtrip():
    dc = DelegationCondition(principal_type="agent", principal_id="agent-99")
    d = dc.to_dict()
    restored = DelegationCondition.from_dict(d)

    assert restored is not None
    assert restored.principal_type == "agent"
    assert restored.principal_id == "agent-99"


def test_delegation_from_dict_none():
    assert DelegationCondition.from_dict(None) is None
    assert DelegationCondition.from_dict({}) is None


# ---------------------------------------------------------------------------
# 15. is_empty()
# ---------------------------------------------------------------------------

def test_is_empty_default():
    assert PolicyConditions().is_empty() is True


def test_is_empty_false_when_valid_from_set():
    assert PolicyConditions(valid_from=_now()).is_empty() is False


def test_is_empty_false_when_valid_until_set():
    assert PolicyConditions(valid_until=_now()).is_empty() is False


def test_is_empty_false_when_source_ips_set():
    assert PolicyConditions(source_ips=["10.0.0.0/8"]).is_empty() is False


def test_is_empty_false_when_required_labels_set():
    assert PolicyConditions(required_labels={"k": "v"}).is_empty() is False


def test_is_empty_false_when_delegation_set():
    assert PolicyConditions(
        delegation=DelegationCondition(principal_id="u1")
    ).is_empty() is False
