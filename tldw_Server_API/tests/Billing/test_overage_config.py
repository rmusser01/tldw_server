"""Tests for overage handling configuration."""
from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Billing.overage_config import OveragePolicy


class TestOveragePolicyFromEnv:
    def test_defaults(self):
        policy = OveragePolicy.from_env()
        assert policy.mode == "notify_only"
        assert policy.grace_percentage == 10.0
        assert policy.notification_threshold == 80.0

    def test_custom_env(self, monkeypatch):
        monkeypatch.setenv("BILLING_OVERAGE_MODE", "hard_block")
        monkeypatch.setenv("BILLING_OVERAGE_GRACE_PCT", "5")
        monkeypatch.setenv("BILLING_OVERAGE_NOTIFY_PCT", "90")
        policy = OveragePolicy.from_env()
        assert policy.mode == "hard_block"
        assert policy.grace_percentage == 5.0
        assert policy.notification_threshold == 90.0

    def test_invalid_env_values_fall_back_to_safe_defaults(self, monkeypatch):
        monkeypatch.setenv("BILLING_OVERAGE_MODE", "hard-block")
        monkeypatch.setenv("BILLING_OVERAGE_GRACE_PCT", "-5")
        monkeypatch.setenv("BILLING_OVERAGE_NOTIFY_PCT", "not-a-number")

        policy = OveragePolicy.from_env()

        assert policy.mode == "notify_only"
        assert policy.grace_percentage == 10.0
        assert policy.notification_threshold == 80.0


class TestShouldBlock:
    def test_hard_block_under_grace(self):
        policy = OveragePolicy(mode="hard_block", grace_percentage=10, notification_threshold=80)
        assert policy.should_block(105) is False

    def test_hard_block_over_grace(self):
        policy = OveragePolicy(mode="hard_block", grace_percentage=10, notification_threshold=80)
        assert policy.should_block(111) is True

    def test_notify_only_never_blocks(self):
        policy = OveragePolicy(mode="notify_only", grace_percentage=10, notification_threshold=80)
        assert policy.should_block(200) is False

    def test_degraded_never_blocks(self):
        policy = OveragePolicy(mode="degraded", grace_percentage=10, notification_threshold=80)
        assert policy.should_block(200) is False


class TestShouldDegrade:
    def test_degraded_mode_over_grace(self):
        policy = OveragePolicy(mode="degraded", grace_percentage=10, notification_threshold=80)
        assert policy.should_degrade(111) is True

    def test_degraded_mode_under_grace(self):
        policy = OveragePolicy(mode="degraded", grace_percentage=10, notification_threshold=80)
        assert policy.should_degrade(105) is False

    def test_hard_block_never_degrades(self):
        policy = OveragePolicy(mode="hard_block", grace_percentage=10, notification_threshold=80)
        assert policy.should_degrade(200) is False


class TestShouldNotify:
    def test_above_threshold(self):
        policy = OveragePolicy(mode="notify_only", grace_percentage=10, notification_threshold=80)
        assert policy.should_notify(85) is True

    def test_at_threshold(self):
        policy = OveragePolicy(mode="notify_only", grace_percentage=10, notification_threshold=80)
        assert policy.should_notify(80) is True

    def test_below_threshold(self):
        policy = OveragePolicy(mode="notify_only", grace_percentage=10, notification_threshold=80)
        assert policy.should_notify(79) is False


class TestEvaluate:
    def test_evaluate_hard_block(self):
        policy = OveragePolicy(mode="hard_block", grace_percentage=10, notification_threshold=80)
        result = policy.evaluate(115)
        assert result["blocked"] is True
        assert result["degraded"] is False
        assert result["notify"] is True
        assert result["mode"] == "hard_block"
        assert result["usage_pct"] == 115

    def test_evaluate_notify_only(self):
        policy = OveragePolicy(mode="notify_only", grace_percentage=10, notification_threshold=80)
        result = policy.evaluate(50)
        assert result["blocked"] is False
        assert result["degraded"] is False
        assert result["notify"] is False
