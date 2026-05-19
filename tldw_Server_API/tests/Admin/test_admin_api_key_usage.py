"""Tests for per-API-key usage attribution in admin_system_ops_service."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def _isolate_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect the system ops JSON store to a temporary directory."""
    store_path = tmp_path / "system_ops.json"
    monkeypatch.setattr(
        "tldw_Server_API.app.services.admin_system_ops_service._STORE_PATH",
        store_path,
    )
    yield


# ---- Import after monkeypatch fixtures are declared ----
from tldw_Server_API.app.services.admin_system_ops_service import (
    get_api_key_usage,
    list_api_key_usage,
    record_api_key_usage,
)


class TestRecordApiKeyUsage:
    def test_first_record_creates_entry(self):
        result = record_api_key_usage(
            "key-1",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.005,
        )

        assert result["key_id"] == "key-1"
        assert result["request_count"] == 1
        assert result["total_tokens"] == 150
        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 50
        assert result["estimated_cost_usd"] == 0.005
        assert result["last_used_at"] is not None
        assert len(result["daily_snapshots"]) == 1

    def test_increments_existing_entry(self):
        record_api_key_usage("key-1", prompt_tokens=100, completion_tokens=50, cost_usd=0.005)
        result = record_api_key_usage("key-1", prompt_tokens=200, completion_tokens=100, cost_usd=0.01)

        assert result["request_count"] == 2
        assert result["total_tokens"] == 450
        assert result["prompt_tokens"] == 300
        assert result["completion_tokens"] == 150
        assert result["estimated_cost_usd"] == 0.015
        # Same day => single daily snapshot
        assert len(result["daily_snapshots"]) == 1
        assert result["daily_snapshots"][0]["requests"] == 2
        assert result["daily_snapshots"][0]["tokens"] == 450

    def test_separate_keys_tracked_independently(self):
        record_api_key_usage("key-1", prompt_tokens=100, completion_tokens=50, cost_usd=0.005)
        record_api_key_usage("key-2", prompt_tokens=200, completion_tokens=100, cost_usd=0.01)

        usage_1 = get_api_key_usage("key-1")
        usage_2 = get_api_key_usage("key-2")

        assert usage_1["total_tokens"] == 150
        assert usage_2["total_tokens"] == 300


class TestGetApiKeyUsage:
    def test_returns_zeroed_default_for_unknown_key(self):
        result = get_api_key_usage("nonexistent")

        assert result["key_id"] == "nonexistent"
        assert result["request_count"] == 0
        assert result["total_tokens"] == 0
        assert result["estimated_cost_usd"] == 0.0
        assert result["daily_snapshots"] == []

    def test_returns_recorded_data(self):
        record_api_key_usage("key-1", prompt_tokens=500, completion_tokens=250, cost_usd=0.05)
        result = get_api_key_usage("key-1")

        assert result["total_tokens"] == 750
        assert result["estimated_cost_usd"] == 0.05


class TestListApiKeyUsage:
    def test_returns_top_keys_by_token_count(self):
        record_api_key_usage("key-small", prompt_tokens=10, completion_tokens=5, cost_usd=0.001)
        record_api_key_usage("key-big", prompt_tokens=10000, completion_tokens=5000, cost_usd=1.0)
        record_api_key_usage("key-medium", prompt_tokens=500, completion_tokens=250, cost_usd=0.05)

        result = list_api_key_usage(limit=2)

        assert len(result) == 2
        assert result[0]["key_id"] == "key-big"
        assert result[1]["key_id"] == "key-medium"

    def test_returns_all_when_limit_exceeds_count(self):
        record_api_key_usage("key-1", prompt_tokens=100, completion_tokens=50, cost_usd=0.005)

        result = list_api_key_usage(limit=10)
        assert len(result) == 1

    def test_empty_when_no_data(self):
        result = list_api_key_usage(limit=10)
        assert result == []


class TestDailySnapshotCap:
    def test_caps_snapshots_at_90_days(self, monkeypatch):
        """Simulate 95 days of usage and verify only 90 snapshots are kept."""
        from datetime import datetime, timezone, timedelta

        base_date = datetime(2026, 1, 1, tzinfo=timezone.utc)

        for day_offset in range(95):
            fake_date = base_date + timedelta(days=day_offset)
            fake_now = fake_date.strftime("%Y-%m-%d")

            # Patch datetime.now to control the date
            with mock.patch(
                "tldw_Server_API.app.services.admin_system_ops_service.datetime"
            ) as mock_dt:
                mock_dt.now.return_value = fake_date
                mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
                # The strftime call is on the return value of datetime.now()
                record_api_key_usage("key-cap", prompt_tokens=10, completion_tokens=5, cost_usd=0.001)

        result = get_api_key_usage("key-cap")
        assert len(result["daily_snapshots"]) <= 90
