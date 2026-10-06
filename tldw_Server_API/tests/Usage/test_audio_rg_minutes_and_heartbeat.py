import asyncio

import pytest

from tldw_Server_API.app.core.Usage import audio_quota


@pytest.mark.asyncio
async def test_check_daily_minutes_prefers_ledger(monkeypatch):
    # Ensure legacy usage path is not invoked when ledger is available
    called_legacy = {"used": False}

    async def _fake_limits(user_id: int):
        return {"daily_minutes": 10}

    async def _fake_ledger_remaining(user_id: int, daily_limit_minutes: float):
        return 2.0  # minutes remaining

    async def _fake_get_daily_minutes_used(user_id: int):
        called_legacy["used"] = True
        return 0.0

    monkeypatch.setattr(audio_quota, "get_limits_for_user", _fake_limits)
    monkeypatch.setattr(audio_quota, "_ledger_remaining_minutes", _fake_ledger_remaining)
    monkeypatch.setattr(audio_quota, "get_daily_minutes_used", _fake_get_daily_minutes_used)

    allowed, remaining = await audio_quota.check_daily_minutes_allow(user_id=1, minutes_requested=3.0)

    assert allowed is False
    assert remaining == 2.0
    assert called_legacy["used"] is False  # ledger path short-circuited legacy usage


@pytest.mark.asyncio
async def test_add_daily_minutes_records_ledger_even_on_legacy_failure(monkeypatch):
    fake_ledger_entries = []

    class _FakeLedger:
        async def initialize(self):
            return None

        async def add(self, entry):
            fake_ledger_entries.append(entry)
            return True

    async def _fake_get_daily_ledger():
        return _FakeLedger()

    class _Pool:
        pool = None

        async def execute(self, *args, **kwargs):
            raise RuntimeError("legacy write failed")

    async def _fake_get_db_pool():
        return _Pool()

    monkeypatch.setattr(audio_quota, "_get_daily_ledger", _fake_get_daily_ledger)
    monkeypatch.setattr(audio_quota, "get_db_pool", _fake_get_db_pool)

    await audio_quota.add_daily_minutes(user_id=7, minutes=1.5)

    assert len(fake_ledger_entries) == 1
    assert fake_ledger_entries[0].units == int(round(1.5 * 60))
