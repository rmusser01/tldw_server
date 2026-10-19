"""Scoped cold-process state for real audio-quota migration tests."""

import pytest


@pytest.fixture
def cold_audio_ledger(monkeypatch):
    """Reset real ledger caches on demand and restore prior state at teardown."""
    from tldw_Server_API.app.core.Usage import audio_quota, quota_checks

    def reset():
        monkeypatch.setattr(audio_quota, "_daily_ledger", None)
        monkeypatch.setattr(audio_quota, "_audio_minutes_legacy_backfill_done", False)
        quota_checks.reset_ledger_cache()

    yield reset
    quota_checks.reset_ledger_cache()
