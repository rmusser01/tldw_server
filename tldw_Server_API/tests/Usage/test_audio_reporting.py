"""Audio quota reporting reads the ledger and reports None as unlimited (spec 2 §9)."""

import io
import sqlite3
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
from tldw_Server_API.app.api.v1.endpoints.audio import audio_streaming
from tldw_Server_API.app.core.Usage import audio_quota

pytestmark = pytest.mark.unit

TEST_API_KEY = "test-api-key-1234567890"


@pytest.fixture()
def ledger(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Ledger seconds the test controls, per period."""
    used = {"today": 0.0, "month": 0.0}

    async def _today(entity_value: str, category: str) -> float:
        """Today's recorded seconds."""
        return used["today"] if category == "minutes" else 0.0

    async def _month(entity_value: str, category: str) -> float:
        """This month's recorded seconds."""
        return used["month"] if category == "minutes" else 0.0

    monkeypatch.setattr(audio_quota, "ledger_used_today", _today)
    monkeypatch.setattr(audio_quota, "ledger_used_this_month", _month)
    return used


@pytest.fixture()
def quotas_on_nothing_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Quotas on, with no limits.* value set for any user."""

    async def _unset(_user_id: int, _key: str) -> None:
        """Every limit is unset, so unlimited."""
        return None

    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    monkeypatch.setattr(audio_quota, "user_quota", _unset)


async def test_daily_and_monthly_minutes_come_from_the_ledger(ledger: dict) -> None:
    """Recorded seconds show up as minutes used today and this month."""
    ledger["today"], ledger["month"] = 600.0, 5400.0
    assert await audio_quota.get_daily_minutes_used(7) == 10.0
    assert await audio_quota.get_monthly_minutes_used(7) == 90.0


async def test_monthly_minutes_exhausted_only_with_a_limit(ledger: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """No monthly limit is never exhausted; a spent one is."""
    limits: dict = {"monthly_minutes": None}

    async def _limits(user_id: int) -> dict:
        """The test's limits."""
        return dict(limits)

    monkeypatch.setattr(audio_quota, "get_limits_for_user", _limits)
    ledger["month"] = 3600.0
    assert await audio_quota.monthly_minutes_exhausted(7) is False
    limits["monthly_minutes"] = 60
    assert await audio_quota.monthly_minutes_exhausted(7) is True
    limits["monthly_minutes"] = 61
    assert await audio_quota.monthly_minutes_exhausted(7) is False


def test_dead_rg_handle_machinery_is_gone() -> None:
    """Nothing tracks per-process audio RG handles any more."""
    for name in ("_rg_job_handles", "_rg_stream_handles", "_get_audio_rg_governor", "active_streams_count"):
        assert not hasattr(audio_quota, name), name


def _limits_app() -> FastAPI:
    """An app serving the real /audio/stream/limits route for user 1."""
    app = FastAPI()
    app.include_router(audio_streaming.router, prefix="/api/v1/audio")
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    return app


def test_stream_limits_report_ledger_minutes_and_null_when_unlimited(
    ledger: dict, quotas_on_nothing_set: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no limits set, the endpoint reports ledger usage and null remaining/active."""
    ledger["today"] = 600.0

    async def _tier(_user_id: int) -> str:
        """A fixed tier."""
        return "free"

    monkeypatch.setattr(audio_streaming, "_get_user_tier", _tier)
    with TestClient(_limits_app()) as client:
        data = client.get("/api/v1/audio/stream/limits").json()
    assert data["used_today_minutes"] == 10.0
    assert data["remaining_minutes"] is None
    assert data["remaining_month_minutes"] is None
    assert data["active_streams"] is None
    assert data["can_start_stream"] is True


def test_stream_limits_error_fallback_is_unlimited(ledger: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing limits lookup reports every limit as unlimited, not the old free-tier numbers."""

    async def _boom(_user_id: int) -> dict:
        """A limits lookup that fails with an expected DB error."""
        raise sqlite3.OperationalError("db down")

    async def _tier(_user_id: int) -> str:
        """A fixed tier."""
        return "free"

    monkeypatch.setattr(audio_streaming, "_get_limits_for_user", _boom)
    monkeypatch.setattr(audio_streaming, "_get_user_tier", _tier)
    with TestClient(_limits_app()) as client:
        data = client.get("/api/v1/audio/stream/limits").json()
    assert data["limits"]
    assert all(value is None for value in data["limits"].values())
    assert data["remaining_minutes"] is None


async def test_owner_processing_limit_none_when_unlimited(quotas_on_nothing_set: None) -> None:
    """The owner processing summary leaves an unset concurrent-jobs limit as null."""
    from tldw_Server_API.app.api.v1.endpoints.audio import audio_jobs

    class _Jobs:
        """A job manager counting two processing jobs."""

        def count_jobs(self, **_kwargs: object) -> int:
            """Two jobs are processing."""
            return 2

    summary = await audio_jobs.owner_processing_summary(123, _Jobs(), None)  # type: ignore[arg-type]
    assert summary.processing == 2
    assert summary.limit is None


def _wav_bytes() -> bytes:
    """A tiny valid WAV upload."""
    buf = io.BytesIO()
    sf.write(buf, np.zeros(1600, dtype=np.float32), 16000, format="WAV")
    return buf.getvalue()


@pytest.mark.parametrize(
    ("exhausted", "period"),
    [(True, "monthly"), (False, "daily")],
)
def test_monthly_breach_message_names_the_month(
    monkeypatch: pytest.MonkeyPatch, bypass_api_limits: Any, exhausted: bool, period: str
) -> None:
    """A minutes denial's 402 names the month when the monthly limit is spent, else the day."""
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", TEST_API_KEY)

    import tldw_Server_API.app.api.v1.endpoints.audio.audio as audio_ep
    import tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions as audio_tx
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter as stt_adapter
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

    async def _user() -> User:
        """The single-user principal."""
        return User(id=1, username="single_user")

    async def _allow(*_args: object, **_kwargs: object) -> tuple[bool, None]:
        """Admit the job."""
        return True, None

    async def _noop(*_args: object, **_kwargs: object) -> None:
        """Do nothing."""
        return None

    async def _deny(_user_id: int, _minutes: float, *, operation_id: str | None = None) -> tuple[bool, float]:
        """Deny the minutes."""
        return False, 0.0

    asked: list[int] = []

    async def _exhausted(user_id: int) -> bool:
        """Record the lookup and return the test's answer."""
        asked.append(user_id)
        return exhausted

    class _Registry:
        """A registry resolving every model to a stub provider."""

        def resolve_provider_for_model(self, _model: object) -> tuple[str, str, None]:
            """Resolve to the external stub."""
            return "external", "external:stub", None

    original_shim = audio_tx._audio_shim_attr

    def _shim(name: str):
        """Serve the fake monthly check; defer everything else."""
        return _exhausted if name == "monthly_minutes_exhausted" else original_shim(name)

    for name in ("can_start_job", "check_daily_minutes_allow"):
        monkeypatch.setattr(audio_ep, name, _allow)
    for name in ("increment_jobs_started", "finish_job", "add_daily_minutes"):
        monkeypatch.setattr(audio_ep, name, _noop)
    monkeypatch.setattr(audio_tx, "_consume_daily_minutes", _deny)
    monkeypatch.setattr(audio_tx, "_audio_shim_attr", _shim)
    monkeypatch.setattr(stt_adapter, "get_stt_provider_registry", lambda: _Registry())
    monkeypatch.setattr(atlib, "convert_to_wav", lambda path, *_a, **_k: path)

    app = FastAPI()
    app.dependency_overrides[get_request_user] = _user
    app.include_router(audio_ep.router, prefix="/api/v1/audio")
    with bypass_api_limits(app), TestClient(app) as client:
        resp = client.post(
            "/api/v1/audio/transcriptions",
            headers={"X-API-KEY": TEST_API_KEY},
            files={"file": ("sample.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
            data={"model": "external:stub", "response_format": "json"},
        )
    assert resp.status_code == 402, resp.text
    assert f"Transcription quota exceeded ({period} minutes)" in resp.text
    assert asked == [1]
