from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.admin import admin_settings
from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    AdminCleanupSettingsUpdate,
    NotesTitleSettingsUpdate,
)
from tldw_Server_API.app.services import admin_settings_service


pytestmark = pytest.mark.unit


class _ExplodingGetSettings:
    def __init__(self, message: str) -> None:
        self.message = message

    def get(self, *_args: object, **_kwargs: object) -> object:
        raise RuntimeError(self.message)


class _ExplodingSetSettings(dict[str, object]):
    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    def __setitem__(self, _key: str, _value: object) -> None:
        raise RuntimeError(self.message)


async def _assert_admin_settings_failure_log_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    fake_settings: object,
    call: object,
    expected_detail: str,
    raw_marker: str,
) -> None:
    monkeypatch.setattr(admin_settings_service, "app_settings", fake_settings)

    messages: list[str] = []
    sink_id = admin_settings_service.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        with pytest.raises(HTTPException) as exc_info:
            await call()  # type: ignore[misc]
    finally:
        admin_settings_service.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == expected_detail
    assert expected_detail in joined
    assert raw_marker not in joined
    assert "config.txt" not in joined


@pytest.fixture
def admin_settings_env(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "unit-test-api-key")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users_test_admin_settings.db'}")


@pytest.mark.asyncio
async def test_risk_weights_persist_to_admin_settings_table(admin_settings_env) -> None:

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool

    await reset_db_pool()

    updated = await admin_settings_service.set_risk_weights(
        {
            "mfa_adoption": {"weight": 7, "cap": 33},
            "failed_logins": {"weight": 4, "cap": 18},
        }
    )

    assert updated["mfa_adoption"] == {"weight": 7, "cap": 33}
    assert updated["failed_logins"] == {"weight": 4, "cap": 18}

    pool = await get_db_pool()
    row = await pool.fetchone(
        "SELECT value_json FROM admin_settings WHERE setting_key = ?",
        "security_risk_weights",
    )

    assert row is not None
    stored = json.loads(row["value_json"])
    assert stored["mfa_adoption"] == {"weight": 7, "cap": 33}
    assert stored["failed_logins"] == {"weight": 4, "cap": 18}

    fetched = await admin_settings_service.get_risk_weights()
    assert fetched["mfa_adoption"] == {"weight": 7, "cap": 33}
    assert fetched["failed_logins"] == {"weight": 4, "cap": 18}


@pytest.mark.asyncio
async def test_set_risk_weights_merges_existing_sections(admin_settings_env) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool

    await reset_db_pool()

    await admin_settings_service.set_risk_weights(
        {
            "mfa_adoption": {"weight": 7, "cap": 33},
            "failed_logins": {"weight": 4, "cap": 18},
        }
    )

    updated = await admin_settings_service.set_risk_weights(
        {
            "mfa_adoption": {"weight": 9, "cap": 44},
        }
    )

    assert updated["mfa_adoption"] == {"weight": 9, "cap": 44}
    assert updated["failed_logins"] == {"weight": 4, "cap": 18}


@pytest.mark.asyncio
async def test_get_cleanup_settings_sanitizes_backend_failure_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _assert_admin_settings_failure_log_sanitized(
        monkeypatch,
        _ExplodingGetSettings("cleanup settings read failed at /private/config.txt"),
        admin_settings_service.get_cleanup_settings,
        "Failed to get cleanup settings",
        "cleanup settings read failed",
    )


@pytest.mark.asyncio
async def test_set_cleanup_settings_sanitizes_backend_failure_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _call() -> dict[str, object]:
        return await admin_settings_service.set_cleanup_settings(AdminCleanupSettingsUpdate(enabled=True))

    await _assert_admin_settings_failure_log_sanitized(
        monkeypatch,
        _ExplodingSetSettings("cleanup settings write failed at /private/config.txt"),
        _call,
        "Failed to set cleanup settings",
        "cleanup settings write failed",
    )


@pytest.mark.asyncio
async def test_get_notes_title_settings_sanitizes_backend_failure_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _assert_admin_settings_failure_log_sanitized(
        monkeypatch,
        _ExplodingGetSettings("notes title read failed at /private/config.txt"),
        admin_settings_service.get_notes_title_settings,
        "Failed to get notes title settings",
        "notes title read failed",
    )


@pytest.mark.asyncio
async def test_set_notes_title_settings_sanitizes_backend_failure_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _call() -> dict[str, object]:
        return await admin_settings_service.set_notes_title_settings(NotesTitleSettingsUpdate(llm_enabled=True))

    await _assert_admin_settings_failure_log_sanitized(
        monkeypatch,
        _ExplodingSetSettings("notes title write failed at /private/config.txt"),
        _call,
        "Failed to set notes title settings",
        "notes title write failed",
    )


@pytest.mark.asyncio
async def test_get_risk_weights_sanitizes_backend_failure_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _raise_backend_failure() -> None:
        raise RuntimeError("risk weights DB failed at /private/admin-settings.db")

    monkeypatch.setattr(admin_settings_service, "_ensure_admin_settings_table", _raise_backend_failure)

    messages: list[str] = []
    sink_id = admin_settings_service.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        with pytest.raises(HTTPException) as exc_info:
            await admin_settings_service.get_risk_weights()
    finally:
        admin_settings_service.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to get risk weights"
    assert "Failed to get risk weights" in joined
    assert "risk weights DB failed" not in joined
    assert "admin-settings.db" not in joined


@pytest.mark.asyncio
async def test_set_risk_weights_sanitizes_backend_failure_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _raise_backend_failure() -> None:
        raise RuntimeError("risk weights write failed at /private/admin-settings.db")

    monkeypatch.setattr(admin_settings_service, "_ensure_admin_settings_table", _raise_backend_failure)

    messages: list[str] = []
    sink_id = admin_settings_service.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        with pytest.raises(HTTPException) as exc_info:
            await admin_settings_service.set_risk_weights({"mfa_adoption": {"weight": 5}})
    finally:
        admin_settings_service.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to set risk weights"
    assert "Failed to set risk weights" in joined
    assert "risk weights write failed" not in joined
    assert "admin-settings.db" not in joined


@pytest.mark.asyncio
async def test_get_security_risk_weights_endpoint_returns_plain_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {"mfa_adoption": {"weight": 9, "cap": 44}}

    async def _fake_get() -> dict[str, dict[str, int]]:
        return expected

    monkeypatch.setattr(admin_settings_service, "get_risk_weights", _fake_get)

    response = await admin_settings.get_security_risk_weights(principal=None)  # type: ignore[arg-type]

    assert response.weights == expected


@pytest.mark.asyncio
async def test_set_security_risk_weights_endpoint_returns_plain_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {"mfa_adoption": {"weight": 5, "cap": 25}}

    async def _fake_set(weights: dict[str, object]) -> dict[str, dict[str, int]]:
        assert weights == expected
        return expected

    monkeypatch.setattr(admin_settings_service, "set_risk_weights", _fake_set)

    response = await admin_settings.set_security_risk_weights(
        payload=admin_settings.RiskWeightsUpdateRequest(weights=expected),
        principal=None,  # type: ignore[arg-type]
    )

    assert response.weights == expected
