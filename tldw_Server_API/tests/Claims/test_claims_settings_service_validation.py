import pytest
from fastapi import HTTPException

from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Claims_Extraction import claims_service
from tldw_Server_API.app.core.config import settings


pytestmark = pytest.mark.unit


_SETTING_KEYS = (
    "CLAIMS_PROMPT_VALIDATION_MODE",
    "CLAIMS_PROMPT_VALIDATION_STRICT",
    "CLAIMS_ALIGNMENT_MODE",
    "CLAIMS_ALIGNMENT_THRESHOLD",
    "CLAIMS_CONTEXT_WINDOW_CHARS",
    "CLAIMS_EXTRACTION_PASSES",
)


@pytest.fixture
def admin_principal() -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        subject="admin",
        roles=["admin"],
        permissions=["system.configure"],
        is_admin=True,
    )


@pytest.fixture
def restore_claims_settings() -> None:
    original_values = {key: settings.get(key) for key in _SETTING_KEYS}
    try:
        yield
    finally:
        for key, value in original_values.items():
            if value is None:
                settings.pop(key, None)
            else:
                settings[key] = value


def test_update_claims_settings_clamps_and_skips_invalid_values(
    admin_principal: AuthPrincipal,
    restore_claims_settings,
) -> None:
    settings["CLAIMS_PROMPT_VALIDATION_MODE"] = "warning"
    settings["CLAIMS_PROMPT_VALIDATION_STRICT"] = True
    settings["CLAIMS_ALIGNMENT_MODE"] = "fuzzy"
    settings["CLAIMS_ALIGNMENT_THRESHOLD"] = 0.5
    settings["CLAIMS_CONTEXT_WINDOW_CHARS"] = 512
    settings["CLAIMS_EXTRACTION_PASSES"] = 3

    result = claims_service.update_claims_settings(
        payload={
            "claims_prompt_validation_mode": "not-a-mode",
            "claims_prompt_validation_strict": "false",
            "claims_alignment_mode": "unknown",
            "claims_alignment_threshold": 1.7,
            "claims_context_window_chars": 999999,
            "claims_extraction_passes": -5,
            "persist": False,
        },
        principal=admin_principal,
    )

    assert result["claims_prompt_validation_mode"] == "warning"
    assert result["claims_prompt_validation_strict"] is False
    assert result["claims_alignment_mode"] == "fuzzy"
    assert result["claims_alignment_threshold"] == pytest.approx(1.0)
    assert result["claims_context_window_chars"] == 20000
    assert result["claims_extraction_passes"] == 1


def test_update_claims_settings_ignores_unparseable_numeric_values(
    admin_principal: AuthPrincipal,
    restore_claims_settings,
) -> None:
    settings["CLAIMS_PROMPT_VALIDATION_MODE"] = "warning"
    settings["CLAIMS_PROMPT_VALIDATION_STRICT"] = False
    settings["CLAIMS_ALIGNMENT_MODE"] = "exact"
    settings["CLAIMS_ALIGNMENT_THRESHOLD"] = 0.42
    settings["CLAIMS_CONTEXT_WINDOW_CHARS"] = 2048
    settings["CLAIMS_EXTRACTION_PASSES"] = 2

    result = claims_service.update_claims_settings(
        payload={
            "claims_prompt_validation_strict": "yes",
            "claims_alignment_threshold": "not-a-number",
            "claims_context_window_chars": "not-an-int",
            "claims_extraction_passes": "nope",
            "persist": False,
        },
        principal=admin_principal,
    )

    assert result["claims_prompt_validation_strict"] is True
    assert result["claims_alignment_threshold"] == pytest.approx(0.42)
    assert result["claims_context_window_chars"] == 2048
    assert result["claims_extraction_passes"] == 2


def test_update_claims_settings_sanitizes_persist_failure(
    admin_principal: AuthPrincipal,
    restore_claims_settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_config_error(_payload):
        raise RuntimeError("config write leaked /private/claims/config/token")

    monkeypatch.setattr(claims_service.setup_manager, "update_config", raise_config_error)

    with pytest.raises(HTTPException) as exc_info:
        claims_service.update_claims_settings(
            payload={
                "claims_prompt_validation_strict": False,
                "persist": True,
            },
            principal=admin_principal,
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to update claims settings"


def test_claims_rebuild_status_sanitizes_service_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_service_error():
        raise RuntimeError("rebuild worker leaked /private/claims/queue/token")

    monkeypatch.setattr(claims_service, "get_claims_rebuild_service", raise_service_error)

    with pytest.raises(HTTPException) as exc_info:
        claims_service.claims_rebuild_status()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to get claims rebuild status"
