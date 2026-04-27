from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.unit
def test_omnivoice_setup_status_strips_sidecar_last_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from tldw_Server_API.app.core.Setup import install_manager
    from tldw_Server_API.app.core.TTS import tts_service_v2

    monkeypatch.setattr(
        install_manager,
        "_get_omnivoice_provider_config",
        lambda: {
            "enabled": True,
            "runtime": "sidecar",
            "extra_params": {"runtime_mode": "real", "model_id": "k2-fsa/OmniVoice"},
        },
    )
    monkeypatch.setattr(
        install_manager,
        "_resolve_omnivoice_runtime_paths",
        lambda _provider_cfg=None: {
            "runtime_base": tmp_path / "models" / "omnivoice_sidecar",
            "venv_dir": tmp_path / "models" / "omnivoice_sidecar" / ".venv",
            "interpreter_path": tmp_path / "models" / "omnivoice_sidecar" / ".venv" / "bin" / "python",
            "runtime_dir": tmp_path / "models" / "omnivoice_sidecar" / "runtime",
            "logs_dir": tmp_path / "models" / "omnivoice_sidecar" / "logs",
            "source_checkout": tmp_path / "OmniVoice",
        },
    )
    monkeypatch.setattr(install_manager, "_hf_repo_cache_dir", lambda _model_id: tmp_path / "hf-cache")
    monkeypatch.setattr(tts_service_v2, "_service_instance", object(), raising=False)
    monkeypatch.setattr(
        install_manager.audio_health,
        "_derive_omnivoice_supervisor_health",
        lambda _service, _payload: {
            "runtime": "sidecar",
            "sidecar_state": "degraded",
            "last_error": "ImportError from /private/omnivoice/runtime.py",
        },
    )

    payload = install_manager.get_omnivoice_setup_status()

    assert payload["sidecar"]["sidecar_state"] == "degraded"  # nosec B101
    assert "last_error" not in payload["sidecar"]  # nosec B101
    assert "/private/omnivoice" not in str(payload)  # nosec B101


@pytest.mark.unit
@pytest.mark.asyncio
async def test_omnivoice_warmup_strips_health_last_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Setup import install_manager
    from tldw_Server_API.app.core.TTS import tts_service_v2
    from tldw_Server_API.app.core.TTS.adapters import omnivoice_sidecar_supervisor

    class _FakeSupervisor:
        sidecar_token = "test-sidecar-token"  # nosec B105

        async def ensure_started(self) -> str:
            return "http://127.0.0.1:8039"

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):  # noqa: ANN001
            return None

        async def post(self, url: str, headers: dict[str, str]):  # noqa: ARG002
            return SimpleNamespace(
                status_code=200,
                json=lambda: {
                    "ready": True,
                    "model_loaded": True,
                    "last_error": "RuntimeError from /private/omnivoice/runtime.py",
                },
            )

    async def _fake_get_tts_service_v2():
        return SimpleNamespace(_get_or_create_omnivoice_supervisor=lambda: _FakeSupervisor())

    monkeypatch.setattr(tts_service_v2, "get_tts_service_v2", _fake_get_tts_service_v2)
    monkeypatch.setattr(
        omnivoice_sidecar_supervisor,
        "create_sidecar_async_client",
        lambda *, timeout: _FakeClient(),  # noqa: ARG005
    )
    monkeypatch.setattr(
        install_manager,
        "get_omnivoice_setup_status",
        lambda: {
            "provider": "omnivoice",
            "sidecar": {
                "runtime": "sidecar",
                "sidecar_state": "ready",
                "last_error": "ImportError from /private/omnivoice/status.py",
            },
        },
    )

    payload = await install_manager.warmup_omnivoice_sidecar_async()

    assert payload["success"] is True  # nosec B101
    assert "last_error" not in payload["health"]  # nosec B101
    assert "last_error" not in payload["omnivoice"]["sidecar"]  # nosec B101
    assert "/private/omnivoice" not in str(payload)  # nosec B101
