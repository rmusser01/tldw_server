import asyncio
import concurrent.futures
from typing import Any

import pytest

from tldw_Server_API.app.core.Claims_Extraction import claims_engine, claims_service

pytestmark = pytest.mark.unit


def test_claims_engine_noncritical_exceptions_do_not_swallow_cancellation():
    assert asyncio.CancelledError not in claims_engine._CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS


def test_claims_service_noncritical_exceptions_do_not_swallow_cancellation():
    assert asyncio.CancelledError not in claims_service._CLAIMS_NONCRITICAL_EXCEPTIONS


@pytest.mark.parametrize("settings_binding", ["snapshot", "rebound"])
@pytest.mark.parametrize("refresh_global_settings", [False, True])
def test_ingestion_llm_timeout_shuts_down_executor_without_waiting(
    monkeypatch, settings_binding, refresh_global_settings
):
    import tldw_Server_API.app.core.Claims_Extraction.ingestion_claims as ingestion_mod
    from tldw_Server_API.app.core import config

    if settings_binding == "rebound":
        monkeypatch.setattr(ingestion_mod, "_settings", config.LazySettings(config.load_settings))
    else:
        # Own the consumed inputs even when another fixture refreshes global config.
        monkeypatch.setattr(ingestion_mod, "_settings", dict(ingestion_mod._settings))

    shutdown_calls: list[dict[str, Any]] = []
    cancel_calls: list[bool] = []

    class _TimeoutFuture:
        def result(self, timeout: float | None = None) -> Any:
            assert timeout == 0.01
            raise concurrent.futures.TimeoutError

        def cancel(self) -> bool:
            cancel_calls.append(True)
            return True

    class _RecordingExecutor:
        def __init__(self, max_workers: int) -> None:
            assert max_workers == 1
            self._shutdown = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            if not self._shutdown:
                self.shutdown(wait=True)
            return False

        def submit(self, fn):
            return _TimeoutFuture()

        def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
            self._shutdown = True
            shutdown_calls.append({"wait": wait, "cancel_futures": cancel_futures})

    monkeypatch.setitem(ingestion_mod._settings, "CLAIMS_LLM_TIMEOUT_SEC", 0.01)
    monkeypatch.setitem(ingestion_mod._settings, "CLAIMS_EXTRACTION_PASSES", 1)
    monkeypatch.setitem(ingestion_mod._settings, "CLAIMS_CONTEXT_WINDOW_CHARS", 0)
    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", _RecordingExecutor)
    monkeypatch.setattr(ingestion_mod, "extract_heuristic_claims_texts", lambda *args, **kwargs: [])
    monkeypatch.setattr(ingestion_mod, "record_claims_provider_request", lambda **kwargs: None)
    monkeypatch.setattr(ingestion_mod, "record_claims_fallback", lambda **kwargs: None)

    if refresh_global_settings:
        config.clear_config_cache()

    ingestion_mod.extract_claims_for_chunks(
        [{"text": "Provider timeout claim.", "metadata": {"chunk_index": 0}}],
        extractor_mode="openai",
        max_per_chunk=1,
    )

    assert cancel_calls == [True]
    assert shutdown_calls == [{"wait": False, "cancel_futures": True}]
