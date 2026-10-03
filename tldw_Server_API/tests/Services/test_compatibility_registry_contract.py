from __future__ import annotations

from tldw_Server_API.app.core.deprecations.runtime_registry import (
    load_compat_registry,
    log_runtime_deprecation,
    reset_runtime_deprecation_cycle,
)


# Compat paths past their sunset dates, removed in TASK-13444.
# Lookups for these keys must come back absent, not raise, so new code
# cannot silently keep depending on them.
EXPIRED_COMPAT_KEYS = (
    "web_scraping_legacy_fallback",
    "llm_chat_legacy_session",
    "auth_db_execute_compat",
)


def test_expired_compat_paths_are_not_registered():
    registry = load_compat_registry()
    for key in EXPIRED_COMPAT_KEYS:
        assert key not in registry  # nosec B101


def test_deprecation_registry_emits_once_per_request_cycle(monkeypatch):
    emitted: list[str] = []

    def _capture_warning(message: str, *args, **kwargs):
        _ = (args, kwargs)
        emitted.append(str(message))

    monkeypatch.setattr(
        "tldw_Server_API.app.core.deprecations.runtime_registry.logger.warning",
        _capture_warning,
    )

    synthetic_key = "__synthetic_emit_once_probe__"
    reset_runtime_deprecation_cycle()
    log_runtime_deprecation(synthetic_key)
    log_runtime_deprecation(synthetic_key)
    assert len(emitted) == 1  # nosec B101

    reset_runtime_deprecation_cycle()
    log_runtime_deprecation(synthetic_key)
    assert len(emitted) == 2  # nosec B101
