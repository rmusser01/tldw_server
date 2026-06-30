from __future__ import annotations

from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


def test_run_pg_rls_auto_ensure_invokes_both_installers_and_logs_combined_result() -> None:
    from tldw_Server_API.app.services.startup_pg_rls import run_pg_rls_auto_ensure

    backend = object()
    calls: list[tuple[str, object]] = []
    logged_messages: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def _ensure_prompt_studio_rls(seen_backend: object) -> bool:
        calls.append(("prompt", seen_backend))
        return True

    def _ensure_chacha_rls(seen_backend: object) -> bool:
        calls.append(("chacha", seen_backend))
        return False

    logger_obj = SimpleNamespace(
        info=lambda message, *args, **kwargs: logged_messages.append((message, args, kwargs))
    )

    result = run_pg_rls_auto_ensure(
        backend,
        ensure_prompt_studio_rls=_ensure_prompt_studio_rls,
        ensure_chacha_rls=_ensure_chacha_rls,
        logger_obj=logger_obj,
    )

    assert result == (True, False)
    assert calls == [("prompt", backend), ("chacha", backend)]
    assert logged_messages == [
        (
            "PG RLS ensure invoked (prompt_studio_applied={}, chacha_applied={})",
            (True, False),
            {},
        )
    ]


def test_run_pg_rls_auto_ensure_propagates_installer_failure_without_logging() -> None:
    from tldw_Server_API.app.services.startup_pg_rls import run_pg_rls_auto_ensure

    logged_messages: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def _boom(_backend: object) -> bool:
        raise RuntimeError("boom")

    logger_obj = SimpleNamespace(
        info=lambda message, *args, **kwargs: logged_messages.append((message, args, kwargs))
    )

    with pytest.raises(RuntimeError, match="boom"):
        run_pg_rls_auto_ensure(
            object(),
            ensure_prompt_studio_rls=_boom,
            ensure_chacha_rls=lambda _backend: True,
            logger_obj=logger_obj,
        )

    assert logged_messages == []
