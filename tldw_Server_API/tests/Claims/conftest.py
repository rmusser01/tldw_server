import os

import pytest


_DISABLE_ROUTES = ("media", "audio", "audio-websocket")


def _merge_routes_disable(existing: str | None) -> str:
    parts = []
    seen = set()
    for raw in ((existing or "").split(",") + list(_DISABLE_ROUTES)):
        token = (raw or "").strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        parts.append(token)
    return ",".join(parts)


@pytest.fixture(autouse=True)
def _claims_suite_minimal_app_env(monkeypatch):
    """Minimal app with the media and audio routes off, for each Claims test only.

    Session scope kept these settings for every later test on the same worker (TASK-13840).
    """
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("ROUTES_DISABLE", _merge_routes_disable(os.environ.get("ROUTES_DISABLE")))
