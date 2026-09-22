"""Guard collection of Prompt Studio tests against changing the shared server app."""

import os
import runpy
from pathlib import Path

import pytest


@pytest.mark.integration
def test_prompt_studio_collection_preserves_shared_app_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing another suite must preserve existing app identity and routing."""
    import tldw_Server_API.app.main as main

    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    keys = ("MINIMAL_TEST_APP", "TEST_MODE", "AUTH_MODE", "CSRF_ENABLED")
    before = {key: os.environ.get(key) for key in keys}
    app = main.app
    routes = tuple(app.routes)
    conftest = Path(__file__).parents[1] / "prompt_studio" / "conftest.py"

    runpy.run_path(str(conftest))

    assert {key: os.environ.get(key) for key in keys} == before  # nosec B101
    assert main.app is app  # nosec B101
    assert tuple(app.routes) == routes  # nosec B101
