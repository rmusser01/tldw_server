"""The app lifespan sets the accounting loop; without it, recording from pool threads would silently stop."""

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Usage import llm_accounting

pytestmark = pytest.mark.integration


def test_the_app_lifespan_sets_a_running_accounting_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """While the app is up, the accounting loop is set and running."""
    from tldw_Server_API.app.main import app

    monkeypatch.setattr(llm_accounting, "_accounting_loop", None)
    with TestClient(app):
        loop = llm_accounting.get_accounting_loop()
        assert loop is not None and loop.is_running()


def test_the_app_lifespan_installs_the_state_carrying_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    """While the app is up, the server loop's default executor carries the LLM call state."""
    from tldw_Server_API.app.core.Usage.llm_roots import LLMStateThreadPoolExecutor
    from tldw_Server_API.app.main import app

    monkeypatch.setattr(llm_accounting, "_accounting_loop", None)
    with TestClient(app):
        loop = llm_accounting.get_accounting_loop()
        assert loop is not None
        assert isinstance(loop._default_executor, LLMStateThreadPoolExecutor)  # noqa: SLF001
