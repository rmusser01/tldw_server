"""Keep the suite offline: no test may run the real git or gh."""

from __future__ import annotations

import pytest
from backlog_py.core import repository as repository_module
from backlog_py.core.remote_ids import RemoteIdState


@pytest.fixture(autouse=True)
def _no_remote_id_lookup(monkeypatch):
    monkeypatch.setattr(repository_module, "collect_remote_ids", lambda _root: RemoteIdState())
