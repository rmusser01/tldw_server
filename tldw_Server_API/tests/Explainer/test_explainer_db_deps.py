"""Tests for the per-user ExplainerDatabase dependency cache."""

from __future__ import annotations

import sqlite3
import threading

import pytest

from tldw_Server_API.app.api.v1.API_Deps import Explainer_DB_Deps as deps_mod
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    deps_mod.cleanup_explainer_db_cache()
    yield
    deps_mod.cleanup_explainer_db_cache()


def _user(user_id: int) -> User:
    return User(id=user_id, username=f"user-{user_id}", email=None, is_active=True)


def test_get_explainer_db_returns_cached_instance_per_user():
    first = deps_mod.get_explainer_db(current_user=_user(1))
    second = deps_mod.get_explainer_db(current_user=_user(1))
    other = deps_mod.get_explainer_db(current_user=_user(2))

    assert first is second
    assert other is not first


def test_eviction_closes_connections_on_every_thread(monkeypatch):
    monkeypatch.setattr(deps_mod, "_MAX_CACHED_EXPLAINER_DB", 1)
    evicted_db = deps_mod.get_explainer_db(current_user=_user(1))
    main_conn = evicted_db.get_connection()
    other_holder: dict[str, object] = {}

    def _open_on_other_thread() -> None:
        other_holder["conn"] = evicted_db.get_connection()

    worker = threading.Thread(target=_open_on_other_thread)
    worker.start()
    worker.join()

    deps_mod.get_explainer_db(current_user=_user(2))

    with pytest.raises(sqlite3.ProgrammingError):
        main_conn.execute("SELECT 1")
    with pytest.raises(sqlite3.ProgrammingError):
        other_holder["conn"].execute("SELECT 1")
