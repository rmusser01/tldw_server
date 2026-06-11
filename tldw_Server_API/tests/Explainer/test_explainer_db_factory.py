"""Tests for the single ExplainerDatabase construction/lifecycle factory."""

from __future__ import annotations

import sqlite3

import pytest

from tldw_Server_API.app.core.DB_Management.Explainer_DB import (
    explainer_db_for_user,
    open_explainer_db,
)

pytestmark = pytest.mark.unit


def test_explainer_db_for_user_resolves_canonical_per_user_path(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

    db = explainer_db_for_user(7)
    try:
        assert db.client_id == "7"
        assert db.db_path == DatabasePaths.get_explainer_db_path(7).resolve()
    finally:
        db.close_connection()


def test_explainer_db_for_user_honors_explicit_path(tmp_path):
    explicit = tmp_path / "Explainer.db"

    db = explainer_db_for_user("9", db_path=explicit)
    try:
        assert db.client_id == "9"
        assert db.db_path == explicit.resolve()
    finally:
        db.close_connection()


def test_open_explainer_db_closes_connection_on_exit(tmp_path):
    with open_explainer_db(7, db_path=tmp_path / "Explainer.db") as db:
        conn = db.get_connection()
        conn.execute("SELECT 1")

    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")
