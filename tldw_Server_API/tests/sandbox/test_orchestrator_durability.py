"""Tests for durable run queue in the sandbox store backends.

Validates that enqueue_run / dequeue_run persist across store instances
(for SQLite) and respect priority ordering.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from tldw_Server_API.app.core.config import clear_config_cache, settings as app_settings
from tldw_Server_API.app.core.Sandbox.store import InMemoryStore, SQLiteStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _configure_sqlite_store(monkeypatch, tmp_path: Path) -> str:
    db_path = str(tmp_path / "sandbox_store.db")
    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "sqlite")
    monkeypatch.setenv("SANDBOX_STORE_DB_PATH", db_path)
    if hasattr(app_settings, "SANDBOX_STORE_BACKEND"):
        monkeypatch.setattr(app_settings, "SANDBOX_STORE_BACKEND", "sqlite")
    if hasattr(app_settings, "SANDBOX_STORE_DB_PATH"):
        monkeypatch.setattr(app_settings, "SANDBOX_STORE_DB_PATH", db_path)
    clear_config_cache()
    return db_path


# ---------------------------------------------------------------------------
# InMemoryStore tests
# ---------------------------------------------------------------------------

class TestInMemoryStoreQueue:
    def test_enqueue_and_dequeue_returns_entry(self):
        store = InMemoryStore()
        store.enqueue_run("run-1", "user-a", priority=0)
        result = store.dequeue_run("worker-1")
        assert result is not None
        assert result["run_id"] == "run-1"
        assert result["user_id"] == "user-a"

    def test_dequeue_returns_highest_priority_first(self):
        store = InMemoryStore()
        store.enqueue_run("run-low", "user-a", priority=0)
        store.enqueue_run("run-high", "user-a", priority=10)
        store.enqueue_run("run-mid", "user-a", priority=5)

        first = store.dequeue_run("worker-1")
        assert first is not None
        assert first["run_id"] == "run-high"

        second = store.dequeue_run("worker-1")
        assert second is not None
        assert second["run_id"] == "run-mid"

        third = store.dequeue_run("worker-1")
        assert third is not None
        assert third["run_id"] == "run-low"

    def test_dequeue_returns_none_when_empty(self):
        store = InMemoryStore()
        result = store.dequeue_run("worker-1")
        assert result is None

    def test_dequeue_fifo_within_same_priority(self):
        store = InMemoryStore()
        store.enqueue_run("run-a", "user-a", priority=5)
        # Tiny sleep so enqueued_at is distinct
        time.sleep(0.001)
        store.enqueue_run("run-b", "user-a", priority=5)

        first = store.dequeue_run("worker-1")
        assert first is not None
        assert first["run_id"] == "run-a"

    def test_remove_from_queue(self):
        store = InMemoryStore()
        store.enqueue_run("run-1", "user-a", priority=0)
        store.enqueue_run("run-2", "user-a", priority=0)
        removed = store.remove_from_queue("run-1")
        assert removed is True
        result = store.dequeue_run("worker-1")
        assert result is not None
        assert result["run_id"] == "run-2"

    def test_remove_nonexistent_returns_false(self):
        store = InMemoryStore()
        assert store.remove_from_queue("ghost") is False


# ---------------------------------------------------------------------------
# SQLiteStore tests
# ---------------------------------------------------------------------------

class TestSQLiteStoreQueue:
    def test_enqueue_and_dequeue_returns_entry(self, monkeypatch, tmp_path):
        db_path = _configure_sqlite_store(monkeypatch, tmp_path)
        store = SQLiteStore(db_path=db_path)
        store.enqueue_run("run-1", "user-a", priority=0)
        result = store.dequeue_run("worker-1")
        assert result is not None
        assert result["run_id"] == "run-1"
        assert result["user_id"] == "user-a"

    def test_dequeue_returns_highest_priority_first(self, monkeypatch, tmp_path):
        db_path = _configure_sqlite_store(monkeypatch, tmp_path)
        store = SQLiteStore(db_path=db_path)
        store.enqueue_run("run-low", "user-a", priority=0)
        store.enqueue_run("run-high", "user-a", priority=10)
        store.enqueue_run("run-mid", "user-a", priority=5)

        first = store.dequeue_run("worker-1")
        assert first is not None
        assert first["run_id"] == "run-high"

        second = store.dequeue_run("worker-1")
        assert second is not None
        assert second["run_id"] == "run-mid"

        third = store.dequeue_run("worker-1")
        assert third is not None
        assert third["run_id"] == "run-low"

    def test_dequeue_returns_none_when_empty(self, monkeypatch, tmp_path):
        db_path = _configure_sqlite_store(monkeypatch, tmp_path)
        store = SQLiteStore(db_path=db_path)
        result = store.dequeue_run("worker-1")
        assert result is None

    def test_enqueued_runs_survive_store_recreation(self, monkeypatch, tmp_path):
        """The key durability test: enqueued runs persist across SQLiteStore instances."""
        db_path = _configure_sqlite_store(monkeypatch, tmp_path)

        store1 = SQLiteStore(db_path=db_path)
        store1.enqueue_run("run-durable", "user-x", priority=5)

        # Create a brand new store instance pointing at the same DB file
        store2 = SQLiteStore(db_path=db_path)
        result = store2.dequeue_run("worker-2")
        assert result is not None
        assert result["run_id"] == "run-durable"
        assert result["user_id"] == "user-x"

    def test_remove_from_queue(self, monkeypatch, tmp_path):
        db_path = _configure_sqlite_store(monkeypatch, tmp_path)
        store = SQLiteStore(db_path=db_path)
        store.enqueue_run("run-1", "user-a", priority=0)
        store.enqueue_run("run-2", "user-a", priority=0)
        removed = store.remove_from_queue("run-1")
        assert removed is True
        result = store.dequeue_run("worker-1")
        assert result is not None
        assert result["run_id"] == "run-2"

    def test_remove_nonexistent_returns_false(self, monkeypatch, tmp_path):
        db_path = _configure_sqlite_store(monkeypatch, tmp_path)
        store = SQLiteStore(db_path=db_path)
        assert store.remove_from_queue("ghost") is False

    def test_dequeue_fifo_within_same_priority(self, monkeypatch, tmp_path):
        db_path = _configure_sqlite_store(monkeypatch, tmp_path)
        store = SQLiteStore(db_path=db_path)
        store.enqueue_run("run-a", "user-a", priority=5)
        store.enqueue_run("run-b", "user-a", priority=5)

        first = store.dequeue_run("worker-1")
        assert first is not None
        assert first["run_id"] == "run-a"
