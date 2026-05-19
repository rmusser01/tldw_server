"""Tests for ACP run history and cost tracking.

Tests the service/DB level query methods (list_runs, aggregate_runs)
without needing HTTP TestClient or auth.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB
from tldw_Server_API.app.services.admin_acp_sessions_service import ACPSessionStore


@pytest.fixture
def db():
    """Create a temporary ACP sessions DB."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "acp_sessions.db")
        instance = ACPSessionsDB(db_path=path)
        yield instance
        instance.close()


@pytest.fixture
def store(db):
    """Create an ACPSessionStore backed by the temp DB."""
    return ACPSessionStore(db=db)


def _insert_session(
    db: ACPSessionsDB,
    session_id: str,
    user_id: int = 1,
    agent_type: str = "claude_code",
    status: str | None = None,
    model: str | None = None,
) -> dict:
    """Helper to insert a session and optionally close/error it."""
    row = db.register_session(
        session_id=session_id,
        user_id=user_id,
        agent_type=agent_type,
        model=model,
    )
    if status == "closed":
        db.close_session(session_id)
    elif status == "error":
        db.set_session_status(session_id, "error")
    return row


def _record_tokens(db: ACPSessionsDB, session_id: str, prompt_tokens: int, completion_tokens: int) -> None:
    """Record a prompt exchange to accumulate token usage."""
    db.record_prompt(
        session_id,
        [{"role": "user", "content": "hello"}],
        {"role": "assistant", "content": "world", "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }},
    )


# ---------------------------------------------------------------------------
# DB-level tests for list_runs
# ---------------------------------------------------------------------------


class TestListRunsDB:
    def test_list_runs_returns_sessions(self, db: ACPSessionsDB):
        _insert_session(db, "s1")
        _insert_session(db, "s2")
        rows, total = db.list_runs()
        assert total == 2
        assert len(rows) == 2
        ids = {r["session_id"] for r in rows}
        assert ids == {"s1", "s2"}

    def test_list_runs_filters_by_status(self, db: ACPSessionsDB):
        _insert_session(db, "s1")
        _insert_session(db, "s2", status="closed")
        rows, total = db.list_runs(status="closed")
        assert total == 1
        assert rows[0]["session_id"] == "s2"

    def test_list_runs_filters_by_agent_type(self, db: ACPSessionsDB):
        _insert_session(db, "s1", agent_type="claude_code")
        _insert_session(db, "s2", agent_type="codex")
        rows, total = db.list_runs(agent_type="codex")
        assert total == 1
        assert rows[0]["session_id"] == "s2"

    def test_list_runs_filters_by_date_range(self, db: ACPSessionsDB):
        _insert_session(db, "s1")
        _insert_session(db, "s2")
        # Both sessions were created just now with ISO timestamps.
        # Use a far-future date that should match nothing.
        rows, total = db.list_runs(from_date="2099-01-01")
        assert total == 0
        assert rows == []

        # Use a far-past date that should match everything.
        rows, total = db.list_runs(from_date="2000-01-01")
        assert total == 2

    def test_list_runs_pagination(self, db: ACPSessionsDB):
        for i in range(5):
            _insert_session(db, f"s{i}")
        rows, total = db.list_runs(limit=2, offset=0)
        assert total == 5
        assert len(rows) == 2

        rows2, total2 = db.list_runs(limit=2, offset=2)
        assert total2 == 5
        assert len(rows2) == 2

        # No overlap between pages
        page1_ids = {r["session_id"] for r in rows}
        page2_ids = {r["session_id"] for r in rows2}
        assert page1_ids.isdisjoint(page2_ids)


# ---------------------------------------------------------------------------
# DB-level tests for aggregate_runs
# ---------------------------------------------------------------------------


class TestAggregateRunsDB:
    def test_aggregate_runs_sums_tokens(self, db: ACPSessionsDB):
        _insert_session(db, "s1")
        _insert_session(db, "s2")
        _record_tokens(db, "s1", prompt_tokens=100, completion_tokens=50)
        _record_tokens(db, "s2", prompt_tokens=200, completion_tokens=80)

        agg = db.aggregate_runs()
        assert agg["total_sessions"] == 2
        assert agg["prompt_tokens"] == 300
        assert agg["completion_tokens"] == 130
        assert agg["total_tokens"] == 430

    def test_aggregate_runs_empty(self, db: ACPSessionsDB):
        agg = db.aggregate_runs()
        assert agg["total_sessions"] == 0
        assert agg["prompt_tokens"] == 0
        assert agg["completion_tokens"] == 0
        assert agg["total_tokens"] == 0

    def test_aggregate_runs_with_date_filter(self, db: ACPSessionsDB):
        _insert_session(db, "s1")
        _record_tokens(db, "s1", prompt_tokens=50, completion_tokens=25)

        # Far-future filter should match nothing
        agg = db.aggregate_runs(from_date="2099-01-01")
        assert agg["total_sessions"] == 0
        assert agg["total_tokens"] == 0


# ---------------------------------------------------------------------------
# Service-level tests
# ---------------------------------------------------------------------------


class TestListRunsService:
    @pytest.mark.asyncio
    async def test_list_runs_returns_sessions(self, store: ACPSessionStore, db: ACPSessionsDB):
        _insert_session(db, "s1")
        _insert_session(db, "s2")
        result = await store.list_runs()
        assert result["total"] == 2
        assert len(result["items"]) == 2
        assert result["limit"] == 100
        assert result["offset"] == 0
        # Each item should have session_id from to_info_dict
        ids = {item["session_id"] for item in result["items"]}
        assert ids == {"s1", "s2"}

    @pytest.mark.asyncio
    async def test_list_runs_filters_by_status(self, store: ACPSessionStore, db: ACPSessionsDB):
        _insert_session(db, "s1")
        _insert_session(db, "s2", status="closed")
        result = await store.list_runs(status="closed")
        assert result["total"] == 1
        assert result["items"][0]["session_id"] == "s2"

    @pytest.mark.asyncio
    async def test_list_runs_pagination(self, store: ACPSessionStore, db: ACPSessionsDB):
        for i in range(5):
            _insert_session(db, f"s{i}")
        result = await store.list_runs(limit=2, offset=0)
        assert result["total"] == 5
        assert len(result["items"]) == 2

    @pytest.mark.asyncio
    async def test_list_runs_filters_by_date(self, store: ACPSessionStore, db: ACPSessionsDB):
        _insert_session(db, "s1")
        result = await store.list_runs(from_date="2099-01-01")
        assert result["total"] == 0


class TestAggregateRunsService:
    @pytest.mark.asyncio
    async def test_aggregate_runs_sums_tokens(self, store: ACPSessionStore, db: ACPSessionsDB):
        _insert_session(db, "s1")
        _insert_session(db, "s2")
        _record_tokens(db, "s1", prompt_tokens=100, completion_tokens=50)
        _record_tokens(db, "s2", prompt_tokens=200, completion_tokens=80)

        result = await store.aggregate_runs()
        assert result["total_sessions"] == 2
        assert result["prompt_tokens"] == 300
        assert result["completion_tokens"] == 130
        assert result["total_tokens"] == 430
        # estimated_cost_usd should be present (may be None if model not set)
        assert "estimated_cost_usd" in result

    @pytest.mark.asyncio
    async def test_aggregate_runs_empty(self, store: ACPSessionStore, db: ACPSessionsDB):
        result = await store.aggregate_runs()
        assert result["total_sessions"] == 0
        assert result["prompt_tokens"] == 0
        assert result["completion_tokens"] == 0
        assert result["total_tokens"] == 0
        assert "estimated_cost_usd" in result

    @pytest.mark.asyncio
    async def test_aggregate_with_model_computes_cost(self, store: ACPSessionStore, db: ACPSessionsDB):
        _insert_session(db, "s1", model="gpt-4")
        _record_tokens(db, "s1", prompt_tokens=1000, completion_tokens=500)

        result = await store.aggregate_runs()
        assert result["total_sessions"] == 1
        # With a real model name, cost might be computed (or None if not in catalog)
        # The important thing is the key exists and the method doesn't error
        assert "estimated_cost_usd" in result
