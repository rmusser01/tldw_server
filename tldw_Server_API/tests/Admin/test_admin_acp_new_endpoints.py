"""Tests for ACP agent metrics, session budget PATCH, and budget enforcement.

Exercises the endpoint handler functions and service layer directly
using isolated ACPSessionStore instances backed by temporary SQLite files.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run a coroutine synchronously."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def _make_store(tmp_path):
    """Create a fresh ACPSessionStore backed by a temp SQLite file."""
    from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB
    from tldw_Server_API.app.services.admin_acp_sessions_service import ACPSessionStore

    db_path = str(tmp_path / "acp_sessions_test.db")
    db = ACPSessionsDB(db_path=db_path)
    store = ACPSessionStore(db=db)
    # Prevent background cleanup from running
    store._cleanup_task = None
    return store


async def _register_session(
    store,
    session_id: str = "sess-001",
    user_id: int = 42,
    agent_type: str = "custom",
    model: str | None = "gpt-4o",
    token_budget: int | None = None,
    auto_terminate: bool = False,
):
    """Register a session on the given store."""
    return await store.register_session(
        session_id=session_id,
        user_id=user_id,
        agent_type=agent_type,
        name="Test session",
        model=model,
        token_budget=token_budget,
        auto_terminate_at_budget=auto_terminate,
    )


async def _add_usage(store, session_id: str, prompt_tokens: int, completion_tokens: int):
    """Record a prompt exchange to accumulate tokens."""
    return await store.record_prompt(
        session_id,
        [{"role": "user", "content": "Hello"}],
        {"role": "assistant", "content": "Hi", "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }},
    )


# ===========================================================================
# 1. GET /admin/acp/sessions
# ===========================================================================


class TestACPAdminSessionsList:
    """Route-level tests for the ACP sessions admin list endpoint."""

    def test_sessions_list_includes_canonical_pagination(self, monkeypatch, tmp_path):
        """Listing ACP sessions preserves total and adds nested pagination."""
        store = _make_store(tmp_path)

        async def _seed():
            await _register_session(store, "s1", user_id=1, agent_type="coder")
            await _register_session(store, "s2", user_id=1, agent_type="coder")
            await _register_session(store, "s3", user_id=1, agent_type="coder")
            await _register_session(store, "s4", user_id=1, agent_type="coder")

        _run(_seed())

        monkeypatch.setattr(
            "tldw_Server_API.app.api.v1.endpoints.admin.admin_acp_agents.get_acp_session_store",
            mock.AsyncMock(return_value=store),
        )

        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_acp_agents

        app = FastAPI()
        app.include_router(admin_acp_agents.router, prefix="/api/v1/admin")

        response = TestClient(app).get("/api/v1/admin/acp/sessions?limit=2&offset=1")

        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 4
        assert len(body["sessions"]) == 2
        assert body["pagination"] == {
            "mode": "offset",
            "limit": 2,
            "offset": 1,
            "total": 4,
            "has_more": True,
            "next_offset": 3,
        }


# ===========================================================================
# 2. GET /admin/acp/agents/metrics
# ===========================================================================


class TestACPAgentMetrics:
    """Tests for the agent metrics aggregation endpoint handler."""

    def test_metrics_empty_store(self, monkeypatch, tmp_path):
        """An empty store returns an empty items list."""
        store = _make_store(tmp_path)

        async def _run_test():
            monkeypatch.setattr(
                "tldw_Server_API.app.api.v1.endpoints.admin.admin_acp_agents.get_acp_session_store",
                mock.AsyncMock(return_value=store),
            )
            from tldw_Server_API.app.api.v1.endpoints.admin.admin_acp_agents import (
                get_acp_agent_metrics,
            )

            resp = await get_acp_agent_metrics()
            assert resp.items == []

        _run(_run_test())

    def test_metrics_single_agent_type(self, monkeypatch, tmp_path):
        """One agent type with sessions yields correct aggregated counts."""
        store = _make_store(tmp_path)

        async def _run_test():
            await _register_session(store, "s1", user_id=1, agent_type="coder", model="gpt-4o")
            await _register_session(store, "s2", user_id=2, agent_type="coder", model="gpt-4o")
            await _add_usage(store, "s1", 100, 50)
            await _add_usage(store, "s2", 200, 100)

            monkeypatch.setattr(
                "tldw_Server_API.app.api.v1.endpoints.admin.admin_acp_agents.get_acp_session_store",
                mock.AsyncMock(return_value=store),
            )
            from tldw_Server_API.app.api.v1.endpoints.admin.admin_acp_agents import (
                get_acp_agent_metrics,
            )

            resp = await get_acp_agent_metrics()
            assert len(resp.items) == 1
            item = resp.items[0]
            assert item.agent_type == "coder"
            assert item.session_count == 2
            assert item.active_sessions == 2
            assert item.total_prompt_tokens == 300
            assert item.total_completion_tokens == 150
            assert item.total_tokens == 450
            # Each record_prompt stores 2 messages (user + assistant) per call
            assert item.total_messages == 4

        _run(_run_test())

    def test_metrics_multiple_agent_types(self, monkeypatch, tmp_path):
        """Multiple agent types produce separate items sorted by total_tokens desc."""
        store = _make_store(tmp_path)

        async def _run_test():
            await _register_session(store, "s1", agent_type="researcher", model="gpt-4o")
            await _register_session(store, "s2", agent_type="coder", model="gpt-4o")
            await _add_usage(store, "s1", 50, 50)    # 100 total
            await _add_usage(store, "s2", 300, 200)   # 500 total

            monkeypatch.setattr(
                "tldw_Server_API.app.api.v1.endpoints.admin.admin_acp_agents.get_acp_session_store",
                mock.AsyncMock(return_value=store),
            )
            from tldw_Server_API.app.api.v1.endpoints.admin.admin_acp_agents import (
                get_acp_agent_metrics,
            )

            resp = await get_acp_agent_metrics()
            assert len(resp.items) == 2
            # Sorted by total_tokens descending
            assert resp.items[0].agent_type == "coder"
            assert resp.items[1].agent_type == "researcher"

        _run(_run_test())

    def test_metrics_includes_closed_sessions_in_count(self, monkeypatch, tmp_path):
        """Closed sessions count toward session_count but not active_sessions."""
        store = _make_store(tmp_path)

        async def _run_test():
            await _register_session(store, "s1", agent_type="coder")
            await _register_session(store, "s2", agent_type="coder")
            await store.close_session("s2")

            monkeypatch.setattr(
                "tldw_Server_API.app.api.v1.endpoints.admin.admin_acp_agents.get_acp_session_store",
                mock.AsyncMock(return_value=store),
            )
            from tldw_Server_API.app.api.v1.endpoints.admin.admin_acp_agents import (
                get_acp_agent_metrics,
            )

            resp = await get_acp_agent_metrics()
            assert len(resp.items) == 1
            item = resp.items[0]
            assert item.session_count == 2
            assert item.active_sessions == 1

        _run(_run_test())

    def test_metrics_response_shape(self, monkeypatch, tmp_path):
        """Verify the full response model shape has all expected fields."""
        store = _make_store(tmp_path)

        async def _run_test():
            await _register_session(store, "s1", agent_type="analyst")
            await _add_usage(store, "s1", 10, 5)

            monkeypatch.setattr(
                "tldw_Server_API.app.api.v1.endpoints.admin.admin_acp_agents.get_acp_session_store",
                mock.AsyncMock(return_value=store),
            )
            from tldw_Server_API.app.api.v1.endpoints.admin.admin_acp_agents import (
                get_acp_agent_metrics,
            )

            resp = await get_acp_agent_metrics()
            item = resp.items[0]
            # All schema fields present
            assert hasattr(item, "agent_type")
            assert hasattr(item, "session_count")
            assert hasattr(item, "active_sessions")
            assert hasattr(item, "total_prompt_tokens")
            assert hasattr(item, "total_completion_tokens")
            assert hasattr(item, "total_tokens")
            assert hasattr(item, "total_messages")
            assert hasattr(item, "last_used_at")
            assert hasattr(item, "total_estimated_cost_usd")

        _run(_run_test())


# ===========================================================================
# 3. PATCH /admin/acp/sessions/{session_id}/budget
# ===========================================================================


class TestACPSessionBudget:
    """Tests for the PATCH budget endpoint handler."""

    def test_set_budget_on_existing_session(self, monkeypatch, tmp_path):
        """Setting a budget returns updated fields."""
        store = _make_store(tmp_path)

        async def _run_test():
            await _register_session(store, "s1")

            monkeypatch.setattr(
                "tldw_Server_API.app.api.v1.endpoints.admin.admin_acp_agents.get_acp_session_store",
                mock.AsyncMock(return_value=store),
            )
            from tldw_Server_API.app.api.v1.endpoints.admin.admin_acp_agents import (
                admin_set_session_budget,
            )
            from tldw_Server_API.app.api.v1.schemas.agent_client_protocol import (
                ACPSessionBudgetRequest,
            )

            body = ACPSessionBudgetRequest(token_budget=5000, auto_terminate_at_budget=True)
            resp = await admin_set_session_budget("s1", body)
            assert resp.session_id == "s1"
            assert resp.token_budget == 5000
            assert resp.auto_terminate_at_budget is True
            assert resp.budget_exhausted is False
            assert resp.total_tokens == 0
            assert resp.budget_remaining == 5000

        _run(_run_test())

    def test_set_budget_null_removes_budget(self, monkeypatch, tmp_path):
        """Setting token_budget to None removes the budget (unlimited)."""
        store = _make_store(tmp_path)

        async def _run_test():
            await _register_session(store, "s1", token_budget=1000, auto_terminate=True)

            monkeypatch.setattr(
                "tldw_Server_API.app.api.v1.endpoints.admin.admin_acp_agents.get_acp_session_store",
                mock.AsyncMock(return_value=store),
            )
            from tldw_Server_API.app.api.v1.endpoints.admin.admin_acp_agents import (
                admin_set_session_budget,
            )
            from tldw_Server_API.app.api.v1.schemas.agent_client_protocol import (
                ACPSessionBudgetRequest,
            )

            body = ACPSessionBudgetRequest(token_budget=None, auto_terminate_at_budget=False)
            resp = await admin_set_session_budget("s1", body)
            assert resp.token_budget is None
            assert resp.budget_remaining is None

        _run(_run_test())

    def test_set_budget_nonexistent_session_404(self, monkeypatch, tmp_path):
        """Patching budget on a missing session raises 404."""
        store = _make_store(tmp_path)

        async def _run_test():
            monkeypatch.setattr(
                "tldw_Server_API.app.api.v1.endpoints.admin.admin_acp_agents.get_acp_session_store",
                mock.AsyncMock(return_value=store),
            )
            from tldw_Server_API.app.api.v1.endpoints.admin.admin_acp_agents import (
                admin_set_session_budget,
            )
            from tldw_Server_API.app.api.v1.schemas.agent_client_protocol import (
                ACPSessionBudgetRequest,
            )
            from fastapi import HTTPException

            body = ACPSessionBudgetRequest(token_budget=1000)
            with pytest.raises(HTTPException) as exc_info:
                await admin_set_session_budget("nonexistent", body)
            assert exc_info.value.status_code == 404

        _run(_run_test())

    def test_budget_remaining_reflects_usage(self, monkeypatch, tmp_path):
        """budget_remaining decreases as tokens are consumed."""
        store = _make_store(tmp_path)

        async def _run_test():
            await _register_session(store, "s1", token_budget=1000, auto_terminate=False)
            await _add_usage(store, "s1", 200, 100)  # 300 total

            monkeypatch.setattr(
                "tldw_Server_API.app.api.v1.endpoints.admin.admin_acp_agents.get_acp_session_store",
                mock.AsyncMock(return_value=store),
            )
            from tldw_Server_API.app.api.v1.endpoints.admin.admin_acp_agents import (
                admin_set_session_budget,
            )
            from tldw_Server_API.app.api.v1.schemas.agent_client_protocol import (
                ACPSessionBudgetRequest,
            )

            body = ACPSessionBudgetRequest(token_budget=1000, auto_terminate_at_budget=False)
            resp = await admin_set_session_budget("s1", body)
            assert resp.total_tokens == 300
            assert resp.budget_remaining == 700

        _run(_run_test())


# ===========================================================================
# 4. Budget enforcement — check_and_enforce_budget()
# ===========================================================================


class TestBudgetEnforcement:
    """Tests for ACPSessionStore.check_and_enforce_budget()."""

    def test_no_budget_set_returns_false(self, tmp_path):
        """Session with no budget is never terminated."""
        store = _make_store(tmp_path)

        async def _run_test():
            await _register_session(store, "s1", token_budget=None)
            await _add_usage(store, "s1", 999999, 999999)
            terminated = await store.check_and_enforce_budget("s1")
            assert terminated is False

        _run(_run_test())

    def test_under_budget_returns_false(self, tmp_path):
        """Session under budget is not terminated."""
        store = _make_store(tmp_path)

        async def _run_test():
            await _register_session(store, "s1", token_budget=1000, auto_terminate=True)
            await _add_usage(store, "s1", 100, 50)  # 150 total
            terminated = await store.check_and_enforce_budget("s1")
            assert terminated is False
            # Session should still be active
            rec = await store.get_session("s1")
            assert rec.status == "active"

        _run(_run_test())

    def test_at_budget_terminates(self, tmp_path):
        """Session at exactly the budget triggers termination."""
        store = _make_store(tmp_path)

        async def _run_test():
            await _register_session(store, "s1", token_budget=300, auto_terminate=True)
            await _add_usage(store, "s1", 200, 100)  # 300 total == budget
            terminated = await store.check_and_enforce_budget("s1")
            assert terminated is True
            rec = await store.get_session("s1")
            assert rec.status == "closed"
            assert rec.budget_exhausted is True

        _run(_run_test())

    def test_above_budget_terminates(self, tmp_path):
        """Session exceeding budget triggers termination."""
        store = _make_store(tmp_path)

        async def _run_test():
            await _register_session(store, "s1", token_budget=100, auto_terminate=True)
            await _add_usage(store, "s1", 200, 100)  # 300 > 100
            terminated = await store.check_and_enforce_budget("s1")
            assert terminated is True
            rec = await store.get_session("s1")
            assert rec.budget_exhausted is True

        _run(_run_test())

    def test_auto_terminate_disabled_does_not_close(self, tmp_path):
        """Exceeding budget with auto_terminate_at_budget=False does not close."""
        store = _make_store(tmp_path)

        async def _run_test():
            await _register_session(store, "s1", token_budget=100, auto_terminate=False)
            await _add_usage(store, "s1", 200, 100)
            terminated = await store.check_and_enforce_budget("s1")
            assert terminated is False
            rec = await store.get_session("s1")
            assert rec.status == "active"

        _run(_run_test())

    def test_nonexistent_session_returns_false(self, tmp_path):
        """Budget check on a missing session returns False."""
        store = _make_store(tmp_path)

        async def _run_test():
            terminated = await store.check_and_enforce_budget("no-such-session")
            assert terminated is False

        _run(_run_test())

    def test_already_closed_session_not_terminated_again(self, tmp_path):
        """A closed session is not re-terminated."""
        store = _make_store(tmp_path)

        async def _run_test():
            await _register_session(store, "s1", token_budget=100, auto_terminate=True)
            await _add_usage(store, "s1", 200, 100)
            await store.close_session("s1")
            # Already closed; should return False
            terminated = await store.check_and_enforce_budget("s1")
            assert terminated is False

        _run(_run_test())
