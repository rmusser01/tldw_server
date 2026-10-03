"""Per-user LLM-token and RAG-query quotas (spec 2 §4)."""

from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import usage_quota_deps
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.RAG.rag_service import transport
from tldw_Server_API.app.core.Usage import quota_checks

pytestmark = pytest.mark.unit


@pytest.fixture()
def rag_limit(monkeypatch: pytest.MonkeyPatch) -> dict:
    """A rag-queries limit and today's usage, both set by the test."""
    state = {"limit": None, "used": 0.0}

    async def _user_quota(_uid: int, _key: str) -> object:
        """The test's limit."""
        return state["limit"]

    async def _used(_uid: str, _category: str) -> float:
        """The test's usage."""
        return state["used"]

    monkeypatch.setattr(quota_checks, "user_quota", _user_quota)
    monkeypatch.setattr(quota_checks, "ledger_used_today", _used)
    return state


def _app() -> FastAPI:
    """A tiny app whose route carries the RAG quota dependency."""
    app = FastAPI()

    @app.get("/q", dependencies=[Depends(usage_quota_deps.require_rag_query_quota(1))])
    async def _q() -> dict:
        """The guarded route."""
        return {"ok": True}

    app.dependency_overrides[get_request_user] = lambda: User(id=7, username="u", email="u@x.test", is_active=True, is_admin=False)
    return app


def test_rag_dependency_402_when_daily_allowance_spent(rag_limit: dict) -> None:
    """Within the allowance passes; at the allowance the route returns 402 limit_exceeded with Retry-After."""
    client = TestClient(_app())
    rag_limit.update(limit=3, used=2.0)
    assert client.get("/q").status_code == 200
    rag_limit["used"] = 3.0
    resp = client.get("/q")
    assert resp.status_code == 402
    assert resp.json()["detail"]["category"] == "rag_queries_day"
    assert int(resp.headers["Retry-After"]) >= 1


async def test_transport_check_refuses_over_allowance(rag_limit: dict) -> None:
    """The MCP path (transport) raises PermissionError when the user's allowance is spent."""
    rag_limit.update(limit=1, used=1.0)
    with pytest.raises(PermissionError):
        await transport.enforce_rag_query_limit_for_org_context(current_user=SimpleNamespace(id=7), units=1)


async def test_transport_logs_a_per_user_row_without_an_org(monkeypatch: pytest.MonkeyPatch) -> None:
    """Usage is recorded per user even when no org resolves (gate the check, never the record)."""
    added: list = []

    class _Ledger:
        """Records ledger writes."""

        async def initialize(self) -> None:
            """No-op init."""

        async def add(self, entry: object) -> bool:
            """Record the entry."""
            added.append(entry)
            return True

    async def _no_org(**_kw: object) -> None:
        """No org context."""
        return None

    monkeypatch.setattr("tldw_Server_API.app.core.DB_Management.Resource_Daily_Ledger.ResourceDailyLedger", _Ledger)
    monkeypatch.setattr(transport, "resolve_org_id_for_rag_context", _no_org)
    await transport.log_rag_queries_for_org_context(current_user=SimpleNamespace(id=7), units=2)
    assert [(e.entity_scope, e.entity_value, e.category, e.units) for e in added] == [("user", "7", "rag_queries", 2)]


async def test_llm_tokens_decision_counts_the_month(monkeypatch: pytest.MonkeyPatch) -> None:
    """A month at its token allowance refuses an estimated request that would exceed it."""

    async def _limit(_uid: int, _key: str) -> int:
        """1000-token allowance."""
        return 1000

    async def _used(_uid: int) -> float:
        """900 tokens used."""
        return 900.0

    monkeypatch.setattr(quota_checks, "user_quota", _limit)
    monkeypatch.setattr(quota_checks, "llm_tokens_this_month", _used)
    allowed = await quota_checks.check_usage(7, "limits.llm_tokens_per_month", 100, lambda: quota_checks.llm_tokens_this_month(7))
    refused = await quota_checks.check_usage(7, "limits.llm_tokens_per_month", 101, lambda: quota_checks.llm_tokens_this_month(7))
    assert allowed.allowed is True and refused.allowed is False
