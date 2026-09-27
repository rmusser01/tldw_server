"""Workflow adapters must act as the account running them, not as user 1.

Three adapters resolved their store from a process-global default pinned to
SINGLE_USER_FIXED_ID, ignoring both the workflow context and AUTH_MODE. Every
account's workflow steps therefore read and wrote user 1's data. This leaks
identically on SQLite and PostgreSQL and needs no misconfiguration.

The cache case is the sharpest: the collection name was a constant and the
entry id came straight from step config, so one workflow could read another
account's cached step output by naming the same key.

Single-user mode legitimately falls back to the fixed id; that is what
resolve_user_id_value is for. The defect was bypassing it entirely.
"""

import pytest

from tldw_Server_API.app.core.AuthNZ import User_DB_Handling
from tldw_Server_API.app.core.Workflows.adapters import _common
from tldw_Server_API.app.core.Workflows.adapters.control import state as state_adapter
from tldw_Server_API.app.core.Workflows.adapters.rag import query as query_adapter
from tldw_Server_API.app.core.Workflows.adapters.rag import search as search_adapter

pytestmark = pytest.mark.unit


class _Collection:
    def get(self, **_kw):
        return {"ids": []}

    def upsert(self, **_kw):
        pass

    def delete(self, **_kw):
        pass


@pytest.fixture()
def chroma_for(monkeypatch):
    """Record which account each adapter opens a Chroma store for."""
    seen: list[str] = []

    class _Manager:
        def get_or_create_collection(self, collection_name):
            return _Collection()

    def _fake(user_id):
        seen.append(str(user_id))
        return _Manager()

    monkeypatch.setattr(_common, "workflow_chroma_manager", _fake)
    monkeypatch.setattr(state_adapter, "workflow_chroma_manager", _fake)
    monkeypatch.setattr(query_adapter, "workflow_chroma_manager", _fake)
    return seen


@pytest.fixture()
def multi_user(monkeypatch):
    """Force multi-user mode so no single-user fallback can apply."""
    monkeypatch.setattr(User_DB_Handling, "is_single_user_mode", lambda: False)


async def test_cache_result_opens_the_store_of_the_running_account(chroma_for):
    """The regression: this landed in user 1's store regardless of the account."""
    out = await state_adapter.run_cache_result_adapter(
        {"key": "quarterly_report", "action": "set", "data": {"secret": 1}},
        {"user_id": "77"},
    )

    assert out.get("error") is None
    assert chroma_for == ["77"]


async def test_semantic_cache_check_opens_the_store_of_the_running_account(chroma_for):
    await query_adapter.run_semantic_cache_check_adapter(
        {"query": "hello"}, {"user_id": "77"}
    )

    assert chroma_for == ["77"]


async def test_cache_result_refuses_when_multi_user_and_no_account(chroma_for, multi_user):
    out = await state_adapter.run_cache_result_adapter(
        {"key": "quarterly_report", "action": "get"}, {}
    )

    assert out["error"] == "missing_user_id"
    assert chroma_for == [], "no account means no store is opened at all"


async def test_semantic_cache_check_refuses_when_multi_user_and_no_account(
    chroma_for, multi_user
):
    out = await query_adapter.run_semantic_cache_check_adapter({"query": "hello"}, {})

    assert out["error"] == "missing_user_id"
    assert out["cache_hit"] is False
    assert chroma_for == []


async def test_rag_search_refuses_when_multi_user_and_no_account(multi_user):
    """Previously searched user 1's media library for whoever ran the step."""
    out = await search_adapter.run_rag_search_adapter({"query": "hello"}, {})

    assert out["metadata"]["error"] == "missing_user_id"
    assert out["documents"] == []


def test_the_user_one_pinned_singleton_is_gone():
    """Deleted rather than fixed, so nothing can reach it again."""
    import tldw_Server_API.app.core.Embeddings.ChromaDB_Library as chroma

    assert not hasattr(chroma, "get_default_chroma_manager")
    assert not hasattr(chroma, "get_default_chroma_client")
