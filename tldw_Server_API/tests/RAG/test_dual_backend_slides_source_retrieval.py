"""Dual-backend coverage for the Notes slides-source retrieval path.

This is the coverage that would have caught TASK-13292.

That defect was a missing `f` prefix on the PostgreSQL branch's SQL, so the driver
received a literal `LENGTH({formatted_text})` and raised a syntax error, which the
surrounding `except Exception` converted into a generic RAGDatabaseError. Notes
contributed zero slides-source candidates on PostgreSQL, every time, from 2026-07-16
until it was fixed -- and nothing caught it, because the only tests touching these
methods replace them with `AsyncMock`
(`tests/RAG_NEW/unit/test_slides_source_retrieval_hardening.py:88`), so no SQL runs on
either backend.

`dual_backend_env` already existed and does exactly what is needed here; it was simply
used by only two files, neither covering Notes. These tests execute the real SQL
against a real ChaChaNotes database on both backends, so a per-backend SQL divergence
in this path fails loudly instead of degrading to an empty result.

Both parametrisations execute wherever a PostgreSQL is reachable -- including the
linux-312/313 CI shards, which already run a postgres service. Where no driver is
present the fixture skips the postgres case and the sqlite one still runs.

Verified to catch the real defect: with the `f` prefix reverted, the postgres case
fails with `RAGDatabaseError: Note source candidate retrieval failed` while the
sqlite case still passes -- exactly the asymmetry that hid the bug.
"""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import (
    NotesDBRetriever,
    RetrievalConfig,
)
from tldw_Server_API.tests.RAG.conftest import DualBackendEnv

pytestmark = pytest.mark.integration

_TITLE = "Quarterly Parity Report"
_BODY = "Revenue rose in the northern region during the second quarter."


def _owner_of(env: DualBackendEnv) -> str:
    """The owner predicate is asymmetric, so the owner id must be the DB's client_id.

    On PostgreSQL these queries bind `AND n.client_id = ?` to owner_user_id; on SQLite
    there is no owner predicate at all, because the deployment invariant is one
    ChaChaNotes file per user. Passing an arbitrary owner therefore passes on SQLite
    and returns nothing on PostgreSQL -- which this suite reproduced on first run, and
    which is separately recorded as review finding rag-12.
    """
    return str(env.chacha_db.client_id)


def _seed_note(env: DualBackendEnv) -> str:
    note_id = env.chacha_db.add_note(title=_TITLE, content=_BODY)
    assert note_id is not None, f"failed to seed a note on {env.label}"
    env.chacha_db.rebuild_full_text_indexes()
    return str(note_id)


def _retriever(env: DualBackendEnv) -> NotesDBRetriever:
    return NotesDBRetriever(
        db_path=env.chacha_db.db_path_str,
        config=RetrievalConfig(max_results=5),
        chacha_db=env.chacha_db,
    )


@pytest.mark.asyncio
async def test_slides_source_candidates_execute_on_both_backends(
    dual_backend_env: DualBackendEnv,
) -> None:
    """The query must actually run -- not raise, and not silently return nothing.

    Before TASK-13292 this raised RAGDatabaseError on PostgreSQL and returned [] on
    SQLite, so asserting only "did not raise" would still have passed on SQLite. The
    non-empty assertion is what makes this meaningful on both.
    """
    env = dual_backend_env
    _seed_note(env)

    documents = await _retriever(env).retrieve_slides_source_candidates_v1(
        query="quarterly",
        owner_user_id=_owner_of(env),
        top_k=5,
    )

    assert documents, (
        f"Notes contributed zero slides-source candidates on {env.label}. A per-backend "
        "SQL divergence here degrades to an empty result rather than an error, which is "
        "how TASK-13292 stayed live for two months."
    )


@pytest.mark.asyncio
async def test_slides_source_projection_executes_on_both_backends(
    dual_backend_env: DualBackendEnv,
) -> None:
    """The sibling projection method uses .format() rather than an f-string.

    Three methods in this file splice their source expression three different ways
    (f-string, f-string, .format()). Exercising the projection on both backends keeps
    the third convention honest too.
    """
    env = dual_backend_env
    _seed_note(env)
    retriever = _retriever(env)

    candidates = await retriever.retrieve_slides_source_candidates_v1(
        query="quarterly",
        owner_user_id=_owner_of(env),
        top_k=5,
    )
    assert candidates, f"no candidates to project on {env.label}"

    documents = await retriever.project_slides_source_documents_v1(
        projections=[(candidates[0], 4000)],
        owner_user_id=_owner_of(env),
    )

    assert documents, f"projection returned nothing on {env.label} for a seeded note"


@pytest.mark.asyncio
async def test_slides_source_sql_carries_no_unsubstituted_placeholder(
    dual_backend_env: DualBackendEnv,
) -> None:
    """Catch the exact TASK-13292 shape at the driver boundary, per backend.

    The unit test in tests/RAG_NEW asserts this against a captured SQL string; here it
    is asserted against the statement a real backend is actually handed.
    """
    env = dual_backend_env
    _seed_note(env)

    seen: list[str] = []
    original = env.chacha_db.execute_query

    def _recording(sql, *args, **kwargs):
        seen.append(sql)
        return original(sql, *args, **kwargs)

    env.chacha_db.execute_query = _recording  # type: ignore[method-assign]
    try:
        await _retriever(env).retrieve_slides_source_candidates_v1(
            query="quarterly",
            owner_user_id=_owner_of(env),
            top_k=5,
        )
    finally:
        env.chacha_db.execute_query = original  # type: ignore[method-assign]

    assert seen, f"no query was executed on {env.label}"
    offenders = [s for s in seen if "{" in s]
    assert not offenders, (
        f"{env.label}: SQL reached the driver with an un-interpolated placeholder -- "
        f"{offenders[0][:160]!r}"
    )
