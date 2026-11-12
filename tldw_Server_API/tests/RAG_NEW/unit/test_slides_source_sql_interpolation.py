"""Regression guard for TASK-13292.

`NotesDBRetriever.retrieve_slides_source_candidates_v1` builds its PostgreSQL SQL
and its SQLite SQL in two separate branches. The PostgreSQL branch was assigned
with a plain triple-quoted string while its body interpolated ``{formatted_text}``,
so PostgreSQL received a literal brace and raised a syntax error. The surrounding
``except Exception`` converted that into a generic RAGDatabaseError, so notes
contributed zero slides-source candidates on that backend, every time.

These tests capture the SQL actually handed to the driver and assert no
un-interpolated placeholder survives on either backend.
"""

from typing import Any

import pytest


from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import NotesDBRetriever

# Suite marker: these are fast, isolated regression guards.
pytestmark = pytest.mark.unit


class _EmptyCursor:
    """Minimal cursor satisfying _read_slides_source_candidate_rows."""

    description: Any = None

    def fetchone(self) -> Any:
        return None


class _CapturingChaChaDB:
    """Stand-in ChaChaNotes DB that records the SQL it is asked to execute."""

    def __init__(self, backend_type: BackendType) -> None:
        self.backend_type = backend_type
        self.captured_sql: str | None = None

    def execute_query(self, sql: str, params: Any, **_kwargs: Any) -> _EmptyCursor:
        self.captured_sql = sql
        return _EmptyCursor()


@pytest.mark.parametrize(
    "backend_type",
    [BackendType.POSTGRESQL, BackendType.SQLITE],
    ids=["postgres", "sqlite"],
)
async def test_slides_source_sql_has_no_uninterpolated_placeholder(
    backend_type: BackendType,
) -> None:
    db = _CapturingChaChaDB(backend_type)
    retriever = NotesDBRetriever(db_path=None, chacha_db=db)

    await retriever.retrieve_slides_source_candidates_v1(
        query="quarterly report",
        owner_user_id="user-1",
        top_k=5,
    )

    assert db.captured_sql is not None, "the retriever never executed a query"

    # The specific defect: the f-prefix was missing, so the placeholder name
    # reached the database verbatim.
    assert "{formatted_text}" not in db.captured_sql, (
        f"{backend_type.value}: SQL contains an un-interpolated {{formatted_text}} "
        "placeholder -- the assignment is missing its f-string prefix"
    )

    # Broader guard: no brace-delimited placeholder of any name should survive,
    # since this SQL uses f-string interpolation rather than str.format().
    assert "{" not in db.captured_sql, (
        f"{backend_type.value}: SQL still contains a literal '{{' -- "
        "PostgreSQL has no brace syntax and will raise a syntax error"
    )


async def test_slides_source_sql_actually_interpolates_the_source_expression() -> None:
    """The interpolated body, not just the absence of braces."""
    db = _CapturingChaChaDB(BackendType.POSTGRESQL)
    retriever = NotesDBRetriever(db_path=None, chacha_db=db)

    await retriever.retrieve_slides_source_candidates_v1(
        query="quarterly report",
        owner_user_id="user-1",
        top_k=5,
    )

    assert db.captured_sql is not None
    # formatted_text builds a '# <title>\n\n<content>' expression; its COALESCE on
    # n.content is the tail that must appear once interpolation has happened.
    assert "COALESCE(n.content" in db.captured_sql, (
        "the source-text expression was not interpolated into the PostgreSQL SQL"
    )


class _FailingChaChaDB:
    """Stand-in ChaChaNotes DB whose driver rejects every query."""

    backend_type = BackendType.POSTGRESQL

    def execute_query(self, sql: str, params: Any, **_kwargs: Any) -> Any:
        raise SyntaxError("syntax error at or near '{' -- SELECT secret_sql_text")


async def test_slides_source_db_error_names_backend_and_cause() -> None:
    """A failure must say which backend failed and why, without leaking the SQL."""
    from tldw_Server_API.app.core.RAG.exceptions import RAGDatabaseError

    retriever = NotesDBRetriever(db_path=None, chacha_db=_FailingChaChaDB())

    with pytest.raises(RAGDatabaseError) as exc_info:
        await retriever.retrieve_slides_source_candidates_v1(
            query="quarterly report",
            owner_user_id="user-1",
            top_k=5,
        )

    message = str(exc_info.value)
    assert "postgresql" in message
    assert "SyntaxError" in message
    assert "secret_sql_text" not in message
