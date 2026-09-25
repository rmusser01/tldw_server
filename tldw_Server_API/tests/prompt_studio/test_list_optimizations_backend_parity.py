"""Regression guard for TASK-13290.

`PromptStudioDatabase` selects `_SQLitePromptStudioDatabase` unless the backend is
PostgreSQL, so SQLite is the default. But `list_optimizations` was implemented only on
`_BackendPromptStudioDatabase`, while the facade delegates to `self._impl`
unconditionally. On the default backend the call raised `AttributeError`, the
endpoint's `_OPTIMIZATION_NONCRITICAL_EXCEPTIONS` tuple swallowed it, and the route
returned HTTP 500 "Failed to list optimizations" for every request on every project.

No test caught it because both endpoint tests substitute a stub class that defines
`list_optimizations(self, *_args, **_kwargs)`. The stub is the reason the gap was
invisible, so this suite deliberately drives a **real** `_SQLitePromptStudioDatabase`
against a temp file.

The root cause is the file's dual-backend triplication: 59 method names implemented
twice with parallel SQL, plus a third `*args/**kwargs` delegating facade that erases
the asymmetry from mypy and from every IDE. The parity test below is the cheap guard
against the next missing method; the decomposition is tracked separately.
"""

from __future__ import annotations

import ast
import pathlib
import tempfile
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import (
    PromptStudioDatabase,
)

pytestmark = pytest.mark.unit

_SOURCE = pathlib.Path(
    "tldw_Server_API/app/core/DB_Management/PromptStudioDatabase.py"
)

# Asymmetries that are deliberate rather than accidental. Anything else appearing on
# one backend class and not the other should fail the parity test below.
_KNOWN_BACKEND_ONLY = {
    "_apply_postgres_migrations",
    "_build_test_case_filters",
    "_convert_sqlite_schema_to_postgres_statements",
    "_ensure_extensions",
    "_ensure_postgres_fts",
    "_initialize_schema_postgres",
    "_transform_sqlite_statement_for_postgres",
    "get_fts_column",
}


@pytest.fixture
def db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        path = Path(tmp.name)
    yield path
    for target in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        try:
            target.unlink()
        except OSError:
            pass


def _class_methods(class_name: str) -> set[str]:
    tree = ast.parse(_SOURCE.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                m.name
                for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError(f"class {class_name} not found")


def test_list_optimizations_exists_on_the_default_backend(db_path) -> None:
    """The endpoint's failure mode: AttributeError swallowed into a 500."""
    db = PromptStudioDatabase(str(db_path), "parity-client")
    try:
        # ADR-051: the facade serves list_optimizations through the shared
        # repository on both backends rather than delegating to self._impl.
        assert callable(getattr(db, "list_optimizations", None)), (
            "list_optimizations is not served on the default backend, so the "
            "optimizations endpoint returns 500 on every request"
        )
        db.list_optimizations(page=1, per_page=1)
    finally:
        db.close_connection()


def test_list_optimizations_returns_the_documented_shape(db_path) -> None:
    """Executed against a real SQLite file -- no stub, which is what hid the bug."""
    db = PromptStudioDatabase(str(db_path), "parity-client")
    try:
        result = db.list_optimizations(page=1, per_page=20)

        assert isinstance(result, dict)
        assert "optimizations" in result and isinstance(result["optimizations"], list)
        assert "pagination" in result
        pagination = result["pagination"]
        assert pagination["page"] == 1
        assert pagination["per_page"] == 20
        assert pagination["total"] == 0
        assert pagination["total_pages"] == 0
    finally:
        db.close_connection()


def test_list_optimizations_validates_pagination(db_path) -> None:
    """Both implementations must reject the same invalid input."""
    from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import InputError

    db = PromptStudioDatabase(str(db_path), "parity-client")
    try:
        with pytest.raises(InputError):
            db.list_optimizations(page=0)
        with pytest.raises(InputError):
            db.list_optimizations(per_page=0)
    finally:
        db.close_connection()


def test_the_two_backend_classes_expose_the_same_public_surface() -> None:
    """The cheap guard against the next missing method.

    A `*args/**kwargs` facade over two independently-written classes means a method
    present on one and absent on the other is invisible to mypy and to every IDE, and
    only surfaces as a 500 in production. 59 names are implemented twice in this file;
    this pins that they stay in step.
    """
    backend = _class_methods("_BackendPromptStudioDatabase")
    sqlite = _class_methods("_SQLitePromptStudioDatabase")

    public_backend = {n for n in backend if not n.startswith("_")}
    public_sqlite = {n for n in sqlite if not n.startswith("_")}

    backend_only = (public_backend - public_sqlite) - _KNOWN_BACKEND_ONLY
    assert not backend_only, (
        "public methods exist on the PostgreSQL class but not the SQLite one, so they "
        f"raise AttributeError on the default backend: {sorted(backend_only)}"
    )
