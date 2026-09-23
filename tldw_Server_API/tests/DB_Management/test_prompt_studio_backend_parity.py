"""Parity between the two Prompt Studio backend implementations (TASK-13318 AC2).

PromptStudioDatabase.py holds two parallel implementations --
_BackendPromptStudioDatabase and _SQLitePromptStudioDatabase -- sharing 60 methods,
behind a PromptStudioDatabase facade that forwards *args/**kwargs and so hides every
signature difference from mypy and every IDE. Drift had already happened: a missing
method (TASK-13290) and seven signature mismatches, with nothing to catch the next.

This is a RATCHET until the planned consolidation lands (one backend-neutral
implementation, following the core/DB_Management/media_db split). The remaining known
mismatches are frozen below with what each one means; a new mismatch fails, and fixing
a known one fails the stale check until it is removed from the list.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management import PromptStudioDatabase as psd

BACKEND = psd._BackendPromptStudioDatabase
SQLITE = psd._SQLitePromptStudioDatabase
FACADE = psd.PromptStudioDatabase

# name -> why it differs. The four latent PUBLIC mismatches were aligned; what is left
# is private and goes away with the consolidation.
KNOWN_SIGNATURE_DRIFT: dict[str, str] = {
    "__init__": "legitimate: the backend variant takes tenant/backend/config",
    "_format_test_case": "ARITY differs, (row) vs (cursor, row): wrong on one side if shared",
    "_row_to_dict": "row is optional on the backend, required on SQLite",
}

# Public methods that exist on one side by design, not by drift.
ONE_SIDED_PUBLIC: dict[str, str] = {
    "get_fts_column": "backend-only: PostgreSQL full-text column resolution",
    "row_to_dict": "SQLite-only public helper",
    "transaction": "SQLite-only: the backend variant delegates transactions to DatabaseBackend",
}


def _own_methods(cls: type) -> set[str]:
    return {name for name, value in vars(cls).items() if callable(value)}


def _signature(cls: type, name: str) -> tuple:
    params = inspect.signature(getattr(cls, name)).parameters.values()
    return tuple(
        (p.name, p.kind.name, p.default is not inspect.Parameter.empty)
        for p in params
        if p.name not in ("self", "cls")
    )


def _mismatched() -> set[str]:
    shared = _own_methods(BACKEND) & _own_methods(SQLITE)
    return {name for name in shared if _signature(BACKEND, name) != _signature(SQLITE, name)}


def test_no_new_signature_drift() -> None:
    new = sorted(_mismatched() - set(KNOWN_SIGNATURE_DRIFT))
    assert not new, (
        "These methods now have different signatures on the PostgreSQL and SQLite Prompt "
        "Studio backends. The facade forwards *args/**kwargs, so a caller will see a "
        f"TypeError on one backend only. Align them: {new}"
    )


def test_known_drift_list_is_not_stale() -> None:
    fixed = sorted(set(KNOWN_SIGNATURE_DRIFT) - _mismatched())
    assert not fixed, (
        f"These no longer differ -- good. Remove them from KNOWN_SIGNATURE_DRIFT: {fixed}"
    )


def test_public_methods_exist_on_both_backends() -> None:
    public_b = {n for n in _own_methods(BACKEND) if not n.startswith("_")}
    public_s = {n for n in _own_methods(SQLITE) if not n.startswith("_")}
    one_sided = sorted((public_b ^ public_s) - set(ONE_SIDED_PUBLIC))
    assert not one_sided, (
        "Public Prompt Studio methods present on only one backend (compare TASK-13290, "
        f"a missing list_optimizations): {one_sided}"
    )


def _facade_delegations() -> set[str]:
    """Attribute names the facade reaches through self._impl."""
    tree = ast.parse(Path(inspect.getsourcefile(psd)).read_text(encoding="utf-8"))
    facade = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PromptStudioDatabase")
    names: set[str] = set()
    for node in ast.walk(facade):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "_impl"
            # Loads only: the facade also ASSIGNS instance state onto _impl (for example
            # tenant_user_id), which is not something the backend class must define.
            and isinstance(node.ctx, ast.Load)
        ):
            names.add(node.attr)
    return names


@pytest.mark.parametrize("backend", [BACKEND, SQLITE], ids=["postgres", "sqlite"])
def test_every_facade_delegation_resolves_on_each_backend(backend: type) -> None:
    missing = sorted(name for name in _facade_delegations() if not hasattr(backend, name))
    assert not missing, (
        f"PromptStudioDatabase forwards these to {backend.__name__}, which lacks them: {missing}"
    )


def test_sqlite_get_prompt_include_deleted_and_bulk_client_id(tmp_path):
    """The two aligned signatures behave as on the backend, not just parse."""
    db = FACADE(str(tmp_path / "ps.db"), "default_client")
    try:
        project = db.create_project(name="p")
        prompt = db.create_prompt(project["id"], "pr")
        conn = db.get_connection()
        conn.execute("UPDATE prompt_studio_prompts SET deleted = 1 WHERE id = ?", (prompt["id"],))
        conn.commit()
        assert db.get_prompt(prompt["id"]) is None
        assert db.get_prompt(prompt["id"], include_deleted=True)["id"] == prompt["id"]

        created = db.create_bulk_test_cases(
            project["id"],
            [{"name": "a", "inputs": {}}, {"name": "b", "inputs": {}, "client_id": "row_client"}],
            client_id=None,
        )
        assert [c["client_id"] for c in created] == ["default_client", "row_client"]
        created = db.create_bulk_test_cases(project["id"], [{"name": "c", "inputs": {}}], client_id="call_client")
        assert created[0]["client_id"] == "call_client"
    finally:
        db.close()
