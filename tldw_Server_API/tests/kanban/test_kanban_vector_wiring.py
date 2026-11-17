# tldw_Server_API/tests/kanban/test_kanban_vector_wiring.py
# Description: Kanban vector indexing lives in core.Kanban and is injected into KanbanDB by the openers.
import os
import subprocess  # nosec B404 - fixed argv below
import sys
from pathlib import Path
from types import SimpleNamespace

from tldw_Server_API.app.api.v1.API_Deps.kanban_deps import _create_kanban_db
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.Kanban_DB import KanbanDB
from tldw_Server_API.app.core.Kanban.vector_search import create_kanban_vector_search
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.kanban_module import KanbanModule
from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import KanbanDBRetriever

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_importing_kanban_db_loads_neither_jobs_nor_embeddings() -> None:
    code = (
        "import sys\nimport tldw_Server_API.app.core.DB_Management.Kanban_DB\n"
        "print(sorted(m for m in sys.modules if m.startswith(("
        "'tldw_Server_API.app.core.Jobs', 'tldw_Server_API.app.core.Embeddings'))))\n"
    )
    result = subprocess.run(  # nosec B603 - fixed argv
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT), "TEST_MODE": "true"},
    )
    assert result.stdout.strip().splitlines()[-1] == "[]"  # 24 modules on dev today


def test_a_kanban_db_without_a_factory_has_no_vector_search(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    db = KanbanDB(db_path=str(DatabasePaths.get_kanban_db_path(1)), user_id="1")
    try:
        assert db.get_vector_search() is None
    finally:
        db.close()


def test_production_openers_wire_the_kanban_vector_index(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    path = str(DatabasePaths.get_kanban_db_path(1))
    retriever = KanbanDBRetriever(path, user_id="1")
    from_init = retriever.kanban_db
    assert from_init is not None
    retriever.close()  # drops kanban_db; the next retrieve() reopens through _get_db (database_retrievers.py:3307)
    dbs: list[KanbanDB] = [
        _create_kanban_db(1),
        from_init,
        retriever._get_db(),
        KanbanModule(ModuleConfig(name="kanban"))._open_db(SimpleNamespace(db_paths={"kanban": path}, user_id="1")),
    ]
    try:
        assert [db._vector_search_factory for db in dbs] == [create_kanban_vector_search] * 4
    finally:
        for db in dbs:
            db.close()
