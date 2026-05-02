import ast
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.RAG.rag_service import agentic_chunker, agentic_execution
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document


pytestmark = pytest.mark.unit


def test_agentic_execution_does_not_import_shell_module():
    source = Path(agentic_execution.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    assert all("agentic_chunker" not in module for module in imported_modules)


def test_agentic_chunker_reexports_core_config_and_toolbox():
    assert agentic_chunker.AgenticConfig is agentic_execution.AgenticConfig
    assert agentic_chunker.AgenticToolbox is agentic_execution.AgenticToolbox


def test_open_section_uses_core_structure_db_resolver_and_falls_back(monkeypatch):
    def raise_database_error(*args, **kwargs):
        raise DatabaseError("structure index unavailable")

    monkeypatch.setattr(agentic_execution, "_get_media_db_for_structure", raise_database_error)

    doc = Document(
        id="doc-1",
        content="Intro\nText\n# Methods\nMethod text\n# Results\nResult text",
        metadata={"media_id": 1, "title": "Paper"},
        source=DataSource.MEDIA_DB,
    )
    toolbox = agentic_execution.AgenticToolbox(
        [doc],
        agentic_execution.AgenticConfig(enable_section_index=True),
    )

    result = toolbox.open_section(doc, "Results")

    assert result is not None
    start, end = result
    assert "Result text" in doc.content[start:end]
