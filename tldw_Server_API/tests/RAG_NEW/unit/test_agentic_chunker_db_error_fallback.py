import pytest

from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.RAG.rag_service import agentic_execution
from tldw_Server_API.app.core.RAG.rag_service.agentic_execution import AgenticConfig
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document


def test_open_section_falls_back_to_heuristics_on_database_error(monkeypatch):
    def raise_database_error(*args, **kwargs):
        raise DatabaseError("structure index unavailable")

    monkeypatch.setattr(agentic_execution, "_get_media_db_for_structure", raise_database_error)

    doc = Document(
        id="doc-1",
        content="Introduction\nOverview text\n# Results\nImportant result text\n# Conclusion",
        metadata={"media_id": 1, "title": "Paper"},
        source=DataSource.MEDIA_DB,
    )
    toolbox = agentic_execution.AgenticToolbox(
        [doc],
        AgenticConfig(enable_section_index=True),
    )

    section = toolbox.open_section(doc, "Results")

    assert section is not None
    start, end = section
    assert "Important result text" in doc.content[start:end]
