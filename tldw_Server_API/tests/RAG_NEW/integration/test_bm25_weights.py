import os
import sqlite3
import tempfile
from datetime import datetime

import pytest

from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import MediaDBRetriever, RetrievalConfig


def _setup_sqlite_media_db(db_path: str):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    # base tables
    cur.execute(
        """
        CREATE TABLE media (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            content TEXT,
            type TEXT,
            url TEXT,
            ingestion_date TEXT,
            transcription_model TEXT
        )
        """
    )
    # FTS5 external-content table
    cur.execute(
        """
        CREATE VIRTUAL TABLE media_fts USING fts5(
            title,
            content,
            content='media',
            content_rowid='id'
        )
        """
    )
    # Insert two contrasting rows
    now = datetime.utcnow().isoformat()
    cur.execute(
        "INSERT INTO media(title, content, type, url, ingestion_date, transcription_model) VALUES(?,?,?,?,?,?)",
        ("alpha beta", "lorem ipsum dolor", "doc", "local://1", now, "none"),
    )
    id1 = cur.lastrowid
    cur.execute(
        "INSERT INTO media_fts(rowid, title, content) VALUES(?,?,?)",
        (id1, "alpha beta", "lorem ipsum dolor"),
    )
    cur.execute(
        "INSERT INTO media(title, content, type, url, ingestion_date, transcription_model) VALUES(?,?,?,?,?,?)",
        ("lorem ipsum", "alpha beta appears in content", "doc", "local://2", now, "none"),
    )
    id2 = cur.lastrowid
    cur.execute(
        "INSERT INTO media_fts(rowid, title, content) VALUES(?,?,?)",
        (id2, "lorem ipsum", "alpha beta appears in content"),
    )
    con.commit()
    con.close()


@pytest.mark.integration
def test_bm25_title_vs_content_weights_flip_order(monkeypatch):
     # Ensure raw SQL fallback is allowed (not production)
    os.environ["tldw_production"] = "false"

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "media.db")
        _setup_sqlite_media_db(db_path)

        # Helper to run retrieval with specific weights via settings monkeypatch
        def run_with_weights(title_w: float, content_w: float):
            # Patch settings consumed inside retriever
            from tldw_Server_API.app.core import config as cfg
            cfg.settings.setdefault("RAG", {})
            cfg.settings["RAG"]["fts_title_weight"] = float(title_w)
            cfg.settings["RAG"]["fts_content_weight"] = float(content_w)

            retr = MediaDBRetriever(db_path=db_path, config=RetrievalConfig(max_results=5))
            try:
                import asyncio
                docs = asyncio.run(retr.retrieve(query="alpha"))
                assert docs, "Expected at least one result"
                return [str(d.id) for d in docs]
            finally:
                retr.close()

        try:
            # Title weight high
            order_title = run_with_weights(5.0, 0.1)
            # Content weight high
            order_content = run_with_weights(0.1, 5.0)

            # Expect that the ordering flips with strong weight changes
            assert order_title[0] != order_content[0], f"Expected different top result; got {order_title[0]} vs {order_content[0]}"
        finally:
            from tldw_Server_API.app.core.DB_Management.backends.factory import reset_managed_sqlite_backends

            reset_managed_sqlite_backends(sqlite_targets=[db_path])
