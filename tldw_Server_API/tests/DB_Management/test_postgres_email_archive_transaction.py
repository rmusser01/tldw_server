"""Real PostgreSQL failures must not roll back accepted archive Media rows."""

import asyncio
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.app.core.Ingestion_Media_Processing import persistence


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ["python", "sql"])
async def test_native_failure_rolls_back_graph_without_losing_archive_media(
    pg_database_config, monkeypatch, failure_kind
):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    real_graph = MediaDatabase.upsert_email_message_graph

    def create(client_id, *, db_path):
        return MediaDatabase(db_path, client_id=client_id, backend=backend)

    def late_failure(db, **kwargs):
        real_graph(db, **kwargs)
        if failure_kind == "sql":
            db.execute_query("SELECT * FROM missing_synthetic_email_table")
        raise ValueError("synthetic failure after nested native writes")

    monkeypatch.setattr(persistence, "create_media_database", create)
    monkeypatch.setattr(persistence, "_is_email_native_persist_enabled", lambda: True)
    monkeypatch.setattr(MediaDatabase, "upsert_email_message_graph", late_failure)
    result = {
        "status": "Success",
        "content": "",
        "children": [
            {
                "status": "Success",
                "content": "Synthetic accepted body",
                "metadata": {
                    "title": "Synthetic subject",
                    "filename": "one.eml",
                    "email": {"subject": "Synthetic subject", "message_id": "<one@example.test>"},
                },
            }
        ],
    }
    try:
        with scoped_context(user_id=42):
            await persistence.persist_doc_item_and_children(
                final_result=result,
                form_data=SimpleNamespace(accept_mbox=True, keywords=[]),
                media_type="email",
                item_input_ref="synthetic.mbox",
                processing_filename="synthetic.mbox",
                chunk_options=None,
                path_kind="upload",
                db_path=":memory:",
                client_id="42",
                loop=asyncio.get_running_loop(),
                claims_context=None,
                email_tenant_id="42",
            )
            assert len(result["child_db_results"]) == 1
            assert backend.execute("SELECT COUNT(*) FROM Media").scalar == 1
            assert backend.execute("SELECT COUNT(*) FROM email_messages").scalar == 0
            assert backend.execute("SELECT COUNT(*) FROM email_sources").scalar == 0
    finally:
        backend.get_pool().close_all()
