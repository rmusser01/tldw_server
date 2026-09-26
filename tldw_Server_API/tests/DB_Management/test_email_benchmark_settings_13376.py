"""Benchmark evidence records the actual bounded PostgreSQL resource budget."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.media_db.runtime import email_benchmark_fixture as fixture
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context


def test_settings_evidence_excludes_unrequested_sensitive_fields():
    db = SimpleNamespace(
        backend_type=BackendType.POSTGRESQL,
        transaction=lambda: nullcontext(None),
        _fetchone_with_connection=lambda *_args: {
            "work_mem": "64MB", "shared_buffers": "128MB", "hash_mem_multiplier": "2",
            "max_parallel_workers_per_gather": "2", "row_security": "on",
            "password": "SYNTHETIC_SECRET_MUST_NOT_BE_EXPORTED",
        },
    )
    settings = fixture.describe_postgres_benchmark_settings(db)
    assert set(settings) == {
        "work_mem", "shared_buffers", "hash_mem_multiplier",
        "max_parallel_workers_per_gather", "row_security",
    }


def test_postgres_settings_refuse_other_backends():
    db = SimpleNamespace(backend_type=BackendType.SQLITE)
    with pytest.raises(ValueError, match="PostgreSQL"):
        fixture.describe_postgres_benchmark_settings(db)


@pytest.mark.integration
@pytest.mark.postgres
def test_settings_evidence_reads_live_transaction_memory_budget(pg_database_config):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(":memory:", client_id="42", backend=backend)
    try:
        with scoped_context(user_id=42):
            with db.transaction() as conn:
                backend.execute("SET LOCAL work_mem = '64MB'", connection=conn)
                settings = fixture.describe_postgres_benchmark_settings(db)
            assert settings["work_mem"] == "64MB"
            assert settings["row_security"] == "on"
            assert len(settings) == 5
    finally:
        db.close_connection()
        backend.get_pool().close_all()
