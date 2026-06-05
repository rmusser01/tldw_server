import threading

import pytest

import tldw_Server_API.app.core.DB_Management.Evaluations_DB as eval_db_module
from tldw_Server_API.app.core.DB_Management.Evaluations_DB import EvaluationsDatabase


@pytest.mark.unit
def test_normalize_postgres_url_prefers_psycopg_when_psycopg2_missing(monkeypatch):
    def _fake_find_spec(name: str):
        if name == "psycopg":
            return object()
        if name == "psycopg2":
            return None
        return None

    monkeypatch.setattr(eval_db_module.importlib.util, "find_spec", _fake_find_spec)

    normalized = eval_db_module._normalize_sqlalchemy_postgres_url(
        "postgresql://user:pass@localhost:5432/evals"
    )
    assert normalized.startswith("postgresql+psycopg://")


@pytest.mark.unit
def test_init_abtest_store_falls_back_when_sqlalchemy_driver_missing(monkeypatch, tmp_path):
    db = EvaluationsDatabase.__new__(EvaluationsDatabase)
    db.db_path = str(tmp_path / "evals.db")
    db._backend = None
    db._uses_shared_content_backend = False
    db._backend_refresh_suspended = False
    db._local = threading.local()
    db._abtest_store = "sentinel"

    monkeypatch.setenv("EVALS_ABTEST_PERSISTENCE", "sqlalchemy")

    import tldw_Server_API.app.core.Evaluations.embeddings_abtest_repository as repo_module

    def _raise_driver_error(*_args, **_kwargs):
        raise ModuleNotFoundError("No module named 'psycopg2'")

    monkeypatch.setattr(repo_module, "get_embeddings_abtest_store", _raise_driver_error)

    EvaluationsDatabase._init_abtest_store(db)
    assert db._abtest_store is None
