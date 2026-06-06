import os
import tempfile
import time
from typing import Any, Dict, Tuple

import pytest
from multiprocessing import Process, Manager

from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory


def _spec_from_db(db: PromptStudioDatabase) -> Dict[str, Any]:
    """Build a serializable spec for child processes to reconnect to the same DB."""
    spec: Dict[str, Any] = {}
    if db.backend_type == BackendType.SQLITE:
        # PromptsDatabase exposes db_path_str via base
        path = getattr(db, "db_path_str", None)
        if not path:
            # Fallback: try _impl chain
            impl = getattr(db, "_impl", None)
            path = getattr(impl, "db_path_str", None)
        spec = {"backend": "sqlite", "sqlite_path": path}
    else:
        backend = db.backend
        cfg = getattr(backend, "config", None)
        spec = {
            "backend": "postgres",
            "pg": {
                "host": cfg.pg_host,
                "port": cfg.pg_port,
                "database": cfg.pg_database,
                "user": cfg.pg_user,
                "password": cfg.pg_password,
            },
        }
    return spec


def _open_db_from_spec(spec: Dict[str, Any]) -> PromptStudioDatabase:
    if spec["backend"] == "sqlite":
        return PromptStudioDatabase(spec["sqlite_path"], client_id="mp-worker")
    pg = spec["pg"]
    cfg = DatabaseConfig(
        backend_type=BackendType.POSTGRESQL,
        pg_host=pg["host"],
        pg_port=int(pg["port"]),
        pg_database=pg["database"],
        pg_user=pg["user"],
        pg_password=pg["password"],
        connect_timeout=2,
    )
    be = DatabaseBackendFactory.create_backend(cfg)
    fd, temp_db_path = tempfile.mkstemp(prefix="prompt_studio_mp_worker_", suffix=".sqlite")
    os.close(fd)
    db = PromptStudioDatabase(db_path=temp_db_path, client_id="mp-worker", backend=be)
    setattr(db, "_temp_sqlite_path", temp_db_path)
    return db


def _worker_acquire_loop(spec: Dict[str, Any], out_ids):
    db = _open_db_from_spec(spec)
    idle_spins = 0
    try:
        while True:
            job = db.acquire_next_job()
            if not job:
                idle_spins += 1
                if idle_spins > 20:  # ~2s total with 0.1s sleeps
                    break
                time.sleep(0.1)
                continue
            idle_spins = 0
            out_ids.append(int(job["id"]))
            db.update_job_status(int(job["id"]), "completed", result={"worker": "mp-worker"})
            # Light delay to encourage interleaving
            time.sleep(0.01)
    finally:
        try:
            db.close()
        except Exception:
            _ = None
        temp_db_path = getattr(db, "_temp_sqlite_path", None)
        if temp_db_path:
            try:
                os.unlink(temp_db_path)
            except OSError:
                _ = None


@pytest.mark.integration
def test_parallel_acquire_distinct_jobs_multiprocessing(
    prompt_studio_dual_backend_db,
    tmp_path,
    pg_database_config,
):
    label, db = prompt_studio_dual_backend_db
    mp_db = None
    backend = None
    temp_db_path = None
    db_to_use = db
    if label == "sqlite":
        mp_db_path = tmp_path / "prompt_studio_mp.sqlite"
        mp_db = PromptStudioDatabase(str(mp_db_path), client_id="mp-session")
        db_to_use = mp_db
    else:
        backend = DatabaseBackendFactory.create_backend(pg_database_config)
        fd, temp_db_path = tempfile.mkstemp(prefix="prompt_studio_mp_session_", suffix=".sqlite")
        os.close(fd)
        mp_db = PromptStudioDatabase(db_path=temp_db_path, client_id="mp-session", backend=backend)
        db_to_use = mp_db

    try:
        # Prepare a moderate number of jobs
        total = 18
        for i in range(total):
            db_to_use.create_job(
                job_type="evaluation",
                entity_id=100 + i,
                payload={"i": i},
                priority=5,
            )

        spec = _spec_from_db(db_to_use)

        # Spawn a handful of workers
        with Manager() as manager:
            out_ids = manager.list()
            procs = [Process(target=_worker_acquire_loop, args=(spec, out_ids)) for _ in range(4)]
            try:
                for p in procs:
                    p.start()
                for p in procs:
                    p.join(timeout=10)
            except KeyboardInterrupt:
                # On interrupt, terminate all child processes promptly
                for p in procs:
                    if p.is_alive():
                        p.terminate()
                for p in procs:
                    try:
                        p.join(2)
                    except Exception:
                        _ = None
                raise
            # Ensure processes are terminated if hanging
            for p in procs:
                if p.is_alive():
                    p.terminate()
                    p.join(2)

            got = list(out_ids)
            assert len(got) == total, f"Expected {total} acquired jobs, got {len(got)} for backend {label}"
            assert len(set(got)) == total, "Duplicate job acquisitions detected across processes"
    finally:
        if mp_db is not None:
            try:
                mp_db.close()
            except Exception:
                _ = None
        if backend is not None:
            try:
                backend.get_pool().close_all()
            except Exception:
                _ = None
        if temp_db_path:
            try:
                os.unlink(temp_db_path)
            except OSError:
                _ = None
