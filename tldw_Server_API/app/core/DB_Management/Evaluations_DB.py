# Evaluations_DB.py - Database management for OpenAI-compatible evaluations API
"""
Database operations for evaluations, runs, and datasets.

Provides CRUD operations and query methods for:
- Evaluation definitions
- Evaluation runs
- Datasets
"""

import json
import importlib.util
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger

from tldw_Server_API.app.api.v1.schemas.evaluation_recipe_schemas import (
    ConfidenceSummary,
    RecommendationSlot,
    RecipeRunRecord,
    ReviewState,
)
from tldw_Server_API.app.api.v1.schemas.evaluation_schemas_unified import RunStatus
from tldw_Server_API.app.core.config import load_comprehensive_config

# Backend abstraction (optional) for PostgreSQL support
from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseBackend,
    QueryResult,
)
from tldw_Server_API.app.core.DB_Management.backends.query_utils import (
    prepare_backend_many_statement,
    prepare_backend_statement,
)
from tldw_Server_API.app.core.DB_Management.content_backend import (
    backend_target_key,
    get_content_backend,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

_EVAL_DB_NONCRITICAL_EXCEPTIONS = (
    ImportError,
    ModuleNotFoundError,
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
    sqlite3.Error,
    json.JSONDecodeError,
)


def _normalize_sqlalchemy_postgres_url(db_url: str) -> str:
    """Prefer psycopg3 SQLAlchemy URLs when psycopg2 is unavailable."""
    if not isinstance(db_url, str):
        return db_url

    lowered = db_url.lower()
    if not (lowered.startswith("postgresql://") or lowered.startswith("postgres://")):
        return db_url
    if lowered.startswith("postgresql+"):
        return db_url

    try:
        has_psycopg = importlib.util.find_spec("psycopg") is not None
        has_psycopg2 = importlib.util.find_spec("psycopg2") is not None
    except _EVAL_DB_NONCRITICAL_EXCEPTIONS:
        return db_url

    if has_psycopg and not has_psycopg2:
        if lowered.startswith("postgresql://"):
            return f"postgresql+psycopg://{db_url[len('postgresql://') :]}"
        if lowered.startswith("postgres://"):
            return f"postgresql+psycopg://{db_url[len('postgres://') :]}"
    return db_url


class _BackendCursorAdapter:
    """Adapter exposing QueryResult via a cursor-like interface."""

    def __init__(self, result: QueryResult):
        self._result = result
        self._index = 0
        self.rowcount = result.rowcount
        self.lastrowid = result.lastrowid
        self.description = result.description

    def fetchall(self):
        return list(self._result.rows)

    def fetchone(self):
        if self._index >= len(self._result.rows):
            return None
        row = self._result.rows[self._index]
        self._index += 1
        return row

    def fetchmany(self, size: Optional[int] = None):
        if size is None or size <= 0:
            size = len(self._result.rows) - self._index
        end = min(self._index + size, len(self._result.rows))
        rows = self._result.rows[self._index:end]
        self._index = end
        return list(rows)

    def close(self):
        self._result = QueryResult(rows=[], rowcount=0)
        self.rowcount = 0
        self.lastrowid = None
        self.description = None


class _EvaluationsBackendCursor:
    """Cursor wrapper that routes SQL through the configured DatabaseBackend."""

    def __init__(self, db: "EvaluationsDatabase", connection: Any):
        self._db = db
        self._conn = connection
        self._adapter: Optional[_BackendCursorAdapter] = None
        self.rowcount: int = -1
        self.lastrowid: Optional[int] = None
        self.description = None

    def execute(self, query: str, params: Optional[Any] = None):
        prepared_query, prepared_params = self._db._prepare_backend_statement(query, params)
        result = self._db.backend.execute(prepared_query, prepared_params, connection=self._conn)
        self._adapter = _BackendCursorAdapter(result)
        self.rowcount = result.rowcount
        self.lastrowid = result.lastrowid
        self.description = result.description
        return self

    def executemany(self, query: str, params_list: list[Any]):
        prepared_query, prepared_params_list = self._db._prepare_backend_many_statement(query, params_list)
        result = self._db.backend.execute_many(prepared_query, prepared_params_list, connection=self._conn)
        self._adapter = _BackendCursorAdapter(result)
        self.rowcount = result.rowcount
        self.lastrowid = result.lastrowid
        self.description = result.description
        return self

    def fetchone(self):
        if not self._adapter:
            return None
        row = self._adapter.fetchone()
        return dict(row) if row else None

    def fetchall(self):
        if not self._adapter:
            return []
        return [dict(r) for r in self._adapter.fetchall()]

    def fetchmany(self, size: Optional[int] = None):
        if not self._adapter:
            return []
        return [dict(r) for r in self._adapter.fetchmany(size)]

    def close(self):
        if self._adapter:
            self._adapter.close()
        self._adapter = None
        self.rowcount = -1
        self.lastrowid = None
        self.description = None


class _EvaluationsBackendConnection:
    """Connection shim exposing sqlite-like helpers for backend usage."""

    def __init__(self, db: "EvaluationsDatabase", connection: Any):
        self._db = db
        self._conn = connection
        self.row_factory = None

    def cursor(self) -> _EvaluationsBackendCursor:
        return _EvaluationsBackendCursor(self._db, self._conn)

    def execute(self, query: str, params: Optional[Any] = None):
        return self.cursor().execute(query, params)

    def executemany(self, query: str, params_list: list[Any]):
        return self.cursor().executemany(query, params_list)

    def commit(self) -> None:
        try:
            self._conn.commit()
        except _EVAL_DB_NONCRITICAL_EXCEPTIONS as exc:
            logger.error(f"Failed to commit evaluations backend connection: {exc}", exc_info=True)
            raise

    def rollback(self) -> None:
        try:
            self._conn.rollback()
        except _EVAL_DB_NONCRITICAL_EXCEPTIONS as exc:
            logger.error(f"Failed to rollback evaluations backend connection: {exc}", exc_info=True)
            raise

    def close(self) -> None:
        return None

class EvaluationsDatabase:
    """Database manager for evaluations system (SQLite or PostgreSQL)."""

    _bootstrapped_backend_targets: set[str] = set()

    def __init__(self, db_path: Optional[str], *, backend: Optional[DatabaseBackend] = None):
        # Default to per-user evaluations DB path when not provided
        if not db_path:
            try:
                uid = DatabasePaths.get_single_user_id()
                db_path = str(DatabasePaths.get_evaluations_db_path(uid))
            except _EVAL_DB_NONCRITICAL_EXCEPTIONS:
                db_path = "Databases/evaluations.db"
        self.db_path = db_path
        # Resolve backend (content backend by default)
        self._backend: Optional[DatabaseBackend] = backend
        self._uses_shared_content_backend = False
        self._local = threading.local()
        self._backend_refresh_suspended = False
        if self._backend is None:
            self._backend, self._uses_shared_content_backend = self._resolve_backend()

        self._abtest_store = None

        if self.backend_type == BackendType.SQLITE:
            self._initialize_database()
            self._apply_migrations()
        else:
            if self._backend is None:
                raise RuntimeError("Evaluations backend is not configured")
            self._ensure_bootstrap_for_backend(self._backend)
        self._init_abtest_store()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections (backend-aware)."""
        if self.backend_type == BackendType.SQLITE:
            # Register explicit adapters to avoid deprecated defaults on Python 3.12+
            with suppress(_EVAL_DB_NONCRITICAL_EXCEPTIONS):
                sqlite3.register_adapter(datetime, lambda d: d.isoformat(sep=" "))
            conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
            conn.row_factory = sqlite3.Row
            with suppress(_EVAL_DB_NONCRITICAL_EXCEPTIONS):
                conn.execute("PRAGMA foreign_keys = ON")
            try:
                yield conn
            finally:
                conn.close()
            return

        if self.backend is None:
            raise RuntimeError("Evaluations backend is not configured")
        backend = self.backend
        raw = backend.get_pool().get_connection()
        previous_backend = getattr(self._local, "backend_pin", None)
        self._local.backend_pin = backend
        try:
            yield _EvaluationsBackendConnection(self, raw)
        finally:
            with suppress(_EVAL_DB_NONCRITICAL_EXCEPTIONS):
                backend.get_pool().return_connection(raw)
            if previous_backend is None:
                with suppress(AttributeError):
                    del self._local.backend_pin
            else:
                self._local.backend_pin = previous_backend

    @property
    def backend(self) -> Optional[DatabaseBackend]:
        pinned_backend = getattr(self._local, "backend_pin", None)
        if pinned_backend is not None:
            return pinned_backend
        if not self._uses_shared_content_backend:
            return self._backend
        if self._backend_refresh_suspended:
            return self._backend
        current_key = self._backend_target_key(self._backend)
        refreshed_backend, uses_shared_backend = self._resolve_backend()
        self._uses_shared_content_backend = uses_shared_backend
        self._backend = refreshed_backend
        refreshed_key = self._backend_target_key(refreshed_backend)
        if (
            uses_shared_backend
            and refreshed_backend is not None
            and refreshed_backend.backend_type == BackendType.POSTGRESQL
            and refreshed_key != current_key
        ):
            self._ensure_bootstrap_for_backend(refreshed_backend)
            self._abtest_store = None
            self._init_abtest_store()
        return self._backend

    @property
    def backend_type(self) -> BackendType:
        backend = self.backend
        return backend.backend_type if backend else BackendType.SQLITE

    def _resolve_backend(self) -> tuple[Optional[DatabaseBackend], bool]:
        try:
            cfg = load_comprehensive_config()
        except _EVAL_DB_NONCRITICAL_EXCEPTIONS:
            return None, False

        try:
            backend = get_content_backend(cfg)
        except _EVAL_DB_NONCRITICAL_EXCEPTIONS:
            return None, False
        if backend is None:
            return None, False
        return backend, backend.backend_type == BackendType.POSTGRESQL

    def _backend_target_key(self, backend: Optional[DatabaseBackend]) -> str | None:
        return backend_target_key(backend)

    def _mark_backend_bootstrapped(self, backend: Optional[DatabaseBackend]) -> None:
        key = self._backend_target_key(backend)
        if key:
            type(self)._bootstrapped_backend_targets.add(key)

    def _ensure_bootstrap_for_backend(self, backend: DatabaseBackend) -> None:
        if backend.backend_type != BackendType.POSTGRESQL:
            return
        key = self._backend_target_key(backend)
        if key in type(self)._bootstrapped_backend_targets:
            return
        previous_suspend = self._backend_refresh_suspended
        self._backend_refresh_suspended = True
        try:
            self._backend = backend
            self._initialize_database_postgres()
        finally:
            self._backend_refresh_suspended = previous_suspend
        if key:
            type(self)._bootstrapped_backend_targets.add(key)

    # --- Backend helpers ---
    def _prepare_backend_statement(self, query: str, params: Optional[Any] = None) -> tuple[str, Optional[Any]]:
        if self.backend_type != BackendType.POSTGRESQL:
            return query, params
        return prepare_backend_statement(self.backend_type, query, params, apply_default_transform=True)

    def _prepare_backend_many_statement(self, query: str, params_list: list[Any]) -> tuple[str, list[Any]]:
        if self.backend_type != BackendType.POSTGRESQL:
            return query, params_list
        # Use shared utility to convert placeholders and normalize parameter lists consistently
        converted_query, converted_params = prepare_backend_many_statement(
            self.backend_type,
            query,
            params_list,
            apply_default_transform=True,
        )
        return converted_query, converted_params

    def _initialize_database(self):
        """Create database tables if they don't exist"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Evaluations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    eval_type TEXT NOT NULL,
                    eval_spec TEXT NOT NULL,
                    dataset_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    created_by TEXT,
                    metadata TEXT,
                    deleted_at TEXT NULL
                )
            """)

            # Evaluation runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    id TEXT PRIMARY KEY,
                    eval_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    target_model TEXT,
                    config TEXT,
                    progress TEXT,
                    results TEXT,
                    error_message TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    webhook_url TEXT,
                    usage TEXT,
                    FOREIGN KEY (eval_id) REFERENCES evaluations(id)
                )
            """)

            # Datasets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    samples TEXT NOT NULL,
                    sample_count INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    created_by TEXT,
                    metadata TEXT
                )
            """)

            # Recipe runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_recipe_runs (
                    run_id TEXT PRIMARY KEY,
                    recipe_id TEXT NOT NULL,
                    recipe_version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    review_state TEXT NOT NULL DEFAULT 'not_required',
                    dataset_snapshot_ref TEXT,
                    dataset_content_hash TEXT,
                    confidence_summary_json TEXT,
                    recommendation_slots_json TEXT,
                    metadata_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_recipe_run_children (
                    parent_run_id TEXT NOT NULL,
                    child_run_id TEXT NOT NULL,
                    child_order INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (parent_run_id, child_run_id),
                    FOREIGN KEY (parent_run_id) REFERENCES evaluation_recipe_runs(run_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS synthetic_eval_draft_samples (
                    sample_id TEXT PRIMARY KEY,
                    recipe_kind TEXT NOT NULL,
                    provenance TEXT NOT NULL,
                    review_state TEXT NOT NULL DEFAULT 'draft',
                    sample_payload_json TEXT NOT NULL,
                    sample_metadata_json TEXT,
                    source_kind TEXT,
                    created_by TEXT,
                    review_summary_json TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS synthetic_eval_review_actions (
                    action_id TEXT PRIMARY KEY,
                    sample_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reviewer_id TEXT,
                    notes TEXT,
                    action_payload_json TEXT,
                    resulting_review_state TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (sample_id) REFERENCES synthetic_eval_draft_samples(sample_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS synthetic_eval_promotions (
                    promotion_id TEXT PRIMARY KEY,
                    sample_id TEXT NOT NULL,
                    dataset_id TEXT,
                    dataset_snapshot_ref TEXT,
                    promoted_by TEXT,
                    promotion_reason TEXT,
                    promotion_metadata_json TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (sample_id) REFERENCES synthetic_eval_draft_samples(sample_id)
                )
            """)

            # Internal evaluations table (for tldw-specific evaluations)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS internal_evaluations (
                    evaluation_id TEXT PRIMARY KEY,
                    evaluation_type TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    input_data TEXT,
                    results TEXT,
                    metadata TEXT,
                    user_id TEXT,
                    status TEXT DEFAULT 'pending',
                    completed_at TEXT,
                    embedding_provider TEXT,
                    embedding_model TEXT
                )
            """)

            # Pipeline presets for RAG pipeline evaluations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_presets (
                    name TEXT PRIMARY KEY,
                    config TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT
                )
            """)

            # Ephemeral collections registry for TTL cleanup
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ephemeral_collections (
                    collection_name TEXT PRIMARY KEY,
                    namespace TEXT,
                    run_id TEXT,
                    ttl_seconds INTEGER DEFAULT 86400,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    deleted_at TEXT NULL
                )
            """)

            # Webhook registrations table (match webhook_manager schema)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS webhook_registrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    url TEXT NOT NULL,
                    secret TEXT NOT NULL,
                    events TEXT NOT NULL,
                    active BOOLEAN DEFAULT 1,
                    retry_count INTEGER DEFAULT 3,
                    timeout_seconds INTEGER DEFAULT 30,
                    total_deliveries INTEGER DEFAULT 0,
                    successful_deliveries INTEGER DEFAULT 0,
                    failed_deliveries INTEGER DEFAULT 0,
                    last_delivery_at TEXT,
                    last_error TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    webhook_id TEXT
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_evals_created ON evaluations(created_at DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_eval ON evaluation_runs(eval_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON evaluation_runs(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_datasets_created ON datasets(created_at DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_eval_recipe_runs_recipe_id ON evaluation_recipe_runs(recipe_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_eval_recipe_runs_status ON evaluation_recipe_runs(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_eval_recipe_runs_created_at ON evaluation_recipe_runs(created_at DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_eval_recipe_children_parent ON evaluation_recipe_run_children(parent_run_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_eval_recipe_children_child ON evaluation_recipe_run_children(child_run_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_synth_eval_samples_recipe ON synthetic_eval_draft_samples(recipe_kind)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_synth_eval_samples_provenance ON synthetic_eval_draft_samples(provenance)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_synth_eval_samples_review_state ON synthetic_eval_draft_samples(review_state)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_synth_eval_samples_created_at ON synthetic_eval_draft_samples(created_at DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_synth_eval_actions_sample_created ON synthetic_eval_review_actions(sample_id, created_at DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_synth_eval_promotions_sample_created ON synthetic_eval_promotions(sample_id, created_at DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_synth_eval_promotions_dataset ON synthetic_eval_promotions(dataset_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_internal_evals_type ON internal_evaluations(evaluation_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_internal_evals_user ON internal_evaluations(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_webhooks_active ON webhook_registrations(active)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pipeline_presets_updated ON pipeline_presets(updated_at DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ephemeral_created ON ephemeral_collections(created_at DESC)")

            # Idempotency mapping table (generic, scoped by user and entity type)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS idempotency_keys (
                    user_id TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(user_id, entity_type, idempotency_key)
                )
                """
            )

            # Embeddings A/B test tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embedding_abtests (
                    test_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_by TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    status TEXT NOT NULL DEFAULT 'pending',
                    config_json TEXT NOT NULL,
                    stats_json TEXT,
                    notes TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embedding_abtest_arms (
                    arm_id TEXT PRIMARY KEY,
                    test_id TEXT NOT NULL,
                    arm_index INTEGER NOT NULL,
                    provider TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    dimensions INTEGER,
                    collection_hash TEXT,
                    pipeline_hash TEXT,
                    collection_name TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    stats_json TEXT,
                    metadata_json TEXT,
                    FOREIGN KEY (test_id) REFERENCES embedding_abtests(test_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embedding_abtest_queries (
                    query_id TEXT PRIMARY KEY,
                    test_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    ground_truth_ids TEXT,
                    metadata_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (test_id) REFERENCES embedding_abtests(test_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embedding_abtest_results (
                    result_id TEXT PRIMARY KEY,
                    test_id TEXT NOT NULL,
                    arm_id TEXT NOT NULL,
                    query_id TEXT NOT NULL,
                    ranked_ids TEXT NOT NULL,
                    scores TEXT,
                    metrics_json TEXT,
                    latency_ms REAL,
                    ranked_distances TEXT,
                    ranked_metadatas TEXT,
                    ranked_documents TEXT,
                    rerank_scores TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (test_id) REFERENCES embedding_abtests(test_id),
                    FOREIGN KEY (arm_id) REFERENCES embedding_abtest_arms(arm_id),
                    FOREIGN KEY (query_id) REFERENCES embedding_abtest_queries(query_id)
                )
            """)

            # Best-effort ALTER TABLE for new diagnostic columns (ignore errors if already exist)
            for _col, sql in [
                ("ranked_distances", "ALTER TABLE embedding_abtest_results ADD COLUMN ranked_distances TEXT"),
                ("ranked_metadatas", "ALTER TABLE embedding_abtest_results ADD COLUMN ranked_metadatas TEXT"),
                ("ranked_documents", "ALTER TABLE embedding_abtest_results ADD COLUMN ranked_documents TEXT"),
                ("rerank_scores", "ALTER TABLE embedding_abtest_results ADD COLUMN rerank_scores TEXT"),
            ]:
                with suppress(_EVAL_DB_NONCRITICAL_EXCEPTIONS):
                    cursor.execute(sql)

            # Ensure embedding_abtest_queries.created_at exists even on older databases (SQLite cannot add non-constant defaults)
            try:
                cursor.execute("PRAGMA table_info(embedding_abtest_queries)")
                columns = {row["name"] for row in cursor.fetchall()}
            except _EVAL_DB_NONCRITICAL_EXCEPTIONS:
                columns = set()
            if "created_at" not in columns:
                try:
                    cursor.execute("ALTER TABLE embedding_abtest_queries ADD COLUMN created_at TEXT")
                    cursor.execute(
                        "UPDATE embedding_abtest_queries SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL"
                    )
                except _EVAL_DB_NONCRITICAL_EXCEPTIONS:
                    logger.warning("Failed to backfill embedding_abtest_queries.created_at column", exc_info=True)

            # Indexes for A/B test tables
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_abtests_created ON embedding_abtests(created_at DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_abtests_status ON embedding_abtests(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_abtest_arms_test ON embedding_abtest_arms(test_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_abtest_results_test ON embedding_abtest_results(test_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_abtest_results_arm ON embedding_abtest_results(arm_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_abtest_results_query ON embedding_abtest_results(query_id)")

            conn.commit()
            logger.info("Evaluations database initialized")

    def _initialize_database_postgres(self) -> None:
        """Provision PostgreSQL tables and indexes to mirror SQLite schema."""
        if self.backend is None:
            raise RuntimeError("Evaluations backend is not configured")
        ddl = """
        CREATE TABLE IF NOT EXISTS evaluations (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            eval_type TEXT NOT NULL,
            eval_spec JSONB NOT NULL,
            dataset_id TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            created_by TEXT,
            metadata JSONB,
            deleted_at TIMESTAMPTZ NULL
        );
        CREATE TABLE IF NOT EXISTS evaluation_runs (
            id TEXT PRIMARY KEY,
            eval_id TEXT NOT NULL,
            status TEXT NOT NULL,
            target_model TEXT,
            config JSONB,
            progress JSONB,
            results JSONB,
            error_message TEXT,
            started_at TIMESTAMPTZ,
            completed_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            webhook_url TEXT,
            usage JSONB,
            FOREIGN KEY (eval_id) REFERENCES evaluations(id)
        );
        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            samples JSONB NOT NULL,
            sample_count INTEGER,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            created_by TEXT,
            metadata JSONB
        );
        CREATE TABLE IF NOT EXISTS evaluation_recipe_runs (
            run_id TEXT PRIMARY KEY,
            recipe_id TEXT NOT NULL,
            recipe_version TEXT NOT NULL,
            status TEXT NOT NULL,
            review_state TEXT NOT NULL DEFAULT 'not_required',
            dataset_snapshot_ref TEXT,
            dataset_content_hash TEXT,
            confidence_summary_json JSONB,
            recommendation_slots_json JSONB,
            metadata_json JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS evaluation_recipe_run_children (
            parent_run_id TEXT NOT NULL,
            child_run_id TEXT NOT NULL,
            child_order INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (parent_run_id, child_run_id),
            FOREIGN KEY (parent_run_id) REFERENCES evaluation_recipe_runs(run_id)
        );
        CREATE TABLE IF NOT EXISTS synthetic_eval_draft_samples (
            sample_id TEXT PRIMARY KEY,
            recipe_kind TEXT NOT NULL,
            provenance TEXT NOT NULL,
            review_state TEXT NOT NULL DEFAULT 'draft',
            sample_payload_json JSONB NOT NULL,
            sample_metadata_json JSONB,
            source_kind TEXT,
            created_by TEXT,
            review_summary_json JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS synthetic_eval_review_actions (
            action_id TEXT PRIMARY KEY,
            sample_id TEXT NOT NULL,
            action TEXT NOT NULL,
            reviewer_id TEXT,
            notes TEXT,
            action_payload_json JSONB,
            resulting_review_state TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            FOREIGN KEY (sample_id) REFERENCES synthetic_eval_draft_samples(sample_id)
        );
        CREATE TABLE IF NOT EXISTS synthetic_eval_promotions (
            promotion_id TEXT PRIMARY KEY,
            sample_id TEXT NOT NULL,
            dataset_id TEXT,
            dataset_snapshot_ref TEXT,
            promoted_by TEXT,
            promotion_reason TEXT,
            promotion_metadata_json JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            FOREIGN KEY (sample_id) REFERENCES synthetic_eval_draft_samples(sample_id)
        );
        -- Unified evaluations table (enabled by default on PostgreSQL)
        CREATE TABLE IF NOT EXISTS evaluations_unified (
            id TEXT PRIMARY KEY,
            evaluation_id TEXT UNIQUE,
            name TEXT NOT NULL,
            evaluation_type TEXT NOT NULL,
            input_data JSONB NOT NULL,
            results JSONB,
            status TEXT NOT NULL DEFAULT 'completed',
            user_id TEXT,
            metadata JSONB,
            embedding_provider TEXT,
            embedding_model TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            completed_at TIMESTAMPTZ,
            eval_spec JSONB
        );
        CREATE TABLE IF NOT EXISTS internal_evaluations (
            evaluation_id TEXT PRIMARY KEY,
            evaluation_type TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            input_data JSONB,
            results JSONB,
            metadata JSONB,
            user_id TEXT,
            status TEXT DEFAULT 'pending',
            completed_at TIMESTAMPTZ,
            embedding_provider TEXT,
            embedding_model TEXT
        );
        CREATE TABLE IF NOT EXISTS pipeline_presets (
            name TEXT PRIMARY KEY,
            config JSONB NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            user_id TEXT
        );
        CREATE TABLE IF NOT EXISTS ephemeral_collections (
            collection_name TEXT PRIMARY KEY,
            namespace TEXT,
            run_id TEXT,
            ttl_seconds INTEGER DEFAULT 86400,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            deleted_at TIMESTAMPTZ
        );
        CREATE TABLE IF NOT EXISTS webhook_registrations (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            url TEXT NOT NULL,
            secret TEXT NOT NULL,
            events TEXT NOT NULL,
            active BOOLEAN DEFAULT TRUE,
            retry_count INTEGER DEFAULT 3,
            timeout_seconds INTEGER DEFAULT 30,
            total_deliveries INTEGER DEFAULT 0,
            successful_deliveries INTEGER DEFAULT 0,
            failed_deliveries INTEGER DEFAULT 0,
            last_delivery_at TIMESTAMPTZ,
            last_error TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            webhook_id TEXT
        );
        CREATE TABLE IF NOT EXISTS embedding_abtests (
            test_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_by TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            status TEXT NOT NULL DEFAULT 'pending',
            config_json JSONB NOT NULL,
            stats_json JSONB,
            notes TEXT
        );
        CREATE TABLE IF NOT EXISTS embedding_abtest_arms (
            arm_id TEXT PRIMARY KEY,
            test_id TEXT NOT NULL,
            arm_index INTEGER NOT NULL,
            provider TEXT NOT NULL,
            model_id TEXT NOT NULL,
            dimensions INTEGER,
            collection_hash TEXT,
            pipeline_hash TEXT,
            collection_name TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            stats_json JSONB,
            metadata_json JSONB,
            FOREIGN KEY (test_id) REFERENCES embedding_abtests(test_id)
        );
        CREATE TABLE IF NOT EXISTS embedding_abtest_queries (
            query_id TEXT PRIMARY KEY,
            test_id TEXT NOT NULL,
            text TEXT NOT NULL,
            ground_truth_ids JSONB,
            metadata_json JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            FOREIGN KEY (test_id) REFERENCES embedding_abtests(test_id)
        );
        CREATE TABLE IF NOT EXISTS embedding_abtest_results (
            result_id TEXT PRIMARY KEY,
            test_id TEXT NOT NULL,
            arm_id TEXT NOT NULL,
            query_id TEXT NOT NULL,
            ranked_ids JSONB NOT NULL,
            scores JSONB,
            metrics_json JSONB,
            latency_ms DOUBLE PRECISION,
            ranked_distances JSONB,
            ranked_metadatas JSONB,
            ranked_documents JSONB,
            rerank_scores JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            FOREIGN KEY (test_id) REFERENCES embedding_abtests(test_id),
            FOREIGN KEY (arm_id) REFERENCES embedding_abtest_arms(arm_id),
            FOREIGN KEY (query_id) REFERENCES embedding_abtest_queries(query_id)
        );
        ALTER TABLE embedding_abtest_queries
            ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();
        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_evals_created ON evaluations(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_runs_eval ON evaluation_runs(eval_id);
        CREATE INDEX IF NOT EXISTS idx_runs_status ON evaluation_runs(status);
        CREATE INDEX IF NOT EXISTS idx_datasets_created ON datasets(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_eval_recipe_runs_recipe_id ON evaluation_recipe_runs(recipe_id);
        CREATE INDEX IF NOT EXISTS idx_eval_recipe_runs_status ON evaluation_recipe_runs(status);
        CREATE INDEX IF NOT EXISTS idx_eval_recipe_runs_created_at ON evaluation_recipe_runs(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_eval_recipe_children_parent ON evaluation_recipe_run_children(parent_run_id);
        CREATE INDEX IF NOT EXISTS idx_eval_recipe_children_child ON evaluation_recipe_run_children(child_run_id);
        CREATE INDEX IF NOT EXISTS idx_synth_eval_samples_recipe ON synthetic_eval_draft_samples(recipe_kind);
        CREATE INDEX IF NOT EXISTS idx_synth_eval_samples_provenance ON synthetic_eval_draft_samples(provenance);
        CREATE INDEX IF NOT EXISTS idx_synth_eval_samples_review_state ON synthetic_eval_draft_samples(review_state);
        CREATE INDEX IF NOT EXISTS idx_synth_eval_samples_created_at ON synthetic_eval_draft_samples(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_synth_eval_actions_sample_created ON synthetic_eval_review_actions(sample_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_synth_eval_promotions_sample_created ON synthetic_eval_promotions(sample_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_synth_eval_promotions_dataset ON synthetic_eval_promotions(dataset_id);
        CREATE INDEX IF NOT EXISTS idx_evals_unified_created ON evaluations_unified(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_evals_unified_status ON evaluations_unified(status);
        CREATE INDEX IF NOT EXISTS idx_evals_unified_type ON evaluations_unified(evaluation_type);
        CREATE INDEX IF NOT EXISTS idx_internal_evals_type ON internal_evaluations(evaluation_type);
        CREATE INDEX IF NOT EXISTS idx_internal_evals_user ON internal_evaluations(user_id);
        CREATE INDEX IF NOT EXISTS idx_webhook_registrations_active ON webhook_registrations(active);
        CREATE INDEX IF NOT EXISTS idx_pipeline_presets_updated ON pipeline_presets(updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_ephemeral_created ON ephemeral_collections(created_at DESC);

        -- Idempotency mapping table (generic, scoped by user and entity type)
        CREATE TABLE IF NOT EXISTS idempotency_keys (
            user_id TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY(user_id, entity_type, idempotency_key)
        );
        """
        with self.backend.transaction() as conn:
            self.backend.create_tables(ddl, connection=conn)

    def _apply_migrations(self):
        """Apply database migrations including the unified schema."""
        try:
            from tldw_Server_API.app.core.DB_Management.migrations_v5_unified_evaluations import (
                migrate_to_unified_evaluations,
            )
            from tldw_Server_API.app.core.DB_Management.migrations_v6_evaluation_recipes import (
                migrate_to_evaluation_recipes,
            )
            from tldw_Server_API.app.core.DB_Management.migrations_v7_synthetic_eval_workflow import (
                migrate_to_synthetic_eval_workflow,
            )

            # Apply the unified evaluations migration
            if migrate_to_unified_evaluations(self.db_path):
                logger.info("Applied unified evaluations migration successfully")
            else:
                logger.warning("Unified evaluations migration already applied or failed")

            if migrate_to_evaluation_recipes(self.db_path):
                logger.info("Applied evaluation recipe migration successfully")
            else:
                logger.warning("Evaluation recipe migration already applied or failed")
            if migrate_to_synthetic_eval_workflow(self.db_path):
                logger.info("Applied synthetic eval workflow migration successfully")
            else:
                logger.warning("Synthetic eval workflow migration already applied or failed")
        except ImportError:
            logger.warning("Migration module not found, skipping")
        except _EVAL_DB_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error applying migrations: {e}")

    def _use_unified_table(self) -> bool:
        """Check if the unified table exists and should be used."""
        if self.backend_type == BackendType.SQLITE:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='evaluations_unified'"
                )
                return cursor.fetchone() is not None
        # PostgreSQL path
        if not self.backend:
            return False
        result = self.backend.execute(

                "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                "WHERE table_schema='public' AND table_name='evaluations_unified')"

        )
        return bool(result.scalar)

    # ============= Evaluation CRUD Operations =============

    def create_evaluation(
        self,
        name: str,
        eval_type: str,
        eval_spec: dict[str, Any],
        description: Optional[str] = None,
        dataset_id: Optional[str] = None,
        created_by: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        eval_id: Optional[str] = None,
    ) -> str:
        """Create a new evaluation definition"""
        eval_id = eval_id or f"eval_{uuid.uuid4().hex[:12]}"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO evaluations (id, name, description, eval_type, eval_spec,
                                       dataset_id, created_by, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                eval_id,
                name,
                description,
                eval_type,
                json.dumps(eval_spec),
                dataset_id,
                created_by,
                json.dumps(metadata) if metadata else None
            ))
            conn.commit()

        logger.info(f"Created evaluation: {eval_id}")
        return eval_id

    def get_evaluation(self, eval_id: str, *, created_by: Optional[str] = None) -> Optional[dict[str, Any]]:
        """Get evaluation by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM evaluations WHERE id = ? AND deleted_at IS NULL"
            params: list[Any] = [eval_id]
            query, params = self._append_user_filter(query, params, "created_by", created_by)
            cursor.execute(query, params)

            row = cursor.fetchone()
            if row:
                return self._row_to_eval_dict(row)
        return None

    def list_evaluations(
        self,
        limit: int = 20,
        after: Optional[str] = None,
        eval_type: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> tuple[list[dict[str, Any]], bool]:
        """List evaluations with pagination"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM evaluations WHERE deleted_at IS NULL"
            params = []

            if eval_type:
                query += " AND eval_type = ?"
                params.append(eval_type)

            query, params = self._append_user_filter(query, params, "created_by", created_by)

            if after:
                query += " AND created_at < (SELECT created_at FROM evaluations WHERE id = ?)"
                params.append(after)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit + 1)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            has_more = len(rows) > limit
            evaluations = [self._row_to_eval_dict(row) for row in rows[:limit]]

            return evaluations, has_more

    def list_evaluations_filtered(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        eval_type: Optional[str] = None,
        created_by: Optional[str] = None,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
        return_has_more: bool = False,
    ) -> Any:
        """List evaluations with optional created_by and date range filters."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM evaluations WHERE deleted_at IS NULL"
            params: list[Any] = []

            if eval_type:
                query += " AND eval_type = ?"
                params.append(eval_type)

            query, params = self._append_user_filter(query, params, "created_by", created_by)

            start_value = self._coerce_datetime_filter(start_date)
            if start_value:
                query += " AND created_at >= ?"
                params.append(start_value)

            end_value = self._coerce_datetime_filter(end_date)
            if end_value:
                query += " AND created_at <= ?"
                params.append(end_value)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit + 1 if return_has_more else limit)
            if offset:
                query += " OFFSET ?"
                params.append(offset)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            if return_has_more:
                has_more = len(rows) > limit
                evaluations = [self._row_to_eval_dict(row) for row in rows[:limit]]
                return evaluations, has_more

            return [self._row_to_eval_dict(row) for row in rows]

    def count_evaluations_filtered(
        self,
        *,
        eval_type: Optional[str] = None,
        created_by: Optional[str] = None,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
    ) -> int:
        """Count evaluations with optional created_by and date range filters."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT COUNT(*) AS count FROM evaluations WHERE deleted_at IS NULL"
            params: list[Any] = []

            if eval_type:
                query += " AND eval_type = ?"
                params.append(eval_type)

            query, params = self._append_user_filter(query, params, "created_by", created_by)

            start_value = self._coerce_datetime_filter(start_date)
            if start_value:
                query += " AND created_at >= ?"
                params.append(start_value)

            end_value = self._coerce_datetime_filter(end_date)
            if end_value:
                query += " AND created_at <= ?"
                params.append(end_value)

            cursor.execute(query, params)
            row = cursor.fetchone()
            if row is None:
                return 0
            if isinstance(row, dict):
                return int(row.get("count", 0) or 0)
            try:
                return int(row[0])
            except _EVAL_DB_NONCRITICAL_EXCEPTIONS:
                return 0

    def _init_abtest_store(self) -> None:
        desired = str(os.getenv("EVALS_ABTEST_PERSISTENCE", "sqlalchemy")).strip().lower()
        if desired not in {"sqlalchemy", "repo"}:
            self._abtest_store = None
            return
        db_url = None
        if self.backend_type == BackendType.SQLITE:
            db_url = self.db_path
        elif self.backend_type == BackendType.POSTGRESQL and self.backend is not None:
            cfg = self.backend.config
            raw = getattr(cfg, "connection_string", None)
            if isinstance(raw, str) and "://" in raw:
                db_url = raw
            else:
                try:
                    from urllib.parse import quote_plus
                    user = cfg.pg_user or "tldw_user"
                    password = cfg.pg_password or ""
                    host = cfg.pg_host or "localhost"
                    port = cfg.pg_port or 5432
                    database = cfg.pg_database or "tldw"
                    sslmode = cfg.pg_sslmode or "prefer"
                    auth = f"{user}:{quote_plus(password)}" if password else user
                    db_url = f"postgresql://{auth}@{host}:{port}/{database}?sslmode={sslmode}"
                except _EVAL_DB_NONCRITICAL_EXCEPTIONS:
                    db_url = None
        if not db_url:
            self._abtest_store = None
            return
        db_url = _normalize_sqlalchemy_postgres_url(db_url)
        try:
            from tldw_Server_API.app.core.Evaluations.embeddings_abtest_repository import (
                get_embeddings_abtest_store,
            )

            self._abtest_store = get_embeddings_abtest_store(db_url)
            logger.debug("Embeddings A/B tests using SQLAlchemy repository backend")
        except _EVAL_DB_NONCRITICAL_EXCEPTIONS as exc:
            self._abtest_store = None
            logger.warning("Falling back to legacy embeddings A/B persistence: {}", exc)

    # ============= Embeddings A/B Test Operations =============

    def create_abtest(self, name: str, config: dict[str, Any], created_by: Optional[str] = None) -> str:
        if self._abtest_store:
            return self._abtest_store.create_abtest(name=name, config=config, created_by=created_by)
        test_id = f"abtest_{uuid.uuid4().hex[:12]}"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO embedding_abtests (test_id, name, created_by, status, config_json)
                VALUES (?, ?, ?, 'pending', ?)
                """,
                (test_id, name, created_by, json.dumps(config))
            )
            conn.commit()
        return test_id

    def upsert_abtest_arm(
        self,
        test_id: str,
        arm_index: int,
        provider: str,
        model_id: str,
        dimensions: Optional[int] = None,
        collection_hash: Optional[str] = None,
        pipeline_hash: Optional[str] = None,
        collection_name: Optional[str] = None,
        status: str = 'pending',
        stats_json: Optional[dict[str, Any]] = None,
        metadata_json: Optional[dict[str, Any]] = None,
    ) -> str:
        if self._abtest_store:
            return self._abtest_store.upsert_abtest_arm(
                test_id=test_id,
                arm_index=arm_index,
                provider=provider,
                model_id=model_id,
                dimensions=dimensions,
                collection_hash=collection_hash,
                pipeline_hash=pipeline_hash,
                collection_name=collection_name,
                status=status,
                stats_json=stats_json,
                metadata_json=metadata_json,
            )
        arm_id = f"arm_{test_id}_{arm_index}"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO embedding_abtest_arms (arm_id, test_id, arm_index, provider, model_id, dimensions,
                    collection_hash, pipeline_hash, collection_name, status, stats_json, metadata_json)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(arm_id) DO UPDATE SET
                    provider=excluded.provider,
                    model_id=excluded.model_id,
                    dimensions=excluded.dimensions,
                    collection_hash=excluded.collection_hash,
                    pipeline_hash=excluded.pipeline_hash,
                    collection_name=excluded.collection_name,
                    status=excluded.status,
                    stats_json=excluded.stats_json,
                    metadata_json=excluded.metadata_json
                """,
                (
                    arm_id, test_id, arm_index, provider, model_id, dimensions,
                    collection_hash, pipeline_hash, collection_name, status,
                    json.dumps(stats_json) if stats_json else None,
                    json.dumps(metadata_json) if metadata_json else None,
                )
            )
            conn.commit()
        return arm_id

    def insert_abtest_queries(self, test_id: str, queries: list[dict[str, Any]]) -> list[str]:
        if self._abtest_store:
            return self._abtest_store.insert_abtest_queries(test_id, queries)
        ids: list[str] = []
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for q in queries:
                qid = f"q_{uuid.uuid4().hex[:10]}"
                ids.append(qid)
                cursor.execute(
                    """
                    INSERT INTO embedding_abtest_queries (
                        query_id,
                        test_id,
                        text,
                        ground_truth_ids,
                        metadata_json,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        qid,
                        test_id,
                        q.get('text', ''),
                        json.dumps(q.get('expected_ids')) if q.get('expected_ids') else None,
                        json.dumps(q.get('metadata')) if q.get('metadata') else None,
                        datetime.utcnow().isoformat(sep=" "),
                    )
                )
            conn.commit()
        return ids

    def set_abtest_status(self, test_id: str, status: str, stats_json: Optional[dict[str, Any]] = None, *, created_by: Optional[str] = None):
        if self._abtest_store:
            self._abtest_store.set_abtest_status(test_id, status, stats_json, created_by=created_by)
            return
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "UPDATE embedding_abtests SET status = ?, stats_json = COALESCE(?, stats_json) WHERE test_id = ?"
            params: list[Any] = [status, json.dumps(stats_json) if stats_json else None, test_id]
            query, params = self._append_user_filter(query, params, "created_by", created_by)
            cursor.execute(query, params)
            conn.commit()

    def insert_abtest_result(
        self,
        test_id: str,
        arm_id: str,
        query_id: str,
        ranked_ids: list[str],
        scores: Optional[list[float]] = None,
        metrics: Optional[dict[str, Any]] = None,
        latency_ms: Optional[float] = None,
        ranked_distances: Optional[list[float]] = None,
        ranked_metadatas: Optional[list[dict[str, Any]]] = None,
        ranked_documents: Optional[list[str]] = None,
        rerank_scores: Optional[list[float]] = None,
    ) -> str:
        if self._abtest_store:
            return self._abtest_store.insert_abtest_result(
                test_id=test_id,
                arm_id=arm_id,
                query_id=query_id,
                ranked_ids=ranked_ids,
                scores=scores,
                metrics=metrics,
                latency_ms=latency_ms,
                ranked_distances=ranked_distances,
                ranked_metadatas=ranked_metadatas,
                ranked_documents=ranked_documents,
                rerank_scores=rerank_scores,
            )
        rid = f"res_{uuid.uuid4().hex[:12]}"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO embedding_abtest_results (
                    result_id, test_id, arm_id, query_id, ranked_ids, scores, metrics_json, latency_ms,
                    ranked_distances, ranked_metadatas, ranked_documents, rerank_scores
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rid, test_id, arm_id, query_id, json.dumps(ranked_ids),
                    json.dumps(scores) if scores else None,
                    json.dumps(metrics) if metrics else None,
                    float(latency_ms) if latency_ms is not None else None,
                    json.dumps(ranked_distances) if ranked_distances else None,
                    json.dumps(ranked_metadatas) if ranked_metadatas else None,
                    json.dumps(ranked_documents) if ranked_documents else None,
                    json.dumps(rerank_scores) if rerank_scores else None,
                )
            )
            conn.commit()
        return rid

    def get_abtest(self, test_id: str, *, created_by: Optional[str] = None) -> Optional[dict[str, Any]]:
        if self._abtest_store:
            return self._abtest_store.get_abtest(test_id, created_by=created_by)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM embedding_abtests WHERE test_id = ?"
            params: list[Any] = [test_id]
            query, params = self._append_user_filter(query, params, "created_by", created_by)
            cursor.execute(query, params)
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def get_abtest_arms(self, test_id: str, *, created_by: Optional[str] = None) -> list[dict[str, Any]]:
        if self._abtest_store:
            return self._abtest_store.get_abtest_arms(test_id, created_by=created_by)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if created_by:
                query = (
                    "SELECT a.* FROM embedding_abtest_arms a "
                    "JOIN embedding_abtests t ON t.test_id = a.test_id "
                    "WHERE a.test_id = ?"
                )
                params: list[Any] = [test_id]
                query, params = self._append_user_filter(query, params, "t.created_by", created_by)
                query += " ORDER BY a.arm_index ASC"
                cursor.execute(query, params)
            else:
                cursor.execute("SELECT * FROM embedding_abtest_arms WHERE test_id = ? ORDER BY arm_index ASC", (test_id,))
            return [dict(r) for r in cursor.fetchall()]

    def find_reusable_abtest_arm(
        self,
        *,
        test_id: str,
        collection_hash: str,
        created_by: Optional[str],
    ) -> Optional[dict[str, Any]]:
        if not collection_hash or not created_by:
            return None
        if self._abtest_store:
            return self._abtest_store.find_reusable_abtest_arm(
                test_id=test_id,
                collection_hash=collection_hash,
                created_by=created_by,
            )
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT a.*
                FROM embedding_abtest_arms a
                JOIN embedding_abtests t ON t.test_id = a.test_id
                WHERE a.collection_hash = ?
                  AND a.status = 'ready'
                  AND a.collection_name IS NOT NULL
                  AND t.created_by = ?
                  AND a.test_id <> ?
                ORDER BY t.created_at DESC
                LIMIT 1
                """,
                (collection_hash, created_by, test_id),
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def get_abtest_queries(self, test_id: str, *, created_by: Optional[str] = None) -> list[dict[str, Any]]:
        if self._abtest_store:
            return self._abtest_store.get_abtest_queries(test_id, created_by=created_by)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if created_by:
                query = (
                    "SELECT q.* FROM embedding_abtest_queries q "
                    "JOIN embedding_abtests t ON t.test_id = q.test_id "
                    "WHERE q.test_id = ?"
                )
                params: list[Any] = [test_id]
                query, params = self._append_user_filter(query, params, "t.created_by", created_by)
                cursor.execute(query, params)
            else:
                cursor.execute("SELECT * FROM embedding_abtest_queries WHERE test_id = ?", (test_id,))
            return [dict(r) for r in cursor.fetchall()]

    def list_abtest_results(self, test_id: str, limit: int = 100, offset: int = 0, *, created_by: Optional[str] = None) -> tuple[list[dict[str, Any]], int]:
        if self._abtest_store:
            return self._abtest_store.list_abtest_results(test_id, limit, offset, created_by=created_by)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if created_by:
                count_query = (
                    "SELECT COUNT(*) FROM embedding_abtest_results r "
                    "JOIN embedding_abtests t ON t.test_id = r.test_id "
                    "WHERE r.test_id = ?"
                )
                count_params: list[Any] = [test_id]
                count_query, count_params = self._append_user_filter(count_query, count_params, "t.created_by", created_by)
                cursor.execute(count_query, count_params)
            else:
                cursor.execute(
                    "SELECT COUNT(*) FROM embedding_abtest_results WHERE test_id = ?",
                    (test_id,)
                )
            total = cursor.fetchone()[0] or 0
            if created_by:
                list_query = (
                    "SELECT r.* FROM embedding_abtest_results r "
                    "JOIN embedding_abtests t ON t.test_id = r.test_id "
                    "WHERE r.test_id = ?"
                )
                list_params: list[Any] = [test_id]
                list_query, list_params = self._append_user_filter(list_query, list_params, "t.created_by", created_by)
                list_query += " ORDER BY r.created_at ASC LIMIT ? OFFSET ?"
                list_params.extend([limit, offset])
                cursor.execute(list_query, list_params)
            else:
                cursor.execute(
                    """
                    SELECT * FROM embedding_abtest_results WHERE test_id = ?
                    ORDER BY created_at ASC LIMIT ? OFFSET ?
                    """,
                    (test_id, limit, offset)
                )
            rows = [dict(r) for r in cursor.fetchall()]
        return rows, total

    def delete_abtest(self, test_id: str, *, delete_idempotency: bool = True, created_by: Optional[str] = None) -> int:
        """Delete an embeddings A/B test and related rows."""
        if self._abtest_store:
            deleted = int(self._abtest_store.delete_abtest(test_id, created_by=created_by) or 0)
            if delete_idempotency:
                try:
                    with self.get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            """
                            DELETE FROM idempotency_keys
                            WHERE entity_type LIKE ? AND (entity_id = ? OR entity_id LIKE ?)
                            """,
                            ("emb_abtest%", test_id, f"{test_id}:%"),
                        )
                        deleted += int(cursor.rowcount or 0)
                        with suppress(_EVAL_DB_NONCRITICAL_EXCEPTIONS):
                            conn.commit()
                except _EVAL_DB_NONCRITICAL_EXCEPTIONS:
                    pass
            return deleted
        deleted = 0
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                if created_by:
                    query = (
                        "DELETE FROM embedding_abtest_results "
                        "WHERE test_id = ? AND EXISTS (SELECT 1 FROM embedding_abtests t WHERE t.test_id = ?"
                    )
                    params: list[Any] = [test_id, test_id]
                    query, params = self._append_user_filter(query, params, "t.created_by", created_by)
                    query += ")"
                    cursor.execute(query, params)
                else:
                    cursor.execute("DELETE FROM embedding_abtest_results WHERE test_id = ?", (test_id,))
                deleted += int(cursor.rowcount or 0)
            except _EVAL_DB_NONCRITICAL_EXCEPTIONS:
                pass
            try:
                if created_by:
                    query = (
                        "DELETE FROM embedding_abtest_queries "
                        "WHERE test_id = ? AND EXISTS (SELECT 1 FROM embedding_abtests t WHERE t.test_id = ?"
                    )
                    params: list[Any] = [test_id, test_id]
                    query, params = self._append_user_filter(query, params, "t.created_by", created_by)
                    query += ")"
                    cursor.execute(query, params)
                else:
                    cursor.execute("DELETE FROM embedding_abtest_queries WHERE test_id = ?", (test_id,))
                deleted += int(cursor.rowcount or 0)
            except _EVAL_DB_NONCRITICAL_EXCEPTIONS:
                pass
            try:
                if created_by:
                    query = (
                        "DELETE FROM embedding_abtest_arms "
                        "WHERE test_id = ? AND EXISTS (SELECT 1 FROM embedding_abtests t WHERE t.test_id = ?"
                    )
                    params: list[Any] = [test_id, test_id]
                    query, params = self._append_user_filter(query, params, "t.created_by", created_by)
                    query += ")"
                    cursor.execute(query, params)
                else:
                    cursor.execute("DELETE FROM embedding_abtest_arms WHERE test_id = ?", (test_id,))
                deleted += int(cursor.rowcount or 0)
            except _EVAL_DB_NONCRITICAL_EXCEPTIONS:
                pass
            try:
                if created_by:
                    query = "DELETE FROM embedding_abtests WHERE test_id = ?"
                    params: list[Any] = [test_id]
                    query, params = self._append_user_filter(query, params, "created_by", created_by)
                    cursor.execute(query, params)
                else:
                    cursor.execute("DELETE FROM embedding_abtests WHERE test_id = ?", (test_id,))
                deleted += int(cursor.rowcount or 0)
            except _EVAL_DB_NONCRITICAL_EXCEPTIONS:
                pass
            if delete_idempotency:
                try:
                    cursor.execute(
                        """
                        DELETE FROM idempotency_keys
                        WHERE entity_type LIKE ? AND (entity_id = ? OR entity_id LIKE ?)
                        """,
                        ("emb_abtest%", test_id, f"{test_id}:%"),
                    )
                    deleted += int(cursor.rowcount or 0)
                except _EVAL_DB_NONCRITICAL_EXCEPTIONS:
                    pass
            with suppress(_EVAL_DB_NONCRITICAL_EXCEPTIONS):
                conn.commit()
        return int(deleted)

    # ============= Idempotency Helpers =============

    def lookup_idempotency(self, entity_type: str, key: str, user_id: Optional[str]) -> Optional[str]:
        """Lookup an existing entity_id by idempotency key scoped to user and entity type."""
        if not key:
            return None
        uid = user_id or ""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "SELECT entity_id FROM idempotency_keys WHERE user_id = ? AND entity_type = ? AND idempotency_key = ?",
                    (uid, entity_type, key),
                )
                row = cursor.fetchone()
                if row:
                    # sqlite3.Row or dict-like from backend adapter
                    return row[0] if not isinstance(row, dict) else row.get("entity_id")
            except _EVAL_DB_NONCRITICAL_EXCEPTIONS:
                return None
        return None

    def record_idempotency(self, entity_type: str, key: str, entity_id: str, user_id: Optional[str]) -> None:
        """Record an idempotency mapping; ignore on conflict."""
        if not key or not entity_id:
            return
        uid = user_id or ""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                # Try INSERT OR IGNORE for SQLite; Postgres path uses ON CONFLICT via backend adapter
                if self.backend_type == BackendType.SQLITE:
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO idempotency_keys (user_id, entity_type, idempotency_key, entity_id)
                        VALUES (?, ?, ?, ?)
                        """,
                        (uid, entity_type, key, entity_id),
                    )
                else:
                    cursor.execute(
                        """
                        INSERT INTO idempotency_keys (user_id, entity_type, idempotency_key, entity_id)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(user_id, entity_type, idempotency_key) DO NOTHING
                        """,
                        (uid, entity_type, key, entity_id),
                    )
                conn = getattr(cursor, "connection", None)
                try:
                    if conn:
                        conn.commit()
                except _EVAL_DB_NONCRITICAL_EXCEPTIONS:
                    pass
            except _EVAL_DB_NONCRITICAL_EXCEPTIONS:
                # Best-effort; safe to ignore failures
                pass

    def update_idempotency_entity(
        self,
        entity_type: str,
        key: str,
        entity_id: str,
        user_id: Optional[str],
    ) -> bool:
        """Update an existing idempotency mapping to a new entity id."""
        if not key or not entity_id:
            return False
        uid = user_id or ""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    UPDATE idempotency_keys
                    SET entity_id = ?
                    WHERE user_id = ? AND entity_type = ? AND idempotency_key = ?
                    """,
                    (entity_id, uid, entity_type, key),
                )
                conn.commit()
                return bool(cursor.rowcount)
            except Exception:
                with suppress(_EVAL_DB_NONCRITICAL_EXCEPTIONS):
                    conn.rollback()
                raise

    def find_latest_recipe_run_by_reuse_hash(
        self,
        *,
        reuse_hash: str,
        owner_user_id: str | None,
        allow_legacy_unowned: bool = False,
    ) -> RecipeRunRecord | None:
        """Fetch the latest completed recipe run matching a reuse hash and owner scope."""
        normalized_reuse_hash = str(reuse_hash or "").strip()
        normalized_owner_user_id = str(owner_user_id or "").strip()
        if not normalized_reuse_hash or not normalized_owner_user_id:
            return None

        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.backend_type == BackendType.POSTGRESQL and self.backend is not None:
                cursor.execute(
                    """
                    SELECT *
                    FROM evaluation_recipe_runs
                    WHERE status = %s
                      AND COALESCE(metadata_json, '{}')::jsonb ->> 'reuse_hash' = %s
                      AND (
                        COALESCE(metadata_json, '{}')::jsonb ->> 'owner_user_id' = %s
                        OR (
                            %s = TRUE
                            AND NOT (COALESCE(metadata_json, '{}')::jsonb ? 'owner_user_id')
                        )
                      )
                    ORDER BY COALESCE(updated_at, created_at) DESC, created_at DESC
                    LIMIT 1
                    """,
                    (
                        RunStatus.COMPLETED.value,
                        normalized_reuse_hash,
                        normalized_owner_user_id,
                        allow_legacy_unowned,
                    ),
                )
            else:
                cursor.execute(
                    """
                    SELECT *
                    FROM evaluation_recipe_runs
                    WHERE status = ?
                      AND json_extract(COALESCE(metadata_json, '{}'), '$.reuse_hash') = ?
                      AND (
                        json_extract(COALESCE(metadata_json, '{}'), '$.owner_user_id') = ?
                        OR (
                            ? = 1
                            AND json_type(COALESCE(metadata_json, '{}'), '$.owner_user_id') IS NULL
                        )
                      )
                    ORDER BY COALESCE(updated_at, created_at) DESC, created_at DESC
                    LIMIT 1
                    """,
                    (
                        RunStatus.COMPLETED.value,
                        normalized_reuse_hash,
                        normalized_owner_user_id,
                        1 if allow_legacy_unowned else 0,
                    ),
                )
            row = cursor.fetchone()
            if row:
                return self._row_to_recipe_run_record(row)
        return None

    def cleanup_idempotency_keys(self, ttl_hours: int = 72) -> int:
        """Remove idempotency keys older than ttl_hours. Returns deleted row count.

        This is intended to be invoked by a periodic maintenance task.
        For SQLite, uses datetime('now', ?). For PostgreSQL, uses NOW() - INTERVAL.
        """
        deleted = 0
        try:
            if self.backend_type == BackendType.POSTGRESQL and self.backend is not None:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        DELETE FROM idempotency_keys
                        WHERE created_at < (NOW() - (INTERVAL '1 hour' * %s))
                        """,
                        (ttl_hours,),
                    )
                    deleted = cursor.rowcount or 0
                    # no commit needed for backend adapter
            else:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    # SQLite: datetime('now', '-{ttl} hours')
                    cursor.execute(
                        "DELETE FROM idempotency_keys WHERE datetime(created_at) < datetime('now', ?)",
                        (f"-{int(ttl_hours)} hours",),
                    )
                    conn.commit()
                    deleted = cursor.rowcount if hasattr(cursor, 'rowcount') else 0
        except _EVAL_DB_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"cleanup_idempotency_keys failed: {e}")
            return 0
        return int(deleted)

    def update_evaluation(self, eval_id: str, updates: dict[str, Any], *, created_by: Optional[str] = None) -> bool:
        """Update evaluation definition"""
        allowed_fields = {"name", "description", "eval_spec", "dataset_id", "metadata"}
        updates = {k: v for k, v in updates.items() if k in allowed_fields}

        if not updates:
            return False

        # Handle metadata merging
        if "metadata" in updates:
            # Get existing evaluation to merge metadata
            existing = self.get_evaluation(eval_id)
            if existing and existing.get("metadata"):
                # Merge existing metadata with updates
                merged_metadata = existing["metadata"].copy()
                merged_metadata.update(updates["metadata"])
                updates["metadata"] = merged_metadata

        # JSON serialize complex fields
        if "eval_spec" in updates:
            updates["eval_spec"] = json.dumps(updates["eval_spec"])
        if "metadata" in updates:
            updates["metadata"] = json.dumps(updates["metadata"])

        updates["updated_at"] = datetime.utcnow().isoformat()

        with self.get_connection() as conn:
            cursor = conn.cursor()

            set_clause = ", ".join([f"{k} = ?" for k in updates])
            values = list(updates.values()) + [eval_id]

            update_evaluation_sql_template = "UPDATE evaluations SET {set_clause} WHERE id = ? AND deleted_at IS NULL"
            query = update_evaluation_sql_template.format_map(locals())  # nosec B608
            params = list(values)
            if created_by:
                query, params = self._append_user_filter(query, params, "created_by", created_by)
            cursor.execute(query, params)

            conn.commit()
            return cursor.rowcount > 0

    def delete_evaluation(self, eval_id: str, *, created_by: Optional[str] = None) -> bool:
        """Soft delete evaluation"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "UPDATE evaluations SET deleted_at = CURRENT_TIMESTAMP WHERE id = ? AND deleted_at IS NULL"
            params: list[Any] = [eval_id]
            query, params = self._append_user_filter(query, params, "created_by", created_by)
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount > 0

    # ============= Run CRUD Operations =============

    def create_run(
        self,
        eval_id: str,
        target_model: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
        webhook_url: Optional[str] = None,
        *,
        run_id: Optional[str] = None,
    ) -> str:
        """Create a new evaluation run.

        Accepts an optional run_id for callers that pre-generate IDs (e.g.,
        tests or idempotent schedulers). Falls back to an auto-generated ID
        when not provided.
        """
        run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO evaluation_runs (id, eval_id, status, target_model, config, webhook_url)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                eval_id,
                "pending",
                target_model,
                json.dumps(config) if config else None,
                webhook_url
            ))
            conn.commit()

        logger.info(f"Created run: {run_id} for evaluation: {eval_id}")
        return run_id

    def get_run(self, run_id: str, *, created_by: Optional[str] = None) -> Optional[dict[str, Any]]:
        """Get run by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if created_by:
                query = (
                    "SELECT r.* FROM evaluation_runs r "
                    "JOIN evaluations e ON e.id = r.eval_id "
                    "WHERE r.id = ? AND e.deleted_at IS NULL"
                )
                params: list[Any] = [run_id]
                query, params = self._append_user_filter(query, params, "e.created_by", created_by)
                cursor.execute(query, params)
            else:
                cursor.execute("SELECT * FROM evaluation_runs WHERE id = ?", (run_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_run_dict(row)
        return None

    def list_runs(
        self,
        eval_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
        after: Optional[str] = None,
        return_has_more: bool = False,
        created_by: Optional[str] = None,
    ) -> Any:
        """List runs with optional filtering.

        By default returns a list of run dicts for ergonomic use in tests and simple call sites.
        When return_has_more=True, returns a tuple of (runs, has_more) for pagination-aware callers.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if created_by:
                query = (
                    "SELECT r.* FROM evaluation_runs r "
                    "JOIN evaluations e ON e.id = r.eval_id "
                    "WHERE e.deleted_at IS NULL"
                )
            else:
                query = "SELECT * FROM evaluation_runs WHERE 1=1"
            params = []

            if eval_id:
                query += " AND r.eval_id = ?" if created_by else " AND eval_id = ?"
                params.append(eval_id)

            if status:
                query += " AND r.status = ?" if created_by else " AND status = ?"
                params.append(status)

            if after:
                query += " AND r.created_at < (SELECT created_at FROM evaluation_runs WHERE id = ?)" if created_by else " AND created_at < (SELECT created_at FROM evaluation_runs WHERE id = ?)"
                params.append(after)

            if created_by:
                query, params = self._append_user_filter(query, params, "e.created_by", created_by)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit + 1)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            has_more = len(rows) > limit
            runs = [self._row_to_run_dict(row) for row in rows[:limit]]
            if return_has_more:
                return runs, has_more
            return runs

    def update_run_status(
        self,
        run_id: str,
        status: str,
        error_message: Optional[str] = None
    ) -> bool:
        """Update run status"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            updates = {"status": status}

            if status == "running" and "started_at" not in updates:
                updates["started_at"] = datetime.utcnow().isoformat()
            elif status in ["completed", "failed", "cancelled"]:
                updates["completed_at"] = datetime.utcnow().isoformat()

            if error_message:
                updates["error_message"] = error_message

            set_clause = ", ".join([f"{k} = ?" for k in updates])
            values = list(updates.values()) + [run_id]

            update_run_sql_template = """
                UPDATE evaluation_runs
                SET {set_clause}
                WHERE id = ?
            """
            update_run_sql = update_run_sql_template.format_map(locals())  # nosec B608
            cursor.execute(update_run_sql, values)

            conn.commit()
            return cursor.rowcount > 0

    def update_run_progress(self, run_id: str, progress: dict[str, Any]) -> bool:
        """Update run progress"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE evaluation_runs
                SET progress = ?
                WHERE id = ?
            """, (json.dumps(progress), run_id))
            conn.commit()
            return cursor.rowcount > 0

    def store_run_results(
        self,
        run_id: str,
        results: dict[str, Any],
        usage: Optional[dict[str, Any]] = None
    ) -> bool:
        """Store run results"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE evaluation_runs
                SET results = ?, usage = ?, status = 'completed', completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (
                json.dumps(results),
                json.dumps(usage) if usage else None,
                run_id
            ))
            conn.commit()
            return cursor.rowcount > 0

    # ============= Dataset CRUD Operations =============

    def create_dataset(
        self,
        name: str,
        samples: list[dict[str, Any]],
        description: Optional[str] = None,
        created_by: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None
    ) -> str:
        """Create a new dataset"""
        dataset_id = f"dataset_{uuid.uuid4().hex[:12]}"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO datasets (id, name, description, samples, sample_count, created_by, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                dataset_id,
                name,
                description,
                json.dumps(samples),
                len(samples),
                created_by,
                json.dumps(metadata) if metadata else None
            ))
            conn.commit()

        logger.info(f"Created dataset: {dataset_id} with {len(samples)} samples")
        return dataset_id

    def get_dataset(
        self,
        dataset_id: str,
        *,
        created_by: Optional[str] = None,
        include_samples: bool = True,
        sample_limit: Optional[int] = None,
        sample_offset: int = 0,
    ) -> Optional[dict[str, Any]]:
        """Get dataset by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM datasets WHERE id = ?"
            params: list[Any] = [dataset_id]
            query, params = self._append_user_filter(query, params, "created_by", created_by)
            cursor.execute(query, params)
            row = cursor.fetchone()
            if row:
                return self._row_to_dataset_dict(
                    row,
                    include_samples=include_samples,
                    sample_limit=sample_limit,
                    sample_offset=sample_offset,
                )
        return None

    def list_datasets(
        self,
        limit: int = 20,
        after: Optional[str] = None,
        offset: int = 0,
        created_by: Optional[str] = None,
    ) -> tuple[list[dict[str, Any]], bool]:
        """List datasets with pagination"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM datasets WHERE 1=1"
            params = []

            if after:
                query += " AND created_at < (SELECT created_at FROM datasets WHERE id = ?)"
                params.append(after)

            query, params = self._append_user_filter(query, params, "created_by", created_by)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit + 1)
            if offset:
                query += " OFFSET ?"
                params.append(offset)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            has_more = len(rows) > limit
            datasets = [self._row_to_dataset_dict(row, include_samples=False)
                       for row in rows[:limit]]

            return datasets, has_more

    def delete_dataset(self, dataset_id: str, *, created_by: Optional[str] = None) -> bool:
        """Delete dataset"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "DELETE FROM datasets WHERE id = ?"
            params: list[Any] = [dataset_id]
            query, params = self._append_user_filter(query, params, "created_by", created_by)
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount > 0

    # ============= Recipe Run CRUD Operations =============

    def create_recipe_run(
        self,
        *,
        recipe_id: str,
        recipe_version: str,
        status: RunStatus | str = RunStatus.PENDING,
        review_state: ReviewState | str = ReviewState.NOT_REQUIRED,
        dataset_snapshot_ref: Optional[str] = None,
        dataset_content_hash: Optional[str] = None,
        confidence_summary: ConfidenceSummary | dict[str, Any] | None = None,
        recommendation_slots: Optional[dict[str, RecommendationSlot | dict[str, Any] | None]] = None,
        metadata: Optional[dict[str, Any]] = None,
        child_run_ids: Optional[list[str]] = None,
        run_id: Optional[str] = None,
    ) -> str:
        """Create a recipe run row and optional child links."""

        run_id = run_id or f"recipe_run_{uuid.uuid4().hex[:12]}"
        status_value = self._coerce_recipe_run_status(status)
        review_value = self._coerce_review_state(review_state)
        if not dataset_snapshot_ref and not dataset_content_hash:
            raise ValueError("A recipe run requires dataset_snapshot_ref or dataset_content_hash")
        recommendation_slots_payload = self._normalize_recommendation_slots(recommendation_slots)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    INSERT INTO evaluation_recipe_runs (
                        run_id,
                        recipe_id,
                        recipe_version,
                        status,
                        review_state,
                        dataset_snapshot_ref,
                        dataset_content_hash,
                        confidence_summary_json,
                        recommendation_slots_json,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        recipe_id,
                        recipe_version,
                        status_value,
                        review_value,
                        dataset_snapshot_ref,
                        dataset_content_hash,
                        self._json_dump_value(confidence_summary),
                        self._json_dump_mapping(recommendation_slots_payload),
                        self._json_dump_mapping(metadata or {}),
                    ),
                )
                if child_run_ids is not None:
                    self._replace_recipe_run_children(cursor, run_id, child_run_ids)
                conn.commit()
            except Exception:
                with suppress(_EVAL_DB_NONCRITICAL_EXCEPTIONS):
                    conn.rollback()
                raise

        return run_id

    def get_recipe_run(self, run_id: str) -> RecipeRunRecord | None:
        """Fetch a recipe run by id."""

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM evaluation_recipe_runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_recipe_run_record(row)
        return None

    def update_recipe_run(
        self,
        run_id: str,
        *,
        status: RunStatus | str | None = None,
        review_state: ReviewState | str | None = None,
        confidence_summary: ConfidenceSummary | dict[str, Any] | None = None,
        recommendation_slots: Optional[dict[str, RecommendationSlot | dict[str, Any] | None]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Update persisted recipe run state with tightly scoped mutable fields."""

        has_updates = any(
            value is not None
            for value in (
                status,
                review_state,
                confidence_summary,
                recommendation_slots,
                metadata,
            )
        )
        if not has_updates:
            return False

        status_value = self._coerce_recipe_run_status(status) if status is not None else None
        review_value = self._coerce_review_state(review_state) if review_state is not None else None
        confidence_payload = (
            self._json_dump_value(confidence_summary)
            if confidence_summary is not None
            else None
        )
        recommendation_payload = (
            self._json_dump_mapping(self._normalize_recommendation_slots(recommendation_slots))
            if recommendation_slots is not None
            else None
        )
        metadata_payload = self._json_dump_mapping(metadata) if metadata is not None else None

        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    UPDATE evaluation_recipe_runs
                    SET
                        status = COALESCE(?, status),
                        review_state = COALESCE(?, review_state),
                        confidence_summary_json = COALESCE(?, confidence_summary_json),
                        recommendation_slots_json = COALESCE(?, recommendation_slots_json),
                        metadata_json = COALESCE(?, metadata_json),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE run_id = ?
                    """,
                    (
                        status_value,
                        review_value,
                        confidence_payload,
                        recommendation_payload,
                        metadata_payload,
                        run_id,
                    ),
                )
                conn.commit()
                return cursor.rowcount > 0
            except Exception:
                with suppress(_EVAL_DB_NONCRITICAL_EXCEPTIONS):
                    conn.rollback()
                raise

    def list_recipe_run_children(self, parent_run_id: str) -> list[str]:
        """List child run ids for a parent recipe run."""

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT child_run_id
                FROM evaluation_recipe_run_children
                WHERE parent_run_id = ?
                ORDER BY child_order ASC, created_at ASC
                """,
                (parent_run_id,),
            )
            rows = cursor.fetchall()
            result: list[str] = []
            for row in rows:
                if isinstance(row, dict):
                    result.append(str(row["child_run_id"]))
                else:
                    result.append(str(row[0]))
            return result

    def set_recipe_run_children(self, parent_run_id: str, child_run_ids: list[str]) -> None:
        """Replace the child run ids for a parent recipe run."""

        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                self._ensure_recipe_run_exists(cursor, parent_run_id)
                self._replace_recipe_run_children(cursor, parent_run_id, child_run_ids)
                conn.commit()
            except Exception:
                with suppress(_EVAL_DB_NONCRITICAL_EXCEPTIONS):
                    conn.rollback()
                raise

    # ============= Helper Methods =============

    def _coerce_recipe_run_status(self, status: RunStatus | str) -> str:
        if isinstance(status, RunStatus):
            return status.value
        return RunStatus(str(status)).value

    def _coerce_review_state(self, review_state: ReviewState | str) -> str:
        if isinstance(review_state, ReviewState):
            return review_state.value
        return ReviewState(str(review_state)).value

    def _ensure_recipe_run_exists(self, cursor: Any, run_id: str) -> None:
        cursor.execute(
            "SELECT 1 FROM evaluation_recipe_runs WHERE run_id = ?",
            (run_id,),
        )
        if cursor.fetchone() is None:
            raise ValueError("parent recipe run does not exist")

    def _replace_recipe_run_children(self, cursor: Any, parent_run_id: str, child_run_ids: list[str]) -> None:
        cursor.execute(
            "DELETE FROM evaluation_recipe_run_children WHERE parent_run_id = ?",
            (parent_run_id,),
        )
        for child_order, child_run_id in enumerate(child_run_ids):
            cursor.execute(
                """
                INSERT INTO evaluation_recipe_run_children (
                    parent_run_id,
                    child_run_id,
                    child_order
                )
                VALUES (?, ?, ?)
                """,
                (parent_run_id, child_run_id, child_order),
            )

    def _normalize_recommendation_slots(
        self,
        recommendation_slots: Optional[dict[str, RecommendationSlot | dict[str, Any] | None]],
    ) -> Optional[dict[str, RecommendationSlot | dict[str, Any]]]:
        if recommendation_slots is None:
            return None
        normalized: dict[str, RecommendationSlot | dict[str, Any]] = {}
        for slot_name, slot_value in recommendation_slots.items():
            if slot_value is None:
                raise ValueError(
                    "RecommendationSlot values cannot be None; use RecommendationSlot(candidate_run_id=None, reason_code=...)"
                )
            normalized[slot_name] = slot_value
        return normalized
    def _json_dump_value(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            payload = value.model_dump(mode="json")
        else:
            payload = value
        return json.dumps(payload, sort_keys=True)

    def _json_dump_mapping(self, value: Optional[dict[str, Any]]) -> Optional[str]:
        if value is None:
            return None
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if item is None:
                normalized[key] = None
            elif hasattr(item, "model_dump"):
                normalized[key] = item.model_dump(mode="json")
            else:
                normalized[key] = item
        return json.dumps(normalized, sort_keys=True)

    def _parse_recipe_datetime(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if value is None:
            return datetime.now(timezone.utc)
        text = str(value).strip()
        if not text:
            return datetime.now(timezone.utc)
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)

    def _row_to_recipe_run_record(self, row: Any) -> RecipeRunRecord:
        confidence_summary = self._json_maybe(row["confidence_summary_json"], default=None)
        recommendation_slots = self._json_maybe(row["recommendation_slots_json"], default={}) or {}
        metadata = self._json_maybe(row["metadata_json"], default={}) or {}
        child_run_ids = self.list_recipe_run_children(str(row["run_id"]))

        payload = {
            "run_id": row["run_id"],
            "recipe_id": row["recipe_id"],
            "recipe_version": row["recipe_version"],
            "status": row["status"],
            "review_state": row["review_state"],
            "dataset_snapshot_ref": row["dataset_snapshot_ref"],
            "dataset_content_hash": row["dataset_content_hash"],
            "confidence_summary": confidence_summary,
            "recommendation_slots": recommendation_slots,
            "child_run_ids": child_run_ids,
            "created_at": self._parse_recipe_datetime(row["created_at"]),
            "updated_at": self._parse_recipe_datetime(row["updated_at"]) if row["updated_at"] else None,
            "metadata": metadata,
        }
        return RecipeRunRecord.model_validate(payload)

    def _ensure_unix_timestamp(self, value: Any, *, fallback_now: bool = False) -> Optional[int]:
        """Convert various timestamp representations to a Unix epoch int.

        Supports:
        - datetime instances (including tz-aware)
        - numeric (int/float) epoch values
        - strings in ISO8601 (with optional 'Z') or '%Y-%m-%d %H:%M:%S'
        Returns None when unparsable unless fallback_now=True, in which case 'now'.
        """
        if value is None:
            return int(datetime.now().timestamp()) if fallback_now else None
        # Already numeric
        if isinstance(value, (int, float)):
            try:
                return int(value)
            except _EVAL_DB_NONCRITICAL_EXCEPTIONS:
                return int(datetime.now().timestamp()) if fallback_now else None
        # datetime instance
        if isinstance(value, datetime):
            try:
                return int(value.timestamp())
            except _EVAL_DB_NONCRITICAL_EXCEPTIONS:
                return int(datetime.now().timestamp()) if fallback_now else None
        # string inputs
        if isinstance(value, str):
            s = value.strip()
            try:
                # Fast path: numeric string
                if s.isdigit():
                    return int(s)
                # Try ISO8601, tolerate trailing 'Z'
                try:
                    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                except ValueError:
                    # Try legacy SQLite format
                    dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
                return int(dt.timestamp())
            except _EVAL_DB_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Failed to parse timestamp value: value={value!r}, error={e}")
                return int(datetime.now().timestamp()) if fallback_now else None
        # Unknown type
        logger.debug(f"Unsupported timestamp type: {type(value)} value={value!r}")
        return int(datetime.now().timestamp()) if fallback_now else None

    def _coerce_datetime_filter(self, value: Optional[Any]) -> Optional[str]:
        """Coerce date filter inputs into ISO-like strings for SQL comparisons."""
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned or None
        dt: Optional[datetime]
        if isinstance(value, (int, float)):
            try:
                dt = datetime.fromtimestamp(value, tz=timezone.utc)
            except _EVAL_DB_NONCRITICAL_EXCEPTIONS:
                return None
        elif isinstance(value, datetime):
            dt = value
        else:
            return None
        try:
            if dt.tzinfo:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        except _EVAL_DB_NONCRITICAL_EXCEPTIONS:
            pass
        return dt.replace(microsecond=0).isoformat(sep=" ")

    def _json_maybe(self, value: Any, *, default: Any = None) -> Any:
        """Return JSON-decoded value.

        - If value is None/falsey: return default.
        - If value is str: json.loads(value).
        - If value is already a JSON-compatible object (dict/list/number/bool): return as-is.
        - Otherwise: return default.
        """
        if value is None or value is False or value == "":
            return default
        if isinstance(value, str):
            try:
                return json.loads(value)
            except _EVAL_DB_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Failed to json.loads value: {value!r}, error={e}")
                return default
        # Accept already-parsed JSON-like structures from JSONB (e.g., PostgreSQL)
        if isinstance(value, (dict, list, int, float, bool)):
            return value
        logger.debug(f"Unsupported JSON field type: {type(value)} value={value!r}")
        return default

    def _user_id_variants(self, user_id: Optional[str]) -> list[str]:
        """Return acceptable user id variants for filtering."""
        if not user_id:
            return []
        raw = str(user_id).strip()
        if not raw:
            return []
        variants = {raw}
        if raw.startswith("user_"):
            core = raw[5:]
            if core:
                variants.add(core)
        elif raw.isdigit():
            variants.add(f"user_{raw}")
        return list(variants)

    def _append_user_filter(
        self,
        query: str,
        params: list[Any],
        field: str,
        user_id: Optional[str],
    ) -> tuple[str, list[Any]]:
        variants = self._user_id_variants(user_id)
        if not variants:
            return query, params
        if len(variants) == 1:
            query += f" AND {field} = ?"
            params.append(variants[0])
            return query, params
        placeholders = ", ".join(["?"] * len(variants))
        query += f" AND {field} IN ({placeholders})"
        params.extend(variants)
        return query, params

    def _row_to_eval_dict(self, row) -> dict[str, Any]:
        """Convert database row to evaluation dictionary"""
        created_timestamp = self._ensure_unix_timestamp(row["created_at"], fallback_now=True)

        return {
            "id": row["id"],
            "object": "evaluation",
            "created": created_timestamp,  # Use 'created' for OpenAI compatibility
            "created_at": created_timestamp,  # Also provide created_at for backwards compatibility
            "name": row["name"],
            "description": row["description"],
            "eval_type": row["eval_type"],
            "eval_spec": self._json_maybe(row["eval_spec"], default={}),
            "dataset_id": row["dataset_id"],
            "created_by": row["created_by"] or "unknown",
            "metadata": self._json_maybe(row["metadata"], default={}),
        }

    def _row_to_run_dict(self, row) -> dict[str, Any]:
        """Convert database row to run dictionary"""
        created_timestamp = self._ensure_unix_timestamp(row["created_at"], fallback_now=True)

        # Parse optional timestamps
        started_at = self._ensure_unix_timestamp(row["started_at"], fallback_now=False)
        completed_at = self._ensure_unix_timestamp(row["completed_at"], fallback_now=False)

        return {
            "id": row["id"],
            "object": "run",
            "created": created_timestamp,  # Use 'created' for OpenAI compatibility
            "created_at": created_timestamp,  # Also provide created_at for backwards compatibility
            "eval_id": row["eval_id"],
            "status": row["status"],
            "target_model": row["target_model"] or "",
            "config": self._json_maybe(row["config"], default={}),
            "progress": self._json_maybe(row["progress"], default=None),
            "results": self._json_maybe(row["results"], default=None),
            "error_message": row["error_message"],
            "started_at": started_at,
            "completed_at": completed_at,
            "usage": self._json_maybe(row["usage"], default=None),
        }

    def _row_to_dataset_dict(
        self,
        row,
        include_samples: bool = True,
        sample_limit: Optional[int] = None,
        sample_offset: int = 0,
    ) -> dict[str, Any]:
        """Convert database row to dataset dictionary"""
        created_timestamp = self._ensure_unix_timestamp(row["created_at"], fallback_now=True)

        result = {
            "id": row["id"],
            "object": "dataset",
            "created": created_timestamp,  # Use 'created' for OpenAI compatibility
            "created_at": created_timestamp,  # Also provide created_at for backwards compatibility
            "name": row["name"],
            "description": row["description"],
            "sample_count": row["sample_count"] or 0,
            "created_by": row["created_by"] or "unknown",
            "metadata": self._json_maybe(row["metadata"], default={}),
        }

        if include_samples:
            samples = self._json_maybe(row["samples"], default=[])
            normalized_offset = max(0, int(sample_offset or 0))
            if sample_limit is not None:
                normalized_limit = max(0, int(sample_limit))
                samples = samples[normalized_offset:normalized_offset + normalized_limit]
            elif normalized_offset:
                samples = samples[normalized_offset:]
            result["samples"] = samples

        return result

    # ============= Unified Evaluation Operations =============

    def store_unified_evaluation(
        self,
        evaluation_id: str,
        name: str,
        evaluation_type: str,
        input_data: dict[str, Any],
        results: dict[str, Any],
        status: str = "completed",
        user_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None
    ) -> bool:
        """Store evaluation in the unified table if it exists, otherwise fall back to internal_evaluations."""
        if self._use_unified_table():
            # Store in unified table
            if self.backend_type == BackendType.POSTGRESQL and self.backend is not None:
                upsert = (
                    "INSERT INTO evaluations_unified ("
                    "id, evaluation_id, name, evaluation_type, input_data, results, status, user_id,"
                    "metadata, embedding_provider, embedding_model, created_at, completed_at, eval_spec"
                    ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW(), %s) "
                    "ON CONFLICT (id) DO UPDATE SET "
                    "name = EXCLUDED.name, evaluation_type = EXCLUDED.evaluation_type, input_data = EXCLUDED.input_data,"
                    "results = EXCLUDED.results, status = EXCLUDED.status, user_id = EXCLUDED.user_id,"
                    "metadata = EXCLUDED.metadata, embedding_provider = EXCLUDED.embedding_provider,"
                    "embedding_model = EXCLUDED.embedding_model, completed_at = EXCLUDED.completed_at"
                )
                try:
                    self.backend.execute(
                        upsert,
                        (
                            evaluation_id,
                            evaluation_id,
                            name or evaluation_type,
                            evaluation_type,
                            json.dumps(input_data),
                            json.dumps(results),
                            status,
                            user_id,
                            json.dumps(metadata) if metadata else None,
                            embedding_provider,
                            embedding_model,
                            json.dumps({}),
                        ),
                    )
                    return True
                except _EVAL_DB_NONCRITICAL_EXCEPTIONS as exc:
                    logger.error(f"Failed to store in unified table (PG): {exc}")
                    return False
            else:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    try:
                        cursor.execute("""
                            INSERT OR REPLACE INTO evaluations_unified (
                                id, evaluation_id, name, evaluation_type,
                                input_data, results, status, user_id,
                                metadata, embedding_provider, embedding_model,
                                created_at, completed_at, eval_spec
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'), ?)
                        """, (
                            evaluation_id,  # Use same ID for both id and evaluation_id
                            evaluation_id,
                            name or evaluation_type,
                            evaluation_type,
                            json.dumps(input_data),
                            json.dumps(results),
                            status,
                            user_id,
                            json.dumps(metadata) if metadata else None,
                            embedding_provider,
                            embedding_model,
                            json.dumps({})  # Empty eval_spec for backward compatibility
                        ))
                        conn.commit()
                        return True
                    except _EVAL_DB_NONCRITICAL_EXCEPTIONS as e:
                        logger.error(f"Failed to store in unified table: {e}")
                        conn.rollback()
                        return False
        else:
            # Store in internal_evaluations table as fallback
            if self.backend_type == BackendType.POSTGRESQL and self.backend is not None:
                upsert = (
                    "INSERT INTO internal_evaluations ("
                    "evaluation_id, evaluation_type, input_data, results, user_id, metadata, status,"
                    "embedding_provider, embedding_model, created_at, completed_at"
                    ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW()) "
                    "ON CONFLICT (evaluation_id) DO UPDATE SET "
                    "evaluation_type = EXCLUDED.evaluation_type, input_data = EXCLUDED.input_data,"
                    "results = EXCLUDED.results, user_id = EXCLUDED.user_id, metadata = EXCLUDED.metadata,"
                    "status = EXCLUDED.status, embedding_provider = EXCLUDED.embedding_provider,"
                    "embedding_model = EXCLUDED.embedding_model, completed_at = EXCLUDED.completed_at"
                )
                try:
                    self.backend.execute(
                        upsert,
                        (
                            evaluation_id,
                            evaluation_type,
                            json.dumps(input_data),
                            json.dumps(results),
                            user_id,
                            json.dumps(metadata) if metadata else None,
                            status,
                            embedding_provider,
                            embedding_model,
                        ),
                    )
                    return True
                except _EVAL_DB_NONCRITICAL_EXCEPTIONS as exc:
                    logger.error(f"Failed to store evaluation (PG): {exc}")
                    return False
            else:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    try:
                        cursor.execute("""
                            INSERT OR REPLACE INTO internal_evaluations (
                                evaluation_id, evaluation_type, input_data, results,
                                user_id, metadata, status, embedding_provider, embedding_model,
                                created_at, completed_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
                        """, (
                            evaluation_id,
                            evaluation_type,
                            json.dumps(input_data),
                            json.dumps(results),
                            user_id,
                            json.dumps(metadata) if metadata else None,
                            status,
                            embedding_provider,
                            embedding_model
                        ))
                        conn.commit()
                        return True
                    except _EVAL_DB_NONCRITICAL_EXCEPTIONS as e:
                        logger.error(f"Failed to store evaluation: {e}")
                        conn.rollback()
                        return False

    def get_unified_evaluation(self, evaluation_id: str) -> Optional[dict[str, Any]]:
        """Get evaluation from unified table if it exists, otherwise from legacy tables."""
        if self._use_unified_table():
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM evaluations_unified
                    WHERE evaluation_id = ? OR id = ?
                """, (evaluation_id, evaluation_id))

                result = cursor.fetchone()
                if result:
                    return dict(result)

        # Fall back to checking internal_evaluations table
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM internal_evaluations
                WHERE evaluation_id = ?
            """, (evaluation_id,))
            result = cursor.fetchone()
            if result:
                return dict(result)

            # Also check the evaluations table
            cursor.execute("SELECT * FROM evaluations WHERE id = ?", (evaluation_id,))
            result = cursor.fetchone()
            if result:
                return dict(result)

        return None

    # ============= Pipeline Presets Operations =============

    def upsert_pipeline_preset(self, name: str, config: dict[str, Any], user_id: Optional[str] = None) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO pipeline_presets (name, config, user_id)
                VALUES (?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    config = excluded.config,
                    updated_at = CURRENT_TIMESTAMP,
                    user_id = COALESCE(excluded.user_id, pipeline_presets.user_id)
                """,
                (name, json.dumps(config), user_id),
            )
            conn.commit()
            return True

    def get_pipeline_preset(self, name: str, *, user_id: Optional[str] = None) -> Optional[dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM pipeline_presets WHERE name = ?"
            params: list[Any] = [name]
            query, params = self._append_user_filter(query, params, "user_id", user_id)
            cursor.execute(query, params)
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "name": row["name"],
                "config": self._json_maybe(row["config"], default={}),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "user_id": row["user_id"],
            }

    def list_pipeline_presets(self, limit: int = 50, offset: int = 0, *, user_id: Optional[str] = None) -> tuple[list[dict[str, Any]], int]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            count_query = "SELECT COUNT(*) FROM pipeline_presets WHERE 1=1"
            count_params: list[Any] = []
            count_query, count_params = self._append_user_filter(count_query, count_params, "user_id", user_id)
            cursor.execute(count_query, count_params)
            total = cursor.fetchone()[0]
            list_query = (
                "SELECT * FROM pipeline_presets WHERE 1=1"
            )
            list_params: list[Any] = []
            list_query, list_params = self._append_user_filter(list_query, list_params, "user_id", user_id)
            list_query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
            list_params.extend([limit, offset])
            cursor.execute(list_query, list_params)
            rows = cursor.fetchall()
            items = [
                {
                    "name": r["name"],
                    "config": self._json_maybe(r["config"], default={}),
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "user_id": r["user_id"],
                }
                for r in rows
            ]
            return items, total

    def delete_pipeline_preset(self, name: str, *, user_id: Optional[str] = None) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "DELETE FROM pipeline_presets WHERE name = ?"
            params: list[Any] = [name]
            query, params = self._append_user_filter(query, params, "user_id", user_id)
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount > 0

    # ============= Ephemeral Collections Operations =============

    def register_ephemeral_collection(
        self, collection_name: str, ttl_seconds: int = 86400, run_id: Optional[str] = None, namespace: Optional[str] = None
    ) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO ephemeral_collections (collection_name, ttl_seconds, run_id, namespace)
                VALUES (?, ?, ?, ?)
                """,
                (collection_name, ttl_seconds, run_id, namespace),
            )
            conn.commit()
            return True

    def list_expired_ephemeral_collections(self) -> list[str]:
        if self.backend_type == BackendType.POSTGRESQL and self.backend is not None:
            result = self.backend.execute(

                    "SELECT collection_name FROM ephemeral_collections "
                    "WHERE deleted_at IS NULL AND (created_at + (ttl_seconds || ' seconds')::interval) <= NOW()"

            )
            return [r["collection_name"] for r in result.rows]
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT collection_name FROM ephemeral_collections
                WHERE deleted_at IS NULL AND datetime(created_at, '+' || ttl_seconds || ' seconds') <= CURRENT_TIMESTAMP
                """
            )
            rows = cursor.fetchall()
            return [r["collection_name"] for r in rows]

    def mark_ephemeral_deleted(self, collection_name: str) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE ephemeral_collections SET deleted_at = CURRENT_TIMESTAMP
                WHERE collection_name = ? AND deleted_at IS NULL
                """,
                (collection_name,),
            )
            conn.commit()
            return cursor.rowcount > 0
