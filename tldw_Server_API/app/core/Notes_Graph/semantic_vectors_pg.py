"""Bounded fixed-table pgvector backend for Notes semantic generations."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from types import MappingProxyType
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseBackend,
    DatabaseError,
)

from .semantic_vectors import (
    SemanticVector,
    SemanticVectorBinding,
    SemanticVectorCapabilityError,
    SemanticVectorCleanup,
    SemanticVectorError,
    SemanticVectorMatch,
)

PGVECTOR_TABLES = MappingProxyType(
    {
        384: "note_semantic_vectors_d384",
        768: "note_semantic_vectors_d768",
        1_024: "note_semantic_vectors_d1024",
        1_536: "note_semantic_vectors_d1536",
        3_072: "note_semantic_vectors_d3072",
    }
)
SEMANTIC_VECTOR_METRIC_LABELS = frozenset({"backend", "operation", "outcome"})
_TENANT_POLICY_PREDICATE = (
    "owner_user_id = current_setting('app.current_user_id', true) "
    "AND dataset_id = current_setting('app.current_dataset_id', true)"
)
_POSTGRES_OPERATION_ERRORS = (
    AttributeError,
    ConnectionError,
    DatabaseError,
    KeyError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
_POSTGRES_RESULT_ERRORS = (
    AttributeError,
    DatabaseError,
    KeyError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _vector_literal(embedding: Iterable[float]) -> str:
    return "[" + ",".join(format(value, ".17g") for value in embedding) + "]"


def _parse_vector(value: object) -> tuple[float, ...]:
    try:
        text = str(value).strip()
    except _POSTGRES_RESULT_ERRORS:
        raise SemanticVectorError(
            "notes_semantic_vector_backend_result_invalid"
        ) from None
    if not text.startswith("[") or not text.endswith("]"):
        raise SemanticVectorError("notes_semantic_vector_backend_result_invalid")
    body = text[1:-1]
    if not body:
        return ()
    try:
        return tuple(float(part) for part in body.split(","))
    except (OverflowError, TypeError, ValueError):
        raise SemanticVectorError("notes_semantic_vector_backend_result_invalid") from None


def _cleanup_confirmation(value: object) -> SemanticVectorCleanup:
    try:
        remaining = int(value or 0)
    except _POSTGRES_RESULT_ERRORS:
        raise SemanticVectorError(
            "notes_semantic_vector_backend_result_invalid"
        ) from None
    return SemanticVectorCleanup(confirmed_absent=remaining == 0)


def _normalized_policy_expression(value: object) -> str:
    try:
        expression = str(value).lower().replace("::text", "")
    except _POSTGRES_RESULT_ERRORS:
        return ""
    return "".join(expression.split()).replace("(", "").replace(")", "")


class PostgresSemanticVectorBackend:
    """Store vectors in operator-bounded shared tables protected by forced RLS."""

    name = "pgvector"

    def __init__(
        self,
        backend: DatabaseBackend,
        *,
        allowed_dimensions: frozenset[int],
    ) -> None:
        self._backend = backend
        self._allowed_dimensions = frozenset(allowed_dimensions)

    async def check_capability(self) -> None:
        await asyncio.to_thread(self._check_capability_sync)

    def _check_capability_sync(self) -> None:
        if self._backend.backend_type is not BackendType.POSTGRESQL:
            raise SemanticVectorCapabilityError("notes_semantic_pgvector_unavailable")
        if not self._allowed_dimensions or not self._allowed_dimensions <= PGVECTOR_TABLES.keys():
            raise SemanticVectorCapabilityError(
                "notes_semantic_pgvector_dimensions_unsupported"
            )
        try:
            with self._backend.transaction() as connection:
                self._backend.execute(
                    "CREATE EXTENSION IF NOT EXISTS vector",
                    connection=connection,
                    log_errors=False,
                )
                installed = bool(
                    self._backend.execute(
                        "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname='vector')",
                        connection=connection,
                        log_errors=False,
                    ).scalar
                )
        except _POSTGRES_OPERATION_ERRORS:
            raise SemanticVectorCapabilityError(
                "notes_semantic_pgvector_extension_unavailable"
            ) from None
        if not installed:
            raise SemanticVectorCapabilityError(
                "notes_semantic_pgvector_extension_unavailable"
            )
        for dimensions in sorted(self._allowed_dimensions):
            self._ensure_dimension_table(dimensions)

    def supports_dimensions(self, dimensions: int) -> bool:
        return dimensions in self._allowed_dimensions and dimensions in PGVECTOR_TABLES

    def _table_exists(self, table: str) -> bool:
        try:
            return bool(self._backend.table_exists(table))
        except _POSTGRES_OPERATION_ERRORS:
            raise SemanticVectorError("notes_semantic_pgvector_operation_failed") from None

    def _ensure_dimension_table(self, dimensions: int) -> None:
        table = PGVECTOR_TABLES.get(dimensions)
        if table is None or dimensions not in self._allowed_dimensions:
            raise SemanticVectorCapabilityError(
                "notes_semantic_pgvector_dimensions_unsupported"
            )
        policy = f"{table}_tenant_isolation"
        index = f"{table}_embedding_hnsw"
        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table} (
                owner_user_id TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                generation_id TEXT NOT NULL,
                vector_id TEXT NOT NULL,
                embedding vector({dimensions}) NOT NULL,
                PRIMARY KEY (owner_user_id, dataset_id, generation_id, vector_id)
            )
        """  # nosec B608 - identifiers and dimensions come only from PGVECTOR_TABLES.
        create_policy_sql = (
            f"CREATE POLICY {policy} ON {table} "  # nosec B608
            f"USING ({_TENANT_POLICY_PREDICATE}) "
            f"WITH CHECK ({_TENANT_POLICY_PREDICATE})"
        )
        try:
            with self._backend.transaction() as connection:
                self._backend.execute(create_table_sql, connection=connection, log_errors=False)
                self._backend.execute(
                    f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY",  # nosec B608
                    connection=connection,
                    log_errors=False,
                )
                self._backend.execute(
                    f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY",  # nosec B608
                    connection=connection,
                    log_errors=False,
                )
                self._backend.execute(
                    f"DROP POLICY IF EXISTS {policy} ON {table}",  # nosec B608
                    connection=connection,
                    log_errors=False,
                )
                self._backend.execute(
                    create_policy_sql,
                    connection=connection,
                    log_errors=False,
                )
                self._backend.execute(
                    f"CREATE INDEX IF NOT EXISTS {index} ON {table} "  # nosec B608
                    "USING hnsw (embedding vector_cosine_ops)",
                    connection=connection,
                    log_errors=False,
                )
        except _POSTGRES_OPERATION_ERRORS:
            raise SemanticVectorCapabilityError(
                "notes_semantic_pgvector_schema_unavailable"
            ) from None
        self._verify_dimension_table(dimensions)

    def _verify_dimension_table(self, dimensions: int) -> None:
        table = PGVECTOR_TABLES[dimensions]
        try:
            schema = self._backend.execute(
                "SELECT c.relrowsecurity,c.relforcerowsecurity,"
                "format_type(a.atttypid,a.atttypmod) AS vector_type "
                "FROM pg_class c JOIN pg_attribute a ON a.attrelid=c.oid "
                "WHERE c.relname=? AND a.attname='embedding'",
                (table,),
                log_errors=False,
            ).rows
            primary_key = self._backend.execute(
                "SELECT pg_get_constraintdef(oid) AS definition FROM pg_constraint "
                "WHERE conrelid=?::regclass AND contype='p'",
                (table,),
                log_errors=False,
            ).rows
            indexes = self._backend.execute(
                "SELECT indexdef FROM pg_indexes WHERE tablename=?",
                (table,),
                log_errors=False,
            ).rows
            policies = self._backend.execute(
                "SELECT policyname,qual,with_check FROM pg_policies WHERE tablename=?",
                (table,),
                log_errors=False,
            ).rows
        except _POSTGRES_OPERATION_ERRORS:
            raise SemanticVectorCapabilityError(
                "notes_semantic_pgvector_schema_unavailable"
            ) from None
        expected_key = "PRIMARY KEY (owner_user_id, dataset_id, generation_id, vector_id)"
        expected_policy = f"{table}_tenant_isolation"
        policy_qual = str(policies[0].get("qual", "")) if len(policies) == 1 else ""
        policy_check = str(policies[0].get("with_check", "")) if len(policies) == 1 else ""
        expected_policy_expression = _normalized_policy_expression(
            _TENANT_POLICY_PREDICATE
        )
        valid = (
            len(schema) == 1
            and schema[0].get("relrowsecurity") is True
            and schema[0].get("relforcerowsecurity") is True
            and schema[0].get("vector_type") == f"vector({dimensions})"
            and any(row.get("definition") == expected_key for row in primary_key)
            and any(
                "USING hnsw (embedding vector_cosine_ops)" in str(row.get("indexdef", ""))
                for row in indexes
            )
            and len(policies) == 1
            and policies[0].get("policyname") == expected_policy
            and _normalized_policy_expression(policy_qual)
            == expected_policy_expression
            and _normalized_policy_expression(policy_check)
            == expected_policy_expression
        )
        if not valid:
            raise SemanticVectorCapabilityError(
                "notes_semantic_pgvector_schema_unavailable"
            )

    @staticmethod
    def _scope(
        backend: DatabaseBackend,
        connection: Any,
        binding: SemanticVectorBinding,
    ) -> None:
        backend.execute(
            "SELECT set_config('app.current_user_id', ?, true)",
            (binding.owner_user_id,),
            connection=connection,
            log_errors=False,
        )
        backend.execute(
            "SELECT set_config('app.current_dataset_id', ?, true)",
            (binding.dataset_id,),
            connection=connection,
            log_errors=False,
        )

    async def create_generation_storage(self, binding: SemanticVectorBinding) -> None:
        await asyncio.to_thread(self._ensure_dimension_table, binding.dimensions)

    def _upsert_sync(
        self,
        binding: SemanticVectorBinding,
        vectors: tuple[SemanticVector, ...],
    ) -> int:
        table = PGVECTOR_TABLES[binding.dimensions]
        if not self._table_exists(table):
            raise SemanticVectorError("notes_semantic_vector_storage_missing")
        sql = (
            f"INSERT INTO {table}(owner_user_id,dataset_id,generation_id,vector_id,embedding) "  # nosec B608
            "VALUES (?,?,?,?,?::vector) "
            "ON CONFLICT(owner_user_id,dataset_id,generation_id,vector_id) "
            "DO UPDATE SET embedding=EXCLUDED.embedding"
        )
        params = [
            (
                binding.owner_user_id,
                binding.dataset_id,
                binding.generation_id,
                vector.vector_id,
                _vector_literal(vector.embedding),
            )
            for vector in vectors
        ]
        try:
            with self._backend.transaction() as connection:
                self._scope(self._backend, connection, binding)
                self._backend.execute_many(sql, params, connection=connection)
        except _POSTGRES_OPERATION_ERRORS:
            raise SemanticVectorError("notes_semantic_pgvector_operation_failed") from None
        return len(vectors)

    async def upsert(
        self,
        binding: SemanticVectorBinding,
        vectors: tuple[SemanticVector, ...],
    ) -> int:
        return await asyncio.to_thread(self._upsert_sync, binding, vectors)

    def _fetch_sync(
        self,
        binding: SemanticVectorBinding,
        vector_ids: tuple[str, ...],
    ) -> tuple[SemanticVector, ...]:
        table = PGVECTOR_TABLES[binding.dimensions]
        if not self._table_exists(table):
            return ()
        sql = (
            f"SELECT vector_id,embedding::text AS embedding_text FROM {table} "  # nosec B608
            "WHERE owner_user_id=? AND dataset_id=? AND generation_id=? "
            "AND vector_id=ANY(?::text[])"
        )
        try:
            with self._backend.transaction() as connection:
                self._scope(self._backend, connection, binding)
                rows = self._backend.execute(
                    sql,
                    (
                        binding.owner_user_id,
                        binding.dataset_id,
                        binding.generation_id,
                        list(vector_ids),
                    ),
                    connection=connection,
                    log_errors=False,
                ).rows
        except _POSTGRES_OPERATION_ERRORS:
            raise SemanticVectorError("notes_semantic_pgvector_operation_failed") from None
        try:
            return tuple(
                SemanticVector(
                    vector_id=str(row["vector_id"]),
                    embedding=_parse_vector(row["embedding_text"]),
                )
                for row in rows
            )
        except SemanticVectorError:
            raise
        except _POSTGRES_RESULT_ERRORS:
            raise SemanticVectorError(
                "notes_semantic_vector_backend_result_invalid"
            ) from None

    async def fetch(
        self,
        binding: SemanticVectorBinding,
        vector_ids: tuple[str, ...],
    ) -> tuple[SemanticVector, ...]:
        return await asyncio.to_thread(self._fetch_sync, binding, vector_ids)

    def _query_sync(
        self,
        binding: SemanticVectorBinding,
        query_vectors: tuple[tuple[float, ...], ...],
        limit: int,
    ) -> tuple[tuple[SemanticVectorMatch, ...], ...]:
        table = PGVECTOR_TABLES[binding.dimensions]
        if not self._table_exists(table):
            return tuple(() for _ in query_vectors)
        sql = (
            "WITH scoped AS MATERIALIZED ("
            f"SELECT vector_id,embedding FROM {table} "  # nosec B608
            "WHERE owner_user_id=? AND dataset_id=? AND generation_id=?"
            ") "
            "SELECT vector_id,(embedding <=> ?::vector) AS distance FROM scoped "
            "ORDER BY embedding <=> ?::vector,vector_id LIMIT ?"
        )
        results: list[Any] = []
        try:
            with self._backend.transaction() as connection:
                self._scope(self._backend, connection, binding)
                for query_vector in query_vectors:
                    literal = _vector_literal(query_vector)
                    results.append(
                        self._backend.execute(
                            sql,
                            (
                                binding.owner_user_id,
                                binding.dataset_id,
                                binding.generation_id,
                                literal,
                                literal,
                                limit,
                            ),
                            connection=connection,
                            log_errors=False,
                        )
                    )
        except _POSTGRES_OPERATION_ERRORS:
            raise SemanticVectorError("notes_semantic_pgvector_operation_failed") from None
        try:
            return tuple(
                tuple(
                    SemanticVectorMatch(
                        vector_id=str(row["vector_id"]),
                        distance=float(row["distance"]),
                    )
                    for row in result.rows
                )
                for result in results
            )
        except _POSTGRES_RESULT_ERRORS:
            raise SemanticVectorError(
                "notes_semantic_vector_backend_result_invalid"
            ) from None

    async def query(
        self,
        binding: SemanticVectorBinding,
        query_vectors: tuple[tuple[float, ...], ...],
        *,
        limit: int,
    ) -> tuple[tuple[SemanticVectorMatch, ...], ...]:
        return await asyncio.to_thread(
            self._query_sync,
            binding,
            query_vectors,
            limit,
        )

    def _delete_ids_sync(
        self,
        binding: SemanticVectorBinding,
        vector_ids: tuple[str, ...],
    ) -> SemanticVectorCleanup:
        table = PGVECTOR_TABLES[binding.dimensions]
        if not self._table_exists(table):
            return SemanticVectorCleanup(confirmed_absent=True)
        predicate = (
            "owner_user_id=? AND dataset_id=? AND generation_id=? "
            "AND vector_id=ANY(?::text[])"
        )
        params = (
            binding.owner_user_id,
            binding.dataset_id,
            binding.generation_id,
            list(vector_ids),
        )
        try:
            with self._backend.transaction() as connection:
                self._scope(self._backend, connection, binding)
                self._backend.execute(
                    f"DELETE FROM {table} WHERE {predicate}",  # nosec B608
                    params,
                    connection=connection,
                    log_errors=False,
                )
                remaining = self._backend.execute(
                    f"SELECT COUNT(*) AS count FROM {table} WHERE {predicate}",  # nosec B608
                    params,
                    connection=connection,
                    log_errors=False,
                ).scalar
        except _POSTGRES_OPERATION_ERRORS:
            raise SemanticVectorError("notes_semantic_pgvector_operation_failed") from None
        return _cleanup_confirmation(remaining)

    async def delete_ids(
        self,
        binding: SemanticVectorBinding,
        vector_ids: tuple[str, ...],
    ) -> SemanticVectorCleanup:
        return await asyncio.to_thread(self._delete_ids_sync, binding, vector_ids)

    def _delete_generation_sync(
        self,
        binding: SemanticVectorBinding,
    ) -> SemanticVectorCleanup:
        table = PGVECTOR_TABLES[binding.dimensions]
        if not self._table_exists(table):
            return SemanticVectorCleanup(confirmed_absent=True)
        predicate = "owner_user_id=? AND dataset_id=? AND generation_id=?"
        params = (
            binding.owner_user_id,
            binding.dataset_id,
            binding.generation_id,
        )
        try:
            with self._backend.transaction() as connection:
                self._scope(self._backend, connection, binding)
                self._backend.execute(
                    f"DELETE FROM {table} WHERE {predicate}",  # nosec B608
                    params,
                    connection=connection,
                    log_errors=False,
                )
                remaining = self._backend.execute(
                    f"SELECT COUNT(*) AS count FROM {table} WHERE {predicate}",  # nosec B608
                    params,
                    connection=connection,
                    log_errors=False,
                ).scalar
        except _POSTGRES_OPERATION_ERRORS:
            raise SemanticVectorError("notes_semantic_pgvector_operation_failed") from None
        return _cleanup_confirmation(remaining)

    async def delete_generation(
        self,
        binding: SemanticVectorBinding,
    ) -> SemanticVectorCleanup:
        return await asyncio.to_thread(self._delete_generation_sync, binding)


__all__ = [
    "PGVECTOR_TABLES",
    "SEMANTIC_VECTOR_METRIC_LABELS",
    "PostgresSemanticVectorBackend",
]
