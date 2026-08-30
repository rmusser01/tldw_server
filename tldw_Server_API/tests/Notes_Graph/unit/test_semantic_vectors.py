"""Unit contracts for the Notes semantic vector facade and factory."""

from __future__ import annotations

import asyncio
import time
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseError
from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import (
    SemanticDimensionState,
    SemanticGenerationState,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_settings import SemanticIndexSettings
from tldw_Server_API.app.core.Notes_Graph.semantic_vectors import (
    NotesSemanticVectorStore,
    SemanticVector,
    SemanticVectorBackend,
    SemanticVectorBinding,
    SemanticVectorBindingError,
    SemanticVectorCapabilityError,
    SemanticVectorCleanup,
    SemanticVectorError,
    SemanticVectorMatch,
    SemanticVectorValidationError,
    create_semantic_vector_store,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_vectors_pg import (
    PostgresSemanticVectorBackend,
)
from tldw_Server_API.tests.Notes_Graph.vector_contract import axis_vector

pytestmark = pytest.mark.unit


class _Authority:
    def __init__(self, *, owner: str = "owner-a", dimensions: int = 384) -> None:
        self.owner_user_id = owner
        self.calls = 0
        self.generation = SimpleNamespace(
            id="generation-a",
            owner_user_id=owner,
            dataset_id="dataset-a",
            state=SemanticGenerationState.STAGING,
            dimension_state=SemanticDimensionState.RESOLVED,
            dimensions=dimensions,
        )

    def get_generation(self, dataset_id: str, generation_id: str):
        self.calls += 1
        if dataset_id != self.generation.dataset_id or generation_id != self.generation.id:
            return None
        return self.generation


class _Backend(SemanticVectorBackend):
    name = "memory"

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def check_capability(self) -> None:
        self.calls.append("capability")

    def supports_dimensions(self, dimensions: int) -> bool:
        return dimensions == 384

    async def create_generation_storage(self, binding: SemanticVectorBinding) -> None:
        self.calls.append("create")

    async def upsert(self, binding: SemanticVectorBinding, vectors: tuple[SemanticVector, ...]) -> int:
        self.calls.append("upsert")
        return len(vectors)

    async def fetch(self, binding: SemanticVectorBinding, vector_ids: tuple[str, ...]):
        self.calls.append("fetch")
        return tuple(
            SemanticVector(vector_id=vector_id, embedding=axis_vector(binding.dimensions, 0))
            for vector_id in reversed(vector_ids)
        )

    async def query(self, binding: SemanticVectorBinding, query_vectors, *, limit: int):
        self.calls.append("query")
        return tuple(
            (SemanticVectorMatch(vector_id="vector-a", distance=0.0),)
            for _ in query_vectors
        )

    async def delete_ids(self, binding: SemanticVectorBinding, vector_ids: tuple[str, ...]):
        self.calls.append("delete_ids")
        return SemanticVectorCleanup(confirmed_absent=True)

    async def delete_generation(self, binding: SemanticVectorBinding):
        self.calls.append("delete_generation")
        return SemanticVectorCleanup(confirmed_absent=True)


class _MalformedResultBackend(_Backend):
    async def query(self, binding: SemanticVectorBinding, query_vectors, *, limit: int):
        return ((SemanticVectorMatch(vector_id="vector-a", distance=object()),),)


class _MalformedVectorIdBackend(_Backend):
    async def fetch(self, binding: SemanticVectorBinding, vector_ids: tuple[str, ...]):
        return (SemanticVector(vector_id="\ud800", embedding=axis_vector(384, 0)),)


class _SlowAuthority(_Authority):
    def get_generation(self, dataset_id: str, generation_id: str):
        time.sleep(0.05)
        return super().get_generation(dataset_id, generation_id)


class _PgResult:
    def __init__(self, *, rows=(), scalar=None) -> None:
        self.rows = rows
        self.scalar = scalar


class _FakePgBackend:
    backend_type = BackendType.POSTGRESQL

    def __init__(
        self,
        *,
        table_error: Exception | None = None,
        table_result: object = True,
        rows=(),
        scalar=0,
    ) -> None:
        self.table_error = table_error
        self.table_result = table_result
        self.rows = rows
        self.scalar = scalar
        self.executions: list[tuple[str, object, object | None]] = []

    def table_exists(self, table_name: str) -> bool:
        if self.table_error is not None:
            raise self.table_error
        return self.table_result  # type: ignore[return-value]

    def transaction(self):
        return nullcontext(object())

    def execute(self, sql: str, params=(), *, connection=None, log_errors=True):
        self.executions.append((sql, params, connection))
        if "COUNT(*)" in sql:
            return _PgResult(scalar=self.scalar)
        if "embedding::text" in sql or " AS distance" in sql:
            return _PgResult(rows=self.rows)
        return _PgResult()

    def execute_many(self, sql: str, params, *, connection=None) -> None:
        return None


class _BadString:
    def __str__(self) -> str:
        raise ValueError("raw conversion detail")


class _IntLike:
    def __int__(self) -> int:
        return 0


class _PgVersionBackend(_FakePgBackend):
    def __init__(self, version: object) -> None:
        super().__init__()
        self.version = version

    def execute(self, sql: str, params=(), *, connection=None, log_errors=True):
        self.executions.append((sql, params, connection))
        if "pg_extension" in sql and "extversion" in sql:
            return _PgResult(rows=({"extversion": self.version},))
        if "pg_extension" in sql:
            return _PgResult(scalar=True)
        return _PgResult()


class _SchemaPgBackend(_FakePgBackend):
    def __init__(self, *, malformed_catalog: bool = False) -> None:
        super().__init__()
        self.connection = object()
        self.malformed_catalog = malformed_catalog

    def transaction(self):
        return nullcontext(self.connection)

    def execute(self, sql: str, params=(), *, connection=None, log_errors=True):
        self.executions.append((sql, params, connection))
        table = "note_semantic_vectors_d384"
        if "format_type" in sql:
            rows = (
                {
                    "schemaname": "public",
                    "tablename": table,
                    "relkind": "r",
                    "relrowsecurity": True,
                    "relforcerowsecurity": True,
                    "vector_type": "vector(384)",
                },
            )
            return _PgResult(rows=None if self.malformed_catalog else rows)
        if "pg_get_constraintdef" in sql:
            return _PgResult(
                rows=(
                    {
                        "definition": "PRIMARY KEY (owner_user_id, dataset_id, generation_id, vector_id)"
                    },
                )
            )
        if "pg_get_indexdef" in sql:
            return _PgResult(
                rows=(
                    {
                        "schemaname": "public",
                        "tablename": table,
                        "indexname": f"{table}_embedding_hnsw",
                        "indexdef": f"CREATE INDEX {table}_embedding_hnsw ON public.{table} USING hnsw (embedding vector_cosine_ops)",
                        "key_definition": "embedding",
                        "access_method": "hnsw",
                        "operator_class": "vector_cosine_ops",
                        "indnkeyatts": 1,
                        "indnatts": 1,
                        "indisvalid": True,
                        "indisready": True,
                        "unqualified": True,
                    },
                )
            )
        if "pg_policies" in sql:
            predicate = (
                "owner_user_id = current_setting('app.current_user_id', true) "
                "AND dataset_id = current_setting('app.current_dataset_id', true)"
            )
            return _PgResult(
                rows=(
                    {
                        "schemaname": "public",
                        "tablename": table,
                        "policyname": f"{table}_tenant_isolation",
                        "permissive": "PERMISSIVE",
                        "roles": ["public"],
                        "cmd": "ALL",
                        "qual": predicate,
                        "with_check": predicate,
                    },
                )
            )
        return _PgResult()


async def _invoke_pg_operation(
    backend: PostgresSemanticVectorBackend,
    operation: str,
) -> None:
    binding = SemanticVectorBinding("owner-a", "dataset-a", "generation-a", 384)
    vector = SemanticVector("vector-a", axis_vector(384, 0))
    if operation == "upsert":
        await backend.upsert(binding, (vector,))
    elif operation == "fetch":
        await backend.fetch(binding, (vector.vector_id,))
    elif operation == "query":
        await backend.query(binding, (vector.embedding,), limit=1)
    elif operation == "delete_ids":
        await backend.delete_ids(binding, (vector.vector_id,))
    else:
        await backend.delete_generation(binding)


@pytest.mark.asyncio
async def test_facade_revalidates_authoritative_binding_before_every_operation() -> None:
    authority = _Authority()
    backend = _Backend()
    store = NotesSemanticVectorStore(authority=authority, backend=backend)
    vector = SemanticVector("vector-a", axis_vector(384, 0))

    await store.create_generation_storage("dataset-a", "generation-a")
    await store.upsert("dataset-a", "generation-a", (vector,))
    assert await store.fetch("dataset-a", "generation-a", ("vector-a",)) == (vector,)
    await store.query("dataset-a", "generation-a", (vector.embedding,), limit=1)
    await store.delete_ids("dataset-a", "generation-a", ("vector-a",))
    await store.delete_generation("dataset-a", "generation-a")

    assert authority.calls == 6
    assert backend.calls == [
        "create",
        "upsert",
        "fetch",
        "query",
        "delete_ids",
        "delete_generation",
    ]


@pytest.mark.asyncio
async def test_facade_authority_lookup_does_not_block_event_loop() -> None:
    store = NotesSemanticVectorStore(authority=_SlowAuthority(), backend=_Backend())
    ticker = asyncio.create_task(asyncio.sleep(0.01))

    await store.create_generation_storage("dataset-a", "generation-a")
    ticker_completed_during_lookup = ticker.done()
    await ticker

    assert ticker_completed_during_lookup is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("generation_change", "code"),
    (
        ({"owner_user_id": "owner-b"}, "notes_semantic_vector_owner_mismatch"),
        ({"dataset_id": "dataset-b"}, "notes_semantic_vector_binding_invalid"),
        ({"dimension_state": SemanticDimensionState.PENDING, "dimensions": None}, "notes_semantic_vector_dimensions_unresolved"),
        ({"dimensions": 768}, "notes_semantic_vector_dimensions_unsupported"),
    ),
)
async def test_facade_rejects_invalid_or_unsupported_authority_binding(
    generation_change: dict[str, object],
    code: str,
) -> None:
    authority = _Authority()
    authority.generation = SimpleNamespace(**{**vars(authority.generation), **generation_change})
    store = NotesSemanticVectorStore(authority=authority, backend=_Backend())

    with pytest.raises(SemanticVectorBindingError) as exc_info:
        await store.create_generation_storage("dataset-a", "generation-a")
    assert exc_info.value.code == code
    assert str(exc_info.value) == code


@pytest.mark.asyncio
async def test_facade_rejects_duplicate_ids_and_sanitizes_backend_results() -> None:
    authority = _Authority()
    store = NotesSemanticVectorStore(authority=authority, backend=_Backend())
    vector = SemanticVector("vector-a", axis_vector(384, 0))

    with pytest.raises(ValueError, match="notes_semantic_vector_ids_duplicate"):
        await store.upsert("dataset-a", "generation-a", (vector, vector))

    fetched = await store.fetch(
        "dataset-a",
        "generation-a",
        ("vector-a", "vector-b"),
    )
    assert [item.vector_id for item in fetched] == ["vector-a", "vector-b"]


@pytest.mark.asyncio
async def test_facade_maps_malformed_backend_distance_to_stable_error() -> None:
    store = NotesSemanticVectorStore(
        authority=_Authority(),
        backend=_MalformedResultBackend(),
    )

    with pytest.raises(SemanticVectorValidationError) as exc_info:
        await store.query(
            "dataset-a",
            "generation-a",
            (axis_vector(384, 0),),
            limit=1,
        )

    assert exc_info.value.code == "notes_semantic_vector_backend_result_invalid"


@pytest.mark.asyncio
async def test_facade_maps_lone_surrogate_input_to_stable_vector_id_error() -> None:
    store = NotesSemanticVectorStore(authority=_Authority(), backend=_Backend())

    with pytest.raises(SemanticVectorValidationError) as exc_info:
        await store.fetch("dataset-a", "generation-a", ("\ud800",))

    assert exc_info.value.code == "notes_semantic_vector_id_invalid"


@pytest.mark.asyncio
async def test_facade_maps_lone_surrogate_backend_id_to_stable_vector_id_error() -> None:
    store = NotesSemanticVectorStore(
        authority=_Authority(),
        backend=_MalformedVectorIdBackend(),
    )

    with pytest.raises(SemanticVectorValidationError) as exc_info:
        await store.fetch("dataset-a", "generation-a", ("vector-a",))

    assert exc_info.value.code == "notes_semantic_vector_id_invalid"


@pytest.mark.asyncio
async def test_facade_rejects_oversized_or_boolean_query_batches_before_authority_io() -> None:
    authority = _Authority()
    backend = _Backend()
    store = NotesSemanticVectorStore(
        authority=authority,
        backend=backend,
        max_query_vectors_per_call=1,
    )

    for query_vectors in ((axis_vector(384, 0), axis_vector(384, 1)), True):
        with pytest.raises(SemanticVectorValidationError) as exc_info:
            await store.query(  # type: ignore[arg-type]
                "dataset-a",
                "generation-a",
                query_vectors,
                limit=1,
            )
        assert exc_info.value.code == "notes_semantic_vector_query_count_invalid"

    assert authority.calls == 0
    assert backend.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    ("upsert", "fetch", "query", "delete_ids", "delete_generation"),
)
async def test_pgvector_operations_map_table_probe_failures_to_stable_error(
    operation: str,
) -> None:
    backend = PostgresSemanticVectorBackend(
        _FakePgBackend(table_error=DatabaseError("sensitive table detail")),
        allowed_dimensions=frozenset({384}),
    )

    with pytest.raises(SemanticVectorError) as exc_info:
        await _invoke_pg_operation(backend, operation)

    assert exc_info.value.code == "notes_semantic_pgvector_operation_failed"
    assert "sensitive table detail" not in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ambiguous",
    (None, 0, 1, "", "false", (), [], object()),
)
async def test_pgvector_table_probe_rejects_ambiguous_absence_results(
    ambiguous: object,
) -> None:
    backend = PostgresSemanticVectorBackend(
        _FakePgBackend(table_result=ambiguous),
        allowed_dimensions=frozenset({384}),
    )

    with pytest.raises(SemanticVectorError) as exc_info:
        await _invoke_pg_operation(backend, "delete_generation")

    assert exc_info.value.code == "notes_semantic_vector_backend_result_invalid"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row",
    (
        {},
        {"vector_id": "vector-a", "embedding_text": _BadString()},
        {"vector_id": _BadString(), "embedding_text": "[1,0]"},
    ),
)
async def test_pgvector_fetch_maps_malformed_rows_to_stable_error(row: dict) -> None:
    backend = PostgresSemanticVectorBackend(
        _FakePgBackend(rows=(row,)),
        allowed_dimensions=frozenset({384}),
    )
    binding = SemanticVectorBinding("owner-a", "dataset-a", "generation-a", 384)

    with pytest.raises(SemanticVectorError) as exc_info:
        await backend.fetch(binding, ("vector-a",))

    assert exc_info.value.code == "notes_semantic_vector_backend_result_invalid"
    assert "raw conversion detail" not in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("delete_ids", "delete_generation"))
@pytest.mark.parametrize(
    "malformed",
    (None, False, True, "0", "1", 0.0, 1.0, -1, _IntLike(), object()),
)
async def test_pgvector_cleanup_maps_malformed_confirmation_to_stable_error(
    operation: str,
    malformed: object,
) -> None:
    backend = PostgresSemanticVectorBackend(
        _FakePgBackend(scalar=malformed),
        allowed_dimensions=frozenset({384}),
    )

    with pytest.raises(SemanticVectorError) as exc_info:
        await _invoke_pg_operation(backend, operation)

    assert exc_info.value.code == "notes_semantic_vector_backend_result_invalid"


@pytest.mark.asyncio
async def test_pgvector_query_maps_malformed_rows_to_stable_result_error() -> None:
    raw_backend = _FakePgBackend(rows=({},))
    backend = PostgresSemanticVectorBackend(
        raw_backend,
        allowed_dimensions=frozenset({384}),
    )
    binding = SemanticVectorBinding("owner-a", "dataset-a", "generation-a", 384)

    with pytest.raises(SemanticVectorError) as exc_info:
        await backend.query(binding, (axis_vector(384, 0),), limit=1)

    assert exc_info.value.code == "notes_semantic_vector_backend_result_invalid"


@pytest.mark.asyncio
async def test_pgvector_query_uses_bounded_iterative_hnsw_with_tenant_filters() -> None:
    raw_backend = _FakePgBackend(rows=())
    backend = PostgresSemanticVectorBackend(
        raw_backend,
        allowed_dimensions=frozenset({384}),
        hnsw_max_scan_tuples=1_234,
    )
    binding = SemanticVectorBinding("owner-a", "dataset-a", "generation-a", 384)

    await backend.query(binding, (axis_vector(384, 0),), limit=1)

    query_sql = next(
        sql for sql, _params, _connection in raw_backend.executions if " AS distance" in sql
    )
    settings = [
        (sql, params)
        for sql, params, _connection in raw_backend.executions
        if "hnsw." in sql
    ]
    assert "MATERIALIZED" not in query_sql
    assert "FROM (SELECT vector_id,(embedding <=> ?::vector) AS distance" in query_sql
    assert "WHERE owner_user_id=? AND dataset_id=? AND generation_id=?" in query_sql
    assert "ORDER BY embedding <=> ?::vector LIMIT ?" in query_sql
    assert ") AS candidates ORDER BY distance,vector_id LIMIT ?" in query_sql
    assert settings == [
        ("SELECT set_config('hnsw.iterative_scan', ?, true)", ("strict_order",)),
        ("SELECT set_config('hnsw.max_scan_tuples', ?, true)", ("1234",)),
    ]


@pytest.mark.asyncio
async def test_pgvector_rejects_extension_versions_before_iterative_hnsw() -> None:
    backend = PostgresSemanticVectorBackend(
        _PgVersionBackend("0.7.4"),
        allowed_dimensions=frozenset({384}),
    )

    with pytest.raises(SemanticVectorCapabilityError) as exc_info:
        await backend.check_capability()

    assert exc_info.value.code == "notes_semantic_pgvector_extension_version_unsupported"


def test_pgvector_verifies_schema_inside_repair_transaction_and_current_schema() -> None:
    raw_backend = _SchemaPgBackend()
    backend = PostgresSemanticVectorBackend(
        raw_backend,
        allowed_dimensions=frozenset({384}),
    )

    backend._ensure_dimension_table(384)

    catalog_queries = [
        (sql, connection)
        for sql, _params, connection in raw_backend.executions
        if any(
            marker in sql
            for marker in (
                "format_type",
                "pg_get_constraintdef",
                "pg_get_indexdef",
                "pg_policies",
            )
        )
    ]
    assert len(catalog_queries) == 4
    assert all("current_schema()" in sql for sql, _connection in catalog_queries)
    assert all(connection is raw_backend.connection for _sql, connection in catalog_queries)


def test_pgvector_maps_malformed_catalog_shape_to_schema_capability_error() -> None:
    backend = PostgresSemanticVectorBackend(
        _SchemaPgBackend(malformed_catalog=True),
        allowed_dimensions=frozenset({384}),
    )

    with pytest.raises(SemanticVectorCapabilityError) as exc_info:
        backend._ensure_dimension_table(384)

    assert exc_info.value.code == "notes_semantic_pgvector_schema_unavailable"


@pytest.mark.asyncio
async def test_factory_returns_typed_capability_failures() -> None:
    authority = _Authority()
    settings = replace(
        SemanticIndexSettings(),
        pgvector_allowed_dimensions=frozenset({384}),
    )

    with pytest.raises(SemanticVectorCapabilityError) as unknown:
        await create_semantic_vector_store("unknown", authority=authority, settings=settings)
    assert unknown.value.code == "notes_semantic_vector_backend_unsupported"
    with pytest.raises(SemanticVectorCapabilityError) as missing_chroma:
        await create_semantic_vector_store("chromadb", authority=authority, settings=settings)
    assert missing_chroma.value.code == "notes_semantic_chroma_unavailable"
    with pytest.raises(SemanticVectorCapabilityError) as missing_postgres:
        await create_semantic_vector_store("pgvector", authority=authority, settings=settings)
    assert missing_postgres.value.code == "notes_semantic_pgvector_unavailable"
