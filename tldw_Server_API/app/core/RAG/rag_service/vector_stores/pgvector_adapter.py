"""
PGVector adapter implementation for VectorStoreAdapter.

Notes:
- Uses psycopg/psycopg2 when available; imports deferred until initialize() to avoid hard dependency at import time.
- Stores each logical collection in a separate table named vs_<sanitized_collection> with a vector column.
- Requires pgvector extension installed in the target database.
"""
import asyncio
import contextlib
import re
from typing import Any, Optional, cast

from loguru import logger
from prometheus_client import Counter, Histogram

from .base import VectorSearchResult, VectorStoreAdapter, VectorStoreConfig

try:
    from pgvector.psycopg import Vector as _PgVector
    from pgvector.psycopg import register_vector as _register_pgvector
except ImportError:  # pragma: no cover - optional dependency
    _register_pgvector = None
    _PgVector = None


class PGVectorError(RuntimeError):
    """Base error for PGVector adapter failures."""


class PGVectorNotInitializedError(PGVectorError):
    """Raised when pgvector connections are not initialized."""

    def __init__(self) -> None:
        super().__init__("PGVector connection not initialized")


class InvalidIndexTypeError(ValueError):
    """Raised when an unsupported index type is requested."""

    def __init__(self) -> None:
        super().__init__("index_type must be one of: hnsw, ivfflat, drop")


class InvalidQueryVectorError(TypeError):
    """Raised when the query vector is not a sequence."""

    def __init__(self) -> None:
        super().__init__("query_vector must be a sequence of floats")


class InvalidQueryVectorValueError(TypeError):
    """Raised when the query vector contains non-numeric values."""

    def __init__(self) -> None:
        super().__init__("query_vector must contain numbers")


class PGVectorAdapter(VectorStoreAdapter):
    # Prometheus metrics (module-level singletons per process)
    _H_UPSERT_LAT = Histogram(
        "pgvector_upsert_latency_seconds",
        "Latency for pgvector upsert operations",
        ["collection"],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, float("inf")),
    )
    _H_QUERY_LAT = Histogram(
        "pgvector_query_latency_seconds",
        "Latency for pgvector search queries",
        ["collection"],
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, float("inf")),
    )
    _H_DELETE_LAT = Histogram(
        "pgvector_delete_latency_seconds",
        "Latency for pgvector delete operations",
        ["collection"],
    )
    _C_ROWS_UPSERTED = Counter(
        "pgvector_rows_upserted_total",
        "Rows upserted into pgvector",
        ["collection"],
    )
    _C_ROWS_DELETED = Counter(
        "pgvector_rows_deleted_total",
        "Rows deleted from pgvector",
        ["collection"],
    )
    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self._conn: Optional[Any] = None  # Single connection fallback
        self._pool: Optional[Any] = None  # psycopg_pool.ConnectionPool when available
        self._driver: Optional[str] = None  # 'psycopg_pool' | 'psycopg' | 'psycopg2'
        self._ef_search = int(self.config.connection_params.get('hnsw_ef_search', 64))
        self._vector_cls: Optional[type] = None  # pgvector.Vector when available

    async def initialize(self) -> None:
        if self._initialized:
            return
        try:
            dsn = self._build_dsn(self.config.connection_params)
            # Prefer psycopg v3 with pooling
            try:
                import psycopg
                try:
                    from psycopg_pool import ConnectionPool
                    try:
                        self._pool = ConnectionPool(
                            conninfo=dsn,
                            min_size=1,
                            max_size=int(self.config.connection_params.get('pool_size', 5)),
                        )
                        self._driver = 'psycopg_pool'
                    except Exception as exc:  # noqa: BLE001 - fallback to single connection
                        logger.debug(
                            "psycopg_pool init failed; falling back to single connection (error_type={})",
                            type(exc).__name__,
                        )
                        self._pool = None
                except ImportError:
                    self._pool = None
                if self._pool is None:
                    # Fallback to single psycopg connection
                    self._conn = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: psycopg.connect(dsn),
                    )
                    self._driver = 'psycopg'
            except ImportError:
                # Final fallback: psycopg2 single connection
                import psycopg2
                self._conn = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: psycopg2.connect(dsn),
                )
                self._driver = 'psycopg2'
            except Exception as exc:  # noqa: BLE001 - fallback to psycopg2 on psycopg failure
                logger.debug(
                    "psycopg connect failed; falling back to psycopg2 (error_type={})",
                    type(exc).__name__,
                )
                import psycopg2
                self._conn = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: psycopg2.connect(dsn),
                )
                self._driver = 'psycopg2'

            await self._register_vector_support()

            # Ensure pgvector extension
            await self._exec("CREATE EXTENSION IF NOT EXISTS vector")
            self._initialized = True
            logger.info("PGVector adapter initialized")
        except Exception:  # noqa: BLE001 - initialization should not raise
            logger.error("Failed to initialize PGVector adapter")
            self._conn = None
            self._pool = None
            self._initialized = False

    def _build_dsn(self, params: dict[str, Any]) -> str:
        # Support both DSN and discrete params
        if params.get('dsn'):
            return str(params['dsn'])
        host = params.get('host', 'localhost')
        port = params.get('port', 5432)
        db = params.get('database', 'postgres')
        user = params.get('user', 'postgres')
        password = params.get('password', '')
        sslmode = params.get('sslmode', 'prefer')
        return f"host={host} port={port} dbname={db} user={user} password={password} sslmode={sslmode}"

    def _sanitize_collection(self, name: str) -> str:
        # Allow only alphanum and underscores; replace others with underscore
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", name)
        return f"vs_{safe}"

    def _borrow_conn(self):
        if self._pool is not None:
            return self._pool.connection()
        if self._conn is not None:
            class _Ctx:
                def __init__(self, conn): self.conn = conn
                def __enter__(self): return self.conn
                def __exit__(self, exc_type, exc, tb): return False
            return _Ctx(self._conn)
        raise PGVectorNotInitializedError()

    async def _register_vector_support(self) -> None:
        """Register pgvector adapters with psycopg when available."""
        if self._vector_cls is not None:
            return
        if _register_pgvector is None or _PgVector is None:
            return
        loop = asyncio.get_event_loop()
        try:
            if self._pool is not None:
                await loop.run_in_executor(None, _register_pgvector, self._pool)
            elif self._conn is not None:
                await loop.run_in_executor(None, _register_pgvector, self._conn)
            else:
                return
            self._vector_cls = _PgVector
            logger.debug("Registered pgvector type with psycopg")
        except Exception:  # noqa: BLE001 - registration best-effort
            logger.debug("pgvector registration failed")
            self._vector_cls = None

    def _serialize_vector(self, vector: list[float]) -> str:
        """Serialize a python list into a pgvector literal."""
        vector_obj: Any = vector
        if self._vector_cls is not None and isinstance(vector_obj, self._vector_cls):
            vector = list(cast(Any, vector_obj))
        if not isinstance(vector, (list, tuple)):
            raise InvalidQueryVectorError()
        parts = []
        for val in vector:
            try:
                parts.append(format(float(val), ".15g"))
            except (TypeError, ValueError) as exc:
                raise InvalidQueryVectorValueError() from exc
        return "[" + ",".join(parts) + "]"

    async def _exec(self, sql: str, params: Optional[tuple] = None) -> None:
        def _run(pool, single, ef):
            ctx = pool if pool is not None else single
            with ctx as conn:
                cur = conn.cursor()
                try:
                    try:
                        cur.execute(f"SET hnsw.ef_search = {int(ef)}")
                    except Exception:  # noqa: BLE001 - best-effort session tuning
                        logger.debug("pgvector._exec: SET hnsw.ef_search failed")
                    cur.execute(sql, params or ())
                    conn.commit()
                except Exception:  # noqa: BLE001 - re-raise after rollback
                    try:
                        conn.rollback()
                    except Exception:  # noqa: BLE001 - rollback best-effort
                        logger.debug("pgvector._exec: rollback failed")
                    raise
                finally:
                    try:
                        cur.close()
                    except Exception:  # noqa: BLE001 - cursor close best-effort
                        logger.debug("pgvector._exec: cursor close failed")
        await asyncio.get_event_loop().run_in_executor(
            None,
            _run,
            self._borrow_conn(),
            None if self._pool else self._borrow_conn(),
            self._ef_search,
        )

    async def _query(self, sql: str, params: Optional[tuple] = None) -> list[tuple]:
        def _run(pool, single, ef):
            ctx = pool if pool is not None else single
            with ctx as conn:
                cur = conn.cursor()
                try:
                    try:
                        cur.execute(f"SET hnsw.ef_search = {int(ef)}")
                    except Exception:  # noqa: BLE001 - best-effort session tuning
                        logger.debug("pgvector._query: SET hnsw.ef_search failed")
                    cur.execute(sql, params or ())
                    rows = cur.fetchall()
                except Exception:
                    try:
                        conn.rollback()
                    except Exception:  # noqa: BLE001 - rollback best-effort
                        logger.debug("pgvector._query: rollback failed")
                    raise
                else:
                    return rows
                finally:
                    try:
                        cur.close()
                    except Exception:  # noqa: BLE001 - cursor close best-effort
                        logger.debug("pgvector._query: cursor close failed")
        return await asyncio.get_event_loop().run_in_executor(
            None,
            _run,
            self._borrow_conn(),
            None if self._pool else self._borrow_conn(),
            self._ef_search,
        )

    async def create_collection(self, collection_name: str, metadata: Optional[dict[str, Any]] = None) -> None:
        tbl = self._sanitize_collection(collection_name)
        dim = int(self.config.embedding_dim)
        metric = self.config.distance_metric or 'cosine'
        # Use ivfflat/hnsw only if configured by DBA; here we keep a basic table
        sql = (
            f"CREATE TABLE IF NOT EXISTS {tbl} ("
            "id TEXT PRIMARY KEY, "
            "content TEXT, "
            "metadata JSONB, "
            f"embedding vector({dim})"
            ")"
        )
        await self._exec(sql)
        # Attempt HNSW index first (pgvector >= 0.7); fallback to IVFFLAT on failure
        ops = 'vector_cosine_ops' if metric == 'cosine' else ('vector_l2_ops' if metric == 'euclidean' else 'vector_ip_ops')
        try:
            await self._exec(
                f"CREATE INDEX IF NOT EXISTS {tbl}_embedding_hnsw ON {tbl} USING hnsw (embedding {ops}) WITH (m=16, ef_construction=200)"
            )
        except Exception:  # noqa: BLE001 - index creation best-effort
            try:
                await self._exec(
                    f"CREATE INDEX IF NOT EXISTS {tbl}_embedding_ivf ON {tbl} USING ivfflat (embedding {ops})"
                )
            except Exception:  # noqa: BLE001 - index creation best-effort
                # If both fail, continue without an ANN index (still usable for brute-force)
                logger.debug("pgvector index creation failed; continuing without ANN index")
        # Analyze to help planner (best-effort)
        try:
            await self._exec(f"ANALYZE {tbl}")
        except Exception:  # noqa: BLE001 - analyze best-effort
            logger.debug("pgvector analyze after collection creation failed")

    async def delete_collection(self, collection_name: str) -> None:
        tbl = self._sanitize_collection(collection_name)
        await self._exec(f"DROP TABLE IF EXISTS {tbl}")

    async def list_collections(self) -> list[str]:
        sql = "SELECT tablename FROM pg_tables WHERE tablename LIKE %s"
        rows = await self._query(sql, ('vs_%',))
        collections = []
        for (name,) in rows:
            if isinstance(name, str) and name.startswith("vs_"):
                collections.append(name[3:])
            else:
                collections.append(str(name))
        return collections

    async def upsert_vectors(
        self,
        collection_name: str,
        ids: list[str],
        vectors: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]]
    ) -> None:
        self._validate_vectors(vectors)
        tbl = self._sanitize_collection(collection_name)
        values = list(zip(ids, documents, metadatas, vectors))
        # Use simple upsert
        def _batch(pool, single, ef):
            ctx = pool if pool is not None else single
            with ctx as conn:
                cur = conn.cursor()
                try:
                    try:
                        cur.execute(f"SET hnsw.ef_search = {int(ef)}")
                    except Exception:  # noqa: BLE001 - best-effort session tuning
                        logger.debug("pgvector.upsert: SET hnsw.ef_search failed")
                    args = [(_id, doc, JsonDumper.dumps(meta), vec) for _id, doc, meta, vec in values]
                    cur.executemany(
                        f"INSERT INTO {tbl}(id, content, metadata, embedding) VALUES (%s, %s, %s, %s) "  # nosec B608
                        f"ON CONFLICT (id) DO UPDATE SET content=EXCLUDED.content, metadata=EXCLUDED.metadata, embedding=EXCLUDED.embedding",
                        args,
                    )
                    conn.commit()
                finally:
                    try:
                        cur.close()
                    except Exception:  # noqa: BLE001 - cursor close best-effort
                        logger.debug("pgvector.upsert: cursor close failed")
        # Observe rows + latency
        with self._H_UPSERT_LAT.labels(collection=tbl).time():
            await asyncio.get_event_loop().run_in_executor(None, _batch, self._borrow_conn(), None if self._pool else self._borrow_conn(), self._ef_search)
        try:
            self._C_ROWS_UPSERTED.labels(collection=tbl).inc(len(values))
        except Exception:  # noqa: BLE001 - metrics best-effort
            logger.debug("pgvector.upsert: metrics increment failed")

    async def delete_vectors(self, collection_name: str, ids: list[str]) -> None:
        tbl = self._sanitize_collection(collection_name)
        def _batch(pool, single, ef):
            ctx = pool if pool is not None else single
            with ctx as conn:
                cur = conn.cursor()
                try:
                    cur.executemany(f"DELETE FROM {tbl} WHERE id=%s", [(i,) for i in ids])  # nosec B608
                    conn.commit()
                    rc = getattr(cur, 'rowcount', 0)
                    return int(rc) if rc is not None else 0
                finally:
                    try:
                        cur.close()
                    except Exception:  # noqa: BLE001 - cursor close best-effort
                        logger.debug("pgvector.delete_vectors: cursor close failed")
        with self._H_DELETE_LAT.labels(collection=tbl).time():
            rc = await asyncio.get_event_loop().run_in_executor(None, _batch, self._borrow_conn(), None if self._pool else self._borrow_conn(), self._ef_search)
        try:
            self._C_ROWS_DELETED.labels(collection=tbl).inc(int(rc))
        except Exception:  # noqa: BLE001 - metrics best-effort
            logger.debug("pgvector.delete_vectors: metrics increment failed")

    async def delete_by_filter(self, collection_name: str, filter: dict[str, Any]) -> int:
        """Delete rows matching a JSONB metadata filter; returns affected row count."""
        tbl = self._sanitize_collection(collection_name)
        if filter and isinstance(filter, dict) and len(filter) > 0:
            where_sql, params = self._build_where_from_filter(filter)
        else:
            # No-op when filter is empty
            return 0
        def _run(pool, single, ef):
            ctx = pool if pool is not None else single
            with ctx as conn:
                cur = conn.cursor()
                try:
                    try:
                        cur.execute(f"SET hnsw.ef_search = {int(ef)}")
                    except Exception:  # noqa: BLE001 - best-effort session tuning
                        logger.debug("pgvector.delete_by_filter: SET hnsw.ef_search failed")
                    cur.execute(f"DELETE FROM {tbl}{where_sql}", tuple(params))  # nosec B608
                    rc = getattr(cur, 'rowcount', 0)
                    conn.commit()
                    return int(rc) if rc is not None else 0
                finally:
                    try:
                        cur.close()
                    except Exception:  # noqa: BLE001 - cursor close best-effort
                        logger.debug("pgvector.delete_by_filter: cursor close failed")
        with self._H_DELETE_LAT.labels(collection=tbl).time():
            rc = await asyncio.get_event_loop().run_in_executor(None, _run, self._borrow_conn(), None if self._pool else self._borrow_conn(), self._ef_search)
        try:
            self._C_ROWS_DELETED.labels(collection=tbl).inc(int(rc or 0))
        except Exception:  # noqa: BLE001 - metrics best-effort
            logger.debug("pgvector.delete_by_filter: metrics increment failed")
        try:
            return int(rc)
        except (TypeError, ValueError):
            return 0

    # Adapter-specific helper: list vectors with pagination
    def _build_where_from_filter(self, filt: dict[str, Any]) -> tuple[str, list[Any]]:
        # Prefer JSON containment for simple equality maps (no operators, no nested dict/list values)
        if (
            isinstance(filt, dict)
            and len(filt) > 0
            and all(not isinstance(v, (dict, list, tuple)) for v in filt.values())
            and all(not str(k).startswith('$') for k in filt)
        ):
            import json as _json
            return ' WHERE metadata @> %s', [_json.dumps(filt)]

        # Fallback: build explicit predicates (supports $and/$or and operators)
        def handle_node(node) -> tuple[list[str], list[Any]]:
            if not isinstance(node, dict):
                return [], []
            local_clauses: list[str] = []
            local_params: list[Any] = []
            for k, v in node.items():
                if k == '$and' and isinstance(v, list):
                    sub_parts = [handle_node(x) for x in v]
                    sub_sql = [f"({ ' AND '.join(p[0]) })" for p in sub_parts if p[0]]
                    sub_params_and: list[Any] = []
                    for p in sub_parts:
                        sub_params_and.extend(p[1])
                    if sub_sql:
                        local_clauses.append(' AND '.join(sub_sql))
                        local_params.extend(sub_params_and)
                elif k == '$or' and isinstance(v, list):
                    sub_parts = [handle_node(x) for x in v]
                    sub_sql = [f"({ ' AND '.join(p[0]) })" for p in sub_parts if p[0]]
                    sub_params_or: list[Any] = []
                    for p in sub_parts:
                        sub_params_or.extend(p[1])
                    if sub_sql:
                        local_clauses.append(' OR '.join(sub_sql))
                        local_params.extend(sub_params_or)
                else:
                    field = str(k)
                    if isinstance(v, dict):
                        for op, val in v.items():
                            if op in ('$eq', 'eq'):
                                local_clauses.append("(metadata->>%s) = %s")
                                local_params.extend([field, str(val)])
                            elif op in ('$neq', 'neq'):
                                local_clauses.append("(metadata->>%s) <> %s")
                                local_params.extend([field, str(val)])
                            elif op in ('$in', 'in') and isinstance(val, (list, tuple)) and val:
                                # Use ANY(array) to safely parametrize lists
                                local_clauses.append("(metadata->>%s) = ANY(%s)")
                                local_params.extend([field, list(map(str, val))])
                            elif op in ('$gt', '$gte', '$lt', '$lte'):
                                cmp = {'$gt': '>', '$gte': '>=', '$lt': '<', '$lte': '<='}[op]
                                local_clauses.append(f"(metadata->>%s)::numeric {cmp} %s")
                                local_params.extend([field, float(val)])
                            else:
                                local_clauses.append("(metadata->>%s) = %s")
                                local_params.extend([field, str(val)])
                    else:
                        local_clauses.append("(metadata->>%s) = %s")
                        local_params.extend([field, str(v)])
            return local_clauses, local_params

        clauses, params = handle_node(filt)
        if clauses:
            return ' WHERE ' + ' AND '.join(clauses), params
        return '', []

    async def list_vectors_paginated(self, collection_name: str, limit: int, offset: int, filter: Optional[dict[str, Any]] = None, order_by: Optional[str] = None, order_dir: str = 'asc') -> dict[str, Any]:
        tbl = self._sanitize_collection(collection_name)
        if filter and isinstance(filter, dict) and len(filter) > 0:
            where_sql, params = self._build_where_from_filter(filter)
        else:
            where_sql, params = '', []
        ob = 'id'
        if order_by and isinstance(order_by, str):
            if order_by.startswith('metadata.'):
                key = order_by.split('.', 1)[1]
                ob = f"(metadata->>'{key}')"
            elif order_by == 'id':
                ob = 'id'
        odir = 'ASC' if str(order_dir).lower() == 'asc' else 'DESC'
        rows = await self._query(
            f"SELECT id, content, metadata FROM {tbl}{where_sql} ORDER BY {ob} {odir} LIMIT %s OFFSET %s",  # nosec B608
            tuple(params + [int(limit), int(offset)]),
        )
        items = []
        for rid, content, metadata in rows:
            items.append({
                'id': str(rid),
                'content': content or '',
                'metadata': metadata if isinstance(metadata, dict) else {},
            })
        if where_sql:
            cnt_rows = await self._query(f"SELECT COUNT(*) FROM {tbl}{where_sql}", tuple(params))  # nosec B608
        else:
            cnt_rows = await self._query(f"SELECT COUNT(*) FROM {tbl}")  # nosec B608
        total = int(cnt_rows[0][0]) if cnt_rows else 0
        return {'items': items, 'total': total}

    # Adapter-specific helper: list vectors including embeddings for duplication
    async def list_vectors_with_embeddings_paginated(self, collection_name: str, limit: int, offset: int, filter: Optional[dict[str, Any]] = None, order_by: Optional[str] = None, order_dir: str = 'asc') -> dict[str, Any]:
        tbl = self._sanitize_collection(collection_name)
        if filter and isinstance(filter, dict) and len(filter) > 0:
            where_sql, params = self._build_where_from_filter(filter)
        else:
            where_sql, params = '', []
        ob = 'id'
        if order_by and isinstance(order_by, str):
            if order_by.startswith('metadata.'):
                key = order_by.split('.', 1)[1]
                ob = f"(metadata->>'{key}')"
            elif order_by == 'id':
                ob = 'id'
        odir = 'ASC' if str(order_dir).lower() == 'asc' else 'DESC'
        rows = await self._query(
            f"SELECT id, content, metadata, embedding FROM {tbl}{where_sql} ORDER BY {ob} {odir} LIMIT %s OFFSET %s",  # nosec B608
            tuple(params + [int(limit), int(offset)]),
        )
        items = []
        for rid, content, metadata, embedding in rows:
            vec = embedding
            try:
                if hasattr(vec, 'tolist'):
                    vec = vec.tolist()
                elif isinstance(vec, tuple):
                    vec = list(vec)
            except (AttributeError, TypeError):
                pass
            items.append({
                'id': str(rid),
                'content': content or '',
                'metadata': metadata if isinstance(metadata, dict) else {},
                'vector': vec if isinstance(vec, list) else [],
            })
        if where_sql:
            cnt_rows = await self._query(f"SELECT COUNT(*) FROM {tbl}{where_sql}", tuple(params))  # nosec B608
        else:
            cnt_rows = await self._query(f"SELECT COUNT(*) FROM {tbl}")  # nosec B608
        total = int(cnt_rows[0][0]) if cnt_rows else 0
        return {'items': items, 'total': total}

    def set_ef_search(self, value: int) -> int:
        with contextlib.suppress(TypeError, ValueError):
            self._ef_search = max(1, int(value))
        return self._ef_search

    async def rebuild_index(
        self,
        collection_name: str,
        index_type: str = 'hnsw',
        metric: Optional[str] = None,
        m: int = 16,
        ef_construction: int = 200,
        lists: int = 100
    ) -> dict[str, Any]:
        """Drop existing ANN index on embedding and create the specified one.

        index_type: 'hnsw' | 'ivfflat' | 'drop'
        metric: 'cosine' | 'euclidean' | 'ip' (defaults to adapter metric)
        """
        tbl = self._sanitize_collection(collection_name)
        # Drop existing embedding indexes
        rows = await self._query(
            "SELECT indexname FROM pg_indexes WHERE tablename = %s",
            (tbl,),
        )
        for (name,) in rows:
            try:
                # Fetch index definition to verify it's on embedding
                defrows = await self._query("SELECT indexdef FROM pg_indexes WHERE indexname = %s", (name,))
                if defrows and 'embedding' in (defrows[0][0] or '').lower():
                    await self._exec(f"DROP INDEX IF EXISTS \"{name}\"")
            except Exception:  # noqa: BLE001 - index drop best-effort
                # Continue dropping best-effort
                logger.debug("pgvector index drop failed during optimize")

        if index_type.lower() == 'drop':
            try:
                await self._exec(f"ANALYZE {tbl}")
            except Exception:  # noqa: BLE001 - analyze best-effort
                logger.debug("pgvector analyze after index drop failed")
            return await self.get_index_info(collection_name)

        op_metric = (metric or self.config.distance_metric or 'cosine').lower()
        ops = 'vector_cosine_ops' if op_metric == 'cosine' else ('vector_l2_ops' if op_metric in ('euclidean','l2') else 'vector_ip_ops')
        if index_type.lower() == 'hnsw':
            await self._exec(
                f"CREATE INDEX IF NOT EXISTS {tbl}_embedding_hnsw ON {tbl} USING hnsw (embedding {ops}) WITH (m={int(m)}, ef_construction={int(ef_construction)})"
            )
        elif index_type.lower() == 'ivfflat':
            await self._exec(
                f"CREATE INDEX IF NOT EXISTS {tbl}_embedding_ivf ON {tbl} USING ivfflat (embedding {ops}) WITH (lists={int(lists)})"
            )
        else:
            raise InvalidIndexTypeError()

        try:
            await self._exec(f"ANALYZE {tbl}")
        except Exception:  # noqa: BLE001 - analyze best-effort
            logger.debug("pgvector analyze after index optimize failed")
        return await self.get_index_info(collection_name)

    # Adapter-specific helper: get a single vector by id
    async def get_vector(self, collection_name: str, vector_id: str) -> Optional[dict[str, Any]]:
        tbl = self._sanitize_collection(collection_name)
        rows = await self._query(
            f"SELECT id, content, metadata FROM {tbl} WHERE id=%s",  # nosec B608
            (vector_id,),
        )
        if not rows:
            return None
        rid, content, metadata = rows[0]
        return {
            'id': str(rid),
            'content': content or '',
            'metadata': metadata if isinstance(metadata, dict) else {},
        }

    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        k: int = 10,
        filter: Optional[dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> list[VectorSearchResult]:
        tbl = self._sanitize_collection(collection_name)
        metric = self.config.distance_metric or 'cosine'
        use_native_vector = self._vector_cls is not None
        # Build distance expression
        if metric == 'cosine':
            op = "<=>"
        elif metric == 'euclidean':
            op = "<->"
        else:
            op = "<#>"
        placeholder = "%s" if use_native_vector else "%s::vector"
        dist_expr = f"embedding {op} {placeholder}"
        sql = f"SELECT id, content, metadata, {dist_expr} AS distance FROM {tbl}"  # nosec B608
        # Build WHERE using rich filter support (equality, $and/$or, $in, numeric cmp)
        vector_param: Any
        if use_native_vector:
            vector_cls = self._vector_cls
            vector_obj: Any = query_vector
            if vector_cls is not None:
                vector_param = vector_obj if isinstance(vector_obj, vector_cls) else vector_cls(query_vector)
            else:
                vector_param = self._serialize_vector(query_vector)
        else:
            vector_param = self._serialize_vector(query_vector)
        params: list[Any] = [vector_param]
        if filter and isinstance(filter, dict) and len(filter) > 0:
            where_sql, where_params = self._build_where_from_filter(filter)
            sql += where_sql
            params.extend(where_params)
        sql += " ORDER BY distance ASC LIMIT %s"
        params.append(int(k))
        with self._H_QUERY_LAT.labels(collection=tbl).time():
            rows = await self._query(sql, tuple(params))
        results: list[VectorSearchResult] = []
        for rid, content, metadata, distance in rows:
            # Convert distance to similarity in [0,1] by heuristic
            try:
                sim = 1.0 / (1.0 + float(distance))
            except (TypeError, ValueError):
                sim = 0.0
            results.append(VectorSearchResult(
                id=str(rid),
                content=content or "",
                metadata=metadata if include_metadata and isinstance(metadata, dict) else {},
                score=sim,
                distance=float(distance) if distance is not None else 0.0,
            ))
        return results

    async def multi_search(
        self,
        collection_patterns: list[str],
        query_vector: list[float],
        k: int = 10,
        filter: Optional[dict[str, Any]] = None
    ) -> list[VectorSearchResult]:
        # Fetch matching tables and aggregate results
        all_tables = await self.list_collections()
        results: list[VectorSearchResult] = []
        for pattern in collection_patterns:
            regex = re.compile('^' + pattern.replace('*', '.*') + '$')
            for tbl in all_tables:
                if regex.match(tbl):
                    results.extend(await self.search(tbl, query_vector, k=k, filter=filter))
        # Sort by score desc and trim
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    async def get_collection_stats(self, collection_name: str) -> dict[str, Any]:
        tbl = self._sanitize_collection(collection_name)
        rows = await self._query(f"SELECT COUNT(*) FROM {tbl}")  # nosec B608
        count = int(rows[0][0]) if rows else 0
        return {
            "collection": collection_name,
            "table": tbl,
            "count": count,
            "dim": self.config.embedding_dim,
            "dimension": self.config.embedding_dim,
            "metric": self.config.distance_metric,
        }

    async def optimize_collection(self, collection_name: str) -> None:
        # VACUUM/ANALYZE basic optimization
        tbl = self._sanitize_collection(collection_name)
        try:
            await self._exec(f"ANALYZE {tbl}")
        except Exception:  # noqa: BLE001 - analyze best-effort
            logger.debug("pgvector analyze in optimize_collection failed")

    async def get_index_info(self, collection_name: str) -> dict[str, Any]:
        tbl = self._sanitize_collection(collection_name)
        # Identify index type on embedding column
        try:
            rows = await self._query("SELECT indexdef FROM pg_indexes WHERE tablename = %s", (tbl,))
            idxdef = " ".join([(r[0] or "") for r in rows])
            idx_type = "hnsw" if "using hnsw" in idxdef.lower() else ("ivfflat" if "using ivfflat" in idxdef.lower() else "none")
        except Exception:  # noqa: BLE001 - index lookup best-effort
            idx_type = "unknown"
        stats = await self.get_collection_stats(collection_name)
        return {
            "backend": "pgvector",
            "index_type": idx_type,
            "dimension": stats.get("dimension", self.config.embedding_dim),
            "count": stats.get("count", 0),
            "ops": "vector_%s_ops" % (self.config.distance_metric or 'cosine'),
            "ef_search": self._ef_search,
        }

    async def close(self) -> None:
        # Close pooled connections first (if any), then single connection fallback
        try:
            if self._pool is not None:
                # psycopg_pool.ConnectionPool exposes close(); run in executor to avoid blocking
                await asyncio.get_event_loop().run_in_executor(None, getattr(self._pool, "close", lambda: None))
        except Exception:  # noqa: BLE001 - close best-effort
            logger.debug("pgvector pool close failed")
        finally:
            self._pool = None
        try:
            if self._conn:
                await asyncio.get_event_loop().run_in_executor(None, self._conn.close)
        except Exception:  # noqa: BLE001 - close best-effort
            logger.debug("pgvector connection close failed")
        finally:
            self._conn = None
        await super().close()

    async def health(self) -> dict[str, Any]:
        ok = False
        info: dict[str, Any] = {"driver": self._driver or "unknown"}
        # Include basic pool stats when psycopg_pool is available
        try:
            if self._pool is not None:
                # Guarded getattr to avoid hard dependency on psycopg_pool internals
                info["pool"] = {
                    "min_size": getattr(self._pool, "min_size", None),
                    "max_size": getattr(self._pool, "max_size", None),
                    "num_connections": getattr(self._pool, "num_connections", None),
                    "num_available": getattr(self._pool, "num_available", None),
                }
        except Exception:  # noqa: BLE001 - pool stats best-effort
            logger.debug("pgvector pool stats collection failed")
        try:
            rows = await self._query("SELECT 1", None)
            ok = bool(rows)
        except Exception:  # noqa: BLE001 - retry on any failure
            try:
                await self.initialize()
                rows2 = await self._query("SELECT 1", None)
                ok = bool(rows2)
            except Exception:  # noqa: BLE001 - final fallback
                ok = False
        info["ok"] = bool(ok)
        return info


class JsonDumper:
    @staticmethod
    def dumps(obj: dict[str, Any]) -> str:
        # Avoid importing json at module top as a micro-optimization
        import json as _json
        try:
            return _json.dumps(obj)
        except (TypeError, ValueError):
            return '{}'
