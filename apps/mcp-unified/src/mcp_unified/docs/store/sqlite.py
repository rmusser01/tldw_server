from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from hashlib import sha256
from importlib import resources
import json
from pathlib import Path
import sqlite3
from typing import Any

from loguru import logger

from ..errors import DocsError
from ..models import AccessScope

_STORE_PACKAGE = "mcp_unified.docs.store"
_SCHEMA_RESOURCE = "schema.sql"
_SCOPE_SENTINEL = ""
_FTS_PHRASE_QUOTE = '"'
_COUNT_SQL = {
    "docs_documents": "SELECT COUNT(*) AS count FROM docs_documents",
    "docs_chunks": "SELECT COUNT(*) AS count FROM docs_chunks",
    "docs_collections": "SELECT COUNT(*) AS count FROM docs_collections",
    "docs_keywords": "SELECT COUNT(*) AS count FROM docs_keywords",
}
_SCOPE_TABLE_INFO_SQL = {
    "docs_documents": "PRAGMA table_info(docs_documents)",
    "docs_collections": "PRAGMA table_info(docs_collections)",
    "docs_keywords": "PRAGMA table_info(docs_keywords)",
    "docs_aliases": "PRAGMA table_info(docs_aliases)",
}
_SCOPE_BACKFILL_SQL = {
    "docs_documents": (
        "UPDATE docs_documents SET owner_scope = ? WHERE owner_scope IS NULL",
        "UPDATE docs_documents SET profile_scope = ? WHERE profile_scope IS NULL",
    ),
    "docs_collections": (
        "UPDATE docs_collections SET owner_scope = ? WHERE owner_scope IS NULL",
        "UPDATE docs_collections SET profile_scope = ? WHERE profile_scope IS NULL",
    ),
    "docs_keywords": (
        "UPDATE docs_keywords SET owner_scope = ? WHERE owner_scope IS NULL",
        "UPDATE docs_keywords SET profile_scope = ? WHERE profile_scope IS NULL",
    ),
    "docs_aliases": (
        "UPDATE docs_aliases SET owner_scope = ? WHERE owner_scope IS NULL",
        "UPDATE docs_aliases SET profile_scope = ? WHERE profile_scope IS NULL",
    ),
}


class DocsCatalogStore:
    """SQLite-backed document corpus store for the runtime-neutral MCP docs package."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def migrate(self) -> None:
        schema = resources.files(_STORE_PACKAGE).joinpath(_SCHEMA_RESOURCE).read_text(encoding="utf-8")
        with closing(self.connect()) as conn:
            conn.executescript(schema)
            self._backfill_scope_sentinels(conn)
            self._ensure_documents_scope_uri_unique_index(conn)
            self._ensure_collection_description_column(conn)
            conn.commit()

    def status(self) -> dict[str, Any]:
        with closing(self.connect()) as conn:
            return {
                "schema_version": self._schema_version(conn),
                "fts_available": self._fts_available(conn),
                "counts": {
                    "documents": self._count(conn, "docs_documents"),
                    "chunks": self._count(conn, "docs_chunks"),
                    "collections": self._count(conn, "docs_collections"),
                    "keywords": self._count(conn, "docs_keywords"),
                },
            }

    def upsert_document(
        self,
        *,
        scope: AccessScope,
        title: str,
        document_type: str,
        canonical_uri: str,
        source_path: str | None,
        source_url: str | None,
        text: str,
        sections: Sequence[Mapping[str, Any]],
        chunks: Sequence[Mapping[str, Any]],
        keywords: Iterable[str],
        collection_names: Iterable[str],
        metadata: Mapping[str, Any],
    ) -> int:
        owner_scope, profile_scope = _scope_values(scope)
        metadata_map = dict(metadata or {})
        metadata_json = _json_dump(metadata_map)
        content_hash = sha256(text.encode("utf-8")).hexdigest()
        package_name = _optional_text(metadata_map.get("package_name") or metadata_map.get("package"))
        package_version = _optional_text(metadata_map.get("version") or metadata_map.get("package_version"))

        with closing(self.connect()) as conn:
            with conn:
                cursor = conn.execute(
                    """
                    INSERT INTO docs_documents (
                        owner_scope,
                        profile_scope,
                        title,
                        document_type,
                        canonical_uri,
                        source_path,
                        source_url,
                        content_hash,
                        text,
                        metadata_json,
                        package_name,
                        package_version
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(owner_scope, profile_scope, canonical_uri)
                    DO UPDATE SET title = excluded.title,
                        document_type = excluded.document_type,
                        source_path = excluded.source_path,
                        source_url = excluded.source_url,
                        content_hash = excluded.content_hash,
                        text = excluded.text,
                        metadata_json = excluded.metadata_json,
                        package_name = excluded.package_name,
                        package_version = excluded.package_version,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id
                    """,
                    (
                        owner_scope,
                        profile_scope,
                        title,
                        document_type,
                        canonical_uri,
                        source_path,
                        source_url,
                        content_hash,
                        text,
                        metadata_json,
                        package_name,
                        package_version,
                    ),
                )
                document_id = int(cursor.fetchone()["id"])

                self._replace_document_rows(conn, document_id, title, sections, chunks)
                self._replace_document_keywords(conn, owner_scope, profile_scope, document_id, keywords)
                self._replace_collection_memberships(
                    conn,
                    owner_scope,
                    profile_scope,
                    document_id,
                    collection_names,
                )
                return document_id

    def search_chunks(
        self,
        scope: AccessScope,
        query: str,
        limit: int,
        offset: int = 0,
        filters: Any = None,
        snippet_length: int = 300,
    ) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []
        match_query = _fts_match_query(query)
        if not match_query:
            return []

        owner_scope, profile_scope = _scope_values(scope)
        bounded_limit = _bounded_non_negative(limit, default=10)
        bounded_offset = _bounded_non_negative(offset, default=0)
        snippet_chars = min(_bounded_positive(snippet_length, default=300), 10_000)
        where_sql, params = self._search_filter_sql(filters, owner_scope, profile_scope)
        sql = "\n".join(
            (
                """
            SELECT
                d.id AS document_id,
                c.id AS chunk_id,
                d.title AS title,
                d.document_type AS document_type,
                d.canonical_uri AS uri,
                d.source_path AS source_path,
                d.source_url AS source_url,
                c.citation AS citation,
                c.text AS chunk_text,
                bm25(docs_chunks_fts) AS score,
                d.metadata_json AS metadata_json
            FROM docs_chunks_fts
            JOIN docs_chunks c ON c.id = docs_chunks_fts.chunk_id
            JOIN docs_documents d ON d.id = c.document_id
            WHERE docs_chunks_fts MATCH ?
              AND d.owner_scope = ?
              AND d.profile_scope = ?
                """,
                where_sql,
                """
            ORDER BY score ASC, c.id ASC
            LIMIT ? OFFSET ?
                """,
            )
        )
        with closing(self.connect()) as conn:
            rows = conn.execute(
                sql,
                (match_query, owner_scope, profile_scope, *params, bounded_limit, bounded_offset),
            ).fetchall()
        return [
            {
                "document_id": int(row["document_id"]),
                "chunk_id": int(row["chunk_id"]),
                "title": row["title"],
                "document_type": row["document_type"],
                "uri": row["uri"],
                "canonical_uri": row["uri"],
                "source_path": row["source_path"],
                "source_url": row["source_url"],
                "citation": row["citation"],
                "snippet": _snippet(row["chunk_text"], snippet_chars),
                "text": row["chunk_text"],
                "score": float(row["score"]),
                "metadata": _json_load(row["metadata_json"]),
            }
            for row in rows
        ]

    def count_search_chunks(self, scope: AccessScope, query: str, filters: Any = None) -> int:
        query = query.strip()
        if not query:
            return 0
        match_query = _fts_match_query(query)
        if not match_query:
            return 0

        owner_scope, profile_scope = _scope_values(scope)
        where_sql, params = self._search_filter_sql(filters, owner_scope, profile_scope)
        sql = "\n".join(
            (
                """
            SELECT COUNT(*) AS count
            FROM docs_chunks_fts
            JOIN docs_chunks c ON c.id = docs_chunks_fts.chunk_id
            JOIN docs_documents d ON d.id = c.document_id
            WHERE docs_chunks_fts MATCH ?
              AND d.owner_scope = ?
              AND d.profile_scope = ?
                """,
                where_sql,
            )
        )
        with closing(self.connect()) as conn:
            row = conn.execute(sql, (match_query, owner_scope, profile_scope, *params)).fetchone()
        return int(row["count"]) if row is not None else 0

    def get_document(self, scope: AccessScope, target: int | str, mode: str = "snippet") -> dict[str, Any]:
        document_id = target if isinstance(target, int) else self._resolve_document_id(scope, target)
        if document_id is None:
            raise DocsError(
                code="document_not_found",
                message="Document not found",
                details={"target": str(target)},
            )

        owner_scope, profile_scope = _scope_values(scope)
        with closing(self.connect()) as conn:
            row = conn.execute(
                """
                SELECT *
                FROM docs_documents
                WHERE id = ? AND owner_scope = ? AND profile_scope = ?
                """,
                (document_id, owner_scope, profile_scope),
            ).fetchone()
            if row is None:
                raise DocsError(
                    code="document_not_found",
                    message="Document not found",
                    details={"target": str(target)},
                )
            document = _document_from_row(row)
            if mode == "full":
                document["text"] = row["text"]
            if mode in {"section", "full"}:
                document["sections"] = self._sections_for_document(conn, int(row["id"]))
            if mode in {"chunk", "chunk_with_neighbors", "full"}:
                document["chunks"] = self._chunks_for_document(conn, int(row["id"]))
            if mode == "snippet":
                chunk = conn.execute(
                    """
                    SELECT text
                    FROM docs_chunks
                    WHERE document_id = ?
                    ORDER BY ordinal ASC, id ASC
                    LIMIT 1
                    """,
                    (document_id,),
                ).fetchone()
                document["snippet"] = chunk["text"] if chunk is not None else ""
            return document

    def list_documents(self, scope: AccessScope, limit: int, offset: int = 0) -> list[dict[str, Any]]:
        owner_scope, profile_scope = _scope_values(scope)
        with closing(self.connect()) as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM docs_documents
                WHERE owner_scope = ? AND profile_scope = ?
                ORDER BY title COLLATE NOCASE ASC, id ASC
                LIMIT ? OFFSET ?
                """,
                (
                    owner_scope,
                    profile_scope,
                    _bounded_non_negative(limit, default=50),
                    _bounded_non_negative(offset, default=0),
                ),
            ).fetchall()
        return [_document_from_row(row) for row in rows]

    def list_collections(self, scope: AccessScope) -> list[dict[str, Any]]:
        owner_scope, profile_scope = _scope_values(scope)
        with closing(self.connect()) as conn:
            rows = conn.execute(
                """
                SELECT col.id, col.name, col.description, COUNT(cm.document_id) AS document_count
                FROM docs_collections col
                LEFT JOIN docs_collection_members cm ON cm.collection_id = col.id
                WHERE col.owner_scope = ? AND col.profile_scope = ?
                GROUP BY col.id, col.name, col.description
                ORDER BY col.name COLLATE NOCASE ASC
                """,
                (owner_scope, profile_scope),
            ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "name": row["name"],
                "description": row["description"],
                "document_count": int(row["document_count"]),
            }
            for row in rows
        ]

    def list_keywords(self, scope: AccessScope) -> list[dict[str, Any]]:
        owner_scope, profile_scope = _scope_values(scope)
        with closing(self.connect()) as conn:
            rows = conn.execute(
                """
                SELECT kw.id, kw.name, COUNT(dk.document_id) AS document_count
                FROM docs_keywords kw
                LEFT JOIN docs_document_keywords dk ON dk.keyword_id = kw.id
                WHERE kw.owner_scope = ? AND kw.profile_scope = ?
                GROUP BY kw.id, kw.name
                ORDER BY kw.name COLLATE NOCASE ASC
                """,
                (owner_scope, profile_scope),
            ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "name": row["name"],
                "keyword": row["name"],
                "document_count": int(row["document_count"]),
            }
            for row in rows
        ]

    def resolve_name(self, scope: AccessScope, name: str) -> list[dict[str, Any]]:
        owner_scope, profile_scope = _scope_values(scope)
        normalized = name.strip()
        if not normalized:
            return []
        like_name = f"%{_escape_like(normalized)}%"

        with closing(self.connect()) as conn:
            document_rows = conn.execute(
                """
                SELECT id, title, canonical_uri
                FROM docs_documents
                WHERE owner_scope = ?
                  AND profile_scope = ?
                  AND (title = ? OR title LIKE ? ESCAPE '\\')
                ORDER BY
                    CASE WHEN title = ? THEN 0 ELSE 1 END,
                    title COLLATE NOCASE ASC,
                    id ASC
                """,
                (owner_scope, profile_scope, normalized, like_name, normalized),
            ).fetchall()
            collection_rows = conn.execute(
                """
                SELECT col.name, col.description, COUNT(cm.document_id) AS document_count
                FROM docs_collections col
                LEFT JOIN docs_collection_members cm ON cm.collection_id = col.id
                WHERE col.owner_scope = ?
                  AND col.profile_scope = ?
                  AND (col.name = ? OR col.name LIKE ? ESCAPE '\\')
                GROUP BY col.id, col.name, col.description
                ORDER BY
                    CASE WHEN col.name = ? THEN 0 ELSE 1 END,
                    col.name COLLATE NOCASE ASC
                """,
                (owner_scope, profile_scope, normalized, like_name, normalized),
            ).fetchall()
            keyword_rows = conn.execute(
                """
                SELECT name
                FROM docs_keywords
                WHERE owner_scope = ?
                  AND profile_scope = ?
                  AND (name = ? OR name LIKE ? ESCAPE '\\')
                ORDER BY
                    CASE WHEN name = ? THEN 0 ELSE 1 END,
                    name COLLATE NOCASE ASC
                """,
                (owner_scope, profile_scope, normalized, like_name, normalized),
            ).fetchall()

        matches: list[dict[str, Any]] = [
            {
                "target_type": "document",
                "id": str(row["id"]),
                "title": row["title"],
                "uri": row["canonical_uri"],
            }
            for row in document_rows
        ]
        matches.extend(
            {
                "target_type": "collection",
                "id": row["name"],
                "title": row["name"],
                "metadata": {
                    "document_count": int(row["document_count"]),
                    "description": row["description"],
                },
            }
            for row in collection_rows
        )
        matches.extend({"target_type": "keyword", "id": row["name"], "title": row["name"]} for row in keyword_rows)
        return matches

    def create_collection(self, *, scope: AccessScope, name: str, description: str = "") -> int:
        owner_scope, profile_scope = _scope_values(scope)
        normalized = _required_name(name, "collection")
        with closing(self.connect()) as conn:
            with conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO docs_collections (owner_scope, profile_scope, name, description)
                    VALUES (?, ?, ?, ?)
                    """,
                    (owner_scope, profile_scope, normalized, description),
                )
                row = conn.execute(
                    """
                    SELECT id
                    FROM docs_collections
                    WHERE owner_scope = ? AND profile_scope = ? AND name = ?
                    """,
                    (owner_scope, profile_scope, normalized),
                ).fetchone()
                collection_id = int(row["id"])
                conn.execute(
                    """
                    UPDATE docs_collections
                    SET description = ?
                    WHERE id = ? AND owner_scope = ? AND profile_scope = ?
                    """,
                    (description, collection_id, owner_scope, profile_scope),
                )
                return collection_id

    def update_collection(self, *, scope: AccessScope, name: str, description: str) -> bool:
        owner_scope, profile_scope = _scope_values(scope)
        normalized = _required_name(name, "collection")
        with closing(self.connect()) as conn:
            with conn:
                cursor = conn.execute(
                    """
                    UPDATE docs_collections
                    SET description = ?
                    WHERE owner_scope = ? AND profile_scope = ? AND name = ?
                    """,
                    (description, owner_scope, profile_scope, normalized),
                )
                return cursor.rowcount > 0

    def set_collection_membership(
        self,
        *,
        scope: AccessScope,
        collection: str,
        document_id: int,
        action: str,
    ) -> str:
        owner_scope, profile_scope = _scope_values(scope)
        normalized_collection = _required_name(collection, "collection")
        normalized_action = action.strip().lower()
        if normalized_action not in {"add", "remove"}:
            raise ValueError("action must be add or remove")

        with closing(self.connect()) as conn:
            with conn:
                self._assert_document_in_scope(conn, owner_scope, profile_scope, document_id)
                collection_id = self._ensure_collection(
                    conn,
                    owner_scope,
                    profile_scope,
                    normalized_collection,
                )
                if normalized_action == "add":
                    cursor = conn.execute(
                        """
                        INSERT OR IGNORE INTO docs_collection_members (collection_id, document_id)
                        VALUES (?, ?)
                        """,
                        (collection_id, document_id),
                    )
                    return "added" if cursor.rowcount else "unchanged"

                cursor = conn.execute(
                    """
                    DELETE FROM docs_collection_members
                    WHERE collection_id = ? AND document_id = ?
                    """,
                    (collection_id, document_id),
                )
                return "removed" if cursor.rowcount else "unchanged"

    def apply_keywords(self, *, scope: AccessScope, document_id: int, keywords: Iterable[str]) -> None:
        owner_scope, profile_scope = _scope_values(scope)
        with closing(self.connect()) as conn:
            with conn:
                self._assert_document_in_scope(conn, owner_scope, profile_scope, document_id)
                self._replace_document_keywords(conn, owner_scope, profile_scope, document_id, keywords)

    def _resolve_document_id(self, scope: AccessScope, name: str) -> int | None:
        owner_scope, profile_scope = _scope_values(scope)
        normalized = name.strip()
        if not normalized:
            return None
        with closing(self.connect()) as conn:
            if normalized.isdecimal():
                row = conn.execute(
                    """
                    SELECT id
                    FROM docs_documents
                    WHERE id = ? AND owner_scope = ? AND profile_scope = ?
                    """,
                    (int(normalized), owner_scope, profile_scope),
                ).fetchone()
                if row is not None:
                    return int(row["id"])

            row = conn.execute(
                """
                SELECT document_id
                FROM docs_aliases
                WHERE owner_scope = ? AND profile_scope = ? AND name = ?
                """,
                (owner_scope, profile_scope, normalized),
            ).fetchone()
            if row is not None:
                return int(row["document_id"])

            row = conn.execute(
                """
                SELECT id
                FROM docs_documents
                WHERE owner_scope = ?
                  AND profile_scope = ?
                  AND (
                      canonical_uri = ?
                      OR source_path = ?
                      OR source_url = ?
                      OR title = ?
                  )
                ORDER BY updated_at DESC, id DESC
                LIMIT 1
                """,
                (owner_scope, profile_scope, normalized, normalized, normalized, normalized),
            ).fetchone()
            return int(row["id"]) if row is not None else None

    @staticmethod
    def _ensure_collection(
        conn: sqlite3.Connection,
        owner_scope: str,
        profile_scope: str,
        name: str,
    ) -> int:
        conn.execute(
            """
            INSERT OR IGNORE INTO docs_collections (owner_scope, profile_scope, name)
            VALUES (?, ?, ?)
            """,
            (owner_scope, profile_scope, name),
        )
        row = conn.execute(
            """
            SELECT id
            FROM docs_collections
            WHERE owner_scope = ? AND profile_scope = ? AND name = ?
            """,
            (owner_scope, profile_scope, name),
        ).fetchone()
        return int(row["id"])

    @staticmethod
    def _assert_document_in_scope(
        conn: sqlite3.Connection,
        owner_scope: str,
        profile_scope: str,
        document_id: int,
    ) -> None:
        row = conn.execute(
            """
            SELECT id
            FROM docs_documents
            WHERE owner_scope = ? AND profile_scope = ? AND id = ?
            """,
            (owner_scope, profile_scope, document_id),
        ).fetchone()
        if row is None:
            raise DocsError(
                code="document_not_found",
                message="Document not found in active scope.",
                details={"document_id": document_id},
            )

    @staticmethod
    def _backfill_scope_sentinels(conn: sqlite3.Connection) -> None:
        for table_name, updates in _SCOPE_BACKFILL_SQL.items():
            if not _table_has_scope_columns(conn, table_name):
                continue
            for update_sql in updates:
                conn.execute(update_sql, (_SCOPE_SENTINEL,))

    @staticmethod
    def _ensure_collection_description_column(conn: sqlite3.Connection) -> None:
        rows = conn.execute("PRAGMA table_info(docs_collections)").fetchall()
        if "description" not in {row["name"] for row in rows}:
            conn.execute("ALTER TABLE docs_collections ADD COLUMN description TEXT NOT NULL DEFAULT ''")

    @staticmethod
    def _ensure_documents_scope_uri_unique_index(conn: sqlite3.Connection) -> None:
        columns = ("owner_scope", "profile_scope", "canonical_uri")
        if _table_has_unique_index(conn, "docs_documents", columns):
            return
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS docs_documents_scope_uri_unique_idx
            ON docs_documents (owner_scope, profile_scope, canonical_uri)
            """
        )

    @staticmethod
    def _schema_version(conn: sqlite3.Connection) -> int:
        try:
            row = conn.execute("SELECT MAX(version) AS version FROM docs_schema_migrations").fetchone()
        except sqlite3.Error as exc:
            logger.debug("Could not read docs schema version: {}", exc)
            return 0
        return int(row["version"] or 0) if row is not None else 0

    @staticmethod
    def _fts_available(conn: sqlite3.Connection) -> bool:
        try:
            conn.execute("SELECT rowid FROM docs_chunks_fts LIMIT 0").fetchall()
        except sqlite3.Error as exc:
            logger.debug("Could not verify docs FTS availability: {}", exc)
            return False
        return True

    @staticmethod
    def _count(conn: sqlite3.Connection, table_name: str) -> int:
        try:
            row = conn.execute(_COUNT_SQL[table_name]).fetchone()
        except KeyError:
            logger.debug("Unknown docs count table requested: {}", table_name)
            return 0
        except sqlite3.Error as exc:
            logger.debug("Could not count docs table {}: {}", table_name, exc)
            return 0
        return int(row["count"]) if row is not None else 0

    @staticmethod
    def _replace_document_rows(
        conn: sqlite3.Connection,
        document_id: int,
        title: str,
        sections: Sequence[Mapping[str, Any]],
        chunks: Sequence[Mapping[str, Any]],
    ) -> None:
        conn.execute("DELETE FROM docs_chunks_fts WHERE document_id = ?", (document_id,))
        conn.execute("DELETE FROM docs_sections WHERE document_id = ?", (document_id,))
        conn.execute("DELETE FROM docs_chunks WHERE document_id = ?", (document_id,))

        for ordinal, section in enumerate(sections):
            conn.execute(
                """
                INSERT INTO docs_sections (
                    document_id,
                    ordinal,
                    heading,
                    level,
                    start_char,
                    end_char,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    ordinal,
                    _optional_text(section.get("heading")),
                    _optional_int(section.get("level")),
                    _optional_int(section.get("start_char")),
                    _optional_int(section.get("end_char")),
                    _json_dump(section.get("metadata") or {}),
                ),
            )

        for ordinal, chunk in enumerate(chunks):
            chunk_text = str(chunk.get("text") or "")
            citation = str(chunk.get("citation") or "")
            cursor = conn.execute(
                """
                INSERT INTO docs_chunks (
                    document_id,
                    ordinal,
                    text,
                    citation,
                    start_char,
                    end_char,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    ordinal,
                    chunk_text,
                    citation,
                    _optional_int(chunk.get("start_char")),
                    _optional_int(chunk.get("end_char")),
                    _json_dump(chunk.get("metadata") or {}),
                ),
            )
            chunk_id = int(cursor.lastrowid)
            conn.execute(
                """
                INSERT INTO docs_chunks_fts (rowid, title, body, citation, chunk_id, document_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (chunk_id, title, chunk_text, citation, chunk_id, document_id),
            )

    @staticmethod
    def _replace_document_keywords(
        conn: sqlite3.Connection,
        owner_scope: str,
        profile_scope: str,
        document_id: int,
        keywords: Iterable[str],
    ) -> None:
        conn.execute("DELETE FROM docs_document_keywords WHERE document_id = ?", (document_id,))
        for keyword in _normalized_names(keywords):
            conn.execute(
                """
                INSERT OR IGNORE INTO docs_keywords (owner_scope, profile_scope, name)
                VALUES (?, ?, ?)
                """,
                (owner_scope, profile_scope, keyword),
            )
            row = conn.execute(
                """
                SELECT id
                FROM docs_keywords
                WHERE owner_scope = ? AND profile_scope = ? AND name = ?
                """,
                (owner_scope, profile_scope, keyword),
            ).fetchone()
            conn.execute(
                """
                INSERT OR IGNORE INTO docs_document_keywords (keyword_id, document_id)
                VALUES (?, ?)
                """,
                (int(row["id"]), document_id),
            )
        conn.execute(
            """
            DELETE FROM docs_keywords
            WHERE owner_scope = ?
              AND profile_scope = ?
              AND id NOT IN (SELECT keyword_id FROM docs_document_keywords)
            """,
            (owner_scope, profile_scope),
        )

    @staticmethod
    def _replace_collection_memberships(
        conn: sqlite3.Connection,
        owner_scope: str,
        profile_scope: str,
        document_id: int,
        collection_names: Iterable[str],
    ) -> None:
        conn.execute("DELETE FROM docs_collection_members WHERE document_id = ?", (document_id,))
        for collection in _normalized_names(collection_names):
            conn.execute(
                """
                INSERT OR IGNORE INTO docs_collections (owner_scope, profile_scope, name)
                VALUES (?, ?, ?)
                """,
                (owner_scope, profile_scope, collection),
            )
            row = conn.execute(
                """
                SELECT id
                FROM docs_collections
                WHERE owner_scope = ? AND profile_scope = ? AND name = ?
                """,
                (owner_scope, profile_scope, collection),
            ).fetchone()
            conn.execute(
                """
                INSERT OR IGNORE INTO docs_collection_members (collection_id, document_id)
                VALUES (?, ?)
                """,
                (int(row["id"]), document_id),
            )

    @staticmethod
    def _search_filter_sql(filters: Any, owner_scope: str, profile_scope: str) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []

        collection = _filter_value(filters, "collection")
        if collection:
            collection = _optional_text(collection)
        if collection:
            clauses.append(
                """
                AND EXISTS (
                    SELECT 1
                    FROM docs_collection_members cm
                    JOIN docs_collections col ON col.id = cm.collection_id
                    WHERE cm.document_id = d.id
                      AND col.owner_scope = ?
                      AND col.profile_scope = ?
                      AND col.name = ?
                )
                """
            )
            params.extend((owner_scope, profile_scope, collection))

        keywords = _filter_value(filters, "keywords") or ()
        if isinstance(keywords, str):
            keywords = (keywords,)
        for keyword in _normalized_names(keywords):
            clauses.append(
                """
                AND EXISTS (
                    SELECT 1
                    FROM docs_document_keywords dk
                    JOIN docs_keywords kw ON kw.id = dk.keyword_id
                    WHERE dk.document_id = d.id
                      AND kw.owner_scope = ?
                      AND kw.profile_scope = ?
                      AND kw.name = ?
                )
                """
            )
            params.extend((owner_scope, profile_scope, keyword))

        document_type = _filter_value(filters, "document_type")
        if document_type:
            document_type = _optional_text(document_type)
        if document_type:
            clauses.append("AND d.document_type = ?")
            params.append(document_type)

        uri_prefix = _filter_value(filters, "uri_prefix")
        if uri_prefix:
            uri_prefix = _optional_text(uri_prefix)
        if uri_prefix:
            clauses.append("AND d.canonical_uri LIKE ? ESCAPE '\\'")
            params.append(f"{_escape_like(uri_prefix)}%")

        package_name = _filter_value(filters, "package")
        if package_name:
            package_name = _optional_text(package_name)
        if package_name:
            clauses.append("AND d.package_name = ?")
            params.append(package_name)

        version = _filter_value(filters, "version")
        if version:
            version = _optional_text(version)
        if version:
            clauses.append("AND d.package_version = ?")
            params.append(version)

        return "\n".join(clauses), params

    @staticmethod
    def _sections_for_document(conn: sqlite3.Connection, document_id: int) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT *
            FROM docs_sections
            WHERE document_id = ?
            ORDER BY ordinal ASC, id ASC
            """,
            (document_id,),
        ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "heading": row["heading"],
                "level": row["level"],
                "start_char": row["start_char"],
                "end_char": row["end_char"],
                "metadata": _json_load(row["metadata_json"]),
            }
            for row in rows
        ]

    @staticmethod
    def _chunks_for_document(conn: sqlite3.Connection, document_id: int) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT *
            FROM docs_chunks
            WHERE document_id = ?
            ORDER BY ordinal ASC, id ASC
            """,
            (document_id,),
        ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "text": row["text"],
                "citation": row["citation"],
                "start_char": row["start_char"],
                "end_char": row["end_char"],
                "metadata": _json_load(row["metadata_json"]),
            }
            for row in rows
        ]


def _scope_values(scope: AccessScope) -> tuple[str, str]:
    return (_normalize_scope_value(scope.owner_scope), _normalize_scope_value(scope.profile_scope))


def _normalize_scope_value(value: str | None) -> str:
    return _SCOPE_SENTINEL if value is None else str(value)


def _json_dump(value: Mapping[str, Any] | object) -> str:
    return json.dumps(value or {}, sort_keys=True, separators=(",", ":"), default=str)


def _json_load(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except (TypeError, ValueError) as exc:
        raise DocsError(
            code="invalid_integer",
            message="Expected an integer-compatible value.",
            details={"value": text},
        ) from exc


def _required_name(value: object, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise DocsError(
            code="invalid_name",
            message=f"{field_name} name must not be empty.",
            details={"field": field_name},
        )
    return text


def _normalized_names(names: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for name in names:
        text = str(name).strip()
        if text and text not in seen:
            normalized.append(text)
            seen.add(text)
    return tuple(normalized)


def _filter_value(filters: Any, name: str) -> Any:
    if filters is None:
        return None
    if isinstance(filters, Mapping):
        return filters.get(name)
    return getattr(filters, name, None)


def _table_has_scope_columns(conn: sqlite3.Connection, table_name: str) -> bool:
    try:
        rows = conn.execute(_SCOPE_TABLE_INFO_SQL[table_name]).fetchall()
    except (KeyError, sqlite3.Error):
        return False
    columns = {row["name"] for row in rows}
    return {"owner_scope", "profile_scope"}.issubset(columns)


def _table_has_unique_index(conn: sqlite3.Connection, table_name: str, columns: tuple[str, ...]) -> bool:
    try:
        indexes = conn.execute(f"PRAGMA index_list({table_name})").fetchall()
    except sqlite3.Error:
        return False
    for index in indexes:
        if not bool(index["unique"]):
            continue
        index_name = index["name"]
        try:
            indexed_columns = conn.execute(f"PRAGMA index_info({index_name})").fetchall()
        except sqlite3.Error:
            continue
        if tuple(row["name"] for row in indexed_columns) == columns:
            return True
    return False


def _fts_match_query(query: str) -> str:
    quoted_terms = []
    for term in query.split():
        escaped = term.replace(_FTS_PHRASE_QUOTE, _FTS_PHRASE_QUOTE * 2)
        quoted_terms.append(f"{_FTS_PHRASE_QUOTE}{escaped}{_FTS_PHRASE_QUOTE}")
    return " ".join(quoted_terms)


def _bounded_non_negative(value: int, default: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return max(result, 0)


def _bounded_positive(value: int, default: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return max(result, 1)


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _snippet(text: str, length: int) -> str:
    if len(text) <= length:
        return text
    return f"{text[:length].rstrip()}..."


def _document_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "title": row["title"],
        "document_type": row["document_type"],
        "canonical_uri": row["canonical_uri"],
        "uri": row["canonical_uri"],
        "source_path": row["source_path"],
        "source_url": row["source_url"],
        "content_hash": row["content_hash"],
        "metadata": _json_load(row["metadata_json"]),
        "package": row["package_name"],
        "version": row["package_version"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
