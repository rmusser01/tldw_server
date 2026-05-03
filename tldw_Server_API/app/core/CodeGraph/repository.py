from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.DB_Management.sqlite_policy import configure_sqlite_connection

from .models import IndexedFile, IndexRunSummary


class CodeGraphRepository:
    """SQLite repository for the native CodeGraph index."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
        with self._connect() as conn:
            conn.executescript(schema)
            conn.commit()

    def counts(self) -> dict[str, int]:
        with self._connect() as conn:
            return {
                "files": self._count(conn, "files"),
                "nodes": self._count(conn, "nodes"),
                "edges": self._count(conn, "edges"),
                "unresolved_refs": self._count(conn, "unresolved_refs"),
            }

    def record_index_run_start(self, *, workspace_key: str, mode: str) -> str:
        run_id = f"run_{uuid.uuid4().hex}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO index_runs(run_id, workspace_key, started_at, mode, status, counters, error_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    workspace_key,
                    _utc_now(),
                    mode,
                    "running",
                    "{}",
                    "[]",
                ),
            )
            conn.commit()
        return run_id

    def finish_index_run(
        self,
        run_id: str,
        *,
        status: str,
        counters: dict[str, int],
        error_summary: list[str] | tuple[str, ...],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE index_runs
                SET finished_at = ?, status = ?, counters = ?, error_summary = ?
                WHERE run_id = ?
                """,
                (
                    _utc_now(),
                    status,
                    json.dumps(counters, sort_keys=True),
                    json.dumps(list(error_summary)),
                    run_id,
                ),
            )
            conn.commit()

    def upsert_file(
        self,
        *,
        path: str,
        language: str,
        size: int,
        content_hash: str,
        modified_at: float,
        status: str,
        errors: list[str] | tuple[str, ...],
        node_count: int = 0,
    ) -> None:
        if Path(path).is_absolute():
            raise ValueError("file path must be workspace-relative")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO files(path, language, size, content_hash, modified_at, indexed_at, node_count, status, errors)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    language = excluded.language,
                    size = excluded.size,
                    content_hash = excluded.content_hash,
                    modified_at = excluded.modified_at,
                    indexed_at = excluded.indexed_at,
                    node_count = excluded.node_count,
                    status = excluded.status,
                    errors = excluded.errors
                """,
                (
                    path,
                    language,
                    int(size),
                    content_hash,
                    float(modified_at),
                    _utc_now(),
                    int(node_count),
                    status,
                    json.dumps(list(errors)),
                ),
            )
            conn.commit()

    def list_files(self, *, limit: int = 100, path_prefix: str | None = None) -> list[IndexedFile]:
        sql = """
            SELECT path, language, size, content_hash, modified_at, indexed_at, node_count, status, errors
            FROM files
        """
        params: list[Any] = []
        if path_prefix:
            sql += " WHERE path LIKE ?"
            params.append(f"{path_prefix}%")
        sql += " ORDER BY path LIMIT ?"
        params.append(max(1, int(limit)))

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_indexed_file_from_row(row) for row in rows]

    def last_index_run(self) -> IndexRunSummary | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT run_id, workspace_key, mode, status, counters, error_summary, started_at, finished_at
                FROM index_runs
                ORDER BY started_at DESC
                LIMIT 1
                """
            ).fetchone()
        return _index_run_from_row(row) if row else None

    def delete_missing_files(self, current_paths: set[str]) -> int:
        existing = {item.path for item in self.list_files(limit=1_000_000)}
        removed = sorted(existing - set(current_paths))
        for path in removed:
            self.delete_file(path)
        return len(removed)

    def delete_file(self, path: str) -> None:
        with self._connect() as conn:
            self._prepare_file_replacement(conn, path)
            conn.execute("DELETE FROM files WHERE path = ?", (path,))
            conn.commit()

    def prepare_file_replacement(self, path: str) -> None:
        with self._connect() as conn:
            self._prepare_file_replacement(conn, path)
            conn.commit()

    def seed_graph_rows_for_test(
        self,
        *,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        unresolved_refs: list[dict[str, Any]],
    ) -> None:
        with self._connect() as conn:
            for node in nodes:
                conn.execute(
                    """
                    INSERT INTO nodes(id, identity_key, kind, name, qualified_name, file_path, language)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        node["id"],
                        node["identity_key"],
                        node["kind"],
                        node["name"],
                        node.get("qualified_name", node["name"]),
                        node["file_path"],
                        node.get("language", "python"),
                    ),
                )
            for edge in edges:
                conn.execute(
                    """
                    INSERT INTO edges(id, source, target, kind, file_path, line, column)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        edge["id"],
                        edge["source"],
                        edge.get("target"),
                        edge["kind"],
                        edge["file_path"],
                        edge.get("line"),
                        edge.get("column"),
                    ),
                )
            for ref in unresolved_refs:
                conn.execute(
                    """
                    INSERT INTO unresolved_refs(from_node_id, reference_name, reference_kind, file_path, line, column, candidates, language)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ref["from_node_id"],
                        ref["reference_name"],
                        ref["reference_kind"],
                        ref["file_path"],
                        ref.get("line"),
                        ref.get("column"),
                        json.dumps(ref.get("candidates", [])),
                        ref.get("language", "python"),
                    ),
                )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        configure_sqlite_connection(conn)
        return conn

    @staticmethod
    def _count(conn: sqlite3.Connection, table: str) -> int:
        count_sql = {
            "files": "SELECT COUNT(*) AS count FROM files",
            "nodes": "SELECT COUNT(*) AS count FROM nodes",
            "edges": "SELECT COUNT(*) AS count FROM edges",
            "unresolved_refs": "SELECT COUNT(*) AS count FROM unresolved_refs",
        }
        row = conn.execute(count_sql[table]).fetchone()
        return int(row["count"])

    @staticmethod
    def _prepare_file_replacement(conn: sqlite3.Connection, path: str) -> None:
        conn.execute("DELETE FROM unresolved_refs WHERE file_path = ?", (path,))
        conn.execute("DELETE FROM edges WHERE file_path = ?", (path,))
        conn.execute("DELETE FROM nodes WHERE file_path = ?", (path,))
        conn.execute(
            """
            DELETE FROM edges
            WHERE source NOT IN (SELECT id FROM nodes)
               OR (target IS NOT NULL AND target NOT IN (SELECT id FROM nodes))
            """
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _indexed_file_from_row(row: sqlite3.Row) -> IndexedFile:
    return IndexedFile(
        path=str(row["path"]),
        language=str(row["language"]),
        size=int(row["size"]),
        content_hash=str(row["content_hash"]),
        modified_at=float(row["modified_at"]),
        indexed_at=str(row["indexed_at"]),
        node_count=int(row["node_count"]),
        status=str(row["status"]),
        errors=tuple(json.loads(row["errors"] or "[]")),
    )


def _index_run_from_row(row: sqlite3.Row) -> IndexRunSummary:
    return IndexRunSummary(
        run_id=str(row["run_id"]),
        workspace_key=str(row["workspace_key"]),
        mode=str(row["mode"]),
        status=str(row["status"]),
        counters=dict(json.loads(row["counters"] or "{}")),
        error_summary=tuple(json.loads(row["error_summary"] or "[]")),
        started_at=str(row["started_at"]) if row["started_at"] else None,
        finished_at=str(row["finished_at"]) if row["finished_at"] else None,
    )
