"""SQLite persistence helpers for native CodeGraph indexes."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.CodeGraph.models import (
    CodeGraphEdge,
    CodeGraphNode,
    CodeGraphUnresolvedRef,
    IndexedFile,
    IndexRunSummary,
    StoredCodeGraphReference,
    codegraph_node_to_dict,
)
from tldw_Server_API.app.core.DB_Management.sqlite_policy import configure_sqlite_connection


@dataclass(frozen=True)
class ImpactTraversal:
    """Bounded graph traversal result for CodeGraph impact queries."""

    nodes: tuple[CodeGraphNode, ...]
    relationships: tuple[dict[str, Any], ...]
    truncated: bool


class CodeGraphRepository:
    """SQLite repository for the native CodeGraph index."""

    def __init__(self, db_path: str | Path) -> None:
        """Store the target SQLite database path without creating it."""
        self.db_path = Path(db_path)

    def initialize(self) -> None:
        """Create parent directories, schema tables, and optional FTS structures."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
        with self._connection() as conn:
            conn.executescript(schema)
            _ensure_schema_compat(conn)
            _create_optional_fts(conn)
            conn.commit()

    def counts(self) -> dict[str, int]:
        """Return row counts for graph inventory tables."""
        with self._connection() as conn:
            return {
                "files": self._count(conn, "files"),
                "nodes": self._count(conn, "nodes"),
                "edges": self._count(conn, "edges"),
                "unresolved_refs": self._count(conn, "unresolved_refs"),
            }

    def record_index_run_start(self, *, workspace_key: str, mode: str) -> str:
        """Insert a running index-run row and return its generated run id."""
        run_id = f"run_{uuid.uuid4().hex}"
        with self._connection() as conn:
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
        """Mark an index run complete with counters and error summary."""
        with self._connection() as conn:
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
        """Insert or update a workspace-relative file inventory row."""
        if Path(path).is_absolute():
            raise ValueError("file path must be workspace-relative")
        with self._connection() as conn:
            _upsert_file(
                conn,
                path=path,
                language=language,
                size=size,
                content_hash=content_hash,
                modified_at=modified_at,
                status=status,
                errors=errors,
                node_count=node_count,
            )
            conn.commit()

    def list_files(
        self,
        *,
        limit: int = 100,
        path_prefix: str | None = None,
        path_pattern: str | None = None,
    ) -> list[IndexedFile]:
        """List indexed files with optional path prefix and glob filtering."""
        sql = """
            SELECT path, language, size, content_hash, modified_at, indexed_at, node_count, status, errors
            FROM files
        """
        params: list[Any] = []
        filters: list[str] = []
        if path_prefix:
            filters.append("path LIKE ? ESCAPE '\\'")
            params.append(f"{_escape_like_literal(path_prefix)}%")
        if path_pattern:
            filters.append("path GLOB ?")
            params.append(path_pattern)
        if filters:
            sql += " WHERE " + " AND ".join(filters)
        sql += " ORDER BY path LIMIT ?"
        params.append(max(1, int(limit)))

        with self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_indexed_file_from_row(row) for row in rows]

    def last_index_run(self) -> IndexRunSummary | None:
        """Return the most recent index run if the database has any run history."""
        with self._connection() as conn:
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
        """Remove files and graph rows not present in the current discovery set."""
        existing = {item.path for item in self.list_files(limit=1_000_000)}
        removed = sorted(existing - set(current_paths))
        if not removed:
            return 0
        with self._connection() as conn:
            _delete_file_rows(conn, removed)
            conn.commit()
        return len(removed)

    def delete_file(self, path: str) -> None:
        """Delete one file inventory row and its dependent graph rows."""
        with self._connection() as conn:
            _delete_file_rows(conn, [path])
            conn.commit()

    def prepare_file_replacement(self, path: str) -> None:
        """Clear graph rows for a file before writing a replacement graph."""
        with self._connection() as conn:
            self._prepare_file_replacement(conn, path)
            conn.commit()

    def replace_file_graph(
        self,
        *,
        path: str,
        nodes: list[CodeGraphNode] | tuple[CodeGraphNode, ...],
        edges: list[CodeGraphEdge] | tuple[CodeGraphEdge, ...],
        unresolved_refs: list[CodeGraphUnresolvedRef] | tuple[CodeGraphUnresolvedRef, ...],
    ) -> None:
        """Replace all graph rows for one workspace-relative file atomically."""
        if Path(path).is_absolute():
            raise ValueError("file path must be workspace-relative")
        with self._connection() as conn:
            self._prepare_file_replacement(conn, path)
            _insert_nodes(conn, nodes)
            _insert_edges(conn, edges)
            _insert_unresolved_refs(conn, unresolved_refs)
            _delete_dangling_edges(conn)
            conn.commit()

    def upsert_file_and_replace_graph(
        self,
        *,
        path: str,
        language: str,
        size: int,
        content_hash: str,
        modified_at: float,
        status: str,
        errors: list[str] | tuple[str, ...],
        node_count: int,
        nodes: list[CodeGraphNode] | tuple[CodeGraphNode, ...],
        edges: list[CodeGraphEdge] | tuple[CodeGraphEdge, ...],
        unresolved_refs: list[CodeGraphUnresolvedRef] | tuple[CodeGraphUnresolvedRef, ...],
    ) -> None:
        """Persist file inventory and replacement graph rows in one transaction."""
        if Path(path).is_absolute():
            raise ValueError("file path must be workspace-relative")
        with self._connection() as conn:
            _upsert_file(
                conn,
                path=path,
                language=language,
                size=size,
                content_hash=content_hash,
                modified_at=modified_at,
                status=status,
                errors=errors,
                node_count=node_count,
            )
            self._prepare_file_replacement(conn, path)
            _insert_nodes(conn, nodes)
            _insert_edges(conn, edges)
            _insert_unresolved_refs(conn, unresolved_refs)
            _delete_dangling_edges(conn)
            conn.commit()

    def search_nodes(
        self,
        query: str,
        *,
        limit: int = 10,
        kind: str | None = None,
        language: str | None = None,
    ) -> list[CodeGraphNode]:
        """Search indexed nodes by exact or fuzzy symbol text with optional filters."""
        text = query.strip()
        if not text:
            return []

        like = f"%{_escape_like_literal(text)}%"
        prefix_like = f"{_escape_like_literal(text)}%"
        params: list[Any] = [
            text,
            text,
            like,
            like,
            like,
            like,
            kind,
            kind,
            language,
            language,
            text,
            text,
            prefix_like,
            max(1, int(limit)),
        ]
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM nodes
                WHERE
                    (
                        lower(name) = lower(?)
                        OR lower(qualified_name) = lower(?)
                        OR name LIKE ? ESCAPE '\\'
                        OR qualified_name LIKE ? ESCAPE '\\'
                        OR signature LIKE ? ESCAPE '\\'
                        OR docstring LIKE ? ESCAPE '\\'
                    )
                    AND (? IS NULL OR kind = ?)
                    AND (? IS NULL OR language = ?)
                ORDER BY
                    CASE
                        WHEN lower(qualified_name) = lower(?) THEN 0
                        WHEN lower(name) = lower(?) THEN 1
                        WHEN name LIKE ? ESCAPE '\\' THEN 2
                        ELSE 3
                    END,
                    file_path,
                    COALESCE(start_line, 0),
                    qualified_name
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [_node_from_row(row) for row in rows]

    def get_node(self, node_id: str) -> CodeGraphNode | None:
        """Fetch one graph node by stable node id."""
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return _node_from_row(row) if row else None

    def find_node_by_symbol(self, symbol: str) -> CodeGraphNode | None:
        """Resolve a symbol name or qualified name to the best matching node."""
        text = symbol.strip()
        if not text:
            return None
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM nodes
                WHERE lower(qualified_name) = lower(?) OR lower(name) = lower(?)
                ORDER BY
                    CASE WHEN lower(qualified_name) = lower(?) THEN 0 ELSE 1 END,
                    file_path,
                    COALESCE(start_line, 0)
                LIMIT 1
                """,
                (text, text, text),
            ).fetchone()
        return _node_from_row(row) if row else None

    def find_module_node(self, qualified_name: str) -> CodeGraphNode | None:
        """Resolve an indexed module node by exact qualified name."""
        text = qualified_name.strip()
        if not text:
            return None
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM nodes
                WHERE kind = 'module' AND qualified_name = ?
                ORDER BY file_path, COALESCE(start_line, 0)
                LIMIT 1
                """,
                (text,),
            ).fetchone()
        return _node_from_row(row) if row else None

    def find_nodes_by_file_and_name(
        self,
        *,
        file_path: str,
        name: str,
    ) -> list[CodeGraphNode]:
        """Find nodes in one file by simple or qualified name."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM nodes
                WHERE file_path = ?
                  AND (name = ? OR qualified_name = ?)
                ORDER BY
                    CASE WHEN name = ? THEN 0 ELSE 1 END,
                    CASE WHEN kind = 'module' THEN 1 ELSE 0 END,
                    COALESCE(start_line, 0),
                    qualified_name,
                    id
                """,
                (file_path, name, name, name),
            ).fetchall()
        return [_node_from_row(row) for row in rows]

    def find_module_node_for_file(self, file_path: str) -> CodeGraphNode | None:
        """Return the module node for a workspace-relative file if one exists."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM nodes
                WHERE file_path = ? AND kind = 'module'
                ORDER BY COALESCE(start_line, 0), qualified_name, id
                LIMIT 1
                """,
                (file_path,),
            ).fetchone()
        return _node_from_row(row) if row else None

    def list_import_nodes(
        self,
        *,
        file_paths: set[str] | frozenset[str] | None = None,
        limit: int | None = None,
    ) -> list[CodeGraphNode]:
        """Return import nodes for cross-file reference resolution."""
        file_scope = _file_path_scope_json(file_paths)
        sql = """
                SELECT *
                FROM nodes
                WHERE kind = 'import'
                  AND (? IS NULL OR file_path IN (SELECT value FROM json_each(?)))
                ORDER BY file_path, COALESCE(start_line, 0), COALESCE(start_column, 0), id
                """
        params: list[Any] = [file_scope, file_scope]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(1, int(limit)))
        with self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_node_from_row(row) for row in rows]

    def list_callers(self, node_id: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """List call edges whose target is the requested node."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    e.id AS edge_id,
                    e.kind AS edge_kind,
                    e.file_path AS edge_file_path,
                    e.line AS edge_line,
                    e.column AS edge_column,
                    e.metadata AS edge_metadata,
                    e.provenance AS edge_provenance,
                    source_node.*,
                    target_node.id AS target_id,
                    target_node.identity_key AS target_identity_key,
                    target_node.kind AS target_kind,
                    target_node.name AS target_name,
                    target_node.qualified_name AS target_qualified_name,
                    target_node.file_path AS target_file_path,
                    target_node.language AS target_language,
                    target_node.start_line AS target_start_line,
                    target_node.end_line AS target_end_line,
                    target_node.start_column AS target_start_column,
                    target_node.end_column AS target_end_column,
                    target_node.signature AS target_signature,
                    target_node.docstring AS target_docstring,
                    target_node.visibility AS target_visibility,
                    target_node.flags AS target_flags,
                    target_node.metadata AS target_metadata
                FROM edges e
                JOIN nodes source_node ON source_node.id = e.source
                JOIN nodes target_node ON target_node.id = e.target
                WHERE e.target = ? AND e.kind = 'calls'
                ORDER BY e.file_path, COALESCE(e.line, 0), source_node.qualified_name
                LIMIT ?
                """,
                (node_id, max(1, int(limit))),
            ).fetchall()
        return [_relationship_from_joined_row(row) for row in rows]

    def list_callees(self, node_id: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """List call edges whose source is the requested node."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    e.id AS edge_id,
                    e.kind AS edge_kind,
                    e.file_path AS edge_file_path,
                    e.line AS edge_line,
                    e.column AS edge_column,
                    e.metadata AS edge_metadata,
                    e.provenance AS edge_provenance,
                    source_node.*,
                    target_node.id AS target_id,
                    target_node.identity_key AS target_identity_key,
                    target_node.kind AS target_kind,
                    target_node.name AS target_name,
                    target_node.qualified_name AS target_qualified_name,
                    target_node.file_path AS target_file_path,
                    target_node.language AS target_language,
                    target_node.start_line AS target_start_line,
                    target_node.end_line AS target_end_line,
                    target_node.start_column AS target_start_column,
                    target_node.end_column AS target_end_column,
                    target_node.signature AS target_signature,
                    target_node.docstring AS target_docstring,
                    target_node.visibility AS target_visibility,
                    target_node.flags AS target_flags,
                    target_node.metadata AS target_metadata
                FROM edges e
                JOIN nodes source_node ON source_node.id = e.source
                JOIN nodes target_node ON target_node.id = e.target
                WHERE e.source = ? AND e.kind = 'calls'
                ORDER BY e.file_path, COALESCE(e.line, 0), target_node.qualified_name
                LIMIT ?
                """,
                (node_id, max(1, int(limit))),
            ).fetchall()
        return [_relationship_from_joined_row(row) for row in rows]

    def list_references_for_resolution(
        self,
        *,
        include_resolved: bool = False,
        file_paths: set[str] | frozenset[str] | None = None,
        limit: int | None = None,
    ) -> list[StoredCodeGraphReference]:
        """List unresolved reference rows used by cross-file resolution."""
        file_scope = _file_path_scope_json(file_paths)
        sql = """
                SELECT *
                FROM unresolved_refs
                WHERE from_node_id IN (SELECT id FROM nodes)
                  AND (? OR (
                      resolved_target IS NULL
                      OR resolved_target NOT IN (SELECT id FROM nodes)
                      OR resolved_edge IS NULL
                      OR resolved_edge NOT IN (SELECT id FROM edges)
                  ))
                  AND (? IS NULL OR file_path IN (SELECT value FROM json_each(?)))
                ORDER BY file_path, COALESCE(line, 0), COALESCE(column, 0), id
            """
        params: list[Any] = [int(include_resolved), file_scope, file_scope]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(1, int(limit)))
        with self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_reference_from_row(row) for row in rows]

    def mark_reference_resolved(
        self,
        ref_id: int,
        *,
        edge: CodeGraphEdge,
        resolution_kind: str,
    ) -> None:
        """Persist a resolved edge and mark the source reference as resolved."""
        self.mark_references_resolved(((ref_id, edge, resolution_kind),))

    def mark_references_resolved(
        self,
        resolutions: Sequence[tuple[int, CodeGraphEdge, str]],
    ) -> None:
        """Persist resolved edges and reference markers in one transaction."""
        items = tuple(resolutions)
        if not items:
            return
        for _ref_id, edge, _resolution_kind in items:
            if edge.target is None:
                raise ValueError("resolved reference edge must have a target")
        resolved_at = _utc_now()
        with self._connection() as conn:
            for _ref_id, edge, _resolution_kind in items:
                _upsert_edge(conn, edge)
            conn.executemany(
                """
                UPDATE unresolved_refs
                SET resolved_target = ?, resolved_edge = ?, resolution_kind = ?, resolved_at = ?
                WHERE id = ?
                """,
                [
                    (edge.target, edge.id, resolution_kind, resolved_at, int(ref_id))
                    for ref_id, edge, resolution_kind in items
                ],
            )
            conn.commit()

    def upsert_edge(self, edge: CodeGraphEdge) -> None:
        """Insert or replace one deterministic graph edge."""
        self.upsert_edges((edge,))

    def upsert_edges(self, edges: Sequence[CodeGraphEdge]) -> None:
        """Insert or replace deterministic graph edges in one transaction."""
        items = tuple(edges)
        if not items:
            return
        for edge in items:
            if edge.target is None:
                raise ValueError("persisted edge must have a target")
        with self._connection() as conn:
            for edge in items:
                _upsert_edge(conn, edge)
            conn.commit()

    def clear_stale_reference_resolutions(self, *, file_paths: set[str] | frozenset[str] | None = None) -> int:
        """Clear resolved-reference markers whose target node or edge is gone."""
        with self._connection() as conn:
            cleared = _clear_stale_reference_resolutions(conn, file_paths=file_paths)
            conn.commit()
        return cleared

    def traverse_impact(
        self,
        node_id: str,
        *,
        depth: int = 2,
        direction: str = "both",
        limit: int = 10,
    ) -> ImpactTraversal:
        """Traverse a bounded incoming/outgoing relationship neighborhood."""
        return self.traverse_impact_many((node_id,), depth=depth, direction=direction, limit=limit)

    def traverse_impact_many(
        self,
        node_ids: tuple[str, ...],
        *,
        depth: int = 2,
        direction: str = "both",
        limit: int = 10,
    ) -> ImpactTraversal:
        """Traverse a bounded relationship neighborhood from multiple root nodes."""
        if direction not in {"incoming", "outgoing", "both"}:
            raise ValueError("direction must be incoming, outgoing, or both")

        root_ids = tuple(dict.fromkeys(str(node_id) for node_id in node_ids if str(node_id)))
        if not root_ids:
            return ImpactTraversal(nodes=(), relationships=(), truncated=False)

        effective_depth = max(1, int(depth))
        effective_limit = max(1, int(limit))
        with self._connection() as conn:
            root_rows = _select_nodes_by_ids_without_anchor(conn, set(root_ids))
            if not root_rows:
                return ImpactTraversal(nodes=(), relationships=(), truncated=False)
            roots_by_id = {str(row["id"]): row for row in root_rows}
            first_root = next((roots_by_id[root_id] for root_id in root_ids if root_id in roots_by_id), root_rows[0])
            anchor_file_path = str(first_root["file_path"])

            seen_node_ids = set(roots_by_id)
            frontier = set(roots_by_id)
            relationships: list[dict[str, Any]] = []
            seen_relationship_ids: set[str] = set()
            truncated = False

            for _level in range(effective_depth):
                if not frontier or truncated:
                    break
                remaining_rows = (effective_limit - len(relationships)) + 1
                if remaining_rows <= 0:
                    truncated = True
                    break
                rows = _select_relationships_for_nodes(
                    conn,
                    frontier,
                    direction,
                    anchor_file_path=anchor_file_path,
                    max_rows=remaining_rows,
                )
                next_frontier: set[str] = set()
                for row in rows:
                    edge_id = str(row["edge_id"])
                    if edge_id in seen_relationship_ids:
                        continue
                    if len(relationships) >= effective_limit:
                        truncated = True
                        break
                    relationship = _relationship_from_joined_row(row)
                    relationships.append(relationship)
                    seen_relationship_ids.add(edge_id)
                    for endpoint in (relationship["source"], relationship["target"]):
                        endpoint_id = str(endpoint["id"])
                        if endpoint_id not in seen_node_ids:
                            seen_node_ids.add(endpoint_id)
                            next_frontier.add(endpoint_id)
                frontier = next_frontier

            node_rows = _select_nodes_by_ids(conn, seen_node_ids, anchor_file_path=anchor_file_path)

        return ImpactTraversal(
            nodes=tuple(_node_from_row(row) for row in node_rows),
            relationships=tuple(relationships),
            truncated=truncated,
        )

    def seed_graph_rows_for_test(
        self,
        *,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        unresolved_refs: list[dict[str, Any]],
    ) -> None:
        """Insert raw graph rows for repository regression tests."""
        with self._connection() as conn:
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

    @contextmanager
    def _connection(self):  # type: ignore[no-untyped-def]
        """Yield a managed SQLite connection for one repository operation."""
        with closing(self._connect()) as conn:
            yield conn

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection using the shared project SQLite policy."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        configure_sqlite_connection(conn)
        return conn

    @staticmethod
    def _count(conn: sqlite3.Connection, table: str) -> int:
        """Return the count for a whitelisted repository table."""
        count_sql = {
            "files": "SELECT COUNT(*) AS count FROM files",
            "nodes": "SELECT COUNT(*) AS count FROM nodes",
            "edges": "SELECT COUNT(*) AS count FROM edges",
            "unresolved_refs": """
                SELECT COUNT(*) AS count
                FROM unresolved_refs
                WHERE resolved_target IS NULL
                   OR resolved_target NOT IN (SELECT id FROM nodes)
                   OR resolved_edge IS NULL
                   OR resolved_edge NOT IN (SELECT id FROM edges)
            """,
        }
        row = conn.execute(count_sql[table]).fetchone()
        return int(row["count"])

    @staticmethod
    def _prepare_file_replacement(conn: sqlite3.Connection, path: str) -> None:
        """Delete graph rows for a file and clear edges left without endpoints."""
        conn.execute("DELETE FROM unresolved_refs WHERE file_path = ?", (path,))
        conn.execute("DELETE FROM edges WHERE file_path = ?", (path,))
        conn.execute("DELETE FROM nodes WHERE file_path = ?", (path,))
        _delete_dangling_edges(conn)
        _clear_stale_reference_resolutions(conn)


def _delete_file_rows(conn: sqlite3.Connection, paths: list[str]) -> None:
    """Delete files and all graph rows that belong to the provided paths."""
    params = [(path,) for path in paths]
    conn.executemany("DELETE FROM unresolved_refs WHERE file_path = ?", params)
    conn.executemany("DELETE FROM edges WHERE file_path = ?", params)
    conn.executemany("DELETE FROM nodes WHERE file_path = ?", params)
    conn.executemany("DELETE FROM files WHERE path = ?", params)
    _delete_dangling_edges(conn)
    _clear_stale_reference_resolutions(conn)


def _ensure_schema_compat(conn: sqlite3.Connection) -> None:
    """Apply additive schema compatibility updates for existing CodeGraph DBs."""
    columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(unresolved_refs)").fetchall()}
    additions = {
        "resolved_target": "ALTER TABLE unresolved_refs ADD COLUMN resolved_target TEXT",
        "resolved_edge": "ALTER TABLE unresolved_refs ADD COLUMN resolved_edge TEXT",
        "resolution_kind": "ALTER TABLE unresolved_refs ADD COLUMN resolution_kind TEXT",
        "resolved_at": "ALTER TABLE unresolved_refs ADD COLUMN resolved_at TEXT",
    }
    for column, statement in additions.items():
        if column not in columns:
            conn.execute(statement)


def _clear_stale_reference_resolutions(
    conn: sqlite3.Connection,
    *,
    file_paths: set[str] | frozenset[str] | None = None,
) -> int:
    """Clear resolved-reference markers when their target node or edge disappeared."""
    file_scope = _file_path_scope_json(file_paths)
    cursor = conn.execute(
        """
        UPDATE unresolved_refs
        SET resolved_target = NULL,
            resolved_edge = NULL,
            resolution_kind = NULL,
            resolved_at = NULL
        WHERE resolved_target IS NOT NULL
          AND (
              resolved_target NOT IN (SELECT id FROM nodes)
              OR resolved_edge IS NULL
              OR resolved_edge NOT IN (SELECT id FROM edges)
          )
          AND (? IS NULL OR file_path IN (SELECT value FROM json_each(?)))
        """,
        (file_scope, file_scope),
    )
    return int(cursor.rowcount or 0)


def _file_path_scope_json(file_paths: set[str] | frozenset[str] | None) -> str | None:
    """Serialize an optional file-path scope for constant SQL predicates."""
    if file_paths is None:
        return None
    return json.dumps(sorted(file_paths))


def _select_relationships_for_nodes(
    conn: sqlite3.Connection,
    node_ids: set[str],
    direction: str,
    *,
    anchor_file_path: str,
    max_rows: int,
) -> list[sqlite3.Row]:
    """Select joined relationships touching a set of node ids in deterministic order."""
    ids_json = json.dumps(sorted(node_ids))

    return conn.execute(
        """
        SELECT
            e.id AS edge_id,
            e.kind AS edge_kind,
            e.file_path AS edge_file_path,
            e.line AS edge_line,
            e.column AS edge_column,
            e.metadata AS edge_metadata,
            e.provenance AS edge_provenance,
            source_node.*,
            target_node.id AS target_id,
            target_node.identity_key AS target_identity_key,
            target_node.kind AS target_kind,
            target_node.name AS target_name,
            target_node.qualified_name AS target_qualified_name,
            target_node.file_path AS target_file_path,
            target_node.language AS target_language,
            target_node.start_line AS target_start_line,
            target_node.end_line AS target_end_line,
            target_node.start_column AS target_start_column,
            target_node.end_column AS target_end_column,
            target_node.signature AS target_signature,
            target_node.docstring AS target_docstring,
            target_node.visibility AS target_visibility,
            target_node.flags AS target_flags,
            target_node.metadata AS target_metadata
        FROM edges e
        JOIN nodes source_node ON source_node.id = e.source
        JOIN nodes target_node ON target_node.id = e.target
        WHERE
            (? IN ('incoming', 'both') AND e.target IN (SELECT value FROM json_each(?)))
            OR (? IN ('outgoing', 'both') AND e.source IN (SELECT value FROM json_each(?)))
        ORDER BY
            CASE WHEN e.file_path = ? THEN 0 ELSE 1 END,
            e.file_path,
            COALESCE(e.line, 0),
            source_node.qualified_name,
            target_node.qualified_name,
            e.id
        LIMIT ?
        """,
        (direction, ids_json, direction, ids_json, anchor_file_path, max(1, int(max_rows))),
    ).fetchall()


def _select_nodes_by_ids(
    conn: sqlite3.Connection,
    node_ids: set[str],
    *,
    anchor_file_path: str,
) -> list[sqlite3.Row]:
    """Select nodes by id in stable source-location order."""
    return conn.execute(
        """
        SELECT *
        FROM nodes
        WHERE id IN (SELECT value FROM json_each(?))
        ORDER BY
            CASE WHEN file_path = ? THEN 0 ELSE 1 END,
            file_path,
            COALESCE(start_line, 0),
            qualified_name,
            id
        """,
        (json.dumps(sorted(node_ids)), anchor_file_path),
    ).fetchall()


def _select_nodes_by_ids_without_anchor(
    conn: sqlite3.Connection,
    node_ids: set[str],
) -> list[sqlite3.Row]:
    """Select nodes by id before an anchor file is known."""
    return conn.execute(
        """
        SELECT *
        FROM nodes
        WHERE id IN (SELECT value FROM json_each(?))
        ORDER BY file_path, COALESCE(start_line, 0), qualified_name, id
        """,
        (json.dumps(sorted(node_ids)),),
    ).fetchall()


def _upsert_file(
    conn: sqlite3.Connection,
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
    """Insert or update a file row using an existing transaction."""
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


def _insert_nodes(
    conn: sqlite3.Connection,
    nodes: list[CodeGraphNode] | tuple[CodeGraphNode, ...],
) -> None:
    """Bulk insert extracted graph nodes."""
    conn.executemany(
        """
        INSERT INTO nodes(
            id, identity_key, kind, name, qualified_name, file_path, language,
            start_line, end_line, start_column, end_column, signature, docstring,
            visibility, flags, metadata
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                node.id,
                node.identity_key,
                node.kind,
                node.name,
                node.qualified_name,
                node.file_path,
                node.language,
                node.start_line,
                node.end_line,
                node.start_column,
                node.end_column,
                node.signature,
                node.docstring,
                node.visibility,
                json.dumps(list(node.flags)),
                json.dumps(node.metadata, sort_keys=True),
            )
            for node in nodes
        ],
    )


def _insert_edges(
    conn: sqlite3.Connection,
    edges: list[CodeGraphEdge] | tuple[CodeGraphEdge, ...],
) -> None:
    """Bulk insert extracted graph edges."""
    conn.executemany(
        """
        INSERT INTO edges(id, source, target, kind, file_path, line, column, metadata, provenance)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                edge.id,
                edge.source,
                edge.target,
                edge.kind,
                edge.file_path,
                edge.line,
                edge.column,
                json.dumps(edge.metadata, sort_keys=True),
                edge.provenance,
            )
            for edge in edges
        ],
    )


def _upsert_edge(conn: sqlite3.Connection, edge: CodeGraphEdge) -> None:
    """Insert or replace a deterministic edge row using an existing transaction."""
    conn.execute(
        """
        INSERT OR REPLACE INTO edges(id, source, target, kind, file_path, line, column, metadata, provenance)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            edge.id,
            edge.source,
            edge.target,
            edge.kind,
            edge.file_path,
            edge.line,
            edge.column,
            json.dumps(edge.metadata, sort_keys=True),
            edge.provenance,
        ),
    )


def _insert_unresolved_refs(
    conn: sqlite3.Connection,
    unresolved_refs: list[CodeGraphUnresolvedRef] | tuple[CodeGraphUnresolvedRef, ...],
) -> None:
    """Bulk insert unresolved references captured by extractors."""
    conn.executemany(
        """
        INSERT INTO unresolved_refs(
            from_node_id, reference_name, reference_kind, file_path, line, column, candidates, language
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                ref.from_node_id,
                ref.reference_name,
                ref.reference_kind,
                ref.file_path,
                ref.line,
                ref.column,
                json.dumps(list(ref.candidates)),
                ref.language,
            )
            for ref in unresolved_refs
        ],
    )


def _escape_like_literal(value: str) -> str:
    """Escape user text before using it inside a SQLite LIKE pattern."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _delete_dangling_edges(conn: sqlite3.Connection) -> None:
    """Remove edges whose source or target node no longer exists."""
    # Stage 1 only stores bounded foreground inventories, so this cleanup favors
    # straightforward correctness; if graph volume grows, switch to a join/exists
    # form or staged node-id table to avoid repeated full-node scans.
    conn.execute(
        """
        DELETE FROM edges
        WHERE source NOT IN (SELECT id FROM nodes)
           OR (target IS NOT NULL AND target NOT IN (SELECT id FROM nodes))
        """
    )


def _create_optional_fts(conn: sqlite3.Connection) -> None:
    """Create the optional nodes FTS table when SQLite was built with FTS5."""
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
                name,
                qualified_name,
                signature,
                docstring,
                metadata,
                content='nodes',
                content_rowid='rowid'
            )
            """
        )
    except sqlite3.OperationalError as exc:
        if "fts5" not in str(exc).lower():
            raise
        logger.warning("CodeGraph FTS5 unavailable; continuing without nodes_fts support")


def _utc_now() -> str:
    """Return the current UTC timestamp in the repository's stored format."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _indexed_file_from_row(row: sqlite3.Row) -> IndexedFile:
    """Convert a files table row to an IndexedFile value object."""
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


def _node_from_row(row: sqlite3.Row) -> CodeGraphNode:
    """Convert a nodes table row to a CodeGraphNode value object."""
    return CodeGraphNode(
        id=str(row["id"]),
        identity_key=str(row["identity_key"]),
        kind=str(row["kind"]),
        name=str(row["name"]),
        qualified_name=str(row["qualified_name"] or row["name"]),
        file_path=str(row["file_path"]),
        language=str(row["language"] or ""),
        start_line=int(row["start_line"]) if row["start_line"] is not None else None,
        end_line=int(row["end_line"]) if row["end_line"] is not None else None,
        start_column=int(row["start_column"]) if row["start_column"] is not None else None,
        end_column=int(row["end_column"]) if row["end_column"] is not None else None,
        signature=str(row["signature"]) if row["signature"] is not None else None,
        docstring=str(row["docstring"]) if row["docstring"] is not None else None,
        visibility=str(row["visibility"]) if row["visibility"] is not None else None,
        flags=tuple(json.loads(row["flags"] or "[]")),
        metadata=dict(json.loads(row["metadata"] or "{}")),
    )


def _target_node_from_joined_row(row: sqlite3.Row) -> CodeGraphNode:
    """Convert target-node aliases from a relationship join row."""
    return CodeGraphNode(
        id=str(row["target_id"]),
        identity_key=str(row["target_identity_key"]),
        kind=str(row["target_kind"]),
        name=str(row["target_name"]),
        qualified_name=str(row["target_qualified_name"] or row["target_name"]),
        file_path=str(row["target_file_path"]),
        language=str(row["target_language"] or ""),
        start_line=int(row["target_start_line"]) if row["target_start_line"] is not None else None,
        end_line=int(row["target_end_line"]) if row["target_end_line"] is not None else None,
        start_column=int(row["target_start_column"]) if row["target_start_column"] is not None else None,
        end_column=int(row["target_end_column"]) if row["target_end_column"] is not None else None,
        signature=str(row["target_signature"]) if row["target_signature"] is not None else None,
        docstring=str(row["target_docstring"]) if row["target_docstring"] is not None else None,
        visibility=str(row["target_visibility"]) if row["target_visibility"] is not None else None,
        flags=tuple(json.loads(row["target_flags"] or "[]")),
        metadata=dict(json.loads(row["target_metadata"] or "{}")),
    )


def _reference_from_row(row: sqlite3.Row) -> StoredCodeGraphReference:
    """Convert an unresolved_refs row to a reference value object."""
    return StoredCodeGraphReference(
        id=int(row["id"]),
        from_node_id=str(row["from_node_id"]),
        reference_name=str(row["reference_name"]),
        reference_kind=str(row["reference_kind"]),
        file_path=str(row["file_path"]),
        line=int(row["line"]) if row["line"] is not None else None,
        column=int(row["column"]) if row["column"] is not None else None,
        candidates=tuple(json.loads(row["candidates"] or "[]")),
        language=str(row["language"]) if row["language"] else None,
        resolved_target=str(row["resolved_target"]) if row["resolved_target"] else None,
        resolved_edge=str(row["resolved_edge"]) if row["resolved_edge"] else None,
        resolution_kind=str(row["resolution_kind"]) if row["resolution_kind"] else None,
        resolved_at=str(row["resolved_at"]) if row["resolved_at"] else None,
    )


def _relationship_from_joined_row(row: sqlite3.Row) -> dict[str, Any]:
    """Serialize a joined edge/source/target row into API-ready shape."""
    source = _node_from_row(row)
    target = _target_node_from_joined_row(row)
    return {
        "id": str(row["edge_id"]),
        "kind": str(row["edge_kind"]),
        "file_path": str(row["edge_file_path"]),
        "line": int(row["edge_line"]) if row["edge_line"] is not None else None,
        "column": int(row["edge_column"]) if row["edge_column"] is not None else None,
        "metadata": dict(json.loads(row["edge_metadata"] or "{}")),
        "provenance": str(row["edge_provenance"]) if row["edge_provenance"] else None,
        "source": codegraph_node_to_dict(source),
        "target": codegraph_node_to_dict(target),
    }


def _index_run_from_row(row: sqlite3.Row) -> IndexRunSummary:
    """Convert an index_runs table row to an IndexRunSummary."""
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
