"""Ownership-aware repository for Explainer workspace persistence."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections import defaultdict
from typing import Any

from tldw_Server_API.app.core.DB_Management.Explainer_DB import (
    ExplainerDatabase,
    InputError,
)
from tldw_Server_API.app.core.Explainer.models import (
    ExplainerCitation,
    ExplainerDepthPreset,
    ExplainerEvidenceState,
    ExplainerGrounding,
    ExplainerMode,
    ExplainerNode,
    ExplainerNodeKind,
    ExplainerNodeStatus,
    ExplainerOutputIntent,
    ExplainerSelectedSource,
    ExplainerSession,
    ExplainerSessionStatus,
    ExplainerSessionSummary,
)

_UNSET = object()


class ExplainerRepository:
    """CRUD repository that enforces user ownership on session access."""

    def __init__(self, db: ExplainerDatabase) -> None:
        self.db = db

    def create_session(
        self,
        *,
        owner_user_id: str,
        title: str,
        mode: str,
        output_intent: str,
        grounding: str,
        depth_preset: str,
        selected_sources: list[ExplainerSelectedSource | dict[str, Any]],
        root_prompt: str,
    ) -> ExplainerSession:
        owner_user_id = _require_text(owner_user_id, "owner_user_id")
        title = _require_text(title, "title")
        root_prompt = _require_text(root_prompt, "root_prompt")
        _require_enum(mode, ExplainerMode, "mode")
        _require_enum(output_intent, ExplainerOutputIntent, "output_intent")
        _require_enum(grounding, ExplainerGrounding, "grounding")
        _require_enum(depth_preset, ExplainerDepthPreset, "depth_preset")

        now = self.db.utcnow_iso()
        session_id = _new_id("exp_sess")
        root_node_id = _new_id("exp_node")
        normalized_sources = [
            _normalize_source(source, fallback_added_at=now)
            for source in selected_sources
        ]

        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO explainer_sessions (
                    id, owner_user_id, title, mode, status, output_intent, grounding,
                    depth_preset, created_at, updated_at, archived_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    session_id,
                    owner_user_id,
                    title,
                    mode,
                    ExplainerSessionStatus.ACTIVE.value,
                    output_intent,
                    grounding,
                    depth_preset,
                    now,
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO explainer_nodes (
                    id, session_id, parent_id, ordinal, title, body, kind, intent,
                    status, evidence_state, outside_knowledge_used, question_options_json,
                    selected_option_id, selected_custom_answer, generation_metadata_json,
                    created_at, updated_at, deleted_at
                )
                VALUES (?, ?, NULL, 0, ?, NULL, ?, ?, ?, ?, 0, NULL, NULL, NULL, NULL, ?, ?, NULL)
                """,
                (
                    root_node_id,
                    session_id,
                    root_prompt,
                    ExplainerNodeKind.QUESTION.value,
                    output_intent,
                    ExplainerNodeStatus.IDLE.value,
                    ExplainerEvidenceState.UNCITED.value,
                    now,
                    now,
                ),
            )
            self._replace_sources(
                conn,
                session_id=session_id,
                owner_user_id=owner_user_id,
                sources=normalized_sources,
                now=now,
            )

        loaded = self.get_session(session_id, owner_user_id=owner_user_id)
        if loaded is None:
            raise RuntimeError("created Explainer session could not be reloaded")
        return loaded

    def list_sessions(
        self,
        *,
        owner_user_id: str,
        include_archived: bool = False,
    ) -> list[ExplainerSession]:
        owner_user_id = _require_text(owner_user_id, "owner_user_id")
        conn = self.db.get_connection()
        session_rows = conn.execute(
            f"""
            SELECT *
            FROM explainer_sessions
            WHERE owner_user_id = ?{_archived_filter(include_archived)}
            ORDER BY updated_at DESC, created_at DESC
            """,
            (owner_user_id,),
        ).fetchall()
        if not session_rows:
            return []

        session_ids = [row["id"] for row in session_rows]
        placeholders = ", ".join("?" for _ in session_ids)
        node_rows = conn.execute(
            f"""
            SELECT *
            FROM explainer_nodes
            WHERE session_id IN ({placeholders}) AND deleted_at IS NULL
            ORDER BY parent_id IS NOT NULL, parent_id, ordinal, created_at
            """,
            session_ids,
        ).fetchall()
        source_rows = conn.execute(
            f"""
            SELECT *
            FROM explainer_selected_sources
            WHERE session_id IN ({placeholders}) AND owner_user_id = ? AND deleted_at IS NULL
            ORDER BY ordinal, added_at
            """,
            (*session_ids, owner_user_id),
        ).fetchall()
        citation_rows = conn.execute(
            f"""
            SELECT *
            FROM explainer_citations
            WHERE session_id IN ({placeholders}) AND owner_user_id = ? AND deleted_at IS NULL
            ORDER BY ordinal, created_at
            """,
            (*session_ids, owner_user_id),
        ).fetchall()

        nodes_by_session: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for row in node_rows:
            nodes_by_session[row["session_id"]].append(row)
        sources_by_session: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for row in source_rows:
            sources_by_session[row["session_id"]].append(row)
        citations_by_session: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for row in citation_rows:
            citations_by_session[row["session_id"]].append(row)

        return [
            self._build_session(
                row,
                nodes_by_session.get(row["id"], []),
                sources_by_session.get(row["id"], []),
                citations_by_session.get(row["id"], []),
            )
            for row in session_rows
        ]

    def list_session_summaries(
        self,
        *,
        owner_user_id: str,
        include_archived: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[ExplainerSessionSummary], int]:
        owner_user_id = _require_text(owner_user_id, "owner_user_id")
        safe_limit = max(1, min(int(limit), 100))
        safe_offset = max(0, int(offset))
        conn = self.db.get_connection()
        total = conn.execute(
            f"""
            SELECT COUNT(*) AS total
            FROM explainer_sessions
            WHERE owner_user_id = ?{_archived_filter(include_archived)}
            """,
            (owner_user_id,),
        ).fetchone()["total"]
        rows = conn.execute(
            f"""
            SELECT
                s.*,
                (
                    SELECT COUNT(*)
                    FROM explainer_nodes n
                    WHERE n.session_id = s.id AND n.deleted_at IS NULL
                ) AS node_count,
                (
                    SELECT COUNT(*)
                    FROM explainer_selected_sources src
                    WHERE src.session_id = s.id
                      AND src.owner_user_id = s.owner_user_id
                      AND src.deleted_at IS NULL
                ) AS selected_source_count
            FROM explainer_sessions s
            WHERE s.owner_user_id = ?{_archived_filter(include_archived, alias="s")}
            ORDER BY s.updated_at DESC, s.created_at DESC
            LIMIT ? OFFSET ?
            """,
            (owner_user_id, safe_limit, safe_offset),
        ).fetchall()
        return (
            [
                ExplainerSessionSummary(
                    id=row["id"],
                    owner_user_id=row["owner_user_id"],
                    title=row["title"],
                    mode=row["mode"],
                    status=row["status"],
                    output_intent=row["output_intent"],
                    grounding=row["grounding"],
                    depth_preset=row["depth_preset"],
                    node_count=int(row["node_count"]),
                    selected_source_count=int(row["selected_source_count"]),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    archived_at=row["archived_at"],
                )
                for row in rows
            ],
            int(total),
        )

    def get_session(
        self,
        session_id: str,
        *,
        owner_user_id: str,
        include_archived: bool = False,
    ) -> ExplainerSession | None:
        session_id = _require_text(session_id, "session_id")
        owner_user_id = _require_text(owner_user_id, "owner_user_id")
        conn = self.db.get_connection()
        session_row = conn.execute(
            f"""
            SELECT *
            FROM explainer_sessions
            WHERE id = ? AND owner_user_id = ?{_archived_filter(include_archived)}
            """,
            (session_id, owner_user_id),
        ).fetchone()
        if session_row is None:
            return None

        node_rows = conn.execute(
            """
            SELECT *
            FROM explainer_nodes
            WHERE session_id = ? AND deleted_at IS NULL
            ORDER BY parent_id IS NOT NULL, parent_id, ordinal, created_at
            """,
            (session_id,),
        ).fetchall()
        source_rows = conn.execute(
            """
            SELECT *
            FROM explainer_selected_sources
            WHERE session_id = ? AND owner_user_id = ? AND deleted_at IS NULL
            ORDER BY ordinal, added_at
            """,
            (session_id, owner_user_id),
        ).fetchall()
        citation_rows = conn.execute(
            """
            SELECT *
            FROM explainer_citations
            WHERE session_id = ? AND owner_user_id = ? AND deleted_at IS NULL
            ORDER BY ordinal, created_at
            """,
            (session_id, owner_user_id),
        ).fetchall()
        return self._build_session(session_row, node_rows, source_rows, citation_rows)

    def update_session(
        self,
        session_id: str,
        *,
        owner_user_id: str,
        title: str | None = None,
        output_intent: str | None = None,
        grounding: str | None = None,
        depth_preset: str | None = None,
        selected_sources: list[ExplainerSelectedSource | dict[str, Any]] | None = None,
    ) -> ExplainerSession | None:
        session = self.get_session(session_id, owner_user_id=owner_user_id)
        if session is None:
            return None

        if output_intent is not None:
            _require_enum(output_intent, ExplainerOutputIntent, "output_intent")
        if grounding is not None:
            _require_enum(grounding, ExplainerGrounding, "grounding")
        if depth_preset is not None:
            _require_enum(depth_preset, ExplainerDepthPreset, "depth_preset")

        next_title = _require_text(title, "title") if title is not None else session.title
        next_output_intent = output_intent or session.output_intent
        next_grounding = grounding or session.grounding
        next_depth_preset = depth_preset or session.depth_preset

        now = self.db.utcnow_iso()
        with self.db.transaction() as conn:
            if any(
                value is not None
                for value in (title, output_intent, grounding, depth_preset)
            ):
                conn.execute(
                    """
                    UPDATE explainer_sessions
                    SET title = ?, output_intent = ?, grounding = ?,
                        depth_preset = ?, updated_at = ?
                    WHERE id = ? AND owner_user_id = ? AND archived_at IS NULL
                    """,
                    (
                        next_title,
                        next_output_intent,
                        next_grounding,
                        next_depth_preset,
                        now,
                        session_id,
                        owner_user_id,
                    ),
                )
            if selected_sources is not None:
                normalized_sources = [
                    _normalize_source(source, fallback_added_at=now)
                    for source in selected_sources
                ]
                self._replace_sources(
                    conn,
                    session_id=session_id,
                    owner_user_id=owner_user_id,
                    sources=normalized_sources,
                    now=now,
                )
                conn.execute(
                    """
                    UPDATE explainer_sessions
                    SET updated_at = ?
                    WHERE id = ? AND owner_user_id = ? AND archived_at IS NULL
                    """,
                    (now, session_id, owner_user_id),
                )

        return self.get_session(session_id, owner_user_id=owner_user_id)

    def archive_session(
        self,
        session_id: str,
        *,
        owner_user_id: str,
    ) -> ExplainerSession | None:
        session = self.get_session(session_id, owner_user_id=owner_user_id)
        if session is None:
            return None
        now = self.db.utcnow_iso()
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE explainer_sessions
                SET status = ?, archived_at = ?, updated_at = ?
                WHERE id = ? AND owner_user_id = ? AND archived_at IS NULL
                """,
                (ExplainerSessionStatus.ARCHIVED.value, now, now, session_id, owner_user_id),
            )
        return self.get_session(session_id, owner_user_id=owner_user_id, include_archived=True)

    def create_node(
        self,
        session_id: str,
        *,
        owner_user_id: str,
        title: str,
        parent_id: str | None = None,
        body: str | None = None,
        kind: str = ExplainerNodeKind.EXPLANATION.value,
        intent: str = ExplainerOutputIntent.EXPLAIN.value,
        status: str = ExplainerNodeStatus.IDLE.value,
        evidence_state: str = ExplainerEvidenceState.UNCITED.value,
        outside_knowledge_used: bool = False,
        citations: list[ExplainerCitation | dict[str, Any]] | None = None,
        generation_metadata: dict[str, Any] | None = None,
    ) -> ExplainerNode | None:
        created = self.create_child_nodes(
            session_id,
            owner_user_id=owner_user_id,
            parent_id=parent_id,
            children=[
                {
                    "title": title,
                    "body": body,
                    "kind": kind,
                    "intent": intent,
                    "status": status,
                    "evidence_state": evidence_state,
                    "outside_knowledge_used": outside_knowledge_used,
                    "citations": citations or [],
                    "generation_metadata": generation_metadata,
                }
            ],
        )
        return None if created is None else created[0]

    def create_child_nodes(
        self,
        session_id: str,
        *,
        owner_user_id: str,
        parent_id: str | None,
        children: list[dict[str, Any]],
    ) -> list[ExplainerNode] | None:
        """Create sibling nodes (with citations and metadata) in one transaction."""
        if not children:
            raise InputError("children is required")
        session = self.get_session(session_id, owner_user_id=owner_user_id)
        if session is None:
            return None
        if parent_id is not None and parent_id not in session.nodes:
            raise InputError("parent_id does not belong to session")

        normalized_children = [_normalize_child_spec(child) for child in children]

        now = self.db.utcnow_iso()
        created_ids: list[str] = []
        with self.db.transaction() as conn:
            ordinal = self._next_child_ordinal(conn, session_id=session_id, parent_id=parent_id)
            for child in normalized_children:
                node_id = _new_id("exp_node")
                conn.execute(
                    """
                    INSERT INTO explainer_nodes (
                        id, session_id, parent_id, ordinal, title, body, kind, intent,
                        status, evidence_state, outside_knowledge_used, question_options_json,
                        selected_option_id, selected_custom_answer, generation_metadata_json,
                        created_at, updated_at, deleted_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?, ?, NULL)
                    """,
                    (
                        node_id,
                        session_id,
                        parent_id,
                        ordinal,
                        child["title"],
                        child["body"],
                        child["kind"],
                        child["intent"],
                        child["status"],
                        child["evidence_state"],
                        1 if child["outside_knowledge_used"] else 0,
                        _json_dumps(child["generation_metadata"]),
                        now,
                        now,
                    ),
                )
                self._replace_node_citations(
                    conn,
                    session_id=session_id,
                    node_id=node_id,
                    owner_user_id=owner_user_id,
                    citations=child["citations"],
                    now=now,
                )
                created_ids.append(node_id)
                ordinal += 1
            conn.execute(
                """
                UPDATE explainer_sessions
                SET updated_at = ?
                WHERE id = ? AND owner_user_id = ?
                """,
                (now, session_id, owner_user_id),
            )
        loaded = self.get_session(session_id, owner_user_id=owner_user_id)
        if loaded is None:
            return None
        return [loaded.nodes[node_id] for node_id in created_ids]

    def transition_node_status(
        self,
        session_id: str,
        node_id: str,
        *,
        owner_user_id: str,
        to_status: str,
        from_statuses: list[str] | set[str] | tuple[str, ...],
    ) -> bool:
        """Atomically move a node's status only if it is currently in an allowed state."""
        session_id = _require_text(session_id, "session_id")
        node_id = _require_text(node_id, "node_id")
        owner_user_id = _require_text(owner_user_id, "owner_user_id")
        _require_enum(to_status, ExplainerNodeStatus, "to_status")
        allowed_from = list(from_statuses)
        if not allowed_from:
            raise InputError("from_statuses is required")
        for value in allowed_from:
            _require_enum(value, ExplainerNodeStatus, "from_statuses")

        now = self.db.utcnow_iso()
        placeholders = ", ".join("?" for _ in allowed_from)
        with self.db.transaction() as conn:
            cursor = conn.execute(
                f"""
                UPDATE explainer_nodes
                SET status = ?, updated_at = ?
                WHERE id = ? AND session_id = ? AND deleted_at IS NULL
                  AND status IN ({placeholders})
                  AND EXISTS (
                    SELECT 1 FROM explainer_sessions s
                    WHERE s.id = explainer_nodes.session_id AND s.owner_user_id = ?
                  )
                """,
                (to_status, now, node_id, session_id, *allowed_from, owner_user_id),
            )
            moved = cursor.rowcount > 0
            if moved:
                conn.execute(
                    """
                    UPDATE explainer_sessions
                    SET updated_at = ?
                    WHERE id = ? AND owner_user_id = ?
                    """,
                    (now, session_id, owner_user_id),
                )
        return moved

    def delete_session(
        self,
        session_id: str,
        *,
        owner_user_id: str,
    ) -> bool:
        """Hard-delete a session and all dependent rows (FK cascade)."""
        session_id = _require_text(session_id, "session_id")
        owner_user_id = _require_text(owner_user_id, "owner_user_id")
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                DELETE FROM explainer_sessions
                WHERE id = ? AND owner_user_id = ?
                """,
                (session_id, owner_user_id),
            )
        return cursor.rowcount > 0

    def update_node(
        self,
        session_id: str,
        node_id: str,
        *,
        owner_user_id: str,
        title: Any = _UNSET,
        body: Any = _UNSET,
        kind: Any = _UNSET,
        intent: Any = _UNSET,
        status: Any = _UNSET,
        evidence_state: Any = _UNSET,
        outside_knowledge_used: Any = _UNSET,
        selected_option_id: Any = _UNSET,
        selected_custom_answer: Any = _UNSET,
        question_options: Any = _UNSET,
        generation_metadata: Any = _UNSET,
        citations: Any = _UNSET,
    ) -> ExplainerNode | None:
        session = self.get_session(session_id, owner_user_id=owner_user_id)
        if session is None or node_id not in session.nodes:
            return None
        existing_node = session.nodes[node_id]

        if title is not _UNSET and title is None:
            raise InputError("title is required")
        if kind is not _UNSET and kind is None:
            raise InputError("kind is required")
        if intent is not _UNSET and intent is None:
            raise InputError("intent is required")
        if status is not _UNSET and status is None:
            raise InputError("status is required")
        if evidence_state is not _UNSET and evidence_state is None:
            raise InputError("evidence_state is required")
        if outside_knowledge_used is not _UNSET and outside_knowledge_used is None:
            raise InputError("outside_knowledge_used is required")
        if kind is not _UNSET:
            _require_enum(kind, ExplainerNodeKind, "kind")
        if intent is not _UNSET:
            _require_enum(intent, ExplainerOutputIntent, "intent")
        if status is not _UNSET:
            _require_enum(status, ExplainerNodeStatus, "status")
        if evidence_state is not _UNSET:
            _require_enum(evidence_state, ExplainerEvidenceState, "evidence_state")
        if not any(
            value is not _UNSET
            for value in (
                title,
                body,
                kind,
                intent,
                status,
                evidence_state,
                outside_knowledge_used,
                selected_option_id,
                selected_custom_answer,
                question_options,
                generation_metadata,
                citations,
            )
        ):
            return existing_node

        now = self.db.utcnow_iso()
        next_outside_knowledge_used = (
            outside_knowledge_used
            if outside_knowledge_used is not _UNSET
            else existing_node.outside_knowledge_used
        )
        next_question_options = (
            question_options
            if question_options is not _UNSET
            else existing_node.question_options
        )
        next_generation_metadata = (
            generation_metadata
            if generation_metadata is not _UNSET
            else existing_node.generation_metadata
        )
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE explainer_nodes
                SET title = ?, body = ?, kind = ?, intent = ?,
                    status = ?, evidence_state = ?,
                    outside_knowledge_used = ?, selected_option_id = ?,
                    selected_custom_answer = ?, question_options_json = ?,
                    generation_metadata_json = ?, updated_at = ?
                WHERE id = ? AND session_id = ? AND deleted_at IS NULL
                """,
                (
                    _require_text(title, "title") if title is not _UNSET else existing_node.title,
                    body if body is not _UNSET else existing_node.body,
                    kind if kind is not _UNSET else existing_node.kind,
                    intent if intent is not _UNSET else existing_node.intent,
                    status if status is not _UNSET else existing_node.status,
                    evidence_state if evidence_state is not _UNSET else existing_node.evidence_state,
                    1 if next_outside_knowledge_used else 0,
                    selected_option_id
                    if selected_option_id is not _UNSET
                    else existing_node.selected_option_id,
                    selected_custom_answer
                    if selected_custom_answer is not _UNSET
                    else existing_node.selected_custom_answer,
                    _json_dumps(next_question_options),
                    _json_dumps(next_generation_metadata),
                    now,
                    node_id,
                    session_id,
                ),
            )
            if citations is not _UNSET:
                self._replace_node_citations(
                    conn,
                    session_id=session_id,
                    node_id=node_id,
                    owner_user_id=owner_user_id,
                    citations=[
                        _normalize_citation(citation)
                        for citation in citations or []
                    ],
                    now=now,
                )
            conn.execute(
                """
                UPDATE explainer_sessions
                SET updated_at = ?
                WHERE id = ? AND owner_user_id = ?
                """,
                (now, session_id, owner_user_id),
            )
        loaded = self.get_session(session_id, owner_user_id=owner_user_id)
        return None if loaded is None else loaded.nodes.get(node_id)

    def delete_node(
        self,
        session_id: str,
        node_id: str,
        *,
        owner_user_id: str,
    ) -> bool:
        session = self.get_session(session_id, owner_user_id=owner_user_id)
        if session is None or node_id not in session.nodes:
            return False
        now = self.db.utcnow_iso()
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE explainer_nodes
                SET deleted_at = ?, updated_at = ?
                WHERE session_id = ? AND deleted_at IS NULL AND id IN (
                    WITH RECURSIVE subtree(id) AS (
                        SELECT id
                        FROM explainer_nodes
                        WHERE id = ? AND session_id = ? AND deleted_at IS NULL
                        UNION ALL
                        SELECT child.id
                        FROM explainer_nodes child
                        JOIN subtree parent ON child.parent_id = parent.id
                        WHERE child.session_id = ? AND child.deleted_at IS NULL
                    )
                    SELECT id FROM subtree
                )
                """,
                (now, now, session_id, node_id, session_id, session_id),
            )
            conn.execute(
                """
                UPDATE explainer_citations
                SET deleted_at = ?
                WHERE session_id = ? AND owner_user_id = ? AND deleted_at IS NULL
                  AND node_id IN (
                    WITH RECURSIVE subtree(id) AS (
                        SELECT id
                        FROM explainer_nodes
                        WHERE id = ? AND session_id = ?
                        UNION ALL
                        SELECT child.id
                        FROM explainer_nodes child
                        JOIN subtree parent ON child.parent_id = parent.id
                        WHERE child.session_id = ?
                    )
                    SELECT id FROM subtree
                  )
                """,
                (now, session_id, owner_user_id, node_id, session_id, session_id),
            )
            conn.execute(
                """
                UPDATE explainer_sessions
                SET updated_at = ?
                WHERE id = ? AND owner_user_id = ?
                """,
                (now, session_id, owner_user_id),
            )
        return True

    def _build_session(
        self,
        session_row: sqlite3.Row,
        node_rows: list[sqlite3.Row],
        source_rows: list[sqlite3.Row],
        citation_rows: list[sqlite3.Row],
    ) -> ExplainerSession:
        citations_by_node: dict[str, list[ExplainerCitation]] = defaultdict(list)
        for row in citation_rows:
            citations_by_node[row["node_id"]].append(
                ExplainerCitation(
                    id=row["id"],
                    source_id=row["source_id"],
                    source_type=row["source_type"],
                    title=row["title"],
                    excerpt=row["excerpt"],
                    location_label=row["location_label"],
                    start_offset=row["start_offset"],
                    end_offset=row["end_offset"],
                    url=row["url"],
                    snapshot_hash=row["snapshot_hash"],
                )
            )

        child_ids_by_parent: dict[str | None, list[str]] = defaultdict(list)
        nodes: dict[str, ExplainerNode] = {}
        for row in node_rows:
            child_ids_by_parent[row["parent_id"]].append(row["id"])
            nodes[row["id"]] = ExplainerNode(
                id=row["id"],
                session_id=row["session_id"],
                parent_id=row["parent_id"],
                ordinal=row["ordinal"],
                title=row["title"],
                body=row["body"],
                kind=row["kind"],
                intent=row["intent"],
                status=row["status"],
                evidence_state=row["evidence_state"],
                outside_knowledge_used=bool(row["outside_knowledge_used"]),
                citations=citations_by_node.get(row["id"], []),
                question_options=_json_loads(row["question_options_json"]),
                selected_option_id=row["selected_option_id"],
                selected_custom_answer=row["selected_custom_answer"],
                generation_metadata=_json_loads(row["generation_metadata_json"]),
                child_node_ids=[],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                deleted_at=row["deleted_at"],
            )
        for parent_id, child_ids in child_ids_by_parent.items():
            if parent_id is not None and parent_id in nodes:
                nodes[parent_id].child_node_ids = child_ids

        selected_sources = [
            ExplainerSelectedSource(
                source_id=row["source_id"],
                source_type=row["source_type"],
                title=row["title"],
                added_at=row["added_at"],
                snapshot_version=row["snapshot_version"],
                metadata=_json_loads(row["metadata_json"]),
            )
            for row in source_rows
        ]

        return ExplainerSession(
            id=session_row["id"],
            owner_user_id=session_row["owner_user_id"],
            title=session_row["title"],
            mode=session_row["mode"],
            status=session_row["status"],
            output_intent=session_row["output_intent"],
            grounding=session_row["grounding"],
            depth_preset=session_row["depth_preset"],
            selected_sources=selected_sources,
            root_node_ids=child_ids_by_parent.get(None, []),
            nodes=nodes,
            created_at=session_row["created_at"],
            updated_at=session_row["updated_at"],
            archived_at=session_row["archived_at"],
        )

    def _replace_sources(
        self,
        conn: sqlite3.Connection,
        *,
        session_id: str,
        owner_user_id: str,
        sources: list[ExplainerSelectedSource],
        now: str,
    ) -> None:
        conn.execute(
            """
            UPDATE explainer_selected_sources
            SET deleted_at = ?
            WHERE session_id = ? AND owner_user_id = ? AND deleted_at IS NULL
            """,
            (now, session_id, owner_user_id),
        )
        for ordinal, source in enumerate(sources):
            conn.execute(
                """
                INSERT INTO explainer_selected_sources (
                    id, session_id, owner_user_id, ordinal, source_id, source_type, title,
                    added_at, snapshot_version, metadata_json, deleted_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    _new_id("exp_src"),
                    session_id,
                    owner_user_id,
                    ordinal,
                    source.source_id,
                    source.source_type,
                    source.title,
                    source.added_at,
                    source.snapshot_version,
                    _json_dumps(source.metadata),
                ),
            )

    def _replace_node_citations(
        self,
        conn: sqlite3.Connection,
        *,
        session_id: str,
        node_id: str,
        owner_user_id: str,
        citations: list[ExplainerCitation],
        now: str,
    ) -> None:
        conn.execute(
            """
            UPDATE explainer_citations
            SET deleted_at = ?
            WHERE session_id = ? AND node_id = ? AND owner_user_id = ? AND deleted_at IS NULL
            """,
            (now, session_id, node_id, owner_user_id),
        )
        for ordinal, citation in enumerate(citations):
            conn.execute(
                """
                INSERT INTO explainer_citations (
                    id, session_id, node_id, owner_user_id, ordinal, source_id, source_type,
                    title, excerpt, location_label, start_offset, end_offset, url,
                    snapshot_hash, created_at, deleted_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    citation.id,
                    session_id,
                    node_id,
                    owner_user_id,
                    ordinal,
                    citation.source_id,
                    citation.source_type,
                    citation.title,
                    citation.excerpt,
                    citation.location_label,
                    citation.start_offset,
                    citation.end_offset,
                    citation.url,
                    citation.snapshot_hash,
                    now,
                ),
            )

    @staticmethod
    def _next_child_ordinal(
        conn: sqlite3.Connection,
        *,
        session_id: str,
        parent_id: str | None,
    ) -> int:
        if parent_id is None:
            row = conn.execute(
                """
                SELECT COALESCE(MAX(ordinal), -1) AS max_ordinal
                FROM explainer_nodes
                WHERE session_id = ? AND parent_id IS NULL AND deleted_at IS NULL
                """,
                (session_id,),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT COALESCE(MAX(ordinal), -1) AS max_ordinal
                FROM explainer_nodes
                WHERE session_id = ? AND parent_id = ? AND deleted_at IS NULL
                """,
                (session_id, parent_id),
            ).fetchone()
        return int(row["max_ordinal"]) + 1


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _archived_filter(include_archived: bool, *, alias: str | None = None) -> str:
    """Shared archived-visibility predicate for session queries."""
    if include_archived:
        return ""
    column = f"{alias}.archived_at" if alias else "archived_at"
    return f" AND {column} IS NULL"


def _require_text(value: str, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise InputError(f"{field_name} is required")
    return text


def _require_enum(value: str, enum_type: type, field_name: str) -> None:
    allowed = {item.value for item in enum_type}
    if value not in allowed:
        raise InputError(f"{field_name} must be one of {sorted(allowed)}")


def _normalize_source(
    source: ExplainerSelectedSource | dict[str, Any],
    *,
    fallback_added_at: str,
) -> ExplainerSelectedSource:
    if isinstance(source, ExplainerSelectedSource):
        return source
    return ExplainerSelectedSource(
        source_id=_require_text(str(source.get("source_id", "")), "source_id"),
        source_type=_require_text(str(source.get("source_type", "")), "source_type"),
        title=_require_text(str(source.get("title", "")), "source title"),
        added_at=str(source.get("added_at") or fallback_added_at),
        snapshot_version=source.get("snapshot_version"),
        metadata=source.get("metadata"),
    )


def _normalize_child_spec(child: dict[str, Any]) -> dict[str, Any]:
    """Validate and default one child-node spec for batch creation."""
    kind = child.get("kind", ExplainerNodeKind.EXPLANATION.value)
    intent = child.get("intent", ExplainerOutputIntent.EXPLAIN.value)
    status = child.get("status", ExplainerNodeStatus.IDLE.value)
    evidence_state = child.get("evidence_state", ExplainerEvidenceState.UNCITED.value)
    _require_enum(kind, ExplainerNodeKind, "kind")
    _require_enum(intent, ExplainerOutputIntent, "intent")
    _require_enum(status, ExplainerNodeStatus, "status")
    _require_enum(evidence_state, ExplainerEvidenceState, "evidence_state")
    generation_metadata = child.get("generation_metadata")
    if generation_metadata is not None and not isinstance(generation_metadata, dict):
        raise InputError("generation_metadata must be an object")
    return {
        "title": _require_text(child.get("title"), "title"),
        "body": child.get("body"),
        "kind": kind,
        "intent": intent,
        "status": status,
        "evidence_state": evidence_state,
        "outside_knowledge_used": bool(child.get("outside_knowledge_used")),
        "citations": [
            _normalize_citation(citation)
            for citation in child.get("citations") or []
        ],
        "generation_metadata": generation_metadata,
    }


def _normalize_citation(
    citation: ExplainerCitation | dict[str, Any],
) -> ExplainerCitation:
    if isinstance(citation, ExplainerCitation):
        return citation
    return ExplainerCitation(
        id=str(citation.get("id") or _new_id("exp_cite")),
        source_id=_require_text(str(citation.get("source_id", "")), "citation source_id"),
        source_type=_require_text(str(citation.get("source_type", "")), "citation source_type"),
        title=_require_text(str(citation.get("title", "")), "citation title"),
        excerpt=_require_text(str(citation.get("excerpt", "")), "citation excerpt"),
        location_label=citation.get("location_label"),
        start_offset=citation.get("start_offset"),
        end_offset=citation.get("end_offset"),
        url=citation.get("url"),
        snapshot_hash=citation.get("snapshot_hash"),
    )


def _json_dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_loads(value: str | None) -> Any:
    if not value:
        return None
    return json.loads(value)
